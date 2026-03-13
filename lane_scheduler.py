from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from django.conf import settings

try:
    import fcntl
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError("lane_scheduler requires fcntl support on this host.") from exc

_LOGGER = logging.getLogger(__name__)

LANE_INTERACTIVE = "interactive"
LANE_NON_INTERACTIVE = "non-interactive"
_LANE_PRIORITY = {
    LANE_INTERACTIVE: 0,
    LANE_NON_INTERACTIVE: 1,
}
_POLL_INTERVAL_SECONDS = 0.05


@dataclass(frozen=True)
class AdmissionTicket:
    lane: str
    token: str
    waited_ms: int


@dataclass(frozen=True)
class WaiterRecord:
    token: str
    lane: str
    pid: int
    created_at: float

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> WaiterRecord:
        token = payload.get("token")
        lane = payload.get("lane")
        pid = payload.get("pid")
        created_at = payload.get("created_at")
        if token is None or lane is None or pid is None or created_at is None:
            raise KeyError("waiter payload is missing required fields")
        return cls(
            token=str(token),
            lane=normalize_lane(str(lane)),
            pid=int(str(pid)),
            created_at=float(str(created_at)),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "token": self.token,
            "lane": self.lane,
            "pid": self.pid,
            "created_at": self.created_at,
        }


class LaneScheduler:
    def __init__(self, *, root_dir: Path, stale_seconds: int) -> None:
        self.root_dir = root_dir
        self.stale_seconds = stale_seconds
        self.waiters_dir = root_dir / "waiters"
        self.control_lock_path = root_dir / "control.lock"
        self.holder_path = root_dir / "holder.json"

    def acquire(self, *, lane: str) -> AdmissionTicket:
        normalized_lane = normalize_lane(lane)
        self._ensure_layout()
        waiter = WaiterRecord(
            token=uuid.uuid4().hex,
            lane=normalized_lane,
            pid=os.getpid(),
            created_at=time.time(),
        )
        waiter_path = self._waiter_path(waiter.token)
        acquired = False
        started_wait = time.monotonic()

        with self._control_lock():
            self._write_json(waiter_path, waiter.as_dict())

        try:
            while True:
                with self._control_lock():
                    self._cleanup_dead_waiters()
                    self._cleanup_stale_holder()
                    if not waiter_path.exists():
                        self._write_json(waiter_path, waiter.as_dict())

                    winner = self._select_next_waiter()
                    if (
                        winner is not None
                        and winner.token == waiter.token
                        and not self.holder_path.exists()
                    ):
                        self._write_json(
                            self.holder_path,
                            {
                                "token": waiter.token,
                                "lane": waiter.lane,
                                "pid": waiter.pid,
                                "claimed_at": time.time(),
                            },
                        )
                        waiter_path.unlink(missing_ok=True)
                        acquired = True
                        waited_ms = int((time.monotonic() - started_wait) * 1000)
                        _LOGGER.info(
                            "lane_scheduler admitted lane=%s wait_ms=%s pid=%s",
                            waiter.lane,
                            waited_ms,
                            waiter.pid,
                        )
                        return AdmissionTicket(
                            lane=waiter.lane,
                            token=waiter.token,
                            waited_ms=waited_ms,
                        )

                time.sleep(_POLL_INTERVAL_SECONDS)
        finally:
            if not acquired:
                with self._control_lock():
                    waiter_path.unlink(missing_ok=True)

    def release(self, ticket: AdmissionTicket) -> None:
        with self._control_lock():
            holder = self._read_holder()
            if holder is None:
                return
            if str(holder.get("token")) == ticket.token:
                self.holder_path.unlink(missing_ok=True)

    def _ensure_layout(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.waiters_dir.mkdir(parents=True, exist_ok=True)
        self.control_lock_path.touch(exist_ok=True)

    @contextmanager
    def _control_lock(self) -> Iterator[None]:
        with self.control_lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _cleanup_dead_waiters(self) -> None:
        for waiter_path in self.waiters_dir.glob("*.json"):
            waiter = self._read_waiter(waiter_path)
            if waiter is None:
                waiter_path.unlink(missing_ok=True)
                continue
            if not _pid_is_alive(waiter.pid):
                _LOGGER.warning(
                    "lane_scheduler removed dead waiter lane=%s pid=%s token=%s",
                    waiter.lane,
                    waiter.pid,
                    waiter.token,
                )
                waiter_path.unlink(missing_ok=True)

    def _cleanup_stale_holder(self) -> None:
        holder = self._read_holder()
        if holder is None:
            return
        try:
            pid_raw = holder["pid"]
            claimed_at_raw = holder["claimed_at"]
            pid = int(str(pid_raw))
            claimed_at = float(str(claimed_at_raw))
        except (KeyError, TypeError, ValueError):
            _LOGGER.warning("lane_scheduler removed invalid holder state")
            self.holder_path.unlink(missing_ok=True)
            return

        if not _pid_is_alive(pid):
            _LOGGER.warning(
                "lane_scheduler reclaimed dead holder lane=%s pid=%s token=%s",
                holder.get("lane", "unknown"),
                pid,
                holder.get("token", "unknown"),
            )
            self.holder_path.unlink(missing_ok=True)
            return

        age_seconds = time.time() - claimed_at
        if age_seconds <= self.stale_seconds:
            return

        _LOGGER.warning(
            "lane_scheduler reclaimed stale holder lane=%s pid=%s age_seconds=%.3f token=%s",
            holder.get("lane", "unknown"),
            holder.get("pid", "unknown"),
            age_seconds,
            holder.get("token", "unknown"),
        )
        self.holder_path.unlink(missing_ok=True)

    def _select_next_waiter(self) -> WaiterRecord | None:
        waiters: list[WaiterRecord] = []
        for waiter_path in self.waiters_dir.glob("*.json"):
            waiter = self._read_waiter(waiter_path)
            if waiter is not None:
                waiters.append(waiter)
        if not waiters:
            return None
        waiters.sort(
            key=lambda waiter: (
                _LANE_PRIORITY[waiter.lane],
                waiter.created_at,
                waiter.token,
            )
        )
        return waiters[0]

    def _read_waiter(self, waiter_path: Path) -> WaiterRecord | None:
        payload = self._read_json(waiter_path)
        if payload is None:
            return None
        try:
            return WaiterRecord.from_dict(payload)
        except (KeyError, TypeError, ValueError):
            _LOGGER.warning("lane_scheduler removed invalid waiter state path=%s", waiter_path)
            return None

    def _read_holder(self) -> dict[str, object] | None:
        return self._read_json(self.holder_path)

    def _read_json(self, path: Path) -> dict[str, object] | None:
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    def _write_json(self, path: Path, payload: dict[str, object]) -> None:
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    def _waiter_path(self, token: str) -> Path:
        return self.waiters_dir / f"{token}.json"


@contextmanager
def admit_local_inference(lane: str) -> Iterator[AdmissionTicket]:
    normalized_lane = normalize_lane(lane)
    if not settings.VOXHELM_LANE_SCHEDULER_ENABLED:
        yield AdmissionTicket(lane=normalized_lane, token="disabled", waited_ms=0)
        return

    scheduler = LaneScheduler(
        root_dir=settings.VOXHELM_LANE_SCHEDULER_DIR,
        stale_seconds=settings.VOXHELM_LANE_SCHEDULER_STALE_SECONDS,
    )
    ticket = scheduler.acquire(lane=normalized_lane)
    try:
        yield ticket
    finally:
        scheduler.release(ticket)


def normalize_lane(lane: str) -> str:
    normalized = lane.strip().lower()
    if normalized not in _LANE_PRIORITY:
        accepted = ", ".join(sorted(_LANE_PRIORITY))
        raise ValueError(f"Unsupported scheduler lane '{lane}'. Accepted values: {accepted}.")
    return normalized


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True

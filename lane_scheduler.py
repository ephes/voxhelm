from __future__ import annotations

import json
import logging
import math
import os
import threading
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


class InferenceCancelled(RuntimeError):
    """Raised when a caller cancelled local inference before or while it ran.

    Raised by the scheduler when ``cancel_event`` is set while the caller is still
    queued, and by the STT service at its cancellation checkpoints. It is not a
    backend-availability error, so it never triggers backend fallback.
    """


def _finite_timestamp(value: object, field: str) -> float:
    """Parse a record timestamp, rejecting NaN and infinities.

    A non-finite ``claimed_at`` would never compare greater than the stale
    window, so the holder could never be reclaimed while its pid stays alive.
    """
    parsed = float(str(value))
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be a finite number")
    return parsed


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
            created_at=_finite_timestamp(created_at, "created_at"),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "token": self.token,
            "lane": self.lane,
            "pid": self.pid,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class HolderRecord:
    token: str
    lane: str
    pid: int
    claimed_at: float

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> HolderRecord:
        token = payload.get("token")
        lane = payload.get("lane")
        pid = payload.get("pid")
        claimed_at = payload.get("claimed_at")
        if token is None or lane is None or pid is None or claimed_at is None:
            raise KeyError("holder payload is missing required fields")
        return cls(
            token=str(token),
            lane=normalize_lane(str(lane)),
            pid=int(str(pid)),
            claimed_at=_finite_timestamp(claimed_at, "claimed_at"),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "token": self.token,
            "lane": self.lane,
            "pid": self.pid,
            "claimed_at": self.claimed_at,
        }


class LaneScheduler:
    """Cross-process admission control with bounded slots per lane.

    Capacity is ``interactive_slots + non_interactive_slots``. The interactive
    slots are reserved: a non-interactive holder can never occupy them, while an
    interactive request may also use an idle non-interactive slot. Holders live
    in ``holders/<token>.json``; a legacy single-slot ``holder.json`` written by
    older code counts as one non-interactive holder.
    """

    def __init__(
        self,
        *,
        root_dir: Path,
        stale_seconds: int,
        interactive_slots: int = 1,
        non_interactive_slots: int = 1,
    ) -> None:
        if non_interactive_slots < 1:
            raise ValueError("non_interactive_slots must be a positive integer.")
        if interactive_slots < 0:
            raise ValueError("interactive_slots must be a non-negative integer.")
        self.root_dir = root_dir
        self.stale_seconds = stale_seconds
        self.interactive_slots = interactive_slots
        self.non_interactive_slots = non_interactive_slots
        self.waiters_dir = root_dir / "waiters"
        self.holders_dir = root_dir / "holders"
        self.control_lock_path = root_dir / "control.lock"
        self.legacy_holder_path = root_dir / "holder.json"

    @property
    def total_slots(self) -> int:
        return self.interactive_slots + self.non_interactive_slots

    def acquire(
        self,
        *,
        lane: str,
        cancel_event: threading.Event | None = None,
    ) -> AdmissionTicket:
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
                if cancel_event is not None and cancel_event.is_set():
                    raise InferenceCancelled(
                        f"Local inference was cancelled while waiting for lane '{waiter.lane}'."
                    )
                with self._control_lock():
                    self._cleanup_dead_waiters()
                    total, non_interactive = self._reclaim_holders()
                    if not waiter_path.exists():
                        self._write_json(waiter_path, waiter.as_dict())

                    winner = self._select_next_waiter(
                        total_holders=total,
                        non_interactive_holders=non_interactive,
                    )
                    if winner is not None and winner.token == waiter.token:
                        holder = HolderRecord(
                            token=waiter.token,
                            lane=waiter.lane,
                            pid=waiter.pid,
                            claimed_at=time.time(),
                        )
                        self._write_json(self._holder_path(holder.token), holder.as_dict())
                        waiter_path.unlink(missing_ok=True)
                        acquired = True
                        waited_ms = int((time.monotonic() - started_wait) * 1000)
                        _LOGGER.info(
                            "lane_scheduler admitted lane=%s wait_ms=%s pid=%s holders=%s/%s",
                            waiter.lane,
                            waited_ms,
                            waiter.pid,
                            total + 1,
                            self.total_slots,
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
            self._holder_path(ticket.token).unlink(missing_ok=True)

    def _ensure_layout(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.waiters_dir.mkdir(parents=True, exist_ok=True)
        self.holders_dir.mkdir(parents=True, exist_ok=True)
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

    def _reclaim_holders(self) -> tuple[int, int]:
        """Reclaim unusable holder files and count the live ones.

        Returns ``(total_holders, non_interactive_holders)``. Must be called with
        the control lock held.
        """
        total = 0
        non_interactive = 0
        for holder_path in sorted(self.holders_dir.glob("*.json")):
            holder = self._reclaim_holder(holder_path)
            if holder is None:
                continue
            total += 1
            if holder.lane == LANE_NON_INTERACTIVE:
                non_interactive += 1
        if self._reclaim_holder(self.legacy_holder_path) is not None:
            # A legacy single-slot holder always occupies a non-interactive slot.
            total += 1
            non_interactive += 1
        return total, non_interactive

    def _reclaim_holder(self, holder_path: Path) -> HolderRecord | None:
        """Return the live holder at ``holder_path``, or remove it and return None."""
        payload = self._read_json(holder_path)
        if payload is None:
            if holder_path.exists():
                _LOGGER.warning("lane_scheduler removed invalid holder state path=%s", holder_path)
                holder_path.unlink(missing_ok=True)
            return None
        try:
            holder = HolderRecord.from_dict(payload)
        except (KeyError, TypeError, ValueError):
            _LOGGER.warning("lane_scheduler removed invalid holder state path=%s", holder_path)
            holder_path.unlink(missing_ok=True)
            return None

        if not _pid_is_alive(holder.pid):
            _LOGGER.warning(
                "lane_scheduler reclaimed dead holder lane=%s pid=%s token=%s",
                holder.lane,
                holder.pid,
                holder.token,
            )
            holder_path.unlink(missing_ok=True)
            return None

        age_seconds = time.time() - holder.claimed_at
        if age_seconds > self.stale_seconds:
            _LOGGER.warning(
                "lane_scheduler reclaimed stale holder lane=%s pid=%s age_seconds=%.3f token=%s",
                holder.lane,
                holder.pid,
                age_seconds,
                holder.token,
            )
            holder_path.unlink(missing_ok=True)
            return None

        return holder

    def _select_next_waiter(
        self,
        *,
        total_holders: int,
        non_interactive_holders: int,
    ) -> WaiterRecord | None:
        """Return the first waiter, in priority order, that the slots admit."""
        waiters: list[WaiterRecord] = []
        for waiter_path in self.waiters_dir.glob("*.json"):
            waiter = self._read_waiter(waiter_path)
            if waiter is not None:
                waiters.append(waiter)
        waiters.sort(
            key=lambda waiter: (
                _LANE_PRIORITY[waiter.lane],
                waiter.created_at,
                waiter.token,
            )
        )
        for waiter in waiters:
            if self._is_admissible(
                lane=waiter.lane,
                total_holders=total_holders,
                non_interactive_holders=non_interactive_holders,
            ):
                return waiter
        return None

    def _is_admissible(
        self,
        *,
        lane: str,
        total_holders: int,
        non_interactive_holders: int,
    ) -> bool:
        if total_holders >= self.total_slots:
            return False
        if lane == LANE_NON_INTERACTIVE:
            return non_interactive_holders < self.non_interactive_slots
        return True

    def _read_waiter(self, waiter_path: Path) -> WaiterRecord | None:
        payload = self._read_json(waiter_path)
        if payload is None:
            return None
        try:
            return WaiterRecord.from_dict(payload)
        except (KeyError, TypeError, ValueError):
            _LOGGER.warning("lane_scheduler removed invalid waiter state path=%s", waiter_path)
            return None

    def _read_json(self, path: Path) -> dict[str, object] | None:
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    def _write_json(self, path: Path, payload: dict[str, object]) -> None:
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    def _waiter_path(self, token: str) -> Path:
        return self.waiters_dir / f"{token}.json"

    def _holder_path(self, token: str) -> Path:
        return self.holders_dir / f"{token}.json"


@contextmanager
def admit_local_inference(
    lane: str,
    *,
    cancel_event: threading.Event | None = None,
) -> Iterator[AdmissionTicket]:
    normalized_lane = normalize_lane(lane)
    if not settings.VOXHELM_LANE_SCHEDULER_ENABLED:
        yield AdmissionTicket(lane=normalized_lane, token="disabled", waited_ms=0)
        return

    scheduler = LaneScheduler(
        root_dir=settings.VOXHELM_LANE_SCHEDULER_DIR,
        stale_seconds=settings.VOXHELM_LANE_SCHEDULER_STALE_SECONDS,
        interactive_slots=settings.VOXHELM_LANE_SCHEDULER_INTERACTIVE_SLOTS,
        non_interactive_slots=settings.VOXHELM_LANE_SCHEDULER_NON_INTERACTIVE_SLOTS,
    )
    ticket = scheduler.acquire(lane=normalized_lane, cancel_event=cancel_event)
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

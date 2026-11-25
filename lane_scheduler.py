from __future__ import annotations

import json
import logging
import math
import os
import stat
import threading
import time
import uuid
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Iterator

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
_MAX_STATE_BYTES = 64 * 1024


class InferenceCancelled(RuntimeError):
    """Raised when a caller cancelled local inference before or while it ran.

    Raised by the scheduler when ``cancel_event`` is set while the caller is still
    queued, and by the STT service at its cancellation checkpoints. It is not a
    backend-availability error, so it never triggers backend fallback.
    """


def _open_regular_file(path: str, flags: int) -> int:
    fd = os.open(path, flags | os.O_NONBLOCK | os.O_NOFOLLOW, 0o666)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError("Lane scheduler file must be regular.")
    except BaseException:
        os.close(fd)
        raise
    return fd


def _finite_timestamp(value: object, field: str) -> float:
    """Parse a record timestamp, rejecting NaN and infinities.

    Record times must remain usable for ordering and operational age warnings.
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
    _holder_handle: IO[str] | None = field(default=None, repr=False, compare=False)


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
    lock_owned: bool = False

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
            lock_owned=payload.get("lock_owned") is True,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "token": self.token,
            "lane": self.lane,
            "pid": self.pid,
            "claimed_at": self.claimed_at,
            "lock_owned": self.lock_owned,
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
        self._warned_old_holders: set[Path] = set()

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
        holder_path = self._holder_path(waiter.token)
        waiter_registration_started = False
        holder_handle: IO[str] | None = None
        ticket: AdmissionTicket | None = None
        started_wait = time.monotonic()

        try:
            with self._control_lock(cancel_event):
                waiter_registration_started = True
                self._write_json(waiter_path, waiter.as_dict())
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    raise InferenceCancelled(
                        f"Local inference was cancelled while waiting for lane '{waiter.lane}'."
                    )
                with self._control_lock(cancel_event):
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
                            lock_owned=True,
                        )
                        waited_ms = int((time.monotonic() - started_wait) * 1000)
                        if cancel_event is not None and cancel_event.is_set():
                            raise InferenceCancelled(
                                "Local inference was cancelled before admission."
                            )
                        holder_handle = holder_path.open("x+", encoding="utf-8")
                        try:
                            fcntl.flock(holder_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                            ticket = AdmissionTicket(
                                lane=waiter.lane,
                                token=waiter.token,
                                waited_ms=waited_ms,
                                _holder_handle=holder_handle,
                            )
                            self._write_json(holder_path, holder.as_dict())
                            waiter_path.unlink(missing_ok=True)
                        except BaseException as exc:
                            # Roll back partial publication under the control lock.
                            self._retire_pending_holder(holder_path, holder_handle, exc)
                            holder_handle = None
                            raise

                if ticket is not None:
                    # Transfer ownership only after the control lock exits.
                    if cancel_event is not None and cancel_event.is_set():
                        raise InferenceCancelled("Local inference was cancelled before admission.")
                    with suppress(Exception):
                        _LOGGER.info(
                            "lane_scheduler admitted lane=%s wait_ms=%s pid=%s holders=%s/%s",
                            waiter.lane,
                            ticket.waited_ms,
                            waiter.pid,
                            total + 1,
                            self.total_slots,
                        )
                    return ticket

                time.sleep(_POLL_INTERVAL_SECONDS)
        except BaseException as exc:
            if holder_handle is not None:
                self._retire_pending_holder(holder_path, holder_handle, exc)
            if waiter_registration_started:
                # The attempt owns its unique waiter even if control lock exit failed.
                try:
                    waiter_path.unlink(missing_ok=True)
                except BaseException:
                    exc.add_note("Could not remove the pending lane waiter.")
            raise

    @staticmethod
    def _retire_pending_holder(path: Path, handle: IO[str], failure: BaseException) -> None:
        # Keep the holder locked while retiring its receipt. Reacquiring the
        # control lock could block after a failed exit.
        try:
            with suppress(FileNotFoundError):
                if os.path.samestat(os.fstat(handle.fileno()), path.stat(follow_symlinks=False)):
                    path.unlink(missing_ok=True)
        except BaseException:
            failure.add_note("Could not remove the pending lane holder.")
        try:
            handle.close()
        except BaseException:
            failure.add_note("Could not close the pending lane holder.")

    def release(self, ticket: AdmissionTicket) -> None:
        path = self._holder_path(ticket.token)
        handle = ticket._holder_handle
        try:
            with self._control_lock():
                with suppress(FileNotFoundError):
                    if handle is not None:
                        named_holder = path.stat(follow_symlinks=False)
                        if handle.closed or not os.path.samestat(
                            os.fstat(handle.fileno()), named_holder
                        ):
                            raise RuntimeError(
                                "Lane scheduler holder receipt changed before release."
                            )
                    path.unlink(missing_ok=True)
        except BaseException as exc:
            if handle is not None:
                try:
                    handle.close()
                except BaseException:
                    exc.add_note("Could not close the lane holder.")
            raise
        else:
            if handle is not None:
                handle.close()

    def _ensure_layout(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.waiters_dir.mkdir(parents=True, exist_ok=True)
        self.holders_dir.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _control_lock(self, cancel_event: threading.Event | None = None) -> Iterator[None]:
        with open(self.control_lock_path, "a+b", opener=_open_regular_file) as handle:
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    raise InferenceCancelled(
                        "Local inference was cancelled while waiting for control."
                    )
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if cancel_event is not None:
                        cancel_event.wait(_POLL_INTERVAL_SECONDS)
                    else:
                        time.sleep(_POLL_INTERVAL_SECONDS)
            try:
                if cancel_event is not None and cancel_event.is_set():
                    raise InferenceCancelled("Local inference was cancelled before admission.")
                try:
                    named_lock = self.control_lock_path.stat(follow_symlinks=False)
                except FileNotFoundError:
                    raise RuntimeError(
                        "Lane scheduler control lock changed while waiting."
                    ) from None
                if not os.path.samestat(os.fstat(handle.fileno()), named_lock):
                    raise RuntimeError("Lane scheduler control lock changed while waiting.")
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _reclaim_holders(self) -> tuple[int, int]:
        """Reclaim unusable holder files and count the live ones.

        Returns ``(total_holders, non_interactive_holders)``. Must be called with
        the control lock held.
        """
        total = 0
        non_interactive = 0
        # glob() suppresses directory I/O errors, which could hide live holders.
        holder_paths = sorted(
            path for path in self.holders_dir.iterdir() if path.name.endswith(".json")
        )
        present_paths = set(holder_paths)
        if self.legacy_holder_path.exists():
            present_paths.add(self.legacy_holder_path)
        self._warned_old_holders.intersection_update(present_paths)
        for holder_path in holder_paths:
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
        with self._open_state(holder_path) as handle:
            if handle is None:
                return None
            payload = self._read_json(handle)
            holder = None
            if payload is not None:
                with suppress(KeyError, TypeError, ValueError):
                    holder = HolderRecord.from_dict(payload)

            if holder is None:
                # Malformed receipts can still belong to a live ticket. Only
                # remove them after locking the same inode we read.
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                _LOGGER.warning("lane_scheduler removed invalid holder state path=%s", holder_path)
                holder_path.unlink(missing_ok=True)
                return None

            if holder.lock_owned:
                # A reused PID cannot keep an abandoned reservation alive.
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    live = True
                else:
                    live = False
            else:
                # Preserve capacity for live pre-lock holders during upgrades.
                live = _pid_is_alive(holder.pid)
            if not live:
                _LOGGER.warning(
                    "lane_scheduler reclaimed dead holder lane=%s pid=%s token=%s",
                    holder.lane,
                    holder.pid,
                    holder.token,
                )
                holder_path.unlink(missing_ok=True)
                return None

        age_seconds = time.time() - holder.claimed_at
        if age_seconds > self.stale_seconds and holder_path not in self._warned_old_holders:
            _LOGGER.warning(
                "lane_scheduler retained old live holder lane=%s pid=%s age_seconds=%.3f token=%s",
                holder.lane,
                holder.pid,
                age_seconds,
                holder.token,
            )
            self._warned_old_holders.add(holder_path)

        return holder

    def _select_next_waiter(
        self,
        *,
        total_holders: int,
        non_interactive_holders: int,
    ) -> WaiterRecord | None:
        """Prune stale waiters and select the best admissible one in one pass."""
        if total_holders >= self.total_slots:
            return None

        batch_slot_available = non_interactive_holders < self.non_interactive_slots
        winner: WaiterRecord | None = None
        best_key: tuple[int, float, str] | None = None
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
                continue
            if waiter.lane == LANE_NON_INTERACTIVE and not batch_slot_available:
                continue
            key = (_LANE_PRIORITY[waiter.lane], waiter.created_at, waiter.token)
            if best_key is None or key < best_key:
                best_key = key
                winner = waiter
        return winner

    def _read_waiter(self, waiter_path: Path) -> WaiterRecord | None:
        with self._open_state(waiter_path) as handle:
            payload = self._read_json(handle) if handle is not None else None
        if payload is None:
            return None
        try:
            return WaiterRecord.from_dict(payload)
        except KeyError, TypeError, ValueError:
            _LOGGER.warning("lane_scheduler removed invalid waiter state path=%s", waiter_path)
            return None

    @contextmanager
    def _open_state(self, path: Path) -> Iterator[IO[bytes] | None]:
        try:
            try:
                handle = open(path, "rb", opener=_open_regular_file)
            except FileNotFoundError:
                yield None
            else:
                with handle:
                    yield handle
        except OSError as exc:
            # Unverifiable state may still represent a live holder.
            _LOGGER.error(
                "lane_scheduler cannot read state path=%s error=%s errno=%s",
                path,
                type(exc).__name__,
                exc.errno,
            )
            raise RuntimeError("Cannot read lane scheduler state; admission refused.") from None

    def _read_json(self, handle: IO[bytes]) -> dict[str, object] | None:
        data = handle.read(_MAX_STATE_BYTES + 1)
        if len(data) > _MAX_STATE_BYTES:
            raise OSError("Lane scheduler state exceeds the size limit.")
        try:
            payload = json.loads(data.decode("utf-8"))
        except ValueError, RecursionError:
            return None
        return payload if isinstance(payload, dict) else None

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

from __future__ import annotations

import functools
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

import pytest

from config.settings import validate_non_negative_int, validate_positive_int
from lane_scheduler import (
    LANE_INTERACTIVE,
    LANE_NON_INTERACTIVE,
    AdmissionTicket,
    InferenceCancelled,
    LaneScheduler,
    admit_local_inference,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEAD_PID = 999999
TIMEOUT = 10.0


def wait_until(predicate: Callable[[], bool], *, timeout: float = TIMEOUT) -> bool:
    """Poll ``predicate`` until it holds, without relying on a fixed sleep."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def waiter_count(scheduler: LaneScheduler) -> int:
    return len(list(scheduler.waiters_dir.glob("*.json")))


def _waiter_count_is(scheduler: LaneScheduler, expected: int) -> bool:
    return waiter_count(scheduler) == expected


def holder_count(scheduler: LaneScheduler) -> int:
    return len(list(scheduler.holders_dir.glob("*.json")))


def write_holder(
    scheduler: LaneScheduler,
    *,
    token: str,
    lane: str = LANE_NON_INTERACTIVE,
    pid: int | None = None,
    age_seconds: float = 0.0,
    path: Path | None = None,
) -> Path:
    scheduler.root_dir.mkdir(parents=True, exist_ok=True)
    scheduler.holders_dir.mkdir(parents=True, exist_ok=True)
    holder_path = path if path is not None else scheduler.holders_dir / f"{token}.json"
    holder_path.write_text(
        json.dumps(
            {
                "token": token,
                "lane": lane,
                "pid": os.getpid() if pid is None else pid,
                "claimed_at": time.time() - age_seconds,
            }
        ),
        encoding="utf-8",
    )
    return holder_path


class OverlapTracker:
    """Records how many holders are active at once, per lane and in total."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active_total = 0
        self.active_non_interactive = 0
        self.max_total = 0
        self.max_non_interactive = 0
        self.admission_order: list[str] = []

    def entered(self, lane: str, name: str) -> None:
        with self._lock:
            self.admission_order.append(name)
            self.active_total += 1
            self.max_total = max(self.max_total, self.active_total)
            if lane == LANE_NON_INTERACTIVE:
                self.active_non_interactive += 1
                self.max_non_interactive = max(
                    self.max_non_interactive, self.active_non_interactive
                )

    def left(self, lane: str) -> None:
        with self._lock:
            self.active_total -= 1
            if lane == LANE_NON_INTERACTIVE:
                self.active_non_interactive -= 1


def test_reserved_interactive_slot_admits_interactive_while_batch_runs(tmp_path: Path) -> None:
    scheduler = LaneScheduler(
        root_dir=tmp_path, stale_seconds=60, interactive_slots=1, non_interactive_slots=1
    )
    holder_ready = threading.Event()
    holder_release = threading.Event()
    interactive_admitted = threading.Event()
    interactive_release = threading.Event()
    second_batch_admitted = threading.Event()

    def holder() -> None:
        ticket = scheduler.acquire(lane=LANE_NON_INTERACTIVE)
        holder_ready.set()
        try:
            holder_release.wait(TIMEOUT)
        finally:
            scheduler.release(ticket)

    def interactive() -> None:
        ticket = scheduler.acquire(lane=LANE_INTERACTIVE)
        interactive_admitted.set()
        try:
            interactive_release.wait(TIMEOUT)
        finally:
            scheduler.release(ticket)

    def second_batch() -> None:
        ticket = scheduler.acquire(lane=LANE_NON_INTERACTIVE)
        second_batch_admitted.set()
        scheduler.release(ticket)

    threads = [
        threading.Thread(target=holder),
        threading.Thread(target=second_batch),
        threading.Thread(target=interactive),
    ]
    threads[0].start()
    assert holder_ready.wait(TIMEOUT)

    threads[1].start()
    assert wait_until(lambda: waiter_count(scheduler) == 1)

    threads[2].start()
    assert interactive_admitted.wait(TIMEOUT)
    assert not second_batch_admitted.is_set()
    assert holder_count(scheduler) == 2

    holder_release.set()
    assert second_batch_admitted.wait(TIMEOUT)

    interactive_release.set()
    for thread in threads:
        thread.join(timeout=TIMEOUT)
    assert holder_count(scheduler) == 0


def test_non_interactive_cap_blocks_a_second_batch_request(tmp_path: Path) -> None:
    scheduler = LaneScheduler(
        root_dir=tmp_path, stale_seconds=60, interactive_slots=1, non_interactive_slots=1
    )
    holder_ready = threading.Event()
    holder_release = threading.Event()
    second_admitted = threading.Event()

    def holder() -> None:
        ticket = scheduler.acquire(lane=LANE_NON_INTERACTIVE)
        holder_ready.set()
        try:
            holder_release.wait(TIMEOUT)
        finally:
            scheduler.release(ticket)

    def second() -> None:
        ticket = scheduler.acquire(lane=LANE_NON_INTERACTIVE)
        second_admitted.set()
        scheduler.release(ticket)

    holder_thread = threading.Thread(target=holder)
    second_thread = threading.Thread(target=second)
    holder_thread.start()
    assert holder_ready.wait(TIMEOUT)
    second_thread.start()
    assert wait_until(lambda: waiter_count(scheduler) == 1)
    assert not second_admitted.is_set()

    holder_release.set()
    assert second_admitted.wait(TIMEOUT)
    holder_thread.join(timeout=TIMEOUT)
    second_thread.join(timeout=TIMEOUT)


def test_interactive_waiter_wins_the_next_free_slot_over_an_older_batch_waiter(
    tmp_path: Path,
) -> None:
    scheduler = LaneScheduler(
        root_dir=tmp_path, stale_seconds=60, interactive_slots=0, non_interactive_slots=1
    )
    holder_ready = threading.Event()
    holder_release = threading.Event()
    tracker = OverlapTracker()

    def holder() -> None:
        ticket = scheduler.acquire(lane=LANE_NON_INTERACTIVE)
        holder_ready.set()
        try:
            holder_release.wait(TIMEOUT)
        finally:
            scheduler.release(ticket)

    def waiter(name: str, lane: str) -> None:
        ticket = scheduler.acquire(lane=lane)
        tracker.entered(lane, name)
        tracker.left(lane)
        scheduler.release(ticket)

    holder_thread = threading.Thread(target=holder)
    holder_thread.start()
    assert holder_ready.wait(TIMEOUT)

    older = threading.Thread(target=waiter, args=("batch", LANE_NON_INTERACTIVE))
    older.start()
    assert wait_until(lambda: waiter_count(scheduler) == 1)
    younger = threading.Thread(target=waiter, args=("interactive", LANE_INTERACTIVE))
    younger.start()
    assert wait_until(lambda: waiter_count(scheduler) == 2)

    holder_release.set()
    holder_thread.join(timeout=TIMEOUT)
    older.join(timeout=TIMEOUT)
    younger.join(timeout=TIMEOUT)

    assert tracker.admission_order == ["interactive", "batch"]


def test_total_bound_is_never_exceeded_under_mixed_load(tmp_path: Path) -> None:
    scheduler = LaneScheduler(
        root_dir=tmp_path, stale_seconds=60, interactive_slots=1, non_interactive_slots=1
    )
    tracker = OverlapTracker()
    lanes = [
        LANE_NON_INTERACTIVE,
        LANE_INTERACTIVE,
        LANE_NON_INTERACTIVE,
        LANE_INTERACTIVE,
        LANE_NON_INTERACTIVE,
        LANE_INTERACTIVE,
    ]

    def worker(index: int, lane: str) -> None:
        ticket = scheduler.acquire(lane=lane)
        tracker.entered(lane, f"{lane}-{index}")
        try:
            time.sleep(0.05)
        finally:
            tracker.left(lane)
            scheduler.release(ticket)

    threads = [
        threading.Thread(target=worker, args=(index, lane)) for index, lane in enumerate(lanes)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=TIMEOUT)

    assert len(tracker.admission_order) == len(lanes)
    assert tracker.max_total <= 2
    assert tracker.max_non_interactive <= 1
    assert holder_count(scheduler) == 0


def test_same_lane_waiters_are_admitted_in_creation_order(tmp_path: Path) -> None:
    scheduler = LaneScheduler(
        root_dir=tmp_path, stale_seconds=60, interactive_slots=1, non_interactive_slots=1
    )
    holder_ready = threading.Event()
    holder_release = threading.Event()
    tracker = OverlapTracker()

    def holder() -> None:
        ticket = scheduler.acquire(lane=LANE_NON_INTERACTIVE)
        holder_ready.set()
        try:
            holder_release.wait(TIMEOUT)
        finally:
            scheduler.release(ticket)

    def waiter(name: str) -> None:
        ticket = scheduler.acquire(lane=LANE_NON_INTERACTIVE)
        tracker.entered(LANE_NON_INTERACTIVE, name)
        tracker.left(LANE_NON_INTERACTIVE)
        scheduler.release(ticket)

    holder_thread = threading.Thread(target=holder)
    holder_thread.start()
    assert holder_ready.wait(TIMEOUT)

    waiters: list[threading.Thread] = []
    for index in range(3):
        thread = threading.Thread(target=waiter, args=(f"batch-{index}",))
        thread.start()
        waiters.append(thread)
        assert wait_until(functools.partial(_waiter_count_is, scheduler, index + 1))

    holder_release.set()
    holder_thread.join(timeout=TIMEOUT)
    for thread in waiters:
        thread.join(timeout=TIMEOUT)

    assert tracker.admission_order == ["batch-0", "batch-1", "batch-2"]
    assert tracker.max_non_interactive == 1


def test_interactive_waiter_is_admitted_first_when_capacity_is_full(tmp_path: Path) -> None:
    scheduler = LaneScheduler(
        root_dir=tmp_path, stale_seconds=60, interactive_slots=1, non_interactive_slots=1
    )
    batch_ready = threading.Event()
    batch_release = threading.Event()
    interactive_ready = threading.Event()
    interactive_release = threading.Event()
    tracker = OverlapTracker()

    def holder(lane: str, ready: threading.Event, release: threading.Event) -> None:
        ticket = scheduler.acquire(lane=lane)
        ready.set()
        try:
            release.wait(TIMEOUT)
        finally:
            scheduler.release(ticket)

    def waiter(name: str, lane: str) -> None:
        ticket = scheduler.acquire(lane=lane)
        tracker.entered(lane, name)
        tracker.left(lane)
        scheduler.release(ticket)

    batch_holder = threading.Thread(
        target=holder, args=(LANE_NON_INTERACTIVE, batch_ready, batch_release)
    )
    batch_holder.start()
    assert batch_ready.wait(TIMEOUT)
    interactive_holder = threading.Thread(
        target=holder, args=(LANE_INTERACTIVE, interactive_ready, interactive_release)
    )
    interactive_holder.start()
    assert interactive_ready.wait(TIMEOUT)
    assert holder_count(scheduler) == 2

    older = threading.Thread(target=waiter, args=("batch", LANE_NON_INTERACTIVE))
    older.start()
    assert wait_until(lambda: waiter_count(scheduler) == 1)
    younger = threading.Thread(target=waiter, args=("interactive", LANE_INTERACTIVE))
    younger.start()
    assert wait_until(lambda: waiter_count(scheduler) == 2)

    batch_release.set()
    older.join(timeout=TIMEOUT)
    younger.join(timeout=TIMEOUT)
    interactive_release.set()
    batch_holder.join(timeout=TIMEOUT)
    interactive_holder.join(timeout=TIMEOUT)

    assert tracker.admission_order == ["interactive", "batch"]


def test_zero_interactive_slots_reproduces_single_slot_serialization(tmp_path: Path) -> None:
    scheduler = LaneScheduler(
        root_dir=tmp_path, stale_seconds=60, interactive_slots=0, non_interactive_slots=1
    )
    tracker = OverlapTracker()
    lanes = [LANE_NON_INTERACTIVE, LANE_INTERACTIVE, LANE_NON_INTERACTIVE, LANE_INTERACTIVE]

    def worker(index: int, lane: str) -> None:
        ticket = scheduler.acquire(lane=lane)
        tracker.entered(lane, f"{lane}-{index}")
        try:
            time.sleep(0.05)
        finally:
            tracker.left(lane)
            scheduler.release(ticket)

    threads = [
        threading.Thread(target=worker, args=(index, lane)) for index, lane in enumerate(lanes)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=TIMEOUT)

    assert len(tracker.admission_order) == len(lanes)
    assert tracker.max_total == 1


def test_stale_holder_is_reclaimed_while_a_live_holder_is_kept(tmp_path: Path) -> None:
    scheduler = LaneScheduler(
        root_dir=tmp_path, stale_seconds=1, interactive_slots=1, non_interactive_slots=1
    )
    stale_path = write_holder(scheduler, token="stale-holder", pid=os.getpid(), age_seconds=5)
    live_path = write_holder(scheduler, token="live-holder")

    started_at = time.monotonic()
    ticket = scheduler.acquire(lane=LANE_INTERACTIVE)
    elapsed = time.monotonic() - started_at
    scheduler.release(ticket)

    assert elapsed < 0.5
    assert not stale_path.exists()
    assert live_path.exists()


def test_dead_holder_pid_is_reclaimed_without_waiting_for_the_stale_timeout(
    tmp_path: Path,
) -> None:
    scheduler = LaneScheduler(
        root_dir=tmp_path, stale_seconds=1800, interactive_slots=1, non_interactive_slots=1
    )
    dead_path = write_holder(scheduler, token="dead-holder", pid=DEAD_PID)

    started_at = time.monotonic()
    ticket = scheduler.acquire(lane=LANE_NON_INTERACTIVE)
    elapsed = time.monotonic() - started_at
    scheduler.release(ticket)

    assert elapsed < 0.5
    assert not dead_path.exists()
    assert holder_count(scheduler) == 0


def test_malformed_holder_files_are_removed(tmp_path: Path) -> None:
    scheduler = LaneScheduler(
        root_dir=tmp_path, stale_seconds=1800, interactive_slots=1, non_interactive_slots=1
    )
    scheduler.holders_dir.mkdir(parents=True, exist_ok=True)
    unparseable = scheduler.holders_dir / "broken.json"
    unparseable.write_text("{not json", encoding="utf-8")
    incomplete = scheduler.holders_dir / "incomplete.json"
    incomplete.write_text(json.dumps({"token": "x", "lane": LANE_INTERACTIVE}), encoding="utf-8")
    bad_lane = scheduler.holders_dir / "bad-lane.json"
    bad_lane.write_text(
        json.dumps({"token": "x", "lane": "nope", "pid": os.getpid(), "claimed_at": time.time()}),
        encoding="utf-8",
    )

    ticket = scheduler.acquire(lane=LANE_NON_INTERACTIVE)
    assert holder_count(scheduler) == 1
    scheduler.release(ticket)

    assert not unparseable.exists()
    assert not incomplete.exists()
    assert not bad_lane.exists()
    assert holder_count(scheduler) == 0


def test_holder_file_with_invalid_utf8_is_reclaimed(tmp_path: Path) -> None:
    scheduler = LaneScheduler(
        root_dir=tmp_path, stale_seconds=1800, interactive_slots=1, non_interactive_slots=1
    )
    scheduler.holders_dir.mkdir(parents=True, exist_ok=True)
    undecodable = scheduler.holders_dir / "undecodable.json"
    undecodable.write_bytes(b'{"token": "\xff\xfe", "lane": "interactive"}')

    started_at = time.monotonic()
    ticket = scheduler.acquire(lane=LANE_NON_INTERACTIVE)
    elapsed = time.monotonic() - started_at
    scheduler.release(ticket)

    assert elapsed < 0.5
    assert not undecodable.exists()
    assert holder_count(scheduler) == 0


def test_holder_with_a_non_finite_claimed_at_is_reclaimed(tmp_path: Path) -> None:
    scheduler = LaneScheduler(
        root_dir=tmp_path, stale_seconds=1800, interactive_slots=1, non_interactive_slots=1
    )
    scheduler.holders_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for token, claimed_at in (("nan-holder", float("nan")), ("inf-holder", float("inf"))):
        path = scheduler.holders_dir / f"{token}.json"
        path.write_text(
            json.dumps(
                {
                    "token": token,
                    "lane": LANE_NON_INTERACTIVE,
                    "pid": os.getpid(),
                    "claimed_at": claimed_at,
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)

    started_at = time.monotonic()
    ticket = scheduler.acquire(lane=LANE_NON_INTERACTIVE)
    elapsed = time.monotonic() - started_at
    scheduler.release(ticket)

    assert elapsed < 0.5
    assert not any(path.exists() for path in paths)


def test_waiter_file_with_invalid_utf8_does_not_block_admission(tmp_path: Path) -> None:
    scheduler = LaneScheduler(
        root_dir=tmp_path, stale_seconds=1800, interactive_slots=1, non_interactive_slots=1
    )
    scheduler.waiters_dir.mkdir(parents=True, exist_ok=True)
    undecodable = scheduler.waiters_dir / "undecodable.json"
    undecodable.write_bytes(b'{"token": "\xff\xfe", "lane": "non-interactive"}')

    started_at = time.monotonic()
    ticket = scheduler.acquire(lane=LANE_NON_INTERACTIVE)
    elapsed = time.monotonic() - started_at
    scheduler.release(ticket)

    assert elapsed < 0.5
    assert not undecodable.exists()
    assert waiter_count(scheduler) == 0


def test_release_never_touches_another_holders_file(tmp_path: Path) -> None:
    scheduler = LaneScheduler(
        root_dir=tmp_path, stale_seconds=1800, interactive_slots=1, non_interactive_slots=1
    )
    ticket = scheduler.acquire(lane=LANE_NON_INTERACTIVE)
    scheduler.holders_dir.joinpath(f"{ticket.token}.json").unlink()
    replacement = write_holder(scheduler, token="replacement-holder")

    scheduler.release(ticket)
    scheduler.release(ticket)

    assert replacement.exists()
    assert holder_count(scheduler) == 1


def test_legacy_holder_json_counts_as_a_non_interactive_holder(tmp_path: Path) -> None:
    scheduler = LaneScheduler(
        root_dir=tmp_path, stale_seconds=1800, interactive_slots=1, non_interactive_slots=1
    )
    legacy = write_holder(scheduler, token="legacy", path=tmp_path / "holder.json")
    assert legacy == scheduler.legacy_holder_path

    interactive_ticket = scheduler.acquire(lane=LANE_INTERACTIVE)
    assert legacy.exists(), "layout creation must not delete a legacy holder"

    batch_admitted = threading.Event()

    def batch() -> None:
        ticket = scheduler.acquire(lane=LANE_NON_INTERACTIVE)
        batch_admitted.set()
        scheduler.release(ticket)

    batch_thread = threading.Thread(target=batch)
    batch_thread.start()
    assert wait_until(lambda: waiter_count(scheduler) == 1)
    assert not batch_admitted.is_set()

    legacy.unlink()
    assert batch_admitted.wait(TIMEOUT)
    batch_thread.join(timeout=TIMEOUT)
    scheduler.release(interactive_ticket)


def test_legacy_holder_json_with_a_dead_pid_is_reclaimed(tmp_path: Path) -> None:
    scheduler = LaneScheduler(
        root_dir=tmp_path, stale_seconds=1800, interactive_slots=1, non_interactive_slots=1
    )
    legacy = write_holder(scheduler, token="legacy", pid=DEAD_PID, path=tmp_path / "holder.json")

    started_at = time.monotonic()
    ticket = scheduler.acquire(lane=LANE_NON_INTERACTIVE)
    elapsed = time.monotonic() - started_at
    scheduler.release(ticket)

    assert elapsed < 0.5
    assert not legacy.exists()


def test_cancel_event_while_queued_raises_and_removes_the_waiter(tmp_path: Path) -> None:
    scheduler = LaneScheduler(
        root_dir=tmp_path, stale_seconds=60, interactive_slots=0, non_interactive_slots=1
    )
    holder_ready = threading.Event()
    holder_release = threading.Event()
    cancel_event = threading.Event()
    cancelled = threading.Event()

    def holder() -> None:
        ticket = scheduler.acquire(lane=LANE_NON_INTERACTIVE)
        holder_ready.set()
        try:
            holder_release.wait(TIMEOUT)
        finally:
            scheduler.release(ticket)

    def waiter() -> None:
        try:
            scheduler.acquire(lane=LANE_INTERACTIVE, cancel_event=cancel_event)
        except InferenceCancelled:
            cancelled.set()

    holder_thread = threading.Thread(target=holder)
    holder_thread.start()
    assert holder_ready.wait(TIMEOUT)
    waiter_thread = threading.Thread(target=waiter)
    waiter_thread.start()
    assert wait_until(lambda: waiter_count(scheduler) == 1)

    cancel_event.set()
    assert cancelled.wait(TIMEOUT)
    waiter_thread.join(timeout=TIMEOUT)
    assert waiter_count(scheduler) == 0

    holder_release.set()
    holder_thread.join(timeout=TIMEOUT)


@pytest.mark.parametrize(
    "interactive_slots,non_interactive_slots",
    [(1, 0), (-1, 1), (0, -1)],
)
def test_constructor_rejects_invalid_slot_counts(
    tmp_path: Path, interactive_slots: int, non_interactive_slots: int
) -> None:
    with pytest.raises(ValueError):
        LaneScheduler(
            root_dir=tmp_path,
            stale_seconds=60,
            interactive_slots=interactive_slots,
            non_interactive_slots=non_interactive_slots,
        )


def test_slot_settings_validators() -> None:
    assert validate_non_negative_int("VOXHELM_LANE_SCHEDULER_INTERACTIVE_SLOTS", 0) == 0
    assert validate_non_negative_int("VOXHELM_LANE_SCHEDULER_INTERACTIVE_SLOTS", 2) == 2
    with pytest.raises(ValueError):
        validate_non_negative_int("VOXHELM_LANE_SCHEDULER_INTERACTIVE_SLOTS", -1)
    assert validate_positive_int("VOXHELM_LANE_SCHEDULER_NON_INTERACTIVE_SLOTS", 1) == 1
    with pytest.raises(ValueError):
        validate_positive_int("VOXHELM_LANE_SCHEDULER_NON_INTERACTIVE_SLOTS", 0)


CHILD_SCRIPT = """
import sys
import time
from pathlib import Path

sys.path.insert(0, sys.argv[1])

from lane_scheduler import LaneScheduler

root = Path(sys.argv[2])
lane = sys.argv[3]
admitted_path = Path(sys.argv[4])
release_arg = sys.argv[5]

scheduler = LaneScheduler(
    root_dir=root,
    stale_seconds=1800,
    interactive_slots=1,
    non_interactive_slots=1,
)
ticket = scheduler.acquire(lane=lane)
try:
    admitted_path.write_text(str(time.time()), encoding="utf-8")
    if release_arg != "-":
        release_path = Path(release_arg)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and not release_path.exists():
            time.sleep(0.01)
finally:
    scheduler.release(ticket)
"""


def waiter_pids(state_dir: Path) -> set[int]:
    """Pids of the processes that currently have a waiter file queued."""
    pids: set[int] = set()
    for path in (state_dir / "waiters").glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except OSError, ValueError:
            continue
        pids.add(int(payload["pid"]))
    return pids


def test_reservation_and_cap_hold_across_real_subprocesses(tmp_path: Path) -> None:
    script_path = tmp_path / "child.py"
    script_path.write_text(CHILD_SCRIPT, encoding="utf-8")
    state_dir = tmp_path / "state"
    batch_admitted = tmp_path / "a.admitted"
    batch_release = tmp_path / "a.release"
    interactive_admitted = tmp_path / "b.admitted"
    interactive_release = tmp_path / "b.release"
    second_batch_admitted = tmp_path / "c.admitted"

    def spawn(lane: str, admitted: Path, release: Path | None) -> subprocess.Popen[bytes]:
        return subprocess.Popen(
            [
                sys.executable,
                str(script_path),
                str(REPO_ROOT),
                str(state_dir),
                lane,
                str(admitted),
                "-" if release is None else str(release),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    processes: list[subprocess.Popen[bytes]] = []
    try:
        batch = spawn(LANE_NON_INTERACTIVE, batch_admitted, batch_release)
        processes.append(batch)
        assert wait_until(batch_admitted.exists), "the batch child was never admitted"

        interactive = spawn(LANE_INTERACTIVE, interactive_admitted, interactive_release)
        processes.append(interactive)
        second_batch = spawn(LANE_NON_INTERACTIVE, second_batch_admitted, None)
        processes.append(second_batch)

        # The reserved interactive slot is honoured across processes: the
        # interactive child runs while the batch child still holds its slot.
        assert wait_until(interactive_admitted.exists), "the interactive child was never admitted"
        assert not batch_release.exists()
        assert batch_admitted.exists()

        # The non-interactive cap is honoured across processes: the second batch
        # child stays queued for as long as the first one holds the only slot.
        assert wait_until(lambda: second_batch.pid in waiter_pids(state_dir))
        assert not wait_until(second_batch_admitted.exists, timeout=0.5)
        assert second_batch.pid in waiter_pids(state_dir)
        assert not batch_release.exists()

        batch_release.touch()
        assert wait_until(second_batch_admitted.exists), "the queued batch child never ran"
        interactive_release.touch()

        for process in processes:
            _, stderr = process.communicate(timeout=60)
            assert process.returncode == 0, stderr.decode("utf-8", "replace")
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=60)

    assert not list((state_dir / "holders").glob("*.json"))


def test_admit_local_inference_is_a_no_op_when_disabled(settings, tmp_path: Path) -> None:
    scheduler_dir = tmp_path / "lane-scheduler"
    settings.VOXHELM_LANE_SCHEDULER_ENABLED = False
    settings.VOXHELM_LANE_SCHEDULER_DIR = scheduler_dir

    with admit_local_inference(LANE_INTERACTIVE) as ticket:
        assert isinstance(ticket, AdmissionTicket)
        assert ticket.token == "disabled"
        assert ticket.lane == LANE_INTERACTIVE

    assert not scheduler_dir.exists()

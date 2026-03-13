from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from lane_scheduler import LANE_INTERACTIVE, LANE_NON_INTERACTIVE, LaneScheduler


def test_lane_scheduler_prioritizes_interactive_waiters(tmp_path: Path) -> None:
    scheduler = LaneScheduler(root_dir=tmp_path, stale_seconds=60)
    ready = threading.Event()
    admission_order: list[str] = []

    def holder() -> None:
        ticket = scheduler.acquire(lane=LANE_NON_INTERACTIVE)
        admission_order.append("holder")
        ready.set()
        try:
            time.sleep(0.15)
        finally:
            scheduler.release(ticket)

    def waiter(name: str, lane: str) -> None:
        ticket = scheduler.acquire(lane=lane)
        admission_order.append(name)
        try:
            time.sleep(0.01)
        finally:
            scheduler.release(ticket)

    holder_thread = threading.Thread(target=holder)
    holder_thread.start()
    assert ready.wait(timeout=1.0)

    batch_thread = threading.Thread(
        target=waiter,
        args=("non_interactive_waiter", LANE_NON_INTERACTIVE),
    )
    interactive_thread = threading.Thread(
        target=waiter,
        args=("interactive_waiter", LANE_INTERACTIVE),
    )
    batch_thread.start()
    time.sleep(0.02)
    interactive_thread.start()

    holder_thread.join()
    batch_thread.join()
    interactive_thread.join()

    assert admission_order == ["holder", "interactive_waiter", "non_interactive_waiter"]


def test_lane_scheduler_reclaims_stale_holder(tmp_path: Path) -> None:
    scheduler = LaneScheduler(root_dir=tmp_path, stale_seconds=1)
    scheduler.root_dir.mkdir(parents=True, exist_ok=True)
    scheduler.waiters_dir.mkdir(parents=True, exist_ok=True)
    scheduler.holder_path.write_text(
        json.dumps(
            {
                "token": "stale-holder",
                "lane": LANE_NON_INTERACTIVE,
                "pid": 999999,
                "claimed_at": time.time() - 5,
            }
        ),
        encoding="utf-8",
    )

    started_at = time.monotonic()
    ticket = scheduler.acquire(lane=LANE_NON_INTERACTIVE)
    elapsed = time.monotonic() - started_at
    scheduler.release(ticket)

    assert elapsed < 0.5
    assert not scheduler.holder_path.exists()


def test_lane_scheduler_reclaims_dead_holder_pid_without_waiting_for_stale_timeout(
    tmp_path: Path,
) -> None:
    scheduler = LaneScheduler(root_dir=tmp_path, stale_seconds=1800)
    scheduler.root_dir.mkdir(parents=True, exist_ok=True)
    scheduler.waiters_dir.mkdir(parents=True, exist_ok=True)
    scheduler.holder_path.write_text(
        json.dumps(
            {
                "token": "dead-holder",
                "lane": LANE_NON_INTERACTIVE,
                "pid": 999999,
                "claimed_at": time.time(),
            }
        ),
        encoding="utf-8",
    )

    started_at = time.monotonic()
    ticket = scheduler.acquire(lane=LANE_INTERACTIVE)
    elapsed = time.monotonic() - started_at
    scheduler.release(ticket)

    assert elapsed < 0.5
    assert not scheduler.holder_path.exists()

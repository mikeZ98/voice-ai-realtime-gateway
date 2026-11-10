"""Cancellation never blocks: the reaper's deadline holds whatever the tasks do.

Barge-in cancels every turn task synchronously, then reaps them with a bounded join. A
task that swallows ``CancelledError`` must be reported as a straggler, never awaited past
the deadline, and must not stop cooperative tasks from being reaped.
"""

from __future__ import annotations

import asyncio
import contextlib
import time

from hypothesis import given, settings
from hypothesis import strategies as st

from voice_ai_realtime_gateway.core.supervisor import JoinReport, TaskSupervisor

DEADLINE_S = 0.02
SLACK_S = 0.5  # scheduler noise on a loaded CI runner; still 25x the deadline


@settings(max_examples=60, deadline=None)
@given(
    cooperative=st.integers(min_value=0, max_value=12),
    stubborn=st.integers(min_value=0, max_value=4),
    yields_before_start=st.integers(min_value=0, max_value=3),
)
def test_cancel_is_synchronous_and_join_respects_its_deadline(
    cooperative: int, stubborn: int, yields_before_start: int
) -> None:
    errors: list[dict[str, object]] = []

    async def scenario() -> None:
        asyncio.get_running_loop().set_exception_handler(lambda _l, ctx: errors.append(dict(ctx)))
        release = asyncio.Event()
        failures: list[BaseException] = []
        sup = TaskSupervisor(on_failure=lambda _task, exc: failures.append(exc))

        async def cooperative_task() -> None:
            await asyncio.Event().wait()

        async def stubborn_task() -> None:
            while not release.is_set():
                with contextlib.suppress(asyncio.CancelledError):
                    await release.wait()

        for index in range(cooperative):
            _ = sup.spawn(cooperative_task(), name=f"cooperative-{index}")
        for index in range(stubborn):
            _ = sup.spawn(stubborn_task(), name=f"stubborn-{index}")
        for _ in range(yields_before_start):
            await asyncio.sleep(0)

        signalled = sup.cancel()  # synchronous: no await between signal and return
        assert signalled == cooperative + stubborn

        started = time.monotonic()
        report = await sup.join(DEADLINE_S)
        elapsed = time.monotonic() - started
        assert elapsed < DEADLINE_S + SLACK_S, "join never waits for a stubborn task"
        # A task cancelled before its first step never runs, so it cannot swallow the
        # cancellation: only tasks that already started can become stragglers.
        stragglers = stubborn if yields_before_start else 0
        assert report == JoinReport(
            finished=cooperative + stubborn - stragglers, stragglers=stragglers
        )

        release.set()  # let the stragglers end so nothing leaks past the scenario
        assert await sup.join(1.0) == JoinReport(finished=stragglers, stragglers=0)
        assert failures == [], "cancellation is not a failure"

    asyncio.run(scenario())
    assert errors == [], f"loop exception handler received: {errors}"

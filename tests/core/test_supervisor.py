from __future__ import annotations

import asyncio
import contextlib
import inspect

import pytest

from voice_ai_realtime_gateway.core.supervisor import (
    JoinReport,
    SupervisorClosedError,
    TaskSupervisor,
)

pytestmark = pytest.mark.usefixtures("strict_loop")

type Failures = list[tuple[str, BaseException]]


def supervisor() -> tuple[TaskSupervisor, Failures]:
    failures: Failures = []
    sup = TaskSupervisor(on_failure=lambda task, exc: failures.append((task.get_name(), exc)))
    return sup, failures


async def forever() -> None:
    await asyncio.Event().wait()


async def test_tasks_are_tracked_until_completion() -> None:
    sup, failures = supervisor()
    gate = asyncio.Event()

    async def work() -> None:
        await gate.wait()

    task = sup.spawn(work(), name="work")
    assert task.get_name() == "work"
    assert len(sup) == 1
    gate.set()
    assert await sup.join(1.0) == JoinReport(finished=1, stragglers=0)
    assert len(sup) == 0
    assert failures == []


async def test_join_on_empty_supervisor_is_immediate() -> None:
    sup, _ = supervisor()
    assert await sup.join(0.001) == JoinReport(finished=0, stragglers=0)


async def test_cancel_is_synchronous_and_absorbed() -> None:
    sup, failures = supervisor()
    for index in range(5):
        _ = sup.spawn(forever(), name=f"t{index}")
    await asyncio.sleep(0)
    assert sup.cancel() == 5
    assert sup.cancel() == 5, "re-signalling pending tasks is idempotent"
    report = await sup.join(1.0)
    assert report == JoinReport(finished=5, stragglers=0)
    assert failures == [], "cancellation is not a failure"


async def test_failures_are_retrieved_and_routed() -> None:
    sup, failures = supervisor()

    async def boom() -> None:
        msg = "kaput"
        raise RuntimeError(msg)

    _ = sup.spawn(boom(), name="boom")
    await sup.join(1.0)
    ((name, error),) = failures
    assert name == "boom"
    assert isinstance(error, RuntimeError)


async def test_clean_completion_is_not_a_failure() -> None:
    sup, failures = supervisor()

    async def ok() -> None:
        return None

    _ = sup.spawn(ok(), name="ok")
    await sup.join(1.0)
    assert failures == []


async def test_stragglers_that_ignore_cancellation_are_reported_not_awaited_forever() -> None:
    sup, _ = supervisor()
    release = asyncio.Event()

    async def stubborn() -> None:
        while not release.is_set():
            with contextlib.suppress(asyncio.CancelledError):
                await release.wait()

    _ = sup.spawn(stubborn(), name="stubborn")
    await asyncio.sleep(0)
    sup.cancel()
    assert await sup.join(0.01) == JoinReport(finished=0, stragglers=1)
    release.set()
    assert await sup.join(1.0) == JoinReport(finished=1, stragglers=0)


async def test_cancelling_the_joiner_does_not_cancel_supervised_tasks() -> None:
    sup, _ = supervisor()
    gate = asyncio.Event()

    async def work() -> None:
        await gate.wait()

    task = sup.spawn(work(), name="work")
    joiner = asyncio.create_task(sup.join(10.0))
    await asyncio.sleep(0)
    joiner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await joiner
    assert not task.done(), "supervised task survives its joiner's cancellation"
    gate.set()
    await sup.join(1.0)


async def test_closed_supervisor_refuses_and_closes_the_coroutine() -> None:
    sup, _ = supervisor()
    sup.close()
    assert sup.closed
    coro = forever()
    with pytest.raises(SupervisorClosedError, match="late"):
        _ = sup.spawn(coro, name="late")
    assert inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED, "no 'never awaited' warning"

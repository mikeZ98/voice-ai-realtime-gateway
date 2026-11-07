"""Shared fixtures: telemetry doubles and a strict event-loop guard."""

from __future__ import annotations

import asyncio
import gc
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from hypothesis import HealthCheck, settings
from hypothesis.configuration import set_hypothesis_home_dir
from hypothesis.database import DirectoryBasedExampleDatabase

from voice_ai_realtime_gateway.telemetry.ids import TraceContext
from voice_ai_realtime_gateway.telemetry.sink import FlightRecorder

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

# --------------------------------------------------------------------------- hygiene
# Hypothesis state lives under ./.cache/ like every other tool cache.
_HYPOTHESIS_HOME = Path(__file__).resolve().parent.parent / ".cache" / "hypothesis"
set_hypothesis_home_dir(_HYPOTHESIS_HOME)
settings.register_profile(
    "gateway",
    database=DirectoryBasedExampleDatabase(_HYPOTHESIS_HOME / "examples"),
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("gateway")


@pytest.fixture
def recorder() -> FlightRecorder:
    return FlightRecorder(capacity=4096)


@pytest.fixture
def trace() -> TraceContext:
    return TraceContext.new()


@pytest.fixture
async def strict_loop() -> AsyncIterator[list[dict[str, object]]]:
    """Fail the test on leaked tasks or anything routed to the loop exception handler.

    Catches "Task exception was never retrieved", "Task was destroyed but it is
    pending!", exceptions raised inside done-callbacks, and unawaited-coroutine leaks.
    """
    loop = asyncio.get_running_loop()
    errors: list[dict[str, object]] = []
    loop.set_exception_handler(lambda _loop, context: errors.append(dict(context)))
    yield errors
    for _ in range(3):
        await asyncio.sleep(0)
    gc.collect()
    current = asyncio.current_task()
    leaked = [task for task in asyncio.all_tasks() if task is not current]
    loop.set_exception_handler(None)
    for task in leaked:
        task.cancel()
    assert not leaked, f"leaked tasks: {leaked}"
    assert not errors, f"loop exception handler received: {errors}"

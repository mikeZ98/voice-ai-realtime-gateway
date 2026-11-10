"""Shared fixtures: deterministic clocks, frame factories and a strict event-loop guard."""

from __future__ import annotations

import asyncio
import gc
import os
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import pytest
from hypothesis import HealthCheck, settings
from hypothesis.configuration import set_hypothesis_home_dir
from hypothesis.database import DirectoryBasedExampleDatabase

from voice_ai_realtime_gateway.core.frames import AudioCodec, AudioFrame
from voice_ai_realtime_gateway.telemetry.ids import TraceContext
from voice_ai_realtime_gateway.telemetry.sink import FlightRecorder

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

# --------------------------------------------------------------------------- hygiene
# Hypothesis state lives under ./.cache/ like every other tool cache. The mutation
# harness points each worker at a private home and a fast, derandomised profile so
# mutants never pollute the shared example database.
_HYPOTHESIS_HOME = Path(
    os.environ.get("VOICE_GATEWAY_HYPOTHESIS_HOME")
    or Path(__file__).resolve().parent.parent / ".cache" / "hypothesis"
)
set_hypothesis_home_dir(_HYPOTHESIS_HOME)
settings.register_profile(
    "gateway",
    database=DirectoryBasedExampleDatabase(_HYPOTHESIS_HOME / "examples"),
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "mutation",
    database=None,
    derandomize=True,
    max_examples=40,
    deadline=None,
    suppress_health_check=list(HealthCheck),
)
settings.load_profile(os.environ.get("VOICE_GATEWAY_HYPOTHESIS_PROFILE", "gateway"))


def pytest_configure(config: pytest.Config) -> None:
    """Create the parent of ``--basetemp`` (pytest only creates the leaf).

    ``--basetemp=.cache/pytest/basetemp`` keeps ``tmp_path`` inside the project; without
    this hook it fails on a fresh clone or in a mutation workspace, where
    ``.cache/pytest/`` does not exist yet.
    """
    basetemp = config.option.basetemp
    if basetemp:
        Path(basetemp).parent.mkdir(parents=True, exist_ok=True)

FRAME_NS = 20_000_000
"""20 ms media frames."""


class FakeClock:
    """Manually advanced monotonic nanosecond clock."""

    def __init__(self, start: int = 1_000_000_000, step: int = 0) -> None:
        self.now = start
        self.step = step

    def __call__(self) -> int:
        current = self.now
        self.now += self.step
        return current


class FrameFactory(Protocol):
    def __call__(
        self,
        nbytes: int = ...,
        *,
        trace: TraceContext | None = ...,
        sequence: int | None = ...,
        origin_ns: int = ...,
    ) -> AudioFrame: ...


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def recorder() -> FlightRecorder:
    return FlightRecorder(capacity=4096)


@pytest.fixture
def trace() -> TraceContext:
    return TraceContext.new()


@pytest.fixture
def make_frame(trace: TraceContext) -> FrameFactory:
    counter = iter(range(1 << 62))

    def factory(
        nbytes: int = 640,
        *,
        trace: TraceContext | None = trace,
        sequence: int | None = None,
        origin_ns: int = 0,
    ) -> AudioFrame:
        return AudioFrame.wrap(
            bytes(nbytes),
            codec=AudioCodec.PCM16,
            sample_rate_hz=16_000,
            channels=1,
            duration_ns=FRAME_NS,
            sequence=next(counter) if sequence is None else sequence,
            origin_ns=origin_ns,
            trace=TraceContext.new() if trace is None else trace,
        )

    return factory


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

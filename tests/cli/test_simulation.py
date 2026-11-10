"""Offline simulation: the production stack under scripted callers, drain and barge-in."""

from __future__ import annotations

import asyncio
import math
import signal
from typing import TYPE_CHECKING

import pytest

from duplex_support import until
from voice_ai_realtime_gateway.cli import simulation as sim
from voice_ai_realtime_gateway.cli.simulation import (
    PLAYOUT_CAPACITY_FRAMES,
    PLAYOUT_PREBUFFER_FRAMES,
    RING_CAPACITY_BYTES_PER_CALL,
    LiveSnapshot,
    PlayoutBuffer,
    Simulation,
    SimulationConfig,
    TelemetryCollector,
    percentiles,
)
from voice_ai_realtime_gateway.deploy.lifecycle import DrainPhase
from voice_ai_realtime_gateway.production.server import CallResult, GatewayServer
from voice_ai_realtime_gateway.telemetry.events import (
    BargeInHandled,
    BufferOverflow,
    EgressDropped,
    ResponseTimeline,
    VadSpeechStarted,
)
from voice_ai_realtime_gateway.transport.control import PlaybackFlush, Pong, encode_control
from voice_ai_realtime_gateway.transport.memory import MemoryWebSocket

if TYPE_CHECKING:
    from voice_ai_realtime_gateway.telemetry.ids import TraceContext


def fast(**overrides: object) -> SimulationConfig:
    """A run that does not wait on the real 20 ms cadence."""
    return SimulationConfig(**{"real_time": False, "ramp_s": 0.0, **overrides})  # type: ignore[arg-type]


# ----------------------------------------------------------------- configuration


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"calls": 0}, "must be positive"),
        ({"turns": 0}, "must be positive"),
        ({"speech_frames": 0}, "must be positive"),
        ({"frame_ms": 0}, "must be positive"),
        ({"barge_in_every": -1}, "barge_in_every"),
        ({"ramp_s": -0.1}, "ramp_s"),
        ({"ramp_s": math.inf}, "ramp_s"),
        ({"drain_timeout_s": 0.0}, "drain_timeout_s"),
        ({"drain_timeout_s": math.nan}, "drain_timeout_s"),
        ({"answer_timeout_s": math.inf}, "answer_timeout_s"),
        ({"reply": "  "}, "reply"),
    ],
)
def test_config_is_validated(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SimulationConfig(**overrides)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("every", "expected"), [(0, [False] * 4), (1, [True] * 4), (2, [False, True, False, True])]
)
def test_interrupt_schedule(every: int, expected: list[bool]) -> None:
    config = SimulationConfig(barge_in_every=every)
    assert [config.interrupts(turn) for turn in range(4)] == expected


def test_reply_frames_follow_the_tone_tts_pace() -> None:
    assert SimulationConfig(reply="x" * 10, frame_ms=20).reply_frames == 4  # 80 ms -> 4 frames
    assert SimulationConfig(reply="x", frame_ms=20).reply_frames == 1
    assert RING_CAPACITY_BYTES_PER_CALL == 768 * 1024


def test_percentiles_are_nearest_rank_in_milliseconds() -> None:
    assert percentiles([]) == {"count": 0, "p50": None, "p99": None, "max": None}
    values = [i * 1_000_000 for i in range(1, 101)]
    assert percentiles(values) == {"count": 100, "p50": 50.0, "p99": 99.0, "max": 100.0}
    assert percentiles([1_500_000]) == {"count": 1, "p50": 1.5, "p99": 1.5, "max": 1.5}


# ----------------------------------------------------------------- telemetry


def timeline(trace: TraceContext, *, token: int | None, egress: int | None) -> ResponseTimeline:
    return ResponseTimeline(
        trace=trace,
        at_ns=0,
        user_span_id=trace.span_id,
        speech_end_ns=1_000,
        transcript_ns=None,
        first_token_ns=token,
        first_audio_ns=None,
        first_egress_ns=egress,
        completed=True,
        generated_chars=1,
        spoken_chars=1,
    )


def test_collector_keeps_only_reported_measurements(trace: TraceContext) -> None:
    collector = TelemetryCollector()
    collector.emit(timeline(trace, token=4_000, egress=9_000))
    collector.emit(timeline(trace, token=None, egress=None))
    for within in (True, False):
        collector.emit(
            BargeInHandled(
                trace=trace,
                at_ns=0,
                detect_to_draining_ns=70,
                budget_ns=100,
                within_budget=within,
                cancelled_tasks=1,
                dropped_frames=2,
                stragglers=0,
            )
        )
    collector.emit(EgressDropped(trace=trace, at_ns=0, sequence=1))
    collector.emit(BufferOverflow(trace=trace, at_ns=0, evicted_frames=1, evicted_bytes=640))
    collector.emit(VadSpeechStarted(trace=trace, at_ns=0))  # ignored
    assert collector.ttft_ns == [3_000]
    assert collector.first_audio_ns == [8_000]
    assert collector.barge_in_ns == [70, 70]
    assert collector.barge_ins_over_budget == 1
    assert (collector.egress_dropped, collector.ring_overflows) == (1, 1)


def snapshot(**overrides: object) -> LiveSnapshot:
    fields: dict[str, object] = {
        "phase": "serving",
        "elapsed_s": 1.0,
        "calls_total": 2,
        "calls_active": 1,
        "calls_completed": 1,
        "calls_refused": 0,
        "buffered_bytes": 0,
        "capacity_bytes": 0,
        "ttft_p50_ms": None,
        "first_audio_p50_ms": None,
        "jitter_p50_ms": None,
        "jitter_p99_ms": None,
        "barge_ins": 0,
        "barge_in_p99_ms": None,
        "frames_out": 0,
        "dropped": 0,
        "close_codes": {},
    }
    fields.update(overrides)
    return LiveSnapshot(**fields)  # type: ignore[arg-type]


def test_snapshot_ratios_are_safe_when_idle() -> None:
    idle = snapshot()
    assert (idle.drop_rate, idle.buffer_fill) == (0.0, 0.0)
    busy = snapshot(frames_out=3, dropped=1, buffered_bytes=256, capacity_bytes=1024)
    assert (busy.drop_rate, busy.buffer_fill) == (0.25, 0.25)


# ----------------------------------------------------------------- runs


async def test_fast_run_is_healthy_and_exercises_barge_in() -> None:
    simulation = Simulation(fast(calls=3, turns=3, barge_in_every=2))
    await simulation.run()
    summary = simulation.summary()
    assert simulation.healthy
    assert summary["calls"] == {"started": 3, "completed": 3, "refused": 0, "failed": 0}
    assert summary["close_codes"] == {"1000": 3}
    assert summary["turns"] == {"answered": 9, "unanswered": 0, "interrupted": 3, "flushed": 3}
    barge = summary["barge_in_ms"]
    assert isinstance(barge, dict)
    assert (barge["count"], barge["over_budget"]) == (3, 0)
    ttft = summary["ttft_ms"]
    assert isinstance(ttft, dict)
    assert ttft["count"] == 12, "9 answers plus 3 responses cancelled by barge-in"
    egress = summary["egress"]
    assert isinstance(egress, dict)
    assert (egress["dropped"], egress["drop_rate"]) == (0, 0.0)
    assert summary["drain"] is None
    assert summary["playout"] is None, "no playout model without real-time pacing"
    assert summary["signals"] == []
    assert simulation.drain_phase is DrainPhase.SERVING
    assert simulation.collector.ttft_ns


async def test_stop_before_any_call_refuses_everyone_with_1013() -> None:
    simulation = Simulation(fast(calls=2, turns=1))
    simulation.request_stop(signal.SIGTERM)
    simulation.request_stop(signal.SIGINT)  # recorded, the drain is already running
    await simulation.run()
    summary = simulation.summary()
    assert summary["signals"] == [signal.SIGTERM, signal.SIGINT]
    assert summary["close_codes"] == {"1013": 2}
    assert summary["calls"] == {"started": 2, "completed": 0, "refused": 2, "failed": 0}
    assert summary["drain"] == {"waited_s": 0.0, "forced": 0}
    assert not simulation.healthy
    assert simulation.drain_phase is DrainPhase.DRAINED


async def test_drain_deadline_closes_live_calls_with_1001(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def keep_streaming_until_closed(self: sim._Client) -> None:
        while not self._closed.is_set():  # a caller who never stops talking
            await self._send(0)

    monkeypatch.setattr(sim._Client, "_await_answer", keep_streaming_until_closed)
    simulation = Simulation(fast(calls=2, turns=1, barge_in_every=0, drain_timeout_s=0.01))
    running = asyncio.create_task(simulation.run())
    await until(lambda: simulation.snapshot().calls_active == 2)
    live = simulation.snapshot()
    assert live.capacity_bytes == 2 * RING_CAPACITY_BYTES_PER_CALL
    simulation.request_stop(signal.SIGTERM)
    await running
    summary = simulation.summary()
    assert summary["close_codes"] == {"1001": 2}, "still mid-conversation at the deadline"
    drain = summary["drain"]
    assert isinstance(drain, dict)
    assert drain["forced"] == 2
    assert summary["calls"] == {"started": 2, "completed": 0, "refused": 2, "failed": 0}


async def test_unanswered_turns_are_reported() -> None:
    simulation = Simulation(fast(calls=1, turns=2, barge_in_every=1, answer_timeout_s=1e-9))
    await simulation.run()
    turns = simulation.summary()["turns"]
    assert turns == {"answered": 0, "unanswered": 2, "interrupted": 0, "flushed": 0}
    assert not simulation.healthy


async def test_a_failing_client_is_counted_not_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    async def explode(self: object) -> None:
        del self
        msg = "client bug"
        raise RuntimeError(msg)

    monkeypatch.setattr(sim._Client, "converse", explode)
    simulation = Simulation(fast(calls=2, turns=1))
    await simulation.run()
    assert simulation.summary()["calls"] == {
        "started": 2,
        "completed": 0,
        "refused": 0,
        "failed": 2,
    }


async def test_real_time_run_measures_jitter() -> None:
    simulation = Simulation(
        SimulationConfig(
            calls=1, turns=1, speech_frames=3, barge_in_every=0, ramp_s=0.0, reply="Hi there."
        )
    )
    await simulation.run()
    assert simulation.healthy
    jitter = simulation.summary()["jitter_ms"]
    assert isinstance(jitter, dict)
    assert jitter["count"] == SimulationConfig(reply="Hi there.").reply_frames - 1
    assert simulation.snapshot().jitter_p99_ms is not None
    playout = simulation.summary()["playout"]
    assert isinstance(playout, dict)
    assert playout["frames"] == SimulationConfig(reply="Hi there.").reply_frames
    assert (playout["prebuffer_ms"], playout["capacity_ms"]) == (40, 200)
    assert playout["overflows"] == 0


async def test_client_ignores_other_control_and_times_unprompted_flushes(
    trace: TraceContext,
) -> None:
    client, _edge = MemoryWebSocket.pair()
    stats = sim._ClientStats()
    script = sim._Client(SimulationConfig(), client, trace, stats, asyncio.Event())
    script._on_control(encode_control(Pong(nonce=7)))
    assert stats.flushes == 0
    script._received = 5
    script._on_control(encode_control(PlaybackFlush()))
    assert (stats.flushes, stats.flush_ns, script._received) == (1, [], 0)
    await script.stop_listening()  # never started: nothing to cancel


async def test_a_call_without_a_close_code_is_neither_counted_nor_completed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def vanish(self: GatewayServer, connection: object, headers: object) -> CallResult:
        del self, connection, headers
        return CallResult(
            admitted=True, close_code=None, close_reason="", tenant=None, outcome=None
        )

    monkeypatch.setattr(GatewayServer, "handle", vanish)
    simulation = Simulation(fast(calls=1, turns=1, barge_in_every=0, answer_timeout_s=1e-9))
    await simulation.run()
    summary = simulation.summary()
    assert summary["close_codes"] == {}
    assert summary["calls"] == {"started": 1, "completed": 0, "refused": 0, "failed": 0}


# ----------------------------------------------------------------- playout model

F = 20_000_000  # one 20 ms frame


def test_playout_buffer_is_validated() -> None:
    for bad in (
        {"frame_ns": 0},
        {"frame_ns": F, "capacity_frames": 0},
        {"frame_ns": F, "prebuffer_frames": -1},
    ):
        with pytest.raises(ValueError, match="positive"):
            PlayoutBuffer(**bad)
    assert (PLAYOUT_PREBUFFER_FRAMES, PLAYOUT_CAPACITY_FRAMES) == (2, 10)


def test_on_cadence_audio_never_underflows_or_overflows() -> None:
    buffer = PlayoutBuffer(F)
    assert [buffer.arrive(k * F) for k in range(50)] == [(False, False)] * 50


def test_a_frame_exactly_at_its_deadline_is_on_time_one_ns_later_underflows() -> None:
    buffer = PlayoutBuffer(F, prebuffer_frames=0)
    assert buffer.arrive(0) == (False, False)
    assert buffer.arrive(F) == (False, False), "due at exactly F"
    assert buffer.arrive(2 * F + 1) == (True, False), "1 ns past its deadline"


def test_playout_resumes_after_an_underflow() -> None:
    buffer = PlayoutBuffer(F)  # 40 ms pre-buffer
    assert buffer.arrive(0) == (False, False)
    assert buffer.arrive(F + 100 * F) == (True, False), "2 s late: the speaker ran dry"
    # Later frames keep the cadence relative to the late one: no cascade of underflows.
    assert [buffer.arrive(101 * F + k * F) for k in range(1, 6)] == [(False, False)] * 5


def test_a_burst_beyond_capacity_overflows() -> None:
    buffer = PlayoutBuffer(F, prebuffer_frames=2, capacity_frames=10)
    results = [buffer.arrive(0) for _ in range(12)]  # 12 frames before playout starts
    assert [overflow for _, overflow in results] == [False] * 10 + [True] * 2
    assert not any(underflow for underflow, _ in results)


def test_frames_played_at_playout_start_free_capacity() -> None:
    buffer = PlayoutBuffer(F, prebuffer_frames=1, capacity_frames=1)
    assert buffer.arrive(0) == (False, False)
    assert buffer.arrive(F) == (False, False), "frame 0 starts playing at exactly F"
    assert buffer.arrive(F) == (False, True), "two unplayed frames exceed a capacity of 1"


def test_reset_starts_a_new_answer() -> None:
    buffer = PlayoutBuffer(F)
    buffer.arrive(0)
    buffer.reset()
    assert buffer.arrive(500 * F) == (False, False), "not late: it is a new answer's first frame"

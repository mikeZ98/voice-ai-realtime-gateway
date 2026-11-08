"""Synthetic concurrency benchmark: N full-duplex calls on one event loop.

Every call runs through the production stack (:class:`GatewayServer`): JWT admission,
session registry, Prometheus metrics, the WebSocket transport (framing, sequencing,
heartbeat, rate limits), energy VAD, ingress rings, speech capture, the
ASR -> LLM -> sentence chunker -> TTS pipeline and the playback ledger. Only the
network and the providers are simulated: each call is a :class:`MemoryWebSocket`
pair, ASR/LLM/TTS are the deterministic doubles, and the TTS emits audio in real time
(one 20 ms frame per 20 ms), as streaming synthesizers do.

Each simulated client streams microphone audio at the real 20 ms cadence on an
absolute schedule (no drift): a spoken utterance, then silence while the agent answers,
for ``--turns`` turns, then hangs up. Starts are spread over ``--ramp`` seconds.

Measured:

* **glass-to-glass**: end of user speech (VAD offset) to the first agent audio packet
  leaving the gateway (the ``voice_gateway_glass_to_glass_seconds`` definition).
* **client turn latency**: last spoken frame sent to first agent audio received, as the
  client sees it (includes the VAD hangover, i.e. endpointing).
* **frame jitter**: deviation of agent-audio inter-arrival times from the 20 ms frame
  cadence, at the client.
* **event-loop lag**: a probe task sleeps 1 ms repeatedly; lag is the overshoot.
* **churn memory**: waves of calls run to completion under ``tracemalloc``; live memory
  attributed to the gateway after each wave (post-GC; the harness's own sample arrays
  excluded) must not grow with the number of calls served.

Usage::

    uv run python -m benchmarks.bench_concurrency                 # 1,000 calls
    uv run python -m benchmarks.bench_concurrency --sessions 200 --churn-waves 3

The JSON report goes to ``.cache/benchmarks/concurrency_report.json``. Exit status is
0 when every threshold holds, 1 otherwise; the report is written either way.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import functools
import gc
import json
import math
import multiprocessing
import os
import platform
import resource
import sys
import time
import tracemalloc
from array import array
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final

from voice_ai_realtime_gateway.core.frames import AudioCodec, AudioFrame
from voice_ai_realtime_gateway.inference.agent import VoiceAgentPipeline
from voice_ai_realtime_gateway.inference.doubles import ScriptedLLM, ToneASR, ToneTTS
from voice_ai_realtime_gateway.production.admission import TokenAuthenticator, TokenSigner
from voice_ai_realtime_gateway.production.observability import GatewayMetrics, SessionRegistry
from voice_ai_realtime_gateway.production.server import GatewayServer, TenantPolicy
from voice_ai_realtime_gateway.telemetry.events import ResponseTimeline
from voice_ai_realtime_gateway.telemetry.ids import SESSION_ID_HEADER, TraceContext
from voice_ai_realtime_gateway.telemetry.sink import FanoutSink
from voice_ai_realtime_gateway.transport.control import Hangup, encode_control
from voice_ai_realtime_gateway.transport.errors import TransportClosedError
from voice_ai_realtime_gateway.transport.framing import encode_packet
from voice_ai_realtime_gateway.transport.memory import MemoryWebSocket
from voice_ai_realtime_gateway.vad.energy import EnergyVAD

if TYPE_CHECKING:
    from collections.abc import Sequence

    from voice_ai_realtime_gateway.production.server import CallContext
    from voice_ai_realtime_gateway.telemetry.events import TelemetryEvent

__all__ = ["BenchConfig", "main", "percentiles", "run_benchmark"]

DEFAULT_REPORT: Final = Path(".cache/benchmarks/concurrency_report.json")
TICK_LAG_BUDGET_MS: Final = 5.0
RETAINED_BYTES_PER_CALL_BUDGET: Final = 256.0
"""Post-GC growth per served call tolerated as measurement noise. A leaked session
(pipeline, rings, transport) costs tens of kilobytes per call, so it fails by ~100x."""
WARMUP_WAVES: Final = 2
"""Churn waves run before measuring: freelists, interned strings and caches settle."""

_KEY: Final = b"benchmark-signing-key-0123456789"
_TENANT: Final = "bench"
_RATE_HZ: Final = 16_000
_SPEECH_AMPLITUDE: Final = 2_000
_NS_PER_MS: Final = 1_000_000
_PROBE_INTERVAL_S: Final = 0.001
_ANSWER_TIMEOUT_S: Final = 10.0
_SPAWN_GRACE_S: Final = 3.0


@dataclass(frozen=True, slots=True, kw_only=True)
class BenchConfig:
    """Benchmark shape."""

    sessions: int = 1_000
    shards: int = 4
    """Event loops (worker processes) the sessions are spread over, as deployed."""
    turns: int = 4
    speech_frames: int = 15
    """Spoken frames per user turn (15 x 20 ms = 300 ms)."""
    ramp_s: float = 2.0
    """Starts are spread over this window; calls outlast it, so all overlap."""
    frame_ms: int = 20
    reply: str = "Sure, let me check that."
    churn_waves: int = 8
    churn_sessions: int = 100

    def __post_init__(self) -> None:
        """Validate.

        Raises:
            ValueError: on a non-positive count or a negative ramp.
        """
        counts = (
            self.sessions,
            self.shards,
            self.turns,
            self.speech_frames,
            self.frame_ms,
            self.churn_sessions,
        )
        if min(counts) < 1 or self.churn_waves < 2 or self.ramp_s < 0:  # noqa: PLR2004 -- a slope needs 2 waves
            msg = "counts must be positive, churn_waves >= 2 and ramp_s >= 0"
            raise ValueError(msg)


@dataclass(slots=True)
class _Samples:
    """Raw measurements (compact arrays; one benchmark holds ~10^5 samples)."""

    glass_to_glass_ns: array[int] = field(default_factory=lambda: array("q"))
    turn_latency_ns: array[int] = field(default_factory=lambda: array("q"))
    jitter_ns: array[int] = field(default_factory=lambda: array("q"))
    lag_ns: array[int] = field(default_factory=lambda: array("q"))
    frames_in: int = 0
    frames_out: int = 0
    completed: int = 0
    failed: int = 0
    unanswered_turns: int = 0
    active: int = 0
    peak_active: int = 0
    last_start_ns: int = 0
    first_end_ns: int = 0
    wall_s: float = 0.0
    registry_empty: bool = True
    peak_rss_bytes: int = 0

    def merge(self, other: _Samples) -> None:
        """Fold another shard's samples into this one."""
        for name in ("glass_to_glass_ns", "turn_latency_ns", "jitter_ns", "lag_ns"):
            getattr(self, name).extend(getattr(other, name))
        for name in ("frames_in", "frames_out", "completed", "failed", "unanswered_turns"):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.peak_active += other.peak_active
        self.last_start_ns = max(self.last_start_ns, other.last_start_ns)
        self.first_end_ns = min(self.first_end_ns or other.first_end_ns, other.first_end_ns)
        self.wall_s = max(self.wall_s, other.wall_s)
        self.registry_empty = self.registry_empty and other.registry_empty
        self.peak_rss_bytes = max(self.peak_rss_bytes, other.peak_rss_bytes)


class _Timeline:
    """Telemetry sink keeping the gateway's own glass-to-glass measurements."""

    __slots__ = ("_samples",)

    def __init__(self, samples: _Samples) -> None:
        self._samples = samples

    def emit(self, event: TelemetryEvent, /) -> None:
        if isinstance(event, ResponseTimeline) and event.glass_to_glass_ns is not None:
            self._samples.glass_to_glass_ns.append(event.glass_to_glass_ns)


def percentiles(values: Sequence[int], *, scale: float = 1e-6) -> dict[str, float | int]:
    """Nearest-rank p50/p95/p99/max of ``values`` (ns), scaled (default: to ms)."""
    if not values:
        return {"count": 0}
    ordered = sorted(values)

    def rank(q: float) -> float:
        return round(ordered[max(0, math.ceil(q * len(ordered)) - 1)] * scale, 3)

    return {
        "count": len(ordered),
        "p50": rank(0.50),
        "p95": rank(0.95),
        "p99": rank(0.99),
        "max": round(ordered[-1] * scale, 3),
    }


@functools.cache
def _pcm(amplitude: int, frame_ms: int) -> bytes:
    """A square-wave PCM16 frame (cached: the client re-sends the same two waveforms)."""
    pair = amplitude.to_bytes(2, "little", signed=True) + (-amplitude).to_bytes(
        2, "little", signed=True
    )
    return pair * (_RATE_HZ * frame_ms // 1000 // 2)


def _frame(trace: TraceContext, *, amplitude: int, sequence: int, frame_ms: int) -> bytes:
    frame = AudioFrame.wrap(
        _pcm(amplitude, frame_ms),
        codec=AudioCodec.PCM16,
        sample_rate_hz=_RATE_HZ,
        channels=1,
        duration_ns=frame_ms * _NS_PER_MS,
        sequence=sequence,
        origin_ns=time.monotonic_ns(),
        trace=trace,
    )
    return encode_packet(frame)


class _Gateway:
    """One gateway process: server + metrics, with the benchmark's provider doubles."""

    def __init__(self, config: BenchConfig, samples: _Samples) -> None:
        self.config = config
        self.sessions = SessionRegistry()
        self.metrics = GatewayMetrics(self.sessions)
        self.server = GatewayServer(
            authenticator=TokenAuthenticator({"k1": _KEY}),
            tenants={
                _TENANT: TenantPolicy(
                    max_concurrent_sessions=config.sessions + config.churn_sessions,
                    max_session_duration_s=600,
                )
            },
            metrics=self.metrics,
            sessions=self.sessions,
            factory=self._pipeline,
            sink=FanoutSink(self.metrics, _Timeline(samples)),
        )
        self.token = TokenSigner("k1", _KEY).sign(
            {
                "aud": "voice-gateway",
                "exp": int(time.time()) + 3_600,
                "tenant": _TENANT,
                "codecs": ["pcm16"],
                "max_duration_s": 600,
            }
        )

    def _pipeline(self, context: CallContext) -> VoiceAgentPipeline:
        return VoiceAgentPipeline(
            transport=context.transport,
            vad=EnergyVAD(),
            asr=ToneASR({_SPEECH_AMPLITUDE: "hello"}),
            llm=ScriptedLLM(self.config.reply),
            tts=ToneTTS(frame_ms=self.config.frame_ms, pace_s=self.config.frame_ms / 1000),
            sink=context.sink,
        )

    def reply_frames(self) -> int:
        """Agent audio frames per answer (ToneTTS: 8 ms per character)."""
        return max(1, math.ceil(len(self.config.reply) * 8 / self.config.frame_ms))


async def _call(gateway: _Gateway, samples: _Samples, *, delay_s: float) -> None:  # noqa: C901, PLR0915 -- one scripted call
    """One client: ``turns`` x (speak, listen to the answer), then hang up."""
    await asyncio.sleep(delay_s)
    config = gateway.config
    trace = TraceContext.new()
    client, edge = MemoryWebSocket.pair(capacity=256)
    headers = {
        "authorization": f"Bearer {gateway.token}",
        SESSION_ID_HEADER.lower(): trace.session_id,
    }
    serving = asyncio.create_task(gateway.server.handle(edge, headers))
    period_s = config.frame_ms / 1000
    expected = gateway.reply_frames()
    received = 0
    answered = asyncio.Event()
    last_spoken_ns = 0
    turn_started = False
    samples.active += 1
    samples.peak_active = max(samples.peak_active, samples.active)
    samples.last_start_ns = time.monotonic_ns()

    async def listen() -> None:
        nonlocal received, turn_started
        previous_ns = 0
        with contextlib.suppress(TransportClosedError):
            while True:
                message = await client.recv()
                if not isinstance(message, bytes):
                    continue
                now = time.monotonic_ns()
                samples.frames_out += 1
                if turn_started:
                    turn_started = False
                    samples.turn_latency_ns.append(now - last_spoken_ns)
                elif previous_ns:
                    samples.jitter_ns.append(abs(now - previous_ns - config.frame_ms * _NS_PER_MS))
                previous_ns = now
                received += 1
                if received >= expected:
                    answered.set()

    listener = asyncio.create_task(listen())
    loop = asyncio.get_running_loop()
    deadline = loop.time()
    sequence = 0

    async def send(amplitude: int) -> None:
        nonlocal deadline, sequence, last_spoken_ns
        deadline += period_s
        await asyncio.sleep(max(0.0, deadline - loop.time()))
        await client.send(
            _frame(trace, amplitude=amplitude, sequence=sequence, frame_ms=config.frame_ms)
        )
        if amplitude:
            last_spoken_ns = time.monotonic_ns()
        sequence += 1
        samples.frames_in += 1

    try:
        for _ in range(config.turns):
            received = 0
            answered.clear()
            for _ in range(config.speech_frames):
                await send(_SPEECH_AMPLITUDE)
            turn_started = True
            give_up = loop.time() + _ANSWER_TIMEOUT_S
            while not answered.is_set() and loop.time() < give_up:
                await send(0)  # the microphone keeps streaming silence
            if not answered.is_set():
                samples.unanswered_turns += 1
            for _ in range(5):  # a short pause before speaking again
                await send(0)
        await client.send(encode_control(Hangup("benchmark done")))
        await serving
        samples.completed += 1
    except Exception:  # noqa: BLE001 -- a failed call is a measured outcome, not a crash
        samples.failed += 1
        serving.cancel()
    finally:
        samples.active -= 1
        samples.first_end_ns = samples.first_end_ns or time.monotonic_ns()
        listener.cancel()
        with contextlib.suppress(asyncio.CancelledError, TransportClosedError):
            await listener
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await serving


async def _probe_lag(samples: _Samples, stop: asyncio.Event) -> None:
    interval_ns = int(_PROBE_INTERVAL_S * 1e9)
    while not stop.is_set():
        started = time.perf_counter_ns()
        await asyncio.sleep(_PROBE_INTERVAL_S)
        samples.lag_ns.append(max(0, time.perf_counter_ns() - started - interval_ns))


async def _concurrency_phase(config: BenchConfig, sessions: int) -> _Samples:
    samples = _Samples()
    gateway = _Gateway(config, samples)
    stop = asyncio.Event()
    probe = asyncio.create_task(_probe_lag(samples, stop))
    started = time.perf_counter()
    async with asyncio.TaskGroup() as group:
        for index in range(sessions):
            group.create_task(_call(gateway, samples, delay_s=config.ramp_s * index / sessions))
    samples.wall_s = time.perf_counter() - started
    stop.set()
    await probe
    samples.registry_empty = len(gateway.sessions) == 0
    return samples


def _shard(config: BenchConfig, sessions: int, start_at: float) -> _Samples:
    """Process entry point: wait for the common start instant, then run one event loop.

    ``time.monotonic`` is system-wide, so every shard starts its ramp together.
    """
    time.sleep(max(0.0, start_at - time.monotonic()))
    result = asyncio.run(_concurrency_phase(config, sessions))
    result.peak_rss_bytes = _peak_rss_bytes()
    return result


def _run_sharded(
    config: BenchConfig, sessions: int, shards: int
) -> tuple[_Samples, list[_Samples]]:
    if shards == 1:
        only = asyncio.run(_concurrency_phase(config, sessions))
        only.peak_rss_bytes = _peak_rss_bytes()
        return only, [only]
    sizes = [sessions // shards + (index < sessions % shards) for index in range(shards)]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=shards, mp_context=context) as pool:
        start_at = time.monotonic() + _SPAWN_GRACE_S
        parts = list(pool.map(_shard, [config] * shards, sizes, [start_at] * shards))
    merged = _Samples()
    for part in parts:
        merged.merge(part)
    return merged, parts


def _sweep(config: BenchConfig, points: Sequence[int]) -> list[dict[str, object]]:
    """Single-event-loop capacity: one fresh process per point, run one at a time."""
    rows: list[dict[str, object]] = []
    context = multiprocessing.get_context("spawn")
    for sessions in points:
        with ProcessPoolExecutor(max_workers=1, mp_context=context) as pool:
            result = pool.submit(_shard, config, sessions, 0.0).result()
        lag = percentiles(result.lag_ns)
        rows.append(
            {
                "sessions_per_loop": sessions,
                "peak_concurrent_sessions": result.peak_active,
                "failed": result.failed,
                "unanswered_turns": result.unanswered_turns,
                "frames_per_s": round((result.frames_in + result.frames_out) / result.wall_s, 1),
                "event_loop_lag_ms": lag,
                "within_lag_budget": bool(result.lag_ns) and lag["p99"] < TICK_LAG_BUDGET_MS,
                "glass_to_glass_ms": percentiles(result.glass_to_glass_ns),
                "frame_jitter_ms": percentiles(result.jitter_ns),
                "peak_rss_bytes": result.peak_rss_bytes,
            }
        )
    return rows


async def _churn_phase(config: BenchConfig) -> dict[str, object]:
    samples = _Samples()
    gateway = _Gateway(config, samples)
    retained: list[int] = []
    tracemalloc.start()
    try:
        for _ in range(WARMUP_WAVES + config.churn_waves):
            async with asyncio.TaskGroup() as group:
                for index in range(config.churn_sessions):
                    group.create_task(
                        _call(
                            gateway, samples, delay_s=config.ramp_s * index / config.churn_sessions
                        )
                    )
            gc.collect()
            retained.append(_gateway_bytes())
    finally:
        tracemalloc.stop()
    waves = retained[WARMUP_WAVES:]
    slope = _slope(waves)
    return {
        "waves": config.churn_waves,
        "calls_per_wave": config.churn_sessions,
        "calls_served": samples.completed,
        "failed": samples.failed,
        "retained_bytes_after_wave": waves,
        "warmup_waves": WARMUP_WAVES,
        "spread_bytes": max(waves) - min(waves),
        "growth_bytes_per_wave": round(slope, 1),
        "retained_bytes_per_call": round(slope / config.churn_sessions, 2),
        "budget_bytes_per_call": RETAINED_BYTES_PER_CALL_BUDGET,
        "registry_empty": len(gateway.sessions) == 0,
    }


def _gateway_bytes() -> int:
    """Live traced bytes, excluding the harness's own sample arrays and tracemalloc."""
    snapshot = tracemalloc.take_snapshot().filter_traces(
        (
            tracemalloc.Filter(inclusive=False, filename_pattern=__file__),
            tracemalloc.Filter(inclusive=False, filename_pattern=tracemalloc.__file__),
        )
    )
    return sum(stat.size for stat in snapshot.statistics("filename"))


def _slope(values: Sequence[int]) -> float:
    """Least-squares growth per step."""
    n = len(values)
    mean_x, mean_y = (n - 1) / 2, sum(values) / n
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in enumerate(values))
    return numerator / sum((x - mean_x) ** 2 for x in range(n))


def _peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def run_benchmark(config: BenchConfig, *, sweep: Sequence[int] = ()) -> dict[str, object]:
    """Run the sharded concurrency phase, the optional sweep and the churn phase."""
    samples, shards = _run_sharded(config, config.sessions, config.shards)
    capacity = _sweep(config, sweep)
    churn = asyncio.run(_churn_phase(config))
    lag = percentiles(samples.lag_ns)
    overlapped = 0 < samples.last_start_ns < samples.first_end_ns
    lag_ok = bool(samples.lag_ns) and lag["p99"] < TICK_LAG_BUDGET_MS
    memory_ok = float(churn["retained_bytes_per_call"]) <= RETAINED_BYTES_PER_CALL_BUDGET  # type: ignore[arg-type]
    calls_ok = samples.failed == 0 and samples.unanswered_turns == 0 and churn["failed"] == 0
    thresholds = {
        "all_sessions_concurrent": overlapped and samples.peak_active == config.sessions,
        "event_loop_lag_p99_under_5ms": lag_ok,
        "memory_bounded_under_churn": memory_ok and bool(churn["registry_empty"]),
        "all_calls_completed": calls_ok and samples.registry_empty,
    }
    return {
        "benchmark": "concurrency",
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": f"{platform.system()}-{platform.machine()}",
            "cpu_count": os.cpu_count(),
            "transport": "MemoryWebSocket (in-process, network-free)",
        },
        "config": asdict(config),
        "concurrency": {
            "sessions": config.sessions,
            "event_loops": config.shards,
            "sessions_per_loop": [shard.peak_active for shard in shards],
            "peak_concurrent_sessions": samples.peak_active,
            "all_sessions_overlapped": overlapped,
            "completed": samples.completed,
            "failed": samples.failed,
            "unanswered_turns": samples.unanswered_turns,
            "wall_s": round(samples.wall_s, 3),
            "frames_in": samples.frames_in,
            "frames_out": samples.frames_out,
            "frames_per_s": round((samples.frames_in + samples.frames_out) / samples.wall_s, 1),
            "registry_empty": samples.registry_empty,
        },
        "glass_to_glass_ms": percentiles(samples.glass_to_glass_ns),
        "client_turn_latency_ms": percentiles(samples.turn_latency_ns),
        "frame_jitter_ms": percentiles(samples.jitter_ns),
        "event_loop_lag_ms": {
            **lag,
            "budget_ms": TICK_LAG_BUDGET_MS,
            "p99_per_loop": [percentiles(shard.lag_ns)["p99"] for shard in shards],
        },
        "single_loop_capacity": capacity,
        "churn_memory": churn,
        "peak_rss_bytes_per_loop": [shard.peak_rss_bytes for shard in shards],
        "thresholds": thresholds,
        "passed": all(thresholds.values()),
    }


def _points(raw: str) -> list[int]:
    return [int(point) for point in raw.split(",") if point.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns the exit status."""
    defaults = BenchConfig()
    parser = argparse.ArgumentParser(description="Synthetic voice-gateway concurrency benchmark.")
    parser.add_argument("--sessions", type=int, default=defaults.sessions)
    parser.add_argument(
        "--shards", type=int, default=defaults.shards, help="event loops (processes)"
    )
    parser.add_argument("--turns", type=int, default=defaults.turns)
    parser.add_argument("--ramp", type=float, default=defaults.ramp_s)
    parser.add_argument("--churn-waves", type=int, default=defaults.churn_waves)
    parser.add_argument("--churn-sessions", type=int, default=defaults.churn_sessions)
    parser.add_argument("--sweep", type=_points, default=[], help="single-loop sizes, e.g. 250,500")
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    arguments = parser.parse_args(argv)
    config = BenchConfig(
        sessions=arguments.sessions,
        shards=arguments.shards,
        turns=arguments.turns,
        ramp_s=arguments.ramp,
        churn_waves=arguments.churn_waves,
        churn_sessions=arguments.churn_sessions,
    )
    report = run_benchmark(config, sweep=arguments.sweep)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    summary = (
        "concurrency",
        "glass_to_glass_ms",
        "frame_jitter_ms",
        "event_loop_lag_ms",
        "thresholds",
    )
    print(json.dumps({key: report[key] for key in summary}, indent=2))
    print(f"report: {arguments.output}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Offline, network-free gateway simulation behind the ``voice-gateway`` CLI.

Every call runs through the production stack (:class:`GatewayServer`: JWT admission,
session registry, metrics, WebSocket transport, energy VAD, the ASR -> LLM -> TTS
pipeline, the playback ledger). Only the network and the providers are simulated: each
call is a :class:`MemoryWebSocket` pair and ASR/LLM/TTS are the deterministic doubles.

Each scripted client speaks ``turns`` utterances and listens to the answers. Every
``barge_in_every``-th turn it interrupts the agent mid-answer: it speaks again as soon as
the first agent frames arrive, so the gateway must cancel the response, flush its
outbound ring and tell the client (``playback.flush``) to drop buffered audio.

Shutdown is the production two-phase drain (:class:`DrainController`): after
:meth:`Simulation.request_stop`, new calls are refused with ``1013`` (try again later),
live calls finish their current answer and hang up normally (``1000``), and only calls
still alive at the drain deadline are closed with ``1001`` (going away). Nothing is cut
with ``1012``.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, final

from voice_ai_realtime_gateway._validation import is_non_negative_finite, is_positive_finite
from voice_ai_realtime_gateway.core.frames import AudioCodec, AudioFrame
from voice_ai_realtime_gateway.deploy.lifecycle import DrainController, DrainPhase, DrainReport
from voice_ai_realtime_gateway.domain.duplex import DEFAULT_EGRESS_RING_BYTES
from voice_ai_realtime_gateway.inference.agent import VoiceAgentPipeline
from voice_ai_realtime_gateway.inference.doubles import ScriptedLLM, ToneASR, ToneTTS
from voice_ai_realtime_gateway.production.admission import TokenAuthenticator, TokenSigner
from voice_ai_realtime_gateway.production.observability import GatewayMetrics, SessionRegistry
from voice_ai_realtime_gateway.production.server import GatewayServer, TenantPolicy
from voice_ai_realtime_gateway.telemetry.events import (
    BargeInHandled,
    BufferOverflow,
    EgressDropped,
    ResponseTimeline,
)
from voice_ai_realtime_gateway.telemetry.ids import SESSION_ID_HEADER, TraceContext
from voice_ai_realtime_gateway.telemetry.sink import FanoutSink
from voice_ai_realtime_gateway.transport.control import (
    Hangup,
    PlaybackFlush,
    decode_server_message,
    encode_control,
)
from voice_ai_realtime_gateway.transport.errors import TransportClosedError
from voice_ai_realtime_gateway.transport.framing import encode_packet
from voice_ai_realtime_gateway.transport.memory import MemoryWebSocket
from voice_ai_realtime_gateway.transport.websocket import DEFAULT_INGRESS_RING_BYTES
from voice_ai_realtime_gateway.vad.energy import EnergyVAD

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from voice_ai_realtime_gateway.production.server import CallContext, CallResult
    from voice_ai_realtime_gateway.telemetry.events import TelemetryEvent

__all__ = [
    "LiveSnapshot",
    "Simulation",
    "SimulationConfig",
    "TelemetryCollector",
    "percentiles",
]

RING_CAPACITY_BYTES_PER_CALL: Final = DEFAULT_INGRESS_RING_BYTES + DEFAULT_EGRESS_RING_BYTES
"""Ingress plus egress ring capacity of one call (the utterance ring is transient)."""

_KEY: Final = b"voice-gateway-cli-signing-key-01"
_TENANT: Final = "cli"
_RATE_HZ: Final = 16_000
_SPEECH_AMPLITUDE: Final = 2_000
_NS_PER_MS: Final = 1_000_000
_PAUSE_FRAMES: Final = 5
_BARGE_AFTER_FRAMES: Final = 2


@dataclass(frozen=True, slots=True, kw_only=True)
class SimulationConfig:
    """Shape of one simulation run.

    Attributes:
        calls: concurrent scripted calls.
        turns: user utterances per call.
        barge_in_every: every n-th turn interrupts the agent mid-answer; 0 disables.
        speech_frames: spoken frames per utterance (15 x 20 ms = 300 ms).
        frame_ms: media frame duration.
        ramp_s: call starts are spread over this window.
        real_time: pace the microphone and the synthesizer at the real frame cadence;
            ``False`` runs as fast as the event loop allows (same media timeline).
        drain_timeout_s: phase-1 drain budget before live calls are closed with 1001.
        answer_timeout_s: a turn with no complete answer by then counts as unanswered.
        reply: what the scripted LLM answers.
    """

    calls: int = 4
    turns: int = 3
    barge_in_every: int = 2
    speech_frames: int = 15
    frame_ms: int = 20
    ramp_s: float = 0.5
    real_time: bool = True
    drain_timeout_s: float = 5.0
    answer_timeout_s: float = 10.0
    reply: str = "Sure, let me check that for you."

    def __post_init__(self) -> None:
        """Validate.

        Raises:
            ValueError: on a non-positive count, a negative ``barge_in_every``, an empty
                reply, or a duration that is negative, zero (timeouts) or not finite.
        """
        if min(self.calls, self.turns, self.speech_frames, self.frame_ms) < 1:
            msg = "calls, turns, speech_frames and frame_ms must be positive"
            raise ValueError(msg)
        if self.barge_in_every < 0:
            msg = f"barge_in_every must be >= 0, got {self.barge_in_every}"
            raise ValueError(msg)
        if not is_non_negative_finite(self.ramp_s):
            msg = f"ramp_s must be finite and non-negative, got {self.ramp_s}"
            raise ValueError(msg)
        if not (
            is_positive_finite(self.drain_timeout_s) and is_positive_finite(self.answer_timeout_s)
        ):
            msg = "drain_timeout_s and answer_timeout_s must be positive and finite"
            raise ValueError(msg)
        if not self.reply.strip():
            msg = "reply must not be empty"
            raise ValueError(msg)

    def interrupts(self, turn: int) -> bool:
        """Whether 0-based ``turn`` is a barge-in turn."""
        return self.barge_in_every > 0 and (turn + 1) % self.barge_in_every == 0

    @property
    def reply_frames(self) -> int:
        """Agent audio frames in one full answer (ToneTTS voices 8 ms per character)."""
        return max(1, math.ceil(len(self.reply) * 8 / self.frame_ms))


def percentiles(values: Sequence[int], *, scale: float = 1e-6) -> dict[str, float | int | None]:
    """Nearest-rank p50/p99/max of nanosecond ``values``, scaled (default: to ms)."""
    if not values:
        return {"count": 0, "p50": None, "p99": None, "max": None}
    ordered = sorted(values)

    def rank(quantile: float) -> float:
        return round(ordered[max(0, math.ceil(quantile * len(ordered)) - 1)] * scale, 3)

    return {"count": len(ordered), "p50": rank(0.50), "p99": rank(0.99), "max": rank(1.0)}


@final
class TelemetryCollector:
    """Telemetry sink that keeps the gateway-side measurements of the run."""

    __slots__ = (
        "barge_in_ns",
        "barge_ins_over_budget",
        "egress_dropped",
        "first_audio_ns",
        "ring_overflows",
        "ttft_ns",
    )

    def __init__(self) -> None:
        """Start empty."""
        self.ttft_ns: list[int] = []
        self.first_audio_ns: list[int] = []
        self.barge_in_ns: list[int] = []
        self.barge_ins_over_budget = 0
        self.egress_dropped = 0
        self.ring_overflows = 0

    def emit(self, event: TelemetryEvent, /) -> None:
        """Record the events the CLI reports on; ignore the rest."""
        if isinstance(event, ResponseTimeline):
            if event.first_token_ns is not None:
                self.ttft_ns.append(event.first_token_ns - event.speech_end_ns)
            if event.first_egress_ns is not None:
                self.first_audio_ns.append(event.first_egress_ns - event.speech_end_ns)
        elif isinstance(event, BargeInHandled):
            self.barge_in_ns.append(event.detect_to_draining_ns)
            self.barge_ins_over_budget += not event.within_budget
        elif isinstance(event, EgressDropped):
            self.egress_dropped += 1
        elif isinstance(event, BufferOverflow):
            self.ring_overflows += 1


@dataclass(frozen=True, slots=True, kw_only=True)
class LiveSnapshot:
    """What the HUD shows, sampled at one instant."""

    phase: str
    elapsed_s: float
    calls_total: int
    calls_active: int
    calls_completed: int
    calls_refused: int
    buffered_bytes: int
    capacity_bytes: int
    ttft_p50_ms: float | None
    first_audio_p50_ms: float | None
    jitter_p50_ms: float | None
    jitter_p99_ms: float | None
    barge_ins: int
    barge_in_p99_ms: float | None
    frames_out: int
    dropped: int
    close_codes: dict[int, int]

    @property
    def drop_rate(self) -> float:
        """Dropped agent frames over all agent frames the gateway tried to send."""
        attempted = self.frames_out + self.dropped
        return self.dropped / attempted if attempted else 0.0

    @property
    def buffer_fill(self) -> float:
        """Buffered bytes over the ring capacity of the live calls."""
        return self.buffered_bytes / self.capacity_bytes if self.capacity_bytes else 0.0


@dataclass(slots=True)
class _ClientStats:
    started: int = 0
    completed: int = 0
    failed: int = 0
    answered: int = 0
    unanswered: int = 0
    interrupts: int = 0
    flushes: int = 0
    frames_in: int = 0
    frames_out: int = 0
    jitter_ns: list[int] = field(default_factory=list)
    flush_ns: list[int] = field(default_factory=list)
    close_codes: Counter[int] = field(default_factory=Counter)


@functools.cache
def _pcm(amplitude: int, frame_ms: int) -> bytes:
    pair = amplitude.to_bytes(2, "little", signed=True) + (-amplitude).to_bytes(
        2, "little", signed=True
    )
    return pair * (_RATE_HZ * frame_ms // 1000 // 2)


@final
class Simulation:
    """One gateway process plus its scripted clients."""

    __slots__ = (
        "_clients",
        "_collector",
        "_config",
        "_drain",
        "_drain_report",
        "_monotonic",
        "_server",
        "_sessions",
        "_signals",
        "_started_s",
        "_stopping",
        "_token",
    )

    def __init__(
        self, config: SimulationConfig, *, monotonic: Callable[[], float] = time.monotonic
    ) -> None:
        """Wire the gateway with the deterministic provider doubles."""
        self._config = config
        self._monotonic = monotonic
        self._sessions = SessionRegistry()
        metrics = GatewayMetrics(self._sessions)
        self._collector = TelemetryCollector()
        self._server = GatewayServer(
            authenticator=TokenAuthenticator({"k1": _KEY}),
            tenants={
                _TENANT: TenantPolicy(
                    max_concurrent_sessions=config.calls, max_session_duration_s=3_600
                )
            },
            metrics=metrics,
            sessions=self._sessions,
            factory=self._pipeline,
            sink=FanoutSink(metrics, self._collector),
        )
        self._drain = DrainController(
            server=self._server,
            sessions=self._sessions,
            drain_timeout_s=config.drain_timeout_s,
            poll_interval_s=0.01,
        )
        self._token = TokenSigner("k1", _KEY).sign(
            {
                "aud": "voice-gateway",
                "exp": int(time.time()) + 86_400,
                "tenant": _TENANT,
                "codecs": ["pcm16"],
                "max_duration_s": 3_600,
            }
        )
        self._clients = _ClientStats()
        self._stopping = asyncio.Event()
        self._signals: list[int] = []
        self._drain_report: DrainReport | None = None
        self._started_s = monotonic()

    # ------------------------------------------------------------------ control

    @property
    def signals(self) -> tuple[int, ...]:
        """Stop requests received, in order."""
        return tuple(self._signals)

    @property
    def collector(self) -> TelemetryCollector:
        """Gateway-side measurements."""
        return self._collector

    def request_stop(self, signum: int) -> None:
        """Begin the graceful drain (idempotent; safe from a signal handler)."""
        self._signals.append(signum)
        if not self._stopping.is_set():
            self._stopping.set()
            _ = self._drain.trigger()  # the run joins it in run()

    async def run(self) -> None:
        """Run every scripted call to its end, then finish the drain if one began."""
        self._started_s = self._monotonic()
        config = self._config
        spacing = config.ramp_s / config.calls
        async with asyncio.TaskGroup() as group:
            for index in range(config.calls):
                group.create_task(self._call(delay_s=index * spacing), name=f"client-{index}")
        if self._stopping.is_set():
            self._drain_report = await self._drain.drain()

    # ------------------------------------------------------------------ reporting

    def snapshot(self) -> LiveSnapshot:
        """Sample the run for the HUD."""
        clients, collector = self._clients, self._collector
        jitter = percentiles(clients.jitter_ns)
        refused = sum(n for code, n in clients.close_codes.items() if code != 1000)  # noqa: PLR2004 -- RFC 6455 normal closure
        return LiveSnapshot(
            phase=self._drain.phase.value,
            elapsed_s=round(self._monotonic() - self._started_s, 3),
            calls_total=self._config.calls,
            calls_active=len(self._sessions),
            calls_completed=clients.completed,
            calls_refused=refused,
            buffered_bytes=self._sessions.ring_bytes(),
            capacity_bytes=len(self._sessions) * RING_CAPACITY_BYTES_PER_CALL,
            ttft_p50_ms=_float(percentiles(collector.ttft_ns)["p50"]),
            first_audio_p50_ms=_float(percentiles(collector.first_audio_ns)["p50"]),
            jitter_p50_ms=_float(jitter["p50"]),
            jitter_p99_ms=_float(jitter["p99"]),
            barge_ins=len(collector.barge_in_ns),
            barge_in_p99_ms=_float(percentiles(collector.barge_in_ns)["p99"]),
            frames_out=clients.frames_out,
            dropped=collector.egress_dropped,
            close_codes=dict(sorted(clients.close_codes.items())),
        )

    def summary(self) -> dict[str, object]:
        """The run's results (the CLI's stdout JSON, minus status and exit code)."""
        snap = self.snapshot()
        clients, collector, drain = self._clients, self._collector, self._drain_report
        return {
            "config": {
                "calls": self._config.calls,
                "turns": self._config.turns,
                "barge_in_every": self._config.barge_in_every,
                "real_time": self._config.real_time,
            },
            "elapsed_s": snap.elapsed_s,
            "signals": list(self._signals),
            "calls": {
                "started": clients.started,
                "completed": clients.completed,
                "refused": snap.calls_refused,
                "failed": clients.failed,
            },
            "close_codes": {str(code): n for code, n in snap.close_codes.items()},
            "turns": {
                "answered": clients.answered,
                "unanswered": clients.unanswered,
                "interrupted": clients.interrupts,
                "flushed": clients.flushes,
            },
            "ttft_ms": percentiles(collector.ttft_ns),
            "time_to_first_audio_ms": percentiles(collector.first_audio_ns),
            "barge_in_ms": {
                **percentiles(collector.barge_in_ns),
                "over_budget": collector.barge_ins_over_budget,
            },
            "client_flush_ms": percentiles(clients.flush_ns),
            "jitter_ms": percentiles(clients.jitter_ns),
            "egress": {
                "frames_out": clients.frames_out,
                "dropped": collector.egress_dropped,
                "drop_rate": round(snap.drop_rate, 6),
                "ring_overflows": collector.ring_overflows,
            },
            "drain": None
            if drain is None
            else {"waited_s": round(drain.waited_s, 3), "forced": drain.forced},
        }

    @property
    def healthy(self) -> bool:
        """Every call completed and every turn was answered, with nothing dropped."""
        clients = self._clients
        return (
            clients.failed == 0
            and clients.unanswered == 0
            and clients.completed == self._config.calls
            and self._collector.egress_dropped == 0
        )

    @property
    def drain_phase(self) -> DrainPhase:
        """Where the drain stands."""
        return self._drain.phase

    # ------------------------------------------------------------------ internals

    def _pipeline(self, context: CallContext) -> VoiceAgentPipeline:
        config = self._config
        return VoiceAgentPipeline(
            transport=context.transport,
            vad=EnergyVAD(),
            asr=ToneASR({_SPEECH_AMPLITUDE: "hello"}),
            llm=ScriptedLLM(config.reply),
            tts=ToneTTS(
                frame_ms=config.frame_ms,
                pace_s=config.frame_ms / 1000 if config.real_time else 0.0,
            ),
            sink=context.sink,
        )

    async def _call(self, *, delay_s: float) -> None:
        await asyncio.sleep(delay_s)
        trace = TraceContext.new()
        client, edge = MemoryWebSocket.pair(capacity=256)
        headers = {
            "authorization": f"Bearer {self._token}",
            SESSION_ID_HEADER.lower(): trace.session_id,
        }
        stats = self._clients
        stats.started += 1
        serving = asyncio.create_task(self._server.handle(edge, headers), name="gateway-call")
        script = _Client(self._config, client, trace, stats, self._stopping)
        try:
            await script.converse()
            result: CallResult = await serving
        except Exception:  # noqa: BLE001 -- a failed call is a reported outcome, not a crash
            stats.failed += 1
            serving.cancel()
            with contextlib.suppress(BaseException):
                await serving
            return
        finally:
            await script.stop_listening()
        if result.close_code is not None:
            stats.close_codes[result.close_code] += 1
        if result.admitted and result.close_code == 1000:  # noqa: PLR2004 -- RFC 6455 normal closure
            stats.completed += 1


def _float(value: float | None) -> float | None:
    return None if value is None else float(value)


@final
class _Client:
    """One scripted caller: speak, listen, sometimes interrupt, hang up."""

    def __init__(
        self,
        config: SimulationConfig,
        socket: MemoryWebSocket,
        trace: TraceContext,
        stats: _ClientStats,
        stopping: asyncio.Event,
    ) -> None:
        self._config = config
        self._socket = socket
        self._trace = trace
        self._stats = stats
        self._stopping = stopping
        self._loop = asyncio.get_running_loop()
        self._deadline = self._loop.time()
        self._sequence = 0
        self._received = 0
        self._closed = asyncio.Event()
        self._previous_ns = 0
        self._interrupt_ns = 0
        self._listener: asyncio.Task[None] | None = None

    async def converse(self) -> None:
        self._listener = asyncio.create_task(self._listen(), name="client-listen")
        config = self._config
        for turn in range(config.turns):
            if self._stopping.is_set() or self._closed.is_set():
                break
            await self._speak()
            if config.interrupts(turn):
                await self._interrupt()
            await self._await_answer()
            for _ in range(_PAUSE_FRAMES):
                await self._send(0)
        if not self._closed.is_set():
            with contextlib.suppress(TransportClosedError):
                await self._socket.send(encode_control(Hangup("cli done")))

    async def stop_listening(self) -> None:
        if self._listener is not None:
            self._listener.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._listener

    async def _speak(self) -> None:
        self._received = 0
        for _ in range(self._config.speech_frames):
            await self._send(_SPEECH_AMPLITUDE)

    async def _interrupt(self) -> None:
        """Wait for the answer to start, then talk over it."""
        if not await self._wait_for(lambda: self._received >= _BARGE_AFTER_FRAMES):
            return
        self._stats.interrupts += 1
        self._interrupt_ns = time.monotonic_ns()
        await self._speak()

    async def _await_answer(self) -> None:
        answered = await self._wait_for(lambda: self._received >= self._config.reply_frames)
        if answered:
            self._stats.answered += 1
        else:
            self._stats.unanswered += 1

    async def _wait_for(self, done: Callable[[], bool]) -> bool:
        """Keep the microphone streaming silence until ``done()`` or the answer timeout."""
        give_up = self._loop.time() + self._config.answer_timeout_s
        while not done():
            if self._closed.is_set() or self._loop.time() >= give_up:
                return False
            await self._send(0)
        return True

    async def _send(self, amplitude: int) -> None:
        config = self._config
        if config.real_time:
            self._deadline += config.frame_ms / 1000
            await asyncio.sleep(max(0.0, self._deadline - self._loop.time()))
        else:
            await asyncio.sleep(0)
        frame = AudioFrame.wrap(
            _pcm(amplitude, config.frame_ms),
            codec=AudioCodec.PCM16,
            sample_rate_hz=_RATE_HZ,
            channels=1,
            duration_ns=config.frame_ms * _NS_PER_MS,
            sequence=self._sequence,
            origin_ns=time.monotonic_ns(),
            trace=self._trace,
        )
        try:
            await self._socket.send(encode_packet(frame))
        except TransportClosedError:
            self._closed.set()
            return
        self._sequence += 1
        self._stats.frames_in += 1

    async def _listen(self) -> None:
        try:
            while True:
                message = await self._socket.recv()
                if isinstance(message, str):
                    self._on_control(message)
                    continue
                now = time.monotonic_ns()
                if self._received and self._previous_ns and self._config.real_time:
                    expected = self._config.frame_ms * _NS_PER_MS
                    self._stats.jitter_ns.append(abs(now - self._previous_ns - expected))
                self._previous_ns = now
                self._received += 1
                self._stats.frames_out += 1
        except TransportClosedError:
            self._closed.set()

    def _on_control(self, text: str) -> None:
        if isinstance(decode_server_message(text), PlaybackFlush):
            self._stats.flushes += 1
            if self._interrupt_ns:
                self._stats.flush_ns.append(time.monotonic_ns() - self._interrupt_ns)
                self._interrupt_ns = 0
            self._received = 0  # audio before the flush belongs to the cancelled answer
            self._previous_ns = 0

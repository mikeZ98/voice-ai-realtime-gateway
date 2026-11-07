"""End-to-end voice agent: Transport -> VAD -> ASR -> LLM -> chunker -> TTS -> Transport.

Topology (one :class:`VoiceAgentPipeline` per call)::

    GatewaySessionRunner task group
    ├── transport / ingress / control / watcher             (Phase 2)
    │     ingress tap ─► SpeechCapture (pre-roll, utterance rings)
    └── service: recognise  ─► StreamingASR per utterance ─► UserTurn ─► conversation
    session-supervised tasks per user turn (all cancelled synchronously by barge-in)
    ├── turn:     await transcript ─► spawn stages ─► await playback ─► commit history
    ├── generate: StreamingLLM ─► SentenceChunker ─► bounded unit queue
    └── speak:    unit queue ─► StreamingTTS ─► SpeechMap ─► session.push_audio
                                                         └─► PlaybackLedger ─► transport

Guarantees:

* **Barge-in aborts generation immediately.** The turn and both stage tasks are owned
  by the session's supervisor, so the barge-in critical section cancels all of them
  synchronously, *before* it aborts the egress ring or leaves ``SPEAKING``. Each stage
  receives ``CancelledError`` at its next resumption, so none can act on the
  interrupted session, and every model stream is closed via ``aclosing``. The TTS
  generator's ``finally`` runs within one event-loop iteration of the cancel.
* **Fault barrier.** A failing LLM or TTS stage is reaped by the session supervisor,
  which faults the session (``FAULT``, socket ``1011``) and cancels the other stage and
  the turn. A failing ASR fails its service, which faults the call the same way. No
  generator is left suspended.
* **Truthful history.** The user turn is recorded as soon as its final transcript
  exists, even if the user then interrupts. The agent turn is recorded only after
  playback ends: in full if it completed, otherwise truncated to the whole words the
  client can have played (:mod:`.playback`).
* **End-to-end tracing.** Each response emits a ``ResponseTimeline`` linking the user
  utterance span to the response span, with speech end (VAD offset), final transcript,
  first token, first synthesized frame and first egress packet. Glass-to-glass latency
  follows from those timestamps.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, final, override

from voice_ai_realtime_gateway.domain.states import DrainReason
from voice_ai_realtime_gateway.inference.capture import DEFAULT_PREROLL_MS, SpeechCapture
from voice_ai_realtime_gateway.inference.chunker import ChunkerConfig, SentenceChunker
from voice_ai_realtime_gateway.inference.conversation import (
    AgentTurn,
    ConversationState,
    UserTurn,
)
from voice_ai_realtime_gateway.inference.playback import PlaybackLedger, SpeechMap
from voice_ai_realtime_gateway.inference.ports import FrameStamper
from voice_ai_realtime_gateway.pipeline.runner import GatewaySessionRunner
from voice_ai_realtime_gateway.telemetry.events import (
    ResponseTimeline,
    TranscriptReceived,
    TtsHalted,
)
from voice_ai_realtime_gateway.transport.websocket import IngressStream
from voice_ai_realtime_gateway.vad.base import VadEvent

if TYPE_CHECKING:
    from collections.abc import Sequence

    from voice_ai_realtime_gateway.core.frames import AudioFrame
    from voice_ai_realtime_gateway.domain.clock import MonotonicClock
    from voice_ai_realtime_gateway.domain.duplex import DuplexSession
    from voice_ai_realtime_gateway.inference.capture import Utterance
    from voice_ai_realtime_gateway.inference.conversation import Turn
    from voice_ai_realtime_gateway.inference.ports import StreamingASR, StreamingLLM, StreamingTTS
    from voice_ai_realtime_gateway.pipeline.runner import RunOutcome
    from voice_ai_realtime_gateway.telemetry.ids import TraceContext
    from voice_ai_realtime_gateway.telemetry.sink import TelemetrySink
    from voice_ai_realtime_gateway.transport.base import VoiceTransport
    from voice_ai_realtime_gateway.vad.base import StreamingVAD, VadDecision

__all__ = ["AgentConfig", "VoiceAgentPipeline"]


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentConfig:
    """Pipeline tuning.

    Attributes:
        preroll_ms: audio kept before VAD onset and fed to the recogniser.
        utterance_frames: per-utterance ASR ring capacity (power of two).
        utterance_bytes: per-utterance ASR ring byte budget.
        max_pending_units: text units the LLM may run ahead of TTS.
        tts_sample_rate_hz: synthesized audio sample rate.
        chunker: sentence boundary policy.
        drain_timeout_s: barge-in and shutdown reaping deadline.
    """

    preroll_ms: int = DEFAULT_PREROLL_MS
    utterance_frames: int = 512
    utterance_bytes: int = 2 * 1024 * 1024
    max_pending_units: int = 4
    tts_sample_rate_hz: int = 16_000
    chunker: ChunkerConfig = field(default_factory=ChunkerConfig)
    drain_timeout_s: float = 0.1

    def __post_init__(self) -> None:
        """Validate.

        Raises:
            ValueError: if ``max_pending_units`` is not positive.
        """
        if self.max_pending_units < 1:
            msg = f"max_pending_units must be positive, got {self.max_pending_units}"
            raise ValueError(msg)


@dataclass(slots=True)
class _Response:
    """Mutable bookkeeping of one response, shared by its stages."""

    utterance: Utterance
    trace: TraceContext
    speech: SpeechMap = field(default_factory=SpeechMap)
    generated: list[str] = field(default_factory=list)
    transcript_ns: int | None = None
    first_token_ns: int | None = None
    first_audio_ns: int | None = None
    completed: bool = False


@final
class VoiceAgentPipeline:
    """One supervised voice-agent call over a :class:`VoiceTransport`."""

    __slots__ = (
        "_asr",
        "_capture",
        "_clock",
        "_config",
        "_conversation",
        "_ledger",
        "_llm",
        "_pending_halt",
        "_runner",
        "_sink",
        "_tts",
    )

    def __init__(  # noqa: PLR0913 -- the pipeline is the composition root of a call
        self,
        *,
        transport: VoiceTransport,
        vad: StreamingVAD,
        asr: StreamingASR,
        llm: StreamingLLM,
        tts: StreamingTTS,
        sink: TelemetrySink,
        clock: MonotonicClock = time.monotonic_ns,
        config: AgentConfig | None = None,
        conversation: ConversationState | None = None,
    ) -> None:
        """Compose the call; nothing runs until :meth:`run`."""
        self._config = AgentConfig() if config is None else config
        self._asr, self._llm, self._tts = asr, llm, tts
        self._sink = sink
        self._clock = clock
        self._conversation = ConversationState() if conversation is None else conversation
        self._ledger = PlaybackLedger(transport, clock=clock)
        self._pending_halt: tuple[TraceContext, int] | None = None
        self._capture = SpeechCapture(
            sink=sink,
            preroll_ms=self._config.preroll_ms,
            max_frames=self._config.utterance_frames,
            max_bytes=self._config.utterance_bytes,
        )
        self._runner = GatewaySessionRunner(
            transport=transport,
            vad=vad,
            sink=sink,
            clock=clock,
            on_user_turn=self._on_user_turn,
            drain_timeout_s=self._config.drain_timeout_s,
            egress=self._ledger,
            on_ingress=self._observe,
            services=(self._recognise,),
        )

    @property
    def session(self) -> DuplexSession:
        """The call's duplex session."""
        return self._runner.session

    @property
    def runner(self) -> GatewaySessionRunner:
        """The transport/VAD/session supervisor."""
        return self._runner

    @property
    def conversation(self) -> ConversationState:
        """What has been said so far."""
        return self._conversation

    @property
    def capture(self) -> SpeechCapture:
        """The utterance segmenter feeding the recogniser."""
        return self._capture

    @property
    def ledger(self) -> PlaybackLedger:
        """Egress playback accounting."""
        return self._ledger

    @override
    def __repr__(self) -> str:
        return (
            f"VoiceAgentPipeline(state={self.session.state.value}, turns={len(self._conversation)})"
        )

    async def run(self) -> RunOutcome:
        """Serve the call until the transport closes (see ``GatewaySessionRunner.run``)."""
        return await self._runner.run()

    # ------------------------------------------------------------------ recognition

    def _observe(self, frame: AudioFrame, decision: VadDecision) -> None:
        # The runner taps ingress *after* the bridge applied the decision, so for a
        # barge-in onset the speak stage has already been cancelled and has recorded when
        # its TTS stream closed. Pair the two; never attribute a halt to a later onset.
        if decision.event is VadEvent.ONSET:
            pending, self._pending_halt = self._pending_halt, None
            if pending is not None and pending[1] >= decision.at_ns:
                trace, halted_at = pending
                self._sink.emit(
                    TtsHalted(trace=trace, at_ns=halted_at, halt_ns=halted_at - decision.at_ns)
                )
        self._capture.observe(frame, decision)

    async def _recognise(self) -> None:
        try:
            async for utterance in self._capture:
                await self._transcribe(utterance)
        finally:
            self._capture.close()

    async def _transcribe(self, utterance: Utterance) -> None:
        finals: list[str] = []
        async with contextlib.aclosing(
            self._asr.transcribe(IngressStream(utterance.ring))
        ) as chunks:
            async for chunk in chunks:
                self._sink.emit(
                    TranscriptReceived(
                        trace=chunk.trace,
                        at_ns=self._clock(),
                        is_final=chunk.is_final,
                        text_chars=len(chunk.text),
                        confidence=chunk.confidence,
                    )
                )
                if chunk.is_final:
                    finals.append(chunk.text)
        text = " ".join(part for part in finals if part).strip()
        turn: UserTurn | None = None
        if text:
            turn = UserTurn(text=text, trace=utterance.trace, at_ns=self._clock())
            self._conversation.append(turn)
        utterance.resolve(turn)

    # ------------------------------------------------------------------ responding

    def _on_user_turn(self, session: DuplexSession) -> None:
        self._pending_halt = None
        utterance = self._capture.last_completed
        if utterance is None:
            session.skip_response()
            return
        _ = session.spawn(
            self._respond(_Response(utterance=utterance, trace=session.trace)),
            name=f"agent-turn:{session.trace.span_id}",
        )

    async def _respond(self, response: _Response) -> None:
        try:
            user_turn = await response.utterance.transcript
            response.transcript_ns = self._clock()
            if user_turn is None:
                self.session.skip_response()
            else:
                await self._converse(
                    response, self._conversation.history_before(user_turn), user_turn
                )
            response.completed = True
        finally:
            self._commit(response)

    async def _converse(
        self, response: _Response, history: Sequence[Turn], user_turn: UserTurn
    ) -> None:
        session = self.session
        units: asyncio.Queue[str | None] = asyncio.Queue(maxsize=self._config.max_pending_units)
        span = response.trace.span_id
        # Both stages are *session* tasks: the barge-in critical section cancels them
        # synchronously, before it aborts the egress ring or changes state, so no stage
        # code can observe a half-interrupted session. A failing stage is reaped by the
        # session supervisor, which faults the session and cancels the other stage.
        generator = session.spawn(
            self._generate(response, history, user_turn.text, units), name=f"agent-llm:{span}"
        )
        speaker = session.spawn(self._speak(response, units), name=f"agent-tts:{span}")
        # ``wait`` (not ``gather``): stage failures reach the session only through its
        # supervisor, which faults the call once and cancels this turn.
        await asyncio.wait((generator, speaker))
        await session.wait_playback_complete()

    async def _generate(
        self,
        response: _Response,
        history: Sequence[Turn],
        user_text: str,
        units: asyncio.Queue[str | None],
    ) -> None:
        chunker = SentenceChunker(self._config.chunker)
        async with contextlib.aclosing(self._llm.stream(history, user_text)) as tokens:
            async for delta in tokens:
                if response.first_token_ns is None:
                    response.first_token_ns = self._clock()
                    self.session.mark_first_token(at_ns=response.first_token_ns)
                response.generated.append(delta.text)
                for unit in chunker.feed(delta.text):
                    await units.put(unit)
        for unit in chunker.flush():
            await units.put(unit)
        await units.put(None)

    async def _speak(self, response: _Response, units: asyncio.Queue[str | None]) -> None:
        session = self.session
        stamper = FrameStamper(
            response.trace, sample_rate_hz=self._config.tts_sample_rate_hz, clock=self._clock
        )
        try:
            while (unit := await units.get()) is not None:
                response.speech.begin_unit(unit)
                async with contextlib.aclosing(self._tts.synthesize(unit, stamper)) as audio:
                    async for synthesized in audio:
                        frame = synthesized.frame
                        if response.first_audio_ns is None:
                            response.first_audio_ns = self._clock()
                            session.begin_response()
                        response.speech.add_frame(frame.duration_ns, synthesized.text_offset)
                        await session.push_audio(frame)
        except asyncio.CancelledError:
            if session.machine.drain_reason is DrainReason.BARGE_IN:
                self._pending_halt = (response.trace, self._clock())  # TTS stream closed now
            raise
        if response.first_audio_ns is None:
            session.skip_response()
        else:
            session.end_response()

    def _commit(self, response: _Response) -> None:
        now = self._clock()
        span = response.trace.span_id
        generated = "".join(response.generated).strip()
        if response.completed:
            spoken = " ".join(unit.strip() for unit in response.speech.units)
        else:
            spoken = response.speech.text_through(self._ledger.heard_ns(span, now))
        if spoken:
            self._conversation.append(
                AgentTurn(
                    text=spoken,
                    trace=response.trace,
                    at_ns=now,
                    interrupted=not response.completed,
                    generated_chars=len(generated),
                )
            )
        utterance = response.utterance
        self._sink.emit(
            ResponseTimeline(
                trace=response.trace,
                at_ns=now,
                user_span_id=utterance.trace.span_id,
                speech_end_ns=utterance.started_ns
                if utterance.ended_ns is None
                else utterance.ended_ns,
                transcript_ns=response.transcript_ns,
                first_token_ns=response.first_token_ns,
                first_audio_ns=response.first_audio_ns,
                first_egress_ns=self._ledger.first_sent_ns(span),
                completed=response.completed,
                generated_chars=len(generated),
                spoken_chars=len(spoken),
            )
        )

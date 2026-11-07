"""End-to-end voice agent over memory sockets: conversation, truncation, aborts, faults."""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import TYPE_CHECKING

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from audio_support import pcm_frame
from conftest import FakeClock
from duplex_support import until
from transport_support import Client
from voice_ai_realtime_gateway.domain.states import SessionState
from voice_ai_realtime_gateway.inference.agent import AgentConfig, VoiceAgentPipeline
from voice_ai_realtime_gateway.inference.conversation import AgentTurn, UserTurn
from voice_ai_realtime_gateway.inference.doubles import ScriptedLLM, ToneASR, ToneTTS
from voice_ai_realtime_gateway.inference.ports import (
    FrameStamper,
    SynthesizedAudio,
    TokenDelta,
    TranscriptChunk,
)
from voice_ai_realtime_gateway.telemetry.events import (
    FirstTokenLatency,
    ResponseTimeline,
    TaskFailed,
    TranscriptReceived,
    VadSpeechStarted,
)
from voice_ai_realtime_gateway.telemetry.ids import TraceContext
from voice_ai_realtime_gateway.telemetry.sink import FlightRecorder
from voice_ai_realtime_gateway.transport.control import Hangup, PlaybackFlush
from voice_ai_realtime_gateway.transport.errors import CloseCode, TransportClosedError
from voice_ai_realtime_gateway.transport.framing import DecodedPacket
from voice_ai_realtime_gateway.transport.memory import MemoryWebSocket
from voice_ai_realtime_gateway.transport.websocket import HeartbeatConfig, WebSocketVoiceTransport
from voice_ai_realtime_gateway.vad.base import VadConfig
from voice_ai_realtime_gateway.vad.energy import EnergyVAD

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Sequence

    from voice_ai_realtime_gateway.core.frames import AudioFrame
    from voice_ai_realtime_gateway.domain.clock import MonotonicClock
    from voice_ai_realtime_gateway.inference.conversation import Turn
    from voice_ai_realtime_gateway.inference.ports import StreamingASR, StreamingLLM, StreamingTTS
    from voice_ai_realtime_gateway.pipeline.runner import RunOutcome

LEXICON = {1_000: "hello", 2_000: "world", 3_000: "again"}
WORDS = {word: amplitude for amplitude, word in LEXICON.items()}
REPLY = "Sure, the weather is mild. Anything else?"
QUIET = HeartbeatConfig(interval_s=3600.0, timeout_s=3600.0)
DEADLINE = 5.0


class Call:
    """A voice agent call plus the client end of its socket."""

    def __init__(  # noqa: PLR0913 -- one knob per collaborator under test
        self,
        *,
        llm: StreamingLLM | None = None,
        tts: StreamingTTS | None = None,
        asr: StreamingASR | None = None,
        clock: MonotonicClock = time.monotonic_ns,
        capacity: int = 64,
        recorder: FlightRecorder | None = None,
    ) -> None:
        server, peer = MemoryWebSocket.pair(capacity=capacity)
        self.recorder = FlightRecorder(capacity=100_000) if recorder is None else recorder
        self.trace = TraceContext.new()
        transport = WebSocketVoiceTransport(
            server, trace=self.trace, sink=self.recorder, clock=clock, heartbeat=QUIET
        )
        self.llm = ScriptedLLM(REPLY) if llm is None else llm
        self.tts = ToneTTS() if tts is None else tts
        self.asr = ToneASR(LEXICON) if asr is None else asr
        self.agent = VoiceAgentPipeline(
            transport=transport,
            vad=EnergyVAD(VadConfig(min_speech_ms=40, hangover_ms=60)),
            asr=self.asr,
            llm=self.llm,
            tts=self.tts,
            sink=self.recorder,
            clock=clock,
        )
        self.client = Client(peer)
        self.sequence = 0
        self.task: asyncio.Task[RunOutcome] = asyncio.create_task(self.agent.run())

    async def frame(self, amplitude: int) -> None:
        await self.client.audio(pcm_frame(self.trace, amplitude=amplitude, sequence=self.sequence))
        self.sequence += 1

    async def say(self, *words: str, trailing_silence: int = 3) -> None:
        for index, word in enumerate(words):
            if index:
                await self.frame(0)
            for _ in range(2):
                await self.frame(WORDS[word])
        for _ in range(trailing_silence):
            await self.frame(0)

    async def receive_audio(self, count: int) -> list[DecodedPacket]:
        packets: list[DecodedPacket] = []
        while len(packets) < count:
            message = await asyncio.wait_for(self.client.receive(), timeout=DEADLINE)
            if isinstance(message, DecodedPacket):
                packets.append(message)
        return packets

    async def until_flush(self) -> int:
        stale = 0
        while not isinstance(
            await asyncio.wait_for(self.client.receive(), timeout=DEADLINE), PlaybackFlush
        ):
            stale += 1
        return stale

    async def hangup(self) -> RunOutcome:
        await self.client.control(Hangup("bye"))
        return await asyncio.wait_for(self.task, timeout=DEADLINE)

    def agent_turns(self) -> list[AgentTurn]:
        return [t for t in self.agent.conversation.turns if isinstance(t, AgentTurn)]

    def user_turns(self) -> list[UserTurn]:
        return [t for t in self.agent.conversation.turns if isinstance(t, UserTurn)]

    def assert_generation_idle(self) -> None:
        """No LLM or TTS stream is open (the ASR may legitimately be mid-utterance)."""
        assert (self.llm.active, self.tts.active) == (0, 0)  # type: ignore[attr-defined]

    def assert_no_orphans(self) -> None:
        """After the call: no model stream of any kind is open."""
        assert (self.asr.active, self.llm.active, self.tts.active) == (0, 0, 0)  # type: ignore[attr-defined]


@pytest.fixture
async def calls() -> AsyncIterator[list[Call]]:
    started: list[Call] = []
    yield started
    for call in started:
        if not call.task.done():
            call.task.cancel()
        with contextlib.suppress(BaseException):
            await call.task


def frames_for(text: str, ms_per_char: int = 8) -> int:
    return max(1, -(-len(text) * ms_per_char // 20))


pytestmark = pytest.mark.usefixtures("strict_loop")


# ------------------------------------------------------------------------ conversation


async def test_full_turn_through_every_stage(calls: list[Call]) -> None:
    call = Call()
    calls.append(call)
    await call.say("hello", "world")
    expected = frames_for("Sure, the weather is mild.") + frames_for("Anything else?")
    packets = await call.receive_audio(expected)
    assert [p.frame.sequence for p in packets] == list(range(expected)), "continuous across units"
    response_span = packets[0].frame.trace
    assert all(p.frame.trace == response_span for p in packets)
    assert response_span.session_id == call.trace.session_id
    await until(lambda: call.agent.session.state is SessionState.LISTENING)
    await until(lambda: len(call.agent_turns()) == 1)
    assert [t.text for t in call.user_turns()] == ["hello world"]
    (agent_turn,) = call.agent_turns()
    assert (agent_turn.text, agent_turn.interrupted) == (REPLY, False)
    assert agent_turn.generated_chars == len(REPLY)
    assert call.llm.calls[-1] == ((), "hello world")  # type: ignore[attr-defined]

    (timeline,) = call.recorder.of_type(ResponseTimeline)
    assert timeline.completed
    assert timeline.user_span_id == call.trace.span_id, "response linked to the user's audio span"
    assert timeline.trace == response_span
    stages = [timeline.speech_end_ns, timeline.transcript_ns, timeline.first_token_ns,
              timeline.first_audio_ns, timeline.first_egress_ns]  # fmt: skip
    assert all(stage is not None for stage in stages)
    assert stages == sorted(stages), "stages happen in pipeline order"  # type: ignore[type-var]
    assert timeline.glass_to_glass_ns is not None
    assert timeline.glass_to_glass_ns >= 0
    assert (timeline.generated_chars, timeline.spoken_chars) == (len(REPLY), len(REPLY))
    assert len(call.recorder.of_type(FirstTokenLatency)) == 1
    transcripts = call.recorder.of_type(TranscriptReceived)
    assert [t.is_final for t in transcripts] == [False, False, True], "interim per word, then final"
    outcome = await call.hangup()
    assert (outcome.turns, outcome.faulted) == (1, False)
    call.assert_no_orphans()
    assert repr(call.agent) == "VoiceAgentPipeline(state=terminated, turns=2)"
    assert call.agent.runner.session is call.agent.session
    assert call.agent.capture.last_completed is not None
    assert call.agent.ledger.first_sent_ns(response_span.span_id) is not None


async def test_second_turn_sees_the_first_exchange_as_context(calls: list[Call]) -> None:
    def reply(history: Sequence[Turn], user_text: str) -> str:
        return f"Turn {len(history) // 2 + 1}: {user_text}."

    call = Call(llm=ScriptedLLM(reply))
    calls.append(call)
    await call.say("hello")
    await until(lambda: len(call.agent_turns()) == 1)
    await call.say("again")
    await until(lambda: len(call.agent_turns()) == 2)
    history, user_text = call.llm.calls[-1]  # type: ignore[attr-defined]
    assert [type(t).__name__ for t in history] == ["UserTurn", "AgentTurn"]
    assert user_text == "again"
    assert [t.text for t in call.agent_turns()] == ["Turn 1: hello.", "Turn 2: again."]
    await call.hangup()


# ------------------------------------------------------------------------ barge-in


async def test_mid_sentence_barge_in_keeps_only_the_words_actually_heard(
    calls: list[Call],
) -> None:
    clock = FakeClock()
    call = Call(clock=clock, capacity=8)
    calls.append(call)
    await call.say("hello")
    await call.receive_audio(3)
    clock.now += 70_000_000  # the client has been playing for 70 ms
    await call.say("again", trailing_silence=0)  # user talks over the agent
    stale = await call.until_flush()
    assert stale <= 16
    await until(lambda: len(call.agent_turns()) == 1)
    (interrupted,) = call.agent_turns()
    # 70 ms heard = 3 whole frames of an 11-frame unit = 7 of 26 chars ("Sure, t") -> "Sure,"
    assert interrupted.text == "Sure,"
    assert interrupted.interrupted
    assert interrupted.generated_chars > len(interrupted.text)
    for hallucination in ("weather", "mild", "Anything"):
        assert hallucination not in interrupted.text
    timeline = call.recorder.of_type(ResponseTimeline)[0]
    assert (timeline.completed, timeline.spoken_chars) == (False, len("Sure,"))
    call.assert_generation_idle()
    assert call.tts.aborted >= 1  # type: ignore[attr-defined]
    for _ in range(3):
        await call.frame(0)  # the interrupting utterance ends: it becomes the next turn
    await until(lambda: len(call.user_turns()) == 2)
    assert [t.text for t in call.user_turns()] == ["hello", "again"]
    turns = call.agent.conversation.turns
    assert [type(t).__name__ for t in turns[:3]] == ["UserTurn", "AgentTurn", "UserTurn"]
    await call.hangup()


async def test_barge_in_before_any_audio_records_no_agent_words(calls: list[Call]) -> None:
    gate = asyncio.Event()

    class SlowLLM:
        active = 0

        async def stream(
            self, history: Sequence[Turn], user_text: str, /
        ) -> AsyncGenerator[TokenDelta, None]:
            del history, user_text
            self.active += 1
            try:
                await gate.wait()
                yield TokenDelta(text="never", index=0)
            finally:
                self.active -= 1

    llm = SlowLLM()
    call = Call(llm=llm)
    calls.append(call)
    await call.say("hello")
    await until(lambda: llm.active == 1)  # agent is thinking
    await call.say("again", trailing_silence=0)
    await call.until_flush()
    await until(lambda: call.recorder.of_type(ResponseTimeline) != [])
    assert call.agent_turns() == []
    timeline = call.recorder.of_type(ResponseTimeline)[0]
    assert (timeline.first_token_ns, timeline.first_audio_ns, timeline.completed) == (
        None,
        None,
        False,
    )
    assert llm.active == 0, "the pending LLM stream was closed"
    await call.hangup()


async def test_tts_halts_within_5_ms_of_the_barge_in(calls: list[Call]) -> None:
    tts = ToneTTS(pace_s=0.004)  # a synthesizer slower than real time per frame
    call = Call(tts=tts, llm=ScriptedLLM(REPLY * 20))
    calls.append(call)
    await call.say("hello")
    await call.receive_audio(2)
    await call.say("again", trailing_silence=0)
    await call.until_flush()
    onset = call.recorder.of_type(VadSpeechStarted)[-1].at_ns
    assert tts.last_closed_ns is not None
    halt_ns = tts.last_closed_ns - onset
    assert 0 <= halt_ns < 5_000_000, f"TTS halted {halt_ns / 1e6:.3f} ms after the barge-in"
    assert tts.aborted == 1
    call.assert_generation_idle()
    await call.hangup()
    call.assert_no_orphans()


# ------------------------------------------------------------------------ edge paths


async def test_unrecognised_speech_skips_the_response(calls: list[Call]) -> None:
    call = Call(asr=ToneASR(LEXICON, silence_peak=50_000))  # hears speech as silence
    calls.append(call)
    await call.say("hello")
    await until(lambda: call.recorder.of_type(ResponseTimeline) != [])
    assert call.agent.conversation.turns == ()
    assert call.agent.session.state is SessionState.LISTENING
    timeline = call.recorder.of_type(ResponseTimeline)[0]
    assert timeline.completed
    assert timeline.transcript_ns is not None
    assert call.llm.calls == type(call.llm.calls)(maxlen=32)  # type: ignore[attr-defined]
    await call.hangup()


async def test_empty_reply_returns_to_listening(calls: list[Call]) -> None:
    call = Call(llm=ScriptedLLM("   "))
    calls.append(call)
    await call.say("hello")
    await until(lambda: call.recorder.of_type(ResponseTimeline) != [])
    assert call.agent_turns() == []
    await until(lambda: call.agent.session.state is SessionState.LISTENING)
    await call.hangup()


async def test_turn_without_a_captured_utterance_is_skipped(recorder: FlightRecorder) -> None:
    server, _peer = MemoryWebSocket.pair()
    transport = WebSocketVoiceTransport(server, trace=TraceContext.new(), sink=recorder)
    agent = VoiceAgentPipeline(
        transport=transport,
        vad=EnergyVAD(),
        asr=ToneASR(LEXICON),
        llm=ScriptedLLM(REPLY),
        tts=ToneTTS(),
        sink=recorder,
    )
    session = agent.session
    session.connect()
    session.connected()
    session.on_user_speech_stopped()
    agent._on_user_turn(session)
    assert session.state is SessionState.LISTENING
    await session.aclose()


async def test_config_is_validated() -> None:
    with pytest.raises(ValueError, match="max_pending_units"):
        AgentConfig(max_pending_units=0)


# ------------------------------------------------------------------------ fault barriers


class FailingLLM:
    def __init__(self) -> None:
        self.active = 0

    async def stream(
        self, history: Sequence[Turn], user_text: str, /
    ) -> AsyncGenerator[TokenDelta, None]:
        del history, user_text
        self.active += 1
        try:
            yield TokenDelta(text="Partial", index=0)
            await asyncio.sleep(0)
            msg = "model backend 503"
            raise RuntimeError(msg)
        finally:
            self.active -= 1


class FailingTTS:
    def __init__(self) -> None:
        self.active = 0

    async def synthesize(
        self, text: str, stamper: FrameStamper, /
    ) -> AsyncGenerator[SynthesizedAudio, None]:
        self.active += 1
        try:
            yield SynthesizedAudio(
                frame=stamper.stamp(bytes(640), duration_ns=20_000_000), text_offset=0
            )
            msg = f"synthesis failed on {text!r}"
            raise OSError(msg)
        finally:
            self.active -= 1


@pytest.mark.parametrize("failing", ["llm", "tts"])
async def test_stage_failure_faults_the_call_without_orphans(
    calls: list[Call], failing: str
) -> None:
    call = Call(llm=FailingLLM()) if failing == "llm" else Call(tts=FailingTTS())
    calls.append(call)
    await call.say("hello")
    outcome = await asyncio.wait_for(call.task, timeout=DEADLINE)
    assert outcome.faulted
    assert outcome.close_code == CloseCode.INTERNAL_ERROR
    failed = {e.task_name.split(":")[0] for e in call.recorder.of_type(TaskFailed)}
    assert failed == {"agent-llm" if failing == "llm" else "agent-tts"}, "only the stage failed"
    call.assert_no_orphans()


async def test_asr_failure_faults_the_call_and_propagates(calls: list[Call]) -> None:
    class BrokenASR:
        active = 0

        async def transcribe(
            self, audio: AsyncIterator[AudioFrame], /
        ) -> AsyncGenerator[TranscriptChunk, None]:
            async for frame in audio:
                if frame.sequence < 0:  # never: keeps this a well-typed async generator
                    yield TranscriptChunk(
                        text="", is_final=True, confidence=0.0, words=(), trace=frame.trace, at_ns=0
                    )
                msg = "recogniser crashed"
                raise ConnectionError(msg)

    call = Call(asr=BrokenASR())
    calls.append(call)
    await call.say("hello")
    with pytest.raises(ExceptionGroup) as excinfo:
        await asyncio.wait_for(call.task, timeout=DEADLINE)
    assert excinfo.group_contains(ConnectionError)
    assert call.agent.session.faulted


# ------------------------------------------------------------------------ races


def reply_for(units: int) -> str:
    return " ".join(f"Sentence number {i} is here." for i in range(units))


def run_call(barge_after: int, reply_units: int, capacity: int, ms_per_char: int) -> Call:
    async def drain(call: Call) -> None:
        with contextlib.suppress(TransportClosedError):
            while True:
                await call.client.receive()

    async def scenario() -> Call:
        errors: list[dict[str, object]] = []
        asyncio.get_running_loop().set_exception_handler(lambda _l, ctx: errors.append(dict(ctx)))
        reply = reply_for(reply_units)
        call = Call(llm=ScriptedLLM(reply), tts=ToneTTS(ms_per_char=ms_per_char), capacity=capacity)
        await call.say("hello")
        responses = 1
        if barge_after:
            total = sum(frames_for(unit, ms_per_char) for unit in reply.split(". "))
            await call.receive_audio(min(barge_after, total))
            await call.say("again")
            responses = 2
        drainer = asyncio.create_task(drain(call))
        await until(
            lambda: len(call.recorder.of_type(ResponseTimeline)) == responses, spins=200_000
        )
        await until(lambda: call.agent.session.state is SessionState.LISTENING, spins=200_000)
        await call.hangup()
        await drainer
        for _ in range(5):
            await asyncio.sleep(0)
        assert not [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        assert errors == []
        return call

    return asyncio.run(scenario())


@settings(max_examples=40, deadline=None)
@given(
    barge_after=st.integers(min_value=0, max_value=12),
    reply_units=st.integers(min_value=1, max_value=4),
    capacity=st.integers(min_value=2, max_value=16),
    ms_per_char=st.integers(min_value=1, max_value=12),
)
def test_barge_in_at_any_point_never_leaks_or_hallucinates(
    barge_after: int, reply_units: int, capacity: int, ms_per_char: int
) -> None:
    call = run_call(barge_after, reply_units, capacity, ms_per_char)
    call.assert_no_orphans()
    reply_words = reply_for(reply_units).split()
    for turn in call.agent_turns():
        words = turn.text.split()
        assert words == reply_words[: len(words)], "agent history is a spoken prefix"
        if not turn.interrupted:
            assert words == reply_words


async def test_many_concurrent_calls_with_barge_ins_stay_isolated(calls: list[Call]) -> None:
    shared = FlightRecorder(capacity=200_000)
    batch = [Call(capacity=4, recorder=shared) for _ in range(12)]
    calls.extend(batch)

    async def converse(index: int, call: Call) -> None:
        await call.say("hello")
        await call.receive_audio(1 + index % 3)
        await call.say("again")
        await until(lambda: len(call.user_turns()) == 2, spins=200_000)
        await call.hangup()

    await asyncio.gather(*(converse(i, c) for i, c in enumerate(batch)))
    sessions = {c.trace.session_id for c in batch}
    assert len(sessions) == 12
    for call in batch:
        call.assert_no_orphans()
        assert [t.text for t in call.user_turns()] == ["hello", "again"]
        assert all(
            t.trace.session_id == call.trace.session_id for t in call.agent.conversation.turns
        )

"""VAD bridge: edges drive the session; onset while SPEAKING is a sub-100 us barge-in."""

from __future__ import annotations

import statistics
import time
from typing import TYPE_CHECKING

import pytest

from audio_support import LOUD, pcm_frame
from duplex_support import RecordingEgress, until
from voice_ai_realtime_gateway.domain.duplex import BARGE_IN_BUDGET_NS, DuplexSession
from voice_ai_realtime_gateway.domain.states import SessionState, SessionTrigger
from voice_ai_realtime_gateway.vad.base import VadConfig, VadDecision, VadEvent
from voice_ai_realtime_gateway.vad.bridge import VadBridge
from voice_ai_realtime_gateway.vad.energy import EnergyVAD

if TYPE_CHECKING:
    from voice_ai_realtime_gateway.core.frames import AudioFrame
    from voice_ai_realtime_gateway.telemetry.ids import TraceContext
    from voice_ai_realtime_gateway.telemetry.sink import FlightRecorder

pytestmark = pytest.mark.usefixtures("strict_loop")

S = SessionState


class ScriptedVAD:
    """Mock VAD replaying a fixed list of edge events."""

    def __init__(self, events: list[VadEvent | None]) -> None:
        self._events = iter(events)
        self._config = VadConfig()
        self._speaking = False

    @property
    def config(self) -> VadConfig:
        return self._config

    def process_chunk(self, frame: AudioFrame, /) -> VadDecision:
        event = next(self._events)
        if event is not None:
            self._speaking = event is VadEvent.ONSET
        return VadDecision(
            is_speech=self._speaking, event=event, level_dbfs=-20.0, at_ns=frame.origin_ns
        )

    def reconfigure(self, config: VadConfig, /) -> None:
        self._config = config

    def reset(self) -> None:
        self._speaking = False


async def test_edges_drive_turn_taking(recorder: FlightRecorder, trace: TraceContext) -> None:
    session = DuplexSession(egress=RecordingEgress(), sink=recorder, trace=trace)
    session.connect()
    session.connected()
    bridge = VadBridge(ScriptedVAD([None, VadEvent.ONSET, None, VadEvent.OFFSET]), session)
    assert isinstance(bridge.vad, ScriptedVAD)
    outcomes = [await bridge.feed(pcm_frame(trace, origin_ns=i)) for i in range(4)]
    assert [o.decision.event for o in outcomes] == [None, VadEvent.ONSET, None, VadEvent.OFFSET]
    assert all(o.barge_in is None for o in outcomes), "onset while LISTENING is not a barge-in"
    assert outcomes[3].turn is not None
    assert outcomes[3].turn.trigger is SessionTrigger.USER_TURN_ENDED
    assert session.state is S.PROCESSING_THOUGHT


async def speaking_session(recorder: FlightRecorder, trace: TraceContext) -> DuplexSession:
    session = DuplexSession(egress=RecordingEgress(gated=True), sink=recorder, trace=trace)
    session.connect()
    session.connected()
    session.on_user_speech_stopped()
    span = session.begin_response()
    for sequence in range(4):
        await session.push_audio(pcm_frame(span, amplitude=LOUD, sequence=sequence))
    await until(lambda: len(session.ring) == 3)  # pump holds one frame inside send()
    return session


async def test_onset_while_speaking_barges_in_and_drains_playback(
    recorder: FlightRecorder, trace: TraceContext
) -> None:
    session = await speaking_session(recorder, trace)
    bridge = VadBridge(EnergyVAD(VadConfig(min_speech_ms=20)), session)
    outcome = await bridge.feed(pcm_frame(trace, amplitude=LOUD, origin_ns=time.monotonic_ns()))
    assert outcome.decision.event is VadEvent.ONSET
    report = outcome.barge_in
    assert report is not None
    assert report.dropped_frames == 3, "queued playback is drained immediately"
    assert report.resumed_listening
    assert len(session.ring) == 0
    assert session.state is S.LISTENING


async def test_vad_triggered_barge_in_is_sub_100_microseconds(
    recorder: FlightRecorder, trace: TraceContext
) -> None:
    latencies: list[int] = []
    for _ in range(40):
        session = await speaking_session(recorder, trace)
        bridge = VadBridge(EnergyVAD(VadConfig(min_speech_ms=20)), session)
        frame = pcm_frame(trace, amplitude=LOUD, origin_ns=time.monotonic_ns())
        outcome = await bridge.feed(frame)
        assert outcome.barge_in is not None
        latencies.append(outcome.barge_in.detect_to_draining_ns)
        await session.aclose()
    assert statistics.median(latencies) < 100_000, f"median {statistics.median(latencies)} ns"
    assert max(latencies) < BARGE_IN_BUDGET_NS

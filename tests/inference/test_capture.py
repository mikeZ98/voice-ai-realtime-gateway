"""Utterance capture: pre-roll, onset/offset segmentation, closing semantics."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from audio_support import LOUD, pcm_frame
from voice_ai_realtime_gateway.inference.capture import SpeechCapture
from voice_ai_realtime_gateway.inference.conversation import UserTurn
from voice_ai_realtime_gateway.vad.base import VadDecision, VadEvent

if TYPE_CHECKING:
    from voice_ai_realtime_gateway.core.frames import AudioFrame
    from voice_ai_realtime_gateway.telemetry.ids import TraceContext
    from voice_ai_realtime_gateway.telemetry.sink import FlightRecorder

pytestmark = pytest.mark.usefixtures("strict_loop")


def decision(event: VadEvent | None, at_ns: int = 0) -> VadDecision:
    return VadDecision(
        is_speech=event is VadEvent.ONSET, event=event, level_dbfs=-20.0, at_ns=at_ns
    )


def frames(trace: TraceContext, count: int) -> list[AudioFrame]:
    return [pcm_frame(trace, amplitude=LOUD, sequence=i, origin_ns=i) for i in range(count)]


async def test_preroll_is_bounded_and_seeds_the_utterance(
    recorder: FlightRecorder, trace: TraceContext
) -> None:
    capture = SpeechCapture(sink=recorder, preroll_ms=40)
    stream = frames(trace, 6)
    for frame in stream[:4]:
        capture.observe(frame, decision(None))
    assert capture.preroll_ns == 40_000_000, "only the last 40 ms is kept"
    capture.observe(stream[4], decision(VadEvent.ONSET))
    utterance = await anext(capture)
    owners = [capture.current]
    assert capture.preroll_ns == 0
    assert [utterance.ring.get_nowait().sequence for _ in range(len(utterance.ring))] == [3, 4]
    assert utterance.started_ns == 3
    assert utterance.trace == trace
    capture.observe(stream[5], decision(None))  # mid-utterance: forwarded, still open
    assert capture.current is utterance
    capture.observe(stream[5], decision(VadEvent.OFFSET, at_ns=99))
    assert [utterance.ring.get_nowait().sequence for _ in range(2)] == [5, 5]
    assert utterance.ring.closed
    assert utterance.ended_ns == 99
    owners.append(capture.current)
    assert owners == [utterance, None]
    assert capture.last_completed is utterance
    assert repr(utterance) == "Utterance(number=0, frames=0, ended=99)"


async def test_zero_preroll_still_keeps_the_onset_frame(
    recorder: FlightRecorder, trace: TraceContext
) -> None:
    capture = SpeechCapture(sink=recorder, preroll_ms=0)
    for frame in frames(trace, 3)[:2]:
        capture.observe(frame, decision(None))
    capture.observe(frames(trace, 3)[2], decision(VadEvent.ONSET))
    utterance = await anext(capture)
    assert len(utterance.ring) == 1


async def test_utterances_are_numbered_in_order(
    recorder: FlightRecorder, trace: TraceContext
) -> None:
    capture = SpeechCapture(sink=recorder)
    for _ in range(2):
        capture.observe(pcm_frame(trace), decision(VadEvent.ONSET))
        capture.observe(pcm_frame(trace), decision(VadEvent.OFFSET))
    assert [(await anext(capture)).number for _ in range(2)] == [0, 1]


async def test_close_aborts_the_open_utterance_and_ends_the_stream(
    recorder: FlightRecorder, trace: TraceContext
) -> None:
    capture = SpeechCapture(sink=recorder)
    capture.observe(pcm_frame(trace), decision(VadEvent.ONSET))
    utterance = await anext(capture)
    capture.close()
    capture.close()
    assert utterance.ring.closed
    assert await utterance.ring.get() is None, "aborted: queued audio dropped"
    assert capture.current is None
    assert [u async for u in capture] == []
    assert [u async for u in capture] == [], "the stream stays ended"
    capture.observe(pcm_frame(trace), decision(VadEvent.ONSET))
    assert capture.current is None, "a closed capture ignores audio"


async def test_resolution_is_idempotent_and_tolerates_a_cancelled_waiter(
    recorder: FlightRecorder, trace: TraceContext
) -> None:
    capture = SpeechCapture(sink=recorder)
    capture.observe(pcm_frame(trace), decision(VadEvent.ONSET))
    utterance = await anext(capture)
    waiter = asyncio.ensure_future(utterance.transcript)
    await asyncio.sleep(0)
    waiter.cancel()  # a cancelled turn task cancels the future it awaited
    await asyncio.sleep(0)
    utterance.resolve(UserTurn(text="late", trace=trace, at_ns=0))
    assert utterance.transcript.cancelled()

    capture.observe(pcm_frame(trace), decision(VadEvent.OFFSET))
    capture.observe(pcm_frame(trace), decision(VadEvent.ONSET))
    second = await anext(capture)
    turn = UserTurn(text="hi", trace=trace, at_ns=0)
    second.resolve(turn)
    second.resolve(None)
    assert await second.transcript is turn


async def test_negative_preroll_is_rejected(recorder: FlightRecorder) -> None:
    with pytest.raises(ValueError, match="preroll_ms"):
        SpeechCapture(sink=recorder, preroll_ms=-1)


async def test_close_without_an_open_utterance(recorder: FlightRecorder) -> None:
    capture = SpeechCapture(sink=recorder)
    capture.close()
    assert [u async for u in capture] == []

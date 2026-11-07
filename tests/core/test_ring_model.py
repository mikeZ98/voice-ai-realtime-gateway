"""Model-based fuzzing: the ring must behave exactly like a bounded deque reference."""

from __future__ import annotations

from collections import deque

import pytest
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, precondition, rule

from voice_ai_realtime_gateway.core.frames import AudioCodec, AudioFrame
from voice_ai_realtime_gateway.core.ring import (
    AudioRingBuffer,
    OverflowPolicy,
    RingClosedError,
    RingEmptyError,
    RingFullError,
)
from voice_ai_realtime_gateway.telemetry.ids import TraceContext

TRACE = TraceContext.new()


def frame(nbytes: int, sequence: int) -> AudioFrame:
    return AudioFrame.wrap(
        bytes(nbytes),
        codec=AudioCodec.PCM16,
        sample_rate_hz=16_000,
        channels=1,
        duration_ns=20_000_000,
        sequence=sequence,
        origin_ns=0,
        trace=TRACE,
    )


class RingModel(RuleBasedStateMachine):
    ring: AudioRingBuffer
    model: deque[AudioFrame]
    capacity_frames: int
    capacity_bytes: int
    policy: OverflowPolicy
    closed: bool
    sequence: int

    @initialize(
        log2_frames=st.integers(min_value=0, max_value=4),
        capacity_bytes=st.integers(min_value=2, max_value=256),
        policy=st.sampled_from(OverflowPolicy),
    )
    def setup(self, log2_frames: int, capacity_bytes: int, policy: OverflowPolicy) -> None:
        self.capacity_frames = 1 << log2_frames
        self.capacity_bytes = capacity_bytes
        self.policy = policy
        self.ring = AudioRingBuffer(
            capacity_frames=self.capacity_frames, capacity_bytes=capacity_bytes, policy=policy
        )
        self.model = deque()
        self.closed = False
        self.sequence = 0

    def _model_fits(self, nbytes: int) -> bool:
        used = sum(f.nbytes for f in self.model)
        return len(self.model) < self.capacity_frames and used + nbytes <= self.capacity_bytes

    @rule(half_words=st.integers(min_value=1, max_value=140))
    def put(self, half_words: int) -> None:
        nbytes = half_words * 2
        item = frame(nbytes, self.sequence)
        self.sequence += 1
        if nbytes > self.capacity_bytes:
            with pytest.raises(ValueError, match="exceeds"):
                self.ring.put_nowait(item)
            return
        if self.closed:
            with pytest.raises(RingClosedError):
                self.ring.put_nowait(item)
            return
        if self.policy is OverflowPolicy.BACKPRESSURE and not self._model_fits(nbytes):
            with pytest.raises(RingFullError):
                self.ring.put_nowait(item)
            return
        expected_evictions = 0
        while not self._model_fits(nbytes):
            self.model.popleft()
            expected_evictions += 1
        self.model.append(item)
        assert self.ring.put_nowait(item) == expected_evictions

    @rule()
    def get(self) -> None:
        if self.model:
            assert self.ring.get_nowait() is self.model.popleft()
        else:
            with pytest.raises(RingClosedError if self.closed else RingEmptyError):
                self.ring.get_nowait()

    @rule()
    def clear(self) -> None:
        assert self.ring.clear() == len(self.model)
        self.model.clear()

    @precondition(lambda self: not self.closed)
    @rule()
    def close(self) -> None:
        self.ring.close()
        self.closed = True

    @rule()
    def abort(self) -> None:
        assert self.ring.abort() == len(self.model)
        self.model.clear()
        self.closed = True

    @rule()
    def reset(self) -> None:
        assert self.ring.reset() == len(self.model)
        self.model.clear()
        self.closed = False

    @invariant()
    def occupancy_matches_model_and_bounds_hold(self) -> None:
        stats = self.ring.stats()
        assert stats.frames == len(self.model) <= self.capacity_frames
        assert stats.nbytes == sum(f.nbytes for f in self.model) <= self.capacity_bytes
        assert stats.high_watermark_frames <= self.capacity_frames
        assert stats.high_watermark_bytes <= self.capacity_bytes
        assert stats.closed is self.closed
        live = [slot for slot in self.ring._slots if slot is not None]
        assert len(live) == len(self.model), "no slot may pin a dequeued frame"


RingModel.TestCase.settings = settings(max_examples=150, stateful_step_count=60)
TestRingModel = RingModel.TestCase

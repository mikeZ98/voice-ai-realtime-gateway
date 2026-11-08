"""Bounded, zero-copy, asyncio-native audio frame ring.

Design
------
* **Preallocated slots.** ``capacity_frames`` (a power of two) reference slots are
  allocated once; head/tail arithmetic is a bit-mask. Frames are stored by reference,
  so payload bytes are never copied (see :class:`~.frames.AudioFrame`).
* **Dual bound.** Admission requires *both* a free slot and room in the
  ``capacity_bytes`` budget, so a burst of large PCM frames cannot exceed the memory
  budget even when slots remain.
* **Overflow policy.**
  :attr:`OverflowPolicy.BACKPRESSURE` -- :meth:`AudioRingBuffer.put` suspends the
  producer until space frees up; :meth:`AudioRingBuffer.put_nowait` raises
  :class:`RingFullError`. Use for egress (TTS -> speaker) where audio must not be lost.
  :attr:`OverflowPolicy.DROP_OLDEST` -- never blocks; evicts the oldest frames to admit
  the newest. Use for live ingress where stale audio is worse than lost audio
  (bounds queueing delay, i.e. prevents bufferbloat under jitter / slow consumers).
* **Epoch fencing.** :meth:`AudioRingBuffer.abort` and :meth:`AudioRingBuffer.reset`
  bump an epoch. A producer or consumer suspended across an epoch change is released
  with :class:`RingClosedError` / ``None`` and can never leak a stale frame into the
  next turn -- the property barge-in relies on.

Concurrency model: confined to one event loop; every mutating method is synchronous
and therefore atomic between ``await`` points. Not thread-safe by design.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from dataclasses import dataclass
from enum import StrEnum, unique
from typing import TYPE_CHECKING, cast, final, override

from voice_ai_realtime_gateway.telemetry.events import BufferOverflow
from voice_ai_realtime_gateway.telemetry.sink import NullSink

if TYPE_CHECKING:
    from voice_ai_realtime_gateway.core.frames import AudioFrame
    from voice_ai_realtime_gateway.telemetry.sink import TelemetrySink

__all__ = [
    "AudioRingBuffer",
    "OverflowPolicy",
    "RingClosedError",
    "RingEmptyError",
    "RingError",
    "RingFullError",
    "RingStats",
]


class RingError(Exception):
    """Base class for ring buffer errors."""


class RingFullError(RingError):
    """A non-blocking put found no room under the BACKPRESSURE policy."""


class RingEmptyError(RingError):
    """A non-blocking get found the ring empty (but still open)."""


class RingClosedError(RingError):
    """The ring is closed / aborted, or was reset while the caller was suspended."""


@unique
class OverflowPolicy(StrEnum):
    """What happens when a frame does not fit."""

    BACKPRESSURE = "backpressure"
    DROP_OLDEST = "drop_oldest"


@dataclass(frozen=True, slots=True, kw_only=True)
class RingStats:
    """Lifetime counters and current occupancy of a ring."""

    frames: int
    nbytes: int
    written: int
    read: int
    evicted: int
    discarded: int
    high_watermark_frames: int
    high_watermark_bytes: int
    epoch: int
    closed: bool


@final
class AudioRingBuffer:
    """Memory-bounded circular buffer of :class:`AudioFrame` references."""

    __slots__ = (
        "_capacity_bytes",
        "_capacity_frames",
        "_closed",
        "_discarded",
        "_epoch",
        "_evicted",
        "_getters",
        "_head",
        "_hwm_bytes",
        "_hwm_frames",
        "_mask",
        "_nbytes",
        "_policy",
        "_putters",
        "_read",
        "_sink",
        "_size",
        "_slots",
        "_written",
    )

    def __init__(
        self,
        *,
        capacity_frames: int,
        capacity_bytes: int,
        policy: OverflowPolicy,
        sink: TelemetrySink | None = None,
    ) -> None:
        """Allocate ``capacity_frames`` slots bounded by ``capacity_bytes`` of payload.

        Raises:
            ValueError: if ``capacity_frames`` is not a positive power of two or
                ``capacity_bytes`` is not positive.
        """
        if capacity_frames < 1 or capacity_frames & (capacity_frames - 1):
            msg = f"capacity_frames must be a positive power of two, got {capacity_frames}"
            raise ValueError(msg)
        if capacity_bytes < 1:
            msg = f"capacity_bytes must be positive, got {capacity_bytes}"
            raise ValueError(msg)
        self._capacity_frames = capacity_frames
        self._capacity_bytes = capacity_bytes
        self._mask = capacity_frames - 1
        self._policy = policy
        self._sink: TelemetrySink = NullSink() if sink is None else sink
        self._slots: list[AudioFrame | None] = [None] * capacity_frames
        self._head = 0
        self._size = 0
        self._nbytes = 0
        self._closed = False
        self._epoch = 0
        self._getters: deque[asyncio.Future[None]] = deque()
        self._putters: deque[asyncio.Future[None]] = deque()
        self._written = 0
        self._read = 0
        self._evicted = 0
        self._discarded = 0
        self._hwm_frames = 0
        self._hwm_bytes = 0

    # ------------------------------------------------------------------ introspection

    @property
    def capacity_frames(self) -> int:
        """Maximum number of queued frames."""
        return self._capacity_frames

    @property
    def capacity_bytes(self) -> int:
        """Maximum total queued payload bytes."""
        return self._capacity_bytes

    @property
    def policy(self) -> OverflowPolicy:
        """Overflow policy."""
        return self._policy

    @property
    def nbytes(self) -> int:
        """Payload bytes currently queued."""
        return self._nbytes

    @property
    def closed(self) -> bool:
        """Whether producers are refused."""
        return self._closed

    @property
    def epoch(self) -> int:
        """Fencing epoch; bumped by :meth:`abort` and :meth:`reset`."""
        return self._epoch

    def __len__(self) -> int:
        """Number of frames currently queued."""
        return self._size

    def stats(self) -> RingStats:
        """Snapshot of occupancy and lifetime counters."""
        return RingStats(
            frames=self._size,
            nbytes=self._nbytes,
            written=self._written,
            read=self._read,
            evicted=self._evicted,
            discarded=self._discarded,
            high_watermark_frames=self._hwm_frames,
            high_watermark_bytes=self._hwm_bytes,
            epoch=self._epoch,
            closed=self._closed,
        )

    @override
    def __repr__(self) -> str:
        return (
            f"AudioRingBuffer(frames={self._size}/{self._capacity_frames}, "
            f"bytes={self._nbytes}/{self._capacity_bytes}, policy={self._policy.value}, "
            f"epoch={self._epoch}, closed={self._closed})"
        )

    # ------------------------------------------------------------------ producer side

    def put_nowait(self, frame: AudioFrame) -> int:
        """Enqueue ``frame`` without suspending.

        Returns:
            Number of frames evicted to make room (always 0 under BACKPRESSURE).

        Raises:
            ValueError: if the frame can never fit the byte budget.
            RingClosedError: if the ring is closed.
            RingFullError: under BACKPRESSURE when there is no room.
        """
        self._admissible(frame)
        if self._closed:
            raise RingClosedError
        if self._policy is OverflowPolicy.BACKPRESSURE and not self._fits(frame):
            raise RingFullError
        return self._insert(frame)

    async def put(self, frame: AudioFrame) -> int:
        """Enqueue ``frame``, suspending under BACKPRESSURE until it fits.

        Cancellation-safe: a cancelled producer leaves no waiter behind and the frame is
        not enqueued.

        Returns:
            Number of frames evicted to make room.

        Raises:
            ValueError: if the frame can never fit the byte budget.
            RingClosedError: if the ring is closed, or aborted/reset while waiting.
        """
        self._admissible(frame)
        epoch = self._epoch
        while True:
            if self._closed or epoch != self._epoch:
                raise RingClosedError
            if self._policy is OverflowPolicy.DROP_OLDEST or self._fits(frame):
                return self._insert(frame)
            await self._wait(self._putters)

    # ------------------------------------------------------------------ consumer side

    def get_nowait(self) -> AudioFrame:
        """Dequeue the oldest frame without suspending.

        Raises:
            RingEmptyError: if the ring is empty and open.
            RingClosedError: if the ring is empty and closed.
        """
        if self._size == 0:
            raise RingClosedError if self._closed else RingEmptyError
        return self._pop()

    async def get(self) -> AudioFrame | None:
        """Dequeue the oldest frame, suspending while the ring is empty.

        Returns:
            The frame, or ``None`` at end-of-stream: the ring is closed and drained, or
            it was aborted/reset while this consumer was suspended.
        """
        epoch = self._epoch
        while True:
            if epoch != self._epoch:
                return None
            if self._size:
                return self._pop()
            if self._closed:
                return None
            await self._wait(self._getters)

    # ------------------------------------------------------------------ lifecycle

    def close(self) -> None:
        """Refuse new frames; consumers drain what is queued, then see end-of-stream."""
        self._closed = True
        self._wake(self._getters)
        self._wake(self._putters)

    def clear(self) -> int:
        """Drop every queued frame, releasing payload references. Returns the count."""
        dropped = self._size
        slots, mask, head = self._slots, self._mask, self._head
        for offset in range(dropped):
            slots[(head + offset) & mask] = None
        self._head = 0
        self._size = 0
        self._nbytes = 0
        self._discarded += dropped
        self._wake(self._putters)
        return dropped

    def abort(self) -> int:
        """Clear, close and fence: every suspended producer/consumer is released.

        Returns:
            Number of queued frames dropped.
        """
        self._epoch += 1
        self._closed = True
        dropped = self.clear()
        self._wake(self._getters)
        return dropped

    def reset(self) -> int:
        """Clear, reopen and fence: waiters from the previous epoch are released.

        Returns:
            Number of queued frames dropped.
        """
        self._epoch += 1
        self._closed = False
        dropped = self.clear()
        self._wake(self._getters)
        return dropped

    # ------------------------------------------------------------------ internals

    def _admissible(self, frame: AudioFrame) -> None:
        if frame.nbytes > self._capacity_bytes:
            msg = f"frame of {frame.nbytes} B exceeds ring budget of {self._capacity_bytes} B"
            raise ValueError(msg)

    def _fits(self, frame: AudioFrame) -> bool:
        return (
            self._size < self._capacity_frames
            and self._nbytes + frame.nbytes <= self._capacity_bytes
        )

    def _insert(self, frame: AudioFrame) -> int:
        evicted = 0
        evicted_bytes = 0
        while not self._fits(frame):
            evicted_bytes += self._pop_raw().nbytes
            evicted += 1
        self._slots[(self._head + self._size) & self._mask] = frame
        self._size += 1
        self._nbytes += frame.nbytes
        self._written += 1
        self._hwm_frames = max(self._hwm_frames, self._size)
        self._hwm_bytes = max(self._hwm_bytes, self._nbytes)
        if evicted:
            self._evicted += evicted
            self._sink.emit(
                BufferOverflow(
                    trace=frame.trace,
                    at_ns=frame.origin_ns,
                    evicted_frames=evicted,
                    evicted_bytes=evicted_bytes,
                )
            )
        self._wake(self._getters)
        return evicted

    def _pop_raw(self) -> AudioFrame:
        head = self._head
        frame = cast("AudioFrame", self._slots[head])
        self._slots[head] = None
        self._head = (head + 1) & self._mask
        self._size -= 1
        self._nbytes -= frame.nbytes
        return frame

    def _pop(self) -> AudioFrame:
        frame = self._pop_raw()
        self._read += 1
        self._wake(self._putters)
        return frame

    @staticmethod
    async def _wait(waiters: deque[asyncio.Future[None]]) -> None:
        waiter = asyncio.get_running_loop().create_future()
        waiters.append(waiter)
        try:
            await waiter
        finally:
            # Woken waiters were already popped by _wake; only a cancelled one may linger.
            # Checking first keeps the hot path cheap: a failed deque.remove() formats
            # repr(future) into its error message (CPython 3.12+), ~4 us per wake-up.
            if waiter.cancelled():
                with contextlib.suppress(ValueError):
                    waiters.remove(waiter)

    @staticmethod
    def _wake(waiters: deque[asyncio.Future[None]]) -> None:
        # Broadcast: every waiter re-checks its own predicate, so no wake-up can be
        # lost to a cancelled waiter. Waiter counts are tiny (1 producer, 1 consumer).
        while waiters:
            waiter = waiters.popleft()
            if not waiter.done():
                waiter.set_result(None)

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING

import pytest

from voice_ai_realtime_gateway.core.ring import (
    AudioRingBuffer,
    OverflowPolicy,
    RingClosedError,
    RingEmptyError,
    RingFullError,
)
from voice_ai_realtime_gateway.telemetry.events import BufferOverflow

if TYPE_CHECKING:
    from conftest import FrameFactory
    from voice_ai_realtime_gateway.telemetry.sink import FlightRecorder

pytestmark = pytest.mark.usefixtures("strict_loop")


def ring(
    frames: int = 4,
    nbytes: int = 4096,
    policy: OverflowPolicy = OverflowPolicy.BACKPRESSURE,
    sink: FlightRecorder | None = None,
) -> AudioRingBuffer:
    return AudioRingBuffer(capacity_frames=frames, capacity_bytes=nbytes, policy=policy, sink=sink)


async def settle() -> None:
    for _ in range(3):
        await asyncio.sleep(0)


# ----------------------------------------------------------------------- construction


@pytest.mark.parametrize("frames", [0, -1, 3, 6, 100])
async def test_capacity_frames_must_be_a_power_of_two(frames: int) -> None:
    with pytest.raises(ValueError, match="power of two"):
        ring(frames=frames)


async def test_capacity_bytes_must_be_positive() -> None:
    with pytest.raises(ValueError, match="capacity_bytes"):
        ring(nbytes=0)


async def test_introspection_and_repr() -> None:
    buf = ring(frames=8, nbytes=1000, policy=OverflowPolicy.DROP_OLDEST)
    assert (buf.capacity_frames, buf.capacity_bytes, buf.policy) == (8, 1000, "drop_oldest")
    assert (len(buf), buf.nbytes, buf.closed, buf.epoch) == (0, 0, False, 0)
    assert repr(buf) == (
        "AudioRingBuffer(frames=0/8, bytes=0/1000, policy=drop_oldest, epoch=0, closed=False)"
    )


# ----------------------------------------------------------------------- zero copy / FIFO


async def test_fifo_order_and_reference_identity(make_frame: FrameFactory) -> None:
    buf = ring()
    frames = [make_frame(16) for _ in range(4)]
    for frame in frames:
        assert buf.put_nowait(frame) == 0
    out = [buf.get_nowait() for _ in range(4)]
    assert all(a is b for a, b in zip(out, frames, strict=True)), "stored by reference"


async def test_wraparound_preserves_order(make_frame: FrameFactory) -> None:
    buf = ring(frames=4)
    seen = []
    for _ in range(25):
        frame = make_frame(8)
        buf.put_nowait(frame)
        seen.append(buf.get_nowait().sequence)
    assert seen == sorted(seen)
    assert buf.stats().written == buf.stats().read == 25


async def test_slots_release_references_on_consume_and_clear(make_frame: FrameFactory) -> None:
    buf = ring()
    frame = make_frame(8)
    buf.put_nowait(frame)
    buf.put_nowait(make_frame(8))
    baseline = sys.getrefcount(frame)
    assert buf.get_nowait() is frame
    assert buf._slots[0] is None, "consumed slot must not pin the frame"
    assert sys.getrefcount(frame) == baseline - 1
    assert buf.clear() == 1
    assert all(slot is None for slot in buf._slots)


# ----------------------------------------------------------------------- bounds


async def test_backpressure_put_nowait_rejects_on_frame_bound(make_frame: FrameFactory) -> None:
    buf = ring(frames=2)
    buf.put_nowait(make_frame(8))
    buf.put_nowait(make_frame(8))
    with pytest.raises(RingFullError):
        buf.put_nowait(make_frame(8))
    assert len(buf) == 2


async def test_backpressure_put_nowait_rejects_on_byte_bound(make_frame: FrameFactory) -> None:
    buf = ring(frames=8, nbytes=100)
    buf.put_nowait(make_frame(64))
    with pytest.raises(RingFullError):
        buf.put_nowait(make_frame(64))
    assert buf.nbytes == 64


async def test_oversized_frame_can_never_be_admitted(make_frame: FrameFactory) -> None:
    for policy in OverflowPolicy:
        buf = ring(nbytes=10, policy=policy)
        with pytest.raises(ValueError, match="exceeds ring budget"):
            buf.put_nowait(make_frame(12))
        with pytest.raises(ValueError, match="exceeds ring budget"):
            await buf.put(make_frame(12))


async def test_drop_oldest_evicts_until_both_bounds_hold(
    make_frame: FrameFactory, recorder: FlightRecorder
) -> None:
    buf = ring(frames=8, nbytes=100, policy=OverflowPolicy.DROP_OLDEST, sink=recorder)
    first, second, third = make_frame(40), make_frame(40), make_frame(90, origin_ns=7)
    buf.put_nowait(first)
    buf.put_nowait(second)
    assert buf.put_nowait(third) == 2
    assert buf.get_nowait() is third
    (overflow,) = recorder.of_type(BufferOverflow)
    assert (overflow.evicted_frames, overflow.evicted_bytes) == (2, 80)
    assert overflow.trace == third.trace
    assert overflow.at_ns == 7
    stats = buf.stats()
    assert stats.evicted == 2
    assert stats.read == 1


async def test_drop_oldest_async_put_never_blocks(make_frame: FrameFactory) -> None:
    buf = ring(frames=2, policy=OverflowPolicy.DROP_OLDEST)
    for _ in range(10):
        await asyncio.wait_for(buf.put(make_frame(8)), timeout=1)
    assert len(buf) == 2
    assert buf.stats().evicted == 8


async def test_high_watermarks_track_peak_occupancy(make_frame: FrameFactory) -> None:
    buf = ring(frames=8, nbytes=1000)
    for _ in range(3):
        buf.put_nowait(make_frame(100))
    for _ in range(3):
        buf.get_nowait()
    stats = buf.stats()
    assert (stats.high_watermark_frames, stats.high_watermark_bytes) == (3, 300)
    assert (stats.frames, stats.nbytes) == (0, 0)


# ----------------------------------------------------------------------- empty / closed


async def test_get_nowait_distinguishes_empty_from_closed(make_frame: FrameFactory) -> None:
    buf = ring()
    with pytest.raises(RingEmptyError):
        buf.get_nowait()
    buf.put_nowait(make_frame(8))
    buf.close()
    buf.get_nowait()
    with pytest.raises(RingClosedError):
        buf.get_nowait()


async def test_close_refuses_producers_but_drains_consumers(make_frame: FrameFactory) -> None:
    buf = ring()
    frame = make_frame(8)
    buf.put_nowait(frame)
    buf.close()
    with pytest.raises(RingClosedError):
        buf.put_nowait(make_frame(8))
    with pytest.raises(RingClosedError):
        await buf.put(make_frame(8))
    assert await buf.get() is frame
    assert await buf.get() is None


async def test_close_wakes_suspended_consumer_with_end_of_stream() -> None:
    buf = ring()
    consumer = asyncio.create_task(buf.get())
    await settle()
    buf.close()
    assert await consumer is None


async def test_close_wakes_suspended_producer_with_closed(make_frame: FrameFactory) -> None:
    buf = ring(frames=1)
    buf.put_nowait(make_frame(8))
    producer = asyncio.create_task(buf.put(make_frame(8)))
    await settle()
    buf.close()
    with pytest.raises(RingClosedError):
        await producer


# ----------------------------------------------------------------------- backpressure


async def test_put_suspends_until_consumer_frees_a_slot(make_frame: FrameFactory) -> None:
    buf = ring(frames=1)
    buf.put_nowait(make_frame(8))
    late = make_frame(8)
    producer = asyncio.create_task(buf.put(late))
    await settle()
    assert not producer.done()
    buf.get_nowait()
    assert await producer == 0
    assert buf.get_nowait() is late


async def test_put_suspends_until_byte_budget_frees(make_frame: FrameFactory) -> None:
    buf = ring(frames=8, nbytes=100)
    buf.put_nowait(make_frame(90))
    producer = asyncio.create_task(buf.put(make_frame(20)))
    await settle()
    assert not producer.done()
    assert buf.clear() == 1
    await producer
    assert buf.nbytes == 20


async def test_get_suspends_until_a_frame_arrives(make_frame: FrameFactory) -> None:
    buf = ring()
    consumer = asyncio.create_task(buf.get())
    await settle()
    assert not consumer.done()
    frame = make_frame(8)
    buf.put_nowait(frame)
    assert await consumer is frame


# ----------------------------------------------------------------------- epoch fencing


async def test_abort_releases_everyone_and_drops_queue(make_frame: FrameFactory) -> None:
    buf = ring(frames=1)
    consumer_buf = ring()
    buf.put_nowait(make_frame(8))
    producer = asyncio.create_task(buf.put(make_frame(8)))
    consumer = asyncio.create_task(consumer_buf.get())
    await settle()
    assert buf.abort() == 1
    assert consumer_buf.abort() == 0
    with pytest.raises(RingClosedError):
        await producer
    assert await consumer is None
    assert buf.closed
    assert buf.epoch == 1
    assert buf.stats().discarded == 1


async def test_reset_fences_stale_waiters_and_reopens(make_frame: FrameFactory) -> None:
    buf = ring(frames=1)
    buf.put_nowait(make_frame(8))
    stale_producer = asyncio.create_task(buf.put(make_frame(8)))
    await settle()
    assert buf.reset() == 1
    with pytest.raises(RingClosedError):
        await stale_producer
    assert len(buf) == 0, "a producer from the previous epoch must not leak a frame"
    assert not buf.closed
    fresh = make_frame(8)
    buf.put_nowait(fresh)
    assert buf.get_nowait() is fresh


async def test_reset_fences_stale_consumer_even_if_new_data_arrives(
    make_frame: FrameFactory,
) -> None:
    buf = ring()
    stale_consumer = asyncio.create_task(buf.get())
    await settle()
    buf.reset()
    buf.put_nowait(make_frame(8))  # new-epoch data lands before the stale consumer runs
    assert await stale_consumer is None
    assert len(buf) == 1, "new-epoch frame is reserved for new-epoch consumers"


# ----------------------------------------------------------------------- cancellation


async def test_cancelled_producer_leaves_no_waiter_and_no_frame(make_frame: FrameFactory) -> None:
    buf = ring(frames=1)
    buf.put_nowait(make_frame(8))
    producer = asyncio.create_task(buf.put(make_frame(8)))
    await settle()
    producer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await producer
    assert not buf._putters
    buf.get_nowait()
    assert len(buf) == 0


async def test_cancelled_consumer_leaves_no_waiter(make_frame: FrameFactory) -> None:
    buf = ring()
    consumer = asyncio.create_task(buf.get())
    await settle()
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer
    assert not buf._getters
    frame = make_frame(8)
    buf.put_nowait(frame)
    assert buf.get_nowait() is frame, "frame was not swallowed by the cancelled consumer"


async def test_wake_skips_waiters_cancelled_before_they_could_unlink(
    make_frame: FrameFactory,
) -> None:
    buf = ring(frames=1)
    buf.put_nowait(make_frame(8))
    cancelled = asyncio.create_task(buf.put(make_frame(8)))
    survivor_frame = make_frame(8)
    survivor = asyncio.create_task(buf.put(survivor_frame))
    await settle()
    cancelled.cancel()  # its waiter future is cancelled but still queued ...
    buf.get_nowait()  # ... when the wake-up broadcast runs.
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    await survivor
    assert buf.get_nowait() is survivor_frame


async def test_woken_then_cancelled_producer_does_not_lose_the_wakeup(
    make_frame: FrameFactory,
) -> None:
    buf = ring(frames=1)
    buf.put_nowait(make_frame(8))
    first = asyncio.create_task(buf.put(make_frame(8)))
    second_frame = make_frame(8)
    second = asyncio.create_task(buf.put(second_frame))
    await settle()
    buf.get_nowait()  # wakes both (broadcast)
    first.cancel()  # first is cancelled after being woken, before it runs
    with pytest.raises(asyncio.CancelledError):
        await first
    await second
    assert buf.get_nowait() is second_frame


# ------------------------------------------------------------------ mutation-derived contracts


async def test_smallest_valid_budget_and_fresh_watermarks() -> None:
    buf = ring(frames=1, nbytes=1)
    assert buf.capacity_bytes == 1
    stats = buf.stats()
    assert (stats.high_watermark_frames, stats.high_watermark_bytes) == (0, 0)


async def test_epoch_advances_by_exactly_one_per_fence() -> None:
    buf = ring()
    epochs = [buf.epoch]
    for fence in (buf.abort, buf.reset, buf.reset, buf.abort):
        fence()
        epochs.append(buf.epoch)
    assert epochs == [0, 1, 2, 3, 4]

"""Concurrency race tests: many producers/consumers under fuzzed interleavings."""

from __future__ import annotations

import asyncio
from collections import defaultdict

from hypothesis import given, settings
from hypothesis import strategies as st

from voice_ai_realtime_gateway.core.frames import AudioCodec, AudioFrame
from voice_ai_realtime_gateway.core.ring import AudioRingBuffer, OverflowPolicy, RingClosedError
from voice_ai_realtime_gateway.telemetry.ids import TraceContext

TRACE = TraceContext.new()


def frame(producer: int, sequence: int) -> AudioFrame:
    return AudioFrame.wrap(
        bytes(16),
        codec=AudioCodec.PCM16,
        sample_rate_hz=16_000,
        channels=1,
        duration_ns=20_000_000,
        sequence=sequence,
        origin_ns=producer,
        trace=TRACE,
    )


async def yield_n(times: int) -> None:
    for _ in range(times):
        await asyncio.sleep(0)


async def run_backpressure_scenario(
    producers: int, consumers: int, per_producer: int, capacity: int, jitter: list[int]
) -> tuple[dict[int, list[int]], int]:
    ring = AudioRingBuffer(
        capacity_frames=capacity, capacity_bytes=16 * capacity, policy=OverflowPolicy.BACKPRESSURE
    )
    received: dict[int, list[int]] = defaultdict(list)
    jitter_cycle = jitter or [0]

    async def produce(producer: int) -> None:
        for sequence in range(per_producer):
            await ring.put(frame(producer, sequence))
            await yield_n(jitter_cycle[(producer + sequence) % len(jitter_cycle)])

    async def consume(consumer: int) -> None:
        step = 0
        while (item := await ring.get()) is not None:
            received[item.origin_ns].append(item.sequence)
            assert len(ring) <= capacity
            step += 1
            await yield_n(jitter_cycle[(consumer + step) % len(jitter_cycle)])

    consumer_tasks = [asyncio.create_task(consume(c)) for c in range(consumers)]
    async with asyncio.TaskGroup() as group:
        for p in range(producers):
            group.create_task(produce(p))
    ring.close()
    await asyncio.gather(*consumer_tasks)
    return received, ring.stats().high_watermark_frames


@settings(max_examples=60)
@given(
    producers=st.integers(min_value=1, max_value=4),
    consumers=st.integers(min_value=1, max_value=4),
    per_producer=st.integers(min_value=0, max_value=40),
    log2_capacity=st.integers(min_value=0, max_value=3),
    jitter=st.lists(st.integers(min_value=0, max_value=3), max_size=8),
)
def test_backpressure_delivers_exactly_once_in_per_producer_order(
    producers: int, consumers: int, per_producer: int, log2_capacity: int, jitter: list[int]
) -> None:
    capacity = 1 << log2_capacity
    received, high_watermark = asyncio.run(
        run_backpressure_scenario(producers, consumers, per_producer, capacity, jitter)
    )
    assert high_watermark <= capacity
    for producer in range(producers):
        delivered = sorted(received.get(producer, []))
        assert delivered == list(range(per_producer)), "lost or duplicated frame"
    if consumers == 1:
        for producer in range(producers):
            assert received.get(producer, []) == list(range(per_producer)), "FIFO violated"


@settings(max_examples=40)
@given(
    frames_before_abort=st.integers(min_value=0, max_value=30),
    jitter=st.lists(st.integers(min_value=0, max_value=3), min_size=1, max_size=6),
)
def test_abort_mid_stream_never_leaks_old_epoch_frames(
    frames_before_abort: int, jitter: list[int]
) -> None:
    async def scenario() -> tuple[int, int]:
        ring = AudioRingBuffer(
            capacity_frames=2, capacity_bytes=1024, policy=OverflowPolicy.BACKPRESSURE
        )
        produced = 0

        async def produce() -> None:
            nonlocal produced
            try:
                while True:
                    await ring.put(frame(0, produced))
                    produced += 1
                    await yield_n(jitter[produced % len(jitter)])
            except RingClosedError:
                return

        async def consume() -> int:
            count = 0
            while await ring.get() is not None:
                count += 1
                if count == frames_before_abort:
                    ring.abort()
                await yield_n(jitter[count % len(jitter)])
            return count

        producer = asyncio.create_task(produce())
        consumer = asyncio.create_task(consume())
        if frames_before_abort == 0:
            await yield_n(2)
            ring.abort()
        consumed = await consumer
        await producer
        ring.reset()
        return consumed, len(ring)

    consumed, leftover = asyncio.run(scenario())
    if frames_before_abort:
        assert consumed == frames_before_abort, "nothing is delivered after the abort"
    assert leftover == 0, "no old-epoch frame survives into the next epoch"

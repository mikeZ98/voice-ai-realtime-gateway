"""In-memory WebSocket: ordering, flow control, close handshake and network loss."""

from __future__ import annotations

import asyncio

import pytest

from voice_ai_realtime_gateway.transport.errors import CloseCode, TransportClosedError
from voice_ai_realtime_gateway.transport.memory import MemoryWebSocket

pytestmark = pytest.mark.usefixtures("strict_loop")


async def test_capacity_must_be_positive() -> None:
    with pytest.raises(ValueError, match="capacity"):
        MemoryWebSocket.pair(capacity=0)


async def test_messages_flow_in_order_both_ways() -> None:
    left, right = MemoryWebSocket.pair()
    await left.send("a")
    await left.send(b"b")
    await right.send("c")
    assert right.pending == 2
    assert [await right.recv(), await right.recv(), await left.recv()] == ["a", b"b", "c"]


async def test_full_buffer_suspends_the_sender_like_tcp_flow_control() -> None:
    left, right = MemoryWebSocket.pair(capacity=1)
    await left.send("1")
    blocked = asyncio.create_task(left.send("2"))
    await asyncio.sleep(0)
    assert not blocked.done()
    assert await right.recv() == "1"
    await blocked
    assert await right.recv() == "2"


async def test_peer_drains_sent_messages_before_observing_the_close() -> None:
    left, right = MemoryWebSocket.pair()
    await left.send("last words")
    await left.close(CloseCode.GOING_AWAY, "bye")
    assert await right.recv() == "last words"
    with pytest.raises(TransportClosedError) as excinfo:
        await right.recv()
    assert (excinfo.value.code, excinfo.value.reason) == (1001, "bye")
    assert (right.closed, right.close_code, right.close_reason) == (True, 1001, "bye")


async def test_closing_side_discards_unread_input_and_refuses_sends() -> None:
    left, right = MemoryWebSocket.pair()
    await right.send("never read")
    await left.close()
    assert left.pending == 0
    with pytest.raises(TransportClosedError):
        await left.recv()
    with pytest.raises(TransportClosedError):
        await left.send("x")
    with pytest.raises(TransportClosedError):
        await right.send("x")
    await left.close(CloseCode.INTERNAL_ERROR)  # idempotent: first close wins
    assert left.close_code == 1000
    await right.close()  # already closed by the peer: no-op
    assert right.close_code == 1000


async def test_close_wakes_pending_receivers_and_blocked_senders() -> None:
    left, right = MemoryWebSocket.pair(capacity=1)
    await left.send("fill")  # right's buffer is now full
    blocked_sender = asyncio.create_task(left.send("overflow"))
    waiting_receiver = asyncio.create_task(left.recv())
    await asyncio.sleep(0)
    assert not blocked_sender.done()
    assert not waiting_receiver.done()
    await right.close(CloseCode.GOING_AWAY, "shutdown")
    with pytest.raises(TransportClosedError):
        await blocked_sender
    with pytest.raises(TransportClosedError) as excinfo:
        await waiting_receiver
    assert excinfo.value.code == 1001


async def test_sever_simulates_network_loss_on_both_ends() -> None:
    left, right = MemoryWebSocket.pair()
    await left.send("in flight")
    await right.sever()
    for end in (left, right):
        with pytest.raises(TransportClosedError) as excinfo:
            await end.recv()
        assert excinfo.value.code == CloseCode.ABNORMAL
        assert end.close_code == CloseCode.ABNORMAL


async def test_sever_after_close_keeps_the_original_close_code() -> None:
    left, right = MemoryWebSocket.pair()
    await left.close(CloseCode.NORMAL, "done")
    await left.sever()
    assert (left.close_code, right.close_code) == (1000, 1000)
    assert repr(left) == "MemoryWebSocket(closed=(1000, 'done'), pending=0)"
    unclosed, _ = MemoryWebSocket.pair()
    assert (unclosed.close_code, unclosed.close_reason) == (None, "")

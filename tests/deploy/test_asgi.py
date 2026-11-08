"""ASGI front end, exercised through the ASGI interface (no sockets, no ASGI server)."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from typing import TYPE_CHECKING, Any

import pytest

from audio_support import pcm_frame
from duplex_support import until
from voice_ai_realtime_gateway.deploy.app import GatewaySettings, build_app, demo_pipeline
from voice_ai_realtime_gateway.deploy.asgi import AsgiWebSocket, GatewayAsgiApp, request_headers
from voice_ai_realtime_gateway.deploy.lifecycle import DrainPhase
from voice_ai_realtime_gateway.production.admission import TokenSigner
from voice_ai_realtime_gateway.production.server import TenantPolicy
from voice_ai_realtime_gateway.transport.control import Hangup, encode_control
from voice_ai_realtime_gateway.transport.errors import TransportClosedError
from voice_ai_realtime_gateway.transport.framing import encode_packet

if TYPE_CHECKING:
    from voice_ai_realtime_gateway.telemetry.ids import TraceContext

pytestmark = pytest.mark.usefixtures("strict_loop")

type Message = dict[str, object]

KEY = b"z" * 32
DEADLINE = 5.0


class Peer:
    """An ASGI server's view of one connection, plus the remote client."""

    def __init__(self) -> None:
        self.inbox: asyncio.Queue[Message] = asyncio.Queue()
        self.outbox: asyncio.Queue[Message] = asyncio.Queue()

    async def receive(self) -> Message:
        return await self.inbox.get()

    async def send(self, message: Message) -> None:
        self.outbox.put_nowait(message)
        if message["type"] == "websocket.close":  # servers report the close back
            self.inbox.put_nowait(
                {"type": "websocket.disconnect", "code": message.get("code", 1000)}
            )

    async def out(self) -> Message:
        return await asyncio.wait_for(self.outbox.get(), DEADLINE)

    def drain_out(self) -> list[Message]:
        messages = []
        while not self.outbox.empty():
            messages.append(self.outbox.get_nowait())
        return messages


def make_app(**overrides: Any) -> GatewayAsgiApp:  # noqa: ANN401 -- settings overrides
    settings = GatewaySettings(
        keys={"k1": KEY},
        tenants={"acme": TenantPolicy(max_concurrent_sessions=5, max_session_duration_s=600)},
        final_scrape_grace_s=0.0,
        **overrides,
    )
    app = build_app(settings, demo_pipeline)
    app._signals = ()
    return app


def token() -> str:
    return TokenSigner("k1", KEY).sign(
        {
            "aud": "voice-gateway",
            "exp": 4_102_444_800,
            "tenant": "acme",
            "codecs": ["pcm16"],
            "max_duration_s": 300,
        }
    )


def scope(
    trace: TraceContext, *, path: str = "/v1/voice", query: str = "", auth: str | None = None
) -> Message:
    headers = [(b"x-session-id", trace.session_id.encode())]
    if auth is not None:
        headers.append((b"authorization", auth.encode()))
    return {"type": "websocket", "path": path, "headers": headers, "query_string": query.encode()}


async def test_full_call_over_asgi(trace: TraceContext) -> None:
    app, peer = make_app(), Peer()
    peer.inbox.put_nowait({"type": "websocket.connect"})
    task = asyncio.create_task(app(scope(trace, auth=f"Bearer {token()}"), peer.receive, peer.send))
    assert (await peer.out())["type"] == "websocket.accept"
    for sequence, amplitude in enumerate([1_000] * 4 + [0] * 16):
        frame = pcm_frame(trace, amplitude=amplitude, sequence=sequence)
        peer.inbox.put_nowait({"type": "websocket.receive", "bytes": encode_packet(frame)})
    reply = await peer.out()
    while "bytes" not in reply:
        reply = await peer.out()
    assert isinstance(reply["bytes"], bytes), "agent audio came back as a binary frame"
    peer.inbox.put_nowait({"type": "websocket.receive", "text": encode_control(Hangup("bye"))})
    await asyncio.wait_for(task, DEADLINE)
    closes = [m for m in peer.drain_out() if m["type"] == "websocket.close"]
    assert closes[-1]["code"] == 1000


async def test_credential_failures_are_refused_before_the_upgrade(trace: TraceContext) -> None:
    app, peer = make_app(), Peer()
    peer.inbox.put_nowait({"type": "websocket.connect"})
    await app(scope(trace, auth="Bearer forged.token.here"), peer.receive, peer.send)
    messages = peer.drain_out()
    assert [m["type"] for m in messages] == ["websocket.close"], "no accept: the server answers 403"
    assert messages[0]["code"] == 1008


async def test_draining_accepts_then_closes_with_try_again_later(trace: TraceContext) -> None:
    app, peer = make_app(), Peer()
    app.server.begin_drain()
    peer.inbox.put_nowait({"type": "websocket.connect"})
    await app(scope(trace, auth=f"Bearer {token()}"), peer.receive, peer.send)
    assert [(m["type"], m.get("code")) for m in peer.drain_out()] == [
        ("websocket.accept", None),
        ("websocket.close", 1013),
    ]


async def test_wrong_path_and_unexpected_first_message(trace: TraceContext) -> None:
    app, peer = make_app(), Peer()
    await app(scope(trace, path="/admin"), peer.receive, peer.send)
    assert peer.drain_out() == [{"type": "websocket.close", "code": 1008}]
    peer.inbox.put_nowait({"type": "websocket.disconnect"})
    await app(scope(trace), peer.receive, peer.send)
    assert peer.drain_out() == []


def test_query_string_credentials_for_browsers(trace: TraceContext) -> None:
    raw = scope(trace, query="access_token=abc.def.ghi&session_id=" + "1" * 32)
    raw["headers"] = [(b"X-Custom", b"v"), (b"bad",), ("str", "str")]
    headers = request_headers(raw)
    assert headers == {
        "x-custom": "v",
        "authorization": "Bearer abc.def.ghi",
        "x-session-id": "1" * 32,
    }
    explicit = request_headers(scope(trace, query="access_token=ignored", auth="Bearer real"))
    assert explicit["authorization"] == "Bearer real", "the header wins over the query string"
    assert request_headers({"type": "websocket"}) == {}


async def test_http_routes_go_to_the_status_app(trace: TraceContext) -> None:
    app, peer = make_app(), Peer()
    await app({"type": "http", "method": "GET", "path": "/drainz"}, peer.receive, peer.send)
    start = peer.drain_out()[0]
    assert start["status"] == 503, "not drained yet"
    del trace


async def test_lifespan_runs_the_reaper_and_drains_on_shutdown() -> None:
    app, peer = make_app(), Peer()
    peer.inbox.put_nowait({"type": "lifespan.startup"})
    peer.inbox.put_nowait({"type": "lifespan.unknown"})  # ignored
    task = asyncio.create_task(app({"type": "lifespan"}, peer.receive, peer.send))
    assert (await peer.out())["type"] == "lifespan.startup.complete"
    assert app._reaper_task is not None
    assert not app._reaper_task.done()
    peer.inbox.put_nowait({"type": "lifespan.shutdown"})
    assert (await peer.out())["type"] == "lifespan.shutdown.complete"
    await task
    assert app.drain.drained
    assert app._reaper_task.cancelled()


async def test_sigusr1_starts_the_drain_in_a_live_process() -> None:
    app, peer = make_app(), Peer()
    app._signals = (signal.SIGUSR1,)
    app._reaper = None
    peer.inbox.put_nowait({"type": "lifespan.startup"})
    task = asyncio.create_task(app({"type": "lifespan"}, peer.receive, peer.send))
    await peer.out()
    os.kill(os.getpid(), signal.SIGUSR1)  # what the preStop hook does
    await until(lambda: app.drain.phase is DrainPhase.DRAINED, spins=100_000)
    assert not app.server.ready
    peer.inbox.put_nowait({"type": "lifespan.shutdown"})
    await peer.out()
    await task


# ------------------------------------------------------------------------ connection adapter


async def test_asgi_websocket_adapter_contract() -> None:
    peer = Peer()
    connection = AsgiWebSocket(peer.receive, peer.send)
    peer.inbox.put_nowait({"type": "websocket.receive", "bytes": b"\x01"})
    peer.inbox.put_nowait({"type": "websocket.receive", "text": "hi"})
    assert [await connection.recv(), await connection.recv()] == [b"\x01", "hi"]
    await connection.send(b"\x02")
    await connection.send("yo")
    assert peer.drain_out() == [
        {"type": "websocket.send", "bytes": b"\x02"},
        {"type": "websocket.send", "text": "yo"},
    ]
    peer.inbox.put_nowait({"type": "websocket.disconnect", "code": 1001})
    with pytest.raises(TransportClosedError) as excinfo:
        await connection.recv()
    assert excinfo.value.code == 1001
    with pytest.raises(TransportClosedError):
        await connection.recv()
    with pytest.raises(TransportClosedError):
        await connection.send("late")
    await connection.close()
    assert peer.drain_out() == [], "closing an already closed socket sends nothing"


async def test_empty_receive_and_failing_sends_map_to_closed() -> None:
    peer = Peer()
    connection = AsgiWebSocket(peer.receive, peer.send)
    peer.inbox.put_nowait({"type": "websocket.receive"})
    with pytest.raises(TransportClosedError) as excinfo:
        await connection.recv()
    assert excinfo.value.code == 1005

    async def vanished(_message: Message) -> None:
        msg = "peer gone"
        raise OSError(msg)

    broken = AsgiWebSocket(peer.receive, vanished)
    with pytest.raises(TransportClosedError) as excinfo:
        await broken.send("x")
    assert excinfo.value.code == 1006
    closing = AsgiWebSocket(peer.receive, vanished)
    with contextlib.suppress(OSError):
        await closing.close(1000, "bye")  # never raises
    await closing.close()

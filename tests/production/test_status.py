"""ASGI status endpoint, exercised through the ASGI interface (no sockets)."""

from __future__ import annotations

import pytest

from voice_ai_realtime_gateway.production.status import StatusApp

pytestmark = pytest.mark.usefixtures("strict_loop")

type Message = dict[str, object]


async def request(
    app: StatusApp, path: str, method: str = "GET"
) -> tuple[int, dict[bytes, bytes], bytes]:
    sent: list[Message] = []

    async def receive() -> Message:
        return {"type": "http.request"}

    async def send(message: Message) -> None:
        sent.append(message)

    await app({"type": "http", "method": method, "path": path}, receive, send)
    start, body = sent
    headers = dict(start["headers"])  # type: ignore[call-overload]
    return start["status"], headers, body["body"]  # type: ignore[return-value]


async def test_health_readiness_and_metrics() -> None:
    state = {"ready": True}
    app = StatusApp(metrics=lambda: "m_total 1.0\n", ready=lambda: state["ready"])
    assert (await request(app, "/healthz"))[::2] == (200, b"ok\n")
    assert (await request(app, "/readyz"))[::2] == (200, b"ready\n")
    state["ready"] = False
    assert (await request(app, "/readyz"))[::2] == (503, b"draining\n")
    status, headers, body = await request(app, "/metrics")
    assert (status, body) == (200, b"m_total 1.0\n")
    assert headers[b"content-type"].startswith(b"text/plain; version=0.0.4")
    assert headers[b"content-length"] == b"12"
    assert headers[b"cache-control"] == b"no-store"


async def test_head_unknown_paths_and_methods() -> None:
    app = StatusApp(metrics=lambda: "", ready=lambda: True)
    status, headers, body = await request(app, "/healthz", "HEAD")
    assert (status, body, headers[b"content-length"]) == (200, b"", b"3")
    assert (await request(app, "/admin"))[0] == 404
    status, headers, _ = await request(app, "/metrics", "POST")
    assert (status, headers[b"allow"]) == (405, b"GET, HEAD")


async def test_lifespan_and_stray_websockets() -> None:
    app = StatusApp(metrics=lambda: "", ready=lambda: True)
    inbox: list[Message] = [
        {"type": "lifespan.startup"},
        {"type": "other"},
        {"type": "lifespan.shutdown"},
    ]
    sent: list[Message] = []

    async def receive() -> Message:
        return inbox.pop(0)

    async def send(message: Message) -> None:
        sent.append(message)

    await app({"type": "lifespan"}, receive, send)
    await app({"type": "websocket", "path": "/"}, receive, send)
    await app({"type": "unknown"}, receive, send)
    assert [m["type"] for m in sent] == [
        "lifespan.startup.complete",
        "lifespan.shutdown.complete",
        "websocket.close",
    ]

"""Zero-dependency ASGI status endpoint: ``/healthz``, ``/readyz``, ``/metrics``.

* ``GET /healthz`` -- liveness: 200 while the process serves requests.
* ``GET /readyz``  -- readiness: 200 while new calls are accepted; 503 while draining,
  so load balancers stop routing before shutdown.
* ``GET /metrics`` -- Prometheus text exposition.

``HEAD`` is supported; other methods get 405 with ``Allow``; unknown paths get 404.
Responses are ``no-store``. The ASGI ``lifespan`` protocol is acknowledged, and other
scope types (for example a WebSocket aimed at the status port) are closed.

Mount it on any ASGI server (uvicorn, hypercorn), typically on a separate port from
the media WebSocket.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, final

from voice_ai_realtime_gateway.telemetry.metrics import CONTENT_TYPE

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    type Message = dict[str, object]
    type Receive = Callable[[], Awaitable[Message]]
    type Send = Callable[[Message], Awaitable[None]]

__all__ = ["StatusApp"]

_TEXT: Final = b"text/plain; charset=utf-8"
_ALLOWED: Final = frozenset({"GET", "HEAD"})
_HTTP_OK: Final = 200
_HTTP_NOT_FOUND: Final = 404
_HTTP_METHOD_NOT_ALLOWED: Final = 405
_HTTP_UNAVAILABLE: Final = 503


@final
class StatusApp:
    """ASGI 3 application exposing health, readiness and metrics."""

    __slots__ = ("_metrics", "_ready")

    def __init__(self, *, metrics: Callable[[], str], ready: Callable[[], bool]) -> None:
        """Serve ``metrics()`` on ``/metrics`` and readiness from ``ready()``."""
        self._metrics = metrics
        self._ready = ready

    async def __call__(self, scope: Mapping[str, object], receive: Receive, send: Send) -> None:
        """Handle one ASGI connection."""
        kind = scope.get("type")
        if kind == "lifespan":
            await self._lifespan(receive, send)
        elif kind == "http":
            await self._http(scope, send)
        elif kind == "websocket":
            await send({"type": "websocket.close", "code": 1008})

    @staticmethod
    async def _lifespan(receive: Receive, send: Send) -> None:
        while True:
            message = await receive()
            if message.get("type") == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            elif message.get("type") == "lifespan.shutdown":
                await send({"type": "lifespan.shutdown.complete"})
                return

    async def _http(self, scope: Mapping[str, object], send: Send) -> None:
        method = scope.get("method")
        path = scope.get("path")
        headers: list[tuple[bytes, bytes]] = [(b"cache-control", b"no-store")]
        if method not in _ALLOWED:
            status, body, content_type = _HTTP_METHOD_NOT_ALLOWED, b"method not allowed\n", _TEXT
            headers.append((b"allow", b"GET, HEAD"))
        elif path == "/healthz":
            status, body, content_type = _HTTP_OK, b"ok\n", _TEXT
        elif path == "/readyz":
            ready = self._ready()
            status = _HTTP_OK if ready else _HTTP_UNAVAILABLE
            body, content_type = (b"ready\n" if ready else b"draining\n"), _TEXT
        elif path == "/metrics":
            status, body, content_type = _HTTP_OK, self._metrics().encode(), CONTENT_TYPE.encode()
        else:
            status, body, content_type = _HTTP_NOT_FOUND, b"not found\n", _TEXT
        headers += [(b"content-type", content_type), (b"content-length", str(len(body)).encode())]
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": b"" if method == "HEAD" else body})

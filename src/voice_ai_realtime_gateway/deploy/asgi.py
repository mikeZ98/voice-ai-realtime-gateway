"""ASGI 3 front end: WebSocket media, HTTP status and lifespan in one app.

Runs under any ASGI server with WebSocket support (uvicorn, hypercorn, daphne). The
gateway itself stays dependency-free; the server is a deployment choice.

* ``websocket`` on ``websocket_path``: admission runs **before** the upgrade.
  Credential and policy failures close pre-accept, which ASGI servers answer with HTTP
  403, so no WebSocket is ever established for them. Capacity, drain and open-circuit
  refusals (``1013``) are accepted and then closed with ``1013``, so the client
  receives the retry signal. Browsers cannot set ``Authorization`` on a WebSocket, so
  ``?access_token=`` (and ``?session_id=``) are accepted as fallbacks.
* ``http``: :class:`~..production.status.StatusApp` (``/healthz``, ``/readyz``,
  ``/drainz``, ``/metrics``).
* ``lifespan``: startup binds the drain signals and starts the zombie reaper; shutdown
  drains (if a ``preStop`` hook has not already) and stops the reaper.

:class:`AsgiWebSocket` adapts an accepted ASGI WebSocket to the gateway's
``WebSocketConnection`` port.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, final
from urllib.parse import parse_qs

from voice_ai_realtime_gateway.deploy.lifecycle import DEFAULT_DRAIN_SIGNALS
from voice_ai_realtime_gateway.production.admission import AdmissionError
from voice_ai_realtime_gateway.telemetry.ids import SESSION_ID_HEADER
from voice_ai_realtime_gateway.transport.errors import CloseCode, TransportClosedError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping, Sequence

    from voice_ai_realtime_gateway.deploy.lifecycle import DrainController
    from voice_ai_realtime_gateway.production.reaper import ZombieReaper
    from voice_ai_realtime_gateway.production.server import GatewayServer
    from voice_ai_realtime_gateway.production.status import StatusApp

    type Message = dict[str, object]
    type Receive = Callable[[], Awaitable[Message]]
    type Send = Callable[[Message], Awaitable[None]]

__all__ = ["AsgiWebSocket", "GatewayAsgiApp", "request_headers"]

_NO_STATUS: int = 1005


@final
class AsgiWebSocket:
    """``WebSocketConnection`` over an accepted ASGI WebSocket."""

    __slots__ = ("_closed", "_receive", "_send")

    def __init__(self, receive: Receive, send: Send) -> None:
        """Wrap the ASGI ``receive`` / ``send`` callables of one connection."""
        self._receive = receive
        self._send = send
        self._closed: tuple[int, str] | None = None

    async def recv(self) -> str | bytes:
        """Next client message.

        Raises:
            TransportClosedError: once the client disconnected or the socket was closed.
        """
        if self._closed is not None:
            raise TransportClosedError(*self._closed)
        message = await self._receive()
        if message.get("type") == "websocket.receive":
            data, text = message.get("bytes"), message.get("text")
            if isinstance(data, bytes):
                return data
            if isinstance(text, str):
                return text
        code = message.get("code")
        self._closed = (code if isinstance(code, int) else _NO_STATUS, "")
        raise TransportClosedError(*self._closed)

    async def send(self, message: str | bytes, /) -> None:
        """Send one message.

        Raises:
            TransportClosedError: if the socket is closed or the server refused the send.
        """
        if self._closed is not None:
            raise TransportClosedError(*self._closed)
        key = "bytes" if isinstance(message, bytes) else "text"
        try:
            await self._send({"type": "websocket.send", key: message})
        except (OSError, RuntimeError) as exc:  # servers signal a vanished peer this way
            self._closed = (CloseCode.ABNORMAL, "")
            raise TransportClosedError(CloseCode.ABNORMAL, "send failed") from exc

    async def close(self, code: int = CloseCode.NORMAL, reason: str = "") -> None:
        """Close the socket (idempotent, never raises)."""
        if self._closed is not None:
            return
        self._closed = (code, reason)
        with contextlib.suppress(OSError, RuntimeError):
            await self._send({"type": "websocket.close", "code": code, "reason": reason})


def request_headers(scope: Mapping[str, object]) -> dict[str, str]:
    """Request headers of an ASGI scope, with query-string credential fallbacks."""
    headers: dict[str, str] = {}
    raw_headers = scope.get("headers")
    for pair in raw_headers if isinstance(raw_headers, list | tuple) else ():
        if isinstance(pair, list | tuple) and len(pair) == 2:  # noqa: PLR2004 -- (name, value)
            name, value = pair
            if isinstance(name, bytes) and isinstance(value, bytes):
                headers[name.decode("latin-1").lower()] = value.decode("latin-1")
    query = scope.get("query_string")
    params = parse_qs(query.decode("latin-1")) if isinstance(query, bytes) else {}
    if "authorization" not in headers and params.get("access_token"):
        headers["authorization"] = f"Bearer {params['access_token'][0]}"
    if SESSION_ID_HEADER.lower() not in headers and params.get("session_id"):
        headers[SESSION_ID_HEADER.lower()] = params["session_id"][0]
    return headers


@final
class GatewayAsgiApp:
    """The deployable gateway: media WebSocket + status + lifecycle."""

    __slots__ = ("_drain", "_path", "_reaper", "_reaper_task", "_server", "_signals", "_status")

    def __init__(  # noqa: PLR0913 -- the composition of one deployable process
        self,
        *,
        server: GatewayServer,
        status: StatusApp,
        drain: DrainController,
        reaper: ZombieReaper | None = None,
        websocket_path: str = "/v1/voice",
        drain_signals: Sequence[int] = DEFAULT_DRAIN_SIGNALS,
    ) -> None:
        """Serve calls on ``websocket_path``; drain on any of ``drain_signals``."""
        self._server = server
        self._status = status
        self._drain = drain
        self._reaper = reaper
        self._path = websocket_path
        self._signals = tuple(drain_signals)
        self._reaper_task: asyncio.Task[None] | None = None

    @property
    def server(self) -> GatewayServer:
        """The gateway server."""
        return self._server

    @property
    def drain(self) -> DrainController:
        """The drain controller."""
        return self._drain

    async def __call__(self, scope: Mapping[str, object], receive: Receive, send: Send) -> None:
        """Dispatch one ASGI connection."""
        kind = scope.get("type")
        if kind == "websocket":
            await self._websocket(scope, receive, send)
        elif kind == "lifespan":
            await self._lifespan(receive, send)
        else:
            await self._status(scope, receive, send)

    async def _websocket(self, scope: Mapping[str, object], receive: Receive, send: Send) -> None:
        if scope.get("path") != self._path:
            await send({"type": "websocket.close", "code": CloseCode.POLICY_VIOLATION})
            return
        if (await receive()).get("type") != "websocket.connect":
            return
        connection = AsgiWebSocket(receive, send)
        try:
            admission = self._server.admit(request_headers(scope))
        except AdmissionError as refusal:
            if refusal.code == CloseCode.TRY_AGAIN_LATER:
                await send({"type": "websocket.accept"})  # so the client sees 1013
            await self._server.refuse(connection, refusal)
            return
        await send({"type": "websocket.accept"})
        await self._server.serve(connection, admission)

    async def _lifespan(self, receive: Receive, send: Send) -> None:
        loop = asyncio.get_running_loop()
        while True:
            message = await receive()
            kind = message.get("type")
            if kind == "lifespan.startup":
                self._drain.install(loop, self._signals)
                if self._reaper is not None:
                    self._reaper_task = loop.create_task(self._reaper.run(), name="zombie-reaper")
                await send({"type": "lifespan.startup.complete"})
            elif kind == "lifespan.shutdown":
                await self._drain.drain()
                self._drain.uninstall(loop)
                if self._reaper_task is not None:
                    self._reaper_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await self._reaper_task
                await send({"type": "lifespan.shutdown.complete"})
                return

"""OpenAI Realtime API adapter (``StreamingLLM``, text output).

Each turn is one **out-of-band** response over a persistent Realtime WebSocket::

    -> {"type": "response.create", "response": {
           "conversation": "none",           # do not mutate server-side state
           "output_modalities": ["text"],
           "instructions": "...",
           "metadata": {"gateway_tag": "<tag>"},
           "input": [ ...message items rebuilt from ConversationState... ]}}
    <- response.created (metadata.gateway_tag = <tag>)  -> remember response.id
    <- response.output_text.delta {response_id, delta}  -> TokenDelta
       (the beta name ``response.text.delta`` is accepted too)
    <- response.done {response: {id, status, status_details}}
    <- error {error: {type, code, message}}

With ``conversation: "none"`` and an explicit ``input`` array, the gateway's
``ConversationState`` stays the single source of truth. That matters after a barge-in
truncated the previous agent turn: the provider never holds the full, unheard reply.
Events of other responses on the shared connection (for example a cancelled previous
turn) are ignored by response id.

Resilience:

* Drop or retryable error **before the first token**: reconnect under the
  :class:`~.resilience.RetryPolicy` and resend the request transparently.
* Drop or retryable error **after tokens were yielded**: tokens already spoken cannot
  be retracted, so the reply ends gracefully (a shorter answer beats a faulted call),
  and a ``ProviderError`` is reported.
* ``invalid_api_key`` / authentication errors and ``insufficient_quota`` are fatal.
* Closing the stream early (barge-in) sends ``response.cancel`` for the active response.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import secrets
import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, NoReturn, final

from voice_ai_realtime_gateway.adapters.resilience import (
    ErrorKind,
    ProviderLink,
    RetryPolicy,
    UpstreamError,
)
from voice_ai_realtime_gateway.inference.conversation import AgentTurn, UserTurn
from voice_ai_realtime_gateway.inference.ports import TokenDelta
from voice_ai_realtime_gateway.transport.errors import TransportClosedError

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Mapping, Sequence

    from voice_ai_realtime_gateway.adapters.resilience import Connector, Sleeper
    from voice_ai_realtime_gateway.domain.clock import MonotonicClock
    from voice_ai_realtime_gateway.inference.conversation import Turn
    from voice_ai_realtime_gateway.telemetry.ids import TraceContext
    from voice_ai_realtime_gateway.telemetry.sink import TelemetrySink
    from voice_ai_realtime_gateway.transport.base import WebSocketConnection

__all__ = [
    "PROVIDER",
    "OpenAIRealtimeLLM",
    "ResponseCreated",
    "ResponseDone",
    "ServerError",
    "TextDelta",
    "build_response_request",
    "classify_openai_error",
    "parse_openai_event",
]

PROVIDER: Final = "openai"
_TEXT_DELTA_TYPES: Final = frozenset({"response.output_text.delta", "response.text.delta"})
_ERROR_KINDS: Final[Mapping[str, ErrorKind]] = MappingProxyType(
    {
        "invalid_api_key": ErrorKind.AUTH,
        "authentication_error": ErrorKind.AUTH,
        "permission_denied": ErrorKind.AUTH,
        "insufficient_quota": ErrorKind.QUOTA,
        "rate_limit_exceeded": ErrorKind.RATE_LIMIT,
        "server_error": ErrorKind.TRANSIENT,
        "server_is_overloaded": ErrorKind.TRANSIENT,
    }
)


# --------------------------------------------------------------------------- wire format


def _item(role: str, content_type: str, text: str) -> dict[str, object]:
    return {"type": "message", "role": role, "content": [{"type": content_type, "text": text}]}


def _history_item(turn: Turn) -> dict[str, object]:
    if isinstance(turn, UserTurn):
        return _item("user", "input_text", turn.text)
    if isinstance(turn, AgentTurn):
        return _item("assistant", "output_text", turn.text)
    note = f"Tool {turn.name}({turn.arguments}) returned: {turn.result}"
    return _item("system", "input_text", note)


def build_response_request(
    history: Sequence[Turn], user_text: str, *, instructions: str, tag: str
) -> str:
    """The ``response.create`` event for one turn (compact JSON)."""
    items = [_history_item(turn) for turn in history]
    items.append(_item("user", "input_text", user_text))
    request = {
        "type": "response.create",
        "response": {
            "conversation": "none",
            "output_modalities": ["text"],
            "instructions": instructions,
            "metadata": {"gateway_tag": tag},
            "input": items,
        },
    }
    return json.dumps(request, separators=(",", ":"))


@dataclass(frozen=True, slots=True, kw_only=True)
class ResponseCreated:
    """``response.created``."""

    response_id: str
    tag: str | None


@dataclass(frozen=True, slots=True, kw_only=True)
class TextDelta:
    """``response.output_text.delta``."""

    response_id: str
    delta: str


@dataclass(frozen=True, slots=True, kw_only=True)
class ResponseDone:
    """``response.done``; ``error_code`` set when the response failed."""

    response_id: str
    status: str
    error_code: str | None


@dataclass(frozen=True, slots=True, kw_only=True)
class ServerError:
    """``error``."""

    type: str
    code: str | None
    message: str


type OpenAIEvent = ResponseCreated | TextDelta | ResponseDone | ServerError


def _malformed(detail: str) -> NoReturn:
    raise UpstreamError(ErrorKind.PROTOCOL, f"malformed event: {detail}", provider=PROVIDER)


def _object(value: object, what: str) -> dict[str, object]:
    if not isinstance(value, dict):
        _malformed(f"{what} must be an object")
    return value


def _string(document: dict[str, object], key: str) -> str:
    value = document.get(key)
    if not isinstance(value, str):
        _malformed(f"{key!r} must be a string")
    return value


def _optional_string(document: dict[str, object], key: str) -> str | None:
    value = document.get(key)
    if value is not None and not isinstance(value, str):
        _malformed(f"{key!r} must be a string or null")
    return value


def parse_openai_event(text: str) -> OpenAIEvent | None:
    """Parse one server event; ``None`` for event types the adapter ignores.

    Raises:
        UpstreamError: ``PROTOCOL`` for anything malformed.
    """
    try:
        document = json.loads(text)
    except (ValueError, RecursionError):
        _malformed("not JSON")
    event = _object(document, "event")
    kind = event.get("type")
    if kind == "response.created":
        response = _object(event.get("response"), "response")
        metadata = response.get("metadata")
        tag = _optional_string(metadata, "gateway_tag") if isinstance(metadata, dict) else None
        return ResponseCreated(response_id=_string(response, "id"), tag=tag)
    if kind in _TEXT_DELTA_TYPES:
        return TextDelta(response_id=_string(event, "response_id"), delta=_string(event, "delta"))
    if kind == "response.done":
        response = _object(event.get("response"), "response")
        details = response.get("status_details")
        error = details.get("error") if isinstance(details, dict) else None
        code = _optional_string(error, "code") if isinstance(error, dict) else None
        return ResponseDone(
            response_id=_string(response, "id"), status=_string(response, "status"), error_code=code
        )
    if kind == "error":
        error = _object(event.get("error"), "error")
        return ServerError(
            type=_string(error, "type"),
            code=_optional_string(error, "code"),
            message=_optional_string(error, "message") or "",
        )
    return None


def classify_openai_error(code: str | None, error_type: str | None = None) -> ErrorKind:
    """Map an OpenAI error ``code`` / ``type`` onto the upstream taxonomy."""
    for key in (code, error_type):
        if key is not None and key in _ERROR_KINDS:
            return _ERROR_KINDS[key]
    return ErrorKind.PROTOCOL


# --------------------------------------------------------------------------- adapter


@final
class _Turn:
    """State of the response currently being streamed."""

    __slots__ = ("done", "response_id", "tag", "yielded")

    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.response_id: str | None = None
        self.yielded = 0
        self.done = False


@final
class OpenAIRealtimeLLM:
    """Streams text replies from the OpenAI Realtime API over a reused connection."""

    __slots__ = (
        "_clock",
        "_connection",
        "_connector",
        "_instructions",
        "_jitter",
        "_policy",
        "_sink",
        "_sleep",
        "_trace",
    )

    def __init__(  # noqa: PLR0913 -- every collaborator is injectable for determinism
        self,
        connector: Connector,
        *,
        sink: TelemetrySink,
        trace: TraceContext,
        instructions: str = "",
        policy: RetryPolicy | None = None,
        clock: MonotonicClock = time.monotonic_ns,
        sleep: Sleeper = asyncio.sleep,
        jitter: Callable[[], float] | None = None,
    ) -> None:
        """Bind to ``connector``; ``instructions`` is the system prompt for every turn."""
        self._connector = connector
        self._sink = sink
        self._trace = trace
        self._instructions = instructions
        self._policy = policy
        self._clock = clock
        self._sleep = sleep
        self._jitter = jitter
        self._connection: WebSocketConnection | None = None

    async def aclose(self) -> None:
        """Close the persistent connection (end of call). Idempotent."""
        connection, self._connection = self._connection, None
        if connection is not None:
            await connection.close()

    async def stream(
        self, history: Sequence[Turn], user_text: str, /
    ) -> AsyncGenerator[TokenDelta, None]:
        """Yield the reply to ``user_text`` given ``history`` as token deltas."""
        link = ProviderLink(
            provider=PROVIDER,
            connector=self._connector,
            sink=self._sink,
            trace=self._trace,
            policy=self._policy,
            clock=self._clock,
            sleep=self._sleep,
            jitter=self._jitter,
        )
        turn = _Turn(secrets.token_hex(8))
        request = build_response_request(
            history, user_text, instructions=self._instructions, tag=turn.tag
        )
        connection = self._connection if self._connection is not None else await link.open()
        self._connection = connection
        try:
            while True:
                try:
                    await connection.send(request)
                    async with contextlib.aclosing(self._receive(connection, turn)) as deltas:
                        async for delta in deltas:
                            yield delta
                except (TransportClosedError, UpstreamError) as failure:
                    # Every failure goes through the link, so retries always pay backoff
                    # and consume the reconnect budget; a rate-limited provider is never
                    # hammered in a tight loop.
                    error = link.classify(failure)
                    self._connection = None
                    await connection.close()
                    if error.fatal or turn.yielded:
                        link.fail(failure)
                        if error.fatal:
                            raise error from failure
                        return  # already speaking: end the reply gracefully
                    connection = await link.reopen(failure)
                    self._connection = connection
                    turn.response_id, turn.done = None, False
                    continue
                return
        finally:
            if not turn.done and turn.response_id is not None and self._connection is not None:
                await self._cancel(self._connection, turn.response_id)

    async def _receive(
        self, connection: WebSocketConnection, turn: _Turn
    ) -> AsyncGenerator[TokenDelta, None]:
        while not turn.done:
            message = await connection.recv()
            if isinstance(message, bytes):
                continue
            delta = self._apply(parse_openai_event(message), turn)
            if delta is not None:
                yield delta

    @staticmethod
    def _apply(event: OpenAIEvent | None, turn: _Turn) -> TokenDelta | None:
        """Advance ``turn`` by one server event; returns a delta to yield, if any."""
        if isinstance(event, ResponseCreated):
            if turn.response_id is None and event.tag in {turn.tag, None}:
                turn.response_id = event.response_id
        elif isinstance(event, TextDelta):
            if event.response_id == turn.response_id:
                turn.yielded += 1
                return TokenDelta(text=event.delta, index=turn.yielded - 1)
        elif isinstance(event, ResponseDone):
            if event.response_id == turn.response_id:
                turn.done = True
                if event.status == "failed":
                    kind = classify_openai_error(event.error_code)
                    detail = f"response failed: {event.error_code}"
                    raise UpstreamError(kind, detail, provider=PROVIDER)
        elif isinstance(event, ServerError):
            kind = classify_openai_error(event.code, event.type)
            raise UpstreamError(kind, event.message or event.type, provider=PROVIDER)
        return None

    @staticmethod
    async def _cancel(connection: WebSocketConnection, response_id: str) -> None:
        cancel = json.dumps({"type": "response.cancel", "response_id": response_id})
        try:
            await connection.send(cancel)
        except TransportClosedError:
            return

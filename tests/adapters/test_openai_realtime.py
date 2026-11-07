"""OpenAI Realtime adapter: request shape, event parsing, retries, graceful truncation."""

from __future__ import annotations

import contextlib
import json
from typing import TYPE_CHECKING

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from provider_support import FakeProvider, instant_sleep
from voice_ai_realtime_gateway.adapters.openai_realtime import (
    OpenAIRealtimeLLM,
    ResponseDone,
    ServerError,
    classify_openai_error,
    parse_openai_event,
)
from voice_ai_realtime_gateway.adapters.resilience import ErrorKind, RetryPolicy, UpstreamError
from voice_ai_realtime_gateway.inference.conversation import AgentTurn, ToolCallTurn, UserTurn
from voice_ai_realtime_gateway.telemetry.events import ProviderError
from voice_ai_realtime_gateway.transport.errors import CloseCode

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from voice_ai_realtime_gateway.inference.conversation import Turn
    from voice_ai_realtime_gateway.telemetry.ids import TraceContext
    from voice_ai_realtime_gateway.telemetry.sink import FlightRecorder
    from voice_ai_realtime_gateway.transport.memory import MemoryWebSocket

pytestmark = pytest.mark.usefixtures("strict_loop")

TOKENS = ["Sure", ",", " the", " weather", " is", " mild."]


def created(rid: str, tag: str | None) -> str:
    response: dict[str, object] = {"id": rid}
    if tag is not None:
        response["metadata"] = {"gateway_tag": tag}
    return json.dumps({"type": "response.created", "response": response})


def delta(rid: str, text: str, *, beta: bool = False) -> str:
    kind = "response.text.delta" if beta else "response.output_text.delta"
    return json.dumps({"type": kind, "response_id": rid, "delta": text})


def done(rid: str, status: str = "completed", code: str | None = None) -> str:
    details = {"error": {"code": code}} if code else None
    return json.dumps(
        {
            "type": "response.done",
            "response": {"id": rid, "status": status, "status_details": details},
        }
    )


def error(code: str | None, kind: str = "invalid_request_error") -> str:
    return json.dumps(
        {"type": "error", "error": {"type": kind, "code": code, "message": f"err {code}"}}
    )


type Script = Callable[[MemoryWebSocket, int, int, str, str], Awaitable[bool]]
"""(ws, connection, request number, response id, tag) -> keep serving?"""


async def stream_all(ws: MemoryWebSocket, _c: int, _n: int, rid: str, tag: str) -> bool:
    await ws.send(created(rid, tag))
    for token in TOKENS:
        await ws.send(delta(rid, token))
    await ws.send(done(rid))
    return True


def realtime(script: Script = stream_all, *refusals: BaseException) -> FakeProvider:
    log: list[dict[str, object]] = []

    async def handler(ws: MemoryWebSocket, connection: int) -> None:
        number = 0
        while True:
            event = json.loads(await ws.recv())
            log.append(event)
            if event["type"] != "response.create":
                continue
            tag = event["response"]["metadata"]["gateway_tag"]
            if not await script(ws, connection, number, f"resp_{connection}_{number}", tag):
                return
            number += 1

    provider = FakeProvider(handler, *refusals)
    provider.log = log  # type: ignore[attr-defined]
    return provider


def llm(
    provider: FakeProvider, recorder: FlightRecorder, trace: TraceContext, **kwargs: object
) -> OpenAIRealtimeLLM:
    return OpenAIRealtimeLLM(
        provider, sink=recorder, trace=trace, instructions="Be brief.",
        sleep=instant_sleep, jitter=lambda: 0.5, **kwargs,  # type: ignore[arg-type]
    )  # fmt: skip


async def reply(adapter: OpenAIRealtimeLLM, history: list[Turn] | None = None) -> str:
    return "".join([d.text async for d in adapter.stream(history or [], "weather?")])


# ------------------------------------------------------------------------ happy path


async def test_out_of_band_request_carries_the_gateway_history(
    recorder: FlightRecorder, trace: TraceContext
) -> None:
    provider = realtime()
    adapter = llm(provider, recorder, trace)
    history: list[Turn] = [
        UserTurn(text="hi", trace=trace, at_ns=0),
        AgentTurn(text="Hello,", trace=trace, at_ns=0, interrupted=True, generated_chars=40),
        ToolCallTurn(
            name="weather", arguments='{"city":"Oslo"}', result="4C", trace=trace, at_ns=0
        ),
    ]
    assert await reply(adapter, history) == "".join(TOKENS)
    (request,) = provider.log  # type: ignore[attr-defined]
    body = request["response"]
    assert (body["conversation"], body["output_modalities"], body["instructions"]) == (
        "none", ["text"], "Be brief."
    )  # fmt: skip
    assert [
        (i["role"], i["content"][0]["type"], i["content"][0]["text"]) for i in body["input"]
    ] == [
        ("user", "input_text", "hi"),
        ("assistant", "output_text", "Hello,"),  # the truncated turn, not the full reply
        ("system", "input_text", 'Tool weather({"city":"Oslo"}) returned: 4C'),
        ("user", "input_text", "weather?"),
    ]
    await adapter.aclose()
    await adapter.aclose()
    await provider.finish()


async def test_connection_is_reused_across_turns(
    recorder: FlightRecorder, trace: TraceContext
) -> None:
    provider = realtime()
    adapter = llm(provider, recorder, trace)
    assert await reply(adapter) == await reply(adapter) == "".join(TOKENS)
    assert provider.connections == 1
    await adapter.aclose()
    await provider.finish()


async def test_foreign_events_and_beta_names_are_handled(
    recorder: FlightRecorder, trace: TraceContext
) -> None:
    async def noisy(ws: MemoryWebSocket, _c: int, _n: int, rid: str, tag: str) -> bool:
        await ws.send(created("resp_other", "someone-else"))
        await ws.send(delta("resp_other", "IGNORED"))
        await ws.send(created(rid, tag))
        await ws.send(b"\x00")
        await ws.send('{"type":"rate_limits.updated","rate_limits":[]}')
        await ws.send(delta(rid, "Beta", beta=True))
        await ws.send(done("resp_other"))
        await ws.send(delta(rid, " ok"))
        await ws.send(done(rid))
        return True

    provider = realtime(noisy)
    adapter = llm(provider, recorder, trace)
    assert await reply(adapter) == "Beta ok"
    await adapter.aclose()
    await provider.finish()


async def test_response_created_without_metadata_is_adopted(
    recorder: FlightRecorder, trace: TraceContext
) -> None:
    async def untagged(ws: MemoryWebSocket, _c: int, _n: int, rid: str, _tag: str) -> bool:
        await ws.send(created(rid, None))
        await ws.send(delta(rid, "fine"))
        await ws.send(done(rid, "incomplete"))
        return True

    provider = realtime(untagged)
    adapter = llm(provider, recorder, trace)
    assert await reply(adapter) == "fine"
    await adapter.aclose()
    await provider.finish()


# ------------------------------------------------------------------------ resilience


def chaos(plan: dict[int, str]) -> Script:
    async def script(ws: MemoryWebSocket, connection: int, number: int, rid: str, tag: str) -> bool:
        action = plan.get(connection, "ok") if number == 0 else "ok"
        if action == "drop_before":
            await ws.sever()
            return False
        if action == "garbage_before":
            await ws.send("{oops")
            return False
        if action.startswith("error:"):
            await ws.send(error(action.split(":", 1)[1]))
            return False
        if action.startswith("fail_done:"):
            await ws.send(created(rid, tag))
            await ws.send(done(rid, "failed", action.split(":", 1)[1]))
            return False
        await ws.send(created(rid, tag))
        for index, token in enumerate(TOKENS):
            if action == "drop_after" and index == 2:
                await ws.sever()
                return False
            await ws.send(delta(rid, token))
        await ws.send(done(rid))
        return True

    return script


@pytest.mark.parametrize(
    ("action", "kind"),
    [
        ("drop_before", "transient"),
        ("garbage_before", "protocol"),
        ("error:rate_limit_exceeded", "rate_limit"),
        ("error:server_error", "transient"),
        ("fail_done:server_error", "transient"),
    ],
)
async def test_failures_before_the_first_token_retry_transparently(
    recorder: FlightRecorder, trace: TraceContext, action: str, kind: str
) -> None:
    provider = realtime(chaos({0: action}))
    adapter = llm(provider, recorder, trace)
    assert await reply(adapter) == "".join(TOKENS)
    assert provider.connections == 2, "every retry reconnects (backoff and budget apply)"
    (problem,) = recorder.of_type(ProviderError)
    assert (problem.kind, problem.fatal) == (kind, False)
    await adapter.aclose()
    await provider.finish()


async def test_drop_after_tokens_ends_the_reply_gracefully(
    recorder: FlightRecorder, trace: TraceContext
) -> None:
    async def slow(ws: MemoryWebSocket, _c: int, _n: int, rid: str, tag: str) -> bool:
        await ws.send(created(rid, tag))
        for token in TOKENS[:2]:
            await ws.send(delta(rid, token))
        await ws.recv()  # wait (forever) for a cancel that never comes before the drop
        return False

    provider = realtime(slow)
    adapter = llm(provider, recorder, trace)
    spoken = []
    async for token in adapter.stream([], "weather?"):
        spoken.append(token.text)
        if len(spoken) == 2:  # the client has demonstrably consumed two tokens
            await provider.servers[0].sever()
    assert "".join(spoken) == "Sure,", "spoken tokens stand; the reply ends, no fault"
    assert recorder.of_type(ProviderError)[-1].kind == "transient"
    await adapter.aclose()
    await provider.finish()


async def test_turn_after_a_dropped_reply_reconnects(
    recorder: FlightRecorder, trace: TraceContext
) -> None:
    provider = realtime(chaos({0: "drop_before"}))
    adapter = llm(provider, recorder, trace)
    assert await reply(adapter) == "".join(TOKENS)
    assert await reply(adapter) == "".join(TOKENS)
    assert provider.connections == 2, "the healthy replacement connection is reused"
    await adapter.aclose()
    await provider.finish()


@pytest.mark.parametrize(
    ("action", "kind"),
    [("error:insufficient_quota", ErrorKind.QUOTA), ("error:invalid_api_key", ErrorKind.AUTH)],
)
async def test_quota_and_auth_errors_are_fatal(
    recorder: FlightRecorder, trace: TraceContext, action: str, kind: ErrorKind
) -> None:
    provider = realtime(chaos({0: action}))
    adapter = llm(provider, recorder, trace)
    with pytest.raises(UpstreamError) as excinfo:
        await reply(adapter)
    assert excinfo.value.kind is kind
    assert provider.connections == 1
    await provider.finish()


async def test_handshake_refusal_is_fatal(recorder: FlightRecorder, trace: TraceContext) -> None:
    provider = realtime(stream_all, UpstreamError(ErrorKind.AUTH, "401"))
    with pytest.raises(UpstreamError):
        await reply(llm(provider, recorder, trace))


async def test_persistent_failures_exhaust_the_budget(
    recorder: FlightRecorder, trace: TraceContext
) -> None:
    provider = realtime(chaos(dict.fromkeys(range(5), "drop_before")))
    adapter = llm(provider, recorder, trace, policy=RetryPolicy(max_reconnects=1))
    with pytest.raises(UpstreamError, match="reconnect budget spent"):
        await reply(adapter)
    await provider.finish()


async def test_barge_in_cancels_the_active_response(
    recorder: FlightRecorder, trace: TraceContext
) -> None:
    provider = realtime()
    adapter = llm(provider, recorder, trace)
    async with contextlib.aclosing(adapter.stream([], "go")) as deltas:
        async for _ in deltas:
            break
    await adapter.aclose()
    await provider.finish()
    assert provider.log[-1] == {"type": "response.cancel", "response_id": "resp_0_0"}  # type: ignore[attr-defined]


async def test_cancel_on_a_connection_the_server_already_closed_is_harmless(
    recorder: FlightRecorder, trace: TraceContext
) -> None:
    async def hang_up(ws: MemoryWebSocket, _c: int, _n: int, rid: str, tag: str) -> bool:
        await ws.send(created(rid, tag))
        await ws.send(delta(rid, "Sure"))
        await ws.close(CloseCode.GOING_AWAY)
        return False

    provider = realtime(hang_up)
    adapter = llm(provider, recorder, trace)
    async with contextlib.aclosing(adapter.stream([], "go")) as deltas:
        async for _ in deltas:
            await provider.servers[0].close()
            break
    await adapter.aclose()
    await provider.finish()


# ------------------------------------------------------------------------ parser contract


def test_parser_contract() -> None:
    assert parse_openai_event('{"type":"session.created"}') is None
    assert parse_openai_event(done("r", "failed", "server_error")) == ResponseDone(
        response_id="r", status="failed", error_code="server_error"
    )
    assert parse_openai_event(
        '{"type":"response.done","response":{"id":"r","status":"completed"}}'
    ) == ResponseDone(response_id="r", status="completed", error_code=None)
    assert parse_openai_event('{"type":"error","error":{"type":"server_error"}}') == ServerError(
        type="server_error", code=None, message=""
    )
    assert classify_openai_error(None, "authentication_error") is ErrorKind.AUTH
    assert classify_openai_error("unknown_code") is ErrorKind.PROTOCOL


@pytest.mark.parametrize(
    "message",
    [
        "nope",
        "[]",
        '{"type":"response.created","response":"x"}',
        '{"type":"response.created","response":{"id":5}}',
        '{"type":"response.created","response":{"id":"r","metadata":{"gateway_tag":7}}}',
        '{"type":"response.output_text.delta","response_id":"r"}',
        '{"type":"response.done","response":{"id":"r"}}',
        '{"type":"error","error":{"type":"x","code":1}}',
        '{"type":"error"}',
    ],
)
def test_malformed_events_are_protocol_errors(message: str) -> None:
    with pytest.raises(UpstreamError) as excinfo:
        parse_openai_event(message)
    assert excinfo.value.kind is ErrorKind.PROTOCOL


@settings(max_examples=400)
@given(
    st.recursive(
        st.none() | st.booleans() | st.integers() | st.text(max_size=6),
        lambda inner: (
            st.lists(inner, max_size=3)
            | st.dictionaries(
                st.sampled_from(
                    [
                        "type",
                        "response",
                        "id",
                        "delta",
                        "response_id",
                        "status",
                        "status_details",
                        "error",
                        "code",
                        "metadata",
                    ]
                ),
                inner,
                max_size=5,
            )
        ),
        max_leaves=15,
    ),
    st.sampled_from(
        ["response.created", "response.output_text.delta", "response.done", "error", "x"]
    ),
)
def test_arbitrary_events_never_escape_as_anything_but_protocol_errors(
    document: object, kind: str
) -> None:
    if isinstance(document, dict):
        document = {**document, "type": kind}
    with contextlib.suppress(UpstreamError):
        parse_openai_event(json.dumps(document))

"""Provider resilience state machine: backoff, classification, fatal vs transient."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

from voice_ai_realtime_gateway.adapters.resilience import (
    ErrorKind,
    ProviderLink,
    ProviderState,
    RetryPolicy,
    UpstreamError,
    classify,
)
from voice_ai_realtime_gateway.telemetry.events import ProviderConnected, ProviderError
from voice_ai_realtime_gateway.transport.errors import CloseCode, TransportClosedError
from voice_ai_realtime_gateway.transport.memory import MemoryWebSocket

if TYPE_CHECKING:
    from voice_ai_realtime_gateway.telemetry.ids import TraceContext
    from voice_ai_realtime_gateway.telemetry.sink import FlightRecorder
    from voice_ai_realtime_gateway.transport.base import WebSocketConnection

pytestmark = pytest.mark.usefixtures("strict_loop")


class ScriptedConnector:
    """Each call pops the next outcome: an exception to raise or a connection to return."""

    def __init__(self, *outcomes: BaseException | None) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    async def __call__(self) -> WebSocketConnection:
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if outcome is not None:
            raise outcome
        connection, _peer = MemoryWebSocket.pair()
        return connection


class Sleeps:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


def link(
    connector: ScriptedConnector,
    recorder: FlightRecorder,
    trace: TraceContext,
    sleeps: Sleeps,
    **policy: float,
) -> ProviderLink:
    return ProviderLink(
        provider="acme",
        connector=connector,
        sink=recorder,
        trace=trace,
        policy=RetryPolicy(**policy),  # type: ignore[arg-type]
        sleep=sleeps,
        jitter=lambda: 0.5,
    )


# ------------------------------------------------------------------------ policy


def test_backoff_is_exponential_capped_and_jittered() -> None:
    policy = RetryPolicy(initial_backoff_s=0.1, max_backoff_s=1.0, multiplier=2.0, jitter=0.2)
    assert [round(policy.backoff_s(n, 0.5), 6) for n in (1, 2, 3, 4, 5)] == [
        0.1,
        0.2,
        0.4,
        0.8,
        1.0,
    ]
    assert policy.backoff_s(1, 0.0) == pytest.approx(0.08)
    assert policy.backoff_s(1, 0.999999) == pytest.approx(0.12, rel=1e-4)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_attempts": 0},
        {"max_reconnects": -1},
        {"initial_backoff_s": 0.0},
        {"max_backoff_s": math.inf},
        {"multiplier": 0.5},
        {"initial_backoff_s": 3.0, "max_backoff_s": 1.0},
        {"jitter": 1.0},
        {"jitter": -0.1},
    ],
)
def test_policy_is_validated(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):  # noqa: PT011 -- several distinct messages
        RetryPolicy(**kwargs)  # type: ignore[arg-type]


# ------------------------------------------------------------------------ classification


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (TransportClosedError(CloseCode.ABNORMAL), ErrorKind.TRANSIENT),
        (TransportClosedError(CloseCode.INTERNAL_ERROR, "boom"), ErrorKind.TRANSIENT),
        (TransportClosedError(CloseCode.POLICY_VIOLATION), ErrorKind.PROTOCOL),
        (TransportClosedError(CloseCode.TRY_AGAIN_LATER), ErrorKind.RATE_LIMIT),
        (ConnectionResetError("reset"), ErrorKind.TRANSIENT),
        (TimeoutError(), ErrorKind.TRANSIENT),
    ],
)
def test_generic_classification(error: BaseException, kind: ErrorKind) -> None:
    classified = classify(error, provider="acme")
    assert classified.kind is kind
    assert classified.provider == "acme"
    assert not classified.fatal


def test_classified_errors_pass_through_and_custom_codes_apply() -> None:
    auth = UpstreamError(ErrorKind.AUTH, "bad key")
    assert classify(auth, provider="x") is auth
    assert auth.fatal
    custom = classify(TransportClosedError(4001), provider="x", close_kinds={4001: ErrorKind.AUTH})
    assert custom.kind is ErrorKind.AUTH
    assert UpstreamError(ErrorKind.QUOTA, "out").fatal
    assert UpstreamError(ErrorKind.TRANSIENT, "x", exhausted=True).fatal
    assert str(auth) == "upstream auth: bad key"


# ------------------------------------------------------------------------ state machine


async def test_first_attempt_success_streams(recorder: FlightRecorder, trace: TraceContext) -> None:
    sleeps = Sleeps()
    provider = link(ScriptedConnector(None), recorder, trace, sleeps)
    states = [provider.state]
    connection = await provider.open()
    states.append(provider.state)
    assert (provider.attempts, sleeps.delays) == (1, [])
    await provider.close(connection)
    states.append(provider.state)
    assert states == [ProviderState.IDLE, ProviderState.STREAMING, ProviderState.CLOSED]
    assert repr(provider) == "ProviderLink(provider=acme, state=closed)"


async def test_transient_failures_back_off_then_recover(
    recorder: FlightRecorder, trace: TraceContext
) -> None:
    sleeps = Sleeps()
    connector = ScriptedConnector(ConnectionRefusedError(), TimeoutError(), None)
    provider = link(connector, recorder, trace, sleeps, initial_backoff_s=0.1, jitter=0.0)
    await provider.open()
    assert sleeps.delays == [0.1, 0.2]
    (connected,) = recorder.of_type(ProviderConnected)
    assert (connected.provider, connected.attempt) == ("acme", 3)
    events = recorder.of_type(ProviderError)
    assert [(e.kind, e.fatal, e.attempt) for e in events] == [
        ("transient", False, 1),
        ("transient", False, 2),
    ]
    assert events[0].provider == "acme"


async def test_rate_limit_honours_retry_after(
    recorder: FlightRecorder, trace: TraceContext
) -> None:
    sleeps = Sleeps()
    limited = UpstreamError(ErrorKind.RATE_LIMIT, "slow down", retry_after_s=3.0)
    provider = link(ScriptedConnector(limited, None), recorder, trace, sleeps, jitter=0.0)
    await provider.open()
    assert sleeps.delays == [3.0]


@pytest.mark.parametrize("kind", [ErrorKind.AUTH, ErrorKind.QUOTA])
async def test_auth_and_quota_fail_immediately(
    recorder: FlightRecorder, trace: TraceContext, kind: ErrorKind
) -> None:
    sleeps = Sleeps()
    connector = ScriptedConnector(UpstreamError(kind, "denied"), None)
    provider = link(connector, recorder, trace, sleeps)
    with pytest.raises(UpstreamError) as excinfo:
        await provider.open()
    assert excinfo.value.kind is kind
    assert (connector.calls, sleeps.delays) == (1, []), "no retry on a fatal error"
    assert provider.state is ProviderState.FAILED
    assert recorder.of_type(ProviderError)[-1].fatal
    await provider.close(None)
    assert provider.state is ProviderState.FAILED, "a failed link stays failed"


async def test_exhausted_attempts_are_fatal(recorder: FlightRecorder, trace: TraceContext) -> None:
    sleeps = Sleeps()
    connector = ScriptedConnector(*[OSError("down")] * 3)
    provider = link(connector, recorder, trace, sleeps, max_attempts=3)
    with pytest.raises(UpstreamError, match="gave up after 3 attempts") as excinfo:
        await provider.open()
    assert excinfo.value.exhausted
    assert excinfo.value.fatal
    assert len(sleeps.delays) == 2
    assert provider.state is ProviderState.FAILED


async def test_unexpected_errors_are_bugs_and_propagate(
    recorder: FlightRecorder, trace: TraceContext
) -> None:
    provider = link(ScriptedConnector(KeyError("bug")), recorder, trace, Sleeps())
    with pytest.raises(KeyError):
        await provider.open()


async def test_reopen_after_a_drop_respects_the_reconnect_budget(
    recorder: FlightRecorder, trace: TraceContext
) -> None:
    sleeps = Sleeps()
    provider = link(ScriptedConnector(None, None, None), recorder, trace, sleeps, max_reconnects=2)
    await provider.open()
    drop = TransportClosedError(CloseCode.ABNORMAL)
    await provider.reopen(drop)
    await provider.reopen(drop)
    assert provider.reconnects == 2
    with pytest.raises(UpstreamError, match="reconnect budget spent") as excinfo:
        await provider.reopen(drop)
    assert excinfo.value.exhausted
    assert provider.state is ProviderState.FAILED


async def test_reopen_after_a_fatal_drop_does_not_retry(
    recorder: FlightRecorder, trace: TraceContext
) -> None:
    connector = ScriptedConnector(None, None)
    provider = link(connector, recorder, trace, Sleeps())
    await provider.open()
    with pytest.raises(UpstreamError) as excinfo:
        await provider.reopen(UpstreamError(ErrorKind.QUOTA, "billing"))
    assert excinfo.value.kind is ErrorKind.QUOTA
    assert connector.calls == 1


async def test_fail_marks_the_stream_failed_without_retrying(
    recorder: FlightRecorder, trace: TraceContext
) -> None:
    provider = link(ScriptedConnector(None), recorder, trace, Sleeps())
    await provider.open()
    error = provider.fail(TransportClosedError(CloseCode.ABNORMAL))
    assert error.kind is ErrorKind.TRANSIENT
    assert provider.state is ProviderState.FAILED
    assert recorder.of_type(ProviderError)[-1].fatal

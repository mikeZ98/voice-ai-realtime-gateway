"""Circuit breakers: thresholds, rolling window, exponential recovery, probing, wiring."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from voice_ai_realtime_gateway.production.circuit import (
    CircuitBreaker,
    CircuitPolicy,
    CircuitRegistry,
    CircuitState,
)
from voice_ai_realtime_gateway.telemetry.events import (
    ProviderConnected,
    ProviderError,
    VadSpeechStarted,
)

if TYPE_CHECKING:
    from conftest import FakeClock
    from voice_ai_realtime_gateway.telemetry.ids import TraceContext

S = 1_000_000_000
POLICY = CircuitPolicy(
    window_s=10,
    buckets=10,
    min_calls=4,
    failure_ratio=0.5,
    open_s=2,
    max_open_s=7,
    backoff=2,
    probes=2,
)


def state(circuit: CircuitBreaker) -> CircuitState:
    return circuit.state  # a call, so assertions are not narrowed across mutations


def breaker(
    clock: FakeClock, transitions: list[tuple[str, CircuitState]] | None = None
) -> CircuitBreaker:
    log = transitions if transitions is not None else []
    return CircuitBreaker(
        "deepgram", POLICY, clock=clock, on_change=lambda n, s: log.append((n, s))
    )


def test_opens_only_above_the_volume_and_ratio_thresholds(clock: FakeClock) -> None:
    circuit = breaker(clock)
    for _ in range(3):
        circuit.record_failure()
    assert state(circuit) is CircuitState.CLOSED, "3 calls < min_calls"
    circuit.record_success()
    circuit.record_failure()
    assert state(circuit) is CircuitState.OPEN, "4 of 5 failed"
    assert not circuit.allow()
    assert not circuit.available()


def test_successes_keep_the_ratio_below_threshold(clock: FakeClock) -> None:
    circuit = breaker(clock)
    for _ in range(10):
        circuit.record_success()
    for _ in range(9):
        circuit.record_failure()
    assert state(circuit) is CircuitState.CLOSED, "9 of 19 failed (< 50 %)"
    circuit.record_failure()
    assert state(circuit) is CircuitState.OPEN, "10 of 20 failed (= 50 %)"


def test_old_outcomes_age_out_of_the_window(clock: FakeClock) -> None:
    circuit = breaker(clock)
    for _ in range(3):
        circuit.record_failure()
    clock.now += 11 * S  # the whole window has rolled over
    circuit.record_failure()
    assert state(circuit) is CircuitState.CLOSED, "only 1 failure is inside the window"


def test_half_open_probes_close_and_reset_backoff(clock: FakeClock) -> None:
    transitions: list[tuple[str, CircuitState]] = []
    circuit = breaker(clock, transitions)
    for _ in range(4):
        circuit.record_failure()
    clock.now += 2 * S
    assert state(circuit) is CircuitState.HALF_OPEN
    assert [circuit.allow() for _ in range(3)] == [True, True, False], "2 probes per period"
    circuit.record_success()
    assert state(circuit) is CircuitState.HALF_OPEN
    circuit.record_success()
    assert state(circuit) is CircuitState.CLOSED
    assert circuit.consecutive_openings == 0
    assert [s for _, s in transitions] == [
        CircuitState.OPEN,
        CircuitState.HALF_OPEN,
        CircuitState.CLOSED,
    ]
    assert repr(circuit) == "CircuitBreaker(name=deepgram, state=closed, opened=0)"


def test_failed_probe_reopens_with_exponential_capped_cooldown(clock: FakeClock) -> None:
    circuit = breaker(clock)
    cooldowns = []
    for _ in range(4):
        circuit.record_failure()
    for _ in range(4):
        cooldowns.append(circuit.current_cooldown_s)
        clock.now += int(circuit.current_cooldown_s * S) - 1
        assert state(circuit) is CircuitState.OPEN, "still cooling down"
        clock.now += 1
        assert state(circuit) is CircuitState.HALF_OPEN
        circuit.record_failure()
    assert cooldowns == [2.0, 4.0, 7.0, 7.0], "2 s, then 4 s, then capped at 7 s"
    assert circuit.cooldown_s() == 7.0
    assert circuit.consecutive_openings == 5
    circuit.record_failure()  # failures while open are not double counted
    circuit.record_success()
    assert state(circuit) is CircuitState.OPEN


def test_probe_slots_expire_so_lost_probes_cannot_wedge_the_breaker(clock: FakeClock) -> None:
    circuit = breaker(clock)
    for _ in range(4):
        circuit.record_failure()
    clock.now += 2 * S
    assert circuit.allow()
    assert circuit.allow()
    assert not circuit.allow(), "probes claimed, outcomes never reported"
    clock.now += 2 * S
    assert circuit.allow(), "a new probe period frees the slots"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window_s": 0},
        {"open_s": 10, "max_open_s": 5},
        {"backoff": 0.5},
        {"buckets": 0},
        {"min_calls": 0},
        {"probes": 0},
        {"failure_ratio": 0},
        {"failure_ratio": 1.5},
    ],
)
def test_policy_is_validated(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):  # noqa: PT011 -- several distinct messages
        CircuitPolicy(**kwargs)  # type: ignore[arg-type]


def test_registry_is_fed_by_telemetry_and_selects_fallbacks(
    clock: FakeClock, trace: TraceContext
) -> None:
    transitions: list[tuple[str, CircuitState]] = []
    circuits = CircuitRegistry(
        POLICY, clock=clock, on_change=lambda n, s: transitions.append((n, s))
    )
    for attempt in range(4):
        circuits.emit(ProviderError(trace=trace, at_ns=0, provider="cartesia", kind="transient",
                                    fatal=False, attempt=attempt, detail=""))  # fmt: skip
    circuits.emit(ProviderConnected(trace=trace, at_ns=0, provider="openai", attempt=1))
    circuits.emit(VadSpeechStarted(trace=trace, at_ns=0))  # unrelated events are ignored
    assert circuits.states() == {"cartesia": CircuitState.OPEN, "openai": CircuitState.CLOSED}
    assert circuits.blocking(["openai", "cartesia"]) == "cartesia"
    assert circuits.blocking(["openai"]) is None
    assert circuits.select("cartesia", "elevenlabs") == "elevenlabs", (
        "fall back to a healthy provider"
    )
    assert circuits.select("cartesia") is None
    assert transitions == [("cartesia", CircuitState.OPEN)]
    assert circuits.breaker("cartesia") is circuits.breaker("cartesia")


def test_blocking_claims_probes_only_when_every_provider_admits(
    clock: FakeClock, trace: TraceContext
) -> None:
    circuits = CircuitRegistry(POLICY, clock=clock)
    for name in ("a", "b"):
        for _ in range(4):
            circuits.breaker(name).record_failure()
    clock.now += 2 * S
    assert circuits.breaker("b").state is CircuitState.HALF_OPEN
    circuits.breaker("b")._probes_admitted = 2  # b's probes are all in flight
    assert circuits.blocking(["a", "b"]) == "b"
    assert circuits.breaker("a")._probes_admitted == 0, "a's probe was not wasted"
    assert circuits.blocking(["a"]) is None
    assert circuits.breaker("a")._probes_admitted == 1
    del trace

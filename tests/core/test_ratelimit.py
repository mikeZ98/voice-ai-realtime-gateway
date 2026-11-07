from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

from voice_ai_realtime_gateway.core.ratelimit import TokenBucket

if TYPE_CHECKING:
    from conftest import FakeClock


def test_burst_then_refusal_then_refill(clock: FakeClock) -> None:
    bucket = TokenBucket(rate_per_s=10.0, burst=3, clock=clock)
    assert [bucket.try_acquire() for _ in range(4)] == [True, True, True, False]
    clock.now += 100_000_000  # 0.1 s -> one token
    assert bucket.try_acquire()
    assert not bucket.try_acquire()
    clock.now += 10_000_000_000
    assert bucket.tokens == 3.0, "refill is capped at the burst size"
    assert repr(bucket) == "TokenBucket(rate_per_s=10.0, burst=3)"


def test_clock_going_backwards_never_mints_tokens(clock: FakeClock) -> None:
    bucket = TokenBucket(rate_per_s=1.0, burst=1, clock=clock)
    assert bucket.try_acquire()
    clock.now -= 5_000_000_000
    assert not bucket.try_acquire()


@pytest.mark.parametrize(
    ("rate", "burst", "match"),
    [
        (0.0, 1, "rate_per_s"),
        (math.inf, 1, "rate_per_s"),
        (math.nan, 1, "rate_per_s"),
        (1.0, 0, "burst"),
    ],
)
def test_parameters_are_validated(rate: float, burst: int, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        TokenBucket(rate_per_s=rate, burst=burst)

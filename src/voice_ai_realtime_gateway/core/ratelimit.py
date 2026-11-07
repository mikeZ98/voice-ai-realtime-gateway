"""Token-bucket rate limiting on a monotonic nanosecond clock.

``rate_per_s`` tokens are added continuously up to ``burst``; each admitted event costs
one token. Refill is computed lazily on access, so the bucket costs O(1) time and a few
words of memory per connection and needs no timer task.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Final, final, override

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["TokenBucket"]

_NS_PER_S: Final = 1_000_000_000


@final
class TokenBucket:
    """Classic token bucket; starts full."""

    __slots__ = ("_burst", "_clock", "_last", "_rate", "_tokens")

    def __init__(
        self,
        *,
        rate_per_s: float,
        burst: int,
        clock: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        """Allow ``rate_per_s`` sustained events with bursts of up to ``burst``.

        Raises:
            ValueError: unless ``rate_per_s`` is positive and finite and ``burst >= 1``.
        """
        if not (math.isfinite(rate_per_s) and rate_per_s > 0):
            msg = f"rate_per_s must be positive and finite, got {rate_per_s}"
            raise ValueError(msg)
        if burst < 1:
            msg = f"burst must be >= 1, got {burst}"
            raise ValueError(msg)
        self._rate = rate_per_s
        self._burst = float(burst)
        self._clock = clock
        self._tokens = float(burst)
        self._last = clock()

    @property
    def tokens(self) -> float:
        """Tokens available right now."""
        self._refill()
        return self._tokens

    def try_acquire(self) -> bool:
        """Consume one token if available; ``False`` means the event must be refused."""
        self._refill()
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False

    def _refill(self) -> None:
        now = self._clock()
        elapsed = max(0, now - self._last)
        self._last = now
        self._tokens = min(self._burst, self._tokens + elapsed * self._rate / _NS_PER_S)

    @override
    def __repr__(self) -> str:
        return f"TokenBucket(rate_per_s={self._rate}, burst={int(self._burst)})"

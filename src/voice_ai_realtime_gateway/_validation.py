"""Finiteness-aware numeric predicates shared by every layer's contracts.

Plain comparisons are not enough to validate a float: ``nan <= 0`` and ``nan < 1`` are
both ``False``, so a guard such as ``if timeout <= 0: raise`` silently admits NaN, and
``inf`` passes any lower bound. These predicates reject both. The module imports nothing
from the package, so the leaf ``telemetry`` layer can use it too.
"""

from __future__ import annotations

import math

__all__ = ["is_non_negative_finite", "is_positive_finite", "is_unit_interval"]


def is_positive_finite(value: float) -> bool:
    """``True`` iff ``value`` is a finite number strictly greater than zero."""
    return math.isfinite(value) and value > 0


def is_non_negative_finite(value: float) -> bool:
    """``True`` iff ``value`` is a finite number greater than or equal to zero."""
    return math.isfinite(value) and value >= 0


def is_unit_interval(value: float) -> bool:
    """``True`` iff ``value`` is a finite number in ``[0, 1]`` (a probability)."""
    return math.isfinite(value) and 0.0 <= value <= 1.0

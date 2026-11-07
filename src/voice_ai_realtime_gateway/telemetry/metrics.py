"""Zero-dependency Prometheus metrics with text exposition format 0.0.4.

Families: :class:`Counter`, :class:`Gauge` (set directly, or computed by a callback at
scrape time), :class:`Histogram` (cumulative ``le`` buckets plus ``_sum`` / ``_count``)
and :class:`Summary` (``quantile`` series over a sliding window of recent observations).

* **Bounded cardinality.** Each family holds at most ``max_series`` label sets. Writes
  to further label sets go to a detached child and are counted in ``dropped_series``,
  so a label value derived from untrusted input cannot exhaust memory.
* **Exact escaping.** Label values escape backslash, double quote and newline; HELP
  text escapes backslash and newline, as the format specification requires.
* **Deterministic output.** Families render in registration order and series in
  sorted label order.

The registry is confined to one event loop, like the rest of the gateway.
"""

from __future__ import annotations

import abc
import itertools
import math
import re
from array import array
from typing import TYPE_CHECKING, Final, final, override

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = [
    "CONTENT_TYPE",
    "DEFAULT_MAX_SERIES",
    "Counter",
    "Gauge",
    "Histogram",
    "MetricsRegistry",
    "Summary",
    "escape_help",
    "escape_label_value",
    "format_value",
]

CONTENT_TYPE: Final = "text/plain; version=0.0.4; charset=utf-8"
DEFAULT_MAX_SERIES: Final = 1000

_METRIC_NAME: Final = re.compile(r"[a-zA-Z_:][a-zA-Z0-9_:]*")
_LABEL_NAME: Final = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")
_RESERVED_LABELS: Final = frozenset({"le", "quantile"})

type LabelValues = tuple[str, ...]


def format_value(value: float) -> str:
    """Render a sample value (``+Inf``, ``-Inf`` and ``NaN`` spelled as Prometheus expects)."""
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    return repr(float(value))


def escape_label_value(value: str) -> str:
    """Escape a label value for the text format."""
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def escape_help(text: str) -> str:
    """Escape HELP text for the text format."""
    return text.replace("\\", "\\\\").replace("\n", "\\n")


def _labels(names: Sequence[str], values: LabelValues, extra: tuple[str, str] | None = None) -> str:
    pairs = [
        f'{name}="{escape_label_value(value)}"' for name, value in zip(names, values, strict=True)
    ]
    if extra is not None:
        pairs.append(f'{extra[0]}="{extra[1]}"')
    return "{" + ",".join(pairs) + "}" if pairs else ""


class _Family[C](abc.ABC):
    """Shared machinery: validation, labelled children, cardinality cap."""

    kind: str = "untyped"
    __slots__ = (
        "_children",
        "_factory",
        "dropped_series",
        "help",
        "labelnames",
        "max_series",
        "name",
    )

    def __init__(
        self,
        name: str,
        help_text: str,
        labelnames: Sequence[str],
        factory: Callable[[], C],
        max_series: int,
    ) -> None:
        if _METRIC_NAME.fullmatch(name) is None:
            msg = f"invalid metric name {name!r}"
            raise ValueError(msg)
        for label in labelnames:
            if _LABEL_NAME.fullmatch(label) is None or label.startswith("__"):
                msg = f"invalid label name {label!r}"
                raise ValueError(msg)
            if label in _RESERVED_LABELS:
                msg = f"label name {label!r} is reserved"
                raise ValueError(msg)
        if len(set(labelnames)) != len(labelnames):
            msg = f"duplicate label names in {labelnames!r}"
            raise ValueError(msg)
        if max_series < 1:
            msg = f"max_series must be positive, got {max_series}"
            raise ValueError(msg)
        self.name = name
        self.help = help_text
        self.labelnames = tuple(labelnames)
        self.max_series = max_series
        self.dropped_series = 0
        self._factory = factory
        self._children: dict[LabelValues, C] = {}
        if not self.labelnames:
            self._children[()] = factory()

    def labels(self, *values: str) -> C:
        """Child for one label set (created on first use)."""
        if len(values) != len(self.labelnames):
            msg = f"{self.name} expects labels {self.labelnames}, got {values!r}"
            raise ValueError(msg)
        child = self._children.get(values)
        if child is None:
            child = self._factory()
            if len(self._children) < self.max_series:
                self._children[values] = child
            else:
                self.dropped_series += 1
        return child

    def _unlabelled(self) -> C:
        if self.labelnames:
            msg = f"{self.name} has labels {self.labelnames}; use .labels(...)"
            raise ValueError(msg)
        return self._children[()]

    def render(self) -> list[str]:
        """Exposition lines for this family."""
        lines = [f"# HELP {self.name} {escape_help(self.help)}", f"# TYPE {self.name} {self.kind}"]
        for values in sorted(self._children):
            lines.extend(self._render_child(values, self._children[values]))
        return lines

    @abc.abstractmethod
    def _render_child(self, values: LabelValues, child: C) -> list[str]:
        """Exposition lines for one labelled child."""


# --------------------------------------------------------------------------- counter


@final
class _CounterValue:
    __slots__ = ("value",)

    def __init__(self) -> None:
        self.value = 0.0

    def inc(self, amount: float = 1.0) -> None:
        """Add a non-negative ``amount``."""
        if amount < 0 or math.isnan(amount):
            msg = f"counters only increase, got {amount}"
            raise ValueError(msg)
        self.value += amount


@final
class Counter(_Family[_CounterValue]):
    """Monotonically increasing count; the name must end in ``_total``."""

    kind = "counter"
    __slots__ = ()

    def __init__(
        self,
        name: str,
        help_text: str,
        labelnames: Sequence[str] = (),
        *,
        max_series: int = DEFAULT_MAX_SERIES,
    ) -> None:
        """Declare the family.

        Raises:
            ValueError: on an invalid name, label set, or a name without ``_total``.
        """
        if not name.endswith("_total"):
            msg = f"counter name {name!r} must end in '_total'"
            raise ValueError(msg)
        super().__init__(name, help_text, labelnames, _CounterValue, max_series)

    def inc(self, amount: float = 1.0) -> None:
        """Increment the unlabelled counter."""
        self._unlabelled().inc(amount)

    @property
    def value(self) -> float:
        """Current value of the unlabelled counter."""
        return self._unlabelled().value

    @override
    def _render_child(self, values: LabelValues, child: _CounterValue) -> list[str]:
        return [f"{self.name}{_labels(self.labelnames, values)} {format_value(child.value)}"]


# --------------------------------------------------------------------------- gauge


@final
class _GaugeValue:
    __slots__ = ("value",)

    def __init__(self) -> None:
        self.value = 0.0

    def set(self, value: float) -> None:
        """Set the current value."""
        self.value = value

    def inc(self, amount: float = 1.0) -> None:
        """Add ``amount`` (may be negative)."""
        self.value += amount

    def dec(self, amount: float = 1.0) -> None:
        """Subtract ``amount``."""
        self.value -= amount


@final
class Gauge(_Family[_GaugeValue]):
    """A value that goes up and down, optionally computed by ``function`` at scrape time."""

    kind = "gauge"
    __slots__ = ("_function",)

    def __init__(
        self,
        name: str,
        help_text: str,
        labelnames: Sequence[str] = (),
        *,
        function: Callable[[], float] | None = None,
        max_series: int = DEFAULT_MAX_SERIES,
    ) -> None:
        """Declare the family; a ``function`` gauge cannot have labels.

        Raises:
            ValueError: on an invalid declaration.
        """
        if function is not None and labelnames:
            msg = "a function gauge cannot have labels"
            raise ValueError(msg)
        super().__init__(name, help_text, labelnames, _GaugeValue, max_series)
        self._function = function

    def set(self, value: float) -> None:
        """Set the unlabelled gauge."""
        self._unlabelled().set(value)

    def inc(self, amount: float = 1.0) -> None:
        """Increment the unlabelled gauge."""
        self._unlabelled().inc(amount)

    def dec(self, amount: float = 1.0) -> None:
        """Decrement the unlabelled gauge."""
        self._unlabelled().dec(amount)

    @property
    def value(self) -> float:
        """Current value (evaluating ``function`` if set)."""
        if self._function is not None:
            return float(self._function())
        return self._unlabelled().value

    @override
    def _render_child(self, values: LabelValues, child: _GaugeValue) -> list[str]:
        value = float(self._function()) if self._function is not None else child.value
        return [f"{self.name}{_labels(self.labelnames, values)} {format_value(value)}"]


# --------------------------------------------------------------------------- histogram


@final
class _HistogramValue:
    __slots__ = ("count", "counts", "sum", "upper_bounds")

    def __init__(self, upper_bounds: tuple[float, ...]) -> None:
        self.upper_bounds = upper_bounds
        self.counts = [0] * len(upper_bounds)
        self.sum = 0.0
        self.count = 0

    def observe(self, value: float) -> None:
        """Record one observation."""
        if math.isnan(value):
            msg = "cannot observe NaN"
            raise ValueError(msg)
        for index, bound in enumerate(self.upper_bounds):
            if value <= bound:
                self.counts[index] += 1
                break
        self.sum += value
        self.count += 1


@final
class Histogram(_Family[_HistogramValue]):
    """Cumulative-bucket distribution."""

    kind = "histogram"
    __slots__ = ("buckets",)

    def __init__(
        self,
        name: str,
        help_text: str,
        buckets: Sequence[float],
        labelnames: Sequence[str] = (),
        *,
        max_series: int = DEFAULT_MAX_SERIES,
    ) -> None:
        """Declare the family with finite, strictly increasing bucket upper bounds.

        Raises:
            ValueError: on an invalid declaration or bucket list.
        """
        bounds = tuple(float(bound) for bound in buckets)
        if not bounds or any(not math.isfinite(bound) for bound in bounds):
            msg = "histogram buckets must be a non-empty list of finite numbers"
            raise ValueError(msg)
        if any(low >= high for low, high in itertools.pairwise(bounds)):
            msg = "histogram buckets must be strictly increasing"
            raise ValueError(msg)
        self.buckets = bounds
        super().__init__(name, help_text, labelnames, lambda: _HistogramValue(bounds), max_series)

    def observe(self, value: float) -> None:
        """Observe into the unlabelled histogram."""
        self._unlabelled().observe(value)

    @property
    def count(self) -> int:
        """Observations recorded by the unlabelled histogram."""
        return self._unlabelled().count

    @override
    def _render_child(self, values: LabelValues, child: _HistogramValue) -> list[str]:
        lines = []
        cumulative = 0
        for bound, count in zip(child.upper_bounds, child.counts, strict=True):
            cumulative += count
            le = _labels(self.labelnames, values, ("le", format_value(bound)))
            lines.append(f"{self.name}_bucket{le} {cumulative}")
        infinite = _labels(self.labelnames, values, ("le", "+Inf"))
        plain = _labels(self.labelnames, values)
        lines.append(f"{self.name}_bucket{infinite} {child.count}")
        lines.append(f"{self.name}_sum{plain} {format_value(child.sum)}")
        lines.append(f"{self.name}_count{plain} {child.count}")
        return lines


# --------------------------------------------------------------------------- summary


@final
class _SummaryValue:
    __slots__ = ("_cursor", "_retained", "_window", "count", "sum")

    def __init__(self, window: int) -> None:
        self._window = array("d", bytes(8 * window))
        self._cursor = 0
        self._retained = 0
        self.sum = 0.0
        self.count = 0

    def observe(self, value: float) -> None:
        """Record one observation in O(1)."""
        if math.isnan(value):
            msg = "cannot observe NaN"
            raise ValueError(msg)
        size = len(self._window)
        self._window[self._cursor] = value
        self._cursor = (self._cursor + 1) % size
        self._retained = min(self._retained + 1, size)
        self.sum += value
        self.count += 1

    def quantile(self, q: float) -> float:
        """Nearest-rank quantile over the window (``NaN`` when empty)."""
        if self._retained == 0:
            return math.nan
        ordered = sorted(self._window[: self._retained])
        return ordered[max(0, math.ceil(q * self._retained) - 1)]


@final
class Summary(_Family[_SummaryValue]):
    """Sliding-window quantiles plus lifetime ``_sum`` / ``_count``."""

    kind = "summary"
    __slots__ = ("quantiles",)

    def __init__(  # noqa: PLR0913 -- a summary is exactly these declarations
        self,
        name: str,
        help_text: str,
        labelnames: Sequence[str] = (),
        *,
        quantiles: Sequence[float] = (0.5, 0.95, 0.99),
        window: int = 1024,
        max_series: int = DEFAULT_MAX_SERIES,
    ) -> None:
        """Declare the family.

        Raises:
            ValueError: on an invalid declaration, quantile or window.
        """
        if any(not 0.0 < q < 1.0 for q in quantiles) or not quantiles:
            msg = "summary quantiles must be a non-empty list within (0, 1)"
            raise ValueError(msg)
        if window < 1:
            msg = f"window must be positive, got {window}"
            raise ValueError(msg)
        self.quantiles = tuple(quantiles)
        super().__init__(name, help_text, labelnames, lambda: _SummaryValue(window), max_series)

    def observe(self, value: float) -> None:
        """Observe into the unlabelled summary."""
        self._unlabelled().observe(value)

    def quantile(self, q: float) -> float:
        """Windowed quantile of the unlabelled summary."""
        return self._unlabelled().quantile(q)

    @property
    def count(self) -> int:
        """Observations recorded by the unlabelled summary."""
        return self._unlabelled().count

    @override
    def _render_child(self, values: LabelValues, child: _SummaryValue) -> list[str]:
        lines = [
            f"{self.name}{_labels(self.labelnames, values, ('quantile', format_value(q)))} "
            f"{format_value(child.quantile(q))}"
            for q in self.quantiles
        ]
        plain = _labels(self.labelnames, values)
        lines.append(f"{self.name}_sum{plain} {format_value(child.sum)}")
        lines.append(f"{self.name}_count{plain} {child.count}")
        return lines


# --------------------------------------------------------------------------- registry


type Family = Counter | Gauge | Histogram | Summary


@final
class MetricsRegistry:
    """Ordered collection of metric families with text exposition."""

    __slots__ = ("_families",)

    def __init__(self) -> None:
        """Create an empty registry."""
        self._families: dict[str, Family] = {}

    def register[F: Family](self, family: F) -> F:
        """Add ``family``.

        Raises:
            ValueError: if a family with the same name is already registered.
        """
        if family.name in self._families:
            msg = f"metric {family.name!r} is already registered"
            raise ValueError(msg)
        self._families[family.name] = family
        return family

    def __contains__(self, name: object) -> bool:
        """Whether a family called ``name`` is registered."""
        return name in self._families

    def render(self) -> str:
        """The full exposition document (ends with a newline)."""
        lines: list[str] = []
        for family in self._families.values():
            lines.extend(family.render())
        return "\n".join(lines) + "\n"

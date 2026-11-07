"""Prometheus exposition: golden output, escaping round trips, cardinality, validation."""

from __future__ import annotations

import math
import re

import pytest
from hypothesis import given
from hypothesis import strategies as st

from voice_ai_realtime_gateway.telemetry.metrics import (
    CONTENT_TYPE,
    Counter,
    Gauge,
    Histogram,
    MetricsRegistry,
    Summary,
    escape_help,
    escape_label_value,
    format_value,
)

SAMPLE = re.compile(r"(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>.*)\})? (?P<value>\S+)")
LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')


def unescape(value: str) -> str:
    return re.sub(r"\\(.)", lambda m: {"n": "\n", "\\": "\\", '"': '"'}[m.group(1)], value)


def parse(document: str) -> list[tuple[str, dict[str, str], str]]:
    """Minimal text-format parser: every non-comment line must be a valid sample."""
    assert document.endswith("\n")
    samples = []
    for line in document.split("\n")[:-1]:  # the format's only line terminator is LF
        if line.startswith("#"):
            assert re.fullmatch(r"# (HELP|TYPE) [a-zA-Z_:][a-zA-Z0-9_:]* .*", line), line
            continue
        match = SAMPLE.fullmatch(line)
        assert match, line
        raw = match.group("labels") or ""
        labels = {k: unescape(v) for k, v in LABEL.findall(raw)}
        if raw:
            rebuilt = ",".join(f'{k}="{escape_label_value(v)}"' for k, v in labels.items())
            assert rebuilt == raw, "labels parse back losslessly"
        samples.append((match.group("name"), labels, match.group("value")))
    return samples


def test_value_formatting() -> None:
    assert [format_value(v) for v in (1, 2.5, math.inf, -math.inf, math.nan)] == [
        "1.0",
        "2.5",
        "+Inf",
        "-Inf",
        "NaN",
    ]
    assert escape_help("a\\b\nc") == "a\\\\b\\nc"
    assert CONTENT_TYPE.startswith("text/plain; version=0.0.4")


def test_golden_exposition_for_every_family() -> None:
    registry = MetricsRegistry()
    drops = registry.register(Counter("drops_total", "Dropped packets", ("reason",)))
    drops.labels("overflow").inc(3)
    drops.labels("network_loss").inc()
    sessions = registry.register(Gauge("sessions", "Active sessions"))
    sessions.set(2)
    sessions.inc()
    sessions.dec(0.5)
    latency = registry.register(Histogram("latency_seconds", "Latency", (0.1, 0.5)))
    for value in (0.05, 0.2, 0.2, 9.0):
        latency.observe(value)
    tokens = registry.register(Summary("ttft_seconds", "First token", quantiles=(0.5, 0.99)))
    for value in (0.3, 0.1, 0.2):
        tokens.observe(value)
    assert registry.render() == (
        "# HELP drops_total Dropped packets\n"
        "# TYPE drops_total counter\n"
        'drops_total{reason="network_loss"} 1.0\n'
        'drops_total{reason="overflow"} 3.0\n'
        "# HELP sessions Active sessions\n"
        "# TYPE sessions gauge\n"
        "sessions 2.5\n"
        "# HELP latency_seconds Latency\n"
        "# TYPE latency_seconds histogram\n"
        'latency_seconds_bucket{le="0.1"} 1\n'
        'latency_seconds_bucket{le="0.5"} 3\n'
        'latency_seconds_bucket{le="+Inf"} 4\n'
        "latency_seconds_sum 9.45\n"
        "latency_seconds_count 4\n"
        "# HELP ttft_seconds First token\n"
        "# TYPE ttft_seconds summary\n"
        'ttft_seconds{quantile="0.5"} 0.2\n'
        'ttft_seconds{quantile="0.99"} 0.3\n'
        "ttft_seconds_sum 0.6000000000000001\n"
        "ttft_seconds_count 3\n"
    )
    assert "sessions" in registry
    assert latency.count == 4
    assert (tokens.count, tokens.quantile(0.5)) == (3, 0.2)
    assert sessions.value == 2.5
    parse(registry.render())


def test_labelled_histogram_and_summary_render_per_series() -> None:
    registry = MetricsRegistry()
    hist = registry.register(Histogram("h", "h", (1.0,), ("provider",)))
    hist.labels("a").observe(0.5)
    summ = registry.register(Summary("s", "s", ("provider",), quantiles=(0.5,), window=2))
    for value in (5.0, 1.0, 2.0):  # window keeps the last two
        summ.labels("b").observe(value)
    samples = parse(registry.render())
    assert ("h_bucket", {"provider": "a", "le": "1.0"}, "1") in samples
    assert ("s", {"provider": "b", "quantile": "0.5"}, "1.0") in samples
    assert ("s_sum", {"provider": "b"}, "8.0") in samples


def test_empty_summary_reports_nan_quantiles() -> None:
    registry = MetricsRegistry()
    registry.register(Summary("s", "s", quantiles=(0.5,)))
    assert 's{quantile="0.5"} NaN' in registry.render()


def test_function_gauge_is_evaluated_at_scrape_time() -> None:
    state = {"n": 1}
    registry = MetricsRegistry()
    gauge = registry.register(Gauge("live", "Live", function=lambda: state["n"]))
    state["n"] = 7
    assert "live 7.0" in registry.render()
    assert gauge.value == 7.0


@given(st.text(max_size=40), st.text(max_size=40))
def test_any_label_value_and_help_text_round_trip(value: str, help_text: str) -> None:
    registry = MetricsRegistry()
    counter = registry.register(Counter("x_total", help_text, ("v",)))
    counter.labels(value).inc()
    samples = parse(registry.render())
    assert samples == [("x_total", {"v": value}, "1.0")]
    assert "\n" not in escape_help(help_text)


def test_cardinality_is_capped() -> None:
    counter = Counter("c_total", "c", ("id",), max_series=2)
    for index in range(5):
        counter.labels(str(index)).inc()
    assert counter.dropped_series == 3
    assert len(parse("\n".join(counter.render()) + "\n")) == 2
    counter.labels("0").inc()  # existing series keep working
    assert counter.dropped_series == 3


@pytest.mark.parametrize(
    ("build", "match"),
    [
        (lambda: Counter("bad name_total", "x"), "invalid metric name"),
        (lambda: Counter("c", "x"), "_total"),
        (lambda: Counter("c_total", "x", ("bad-label",)), "invalid label"),
        (lambda: Counter("c_total", "x", ("__internal",)), "invalid label"),
        (lambda: Counter("c_total", "x", ("le",)), "reserved"),
        (lambda: Counter("c_total", "x", ("a", "a")), "duplicate"),
        (lambda: Counter("c_total", "x", max_series=0), "max_series"),
        (lambda: Gauge("g", "x", ("a",), function=lambda: 1.0), "function gauge"),
        (lambda: Histogram("h", "x", ()), "non-empty"),
        (lambda: Histogram("h", "x", (1.0, math.inf)), "finite"),
        (lambda: Histogram("h", "x", (2.0, 1.0)), "strictly increasing"),
        (lambda: Summary("s", "x", quantiles=(1.0,)), "quantiles"),
        (lambda: Summary("s", "x", quantiles=()), "quantiles"),
        (lambda: Summary("s", "x", window=0), "window"),
    ],
)
def test_invalid_declarations_are_rejected(build: object, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        build()  # type: ignore[operator]


def test_invalid_writes_are_rejected() -> None:
    counter = Counter("c_total", "c", ("a",))
    with pytest.raises(ValueError, match="expects labels"):
        counter.labels()
    with pytest.raises(ValueError, match=r"use \.labels"):
        counter.inc()
    with pytest.raises(ValueError, match="only increase"):
        Counter("d_total", "d").inc(-1)
    with pytest.raises(ValueError, match="NaN"):
        Histogram("h", "h", (1.0,)).observe(math.nan)
    with pytest.raises(ValueError, match="NaN"):
        Summary("s", "s").observe(math.nan)
    registry = MetricsRegistry()
    registry.register(Gauge("g", "g"))
    with pytest.raises(ValueError, match="already registered"):
        registry.register(Gauge("g", "g"))
    assert Counter("e_total", "e").value == 0.0

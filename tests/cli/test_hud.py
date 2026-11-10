"""Rich HUD: pure rendering, stderr-only output, refresh lifecycle."""

from __future__ import annotations

import asyncio
import io

import pytest
from rich.console import Console

from voice_ai_realtime_gateway.cli.hud import TelemetryHud, phase_style, render
from voice_ai_realtime_gateway.cli.simulation import LiveSnapshot


def snapshot(**overrides: object) -> LiveSnapshot:
    fields: dict[str, object] = {
        "phase": "serving",
        "elapsed_s": 2.5,
        "calls_total": 4,
        "calls_active": 2,
        "calls_completed": 1,
        "calls_refused": 1,
        "buffered_bytes": 1_000,
        "capacity_bytes": 4_000,
        "ttft_p50_ms": 1.25,
        "first_audio_p50_ms": 21.5,
        "jitter_p50_ms": 0.4,
        "jitter_p99_ms": 1.1,
        "barge_ins": 3,
        "barge_in_p99_ms": 0.12,
        "frames_out": 99,
        "dropped": 1,
        "close_codes": {1000: 1, 1013: 1},
    }
    fields.update(overrides)
    return LiveSnapshot(**fields)  # type: ignore[arg-type]


def text_of(snap: LiveSnapshot) -> str:
    console = Console(file=io.StringIO(), width=140, color_system=None)
    console.print(render(snap))
    output = console.file
    assert isinstance(output, io.StringIO)
    return output.getvalue()


def test_render_shows_every_metric() -> None:
    text = text_of(snapshot())
    for expected in (
        "SERVING",
        "TTFT p50",
        "1.25 ms",
        "21.50 ms",
        "jitter p99",
        "1.10 ms",
        "3 (0.12 ms)",
        "1,000 B / 4,000 B",
        "1.0000% (1 of 100)",
        "1000x1, 1013x1",
        "2",
    ):
        assert expected in text, expected


def test_render_handles_missing_measurements() -> None:
    text = text_of(snapshot(ttft_p50_ms=None, jitter_p50_ms=None, close_codes={}))
    assert "TTFT p50" in text
    assert " - " in text or "-\n" in text or "  -" in text


@pytest.mark.parametrize(
    ("phase", "style"),
    [
        ("serving", "bold green"),
        ("draining", "bold yellow"),
        ("flushing", "bold magenta"),
        ("drained", "bold cyan"),
        ("mystery", "bold white"),
    ],
)
def test_phase_styles(phase: str, style: str) -> None:
    assert phase_style(phase) == style


@pytest.mark.parametrize("buffered", [0, 2_500, 3_900])  # green, yellow, red fill
def test_render_colours_the_buffer_fill(buffered: int) -> None:
    assert "buffer fill" in text_of(snapshot(buffered_bytes=buffered))


def test_refresh_rate_is_validated() -> None:
    with pytest.raises(ValueError, match="refresh_hz"):
        TelemetryHud(snapshot, refresh_hz=0)


async def test_hud_draws_to_its_console_and_stops_cleanly() -> None:
    sink = io.StringIO()
    console = Console(file=sink, width=140, force_terminal=True, color_system=None)
    samples = 0

    def sample() -> LiveSnapshot:
        nonlocal samples
        samples += 1
        return snapshot(phase="draining" if samples > 1 else "serving")

    hud = TelemetryHud(sample, console=console, refresh_hz=200)
    assert hud.console is console
    hud.refresh()  # not live yet: a no-op
    assert samples == 0
    async with hud:
        await asyncio.sleep(0.03)
    assert samples > 2, "the pump redraws while running"
    assert "DRAINING" in sink.getvalue()


def test_default_console_is_stderr() -> None:
    assert TelemetryHud(snapshot).console.stderr


async def test_exit_without_enter_is_a_no_op() -> None:
    hud = TelemetryHud(snapshot, console=Console(file=io.StringIO()))
    await hud.__aexit__(None, None, None)
    hud.refresh()

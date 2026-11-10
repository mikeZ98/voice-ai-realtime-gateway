"""Rich terminal HUD for the ``voice-gateway`` CLI (the optional ``[hud]`` extra).

Renders only to **stderr**, so the CLI's stdout carries nothing but its JSON summary.
:func:`render` is a pure function of a :class:`LiveSnapshot`; :class:`TelemetryHud`
redraws it at a fixed rate from a background task while the simulation runs.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Final, Self, final

from rich.console import Console, Group
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import TracebackType

    from rich.live import Live

    from voice_ai_realtime_gateway.cli.simulation import LiveSnapshot

__all__ = ["TelemetryHud", "phase_style", "render"]

_PHASE_STYLES: Final = {
    "serving": "bold green",
    "draining": "bold yellow",
    "flushing": "bold magenta",
    "drained": "bold cyan",
}
_FILL_WARN: Final = 0.5
_FILL_ALERT: Final = 0.8


def phase_style(phase: str) -> str:
    """Colour of a drain phase."""
    return _PHASE_STYLES.get(phase, "bold white")


def _ms(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f} ms"


def _fill_style(fill: float) -> str:
    if fill >= _FILL_ALERT:
        return "red"
    return "yellow" if fill >= _FILL_WARN else "green"


def render(snapshot: LiveSnapshot) -> Panel:
    """Draw one frame of the HUD."""
    header = Text.assemble(
        ("voice-gateway ", "bold"),
        (snapshot.phase.upper(), phase_style(snapshot.phase)),
        f"   {snapshot.elapsed_s:.1f}s",
    )
    latency = Table.grid(padding=(0, 2))
    latency.add_column(style="dim")
    latency.add_column(justify="right")
    latency.add_row("TTFT p50", _ms(snapshot.ttft_p50_ms))
    latency.add_row("first audio p50", _ms(snapshot.first_audio_p50_ms))
    latency.add_row("jitter p50", _ms(snapshot.jitter_p50_ms))
    latency.add_row("jitter p99", _ms(snapshot.jitter_p99_ms))
    latency.add_row("barge-ins (p99)", f"{snapshot.barge_ins} ({_ms(snapshot.barge_in_p99_ms)})")

    fill = snapshot.buffer_fill
    buffers = Table.grid(padding=(0, 2))
    buffers.add_column(style="dim")
    buffers.add_column(justify="right")
    buffers.add_row("queue depth", f"{snapshot.buffered_bytes:,} B / {snapshot.capacity_bytes:,} B")
    buffers.add_row(
        "buffer fill",
        ProgressBar(total=1.0, completed=fill, width=24, complete_style=_fill_style(fill)),
    )
    attempted = snapshot.frames_out + snapshot.dropped
    buffers.add_row("drop rate", f"{snapshot.drop_rate:.4%} ({snapshot.dropped} of {attempted})")

    sockets = Table.grid(padding=(0, 2))
    sockets.add_column(style="dim")
    sockets.add_column(justify="right")
    sockets.add_row("open", str(snapshot.calls_active))
    sockets.add_row("completed", f"{snapshot.calls_completed} / {snapshot.calls_total}")
    sockets.add_row("refused", str(snapshot.calls_refused))
    codes = ", ".join(f"{code}x{count}" for code, count in snapshot.close_codes.items())
    sockets.add_row("close codes", codes or "-")

    body = Table.grid(padding=(0, 4), expand=True)
    body.add_row(
        Panel(latency, title="latency", border_style="dim"),
        Panel(buffers, title="buffers", border_style="dim"),
        Panel(sockets, title="websockets", border_style="dim"),
    )
    return Panel(Group(header, body), border_style=phase_style(snapshot.phase))


@final
class TelemetryHud:
    """Redraws :func:`render` of ``sample()`` at ``refresh_hz`` on stderr."""

    __slots__ = ("_console", "_interval_s", "_live", "_sample", "_task")

    def __init__(
        self,
        sample: Callable[[], LiveSnapshot],
        *,
        console: Console | None = None,
        refresh_hz: float = 4.0,
    ) -> None:
        """Bind the HUD to ``sample`` (stderr console unless one is given)."""
        if not refresh_hz > 0:
            msg = f"refresh_hz must be positive, got {refresh_hz}"
            raise ValueError(msg)
        self._sample = sample
        self._console = Console(stderr=True) if console is None else console
        self._interval_s = 1.0 / refresh_hz
        self._live: Live | None = None
        self._task: asyncio.Task[None] | None = None

    @property
    def console(self) -> Console:
        """Where the HUD draws."""
        return self._console

    def refresh(self) -> None:
        """Draw the current snapshot now."""
        if self._live is not None:
            self._live.update(render(self._sample()), refresh=True)

    async def __aenter__(self) -> Self:
        """Start drawing."""
        from rich.live import Live  # noqa: PLC0415 -- only when the HUD actually runs

        self._live = Live(
            render(self._sample()),
            console=self._console,
            auto_refresh=False,
            redirect_stdout=False,
            redirect_stderr=False,
            transient=False,
        )
        self._live.start()
        self._task = asyncio.create_task(self._pump(), name="hud-refresh")
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Draw a final frame and stop (a no-op if the HUD was never entered)."""
        task, live = self._task, self._live
        if task is None or live is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        self.refresh()
        live.stop()
        self._task = self._live = None

    async def _pump(self) -> None:
        while True:
            await asyncio.sleep(self._interval_s)
            self.refresh()

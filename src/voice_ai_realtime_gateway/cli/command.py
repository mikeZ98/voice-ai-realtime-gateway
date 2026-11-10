"""``voice-gateway``: run the gateway offline against scripted callers.

Contract:

* **stdout** carries exactly one line: the JSON run summary.
* **stderr** carries the HUD (``--hud``, needs the ``[hud]`` extra) and error messages.
* **Exit status**: ``0`` every call completed and every turn was answered; ``1`` the run
  was unhealthy (failed calls, unanswered turns or dropped audio); ``2`` usage or
  configuration error; ``128 + signum`` stopped by a signal (130 for ``SIGINT``, 143 for
  ``SIGTERM``).
* **Signals**: the first ``SIGINT``/``SIGTERM`` starts the graceful drain. New calls are
  refused with ``1013``, live calls finish the current answer (in-flight audio is
  delivered) and hang up with ``1000``, and calls still alive at ``--drain-timeout`` are
  closed with ``1001``. Later signals are recorded; the drain is already bounded.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib.util
import json
import signal
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Final

from voice_ai_realtime_gateway.cli.simulation import Simulation, SimulationConfig

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

__all__ = ["EXIT_UNHEALTHY", "EXIT_USAGE", "SIGNAL_EXIT_BASE", "build_parser", "main"]

EXIT_UNHEALTHY: Final = 1
EXIT_USAGE: Final = 2
SIGNAL_EXIT_BASE: Final = 128
STOP_SIGNALS: Final = (signal.SIGINT, signal.SIGTERM)


def build_parser() -> argparse.ArgumentParser:
    """The CLI's argument parser."""
    defaults = SimulationConfig()
    parser = argparse.ArgumentParser(
        prog="voice-gateway",
        description=(
            "Run the full-duplex voice gateway offline: scripted callers, deterministic "
            "STT/LLM/TTS doubles, the production server, VAD, barge-in and graceful drain. "
            "Prints one JSON summary line on stdout."
        ),
    )
    parser.add_argument("--calls", type=int, default=defaults.calls, help="concurrent calls")
    parser.add_argument("--turns", type=int, default=defaults.turns, help="utterances per call")
    parser.add_argument(
        "--barge-in-every",
        type=int,
        default=defaults.barge_in_every,
        metavar="N",
        help="interrupt the agent on every N-th turn (0: never)",
    )
    parser.add_argument("--speech-frames", type=int, default=defaults.speech_frames)
    parser.add_argument("--ramp", type=float, default=defaults.ramp_s, metavar="SECONDS")
    parser.add_argument(
        "--drain-timeout", type=float, default=defaults.drain_timeout_s, metavar="SECONDS"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="do not pace audio at the real 20 ms cadence (same media timeline, no waiting)",
    )
    parser.add_argument(
        "--hud",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="live terminal HUD on stderr (default: on when stderr is a TTY and rich is installed)",
    )
    parser.add_argument("--report", type=Path, metavar="PATH", help="also write the summary here")
    return parser


def _config(args: argparse.Namespace) -> SimulationConfig:
    return SimulationConfig(
        calls=args.calls,
        turns=args.turns,
        barge_in_every=args.barge_in_every,
        speech_frames=args.speech_frames,
        ramp_s=args.ramp,
        drain_timeout_s=args.drain_timeout,
        real_time=not args.fast,
    )


def hud_available() -> bool:
    """Whether the optional ``rich`` dependency is installed."""
    return importlib.util.find_spec("rich") is not None


@contextlib.contextmanager
def _stop_signals_to(simulation: Simulation) -> Iterator[None]:
    loop = asyncio.get_running_loop()
    for sig in STOP_SIGNALS:
        loop.add_signal_handler(sig, simulation.request_stop, int(sig))
    try:
        yield
    finally:
        for sig in STOP_SIGNALS:
            loop.remove_signal_handler(sig)


async def _execute(simulation: Simulation, *, hud: bool) -> None:
    with _stop_signals_to(simulation):
        if not hud:
            await simulation.run()
            return
        from voice_ai_realtime_gateway.cli.hud import TelemetryHud  # noqa: PLC0415 -- optional extra

        async with TelemetryHud(simulation.snapshot):
            await simulation.run()


def _outcome(simulation: Simulation) -> tuple[str, int]:
    if simulation.signals:
        return "drained", SIGNAL_EXIT_BASE + simulation.signals[0]
    if simulation.healthy:
        return "completed", 0
    return "unhealthy", EXIT_UNHEALTHY


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI; returns the process exit status."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # --help (0) or a usage error (2)
        return int(exc.code or 0)
    try:
        config = _config(args)
    except ValueError as exc:
        sys.stderr.write(f"voice-gateway: invalid configuration: {exc}\n")
        return EXIT_USAGE
    hud = args.hud if args.hud is not None else (sys.stderr.isatty() and hud_available())
    if hud and not hud_available():
        sys.stderr.write(
            "voice-gateway: --hud needs the optional 'rich' dependency: "
            "install voice-ai-realtime-gateway[hud]\n"
        )
        return EXIT_USAGE

    async def run() -> Simulation:
        simulation = Simulation(config)
        await _execute(simulation, hud=hud)
        return simulation

    simulation = asyncio.run(run())
    status, exit_code = _outcome(simulation)
    summary = {"status": status, "exit_code": exit_code, **simulation.summary()}
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    sys.stdout.write(json.dumps(summary, sort_keys=True) + "\n")
    sys.stdout.flush()
    return exit_code

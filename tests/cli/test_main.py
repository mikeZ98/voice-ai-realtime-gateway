"""voice-gateway CLI contract: one JSON line on stdout, exit codes, signals, HUD gating."""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import signal
from typing import TYPE_CHECKING, Any

import pytest

from voice_ai_realtime_gateway.cli import command as cli
from voice_ai_realtime_gateway.cli.simulation import Simulation

if TYPE_CHECKING:
    from pathlib import Path

FAST = ["--fast", "--ramp", "0", "--no-hud"]


def run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict[str, Any], str]:
    code = cli.main(list(argv))
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert len(lines) == 1, f"stdout must be exactly one JSON line, got {captured.out!r}"
    return code, json.loads(lines[0]), captured.err


def test_completed_run(capsys: pytest.CaptureFixture[str]) -> None:
    code, summary, err = run(capsys, *FAST, "--calls", "2", "--turns", "2")
    assert (code, summary["status"], summary["exit_code"]) == (0, "completed", 0)
    assert summary["close_codes"] == {"1000": 2}
    assert summary["config"] == {"barge_in_every": 2, "calls": 2, "real_time": False, "turns": 2}
    assert err == ""


def test_unhealthy_run_exits_one(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Simulation, "healthy", property(lambda _self: False))
    code, summary, _ = run(capsys, *FAST, "--calls", "1", "--turns", "1")
    assert (code, summary["status"], summary["exit_code"]) == (1, "unhealthy", 1)


def test_report_file_matches_stdout(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    report = tmp_path / "nested" / "run.json"
    code, summary, _ = run(capsys, *FAST, "--calls", "1", "--turns", "1", "--report", str(report))
    assert code == 0
    assert json.loads(report.read_text()) == summary


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_signals_drain_and_exit_128_plus_signum(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, sig: signal.Signals
) -> None:
    real_run = Simulation.run

    async def run_and_get_signalled(self: Simulation) -> None:
        running = asyncio.create_task(real_run(self))
        await asyncio.sleep(0)
        os.kill(os.getpid(), sig)  # delivered through the loop's signal handler
        await running

    monkeypatch.setattr(Simulation, "run", run_and_get_signalled)
    code, summary, _ = run(capsys, *FAST, "--calls", "2", "--turns", "50", "--ramp", "1")
    assert code == summary["exit_code"] == 128 + sig
    assert summary["status"] == "drained"
    assert summary["signals"] == [int(sig)]
    assert "1012" not in summary["close_codes"]
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler, "handlers removed"


def test_hud_renders_to_stderr_only(capsys: pytest.CaptureFixture[str]) -> None:
    code, summary, err = run(
        capsys, "--fast", "--ramp", "0", "--hud", "--calls", "1", "--turns", "1"
    )
    assert (code, summary["status"]) == (0, "completed")
    assert "voice-gateway" in err
    assert "TTFT p50" in err


def test_hud_defaults_to_off_when_stderr_is_not_a_tty(capsys: pytest.CaptureFixture[str]) -> None:
    _, _, err = run(capsys, "--fast", "--ramp", "0", "--calls", "1", "--turns", "1")
    assert err == ""


def test_hud_without_rich_is_a_usage_error(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "hud_available", lambda: False)
    assert cli.main(["--hud"]) == cli.EXIT_USAGE
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "voice-ai-realtime-gateway[hud]" in captured.err


def test_rich_is_detected() -> None:
    assert cli.hud_available()


def test_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--help"]) == 0
    assert "voice-gateway" in capsys.readouterr().out


@pytest.mark.parametrize("argv", [["--calls", "many"], ["--bogus"]])
def test_usage_errors_exit_two(capsys: pytest.CaptureFixture[str], argv: list[str]) -> None:
    assert cli.main(argv) == cli.EXIT_USAGE
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "usage:" in captured.err


@pytest.mark.parametrize("argv", [["--calls", "0"], ["--drain-timeout", "nan"], ["--ramp", "-1"]])
def test_invalid_configuration_exits_two(
    capsys: pytest.CaptureFixture[str], argv: list[str]
) -> None:
    assert cli.main(argv) == cli.EXIT_USAGE
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "invalid configuration" in captured.err


def test_module_entry_point_is_the_cli() -> None:
    module = importlib.import_module("voice_ai_realtime_gateway.__main__")
    assert module.main is cli.main

#!/usr/bin/env python3
"""Real-time latency benchmark behind the README tables.

Each run is a fresh child process executing the real ``voice-gateway`` CLI in real time
(20 ms audio cadence, real-time synthesis): the production server, transport, VAD,
pipeline and drain, with scripted callers who interrupt the agent on every second turn.
A fresh process per run makes peak RSS attributable to that run alone.

Reported per scenario (median over ``--repeat`` runs):

* **TTFT**: end of user speech (VAD offset) to the LLM's first token.
* **first audio**: end of user speech to the first agent audio packet leaving the
  gateway (glass-to-glass at the gateway, including the 20 ms TTS frame).
* **barge-in**: VAD onset to the session draining, i.e. TTS task cancelled and the
  outbound ring flushed, measured inside the gateway. Gated: p99 must stay < 5 ms.
* **client flush**: the caller starting to talk over the agent until it receives
  ``playback.flush`` (includes the VAD's onset detection).
* **jitter buffer**: underflow and overflow rates of the client playout model
  (40 ms pre-buffer, 200 ms capacity; see ``PlayoutBuffer``).
* **peak RSS** of the gateway process under the scenario's load.

Usage::

    uv run python scripts/benchmark.py [--calls 1 25 100 250] [--turns 3] [--repeat 3]
                                       [--json .cache/benchmarks/latency.json]

Exit status is 0 when every run was healthy and every barge-in p99 stayed under 5 ms.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Final

ROOT: Final = Path(__file__).resolve().parent.parent
CACHE: Final = ROOT / ".cache" / "benchmarks"
BARGE_IN_BUDGET_MS: Final = 5.0


def run_once(calls: int, turns: int, ramp_s: float) -> dict[str, Any]:
    """One real-time CLI run in a child process; returns its summary plus peak RSS."""
    CACHE.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=CACHE, suffix=".json", delete=False) as handle:
        report = Path(handle.name)
    cmd = [
        sys.executable, "-m", "voice_ai_realtime_gateway", "--no-hud",
        "--calls", str(calls), "--turns", str(turns), "--barge-in-every", "2",
        "--ramp", str(ramp_s), "--report", str(report),
    ]  # fmt: skip
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.DEVNULL)
    _, status, usage = os.wait4(proc.pid, 0)
    proc.returncode = os.waitstatus_to_exitcode(status)
    summary: dict[str, Any] = json.loads(report.read_text())
    report.unlink()
    # ru_maxrss is bytes on macOS and KiB on Linux.
    rss = usage.ru_maxrss if sys.platform == "darwin" else usage.ru_maxrss * 1024
    summary["peak_rss_mib"] = rss / 2**20
    summary["process_exit"] = proc.returncode
    return summary


def median(values: list[float]) -> float:
    """Median of ``values``."""
    return statistics.median(values)


def stat(runs: list[dict[str, Any]], key: str, field: str) -> float:
    """Median over runs of one percentile field (runs without samples are skipped)."""
    return median([float(run[key][field]) for run in runs if run[key][field] is not None])


def row(calls: int, runs: list[dict[str, Any]]) -> str:
    """One Markdown table row for a scenario."""
    playout = [run["playout"] for run in runs]
    answered = sum(run["turns"]["answered"] for run in runs)
    turns = answered + sum(run["turns"]["unanswered"] for run in runs)
    cells = [
        f"{calls:,}",
        f"{stat(runs, 'ttft_ms', 'p50'):.2f} / {stat(runs, 'ttft_ms', 'p99'):.2f}",
        (
            f"{stat(runs, 'time_to_first_audio_ms', 'p50'):.1f} / "
            f"{stat(runs, 'time_to_first_audio_ms', 'p99'):.1f}"
        ),
        f"{stat(runs, 'barge_in_ms', 'p50'):.3f} / {stat(runs, 'barge_in_ms', 'p99'):.3f}",
        f"{stat(runs, 'client_flush_ms', 'p99'):.1f}",
        f"{stat(runs, 'jitter_ms', 'p99'):.2f}",
        f"{median([p['underflow_rate'] for p in playout]):.4%}",
        f"{median([p['overflow_rate'] for p in playout]):.4%}",
        str(sum(run["egress"]["dropped"] for run in runs)),
        f"{median([run['peak_rss_mib'] for run in runs]):.0f}",
        f"{answered:,} / {turns:,}",
    ]
    return "| " + " | ".join(cells) + " |"


HEADER: Final = (
    "| Calls | TTFT p50 / p99 (ms) | First audio p50 / p99 (ms) | Barge-in p50 / p99 (ms) "
    "| Client flush p99 (ms) | Jitter p99 (ms) | Underflow | Overflow | Dropped "
    "| Peak RSS (MiB) | Turns answered |\n"
    "|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"
)


def healthy(run: dict[str, Any]) -> bool:
    """A run counts if the CLI was healthy and barge-in p99 stayed in budget."""
    barge_p99 = run["barge_in_ms"]["p99"]
    return run["exit_code"] == 0 and (barge_p99 is None or barge_p99 < BARGE_IN_BUDGET_MS)


def main() -> int:
    """Run every scenario and print the table; non-zero exit on a failed gate."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--calls", type=int, nargs="+", default=[1, 25, 100, 250])
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--ramp", type=float, default=1.0, help="seconds to spread call starts")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    print(
        f"<!-- {platform.platform()} | Python {platform.python_version()} | "
        f"{os.cpu_count()} CPUs | turns={args.turns} | repeat={args.repeat} -->"
    )
    print(HEADER)
    raw: dict[str, list[dict[str, Any]]] = {}
    ok = True
    for calls in args.calls:
        runs = [run_once(calls, args.turns, args.ramp) for _ in range(args.repeat)]
        raw[str(calls)] = runs
        ok = ok and all(healthy(run) for run in runs)
        print(row(calls, runs), flush=True)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(raw, indent=2) + "\n")
    if not ok:
        print(f"benchmark: FAILED (unhealthy run or barge-in p99 >= {BARGE_IN_BUDGET_MS} ms)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

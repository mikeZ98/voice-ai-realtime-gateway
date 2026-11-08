"""Benchmark harness smoke tests: tiny runs, real code paths, network-free."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from benchmarks import bench_concurrency as bench

if TYPE_CHECKING:
    from pathlib import Path

TINY = bench.BenchConfig(
    sessions=6, shards=1, turns=1, speech_frames=6, ramp_s=0.05, churn_waves=2, churn_sessions=2
)


def test_percentiles_use_nearest_rank_in_milliseconds() -> None:
    assert bench.percentiles([]) == {"count": 0}
    stats = bench.percentiles([n * 1_000_000 for n in range(1, 101)])
    assert stats == {"count": 100, "p50": 50.0, "p95": 95.0, "p99": 99.0, "max": 100.0}
    assert bench.percentiles([7_000_000])["p99"] == 7.0


@pytest.mark.parametrize(
    "overrides", [{"sessions": 0}, {"shards": 0}, {"churn_waves": 1}, {"ramp_s": -1.0}]
)
def test_config_is_validated(overrides: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="positive"):
        bench.BenchConfig(**overrides)  # type: ignore[arg-type]


def test_single_loop_run_reports_every_metric() -> None:
    report = bench.run_benchmark(TINY)
    concurrency = report["concurrency"]
    assert isinstance(concurrency, dict)
    assert concurrency["completed"] == 6
    assert concurrency["peak_concurrent_sessions"] == 6
    assert concurrency["registry_empty"] is True
    thresholds = report["thresholds"]
    assert isinstance(thresholds, dict)
    assert thresholds["all_calls_completed"] is True
    assert thresholds["all_sessions_concurrent"] is True
    for key in ("glass_to_glass_ms", "client_turn_latency_ms", "frame_jitter_ms"):
        stats = report[key]
        assert isinstance(stats, dict)
        assert stats["count"] > 0
    g2g = report["glass_to_glass_ms"]
    assert isinstance(g2g, dict)
    assert g2g["p50"] >= TINY.frame_ms * 0.9, "real-time TTS: first frame takes one frame"
    churn = report["churn_memory"]
    assert isinstance(churn, dict)
    assert churn["calls_served"] == (bench.WARMUP_WAVES + 2) * 2
    assert len(churn["retained_bytes_after_wave"]) == 2


def test_sharded_cli_run_writes_the_json_report(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    status = bench.main(
        [
            "--sessions", "4", "--shards", "2", "--turns", "1", "--ramp", "0.05",
            "--churn-waves", "2", "--churn-sessions", "2", "--sweep", "2",
            "--output", str(output),
        ]
    )  # fmt: skip
    report = json.loads(output.read_text())
    assert status in {0, 1}
    assert status == (0 if report["passed"] else 1)
    assert report["concurrency"]["event_loops"] == 2
    assert report["concurrency"]["sessions_per_loop"] == [2, 2]
    assert report["concurrency"]["all_sessions_overlapped"] is True
    assert len(report["event_loop_lag_ms"]["p99_per_loop"]) == 2
    (point,) = report["single_loop_capacity"]
    assert point["sessions_per_loop"] == 2
    assert point["failed"] == 0
    assert all(rss > 0 for rss in report["peak_rss_bytes_per_loop"])

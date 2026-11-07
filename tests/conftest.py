"""Shared test configuration."""

from __future__ import annotations

from pathlib import Path

from hypothesis import HealthCheck, settings
from hypothesis.configuration import set_hypothesis_home_dir
from hypothesis.database import DirectoryBasedExampleDatabase

# --------------------------------------------------------------------------- hygiene
# Hypothesis state lives under ./.cache/ like every other tool cache.
_HYPOTHESIS_HOME = Path(__file__).resolve().parent.parent / ".cache" / "hypothesis"
set_hypothesis_home_dir(_HYPOTHESIS_HOME)
settings.register_profile(
    "gateway",
    database=DirectoryBasedExampleDatabase(_HYPOTHESIS_HOME / "examples"),
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("gateway")

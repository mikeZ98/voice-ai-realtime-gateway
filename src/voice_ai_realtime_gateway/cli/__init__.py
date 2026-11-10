"""``voice-gateway`` command line: offline simulation, JSON summary, optional Rich HUD."""

from voice_ai_realtime_gateway.cli.command import main
from voice_ai_realtime_gateway.cli.simulation import LiveSnapshot, Simulation, SimulationConfig

__all__ = ["LiveSnapshot", "Simulation", "SimulationConfig", "main"]

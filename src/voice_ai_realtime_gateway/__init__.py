"""Ultra-low-latency full-duplex voice agent gateway.

Layering (imports only point downwards):

* :mod:`voice_ai_realtime_gateway.domain` -- session state machine, Lamport clock,
  barge-in aware duplex controller.
* :mod:`voice_ai_realtime_gateway.core` -- bounded zero-copy audio frame ring.
* :mod:`voice_ai_realtime_gateway.telemetry` -- trace correlation IDs, typed events,
  sinks and latency windows.
"""

from importlib.metadata import version

__all__ = ["__version__"]

__version__: str = version("voice-ai-realtime-gateway")

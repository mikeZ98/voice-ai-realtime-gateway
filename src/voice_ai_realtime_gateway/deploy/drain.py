"""Kubernetes ``preStop`` hook: drain this pod before ``SIGTERM`` arrives.

Usage (container exec hook)::

    python -m voice_ai_realtime_gateway.deploy.drain --pid 1 --timeout 45

1. Send ``SIGUSR1`` to the gateway process; its :class:`~.lifecycle.DrainController`
   turns ``/readyz`` to 503 and lets live calls finish.
2. Poll ``/drainz`` on localhost until it answers 200 (drain complete) or the timeout
   passes. Kubernetes sends ``SIGTERM`` only after this hook returns, so uvicorn never
   cuts a live call with 1012.

Exit status: 0 when drained, 1 on timeout (Kubernetes proceeds either way, bounded by
``terminationGracePeriodSeconds``). The probe only talks to loopback addresses.
"""

from __future__ import annotations

import argparse
import os
import signal
import time
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = ["main", "probe_status"]

_LOOPBACK: Final = ("http://127.0.0.1:", "http://localhost:", "http://[::1]:")
_HTTP_OK: Final = 200


def probe_status(url: str, *, timeout_s: float = 2.0) -> int:
    """HTTP status of ``GET url`` on loopback, or 0 when unreachable.

    Raises:
        ValueError: if ``url`` is not a plain-HTTP loopback URL.
    """
    if not url.startswith(_LOOPBACK):
        msg = f"refusing to probe non-loopback URL {url!r}"
        raise ValueError(msg)
    # The scheme and host were pinned to loopback HTTP above, so no file:// or remote URL.
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:  # noqa: S310  # nosec B310
            return int(response.status)
    except urllib.error.HTTPError as error:
        return int(error.code)
    except (urllib.error.URLError, OSError):
        return 0


def main(
    argv: Sequence[str] | None = None,
    *,
    kill: Callable[[int, int], None] = os.kill,
    probe: Callable[[str], int] = probe_status,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    """Run the hook; returns the process exit status."""
    parser = argparse.ArgumentParser(description="Drain the voice gateway before SIGTERM.")
    parser.add_argument("--pid", type=int, default=1)
    parser.add_argument("--signal", default="SIGUSR1", choices=["SIGUSR1", "SIGUSR2", "SIGTERM"])
    parser.add_argument("--url", default="http://127.0.0.1:8080/drainz")
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--interval", type=float, default=0.5)
    arguments = parser.parse_args(argv)
    kill(arguments.pid, signal.Signals[arguments.signal])
    deadline = clock() + arguments.timeout
    while clock() < deadline:
        if probe(arguments.url) == _HTTP_OK:
            return 0
        sleep(arguments.interval)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

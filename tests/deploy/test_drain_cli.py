"""preStop drain hook: signal, poll, exit status, loopback-only probing."""

from __future__ import annotations

import signal
import urllib.error
from typing import TYPE_CHECKING, Any, Self

import pytest

from voice_ai_realtime_gateway.deploy import drain

if TYPE_CHECKING:
    from collections.abc import Iterator


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, delay: float) -> None:
        self.now += delay


def test_signals_then_polls_until_drained() -> None:
    kills: list[tuple[int, int]] = []
    answers: Iterator[int] = iter([0, 503, 503, 200])
    clock = Clock()
    status = drain.main(
        ["--pid", "7", "--url", "http://127.0.0.1:9/drainz", "--interval", "1"],
        kill=lambda pid, sig: kills.append((pid, sig)),
        probe=lambda _url: next(answers),
        sleep=clock.sleep,
        clock=clock,
    )
    assert status == 0
    assert kills == [(7, signal.SIGUSR1)]
    assert clock.now == 3.0


def test_gives_up_after_the_timeout() -> None:
    clock = Clock()
    status = drain.main(
        ["--timeout", "3", "--signal", "SIGTERM"],
        kill=lambda _pid, sig: None if sig == signal.SIGTERM else pytest.fail("wrong signal"),
        probe=lambda _url: 503,
        sleep=clock.sleep,
        clock=clock,
    )
    assert status == 1


class Response:
    status = 200

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None


def test_probe_maps_http_outcomes(monkeypatch: pytest.MonkeyPatch) -> None:
    outcomes: list[Any] = [
        Response(),
        urllib.error.HTTPError("u", 503, "draining", {}, None),  # type: ignore[arg-type]
        urllib.error.URLError("refused"),
        OSError("reset"),
    ]

    def fake_urlopen(url: str, timeout: float) -> Response:
        assert url.startswith("http://127.0.0.1:")
        assert timeout == 2.0
        outcome = outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome  # type: ignore[no-any-return]

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert [drain.probe_status("http://127.0.0.1:8080/drainz") for _ in range(4)] == [
        200,
        503,
        0,
        0,
    ]


@pytest.mark.parametrize(
    "url", ["https://127.0.0.1:1/x", "http://example.com/drainz", "file:///etc/passwd"]
)
def test_probe_refuses_non_loopback_urls(url: str) -> None:
    with pytest.raises(ValueError, match="non-loopback"):
        drain.probe_status(url)

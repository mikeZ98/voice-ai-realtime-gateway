from __future__ import annotations

import socket
from importlib.metadata import version

import pytest
from pytest_socket import SocketBlockedError

import voice_ai_realtime_gateway


def test_version_matches_distribution_metadata() -> None:
    assert voice_ai_realtime_gateway.__version__ == version("voice-ai-realtime-gateway")


@pytest.mark.parametrize("family", [socket.AF_INET, socket.AF_INET6])
def test_unit_suite_is_network_free(family: socket.AddressFamily) -> None:
    with (
        pytest.warns(UserWarning, match="socket"),
        pytest.raises(SocketBlockedError),
    ):
        socket.socket(family, socket.SOCK_STREAM)

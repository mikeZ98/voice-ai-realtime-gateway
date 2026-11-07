from __future__ import annotations

import importlib
import socket
from importlib.metadata import version

import pytest
from pytest_socket import SocketBlockedError

import voice_ai_realtime_gateway

PUBLIC_PACKAGES = (
    "voice_ai_realtime_gateway.adapters",
    "voice_ai_realtime_gateway.core",
    "voice_ai_realtime_gateway.domain",
    "voice_ai_realtime_gateway.inference",
    "voice_ai_realtime_gateway.pipeline",
    "voice_ai_realtime_gateway.production",
    "voice_ai_realtime_gateway.telemetry",
    "voice_ai_realtime_gateway.transport",
    "voice_ai_realtime_gateway.vad",
)


def test_version_matches_distribution_metadata() -> None:
    assert voice_ai_realtime_gateway.__version__ == version("voice-ai-realtime-gateway")


@pytest.mark.parametrize("module_name", PUBLIC_PACKAGES)
def test_public_api_is_exact_and_resolvable(module_name: str) -> None:
    module = importlib.import_module(module_name)
    exported = module.__all__
    assert len(set(exported)) == len(exported), "__all__ must be duplicate-free"
    for name in exported:
        assert getattr(module, name) is not None


@pytest.mark.parametrize("family", [socket.AF_INET, socket.AF_INET6])
def test_unit_suite_is_network_free(family: socket.AddressFamily) -> None:
    with (
        pytest.warns(UserWarning, match="socket"),
        pytest.raises(SocketBlockedError),
    ):
        socket.socket(family, socket.SOCK_STREAM)

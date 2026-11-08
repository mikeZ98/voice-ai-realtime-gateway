"""Settings from the environment and process wiring."""

from __future__ import annotations

import base64
import json

import pytest

from voice_ai_realtime_gateway.core.frames import AudioCodec
from voice_ai_realtime_gateway.deploy.app import GatewaySettings, create_demo_app
from voice_ai_realtime_gateway.deploy.asgi import GatewayAsgiApp

SECRET = base64.urlsafe_b64encode(b"q" * 32).decode().rstrip("=")
TENANTS = json.dumps(
    {
        "acme": {
            "max_concurrent_sessions": 50,
            "max_session_duration_s": 1800,
            "codecs": ["pcm16"],
        },
        "beta": {"max_concurrent_sessions": 2, "max_session_duration_s": 60},
    }
)
BAD_CODEC = json.dumps(
    {"a": {"max_concurrent_sessions": 1, "max_session_duration_s": 1, "codecs": ["mp3"]}}
)
BASE = {"GATEWAY_KEYS": f"k1={SECRET}, k2={SECRET}", "GATEWAY_TENANTS": TENANTS}


def test_settings_from_a_complete_environment() -> None:
    settings = GatewaySettings.from_env(
        {
            **BASE,
            "GATEWAY_AUDIENCE": "aud",
            "GATEWAY_WEBSOCKET_PATH": "/ws",
            "GATEWAY_DRAIN_TIMEOUT_S": "45",
            "GATEWAY_FINAL_SCRAPE_GRACE_S": "0",
            "GATEWAY_MEDIA_TIMEOUT_S": "7.5",
            "GATEWAY_REAPER_GRACE_S": "12",
            "GATEWAY_REQUIRED_PROVIDERS": "deepgram, openai,,cartesia",
        }
    )
    assert settings.keys == {"k1": b"q" * 32, "k2": b"q" * 32}
    assert settings.tenants["acme"].allowed_codecs == frozenset({AudioCodec.PCM16})
    assert settings.tenants["beta"].allowed_codecs == frozenset(AudioCodec), "default: all codecs"
    assert (settings.audience, settings.websocket_path) == ("aud", "/ws")
    assert (settings.drain_timeout_s, settings.final_scrape_grace_s, settings.media_timeout_s) == (
        45,
        0,
        7.5,
    )
    assert settings.reaper_grace_s == 12
    assert settings.required_providers == ("deepgram", "openai", "cartesia")


def test_defaults() -> None:
    settings = GatewaySettings.from_env(BASE)
    assert (settings.drain_timeout_s, settings.final_scrape_grace_s) == (30.0, 15.0)
    assert (settings.media_timeout_s, settings.reaper_grace_s, settings.required_providers) == (
        10.0,
        30.0,
        (),
    )


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"GATEWAY_KEYS": ""}, "GATEWAY_KEYS is required"),
        ({"GATEWAY_TENANTS": ""}, "GATEWAY_TENANTS is required"),
        ({"GATEWAY_KEYS": "nokey"}, "kid=base64url"),
        ({"GATEWAY_KEYS": "=abc"}, "kid=base64url"),
        ({"GATEWAY_KEYS": "k1=@@@"}, "not base64url"),
        ({"GATEWAY_KEYS": " , "}, "at least one key"),
        ({"GATEWAY_TENANTS": "{"}, "must be JSON"),
        ({"GATEWAY_TENANTS": "[]"}, "non-empty JSON object"),
        ({"GATEWAY_TENANTS": '{"a": []}'}, "tenant 'a' is invalid"),
        ({"GATEWAY_TENANTS": '{"a": {"max_concurrent_sessions": 1}}'}, "tenant 'a' is invalid"),
        ({"GATEWAY_TENANTS": BAD_CODEC}, "tenant 'a' is invalid"),
        ({"GATEWAY_DRAIN_TIMEOUT_S": "soon"}, "GATEWAY_DRAIN_TIMEOUT_S"),
        ({"GATEWAY_DRAIN_TIMEOUT_S": "-1"}, "GATEWAY_DRAIN_TIMEOUT_S"),
        ({"GATEWAY_MEDIA_TIMEOUT_S": "inf"}, "GATEWAY_MEDIA_TIMEOUT_S"),
    ],
)
def test_invalid_environments_name_the_variable(overrides: dict[str, str], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        GatewaySettings.from_env({**BASE, **overrides})


def test_short_keys_are_rejected_at_wiring_time(monkeypatch: pytest.MonkeyPatch) -> None:
    short = base64.urlsafe_b64encode(b"x" * 8).decode()
    monkeypatch.setattr("os.environ", {**BASE, "GATEWAY_KEYS": f"k1={short}"})
    with pytest.raises(ValueError, match="32 bytes"):
        create_demo_app()


def test_demo_app_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("os.environ", dict(BASE))
    app = create_demo_app()
    assert isinstance(app, GatewayAsgiApp)
    assert app.server.ready

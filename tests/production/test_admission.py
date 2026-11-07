"""Token admission: signatures, algorithms, time bounds, claims, header parsing, fuzzing."""

from __future__ import annotations

import base64
import json
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from voice_ai_realtime_gateway.core.frames import AudioCodec
from voice_ai_realtime_gateway.production.admission import (
    AdmissionError,
    TokenAuthenticator,
    TokenSigner,
    bearer_token,
)

KEY = b"k" * 32
OTHER = b"o" * 32
NOW = 1_800_000_000.0
SID = "0123456789abcdef" * 2


def claims(**overrides: Any) -> dict[str, Any]:  # noqa: ANN401 -- arbitrary claim values
    base: dict[str, Any] = {
        "aud": "voice-gateway",
        "exp": NOW + 600,
        "iat": NOW,
        "tenant": "acme",
        "codecs": ["pcm16"],
        "max_duration_s": 900,
        "jti": "t-1",
    }
    base.update(overrides)
    return {k: v for k, v in base.items() if v is not ...}


def authenticator(**kwargs: Any) -> TokenAuthenticator:  # noqa: ANN401
    return TokenAuthenticator({"k1": KEY, "k2": OTHER}, wall_clock=lambda: NOW, **kwargs)


def refused(token: str, reason: str, auth: TokenAuthenticator | None = None) -> None:
    with pytest.raises(AdmissionError) as excinfo:
        (auth or authenticator()).verify(token)
    assert excinfo.value.reason == reason


def test_valid_token_yields_its_grant() -> None:
    grant = authenticator().verify(
        TokenSigner("k1", KEY).sign(claims(sid=SID, codecs=["pcm16", "opus"]))
    )
    assert grant.tenant == "acme"
    assert grant.codecs == frozenset({AudioCodec.PCM16, AudioCodec.OPUS})
    assert (grant.max_duration_s, grant.expires_at) == (900.0, int(NOW + 600))
    assert (grant.session_id, grant.token_id, grant.key_id) == (SID, "t-1", "k1")


def test_key_rotation_accepts_every_active_key() -> None:
    assert authenticator().verify(TokenSigner("k2", OTHER).sign(claims())).key_id == "k2"
    unbound = authenticator().verify(TokenSigner("k1", KEY).sign(claims(jti=...)))
    assert (unbound.session_id, unbound.token_id) == (None, None)


def test_audience_may_be_a_list() -> None:
    assert (
        authenticator()
        .verify(TokenSigner("k1", KEY).sign(claims(aud=["x", "voice-gateway"])))
        .tenant
        == "acme"
    )


# ------------------------------------------------------------------------ forgery


def test_tampered_claims_fail_the_signature() -> None:
    header, _, signature = TokenSigner("k1", KEY).sign(claims()).split(".")
    forged = (
        base64.urlsafe_b64encode(json.dumps(claims(tenant="evil")).encode()).rstrip(b"=").decode()
    )
    refused(f"{header}.{forged}.{signature}", "bad signature")


def test_key_id_and_secret_must_match() -> None:
    refused(TokenSigner("k1", OTHER).sign(claims()), "bad signature")
    refused(TokenSigner("nope", KEY).sign(claims()), "unknown key")
    refused(TokenSigner("k1", KEY).sign(claims(), header={"kid": 7}), "unknown key")


@pytest.mark.parametrize("alg", ["none", "RS256", "HS512", "hs256", None])
def test_only_hs256_is_accepted(alg: str | None) -> None:
    refused(TokenSigner("k1", KEY).sign(claims(), header={"alg": alg}), "unsupported algorithm")


@pytest.mark.parametrize(
    "token",
    [
        "",
        "a.b",
        "a.b.c.d",
        "a+b.c.d",
        "a.b.c=",
        "é.a.b",
        "x" * 5000,
        f"{base64.urlsafe_b64encode(b'[1]').decode().rstrip('=')}.e30.e30",
        f"{base64.urlsafe_b64encode(b'{bad').decode().rstrip('=')}.e30.e30",
        "A.e30.e30",
    ],
)
def test_malformed_tokens_are_refused(token: str) -> None:
    refused(token, "malformed token")


def test_claims_segment_must_be_a_json_object() -> None:
    signer = TokenSigner("k1", KEY)
    header, _, _ = signer.sign(claims()).split(".")
    body = base64.urlsafe_b64encode(b"[]").rstrip(b"=").decode()
    import hashlib  # noqa: PLC0415
    import hmac  # noqa: PLC0415

    signature = hmac.new(KEY, f"{header}.{body}".encode(), hashlib.sha256).digest()
    sig = base64.urlsafe_b64encode(signature).rstrip(b"=").decode()
    refused(f"{header}.{body}.{sig}", "malformed token")


# ------------------------------------------------------------------------ time bounds


def test_expiry_honours_the_leeway() -> None:
    signer = TokenSigner("k1", KEY)
    assert authenticator(leeway_s=30).verify(signer.sign(claims(exp=NOW - 10))).tenant == "acme"
    refused(signer.sign(claims(exp=NOW - 30)), "token expired", authenticator(leeway_s=30))
    refused(signer.sign(claims(exp=...)), "invalid claims")
    refused(signer.sign(claims(exp="soon")), "invalid claims")
    refused(signer.sign(claims(exp=True)), "invalid claims")


@pytest.mark.parametrize("name", ["nbf", "iat"])
def test_future_tokens_are_not_yet_valid(name: str) -> None:
    signer = TokenSigner("k1", KEY)
    refused(signer.sign(claims(**{name: NOW + 120})), "token not yet valid")
    assert authenticator().verify(signer.sign(claims(**{name: NOW + 20}))).tenant == "acme"


# ------------------------------------------------------------------------ claims


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"aud": "other"}, "wrong audience"),
        ({"aud": ...}, "wrong audience"),
        ({"tenant": ""}, "invalid claims"),
        ({"tenant": "a b"}, "invalid claims"),
        ({"tenant": "x" * 65}, "invalid claims"),
        ({"tenant": 5}, "invalid claims"),
        ({"codecs": []}, "invalid claims"),
        ({"codecs": "pcm16"}, "invalid claims"),
        ({"codecs": ["mp3"]}, "invalid claims"),
        ({"codecs": [1]}, "invalid claims"),
        ({"max_duration_s": 0}, "invalid claims"),
        ({"max_duration_s": 86_401}, "invalid claims"),
        ({"max_duration_s": True}, "invalid claims"),
        ({"max_duration_s": "long"}, "invalid claims"),
        ({"sid": "short"}, "invalid claims"),
        ({"sid": 5}, "invalid claims"),
        ({"jti": 5}, "invalid claims"),
    ],
)
def test_invalid_claims_are_refused(overrides: dict[str, Any], reason: str) -> None:
    refused(TokenSigner("k1", KEY).sign(claims(**overrides)), reason)


def test_configuration_is_validated() -> None:
    with pytest.raises(ValueError, match="at least one key"):
        TokenAuthenticator({})
    with pytest.raises(ValueError, match="32 bytes"):
        TokenAuthenticator({"k": b"short"})
    with pytest.raises(ValueError, match="32 bytes"):
        TokenSigner("k", b"short")
    with pytest.raises(ValueError, match="leeway"):
        TokenAuthenticator({"k": KEY}, leeway_s=-1)


# ------------------------------------------------------------------------ headers


def test_bearer_token_extraction() -> None:
    assert bearer_token({"authorization": "Bearer abc.def-ghi_j"}) == "abc.def-ghi_j"
    assert bearer_token({"Authorization": "bearer x.y.z"}) == "x.y.z"
    with pytest.raises(AdmissionError, match="missing"):
        bearer_token({"X-Other": "1"})
    with pytest.raises(AdmissionError, match="scheme"):
        bearer_token({"Authorization": "Basic dXNlcjpwYXNz"})


# ------------------------------------------------------------------------ fuzzing


@settings(max_examples=500)
@given(st.text(max_size=300))
def test_arbitrary_strings_never_escape_as_anything_but_refusals(token: str) -> None:
    with pytest.raises(AdmissionError):
        authenticator().verify(token)


@settings(max_examples=500)
@given(position=st.integers(min_value=0), replacement=st.sampled_from("ABCxyz019_-."))
def test_any_single_character_change_is_refused(position: int, replacement: str) -> None:
    token = TokenSigner("k1", KEY).sign(claims())
    index = position % len(token)
    if token[index] == replacement:
        return
    with pytest.raises(AdmissionError):
        authenticator().verify(token[:index] + replacement + token[index + 1 :])

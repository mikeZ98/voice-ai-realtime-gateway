"""Session admission with HMAC-SHA256 signed bearer tokens (compact JWS / JWT ``HS256``).

Token: ``base64url(header) "." base64url(claims) "." base64url(HMAC-SHA256(key, h "." c))``

Header: ``{"alg": "HS256", "typ": "JWT", "kid": "<key id>"}``. ``alg`` must be exactly
``HS256`` (``none`` and algorithm-confusion attempts are rejected) and ``kid`` selects
one of several active keys, enabling rotation without downtime.

Claims:

==================  ========  ===================================================
claim               required  meaning
==================  ========  ===================================================
``aud``             yes       must equal (or contain) the configured audience
``exp``             yes       expiry, seconds since the epoch
``nbf`` / ``iat``   no        not-before / issued-at (rejected if in the future)
``tenant``          yes       tenant id ``[A-Za-z0-9._-]{1,64}``
``codecs``          yes       non-empty list of allowed ingress codecs
``max_duration_s``  yes       session duration cap, ``0 < d <= 86400``
``sid``             no        binds the token to one ``X-Session-ID``
``jti``             no        token id, for audit
==================  ========  ===================================================

Verification order is fail-fast and side-channel aware. Size, alphabet and structure
come first. The signature is compared in constant time (``hmac.compare_digest``)
**before** any claim is interpreted. Then come time bounds (with a bounded leeway) and
claim validation. Every failure raises :class:`AdmissionError` with a short, public
reason that never echoes token contents.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import math
import re
import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, NoReturn, final

from voice_ai_realtime_gateway.core.frames import AudioCodec
from voice_ai_realtime_gateway.telemetry.ids import InvalidTraceIdError, parse_session_id

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from voice_ai_realtime_gateway.telemetry.ids import SessionId

__all__ = [
    "DEFAULT_AUDIENCE",
    "MIN_KEY_BYTES",
    "AdmissionError",
    "AdmissionGrant",
    "TokenAuthenticator",
    "TokenSigner",
    "bearer_token",
]

DEFAULT_AUDIENCE: Final = "voice-gateway"
MIN_KEY_BYTES: Final = 32
MAX_DURATION_S: Final = 86_400.0
_SEGMENT: Final = re.compile(r"[A-Za-z0-9_-]+")
_TENANT: Final = re.compile(r"[A-Za-z0-9._-]{1,64}")
_BEARER: Final = re.compile(r"Bearer ([A-Za-z0-9_.-]+)", re.IGNORECASE)


class AdmissionError(Exception):
    """A connection was refused; ``reason`` is safe to send to the client."""

    def __init__(self, reason: str) -> None:
        """Refuse with a short public ``reason``."""
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True, kw_only=True)
class AdmissionGrant:
    """What a verified token entitles the connection to."""

    tenant: str
    codecs: frozenset[AudioCodec]
    max_duration_s: float
    expires_at: int
    session_id: SessionId | None
    token_id: str | None
    key_id: str


def _refuse(reason: str) -> NoReturn:
    raise AdmissionError(reason)


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64decode(segment: str) -> bytes:
    if _SEGMENT.fullmatch(segment) is None:
        _refuse("malformed token")
    try:
        return base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4))
    except (binascii.Error, ValueError):
        _refuse("malformed token")


def _json_object(raw: bytes) -> dict[str, object]:
    try:
        document = json.loads(raw)
    except (ValueError, RecursionError):
        _refuse("malformed token")
    if not isinstance(document, dict):
        _refuse("malformed token")
    return document


def bearer_token(headers: Mapping[str, str]) -> str:
    """Extract the bearer token from an ``Authorization`` header (case-insensitive name).

    Raises:
        AdmissionError: if the header is missing or not a bearer credential.
    """
    for name, value in headers.items():
        if name.lower() == "authorization":
            match = _BEARER.fullmatch(value.strip())
            if match is None:
                _refuse("unsupported authorization scheme")
            return match.group(1)
    _refuse("missing bearer token")


@final
class TokenSigner:
    """Issues tokens (for the control plane and tests)."""

    __slots__ = ("_key", "_key_id")

    def __init__(self, key_id: str, key: bytes) -> None:
        """Sign with ``key`` under ``key_id``.

        Raises:
            ValueError: if ``key`` is shorter than :data:`MIN_KEY_BYTES`.
        """
        if len(key) < MIN_KEY_BYTES:
            msg = f"HMAC keys must be at least {MIN_KEY_BYTES} bytes"
            raise ValueError(msg)
        self._key_id = key_id
        self._key = key

    def sign(
        self, claims: Mapping[str, object], *, header: Mapping[str, object] | None = None
    ) -> str:
        """Encode and sign ``claims`` (``header`` overrides are for negative tests)."""
        head = {"alg": "HS256", "typ": "JWT", "kid": self._key_id, **(header or {})}
        signing_input = (
            _b64encode(json.dumps(head, separators=(",", ":")).encode())
            + "."
            + _b64encode(json.dumps(dict(claims), separators=(",", ":")).encode())
        )
        signature = hmac.new(self._key, signing_input.encode("ascii"), hashlib.sha256).digest()
        return f"{signing_input}.{_b64encode(signature)}"


@final
class TokenAuthenticator:
    """Verifies admission tokens against a set of active keys."""

    __slots__ = ("_audience", "_keys", "_leeway_s", "_max_token_chars", "_wall_clock")

    def __init__(
        self,
        keys: Mapping[str, bytes],
        *,
        audience: str = DEFAULT_AUDIENCE,
        leeway_s: float = 30.0,
        max_token_chars: int = 4096,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        """Accept tokens signed by any key in ``keys`` (``kid`` -> secret).

        Raises:
            ValueError: on an empty key set, a short key, or a negative leeway.
        """
        if not keys:
            msg = "at least one key is required"
            raise ValueError(msg)
        if any(len(key) < MIN_KEY_BYTES for key in keys.values()):
            msg = f"HMAC keys must be at least {MIN_KEY_BYTES} bytes"
            raise ValueError(msg)
        if not (math.isfinite(leeway_s) and leeway_s >= 0):
            msg = "leeway_s must be finite and non-negative"
            raise ValueError(msg)
        self._keys = MappingProxyType(dict(keys))
        self._audience = audience
        self._leeway_s = leeway_s
        self._max_token_chars = max_token_chars
        self._wall_clock = wall_clock

    def verify(self, token: str) -> AdmissionGrant:
        """Verify ``token`` and return the grant it carries.

        Raises:
            AdmissionError: for any malformed, forged, expired or out-of-policy token.
        """
        if len(token) > self._max_token_chars or not token.isascii():
            _refuse("malformed token")
        segments = token.split(".")
        if len(segments) != 3:  # noqa: PLR2004 -- header.claims.signature
            _refuse("malformed token")
        encoded_header, encoded_claims, encoded_signature = segments
        header = _json_object(_b64decode(encoded_header))
        if header.get("alg") != "HS256":
            _refuse("unsupported algorithm")
        key_id = header.get("kid")
        key = self._keys.get(key_id) if isinstance(key_id, str) else None
        if key is None or not isinstance(key_id, str):
            _refuse("unknown key")
        expected = hmac.new(
            key, f"{encoded_header}.{encoded_claims}".encode("ascii"), hashlib.sha256
        ).digest()
        if not hmac.compare_digest(expected, _b64decode(encoded_signature)):
            _refuse("bad signature")
        claims = _json_object(_b64decode(encoded_claims))
        expires_at = self._check_times(claims)
        return AdmissionGrant(
            tenant=self._tenant(claims),
            codecs=self._codecs(claims),
            max_duration_s=self._duration(claims),
            expires_at=expires_at,
            session_id=self._session(claims),
            token_id=self._optional_text(claims, "jti"),
            key_id=key_id,
        )

    # ------------------------------------------------------------------ claims

    @staticmethod
    def _timestamp(claims: Mapping[str, object], name: str) -> float | None:
        value = claims.get(name)
        if value is None:
            return None
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
        ):
            _refuse("invalid claims")
        return float(value)

    def _check_times(self, claims: Mapping[str, object]) -> int:
        audience = claims.get("aud")
        audiences = audience if isinstance(audience, list) else [audience]
        if self._audience not in audiences:
            _refuse("wrong audience")
        now = self._wall_clock()
        expires = self._timestamp(claims, "exp")
        if expires is None:
            _refuse("invalid claims")
        if now - self._leeway_s >= expires:
            _refuse("token expired")
        for name in ("nbf", "iat"):
            moment = self._timestamp(claims, name)
            if moment is not None and moment > now + self._leeway_s:
                _refuse("token not yet valid")
        return int(expires)

    @staticmethod
    def _tenant(claims: Mapping[str, object]) -> str:
        tenant = claims.get("tenant")
        if not isinstance(tenant, str) or _TENANT.fullmatch(tenant) is None:
            _refuse("invalid claims")
        return tenant

    @staticmethod
    def _codecs(claims: Mapping[str, object]) -> frozenset[AudioCodec]:
        raw = claims.get("codecs")
        if not isinstance(raw, list) or not raw:
            _refuse("invalid claims")
        codecs: set[AudioCodec] = set()
        for value in raw:
            if not isinstance(value, str) or value not in AudioCodec:
                _refuse("invalid claims")
            codecs.add(AudioCodec(value))
        return frozenset(codecs)

    @staticmethod
    def _duration(claims: Mapping[str, object]) -> float:
        value = claims.get("max_duration_s")
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
            or not 0 < value <= MAX_DURATION_S
        ):
            _refuse("invalid claims")
        return float(value)

    @staticmethod
    def _session(claims: Mapping[str, object]) -> SessionId | None:
        value = claims.get("sid")
        if value is None:
            return None
        if not isinstance(value, str):
            _refuse("invalid claims")
        try:
            return parse_session_id(value)
        except InvalidTraceIdError:
            _refuse("invalid claims")

    @staticmethod
    def _optional_text(claims: Mapping[str, object], name: str) -> str | None:
        value = claims.get(name)
        if value is not None and not isinstance(value, str):
            _refuse("invalid claims")
        return value

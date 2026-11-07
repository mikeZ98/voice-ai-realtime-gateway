"""Production hardening: token admission, metrics, status endpoint and the gateway server."""

from voice_ai_realtime_gateway.production.admission import (
    DEFAULT_AUDIENCE,
    MIN_KEY_BYTES,
    AdmissionError,
    AdmissionGrant,
    TokenAuthenticator,
    TokenSigner,
    bearer_token,
)
from voice_ai_realtime_gateway.production.observability import (
    GatewayMetrics,
    SessionRegistry,
    TrackedCall,
)
from voice_ai_realtime_gateway.production.server import (
    Admission,
    CallContext,
    CallResult,
    GatewayServer,
    PipelineFactory,
    ServerConfig,
    TenantPolicy,
)
from voice_ai_realtime_gateway.production.status import StatusApp

__all__ = [
    "DEFAULT_AUDIENCE",
    "MIN_KEY_BYTES",
    "Admission",
    "AdmissionError",
    "AdmissionGrant",
    "CallContext",
    "CallResult",
    "GatewayMetrics",
    "GatewayServer",
    "PipelineFactory",
    "ServerConfig",
    "SessionRegistry",
    "StatusApp",
    "TenantPolicy",
    "TokenAuthenticator",
    "TokenSigner",
    "TrackedCall",
    "bearer_token",
]

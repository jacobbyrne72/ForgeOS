"""Gateway: the one call path from hive to any model, any provider.

Everything a caller needs is re-exported here so nothing outside this package
has to know whether a call went through HTTP or litellm, or reach into the
`client`/`health` submodules directly.
"""

from .client import (
    Gateway,
    GatewayRequest,
    GatewayResponse,
    HttpTransport,
    LiteLLMTransport,
    ModelUnavailableError,
    RateLimitError,
    RawCallResult,
    Transport,
    TransportError,
    assemble_prompt,
    default_transports,
    rate_limit_saturation,
)
from .dead_models import DeadModel, DeadModelStore
from .health import HealthTracker, ProviderHealth

__all__ = [
    "DeadModel",
    "DeadModelStore",
    "Gateway",
    "GatewayRequest",
    "GatewayResponse",
    "HealthTracker",
    "HttpTransport",
    "LiteLLMTransport",
    "ModelUnavailableError",
    "ProviderHealth",
    "RateLimitError",
    "RawCallResult",
    "Transport",
    "TransportError",
    "assemble_prompt",
    "default_transports",
    "rate_limit_saturation",
]

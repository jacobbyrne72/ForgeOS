"""Knowledge layer — three tiers, borrowed from the LLM-wiki pattern.

    raw       immutable sources, content-addressed (transcripts, vendor source, artifacts)
    claims    extracted, labelled, corroboration-counted  <- this package
    rules     promoted claims that agents may actually follow

The tier boundary is the point. Retrieval reads raw; agents read only promoted
rules. Nothing crosses from raw to rule without clearing the gate in claims.py.
"""

from .claims import (
    CLAIMS_SCHEMA,
    MIN_SOURCES_TO_PROMOTE,
    NEVER_PROMOTABLE,
    Claim,
    ClaimStore,
    ClaimType,
    PromotionResult,
    VerificationStatus,
    claim_key,
)

__all__ = [
    "CLAIMS_SCHEMA",
    "Claim",
    "ClaimStore",
    "ClaimType",
    "MIN_SOURCES_TO_PROMOTE",
    "NEVER_PROMOTABLE",
    "PromotionResult",
    "VerificationStatus",
    "claim_key",
]

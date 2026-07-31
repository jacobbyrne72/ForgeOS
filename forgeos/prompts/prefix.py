"""Byte-stable prompt prefixes and the cache-hit guarantee.

Cached input tokens cost roughly 10% of fresh ones, but a provider only serves
a cache hit when the prompt PREFIX is byte-identical to a previous call. So a
prompt has two parts:

    STABLE PREFIX   never varies for a given (role, version) -- role contract,
                     tool protocol, output schema, safety policy, evidence
                     requirements.
    VOLATILE TAIL    varies per task -- the specific task, its context
                     manifest, prior blocker.

Dynamic compression of the PREFIX is actively harmful: it changes bytes every
call and destroys the cache, so an 89% compression saving replaces a 90% cache
saving and comes out behind. Compress only the tail.

A prefix must be pure ASCII-safe deterministic text: no timestamps, no random
ids, no dict-ordering dependence, nothing that varies between processes or
between two builds in the same process. `fingerprint` exists to catch drift
mechanically instead of by review.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import tiktoken
from pydantic import BaseModel, field_validator

from forgeos.settings import Role

# ~4 chars/token is the standard rough approximation for English text --
# good enough to size a prefix for the terse-prefix sanity check below, not
# to bill it or to gate a hard per-provider cutoff. The cache floor further
# down needs an exact count instead, which is what the tiktoken encoding is
# for -- it reads a local encoding table, it never calls out to a live
# provider.
_CHARS_PER_TOKEN = 4


class StablePrefix(BaseModel):
    """The part of a prompt that must be byte-identical across calls.

    Identity is `(role, version)`. A new version is a new cache lineage --
    the correct way to change what a role says is to bump `version`, never to
    edit `text` under a version number that has already shipped.
    """

    role: Role
    version: int
    text: str

    @field_validator("text")
    @classmethod
    def _ascii_and_nonblank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("prefix text must not be blank")
        if not v.isascii():
            raise ValueError("prefix text must be ASCII-safe")
        return v

    @field_validator("version")
    @classmethod
    def _positive_version(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("version must be > 0")
        return v

    @property
    def fingerprint(self) -> str:
        """sha256 of the exact bytes. Two builds of the same prefix must match,
        in this process and in a fresh one -- that is the whole point of using
        a content hash instead of e.g. Python's salted `hash()`.
        """
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    @property
    def token_estimate(self) -> int:
        return max(1, len(self.text) // _CHARS_PER_TOKEN)

    @property
    def key(self) -> tuple[Role, int]:
        return (self.role, self.version)


class Prompt(BaseModel):
    """A built prompt: stable prefix then volatile tail, in that order.

    `cacheable_prefix_len` is the byte length of `full_text` a provider can
    serve from cache. It is exactly `len(prefix text)` because the tail is
    appended immediately after it and never overlaps or reorders it.
    """

    full_text: str
    prefix_fingerprint: str
    cacheable_prefix_len: int


def build_prompt(prefix: StablePrefix, tail: str) -> Prompt:
    """Concatenate prefix then tail, prefix first, unmodified, every call.

    That ordering -- and only ever appending, never editing, the prefix --
    is the entire cache-hit guarantee. Changing a single byte of the prefix
    for one call and not the next turns every call into a cache miss.
    """
    return Prompt(
        full_text=prefix.text + tail,
        prefix_fingerprint=prefix.fingerprint,
        cacheable_prefix_len=len(prefix.text),
    )


# --------------------------------------------------- provider cache mechanics


@dataclass(frozen=True)
class ProviderCacheRules:
    """One provider's real cache mechanics, as data.

    Every field comes from a provider's own docs (see
    docs/research/cost-reduction.md), not from guessing at what "should"
    work. Branch on these fields, never on a `provider == "anthropic"`
    string check scattered through call sites -- a rule that changes (a
    raised floor, a new TTL tier) should mean editing one row here, not
    hunting every place a provider name was compared.
    """

    provider: str
    max_cache_breakpoints: int | None  # None: caching is automatic, no explicit-marker limit applies
    min_cacheable_tokens: int
    default_ttl_seconds: int
    extended_ttl_seconds: int | None  # None: no opt-in longer TTL exists
    requires_byte_identical_prefix: bool


PROVIDER_CACHE_RULES: dict[str, ProviderCacheRules] = {
    "anthropic": ProviderCacheRules(
        provider="anthropic",
        max_cache_breakpoints=4,
        min_cacheable_tokens=1024,
        default_ttl_seconds=300,
        extended_ttl_seconds=3600,
        requires_byte_identical_prefix=True,
    ),
    "openai": ProviderCacheRules(
        provider="openai",
        max_cache_breakpoints=None,
        min_cacheable_tokens=1024,
        default_ttl_seconds=300,
        extended_ttl_seconds=None,
        requires_byte_identical_prefix=True,
    ),
}

# Reference encoding for the cache-floor check only -- not a billing count.
# Neither provider's real tokeniser is what matters here: Anthropic's isn't
# exposed through tiktoken at all, and OpenAI's varies by model. cl100k_base
# is a stable, local encoding close enough to gate a hard threshold like "at
# least 1024 tokens" without depending on a live provider call.
_CACHE_FLOOR_ENCODING = tiktoken.get_encoding("cl100k_base")

# 4+ consecutive digits reads as a year, an epoch value, or a generated id --
# the same heuristic the role-prefix test suite already checks by hand.
# Anything that varies between processes or calls belongs in the tail, never
# the prefix; this is the mechanical check for the most common way that rule
# gets broken by accident.
_TIMESTAMP_LIKE_RE = re.compile(r"\d{4,}")


def meets_cache_floor(text: str, provider: str) -> bool:
    """Whether `text` is long enough to ever be served from `provider`'s cache.

    Below the floor, cache mechanics don't degrade gracefully -- they don't
    apply at all. OpenAI's automatic caching simply never activates under
    1024 tokens; Anthropic refuses to write a cache entry that short. A
    byte-identical prefix under the floor still never earns a cache hit.

    Raises KeyError for a provider with no entry in `PROVIDER_CACHE_RULES`:
    this is a check made deliberately before relying on caching, so
    silently assuming a floor for an unmodeled provider would be worse than
    refusing to answer.
    """
    rules = PROVIDER_CACHE_RULES[provider]
    tokens = len(_CACHE_FLOOR_ENCODING.encode(text, disallowed_special=()))
    return tokens >= rules.min_cacheable_tokens


def validate_prefix(prefix: StablePrefix, provider: str, *, breakpoints: int = 1) -> list[str]:
    """Actionable findings for marking `prefix` as a cache breakpoint on `provider`.

    Empty list means it is safe to mark as assembled. Every finding is a
    plain sentence naming the problem and why it matters -- this is meant to
    be read by whoever is about to wire a cache_control marker, not parsed
    by a machine.

    `breakpoints` is how many cache breakpoints the caller intends to mark
    across the full assembled prompt. Today's `build_prompt` always produces
    exactly one -- the prefix/tail boundary -- but Anthropic allows marking
    several independent blocks in one request (e.g. a system block and a
    tool-definitions block), so this stays a parameter instead of a
    hardcoded 1.
    """
    findings: list[str] = []

    rules = PROVIDER_CACHE_RULES.get(provider)
    if rules is None:
        findings.append(
            f"no cache rules modeled for provider {provider!r} -- add it to "
            "PROVIDER_CACHE_RULES before relying on this check"
        )
    else:
        if rules.max_cache_breakpoints is not None and breakpoints > rules.max_cache_breakpoints:
            findings.append(
                f"{breakpoints} cache breakpoints requested but {provider} allows at most "
                f"{rules.max_cache_breakpoints}; the excess will be ignored or rejected, not "
                "silently combined into one"
            )
        if not meets_cache_floor(prefix.text, provider):
            findings.append(
                f"prefix is below {provider}'s {rules.min_cacheable_tokens}-token cache "
                "floor -- it will never be served from cache, and marking it as a "
                "breakpoint anyway pays a write surcharge for a discount that can never land"
            )

    match = _TIMESTAMP_LIKE_RE.search(prefix.text)
    if match:
        findings.append(
            f"prefix contains a non-deterministic-looking segment {match.group()!r} at "
            f"offset {match.start()} (4+ consecutive digits reads as a timestamp, epoch "
            "value, or generated id) -- a prefix that varies between calls destroys the "
            "cache it exists to earn"
        )

    return findings


class PrefixRegistry:
    """Registered `(role, version)` -> `StablePrefix`, with drift detection.

    Silent prefix drift -- the same `(role, version)` key quietly pointing at
    different text between two calls -- is the exact failure this class
    exists to catch. Bumping the version is the correct way to change a
    prefix; `verify_stability` is the guard against doing it by accident.
    """

    def __init__(self) -> None:
        self._prefixes: dict[tuple[Role, int], StablePrefix] = {}

    def register(self, prefix: StablePrefix) -> None:
        self._prefixes[prefix.key] = prefix

    def get(self, role: Role, version: int) -> StablePrefix | None:
        return self._prefixes.get((role, version))

    def verify_stability(self, role: Role, version: int, text: str) -> bool:
        """False iff `(role, version)` is already registered with different text.

        An unregistered `(role, version)` is not drift -- there is nothing to
        compare against yet, so this returns True. Only a registered key whose
        text has changed underneath it is drift.
        """
        existing = self.get(role, version)
        if existing is None:
            return True
        return existing.text == text


__all__ = [
    "PROVIDER_CACHE_RULES",
    "PrefixRegistry",
    "ProviderCacheRules",
    "Prompt",
    "StablePrefix",
    "build_prompt",
    "meets_cache_floor",
    "validate_prefix",
]

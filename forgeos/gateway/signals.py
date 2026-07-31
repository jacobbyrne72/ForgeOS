"""Per-provider error-text -> exhaustion-signal corpus.

`core/quota.py` already has the right *state machine*: `ExhaustionSignal`
(VENDOR_EXHAUSTED rotates, RATE_LIMITED pauses for `retry_after`, NETWORK_ERROR
does neither) plus `should_rotate` branching on it. What it lacks is a wide,
maintained answer to "what does provider X's error body actually say when it
means quota-exhausted vs. merely rate-limited" -- today callers have to infer
that per-provider, ad hoc, from thin evidence (mostly a bare HTTP status code).

This module is that answer: a documented table of real per-provider error
patterns (`docs/research/router-9router.md` Port #1), mined from
`vendor/9router/open-sse/config/errorConfig.js` (`ERROR_RULES`, its own
provider-agnostic classification table) and its executors' provider-specific
`parseError` overrides (`open-sse/executors/{codex,gemini-cli}.js`), plus each
vendor's own documented API error format. It intentionally ports only DATA --
error strings and status codes -- never any of 9router's account-pooling,
credential, or interception machinery. See
`docs/research/router-9router.md`'s DO-NOT-PORT section: nothing here reads a
credential, installs a certificate, spoofs a client fingerprint, rotates an
IP, or harvests an OAuth token. It classifies text that ForgeOS's own,
already-authorized transports (`gateway/client.py`) received directly from a
provider they called on the operator's own behalf.

Safety property -- UNKNOWN is a first-class outcome, not an edge case:
`classify_provider_error` returns `None`, distinct from all three
`ExhaustionSignal` members, whenever a provider/status/text combination
matches nothing in the corpus. A caller MUST treat `None` as "learned
nothing" and leave existing quota/health state untouched -- never default it
to `VENDOR_EXHAUSTED` (which would rotate a perfectly good seat off for a
message that just happened to look scary) or to `RATE_LIMITED` (which would
park a seat that was never actually told to back off). This mirrors
`parse_usage_report`'s own rule in `core/quota.py`: an absent fact must stay
absent, because a caller trusts what this module hands it.
"""

from __future__ import annotations

import re
from typing import NamedTuple

from ..core.quota import ExhaustionSignal


class _Rule(NamedTuple):
    """One row of the corpus.

    Matches when: `status` is None or equals the observed status code, AND
    `text_any` is empty or at least one of its substrings is present, AND
    `text_all` is empty or every one of its substrings is present. At least
    one of `status`/`text_any`/`text_all` must be set -- an empty rule would
    match everything, silently swallowing whatever came after it.

    Text rules are listed before status-only rules within each provider's
    table, and checked in that order: a specific phrase should win over a
    bare status code that many different conditions share. Same convention
    9router's own `ERROR_RULES` uses (`errorConfig.js`: "text rules first,
    then status rules").
    """

    signal: ExhaustionSignal
    status: int | None = None
    text_any: tuple[str, ...] = ()
    text_all: tuple[str, ...] = ()


def _matches(rule: _Rule, status_code: int | None, low_text: str) -> bool:
    if rule.status is not None and rule.status != status_code:
        return False
    if rule.text_any and not any(s in low_text for s in rule.text_any):
        return False
    if rule.text_all and not all(s in low_text for s in rule.text_all):
        return False
    return True


# --------------------------------------------------------------- the corpus
#
# Every entry below is either a direct receipt (a real error shape mined from
# vendor/9router) or a vendor's own publicly documented error format. Entries
# that 9router itself could not disambiguate (its `ERROR_RULES` only needs a
# boolean "should this account fall back", not a three-way quota/rate/network
# split) are deliberately narrowed here rather than guessed wide -- see the
# module docstring's safety property.

_ANTHROPIC_RULES: tuple[_Rule, ...] = (
    # Claude subscription (Claude Code CLI / claude.ai Pro-Max OAuth) usage
    # cap. Not an HTTP status by itself -- this is the same "usage limit"
    # phrasing `core.quota.parse_usage_report` already keys off of for CLI
    # usage reports, kept consistent here for the HTTP-error-body case.
    _Rule(ExhaustionSignal.VENDOR_EXHAUSTED, text_any=("usage limit", "usage_limit")),
    # Anthropic's documented capacity signal (status 529, body
    # `{"type":"error","error":{"type":"overloaded_error",...}}`). This is
    # NOT the caller's quota -- Anthropic is momentarily out of capacity for
    # everyone -- so it pauses (RATE_LIMITED) rather than rotates: the seat
    # itself is fine.
    _Rule(ExhaustionSignal.RATE_LIMITED, text_any=("overloaded_error", "overloaded")),
    # Anthropic's documented rate-limit error type.
    _Rule(ExhaustionSignal.RATE_LIMITED, text_any=("rate_limit_error",)),
    _Rule(ExhaustionSignal.RATE_LIMITED, status=529),
    _Rule(ExhaustionSignal.RATE_LIMITED, status=429),
)

_OPENAI_RULES: tuple[_Rule, ...] = (
    # OpenAI's own documented distinction (platform.openai.com/docs/guides/
    # error-codes): a 429 body's `code` field is `insufficient_quota` for a
    # billing/quota cutoff (no self-heal -- needs the account topped up) vs
    # `rate_limit_exceeded` for ordinary RPM/TPM throttling (self-heals).
    # Conflating them was the exact failure mode this corpus exists to fix:
    # treating a dead billing key as an ordinary 30s-retry rate limit means
    # retrying it forever, once every window, for nothing.
    _Rule(ExhaustionSignal.VENDOR_EXHAUSTED, text_any=("insufficient_quota",)),
    # Human-readable default message for the same code -- mined verbatim
    # from vendor/9router/open-sse/config/errorConfig.js
    # DEFAULT_ERROR_MESSAGES[403] = "You exceeded your current quota".
    _Rule(ExhaustionSignal.VENDOR_EXHAUSTED, text_any=("exceeded your current quota",)),
    _Rule(ExhaustionSignal.VENDOR_EXHAUSTED, text_any=("billing_hard_limit_reached", "billing hard limit")),
    _Rule(ExhaustionSignal.RATE_LIMITED, text_any=("rate_limit_exceeded",)),
    _Rule(ExhaustionSignal.RATE_LIMITED, status=429),
)

_DEEPSEEK_RULES: tuple[_Rule, ...] = (
    # DeepSeek is balance-based, not window-quota-based -- it has no "resets
    # in Nh" concept at all (receipt: vendor/9router/open-sse/services/
    # usage/deepseek.js polls GET /user/balance, never a reset timestamp).
    # "Insufficient Balance" is DeepSeek's documented 402 body for a drained
    # account: gone until the operator adds funds, which is exactly what
    # VENDOR_EXHAUSTED with no known reset means.
    _Rule(ExhaustionSignal.VENDOR_EXHAUSTED, text_any=("insufficient balance",)),
    _Rule(ExhaustionSignal.VENDOR_EXHAUSTED, status=402),
    _Rule(ExhaustionSignal.RATE_LIMITED, text_any=("rate limit",)),
    _Rule(ExhaustionSignal.RATE_LIMITED, status=429),
)

_OPENROUTER_RULES: tuple[_Rule, ...] = (
    # OpenRouter's documented credit-exhaustion shape: 402 with a message
    # about running out of credits. Distinct from a stale free-tier model
    # slug (404), which `gateway/client.py` already classifies separately as
    # `ModelUnavailableError` -- not this module's concern.
    _Rule(ExhaustionSignal.VENDOR_EXHAUSTED, text_any=("out of credits", "not enough credits", "insufficient_quota")),
    _Rule(ExhaustionSignal.VENDOR_EXHAUSTED, status=402),
    _Rule(ExhaustionSignal.RATE_LIMITED, text_any=("rate limit", "rate-limited", "rate_limit")),
    _Rule(ExhaustionSignal.RATE_LIMITED, status=429),
)

_GOOGLE_RULES: tuple[_Rule, ...] = (
    # Google/Vertex overload one gRPC status for both meanings: RPM/TPM
    # throttling and a daily/monthly quota cap both surface as
    # `"status": "RESOURCE_EXHAUSTED"` at HTTP 429 (ai.google.dev/gemini-api/
    # docs/rate-limits). The message text is the only disambiguator Google
    # gives; "per day"/"daily" in it means the caller hit the long-window
    # cap, not the short one.
    _Rule(ExhaustionSignal.VENDOR_EXHAUSTED, text_all=("resource_exhausted", "per day")),
    _Rule(ExhaustionSignal.VENDOR_EXHAUSTED, text_all=("resource_exhausted", "daily")),
    # Otherwise RESOURCE_EXHAUSTED is the ordinary per-minute limit, which
    # typically carries a short `RetryInfo.retryDelay` (receipt:
    # vendor/9router/open-sse/executors/gemini-cli.js:37-54).
    _Rule(ExhaustionSignal.RATE_LIMITED, text_any=("resource_exhausted",)),
    _Rule(ExhaustionSignal.RATE_LIMITED, status=429),
)

# Provider-agnostic fallback -- used when the provider isn't one of the named
# tables above, or as the last tier after a named provider's specific rules
# don't match. Mined directly from vendor/9router/open-sse/config/
# errorConfig.js `ERROR_RULES`, restricted to the phrases that map
# unambiguously onto our three-way split. 9router's own "no credentials" /
# "request not allowed" / "improperly formed request" / bare 401 / 402 / 403
# / 404 rules are deliberately NOT ported here: those are auth/config/
# malformed-request signals, not evidence about quota state, and guessing
# VENDOR_EXHAUSTED or RATE_LIMITED for a broken request that rotating won't
# fix is exactly the wrong call this corpus exists to prevent.
_GENERIC_RULES: tuple[_Rule, ...] = (
    _Rule(ExhaustionSignal.VENDOR_EXHAUSTED, text_any=("quota exceeded",)),
    _Rule(ExhaustionSignal.RATE_LIMITED, text_any=("rate limit", "too many requests")),
    # 9router applies the same exponential backoff to "overloaded" and
    # "capacity" as it does to a rate limit (errorConfig.js `ERROR_RULES`) --
    # vendor-side capacity, not the caller's exhausted seat, so pause rather
    # than rotate.
    _Rule(ExhaustionSignal.RATE_LIMITED, text_any=("overloaded", "capacity")),
    # Bare 429 with no recognizable text: HTTP's own "Too Many Requests"
    # status, which is what `gateway/client.py` already infers today for any
    # unrecognized provider. Kept as the final tier so a real quota-exhaustion
    # phrase above always wins first.
    _Rule(ExhaustionSignal.RATE_LIMITED, status=429),
)

_PROVIDER_RULES: dict[str, tuple[_Rule, ...]] = {
    "anthropic": _ANTHROPIC_RULES,
    "openai": _OPENAI_RULES,
    "deepseek": _DEEPSEEK_RULES,
    "openrouter": _OPENROUTER_RULES,
    "google": _GOOGLE_RULES,
}


def _first_match(rules: tuple[_Rule, ...], status_code: int | None, low_text: str) -> ExhaustionSignal | None:
    for rule in rules:
        if _matches(rule, status_code, low_text):
            return rule.signal
    return None


def classify_provider_error(provider: str, status_code: int | None, text: str) -> ExhaustionSignal | None:
    """What a provider's error actually told us, or `None` if it told us nothing.

    Tries `provider`'s own table first (more specific beats generic), then
    falls back to the provider-agnostic table so a provider not yet in the
    corpus -- or one whose specific rules didn't match this particular body --
    still gets the benefit of patterns real vendors converge on.

    Returns `None`, not a guess, when nothing matches. See the module
    docstring's safety property: a caller must treat `None` as "no new
    information" and leave whatever quota/health state it already has alone.
    """
    low_text = (text or "").lower()
    specific = _PROVIDER_RULES.get(provider.lower())
    if specific is not None:
        matched = _first_match(specific, status_code, low_text)
        if matched is not None:
            return matched
    return _first_match(_GENERIC_RULES, status_code, low_text)


# ----------------------------------------------------------------- retry-after
#
# Distinct from `core.quota._RESET_PATTERNS`: those parse hours-scale
# subscription-window resets out of a CLI usage report ("resets in 2h 15m").
# These parse second-scale HTTP backoff hints a provider embeds in a 429/503
# error BODY, for use only when the `Retry-After` header itself is absent.

_RETRY_AFTER_TEXT_PATTERNS: tuple[re.Pattern, ...] = (
    # Google/Vertex RetryInfo, inside error.details[]: `"retryDelay": "41s"`
    # (receipt: vendor/9router/open-sse/executors/gemini-cli.js:37-54).
    re.compile(r'retrydelay"?\s*:\s*"?(?P<seconds>\d+(?:\.\d+)?)\s*s', re.I),
    # A provider that echoes the Retry-After value as body text instead of
    # (or in addition to) the header.
    re.compile(r"retry[- ]after[:\s]+(?P<seconds>\d+(?:\.\d+)?)", re.I),
    # "please try again in 20s" / "retry after 20 seconds" (OpenAI-style and
    # generic natural-language phrasing).
    re.compile(r"(?:try again|retry)\s+(?:in|after)\s+(?P<seconds>\d+(?:\.\d+)?)\s*s(?:ec(?:ond)?s?)?\b", re.I),
)


def extract_retry_after(text: str) -> float | None:
    """Seconds to wait before retrying, read from an error body's own text.

    Returns `None` when the text carries no such hint, rather than inventing
    a backoff -- the same rule `parse_usage_report` follows for reset times
    in `core/quota.py`: a fabricated number is worse than none, because a
    caller would trust it. The caller (`gateway/client.py`'s `HttpTransport`)
    already falls back to a fixed default when this returns `None`; this
    function's job is only to surface a real number when one is actually
    present, never to pick one.
    """
    if not text:
        return None
    for pattern in _RETRY_AFTER_TEXT_PATTERNS:
        match = pattern.search(text)
        if match:
            return float(match.group("seconds"))
    return None


__all__ = [
    "classify_provider_error",
    "extract_retry_after",
]

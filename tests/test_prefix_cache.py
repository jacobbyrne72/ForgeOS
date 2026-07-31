"""Tests for provider cache mechanics layered on the byte-stable prefix.

Cached input tokens cost roughly 10% of fresh ones, but only if the
provider's own cache mechanics are respected: Anthropic will not accept
more than 4 explicit cache breakpoints and refuses to write a cache entry
under 1024 tokens, and OpenAI's automatic caching simply never activates
below that same 1024-token floor. These tests hold `prefix.py`'s modeling
of those rules to the provider docs cited in docs/research/cost-reduction.md,
and prove the one guarantee the whole module exists for: two independently
built prefixes from the same inputs are byte-identical, and a changed tail
never touches the prefix's bytes.
"""

from __future__ import annotations

import pytest

from forgeos.prompts.prefix import (
    PROVIDER_CACHE_RULES,
    ProviderCacheRules,
    StablePrefix,
    build_prompt,
    meets_cache_floor,
    validate_prefix,
)
from forgeos.settings import Role


def _long_text(repeats: int) -> str:
    """ASCII text comfortably over a token count for a given `repeats`."""
    return "stable prefix content word " * repeats


# ------------------------------------------------------- ProviderCacheRules


def test_anthropic_rules_match_documented_limits():
    rules = PROVIDER_CACHE_RULES["anthropic"]
    assert rules.max_cache_breakpoints == 4
    assert rules.min_cacheable_tokens == 1024
    assert rules.default_ttl_seconds == 300
    assert rules.extended_ttl_seconds == 3600
    assert rules.requires_byte_identical_prefix is True


def test_openai_rules_match_documented_limits():
    rules = PROVIDER_CACHE_RULES["openai"]
    assert rules.max_cache_breakpoints is None
    assert rules.min_cacheable_tokens == 1024
    assert rules.requires_byte_identical_prefix is True


def test_provider_cache_rules_is_real_data_not_a_string_branch():
    """Modeled as data (ProviderCacheRules instances), not an if/elif chain
    keyed on provider name strings scattered through call sites."""
    assert isinstance(PROVIDER_CACHE_RULES["anthropic"], ProviderCacheRules)
    assert isinstance(PROVIDER_CACHE_RULES["openai"], ProviderCacheRules)


# ----------------------------------------------------------- cache floor


def test_meets_cache_floor_true_for_a_long_prefix():
    long_text = _long_text(1000)
    assert meets_cache_floor(long_text, "anthropic") is True
    assert meets_cache_floor(long_text, "openai") is True


def test_meets_cache_floor_false_for_a_short_prefix():
    short_text = "you are a helpful assistant."
    assert meets_cache_floor(short_text, "anthropic") is False
    assert meets_cache_floor(short_text, "openai") is False


def test_meets_cache_floor_raises_for_unmodeled_provider():
    with pytest.raises(KeyError):
        meets_cache_floor("anything", "some-unmodeled-provider")


# -------------------------------------------------------------- validate_prefix


def test_validate_prefix_clean_for_a_long_ascii_prefix():
    prefix = StablePrefix(role=Role.IMPLEMENTER, version=1, text=_long_text(1000))
    assert validate_prefix(prefix, "anthropic") == []


def test_validate_prefix_flags_below_floor():
    prefix = StablePrefix(role=Role.IMPLEMENTER, version=1, text="short prefix text")
    findings = validate_prefix(prefix, "anthropic")
    assert any("floor" in f for f in findings)


def test_validate_prefix_flags_too_many_breakpoints():
    prefix = StablePrefix(role=Role.IMPLEMENTER, version=1, text=_long_text(1000))
    findings = validate_prefix(prefix, "anthropic", breakpoints=5)
    assert any("breakpoint" in f for f in findings)


def test_validate_prefix_within_breakpoint_limit_is_clean():
    prefix = StablePrefix(role=Role.IMPLEMENTER, version=1, text=_long_text(1000))
    assert validate_prefix(prefix, "anthropic", breakpoints=4) == []


def test_validate_prefix_openai_has_no_breakpoint_ceiling():
    """OpenAI caches automatically -- there is no explicit-marker limit to
    exceed, so any breakpoint count is fine to request."""
    prefix = StablePrefix(role=Role.IMPLEMENTER, version=1, text=_long_text(1000))
    findings = validate_prefix(prefix, "openai", breakpoints=99)
    assert not any("breakpoint" in f for f in findings)


def test_validate_prefix_flags_nondeterministic_looking_segment():
    text = _long_text(50) + " epoch 1753900800 marker"
    prefix = StablePrefix(role=Role.IMPLEMENTER, version=1, text=text)
    findings = validate_prefix(prefix, "anthropic")
    assert any("non-deterministic" in f for f in findings)


def test_validate_prefix_flags_unmodeled_provider():
    prefix = StablePrefix(role=Role.IMPLEMENTER, version=1, text=_long_text(1000))
    findings = validate_prefix(prefix, "some-unmodeled-provider")
    assert any("no cache rules modeled" in f for f in findings)


def test_validate_prefix_unmodeled_provider_still_flags_timestamp():
    """An unrecognized provider skips the floor/breakpoint checks (nothing
    is modeled to check against) but the deterministic-text check is
    provider-agnostic and still runs."""
    text = _long_text(50) + " epoch 1753900800 marker"
    prefix = StablePrefix(role=Role.IMPLEMENTER, version=1, text=text)
    findings = validate_prefix(prefix, "some-unmodeled-provider")
    assert any("no cache rules modeled" in f for f in findings)
    assert any("non-deterministic" in f for f in findings)


# ------------------------------------------------- the cache-hit guarantee


def test_build_prompt_is_byte_identical_across_two_independent_assemblies():
    """THE guarantee the whole module exists for: build the same prefix
    twice from scratch (two separate StablePrefix constructions, not the
    same object) and assemble each with a different tail. The prefix bytes
    -- what a provider would actually serve from cache -- must be identical
    both times, and the tail change must never touch them.
    """
    text = _long_text(1000)
    prefix_a = StablePrefix(role=Role.PLANNER, version=1, text=text)
    prefix_b = StablePrefix(role=Role.PLANNER, version=1, text=text)

    prompt_1 = build_prompt(prefix_a, "\n\nTASK: short.")
    prompt_2 = build_prompt(prefix_b, "\n\nTASK: a completely different, much longer task body.")

    prefix_bytes_1 = prompt_1.full_text[: prompt_1.cacheable_prefix_len].encode("utf-8")
    prefix_bytes_2 = prompt_2.full_text[: prompt_2.cacheable_prefix_len].encode("utf-8")

    assert prefix_bytes_1 == prefix_bytes_2
    assert prompt_1.prefix_fingerprint == prompt_2.prefix_fingerprint
    assert prompt_1.cacheable_prefix_len == prompt_2.cacheable_prefix_len
    assert meets_cache_floor(text, "anthropic") is True

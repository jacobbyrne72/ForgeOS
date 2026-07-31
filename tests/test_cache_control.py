"""Anthropic prompt-cache breakpoints — the discount that never fired.

ForgeOS assembles every prompt as `prefix + tail` and keeps the prefix
byte-stable, and the README has long claimed "~90% off cached input". On OpenAI
and DeepSeek that works: they cache automatically on an exact prefix match.

**Anthropic does not.** It caches only a block explicitly marked with
`cache_control: {"type": "ephemeral"}`, and the codebase emitted that marker
exactly nowhere. So on Claude — the largest subscription spend here — the
discount had never once applied. The byte-stable prefix was necessary and not
sufficient, and nothing in the test suite noticed.

The guard that matters as much as the feature: a cache **write** is billed at
roughly 1.25x normal input. Marking a block too short for the provider to cache
buys the surcharge and no discount, so a minimum-token floor is not an
optimisation — it is what stops this feature costing money.
"""

from __future__ import annotations

import pytest

from forgeos.gateway.client import (
    CACHE_CONTROL_MIN_TOKENS,
    HttpTransport,
    _content_blocks,
)


def _long(tokens: int = CACHE_CONTROL_MIN_TOKENS + 200) -> str:
    """A prefix comfortably over the cacheable floor."""
    return "stable role contract. " * tokens


# ------------------------------------------------------- the marker itself


def test_a_long_prefix_is_marked_as_a_cache_breakpoint():
    prefix = _long()
    blocks = _content_blocks(prefix + "the task", prefix, mark_cache=True)
    assert isinstance(blocks, list)
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}
    assert blocks[0]["text"] == prefix


def test_the_tail_is_a_separate_unmarked_block():
    """Marking the tail would defeat the point — it changes every call, so it
    could never be read back from cache."""
    prefix = _long()
    blocks = _content_blocks(prefix + "the volatile task", prefix, mark_cache=True)
    assert blocks[1]["text"] == "the volatile task"
    assert "cache_control" not in blocks[1]


def test_prefix_and_tail_together_reproduce_the_prompt_exactly():
    """The wire payload must carry the same text as before, byte for byte. A
    caching optimisation that alters the prompt is a correctness change."""
    prefix = _long()
    prompt = prefix + "do the thing"
    blocks = _content_blocks(prompt, prefix, mark_cache=True)
    assert "".join(b["text"] for b in blocks) == prompt


# ----------------------------------------------------------- the cost guard


def test_a_prefix_below_the_floor_is_not_marked():
    """The load-bearing guard. Anthropic refuses to cache a short block, and a
    cache write costs ~1.25x normal input — so marking it is a pure surcharge."""
    short = "you are a helpful assistant."
    out = _content_blocks(short + "task", short, mark_cache=True)
    assert isinstance(out, str), "a sub-floor prefix must not be marked"


def test_the_floor_matches_anthropics_documented_minimum():
    assert CACHE_CONTROL_MIN_TOKENS >= 1024


# ------------------------------------------------- never change other wires


def test_a_provider_without_cache_control_gets_a_plain_string():
    """OpenAI, DeepSeek and local models must see exactly the payload they saw
    before. Sending Anthropic content blocks to them is ignored at best and a
    400 at worst."""
    prefix = _long()
    out = _content_blocks(prefix + "task", prefix, mark_cache=False)
    assert isinstance(out, str)
    assert out == prefix + "task"


def test_no_prefix_means_a_plain_string():
    out = _content_blocks("just a prompt", "", mark_cache=True)
    assert isinstance(out, str)


def test_a_prefix_that_is_not_actually_a_prefix_is_refused():
    """If the caller's prefix does not match the prompt's opening bytes,
    splitting on it would send different text than intended."""
    out = _content_blocks("totally different text", _long(), mark_cache=True)
    assert isinstance(out, str)


# --------------------------------------------------- capability is derived


@pytest.mark.parametrize("name,serves,expected", [
    ("anthropic", {"anthropic"}, True),
    ("openrouter", {"openrouter"}, True),
    ("deepseek", {"deepseek"}, False),
    ("openai", {"openai"}, False),
    ("gateway", set(), True),          # fan-out may route to Anthropic
    ("ollama", {"ollama"}, False),
])
def test_cache_control_support_is_derived_from_what_a_transport_serves(
    name, serves, expected
):
    """Derived, not configured — so adding a provider cannot silently opt it in
    to markers its API will reject."""
    t = HttpTransport(base_url="https://x", name=name, serves=serves or None)
    assert t.supports_cache_control is expected


def test_litellm_never_gets_content_blocks():
    """litellm shapes payloads per provider itself; handing it Anthropic blocks
    would double-encode them."""
    from forgeos.gateway.client import LiteLLMTransport

    assert LiteLLMTransport.supports_cache_control is False

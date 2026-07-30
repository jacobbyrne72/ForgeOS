"""Tests for the prompt layer: byte-stable prefixes and the cache-hit guarantee.

Cached input tokens cost roughly 10% of fresh ones, but only when the prompt
prefix is byte-identical to a previous call. These tests exist to catch the
two ways that guarantee breaks silently: a prefix that is not actually
deterministic (varies by process or by rebuild), and a prefix that drifts
under a version number that was already shipped.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from hive.prompts.prefix import PrefixRegistry, StablePrefix, build_prompt
from hive.prompts.roles import ROLE_PREFIXES, PREFIX_VERSION, default_prefix_registry, role_prefix
from hive.settings import Role

REPO_ROOT = Path(__file__).resolve().parents[1]

# Domain vocabulary that must never leak into hive -- it is domain-agnostic
# infrastructure, not a script for whatever project it happens to be pointed
# at. "trading" is the example named explicitly; the rest are the same
# category of leak.
_DOMAIN_WORDS = ("trading", "trade", "forex", "ftmo", "backtest", "finance", "stock", "crypto")


# ------------------------------------------------------------------ coverage


def test_every_role_has_a_prefix():
    for role in Role:
        prefix = role_prefix(role)
        assert isinstance(prefix, StablePrefix)
        assert prefix.role is role


def test_role_prefixes_dict_has_no_extra_or_missing_roles():
    assert set(ROLE_PREFIXES) == set(Role)


# --------------------------------------------------------------- determinism


def test_fingerprint_stable_across_repeated_construction():
    for role in Role:
        text = role_prefix(role).text
        a = StablePrefix(role=role, version=PREFIX_VERSION, text=text)
        b = StablePrefix(role=role, version=PREFIX_VERSION, text=text)
        assert a.fingerprint == b.fingerprint
        assert a.fingerprint == role_prefix(role).fingerprint


def test_fingerprint_stable_across_a_fresh_process():
    """The in-process fingerprints must match a brand new interpreter's.

    A hash built from something process-local (salted `hash()`, object id,
    dict-iteration order over an unordered container) would pass every
    in-process test and still break the cache in production, because
    production calls are never all the same process.
    """
    script = (
        "import json\n"
        "from hive.prompts.roles import ROLE_PREFIXES\n"
        "print(json.dumps({r.value: p.fingerprint for r, p in ROLE_PREFIXES.items()}))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    fresh = json.loads(result.stdout.strip())

    in_process = {role.value: role_prefix(role).fingerprint for role in Role}
    assert fresh == in_process


# ------------------------------------------------------------ drift detection


def test_verify_stability_true_for_unregistered_role_and_version():
    registry = PrefixRegistry()
    assert registry.verify_stability(Role.IMPLEMENTER, 1, "anything") is True


def test_verify_stability_true_when_text_matches_registered():
    registry = PrefixRegistry()
    prefix = StablePrefix(role=Role.VERIFIER, version=1, text="Verify with real output.")
    registry.register(prefix)
    assert registry.verify_stability(Role.VERIFIER, 1, "Verify with real output.") is True


def test_verify_stability_catches_drift_for_a_reused_version():
    registry = PrefixRegistry()
    original = StablePrefix(role=Role.VERIFIER, version=1, text="Verify with real output.")
    registry.register(original)

    # Same (role, version) key, different text: this is the drift case.
    assert registry.verify_stability(Role.VERIFIER, 1, "Verify with real output, mostly.") is False

    # A different version is a new lineage, not drift.
    assert registry.verify_stability(Role.VERIFIER, 2, "Verify with real output, mostly.") is True


def test_default_prefix_registry_matches_every_registered_role():
    registry = default_prefix_registry()
    for role in Role:
        prefix = role_prefix(role)
        assert registry.get(role, PREFIX_VERSION) is not None
        assert registry.verify_stability(role, PREFIX_VERSION, prefix.text) is True


# --------------------------------------------------------------- ascii/domain


@pytest.mark.parametrize("role", list(Role))
def test_prefix_is_ascii(role):
    assert role_prefix(role).text.isascii()


@pytest.mark.parametrize("role", list(Role))
def test_prefix_has_no_digit_heavy_timestamp(role):
    text = role_prefix(role).text
    # 4+ consecutive digits catches years, epoch-ish numbers, and dates
    # (which also contain a 4-digit year) without false-flagging a stray
    # single digit.
    assert not re.search(r"\d{4,}", text), text


@pytest.mark.parametrize("role", list(Role))
def test_prefix_has_no_domain_vocabulary(role):
    lowered = role_prefix(role).text.lower()
    for word in _DOMAIN_WORDS:
        assert word not in lowered, f"{role}: found domain word {word!r}"


@pytest.mark.parametrize("role", list(Role))
def test_prefix_is_nonempty_and_terse(role):
    prefix = role_prefix(role)
    assert prefix.text
    # Terse is part of the spec, not just a nice-to-have -- every extra
    # sentence here is paid for on every call, cache hit or not, by the
    # weakest model in the fleet. A few hundred tokens is a contract, not
    # a soft target.
    assert prefix.token_estimate < 500


# ------------------------------------------------------------- cache-hit test


def test_build_prompt_puts_prefix_first():
    prefix = role_prefix(Role.IMPLEMENTER)
    prompt = build_prompt(prefix, "\n\nTASK: fix the thing.")
    assert prompt.full_text.startswith(prefix.text)
    assert prompt.full_text == prefix.text + "\n\nTASK: fix the thing."


def test_build_prompt_prefix_bytes_are_identical_when_the_tail_changes():
    """This is the actual cache-hit guarantee: the byte range a provider would
    serve from cache must not move or change when only the tail varies.
    """
    prefix = role_prefix(Role.PLANNER)

    prompt_a = build_prompt(prefix, "\n\nTASK: decompose objective A.")
    prompt_b = build_prompt(prefix, "\n\nTASK: decompose an entirely different objective B, much longer.")

    assert prompt_a.cacheable_prefix_len == prompt_b.cacheable_prefix_len == len(prefix.text)
    assert prompt_a.prefix_fingerprint == prompt_b.prefix_fingerprint == prefix.fingerprint

    prefix_bytes_a = prompt_a.full_text[: prompt_a.cacheable_prefix_len]
    prefix_bytes_b = prompt_b.full_text[: prompt_b.cacheable_prefix_len]
    assert prefix_bytes_a == prefix_bytes_b == prefix.text


def test_build_prompt_fingerprint_changes_if_prefix_text_changes():
    a = StablePrefix(role=Role.SUMMARIZER, version=1, text="Summarize the tail only.")
    b = StablePrefix(role=Role.SUMMARIZER, version=1, text="Summarize the tail only!")
    prompt_a = build_prompt(a, "tail")
    prompt_b = build_prompt(b, "tail")
    assert prompt_a.prefix_fingerprint != prompt_b.prefix_fingerprint


# ------------------------------------------------------------- prefix.py unit


def test_stable_prefix_rejects_blank_text():
    with pytest.raises(ValueError):
        StablePrefix(role=Role.EXPLORER, version=1, text="   ")


def test_stable_prefix_rejects_non_ascii_text():
    with pytest.raises(ValueError):
        StablePrefix(role=Role.EXPLORER, version=1, text="café")


def test_stable_prefix_rejects_nonpositive_version():
    with pytest.raises(ValueError):
        StablePrefix(role=Role.EXPLORER, version=0, text="ok")


def test_stable_prefix_key_is_role_and_version():
    p = StablePrefix(role=Role.REVIEWER, version=3, text="ok")
    assert p.key == (Role.REVIEWER, 3)


def test_prefix_registry_get_returns_none_when_unregistered():
    assert PrefixRegistry().get(Role.MANAGER, 1) is None

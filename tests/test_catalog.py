"""Tests for the model price catalog: dual-source parsing (models.dev's
per-provider nesting and BerriAI/litellm's flat model_prices_and_context_window.json)
plus the provenance/staleness stamp layered on top so a caller never prices a
call off a number nobody has checked in months.
"""

from __future__ import annotations

import json
import time

import pytest

import forgeos.catalog as catalog_module
from forgeos.catalog import (
    DEFAULT_CATALOG_PATHS,
    LITELLM_URL,
    MODELS_DEV_URL,
    Catalog,
    ModelCard,
    _looks_like_models_dev,
    _parse_catalog_json,
    _parse_litellm,
    _parse_models_dev,
)

MODELS_DEV_FIXTURE = {
    "openai": {
        "models": {
            "gpt-4": {
                "name": "GPT-4",
                "cost": {"input": 30.0, "output": 60.0, "cache_read": 15.0},
                "limit": {"context": 8192, "output": 4096},
                "reasoning": False,
                "tool_call": True,
            }
        }
    }
}

# A slice of litellm's real shape: flat {model_name: {...}}, one
# litellm_provider per entry, prices per TOKEN not per 1M, plus the
# "sample_spec" pseudo-entry litellm ships to document the schema (no
# litellm_provider field, so it is not a real model).
LITELLM_FIXTURE = {
    "gpt-4": {
        "max_tokens": 8192,
        "max_input_tokens": 8192,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00003,
        "output_cost_per_token": 0.00006,
        "litellm_provider": "openai",
        "mode": "chat",
        "supports_function_calling": True,
    },
    "claude-3-7-sonnet-20250219": {
        "max_input_tokens": 200000,
        "max_output_tokens": 64000,
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000015,
        "cache_read_input_token_cost": 0.0000003,
        "cache_creation_input_token_cost": 0.00000375,
        "litellm_provider": "anthropic",
        "mode": "chat",
        "supports_vision": True,
    },
    "sample_spec": {
        "max_tokens": "max_output_tokens_int",
    },
}


# ------------------------------------------------------------ shape detection


def test_looks_like_models_dev_true_for_models_dev_shape():
    assert _looks_like_models_dev(MODELS_DEV_FIXTURE) is True


def test_looks_like_models_dev_false_for_litellm_shape():
    assert _looks_like_models_dev(LITELLM_FIXTURE) is False


def test_parse_catalog_json_dispatches_by_shape():
    md_cards = _parse_catalog_json(MODELS_DEV_FIXTURE)
    assert md_cards and all(c.source == "models_dev" for c in md_cards)

    ll_cards = _parse_catalog_json(LITELLM_FIXTURE)
    assert ll_cards and all(c.source == "litellm" for c in ll_cards)


# --------------------------------------------------------------- models.dev


def test_parse_models_dev_reads_cost_and_limits():
    cards = _parse_models_dev(MODELS_DEV_FIXTURE)
    assert len(cards) == 1
    c = cards[0]
    assert c.ref == "openai/gpt-4"
    assert c.input_cost_per_1m == 30.0
    assert c.output_cost_per_1m == 60.0
    assert c.cache_read_cost_per_1m == 15.0
    assert c.context == 8192
    assert c.tool_call is True
    assert c.source == "models_dev"
    assert c.source_url == MODELS_DEV_URL


# ------------------------------------------------------------------ litellm


def test_parse_litellm_converts_per_token_to_per_1m():
    cards = _parse_litellm(LITELLM_FIXTURE)
    by_ref = {c.ref: c for c in cards}
    gpt4 = by_ref["openai/gpt-4"]
    assert gpt4.input_cost_per_1m == pytest.approx(30.0)
    assert gpt4.output_cost_per_1m == pytest.approx(60.0)
    assert gpt4.tool_call is True
    assert gpt4.context == 8192
    assert gpt4.max_output == 4096


def test_parse_litellm_reads_cache_prices_and_scales_them_too():
    """The whole point of pulling in a real cache-price field: pricing cached
    input at the fresh rate would make the biggest saving in the harness
    read as zero (see ModelCard.cache_discount)."""
    cards = _parse_litellm(LITELLM_FIXTURE)
    by_ref = {c.ref: c for c in cards}
    claude = by_ref["anthropic/claude-3-7-sonnet-20250219"]
    assert claude.cache_read_cost_per_1m == pytest.approx(0.3)
    assert claude.cache_write_cost_per_1m == pytest.approx(3.75)
    assert claude.attachment is True


def test_parse_litellm_skips_entries_without_litellm_provider():
    """"sample_spec" documents litellm's own schema; it is not a real model."""
    cards = _parse_litellm(LITELLM_FIXTURE)
    assert "sample_spec" not in {c.model_id for c in cards}
    assert len(cards) == 2


def test_parse_litellm_stamps_source_and_fetched_at():
    cards = _parse_litellm(LITELLM_FIXTURE, fetched_at=12345.0)
    assert all(
        c.source == "litellm" and c.source_url == LITELLM_URL and c.fetched_at == 12345.0
        for c in cards
    )


# ---------------------------------------------------------------- provenance


def test_provenance_word_measured_for_a_fetched_and_stamped_card():
    card = ModelCard(model_id="m", provider="p", source="litellm", fetched_at=time.time())
    assert card.provenance_word == "measured"


def test_provenance_word_modelled_for_a_named_source_without_a_timestamp():
    """A `source` set without `fetched_at` (or a source outside the known
    fetch formats, e.g. a future hand-entered fallback) is not a measurement
    -- it mirrors forgeos.economy.savings.Provenance.MODELLED."""
    card = ModelCard(model_id="m", provider="p", source="litellm")
    assert card.provenance_word == "modelled"

    manual = ModelCard(model_id="m", provider="p", source="manual", fetched_at=time.time())
    assert manual.provenance_word == "modelled"


def test_provenance_word_unknown_for_a_bare_card():
    card = ModelCard(model_id="m", provider="p")
    assert card.provenance_word == "unknown"


def test_parsed_cards_are_measured():
    """The whole point of the parsers stamping source + source_url + fetched_at
    together: a card that actually came from a live fetch must read as
    measured, not merely as "has some fields filled in"."""
    for c in _parse_models_dev(MODELS_DEV_FIXTURE, fetched_at=time.time()):
        assert c.provenance_word == "measured"
    for c in _parse_litellm(LITELLM_FIXTURE, fetched_at=time.time()):
        assert c.provenance_word == "measured"


def test_fresh_card_is_not_stale():
    card = ModelCard(model_id="m", provider="p", fetched_at=time.time())
    assert card.age_days == pytest.approx(0.0, abs=0.01)
    assert card.is_stale() is False


def test_old_card_is_stale():
    sixty_days_ago = time.time() - 60 * 86400
    card = ModelCard(model_id="m", provider="p", fetched_at=sixty_days_ago)
    assert card.age_days == pytest.approx(60.0, abs=0.1)
    assert card.is_stale(max_age_days=30.0) is True
    assert card.is_stale(max_age_days=90.0) is False


def test_unstamped_card_has_no_age_and_counts_as_stale():
    """A price this module cannot date must never read as fresh just because
    no one checked -- see ModelCard.is_stale's docstring."""
    card = ModelCard(model_id="m", provider="p")
    assert card.fetched_at == 0.0
    assert card.source == ""
    assert card.age_days is None
    assert card.is_stale() is True


def test_catalog_stale_filters_by_age():
    fresh = ModelCard(model_id="a", provider="p", fetched_at=time.time())
    old = ModelCard(model_id="b", provider="p", fetched_at=time.time() - 60 * 86400)
    cat = Catalog([fresh, old])
    stale_refs = {c.model_id for c in cat.stale(max_age_days=30.0)}
    assert stale_refs == {"b"}


# ---------------------------------------------------------- from_file/discover


def test_from_file_stamps_fetched_at_from_mtime(tmp_path):
    path = tmp_path / "litellm.json"
    path.write_text(json.dumps(LITELLM_FIXTURE), encoding="utf-8")
    cat = Catalog.from_file(path)
    mtime = path.stat().st_mtime
    assert len(cat) == 2
    for c in cat.all():
        assert c.fetched_at == pytest.approx(mtime, abs=1.0)
        assert c.source == "litellm"
        assert c.is_stale() is False


def test_from_file_detects_models_dev_shape(tmp_path):
    path = tmp_path / "models_dev.json"
    path.write_text(json.dumps(MODELS_DEV_FIXTURE), encoding="utf-8")
    cat = Catalog.from_file(path)
    card = cat.get("openai/gpt-4")
    assert card is not None
    assert card.source == "models_dev"


def test_discover_merges_extra_paths_and_later_path_wins_on_ref_collision(tmp_path, monkeypatch):
    """Two cache files both define openai/gpt-4 at a different price. The
    later path in merge order wins -- the same mechanism that makes the
    litellm cache path (last in DEFAULT_CATALOG_PATHS) take precedence over
    an older models.dev cache in real use.
    """
    monkeypatch.setattr(catalog_module, "DEFAULT_CATALOG_PATHS", ())

    stale_fixture = json.loads(json.dumps(MODELS_DEV_FIXTURE))
    stale_fixture["openai"]["models"]["gpt-4"]["cost"]["input"] = 999.0
    older = tmp_path / "older.json"
    newer = tmp_path / "newer.json"
    older.write_text(json.dumps(stale_fixture), encoding="utf-8")
    newer.write_text(json.dumps(MODELS_DEV_FIXTURE), encoding="utf-8")

    cat = Catalog.discover(extra_paths=(older, newer))
    assert cat.get("openai/gpt-4").input_cost_per_1m == 30.0


def test_discover_keeps_an_unknown_models_prior_when_a_fresh_fetch_does_not_mention_it(
    tmp_path, monkeypatch
):
    """A model absent from the freshly-fetched file must not vanish, and its
    own provenance (source, source_url, fetched_at) must be left exactly as
    it was -- merging is additive-with-override, never destructive.
    """
    monkeypatch.setattr(catalog_module, "DEFAULT_CATALOG_PATHS", ())

    hand_entered = {
        "custom": {
            "models": {
                "house-model": {
                    "cost": {"input": 1.5, "output": 3.0},
                }
            }
        }
    }
    prior_path = tmp_path / "prior.json"
    fresh_path = tmp_path / "fresh.json"
    prior_path.write_text(json.dumps(hand_entered), encoding="utf-8")
    fresh_path.write_text(json.dumps(MODELS_DEV_FIXTURE), encoding="utf-8")

    cat = Catalog.discover(extra_paths=(prior_path, fresh_path))

    prior_card = cat.get("custom/house-model")
    assert prior_card is not None
    assert prior_card.input_cost_per_1m == 1.5
    assert prior_card.fetched_at == pytest.approx(prior_path.stat().st_mtime, abs=1.0)

    fresh_card = cat.get("openai/gpt-4")
    assert fresh_card is not None
    assert fresh_card.input_cost_per_1m == 30.0


def test_discover_tolerates_a_corrupt_file(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog_module, "DEFAULT_CATALOG_PATHS", ())
    bad = tmp_path / "corrupt.json"
    bad.write_text("{not json", encoding="utf-8")
    cat = Catalog.discover(extra_paths=(bad,))
    assert len(cat) == 0  # must not raise


def test_default_catalog_paths_checks_litellm_last():
    """Precedence claim, checked mechanically instead of just in a comment:
    discover() overwrites earlier refs with later ones, so litellm's cache
    path must be the last entry to actually win on a shared ref."""
    assert DEFAULT_CATALOG_PATHS[-1].name == "litellm_prices.json"

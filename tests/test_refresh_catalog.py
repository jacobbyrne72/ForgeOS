"""Tests for tools/refresh_catalog.py: source selection (litellm vs
models.dev), the atomic install, and the price-change report -- all without
a real network call, per the house rule.

`tools/refresh_catalog.py` is not a package member (no tools/__init__.py; it
is meant to be run directly, per its own docstring). On this machine a
different, unrelated `tools` package is installed in site-packages and
shadows the repo's `tools/` directory for `import tools.refresh_catalog`
(a regular package always wins over an implicit namespace package,
regardless of sys.path order), so this loads the file by path instead of by
dotted import -- see `_load_refresh_catalog`.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from forgeos.catalog import DEFAULT_CATALOG_PATHS, LITELLM_URL, MODELS_DEV_URL, Catalog

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_refresh_catalog():
    path = REPO_ROOT / "tools" / "refresh_catalog.py"
    spec = importlib.util.spec_from_file_location("_test_refresh_catalog", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rc():
    return _load_refresh_catalog()


LITELLM_PAYLOAD = {
    "gpt-4": {
        "max_input_tokens": 8192,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.00003,
        "output_cost_per_token": 0.00006,
        "litellm_provider": "openai",
        "mode": "chat",
    },
}


# ------------------------------------------------------------ source wiring


def test_default_source_is_litellm(rc):
    assert rc.DEFAULT_SOURCE == "litellm"
    assert set(rc.SOURCES) == {"litellm", "models_dev"}


def test_litellm_target_is_checked_last_by_catalog(rc):
    """Mechanical proof that litellm really is wired in "as the price
    source": its install path must be the one forgeos/catalog.py checks
    last, so it wins over models.dev on any ref both know."""
    assert rc.SOURCES["litellm"]["target"] == DEFAULT_CATALOG_PATHS[-1]


def test_models_dev_target_is_still_a_known_catalog_path(rc):
    assert rc.SOURCES["models_dev"]["target"] in DEFAULT_CATALOG_PATHS


def test_sources_have_distinct_urls_and_targets(rc):
    litellm, models_dev = rc.SOURCES["litellm"], rc.SOURCES["models_dev"]
    assert litellm["url"] != models_dev["url"]
    assert litellm["target"] != models_dev["target"]


def test_source_urls_come_from_catalog_not_a_local_copy(rc):
    """The URL this tool fetches from must be the exact same constant
    forgeos/catalog.py stamps onto every parsed ModelCard.source_url --
    otherwise "what we fetched" and "what a card claims it was fetched
    from" could silently drift apart.
    """
    assert rc.SOURCES["litellm"]["url"] == LITELLM_URL
    assert rc.SOURCES["models_dev"]["url"] == MODELS_DEV_URL


# -------------------------------------------------------------------- install


def test_install_is_atomic_and_readable_by_catalog(tmp_path, rc):
    target = tmp_path / "install_test.json"
    rc.install(LITELLM_PAYLOAD, target)

    assert target.exists()
    cat = Catalog.from_file(target)
    card = cat.get("openai/gpt-4")
    assert card is not None
    assert card.input_cost_per_1m == pytest.approx(30.0)
    assert card.source == "litellm"


def test_install_no_leftover_temp_file(tmp_path, rc):
    target = tmp_path / "install_test.json"
    rc.install(LITELLM_PAYLOAD, target)
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == []


# ---------------------------------------------------------------------- main


def test_main_dry_run_does_not_write(tmp_path, monkeypatch, rc):
    fake_target = tmp_path / "would_install.json"
    monkeypatch.setitem(rc.SOURCES["litellm"], "target", fake_target)
    monkeypatch.setattr(rc, "fetch", lambda url, timeout: LITELLM_PAYLOAD)
    monkeypatch.setattr(sys, "argv", ["refresh_catalog.py", "--dry-run"])

    assert rc.main() == 0
    assert not fake_target.exists()


def test_main_installs_when_not_a_dry_run(tmp_path, monkeypatch, rc):
    fake_target = tmp_path / "installed.json"
    monkeypatch.setitem(rc.SOURCES["litellm"], "target", fake_target)
    monkeypatch.setattr(rc, "fetch", lambda url, timeout: LITELLM_PAYLOAD)
    monkeypatch.setattr(sys, "argv", ["refresh_catalog.py"])

    assert rc.main() == 0
    assert fake_target.exists()
    card = Catalog.from_file(fake_target).get("openai/gpt-4")
    assert card is not None
    assert card.output_cost_per_1m == pytest.approx(60.0)


def test_main_source_flag_selects_the_models_dev_url(tmp_path, monkeypatch, rc):
    fake_target = tmp_path / "models_dev_installed.json"
    monkeypatch.setitem(rc.SOURCES["models_dev"], "target", fake_target)

    seen_urls = []

    def fake_fetch(url, timeout):
        seen_urls.append(url)
        return {"openai": {"models": {"gpt-4": {"cost": {"input": 1.0, "output": 2.0}}}}}

    monkeypatch.setattr(rc, "fetch", fake_fetch)
    monkeypatch.setattr(sys, "argv", ["refresh_catalog.py", "--source", "models_dev", "--dry-run"])

    assert rc.main() == 0
    assert seen_urls == [rc.SOURCES["models_dev"]["url"]]
    assert not fake_target.exists()  # --dry-run: confirms the source-target wiring without installing


def test_main_url_override_takes_precedence_over_source_default(tmp_path, monkeypatch, rc):
    seen_urls = []

    def fake_fetch(url, timeout):
        seen_urls.append(url)
        return LITELLM_PAYLOAD

    monkeypatch.setattr(rc, "fetch", fake_fetch)
    monkeypatch.setattr(
        sys, "argv", ["refresh_catalog.py", "--url", "https://example.invalid/prices.json", "--dry-run"]
    )

    assert rc.main() == 0
    assert seen_urls == ["https://example.invalid/prices.json"]


def test_main_never_writes_on_a_failed_fetch(tmp_path, monkeypatch, rc):
    fake_target = tmp_path / "never_written.json"
    monkeypatch.setitem(rc.SOURCES["litellm"], "target", fake_target)

    def failing_fetch(url, timeout):
        raise RuntimeError("simulated network failure")

    monkeypatch.setattr(rc, "fetch", failing_fetch)
    monkeypatch.setattr(sys, "argv", ["refresh_catalog.py"])

    assert rc.main() == 1
    assert not fake_target.exists()


def test_main_never_writes_when_payload_parses_to_zero_models(tmp_path, monkeypatch, rc):
    fake_target = tmp_path / "never_written_empty.json"
    monkeypatch.setitem(rc.SOURCES["litellm"], "target", fake_target)
    monkeypatch.setattr(rc, "fetch", lambda url, timeout: {"sample_spec": {"max_tokens": "x"}})
    monkeypatch.setattr(sys, "argv", ["refresh_catalog.py"])

    assert rc.main() == 1
    assert not fake_target.exists()


# -------------------------------------------------------------------- report


def test_report_flags_cheaper_dearer_added_and_removed(capsys, rc):
    before = {
        "openai/gpt-4": (30.0, 60.0, 0.0),
        "openai/gpt-3.5": (1.0, 2.0, 0.0),
        "openai/o1-legacy": (5.0, 10.0, 0.0),
    }
    after = {
        "openai/gpt-4": (15.0, 30.0, 0.0),   # cheaper
        "openai/gpt-3.5": (2.0, 4.0, 0.0),   # dearer
        "openai/gpt-5": (3.0, 6.0, 0.0),     # new
        # o1-legacy is gone
    }
    rc.report(before, after, "", 15)
    out = capsys.readouterr().out
    assert "CHEAPER (1)" in out
    assert "MORE EXPENSIVE (1)" in out
    assert "NEW (1)" in out
    assert "gpt-5" in out
    assert "GONE (1)" in out
    assert "o1-legacy" in out


def test_report_provider_filter_scopes_the_diff(capsys, rc):
    before = {"openai/gpt-4": (30.0, 60.0, 0.0)}
    after = {"openai/gpt-4": (15.0, 30.0, 0.0), "anthropic/claude": (3.0, 15.0, 0.0)}
    rc.report(before, after, "anthropic", 15)
    out = capsys.readouterr().out
    # openai's price drop is real but out of scope for an anthropic-filtered
    # report; anthropic/claude is new and in scope.
    assert "CHEAPER" not in out
    assert "NEW (1)" in out


def test_report_no_changes_says_so(capsys, rc):
    same = {"openai/gpt-4": (30.0, 60.0, 0.0)}
    rc.report(same, dict(same), "", 15)
    assert "no price changes" in capsys.readouterr().out

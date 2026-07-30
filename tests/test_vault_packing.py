"""Vault + packing tests.

Portability is the theme for the vault half: forgeos must work identically on a machine
that has never installed Obsidian, codex-brain, or Hermes. A tool that only runs on
its author's laptop is not open source in any useful sense.
"""

from __future__ import annotations

import json
import os

import pytest

from forgeos.economy.packing import (
    BOUNDARY_RE,
    MAX_SOURCES,
    MAX_SOURCES_PER_DIR,
    Overflow,
    PackReport,
    SectionBudget,
    allocate_sections,
    diversify,
    expand_to_boundary,
    find_tensions,
    next_reads,
    spend,
)
from forgeos.knowledge.vault import (
    SUBDIRS,
    VAULT_ENV,
    Vault,
    obsidian_vaults,
    resolve_vault_path,
)
from forgeos.policy import pack_command, ripper_command


# ============================================================== portability


def test_ripper_command_degrades_to_builtin_advice_without_the_external_tool(monkeypatch):
    """Suggesting a command that does not exist reads as a broken tool, not a rule."""
    monkeypatch.setenv("FORGEOS_DIGEST_TOOL", "")
    monkeypatch.setattr(os.path, "expanduser", lambda p: "/nonexistent" + p.lstrip("~"))
    cmd = ripper_command("big.py")
    assert "digest.py" not in cmd
    assert "offset" in cmd or "rg " in cmd


def test_pack_command_degrades_to_ripgrep(monkeypatch):
    monkeypatch.setenv("FORGEOS_PACK_TOOL", "")
    monkeypatch.setattr(os.path, "expanduser", lambda p: "/nonexistent" + p.lstrip("~"))
    assert "rg " in pack_command("find the retry parser")


def test_configured_digest_tool_is_used_when_present(monkeypatch, tmp_path):
    tool = tmp_path / "digest.py"
    tool.write_text("# stub", encoding="utf-8")
    monkeypatch.setenv("FORGEOS_DIGEST_TOOL", str(tool))
    assert str(tool) in ripper_command("big.py")


def test_no_authors_home_directory_is_hardcoded_anywhere():
    """A path from the author's machine in shipped code is an open-source smell and
    a bug for every other user."""
    import pathlib

    offenders = []
    for p in pathlib.Path("forgeos").rglob("*.py"):
        text = p.read_text(encoding="utf-8", errors="replace").lower()
        if "users\\\\byrne" in text or "users/byrne" in text:
            offenders.append(str(p))
    assert offenders == [], f"hardcoded author paths in: {offenders}"


# ==================================================================== vault


def test_vault_is_created_where_configured(tmp_path):
    v = Vault(tmp_path / "myvault")
    assert v.created
    for sub in SUBDIRS:
        assert (v.path / sub).is_dir()
    assert (v.path / "index.md").exists()
    assert (v.path / "log.md").exists()
    assert (v.path / "SCHEMA.md").exists()


def test_env_var_selects_the_vault(monkeypatch, tmp_path):
    monkeypatch.setenv(VAULT_ENV, str(tmp_path / "envvault"))
    path, source, _ = resolve_vault_path()
    assert path == tmp_path / "envvault"
    assert VAULT_ENV in source


def test_explicit_argument_beats_the_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv(VAULT_ENV, str(tmp_path / "envvault"))
    path, source, _ = resolve_vault_path(str(tmp_path / "explicit"))
    assert path == tmp_path / "explicit"
    assert source == "configured"


def test_falls_back_to_a_plain_folder_without_obsidian(monkeypatch):
    """forgeos must not require a GUI app to store markdown."""
    monkeypatch.delenv(VAULT_ENV, raising=False)
    monkeypatch.setattr("forgeos.knowledge.vault.obsidian_vaults", lambda: [])
    path, source, managed = resolve_vault_path()
    assert source == "default"
    assert managed is False
    assert ".forgeos" in str(path)


def test_existing_obsidian_vault_is_used_as_a_subfolder(monkeypatch, tmp_path):
    """Notes land where the human already looks — but never scattered through
    a vault root they have organised themselves."""
    monkeypatch.delenv(VAULT_ENV, raising=False)
    monkeypatch.setattr("forgeos.knowledge.vault.obsidian_vaults", lambda: [tmp_path / "MyNotes"])
    path, source, managed = resolve_vault_path()
    assert path == tmp_path / "MyNotes" / "forgeos"
    assert source == "obsidian"
    assert managed is True


def test_malformed_obsidian_config_is_not_fatal(monkeypatch, tmp_path):
    cfg = tmp_path / "obsidian.json"
    cfg.write_text("{not valid json", encoding="utf-8")
    monkeypatch.setattr(
        "forgeos.knowledge.vault._OBSIDIAN_CONFIG",
        {__import__("platform").system(): str(cfg)},
    )
    assert obsidian_vaults() == []


def test_obsidian_config_listing_skips_vaults_that_no_longer_exist(monkeypatch, tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    cfg = tmp_path / "obsidian.json"
    cfg.write_text(json.dumps({"vaults": {
        "a": {"path": str(real), "ts": 2},
        "b": {"path": str(tmp_path / "deleted"), "ts": 9},
    }}), encoding="utf-8")
    monkeypatch.setattr(
        "forgeos.knowledge.vault._OBSIDIAN_CONFIG",
        {__import__("platform").system(): str(cfg)},
    )
    assert obsidian_vaults() == [real]


def test_page_is_written_and_indexed(tmp_path):
    v = Vault(tmp_path / "v")
    v.write_page("wiki", "Retry parsing", "How Retry-After is normalised.", "Body text.")
    entries = v.index_entries()
    assert len(entries) == 1
    assert "Retry parsing" in entries[0]
    assert "How Retry-After is normalised." in entries[0]


def test_page_without_a_summary_is_refused(tmp_path):
    """An unsummarised page cannot be judged for relevance without reading it,
    which defeats the point of having an index."""
    v = Vault(tmp_path / "v")
    with pytest.raises(ValueError):
        v.write_page("wiki", "Title", "   ", "body")


def test_rewriting_a_page_updates_its_index_line_rather_than_duplicating(tmp_path):
    v = Vault(tmp_path / "v")
    v.write_page("wiki", "Retry parsing", "First summary.", "a")
    v.write_page("wiki", "Retry parsing", "Second summary.", "b")
    entries = v.index_entries()
    assert len(entries) == 1
    assert "Second summary." in entries[0]


def test_unknown_folder_is_refused(tmp_path):
    v = Vault(tmp_path / "v")
    with pytest.raises(ValueError):
        v.write_page("nowhere", "T", "s", "b")


def test_log_is_append_only(tmp_path):
    v = Vault(tmp_path / "v")
    v.append_log("ingest", "first")
    v.append_log("ingest", "second")
    text = (v.path / "log.md").read_text(encoding="utf-8")
    assert "first" in text and "second" in text
    assert text.index("first") < text.index("second")


def test_read_page_refuses_to_escape_the_vault(tmp_path):
    """A note reference must not become an arbitrary file read."""
    v = Vault(tmp_path / "v")
    with pytest.raises(ValueError):
        v.read_page("../../../etc/passwd")


# ================================================================== packing


def test_section_budget_splits_by_role_and_never_exceeds_the_total():
    b = allocate_sections(10_000)
    assert b.total <= 10_000
    assert b.evidence > b.cards > 0
    assert b.tensions > 0 and b.next_reads > 0


def test_tiny_budget_still_yields_a_valid_split():
    b = allocate_sections(10)
    assert b.total <= 10
    assert isinstance(b, SectionBudget)


def test_boundary_expansion_never_starts_mid_function():
    lines = [
        "import os", "", "def alpha():", "    return 1", "", "def beta():",
        "    x = 2", "    y = 3", "    return x + y", "", "def gamma():", "    pass",
    ]
    start, end = expand_to_boundary(lines, 7, 8)  # mid-body of beta()
    assert BOUNDARY_RE.match(lines[start]), f"expanded start is mid-body: {lines[start]!r}"


def test_boundary_expansion_is_bounded_on_a_file_with_no_boundaries():
    """The guard against the fix becoming the bug: no boundaries must not mean
    swallowing the entire file."""
    lines = ["    filler line" for _ in range(500)]
    start, end = expand_to_boundary(lines, 250, 251, max_search=10)
    assert start >= 240
    assert end <= 262


def test_boundary_expansion_handles_an_empty_file():
    assert expand_to_boundary([], 0, 5) == (0, 0)


def test_diversify_caps_sources_per_directory():
    sources = [{"path": f"src/router/f{i}.py", "tokens": 10} for i in range(6)]
    kept, overflow = diversify(sources)
    assert len(kept) == MAX_SOURCES_PER_DIR
    assert len(overflow) == 3
    assert all("already has" in o.reason for o in overflow)


def test_diversify_spreads_across_directories():
    sources = [{"path": f"src/mod{i}/f.py", "tokens": 10} for i in range(6)]
    kept, overflow = diversify(sources)
    assert len(kept) == 6
    assert overflow == []


def test_diversify_respects_the_global_cap():
    sources = [{"path": f"src/mod{i}/f.py", "tokens": 10} for i in range(40)]
    kept, _ = diversify(sources)
    assert len(kept) == MAX_SOURCES


def test_tension_detection_surfaces_disagreement_between_sources():
    """The technique naive packing lacks: two sources that contradict each other."""
    sources = [
        {"path": "docs/a.md", "text": "The retry parser works correctly for HTTP-date."},
        {"path": "docs/b.md", "text": "The retry parser fails on HTTP-date inputs."},
    ]
    tensions = find_tensions(sources, ["retry parser"])
    assert len(tensions) == 1
    t = tensions[0]
    assert t.positive_source == "docs/a.md"
    assert t.negative_source == "docs/b.md"
    assert "CONFLICT" in t.render()


def test_a_source_contradicting_itself_is_not_a_tension():
    """'X failed, then X passed' is narrative, not disagreement between sources."""
    sources = [{"path": "log.md", "text": "retry failed at first\nretry works now"}]
    assert find_tensions(sources, ["retry"]) == []


def test_agreeing_sources_produce_no_tension():
    sources = [
        {"path": "a.md", "text": "parser works"},
        {"path": "b.md", "text": "parser passes all cases"},
    ]
    assert find_tensions(sources, ["parser"]) == []


def test_next_reads_names_what_was_excluded():
    """Silent truncation makes a partial view look complete."""
    overflow = [Overflow(path=f"src/f{i}.py", reason="over cap", est_tokens=100)
                for i in range(10)]
    lines = next_reads(overflow, limit=3)
    assert len(lines) == 4
    assert "and 7 more excluded" in lines[-1]


def test_next_reads_of_nothing_is_empty():
    assert next_reads([]) == []


def test_spend_preserves_rank_order_rather_than_packing_tightly():
    """Skipping ahead to smaller items would discard the ranking that made the
    pack relevant in the first place."""
    section: list = []
    items = [{"tokens": 80}, {"tokens": 500}, {"tokens": 5}]
    used = spend(section, items, 100)
    assert used == 80
    assert len(section) == 1  # stopped at the 500, did not skip to the 5


def test_pack_report_maths_and_zero_baseline():
    r = PackReport(kept_sources=5, excluded_sources=2, tensions_found=1,
                   baseline_tokens=40_000, actual_tokens=6_000,
                   sections=allocate_sections(6_000))
    assert r.saved_tokens == 34_000
    assert r.saved_pct == pytest.approx(85.0)
    assert "conflict" in r.render()

    empty = PackReport(kept_sources=0, excluded_sources=0, tensions_found=0,
                       baseline_tokens=0, actual_tokens=0,
                       sections=allocate_sections(0))
    assert empty.saved_pct == 0.0

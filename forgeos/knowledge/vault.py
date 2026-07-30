"""Knowledge vault — the three-layer wiki, stored as plain markdown.

Layout follows the LLM-wiki pattern: immutable raw sources, an LLM-owned wiki, and
a schema file describing the conventions, with `index.md` as the existence catalog
and `log.md` as an append-only history.

Two properties make it worth having:

- **`index.md` is the cheap existence check.** An agent reads one small catalog to
  learn what is known, then fetches only the page it needs. Without it the choice is
  between reading everything and not knowing what exists.
- **`log.md` is append-only.** History that can be rewritten is not history.

Plain markdown on purpose. If Obsidian is installed, forgeos uses an existing vault so
notes appear in a tool the human already has open; if not, it creates a plain folder
that is still perfectly readable in any editor. Nothing here requires Obsidian —
depending on a specific GUI to store text would be a poor trade.
"""

from __future__ import annotations

import json
import os
import platform
import re
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

VAULT_ENV = "FORGEOS_VAULT"

# Where Obsidian records the vaults a user already has. Reading it means forgeos can
# offer an existing vault instead of creating a competing one.
_OBSIDIAN_CONFIG = {
    "Windows": "~/AppData/Roaming/obsidian/obsidian.json",
    "Darwin": "~/Library/Application Support/obsidian/obsidian.json",
    "Linux": "~/.config/obsidian/obsidian.json",
}

SUBDIRS = ("raw", "wiki", "sessions", "decisions")

_SCHEMA_MD = """# Vault schema

Conventions for this vault. Written for humans and agents alike.

## Layers

| Folder | Owner | Rule |
|---|---|---|
| `raw/` | immutable | Source material exactly as captured. Never edited, only added to. |
| `wiki/` | agents | Distilled pages. May be rewritten as understanding improves. |
| `sessions/` | agents | One note per working session. |
| `decisions/` | humans + agents | Why a choice was made, and what would reverse it. |

## Files

- `index.md` — catalog of every page with a one-line summary. Read this FIRST; it is
  how an agent learns what exists without reading everything.
- `log.md` — append-only history. Never edit an existing entry.

## Rules

1. Every page opens with a one-line summary so a reader can judge relevance in one line.
2. Reference material is indexed and quoted verbatim, never summarised — for an API,
   the exact signature IS the payload, and a summary silently loses it.
3. Prose is distilled; structure is introspected. Do not chunk something you can query.
4. An outside claim enters as unverified and stays that way until evidence and a
   contradiction check promote it.
"""


class VaultInfo(BaseModel):
    path: Path
    created: bool = False
    obsidian_detected: bool = False
    obsidian_managed: bool = False
    source: str = ""

    model_config = {"arbitrary_types_allowed": True}


def obsidian_installed() -> bool:
    cfg = _OBSIDIAN_CONFIG.get(platform.system())
    return bool(cfg) and Path(os.path.expanduser(cfg)).exists()


def obsidian_vaults() -> list[Path]:
    """Vaults Obsidian already knows about, newest-opened first.

    Best-effort: a missing or malformed config yields an empty list rather than an
    error. Not finding a vault is normal, not a failure.
    """
    cfg = _OBSIDIAN_CONFIG.get(platform.system())
    if not cfg:
        return []
    p = Path(os.path.expanduser(cfg))
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []

    entries = data.get("vaults")
    if not isinstance(entries, dict):
        return []

    found: list[tuple[int, Path]] = []
    for meta in entries.values():
        if not isinstance(meta, dict):
            continue
        raw = meta.get("path")
        if not raw:
            continue
        vp = Path(str(raw))
        if vp.exists():
            found.append((int(meta.get("ts") or 0), vp))

    found.sort(key=lambda t: -t[0])
    return [p for _, p in found]


def resolve_vault_path(configured: str | None = None) -> tuple[Path, str, bool]:
    """Decide where the vault lives. Returns (path, source, obsidian_managed).

    Order of preference, most explicit first:
      1. an explicit argument      — the operator said so
      2. $FORGEOS_VAULT               — the operator said so, persistently
      3. an existing Obsidian vault — put notes where they will actually be read
      4. ~/.forgeos/vault             — a plain folder that needs no extra software
    """
    if configured:
        return Path(os.path.expanduser(configured)), "configured", False

    env = os.environ.get(VAULT_ENV, "").strip()
    if env:
        return Path(os.path.expanduser(env)), f"${VAULT_ENV}", False

    for vp in obsidian_vaults():
        # A subfolder, never the vault root — forgeos must not scatter files through
        # notes a human has organised.
        return vp / "forgeos", "obsidian", True

    return Path(os.path.expanduser("~/.forgeos/vault")), "default", False


class Vault:
    def __init__(self, path: str | Path | None = None, *, create: bool = True):
        resolved, source, managed = resolve_vault_path(str(path) if path else None)
        self.path = resolved
        self.source = source
        self.obsidian_managed = managed
        self.created = False
        if create:
            self.created = self._ensure()

    # ------------------------------------------------------------- layout

    def _ensure(self) -> bool:
        existed = self.path.exists()
        for sub in SUBDIRS:
            (self.path / sub).mkdir(parents=True, exist_ok=True)

        schema = self.path / "SCHEMA.md"
        if not schema.exists():
            schema.write_text(_SCHEMA_MD, encoding="utf-8")

        index = self.path / "index.md"
        if not index.exists():
            index.write_text(
                "# Index\n\nOne line per page. Read this before reading anything else.\n\n",
                encoding="utf-8",
            )

        log = self.path / "log.md"
        if not log.exists():
            log.write_text("# Log\n\nAppend-only. Never edit an existing entry.\n\n",
                           encoding="utf-8")
        return not existed

    def info(self) -> VaultInfo:
        return VaultInfo(
            path=self.path,
            created=self.created,
            obsidian_detected=obsidian_installed(),
            obsidian_managed=self.obsidian_managed,
            source=self.source,
        )

    # -------------------------------------------------------------- write

    @staticmethod
    def _slug(title: str) -> str:
        s = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        return s[:80] or "untitled"

    def write_page(self, folder: str, title: str, summary: str, body: str,
                   *, sources: list[str] | None = None) -> Path:
        """Write a page and register it in the index.

        The summary is mandatory and goes first — an unsummarised page cannot be
        judged for relevance without reading it, which defeats the index entirely.
        """
        if folder not in SUBDIRS:
            raise ValueError(f"unknown folder {folder!r}; expected one of {SUBDIRS}")
        if not summary.strip():
            raise ValueError("a page without a one-line summary cannot be indexed usefully")

        page = self.path / folder / f"{self._slug(title)}.md"
        parts = [f"# {title}", "", f"> {summary.strip()}", "", body.rstrip(), ""]
        if sources:
            parts += ["", "## Sources", *[f"- {s}" for s in sources], ""]
        page.write_text("\n".join(parts), encoding="utf-8")

        self._index_entry(folder, title, summary, page)
        return page

    def _index_entry(self, folder: str, title: str, summary: str, page: Path) -> None:
        index = self.path / "index.md"
        rel = page.relative_to(self.path).as_posix()
        line = f"- [{title}]({rel}) — {summary.strip()}\n"
        current = index.read_text(encoding="utf-8") if index.exists() else "# Index\n\n"
        if rel in current:  # replace the existing line rather than duplicating it
            current = "\n".join(
                line.rstrip("\n") if rel in ln else ln for ln in current.splitlines()
            ) + "\n"
        else:
            current = current.rstrip("\n") + "\n" + line
        index.write_text(current, encoding="utf-8")

    def append_log(self, kind: str, message: str, *, at: datetime | None = None) -> None:
        """Append one dated entry. Never rewrites an existing line."""
        ts = (at or datetime.now(timezone.utc)).strftime("%Y-%m-%d %H:%M")
        with (self.path / "log.md").open("a", encoding="utf-8") as fh:
            fh.write(f"\n## [{ts}] {kind}\n\n{message.strip()}\n")

    # --------------------------------------------------------------- read

    def index_entries(self) -> list[str]:
        index = self.path / "index.md"
        if not index.exists():
            return []
        return [ln for ln in index.read_text(encoding="utf-8").splitlines()
                if ln.startswith("- [")]

    def pages(self, folder: str | None = None) -> list[Path]:
        folders = [folder] if folder else list(SUBDIRS)
        out: list[Path] = []
        for f in folders:
            d = self.path / f
            if d.exists():
                out.extend(sorted(d.glob("*.md")))
        return out

    def read_page(self, path: str | Path) -> str:
        p = Path(path)
        if not p.is_absolute():
            p = self.path / p
        # Confine reads to the vault; a traversal here would turn a note reference
        # into an arbitrary file read.
        root = self.path.resolve()
        resolved = p.resolve()
        # Containment, not a string prefix: "~/.forgeos/vault_backup" starts with
        # "~/.forgeos/vault" and would have passed a startswith() check.
        if not resolved.is_relative_to(root):
            raise ValueError("refusing to read outside the vault")
        return resolved.read_text(encoding="utf-8")


__all__ = ["SUBDIRS", "VAULT_ENV", "Vault", "VaultInfo", "obsidian_installed",
           "obsidian_vaults", "resolve_vault_path"]

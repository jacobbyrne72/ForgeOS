"""Adapter auto-discovery — scan PATH and plugins dirs."""

from __future__ import annotations
import importlib.metadata
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .._sqlite import connect

ADAPTER_DIRS = [
    Path.home() / ".forgeos" / "plugins",
    Path.cwd() / ".forgeos-live" / "plugins",
    Path.cwd() / "adapter_plugins",
]
BUILTIN_ADAPTERS = {"omc-team": "omc", "ollama": "ollama", "gateway": "gateway", "acp": "acp"}
ADAPTER_CACHE_TTL = 300  # 5 minutes


@dataclass
class DiscoveredAdapter:
    name: str
    kind: str
    path: str
    source: str
    version: str = ""
    usable: bool = False


def _connect_cache(home: Path) -> Any:
    p = home / "adapter_discovery.db"
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = connect(str(p))
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS adapter_cache (
            name TEXT PRIMARY KEY, kind TEXT, path TEXT, source TEXT,
            version TEXT, usable INTEGER, discovered_at REAL
        );""")
    conn.commit()
    return conn


def discover_adapters(
    home: Path | None = None,
    force_refresh: bool = False,
) -> list[DiscoveredAdapter]:
    home = home or Path.home() / ".forgeos"
    cache = _connect_cache(home)
    now = time.time()
    if not force_refresh:
        cached = cache.execute(
            "SELECT name, kind, path, source, version, usable FROM adapter_cache WHERE discovered_at > ?",
            (now - ADAPTER_CACHE_TTL,),
        ).fetchall()
        if cached:
            return [
                DiscoveredAdapter(
                    name=r[0], kind=r[1], path=r[2], source=r[3], version=r[4], usable=bool(r[5])
                )
                for r in cached
            ]
    results: list[DiscoveredAdapter] = []
    seen: set[str] = set()
    for name, kind in BUILTIN_ADAPTERS.items():
        results.append(DiscoveredAdapter(name=name, kind=kind, path="builtin", source="builtin"))
        seen.add(name)
    _scan_path(results, seen)
    _scan_plugins(results, seen, home)
    _scan_entry_points(results, seen)
    for a in results:
        a.usable = _validate_adapter(a)
    cache.execute("DELETE FROM adapter_cache")
    for a in results:
        cache.execute(
            "INSERT INTO adapter_cache VALUES (?,?,?,?,?,?,?)",
            (a.name, a.kind, a.path, a.source, a.version, int(a.usable), now),
        )
    cache.commit()
    return results


def _scan_path(results: list, seen: set) -> None:
    for d in os.environ.get("PATH", "").split(os.pathsep):
        d = d.strip()
        if not d or not Path(d).is_dir():
            continue
        for entry in sorted(Path(d).iterdir()):
            if entry.name.startswith("forgeos_") or entry.name.endswith("_adapter"):
                if entry.name not in seen:
                    seen.add(entry.name)
                    results.append(
                        DiscoveredAdapter(name=entry.name, kind="custom", path=str(entry), source="path")
                    )


def _scan_plugins(results: list, seen: set, home: Path) -> None:
    for plugins_dir in ADAPTER_DIRS:
        if not plugins_dir.is_dir():
            continue
        for entry in sorted(plugins_dir.iterdir()):
            if entry.name.startswith(".") or entry.name in seen:
                continue
            if entry.is_dir():
                seen.add(entry.name)
                results.append(
                    DiscoveredAdapter(name=entry.name, kind="plugin", path=str(entry), source="plugin")
                )
            elif entry.suffix == ".py":
                seen.add(entry.name)
                results.append(
                    DiscoveredAdapter(name=entry.stem, kind="adapter", path=str(entry), source="plugin")
                )


def _scan_entry_points(results: list, seen: set) -> None:
    try:
        eps = importlib.metadata.entry_points(group="forgeos.adapters")
        for ep in eps:
            if ep.name not in seen:
                seen.add(ep.name)
                results.append(
                    DiscoveredAdapter(name=ep.name, kind="entry_point", path=ep.value, source="entry_point")
                )
    except Exception:
        pass


def _validate_adapter(a: DiscoveredAdapter) -> bool:
    try:
        if a.source == "builtin" or a.path == "builtin":
            return True
        if a.source == "path":
            return Path(a.path).is_file()
        if a.source == "plugin" and Path(a.path).is_dir():
            return (Path(a.path) / "__init__.py").exists()
        return Path(a.path).is_file() or Path(a.path).is_dir()
    except Exception:
        return False

"""Refresh the model price catalogue, and show what moved.

Every routing decision ForgeOS makes is priced against this catalogue: which
worker is "cheapest capable", what the preflight estimate says, whether the
governor refuses a call. A stale catalogue does not merely report old numbers —
it routes work to the wrong model and mis-estimates every call, silently.

Two upstream price tables are supported, chosen with --source (default litellm):

    litellm      BerriAI/litellm's model_prices_and_context_window.json --
                 auto-synced by their CI, the de facto community-canonical
                 pricing table. This is the primary source now: its install
                 path is checked LAST by forgeos/catalog.py, so it wins over
                 models.dev on any model ref both know.
    models_dev   models.dev/api.json, the original bundled source. Still
                 supported so an existing cache keeps working.

Every installed file's own mtime becomes its staleness stamp: `forgeos/catalog.py`
reads it back into `ModelCard.fetched_at`, so `card.is_stale()` /
`Catalog.stale()` can catch a routing decision about to price off a number
nobody has checked in months.

    python tools/refresh_catalog.py --dry-run              # litellm, show the diff, write nothing
    python tools/refresh_catalog.py                        # litellm, fetch and install
    python tools/refresh_catalog.py --source models_dev     # refresh the older source instead
    python tools/refresh_catalog.py --provider openai

Prints a price-change report, because a refresh that silently swaps numbers under
a routing system is exactly the sort of unexplained behaviour change this project
exists to avoid. Cheaper models are the point; you should be able to see them.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from forgeos.catalog import (  # noqa: E402
    DEFAULT_CATALOG_PATHS,
    LITELLM_URL,
    MODELS_DEV_URL,
    Catalog,
    default_catalog,
)

# Each upstream price table's fetch URL and install path. The URL comes from
# forgeos/catalog.py, not a local copy -- that module stamps the same URL
# onto every ModelCard's `source_url`, so "what this tool fetches" and "what
# a parsed card says it was fetched from" can never drift apart. The install
# path is this tool's own concern: forgeos/catalog.py checks the litellm
# path LAST among DEFAULT_CATALOG_PATHS, so it is the one that wins on a ref
# both tables know -- see the module docstring.
SOURCES: dict[str, dict[str, object]] = {
    "litellm": {
        "url": LITELLM_URL,
        "target": Path(os.path.expanduser("~/.forgeos/litellm_prices.json")),
    },
    "models_dev": {
        "url": MODELS_DEV_URL,
        "target": Path(os.path.expanduser("~/.forgeos/models_dev_cache.json")),
    },
}
DEFAULT_SOURCE = "litellm"


def current_prices() -> dict[str, tuple[float, float, float]]:
    """ref -> (input, output, cache_read) per 1M, from whatever is installed now."""
    try:
        cat = default_catalog()
    except Exception:
        return {}
    return {
        m.ref: (m.input_cost_per_1m, m.output_cost_per_1m, m.cache_read_cost_per_1m)
        for m in cat.all()
    }


def fetch(url: str, timeout: float) -> dict:
    r = httpx.get(url, timeout=timeout, follow_redirects=True)
    r.raise_for_status()
    return r.json()


def install(payload: dict, target: Path) -> None:
    """Atomic write. A half-written catalogue is worse than a stale one — every
    price lookup would silently fall back to 'unknown' mid-file."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        shutil.move(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def report(before: dict, after: dict, provider_filter: str, limit: int) -> None:
    cheaper, dearer, added, removed = [], [], [], []

    for ref, (i2, o2, c2) in after.items():
        if provider_filter and not ref.startswith(provider_filter + "/"):
            continue
        if ref not in before:
            added.append(ref)
            continue
        i1, o1, _c1 = before[ref]
        # Compare on input+output; a model is "cheaper" only if the pair falls.
        # Judging on input alone would call a model cheaper when its output
        # price — the expensive side of most bills — had gone up.
        old, new = i1 + o1, i2 + o2
        if old <= 0:
            continue
        delta = (new - old) / old
        if delta < -0.001:
            cheaper.append((delta, ref, i1, o1, i2, o2))
        elif delta > 0.001:
            dearer.append((delta, ref, i1, o1, i2, o2))

    for ref in before:
        if provider_filter and not ref.startswith(provider_filter + "/"):
            continue
        if ref not in after:
            removed.append(ref)

    cheaper.sort()
    dearer.sort(reverse=True)

    def show(rows, title):
        if not rows:
            return
        print(f"\n{title} ({len(rows)})")
        print(f"  {'model':44} {'in/1M':>16} {'out/1M':>16} {'change':>8}")
        for delta, ref, i1, o1, i2, o2 in rows[:limit]:
            print(f"  {ref[:44]:44} {i1:>7.3f}->{i2:<8.3f} {o1:>7.3f}->{o2:<8.3f} {delta:>+7.1%}")
        if len(rows) > limit:
            print(f"  ... and {len(rows) - limit} more (raise --limit to see them)")

    show(cheaper, "CHEAPER")
    show(dearer, "MORE EXPENSIVE")
    if added:
        print(f"\nNEW ({len(added)}): {', '.join(sorted(added)[:8])}"
              + (" ..." if len(added) > 8 else ""))
    if removed:
        # Named, not hidden: a model that vanished from the catalogue is one the
        # router may still be configured to reach for.
        print(f"\nGONE ({len(removed)}): {', '.join(sorted(removed)[:8])}"
              + (" ..." if len(removed) > 8 else ""))
    if not any((cheaper, dearer, added, removed)):
        print("\nno price changes")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=sorted(SOURCES), default=DEFAULT_SOURCE,
                     help=f"which upstream price table to refresh (default: {DEFAULT_SOURCE})")
    ap.add_argument("--url", default=None, help="override the chosen source's URL")
    ap.add_argument("--dry-run", action="store_true", help="show the diff, write nothing")
    ap.add_argument("--provider", default="", help="only report on this provider")
    ap.add_argument("--limit", type=int, default=15)
    ap.add_argument("--timeout", type=float, default=60.0)
    args = ap.parse_args()

    source = SOURCES[args.source]
    url = args.url or source["url"]
    target = source["target"]

    for p in DEFAULT_CATALOG_PATHS:
        if p.exists():
            age = (time.time() - p.stat().st_mtime) / 86400
            print(f"installed: {p}  ({p.stat().st_size / 1e6:.1f}MB, {age:.0f} days old)")

    before = current_prices()
    print(f"\nfetching [{args.source}] {url} ...")
    try:
        payload = fetch(url, args.timeout)
    except Exception as exc:
        # Never clobber a working catalogue with a failed fetch. Stale and usable
        # beats fresh and absent.
        print(f"FAILED: {exc.__class__.__name__}: {exc}")
        print("the installed catalogue is untouched")
        return 1

    fd, tmp = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        tmp_path.write_text(json.dumps(payload), encoding="utf-8")
        after = {
            m.ref: (m.input_cost_per_1m, m.output_cost_per_1m, m.cache_read_cost_per_1m)
            for m in Catalog.from_file(tmp_path).all()
        }
    finally:
        tmp_path.unlink(missing_ok=True)

    print(f"fetched {len(after)} models (installed catalogue has {len(before)})")
    if not after:
        print("FAILED: the fetched payload parsed to zero models; not installing")
        return 1

    report(before, after, args.provider, args.limit)

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    install(payload, target)
    print(f"\ninstalled -> {target}")
    if args.source == "litellm":
        print("forgeos/catalog.py checks this path last, so it wins over models.dev "
              "on any overlapping model ref.")
    else:
        print("forgeos/catalog.py checks this path first among models.dev caches; "
              "litellm's cache (if installed) still wins on an overlapping ref.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

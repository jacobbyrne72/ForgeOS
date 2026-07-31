"""Refresh the model price catalogue, and show what moved.

Every routing decision ForgeOS makes is priced against this catalogue: which
worker is "cheapest capable", what the preflight estimate says, whether the
governor refuses a call. A stale catalogue does not merely report old numbers —
it routes work to the wrong model and mis-estimates every call, silently.

The bundled copy was seeded from a Hermes cache that was **51 days old**, which
is several provider price cuts ago. This fetches current pricing from models.dev
and writes `~/.forgeos/models_dev_cache.json`, the first path
`forgeos/catalog.py` looks at, so it takes precedence over any older copy.

    python tools/refresh_catalog.py --dry-run     # show the diff, write nothing
    python tools/refresh_catalog.py               # fetch and install
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

from forgeos.catalog import DEFAULT_CATALOG_PATHS, Catalog, default_catalog  # noqa: E402

SOURCE_URL = "https://models.dev/api.json"
TARGET = Path(os.path.expanduser("~/.forgeos/models_dev_cache.json"))


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
    ap.add_argument("--url", default=SOURCE_URL)
    ap.add_argument("--dry-run", action="store_true", help="show the diff, write nothing")
    ap.add_argument("--provider", default="", help="only report on this provider")
    ap.add_argument("--limit", type=int, default=15)
    ap.add_argument("--timeout", type=float, default=60.0)
    args = ap.parse_args()

    for p in DEFAULT_CATALOG_PATHS:
        if p.exists():
            age = (time.time() - p.stat().st_mtime) / 86400
            print(f"installed: {p}  ({p.stat().st_size / 1e6:.1f}MB, {age:.0f} days old)")

    before = current_prices()
    print(f"\nfetching {args.url} ...")
    try:
        payload = fetch(args.url, args.timeout)
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

    install(payload, TARGET)
    print(f"\ninstalled -> {TARGET}")
    print("forgeos/catalog.py reads this path first, so it now takes precedence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

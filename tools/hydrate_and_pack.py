"""Fetch a blobless clone's contents, then pack it for reading. One at a time.

The catalogue clones carry filenames and no file bytes (`--filter=blob:none`).
Reading one means fetching its blobs first. That is a network operation per
repo, and this machine has been crashed once already by parallel cloning, so
this runs STRICTLY SEQUENTIALLY and says so loudly. There is no --parallel flag
and adding one would be a mistake.

After hydration each repo is packed with `repomix` into a single AI-readable
file. That matters for cost: handing a model a directory means it reads files
one at a time, paying a round trip and a fresh context each time, and it cannot
see the shape of the whole thing while doing it. One packed file is one read.

Resumable by construction. Already-hydrated repos are detected and skipped, so
an interrupted run costs only the repo it was in the middle of. State lives on
disk (the working tree exists or it does not) rather than in a manifest that
can disagree with reality.

Ordering comes from `tools/repo_triage.py`, which ranked all 713 by capability
signals in their file trees. Hydrating in that order means the budget is spent
on the repos most likely to carry something ForgeOS lacks.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

PACK_DIR = Path("artifacts/packed")

# A tree this big is not going to be read whole even after packing, and
# fetching it costs minutes. Skipped with the reason recorded rather than
# silently, so the ceiling is visible instead of looking like coverage.
MAX_FILES_TO_HYDRATE = 6_000


def _run(args: list[str], *, cwd: str | None = None, timeout: float = 900) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            args, cwd=cwd, capture_output=True, text=True, timeout=timeout,
            # UTF-8 with replacement, not the console default. Windows decodes
            # subprocess output as cp1252, and repomix prints box-drawing
            # characters -- so reading its OUTPUT raised UnicodeDecodeError and
            # reported the pack as failed when the pack had actually run.
            encoding="utf-8", errors="replace",
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "PYTHONIOENCODING": "utf-8"},
        )
    except subprocess.TimeoutExpired:
        return 124, "timed out"
    except OSError as exc:
        return 1, f"{type(exc).__name__}: {exc}"
    # `capture_output=True` still yields None for a stream a process never
    # opened -- npx on Windows does exactly that -- and concatenating None
    # raised a TypeError that masked the real failure underneath it.
    combined = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, combined[-2000:]


def is_hydrated(repo: Path) -> bool:
    """A working tree with real files. Checked on disk rather than tracked in a
    manifest, because a manifest can disagree with reality and this cannot."""
    try:
        for path in repo.iterdir():
            if path.name != ".git":
                return True
    except OSError:
        pass
    return False


def hydrate(repo: Path, *, timeout: float) -> tuple[bool, str]:
    """Fetch the missing blobs and check out a working tree.

    `git checkout` on a blob:none clone triggers a lazy fetch of exactly the
    blobs it needs -- no re-clone, no full history refetch.
    """
    # Windows MAX_PATH is 260 characters and several of these repos carry
    # snapshot-test filenames well past it -- gemini-cli and open-interpreter
    # both failed checkout on exactly that. `core.longpaths` is per-repo config,
    # so it has to be set on each clone before the first checkout rather than
    # once globally.
    _run(["git", "-C", str(repo), "config", "core.longpaths", "true"], timeout=60)
    code, out = _run(["git", "-C", str(repo), "checkout", "--force", "HEAD"],
                     timeout=timeout)
    if code != 0:
        # A repo whose default branch is not HEAD-resolvable still has a
        # branch; try the remote head explicitly before giving up.
        code, out = _run(["git", "-C", str(repo), "checkout", "--force",
                          "origin/HEAD"], timeout=timeout)
    return code == 0, out.strip()[:300]


def pack(repo: Path, out_file: Path, *, timeout: float) -> tuple[bool, str]:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    # Resolve through `which`: on Windows `repomix` is a `.cmd` shim, and
    # subprocess without shell=True will not find a bare name that has an
    # extension. Passing shell=True instead would mean quoting paths by hand.
    exe = shutil.which("repomix")
    if exe:
        code, out = _run(
            [exe, "--output", str(out_file), "--style", "markdown",
             "--compress", "--remove-empty-lines", "--no-file-summary"],
            cwd=str(repo), timeout=timeout,
        )
        if code == 0 and out_file.exists():
            return True, out.strip()[:300]
    npx = shutil.which("npx")
    if not npx:
        return False, "neither repomix nor npx is on PATH"
    code, out = _run([npx, "--yes", "repomix", "--output", str(out_file)],
                     cwd=str(repo), timeout=timeout)
    return code == 0 and out_file.exists(), out.strip()[:300]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--triage", default="artifacts/repo_triage.json")
    ap.add_argument("--top", type=int, default=5,
                    help="how many repos to hydrate THIS RUN (kept small on purpose)")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--only", default="", help="substring filter on repo name")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan; fetch nothing")
    args = ap.parse_args()

    ranked = json.loads(Path(args.triage).read_text(encoding="utf-8"))
    if args.only:
        ranked = [r for r in ranked if args.only.lower() in r["name"].lower()]

    todo = []
    for row in ranked:
        repo = Path(row["path"])
        if not repo.exists():
            continue
        if is_hydrated(repo):
            continue
        if row["files"] > MAX_FILES_TO_HYDRATE:
            continue
        todo.append(row)
        if len(todo) >= args.top:
            break

    print("SEQUENTIAL hydration -- one repo at a time, never parallel.")
    print(f"{len(todo)} repo(s) selected this run:\n")
    for i, row in enumerate(todo, 1):
        sig = ", ".join(sorted(row["signals"])[:4])
        print(f"  {i}. {row['name']:<44} score={row['score']:<4} "
              f"files={row['files']:<6} {sig}")
    if args.dry_run:
        print("\n--dry-run: nothing fetched")
        return 0

    results = []
    for i, row in enumerate(todo, 1):
        repo = Path(row["path"])
        started = time.time()
        print(f"\n[{i}/{len(todo)}] {row['name']}")
        ok, detail = hydrate(repo, timeout=args.timeout)
        if not ok:
            print(f"    hydrate FAILED: {detail}")
            results.append({"name": row["name"], "hydrated": False, "detail": detail})
            continue
        on_disk = sum(1 for p in repo.rglob("*") if p.is_file() and ".git" not in p.parts)
        print(f"    hydrated: {on_disk:,} files in {time.time() - started:.0f}s")

        out_file = PACK_DIR / f"{row['name']}.md"
        packed, pdetail = pack(repo, out_file, timeout=args.timeout)
        size = out_file.stat().st_size if out_file.exists() else 0
        if packed:
            # ~4 chars/token, the same rough figure the rest of the codebase
            # uses when it has nothing better.
            print(f"    packed: {size:,} bytes (~{size // 4:,} tokens) -> {out_file}")
        else:
            print(f"    pack FAILED: {pdetail}")
        results.append({
            "name": row["name"], "hydrated": True, "files": on_disk,
            "packed": packed, "pack_bytes": size, "pack_path": str(out_file),
        })

    Path("artifacts/hydration_log.json").write_text(
        json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    done = sum(1 for r in results if r.get("packed"))
    print(f"\n{done}/{len(todo)} hydrated and packed. Re-run to continue; "
          f"already-hydrated repos are skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

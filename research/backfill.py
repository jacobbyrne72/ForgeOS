"""Backfill transcripts for enumerated videos that were never fetched.

The first pass applied a relevance filter and kept 495 of 3,406 enumerated
videos. That filter was tuned once and got at least one channel badly wrong
(a keyword list without LLM vocabulary matched 1 of 17 Karpathy videos), so
the safer move is to widen coverage per channel rather than trust it twice.

Round-robins across channels so every channel gains, instead of one prolific
channel consuming the whole budget. Costs no model tokens — yt-dlp only.

    python research/backfill.py --per-channel 60
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRANSCRIPTS = ROOT / "transcripts"
ENUM = ROOT / "enumerated_deep.tsv"


def already_have() -> set[str]:
    return {p.name.split("_")[0] for p in TRANSCRIPTS.glob("*.txt")}


def load_rows() -> list[tuple[str, str, str]]:
    rows = []
    for line in ENUM.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split("|||")
        if len(parts) >= 3:
            rows.append((parts[0], parts[1], parts[2]))
    return rows


def _written(video_id: str) -> Path | None:
    """Any subtitle file this id actually produced, whatever suffix yt-dlp chose."""
    for p in TRANSCRIPTS.glob(f"{video_id}*"):
        if p.suffix.lower() in (".vtt", ".srt", ".txt") and p.stat().st_size > 0:
            return p
    return None


def fetch(video_id: str) -> tuple[bool, str]:
    """Subtitles only — never the media. Returns (wrote_a_file, reason).

    Success is a FILE ON DISK, not an exit code. `--ignore-errors` makes yt-dlp
    exit 0 when a video simply has no subtitles, and it exits 0 under rate
    limiting too — so an exit-code check reports a perfect run while writing
    nothing at all. That is exactly what happened on the first attempt here:
    1,087 "successes", zero files, because YouTube was returning 429 the whole
    time. Silence is not success.
    """
    out = TRANSCRIPTS / f"{video_id}_{video_id}"
    cmd = [
        "yt-dlp", "--skip-download",
        "--write-auto-subs", "--write-subs", "--sub-langs", "en.*",
        "--sub-format", "vtt",
        "--no-warnings",
        # Politeness, because the throttle is the binding constraint, not our speed.
        "--sleep-requests", "1",
        "--retries", "3", "--retry-sleep", "exp=5:120",
        "-o", str(out),
        f"https://www.youtube.com/watch?v={video_id}",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except Exception as exc:
        return False, f"spawn failed: {exc.__class__.__name__}"

    if _written(video_id) is not None:
        return True, "ok"
    err = (r.stderr or "").strip().splitlines()
    last = err[-1][:120] if err else "no subtitle file written"
    if "429" in (r.stderr or ""):
        return False, "rate limited (429)"
    return False, last


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-channel", type=int, default=60)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    have = already_have()
    rows = load_rows()

    by_channel: dict[str, list[str]] = {}
    for chan, vid, _title in rows:
        if vid not in have:
            by_channel.setdefault(chan, []).append(vid)

    queue: list[tuple[str, str]] = []
    for chan, vids in by_channel.items():
        for vid in vids[: args.per_channel]:
            queue.append((chan, vid))

    print(f"have {len(have)} transcripts; {len(queue)} queued "
          f"across {len(by_channel)} channels", flush=True)
    if args.dry_run:
        for chan, vids in sorted(by_channel.items()):
            print(f"  {chan:28} missing {len(vids):>4}, taking {min(len(vids), args.per_channel)}")
        return 0

    ok = fail = 0
    throttled = 0
    reasons: dict[str, int] = {}
    for i, (chan, vid) in enumerate(queue, 1):
        wrote, why = fetch(vid)
        if wrote:
            ok += 1
            throttled = 0
        else:
            fail += 1
            reasons[why] = reasons.get(why, 0) + 1
            if "rate limited" in why:
                throttled += 1
                # Back off hard rather than grinding through a thousand 429s and
                # calling it a run. Give up rather than pretend.
                if throttled >= 8:
                    print(f"  stopping at {i}/{len(queue)}: rate limited "
                          f"{throttled} times in a row — try again later",
                          flush=True)
                    break
                time.sleep(min(60, 5 * throttled))
        if i % 25 == 0:
            print(f"  {i}/{len(queue)}  wrote={ok} failed={fail}", flush=True)

    print(f"done: wrote={ok} failed={fail}, transcripts now {len(already_have())}",
          flush=True)
    if reasons:
        print("failure reasons:", flush=True)
        for why, n in sorted(reasons.items(), key=lambda kv: -kv[1])[:8]:
            print(f"  {n:>5}  {why}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Filter enumerated video titles down to the ones worth spending transcript budget on.

Written in Python rather than grep because YouTube titles contain emoji, which makes
the raw file invalid UTF-8 — grep silently dropped every line, which looked exactly
like "nothing matched". Decoding with errors="replace" is the fix.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

R = Path(r"C:\Users\byrne\Downloads\hive\research")

INCLUDE = re.compile(
    r"agent|multi[- ]?agent|swarm|orchestrat|coding agent|claude code|codex|gemini cli"
    r"|opencode|aider|\bmcp\b|\bacp\b|context engineer|prompt engineer|memory|\brag\b"
    r"|eval|rout(e|ing)|model select|token|subagent|worktree|sandbox|automation|n8n"
    r"|langgraph|pydantic ai|crew ?ai|openrouter|litellm|cheap|cost|spec|skill|harness"
    r"|parallel|context|cli|build|architect|workflow|pipeline|local model|ollama|vllm",
    re.I,
)
EXCLUDE = re.compile(
    r"will change everything|is insane|breaking news|destroys|make money|top \d+ tools"
    r"|daily ai news|reaction|rumou?r|shocking|you won'?t believe|giveaway",
    re.I,
)


def main() -> int:
    raw = R / "enumerated.tsv"
    if not raw.exists():
        print(f"missing {raw}", file=sys.stderr)
        return 1

    text = raw.read_text(encoding="utf-8", errors="replace")
    # yt-dlp's --print does NOT expand \t in the template, so the id/title separator
    # arrived as the two characters backslash-t while the channel prefix is a real tab.
    text = text.replace("\\t", "\t")
    kept: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    total = 0

    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        channel, vid, title = parts[0], parts[1], "\t".join(parts[2:])
        total += 1
        if not INCLUDE.search(title) or EXCLUDE.search(title):
            continue
        if vid in seen:
            continue
        seen.add(vid)
        kept.append((vid, channel, title))

    (R / "matched.tsv").write_text(
        "\n".join(f"{v}\t{c}\t{t}" for v, c, t in kept), encoding="utf-8"
    )
    (R / "matched_ids.txt").write_text(
        "\n".join(f"https://youtu.be/{v}" for v, _, _ in kept) + "\n", encoding="utf-8"
    )

    print(f"enumerated : {total}")
    print(f"kept       : {len(kept)}")
    print(f"dropped    : {total - len(kept)}")
    print("\nkept per channel:")
    for ch, n in Counter(c for _, c, _ in kept).most_common():
        print(f"  {ch:<26} {n}")
    print("\nsample kept titles:")
    for _, ch, t in kept[:12]:
        # Windows console is cp1252; emoji in titles would crash the print, not the run.
        safe = t[:88].encode("ascii", "replace").decode("ascii")
        print(f"  [{ch}] {safe}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

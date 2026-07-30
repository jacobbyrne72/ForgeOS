"""Filter the deep enumeration into a weighted corpus.

Two things this does that a flat keyword filter does not:

1. **Per-channel caps by tier.** Taking 200 videos from a news channel and 200 from
   the best harness-engineering channel would defeat the whole point of ranking them.
   A discovery channel earns a small quota; a primary-builder channel earns a large one.
2. **Tags each kept video with its technical layer**, so model-internals material and
   agent-harness material never collapse into one undifferentiated pile — they answer
   different questions and belong in separately searchable collections.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path

R = Path(r"C:\Users\byrne\Downloads\forgeos\research")
RAW = R / "enumerated_deep.tsv"

# channel -> (quota, technical_layer). Quota reflects signal density, not volume.
CHANNELS: dict[str, tuple[int, str]] = {
    # Primary builders / harness engineering — the material this project is about.
    "indydevdan": (70, "agent_harness"),
    "colemedin": (45, "agent_harness"),
    "AIJasonZ": (35, "agent_harness"),
    "AllAboutAI": (35, "agent_harness"),
    "daveebbelaar": (40, "ai_engineering"),
    "jamesbriggs": (35, "context_retrieval"),
    # Model internals — a different corpus, deliberately tagged apart.
    "AndrejKarpathy": (25, "model_training"),
    "samwitteveenai": (25, "ai_engineering"),
    "AssemblyAI": (15, "ai_engineering"),
    "LangChain": (30, "framework_canonical"),
    # Workflow / automation.
    "MervinPraison": (25, "workflow"),
    "davidondrej": (25, "workflow"),
    "rileybrownai": (15, "operator_workflow"),
    "hyperautomationlabs1045": (20, "workflow"),
    # Discovery only — small quotas, high dedup, treated as leads not evidence.
    "matthew_berman": (12, "discovery"),
    "mreflow": (8, "discovery"),
    "WorldofAI": (8, "discovery"),
    "SkillLeapAI": (6, "discovery"),
    "aiadvantage": (6, "discovery"),
    "futurepedia": (4, "discovery"),
    "Aitrepreneur": (12, "local_models"),
}

INCLUDE = re.compile(
    r"agent|multi[- ]?agent|swarm|orchestrat|coding agent|claude code|codex|gemini cli"
    r"|opencode|aider|\bmcp\b|\bacp\b|context engineer|prompt engineer|memory|\brag\b"
    r"|eval|rout(e|ing)|model select|token|subagent|worktree|sandbox|n8n|langgraph"
    r"|pydantic ai|crew ?ai|openrouter|litellm|cheap|cost|spec|skill|harness|parallel"
    r"|context|architect|pipeline|local model|ollama|vllm|transformer|attention|tokeniz"
    r"|fine[- ]?tun|train|gpt from scratch|inference|quantiz|embedding|vector|retriev"
    r"|benchmark|test|debug|refactor|workflow|automat"
    # Model-internals vocabulary. Omitting these cost the single most important
    # channel almost its entire catalogue on the first pass — titles like
    # "Deep Dive into LLMs" and "Let's reproduce GPT-2" matched nothing.
    r"|\bllm\b|\bllms\b|language model|neural|\bgpt\b|nanogpt|backprop|gradient"
    r"|from scratch|deep dive|micrograd|autograd|\bcuda\b|scaling law|reinforcement"
    r"|diffusion|\blora\b|\brl\b|reason",
    re.I,
)
EXCLUDE = re.compile(
    r"will change everything|is insane|breaking news|destroys|make money|top \d+ tools"
    r"|daily ai news|reaction|rumou?r|shocking|you won'?t believe|giveaway|clickbait"
    r"|\bnews\b.*\bround ?up\b|weekly news",
    re.I,
)


def main() -> int:
    if not RAW.exists():
        print(f"missing {RAW}")
        return 1

    text = RAW.read_text(encoding="utf-8", errors="replace")
    by_channel: dict[str, list[tuple[str, str]]] = defaultdict(list)
    seen: set[str] = set()
    total = 0

    for line in text.splitlines():
        parts = line.split("|||")
        if len(parts) < 3:
            continue
        channel, vid, title = parts[0].strip(), parts[1].strip(), "|||".join(parts[2:]).strip()
        total += 1
        if channel not in CHANNELS:
            continue
        if not INCLUDE.search(title) or EXCLUDE.search(title):
            continue
        if vid in seen:
            continue
        seen.add(vid)
        by_channel[channel].append((vid, title))

    kept: list[tuple[str, str, str, str]] = []
    for channel, items in by_channel.items():
        quota, layer = CHANNELS[channel]
        # Enumeration is newest-first, so the head of the list is the most current
        # material — which matters when tooling changes every few months.
        for vid, title in items[:quota]:
            kept.append((vid, channel, layer, title))

    (R / "corpus.tsv").write_text(
        "\n".join(f"{v}\t{c}\t{lay}\t{t}" for v, c, lay, t in kept), encoding="utf-8"
    )
    (R / "corpus_ids.txt").write_text(
        "\n".join(f"https://youtu.be/{v}" for v, _, _, _ in kept) + "\n", encoding="utf-8"
    )

    print(f"enumerated : {total}")
    print(f"matched    : {sum(len(v) for v in by_channel.values())}")
    print(f"kept (quota-capped): {len(kept)}")
    print("\nby technical layer:")
    for layer, n in Counter(lay for _, _, lay, _ in kept).most_common():
        print(f"  {layer:<22} {n}")
    print("\nby channel:")
    for ch, n in Counter(c for _, c, _, _ in kept).most_common():
        print(f"  {ch:<26} {n}/{CHANNELS[ch][0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

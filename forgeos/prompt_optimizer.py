"""Prompt length optimizer - trims prompts to save tokens.

Reduces prompt length by removing excessive whitespace,
truncating long history to the most recent entries,
and stripping low-value prefix text.

Consolidates prompt_optimizer + prompt_shrinker + prompt_summarizer
(2026-07-31): three independent "reduce this prompt" implementations that
shared the same whitespace-cleanup core (now _clean_whitespace()).
optimize_prompt() keeps its history-trimming and identity-line dedup;
shrink_prompt() keeps prompt_shrinker.py's hard token-budget truncation;
remove_filler_words() is prompt_summarizer.py's distinct technique (strip
conversational filler). Both donor modules are gone, neither was wired
anywhere or used by any other module. Behavior of optimize_prompt() and
shrink_prompt() is unchanged from before the merge.
"""
from __future__ import annotations

import re

_FILLER_WORDS = [
    "basically", "essentially", "honestly", "to be honest",
    "I mean", "kind of", "sort of", "you know",
    "well", "so", "like", "just", "actually",
]


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token."""
    return max(1, len(text) // 4)


estimate_prompt_tokens = estimate_tokens  # back-compat name from prompt_shrinker.py


def _clean_whitespace(text: str) -> str:
    """Collapse blank-line runs and strip trailing/edge whitespace.

    Shared by optimize_prompt() and shrink_prompt() — both prompt_optimizer.py
    and prompt_shrinker.py did this identically before the merge.
    """
    cleaned = re.sub(r'\n{3,}', '\n\n', text)
    cleaned = re.sub(r'[ \t]+$', '', cleaned, flags=re.MULTILINE)
    return cleaned.strip()


def remove_filler_words(text: str) -> str:
    """Strip conversational filler words/phrases (from prompt_summarizer.py)."""
    shortened = text
    for filler in _FILLER_WORDS:
        shortened = shortened.replace(" " + filler + " ", " ")
        shortened = shortened.replace(" " + filler + "., ", ". ")
        shortened = shortened.replace(" " + filler + ".", ".")
    return re.sub(r'\s+', ' ', shortened).strip()


def optimize_prompt(
    prompt: str,
    max_tokens: int = 8192,
    trim_history_to: int = 10,
    remove_filler: bool = False,
) -> tuple[str, dict]:
    """Optimize a prompt to minimize token usage.

    Returns (optimized_prompt, stats dict).
    """
    orig_tokens = estimate_tokens(prompt)

    optimized = _clean_whitespace(prompt)

    # Truncate history entries to last N
    hist_pat = r'(Previous turns?:.*?)(\n\n|\Z)'   
    hist_m = re.search(hist_pat, optimized, re.DOTALL)
    if hist_m:
        hist_text = hist_m.group(1)
        entries = re.split(r'(?<=\n)\s*-\s*', hist_text)
        if len(entries) > trim_history_to:
            kept = entries[-trim_history_to:]
            truncated = "- " + chr(10) + "- ".join(kept)
            optimized = optimized[:hist_m.start()] + truncated + optimized[hist_m.end():]

    # 5. Compress repeated system identity messages
    dup_pat = r'(You are [^\n]{0,80}?)\1'
    optimized = re.sub(dup_pat, r'\1', optimized, flags=re.DOTALL | re.IGNORECASE)

    if remove_filler:
        optimized = remove_filler_words(optimized)

    final_tokens = estimate_tokens(optimized)
    tokens_saved = max(0, orig_tokens - final_tokens)

    stats = {
        "original_tokens": orig_tokens,
        "optimized_tokens": final_tokens,
        "tokens_saved": tokens_saved,
        "savings_pct": round(tokens_saved / max(1, orig_tokens) * 100, 1),
        "under_budget": final_tokens <= max_tokens,
    }

    return optimized, stats


def shrink_prompt(prompt: str, target_tokens: int = 2048) -> tuple[str, dict]:
    """Clean whitespace, then hard-truncate to a strict token ceiling.

    (from prompt_shrinker.py — unlike optimize_prompt(), which only reports
    whether max_tokens was met, this always enforces target_tokens.)
    """
    original_tokens = estimate_tokens(prompt)

    shrunk = _clean_whitespace(prompt)

    tokens = estimate_tokens(shrunk)
    if tokens > target_tokens:
        ratio = target_tokens / tokens
        char_limit = int(len(shrunk) * ratio)
        shrunk = shrunk[:char_limit]
        last_nl = shrunk.rfind('\n')
        if last_nl > 0:
            shrunk = shrunk[:last_nl]
        shrunk += "\n... (truncated to save tokens)"

    final_tokens = estimate_tokens(shrunk)
    saved = original_tokens - final_tokens

    return shrunk, dict(
        original_tokens=original_tokens,
        shrunk_tokens=final_tokens,
        tokens_saved=saved,
        savings_pct=round(saved / max(1, original_tokens) * 100, 1),
    )
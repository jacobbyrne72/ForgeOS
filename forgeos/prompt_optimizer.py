"""Prompt length optimizer - trims prompts to save tokens.

Reduces prompt length by removing excessive whitespace,
truncating long history to the most recent entries,
and stripping low-value prefix text.
"""
from __future__ import annotations

import re


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token."""
    return max(1, len(text) // 4)


def optimize_prompt(
    prompt: str,
    max_tokens: int = 8192,
    trim_history_to: int = 10,
) -> tuple[str, dict]:
    """Optimize a prompt to minimize token usage.

    Returns (optimized_prompt, stats dict).
    """
    original = prompt
    orig_tokens = estimate_tokens(original)

    # 1. Remove excessive whitespace (3+ newlines become 2)
    optimized = re.sub(r'\n{3,}', '\n\n', prompt)
    # 2. Remove trailing whitespace per line
    optimized = re.sub(r'[ \t]+$', '', optimized, flags=re.MULTILINE)
    # 3. Strip leading/trailing blank lines
    optimized = optimized.strip()

    # 4. Truncate history entries to last N
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
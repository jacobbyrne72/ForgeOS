"""Prompt shrinker - reduces input token count before API calls."""
from __future__ import annotations

import re


def shrink_prompt(prompt: str, target_tokens: int = 2048):
    """Shrink prompt to fit within token budget."""
    original_tokens = max(1, len(prompt) // 4)

    # 1. Remove excessive blank lines
    shrunk = re.sub(r"\n{3,}", "\n\n", prompt)

    # 2. Strip trailing whitespace per line
    shrunk = re.sub(r"[ \t]+$", "", shrunk, flags=re.MULTILINE)

    # 3. Strip leading/trailing whitespace
    shrunk = shrunk.strip()

    # 4. Truncate to target tokens if needed
    tokens = max(1, len(shrunk) // 4)
    if tokens > target_tokens:
        ratio = target_tokens / tokens
        char_limit = int(len(shrunk) * ratio)
        shrunk = shrunk[:char_limit]
        last_nl = shrunk.rfind('\n')
        if last_nl > 0:
            shrunk = shrunk[:last_nl]
        shrunk += "\n... (truncated to save tokens)"

    final_tokens = max(1, len(shrunk) // 4)
    saved = original_tokens - final_tokens

    return shrunk, dict(
        original_tokens=original_tokens,
        shrunk_tokens=final_tokens,
        tokens_saved=saved,
        savings_pct=round(saved / max(1, original_tokens) * 100, 1),
    )


def estimate_prompt_tokens(prompt: str):
    """Estimate token count for a prompt."""
    return max(1, len(prompt) // 4)

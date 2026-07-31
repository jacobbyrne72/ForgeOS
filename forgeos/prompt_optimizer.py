"""Prompt optimizer - reduce token billing by trimming and structuring prompts."""
from __future__ import annotations

import re
from typing import Optional


def optimize_prompt(
    prompt: str,
    target_tokens: Optional[int] = None,
    remove_empty_lines: bool = True,
    remove_duplicate_lines: bool = True,
    strip_whitespace: bool = True,
    collapse_multiple_spaces: bool = True,
) -> tuple[str, dict]:
    """Optimize a prompt to reduce token billing while preserving meaning."""
    original = prompt
    stats = {"original_tokens_est": len(prompt) // 4, "changes": []}

    if strip_whitespace:
        prompt = prompt.strip()
        stats["changes"].append("strip_whitespace")

    if remove_empty_lines:
        lines = prompt.split("\n")
        filtered = [l for l in lines if l.strip()]
        if len(filtered) < len(lines):
            stats["changes"].append("removed_empty_lines")
        prompt = "\n".join(filtered)

    if remove_duplicate_lines:
        lines = prompt.split("\n")
        seen = set()
        unique = []
        for l in lines:
            key = l.strip().lower()
            if key not in seen:
                seen.add(key)
                unique.append(l)
        if len(unique) < len(lines):
            stats["changes"].append("removed_duplicate_lines")
        prompt = "\n".join(unique)

    if collapse_multiple_spaces:
        collapsed = re.sub(r" {2,}", " ", prompt)
        if collapsed != prompt:
            stats["changes"].append("collapsed_multiple_spaces")
        prompt = collapsed

    if target_tokens is not None:
        est = len(prompt) // 4
        if est > target_tokens:
            words = prompt.split()
            truncated = []
            running = 0
            for w in words:
                if running + len(w) // 4 > target_tokens:
                    break
                truncated.append(w)
                running += len(w) // 4
            prompt = " ".join(truncated)
            stats["changes"].append("truncated_to_target")

    stats["optimized_tokens_est"] = max(1, len(prompt) // 4)
    stats["tokens_saved"] = stats["original_tokens_est"] - stats["optimized_tokens_est"]
    stats["savings_pct"] = round(
        stats["tokens_saved"] / max(stats["original_tokens_est"], 1) * 100, 1
    )
    return prompt, stats


def estimate_tokens(text: str, tokens_per_char: float = 0.25) -> int:
    """Quick token estimate (4 chars ~= 1 token for English text)."""
    return max(1, int(len(text) * tokens_per_char))

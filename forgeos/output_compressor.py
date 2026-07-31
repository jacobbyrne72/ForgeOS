"""Compress and truncate model outputs to reduce token billing.

Consolidates output_compressor + response_compressor + response_truncator
(2026-07-31): three independent "shrink this response" implementations.
compress_output() and truncate_response() keep their original, unmodified
behavior — both are wired into separate `forge` CLI commands with genuinely
different truncation-boundary heuristics (last-newline vs. sentence-ending
punctuation), which is a real difference, not duplication.
compress_by_sentence_ratio() is response_compressor.py's distinct extractive
technique (keep the first N% of sentences rather than truncate to a token
budget), preserved as a function; the per-call stats-tracking class it
shipped in was dropped since nothing else here is stateful. Both donor
modules are gone; neither was wired anywhere else.
"""
from __future__ import annotations

import re


def compress_output(output: str, target_tokens: int = 4096):
    """Compress output to fit within token budget."""
    original_tokens = max(1, len(output) // 4)

    # 1. Collapse 3+ blank lines to 2
    compressed = re.sub(r"\n{3,}", "\n\n", output)

    # 2. Strip trailing whitespace per line
    compressed = re.sub(r"[ \t]+$", "", compressed, flags=re.MULTILINE)

    # 3. Strip leading/trailing blank lines
    compressed = compressed.strip()

    # 4. Truncate to target tokens
    tokens = max(1, len(compressed) // 4)
    if tokens > target_tokens:
        compression_ratio = target_tokens / tokens
        char_limit = int(len(compressed) * compression_ratio)
        truncated = compressed[:char_limit]
        last_newline = truncated.rfind("\n")
        if last_newline > 0:
            truncated = truncated[:last_newline]
        compressed = truncated + "\n... (token budget enforced)"

    final_tokens = max(1, len(compressed) // 4)
    tokens_saved = original_tokens - final_tokens

    return compressed, dict(
        original_tokens=original_tokens,
        compressed_tokens=final_tokens,
        tokens_saved=tokens_saved,
        savings_pct=round(tokens_saved / max(1, original_tokens) * 100, 1),
    )


def estimate_output_tokens(output: str):
    """Estimate token count for a model output."""
    return max(1, len(output) // 4)


def truncate_response(text: str, max_tokens: int = 4096) -> tuple[str, dict]:
    """Truncate response to token budget. Saves cost on over-length outputs.

    Cuts at a sentence boundary when possible. (from response_truncator.py,
    unchanged — a different boundary heuristic from compress_output()'s
    last-newline cut above, kept as its own function rather than merged.)
    """
    estimated_tokens = max(1, len(text) // 4)
    if estimated_tokens <= max_tokens:
        return text, dict(
            truncated=False,
            original_tokens=estimated_tokens,
            final_tokens=estimated_tokens,
            tokens_saved=0,
        )

    # Truncate at budget limit
    char_limit = max_tokens * 4
    truncated = text[:char_limit]

    # Cut at last sentence boundary if possible
    for sent_end in [". ", "! ", "? ", "\n\n"]:
        idx = truncated.rfind(sent_end)
        if idx > char_limit // 2:
            truncated = truncated[:idx + 1]
            break

    truncated += "\n...(truncated to save tokens)"
    final_tokens = max(1, len(truncated) // 4)
    saved = estimated_tokens - final_tokens

    return truncated, dict(
        truncated=True,
        original_tokens=estimated_tokens,
        final_tokens=final_tokens,
        tokens_saved=saved,
        savings_pct=round(saved / max(1, estimated_tokens) * 100, 1),
    )


def compress_by_sentence_ratio(text: str, max_ratio: float = 0.7) -> str:
    """Extractive compression: keep the first max_ratio fraction of sentences.

    A different technique from the token-budget truncation above — no token
    target, just a sentence-count ratio. (from response_compressor.py's
    ResponseCompressor.compress.)
    """
    if not text or len(text.split()) <= 10:
        return text
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) <= 2:
        return text
    kept = sentences[:max(2, int(len(sentences) * max_ratio))]
    return ' '.join(kept)

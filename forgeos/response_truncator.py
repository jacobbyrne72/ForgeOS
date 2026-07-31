"""Response truncation middleware - enforces token budgets on model responses."""
from __future__ import annotations


def truncate_response(text: str, max_tokens: int = 4096) -> tuple[str, dict]:
    """Truncate response to token budget. Saves cost on over-length outputs."""
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

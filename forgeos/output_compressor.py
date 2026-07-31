"""Compress model outputs to reduce token billing."""

from __future__ import annotations



import re





def compress_output(output: str, target_tokens: int = 4096):

    """Compress output to fit within token budget."""

    original_tokens = max(1, len(output) // 4)



    # 1. Collapse 3+ blank lines to 2

    compressed = re.sub(r"\n{3,}", "\n\n", output)



    # 2. Strip trailing whitespace per line

    compressed = re.sub(r"[ 	]+$", "", compressed, flags=re.MULTILINE)



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

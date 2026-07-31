"""Pre-API prompt summarizer — shorten user prompts to save tokens."""
from __future__ import annotations


class PromptSummarizer:
    """Summarize verbose prompts to reduce API token usage."""

    def __init__(self, max_output_tokens: int = 500):
        self.max_output_tokens = max_output_tokens
        self.summaries_created = 0

    def summarize(self, prompt: str) -> str:
        """Summarize a prompt to its essential meaning."""
        if not prompt:
            return prompt

        # Remove redundant filler words
        filler_words = [
            "basically", "essentially", "honestly", "to be honest",
            "I mean", "kind of", "sort of", "you know",
            "well", "so", "like", "just", "actually"
        ]
        shortened = prompt
        for filler in filler_words:
            shortened = shortened.replace(" " + filler + " ", " ")
            shortened = shortened.replace(" " + filler + "., ", ". ")
            shortened = shortened.replace(" " + filler + ".", ".")

        # Collapse multiple spaces
        import re
        shortened = re.sub(r'\s+', ' ', shortened).strip()

        # If still too long, truncate
        if len(shortened) > self.max_output_tokens * 4:
            shortened = shortened[:self.max_output_tokens * 4]

        self.summaries_created += 1
        return shortened

    def estimate_savings(self, original: str, summarized: str) -> dict:
        """Estimate token savings from summarization."""
        orig_tokens = max(1, len(original) // 4)
        sum_tokens = max(1, len(summarized) // 4)
        saved = orig_tokens - sum_tokens
        return dict(
            original_tokens=orig_tokens,
            summarized_tokens=sum_tokens,
            tokens_saved=saved,
            savings_pct=round(saved / orig_tokens * 100, 1) if orig_tokens > 0 else 0,
            cost_saved=round(saved * 0.03 / 1000, 6),
        )

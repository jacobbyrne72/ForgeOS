"""Compress model responses to reduce output token billing."""
from __future__ import annotations
import re

class ResponseCompressor:
    def __init__(self, max_output_tokens=4096):
        self.max_output_tokens = max_output_tokens
        self._compressions = 0
        self._tokens_saved = 0

    def compress(self, text, max_ratio=0.7):
        if not text or len(text.split()) <= 10:
            return text
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) <= 2:
            return text
        kept = sentences[:max(2, int(len(sentences) * max_ratio))]
        result = ' '.join(kept)
        self._compressions += 1
        self._tokens_saved += max(0, len(text.split()) - len(result.split()))
        return result

    def stats(self):
        return dict(
            compressions=self._compressions,
            tokens_saved=self._tokens_saved,
        )

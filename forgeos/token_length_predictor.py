"""Predict token count before API call to avoid overpaying for oversized inputs."""
from __future__ import annotations


class TokenLengthPredictor:
    """Estimate token count without calling the API."""

    def __init__(self, bytes_per_token: float = 4.0):
        self.bytes_per_token = bytes_per_token

    def predict_tokens(self, text: str) -> int:
        if not text:
            return 0
        # Use byte length / bytes_per_token as estimate
        return max(1, int(len(text.encode('utf-8')) / self.bytes_per_token))

    def is_undersized(self, text: str, min_tokens: int = 10) -> bool:
        return self.predict_tokens(text) < min_tokens

    def is_oversized(self, text: str, max_tokens: int = 4096) -> bool:
        return self.predict_tokens(text) > max_tokens

    def truncate_to_budget(self, text: str, max_tokens: int = 4096) -> str:
        """Truncate text to fit within token budget."""
        tokens = self.predict_tokens(text)
        if tokens <= max_tokens:
            return text
        # Calculate character budget
        char_budget = int(max_tokens * self.bytes_per_token)
        return text[:char_budget]

    def cost_estimate(self, text: str, cost_per_1k: float = 0.03) -> float:
        """Estimate cost for a text input."""
        tokens = self.predict_tokens(text)
        return round(tokens * cost_per_1k / 1000.0, 6)

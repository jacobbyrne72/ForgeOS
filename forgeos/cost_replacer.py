"""Cost replacer — replaces expensive model calls with cheaper equivalents.

Identifies patterns where expensive calls can be replaced
with local/cheaper alternatives, cutting per-operation cost.
"""
from __future__ import annotations

import re

# Patterns where replacements save cost
REPLACEMENTS = {
    "summarize": {
        "expensive": "Use GPT-4 to summarize this text",
        "cheap": "Use the summarizer function to condense this text",
        "savings_usd": 0.03,  # per replacement
        "saves_tokens": 800,
    },
    "translate": {
        "expensive": "Translate this text to Spanish",
        "cheap": "Use the local translator for this text",
        "savings_usd": 0.02,
        "saves_tokens": 600,
    },
    "extract": {
        "expensive": "Extract key entities from this text",
        "cheap": "Use regex extraction for key entities",
        "savings_usd": 0.015,
        "saves_tokens": 500,
    },
    "classify": {
        "expensive": "Classify this text into categories",
        "cheap": "Use keyword-based classification",
        "savings_usd": 0.01,
        "saves_tokens": 300,
    },
    "format": {
        "expensive": "Reformat this JSON for readability",
        "cheap": "Use json.dumps with indent parameter",
        "savings_usd": 0.005,
        "saves_tokens": 200,
    },
    "review": {
        "expensive": "Review this code for errors",
        "cheap": "Run syntax check + lint on this code",
        "savings_usd": 0.025,
        "saves_tokens": 700,
    },
}

class CostReplacer:
    def __init__(self):
        self.replacements_applied = 0
        self.total_saved_usd = 0.0
        self.total_saved_tokens = 0

    def scan(self, text: str) -> list[dict]:
        """Scan text for opportunities to replace expensive calls."""
        opportunities = []
        for name, rep in REPLACEMENTS.items():
            pattern = re.escape(rep["expensive"])
            if re.search(pattern, text, re.IGNORECASE):
                opportunities.append({
                    "type": name,
                    "original": rep["expensive"],
                    "replacement": rep["cheap"],
                    "savings_usd": rep["savings_usd"],
                    "saves_tokens": rep["saves_tokens"],
                })
        return opportunities

    def replace(self, text: str) -> tuple[str, list[dict]]:
        """Replace all expensive patterns in text with cheap equivalents.

        Returns (modified_text, list_of_replacements).
        """
        all_replacements = []
        modified = text

        for name, rep in REPLACEMENTS.items():
            if rep["expensive"].lower() in modified.lower():
                # Case-insensitive replace
                pat = re.compile(re.escape(rep["expensive"]), re.IGNORECASE)
                modified = pat.sub(rep["cheap"], modified)
                all_replacements.append({
                    "type": name,
                    "savings_usd": rep["savings_usd"],
                    "saves_tokens": rep["saves_tokens"],
                })
                self.replacements_applied += 1
                self.total_saved_usd += rep["savings_usd"]
                self.total_saved_tokens += rep["saves_tokens"]

        return modified, all_replacements

    def report(self) -> dict:
        return {
            "replacements_applied": self.replacements_applied,
            "total_saved_usd": round(self.total_saved_usd, 4),
            "total_saved_tokens": self.total_saved_tokens,
            "avg_savings_usd": round(
                self.total_saved_usd / max(1, self.replacements_applied), 4
            ),
        }

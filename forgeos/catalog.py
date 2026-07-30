"""Model catalog — pricing and capability for every model we can reach.

Seeded from the models.dev cache Hermes already maintains
(`~/.hermes/models_dev_cache.json`, ~140 providers). That file is 2.2MB, so it is
loaded from disk into memory and indexed — never pasted into a prompt.

This answers "what can we reach, what does it cost, what can it do" from data
rather than from assumption. Routing without it is guessing, and guessing is what
sends easy work to expensive models.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field

# Where a models.dev-shaped catalog might already exist. All optional: forgeos works
# with an empty catalog, it just cannot price a call it has never heard of. Set
# FORGEOS_MODEL_CATALOG to point at your own copy.
CATALOG_ENV = "FORGEOS_MODEL_CATALOG"

DEFAULT_CATALOG_PATHS = tuple(
    Path(os.path.expanduser(p))
    for p in (
        "~/.forgeos/models_dev_cache.json",
        "~/.cache/forgeos/models_dev_cache.json",
        # Populated by Hermes on machines that run it; harmless absent elsewhere.
        "~/.hermes/models_dev_cache.json",
        "~/.hermes/provider_models_cache.json",
    )
)


class ModelCard(BaseModel):
    """One model's measurable facts. No opinions, no vibes."""

    model_id: str
    provider: str
    name: str = ""
    family: str = ""
    input_cost_per_1m: float = 0.0
    output_cost_per_1m: float = 0.0
    # Cached input is typically ~10x cheaper than fresh input. Pricing it at the
    # fresh rate would make the single biggest saving in this harness read as zero,
    # so the discount has to be first-class in the cost model, not a footnote.
    cache_read_cost_per_1m: float = 0.0
    cache_write_cost_per_1m: float = 0.0
    context: int = 0
    max_output: int = 0
    reasoning: bool = False
    tool_call: bool = False
    attachment: bool = False
    open_weights: bool = False
    modalities_in: list[str] = Field(default_factory=list)
    modalities_out: list[str] = Field(default_factory=list)
    knowledge: str = ""

    @property
    def ref(self) -> str:
        return f"{self.provider}/{self.model_id}"

    @property
    def is_free(self) -> bool:
        return self.input_cost_per_1m == 0.0 and self.output_cost_per_1m == 0.0

    @property
    def cache_read_rate(self) -> float:
        """Cached-input price, falling back to the fresh rate when unpublished.

        Falling back to the FRESH rate is deliberate: it never understates cost, so
        a missing cache price can only make an estimate conservative, never make a
        budget check pass when it should have failed.
        """
        return self.cache_read_cost_per_1m or self.input_cost_per_1m

    @property
    def cache_discount(self) -> float:
        """Fraction saved per cached input token, 0.0 when there is no discount."""
        if not self.input_cost_per_1m:
            return 0.0
        return max(0.0, 1.0 - (self.cache_read_rate / self.input_cost_per_1m))

    def cost_micros(self, tokens_in: int, tokens_out: int, tokens_cached_in: int = 0) -> int:
        """Microdollars for a call of this shape. The preflight estimate.

        `tokens_in` is FRESH input; cached input is passed separately and priced at
        the cache-read rate.
        """
        usd = (
            (tokens_in / 1_000_000) * self.input_cost_per_1m
            + (tokens_cached_in / 1_000_000) * self.cache_read_rate
            + (tokens_out / 1_000_000) * self.output_cost_per_1m
        )
        return int(round(usd * 1_000_000))

    def fits(self, tokens_in: int) -> bool:
        """Whether the payload fits. A model that cannot hold the context is not a candidate."""
        return self.context == 0 or tokens_in <= self.context


def _parse_models_dev(raw: dict) -> list[ModelCard]:
    """Parse the models.dev shape: {provider_id: {models: {model_id: {...}}}}."""
    cards: list[ModelCard] = []
    for provider_id, pdata in raw.items():
        if not isinstance(pdata, dict):
            continue
        models = pdata.get("models")
        if not isinstance(models, dict):
            continue
        for model_id, m in models.items():
            if not isinstance(m, dict):
                continue
            cost = m.get("cost") or {}
            limit = m.get("limit") or {}
            mod = m.get("modalities") or {}
            cards.append(
                ModelCard(
                    model_id=model_id,
                    provider=provider_id,
                    name=m.get("name") or model_id,
                    family=m.get("family") or "",
                    # models.dev quotes cost per 1M tokens.
                    input_cost_per_1m=float(cost.get("input") or 0.0),
                    output_cost_per_1m=float(cost.get("output") or 0.0),
                    cache_read_cost_per_1m=float(cost.get("cache_read") or 0.0),
                    cache_write_cost_per_1m=float(cost.get("cache_write") or 0.0),
                    context=int(limit.get("context") or 0),
                    max_output=int(limit.get("output") or 0),
                    reasoning=bool(m.get("reasoning")),
                    tool_call=bool(m.get("tool_call")),
                    attachment=bool(m.get("attachment")),
                    open_weights=bool(m.get("open_weights")),
                    modalities_in=list(mod.get("input") or []),
                    modalities_out=list(mod.get("output") or []),
                    knowledge=str(m.get("knowledge") or ""),
                )
            )
    return cards


class Catalog:
    def __init__(self, cards: list[ModelCard] | None = None):
        self._cards: dict[str, ModelCard] = {}
        for c in cards or []:
            self._cards[c.ref] = c

    # ------------------------------------------------------------------ load

    @classmethod
    def from_file(cls, path: str | Path) -> Catalog:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(_parse_models_dev(raw))

    @classmethod
    def discover(cls, extra_paths: tuple[Path, ...] = ()) -> Catalog:
        """Load from whichever known cache exists. Empty catalog if none do.

        An empty catalog must never be a crash — forgeos has to work on a machine
        that has not run Hermes.
        """
        cards: list[ModelCard] = []
        for p in (*extra_paths, *DEFAULT_CATALOG_PATHS):
            try:
                if p.exists():
                    cards.extend(_parse_models_dev(json.loads(p.read_text(encoding="utf-8"))))
            except (json.JSONDecodeError, OSError):
                continue  # a corrupt cache must not take the harness down
        return cls(cards)

    # ----------------------------------------------------------------- query

    def __len__(self) -> int:
        return len(self._cards)

    def get(self, ref: str) -> ModelCard | None:
        return self._cards.get(ref)

    def all(self) -> list[ModelCard]:
        return list(self._cards.values())

    def providers(self) -> set[str]:
        return {c.provider for c in self._cards.values()}

    def free(self) -> list[ModelCard]:
        return [c for c in self._cards.values() if c.is_free]

    def find(
        self,
        *,
        providers: set[str] | None = None,
        needs_reasoning: bool = False,
        needs_tools: bool = False,
        min_context: int = 0,
        free_only: bool = False,
    ) -> list[ModelCard]:
        out = []
        for c in self._cards.values():
            if providers is not None and c.provider not in providers:
                continue
            if needs_reasoning and not c.reasoning:
                continue
            if needs_tools and not c.tool_call:
                continue
            if min_context and c.context and c.context < min_context:
                continue
            if free_only and not c.is_free:
                continue
            out.append(c)
        return out

    def cheapest(
        self,
        tokens_in: int = 4_000,
        tokens_out: int = 1_000,
        **filters,
    ) -> list[ModelCard]:
        """Candidates ordered by estimated cost for a call of this shape.

        Sorting by a realistic call shape rather than by headline input price —
        a model with cheap input and ruinous output is not actually cheap.
        """
        cands = [c for c in self.find(**filters) if c.fits(tokens_in)]
        cands.sort(key=lambda c: (c.cost_micros(tokens_in, tokens_out), -c.context))
        return cands


@lru_cache(maxsize=1)
def default_catalog() -> Catalog:
    """Process-wide catalog. Cached because parsing 2.2MB per call is waste."""
    return Catalog.discover()


__all__ = ["Catalog", "ModelCard", "default_catalog", "DEFAULT_CATALOG_PATHS"]


if __name__ == "__main__":  # pragma: no cover
    cat = default_catalog()
    print(f"{len(cat)} models across {len(cat.providers())} providers")
    print(f"free: {len(cat.free())}")
    print("\ncheapest that can reason + call tools, 8k in / 2k out:")
    for c in cat.cheapest(8_000, 2_000, needs_reasoning=True, needs_tools=True)[:8]:
        print(f"  {c.ref:52} {c.cost_micros(8_000, 2_000):>8} micros  ctx={c.context}")

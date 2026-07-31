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
import time
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field

# Where a models.dev- or litellm-shaped catalog might already exist. All optional:
# forgeos works with an empty catalog, it just cannot price a call it has never
# heard of. Set FORGEOS_MODEL_CATALOG to point at your own copy.
CATALOG_ENV = "FORGEOS_MODEL_CATALOG"

# Canonical upstream URL per supported price-table format. Single source of
# truth: `tools/refresh_catalog.py` fetches from these, and every `ModelCard`
# parsed from that shape is stamped with the matching one as `source_url`, so
# "where did this price come from" can never drift from "where would refresh
# actually fetch it from."
MODELS_DEV_URL = "https://models.dev/api.json"
LITELLM_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"

# `source` values that count as an automated, current fetch rather than a
# hand-entered guess. See `ModelCard.provenance_word`.
_MEASURED_SOURCES = frozenset({"models_dev", "litellm"})

DEFAULT_CATALOG_PATHS = tuple(
    Path(os.path.expanduser(p))
    for p in (
        "~/.forgeos/models_dev_cache.json",
        "~/.cache/forgeos/models_dev_cache.json",
        # Populated by Hermes on machines that run it; harmless absent elsewhere.
        "~/.hermes/models_dev_cache.json",
        "~/.hermes/provider_models_cache.json",
        # BerriAI/litellm's community-canonical, CI-synced pricing table. Last
        # on purpose: `discover()` merges every path in order and a later
        # entry overwrites an earlier one on a matching ref, so this is the
        # one that wins when it and an older models.dev cache both know a
        # model. See tools/refresh_catalog.py.
        "~/.forgeos/litellm_prices.json",
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
    # Provenance: where this price came from and how old it is. "" / 0.0 means
    # unstamped -- a hand-built card (tests, model_ranker.py) or a cache file
    # that did not go through Catalog.from_file/discover. An unstamped price
    # must read as unknown freshness, never as fresh; see `is_stale`.
    source: str = ""            # "litellm" | "models_dev" | "" (unstamped)
    source_url: str = ""        # the exact upstream URL `source` was fetched from
    fetched_at: float = 0.0

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

    @property
    def age_days(self) -> float | None:
        """Days since this price was fetched, or None if provenance is unknown."""
        if not self.fetched_at:
            return None
        return max(0.0, (time.time() - self.fetched_at) / 86400)

    def is_stale(self, max_age_days: float = 30.0) -> bool:
        """True if this price is older than `max_age_days`.

        An unstamped card (age unknown) also counts as stale -- a price this
        module cannot date must never be treated as fresh just because no one
        checked.
        """
        age = self.age_days
        return age is None or age > max_age_days

    @property
    def provenance_word(self) -> str:
        """"measured", "modelled", or "unknown" -- the same three-way honesty
        distinction `forgeos.economy.savings.Provenance` uses for a figure's
        evidence strength (that module imports FROM this one, so importing
        its enum back here would be circular; this mirrors the vocabulary
        without the import).

        A card is only MEASURED when it came from a known live source AND
        carries a fetch timestamp -- `source` set without `fetched_at` (or
        vice versa) is an inconsistent half-stamp, not a measurement. Any
        other named `source` (e.g. a future hand-entered fallback price) is
        MODELLED: someone typed a number in, nothing was fetched. No
        `source` at all is UNKNOWN.
        """
        if self.source in _MEASURED_SOURCES and self.fetched_at:
            return "measured"
        if self.source:
            return "modelled"
        return "unknown"


def _parse_models_dev(raw: dict, *, fetched_at: float = 0.0) -> list[ModelCard]:
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
                    source="models_dev",
                    source_url=MODELS_DEV_URL,
                    fetched_at=fetched_at,
                )
            )
    return cards


def _parse_litellm(raw: dict, *, fetched_at: float = 0.0) -> list[ModelCard]:
    """Parse BerriAI/litellm's `model_prices_and_context_window.json` shape:
    a flat `{model_name: {...}}` map, one `litellm_provider` field per entry,
    rather than models.dev's per-provider nesting.

    litellm quotes cost PER TOKEN, not per 1M like models.dev -- every cost
    field here is scaled by 1e6 on the way into a ModelCard so the two
    formats compare on the same unit everywhere else in the codebase (see
    `ModelCard.cost_micros`).
    """
    cards: list[ModelCard] = []
    for model_id, m in raw.items():
        if not isinstance(m, dict):
            continue
        provider = m.get("litellm_provider")
        if not provider:
            continue  # e.g. the "sample_spec" schema-documentation entry, not a real model
        cards.append(
            ModelCard(
                model_id=model_id,
                provider=str(provider),
                name=model_id,
                input_cost_per_1m=float(m.get("input_cost_per_token") or 0.0) * 1_000_000,
                output_cost_per_1m=float(m.get("output_cost_per_token") or 0.0) * 1_000_000,
                cache_read_cost_per_1m=float(m.get("cache_read_input_token_cost") or 0.0) * 1_000_000,
                cache_write_cost_per_1m=float(m.get("cache_creation_input_token_cost") or 0.0) * 1_000_000,
                context=int(m.get("max_input_tokens") or m.get("max_tokens") or 0),
                max_output=int(m.get("max_output_tokens") or 0),
                reasoning=bool(m.get("supports_reasoning")),
                tool_call=bool(m.get("supports_function_calling")),
                attachment=bool(m.get("supports_vision") or m.get("supports_pdf_input")),
                source="litellm",
                source_url=LITELLM_URL,
                fetched_at=fetched_at,
            )
        )
    return cards


def _looks_like_models_dev(raw: dict) -> bool:
    """models.dev nests models under a per-provider "models" key; litellm is a
    flat `{model_name: {...}}` map with no such nesting. Any provider-shaped
    entry is enough to tell the two apart.
    """
    return any(isinstance(v, dict) and isinstance(v.get("models"), dict) for v in raw.values())


def _parse_catalog_json(raw: dict, *, fetched_at: float = 0.0) -> list[ModelCard]:
    """Dispatch to the right parser by shape, so `from_file`/`discover` work on
    either a models.dev cache or a litellm `model_prices_and_context_window.json`
    without the caller having to say which.
    """
    if _looks_like_models_dev(raw):
        return _parse_models_dev(raw, fetched_at=fetched_at)
    return _parse_litellm(raw, fetched_at=fetched_at)


class Catalog:
    def __init__(self, cards: list[ModelCard] | None = None):
        self._cards: dict[str, ModelCard] = {}
        for c in cards or []:
            self._cards[c.ref] = c

    # ------------------------------------------------------------------ load

    @classmethod
    def from_file(cls, path: str | Path) -> Catalog:
        """Load and parse one cache file, stamping every card with its mtime.

        The file's own mtime IS the staleness stamp -- `tools/refresh_catalog.py`
        installs via an atomic rename, so mtime is "when this price table was
        fetched," not "when the file happened to be touched." No separate
        sidecar or metadata format needed to answer `card.age_days`.
        """
        path = Path(path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(_parse_catalog_json(raw, fetched_at=path.stat().st_mtime))

    @classmethod
    def discover(cls, extra_paths: tuple[Path, ...] = ()) -> Catalog:
        """Load from whichever known cache exists. Empty catalog if none do.

        An empty catalog must never be a crash — forgeos has to work on a machine
        that has not run Hermes. Later paths win on a matching ref (see
        `DEFAULT_CATALOG_PATHS`), each card keeping its own file's provenance.
        """
        cards: list[ModelCard] = []
        for p in (*extra_paths, *DEFAULT_CATALOG_PATHS):
            try:
                if p.exists():
                    raw = json.loads(p.read_text(encoding="utf-8"))
                    cards.extend(_parse_catalog_json(raw, fetched_at=p.stat().st_mtime))
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

    def stale(self, max_age_days: float = 30.0) -> list[ModelCard]:
        """Cards whose price provenance is older than `max_age_days`, or
        unstamped -- the check a caller (e.g. the ledger) makes before pricing
        a call against a number that might not be real anymore.
        """
        return [c for c in self._cards.values() if c.is_stale(max_age_days)]

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


__all__ = [
    "Catalog",
    "ModelCard",
    "default_catalog",
    "DEFAULT_CATALOG_PATHS",
    "LITELLM_URL",
    "MODELS_DEV_URL",
]


if __name__ == "__main__":  # pragma: no cover
    cat = default_catalog()
    print(f"{len(cat)} models across {len(cat.providers())} providers")
    print(f"free: {len(cat.free())}")
    print(f"stale (>30d old or unstamped): {len(cat.stale())}  "
          "-- run tools/refresh_catalog.py to update")
    print("\ncheapest that can reason + call tools, 8k in / 2k out:")
    for c in cat.cheapest(8_000, 2_000, needs_reasoning=True, needs_tools=True)[:8]:
        print(f"  {c.ref:52} {c.cost_micros(8_000, 2_000):>8} micros  ctx={c.context}")

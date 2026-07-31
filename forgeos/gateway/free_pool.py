"""Resolve a free-tier worker to models that actually exist and can be reached.

The registry ships a `gateway.free` worker pinned to `model="auto:free"` — a
placeholder that resolves to nothing. Meanwhile the catalogue lists 748 free
models with >=100k context (676 of them tool-calling). The cheapest possible call
is a free one, and none of them were reachable.

Three filters, each of which exists because skipping it costs real money:

1. **Only providers this machine can actually call.** A free NVIDIA model is not
   free if there is no NVIDIA credential — it is a guaranteed 401 plus a retry
   plus an escalation to a paid tier. Free-and-unreachable is worse than absent,
   because the router prefers it.
2. **Only models not already known dead.** Free-tier slugs rot constantly: this
   repo watched OpenRouter retire five `:free` models that every catalogue still
   listed at $0. `DeadModelStore` remembers them across runs so we stop paying a
   request to rediscover the same corpse.
3. **Ordered, not shuffled.** Deterministic ordering makes a run reproducible and
   makes a regression attributable. Random selection would turn "why did this
   cost more today" into an unanswerable question.

A free model is still a *request*, and requests have rate limits. This returns an
ordered candidate list, not one pick, so the caller can fall through — which is
what a free pool has to do anyway.
"""

from __future__ import annotations

from ..catalog import Catalog, ModelCard
from ..settings import ProviderKind, Settings

# Free tiers are throttled far harder than paid ones, so a free model with a tiny
# context window buys a truncated prompt and a retry. Below this it is not worth
# the round trip.
DEFAULT_MIN_CONTEXT = 100_000


def usable_providers(settings: Settings) -> set[str]:
    """Providers with a credential (or needing none) AND not disabled.

    `Provider.usable` already encodes enabled + installed + authenticated, which
    is exactly the question here — a provider failing any of those cannot serve a
    free model any more than a paid one.
    """
    return {
        p.name
        for p in settings.providers.values()
        if p.usable or p.kind is ProviderKind.LOCAL
    }


def free_candidates(
    catalog: Catalog,
    settings: Settings,
    *,
    min_context: int = DEFAULT_MIN_CONTEXT,
    needs_tools: bool = False,
    is_dead=None,
    limit: int = 24,
) -> list[ModelCard]:
    """Free models this machine could actually call, best first.

    `is_dead` is an optional `(model_ref) -> bool` — pass
    `DeadModelStore.is_dead`-shaped callable to skip slugs already proven gone.
    Left None only so this stays testable without a database.

    Ordering: larger context first (a free model's usual failure is running out
    of room), then tool-capable, then by ref for a stable tiebreak. Cost is not a
    tiebreak here because every candidate is $0 — ranking them by price would be
    sorting zeros and pretending it meant something.
    """
    allowed = usable_providers(settings)

    out: list[ModelCard] = []
    for card in catalog.all():
        if not card.is_free:
            continue
        if card.provider not in allowed:
            continue
        if card.context < min_context:
            continue
        if needs_tools and not card.tool_call:
            continue
        if is_dead is not None and is_dead(card.ref):
            continue
        out.append(card)

    out.sort(key=lambda m: (-m.context, not m.tool_call, m.ref))
    return out[:limit]


def resolve_free_ref(
    model_ref: str,
    catalog: Catalog,
    settings: Settings,
    **kw,
) -> list[str]:
    """Turn a placeholder like `auto:free` into real, ordered model refs.

    A concrete ref passes through untouched — a caller that named a specific
    model meant it, and silently substituting a different one would make the
    ledger's model attribution a lie.
    """
    if model_ref and not model_ref.endswith(":free") and model_ref != "auto:free":
        return [model_ref]
    return [c.ref for c in free_candidates(catalog, settings, **kw)]


__all__ = ["DEFAULT_MIN_CONTEXT", "free_candidates", "resolve_free_ref", "usable_providers"]

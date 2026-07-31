from __future__ import annotations

from forgeos.catalog import Catalog, ModelCard
from forgeos.gateway.client import Gateway
from forgeos.gateway.dead_models import DeadModelStore
from forgeos.gateway.free_pool import resolve_free_ref
from forgeos.ledger import Ledger
from forgeos.settings import AuthMode, Provider, ProviderKind, Settings


def _settings() -> Settings:
    return Settings(
        providers={
            "openrouter": Provider(
                name="openrouter",
                kind=ProviderKind.API,
                auth=AuthMode.NONE,
            )
        }
    )


def _card(model_id: str, *, context: int = 100_000) -> ModelCard:
    return ModelCard(
        model_id=model_id,
        provider="openrouter",
        input_cost_per_1m=0,
        output_cost_per_1m=0,
        context=context,
    )


def test_concrete_free_slug_is_not_replaced_by_the_pool():
    catalog = Catalog([_card("chosen:free"), _card("other:free", context=200_000)])

    assert resolve_free_ref(
        "openrouter/chosen:free", catalog, _settings()
    ) == ["openrouter/chosen:free"]


def test_gateway_resolves_auto_free_and_skips_persistently_dead_models():
    catalog = Catalog([_card("dead:free"), _card("live:free")])
    ledger = Ledger(":memory:")
    dead = DeadModelStore(":memory:")

    class Transport:
        name = "openrouter"
        serves = {"openrouter"}

    dead.mark_dead("openrouter", "openrouter/dead:free", reason="retired")
    gateway = Gateway(
        catalog=catalog,
        ledger=ledger,
        settings=_settings(),
        transports=[Transport()],
        dead_models=dead,
    )
    try:
        assert gateway.resolve_model_refs("auto:free") == ["openrouter/live:free"]
    finally:
        gateway.close()
        ledger.close()

"""Per-provider multi-key support: `forgeos/gateway/keyring.py`.

One key going bad (rate-limited, out of credit, revoked) must not take the
whole provider down when a second key is configured for it. These tests pin
the four things that matter about that: quarantine is per-key and the right
DURATION for the right failure (429 self-heals fast, 402 self-heals slow, 401
never self-heals this session), selection always rotates to a key that isn't
quarantined, selection is deterministic (no `random.choice` anywhere), and a
credential's VALUE never appears anywhere except the one accessor built to
hand it to a request header.
"""

from __future__ import annotations

import json

from forgeos.gateway.keyring import (
    DEFAULT_EXHAUSTED_BACKOFF_SECONDS,
    DEFAULT_RATE_LIMIT_BACKOFF_SECONDS,
    KeyRing,
    KeyState,
)
from forgeos.settings import AuthMode, Provider, ProviderKind


def _ring(env: dict[str, str], base: str = "OPENROUTER_API_KEY", t: float = 1_000.0) -> tuple[KeyRing, dict]:
    clock = {"t": t}
    ring = KeyRing("openrouter", base, environ=env, clock=lambda: clock["t"])
    return ring, clock


# ------------------------------------------------------------- discovery


def test_a_single_key_is_discovered_under_the_base_name():
    ring, _ = _ring({"OPENROUTER_API_KEY": "sk-one"})
    assert ring.names() == ["OPENROUTER_API_KEY"]
    assert len(ring) == 1


def test_numbered_keys_are_discovered_contiguously_from_2():
    ring, _ = _ring({
        "OPENROUTER_API_KEY": "sk-one",
        "OPENROUTER_API_KEY_2": "sk-two",
        "OPENROUTER_API_KEY_3": "sk-three",
    })
    assert ring.names() == ["OPENROUTER_API_KEY", "OPENROUTER_API_KEY_2", "OPENROUTER_API_KEY_3"]


def test_numbering_stops_at_the_first_gap():
    """BASE_4 must not be picked up if BASE_3 is missing -- the numbering is a
    priority order, not just a set of names to union together."""
    ring, _ = _ring({
        "OPENROUTER_API_KEY": "sk-one",
        "OPENROUTER_API_KEY_2": "sk-two",
        "OPENROUTER_API_KEY_4": "sk-four",  # gap at _3
    })
    assert ring.names() == ["OPENROUTER_API_KEY", "OPENROUTER_API_KEY_2"]


def test_numbered_keys_work_even_without_a_base_key():
    ring, _ = _ring({"OPENROUTER_API_KEY_2": "sk-two"})
    assert ring.names() == ["OPENROUTER_API_KEY_2"]


def test_a_comma_separated_base_value_becomes_multiple_synthetic_slots():
    ring, _ = _ring({"OPENROUTER_API_KEY": "sk-a, sk-b,sk-c"})
    assert ring.names() == ["OPENROUTER_API_KEY[0]", "OPENROUTER_API_KEY[1]", "OPENROUTER_API_KEY[2]"]
    assert ring.reveal("OPENROUTER_API_KEY[1]") == "sk-b"


def test_comma_list_and_numbered_vars_combine_in_order():
    ring, _ = _ring({
        "OPENROUTER_API_KEY": "sk-a,sk-b",
        "OPENROUTER_API_KEY_2": "sk-c",
    })
    assert ring.names() == ["OPENROUTER_API_KEY[0]", "OPENROUTER_API_KEY[1]", "OPENROUTER_API_KEY_2"]


def test_an_unconfigured_provider_discovers_no_keys():
    ring, _ = _ring({})
    assert ring.names() == []
    assert ring.usable is False
    assert ring.select() is None


def test_for_provider_uses_the_providers_env_key_as_the_base():
    provider = Provider(
        name="openrouter", kind=ProviderKind.API, auth=AuthMode.API_KEY,
        env_key="OPENROUTER_API_KEY",
    )
    ring = KeyRing.for_provider(provider, environ={"OPENROUTER_API_KEY": "sk-one"})
    assert ring.provider == "openrouter"
    assert ring.names() == ["OPENROUTER_API_KEY"]


def test_for_provider_rejects_a_provider_with_no_env_key():
    provider = Provider(name="ollama", kind=ProviderKind.LOCAL, auth=AuthMode.NONE)
    try:
        KeyRing.for_provider(provider)
    except ValueError as exc:
        assert "ollama" in str(exc)
    else:
        raise AssertionError("expected ValueError")


# --------------------------------------------------------- rotation on 429


def test_selection_starts_with_the_first_key_in_priority_order():
    ring, _ = _ring({"OPENROUTER_API_KEY": "sk-one", "OPENROUTER_API_KEY_2": "sk-two"})
    assert ring.select() == "OPENROUTER_API_KEY"


def test_a_rate_limited_key_is_skipped_in_favour_of_the_next_healthy_one():
    ring, clock = _ring({"OPENROUTER_API_KEY": "sk-one", "OPENROUTER_API_KEY_2": "sk-two"})

    state = ring.record_failure("OPENROUTER_API_KEY", 429, retry_after_seconds=60.0)
    assert state is KeyState.RATE_LIMITED

    assert ring.select() == "OPENROUTER_API_KEY_2"


def test_the_rate_limited_key_comes_back_once_its_window_passes():
    ring, clock = _ring({"OPENROUTER_API_KEY": "sk-one", "OPENROUTER_API_KEY_2": "sk-two"})
    ring.record_failure("OPENROUTER_API_KEY", 429, retry_after_seconds=60.0)
    assert ring.select() == "OPENROUTER_API_KEY_2"

    clock["t"] += 60.0
    # Priority order is fixed: once key 1 is healthy again it is chosen over
    # key 2 again, exactly as if it had never failed.
    assert ring.select() == "OPENROUTER_API_KEY"


def test_a_third_key_is_used_once_the_first_two_are_both_rate_limited():
    ring, _ = _ring({
        "OPENROUTER_API_KEY": "sk-one",
        "OPENROUTER_API_KEY_2": "sk-two",
        "OPENROUTER_API_KEY_3": "sk-three",
    })
    ring.record_failure("OPENROUTER_API_KEY", 429, retry_after_seconds=60.0)
    ring.record_failure("OPENROUTER_API_KEY_2", 429, retry_after_seconds=60.0)
    assert ring.select() == "OPENROUTER_API_KEY_3"


def test_a_provider_given_retry_after_is_honoured_exactly_not_escalated():
    """429 is busy, not broken -- a provider that says 'retry in 5s' should be
    retried in 5s, not backed off to the module's own 30s default."""
    ring, clock = _ring({"OPENROUTER_API_KEY": "sk-one"})
    ring.record_failure("OPENROUTER_API_KEY", 429, retry_after_seconds=5.0)

    clock["t"] += 4.999
    assert ring.select() is None
    clock["t"] += 0.002
    assert ring.select() == "OPENROUTER_API_KEY"


def test_repeated_429s_without_a_provider_hint_back_off_exponentially():
    ring, clock = _ring({"OPENROUTER_API_KEY": "sk-one"})

    ring.record_failure("OPENROUTER_API_KEY", 429)
    first_window = ring.snapshot()[0].quarantined_until - clock["t"]
    assert first_window == DEFAULT_RATE_LIMIT_BACKOFF_SECONDS

    clock["t"] += first_window  # let it come back, then fail again
    ring.record_failure("OPENROUTER_API_KEY", 429)
    second_window = ring.snapshot()[0].quarantined_until - clock["t"]
    assert second_window == DEFAULT_RATE_LIMIT_BACKOFF_SECONDS * 2


def test_a_success_clears_the_quarantine_and_the_backoff_escalation():
    ring, clock = _ring({"OPENROUTER_API_KEY": "sk-one"})
    ring.record_failure("OPENROUTER_API_KEY", 429)
    assert ring.select() is None

    ring.record_success("OPENROUTER_API_KEY")
    assert ring.select() == "OPENROUTER_API_KEY"
    assert ring.snapshot()[0].state is KeyState.HEALTHY
    assert ring.snapshot()[0].backoff_seconds == 0.0


# ------------------------------------------------------- permanent quarantine (401/403)


def test_401_quarantines_a_key_permanently_for_the_session():
    ring, clock = _ring({"OPENROUTER_API_KEY": "sk-one", "OPENROUTER_API_KEY_2": "sk-two"})
    state = ring.record_failure("OPENROUTER_API_KEY", 401)
    assert state is KeyState.INVALID

    assert ring.select() == "OPENROUTER_API_KEY_2"
    clock["t"] += 10_000_000  # no amount of waiting revives an invalid key
    assert ring.select() == "OPENROUTER_API_KEY_2"
    assert ring.snapshot()[0].available(clock["t"]) is False


def test_403_is_also_permanent_quarantine():
    ring, _ = _ring({"OPENROUTER_API_KEY": "sk-one"})
    state = ring.record_failure("OPENROUTER_API_KEY", 403)
    assert state is KeyState.INVALID
    assert ring.usable is False


# --------------------------------------------------------------- 402: exhausted


def test_402_quarantines_long_not_like_a_rate_limit():
    ring, clock = _ring({"OPENROUTER_API_KEY": "sk-one"})
    state = ring.record_failure("OPENROUTER_API_KEY", 402)
    assert state is KeyState.EXHAUSTED

    window = ring.snapshot()[0].quarantined_until - clock["t"]
    assert window == DEFAULT_EXHAUSTED_BACKOFF_SECONDS
    assert window > DEFAULT_RATE_LIMIT_BACKOFF_SECONDS * 10  # not a few seconds

    # Does not fix itself in seconds.
    clock["t"] += 5.0
    assert ring.select() is None


def test_402_eventually_self_heals_unlike_401():
    ring, clock = _ring({"OPENROUTER_API_KEY": "sk-one"})
    ring.record_failure("OPENROUTER_API_KEY", 402)
    clock["t"] += DEFAULT_EXHAUSTED_BACKOFF_SECONDS
    assert ring.select() == "OPENROUTER_API_KEY"


# ------------------------------------------------------- all keys exhausted


def test_all_keys_quarantined_makes_the_provider_report_unusable():
    ring, _ = _ring({"OPENROUTER_API_KEY": "sk-one", "OPENROUTER_API_KEY_2": "sk-two"})
    ring.record_failure("OPENROUTER_API_KEY", 401)
    ring.record_failure("OPENROUTER_API_KEY_2", 402)

    assert ring.usable is False
    assert ring.select() is None
    assert ring.select_and_reveal() is None


def test_a_provider_with_one_healthy_key_among_several_dead_ones_stays_usable():
    ring, _ = _ring({
        "OPENROUTER_API_KEY": "sk-one",
        "OPENROUTER_API_KEY_2": "sk-two",
        "OPENROUTER_API_KEY_3": "sk-three",
    })
    ring.record_failure("OPENROUTER_API_KEY", 401)
    ring.record_failure("OPENROUTER_API_KEY_2", 429, retry_after_seconds=60.0)

    assert ring.usable is True
    assert ring.select() == "OPENROUTER_API_KEY_3"


# --------------------------------------------------- unclassified failures


def test_an_unrelated_http_error_does_not_change_the_keys_state():
    """A 500 or a network blip says nothing about whether THIS key is good --
    it is a transport problem, not a credential one (the same split
    `dead_models.py` already draws between a model problem and a transport
    problem)."""
    ring, _ = _ring({"OPENROUTER_API_KEY": "sk-one"})
    state = ring.record_failure("OPENROUTER_API_KEY", 500)
    assert state is KeyState.HEALTHY
    assert ring.select() == "OPENROUTER_API_KEY"


# ------------------------------------------------------------- determinism


def test_selection_is_deterministic_given_the_same_state():
    """No random.choice anywhere: two rings built from identical env/state
    must make the identical choice, repeatedly."""
    env = {"OPENROUTER_API_KEY": "sk-one", "OPENROUTER_API_KEY_2": "sk-two",
           "OPENROUTER_API_KEY_3": "sk-three"}
    ring_a, _ = _ring(env)
    ring_b, _ = _ring(env)

    for ring in (ring_a, ring_b):
        ring.record_failure("OPENROUTER_API_KEY", 429, retry_after_seconds=60.0)

    picks_a = [ring_a.select() for _ in range(10)]
    picks_b = [ring_b.select() for _ in range(10)]
    assert picks_a == picks_b
    assert len(set(picks_a)) == 1
    assert picks_a[0] == "OPENROUTER_API_KEY_2"


# ------------------------------------------------------------------ secrets


def test_the_key_value_never_appears_in_repr_status_or_snapshot():
    secret = "sk-do-not-leak-me-0123456789"
    ring, _ = _ring({"OPENROUTER_API_KEY": secret, "OPENROUTER_API_KEY_2": "sk-two"})
    ring.record_failure("OPENROUTER_API_KEY", 401, reason="credential rejected")
    ring.record_failure("OPENROUTER_API_KEY_2", 429, retry_after_seconds=30.0, reason="slow down")

    blob = "".join([
        repr(ring),
        json.dumps(ring.status()),
        json.dumps([r.model_dump(mode="json") for r in ring.snapshot()]),
        json.dumps(ring.names()),
    ])
    assert secret not in blob
    assert "sk-two" not in blob


def test_reveal_is_the_only_way_to_get_a_value_back():
    secret = "sk-do-not-leak-me-0123456789"
    ring, _ = _ring({"OPENROUTER_API_KEY": secret})
    assert ring.reveal("OPENROUTER_API_KEY") == secret
    picked = ring.select_and_reveal()
    assert picked == ("OPENROUTER_API_KEY", secret)


def test_a_key_record_pydantic_model_has_no_field_that_can_hold_a_value():
    """Structural guarantee, not just a behavioural one: nothing on the public
    state model even has a slot for a secret to be assigned into by mistake."""
    from forgeos.gateway.keyring import KeyRecord

    fields = set(KeyRecord.model_fields)
    assert fields == {
        "name", "state", "quarantined_until", "reason",
        "consecutive_failures", "backoff_seconds", "checked_at",
    }

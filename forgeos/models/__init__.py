"""Model-trait registry — how to prompt each model for the countermeasure its
own failure mode needs.

A trait is a CLAIM about a model's behavior, not a fact: it carries
`TraitEvidence` (MEASURED / DOCUMENTED / OBSERVED) the same way
`forgeos.knowledge.claims` labels outside claims, and for the same reason -- an
unverified impression that quietly becomes a permanent hard rule is how a
harness mis-prompts a model forever with no diff to revert. `record_outcome`
is the unlearning path: this harness's own contradicting results downgrade or
remove a trait, they never just pile on top of it.
"""

from .traits import (
    BENCHMARK_SHAPED_PATTERN,
    SEED_MODEL_PROFILES,
    ModelProfile,
    PromptDiscipline,
    Tendency,
    Trait,
    TraitEvidence,
    all_profiles,
    discipline_for,
    discipline_instructions,
    get_profile,
    instructions_token_estimate,
    record_outcome,
    register_profile,
    reset_registry,
)

__all__ = [
    "BENCHMARK_SHAPED_PATTERN",
    "SEED_MODEL_PROFILES",
    "ModelProfile",
    "PromptDiscipline",
    "Tendency",
    "Trait",
    "TraitEvidence",
    "all_profiles",
    "discipline_for",
    "discipline_instructions",
    "get_profile",
    "instructions_token_estimate",
    "record_outcome",
    "register_profile",
    "reset_registry",
]

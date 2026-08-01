"""Prompt layer: byte-stable prefixes for cache hits, one per Role, plus the
Model-Native Prompt Compiler (`PromptIR` + per-family `render`).

See `prefix.py` for the mechanism (StablePrefix, build_prompt, PrefixRegistry),
`roles.py` for the concrete text bound to each Role, `ir.py` for the
provider-neutral prompt semantics, and `renderers.py` for the per-vendor
house-style rendering built on top of it.
"""

from __future__ import annotations

from forgeos.prompts.ir import (
    Authority,
    ContextIR,
    Execution,
    Job,
    Mission,
    Output,
    PromptIR,
    Verification,
    ir_from_task,
)
from forgeos.prompts.prefix import PrefixRegistry, Prompt, StablePrefix, build_prompt
from forgeos.prompts.renderers import RenderedPrompt, render, resolve_family
from forgeos.prompts.roles import PREFIX_VERSION, ROLE_PREFIXES, default_prefix_registry, role_prefix

__all__ = [
    "PREFIX_VERSION",
    "ROLE_PREFIXES",
    "Authority",
    "ContextIR",
    "Execution",
    "Job",
    "Mission",
    "Output",
    "PrefixRegistry",
    "Prompt",
    "PromptIR",
    "RenderedPrompt",
    "StablePrefix",
    "Verification",
    "build_prompt",
    "default_prefix_registry",
    "ir_from_task",
    "render",
    "resolve_family",
    "role_prefix",
]

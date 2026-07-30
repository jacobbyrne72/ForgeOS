"""Prompt layer: byte-stable prefixes for cache hits, one per Role.

See `prefix.py` for the mechanism (StablePrefix, build_prompt, PrefixRegistry)
and `roles.py` for the concrete text bound to each Role.
"""

from __future__ import annotations

from hive.prompts.prefix import PrefixRegistry, Prompt, StablePrefix, build_prompt
from hive.prompts.roles import PREFIX_VERSION, ROLE_PREFIXES, default_prefix_registry, role_prefix

__all__ = [
    "PREFIX_VERSION",
    "ROLE_PREFIXES",
    "PrefixRegistry",
    "Prompt",
    "StablePrefix",
    "build_prompt",
    "default_prefix_registry",
    "role_prefix",
]

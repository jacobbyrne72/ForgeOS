"""Deterministic control plane.

Everything in this package is ordinary code that consumes zero model tokens.
That is deliberate: every scheduling, budgeting and classification decision moved
out of a model is a permanent saving, not a one-off. The LLM manager sits *below*
this layer and is woken only for genuine ambiguity.
"""

from .governor import (
    DEFAULT_LOOP_THRESHOLD,
    FOCUS_AT,
    NARROW_AT,
    Action,
    Decision,
    Governor,
    LoopSignal,
    Remedy,
)

__all__ = [
    "Action",
    "DEFAULT_LOOP_THRESHOLD",
    "Decision",
    "FOCUS_AT",
    "Governor",
    "LoopSignal",
    "NARROW_AT",
    "Remedy",
]

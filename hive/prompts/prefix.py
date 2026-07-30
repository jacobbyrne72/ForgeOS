"""Byte-stable prompt prefixes and the cache-hit guarantee.

Cached input tokens cost roughly 10% of fresh ones, but a provider only serves
a cache hit when the prompt PREFIX is byte-identical to a previous call. So a
prompt has two parts:

    STABLE PREFIX   never varies for a given (role, version) -- role contract,
                     tool protocol, output schema, safety policy, evidence
                     requirements.
    VOLATILE TAIL    varies per task -- the specific task, its context
                     manifest, prior blocker.

Dynamic compression of the PREFIX is actively harmful: it changes bytes every
call and destroys the cache, so an 89% compression saving replaces a 90% cache
saving and comes out behind. Compress only the tail.

A prefix must be pure ASCII-safe deterministic text: no timestamps, no random
ids, no dict-ordering dependence, nothing that varies between processes or
between two builds in the same process. `fingerprint` exists to catch drift
mechanically instead of by review.
"""

from __future__ import annotations

import hashlib

from pydantic import BaseModel, field_validator

from hive.settings import Role

# No tokenizer dependency here on purpose -- this module has no business
# calling out to a provider just to size a string. ~4 chars/token is the
# standard rough approximation for English text: good enough to size a
# prefix for a sanity check, not to bill it.
_CHARS_PER_TOKEN = 4


class StablePrefix(BaseModel):
    """The part of a prompt that must be byte-identical across calls.

    Identity is `(role, version)`. A new version is a new cache lineage --
    the correct way to change what a role says is to bump `version`, never to
    edit `text` under a version number that has already shipped.
    """

    role: Role
    version: int
    text: str

    @field_validator("text")
    @classmethod
    def _ascii_and_nonblank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("prefix text must not be blank")
        if not v.isascii():
            raise ValueError("prefix text must be ASCII-safe")
        return v

    @field_validator("version")
    @classmethod
    def _positive_version(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("version must be > 0")
        return v

    @property
    def fingerprint(self) -> str:
        """sha256 of the exact bytes. Two builds of the same prefix must match,
        in this process and in a fresh one -- that is the whole point of using
        a content hash instead of e.g. Python's salted `hash()`.
        """
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    @property
    def token_estimate(self) -> int:
        return max(1, len(self.text) // _CHARS_PER_TOKEN)

    @property
    def key(self) -> tuple[Role, int]:
        return (self.role, self.version)


class Prompt(BaseModel):
    """A built prompt: stable prefix then volatile tail, in that order.

    `cacheable_prefix_len` is the byte length of `full_text` a provider can
    serve from cache. It is exactly `len(prefix text)` because the tail is
    appended immediately after it and never overlaps or reorders it.
    """

    full_text: str
    prefix_fingerprint: str
    cacheable_prefix_len: int


def build_prompt(prefix: StablePrefix, tail: str) -> Prompt:
    """Concatenate prefix then tail, prefix first, unmodified, every call.

    That ordering -- and only ever appending, never editing, the prefix --
    is the entire cache-hit guarantee. Changing a single byte of the prefix
    for one call and not the next turns every call into a cache miss.
    """
    return Prompt(
        full_text=prefix.text + tail,
        prefix_fingerprint=prefix.fingerprint,
        cacheable_prefix_len=len(prefix.text),
    )


class PrefixRegistry:
    """Registered `(role, version)` -> `StablePrefix`, with drift detection.

    Silent prefix drift -- the same `(role, version)` key quietly pointing at
    different text between two calls -- is the exact failure this class
    exists to catch. Bumping the version is the correct way to change a
    prefix; `verify_stability` is the guard against doing it by accident.
    """

    def __init__(self) -> None:
        self._prefixes: dict[tuple[Role, int], StablePrefix] = {}

    def register(self, prefix: StablePrefix) -> None:
        self._prefixes[prefix.key] = prefix

    def get(self, role: Role, version: int) -> StablePrefix | None:
        return self._prefixes.get((role, version))

    def verify_stability(self, role: Role, version: int, text: str) -> bool:
        """False iff `(role, version)` is already registered with different text.

        An unregistered `(role, version)` is not drift -- there is nothing to
        compare against yet, so this returns True. Only a registered key whose
        text has changed underneath it is drift.
        """
        existing = self.get(role, version)
        if existing is None:
            return True
        return existing.text == text


__all__ = [
    "PrefixRegistry",
    "Prompt",
    "StablePrefix",
    "build_prompt",
]

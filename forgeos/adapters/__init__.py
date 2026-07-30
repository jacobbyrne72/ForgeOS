"""One execution backend per module.

`base.WorkerAdapter` is the contract every backend implements; everything
else in this package is a concrete backend behind it. Importing this package
must never require any particular backend's SDK to be installed — each
adapter module only imports its own dependency lazily, inside the methods
that actually need it.
"""

from __future__ import annotations

from .acp import ACPAdapter
from .base import EventKind, WorkerAdapter, WorkerCapabilities, WorkerEvent, WorkerUsage

__all__ = [
    "ACPAdapter",
    "EventKind",
    "WorkerAdapter",
    "WorkerCapabilities",
    "WorkerEvent",
    "WorkerUsage",
]

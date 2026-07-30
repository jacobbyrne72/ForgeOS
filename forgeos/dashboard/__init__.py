"""forgeos.dashboard — read-only operator view over the ledger, event log,
avoidance log and lease store, plus one write action: a manual per-job halt
flag (a flag, never a delete, never a budget edit).

Binds 127.0.0.1 only. There is no auth layer here, so a non-loopback bind
would expose every job's cost data -- and the halt endpoint -- to the network.
"""

from .app import HOST, PORT, create_app

__all__ = ["HOST", "PORT", "create_app"]

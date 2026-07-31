"""Compatibility entry point for :mod:`forgeos.forgebench_table`."""

from __future__ import annotations

import sys
from pathlib import Path

# Running this file directly puts ``tools/`` on sys.path, not the repository
# root. Add the root explicitly so the same implementation works from a fresh
# checkout without requiring an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forgeos.forgebench_table import main


if __name__ == "__main__":
    raise SystemExit(main())

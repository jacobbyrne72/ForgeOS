#!/usr/bin/env python
"""PreToolUse gate. Exit 2 = hard deny.

Wire into Claude Code settings.json:

    {"hooks": {"PreToolUse": [{"matcher": "Read",
      "hooks": [{"type": "command",
                 "command": "python C:/Users/byrne/Downloads/forgeos/hooks/ripper_gate.py"}]}]}}

Reads the tool-call payload as JSON on stdin. Prints the reason and the exact
replacement command on stderr, which is what the agent sees — a denial without a
next step just gets worked around, so the suggestion is the important half.

Fails OPEN on any internal error. A broken gate must not become an outage: blocking
every read because this script has a bug would be worse than the waste it prevents.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            return 0
        payload = json.loads(raw)
    except Exception:
        return 0  # unreadable payload is not the agent's fault

    try:
        from forgeos.policy import check_tool_call

        tool_name = payload.get("tool_name") or payload.get("toolName") or ""
        tool_input = payload.get("tool_input") or payload.get("toolInput") or {}
        decision = check_tool_call(str(tool_name), dict(tool_input))
    except Exception as exc:  # fail open, but say so
        print(f"ripper_gate: internal error, allowing ({type(exc).__name__})", file=sys.stderr)
        return 0

    if decision.allowed:
        return 0

    print(f"DENIED: {decision.reason}", file=sys.stderr)
    if decision.suggestion:
        print(f"\nDo this instead:\n{decision.suggestion}", file=sys.stderr)
    print(
        "\n(A read with offset/limit always passes — name the range you need.)",
        file=sys.stderr,
    )
    return decision.exit_code


if __name__ == "__main__":
    raise SystemExit(main())

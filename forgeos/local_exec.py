"""Local executor — runs code locally when safe, avoiding API costs.

Avoids API calls entirely for code that can be executed locally.
Uses AST analysis to detect unsafe operations before execution.
"""
from __future__ import annotations

import ast
import builtins
import io
from contextlib import redirect_stdout, redirect_stderr
from .cost_router import CostRouter


SAFE_FUNCTIONS = {
    "len", "range", "enumerate", "zip", "map", "filter",
    "sum", "min", "max", "sorted", "reversed", "abs",
    "int", "float", "str", "bool", "list", "dict", "tuple",
    "set", "print", "round", "isinstance", "type",
}

SAFE_IMPORTS = {"json", "re", "datetime", "os", "sys"}
SAFE_ATTRIBUTES: set[str] = set()


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Import only the small standard-library allowlist used by local code."""
    if level != 0 or name.split(".")[0] not in SAFE_IMPORTS:
        raise ImportError(f"local import is not allowed: {name}")
    return builtins.__import__(name, globals, locals, fromlist, level)


class LocalExecutor:
    """Execute code locally when safe, avoiding API costs entirely."""

    def __init__(self):
        self.executor = CostRouter()
        self.local_calls = 0
        self.api_calls = 0
        self.local_savings = 0.0

    def can_execute_locally(self, code: str) -> tuple[bool, str]:
        """Check if code can be safely executed locally."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False, "syntax_error"

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id not in SAFE_FUNCTIONS:
                        return False, "unsafe_function:" + node.func.id
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr not in SAFE_ATTRIBUTES:
                        return False, "unsafe_attribute:" + node.func.attr
                else:
                    return False, "unsafe_callable_expression"
            if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name.split(".")[0] for a in node.names]
                else:
                    names = [node.module] if node.module else []
                for name in names:
                    if name not in SAFE_IMPORTS:
                        return False, "unsafe_import:" + str(name)

        return True, "safe"

    def execute(self, code: str) -> dict:
        """Try local execution first. Returns cost and output info."""
        can_exec, reason = self.can_execute_locally(code)

        if can_exec:
            self.local_calls += 1
            try:
                stdout_capture = io.StringIO()
                stderr_capture = io.StringIO()
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    safe_builtins = {
                        name: getattr(builtins, name) for name in SAFE_FUNCTIONS
                    }
                    safe_builtins["__import__"] = _safe_import
                    exec(
                        compile(code, "<local>", "exec"),
                        {"__builtins__": safe_builtins},
                    )
                return {
                    "executed_locally": True,
                    "cost": 0.0,
                    "output": stdout_capture.getvalue(),
                    "error": None,
                    "reason": "executed_locally",
                }
            except Exception as e:
                return {
                    "executed_locally": False,
                    "cost": 0.0,
                    "output": None,
                    "error": str(e),
                    "reason": "local_failed:" + str(e),
                }
        else:
            self.api_calls += 1
            return {
                "executed_locally": False,
                "cost": 0.03,
                "output": None,
                "error": None,
                "reason": "deferred_to_api:" + reason,
            }

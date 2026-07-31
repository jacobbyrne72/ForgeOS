"""Every registered subcommand must actually run something.

This file exists because of a specific failure. `main()` built its `dispatch`
dict and then ended — the `return dispatch[args.command](args)` had drifted
outside the function, where it was unreachable. So every `forge` command parsed
its arguments, did nothing at all, and **exited 0**.

`forge doctor` printed nothing and reported success. The README called it a
"Full CLI" and "production-hardened". A green exit code with nothing behind it
is the precise failure this whole project is built to prevent, and it shipped in
the project's own front door.

Argument parsing is not execution, and an exit code is not evidence. These tests
check the wiring structurally — no subprocess, no real work — so the class of bug
cannot come back silently as more commands are added.
"""

from __future__ import annotations

import ast
import inspect
import pathlib

import pytest

from forgeos import cli

CLI_SRC = pathlib.Path(inspect.getfile(cli)).read_text(encoding="utf-8")
TREE = ast.parse(CLI_SRC)


def _main_node() -> ast.FunctionDef:
    for node in TREE.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            return node
    raise AssertionError("cli.main() not found")


def _registered_subcommands() -> set[str]:
    """Subcommand names passed to `sub.add_parser("name", ...)`."""
    names = set()
    for node in ast.walk(_main_node()):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_parser"
                and node.args
                and isinstance(node.args[0], ast.Constant)):
            names.add(node.args[0].value)
    return names


def _dispatch_keys() -> set[str]:
    """Keys of the `dispatch = {...}` literal inside main()."""
    for node in ast.walk(_main_node()):
        if (isinstance(node, ast.Assign)
                and any(getattr(t, "id", "") == "dispatch" for t in node.targets)
                and isinstance(node.value, ast.Dict)):
            return {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
    raise AssertionError("dispatch dict not found in main()")


# ------------------------------------------------------- the bug that shipped


def test_main_actually_calls_a_handler():
    """The regression itself. main() must END by invoking something, not by
    building a table and falling off the end."""
    main = _main_node()
    returns = [n for n in ast.walk(main) if isinstance(n, ast.Return)]
    calling = [
        r for r in returns
        if isinstance(r.value, ast.Call)
        and not (isinstance(r.value.func, ast.Attribute)
                 and r.value.func.attr in {"print_help"})
    ]
    assert calling, (
        "main() never calls a handler — every command would parse, do nothing, "
        "and exit 0"
    )


def test_no_unreachable_dispatch_outside_main():
    """The orphaned `return dispatch[...]` sat at module level after another
    function, where Python never reached it."""
    for node in TREE.body:
        if isinstance(node, ast.FunctionDef) and node.name != "main":
            for inner in ast.walk(node):
                if (isinstance(inner, ast.Subscript)
                        and isinstance(inner.value, ast.Name)
                        and inner.value.id == "dispatch"):
                    raise AssertionError(
                        f"dispatch[...] referenced inside {node.name}() — "
                        "that is the orphaned-return bug returning"
                    )


# --------------------------------------------------------------- completeness


@pytest.mark.parametrize("name", sorted(_registered_subcommands()))
def test_every_registered_subcommand_has_a_handler(name):
    """A command in --help that cannot run is worse than an absent one: the user
    reasonably believes it worked."""
    assert name in _dispatch_keys(), (
        f"'forge {name}' is registered in --help but has no dispatch entry"
    )


@pytest.mark.parametrize("name", sorted(_dispatch_keys()))
def test_every_dispatch_entry_is_reachable(name):
    """The mirror image: a handler nobody can invoke is dead weight that reads
    as a feature."""
    assert name in _registered_subcommands(), (
        f"dispatch has '{name}' but no subparser registers it — unreachable"
    )


@pytest.mark.parametrize("name", sorted(_dispatch_keys()))
def test_every_handler_exists_and_is_callable(name):
    for node in ast.walk(_main_node()):
        if (isinstance(node, ast.Assign)
                and any(getattr(t, "id", "") == "dispatch" for t in node.targets)
                and isinstance(node.value, ast.Dict)):
            for k, v in zip(node.value.keys, node.value.values):
                if isinstance(k, ast.Constant) and k.value == name:
                    fn = getattr(cli, v.id, None)
                    assert callable(fn), f"{v.id} is not a callable in cli.py"
                    return
    raise AssertionError(f"handler for {name} not found")


# -------------------------------------------------------------- the cost flag


def test_run_defines_the_budget_flag_it_reads():
    """`cmd_run` read `args.budget` behind a hasattr() guard while no parser
    defined it — so a cost-governed harness had no way to cap a run's spend, and
    the guard hid it."""
    assert "--budget" in CLI_SRC, "forge run must expose a spend ceiling"
    src = inspect.getsource(cli.cmd_run)
    if "args.budget" in src:
        main_src = inspect.getsource(cli.main)
        assert '"--budget"' in main_src or "'--budget'" in main_src, (
            "cmd_run reads args.budget but main() never registers the flag"
        )


def test_run_delegates_to_guarded_team_runner(monkeypatch):
    """The public ``run`` verb must execute the same budget/review path as
    ``python -m forgeos team`` rather than merely printing a projection."""
    from forgeos import __main__ as runtime_cli

    captured = {}

    def fake_run_team(objective, **kwargs):
        captured["objective"] = objective
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(runtime_cli, "_run_team", fake_run_team)
    assert cli.main([
        "run", "add a retry helper", "--cwd", "repo", "--budget-usd", "2.5",
        "--state-dir", "state", "--dry-run",
    ]) == 0
    assert captured == {
        "objective": "add a retry helper",
        "cwd": "repo",
        "budget_usd": 2.5,
        "state_dir": "state",
        "dry_run": True,
    }


def test_forgebench_forwards_json_receipt_path(monkeypatch):
    from types import SimpleNamespace
    from forgeos import forgebench

    captured = {}

    def fake_main(argv):
        captured["argv"] = argv
        return 0

    monkeypatch.setattr(forgebench, "main", fake_main)
    args = SimpleNamespace(
        model="p/m", budget_usd=1.25, dry_run=True, skip_baseline=False,
        ledger_path=":memory:", json_out="receipt.json",
    )

    assert cli.cmd_forgebench(args) == 0
    assert "--json-out" in captured["argv"]
    assert captured["argv"][captured["argv"].index("--json-out") + 1] == "receipt.json"


def test_fleet_is_safe_on_windows_cp1252_console(monkeypatch, capsys):
    """The fleet screenshot must not crash on the default Windows console."""
    from types import SimpleNamespace

    from forgeos.settings import AuthMode, ProviderKind, Settings

    fake = SimpleNamespace(providers={
        "ollama": SimpleNamespace(
            name="ollama", kind=ProviderKind.LOCAL, auth=AuthMode.NONE, enabled=True,
        ),
        "claude": SimpleNamespace(
            name="claude", kind=ProviderKind.CLI, auth=AuthMode.SUBSCRIPTION, enabled=True,
        ),
    })
    monkeypatch.setattr(Settings, "load", classmethod(lambda cls: fake))

    assert cli.main(["fleet"]) == 0
    output = capsys.readouterr().out
    output.encode("cp1252")
    assert "->" in output


def test_unknown_command_fails_loudly():
    """An unhandled command must not exit 0. Silence reading as success is the
    whole bug."""
    src = inspect.getsource(cli.main)
    assert "return 2" in src or "raise" in src, (
        "main() must return non-zero for a command it cannot handle"
    )

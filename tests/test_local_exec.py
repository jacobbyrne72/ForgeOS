from forgeos.local_exec import LocalExecutor


def test_local_executor_runs_allowlisted_code():
    result = LocalExecutor().execute("print(sum([1, 2, 3]))")

    assert result["executed_locally"] is True
    assert result["output"] == "6\n"


def test_local_executor_rejects_implicit_builtins_file_access():
    executor = LocalExecutor()

    allowed, reason = executor.can_execute_locally("__builtins__.open('secret.txt')")

    assert allowed is False
    assert reason == "unsafe_attribute:open"


def test_local_executor_rejects_indirect_callable_bypass():
    executor = LocalExecutor()

    allowed, reason = executor.can_execute_locally(
        "(__builtins__['open'])('secret.txt')"
    )

    assert allowed is False
    assert reason == "unsafe_callable_expression"

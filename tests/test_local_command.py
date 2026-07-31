import asyncio
import sys

from forgeos.adapters.factory import build_adapter
from forgeos.adapters.local_command import LocalCommandAdapter
from forgeos.registry import Adapter, CostTier, WorkerProfile


def _run(coro):
    return asyncio.run(coro)


def test_local_command_adapter_delivers_prompt_and_cleans_up():
    adapter = LocalCommandAdapter(
        sys.executable,
        ["-c", "import sys; print(sys.stdin.read().upper(), end='')"],
    )
    session_id = _run(adapter.start("task-1", ".", "local"))
    events = _run(_collect(adapter.send(session_id, "hello")))

    assert [event.kind.value for event in events] == ["tool_call", "tool_update", "done"]
    assert events[0].tool_kind == "execute"
    assert events[1].text == "HELLO"
    _run(adapter.close(session_id))


def test_local_command_checkpoint_contains_configuration_but_not_prompt():
    adapter = LocalCommandAdapter("python", ["-V"])
    session_id = _run(adapter.start("task-1", ".", "local"))
    checkpoint = _run(adapter.checkpoint(session_id))

    assert checkpoint["task_id"] == "task-1"
    assert checkpoint["command"] == "python"
    assert "prompt" not in checkpoint
    _run(adapter.close(session_id))


def test_factory_builds_local_command_profile_without_health_probe():
    profile = WorkerProfile(
        worker_id="local.python",
        adapter=Adapter.LOCAL_COMMAND,
        command=sys.executable,
        args=["-V"],
        tier=CostTier.FREE,
    )
    adapter, reason = build_adapter(profile, check_health=False)

    assert isinstance(adapter, LocalCommandAdapter)
    assert reason == "local command adapter built"


async def _collect(events):
    return [event async for event in events]

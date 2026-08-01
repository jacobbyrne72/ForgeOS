from forgeos.recovery import RECOVERY_SCHEMA, build_recovery


def _snapshot(*, jobs=None, queue=None):
    return {
        "schema": "forgeos.dashboard_snapshot.v1",
        "captured_at": 123.0,
        "jobs": {"jobs": jobs or []},
        "queue": queue or {},
    }


def test_recovery_is_clear_when_no_persisted_work_needs_attention(tmp_path):
    report = build_recovery(_snapshot(), state_dir=tmp_path)

    assert report["schema"] == RECOVERY_SCHEMA
    assert report["status"] == "clear"
    assert report["action_count"] == 0
    assert report["next_action"] is None
    assert report["summary"] == {
        "unfinished_jobs": 0,
        "unfinished_tasks": 0,
        "halted_jobs": 0,
        "stale_queue": False,
    }


def test_recovery_recommends_confirmation_gated_resume_for_halted_job(tmp_path):
    report = build_recovery(
        _snapshot(jobs=[{
            "id": "job-123",
            "objective": "finish the operator surface",
            "task_count": 3,
            "task_counts_by_state": {"done": 1, "running": 2},
            "halted": True,
        }]),
        state_dir=tmp_path,
    )

    action = report["next_action"]
    assert report["status"] == "attention"
    assert report["summary"]["unfinished_tasks"] == 2
    assert action["kind"] == "resume_halted_job"
    assert action["severity"] == "blocked"
    assert action["requires_operator_confirmation"] is True
    assert action["argv"] == [
        "python", "-m", "forgeos", "resume", "job-123",
        "--state-dir", str(tmp_path),
    ]


def test_recovery_surfaces_stale_queue_without_recommending_restart(tmp_path):
    report = build_recovery(
        _snapshot(queue={
            "queue_dir": str(tmp_path / "queue"),
            "heartbeat_valid": True,
            "stale": True,
            "owner_active": False,
        }),
        state_dir=tmp_path,
    )

    action = report["next_action"]
    assert action["kind"] == "inspect_stale_queue"
    assert action["argv"][-1] == "--json"
    assert action["requires_operator_confirmation"] is False
    assert "watch" not in action["argv"]

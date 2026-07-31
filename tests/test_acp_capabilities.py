from types import SimpleNamespace

from forgeos.adapters.acp import _supports_resume_session


def test_acp_resume_capability_accepts_an_empty_capability_object():
    init = SimpleNamespace(
        agent_capabilities=SimpleNamespace(
            session_capabilities=SimpleNamespace(resume={}),
        ),
    )

    assert _supports_resume_session(init)


def test_acp_resume_capability_is_false_when_omitted():
    init = SimpleNamespace(agent_capabilities=SimpleNamespace())

    assert not _supports_resume_session(init)

"""The dashboard chat bar: intent in, digest-gated execution out.

The property under test is the same one `manager_chat` enforces everywhere
else: NOTHING EXECUTES WITHOUT AN APPROVAL BOUND TO THE EXACT PLAN THAT WAS
SHOWN. The browser surface adds one more attacker to the model -- any web page
the operator visits can POST to 127.0.0.1 -- and the digest is what defeats it:
a cross-origin page cannot read the propose response, so it never learns the
digest, so it can never run anything.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from forgeos.dashboard.app import create_app


@pytest.fixture
def client(tmp_path):
    app = create_app(tmp_path / "state")
    with TestClient(app) as c:
        yield c


def _propose(client, objective="add a retry helper", max_usd=1.0):
    resp = client.post("/api/chat/propose",
                       json={"objective": objective, "max_usd": max_usd})
    assert resp.status_code == 200, resp.text
    return resp.json()


# ------------------------------------------------------------- the spend gate


def test_nothing_runs_without_a_proposal(client):
    resp = client.post("/api/chat/run", json={"digest": "0" * 16})
    assert resp.status_code == 409


def test_a_wrong_digest_never_runs(client):
    """The blind-POST attacker: it can call propose, it cannot read the
    response, so the best it has is a guess."""
    _propose(client)
    resp = client.post("/api/chat/run", json={"digest": "f" * 16})
    assert resp.status_code == 403


def test_an_empty_digest_never_runs(client):
    _propose(client)
    assert client.post("/api/chat/run", json={"digest": ""}).status_code in (403, 422)


def test_the_shown_digest_runs_a_dry_preview(client):
    proposal = _propose(client)
    resp = client.post("/api/chat/run",
                       json={"digest": proposal["digest"], "dry_run": True})
    assert resp.status_code == 200
    body = resp.json()
    assert body["dry_run"] is True and body["ran"] is False
    assert isinstance(body["tasks"], list) and body["tasks"], "a preview with no tasks"


def test_a_new_proposal_voids_the_old_digest(client):
    """Approve-then-swap: get a digest for a cheap plan, propose an expensive
    one, replay the old digest. The approval must not carry."""
    first = _propose(client, objective="cheap thing", max_usd=0.10)
    _propose(client, objective="expensive thing", max_usd=25.0)
    resp = client.post("/api/chat/run", json={"digest": first["digest"]})
    assert resp.status_code == 403


def test_the_rendered_blueprint_shows_what_approval_means(client):
    body = _propose(client, max_usd=2.5)
    for needed in ("GOAL", "CEILING", "DIGEST"):
        assert needed in body["rendered"], f"a human cannot approve without {needed}"
    assert body["digest"] in body["rendered"], "the digest shown must be the one accepted"


# ----------------------------------------------------------------- input guard


def test_an_empty_objective_is_refused(client):
    resp = client.post("/api/chat/propose", json={"objective": "   "})
    assert resp.status_code == 422


def test_an_absurd_budget_is_refused_at_the_door(client):
    """The chat bar must not be the one place a $10,000 cap can be typed."""
    assert client.post("/api/chat/propose",
                       json={"objective": "x", "max_usd": 10_000}).status_code == 422
    assert client.post("/api/chat/propose",
                       json={"objective": "x", "max_usd": 0}).status_code == 422


def test_the_budget_that_runs_is_the_one_that_was_approved(client):
    """A second request cannot widen what was shown. The run reads the cap from
    the approved contract, and the run body has no budget field at all."""
    proposal = _propose(client, max_usd=0.25)
    resp = client.post("/api/chat/run",
                       json={"digest": proposal["digest"], "dry_run": True,
                             "budget_usd": 99.0})  # ignored: not part of the model
    assert resp.status_code == 200

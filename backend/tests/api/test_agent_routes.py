"""Tests for the agent propose/approve routes.

The routes own the HTTP contract: request validation, the
error-kind to status-code mapping, and the stateless plan+evidence
round trip. The planner flow itself is owned by the planner tests
and is faked here by monkeypatching the names agent.py imported.

A minimal FastAPI app mounts only the agent router with inert
dependency overrides. Building the real app would run the lifespan,
which eagerly constructs both transports; none of that machinery is
under test here.
"""

from __future__ import annotations

from typing import Any

import pytest
from app.api import agent
from app.api.deps import (
    get_agent_error_recorder,
    get_deepseek_transport,
    get_embedder,
    get_playwright_transport,
)
from app.core.db import get_db
from app.schemas.agent_plan import (
    Evidence,
    MarkForRevisionStep,
    Plan,
    PlanProposal,
)
from app.schemas.tools import MarkForRevisionInput
from app.services.agent_error_recorder import NoOpAgentErrorRecorder
from app.services.agent_orchestrator import AgentOrchestratorError
from app.services.agent_planner import PlannerErrorKind, PlannerServiceError
from fastapi import FastAPI
from fastapi.testclient import TestClient

TARGET_PATH = "Python > Data Types > Integers"


def _proposal() -> PlanProposal:
    """One grounded single-step proposal, the shape propose returns."""
    plan = Plan(steps=[MarkForRevisionStep(args=MarkForRevisionInput(path=TARGET_PATH))])
    evidence = [Evidence(tool="get_weak_topics", result={"topics": [{"path": TARGET_PATH}]})]
    return PlanProposal(plan=plan, evidence=evidence)


@pytest.fixture
def client() -> TestClient:
    """A test client over a minimal app carrying only the agent router.

    Every dependency is overridden with an inert stand-in: the
    service functions are monkeypatched per test, so nothing behind
    the dependencies is ever reached.
    """
    app = FastAPI()
    app.include_router(agent.router, prefix="/api")
    app.dependency_overrides[get_db] = lambda: None
    app.dependency_overrides[get_playwright_transport] = object
    app.dependency_overrides[get_deepseek_transport] = object
    app.dependency_overrides[get_embedder] = object
    app.dependency_overrides[get_agent_error_recorder] = NoOpAgentErrorRecorder
    return TestClient(app)


def test_propose_returns_plan_and_evidence(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    proposal = _proposal()

    async def fake_propose(**kwargs: Any) -> PlanProposal:
        return proposal

    monkeypatch.setattr(agent, "propose_plan", fake_propose)

    response = client.post("/api/agent/propose", json={"transport_kind": "deepseek"})

    assert response.status_code == 200
    body = response.json()
    assert body["plan"]["steps"][0]["tool"] == "mark_for_revision"
    assert body["plan"]["steps"][0]["args"]["path"] == TARGET_PATH
    assert body["evidence"][0]["tool"] == "get_weak_topics"


@pytest.mark.parametrize(
    ("kind", "expected_status"),
    [
        ("no_data", 422),
        ("transport_failed", 502),
        ("parse_failed", 502),
        ("disallowed_tool", 502),
        ("ungrounded", 502),
        ("tool_handler_failed", 500),
        ("unexpected", 500),
    ],
)
def test_propose_maps_error_kind_to_status(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    kind: PlannerErrorKind,
    expected_status: int,
) -> None:
    async def fake_propose(**kwargs: Any) -> PlanProposal:
        raise PlannerServiceError(f"{kind} occurred", kind=kind)

    monkeypatch.setattr(agent, "propose_plan", fake_propose)

    response = client.post("/api/agent/propose", json={"transport_kind": "deepseek"})

    assert response.status_code == expected_status
    assert response.json()["detail"] == f"{kind} occurred"


def test_propose_rejects_unknown_transport_kind(client: TestClient) -> None:
    response = client.post("/api/agent/propose", json={"transport_kind": "carrier_pigeon"})

    assert response.status_code == 422


def test_approve_executes_and_returns_204(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    async def fake_approve(**kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(agent, "approve_plan", fake_approve)
    proposal = _proposal()

    response = client.post("/api/agent/approve", json=proposal.model_dump(mode="json"))

    assert response.status_code == 204
    assert response.content == b""
    assert captured["plan"] == proposal.plan
    assert captured["evidence"] == proposal.evidence


def test_approve_maps_ungrounded_to_502(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_approve(**kwargs: Any) -> None:
        raise PlannerServiceError("plan is not grounded", kind="ungrounded")

    monkeypatch.setattr(agent, "approve_plan", fake_approve)

    response = client.post("/api/agent/approve", json=_proposal().model_dump(mode="json"))

    assert response.status_code == 502
    assert response.json()["detail"] == "plan is not grounded"


def test_approve_maps_orchestrator_failure_to_500(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_approve(**kwargs: Any) -> None:
        raise AgentOrchestratorError("mutate step failed")

    monkeypatch.setattr(agent, "approve_plan", fake_approve)

    response = client.post("/api/agent/approve", json=_proposal().model_dump(mode="json"))

    assert response.status_code == 500
    assert response.json()["detail"] == "mutate step failed"


def test_approve_rejects_unknown_tool_at_validation(client: TestClient) -> None:
    """The closed step vocabulary holds at the wire: an unknown tool
    is unconstructable, so it dies in request validation before the
    handler runs."""
    body = {
        "plan": {"steps": [{"kind": "mutate", "tool": "drop_tables", "args": {}}]},
        "evidence": [],
    }

    response = client.post("/api/agent/approve", json=body)

    assert response.status_code == 422

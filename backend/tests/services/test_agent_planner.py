"""Tests for agent_planner: propose and approve.

FakeTransport drives the planner loop without a real LLM. The flow
mirrors the diagnostic service (open chat, tool-call loop, terminal
parse, close) with two planner-specific behaviors under test: the
read allowlist gates registry dispatch, and get_weak_topics results
are retained as Evidence that grounds the plan.

The propose flow's no_data guard runs the real get_weak_topics, so
the happy paths seed a committed Session plus a LearnedItem with an
INCORRECT verdict. session_id is NOT NULL on learned_item, so the
seed scaffolds a Session first. The seeded topic path, the
get_weak_topics result path, and the PLAN path the fake LLM emits
are one constant per test, since grounding requires the plan target
to appear in the gathered evidence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from app.models import (
    Domain,
    DomainKind,
    LearnedItem,
    Session,
    Topic,
    TopicStatus,
    TransportKind,
)
from app.models.enums import GradingVerdict, LearningMode, SessionState
from app.schemas.agent_plan import Evidence, MarkForRevisionStep, Plan, PlanProposal
from app.schemas.tools import (
    CreateDomainCall,
    CreateDomainInput,
    GetWeakTopicsCall,
    GetWeakTopicsInput,
    GetWeakTopicsOutput,
    WeakTopicInfo,
)
from app.services.agent_error_recorder import NoOpAgentErrorRecorder
from app.services.agent_orchestrator import AgentOrchestratorError
from app.services.agent_planner import (
    PlannerServiceError,
    approve_plan,
    propose_plan,
)
from app.transport.base import TransportError, TransportResponse

from tests.services.fakes import FakeEmbedder, FakeTransport

if TYPE_CHECKING:
    from sqlalchemy.orm import Session as DbSession

WEAK_PATH = "Python > Data Types > Integers"
OTHER_PATH = "Python > Async > Tasks"


def _seed_weak_topic(db: DbSession, path: str = WEAK_PATH) -> None:
    """Seed a topic with one INCORRECT graded item so it reads as weak.

    get_weak_topics inner-joins LearnedItem to Topic on topic_id,
    keeps verdicts that are not null, and surfaces topics with a
    non-zero weakness score. learned_item.session_id is NOT NULL, so
    a committed Session is scaffolded first and the items point at it.

    Two INCORRECT items, not one: the planner's no_data guard reads
    with min_attempts=1, but the LLM's own get_weak_topics call uses
    the tool default min_attempts=2. A single item clears the guard
    yet falls below the LLM read's threshold, leaving the plan
    ungrounded. Two items clear both, so the topic the guard found is
    the same topic the evidence contains. Committing makes this
    durable state both reads see.
    """
    domain = Domain(name=path.split(" > ", 1)[0], kind=DomainKind.LANGUAGE, description=None)
    topic = Topic(
        path=path,
        domain=path.split(" > ", 1)[0],
        name=path.rsplit(" > ", 1)[-1],
        status=TopicStatus.LEARNED,
    )
    db.add(domain)
    db.add(topic)
    db.flush()

    session = Session(
        topic_id=topic.id,
        mode_used=LearningMode.FLASHCARD,
        state=SessionState.COMPLETED,
        transport_kind=TransportKind.DEEPSEEK,
    )
    db.add(session)
    db.flush()

    for question, given in (("What is 7 // 2?", "3.5"), ("What is 7 % 2?", "0")):
        db.add(
            LearnedItem(
                session_id=session.id,
                topic_id=topic.id,
                question=question,
                answer="correct",
                your_answer=given,
                mode=LearningMode.FLASHCARD,
                grading_verdict=GradingVerdict.INCORRECT,
            )
        )
    db.commit()


def _status_of(db: DbSession, path: str) -> TopicStatus:
    """Read one topic's current status, asserting it exists."""
    return db.query(Topic).filter(Topic.path == path).one().status


def _plan_response(path: str) -> str:
    """A terminal PLAN marking one path for revision."""
    return (
        f'---PLAN---\n[{{"tool": "mark_for_revision", "args": {{"path": "{path}"}}}}]\n---END---\n'
    )


def _tool_call_response() -> str:
    """A TOOL_CALL block requesting get_weak_topics."""
    return '---TOOL_CALL---\n{"name": "get_weak_topics", "args": {}}\n---END---\n'


def _evidence_for(path: str) -> list[Evidence]:
    """Evidence as approve receives it: one get_weak_topics result with path."""
    output = GetWeakTopicsOutput(
        topics=[
            WeakTopicInfo(
                topic_path=path,
                incorrect_count=1,
                partial_count=0,
                correct_count=0,
                samples=[],
            )
        ]
    )
    return [Evidence(tool="get_weak_topics", result=output.model_dump(mode="json"))]


# ---------- propose: happy path ----------


async def test_propose_tool_call_then_plan_returns_grounded_proposal(db: DbSession) -> None:
    """LLM calls get_weak_topics, then emits a plan targeting that path.

    The full propose flow: guard passes, chat opens, the tool call
    runs and its result is kept as evidence, the terminal plan parses,
    and groundedness passes because the plan path is in the evidence.
    """
    _seed_weak_topic(db)
    transport = FakeTransport([_tool_call_response(), _plan_response(WEAK_PATH)])

    proposal = await propose_plan(
        db=db,
        transport=transport,
        embedder=FakeEmbedder(),
        transport_kind=TransportKind.DEEPSEEK,
    )

    assert isinstance(proposal, PlanProposal)
    assert len(proposal.plan.steps) == 1
    step = proposal.plan.steps[0]
    assert isinstance(step, MarkForRevisionStep)
    assert step.args.path == WEAK_PATH
    # The tool result was retained as evidence.
    assert len(proposal.evidence) == 1
    assert proposal.evidence[0].tool == "get_weak_topics"
    # Propose mutates nothing: the topic keeps its committed status.
    assert _status_of(db, WEAK_PATH) is TopicStatus.LEARNED
    # One chat opened, one tool result sent back.
    assert len(transport.chats) == 1
    assert len(transport.chats[0].tool_results_received) == 1
    # The chat advertises exactly the allowlisted surface, so the
    # transport can never offer a tool the propose gate would reject.
    assert transport.chats[0].tool_names == ("get_weak_topics",)


async def test_propose_native_tool_calls_then_plan(db: DbSession) -> None:
    """DeepSeek-style native tool_calls populate evidence the same way."""
    _seed_weak_topic(db)
    call = GetWeakTopicsCall(args=GetWeakTopicsInput(), id="call_1")
    transport = FakeTransport(
        [TransportResponse(text="", tool_calls=[call]), _plan_response(WEAK_PATH)]
    )

    proposal = await propose_plan(
        db=db,
        transport=transport,
        embedder=FakeEmbedder(),
        transport_kind=TransportKind.DEEPSEEK,
    )

    assert len(proposal.evidence) == 1
    step = proposal.plan.steps[0]
    assert isinstance(step, MarkForRevisionStep)
    assert step.args.path == WEAK_PATH


# ---------- propose: no_data guard ----------


async def test_propose_no_weak_topics_raises_no_data_before_transport(db: DbSession) -> None:
    """Falsifying test: no weak topics → guard fires → transport never called.

    A bare topic with no graded items yields no weak topics, so the
    guard must fire before the chat opens. If chats is non-empty, the
    guard ran too late.
    """
    domain = Domain(name="Python", kind=DomainKind.LANGUAGE, description=None)
    topic = Topic(path=WEAK_PATH, domain="Python", name="Integers", status=TopicStatus.LEARNED)
    db.add(domain)
    db.add(topic)
    db.commit()

    transport = FakeTransport([_plan_response(WEAK_PATH)])

    with pytest.raises(PlannerServiceError) as exc_info:
        await propose_plan(
            db=db,
            transport=transport,
            embedder=FakeEmbedder(),
            transport_kind=TransportKind.DEEPSEEK,
        )

    assert exc_info.value.kind == "no_data"
    assert len(transport.chats) == 0


async def test_propose_empty_db_raises_no_data(db: DbSession) -> None:
    """No topics at all: guard fires, no transport call."""
    transport = FakeTransport([_plan_response(WEAK_PATH)])

    with pytest.raises(PlannerServiceError) as exc_info:
        await propose_plan(
            db=db,
            transport=transport,
            embedder=FakeEmbedder(),
            transport_kind=TransportKind.DEEPSEEK,
        )

    assert exc_info.value.kind == "no_data"
    assert len(transport.chats) == 0


# ---------- propose: disallowed tool ----------


async def test_propose_disallowed_tool_call_raises(db: DbSession) -> None:
    """A registry tool outside the allowlist is rejected before dispatch.

    create_domain is a real teaching-surface tool that commits its own
    writes. The planner must not let the LLM reach it: an unapproved
    write during the propose loop violates the mutate-after-approval
    contract. The allowlist gates it before execute_tool_call runs, so
    no domain is created.
    """
    _seed_weak_topic(db)
    bad_call = CreateDomainCall(
        args=CreateDomainInput(name="Injected", kind=DomainKind.LANGUAGE),
        id="call_x",
    )
    transport = FakeTransport(
        [TransportResponse(text="", tool_calls=[bad_call]), _plan_response(WEAK_PATH)]
    )

    with pytest.raises(PlannerServiceError) as exc_info:
        await propose_plan(
            db=db,
            transport=transport,
            embedder=FakeEmbedder(),
            transport_kind=TransportKind.DEEPSEEK,
        )

    assert exc_info.value.kind == "disallowed_tool"
    # The disallowed tool never executed: no Injected domain landed.
    assert db.query(Domain).filter(Domain.name == "Injected").one_or_none() is None


# ---------- propose: ungrounded plan ----------


async def test_propose_plan_targeting_unevidenced_path_raises_ungrounded(db: DbSession) -> None:
    """A plan path absent from the gathered evidence fails grounding.

    The LLM reads weak topics (which contain WEAK_PATH) but emits a
    plan targeting OTHER_PATH. The groundedness guard rejects it: the
    plan must target a path that appeared in the evidence.
    """
    _seed_weak_topic(db)
    transport = FakeTransport([_tool_call_response(), _plan_response(OTHER_PATH)])

    with pytest.raises(PlannerServiceError) as exc_info:
        await propose_plan(
            db=db,
            transport=transport,
            embedder=FakeEmbedder(),
            transport_kind=TransportKind.DEEPSEEK,
        )

    assert exc_info.value.kind == "ungrounded"


async def test_propose_plan_with_no_tool_call_is_ungrounded(db: DbSession) -> None:
    """A plan emitted with no prior tool call has empty evidence, so ungrounded.

    The LLM skips the tool call and jumps straight to a plan. Evidence
    is empty, so no path can be grounded, so any plan fails the guard.
    This makes "call the tool first" structural rather than a prompt
    promise.
    """
    _seed_weak_topic(db)
    transport = FakeTransport([_plan_response(WEAK_PATH)])

    with pytest.raises(PlannerServiceError) as exc_info:
        await propose_plan(
            db=db,
            transport=transport,
            embedder=FakeEmbedder(),
            transport_kind=TransportKind.DEEPSEEK,
        )

    assert exc_info.value.kind == "ungrounded"


# ---------- propose: transport and parse failures ----------


async def test_propose_transport_failure_on_start_raises(db: DbSession) -> None:
    """Transport fails opening the chat."""
    _seed_weak_topic(db)
    transport = FakeTransport([], raise_on_send=TransportError("boom"))

    with pytest.raises(PlannerServiceError) as exc_info:
        await propose_plan(
            db=db,
            transport=transport,
            embedder=FakeEmbedder(),
            transport_kind=TransportKind.DEEPSEEK,
        )

    assert exc_info.value.kind == "transport_failed"


async def test_propose_unparseable_terminal_raises_parse_failed(db: DbSession) -> None:
    """LLM returns a teaching turn instead of a plan: parse failure.

    parse_plan_response only accepts TOOL_CALL or PLAN, so a TOPIC
    turn dies inside it. This is why the planner has no
    wrong_response_kind: a wrong terminal is unconstructable as
    anything but a parse failure.
    """
    _seed_weak_topic(db)
    turn = "---TOPIC---\nx\n---DIFFICULTY---\nbeginner\n---END---\n"
    transport = FakeTransport([turn])

    with pytest.raises(PlannerServiceError) as exc_info:
        await propose_plan(
            db=db,
            transport=transport,
            embedder=FakeEmbedder(),
            transport_kind=TransportKind.DEEPSEEK,
        )

    assert exc_info.value.kind == "parse_failed"


# ---------- approve ----------


async def test_approve_executes_grounded_plan(db: DbSession) -> None:
    """approve re-checks grounding then commits the mutation.

    The plan targets WEAK_PATH, the evidence contains it, the topic
    exists. After approval the topic flips to needs_revision and a
    fresh read confirms the commit.
    """
    _seed_weak_topic(db)
    plan = Plan(steps=[MarkForRevisionStep.model_validate({"args": {"path": WEAK_PATH}})])

    await approve_plan(
        db=db,
        recorder=NoOpAgentErrorRecorder(),
        plan=plan,
        evidence=_evidence_for(WEAK_PATH),
    )

    assert _status_of(db, WEAK_PATH) is TopicStatus.NEEDS_REVISION


async def test_approve_ungrounded_plan_raises_before_execution(db: DbSession) -> None:
    """approve rejects a plan whose target is not in the evidence.

    The plan targets WEAK_PATH but the evidence grounds only
    OTHER_PATH. The guard fires before run_plan, so the topic keeps
    its committed status: nothing executed.
    """
    _seed_weak_topic(db)
    plan = Plan(steps=[MarkForRevisionStep.model_validate({"args": {"path": WEAK_PATH}})])

    with pytest.raises(PlannerServiceError) as exc_info:
        await approve_plan(
            db=db,
            recorder=NoOpAgentErrorRecorder(),
            plan=plan,
            evidence=_evidence_for(OTHER_PATH),
        )

    assert exc_info.value.kind == "ungrounded"
    assert _status_of(db, WEAK_PATH) is TopicStatus.LEARNED


async def test_approve_grounded_plan_for_absent_topic_raises_orchestrator_error(
    db: DbSession,
) -> None:
    """Evidence grounds a path, but the topic was deleted before approve.

    Grounding passes (the path is in the evidence), but the strict
    mutate core raises because the topic no longer exists. The
    orchestrator rolls back and raises AgentOrchestratorError, which
    propose_plan deliberately does not wrap: a mutation failure is the
    orchestrator's contract, mapped separately by the route. This pins
    the seam between the stateless groundedness guard (evidence-based)
    and the core's existence check (DB-based).
    """
    # Evidence references a path with no matching committed topic.
    ghost = "Ghost > Deleted > Topic"
    plan = Plan(steps=[MarkForRevisionStep.model_validate({"args": {"path": ghost}})])

    with pytest.raises(AgentOrchestratorError):
        await approve_plan(
            db=db,
            recorder=NoOpAgentErrorRecorder(),
            plan=plan,
            evidence=_evidence_for(ghost),
        )

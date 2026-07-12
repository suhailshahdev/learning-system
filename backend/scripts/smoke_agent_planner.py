"""Smoke test for the bounded planner's propose/approve round trip.

Exercises propose_plan and approve_plan end to end against a real
LLM transport and real Postgres: the LLM reads weak topics through
the tool-call loop, emits a mutate-only plan grounded in that
evidence, and the approved plan's mutations commit atomically.

Seeds two weak topics under a recognizable prefix: topic rows plus a
synthetic completed session and two INCORRECT learned items each.
Two items per topic, not one: the planner's no_data guard reads with
min_attempts=1 but the LLM's own get_weak_topics call uses the tool
default min_attempts=2, so two items clear both thresholds.

Hard checks are the mechanical invariants: the plan is non-empty,
mutate-only, and every target appears in the gathered evidence;
approve flips seeded targets to NEEDS_REVISION; a tampered plan
whose target is absent from the evidence is rejected as ungrounded
with no database change. Which weak topic the LLM picks is its call
and is reported for eyeballing, not asserted: a real database can
hold other legitimately weak topics.

Approve safety on a shared database: only smoke-prefixed targets are
approved. If the plan targets other topics, those steps are reported
and left unexecuted so the smoke never mutates real data.

By default runs against DeepSeek. Pass --transport=playwright for
claude.ai, or --all for both.

Run with the Postgres container up and migrated to head:

    docker compose up -d
    cd backend && uv run python scripts/smoke_agent_planner.py

The script cleans up its rows at start and exit (learned items,
session, topics, domain under the smoke prefix), so re-runs start
clean and a mid-run failure still cleans up.

Requires:
  - DeepSeek path: DEEPSEEK_API_KEY in .env or process environment.
  - Playwright path: persistent Chrome profile logged in to claude.ai.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import TYPE_CHECKING, Any, Literal

from app.core.config import get_settings
from app.core.db import SessionLocal
from app.models import (
    Domain,
    DomainKind,
    GradingVerdict,
    LearnedItem,
    LearningMode,
    Session,
    SessionState,
    Topic,
    TopicStatus,
    TransportKind,
)
from app.schemas.agent_plan import MarkForRevisionStep, Plan
from app.schemas.tools import MarkForRevisionInput
from app.services.agent_error_recorder import WritingAgentErrorRecorder
from app.services.agent_orchestrator import AgentOrchestratorError
from app.services.agent_planner import (
    PlannerServiceError,
    approve_plan,
    propose_plan,
)
from app.services.embedding_service import OpenRouterEmbedder
from app.transport.deepseek_impl import DeepseekTransport
from app.transport.playwright_impl import PlaywrightClaudeTransport
from sqlalchemy import delete, select, update

if TYPE_CHECKING:
    from app.schemas.agent_plan import Evidence, PlanProposal
    from app.transport.base import LLMTransport

# Recognizable prefix so cleanup can find exactly what this smoke
# wrote without touching real rows. Every path here lives under it.
_SMOKE_PREFIX = "SmokePlanner"
_WEAK_A = f"{_SMOKE_PREFIX} > Loops > OffByOne"
_WEAK_B = f"{_SMOKE_PREFIX} > Recursion > BaseCase"
# Never seeded and never in evidence, so a plan targeting it must be
# rejected as ungrounded. Under the prefix so cleanup would catch it
# even if a bug created it.
_NEVER_SEEDED = f"{_SMOKE_PREFIX} > Ghost > NeverSeeded"

TransportChoice = Literal["deepseek", "playwright", "all"]


def _require_postgres() -> None:
    """Abort unless the configured database is Postgres.

    The approve leg proves commit semantics on the production
    database. Running against SQLite would pass without testing what
    it claims to test, so refuse rather than give a false green.
    """
    url = get_settings().database_url
    if not url.startswith("postgresql"):
        print(f"FAIL: smoke requires Postgres, got {url!r}.")
        print("Start the container and set the database URL to Postgres.")
        sys.exit(1)


def _cleanup() -> None:
    """Delete everything the smoke wrote, children before parents.

    Runs at start and at exit so a crashed prior run cannot poison
    this one and a mid-run failure still cleans up. Matches rows by
    the smoke prefix, so it cannot touch real data.
    """
    session = SessionLocal()
    try:
        topic_ids = (
            session.execute(select(Topic.id).where(Topic.path.like(f"{_SMOKE_PREFIX}%")))
            .scalars()
            .all()
        )
        if topic_ids:
            session.execute(delete(LearnedItem).where(LearnedItem.topic_id.in_(topic_ids)))
            session.execute(delete(Session).where(Session.topic_id.in_(topic_ids)))
            session.execute(delete(Topic).where(Topic.id.in_(topic_ids)))
        session.execute(delete(Domain).where(Domain.name == _SMOKE_PREFIX))
        session.commit()
    except Exception as e:
        session.rollback()
        print(f"WARN: cleanup failed: {e}")
    finally:
        session.close()


def _seed() -> None:
    """Seed two weak topics: rows, one synthetic session, four items.

    Topics start LEARNED so the approve leg's flip to NEEDS_REVISION
    is observable. The learned items carry INCORRECT verdicts so the
    real get_weak_topics surfaces both topics as weak.
    """
    session = SessionLocal()
    try:
        session.add(Domain(name=_SMOKE_PREFIX, kind=DomainKind.LANGUAGE, description=None))
        topics: list[Topic] = []
        for path in (_WEAK_A, _WEAK_B):
            topic = Topic(
                path=path,
                domain=_SMOKE_PREFIX,
                name=path.rsplit(" > ", 1)[-1],
                status=TopicStatus.LEARNED,
            )
            session.add(topic)
            topics.append(topic)
        session.flush()

        synthetic = Session(
            topic_id=topics[0].id,
            mode_used=LearningMode.FLASHCARD,
            state=SessionState.COMPLETED,
            transport_kind=TransportKind.DEEPSEEK,
        )
        session.add(synthetic)
        session.flush()

        for topic in topics:
            for n in (1, 2):
                session.add(
                    LearnedItem(
                        session_id=synthetic.id,
                        topic_id=topic.id,
                        question=f"{topic.name} smoke question {n}",
                        answer="the correct answer",
                        your_answer="a wrong answer",
                        mode=LearningMode.FLASHCARD,
                        grading_verdict=GradingVerdict.INCORRECT,
                    )
                )
        session.commit()
    finally:
        session.close()
    print(f"Seeded {_WEAK_A!r} and {_WEAK_B!r}: LEARNED, two INCORRECT items each.\n")


def _reset_seeded_statuses() -> None:
    """Set both seeded topics back to LEARNED on a short session.

    A prior transport's approve leg flips them to NEEDS_REVISION;
    every run starts from the same LEARNED baseline so the flip
    assertion stays meaningful.
    """
    session = SessionLocal()
    try:
        session.execute(
            update(Topic)
            .where(Topic.path.in_([_WEAK_A, _WEAK_B]))
            .values(status=TopicStatus.LEARNED)
        )
        session.commit()
    finally:
        session.close()


def _status_of(path: str) -> TopicStatus | None:
    """Read one topic's status on a fresh session, or None if absent."""
    session = SessionLocal()
    try:
        return session.execute(select(Topic.status).where(Topic.path == path)).scalar_one_or_none()
    finally:
        session.close()


def _evidence_paths(evidence: list[Evidence]) -> set[str]:
    """Paths present in the gathered evidence.

    get_weak_topics evidence carries a GetWeakTopicsOutput dump:
    result["topics"] is a list of WeakTopicInfo dicts keyed by
    topic_path. Re-derived here rather than importing the service's
    private guard helper, so the smoke checks the contract from the
    outside.
    """
    paths: set[str] = set()
    for item in evidence:
        topics = item.result.get("topics")
        if not isinstance(topics, list):
            continue
        for entry in topics:
            if isinstance(entry, dict):
                path = entry.get("topic_path")
                if isinstance(path, str):
                    paths.add(path)
    return paths


def _plan_targets(plan: Plan) -> list[str]:
    """Paths targeted by the plan's mark_for_revision steps."""
    return [step.args.path for step in plan.steps if isinstance(step, MarkForRevisionStep)]


def _print_proposal(evidence: list[Evidence], plan: Plan) -> None:
    """Print the round trip's artifacts for eyeballing."""
    print(f"  Evidence entries: {len(evidence)}")
    for item in evidence:
        print(f"    {item.tool}: {sorted(_evidence_paths([item]))}")
    print(f"  Plan steps: {len(plan.steps)}")
    for target in _plan_targets(plan):
        print(f"    mark_for_revision -> {target!r}")


async def _approve(plan: Plan, evidence: list[Evidence]) -> None:
    """Run approve_plan on a short session with a writing recorder."""
    db = SessionLocal()
    recorder = WritingAgentErrorRecorder(SessionLocal)
    try:
        await approve_plan(db=db, recorder=recorder, plan=plan, evidence=evidence)
    finally:
        db.close()


def _check_invariants(name: str, proposal: PlanProposal) -> list[str]:
    """Hard checks that hold regardless of which topic the LLM picked."""
    failures: list[str] = []
    targets = _plan_targets(proposal.plan)
    if not proposal.plan.steps:
        failures.append(f"{name}: plan is empty; no_data guard should have prevented this.")
    if len(targets) != len(proposal.plan.steps):
        failures.append(f"{name}: plan contains non-mutate steps.")
    if not proposal.evidence:
        failures.append(f"{name}: no evidence gathered; plan cannot be grounded.")
    evidenced = _evidence_paths(proposal.evidence)
    ungrounded = [t for t in targets if t not in evidenced]
    if ungrounded:
        failures.append(f"{name}: targets missing from evidence: {ungrounded!r}")
    return failures


async def _check_tamper_rejection(name: str, evidence: list[Evidence]) -> list[str]:
    """A target absent from the evidence must be rejected, untouched.

    Exercises the approve-side groundedness re-check against real
    LLM-produced evidence: the tampered plan must raise ungrounded
    before anything executes, and the seeded topics must be
    unchanged afterwards.
    """
    failures: list[str] = []
    tampered = Plan(steps=[MarkForRevisionStep(args=MarkForRevisionInput(path=_NEVER_SEEDED))])
    try:
        await _approve(tampered, evidence)
        failures.append(f"{name}: tampered plan was approved; expected ungrounded rejection.")
    except PlannerServiceError as exc:
        if exc.kind != "ungrounded":
            failures.append(
                f"{name}: tampered plan rejected with kind={exc.kind!r}, expected 'ungrounded'."
            )
    for path in (_WEAK_A, _WEAK_B):
        if _status_of(path) is not TopicStatus.LEARNED:
            failures.append(f"{name}: {path!r} changed by the rejected tampered plan.")
    return failures


async def _check_approve(
    name: str, smoke_targets: list[str], evidence: list[Evidence]
) -> list[str]:
    """Approve the smoke-prefixed targets and assert the status flips."""
    failures: list[str] = []
    approved = Plan(
        steps=[MarkForRevisionStep(args=MarkForRevisionInput(path=t)) for t in smoke_targets]
    )
    try:
        await _approve(approved, evidence)
    except (PlannerServiceError, AgentOrchestratorError) as exc:
        failures.append(f"{name}: approve failed: {exc}")
        return failures

    for target in smoke_targets:
        if _status_of(target) is not TopicStatus.NEEDS_REVISION:
            failures.append(f"{name}: {target!r} not NEEDS_REVISION after approve.")
    return failures


async def _smoke_one(
    name: str,
    transport: LLMTransport[Any],
    kind: TransportKind,
    embedder: OpenRouterEmbedder,
) -> list[str]:
    """Propose, verify invariants, tamper-reject, then approve.

    Returns the failures it found (empty list means pass) so the
    caller can aggregate across transports before deciding exit code.
    """
    _reset_seeded_statuses()
    print(f"[{name}] propose...")

    db = SessionLocal()
    try:
        proposal = await propose_plan(
            db=db,
            transport=transport,
            embedder=embedder,
            transport_kind=kind,
        )
    except PlannerServiceError as exc:
        return [f"{name}: propose failed, kind={exc.kind}: {exc.message}"]
    finally:
        db.close()

    _print_proposal(proposal.evidence, proposal.plan)

    failures = _check_invariants(name, proposal)
    if failures:
        return failures

    # Which topics the LLM picked is its call: report, don't assert.
    targets = _plan_targets(proposal.plan)
    smoke_targets = [t for t in targets if t.startswith(_SMOKE_PREFIX)]
    other_targets = [t for t in targets if not t.startswith(_SMOKE_PREFIX)]
    if other_targets:
        print(f"  [check] plan also targets non-smoke topics: {other_targets!r}.")
        print("           Grounded, so legitimate; excluded from approve for data safety.")

    failures += await _check_tamper_rejection(name, proposal.evidence)

    # Approve only smoke-prefixed targets so the smoke never mutates
    # real topics on a shared database.
    if not smoke_targets:
        print("  [no-evidence] plan contains no smoke-prefixed targets; approve leg skipped.")
        return failures

    failures += await _check_approve(name, smoke_targets, proposal.evidence)

    if not failures:
        print(f"  PASS [{name}]: propose grounded, tamper rejected, approve committed.\n")
    return failures


async def run(choice: TransportChoice) -> int:
    """Seed, run the chosen transport(s), clean up, aggregate."""
    _require_postgres()
    _cleanup()
    _seed()

    settings = get_settings()
    failures: list[str] = []
    try:
        async with OpenRouterEmbedder(
            api_key=settings.openrouter_api_key.get_secret_value(),
            model=settings.openrouter_embedding_model,
        ) as embedder:
            if choice in {"deepseek", "all"}:
                async with DeepseekTransport(
                    api_key=settings.deepseek_api_key.get_secret_value(),
                    default_model=settings.deepseek_model,
                ) as ds:
                    failures += await _smoke_one(
                        f"DeepSeek/{settings.deepseek_model}",
                        ds,
                        TransportKind.DEEPSEEK,
                        embedder,
                    )
            if choice in {"playwright", "all"}:
                async with PlaywrightClaudeTransport(settings.chrome_profile_path) as pw:
                    failures += await _smoke_one(
                        "Playwright/claude.ai",
                        pw,
                        TransportKind.CLAUDE_PLAYWRIGHT,
                        embedder,
                    )
    finally:
        _cleanup()

    if failures:
        print("\nSMOKE FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nSMOKE PASSED: planner propose/approve round trip holds end to end.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Bounded-planner smoke test.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--transport",
        choices=["deepseek", "playwright"],
        default="deepseek",
        help="Transport to exercise (default: deepseek).",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Run against both transports.",
    )
    args = parser.parse_args()
    choice: TransportChoice = "all" if args.all else args.transport
    sys.exit(asyncio.run(run(choice)))


if __name__ == "__main__":
    main()

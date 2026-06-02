"""Agent orchestrator.

Walks a Plan: runs read steps during planning to gather evidence,
then (only on approval) runs mutate steps inside one transaction.
Either every mutation applies or none do.

Two session strategies live here, and reversing them breaks the
contract (see the module's tests and the smoke). The plan's
mutations run on the request session passed in by the caller: the
orchestrator is the holder, nothing outer will roll back underneath
it, so it owns one commit at the end and one rollback on failure
(the start_session boundary pattern). Error logging runs on an
independent session through the injected recorder: an error record
must survive the plan's own rollback, so it cannot share the session
the rollback discards.

No planner or LLM yet. The caller hands in a Plan directly and a
boolean approve gate stands in for the real propose and approve
round trip. Dispatch is hardcoded to the two tools this slice
exercises and will be replaced with specialist dispatch in a later
step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.schemas.agent_plan import Evidence, Plan, PlanStep
from app.schemas.tools import GetWeakTopicsInput
from app.services.agent_error_recorder import AgentErrorData
from app.services.agent_tools import stage_topic_upsert
from app.services.tools.handlers import get_weak_topics

if TYPE_CHECKING:
    from sqlalchemy.orm import Session as DbSession

    from app.services.agent_error_recorder import AgentErrorRecorder


class AgentOrchestratorError(Exception):
    """An orchestrator run failed.

    Raised after the plan's transaction has been rolled back and the
    failure has been recorded on the independent session. The caller
    sees one error type, the staged plan has left no partial writes.
    """

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.cause = cause


async def run_plan(
    *,
    db: DbSession,
    recorder: AgentErrorRecorder,
    plan: Plan,
    approve: bool,
) -> list[Evidence]:
    """Run a plan's reads, then its mutations if approved.

    Two passes. The read pass runs every read step and collects one
    Evidence per step. No writes happen, so no approval is needed to
    reach the evidence. The mutate pass runs only when approve is
    True: it executes every mutate step on the request session and
    commits once at the end. Any failure in the mutate pass rolls the
    whole transaction back, records the failure on the independent
    session, and raises.

    Returns the collected evidence. The caller (the smoke and the
    tests) uses the evidence to assert the read pass ran and inspects
    the database to assert the mutate pass applied or rolled back as
    a unit.

    The approve gate is current stand-in for the propose/approve
    round trip. With approve False the function returns after the
    read pass having mutated nothing, which is the proposal phase.
    With approve True it runs the mutations, which is the execute
    phase. This will be split into the real two-call shape in a
    later step.
    """
    evidence = await _run_read_pass(db, plan)

    if not approve:
        return evidence

    await _run_mutate_pass(db=db, recorder=recorder, plan=plan)
    return evidence


async def _run_read_pass(db: DbSession, plan: Plan) -> list[Evidence]:
    """Run every read step, returning one Evidence each.

    Read steps must not mutate. A mutate step reached here is a plan
    construction bug, not a runtime condition to tolerate: the read
    pass runs before any approval, so a mutation here would be an
    unapproved write. The kind check skips them, the mutate pass is
    the only place mutations run.

    Async because the read tools are async handlers (they share the
    teaching-loop handler signatures). The dispatched handler does
    sync work under an async def, awaiting it keeps the read pass
    honest about calling an async surface rather than reaching past
    it.
    """
    evidence: list[Evidence] = []
    for step in plan.steps:
        if step.kind != "read":
            continue
        result = await _dispatch_read(db, step)
        evidence.append(Evidence(tool=step.tool, result=result))
    return evidence


async def _run_mutate_pass(
    *,
    db: DbSession,
    recorder: AgentErrorRecorder,
    plan: Plan,
) -> None:
    """Run every mutate step in one transaction, all-or-nothing.

    Each mutate step stages its write through the commit-free agent
    tools, which flush so later steps see earlier ones. After all
    steps succeed the transaction commits once. Any failure rolls the
    whole transaction back so no partial plan persists, records the
    failure on the independent recorder session, and raises.

    The rollback comes before the record call deliberately: the
    record runs on its own session and is unaffected by the rollback,
    but rolling back first discards the staged mutations promptly
    rather than leaving them pending while the record write happens.
    """
    failed_index: int | None = None
    failed_tool: str | None = None
    try:
        for index, step in enumerate(plan.steps):
            if step.kind != "mutate":
                continue
            failed_index = index
            failed_tool = step.tool
            _dispatch_mutate(db, step)
        db.commit()
    except Exception as e:
        db.rollback()
        recorder.record(
            AgentErrorData(
                kind="agent.plan.mutate_failed",
                message=str(e),
                context={
                    "failed_step_index": failed_index,
                    "failed_tool": failed_tool,
                },
            )
        )
        raise AgentOrchestratorError(
            f"Plan mutation failed at step {failed_index} ({failed_tool!r}): {e}",
            cause=e,
        ) from e


async def _dispatch_read(db: DbSession, step: PlanStep) -> dict[str, object]:
    """Dispatch one read step to its tool. Currently hardcoded.

    One read tool in this slice: get_weak_topics. Dispatch is a
    literal name check rather than a registry and will be replaced
    with specialist dispatch once there is more than one read tool
    worth routing.

    get_weak_topics is read-only and does not use the embedder, so it
    is awaited directly rather than routed through execute_tool_call
    (which threads an embedder the teaching loop's retrieval tool
    needs). Calling the handler keeps the read on the one code path
    that already produces this output.
    """
    if step.tool == "get_weak_topics":
        args = GetWeakTopicsInput.model_validate(step.args)
        output = await get_weak_topics(db, args)
        return output.model_dump(mode="json")
    raise AgentOrchestratorError(f"Unknown read tool {step.tool!r}.")


def _dispatch_mutate(db: DbSession, step: PlanStep) -> None:
    """Dispatch one mutate step to its tool. Currently hardcoded.

    One mutate tool this slice: stage_topic_upsert, which flushes
    without committing so the orchestrator owns the transaction.
    """
    if step.tool == "stage_topic_upsert":
        path = str(step.args["path"])
        stage_topic_upsert(db, path=path)
        return
    raise AgentOrchestratorError(f"Unknown mutate tool {step.tool!r}.")

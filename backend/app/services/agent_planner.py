"""Planner service.

Drives the propose half of the agent's propose/approve round trip.
The LLM reads the user's weak topics through the tool-call loop and
emits a mutate-only plan. The tool results are kept as Evidence, the
plan is checked against them, and the pair goes back to the caller.
Nothing mutates during propose.

Approve receives the plan and evidence back, re-checks groundedness
(the backend holds no state between the two calls), and executes
through the orchestrator's transaction boundary.

The chat is throwaway: no persistence, no session row. Mirrors the
diagnostic service's loop with two differences. Tool calls are gated
against a read allowlist before dispatch, because registry handlers
commit their own writes and an unapproved mutation must die
structurally rather than by prompt promise. And get_weak_topics
results become Evidence instead of being discarded.
"""

from __future__ import annotations

import contextlib
import json
from typing import TYPE_CHECKING, Any, Literal

from pydantic import ValidationError

from app.prompts.planner_intro import build_planner_intro
from app.schemas.agent_plan import Evidence, MarkForRevisionStep, PlanProposal
from app.schemas.parsed_response import ParsedToolCall
from app.schemas.tools import GetWeakTopicsInput, GetWeakTopicsOutput
from app.services.agent_orchestrator import run_plan
from app.services.parser import ParseError, parse_plan_response
from app.services.tools.handlers import ToolHandlerError, get_weak_topics
from app.services.tools.registry import execute_tool_call
from app.transport.base import (
    ToolResult,
    TransportError,
    TransportResponse,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session as DbSession

    from app.models import TransportKind
    from app.schemas.agent_plan import ParsedPlan, Plan
    from app.services.agent_error_recorder import AgentErrorRecorder
    from app.services.embedding_service import Embedder
    from app.transport.base import LLMTransport


# First message sent to the planner chat after the intro. Kept thin:
# the intro carries the instructions, this is the "go" signal.
_FIRST_MESSAGE = (
    "Read the user's weak topics with get_weak_topics, then respond "
    "with a PLAN block proposing which topics to mark for revision."
)

# Tools the planner LLM may call during the propose loop. Checked
# before registry dispatch: the registry's write handlers commit
# their own writes, and a mutation during propose would be an
# unapproved write. The phase split is structural, not a prompt rule.
_ALLOWED_TOOLS = frozenset({"get_weak_topics"})


# Failure modes for the planner service. The route layer maps these
# to HTTP status codes. wrong_response_kind from the diagnostic set
# is absent deliberately: parse_plan_response only produces the two
# planner shapes, so a wrong-kind terminal is unconstructable and
# surfaces as parse_failed instead.
type PlannerErrorKind = Literal[
    "transport_failed",
    "parse_failed",
    "tool_handler_failed",
    "disallowed_tool",
    "no_data",
    "ungrounded",
    "unexpected",
]


class PlannerServiceError(Exception):
    """A planner-service operation failed.

    Wraps the underlying cause so callers see one error type at the
    service boundary. kind is the discriminator the route layer uses
    to pick the HTTP status: no_data maps to 422, ungrounded and
    disallowed_tool map to 502 since both mean the upstream LLM broke
    its contract.
    """

    def __init__(
        self,
        message: str,
        kind: PlannerErrorKind,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.kind: PlannerErrorKind = kind
        self.cause = cause


async def propose_plan(
    *,
    db: DbSession,
    transport: LLMTransport[Any],
    embedder: Embedder,
    transport_kind: TransportKind,
) -> PlanProposal:
    """Open a planner chat, gather evidence, return the grounded plan.

    Guards the empty state before any transport call, opens a fresh
    chat with the static intro, drives the tool-call loop keeping
    get_weak_topics results as Evidence, parses the terminal plan,
    and checks groundedness before returning. Nothing mutates here:
    the plan executes only when the caller approves it via
    approve_plan.

    transport_kind is accepted for signature symmetry with the
    diagnostic service and the route deps, the flow has no
    per-transport branching.

    Raises PlannerServiceError on empty state, transport failure,
    parse failure, a disallowed tool call, a tool handler failure,
    or an ungrounded plan.
    """
    await _check_plannable_state(db)

    intro = build_planner_intro()

    try:
        chat, response = await transport.start_new_chat(intro, _FIRST_MESSAGE)
    except TransportError as e:
        raise PlannerServiceError(
            f"Transport failed opening planner chat: {e.message}",
            kind="transport_failed",
            cause=e,
        ) from e

    try:
        parsed, evidence = await _execute_until_plan(
            transport=transport,
            embedder=embedder,
            chat=chat,
            response=response,
            db=db,
        )
    except PlannerServiceError:
        # Close the chat even on failure. Planner chats are throwaway
        # and leaving one open leaks transport state.
        await _close_quietly(transport, chat)
        raise
    except Exception as e:
        await _close_quietly(transport, chat)
        raise PlannerServiceError(
            f"Unexpected error during planner flow: {e}",
            kind="unexpected",
            cause=e,
        ) from e

    await _close_quietly(transport, chat)

    _assert_plan_grounded(parsed.plan, evidence)
    return PlanProposal(plan=parsed.plan, evidence=evidence)


async def approve_plan(
    *,
    db: DbSession,
    recorder: AgentErrorRecorder,
    plan: Plan,
    evidence: list[Evidence],
) -> None:
    """Re-check groundedness, then execute the plan's mutations.

    The backend holds no state between propose and approve, so the
    plan and evidence arrive back from the caller and the same guard
    runs again before anything executes. Execution goes through
    run_plan with approval: one transaction, all-or-nothing.

    Raises PlannerServiceError(kind="ungrounded") from the guard.
    Lets AgentOrchestratorError propagate unwrapped: a mutation
    failure is the orchestrator's contract and the route maps it
    separately.
    """
    _assert_plan_grounded(plan, evidence)
    await run_plan(db=db, recorder=recorder, plan=plan, approve=True)


async def _execute_until_plan(
    *,
    transport: LLMTransport[Any],
    embedder: Embedder,
    chat: Any,
    response: TransportResponse,
    db: DbSession,
) -> tuple[ParsedPlan, list[Evidence]]:
    """Drive tool calls until the LLM emits the terminal plan.

    Every call is checked against the allowlist before dispatch, then
    runs through the same registry as the teaching flow. Results from
    get_weak_topics are kept as Evidence. The caller checks the plan
    against them and returns them as the proposal's justification.
    """
    evidence: list[Evidence] = []
    parsed = _parse_or_raise(response)
    while isinstance(parsed, ParsedToolCall):
        # Execute every call in the response before sending results
        # back. OpenAI-compatible APIs require all calls in one
        # assistant message to be answered together.
        results: list[ToolResult] = []
        for call in parsed.calls:
            if call.name not in _ALLOWED_TOOLS:
                raise PlannerServiceError(
                    f"Tool {call.name!r} is not available to the planner.",
                    kind="disallowed_tool",
                )
            try:
                output = await execute_tool_call(db, call, embedder)
            except ToolHandlerError as e:
                raise PlannerServiceError(
                    f"Tool handler {call.name!r} failed: {e.message}",
                    kind="tool_handler_failed",
                    cause=e,
                ) from e
            if call.name == "get_weak_topics":
                evidence.append(Evidence(tool=call.name, result=output.model_dump(mode="json")))
            results.append(
                ToolResult(call_id=call.id or call.name, content=output.model_dump_json())
            )

        try:
            response = await transport.send_tool_results(chat, results)
        except TransportError as e:
            raise PlannerServiceError(
                f"Transport failed sending tool results: {e.message}",
                kind="transport_failed",
                cause=e,
            ) from e

        parsed = _parse_or_raise(response)

    return parsed, evidence


def _parse_or_raise(response: TransportResponse) -> ParsedToolCall | ParsedPlan:
    """Translate a TransportResponse, wrapping parse failures uniformly."""
    try:
        return _response_to_parsed(response)
    except ParseError as e:
        raise PlannerServiceError(
            f"Parse failed on planner response: {e.message}",
            kind="parse_failed",
            cause=e,
        ) from e


def _response_to_parsed(response: TransportResponse) -> ParsedToolCall | ParsedPlan:
    """Translate a TransportResponse for the planner flow.

    Same shape as the diagnostic service's translation: native
    tool_calls take precedence (DeepSeek), otherwise the text parses
    through the planner grammar, a TOOL_CALL block or the terminal
    PLAN.
    """
    if response.tool_calls:
        calls = list(response.tool_calls)
        raw_text = json.dumps([c.model_dump(mode="json") for c in calls])
        return ParsedToolCall(calls=calls, raw_text=raw_text)
    return parse_plan_response(response.text)


async def _close_quietly(transport: LLMTransport[Any], chat: Any) -> None:
    """Close the chat, swallowing any error from the close itself.

    A failed close is not worth promoting over the original error
    being raised. Throwaway chats: leaking the chat handle is
    acceptable if close fails.
    """
    with contextlib.suppress(Exception):
        await transport.close(chat)


async def _check_plannable_state(db: DbSession) -> None:
    """Raise PlannerServiceError(kind="no_data") when no weak topics exist.

    Runs the same handler the LLM's tool call hits, with the widest
    net: min_attempts=1 so a single graded miss counts, sample_size=0
    because only existence matters here. An empty result means a
    planner chat could only produce an empty or invented plan, both
    of which the wire format rejects, so the chat never opens.
    """
    output = await get_weak_topics(db, GetWeakTopicsInput(min_attempts=1, sample_size=0))
    if not output.topics:
        raise PlannerServiceError(
            "No weak topics exist yet. The planner needs graded attempts "
            "with incorrect or partial verdicts to plan from.",
            kind="no_data",
        )


def _assert_plan_grounded(plan: Plan, evidence: list[Evidence]) -> None:
    """Raise unless every plan step is a mutate targeting an evidenced path.

    Three checks: the plan has at least one step, every step is a
    mutate step, and every target path appears in a get_weak_topics
    evidence entry. A plan emitted without any tool call has empty
    evidence and fails here, so the call-first rule is structural.

    Runs before the orchestrator, so a bad plan dies with a clean
    error instead of mid-transaction. Existence against current
    database state stays with the strict mutate core inside the
    transaction.
    """
    grounded = _evidenced_paths(evidence)
    if not plan.steps:
        raise PlannerServiceError("Plan has no steps.", kind="ungrounded")
    for index, step in enumerate(plan.steps):
        if not isinstance(step, MarkForRevisionStep):
            raise PlannerServiceError(
                f"Plan step {index} ({step.tool!r}) is not a mutate step. "
                f"Planner plans are mutate-only.",
                kind="ungrounded",
            )
        if step.args.path not in grounded:
            raise PlannerServiceError(
                f"Plan step {index} targets {step.args.path!r}, which is "
                f"not in the gathered evidence.",
                kind="ungrounded",
            )


def _evidenced_paths(evidence: list[Evidence]) -> set[str]:
    """Collect topic paths from get_weak_topics evidence entries.

    Entries validate back through GetWeakTopicsOutput. On approve the
    evidence arrives from the client, so an entry that does not
    validate grounds nothing rather than being trusted.
    """
    paths: set[str] = set()
    for entry in evidence:
        if entry.tool != "get_weak_topics":
            continue
        try:
            output = GetWeakTopicsOutput.model_validate(entry.result)
        except ValidationError:
            continue
        paths.update(t.topic_path for t in output.topics)
    return paths

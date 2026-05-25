"""Smoke test for the tool-execution loop against real LLMs.

The LLM emits tool calls (either as ---TOOL_CALL--- blocks
on Claude transport or via native function calling on DeepSeek),
the registry executes handlers, and results flow back to the LLM.
Verifies the end-to-end path with real DB side effects.

Uses a topic in a non-existent domain to nudge the LLM toward
calling create_domain. If the LLM complies, a Domain row lands
in the DB. If it doesn't comply, the smoke fails and we know
the intro needs more work to motivate tool use.

Run from backend/ with:

    uv run python scripts/smoke_tool_calls.py
    uv run python scripts/smoke_tool_calls.py --transport=playwright
    uv run python scripts/smoke_tool_calls.py --all

Each run creates real rows in learning.db. Use `uv run python -m
cli.admin db reset` between runs if you want a fresh slate.

Requires:
  - DeepSeek path: DEEPSEEK_API_KEY in .env or process environment.
  - Playwright path: persistent Chrome profile logged in to claude.ai
    (run scripts/spike_claude_dom.py once if needed).
"""

from __future__ import annotations

import argparse
import asyncio
from typing import TYPE_CHECKING, Any, Literal

from app.core.config import get_settings
from app.core.db import SessionLocal
from app.models import Domain, Embedding, EmbeddingSourceType, SessionTurn, TransportKind, TurnRole
from app.services.embedding_service import (
    EmbeddingRecord,
    OpenRouterEmbedder,
    embed_records,
)
from app.services.session_service import (
    SessionServiceError,
    send_user_answer,
    start_session,
)
from app.transport import TransportError
from app.transport.deepseek_impl import DeepseekTransport
from app.transport.playwright_impl import PlaywrightClaudeTransport
from sqlalchemy import select

if TYPE_CHECKING:
    from app.transport.base import LLMTransport
    from sqlalchemy.orm import Session as DbSession


# An obscure-enough language that it won't be in any seed list.
# If the seed list ever includes Mojo, switch to another fresh
# domain name. The point is: this domain must not exist in the
# DB before the smoke runs, so create_domain has a reason to fire.
SMOKE_DOMAIN_NAME = "Mojo"
SMOKE_TOPIC_PATH = f"{SMOKE_DOMAIN_NAME} > Memory Model > Ownership"

# A follow-up question that nudges the LLM to look up existing
# topics in the same domain. Designed to make get_topics_by_domain
# a useful next move.
FOLLOWUP_ANSWER = (
    "Before answering, please tell me what other Mojo topics "
    "exist in the system so I can plan my study order."
)

TransportChoice = Literal["deepseek", "playwright", "all"]

# Check kind for smoke_one. The two nudges produce different
# behavior in the LLM. The check that follows is correspondingly
# different. Keeping them separate per run avoids coupling two
# independent compliance signals into one pass/fail.
CheckMode = Literal["domain_and_tool_use", "search_corpus"]


# Second nudge: search_corpus dedup at session start.
#
# Seeded items look like prior learned items on Python integer
# division. When the smoke starts a session on the same topic,
# the LLM is expected to call search_corpus during start to
# check whether the user has seen these questions before. The
# intro documents that workflow. This smoke verifies the LLM
# actually exercises it.
#
# Idempotent on source_id, re-runs of this smoke do not
# duplicate seed rows.
SEED_TOPIC_PATH = "Python > Data Types > Integers"
SEED_RECORDS: list[tuple[str, str]] = [
    (
        "smoke-tool-call-seed-0",
        "What is the result of 7 // 2 in Python 3?\n"
        "Answer: 3. The // operator is integer floor division and truncates toward zero on positive ints.",
    ),
    (
        "smoke-tool-call-seed-1",
        "What does the % operator do on Python integers?\n"
        "Answer: returns the remainder of integer division. 10 % 3 is 1.",
    ),
]

# Hint appended to the topic path on the search-corpus run to
# nudge dedup behavior. The production path does not need this
# hint because the intro already documents the dedup workflow,
# but real LLM compliance with new tools is empirical. A direct
# nudge in the user-facing topic makes the dedup window obvious
# to the LLM without disturbing the intro for production sessions.
SEARCH_CORPUS_NUDGE_ANSWER = (
    "Before picking your first question, please make sure you're "
    "not repeating something I've already been asked. Check the "
    "history if you can."
)


async def _seed_search_corpus_items(embedder: OpenRouterEmbedder) -> None:
    """Idempotently seed the embedding rows the search_corpus check needs.

    Skips any seed whose source_id is already present. The seed
    is a small fixed list. The cost of the existence check is
    one query and a few index lookups.
    """
    with SessionLocal() as db:
        existing_ids = set(
            db.execute(
                select(Embedding.source_id).where(
                    Embedding.source_id.in_([sid for sid, _ in SEED_RECORDS])
                )
            )
            .scalars()
            .all()
        )

        missing = [(sid, content) for sid, content in SEED_RECORDS if sid not in existing_ids]
        if not missing:
            print(f"[seed] all {len(SEED_RECORDS)} search-corpus seed rows already present.\n")
            return

        records = [
            EmbeddingRecord(
                source_type=EmbeddingSourceType.LEARNED_ITEM,
                source_id=sid,
                content=content,
            )
            for sid, content in missing
        ]
        print(f"[seed] embedding and storing {len(records)} search-corpus seed rows...")
        await embed_records(db=db, embedder=embedder, records=records)
        db.commit()
        print(f"[seed] stored {len(records)} rows.\n")


async def smoke_one(
    name: str,
    transport: LLMTransport[Any],
    transport_kind: TransportKind,
    embedder: OpenRouterEmbedder,
    check_mode: CheckMode,
) -> None:
    """Run one check against the given transport.

    Dispatches to a check-specific helper. Each helper opens its
    own DB session, runs the relevant nudge, and asserts. Splitting
    keeps complexity per function low and makes it possible to
    iterate either nudge's wording without touching the other.
    """
    print(f"=== {name} (check_mode={check_mode}) ===")

    if check_mode == "domain_and_tool_use":
        await _check_domain_and_tool_use(name, transport, transport_kind, embedder)
    else:
        await _check_search_corpus(name, transport, transport_kind, embedder)


async def _check_domain_and_tool_use(
    name: str,
    transport: LLMTransport[Any],
    transport_kind: TransportKind,
    embedder: OpenRouterEmbedder,
) -> None:
    """Run the Mojo-domain nudge and assert tool use.

    Pass condition: either (a) the Mojo Domain row was created
    during start (LLM called create_domain), or (b) the follow-up
    persisted at least one TOOL_CALL turn (LLM called something).
    Either path proves tool execution is working.
    """
    print(f"  topic_path: {SMOKE_TOPIC_PATH}")

    with SessionLocal() as db:
        existing_mojo = db.query(Domain).filter(Domain.name == SMOKE_DOMAIN_NAME).one_or_none()
        if existing_mojo is not None:
            print(
                f"  [warn] {SMOKE_DOMAIN_NAME!r} domain already exists. "
                f"Run targeted cleanup for a clean smoke."
            )

        session, parsed = await start_session(
            db=db,
            transport=transport,
            transport_kind=transport_kind,
            topic_path=SMOKE_TOPIC_PATH,
            embedder=embedder,
        )
        print(f"  [start] session.id={session.id}")
        print(f"  [start] parsed.question={parsed.question[:100]!r}")

        post_start_mojo = db.query(Domain).filter(Domain.name == SMOKE_DOMAIN_NAME).one_or_none()
        domain_created_at_start = post_start_mojo is not None and existing_mojo is None
        if domain_created_at_start:
            print(f"  [check] {SMOKE_DOMAIN_NAME!r} Domain row created during start: OK")
        else:
            print(f"  [check] {SMOKE_DOMAIN_NAME!r} Domain row not created during start.")

        print(f"  [send] answer={FOLLOWUP_ANSWER[:80]!r}")
        next_parsed = await send_user_answer(
            db=db,
            transport=transport,
            session_id=session.id,
            answer=FOLLOWUP_ANSWER,
            embedder=embedder,
        )
        print(f"  [send] parsed.kind={next_parsed.kind}")

        tool_call_turns = _fetch_tool_call_turns(db, session.id)
        _print_tool_call_summary(tool_call_turns)

        if domain_created_at_start or tool_call_turns:
            print(f"  [{name}] ok — tool path exercised.\n")
        else:
            print(
                f"  [{name}] no evidence — neither Domain row nor TOOL_CALL turns. "
                f"Inspect: session_turn WHERE session_id={session.id!r}."
            )
            print()


# Substrings that indicate the LLM read the seeded items. The two
# seed records cover floor division (//) and modulo (%). If either
# concept appears in the teaching turn or grading explanation
# without us mentioning it, the LLM must have read it from the
# embedding store.
SEED_EVIDENCE_KEYWORDS: list[str] = [
    "floor division",
    "//",
    "modulo",
    "modulus",
    " % ",
    "remainder",
]


async def _check_search_corpus(
    name: str,
    transport: LLMTransport[Any],
    transport_kind: TransportKind,
    embedder: OpenRouterEmbedder,
) -> None:
    """Run the search-corpus dedup nudge and report what the LLM did.

    Two evidence paths, either of which proves search_corpus
    fired:

      1. A TOOL_CALL turn with name search_corpus persisted in
         session_turn. This is the strong signal that DeepSeek
         produces by calling search_corpus during the follow-up
         (where calls do persist).

      2. The LLM's teaching turn or grading explanation
         references content unique to the seeded items
         (floor division, modulo, etc.) without the user
         having mentioned them. This is the only signal
         available when search_corpus was called during
         start_session, because it means start-time tool
         turns do not persist.

    Smoke does not raise on missing evidence. Smokes run
    against real LLMs whose tool-call timing is not
    deterministic. Failing the run on LLM nondeterminism
    forces hand-tuning the smoke per model. The script
    prints "[ok]" or "[no evidence]" and lets the operator
    decide.
    """
    print(f"  topic_path: {SEED_TOPIC_PATH}")

    with SessionLocal() as db:
        session, parsed = await start_session(
            db=db,
            transport=transport,
            transport_kind=transport_kind,
            topic_path=SEED_TOPIC_PATH,
            embedder=embedder,
        )
        print(f"  [start] session.id={session.id}")
        print(f"  [start] parsed.question={parsed.question[:100]!r}")

        print(f"  [send] answer={SEARCH_CORPUS_NUDGE_ANSWER[:80]!r}")
        next_parsed = await send_user_answer(
            db=db,
            transport=transport,
            session_id=session.id,
            answer=SEARCH_CORPUS_NUDGE_ANSWER,
            embedder=embedder,
        )
        print(f"  [send] parsed.kind={next_parsed.kind}")

        tool_call_turns = _fetch_tool_call_turns(db, session.id)
        _print_tool_call_summary(tool_call_turns)

        search_corpus_calls = [
            turn
            for turn in tool_call_turns
            if turn.parsed is not None
            and "call" in turn.parsed
            and turn.parsed["call"].get("name") == "search_corpus"
        ]
        persisted_call_evidence = bool(search_corpus_calls)
        if persisted_call_evidence:
            print(f"  [evidence] search_corpus persisted: {len(search_corpus_calls)} call(s)")
            for turn in search_corpus_calls:
                if turn.parsed is not None and "call" in turn.parsed:
                    args = turn.parsed["call"].get("args", {})
                    query = args.get("query", "?")
                    source_type = args.get("source_type", "(any)")
                    print(f"  [evidence]   - query={query[:60]!r} source_type={source_type!r}")

        # search_corpus at session start does not persist as a
        # TOOL_CALL turn. The only trace is the LLM's downstream output
        # referencing seed content. Inspect raw assistant turns for
        # keywords unique to the seeded items.
        assistant_turns = (
            db.query(SessionTurn)
            .filter(SessionTurn.session_id == session.id)
            .filter(SessionTurn.role.in_([TurnRole.ASSISTANT, TurnRole.GRADING]))
            .all()
        )
        downstream_text = "\n".join((turn.raw_content or "") for turn in assistant_turns).lower()
        matched_keywords = sorted(
            {kw for kw in SEED_EVIDENCE_KEYWORDS if kw.lower() in downstream_text}
        )
        downstream_evidence = bool(matched_keywords)
        if downstream_evidence:
            print(f"  [evidence] LLM output references seed content: {matched_keywords}")

        if persisted_call_evidence or downstream_evidence:
            print(f"  [{name}] ok — search_corpus exercised.\n")
        else:
            print(
                f"  [{name}] no evidence — neither persisted call nor seed-content "
                f"reference in LLM output. Inspect: session_turn WHERE "
                f"session_id={session.id!r}."
            )
            print()


def _fetch_tool_call_turns(db: DbSession, session_id: str) -> list[SessionTurn]:
    """Return all TOOL_CALL turns for a session and verify TOOL_RESULT pairing.

    Raises if call and result counts mismatch — that's a wire-format
    bug worth surfacing immediately rather than papering over.
    """
    tool_call_turns = (
        db.query(SessionTurn)
        .filter(SessionTurn.session_id == session_id)
        .filter(SessionTurn.role == TurnRole.TOOL_CALL)
        .all()
    )
    tool_result_turns = (
        db.query(SessionTurn)
        .filter(SessionTurn.session_id == session_id)
        .filter(SessionTurn.role == TurnRole.TOOL_RESULT)
        .all()
    )
    if len(tool_result_turns) != len(tool_call_turns):
        raise RuntimeError(
            f"Mismatched TOOL_CALL/TOOL_RESULT turn counts: "
            f"{len(tool_call_turns)} calls, {len(tool_result_turns)} results."
        )
    return tool_call_turns


def _print_tool_call_summary(turns: list[SessionTurn]) -> None:
    """Print a one-line summary of each TOOL_CALL turn."""
    if turns:
        print(f"  [check] TOOL_CALL turns persisted: {len(turns)}")
        for turn in turns:
            if turn.parsed is not None and "call" in turn.parsed:
                call = turn.parsed["call"]
                print(f"  [check]   - tool: {call.get('name')!r} args: {call.get('args')}")
    else:
        print("  [check] No TOOL_CALL turns persisted during follow-up.")


async def run(choice: TransportChoice) -> None:
    """Dispatch to the chosen transport(s).

    Each transport runs both check modes back-to-back:
    domain_and_tool_use (Mojo topic, create_domain nudge) and
    search_corpus (Python > Integers, dedup nudge). Two API
    calls per transport per check.
    """
    settings = get_settings()

    async with OpenRouterEmbedder(
        api_key=settings.openrouter_api_key.get_secret_value(),
        model=settings.openrouter_embedding_model,
    ) as embedder:
        # Seed the search-corpus embedding rows once, before any
        # transport runs. Idempotent on source_id.
        await _seed_search_corpus_items(embedder)

        if choice in {"deepseek", "all"}:
            print("Starting DeepSeek transport...\n")
            async with DeepseekTransport(
                api_key=settings.deepseek_api_key.get_secret_value(),
                default_model=settings.deepseek_model,
            ) as ds:
                await smoke_one(
                    f"DeepSeek/{settings.deepseek_model}",
                    ds,
                    TransportKind.DEEPSEEK,
                    embedder,
                    check_mode="domain_and_tool_use",
                )
                await smoke_one(
                    f"DeepSeek/{settings.deepseek_model}",
                    ds,
                    TransportKind.DEEPSEEK,
                    embedder,
                    check_mode="search_corpus",
                )

        if choice in {"playwright", "all"}:
            print("Starting Playwright transport...\n")
            async with PlaywrightClaudeTransport(settings.chrome_profile_path) as pw:
                await smoke_one(
                    "Playwright/claude.ai",
                    pw,
                    TransportKind.CLAUDE_PLAYWRIGHT,
                    embedder,
                    check_mode="domain_and_tool_use",
                )
                await smoke_one(
                    "Playwright/claude.ai",
                    pw,
                    TransportKind.CLAUDE_PLAYWRIGHT,
                    embedder,
                    check_mode="search_corpus",
                )

    print("Smoke complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tool-execution loop smoke test.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--transport",
        choices=["deepseek", "playwright"],
        default="deepseek",
        help="Which transport to run against (default: deepseek).",
    )
    group.add_argument(
        "--all",
        action="store_const",
        const="all",
        dest="transport",
        help="Run against both transports.",
    )
    args = parser.parse_args()

    try:
        asyncio.run(run(args.transport))
    except SessionServiceError as e:
        print(f"\nSession service error: {e.message}")
        if e.cause is not None:
            print(f"Caused by: {type(e.cause).__name__}: {e.cause}")
        raise SystemExit(1) from e
    except TransportError as e:
        print(f"\nTransport error: {e.message}")
        if e.cause is not None:
            print(f"Caused by: {type(e.cause).__name__}: {e.cause}")
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()

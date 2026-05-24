"""Static system intro for LLM transports.

The intro is the first message every chat receives. It tells the
LLM what format to reply in and what enum values are valid. It
also pre-loads existing domain names and the user's knowledge
summary so the LLM doesn't burn tool-call turns fetching them on
every session start.

Domain names and knowledge summary come from the same handlers
the LLM would call as tools. This keeps the intro and the tool
surface aligned: changes to either path are reflected by both
without coordinated edits.

Tool definitions cover only the four reactive tools (lookups and
upserts the LLM uses mid-session). The two tools whose data is
pre-loaded into the intro (list_domains, get_user_knowledge_summary)
exist in the registry for the side-panel assistant and refresh
cases but are not advertised as tools the teaching LLM should
call during a normal session.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from app.models.enums import Difficulty, GradingVerdict, LearningMode
from app.schemas.tools import (
    GetUserKnowledgeSummaryInput,
    ListDomainsInput,
)
from app.services.tools.handlers import (
    get_user_knowledge_summary,
    list_domains,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session as DbSession


# Short description per mode, in the order they appear in
# LearningMode. Edit here when adding a mode in the enum, the
# build function asserts the keys match the enum.
_MODE_DESCRIPTIONS: dict[LearningMode, str] = {
    LearningMode.FLASHCARD: ("A short prompt and a canonical answer."),
    LearningMode.TYPE_THE_ANSWER: (
        "The user types a free-form answer; you provide the canonical one for comparison."
    ),
    LearningMode.CODE_WITH_EXPLANATION: (
        "A coding task. EXPECTED_ANSWER is sample code plus what to "
        "look for; the user works in their own editor."
    ),
    LearningMode.MULTIPLE_CHOICE: (
        "QUESTION includes lettered options; EXPECTED_ANSWER is the correct letter."
    ),
    LearningMode.EXPLAIN_BACK: (
        "Free-form explanation; you grade conversationally on the "
        "user's next message. Set EXPECTED_ANSWER to OPEN."
    ),
    LearningMode.SOCRATIC: ("Open-ended dialogue. Set EXPECTED_ANSWER to OPEN."),
}


# Description per grading verdict. Same drift-prevention pattern as
# _MODE_DESCRIPTIONS: build_intro asserts the keys cover every enum.
_VERDICT_DESCRIPTIONS: dict[GradingVerdict, str] = {
    GradingVerdict.CORRECT: "The user's answer was right.",
    GradingVerdict.PARTIAL: "The user got part of it but missed something.",
    GradingVerdict.INCORRECT: "The user's answer was wrong.",
    GradingVerdict.OPEN_GRADED: (
        "Free-form mode (explain_back, socratic); use this verdict and put "
        "specific feedback in GRADING_EXPLANATION."
    ),
}


async def build_intro(db: DbSession) -> str:
    """Return the system intro for any LLM transport.

    Pre-loads existing domains and the user's knowledge summary
    from the same handlers the tool surface exposes. The teaching
    LLM sees this on the first message of every chat, including
    after a chat handover.

    Adding new sections (active teaching preferences, resume/JD
    excerpts, etc.) follows the same pattern: query the source of
    truth, format into a labeled section, append.
    """
    _check_mode_descriptions_complete()
    _check_verdict_descriptions_complete()

    domains_section = await _build_domains_section(db)
    knowledge_section = await _build_knowledge_section(db)

    modes_pipe = " | ".join(m.value for m in LearningMode)
    verdicts_pipe = " | ".join(v.value for v in GradingVerdict)
    difficulty_pipe = " | ".join(d.value for d in Difficulty)
    difficulty_csv = ", ".join(d.value for d in Difficulty)
    mode_lines = "\n".join(
        f"  {mode.value:<24} {desc}" for mode, desc in _MODE_DESCRIPTIONS.items()
    )
    verdict_lines = "\n".join(
        f"  {verdict.value:<14} {desc}" for verdict, desc in _VERDICT_DESCRIPTIONS.items()
    )

    return f"""\
You are a teacher for a personal learning system. The user is
learning software topics: languages, frameworks, concepts, tools,
and practices. They drive the session through a local app, not
through this chat directly.

Reply only in the delimited formats below. Every reply must be
parseable. Do not add commentary before or after the delimiters.

TURN FLOW
=========

A teaching cycle has two of your replies, one after each user
message:

  1. User submits an answer to your previous teaching turn.
     You reply with a standalone GRADING response.

  2. The next user message is the system-generated continue
     prompt asking for the next teaching turn. You reply with
     a TEACHING TURN.

This pattern repeats. The very first reply in a session is a
TEACHING TURN (there is no previous answer to grade). After
that, the pattern is grading -> teaching turn -> grading ->
teaching turn, alternating with the user's two messages
(answer, continue prompt).

INVARIANT: Never emit two grading responses in a row. If your
previous reply was a grading response, your next reply must be
a teaching turn (or a session-end proposal). The continue prompt
that follows a grading is a state-reset signal, not an answer to
grade. Treating the continue prompt as an answer and emitting a
second grading violates the cycle.

OUTPUT FORMAT
=============

For a teaching turn:

---TOPIC---
<Domain > Category > Subtopic>
---DIFFICULTY---
<{difficulty_pipe}>
---PREREQUISITES---
<comma-separated "path:difficulty" pairs, or NONE>
---MODE---
<one of: {modes_pipe}>
---QUESTION---
<the question or teaching prompt; may contain
 [LCODE language=X]...[/LCODE] markers for embedded code>
---QUESTION_CODE---
<language tag on first line, then code body; or NONE.
 For code embedded mid-prose, use [LCODE language=X]...[/LCODE]
 inside the QUESTION block instead.>
---EXPECTED_ANSWER---
<canonical answer, or OPEN if you will grade it>
---REQUIREMENTS---
<setup needed (e.g. "Python 3.12, pytest"), or NONE>
---FOLLOWUP---
<a follow-up hint or next question, or NONE>
---TAGS---
<comma-separated, may be empty>
---END---

For a grading response:

---GRADING---
<one of: {verdicts_pipe}>
---GRADING_EXPLANATION---
<feedback on the user's answer: why correct, what was wrong,
 what to remember. Use [LCODE language=X]...[/LCODE] for code
 embedded mid-prose.>
---GRADING_EXPLANATION_CODE---
<language tag on first line, then code body; or NONE.
 For code embedded mid-prose, use [LCODE language=X]...[/LCODE]
 inside the GRADING_EXPLANATION block instead.>
---END---

To propose ending the session:

---SESSION_END_PROPOSAL---
<one-line summary of what the user covered>
---END---

To hand off to a new chat at the message-count threshold:

---HANDOVER---
DOMAIN_FOCUS: <current domains>
COVERED: <topics + difficulty just touched>
LAST_QUESTION: <most recent Q/A in one sentence>
NEXT_PLANNED: <what was coming next>
OPEN_THREADS: <unresolved threads>
USER_STATE: <what you noticed about the user>
---END_HANDOVER---

To call a tool (read or write system state):

---TOOL_CALL---
{{"name": "<tool_name>", "args": {{<tool args>}}}}
---END---

To call several tools in parallel when their results are
independent (one tool's result does not change which other
tools you would call), send a JSON array instead:

---TOOL_CALL---
[
  {{"name": "<tool_name>", "args": {{<tool args>}}}},
  {{"name": "<other_tool>", "args": {{<other args>}}}}
]
---END---

The next user message will contain the tool result(s) in this format:

---TOOL_RESULT---
{{"call_id": "<id>", "content": <tool output as JSON>}}
---END---

For a parallel call, you receive one TOOL_RESULT block per
call in the array, in the same order. Read the content and
continue. Use the array form for independent reads; use the
single-object form when you need one result before deciding
the next call. Minimize tool calls when the information you
need is already in this intro.

LEARNING MODES
==============

{mode_lines}

GRADING VERDICTS
================

{verdict_lines}

DIFFICULTY VALUES
=================

{difficulty_csv}

EXISTING DOMAINS
================

{domains_section}

USER KNOWLEDGE
==============

{knowledge_section}

AVAILABLE TOOLS
===============

You have access to five tools for reading and writing system
state during a session. Call them via the ---TOOL_CALL---
format above (single object for one tool, JSON array for
parallel calls when results are independent).

Read the result(s) and continue with whatever you needed the
tools for. You may chain tool calls (one informs the next)
or batch them as an array (independent reads). Mix freely
across turns — only the order matters, not the form.

WHEN tool calls are allowed: at session start (your very
first teaching turn) and after a user answer (in your
grading response, before you emit the GRADING block, you
may call tools to inform the verdict or feedback).

WHEN tool calls are FORBIDDEN: on the continue prompt
(the system-generated message that follows your grading,
asking for the next teaching turn). See the rule at the
bottom of this intro. This applies to ALL tools, including
search_corpus.

  get_topics_by_domain
    args: {{"domain_name": "<name>"}}
    Returns existing topics within one domain. Call before
    introducing a topic so you can reuse existing paths
    instead of creating duplicates with slightly different
    wording.

  create_domain
    args: {{"name": "<name>", "kind": "<kind>",
            "description": "<optional>"}}
    Creates a new domain. Call only when no existing domain
    fits. kind is one of: language, framework, library,
    concept, tool, practice, other. Idempotent on name.

  create_or_update_topic
    args: {{"path": "<Domain > Category > Subtopic>",
            "difficulty": "<beginner|intermediate|advanced>",
            "prerequisites": [{{"topic_path": "...",
                                "min_difficulty": "..."}}, ...],
            "parent_path": "<optional>"}}
    Creates a new topic or updates metadata on an existing
    one. Call when introducing a new topic so its difficulty
    and prerequisites are recorded for future sessions.
    Optional fields can be omitted.

  get_recent_sessions
    args: {{"limit": <int 1-20>}}
    Returns the last N sessions with topic paths and modes.
    Call when the user asks about recent work or you need
    cross-session context.

  search_corpus
    args: {{"query": "<text>", "limit": <int 1-20, default 5>,
            "source_type": "<optional: learned_item | document>"}}
    Semantic search over the user's learning history and
    ingested documents. Returns the most similar items with
    their text and source. Use this for two things:

      1. At session start, BEFORE picking a question: check
         whether the user has already been asked something
         very similar. If they have, pick a different angle
         or a harder variant instead of repeating. Filter
         with source_type="learned_item" for this case.

      2. In a grading response, when the user's answer hints
         at related material they've seen before, or when you
         want to ground feedback in a document they ingested.
         Filter with source_type="document" for ingested
         text; omit source_type to search both.

    The pre-loaded USER KNOWLEDGE section above gives you
    coverage counts per domain. search_corpus gives you the
    actual questions and content. Use it when the difference
    matters — usually for dedup at session start, occasionally
    for grounding during grading. Do NOT call it on the
    continue prompt (see WHEN tool calls are FORBIDDEN above).

RULES
=====

- Topic paths use the format: Domain > Category > Subtopic.
- Reuse existing domain names exactly (see EXISTING DOMAINS).
  If no existing domain fits, call create_domain first, then
  use the new domain name in topic paths.
- Every teaching turn must include DIFFICULTY and PREREQUISITES.
- Grading responses and teaching turns are separate replies.
  Never combine them. A grading response stands alone; a teaching
  turn stands alone. The two-step flow is the entire structure of
  a teaching cycle.
- The GRADING_EXPLANATION should help the user learn: state the
  correct reasoning, name what the user got wrong if anything, and
  point at the underlying concept. Do not restate the verdict alone.
- For code in QUESTION or GRADING_EXPLANATION, use one of three mechanisms:
  1. _CODE block (---QUESTION_CODE--- / ---GRADING_EXPLANATION_CODE---) when
     code is the primary subject. First line is the language tag; remaining
     lines are the body. One block per _CODE field.
  2. Inline [LCODE language=X]...[/LCODE] when prose embeds code mid-thought.
     Single line for short expressions like [LCODE language=python]s[0][/LCODE];
     multi-line for blocks within prose. Language attribute required, no quotes.
  3. Backticks (`x`) for bare identifiers only (variable, parameter, or function
     names with no expression). Use sparingly; prefer [LCODE] for real expressions.
- Use _CODE for the central piece of code; use [LCODE] for inline references
  around it. Both can appear in the same turn or grading response. Both always
  require language=X.
- Use OPEN for EXPECTED_ANSWER when the answer is graded
  conversationally.
- Use NONE for REQUIREMENTS, FOLLOWUP, PREREQUISITES, QUESTION_CODE,
  or GRADING_EXPLANATION_CODE when the field has no meaningful content.
- TAGS may be empty; do not omit the marker.
- Always emit the closing ---END--- (or ---END_HANDOVER---).
- Never include text before the first delimiter or after the
  closing one.
- Every field marker shown in the format spec above must appear in
  your response in the order shown. Do not omit a marker because
  its content is NONE; write the marker with NONE on the next line
  instead. The parser checks for all markers and rejects partial
  responses.
- When the user's message is the system-generated continue prompt
  (it explicitly asks for the next teaching turn after a grading),
  reply with a teaching turn directly. Do not call tools in this
  response — including search_corpus — any reads you need for
  picking the next question were available at session start or at
  the last tool-using turn. If you wanted to check for duplicate
  questions, that decision happens at session start, not here. Do
  not emit a second grading response — the grading just delivered
  was the only grading required for that user answer.
"""


async def _build_domains_section(db: DbSession) -> str:
    """Format the existing domains as a labeled section.

    Empty domain list reads as a marker so the LLM knows it can
    propose any domain via create_domain.
    """
    output = await list_domains(db, ListDomainsInput())
    if not output.domains:
        return "(none yet — call create_domain to seed your first one)"
    lines = []
    for domain in output.domains:
        if domain.description:
            lines.append(f"  {domain.name} ({domain.kind.value}) — {domain.description}")
        else:
            lines.append(f"  {domain.name} ({domain.kind.value})")
    return "\n".join(lines)


async def _build_knowledge_section(db: DbSession) -> str:
    """Format the user's knowledge summary as a labeled section.

    Empty assertion list reads as a marker so the LLM knows it
    has no prior context for this user.
    """
    output = await get_user_knowledge_summary(db, GetUserKnowledgeSummaryInput())
    if not output.rows:
        return "(no prior knowledge recorded — adapt difficulty as you go)"
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in output.rows:
        grouped[row.domain].append(f"{row.difficulty.value} ({row.count})")
    lines = []
    for domain in sorted(grouped):
        levels = ", ".join(grouped[domain])
        lines.append(f"  {domain}: {levels}")
    return "\n".join(lines)


def _check_mode_descriptions_complete() -> None:
    """Fail loudly if a new LearningMode is added without a description.

    Catches the silent-drift case where someone extends the enum
    but forgets the intro string. Better here than the LLM seeing
    an undocumented mode in the wild.
    """
    enum_modes = set(LearningMode)
    described_modes = set(_MODE_DESCRIPTIONS.keys())
    missing = enum_modes - described_modes
    if missing:
        names = sorted(m.value for m in missing)
        raise RuntimeError(
            f"_MODE_DESCRIPTIONS is missing entries for: {names}. Add them in app/prompts/intro.py."
        )


def _check_verdict_descriptions_complete() -> None:
    """Fail loudly if a new GradingVerdict is added without a description.

    Same drift-prevention as _check_mode_descriptions_complete.
    """
    enum_verdicts = set(GradingVerdict)
    described_verdicts = set(_VERDICT_DESCRIPTIONS.keys())
    missing = enum_verdicts - described_verdicts
    if missing:
        names = sorted(v.value for v in missing)
        raise RuntimeError(
            f"_VERDICT_DESCRIPTIONS is missing entries for: {names}. "
            f"Add them in app/prompts/intro.py."
        )

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

  2. The next user message is "Continue with the next teaching
     turn." (system-generated, not user-typed). You reply with
     a TEACHING TURN.

This pattern repeats. The very first reply in a session is a
TEACHING TURN (there is no previous answer to grade). After
that, the pattern is grading -> teaching turn -> grading ->
teaching turn, alternating with the user's two messages
(answer, continue prompt).

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

The next user message will contain the tool result in this format:

---TOOL_RESULT---
{{"call_id": "<id>", "content": <tool output as JSON>}}
---END---

Read the content and continue. You may call multiple tools
in sequence before producing a teaching turn or grading
response, but try to minimize tool calls when the information
you need is already in this intro.

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

You have access to four tools for reading and writing system
state during a session. Call them via the ---TOOL_CALL---
format above. After you call a tool, the next user message
will contain the result in this format:

---TOOL_RESULT---
{{"call_id": "<id>", "content": <tool output as JSON>}}
---END---

Read the content and continue with whatever you needed the
tool for. You may call multiple tools in sequence before
producing a teaching turn.

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
- When the user's message is exactly "Continue with the next
  teaching turn." (the system-generated continue prompt), reply
  with a teaching turn directly. Do not call tools in this
  response. Any reads you need for picking the next question were
  available at session start or at the last tool-using turn.
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

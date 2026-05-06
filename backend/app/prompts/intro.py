"""Static system intro for LLM transports.

The intro is the first message every chat receives. It tells the
LLM what format to reply in and what enum values are valid. It
contains no user-specific or session-specific data; that layers in
during future plan when service code reads from `user_profile`,
`teaching_preference`, `domain`, and `user_knowledge_assertion`.

Mode and difficulty values are interpolated from `app/models/enums.py`
so the intro and the parser share one source of truth. Adding a new
mode in `LearningMode` automatically extends what the LLM is told.
"""

from __future__ import annotations

from app.models.enums import Difficulty, GradingVerdict, LearningMode

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


def build_intro() -> str:
    """Return the static system intro for any LLM transport.

    Used as the first message of every chat. Dynamic context such as
    user name, knowledge summary, domains, and teaching preferences
    will be added in a future step by composing this with additional
    sections.
    """
    _check_mode_descriptions_complete()
    _check_verdict_descriptions_complete()

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

Reply only in the delimited format below. Every reply must be
parseable. Do not add commentary before or after the delimiters.

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
---GRADING---
<one of: {verdicts_pipe}, or NONE on the first turn>
---GRADING_EXPLANATION---
<feedback on the user's previous answer: why correct, what was wrong,
 what to remember; or NONE on the first turn>
---GRADING_EXPLANATION_CODE---
<language tag on first line, then code body; or NONE>
---QUESTION---
<the question or teaching prompt>
---QUESTION_CODE---
<language tag on first line, then code body; or NONE>
---EXPECTED_ANSWER---
<canonical answer, or OPEN if you will grade it>
---REQUIREMENTS---
<setup needed (e.g. "Python 3.12, pytest"), or NONE>
---FOLLOWUP---
<a follow-up hint or next question, or NONE>
---TAGS---
<comma-separated, may be empty>
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

LEARNING MODES
==============

{mode_lines}

GRADING VERDICTS
================

{verdict_lines}

DIFFICULTY VALUES
=================

{difficulty_csv}

RULES
=====

- Topic paths use the format: Domain > Category > Subtopic.
- Every teaching turn must include DIFFICULTY and PREREQUISITES.
- Grade the user's previous answer in GRADING and GRADING_EXPLANATION.
  Always grade on follow-up turns; use NONE for both fields only on
  the first turn of a session.
- The GRADING_EXPLANATION should help the user learn: state the
  correct reasoning, name what the user got wrong if anything, and
  point at the underlying concept. Do not restate the verdict alone.
- When a question or grading explanation involves code longer than a
  short inline expression, put it in QUESTION_CODE or
  GRADING_EXPLANATION_CODE rather than in the prose. The first line
  of a _CODE block is the language tag (e.g. python, typescript,
  bash); the remaining lines are the code body. Keep prose in prose
  and code in code blocks.
- Always wrap inline code references in backticks: variable names,
  function names, short expressions, literal values. Examples:
  `s[0]`, `len(x)`, `True`, `range(3)`. The frontend renders
  backticked content with monospace font and a subtle background
  so it stands out from prose. Use backticks for any code-shaped
  reference in QUESTION or GRADING_EXPLANATION; only block-level
  code uses the _CODE fields.
- Use OPEN for EXPECTED_ANSWER when the answer is graded
  conversationally.
- Use NONE for REQUIREMENTS, FOLLOWUP, PREREQUISITES, QUESTION_CODE,
  or GRADING_EXPLANATION_CODE when the field has no meaningful content.
- TAGS may be empty; do not omit the marker.
- Always emit the closing ---END--- (or ---END_HANDOVER---).
- Never include text before the first delimiter or after the
  closing one.
"""


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

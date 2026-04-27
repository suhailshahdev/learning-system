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

from app.models.enums import Difficulty, LearningMode

# Short description per mode, in the order they appear in
# LearningMode. Edit here when adding a mode in the enum; the
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


def build_intro() -> str:
    """Return the static system intro for any LLM transport.

    Used as the first message of every chat. Dynamic context such as
    user name, knowledge summary, domains, and teaching preferences
    will be added in a future step by composing this with additional
    sections.
    """
    _check_mode_descriptions_complete()

    modes_pipe = " | ".join(m.value for m in LearningMode)
    difficulty_pipe = " | ".join(d.value for d in Difficulty)
    difficulty_csv = ", ".join(d.value for d in Difficulty)
    mode_lines = "\n".join(
        f"  {mode.value:<24} {desc}" for mode, desc in _MODE_DESCRIPTIONS.items()
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
---QUESTION---
<the question or teaching prompt>
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

DIFFICULTY VALUES
=================

{difficulty_csv}

RULES
=====

- Topic paths use the format: Domain > Category > Subtopic.
- Every teaching turn must include DIFFICULTY and PREREQUISITES.
- Use OPEN for EXPECTED_ANSWER when the answer is graded
  conversationally.
- Use NONE for REQUIREMENTS, FOLLOWUP, or PREREQUISITES when the
  field has no meaningful content.
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

"""Retest grading intro and per-question prompt.

Used by the retest flow when a user submits an answer. The LLM
receives a self-contained grading task: the question the user
was asked, the canonical expected answer (or None for free-form
modes), the user's new answer. The LLM produces a single
GRADING response and the chat closes.

Each grading call opens a fresh chat. There is no conversation
history across questions in a retest. This trades a small
per-question round-trip cost for radical simplicity: no
handover logic, no chat lifetime management, no
threshold tracking. The retest can be 30 questions or 300,
the per-call structure is identical.

The intro is static text. No domain list, no knowledge summary,
no tool surface. Grading is a closed task and the LLM does not
need to read or write system state.
"""

from __future__ import annotations

from app.models.enums import GradingVerdict

# Description per grading verdict. Same content as in intro.py's
# _VERDICT_DESCRIPTIONS but local here to keep the retest intro
# independent. Drift between the two is unlikely (the verdicts
# themselves rarely change) and the cost of duplication is one
# small dict.
_VERDICT_DESCRIPTIONS: dict[GradingVerdict, str] = {
    GradingVerdict.CORRECT: "The user's answer was right.",
    GradingVerdict.PARTIAL: "The user got part of it but missed something.",
    GradingVerdict.INCORRECT: "The user's answer was wrong.",
    GradingVerdict.OPEN_GRADED: (
        "Free-form mode (explain_back, socratic) where there is no canonical "
        "answer; use this verdict and put your judgment in GRADING_EXPLANATION."
    ),
}


def build_retest_grading_intro() -> str:
    """Return the static intro for a retest grading chat.

    Stateless: same text every call. The per-question payload
    arrives as the chat's first message.
    """
    _check_verdict_descriptions_complete()

    verdicts_pipe = " | ".join(v.value for v in GradingVerdict)
    verdict_lines = "\n".join(
        f"  {verdict.value:<14} {desc}" for verdict, desc in _VERDICT_DESCRIPTIONS.items()
    )

    return f"""\
You are grading a single answer in a retest session. The user is
revisiting a question they were asked before. You will receive
the question, the canonical expected answer (or NONE for
free-form modes), and the user's new answer.

Your reply is a single GRADING response in the format below.
You do not propose follow-up questions. You do not start a new
teaching turn. The retest controls the question stream; your
only job is to judge this one answer.

OUTPUT FORMAT
=============

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

GRADING VERDICTS
================

{verdict_lines}

RULES
=====

- Reply only in the delimited format above. Do not add commentary
  before the first delimiter or after the closing ---END---.
- Every marker must appear in your response in the order shown.
  Write the marker with NONE on the next line if the field has
  no meaningful content (only GRADING_EXPLANATION_CODE may be
  NONE in practice; GRADING_EXPLANATION must always have content).
- When the canonical expected answer is NONE, the question is in
  a free-form mode. Use the open_graded verdict and put your
  judgment of the user's explanation in GRADING_EXPLANATION.
- The GRADING_EXPLANATION should help the user learn from the
  retest, not just label the answer. State the correct reasoning,
  name what the user got wrong, point at the underlying concept.
- Do not output any other format. No teaching turns, no
  tool calls, no session-end proposals, no handovers. The retest
  flow does not use these and the parser will reject them.
"""


def build_retest_grading_prompt(
    *,
    question: str,
    expected_answer: str | None,
    user_answer: str,
) -> str:
    """Return the per-question grading payload sent as the chat's first message.

    Carries the question, the canonical expected answer (or
    NONE for free-form modes), and the user's new answer.
    The LLM sees this immediately after the intro and replies
    with a GRADING response.

    Plain text with labeled sections. No delimiters since this
    is input to the LLM, not parsed output from it. The labeled
    shape keeps the LLM's attention on the right fields without
    burning structured-format complexity.
    """
    expected = expected_answer if expected_answer is not None else "NONE (open-graded mode)"
    return f"""\
QUESTION
========

{question}

CANONICAL EXPECTED ANSWER
=========================

{expected}

USER'S NEW ANSWER
=================

{user_answer}

Grade this answer now. Reply with one GRADING block."""


def _check_verdict_descriptions_complete() -> None:
    """Fail loudly if a new GradingVerdict is added without a description.

    Same drift-prevention as intro.py. Catches the case where
    the enum grows but this module is not updated.
    """
    enum_verdicts = set(GradingVerdict)
    described_verdicts = set(_VERDICT_DESCRIPTIONS.keys())
    missing = enum_verdicts - described_verdicts
    if missing:
        names = sorted(v.value for v in missing)
        raise RuntimeError(
            f"_VERDICT_DESCRIPTIONS is missing entries for: {names}. "
            f"Add them in app/prompts/retest_grading_intro.py."
        )

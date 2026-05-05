"""Per-turn prompt sent on every user answer.

The system intro and the first prompt have already established
the topic and the format. The per-turn prompt just delivers the
user's answer plus a short reminder. The chat already holds
context from prior turns and restating it here would be noise.
"""

from __future__ import annotations


def build_turn_prompt(user_answer: str) -> str:
    """Build the prompt for a follow-up turn.

    The user_answer is the text the user submitted in the local
    app. The reminder anchors the LLM back to the delimited
    format in case earlier turns drifted, and reinforces that
    grading is required on every follow-up turn.
    """
    return f"""\
{user_answer}

Grade the user's answer above in GRADING and GRADING_EXPLANATION.
Then continue with the next teaching turn. Reply in the delimited
format declared in the system intro.
"""

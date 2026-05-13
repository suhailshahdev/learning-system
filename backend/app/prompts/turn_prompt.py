"""Per-turn prompts for the split-roundtrip teaching flow.

After split, a teaching cycle has two LLM round trips:
the user's answer produces a grading response, then the system's
continue prompt produces the next teaching turn. Each round trip
has its own prompt builder.

The system intro and the first prompt have already established
the topic and the format. The per-turn prompts just deliver the
user's input plus a short reminder. The chat already holds
context from prior turns and restating it here would be noise.
"""

from __future__ import annotations


def build_turn_prompt(user_answer: str) -> str:
    """Build the prompt for a user answer.

    The user_answer is the text the user submitted in the local
    app. The reminder anchors the LLM back to the standalone
    GRADING response shape declared in the system intro.
    """
    return f"""\
{user_answer}

Reply with a standalone grading response for the answer above.
Use the ---GRADING--- delimited format declared in the system
intro. Do not include a teaching turn in this response, the
next teaching turn comes after the continue prompt.
"""


def build_continue_prompt() -> str:
    """Build the prompt that triggers the next teaching turn.

    Sent after the user has read the grading response and the
    backend received the continue signal from the frontend. The
    chat already holds context from the grading just emitted,
    this prompt just signals "ready for the next teaching turn."
    """
    return """\
Continue with the next teaching turn. Reply in the standalone
teaching-turn delimited format declared in the system intro.
"""

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
    backend received the continue signal from the frontend.

    Wording is deliberately firm: DeepSeek pro has been observed
    emitting a second grading response in this position when the
    prompt was a polite request. The explicit "do not grade
    again" instruction addresses that compliance failure.
    """
    return """\
The previous grading response is complete and final. The user has
read it. They are not seeking more feedback on their answer.

Your task now is to produce the next teaching turn: a new question
on the same topic or a related one, in the standalone teaching-turn
format.

Your reply MUST begin with the ---TOPIC--- delimiter. If your reply
begins with ---GRADING--- you have misread the cycle — the grading
phase ended with your previous response and the cycle is now at the
teaching-turn phase.

Format: ---TOPIC--- ... ---END--- as declared in the system intro.
Do not produce any other response kind.
"""

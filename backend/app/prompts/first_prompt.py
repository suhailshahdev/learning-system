"""First user prompt sent after the system intro.

The intro tells the LLM how to format replies. The first prompt
tells it what to teach. Together they bootstrap a session.
"""

from __future__ import annotations


def build_first_prompt(topic_path: str) -> str:
    """Build the first prompt for a fresh session.

    The LLM has already received the system intro by the time this
    prompt is sent. The reply must conform to the delimited format
    declared in the intro, which the parser enforces.
    """
    return f"""\
Begin teaching the topic: {topic_path}

Pick a starting question at a difficulty appropriate for someone
just starting on this topic. Pick whichever learning mode best
fits the question.

Reply in the delimited format declared in the system intro.
"""

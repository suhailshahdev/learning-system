"""Prompt asking the dying chat to emit a handover block.

Sent as the next user turn in a chat that has reached its
message-count threshold. The reply is parsed by parse_response
and consumed by the session engine when opening the next chat.
"""

from __future__ import annotations


def build_handover_request() -> str:
    """Build the handover-request message for the dying chat.

    The message tells the LLM that the session is continuing in a
    new chat and that it should produce a summary the next chat
    can use to pick up where this one left off. The format is the
    one declared in the system intro and the parser enforces it.
    """
    return """\
We are starting a new chat for this session and need a handover
so the next chat can continue without losing context. Produce a
handover block summarizing the session so far.

Use this exact format:

---HANDOVER---
DOMAIN_FOCUS: <current domain or domains>
COVERED: <topics covered with difficulty, brief>
LAST_QUESTION: <the most recent question and the user's answer, one sentence each>
NEXT_PLANNED: <what was coming next>
OPEN_THREADS: <unresolved threads, or NONE>
USER_STATE: <anything notable about how the user is progressing>
---END_HANDOVER---

Reply with the handover block only. No other text before or
after."""

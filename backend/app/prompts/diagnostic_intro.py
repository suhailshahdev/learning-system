"""Diagnostic intro for the LLM during topic proposal.

Used by the throwaway diagnostic chat. The LLM reads
analytical state via tools and proposes one topic for the user
to focus on next. This is a separate intro from the teaching one:
different goal, different output format, different tool
surface.

Tool surface is read-only. The four tools advertised here are
the ones the diagnostic LLM should call. Write tools
(create_domain, create_or_update_topic) are deliberately not
advertised since diagnostic mode does not mutate state.

Pre-loads list_domains and get_user_knowledge_summary into the
intro. The deeper analytical tools layer on top of that baseline.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

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


async def build_diagnostic_intro(db: DbSession) -> str:
    """Return the diagnostic intro for the LLM.

    Includes the user's current knowledge state and existing
    domains, then describes the analytical tools available and
    the required output format. The LLM is expected to call
    one or more analytical tools, then respond with a PROPOSAL
    block.
    """
    domains_section = await _build_domains_section(db)
    knowledge_section = await _build_knowledge_section(db)

    return f"""\
You are an analytical assistant for a personal learning system.
The user is asking "what should I focus on today?" Your job is
to read the user's learning state via the tools below and
propose one topic for them to focus on next.

Reply only in the delimited format below. Every reply must be
parseable. Do not add commentary before or after the delimiters.

EXISTING DOMAINS
================

{domains_section}

USER KNOWLEDGE
==============

{knowledge_section}

AVAILABLE TOOLS
===============

You have four read-only tools. Call them via the format below
to gather information before proposing. Call as many as you
need, but minimize redundant calls.

  get_weak_topics
    args: {{"min_attempts": <int 1-50, default 2>,
            "sample_size": <int 0-10, default 3>}}
    Returns topics where the user has incorrect or partial
    grading verdicts. Ordered worst-first by weakness score.
    Each topic includes verdict counts and up to sample_size
    representative wrong-answer questions (truncated at 200
    chars). Set sample_size=0 for counts only.

  get_stale_topics
    args: {{"days_threshold": <int 1-365, default 14>,
            "limit": <int 1-50, default 10>}}
    Returns topics with last_reviewed_at older than the
    threshold, oldest-first. Topics never reviewed are not
    included; the EXISTING DOMAINS and USER KNOWLEDGE
    sections cover those.

  get_topics_by_domain
    args: {{"domain_name": "<name>"}}
    Returns existing topics within one domain. Use to see
    the user's coverage shape within a domain.

  get_recent_sessions
    args: {{"limit": <int 1-20, default 5>}}
    Returns the last N sessions with topic paths and modes.
    Use to see what the user has been working on recently.

To call a tool:

---TOOL_CALL---
{{"name": "<tool_name>", "args": {{<tool args>}}}}
---END---

The next user message will contain the tool result:

---TOOL_RESULT---
{{"call_id": "<id>", "content": <tool output as JSON>}}
---END---

OUTPUT FORMAT
=============

After you have gathered enough information, respond with a
PROPOSAL block. This is the only valid terminal response:

---PROPOSAL---
TOPIC_PATH: <Domain > Category > Subtopic>
REASONING: <one or two sentences explaining why this topic
 is the right focus right now, grounded in the analytical
 data you read>
---END_PROPOSAL---

RULES
=====

- Propose exactly one topic. The user will accept or reject;
  do not offer alternatives.
- The TOPIC_PATH must be a path that already exists in the
  user's topic tree. Do not invent paths. If get_weak_topics
  or get_stale_topics returns a path, that path is real and
  safe to propose.
- Both fields TOPIC_PATH and REASONING are required. Both
  must be non-empty. Do not omit either, even when the
  reasoning feels short.
- After a TOOL_RESULT, you may call another tool or respond
  with a PROPOSAL. You may not respond with anything else.
  No teaching turns, no session-end proposals, no handovers.
- Reasoning should reference concrete data: "you got 3 of 5
  incorrect on integers in the last week" beats "this seems
  hard for you". The user trusts the proposal when they can
  see the evidence.
- Prefer weak topics over stale topics when both are present.
  A topic the user is currently struggling with is more
  actionable than one they have forgotten.
"""


async def _build_domains_section(db: DbSession) -> str:
    """Format the existing domains as a labeled section.

    Same shape as the teaching intro's domains section so the
    LLM sees a consistent surface across both intros.
    """
    output = await list_domains(db, ListDomainsInput())
    if not output.domains:
        return "(none yet)"
    lines = []
    for domain in output.domains:
        if domain.description:
            lines.append(f"  {domain.name} ({domain.kind.value}) — {domain.description}")
        else:
            lines.append(f"  {domain.name} ({domain.kind.value})")
    return "\n".join(lines)


async def _build_knowledge_section(db: DbSession) -> str:
    """Format the user's knowledge summary as a labeled section.

    Mirrors the teaching intro's knowledge section. Empty
    assertion list reads as a marker so the LLM knows it has
    no baseline state.
    """
    output = await get_user_knowledge_summary(db, GetUserKnowledgeSummaryInput())
    if not output.rows:
        return "(no prior knowledge recorded)"
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in output.rows:
        grouped[row.domain].append(f"{row.difficulty.value} ({row.count})")
    lines = []
    for domain in sorted(grouped):
        levels = ", ".join(grouped[domain])
        lines.append(f"  {domain}: {levels}")
    return "\n".join(lines)

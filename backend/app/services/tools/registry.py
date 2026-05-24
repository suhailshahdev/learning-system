"""Tool name to handler mapping.

The registry is the single source of truth for which tools exist
and how to invoke them. Both transports route ToolCall values
through `execute_tool_call`, which dispatches on the discriminated
`name` field to the right handler.

Handler signatures match the discriminated union: each handler
takes its corresponding input type and returns its corresponding
output type. mypy verifies the registry mapping at build time
through the explicit `cast` in execute_tool_call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.schemas.tools import (
    CreateDomainCall,
    CreateOrUpdateTopicCall,
    GetRecentSessionsCall,
    GetStaleTopicsCall,
    GetTopicsByDomainCall,
    GetUserKnowledgeSummaryCall,
    GetWeakTopicsCall,
    ListDomainsCall,
    SearchCorpusCall,
    ToolCall,
)
from app.services.tools.handlers import (
    create_domain,
    create_or_update_topic,
    get_recent_sessions,
    get_stale_topics,
    get_topics_by_domain,
    get_user_knowledge_summary,
    get_weak_topics,
    list_domains,
    search_corpus_tool,
)

if TYPE_CHECKING:
    from pydantic import BaseModel
    from sqlalchemy.orm import Session as DbSession

    from app.services.embedding_service import Embedder


# Map from tool name (the discriminator) to the handler function.
# Static dict because the surface is fixed. If tools become
# pluggable in the future, this becomes a registry class with
# register/unregister methods.
# Name -> handler for the (db, args) handlers. search_corpus is not
# here: it needs an embedder, so it has a different call shape and is
# dispatched directly in execute_tool_call's match block. If a third
# dependency-carrying handler appears, this split should become a
# ToolContext passed to every handler.
HANDLERS = {
    "list_domains": list_domains,
    "create_domain": create_domain,
    "get_topics_by_domain": get_topics_by_domain,
    "create_or_update_topic": create_or_update_topic,
    "get_user_knowledge_summary": get_user_knowledge_summary,
    "get_recent_sessions": get_recent_sessions,
    "get_weak_topics": get_weak_topics,
    "get_stale_topics": get_stale_topics,
}


async def execute_tool_call(db: DbSession, call: ToolCall, embedder: Embedder) -> BaseModel:  # noqa: PLR0911
    """Dispatch a ToolCall to the matching handler.

    Pydantic narrows `call` based on its discriminator. The match
    block lets each branch pass the type-correct input shape to
    its handler. Returns the handler's output as a Pydantic model,
    the caller serializes it to whatever wire format the transport
    needs.

    embedder is needed only by search_corpus, the one handler that
    embeds its query. The other handlers ignore it.
    """
    match call:
        case ListDomainsCall():
            return await list_domains(db, call.args)
        case CreateDomainCall():
            return await create_domain(db, call.args)
        case GetTopicsByDomainCall():
            return await get_topics_by_domain(db, call.args)
        case CreateOrUpdateTopicCall():
            return await create_or_update_topic(db, call.args)
        case GetUserKnowledgeSummaryCall():
            return await get_user_knowledge_summary(db, call.args)
        case GetRecentSessionsCall():
            return await get_recent_sessions(db, call.args)
        case GetWeakTopicsCall():
            return await get_weak_topics(db, call.args)
        case GetStaleTopicsCall():
            return await get_stale_topics(db, call.args)
        case SearchCorpusCall():
            return await search_corpus_tool(db, call.args, embedder)

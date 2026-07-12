"""DeepSeek chat completions API transport implementation.

Stateless HTTP transport for the DeepSeek chat completions API.
Each send posts the full message history and gets back the next
assistant reply. The handle holds the history locally since there
is no server-side chat to manage.

The endpoint is OpenAI-compatible. We use httpx directly instead
of the OpenAI SDK since there is only one endpoint with a fixed
request shape.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Self

import httpx
from pydantic import BaseModel, TypeAdapter, ValidationError

from app.schemas.tools import (
    CreateDomainInput,
    CreateOrUpdateTopicInput,
    GetRecentSessionsInput,
    GetStaleTopicsInput,
    GetTopicsByDomainInput,
    GetWeakTopicsInput,
    ToolCall,
)
from app.transport.base import (
    ChatResumeMetadata,
    ToolResult,
    TransportError,
    TransportResponse,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import TracebackType

    from app.transport.base import LLMTransport


DEEPSEEK_BASE_URL = "https://api.deepseek.com"
CHAT_COMPLETIONS_PATH = "/chat/completions"

CONNECT_TIMEOUT_S = 10.0
READ_TIMEOUT_S = 120.0
TOTAL_TIMEOUT_S = 120.0


# Module-level adapter so the discriminated-union validator is built
# once. Same pattern as the parser uses.
_TOOL_CALL_ADAPTER: TypeAdapter[ToolCall] = TypeAdapter(ToolCall)


# Every tool this transport can advertise natively, keyed by name.
# Which subset a given chat advertises is the caller's decision via
# start_new_chat/resume_chat tool_names; the catalog only supplies
# the definitions. The two pre-loaded tools (list_domains,
# get_user_knowledge_summary) are absent because their data ships in
# the intro already.
_TOOL_CATALOG: dict[str, tuple[str, type[BaseModel]]] = {
    "get_topics_by_domain": (
        "Returns existing topics within one domain. "
        "Call before introducing a topic to reuse paths.",
        GetTopicsByDomainInput,
    ),
    "create_domain": (
        "Creates a new domain. Idempotent on name. Call only when no existing domain fits.",
        CreateDomainInput,
    ),
    "create_or_update_topic": (
        "Upserts a topic by path. Records difficulty, prerequisites, and parent_path.",
        CreateOrUpdateTopicInput,
    ),
    "get_recent_sessions": (
        "Returns the last N sessions with topic paths and modes.",
        GetRecentSessionsInput,
    ),
    "get_weak_topics": (
        "Returns topics with incorrect or partial grading verdicts, "
        "ordered worst-first by weakness score.",
        GetWeakTopicsInput,
    ),
    "get_stale_topics": (
        "Returns topics whose last review is older than the threshold, oldest-first.",
        GetStaleTopicsInput,
    ),
}


def _tools_param_for(tool_names: Sequence[str]) -> list[dict[str, Any]]:
    """Build the API's `tools` parameter for the named tools.

    OpenAI-compatible APIs expect each tool as:
        {"type": "function",
         "function": {"name": ..., "description": ..., "parameters": <JSON schema>}}

    An unknown name is a caller bug: fail loudly at chat open rather
    than advertise a partial surface.
    """
    unknown = [name for name in tool_names if name not in _TOOL_CATALOG]
    if unknown:
        raise TransportError(f"Unknown tool names for DeepSeek advertisement: {unknown!r}")
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": _TOOL_CATALOG[name][0],
                "parameters": _TOOL_CATALOG[name][1].model_json_schema(),
            },
        }
        for name in tool_names
    ]


Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class Message:
    """One message in a DeepSeek chat history.

    Frozen so handles can be safely shared across awaits without
    accidental mutation. The `role` literal matches the API's
    expected values, mypy catches typos at the call site.

    For role="tool" messages, tool_call_id is the id from the
    earlier assistant tool_call this message responds to. Required
    by the API for correlation. None for non-tool messages.

    For role="assistant" messages that contain tool_calls,
    raw_tool_calls preserves the API's response shape so the
    history can be replayed verbatim. None when the assistant
    message was plain text.
    """

    role: Role
    content: str
    tool_call_id: str | None = None
    raw_tool_calls: list[dict[str, Any]] | None = None

    def to_wire(self) -> dict[str, Any]:
        """Serialize to the JSON shape the API expects."""
        wire: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_call_id is not None:
            wire["tool_call_id"] = self.tool_call_id
        if self.raw_tool_calls is not None:
            wire["tool_calls"] = self.raw_tool_calls
        return wire


@dataclass
class DeepseekChatHandle:
    """Per-chat state for the DeepSeek transport.

    Holds the running message history and the model name for this
    chat. message_count mirrors the same field on the Playwright
    handle so the session engine can read it the same way on either
    transport.
    """

    model: str
    history: list[Message] = field(default_factory=list)
    message_count: int = 0
    # The chat's native tool surface, built once at open/resume from
    # the caller's tool_names. None advertises nothing. Rides every
    # API call for the chat's lifetime so the offer stays constant.
    tools: list[dict[str, Any]] | None = None


class DeepseekTransport:
    """Chat completions API transport for DeepSeek.

    Owns one long-lived HTTP client for the transport's lifetime so
    connections are reused. Use as an async context manager or call
    start() and shutdown() explicitly.
    """

    def __init__(self, api_key: str, default_model: str = "deepseek-v4-flash") -> None:
        self._api_key = api_key
        self._default_model = default_model
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.shutdown()

    async def start(self) -> None:
        """Open the HTTP client used for every request."""
        timeout = httpx.Timeout(
            timeout=TOTAL_TIMEOUT_S,
            connect=CONNECT_TIMEOUT_S,
            read=READ_TIMEOUT_S,
        )
        self._client = httpx.AsyncClient(
            base_url=DEEPSEEK_BASE_URL,
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )

    async def shutdown(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def start_new_chat(
        self,
        system_intro: str,
        first_message: str,
        tool_names: Sequence[str] | None = None,
    ) -> tuple[DeepseekChatHandle, TransportResponse]:
        if self._client is None:
            raise TransportError("Transport not started. Call start() first.")

        handle = DeepseekChatHandle(model=self._default_model)
        if tool_names is not None:
            handle.tools = _tools_param_for(tool_names)
        handle.history.append(Message(role="system", content=system_intro))
        response = await self._send_and_capture(handle, first_message)
        return handle, response

    async def resume_chat(
        self, metadata: ChatResumeMetadata, tool_names: Sequence[str] | None = None
    ) -> DeepseekChatHandle:
        if self._client is None:
            raise TransportError("Transport not started. Call start() first.")

        if not metadata.prior_messages:
            raise TransportError("Cannot resume DeepSeek chat with empty prior_messages.")

        handle = DeepseekChatHandle(model=self._default_model)
        if tool_names is not None:
            handle.tools = _tools_param_for(tool_names)
        handle.history = [Message(role=m.role, content=m.content) for m in metadata.prior_messages]
        handle.message_count = metadata.message_count
        return handle

    async def send(self, chat: DeepseekChatHandle, message: str) -> TransportResponse:
        if self._client is None:
            raise TransportError("Transport not started. Call start() first.")
        return await self._send_and_capture(chat, message)

    async def close(self, chat: DeepseekChatHandle) -> None:
        # Stateless API with no per-chat resources. This method exists only
        # to satisfy the protocol uniformly, the handle is cleaned up by the
        # caller.
        return None

    async def send_tool_results(
        self, chat: DeepseekChatHandle, results: list[ToolResult]
    ) -> TransportResponse:
        """Send tool execution results back as `tool` role messages.

        Required by the OpenAI-compatible API after the assistant
        responded with tool_calls: each tool result must be sent
        as a separate message with role="tool" and the matching
        tool_call_id. The next response can be plain text or
        another round of tool_calls.
        """
        if self._client is None:
            raise TransportError("Transport not started. Call start() first.")

        next_history = [
            *chat.history,
            *[
                Message(role="tool", content=result.content, tool_call_id=result.call_id)
                for result in results
            ],
        ]
        return await self._post_chat_completion(chat, next_history)

    async def _send_and_capture(self, chat: DeepseekChatHandle, message: str) -> TransportResponse:
        """Post one user turn, append the assistant reply to history.

        Mutates the handle's history with both the new user message
        and the assistant response on success. On failure the history
        is left as it was so the caller can retry.

        The assistant response may be plain text (returned as
        TransportResponse.text) or a list of tool calls (returned as
        TransportResponse.tool_calls). The two are mutually exclusive
        in any single API response.
        """
        next_history = [*chat.history, Message(role="user", content=message)]
        return await self._post_chat_completion(chat, next_history)

    async def _post_chat_completion(
        self, chat: DeepseekChatHandle, next_history: list[Message]
    ) -> TransportResponse:
        """Post a chat completion request and parse the response.

        Shared by _send_and_capture (for user messages) and
        send_tool_results (for tool result messages). Encapsulates
        the API contract: payload shape, error handling, and
        parsing the response into either text or tool calls.
        """
        if self._client is None:
            raise TransportError("Transport not started.")

        # Thinking mode requires the caller to preserve `reasoning_content`
        # across tool-call round trips within a single user turn. The helper
        # loop in session_service does not currently track that field, so
        # any tool-using session against a thinking-mode model fails on the
        # second API call with HTTP 400. Disabling thinking mode keeps the
        # rest of the architecture working. Trigger to revisit: LLM tool-
        # selection quality drops measurably and chain-of-thought is the
        # likely fix.
        payload: dict[str, Any] = {
            "model": chat.model,
            "messages": [m.to_wire() for m in next_history],
            "stream": False,
            "thinking": {"type": "disabled"},
        }
        # Omit the key entirely when the chat advertises nothing; an
        # empty tools list is not the same thing to the API.
        if chat.tools is not None:
            payload["tools"] = chat.tools

        try:
            response = await self._client.post(CHAT_COMPLETIONS_PATH, json=payload)
            response.raise_for_status()
            data: dict[str, Any] = response.json()
        except httpx.TimeoutException as e:
            raise TransportError("DeepSeek request timed out.", cause=e) from e
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            body = e.response.text[:200]
            raise TransportError(f"DeepSeek HTTP {status}: {body}", cause=e) from e
        except httpx.RequestError as e:
            raise TransportError("Network error reaching DeepSeek.", cause=e) from e
        except ValueError as e:
            raise TransportError("DeepSeek returned malformed JSON.", cause=e) from e

        choices = data.get("choices") or []
        if not choices:
            raise TransportError("DeepSeek returned no choices.")

        try:
            message_data = choices[0]["message"]
        except (KeyError, TypeError) as e:
            raise TransportError("DeepSeek response missing choices[0].message.", cause=e) from e

        # The API returns either content (text) or tool_calls or both.
        # In practice, when tool_calls is present, content is null or
        # empty. We surface tool_calls when present and ignore any
        # accompanying preamble text.
        raw_tool_calls = message_data.get("tool_calls")
        if raw_tool_calls:
            tool_calls = self._parse_tool_calls(raw_tool_calls)
            chat.history = [
                *next_history,
                Message(
                    role="assistant",
                    content=message_data.get("content") or "",
                    raw_tool_calls=raw_tool_calls,
                ),
            ]
            chat.message_count += 1
            return TransportResponse(text="", tool_calls=tool_calls)

        assistant_text = message_data.get("content")
        if not isinstance(assistant_text, str):
            raise TransportError(
                f"DeepSeek returned non-string content: {type(assistant_text).__name__}."
            )

        chat.history = [*next_history, Message(role="assistant", content=assistant_text)]
        chat.message_count += 1
        return TransportResponse(text=assistant_text)

    def _parse_tool_calls(self, raw_tool_calls: list[dict[str, Any]]) -> list[ToolCall]:
        """Validate API tool_calls into the discriminated ToolCall union.

        DeepSeek's API returns each tool call as:
            {"id": "...", "type": "function",
             "function": {"name": "...", "arguments": "<JSON string>"}}

        We extract id, name, and arguments and feed through the same
        TypeAdapter the parser uses for Claude transport, so both
        paths converge on identical ToolCall values. The id is
        required when sending tool results back to the API as
        tool_call_id, so any missing id is a hard error here.
        """
        out: list[ToolCall] = []
        for raw in raw_tool_calls:
            try:
                call_id = raw["id"]
                function = raw["function"]
                name = function["name"]
                arguments_str = function["arguments"]
            except (KeyError, TypeError) as e:
                raise TransportError(f"Malformed tool_call from DeepSeek: {raw!r}", cause=e) from e

            try:
                args = json.loads(arguments_str)
            except json.JSONDecodeError as e:
                raise TransportError(
                    f"Tool call {name!r} has invalid JSON arguments: {e.msg}", cause=e
                ) from e

            try:
                call = _TOOL_CALL_ADAPTER.validate_python(
                    {"name": name, "args": args, "id": call_id}
                )
            except ValidationError as e:
                raise TransportError(
                    f"Tool call {name!r} failed schema validation: {e}", cause=e
                ) from e

            out.append(call)
        return out


_: type[LLMTransport[DeepseekChatHandle]] = DeepseekTransport

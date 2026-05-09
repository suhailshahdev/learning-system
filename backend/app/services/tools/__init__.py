"""LLM tool surface for system actions.

Six tools let the teaching LLM read and write system state during
a session. Tools are for actions on system state (read domains,
upsert topics, read knowledge summaries), not for grading or
question generation, which stay as structured text via ParsedTurn.

The registry maps tool names to async handler functions. Both
transports converge on this registry: DeepSeek normalizes native
function-call API responses into ToolCall values, and the Claude
transport's parser extracts ---TOOL_CALL--- blocks into the same
shape. One handler implementation serves both paths.

Each handler is responsible for its own commit/rollback. Tool
calls are agent actions that should persist independently of the
teaching turn that triggered them.
"""

from app.services.tools.registry import HANDLERS, execute_tool_call

__all__ = ["HANDLERS", "execute_tool_call"]

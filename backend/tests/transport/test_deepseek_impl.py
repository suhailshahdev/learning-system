"""Tests for DeepSeek's native tool advertisement.

The catalog must cover every tool a service can expose through a
per-chat surface, otherwise the transport rejects the chat before
the model receives it.
"""

from app.schemas.tools import SearchCorpusInput
from app.transport import deepseek_impl


def test_search_corpus_has_native_tool_schema() -> None:
    tools = deepseek_impl._tools_param_for(("search_corpus",))

    assert len(tools) == 1
    function = tools[0]["function"]
    assert function["name"] == "search_corpus"
    assert function["parameters"] == SearchCorpusInput.model_json_schema()

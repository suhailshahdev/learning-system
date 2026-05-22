"""Tests for the embedding service.

Covers embedder logic (batching, response ordering, dimension and
count validation, error wrapping) and embed_records' row construction.
Does not test vector persistence: the embedding column is a pgvector
type with no SQLite equivalent, so the conftest in-memory SQLite DB
cannot store it. Persistence against real Postgres is covered by the
smoke. These tests verify everything up to the write.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
import pytest
from app.models import EmbeddingSourceType
from app.models.embedding import EMBEDDING_DIM
from app.services.embedding_service import (
    EmbeddingError,
    EmbeddingRecord,
    OpenRouterEmbedder,
    embed_records,
)

from tests.services.fakes import FakeEmbedder

if TYPE_CHECKING:
    from sqlalchemy.orm import Session as DbSession


def _vec(fill: float = 0.1) -> list[float]:
    """A correctly-sized vector for canned responses."""
    return [fill] * EMBEDDING_DIM


def _mock_embedder(handler: httpx.MockTransport) -> OpenRouterEmbedder:
    """An OpenRouterEmbedder whose client routes through a mock transport."""
    embedder = OpenRouterEmbedder(api_key="test-key", model="openai/text-embedding-3-small")
    embedder._client = httpx.AsyncClient(
        base_url="https://openrouter.ai/api/v1",
        transport=handler,
    )
    return embedder


async def test_embed_texts_returns_input_order_despite_shuffled_response() -> None:
    """The API may return items out of index order, we sort by index."""

    def respond(_request: httpx.Request) -> httpx.Response:
        # Return items deliberately out of order: index 1 before index 0.
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": _vec(0.2)},
                    {"index": 0, "embedding": _vec(0.1)},
                ]
            },
        )

    embedder = _mock_embedder(httpx.MockTransport(respond))
    vectors = await embedder.embed_texts(["first", "second"])

    assert len(vectors) == 2
    assert vectors[0][0] == pytest.approx(0.1)  # index 0 -> first
    assert vectors[1][0] == pytest.approx(0.2)  # index 1 -> second


async def test_embed_texts_empty_returns_empty_without_calling_api() -> None:
    """Empty input short-circuits; no HTTP call is made."""

    def fail(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("API should not be called for empty input.")

    embedder = _mock_embedder(httpx.MockTransport(fail))
    assert await embedder.embed_texts([]) == []


async def test_embed_texts_wrong_dimension_raises() -> None:
    """A vector of the wrong width is a hard error."""

    def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]})

    embedder = _mock_embedder(httpx.MockTransport(respond))
    with pytest.raises(EmbeddingError, match="dim vector, expected"):
        await embedder.embed_texts(["x"])


async def test_embed_texts_count_mismatch_raises() -> None:
    """Fewer embeddings than inputs is a hard error."""

    def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": _vec()}]})

    embedder = _mock_embedder(httpx.MockTransport(respond))
    with pytest.raises(EmbeddingError, match="embeddings for 2 inputs"):
        await embedder.embed_texts(["a", "b"])


async def test_embed_texts_http_error_wraps() -> None:
    """An HTTP error status becomes an EmbeddingError with the status."""

    def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="rate limited")

    embedder = _mock_embedder(httpx.MockTransport(respond))
    with pytest.raises(EmbeddingError, match="HTTP 429"):
        await embedder.embed_texts(["x"])


async def test_embed_texts_malformed_json_wraps() -> None:
    """A non-JSON body becomes an EmbeddingError."""

    def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json")

    embedder = _mock_embedder(httpx.MockTransport(respond))
    with pytest.raises(EmbeddingError, match="malformed JSON"):
        await embedder.embed_texts(["x"])


async def test_embed_texts_not_started_raises() -> None:
    """Calling embed_texts before start() is a clear error."""
    embedder = OpenRouterEmbedder(api_key="k", model="m")
    with pytest.raises(EmbeddingError, match="not started"):
        await embedder.embed_texts(["x"])


async def test_embed_records_builds_rows_with_source_and_version(db: DbSession) -> None:
    """embed_records builds one row per record, stamped with model_version.

    Uses FakeEmbedder so no network. Does not assert vector round-trip,
    inspects the returned Embedding objects. Whether the SQLite flush
    tolerates the vector column is what this test establishes.
    """
    records = [
        EmbeddingRecord(
            source_type=EmbeddingSourceType.LEARNED_ITEM,
            source_id="item-1",
            content="what is a list comprehension",
        ),
        EmbeddingRecord(
            source_type=EmbeddingSourceType.DOCUMENT_CHUNK,
            source_id="doc-1",
            content="a chunk of pasted notes",
        ),
    ]
    rows = await embed_records(db=db, embedder=FakeEmbedder(), records=records)

    assert len(rows) == 2
    assert rows[0].source_type is EmbeddingSourceType.LEARNED_ITEM
    assert rows[0].source_id == "item-1"
    assert rows[0].embedding_model_version == "fake-embedder-v1"
    assert len(rows[0].embedding) == EMBEDDING_DIM
    assert rows[1].source_type is EmbeddingSourceType.DOCUMENT_CHUNK


async def test_embed_records_empty_returns_empty() -> None:
    """No records means no work and no rows."""
    assert await embed_records(db=None, embedder=FakeEmbedder(), records=[]) == []  # type: ignore[arg-type]


async def test_embed_records_propagates_embedder_failure() -> None:
    """A failing embedder raises EmbeddingError out of embed_records.

    This is the error-isolation contract approve_session relies on:
    embed_records raises, the caller catches and logs.
    """
    records = [
        EmbeddingRecord(
            source_type=EmbeddingSourceType.LEARNED_ITEM,
            source_id="item-1",
            content="x",
        )
    ]
    with pytest.raises(EmbeddingError, match="forced failure"):
        await embed_records(db=None, embedder=FakeEmbedder(raise_on_embed=True), records=records)  # type: ignore[arg-type]

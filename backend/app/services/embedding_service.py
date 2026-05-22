"""Embedding service for semantic retrieval.

Turns text into vectors and writes them to the embedding table.
Used on two paths: embed-on-approve (new learned items, called
post-commit from the session service) and backfill (existing items,
called from a script). Both go through embed_records so the write
shape is identical.

The Embedder protocol abstracts the provider. OpenRouterEmbedder is
the default, calling OpenRouter's OpenAI-compatible embeddings
endpoint over httpx. Swapping to a local model (sentence-transformers)
or another gateway is a new Embedder implementation and a config
switch, no change here.

Embedding failures are isolated by the caller, not raised through
approval: a missing embedding is a recoverable gap to backfill, not
a lost item. This service raises EmbeddingError, approve_session
catches it and logs to error_log rather than failing the approve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, Self

import httpx

from app.models import Embedding, EmbeddingSourceType, ErrorLog
from app.models.embedding import EMBEDDING_DIM

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import TracebackType

    from sqlalchemy.orm import Session as DbSession

    from app.models import LearnedItem


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
EMBEDDINGS_PATH = "/embeddings"

CONNECT_TIMEOUT_S = 10.0
READ_TIMEOUT_S = 60.0
TOTAL_TIMEOUT_S = 60.0

# The embeddings endpoint accepts many inputs per request. Chunk
# anything larger so a big backfill stays under the per-request input
# cap. A single approve is far below this, the cap matters only for
# backfill.
MAX_INPUTS_PER_REQUEST = 256


class EmbeddingError(Exception):
    """An embedding operation failed.

    Wraps the underlying cause so callers see one error type at the
    service boundary, mirroring SessionServiceError and TransportError.
    """

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.cause = cause


@dataclass(frozen=True)
class EmbeddingRecord:
    """One thing to embed: its text and where it came from.

    source_type and source_id become the embedding row's polymorphic
    reference. content is the exact text embedded and stored.
    """

    source_type: EmbeddingSourceType
    source_id: str
    content: str


class Embedder(Protocol):
    """Turns a batch of texts into vectors, in input order.

    Returns one vector per input text. The provider and model are the
    implementation's concern, callers depend only on this shape.
    """

    @property
    def model_version(self) -> str:
        """Identifier of the model producing vectors, stored per row."""
        ...

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed each text, returning vectors in the same order."""
        ...


class OpenRouterEmbedder:
    """Embedder backed by OpenRouter's OpenAI-compatible embeddings endpoint.

    Owns one long-lived HTTP client for its lifetime so connections
    are reused. Use as an async context manager or call start() and
    shutdown() explicitly. Mirrors the DeepSeek transport's client
    lifecycle and error handling.
    """

    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model
        self._client: httpx.AsyncClient | None = None

    @property
    def model_version(self) -> str:
        return self._model

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
            base_url=OPENROUTER_BASE_URL,
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

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if self._client is None:
            raise EmbeddingError("Embedder not started. Call start() first.")
        if not texts:
            return []

        vectors: list[list[float]] = []
        for start in range(0, len(texts), MAX_INPUTS_PER_REQUEST):
            batch = texts[start : start + MAX_INPUTS_PER_REQUEST]
            vectors.extend(await self._embed_one_batch(batch))
        return vectors

    async def _embed_one_batch(self, batch: list[str]) -> list[list[float]]:
        """Post one batch of inputs and return vectors in input order."""
        if self._client is None:
            raise EmbeddingError("Embedder not started.")

        payload = {"model": self._model, "input": batch}

        try:
            response = await self._client.post(EMBEDDINGS_PATH, json=payload)
            response.raise_for_status()
            data = response.json()
        except httpx.TimeoutException as e:
            raise EmbeddingError("OpenRouter embeddings request timed out.", cause=e) from e
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            body = e.response.text[:200]
            raise EmbeddingError(f"OpenRouter embeddings HTTP {status}: {body}", cause=e) from e
        except httpx.RequestError as e:
            raise EmbeddingError("Network error reaching OpenRouter.", cause=e) from e
        except ValueError as e:
            raise EmbeddingError("OpenRouter returned malformed JSON.", cause=e) from e

        # Response shape: {"data": [{"embedding": [...], "index": N}, ...]}.
        # The API may return items out of order, so sort by index before
        # extracting vectors to guarantee input-order alignment.
        items = data.get("data")
        if not isinstance(items, list) or len(items) != len(batch):
            raise EmbeddingError(
                f"OpenRouter returned {len(items) if isinstance(items, list) else 'no'} "
                f"embeddings for {len(batch)} inputs."
            )
        try:
            ordered = sorted(items, key=lambda item: item["index"])
            vectors = [item["embedding"] for item in ordered]
        except (KeyError, TypeError) as e:
            raise EmbeddingError("OpenAI embedding item missing index/embedding.", cause=e) from e

        for vec in vectors:
            if len(vec) != EMBEDDING_DIM:
                raise EmbeddingError(
                    f"OpenRouter returned {len(vec)}-dim vector, expected {EMBEDDING_DIM}."
                )
        return vectors


async def embed_records(
    db: DbSession,
    embedder: Embedder,
    records: list[EmbeddingRecord],
) -> list[Embedding]:
    """Embed records and write one embedding row each.

    Shared by embed-on-approve and backfill. Embeds all record texts
    in order, builds an Embedding row per record with the embedder's
    model_version stamped on, adds them to the session, and flushes.
    Does not commit: the caller owns the transaction boundary.

    Raises EmbeddingError if the embedder fails. The caller decides
    whether that aborts its work (backfill) or is logged and swallowed
    (approve).
    """
    if not records:
        return []

    vectors = await embedder.embed_texts([r.content for r in records])
    if len(vectors) != len(records):
        raise EmbeddingError(
            f"Embedder returned {len(vectors)} vectors for {len(records)} records."
        )

    rows = [
        Embedding(
            source_type=record.source_type,
            source_id=record.source_id,
            content=record.content,
            embedding=vector,
            embedding_model_version=embedder.model_version,
        )
        for record, vector in zip(records, vectors, strict=True)
    ]
    for row in rows:
        db.add(row)
    db.flush()
    return rows


def records_from_learned_items(items: Sequence[LearnedItem]) -> list[EmbeddingRecord]:
    """Build embedding records from learned items.

    Embeds question and answer together: the question alone serves
    dedup ("have I asked this"), but concatenating the answer makes
    retrieval hit on items whose answer discusses a topic the
    question did not name. Shared by the approve path and the
    backfill script so both embed the same text shape.
    """
    return [
        EmbeddingRecord(
            source_type=EmbeddingSourceType.LEARNED_ITEM,
            source_id=item.id,
            content=f"{item.question}\n{item.answer}",
        )
        for item in items
    ]


def log_embedding_failure(db: DbSession, session_id: str | None, exc: EmbeddingError) -> None:
    """Write an error_log row for a failed embedding and commit it.

    Embedding is best-effort: a failure leaves a backfillable gap,
    it does not fail the operation that triggered it. Callers catch
    EmbeddingError and call this instead of propagating. Mirrors the
    session service's error-log helper but lives here so the
    embedding concern owns its own failure recording.

    Errors inside this helper are swallowed so they cannot mask the
    original flow.
    """
    try:
        row = ErrorLog(
            session_id=session_id,
            kind="embedding.failed",
            message=exc.message,
            context={"cause": type(exc.cause).__name__ if exc.cause else None},
        )
        db.add(row)
        db.commit()
    except Exception:
        db.rollback()

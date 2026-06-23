"""Dependency-free retrieval adapter interfaces and local executor."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Protocol, Sequence, runtime_checkable

from eigentruth.control.actions import (
    ActionExecutionStatus,
    ActionExecutor,
    ActionRequest,
    ActionResult,
    DryRunActionExecutor,
)
from eigentruth.control.policy import ControlAction

_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+")
_FTS_SCHEMA_VERSION = "1"


@dataclass(frozen=True)
class RetrievalQuery:
    """One dependency-free retrieval query."""

    query: str
    claim_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("retrieval query must be non-empty.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "query": self.query,
            "claim_id": self.claim_id,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RetrievalQuery":
        """Build a retrieval query from JSON-like data."""
        claim_id = data.get("claim_id")
        return cls(
            query=str(data["query"]),
            claim_id=None if claim_id is None else str(claim_id),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class RetrievalHit:
    """One retrieval hit returned by a retriever."""

    text: str
    source: str | None = None
    score: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("retrieval hit text must be non-empty.")
        score = float(self.score)
        if not (0.0 <= score <= 1.0):
            raise ValueError("retrieval hit score must be in [0, 1].")
        object.__setattr__(self, "score", score)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "text": self.text,
            "source": self.source,
            "score": self.score,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RetrievalHit":
        """Build a retrieval hit from JSON-like data."""
        text = data.get("text", data.get("content"))
        if text is None:
            raise ValueError("retrieval hit mapping must contain 'text' or 'content'.")
        source = data.get("source")
        metadata = dict(data.get("metadata", {}))
        for key, value in data.items():
            metadata_key = str(key)
            if metadata_key not in {"text", "content", "source", "score", "metadata"} and metadata_key not in metadata:
                metadata[metadata_key] = value
        return cls(
            text=str(text),
            source=None if source is None else str(source),
            score=float(data.get("score", 1.0)),
            metadata=metadata,
        )


class _IndexedRetrievalDocument(NamedTuple):
    hit: RetrievalHit
    tokens: tuple[str, ...]


class _FTSConnectionState(NamedTuple):
    connection: sqlite3.Connection
    index_reused: bool
    index_path: Path | None
    document_fingerprint: str


@runtime_checkable
class Retriever(Protocol):
    """Interface for local or external retrieval implementations."""

    def retrieve(self, query: RetrievalQuery, *, limit: int = 5) -> Sequence[RetrievalHit]:
        """Return retrieval hits for one query."""
        ...


@dataclass(frozen=True)
class InMemoryRetriever:
    """Token-overlap retriever over local text snippets."""

    documents: Sequence[RetrievalHit | Mapping[str, Any] | str]
    min_overlap: float = 0.2

    def __post_init__(self) -> None:
        if not (0.0 <= self.min_overlap <= 1.0):
            raise ValueError("min_overlap must be in [0, 1].")
        documents = tuple(_coerce_hit(item) for item in self.documents)
        object.__setattr__(self, "documents", documents)
        object.__setattr__(
            self,
            "_indexed_documents",
            tuple(_IndexedRetrievalDocument(document, _tokens(document.text)) for document in documents),
        )

    def retrieve(self, query: RetrievalQuery, *, limit: int = 5) -> tuple[RetrievalHit, ...]:
        """Return top local documents by lexical token overlap."""
        if limit <= 0:
            return ()
        query_tokens = _tokens(query.query)
        scored: list[tuple[float, RetrievalHit]] = []
        for indexed in self._indexed_documents:
            document = indexed.hit
            overlap = _token_overlap(query_tokens, indexed.tokens)
            if overlap < self.min_overlap:
                continue
            score = min(1.0, overlap * document.score)
            metadata = {
                **dict(document.metadata),
                "token_overlap": overlap,
                "retriever": type(self).__name__,
            }
            scored.append((score, RetrievalHit(document.text, document.source, score, metadata)))
        scored.sort(key=lambda item: item[0], reverse=True)
        return tuple(hit for _, hit in scored[:limit])


@dataclass(frozen=True)
class SQLiteFTSRetriever:
    """SQLite FTS5 candidate retriever with token-overlap scoring fallback."""

    documents: Sequence[RetrievalHit | Mapping[str, Any] | str]
    min_overlap: float = 0.2
    index_path: str | Path | None = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.min_overlap <= 1.0):
            raise ValueError("min_overlap must be in [0, 1].")
        documents = tuple(_coerce_hit(item) for item in self.documents)
        index_path = None if self.index_path is None else Path(self.index_path)
        object.__setattr__(self, "documents", documents)
        object.__setattr__(self, "index_path", index_path)
        object.__setattr__(self, "_fallback", InMemoryRetriever(documents, min_overlap=self.min_overlap))
        document_fingerprint = ""
        try:
            document_fingerprint = _documents_fingerprint(documents)
            state = _build_fts_connection(
                documents,
                index_path=index_path,
                document_fingerprint=document_fingerprint,
            )
        except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
            object.__setattr__(self, "_connection", None)
            object.__setattr__(self, "_available", False)
            object.__setattr__(self, "_fallback_reason", str(exc))
            object.__setattr__(self, "_index_reused", False)
            object.__setattr__(self, "_document_fingerprint", document_fingerprint)
        else:
            object.__setattr__(self, "_connection", state.connection)
            object.__setattr__(self, "_available", True)
            object.__setattr__(self, "_fallback_reason", None)
            object.__setattr__(self, "_index_reused", state.index_reused)
            object.__setattr__(self, "_document_fingerprint", state.document_fingerprint)

    @property
    def available(self) -> bool:
        """Return whether SQLite FTS5 is available in this Python build."""
        return bool(self._available)

    @property
    def fallback_reason(self) -> str | None:
        """Return why FTS was unavailable, if the retriever is using fallback."""
        return self._fallback_reason

    @property
    def index_reused(self) -> bool:
        """Return whether an existing persistent FTS index was reused."""
        return bool(self._index_reused)

    @property
    def document_fingerprint(self) -> str:
        """Return the fingerprint used to validate a persistent FTS index."""
        return str(self._document_fingerprint)

    def retrieve(self, query: RetrievalQuery, *, limit: int = 5) -> tuple[RetrievalHit, ...]:
        """Return top local documents via SQLite FTS5 candidates and token overlap."""
        if limit <= 0:
            return ()
        if not self.available:
            return tuple(self._fallback.retrieve(query, limit=limit))
        query_tokens = _tokens(query.query)
        fts_query = _fts_query(query_tokens)
        if not fts_query:
            return ()
        try:
            rows = tuple(self._connection.execute(  # type: ignore[union-attr]
                """
                SELECT text, source, metadata_json, base_score
                FROM documents
                WHERE text MATCH ?
                """,
                (fts_query,),
            ))
        except sqlite3.Error:
            return tuple(self._fallback.retrieve(query, limit=limit))

        scored: list[tuple[float, RetrievalHit]] = []
        for text, source, metadata_json, base_score in rows:
            overlap = _token_overlap(query_tokens, _tokens(str(text)))
            if overlap < self.min_overlap:
                continue
            try:
                metadata = dict(_json_loads_mapping(str(metadata_json)))
            except ValueError:
                metadata = {}
            document_score = float(base_score)
            score = min(1.0, overlap * document_score)
            hit_metadata = {
                **metadata,
                "token_overlap": overlap,
                "retriever": type(self).__name__,
                "retriever_backend": "sqlite_fts",
            }
            scored.append((
                score,
                RetrievalHit(str(text), None if source is None else str(source), score, hit_metadata),
            ))
        scored.sort(key=lambda item: item[0], reverse=True)
        return tuple(hit for _, hit in scored[:limit])


@dataclass(frozen=True)
class RetrievalActionExecutor:
    """Execute retrieve actions against a dependency-free retriever."""

    retriever: Retriever
    fallback_executor: ActionExecutor = field(default_factory=DryRunActionExecutor)
    limit: int = 5

    def execute(
        self,
        request: ActionRequest,
        context: Mapping[str, Any] | None = None,
    ) -> ActionResult:
        """Execute one retrieve action locally, or fall back for other actions."""
        if request.action is not ControlAction.RETRIEVE:
            return self.fallback_executor.execute(request, context=context)

        queries = _queries_from_request(request)
        if not queries:
            return ActionResult(
                action=request.action,
                status=ActionExecutionStatus.SKIPPED,
                output={"hits": (), "queries": (), "reason": "no retrieval targets"},
                metadata={
                    "executor": type(self).__name__,
                    "request_metadata": dict(request.metadata),
                    "context": dict(context or {}),
                    "side_effects": False,
                },
                request_id=request.request_id,
            )

        limit = _limit_from_payload(request.payload, default=self.limit)
        hits_by_query = []
        all_hits = []
        for query in queries:
            hits = tuple(self.retriever.retrieve(query, limit=limit))
            hit_dicts = tuple(hit.to_dict() for hit in hits)
            hits_by_query.append({"query": query.to_dict(), "hits": hit_dicts})
            all_hits.extend(hit_dicts)

        return ActionResult(
            action=request.action,
            status=ActionExecutionStatus.SUCCEEDED,
            output={
                "queries": tuple(query.to_dict() for query in queries),
                "hits": tuple(all_hits),
                "hits_by_query": tuple(hits_by_query),
            },
            metadata={
                "executor": type(self).__name__,
                "request_metadata": dict(request.metadata),
                "context": dict(context or {}),
                "side_effects": False,
            },
            request_id=request.request_id,
        )

    def execute_many(
        self,
        requests: Sequence[ActionRequest],
        context: Mapping[str, Any] | None = None,
    ) -> tuple[ActionResult, ...]:
        """Execute multiple action requests."""
        return tuple(self.execute(request, context=context) for request in requests)


def _coerce_hit(value: RetrievalHit | Mapping[str, Any] | str) -> RetrievalHit:
    if isinstance(value, RetrievalHit):
        return value
    if isinstance(value, str):
        return RetrievalHit(value)
    return RetrievalHit.from_dict(value)


def _build_fts_connection(
    documents: Sequence[RetrievalHit],
    *,
    index_path: Path | None,
    document_fingerprint: str,
) -> _FTSConnectionState:
    if index_path is None:
        connection = sqlite3.connect(":memory:")
        try:
            _initialize_fts_connection(
                connection,
                documents,
                document_fingerprint=document_fingerprint,
            )
        except sqlite3.Error:
            connection.close()
            raise
        return _FTSConnectionState(
            connection=connection,
            index_reused=False,
            index_path=None,
            document_fingerprint=document_fingerprint,
        )

    index_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(index_path))
    try:
        if _can_reuse_fts_connection(connection, document_fingerprint=document_fingerprint):
            return _FTSConnectionState(
                connection=connection,
                index_reused=True,
                index_path=index_path,
                document_fingerprint=document_fingerprint,
            )
        _initialize_fts_connection(
            connection,
            documents,
            document_fingerprint=document_fingerprint,
        )
    except sqlite3.Error:
        connection.close()
        raise
    return _FTSConnectionState(
        connection=connection,
        index_reused=False,
        index_path=index_path,
        document_fingerprint=document_fingerprint,
    )


def _can_reuse_fts_connection(connection: sqlite3.Connection, *, document_fingerprint: str) -> bool:
    try:
        rows = dict(connection.execute("SELECT key, value FROM index_metadata"))
        count_row = connection.execute("SELECT count(*) FROM documents").fetchone()
    except sqlite3.Error:
        return False
    if not count_row:
        return False
    return (
        rows.get("schema_version") == _FTS_SCHEMA_VERSION
        and rows.get("document_fingerprint") == document_fingerprint
        and rows.get("n_documents") == str(count_row[0])
    )


def _initialize_fts_connection(
    connection: sqlite3.Connection,
    documents: Sequence[RetrievalHit],
    *,
    document_fingerprint: str,
) -> None:
    connection.execute("DROP TABLE IF EXISTS documents")
    connection.execute("DROP TABLE IF EXISTS index_metadata")
    connection.execute(
        """
        CREATE VIRTUAL TABLE documents USING fts5(
            text,
            source UNINDEXED,
            metadata_json UNINDEXED,
            base_score UNINDEXED
        )
        """
    )
    connection.execute("CREATE TABLE index_metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.executemany(
        "INSERT INTO documents(text, source, metadata_json, base_score) VALUES (?, ?, ?, ?)",
        (
            (
                document.text,
                document.source,
                _json_dumps_mapping(document.metadata),
                float(document.score),
            )
            for document in documents
        ),
    )
    connection.executemany(
        "INSERT INTO index_metadata(key, value) VALUES (?, ?)",
        (
            ("schema_version", _FTS_SCHEMA_VERSION),
            ("document_fingerprint", document_fingerprint),
            ("n_documents", str(len(documents))),
        ),
    )
    connection.commit()


def _documents_fingerprint(documents: Sequence[RetrievalHit]) -> str:
    hasher = sha256()
    for document in documents:
        hasher.update(
            _json_dumps_mapping({
                "text": document.text,
                "source": document.source,
                "score": float(document.score),
                "metadata": dict(document.metadata),
            }).encode("utf-8")
        )
        hasher.update(b"\n")
    return hasher.hexdigest()


def _fts_query(tokens: Sequence[str]) -> str:
    unique_tokens = sorted(set(tokens))
    return " OR ".join(f'"{token}"' for token in unique_tokens)


def _json_dumps_mapping(payload: Mapping[str, Any]) -> str:
    return json.dumps(dict(payload), sort_keys=True, separators=(",", ":"))


def _json_loads_mapping(payload: str) -> Mapping[str, Any]:
    value = json.loads(payload)
    if not isinstance(value, Mapping):
        raise ValueError("JSON payload is not a mapping.")
    return value


def _queries_from_request(request: ActionRequest) -> tuple[RetrievalQuery, ...]:
    payload = dict(request.payload)
    targets = payload.get("retrieval_targets", ())
    queries: list[RetrievalQuery] = []
    if isinstance(targets, Mapping):
        targets = (targets,)
    if isinstance(targets, Sequence) and not isinstance(targets, str):
        for target in targets:
            if isinstance(target, Mapping):
                text = str(target.get("text", "")).strip()
                if not text:
                    continue
                raw_claim_id = target.get("claim_id")
                claim_id = None if raw_claim_id is None else str(raw_claim_id)
                queries.append(
                    RetrievalQuery(
                        query=text,
                        claim_id=claim_id,
                        metadata={"target": dict(target)},
                    )
                )
            elif isinstance(target, str) and target.strip():
                queries.append(RetrievalQuery(query=target.strip()))
    raw_query = payload.get("query")
    if not queries and isinstance(raw_query, str) and raw_query.strip():
        queries.append(RetrievalQuery(query=raw_query.strip()))
    return tuple(queries)


def _limit_from_payload(payload: Mapping[str, Any], *, default: int) -> int:
    try:
        value = int(payload.get("limit", default))
    except (TypeError, ValueError):
        return max(1, int(default))
    return max(1, value)


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(match.group(0).casefold() for match in _TOKEN_RE.finditer(text))


def _token_overlap(query_tokens: Sequence[str], document_tokens: Sequence[str]) -> float:
    if not query_tokens:
        return 0.0
    document_set = set(document_tokens)
    if not document_set:
        return 0.0
    return sum(1 for token in query_tokens if token in document_set) / len(query_tokens)

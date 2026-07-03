"""Dependency-free retrieval adapter interfaces and local executor."""

from __future__ import annotations

import json
import re
import sqlite3
import urllib.error
import urllib.parse
import urllib.request
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
from eigentruth.json_utils import to_jsonable
from eigentruth.verify.groundedness import (
    EvidenceDocument,
    EvidenceQualityAssessment,
    EvidenceQualityPolicy,
    EvidenceQualitySummary,
    summarize_evidence_quality,
)
from eigentruth.verify.protocols import Claim, VerificationResult
from eigentruth.verify.triples import (
    ClaimTriple,
    ClaimTripleExtractor,
    RuleBasedTripleExtractor,
    TripleEvidenceVerifier,
    extract_claim_triples,
)

_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+")
_FTS_SCHEMA_VERSION = "2"
_RETRIEVAL_INDEX_TEXT_METADATA_KEY = "retrieval_index_text"


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


@dataclass(frozen=True)
class TripleSlotRetrievalPlan:
    """Slot-aware retrieval queries derived from extracted claim triples."""

    claim_id: str | None
    triples: Sequence[ClaimTriple | Mapping[str, Any]] = ()
    queries: Sequence[RetrievalQuery | Mapping[str, Any]] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        claim_id = None if self.claim_id is None else str(self.claim_id)
        triples = tuple(
            item if isinstance(item, ClaimTriple) else ClaimTriple.from_dict(item)
            for item in self.triples
        )
        queries = tuple(
            item if isinstance(item, RetrievalQuery) else RetrievalQuery.from_dict(item)
            for item in self.queries
        )
        object.__setattr__(self, "claim_id", claim_id)
        object.__setattr__(self, "triples", triples)
        object.__setattr__(self, "queries", queries)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def query_count(self) -> int:
        """Return the number of planned retrieval queries."""
        return len(self.queries)

    @property
    def triple_count(self) -> int:
        """Return the number of extracted triples behind this plan."""
        return len(self.triples)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready retrieval plan."""
        return {
            "claim_id": self.claim_id,
            "triple_count": self.triple_count,
            "query_count": self.query_count,
            "triples": tuple(triple.to_dict() for triple in self.triples),
            "queries": tuple(query.to_dict() for query in self.queries),
            "metadata": to_jsonable(dict(self.metadata)),
        }


@dataclass(frozen=True)
class TripleSlotRetrievalBindingReport:
    """Retrieved evidence bound back to the claim triple slots it can audit."""

    claim_id: str | None
    plan: TripleSlotRetrievalPlan | Mapping[str, Any]
    hits: Sequence[RetrievalHit | Mapping[str, Any] | str] = ()
    verification_result: VerificationResult | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        claim_id = None if self.claim_id is None else str(self.claim_id)
        raw_plan = self.plan
        plan = (
            raw_plan
            if isinstance(raw_plan, TripleSlotRetrievalPlan)
            else TripleSlotRetrievalPlan(
                claim_id=raw_plan.get("claim_id"),
                triples=tuple(raw_plan.get("triples", ())),
                queries=tuple(raw_plan.get("queries", ())),
                metadata=dict(raw_plan.get("metadata", {})),
            )
        )
        hits = tuple(_coerce_hit(hit) for hit in self.hits)
        object.__setattr__(self, "claim_id", claim_id)
        object.__setattr__(self, "plan", plan)
        object.__setattr__(self, "hits", hits)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def hit_count(self) -> int:
        """Return the number of retrieval hits considered by the binding."""
        return len(self.hits)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready binding report."""
        return {
            "claim_id": self.claim_id,
            "plan": self.plan.to_dict(),
            "hit_count": self.hit_count,
            "hits": tuple(hit.to_dict() for hit in self.hits),
            "verification_result": _verification_result_to_dict(self.verification_result),
            "metadata": to_jsonable(dict(self.metadata)),
        }


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


def plan_triple_slot_retrieval(
    claim: Claim,
    *,
    extractor: ClaimTripleExtractor | None = None,
    include_object: bool = False,
    max_queries_per_triple: int = 2,
) -> TripleSlotRetrievalPlan:
    """Return retrieval queries focused on a claim triple's subject/predicate slots.

    The default plan intentionally omits the claim object from the query. That
    helps retrieval find independent evidence for the property being claimed
    instead of merely finding documents that repeat a generated answer.
    """
    if int(max_queries_per_triple) <= 0:
        raise ValueError("max_queries_per_triple must be positive.")
    active_extractor = RuleBasedTripleExtractor() if extractor is None else extractor
    triples = tuple(extract_claim_triples(claim, extractor=active_extractor))
    queries: list[RetrievalQuery] = []
    seen: set[str] = set()
    for triple_index, triple in enumerate(triples):
        for query_index, (query_text, strategy) in enumerate(_triple_slot_query_variants(
            triple,
            include_object=include_object,
        )):
            if query_index >= int(max_queries_per_triple):
                break
            query_key = _query_dedupe_key(query_text)
            if not query_key or query_key in seen:
                continue
            seen.add(query_key)
            queries.append(RetrievalQuery(
                query=query_text,
                claim_id=claim.claim_id,
                metadata={
                    "query_type": "triple_slot",
                    "strategy": strategy,
                    "triple_index": triple_index,
                    "triple": triple.to_dict(),
                    "include_object": bool(include_object),
                    "omitted_object": None if include_object else triple.object,
                },
            ))
    return TripleSlotRetrievalPlan(
        claim_id=claim.claim_id,
        triples=triples,
        queries=tuple(queries),
        metadata={
            "planner": "plan_triple_slot_retrieval",
            "include_object": bool(include_object),
            "max_queries_per_triple": int(max_queries_per_triple),
        },
    )


def plan_triple_slot_retrieval_queries(
    claim: Claim,
    *,
    extractor: ClaimTripleExtractor | None = None,
    include_object: bool = False,
    max_queries_per_triple: int = 2,
) -> tuple[RetrievalQuery, ...]:
    """Return just the queries from ``plan_triple_slot_retrieval``."""
    return tuple(
        plan_triple_slot_retrieval(
            claim,
            extractor=extractor,
            include_object=include_object,
            max_queries_per_triple=max_queries_per_triple,
        ).queries
    )


def bind_triple_slot_retrieval_hits(
    claim: Claim,
    hits: Sequence[RetrievalHit | Mapping[str, Any] | str],
    *,
    plan: TripleSlotRetrievalPlan | Mapping[str, Any] | None = None,
    extractor: ClaimTripleExtractor | None = None,
    min_slot_coverage: float = 1.0,
    refute_object_mismatch: bool = False,
) -> TripleSlotRetrievalBindingReport:
    """Bind retrieval hits back to claim triples and run a slot audit."""
    active_plan = (
        plan_triple_slot_retrieval(claim, extractor=extractor)
        if plan is None
        else (
            plan
            if isinstance(plan, TripleSlotRetrievalPlan)
            else TripleSlotRetrievalPlan(
                claim_id=plan.get("claim_id"),
                triples=tuple(plan.get("triples", ())),
                queries=tuple(plan.get("queries", ())),
                metadata=dict(plan.get("metadata", {})),
            )
        )
    )
    active_extractor = RuleBasedTripleExtractor() if extractor is None else extractor
    hit_objects = tuple(_coerce_hit(hit) for hit in hits)
    verifier = TripleEvidenceVerifier(
        evidence=tuple(_evidence_document_from_hit(hit) for hit in hit_objects),
        extractor=active_extractor,
        min_slot_coverage=min_slot_coverage,
        refute_object_mismatch=refute_object_mismatch,
    )
    result = verifier.verify(claim)
    return TripleSlotRetrievalBindingReport(
        claim_id=claim.claim_id,
        plan=active_plan,
        hits=hit_objects,
        verification_result=result,
        metadata={
            "binder": "bind_triple_slot_retrieval_hits",
            "min_slot_coverage": float(min_slot_coverage),
            "refute_object_mismatch": bool(refute_object_mismatch),
        },
    )


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
            tuple(
                _IndexedRetrievalDocument(document, _tokens(_retrieval_index_text(document)))
                for document in documents
            ),
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
                **_retrieval_result_metadata(document),
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
                SELECT text, index_text, source, metadata_json, base_score
                FROM documents
                WHERE index_text MATCH ?
                """,
                (fts_query,),
            ))
        except sqlite3.Error:
            return tuple(self._fallback.retrieve(query, limit=limit))

        scored: list[tuple[float, RetrievalHit]] = []
        for text, index_text, source, metadata_json, base_score in rows:
            overlap = _token_overlap(query_tokens, _tokens(str(index_text)))
            if overlap < self.min_overlap:
                continue
            try:
                metadata = dict(_json_loads_mapping(str(metadata_json)))
            except ValueError:
                metadata = {}
            document_score = float(base_score)
            score = min(1.0, overlap * document_score)
            document = RetrievalHit(
                str(text),
                None if source is None else str(source),
                document_score,
                metadata,
            )
            hit_metadata = {
                **_retrieval_result_metadata(document),
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
class HTTPJSONRetriever:
    """HTTP JSON retriever shell for external search services.

    The retriever is stdlib-only and expects a JSON list or an object containing
    one of ``hits``, ``results``, or ``documents``. It performs no retries and
    raises on transport/parse errors so executor boundaries can fail closed and
    record the error in traceable action results.
    """

    endpoint: str
    query_param: str = "q"
    limit_param: str = "limit"
    headers: Mapping[str, str] = field(default_factory=dict)
    timeout_seconds: float = 5.0
    max_response_bytes: int = 2_000_000
    hit_keys: Sequence[str] = ("hits", "results", "documents")

    def __post_init__(self) -> None:
        endpoint = str(self.endpoint).strip()
        parsed = urllib.parse.urlparse(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("endpoint must be an absolute http(s) URL.")
        query_param = str(self.query_param).strip()
        limit_param = str(self.limit_param).strip()
        if not query_param:
            raise ValueError("query_param must be non-empty.")
        if not limit_param:
            raise ValueError("limit_param must be non-empty.")
        timeout_seconds = float(self.timeout_seconds)
        if timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be positive.")
        max_response_bytes = int(self.max_response_bytes)
        if max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be positive.")
        object.__setattr__(self, "endpoint", endpoint)
        object.__setattr__(self, "query_param", query_param)
        object.__setattr__(self, "limit_param", limit_param)
        object.__setattr__(self, "headers", {str(key): str(value) for key, value in self.headers.items()})
        object.__setattr__(self, "timeout_seconds", timeout_seconds)
        object.__setattr__(self, "max_response_bytes", max_response_bytes)
        object.__setattr__(self, "hit_keys", _non_empty_strings(self.hit_keys, name="hit_keys"))

    def retrieve(self, query: RetrievalQuery, *, limit: int = 5) -> tuple[RetrievalHit, ...]:
        """Return hits from a caller-provided HTTP JSON endpoint."""
        if limit <= 0:
            return ()
        request = urllib.request.Request(
            _url_with_query(
                self.endpoint,
                {
                    self.query_param: query.query,
                    self.limit_param: str(limit),
                },
            ),
            headers=dict(self.headers),
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                status = int(getattr(response, "status", 200))
                if status < 200 or status >= 300:
                    raise RuntimeError(f"retrieval endpoint returned HTTP {status}.")
                raw = response.read(self.max_response_bytes + 1)
                if len(raw) > self.max_response_bytes:
                    raise RuntimeError("retrieval response exceeded max_response_bytes.")
                charset = _response_charset(response)
        except (OSError, urllib.error.URLError) as exc:
            raise RuntimeError(f"retrieval endpoint request failed: {exc}") from exc
        try:
            payload = json.loads(raw.decode(charset))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"retrieval endpoint returned invalid JSON: {exc}") from exc

        hits = []
        for item in _hits_from_payload(payload, hit_keys=self.hit_keys):
            hit = _coerce_hit(item)
            metadata = {
                **dict(hit.metadata),
                "retriever": type(self).__name__,
                "retriever_backend": "http_json",
                "endpoint": self.endpoint,
                "status_code": status,
            }
            if query.claim_id is not None:
                metadata["claim_id"] = query.claim_id
            hits.append(RetrievalHit(hit.text, hit.source, hit.score, metadata))
            if len(hits) >= limit:
                break
        return tuple(hits)


@dataclass(frozen=True)
class ProvenanceFilteredRetriever:
    """Filter retriever hits before they become verifier evidence.

    This wrapper keeps external retrieval integrations dependency-free while
    making the evidence trust boundary explicit: callers can require source
    provenance, source allow/deny prefixes, minimum score, metadata tags, and a
    per-source hit cap before hits are exposed to the control loop.
    """

    retriever: Retriever
    min_score: float = 0.0
    require_source: bool = False
    allowed_source_prefixes: Sequence[str] = ()
    denied_source_prefixes: Sequence[str] = ()
    required_metadata: Mapping[str, Any] = field(default_factory=dict)
    max_hits_per_source: int | None = None
    fetch_multiplier: int = 3

    def __post_init__(self) -> None:
        min_score = float(self.min_score)
        if not (0.0 <= min_score <= 1.0):
            raise ValueError("min_score must be in [0, 1].")
        fetch_multiplier = int(self.fetch_multiplier)
        if fetch_multiplier <= 0:
            raise ValueError("fetch_multiplier must be positive.")
        max_hits_per_source = self.max_hits_per_source
        if max_hits_per_source is not None:
            max_hits_per_source = int(max_hits_per_source)
            if max_hits_per_source <= 0:
                raise ValueError("max_hits_per_source must be positive when set.")
        object.__setattr__(self, "min_score", min_score)
        object.__setattr__(
            self,
            "allowed_source_prefixes",
            _non_empty_strings_or_empty(self.allowed_source_prefixes, name="allowed_source_prefixes"),
        )
        object.__setattr__(
            self,
            "denied_source_prefixes",
            _non_empty_strings_or_empty(self.denied_source_prefixes, name="denied_source_prefixes"),
        )
        object.__setattr__(self, "required_metadata", dict(self.required_metadata))
        object.__setattr__(self, "max_hits_per_source", max_hits_per_source)
        object.__setattr__(self, "fetch_multiplier", fetch_multiplier)

    def retrieve(self, query: RetrievalQuery, *, limit: int = 5) -> tuple[RetrievalHit, ...]:
        """Return filtered hits from the wrapped retriever."""
        if limit <= 0:
            return ()
        candidate_limit = max(limit, limit * self.fetch_multiplier)
        candidates = tuple(self.retriever.retrieve(query, limit=candidate_limit))
        accepted: list[RetrievalHit] = []
        source_counts: dict[str, int] = {}
        for hit in candidates:
            if not self._accepts(hit):
                continue
            source_key = hit.source or "<missing-source>"
            if self.max_hits_per_source is not None:
                source_count = source_counts.get(source_key, 0)
                if source_count >= self.max_hits_per_source:
                    continue
                source_counts[source_key] = source_count + 1
            metadata = {
                **dict(hit.metadata),
                "provenance_filter": {
                    "retriever": type(self).__name__,
                    "wrapped_retriever": type(self.retriever).__name__,
                    "min_score": self.min_score,
                    "require_source": self.require_source,
                    "allowed_source_prefixes": self.allowed_source_prefixes,
                    "denied_source_prefixes": self.denied_source_prefixes,
                    "required_metadata": dict(self.required_metadata),
                    "max_hits_per_source": self.max_hits_per_source,
                },
            }
            accepted.append(RetrievalHit(hit.text, hit.source, hit.score, metadata))
            if len(accepted) >= limit:
                break
        return tuple(accepted)

    def _accepts(self, hit: RetrievalHit) -> bool:
        if hit.score < self.min_score:
            return False
        source = hit.source
        if self.require_source and not source:
            return False
        if source and any(source.startswith(prefix) for prefix in self.denied_source_prefixes):
            return False
        if self.allowed_source_prefixes and (not source or not any(
            source.startswith(prefix) for prefix in self.allowed_source_prefixes
        )):
            return False
        metadata = dict(hit.metadata)
        return all(metadata.get(str(key)) == value for key, value in self.required_metadata.items())


@dataclass(frozen=True)
class RetrievalActionExecutor:
    """Execute retrieve actions against a dependency-free retriever."""

    retriever: Retriever
    fallback_executor: ActionExecutor = field(default_factory=DryRunActionExecutor)
    limit: int = 5
    evidence_quality_policy: EvidenceQualityPolicy | Mapping[str, Any] | None = None

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
        errors = []
        quality_assessments: list[EvidenceQualityAssessment] = []
        for query in queries:
            try:
                hits = tuple(self.retriever.retrieve(query, limit=limit))
            except Exception as exc:  # noqa: BLE001 - executor boundary must fail closed.
                error = {
                    "query": query.to_dict(),
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
                errors.append(error)
                hits_by_query.append({"query": query.to_dict(), "hits": (), "error": error})
                continue
            hit_dicts = tuple(hit.to_dict() for hit in hits)
            query_result: dict[str, Any] = {"query": query.to_dict(), "hits": hit_dicts}
            quality_summary = _quality_summary_for_query(
                hits,
                query=query,
                policy=self.evidence_quality_policy,
            )
            if quality_summary is not None:
                query_result["evidence_quality"] = quality_summary.to_dict()
                quality_assessments.extend(quality_summary.assessments)
            hits_by_query.append(query_result)
            all_hits.extend(hit_dicts)
        output: dict[str, Any] = {
            "queries": tuple(query.to_dict() for query in queries),
            "hits": tuple(all_hits),
            "hits_by_query": tuple(hits_by_query),
            "errors": tuple(errors),
        }
        if self.evidence_quality_policy is not None:
            output["evidence_quality"] = EvidenceQualitySummary.from_assessments(
                tuple(quality_assessments),
                document_count=len(all_hits),
            ).to_dict()

        return ActionResult(
            action=request.action,
            status=ActionExecutionStatus.FAILED if errors else ActionExecutionStatus.SUCCEEDED,
            output=output,
            metadata={
                "executor": type(self).__name__,
                "retriever": type(self.retriever).__name__,
                "request_metadata": dict(request.metadata),
                "context": dict(context or {}),
                "side_effects": False,
                "fail_closed": bool(errors),
            },
            request_id=request.request_id,
            error=None if not errors else f"retrieval failed for {len(errors)} of {len(queries)} queries",
        )

    def execute_many(
        self,
        requests: Sequence[ActionRequest],
        context: Mapping[str, Any] | None = None,
    ) -> tuple[ActionResult, ...]:
        """Execute multiple action requests."""
        return tuple(self.execute(request, context=context) for request in requests)


def _quality_summary_for_query(
    hits: Sequence[RetrievalHit],
    *,
    query: RetrievalQuery,
    policy: EvidenceQualityPolicy | Mapping[str, Any] | None,
) -> EvidenceQualitySummary | None:
    if policy is None:
        return None
    return summarize_evidence_quality(
        tuple(hit.to_dict() for hit in hits),
        policy=policy,
        features=_quality_features_from_query(query),
    )


def _quality_features_from_query(query: RetrievalQuery) -> Mapping[str, Any]:
    metadata = dict(query.metadata)
    features = metadata.get("features", metadata.get("claim_features", {}))
    if not isinstance(features, Mapping):
        features = {}
    target = metadata.get("target")
    if isinstance(target, Mapping):
        target_features = target.get("features", target.get("claim_features", {}))
        if isinstance(target_features, Mapping) and target_features:
            return target_features
        target_metadata = target.get("metadata", {})
        if isinstance(target_metadata, Mapping):
            target_metadata_features = target_metadata.get("features", target_metadata.get("claim_features", {}))
            if isinstance(target_metadata_features, Mapping):
                return target_metadata_features
    return features if isinstance(features, Mapping) else {}


def _triple_slot_query_variants(
    triple: ClaimTriple,
    *,
    include_object: bool,
) -> tuple[tuple[str, str], ...]:
    subjects = _query_subject_variants(triple)
    predicate = _clean_query_part(_predicate_query_text(triple.predicate))
    object_value = _clean_query_part(triple.object)
    if not subjects or not predicate:
        return ()

    variants: list[tuple[str, str]] = []
    normalized_predicate = _normalized_predicate(triple.predicate)
    for subject_index, subject in enumerate(subjects):
        suffix = "" if subject_index == 0 else "_alias"
        if normalized_predicate in {"capital_of", "official_language_of", "currency_of"}:
            variants.append((_clean_query_part(f"{predicate} of {subject}"), f"property_of_subject{suffix}"))
        elif normalized_predicate == "located_in":
            variants.append((_clean_query_part(f"{subject} located in"), f"subject_predicate{suffix}"))
        elif normalized_predicate == "equals":
            variants.append((_clean_query_part(f"{subject} equals"), f"subject_predicate{suffix}"))
        else:
            variants.append((_clean_query_part(f"{subject} {predicate}"), f"subject_predicate{suffix}"))

        variants.append((_clean_query_part(f"{subject} {predicate}"), f"subject_predicate_generic{suffix}"))
        if include_object and object_value:
            variants.append((_clean_query_part(f"{subject} {predicate} {object_value}"), f"full_triple{suffix}"))
    return tuple((query, strategy) for query, strategy in _unique_query_variants(variants) if query)


def _predicate_query_text(value: str) -> str:
    normalized = _normalized_predicate(value)
    mapping = {
        "capital_of": "capital",
        "official_language_of": "official language",
        "currency_of": "currency",
        "founder": "founder",
        "located_in": "located in",
        "equals": "equals",
    }
    return mapping.get(normalized, normalized.replace("_", " "))


def _query_subject_variants(triple: ClaimTriple) -> tuple[str, ...]:
    values = [triple.subject]
    metadata = dict(triple.metadata)
    for key in ("subject_aliases", "subject_alias", "aliases", "alias"):
        raw = metadata.get(key)
        if raw is None:
            continue
        if isinstance(raw, (str, bytes, bytearray)):
            candidates = (raw,)
        else:
            try:
                candidates = tuple(raw)
            except TypeError:
                candidates = (raw,)
        values.extend(str(candidate) for candidate in candidates)
    seen: set[str] = set()
    unique = []
    for value in values:
        text = _clean_query_part(str(value))
        key = _query_dedupe_key(text)
        if not key or key in seen:
            continue
        unique.append(text)
        seen.add(key)
    return tuple(unique)


def _normalized_predicate(value: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", str(value).casefold()).strip("_")


def _clean_query_part(value: Any) -> str:
    text = " ".join(str(value).replace("_", " ").split())
    return text.strip(" \t\r\n,;:.!?\"'()[]{}")


def _unique_query_variants(values: Sequence[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    unique: list[tuple[str, str]] = []
    seen: set[str] = set()
    for query, strategy in values:
        key = _query_dedupe_key(query)
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append((query, strategy))
    return tuple(unique)


def _query_dedupe_key(value: str) -> str:
    return " ".join(_tokens(value))


def _evidence_document_from_hit(hit: RetrievalHit) -> EvidenceDocument:
    metadata = dict(hit.metadata)
    metadata.setdefault("retrieval_score", hit.score)
    if hit.source is not None:
        metadata.setdefault("retrieval_source", hit.source)
    return EvidenceDocument(hit.text, source=hit.source, metadata=metadata)


def _verification_result_to_dict(result: VerificationResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "status": result.status.value,
        "confidence": result.confidence,
        "evidence": tuple(result.evidence),
        "explanation": result.explanation,
        "metadata": to_jsonable(dict(result.metadata)),
    }


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
            index_text,
            text UNINDEXED,
            source UNINDEXED,
            metadata_json UNINDEXED,
            base_score UNINDEXED
        )
        """
    )
    connection.execute("CREATE TABLE index_metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.executemany(
        "INSERT INTO documents(index_text, text, source, metadata_json, base_score) VALUES (?, ?, ?, ?, ?)",
        (
            (
                _retrieval_index_text(document),
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


def _retrieval_index_text(document: RetrievalHit) -> str:
    """Return text used for retrieval indexing without changing evidence text."""
    raw_index_text = document.metadata.get(_RETRIEVAL_INDEX_TEXT_METADATA_KEY)
    values: list[str] = [document.text]
    if isinstance(raw_index_text, str):
        stripped = raw_index_text.strip()
        if stripped:
            values.append(stripped)
    elif isinstance(raw_index_text, Sequence) and not isinstance(raw_index_text, (bytes, bytearray)):
        values.extend(str(item).strip() for item in raw_index_text if str(item).strip())
    return " ".join(values)


def _retrieval_result_metadata(document: RetrievalHit) -> dict[str, Any]:
    metadata = dict(document.metadata)
    index_text = metadata.pop(_RETRIEVAL_INDEX_TEXT_METADATA_KEY, None)
    if index_text is None:
        return metadata
    if isinstance(index_text, str):
        index_parts = (index_text,)
    elif isinstance(index_text, Sequence) and not isinstance(index_text, (bytes, bytearray)):
        index_parts = tuple(str(item) for item in index_text)
    else:
        index_parts = (str(index_text),)
    joined = " ".join(part.strip() for part in index_parts if part.strip())
    if joined:
        metadata["retrieval_index_text_used"] = True
        metadata["retrieval_index_text_sha256"] = sha256(joined.encode("utf-8")).hexdigest()
        metadata["retrieval_index_text_chars"] = len(joined)
    return metadata


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


def _url_with_query(endpoint: str, params: Mapping[str, str]) -> str:
    parsed = urllib.parse.urlsplit(endpoint)
    query_items = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    query_items.extend((str(key), str(value)) for key, value in params.items())
    return urllib.parse.urlunsplit((
        parsed.scheme,
        parsed.netloc,
        parsed.path,
        urllib.parse.urlencode(query_items),
        parsed.fragment,
    ))


def _response_charset(response: Any) -> str:
    headers = getattr(response, "headers", None)
    if headers is not None:
        get_content_charset = getattr(headers, "get_content_charset", None)
        if callable(get_content_charset):
            charset = get_content_charset()
            if charset:
                return str(charset)
    return "utf-8"


def _hits_from_payload(payload: Any, *, hit_keys: Sequence[str]) -> tuple[RetrievalHit | Mapping[str, Any] | str, ...]:
    if isinstance(payload, Mapping):
        for key in hit_keys:
            value = payload.get(key)
            if _is_hit_sequence(value):
                return tuple(value)
        if "text" in payload or "content" in payload:
            return (payload,)
        raise RuntimeError(f"retrieval JSON object did not contain any hit list keys: {tuple(hit_keys)!r}.")
    if _is_hit_sequence(payload):
        return tuple(payload)
    raise RuntimeError("retrieval JSON payload must be a hit list or object containing a hit list.")


def _is_hit_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _non_empty_strings(values: Sequence[Any], *, name: str) -> tuple[str, ...]:
    strings = tuple(str(value).strip() for value in values)
    if not strings or any(not value for value in strings):
        raise ValueError(f"{name} must contain non-empty strings.")
    return strings


def _non_empty_strings_or_empty(values: Sequence[Any], *, name: str) -> tuple[str, ...]:
    strings = tuple(str(value).strip() for value in values)
    if any(not value for value in strings):
        raise ValueError(f"{name} must contain non-empty strings.")
    return strings


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

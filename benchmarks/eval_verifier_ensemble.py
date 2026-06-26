"""Evaluate calibrated internal diagnostics combined with retrieval verification.

The script consumes score dumps and claim/evidence metadata, then compares a
single conformal internal gate against a simple evidence-aware policy:

* refuted claim -> trigger
* supported claim -> suppress an internal trigger
* otherwise -> keep the internal trigger

This is intentionally dependency-free. It is a controlled benchmark harness for
later production verifiers, not a replacement for a real search/RAG backend.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, MutableMapping, Sequence

import torch

from eigentruth.adapters import (
    CachedRetriever,
    InMemoryRetriever,
    InMemoryWorldModelAdapter,
    QuestionAnswerVerifier,
    RetrievalQuery,
    SQLiteStateSource,
    StateTransitionVerifier,
    StructuredFactVerifier,
    StructuredStateVerifier,
    combine_cache_stats,
)
from eigentruth.calibration import DEFAULT_SCORE_DIRECTIONS
from eigentruth.control import (
    ControlAction,
    RiskDecision,
    RiskLevel,
    StagedVerificationPolicy,
    VerificationStageDecision,
)
from eigentruth.eval.conformal import directional_conformal_threshold
from eigentruth.eval.score_dump import (
    load_score_dump_statement_scores,
    score_dump_cache_summary,
    score_dump_file_metadata,
)
from eigentruth.verify import (
    CachedVerifier,
    Claim,
    GroundednessVerifier,
    JsonTraceCache,
    SelfConsistencyVerifier,
    TripleEvidenceVerifier,
    VerificationResult,
    VerificationStatus,
    stable_cache_key,
)
from eigentruth.verify.features import flag_value_enabled

ALPHAS = (0.05, 0.10, 0.20)
TOLERANCE = 0.03


@dataclass(frozen=True)
class ClaimEvidenceRecord:
    """One claim plus local evidence sources for verifier ensemble evaluation."""

    claim: Claim
    initial_evidence: Sequence[Mapping[str, Any] | str] = ()
    retrieval_documents: Sequence[Mapping[str, Any] | str] = ()
    selfcheck_samples: Sequence[Mapping[str, Any] | str] = ()
    refutations: Mapping[str, Sequence[str] | str] = field(default_factory=dict)
    state: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError("scores name cannot be empty.")
    return name, Path(path)


def _parse_alphas(value: str | None) -> tuple[float, ...]:
    if value is None:
        return ALPHAS
    alphas = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not alphas:
        raise ValueError("--alphas must contain at least one alpha value.")
    if any(not (0.0 < alpha < 1.0) for alpha in alphas):
        raise ValueError("alphas must be in (0, 1).")
    return alphas


def _parse_csv(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _load_scores(
    path: Path,
    signal: str,
    *,
    cache: MutableMapping[str, Any] | None = None,
) -> dict[str, Any]:
    dump = load_score_dump_statement_scores(path, (signal,), cache=cache)
    labels = torch.tensor(dump.labels, dtype=torch.int64)
    scores = torch.tensor(dump.scores[signal], dtype=torch.float64)
    return {
        "config": dict(dump.config),
        "labels": labels,
        "scores": scores,
        "statements": tuple(dump.statements),
        "score_dump_summary": dict(dump.summary),
        "score_dump_source_format": dump.source_format,
    }


def _load_fixture(path: Path | None) -> Mapping[str, Any]:
    if path is None:
        return {}
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        return {"records": payload}
    if not isinstance(payload, Mapping):
        raise ValueError("claim fixture must be a JSON object or list.")
    return payload


def _load_qa_verifier(path: Path | None) -> QuestionAnswerVerifier | None:
    if path is None:
        return None
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        payload = {"documents": payload}
    if not isinstance(payload, Mapping):
        raise ValueError("QA corpus must be a JSON object or list.")
    return QuestionAnswerVerifier.from_corpus(payload)


def _load_fact_verifier(path: Path | None) -> StructuredFactVerifier | None:
    if path is None:
        return None
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        payload = {"documents": payload}
    if not isinstance(payload, Mapping):
        raise ValueError("fact corpus must be a JSON object or list.")
    return StructuredFactVerifier.from_corpus(payload)


def _load_state_source(path: Path | None) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Load a local structured state source.

    The file may be either a raw JSON object used as state, or an object with
    explicit ``state`` and optional ``state_checks`` / ``state_transitions``
    fields. It may also contain a ``sqlite`` state-source spec with
    ``database_path`` and ``queries`` fields.
    """
    if path is None:
        return {}, {}, {}
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, Mapping):
        raise ValueError("state source must be a JSON object.")
    if "sqlite" in payload:
        state = _load_sqlite_state_source(payload["sqlite"], base_path=path.parent)
        extra_state = payload.get("state", {})
        if not isinstance(extra_state, Mapping):
            raise ValueError("state source 'state' must be a JSON object when present.")
        raw_checks = payload.get("state_checks", {})
        if not isinstance(raw_checks, Mapping):
            raise ValueError("state source 'state_checks' must be a JSON object.")
        raw_transitions = payload.get("state_transitions", {})
        if not isinstance(raw_transitions, Mapping):
            raise ValueError("state source 'state_transitions' must be a JSON object.")
        return (
            dict(_merge_state_mappings(state, extra_state)),
            dict(raw_checks),
            dict(raw_transitions),
        )
    if "state" in payload:
        state = payload.get("state", {})
        if not isinstance(state, Mapping):
            raise ValueError("state source 'state' must be a JSON object.")
        raw_checks = payload.get("state_checks", {})
        if not isinstance(raw_checks, Mapping):
            raise ValueError("state source 'state_checks' must be a JSON object.")
        raw_transitions = payload.get("state_transitions", {})
        if not isinstance(raw_transitions, Mapping):
            raise ValueError("state source 'state_transitions' must be a JSON object.")
        return dict(state), dict(raw_checks), dict(raw_transitions)
    return dict(payload), {}, {}


def _load_sqlite_state_source(spec: Any, *, base_path: Path) -> Mapping[str, Any]:
    if not isinstance(spec, Mapping):
        raise ValueError("state source 'sqlite' must be a JSON object.")
    raw_database_path = spec.get("database_path", spec.get("path"))
    if raw_database_path is None:
        raise ValueError("state source 'sqlite' must contain database_path or path.")
    queries = spec.get("queries")
    if not isinstance(queries, Sequence) or isinstance(queries, (str, bytes, bytearray)):
        raise ValueError("state source 'sqlite.queries' must be a list.")
    database_path = Path(str(raw_database_path))
    if not database_path.is_absolute():
        database_path = base_path / database_path
    return SQLiteStateSource(database_path, queries=tuple(queries)).load_state()


def _records_from_dump_and_fixture(
    *,
    dump: Mapping[str, Any],
    fixture: Mapping[str, Any],
    expected_count: int,
) -> tuple[ClaimEvidenceRecord, ...]:
    fixture_records = tuple(fixture.get("records", ()))
    if fixture_records and len(fixture_records) != expected_count:
        raise ValueError(
            f"claim fixture has {len(fixture_records)} records but score dump has {expected_count} labels."
        )
    dump_statements = tuple(dump.get("statements", ()))
    if not fixture_records and len(dump_statements) != expected_count:
        raise ValueError(
            "verifier ensemble requires either a claim fixture with one record per score "
            "or score dumps containing a 'statements' list."
        )

    records = []
    global_initial = tuple(fixture.get("initial_evidence", ()))
    global_documents = tuple(fixture.get("retrieval_documents", ()))
    global_selfcheck_samples = _selfcheck_samples_from_mapping(fixture)
    global_refutations = dict(fixture.get("refutations", {}))
    for idx in range(expected_count):
        raw_record = dict(fixture_records[idx]) if fixture_records else {}
        statement = dict(dump_statements[idx]) if dump_statements else {}
        claim_text = str(
            raw_record.get("claim")
            or raw_record.get("text")
            or statement.get("claim")
            or statement.get("text")
            or statement.get("answer")
            or ""
        ).strip()
        if not claim_text:
            raise ValueError(f"record {idx} is missing claim text.")
        claim_id = str(raw_record.get("claim_id") or statement.get("claim_id") or f"c{idx + 1}")
        claim_metadata = dict(statement.get("metadata", {}))
        claim_metadata.update(dict(raw_record.get("claim_metadata", {})))
        if "state_check" in statement:
            claim_metadata["state_check"] = statement["state_check"]
        if "state_check" in raw_record:
            claim_metadata["state_check"] = raw_record["state_check"]
        if "state_transition" in statement:
            claim_metadata["state_transition"] = statement["state_transition"]
        if "state_transition" in raw_record:
            claim_metadata["state_transition"] = raw_record["state_transition"]
        for key in ("features", "requires_triple_audit", "triples", "claim_triples"):
            if key in statement:
                claim_metadata[key] = statement[key]
            if key in raw_record:
                claim_metadata[key] = raw_record[key]
        record_state = _merge_state_mappings(
            statement.get("state", {}),
            raw_record.get("state", {}),
        )
        records.append(
            ClaimEvidenceRecord(
                claim=Claim(
                    text=claim_text,
                    claim_id=claim_id,
                    metadata=claim_metadata,
                ),
                initial_evidence=tuple(raw_record.get("initial_evidence", global_initial)),
                retrieval_documents=tuple(raw_record.get("retrieval_documents", global_documents)),
                selfcheck_samples=_selfcheck_samples_from_mapping(
                    raw_record,
                    fallback=_selfcheck_samples_from_mapping(statement, fallback=global_selfcheck_samples),
                ),
                refutations={
                    **global_refutations,
                    **dict(raw_record.get("refutations", {})),
                },
                state=record_state,
                metadata={
                    "index": idx,
                    "statement": statement,
                    **dict(raw_record.get("metadata", {})),
                },
            )
        )
    return tuple(records)


def _selfcheck_samples_from_mapping(
    data: Mapping[str, Any],
    *,
    fallback: Sequence[Mapping[str, Any] | str] = (),
) -> tuple[Mapping[str, Any] | str, ...]:
    """Return self-check samples from fixture or score-dump metadata."""
    for key in ("selfcheck_samples", "sampled_responses", "samples"):
        if key in data:
            return _as_sample_sequence(data[key], name=key)
    return tuple(fallback)


def _as_sample_sequence(value: Any, *, name: str) -> tuple[Mapping[str, Any] | str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, Mapping)):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        samples = []
        for item in value:
            if not isinstance(item, (str, Mapping)):
                raise ValueError(f"{name} items must be strings or JSON objects.")
            samples.append(item)
        return tuple(samples)
    raise ValueError(f"{name} must be a string, JSON object, or sequence of those values.")


def _merge_state_mappings(*values: Any) -> Mapping[str, Any]:
    merged: dict[str, Any] = {}
    for value in values:
        if not isinstance(value, Mapping):
            continue
        for key, item in value.items():
            if isinstance(item, Mapping) and isinstance(merged.get(key), Mapping):
                merged[key] = {**dict(merged[key]), **dict(item)}
            else:
                merged[key] = item
    return merged


def _stage_diagnostic_decision(
    *,
    score: float,
    threshold: float,
    direction: str,
    signal: str,
) -> RiskDecision:
    if direction == "higher":
        internal_trigger = score > threshold
        signed_margin = score - threshold
    elif direction == "lower":
        internal_trigger = score < threshold
        signed_margin = threshold - score
    else:
        raise ValueError("direction must be 'higher' or 'lower'.")
    margin_confidence = min(1.0, max(0.5, 0.5 + abs(signed_margin) / (2.0 * (abs(threshold) + 1.0))))
    diagnostics = {
        signal: score,
        "threshold": threshold,
        "direction": direction,
        "internal_trigger": internal_trigger,
        "signed_margin": signed_margin,
    }
    if internal_trigger:
        return RiskDecision(
            action=ControlAction.RETRIEVE,
            risk_level=RiskLevel.MEDIUM,
            confidence=margin_confidence,
            reason="internal diagnostic exceeded staged verifier threshold",
            diagnostics=diagnostics,
        )
    return RiskDecision(
        action=ControlAction.ACCEPT,
        risk_level=RiskLevel.LOW,
        confidence=margin_confidence,
        reason="internal diagnostic below staged verifier threshold",
        diagnostics=diagnostics,
    )


def _stage_record_payload(
    *,
    decision: VerificationStageDecision,
    diagnostic_decision: RiskDecision,
    score: float,
    threshold: float,
    direction: str,
    alpha: float,
    signal: str,
) -> dict[str, Any]:
    return {
        **decision.to_dict(),
        "diagnostic_decision": diagnostic_decision.to_dict(),
        "score": score,
        "threshold": threshold,
        "direction": direction,
        "alpha": alpha,
        "signal": signal,
    }


def _record_metadata(
    record: ClaimEvidenceRecord,
    stage_payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    metadata = dict(record.metadata)
    if stage_payload is not None:
        metadata["staged_verification"] = dict(stage_payload)
    return metadata


def _staged_skip_record(
    record: ClaimEvidenceRecord,
    stage_payload: Mapping[str, Any],
) -> dict[str, Any]:
    result = VerificationResult(
        status=VerificationStatus.NOT_APPLICABLE,
        confidence=0.0,
        evidence=(),
        explanation="Verification skipped by staged policy.",
        metadata={"staged_verification": dict(stage_payload)},
    )
    payload = _verification_to_dict(result)
    return {
        "claim": {
            "text": record.claim.text,
            "claim_id": record.claim.claim_id,
            "metadata": dict(record.claim.metadata),
        },
        "initial": payload,
        "final": payload,
        "qa": None,
        "state": None,
        "transition": None,
        "selfcheck": None,
        "retrieval_hits": (),
        "route": _route_metadata(
            selected_route="staged_skip",
            selected_verifier="StagedVerificationPolicy",
            attempted_routes=(),
            used_retrieval=False,
            route_timings=(),
        ),
        "metadata": _record_metadata(record, stage_payload),
    }


def _verify_records(
    records: Sequence[ClaimEvidenceRecord],
    *,
    verifier_min_overlap: float,
    retriever_min_overlap: float,
    retrieval_limit: int,
    selfcheck_min_samples: int,
    selfcheck_min_overlap: float,
    selfcheck_support_threshold: float,
    selfcheck_refute_threshold: float,
    selfcheck_early_stop: bool,
    selfcheck_max_samples: int | None,
    enable_triple_evidence: bool = False,
    triple_min_slot_coverage: float = 1.0,
    qa_verifier: QuestionAnswerVerifier | None = None,
    fact_verifier: StructuredFactVerifier | None = None,
    state_verifier: StructuredStateVerifier | None = None,
    state_checks: Mapping[str, Any] | None = None,
    transition_verifier: StateTransitionVerifier | None = None,
    state_transitions: Mapping[str, Any] | None = None,
    cache_stats: dict[str, Any] | None = None,
    stage_policy: StagedVerificationPolicy | None = None,
    stage_scores: torch.Tensor | None = None,
    stage_threshold: float | None = None,
    stage_direction: str = "higher",
    stage_alpha: float | None = None,
    stage_signal: str | None = None,
) -> tuple[dict[str, Any], ...]:
    if stage_policy is not None:
        if stage_scores is None or stage_threshold is None or stage_alpha is None or stage_signal is None:
            raise ValueError("staged verification requires scores, threshold, alpha, and signal.")
        if stage_scores.numel() != len(records):
            raise ValueError("staged verification scores must match records length.")
    verified = []
    state_checks = {} if state_checks is None else state_checks
    state_transitions = {} if state_transitions is None else state_transitions
    qa_runner = CachedVerifier(qa_verifier) if qa_verifier is not None else None
    fact_runner = CachedVerifier(fact_verifier) if fact_verifier is not None else None
    state_runner = CachedVerifier(state_verifier) if state_verifier is not None else None
    transition_runner = CachedVerifier(transition_verifier) if transition_verifier is not None else None
    groundedness_runners: dict[str, CachedVerifier] = {}
    triple_evidence_runners: dict[str, CachedVerifier] = {}
    retrieval_qa_runners: dict[str, CachedVerifier] = {}
    selfcheck_runners: dict[str, CachedVerifier] = {}
    retrievers: dict[str, CachedRetriever] = {}

    def groundedness_runner(
        evidence: Sequence[Mapping[str, Any] | str],
        refutations: Mapping[str, Sequence[str] | str],
    ) -> CachedVerifier:
        key = stable_cache_key({
            "evidence": evidence,
            "refutations": refutations,
            "min_overlap": verifier_min_overlap,
        })
        runner = groundedness_runners.get(key)
        if runner is None:
            runner = CachedVerifier(
                GroundednessVerifier(
                    evidence=evidence,
                    refutations=refutations,
                    min_overlap=verifier_min_overlap,
                )
            )
            groundedness_runners[key] = runner
        return runner

    def triple_evidence_runner(
        evidence: Sequence[Mapping[str, Any] | str],
    ) -> CachedVerifier:
        key = stable_cache_key({
            "evidence": evidence,
            "min_slot_coverage": triple_min_slot_coverage,
        })
        runner = triple_evidence_runners.get(key)
        if runner is None:
            runner = CachedVerifier(
                TripleEvidenceVerifier(
                    evidence=evidence,
                    min_slot_coverage=triple_min_slot_coverage,
                )
            )
            triple_evidence_runners[key] = runner
        return runner

    def retrieval_qa_runner(
        documents: Sequence[Mapping[str, Any] | str],
    ) -> CachedVerifier | None:
        key = stable_cache_key({"documents": documents, "verifier": "QuestionAnswerVerifier"})
        runner = retrieval_qa_runners.get(key)
        if runner is not None:
            return runner
        try:
            verifier = QuestionAnswerVerifier.from_corpus({"documents": documents})
        except ValueError:
            return None
        runner = CachedVerifier(verifier)
        retrieval_qa_runners[key] = runner
        return runner

    def selfcheck_runner(samples: Sequence[Mapping[str, Any] | str]) -> CachedVerifier:
        key = stable_cache_key({
            "samples": samples,
            "min_samples": selfcheck_min_samples,
            "min_overlap": selfcheck_min_overlap,
            "support_threshold": selfcheck_support_threshold,
            "refute_threshold": selfcheck_refute_threshold,
            "early_stop": selfcheck_early_stop,
            "max_samples": selfcheck_max_samples,
        })
        runner = selfcheck_runners.get(key)
        if runner is None:
            runner = CachedVerifier(
                SelfConsistencyVerifier(
                    samples=samples,
                    min_samples=selfcheck_min_samples,
                    min_overlap=selfcheck_min_overlap,
                    support_threshold=selfcheck_support_threshold,
                    refute_threshold=selfcheck_refute_threshold,
                    early_stop=selfcheck_early_stop,
                    max_samples=selfcheck_max_samples,
                )
            )
            selfcheck_runners[key] = runner
        return runner

    def retriever_for(documents: Sequence[Mapping[str, Any] | str]) -> CachedRetriever:
        key = stable_cache_key({"documents": documents, "min_overlap": retriever_min_overlap})
        retriever = retrievers.get(key)
        if retriever is None:
            retriever = CachedRetriever(InMemoryRetriever(documents, min_overlap=retriever_min_overlap))
            retrievers[key] = retriever
        return retriever

    for record_index, record in enumerate(records):
        stage_payload = None
        if stage_policy is not None:
            score = float(stage_scores[record_index].item())  # type: ignore[index]
            diagnostic_decision = _stage_diagnostic_decision(
                score=score,
                threshold=float(stage_threshold),
                direction=stage_direction,
                signal=stage_signal,
            )
            stage_decision = stage_policy.decide(
                diagnostic_decision,
                claims=(record.claim,),
                context={"record_index": record_index},
            )
            stage_payload = _stage_record_payload(
                decision=stage_decision,
                diagnostic_decision=diagnostic_decision,
                score=score,
                threshold=float(stage_threshold),
                direction=stage_direction,
                alpha=float(stage_alpha),
                signal=stage_signal,
            )
            if not stage_decision.run_verifier:
                verified.append(_staged_skip_record(record, stage_payload))
                continue
        qa_result = None
        fact_result = None
        state_result = None
        selfcheck_result = None
        retrieval_qa_result = None
        attempted_routes = []
        route_timings: list[dict[str, Any]] = []
        if qa_runner is not None:
            attempted_routes.append("structured_qa")
            qa_result = _timed_verify(
                route_timings,
                route="structured_qa",
                runner=qa_runner,
                claim=record.claim,
                context={"statement": record.metadata.get("statement", {})},
            )
            if qa_result.status in {VerificationStatus.SUPPORTED, VerificationStatus.REFUTED}:
                verified.append({
                    "claim": {
                        "text": record.claim.text,
                        "claim_id": record.claim.claim_id,
                        "metadata": dict(record.claim.metadata),
                    },
                    "initial": _verification_to_dict(qa_result),
                    "final": _verification_to_dict(qa_result),
                    "qa": _verification_to_dict(qa_result),
                    "state": None,
                    "transition": None,
                    "selfcheck": None,
                    "retrieval_hits": (),
                    "route": _route_metadata(
                        selected_route="structured_qa",
                        selected_verifier="QuestionAnswerVerifier",
                        attempted_routes=attempted_routes,
                        used_retrieval=False,
                        route_timings=route_timings,
                    ),
                    "metadata": _record_metadata(record, stage_payload),
                })
                continue

        if fact_runner is not None:
            attempted_routes.append("structured_fact")
            fact_result = _timed_verify(
                route_timings,
                route="structured_fact",
                runner=fact_runner,
                claim=record.claim,
                context={"statement": record.metadata.get("statement", {})},
            )
            if fact_result.status in {VerificationStatus.SUPPORTED, VerificationStatus.REFUTED}:
                verified.append({
                    "claim": {
                        "text": record.claim.text,
                        "claim_id": record.claim.claim_id,
                        "metadata": dict(record.claim.metadata),
                    },
                    "initial": _verification_to_dict(fact_result),
                    "final": _verification_to_dict(fact_result),
                    "qa": None if qa_result is None else _verification_to_dict(qa_result),
                    "fact": _verification_to_dict(fact_result),
                    "state": None,
                    "transition": None,
                    "selfcheck": None,
                    "retrieval_hits": (),
                    "route": _route_metadata(
                        selected_route="structured_fact",
                        selected_verifier="StructuredFactVerifier",
                        attempted_routes=attempted_routes,
                        used_retrieval=False,
                        route_timings=route_timings,
                    ),
                    "metadata": _record_metadata(record, stage_payload),
                })
                continue

        if transition_runner is not None and _record_has_state_transition(record, state_transitions):
            attempted_routes.append("state_transition")
            transition_result = _timed_verify(
                route_timings,
                route="state_transition",
                runner=transition_runner,
                claim=record.claim,
                context=_transition_context(record, state_transitions),
            )
            if transition_result.status in {VerificationStatus.SUPPORTED, VerificationStatus.REFUTED}:
                verified.append({
                    "claim": {
                        "text": record.claim.text,
                        "claim_id": record.claim.claim_id,
                        "metadata": dict(record.claim.metadata),
                    },
                    "initial": _verification_to_dict(transition_result),
                    "final": _verification_to_dict(transition_result),
                    "qa": None if qa_result is None else _verification_to_dict(qa_result),
                    "state": None,
                    "transition": _verification_to_dict(transition_result),
                    "selfcheck": None,
                    "retrieval_hits": (),
                    "route": _route_metadata(
                        selected_route="state_transition",
                        selected_verifier="StateTransitionVerifier",
                        attempted_routes=attempted_routes,
                        used_retrieval=False,
                        route_timings=route_timings,
                    ),
                    "metadata": _record_metadata(record, stage_payload),
                })
                continue

        if state_runner is not None and _record_has_state_check(record, state_checks):
            attempted_routes.append("structured_state")
            state_result = _timed_verify(
                route_timings,
                route="structured_state",
                runner=state_runner,
                claim=record.claim,
                context=_state_context(record, state_checks),
            )
            if state_result.status in {VerificationStatus.SUPPORTED, VerificationStatus.REFUTED}:
                verified.append({
                    "claim": {
                        "text": record.claim.text,
                        "claim_id": record.claim.claim_id,
                        "metadata": dict(record.claim.metadata),
                    },
                    "initial": _verification_to_dict(state_result),
                    "final": _verification_to_dict(state_result),
                    "qa": None if qa_result is None else _verification_to_dict(qa_result),
                    "state": _verification_to_dict(state_result),
                    "transition": None,
                    "selfcheck": None,
                    "retrieval_hits": (),
                    "route": _route_metadata(
                        selected_route="structured_state",
                        selected_verifier="StructuredStateVerifier",
                        attempted_routes=attempted_routes,
                        used_retrieval=False,
                        route_timings=route_timings,
                    ),
                    "metadata": _record_metadata(record, stage_payload),
                })
                continue

        triple_result = None
        if enable_triple_evidence and _record_has_triple_evidence(record):
            attempted_routes.append("triple_evidence")
            triple_result = _timed_verify(
                route_timings,
                route="triple_evidence",
                runner=triple_evidence_runner(record.initial_evidence),
                claim=record.claim,
            )
            if triple_result.status is not VerificationStatus.NOT_APPLICABLE:
                verified.append({
                    "claim": {
                        "text": record.claim.text,
                        "claim_id": record.claim.claim_id,
                        "metadata": dict(record.claim.metadata),
                    },
                    "initial": _verification_to_dict(triple_result),
                    "final": _verification_to_dict(triple_result),
                    "qa": None if qa_result is None else _verification_to_dict(qa_result),
                    "state": None if state_result is None else _verification_to_dict(state_result),
                    "transition": None,
                    "triple_evidence": _verification_to_dict(triple_result),
                    "selfcheck": None,
                    "retrieval_hits": (),
                    "route": _route_metadata(
                        selected_route="triple_evidence",
                        selected_verifier="TripleEvidenceVerifier",
                        attempted_routes=attempted_routes,
                        used_retrieval=False,
                        route_timings=route_timings,
                    ),
                    "metadata": _record_metadata(record, stage_payload),
                })
                continue

        attempted_routes.append("groundedness")
        initial = _timed_verify(
            route_timings,
            route="groundedness",
            runner=groundedness_runner(record.initial_evidence, record.refutations),
            claim=record.claim,
        )
        hits = ()
        final = initial
        selected_route = "groundedness"
        selected_verifier = "GroundednessVerifier"
        selected_retrieval_hits: tuple[Mapping[str, Any], ...] = ()
        if initial.status is VerificationStatus.INSUFFICIENT_EVIDENCE and record.selfcheck_samples:
            attempted_routes.append("self_consistency")
            selfcheck_result = _timed_verify(
                route_timings,
                route="self_consistency",
                runner=selfcheck_runner(record.selfcheck_samples),
                claim=record.claim,
            )
            final = selfcheck_result
            selected_route = "self_consistency"
            selected_verifier = "SelfConsistencyVerifier"
        if initial.status is VerificationStatus.INSUFFICIENT_EVIDENCE and record.retrieval_documents:
            if final.status not in {
                VerificationStatus.INSUFFICIENT_EVIDENCE,
                VerificationStatus.NOT_APPLICABLE,
            }:
                selected_route = "self_consistency"
                selected_verifier = "SelfConsistencyVerifier"
            else:
                final = initial if selfcheck_result is None else selfcheck_result
        if final.status in {
            VerificationStatus.INSUFFICIENT_EVIDENCE,
            VerificationStatus.NOT_APPLICABLE,
        } and record.retrieval_documents:
            fixture_documents = _retrieval_document_payloads(record.retrieval_documents)
            if fixture_documents:
                runner = retrieval_qa_runner(fixture_documents)
                if runner is not None:
                    attempted_routes.append("retrieval_structured_qa")
                    retrieval_qa_result = _timed_verify(
                        route_timings,
                        route="retrieval_structured_qa",
                        runner=runner,
                        claim=record.claim,
                        context={"statement": record.metadata.get("statement", {})},
                    )
                    if retrieval_qa_result.status in {
                        VerificationStatus.SUPPORTED,
                        VerificationStatus.REFUTED,
                    }:
                        final = retrieval_qa_result
                        selected_route = "retrieval_structured_qa"
                        selected_verifier = "QuestionAnswerVerifier"
                        selected_retrieval_hits = fixture_documents
            hit_documents: tuple[Mapping[str, Any], ...] = ()
            if final.status in {
                VerificationStatus.INSUFFICIENT_EVIDENCE,
                VerificationStatus.NOT_APPLICABLE,
            }:
                retriever = retriever_for(record.retrieval_documents)
                hits = _timed_retrieve(
                    route_timings,
                    retriever=retriever,
                    query=RetrievalQuery(query=record.claim.text, claim_id=record.claim.claim_id),
                    limit=retrieval_limit,
                )
                hit_documents = tuple(hit.to_dict() for hit in hits)
            qa_documents = hit_documents
            if qa_documents and retrieval_qa_result is None:
                runner = retrieval_qa_runner(qa_documents)
                if runner is not None:
                    attempted_routes.append("retrieval_structured_qa")
                    retrieval_qa_result = _timed_verify(
                        route_timings,
                        route="retrieval_structured_qa",
                        runner=runner,
                        claim=record.claim,
                        context={"statement": record.metadata.get("statement", {})},
                    )
                    if retrieval_qa_result.status in {
                        VerificationStatus.SUPPORTED,
                        VerificationStatus.REFUTED,
                    }:
                        final = retrieval_qa_result
                        selected_route = "retrieval_structured_qa"
                        selected_verifier = "QuestionAnswerVerifier"
                        selected_retrieval_hits = qa_documents
                        _retag_retrieval_timings(
                            route_timings,
                            from_route="retrieval_groundedness",
                            to_route="retrieval_structured_qa",
                        )
                if final.status in {
                    VerificationStatus.INSUFFICIENT_EVIDENCE,
                    VerificationStatus.NOT_APPLICABLE,
                } and hit_documents:
                    attempted_routes.append("retrieval_groundedness")
                    final_evidence = tuple(record.initial_evidence) + hit_documents
                    final = _timed_verify(
                        route_timings,
                        route="retrieval_groundedness",
                        runner=groundedness_runner(final_evidence, record.refutations),
                        claim=record.claim,
                    )
                    selected_route = "retrieval_groundedness"
                    selected_verifier = "GroundednessVerifier"
                    selected_retrieval_hits = hit_documents
            if not hit_documents and final.status in {
                VerificationStatus.INSUFFICIENT_EVIDENCE,
                VerificationStatus.NOT_APPLICABLE,
            }:
                attempted_routes.append("retrieval_groundedness")
        verified.append({
            "claim": {
                "text": record.claim.text,
                "claim_id": record.claim.claim_id,
                "metadata": dict(record.claim.metadata),
            },
            "initial": _verification_to_dict(initial),
            "final": _verification_to_dict(final),
            "qa": None if qa_result is None else _verification_to_dict(qa_result),
            "fact": None if fact_result is None else _verification_to_dict(fact_result),
            "state": None if state_result is None else _verification_to_dict(state_result),
            "transition": None,
            "triple_evidence": None if triple_result is None else _verification_to_dict(triple_result),
            "selfcheck": None if selfcheck_result is None else _verification_to_dict(selfcheck_result),
            "retrieval_qa": None if retrieval_qa_result is None else _verification_to_dict(retrieval_qa_result),
            "retrieval_hits": selected_retrieval_hits,
            "route": _route_metadata(
                selected_route=selected_route,
                selected_verifier=selected_verifier,
                attempted_routes=attempted_routes,
                used_retrieval=bool(selected_retrieval_hits),
                route_timings=route_timings,
            ),
            "metadata": _record_metadata(record, stage_payload),
        })
    if cache_stats is not None:
        qa_stats = {} if qa_runner is None else qa_runner.stats.to_dict()
        fact_stats = {} if fact_runner is None else fact_runner.stats.to_dict()
        state_stats = {} if state_runner is None else state_runner.stats.to_dict()
        transition_stats = {} if transition_runner is None else transition_runner.stats.to_dict()
        groundedness_stats = combine_cache_stats(
            *(runner.stats.to_dict() for runner in groundedness_runners.values())
        )
        triple_evidence_stats = combine_cache_stats(
            *(runner.stats.to_dict() for runner in triple_evidence_runners.values())
        )
        retrieval_qa_stats = combine_cache_stats(
            *(runner.stats.to_dict() for runner in retrieval_qa_runners.values())
        )
        selfcheck_stats = combine_cache_stats(
            *(runner.stats.to_dict() for runner in selfcheck_runners.values())
        )
        retriever_stats = combine_cache_stats(*(retriever.stats.to_dict() for retriever in retrievers.values()))
        cache_stats.update({
            "qa_verifier": qa_stats,
            "fact_verifier": fact_stats,
            "state_verifier": state_stats,
            "transition_verifier": transition_stats,
            "groundedness_verifiers": {
                **groundedness_stats,
                "instances": len(groundedness_runners),
            },
            "triple_evidence_verifiers": {
                **triple_evidence_stats,
                "instances": len(triple_evidence_runners),
            },
            "retrieval_qa_verifiers": {
                **retrieval_qa_stats,
                "instances": len(retrieval_qa_runners),
            },
            "selfcheck_verifiers": {
                **selfcheck_stats,
                "instances": len(selfcheck_runners),
            },
            "retrievers": {
                **retriever_stats,
                "instances": len(retrievers),
            },
            "total": combine_cache_stats(
                qa_stats,
                state_stats,
                transition_stats,
                groundedness_stats,
                triple_evidence_stats,
                retrieval_qa_stats,
                selfcheck_stats,
                retriever_stats,
            ),
        })
    return tuple(verified)


def _record_has_state_check(record: ClaimEvidenceRecord, state_checks: Mapping[str, Any]) -> bool:
    metadata = record.claim.metadata if isinstance(record.claim.metadata, Mapping) else {}
    if "state_check" in metadata or any(key in metadata for key in ("path", "key", "field")):
        return True
    return record.claim.claim_id is not None and record.claim.claim_id in state_checks


def _record_has_triple_evidence(record: ClaimEvidenceRecord) -> bool:
    metadata = record.claim.metadata if isinstance(record.claim.metadata, Mapping) else {}
    if flag_value_enabled(metadata.get("requires_triple_audit")):
        return True
    if metadata.get("triples") is not None or metadata.get("claim_triples") is not None:
        return True
    features = metadata.get("features", {})
    if isinstance(features, Mapping):
        return any(
            flag_value_enabled(features.get(key))
            for key in ("has_number", "has_citation", "is_time_sensitive")
        )
    return False


def _retrieval_document_payloads(
    documents: Sequence[Mapping[str, Any] | str],
) -> tuple[Mapping[str, Any], ...]:
    """Return retrieval documents in the same JSON shape as retriever hits."""
    payloads = []
    for item in documents:
        if isinstance(item, str):
            text = item.strip()
            if text:
                payloads.append({"text": text, "source": None, "score": 1.0, "metadata": {}})
            continue
        text = item.get("text", item.get("content"))
        if text is None or not str(text).strip():
            continue
        source = item.get("source")
        try:
            score = float(item.get("score", 1.0))
        except (TypeError, ValueError):
            score = 1.0
        payloads.append({
            "text": str(text),
            "source": None if source is None else str(source),
            "score": score if math.isfinite(score) else 1.0,
            "metadata": dict(item.get("metadata", {})),
        })
    return tuple(payloads)


def _state_context(record: ClaimEvidenceRecord, state_checks: Mapping[str, Any]) -> dict[str, Any]:
    context: dict[str, Any] = {"statement": record.metadata.get("statement", {})}
    if record.state:
        context["state"] = dict(record.state)
    if state_checks:
        context["state_checks"] = dict(state_checks)
    return context


def _record_has_state_transition(record: ClaimEvidenceRecord, state_transitions: Mapping[str, Any]) -> bool:
    metadata = record.claim.metadata if isinstance(record.claim.metadata, Mapping) else {}
    if "state_transition" in metadata:
        return True
    if "action" in metadata and any(key in metadata for key in ("postcondition", "state_check", "check")):
        return True
    return record.claim.claim_id is not None and record.claim.claim_id in state_transitions


def _transition_context(record: ClaimEvidenceRecord, state_transitions: Mapping[str, Any]) -> dict[str, Any]:
    context: dict[str, Any] = {"statement": record.metadata.get("statement", {})}
    if record.state:
        context["state"] = dict(record.state)
    if state_transitions:
        context["state_transitions"] = dict(state_transitions)
    return context


def _transition_routes_enabled(
    records: Sequence[ClaimEvidenceRecord],
    global_state: Mapping[str, Any],
    state_transitions: Mapping[str, Any],
) -> bool:
    if global_state or state_transitions:
        return any(_record_has_state_transition(record, state_transitions) for record in records)
    return any(record.state and _record_has_state_transition(record, state_transitions) for record in records)


def _state_routes_enabled(
    records: Sequence[ClaimEvidenceRecord],
    global_state: Mapping[str, Any],
    state_checks: Mapping[str, Any],
) -> bool:
    if global_state or state_checks:
        return any(_record_has_state_check(record, state_checks) for record in records)
    return any(record.state and _record_has_state_check(record, state_checks) for record in records)


def _route_metadata(
    *,
    selected_route: str,
    selected_verifier: str,
    attempted_routes: Sequence[str],
    used_retrieval: bool,
    route_timings: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    payload = {
        "selected_route": selected_route,
        "selected_verifier": selected_verifier,
        "attempted_routes": tuple(attempted_routes),
        "attempted_route_count": len(tuple(attempted_routes)),
        "used_retrieval": used_retrieval,
    }
    if route_timings is not None:
        timings = tuple(dict(item) for item in route_timings)
        payload.update({
            "attempted_route_timings": timings,
            "total_duration_seconds": _sum_timing_durations(timings),
            "selected_route_duration_seconds": _sum_timing_durations(timings, route=selected_route),
        })
    return payload


def _timed_verify(
    route_timings: list[dict[str, Any]],
    *,
    route: str,
    runner: Any,
    claim: Claim,
    context: Mapping[str, Any] | None = None,
) -> VerificationResult:
    started = perf_counter()
    result = runner.verify(claim, context=context)
    duration = perf_counter() - started
    route_timings.append({
        "route": route,
        "operation": "verify",
        "duration_seconds": duration,
        "status": result.status.value,
    })
    return result


def _timed_retrieve(
    route_timings: list[dict[str, Any]],
    *,
    retriever: CachedRetriever,
    query: RetrievalQuery,
    limit: int,
) -> tuple[Any, ...]:
    started = perf_counter()
    hits = tuple(retriever.retrieve(query, limit=limit))
    duration = perf_counter() - started
    route_timings.append({
        "route": "retrieval_groundedness",
        "operation": "retrieve",
        "duration_seconds": duration,
        "hit_count": len(hits),
    })
    return hits


def _retag_retrieval_timings(
    route_timings: list[dict[str, Any]],
    *,
    from_route: str,
    to_route: str,
) -> None:
    """Assign shared retrieval latency to the route that consumed the hits."""
    for item in route_timings:
        if item.get("operation") == "retrieve" and item.get("route") == from_route:
            item["route"] = to_route


def _sum_timing_durations(
    route_timings: Sequence[Mapping[str, Any]],
    *,
    route: str | None = None,
) -> float | None:
    durations = []
    for item in route_timings:
        if route is not None and item.get("route") != route:
            continue
        duration = _finite_float(item.get("duration_seconds"))
        if duration is not None:
            durations.append(duration)
    if not durations:
        return None
    return float(sum(durations))


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _verification_to_dict(result: VerificationResult) -> dict[str, Any]:
    return {
        "status": result.status.value,
        "confidence": result.confidence,
        "evidence": tuple(result.evidence),
        "explanation": result.explanation,
        "metadata": dict(result.metadata),
    }


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(float(value) for value in values) if values else float("nan")


def _status_counts(verified_records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {status.value: 0 for status in VerificationStatus}
    for record in verified_records:
        status = str(record["final"]["status"])
        counts[status] = counts.get(status, 0) + 1
    return counts


def _verification_quality(
    verified_records: Sequence[Mapping[str, Any]],
    labels: torch.Tensor,
) -> dict[str, Any]:
    if labels.numel() != len(verified_records):
        raise ValueError("labels and verified records must have the same length.")
    matrix = {
        "true": {status.value: 0 for status in VerificationStatus},
        "false": {status.value: 0 for status in VerificationStatus},
    }
    for label, record in zip(labels.tolist(), verified_records):
        label_key = "false" if int(label) == 1 else "true"
        status = str(record["final"]["status"])
        matrix[label_key][status] = matrix[label_key].get(status, 0) + 1

    true_total = int((labels == 0).sum().item())
    false_total = int((labels == 1).sum().item())
    true_supported = matrix["true"][VerificationStatus.SUPPORTED.value]
    true_refuted = matrix["true"][VerificationStatus.REFUTED.value]
    false_supported = matrix["false"][VerificationStatus.SUPPORTED.value]
    false_refuted = matrix["false"][VerificationStatus.REFUTED.value]
    decided_total = true_supported + true_refuted + false_supported + false_refuted
    correct_decisions = true_supported + false_refuted
    wrong_decisions = true_refuted + false_supported
    insufficient_total = (
        matrix["true"][VerificationStatus.INSUFFICIENT_EVIDENCE.value]
        + matrix["false"][VerificationStatus.INSUFFICIENT_EVIDENCE.value]
    )

    return {
        "label_status_matrix": matrix,
        "true_supported_rate": _safe_div(true_supported, true_total),
        "true_refuted_rate": _safe_div(true_refuted, true_total),
        "false_refuted_rate": _safe_div(false_refuted, false_total),
        "false_supported_rate": _safe_div(false_supported, false_total),
        "insufficient_evidence_rate": _safe_div(insufficient_total, len(verified_records)),
        "decision_accuracy": _safe_div(correct_decisions, decided_total),
        "decision_error_rate": _safe_div(wrong_decisions, decided_total),
        "n_decided_supported_or_refuted": decided_total,
    }


def _route_summary(
    verified_records: Sequence[Mapping[str, Any]],
    labels: torch.Tensor,
) -> dict[str, Any]:
    if labels.numel() != len(verified_records):
        raise ValueError("labels and verified records must have the same length.")
    status_values = tuple(status.value for status in VerificationStatus)
    selected_counts: dict[str, int] = {}
    attempted_counts: dict[str, int] = {}
    by_route: dict[str, dict[str, Any]] = {}
    records_by_route: dict[str, list[Mapping[str, Any]]] = {}
    for label, record in zip(labels.tolist(), verified_records):
        route = record.get("route", {})
        if not isinstance(route, Mapping):
            route = {}
        selected_route = str(route.get("selected_route", "unknown"))
        records_by_route.setdefault(selected_route, []).append(record)
        selected_counts[selected_route] = selected_counts.get(selected_route, 0) + 1
        for attempted in route.get("attempted_routes", ()):
            route_name = str(attempted)
            attempted_counts[route_name] = attempted_counts.get(route_name, 0) + 1

        if selected_route not in by_route:
            by_route[selected_route] = {
                "selected": 0,
                "statuses": {status: 0 for status in status_values},
                "labels": {"true": 0, "false": 0},
                "used_retrieval": 0,
            }
        payload = by_route[selected_route]
        payload["selected"] += 1
        status = str(record["final"]["status"])
        payload["statuses"][status] = payload["statuses"].get(status, 0) + 1
        label_key = "false" if int(label) == 1 else "true"
        payload["labels"][label_key] = payload["labels"].get(label_key, 0) + 1
        if route.get("used_retrieval"):
            payload["used_retrieval"] += 1

    for route_name, payload in by_route.items():
        selected = int(payload["selected"])
        payload["rates"] = {
            "supported": _safe_div(payload["statuses"].get(VerificationStatus.SUPPORTED.value, 0), selected),
            "refuted": _safe_div(payload["statuses"].get(VerificationStatus.REFUTED.value, 0), selected),
            "insufficient_evidence": _safe_div(
                payload["statuses"].get(VerificationStatus.INSUFFICIENT_EVIDENCE.value, 0),
                selected,
            ),
            "error": _safe_div(payload["statuses"].get(VerificationStatus.ERROR.value, 0), selected),
        }
        payload.update(_route_cost_metrics(records_by_route.get(route_name, ())))
    return {
        "selected_counts": selected_counts,
        "attempted_counts": attempted_counts,
        "by_route": by_route,
    }


def _route_quality(
    verified_records: Sequence[Mapping[str, Any]],
    labels: torch.Tensor,
) -> dict[str, Any]:
    """Return label-conditioned verifier quality broken down by selected route."""
    if labels.numel() != len(verified_records):
        raise ValueError("labels and verified records must have the same length.")
    grouped: dict[str, dict[str, Any]] = {}
    for label, record in zip(labels.tolist(), verified_records):
        route = _selected_route(record)
        payload = grouped.setdefault(route, {"records": [], "labels": []})
        payload["records"].append(record)
        payload["labels"].append(int(label))

    quality: dict[str, Any] = {}
    total = len(verified_records)
    for route, payload in sorted(grouped.items()):
        route_labels = torch.tensor(payload["labels"], dtype=torch.int64)
        route_quality = _verification_quality(payload["records"], route_labels)
        selected = len(payload["records"])
        quality[route] = {
            "selected": selected,
            "selection_rate": _safe_div(selected, total),
            "n_true": int((route_labels == 0).sum().item()),
            "n_false": int((route_labels == 1).sum().item()),
            **route_quality,
            **_route_cost_metrics(payload["records"]),
        }
    return quality


def _route_cost_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize route-level verification cost from per-record route metadata."""
    total_durations = []
    selected_route_durations = []
    attempted_route_counts = []
    used_retrieval_count = 0
    retrieval_hit_count = 0
    for record in records:
        route = record.get("route", {})
        if not isinstance(route, Mapping):
            route = {}
        total_duration = _finite_float(route.get("total_duration_seconds"))
        if total_duration is not None:
            total_durations.append(total_duration)
        selected_duration = _finite_float(route.get("selected_route_duration_seconds"))
        if selected_duration is not None:
            selected_route_durations.append(selected_duration)
        attempted_count = _finite_float(route.get("attempted_route_count"))
        if attempted_count is not None:
            attempted_route_counts.append(attempted_count)
        if route.get("used_retrieval"):
            used_retrieval_count += 1
        hits = record.get("retrieval_hits", ())
        if isinstance(hits, Sequence) and not isinstance(hits, (str, bytes, bytearray)):
            retrieval_hit_count += len(hits)

    selected = len(records)
    return {
        "duration_observations": len(total_durations),
        "total_duration_seconds": None if not total_durations else float(sum(total_durations)),
        "mean_duration_seconds": _mean_or_none(total_durations),
        "p95_duration_seconds": _percentile_or_none(total_durations, 95.0),
        "p99_duration_seconds": _percentile_or_none(total_durations, 99.0),
        "max_duration_seconds": None if not total_durations else max(total_durations),
        "selected_route_duration_observations": len(selected_route_durations),
        "total_selected_route_duration_seconds": (
            None if not selected_route_durations else float(sum(selected_route_durations))
        ),
        "mean_selected_route_duration_seconds": _mean_or_none(selected_route_durations),
        "p95_selected_route_duration_seconds": _percentile_or_none(selected_route_durations, 95.0),
        "p99_selected_route_duration_seconds": _percentile_or_none(selected_route_durations, 99.0),
        "attempted_route_count_observations": len(attempted_route_counts),
        "total_attempted_route_count": None if not attempted_route_counts else float(sum(attempted_route_counts)),
        "mean_attempted_route_count": _mean_or_none(attempted_route_counts),
        "used_retrieval_count": used_retrieval_count,
        "retrieval_use_rate": _safe_div(used_retrieval_count, selected),
        "retrieval_hit_count": retrieval_hit_count,
        "mean_retrieval_hits": _safe_div(retrieval_hit_count, selected),
    }


def _selfcheck_execution_summary(verified_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize self-consistency sample processing from verification metadata."""
    records_with_selfcheck = 0
    early_stopped_records = 0
    available_samples = 0
    considered_samples = 0
    processed_samples = 0
    skipped_samples = 0
    for record in verified_records:
        payload = record.get("selfcheck")
        if not isinstance(payload, Mapping):
            continue
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, Mapping):
            metadata = {}
        records_with_selfcheck += 1
        considered = _non_negative_int(metadata.get("sample_count"))
        available = _non_negative_int(metadata.get("available_sample_count"))
        processed = _non_negative_int(metadata.get("processed_sample_count"))
        skipped = _non_negative_int(metadata.get("skipped_sample_count"))
        considered_samples += considered if considered is not None else 0
        available_samples += available if available is not None else (considered or 0)
        processed_samples += processed if processed is not None else (considered or 0)
        skipped_samples += skipped if skipped is not None else 0
        if metadata.get("early_stop"):
            early_stopped_records += 1
    return {
        "executed_records": records_with_selfcheck,
        "early_stopped_records": early_stopped_records,
        "available_samples": available_samples,
        "considered_samples": considered_samples,
        "processed_samples": processed_samples,
        "skipped_samples": skipped_samples,
        "processing_rate": _safe_div(processed_samples, considered_samples),
    }


def _staged_verification_summary(
    verified_records: Sequence[Mapping[str, Any]],
    *,
    enabled: bool,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "enabled": enabled,
        "total_records": len(verified_records),
        "verified_records": 0,
        "skipped_records": 0,
        "skip_rate": None,
        "reason_counts": {},
        "risk_level_counts": {},
        "action_counts": {},
        "triggered_claim_count": 0,
        "triggered_feature_counts": {},
        "triggered_metadata_counts": {},
    }
    if not enabled:
        return summary

    reason_counts: dict[str, int] = {}
    risk_level_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    triggered_feature_counts: dict[str, int] = {}
    triggered_metadata_counts: dict[str, int] = {}
    triggered_claim_ids: set[str] = set()
    for record in verified_records:
        metadata = record.get("metadata", {})
        if not isinstance(metadata, Mapping):
            metadata = {}
        stage = metadata.get("staged_verification", {})
        if not isinstance(stage, Mapping):
            continue
        if stage.get("run_verifier"):
            summary["verified_records"] += 1
        else:
            summary["skipped_records"] += 1
        reason = str(stage.get("reason", "unknown"))
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        diagnostic = stage.get("diagnostic_decision", {})
        if isinstance(diagnostic, Mapping):
            risk_level = str(diagnostic.get("risk_level", "unknown"))
            action = str(diagnostic.get("action", "unknown"))
            risk_level_counts[risk_level] = risk_level_counts.get(risk_level, 0) + 1
            action_counts[action] = action_counts.get(action, 0) + 1
        for claim_id in stage.get("triggered_claim_ids", ()):
            triggered_claim_ids.add(str(claim_id))
        for features in _mapping_values(stage.get("triggered_features", {})):
            for feature in features:
                feature_name = str(feature)
                triggered_feature_counts[feature_name] = triggered_feature_counts.get(feature_name, 0) + 1
        for keys in _mapping_values(stage.get("triggered_metadata", {})):
            for key in keys:
                key_name = str(key)
                triggered_metadata_counts[key_name] = triggered_metadata_counts.get(key_name, 0) + 1

    summary["skip_rate"] = _safe_div(int(summary["skipped_records"]), len(verified_records))
    summary["reason_counts"] = reason_counts
    summary["risk_level_counts"] = risk_level_counts
    summary["action_counts"] = action_counts
    summary["triggered_claim_count"] = len(triggered_claim_ids)
    summary["triggered_feature_counts"] = triggered_feature_counts
    summary["triggered_metadata_counts"] = triggered_metadata_counts
    return summary


def _mapping_values(value: Any) -> tuple[Sequence[Any], ...]:
    if not isinstance(value, Mapping):
        return ()
    return tuple(
        tuple(item) if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)) else (item,)
        for item in value.values()
    )


def _non_negative_int(value: Any) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _safe_div(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _verified_trigger(internal_trigger: bool, final_status: str) -> bool:
    if final_status == VerificationStatus.REFUTED.value:
        return True
    if final_status == VerificationStatus.SUPPORTED.value:
        return False
    return internal_trigger


def _rate(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return float("nan")
    return float(values.to(torch.float64).mean().item())


def _evaluate_alpha(
    *,
    scores: torch.Tensor,
    labels: torch.Tensor,
    verified_records: Sequence[Mapping[str, Any]],
    alpha: float,
    direction: str,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    true_idx = torch.nonzero(labels == 0, as_tuple=False).flatten()
    false_idx = torch.nonzero(labels == 1, as_tuple=False).flatten()
    false_alarm_internal = []
    detection_internal = []
    false_alarm_verified = []
    detection_verified = []
    suppressed_false_alarms = []
    rescued_detections = []

    final_statuses = tuple(str(record["final"]["status"]) for record in verified_records)
    selected_routes = tuple(_selected_route(record) for record in verified_records)
    route_impacts = _empty_route_impact_accumulator(selected_routes, labels)
    for repeat in range(repeats):
        generator = torch.Generator().manual_seed(seed + repeat)
        perm = true_idx[torch.randperm(true_idx.numel(), generator=generator)]
        half = true_idx.numel() // 2
        calib_idx = perm[:half]
        test_true_idx = perm[half:]
        threshold = directional_conformal_threshold(scores[calib_idx], alpha, direction)

        internal_true = _internal_triggers(scores, test_true_idx, threshold, direction)
        internal_false = _internal_triggers(scores, false_idx, threshold, direction)
        verified_true = _verified_triggers(internal_true, test_true_idx, final_statuses)
        verified_false = _verified_triggers(internal_false, false_idx, final_statuses)

        false_alarm_internal.append(_rate(internal_true))
        detection_internal.append(_rate(internal_false))
        false_alarm_verified.append(_rate(verified_true))
        detection_verified.append(_rate(verified_false))
        suppressed_false_alarms.append(_rate(internal_true & ~verified_true))
        rescued_detections.append(_rate(~internal_false & verified_false))
        _accumulate_route_control_impact(
            route_impacts,
            selected_routes=selected_routes,
            test_true_idx=test_true_idx,
            false_idx=false_idx,
            internal_true=internal_true,
            internal_false=internal_false,
            verified_true=verified_true,
            verified_false=verified_false,
        )

    internal_fa = _mean(false_alarm_internal)
    verified_fa = _mean(false_alarm_verified)
    return {
        "alpha": alpha,
        "internal": {
            "false_alarm": internal_fa,
            "coverage": 1.0 - internal_fa,
            "detection": _mean(detection_internal),
            "pass": abs(internal_fa - alpha) <= TOLERANCE,
        },
        "verified": {
            "false_alarm": verified_fa,
            "coverage": 1.0 - verified_fa,
            "detection": _mean(detection_verified),
            "pass": verified_fa <= alpha + TOLERANCE,
        },
        "delta": {
            "false_alarm": verified_fa - internal_fa,
            "detection": _mean(detection_verified) - _mean(detection_internal),
            "suppressed_false_alarm_rate": _mean(suppressed_false_alarms),
            "rescued_detection_rate": _mean(rescued_detections),
        },
        "route_control_impact": _finalize_route_control_impact(route_impacts),
        "repeats": repeats,
    }


def _internal_triggers(
    scores: torch.Tensor,
    idx: torch.Tensor,
    threshold: float,
    direction: str,
) -> torch.Tensor:
    if idx.numel() == 0:
        return torch.zeros(0, dtype=torch.bool)
    if direction == "higher":
        return scores[idx] > threshold
    if direction == "lower":
        return scores[idx] < threshold
    raise ValueError("direction must be 'higher' or 'lower'.")


def _verified_triggers(
    internal_triggers: torch.Tensor,
    idx: torch.Tensor,
    final_statuses: Sequence[str],
) -> torch.Tensor:
    values = [
        _verified_trigger(bool(trigger), final_statuses[int(item)])
        for trigger, item in zip(internal_triggers.tolist(), idx.tolist())
    ]
    return torch.tensor(values, dtype=torch.bool)


def _selected_route(record: Mapping[str, Any]) -> str:
    route = record.get("route", {})
    if not isinstance(route, Mapping):
        return "unknown"
    return str(route.get("selected_route", "unknown"))


def _empty_route_impact_accumulator(
    selected_routes: Sequence[str],
    labels: torch.Tensor,
) -> dict[str, dict[str, Any]]:
    routes = sorted(set(selected_routes))
    return {
        route: {
            "n_selected": sum(1 for item in selected_routes if item == route),
            "n_true": sum(
                1 for index, item in enumerate(selected_routes)
                if item == route and int(labels[index].item()) == 0
            ),
            "n_false": sum(
                1 for index, item in enumerate(selected_routes)
                if item == route and int(labels[index].item()) == 1
            ),
            "false_alarm_internal": [],
            "false_alarm_verified": [],
            "detection_internal": [],
            "detection_verified": [],
            "suppressed_false_alarm_rate": [],
            "rescued_detection_rate": [],
        }
        for route in routes
    }


def _accumulate_route_control_impact(
    route_impacts: dict[str, dict[str, Any]],
    *,
    selected_routes: Sequence[str],
    test_true_idx: torch.Tensor,
    false_idx: torch.Tensor,
    internal_true: torch.Tensor,
    internal_false: torch.Tensor,
    verified_true: torch.Tensor,
    verified_false: torch.Tensor,
) -> None:
    test_true_items = test_true_idx.tolist()
    false_items = false_idx.tolist()
    for route, payload in route_impacts.items():
        true_positions = [
            pos for pos, item in enumerate(test_true_items)
            if selected_routes[int(item)] == route
        ]
        if true_positions:
            positions = torch.tensor(true_positions, dtype=torch.int64)
            route_internal_true = internal_true[positions]
            route_verified_true = verified_true[positions]
            payload["false_alarm_internal"].append(_rate(route_internal_true))
            payload["false_alarm_verified"].append(_rate(route_verified_true))
            payload["suppressed_false_alarm_rate"].append(_rate(route_internal_true & ~route_verified_true))

        false_positions = [
            pos for pos, item in enumerate(false_items)
            if selected_routes[int(item)] == route
        ]
        if false_positions:
            positions = torch.tensor(false_positions, dtype=torch.int64)
            route_internal_false = internal_false[positions]
            route_verified_false = verified_false[positions]
            payload["detection_internal"].append(_rate(route_internal_false))
            payload["detection_verified"].append(_rate(route_verified_false))
            payload["rescued_detection_rate"].append(_rate(~route_internal_false & route_verified_false))


def _finalize_route_control_impact(route_impacts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    finalized: dict[str, Any] = {}
    for route, payload in route_impacts.items():
        internal_false_alarm = _mean_or_none(payload["false_alarm_internal"])
        verified_false_alarm = _mean_or_none(payload["false_alarm_verified"])
        internal_detection = _mean_or_none(payload["detection_internal"])
        verified_detection = _mean_or_none(payload["detection_verified"])
        finalized[route] = {
            "n_selected": payload["n_selected"],
            "n_true": payload["n_true"],
            "n_false": payload["n_false"],
            "internal": {
                "false_alarm": internal_false_alarm,
                "detection": internal_detection,
            },
            "verified": {
                "false_alarm": verified_false_alarm,
                "detection": verified_detection,
            },
            "delta": {
                "false_alarm": _optional_delta(verified_false_alarm, internal_false_alarm),
                "detection": _optional_delta(verified_detection, internal_detection),
                "suppressed_false_alarm_rate": _mean_or_none(payload["suppressed_false_alarm_rate"]),
                "rescued_detection_rate": _mean_or_none(payload["rescued_detection_rate"]),
            },
        }
    return finalized


def _mean_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return _mean(values)


def _percentile_or_none(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    if not (0.0 <= percentile <= 100.0):
        raise ValueError("percentile must be between 0 and 100.")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (percentile / 100.0) * (len(ordered) - 1)
    lower_index = math.floor(rank)
    upper_index = math.ceil(rank)
    if lower_index == upper_index:
        return ordered[lower_index]
    lower = ordered[lower_index]
    upper = ordered[upper_index]
    return lower + (upper - lower) * (rank - lower_index)


def _optional_delta(after: float | None, before: float | None) -> float | None:
    if after is None or before is None:
        return None
    return after - before


def _verification_trace_cache(cache_dir: Path | None) -> JsonTraceCache | None:
    if cache_dir is None:
        return None
    return JsonTraceCache(
        Path(cache_dir) / "verifier-ensemble-verified-records.json",
        cache_type="verifier_ensemble_verified_records",
    )


def _verification_trace_cache_key(
    *,
    name: str,
    score_path: Path,
    signal: str,
    claims_path: Path | None,
    qa_corpus_path: Path | None,
    fact_corpus_path: Path | None,
    state_path: Path | None,
    records: Sequence[ClaimEvidenceRecord],
    global_state: Mapping[str, Any],
    global_state_checks: Mapping[str, Any],
    global_state_transitions: Mapping[str, Any],
    verifier_min_overlap: float,
    retriever_min_overlap: float,
    retrieval_limit: int,
    selfcheck_min_samples: int,
    selfcheck_min_overlap: float,
    selfcheck_support_threshold: float,
    selfcheck_refute_threshold: float,
    selfcheck_early_stop: bool,
    selfcheck_max_samples: int | None,
    enable_triple_evidence: bool,
    triple_min_slot_coverage: float,
    min_world_model_confidence: float,
    staged_verification: bool,
    staged_alpha: float,
    staged_direction: str,
    staged_policy: StagedVerificationPolicy | None,
) -> tuple[str, dict[str, Any]]:
    material = {
        "schema_version": 1,
        "cache_type": "verifier_ensemble_verified_records",
        "builder": "eval_verifier_ensemble:verified_records:v4",
        "name": name,
        "signal": signal,
        "score_dump": _path_fingerprint(score_path),
        "claims_fixture": _path_fingerprint(claims_path),
        "qa_corpus": _path_fingerprint(qa_corpus_path),
        "fact_corpus": _path_fingerprint(fact_corpus_path),
        "state_source": _path_fingerprint(state_path),
        "records": tuple(_record_cache_material(record) for record in records),
        "global_state": dict(global_state),
        "global_state_checks": dict(global_state_checks),
        "global_state_transitions": dict(global_state_transitions),
        "verifier": {
            "min_overlap": float(verifier_min_overlap),
        },
        "retriever": {
            "type": "InMemoryRetriever",
            "min_overlap": float(retriever_min_overlap),
            "limit": int(retrieval_limit),
        },
        "selfcheck_verifier": {
            "type": "SelfConsistencyVerifier",
            "min_samples": int(selfcheck_min_samples),
            "min_overlap": float(selfcheck_min_overlap),
            "support_threshold": float(selfcheck_support_threshold),
            "refute_threshold": float(selfcheck_refute_threshold),
            "early_stop": bool(selfcheck_early_stop),
            "max_samples": selfcheck_max_samples,
        },
        "triple_evidence_verifier": {
            "type": "TripleEvidenceVerifier",
            "enabled": bool(enable_triple_evidence),
            "min_slot_coverage": float(triple_min_slot_coverage),
        },
        "transition_verifier": {
            "type": "StateTransitionVerifier",
            "min_prediction_confidence": float(min_world_model_confidence),
        },
        "staged_verification": {
            "enabled": bool(staged_verification),
            "alpha": float(staged_alpha),
            "direction": staged_direction,
            "policy": None if staged_policy is None else staged_policy.to_dict(),
        },
    }
    key = hashlib.sha256(stable_cache_key(material).encode("utf-8")).hexdigest()
    return key, material


def _record_cache_material(record: ClaimEvidenceRecord) -> dict[str, Any]:
    return {
        "claim": {
            "text": record.claim.text,
            "claim_id": record.claim.claim_id,
            "span": record.claim.span,
            "metadata": dict(record.claim.metadata),
        },
        "initial_evidence": tuple(record.initial_evidence),
        "retrieval_documents": tuple(record.retrieval_documents),
        "selfcheck_samples": tuple(record.selfcheck_samples),
        "refutations": dict(record.refutations),
        "state": dict(record.state),
        "metadata": dict(record.metadata),
    }


def _path_fingerprint(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    path = Path(path)
    try:
        data = path.read_bytes()
    except OSError as exc:
        return {
            "path": str(path),
            "missing": True,
            "error": str(exc),
        }
    return {
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _load_verified_records_from_cache(
    cache: JsonTraceCache | None,
    key: str,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]] | None:
    if cache is None:
        return None
    record = cache.get_record(key)
    if record is None:
        return None
    if not isinstance(record.payload, Mapping):
        return None
    raw_records = record.payload.get("verified_records")
    if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes, bytearray)):
        return None
    cache_stats = record.payload.get("cache_stats", {})
    if not isinstance(cache_stats, Mapping):
        cache_stats = {}
    return tuple(dict(item) for item in raw_records if isinstance(item, Mapping)), dict(cache_stats)


def _trace_cache_stats(
    *,
    enabled: bool,
    hit: bool,
    cache: JsonTraceCache | None,
    key: str | None,
) -> dict[str, Any]:
    return {
        "enabled": enabled,
        "hit": hit,
        "key": key,
        "path": None if cache is None else str(cache.path),
        "records": None if cache is None else cache.summary()["records"],
    }


def _write_verified_records_jsonl(
    stream: Any,
    *,
    run_name: str,
    score_path: Path,
    signal: str,
    labels: torch.Tensor,
    scores: torch.Tensor,
    records: Sequence[Mapping[str, Any]],
) -> None:
    """Write one compact verified-record detail per line."""
    label_values = labels.tolist()
    score_values = scores.tolist()
    for record_index, record in enumerate(records):
        stream.write(
            json.dumps(
                {
                    "schema_version": 1,
                    "workflow": "verifier_ensemble_verified_record",
                    "run": run_name,
                    "score_path": str(score_path),
                    "signal": signal,
                    "record_index": record_index,
                    "label": int(label_values[record_index]),
                    "score": float(score_values[record_index]),
                    "record": dict(record),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )


def build_verifier_ensemble_report(
    score_dumps: Sequence[tuple[str, Path]],
    *,
    signal: str,
    claims_path: Path | None = None,
    qa_corpus_path: Path | None = None,
    fact_corpus_path: Path | None = None,
    state_path: Path | None = None,
    direction: str | None = None,
    alphas: Sequence[float] = ALPHAS,
    repeats: int = 20,
    seed: int = 0,
    verifier_min_overlap: float = 0.65,
    retriever_min_overlap: float = 0.2,
    retrieval_limit: int = 5,
    selfcheck_min_samples: int = 2,
    selfcheck_min_overlap: float = 0.65,
    selfcheck_support_threshold: float = 0.60,
    selfcheck_refute_threshold: float = 0.50,
    selfcheck_early_stop: bool = False,
    selfcheck_max_samples: int | None = None,
    enable_triple_evidence: bool = False,
    triple_min_slot_coverage: float = 1.0,
    min_world_model_confidence: float = 0.0,
    verification_cache_dir: Path | None = None,
    staged_verification: bool = False,
    staged_alpha: float = 0.10,
    staged_verify_risk_levels: Sequence[str] = ("medium", "high", "unknown"),
    staged_verify_actions: Sequence[str] = (
        "retrieve",
        "rewrite",
        "steer_regenerate",
        "execute_tool",
        "abstain",
        "clarify",
    ),
    staged_verify_feature_flags: Sequence[str] = (
        "has_number",
        "has_citation",
        "is_time_sensitive",
    ),
    staged_verify_metadata_keys: Sequence[str] = ("requires_verification",),
    verified_records_path: str | Path | None = None,
) -> dict[str, Any]:
    if not score_dumps:
        raise ValueError("at least one score dump is required.")
    if repeats < 1:
        raise ValueError("repeats must be >= 1.")
    if any(not (0.0 < float(alpha) < 1.0) for alpha in alphas):
        raise ValueError("alphas must be in (0, 1).")
    if selfcheck_min_samples < 1:
        raise ValueError("selfcheck_min_samples must be >= 1.")
    if not (0.0 <= selfcheck_min_overlap <= 1.0):
        raise ValueError("selfcheck_min_overlap must be in [0, 1].")
    if not (0.0 <= selfcheck_support_threshold <= 1.0):
        raise ValueError("selfcheck_support_threshold must be in [0, 1].")
    if not (0.0 <= selfcheck_refute_threshold <= 1.0):
        raise ValueError("selfcheck_refute_threshold must be in [0, 1].")
    if selfcheck_max_samples is not None and selfcheck_max_samples < selfcheck_min_samples:
        raise ValueError("selfcheck_max_samples must be >= selfcheck_min_samples when set.")
    if not (0.0 <= triple_min_slot_coverage <= 1.0):
        raise ValueError("triple_min_slot_coverage must be in [0, 1].")
    if not (0.0 <= min_world_model_confidence <= 1.0):
        raise ValueError("min_world_model_confidence must be in [0, 1].")
    if not (0.0 < float(staged_alpha) < 1.0):
        raise ValueError("staged_alpha must be in (0, 1).")

    fixture = _load_fixture(claims_path)
    qa_verifier = _load_qa_verifier(qa_corpus_path)
    fact_verifier = _load_fact_verifier(fact_corpus_path)
    source_state, source_state_checks, source_state_transitions = _load_state_source(state_path)
    fixture_state = fixture.get("state", {})
    if not isinstance(fixture_state, Mapping):
        raise ValueError("claim fixture 'state' must be a JSON object when present.")
    fixture_state_checks = fixture.get("state_checks", {})
    if not isinstance(fixture_state_checks, Mapping):
        raise ValueError("claim fixture 'state_checks' must be a JSON object when present.")
    fixture_state_transitions = fixture.get("state_transitions", {})
    if not isinstance(fixture_state_transitions, Mapping):
        raise ValueError("claim fixture 'state_transitions' must be a JSON object when present.")
    global_state = _merge_state_mappings(source_state, fixture_state)
    global_state_checks = {**dict(source_state_checks), **dict(fixture_state_checks)}
    global_state_transitions = {**dict(source_state_transitions), **dict(fixture_state_transitions)}
    trace_cache = _verification_trace_cache(verification_cache_dir)
    stage_policy = (
        StagedVerificationPolicy(
            verify_risk_levels=tuple(staged_verify_risk_levels),
            verify_actions=tuple(staged_verify_actions),
            verify_claim_feature_flags=tuple(staged_verify_feature_flags),
            verify_claim_metadata_keys=tuple(staged_verify_metadata_keys),
        )
        if staged_verification
        else None
    )
    runs = []
    score_dump_metadata_cache = {}
    any_state_enabled = False
    any_transition_enabled = False
    any_selfcheck_enabled = False
    any_triple_evidence_enabled = False
    any_fact_enabled = fact_verifier is not None
    verified_record_counts: dict[str, int] = {}
    verified_record_total = 0
    verified_records_sidecar_path = None if verified_records_path is None else Path(verified_records_path)
    verified_records_tmp_path = (
        None
        if verified_records_sidecar_path is None
        else verified_records_sidecar_path.with_name(f"{verified_records_sidecar_path.name}.tmp")
    )
    verified_records_stream = None
    if verified_records_tmp_path is not None:
        verified_records_tmp_path.parent.mkdir(parents=True, exist_ok=True)
        verified_records_stream = verified_records_tmp_path.open("w", encoding="utf-8")

    try:
        for name, path in score_dumps:
            dump = _load_scores(path, signal, cache=score_dump_metadata_cache)
            labels = dump["labels"]
            scores = dump["scores"]
            resolved_direction = direction or DEFAULT_SCORE_DIRECTIONS.get(signal, "higher")
            records = _records_from_dump_and_fixture(
                dump=dump,
                fixture=fixture,
                expected_count=int(labels.numel()),
            )
            stage_threshold = None
            if stage_policy is not None:
                true_scores = scores[labels == 0]
                if true_scores.numel() == 0:
                    raise ValueError("staged verification requires at least one true-labeled calibration score.")
                stage_threshold = directional_conformal_threshold(
                    true_scores,
                    float(staged_alpha),
                    resolved_direction,
                )
            transition_enabled = _transition_routes_enabled(records, global_state, global_state_transitions)
            state_enabled = _state_routes_enabled(records, global_state, global_state_checks)
            any_transition_enabled = any_transition_enabled or transition_enabled
            any_state_enabled = any_state_enabled or state_enabled
            selfcheck_enabled = any(record.selfcheck_samples for record in records)
            any_selfcheck_enabled = any_selfcheck_enabled or selfcheck_enabled
            triple_evidence_enabled = bool(
                enable_triple_evidence
                and any(_record_has_triple_evidence(record) for record in records)
            )
            any_triple_evidence_enabled = any_triple_evidence_enabled or triple_evidence_enabled
            transition_verifier = (
                StateTransitionVerifier(
                    world_model=InMemoryWorldModelAdapter(StructuredStateVerifier({})),
                    state=global_state,
                    min_prediction_confidence=float(min_world_model_confidence),
                )
                if transition_enabled
                else None
            )
            state_verifier = StructuredStateVerifier(global_state) if state_enabled else None
            run_cache_stats: dict[str, Any] = {}
            trace_key, trace_material = _verification_trace_cache_key(
                name=name,
                score_path=path,
                signal=signal,
                claims_path=claims_path,
                qa_corpus_path=qa_corpus_path,
                fact_corpus_path=fact_corpus_path,
                state_path=state_path,
                records=records,
                global_state=global_state,
                global_state_checks=global_state_checks,
                global_state_transitions=global_state_transitions,
                verifier_min_overlap=verifier_min_overlap,
                retriever_min_overlap=retriever_min_overlap,
                retrieval_limit=retrieval_limit,
                selfcheck_min_samples=selfcheck_min_samples,
                selfcheck_min_overlap=selfcheck_min_overlap,
                selfcheck_support_threshold=selfcheck_support_threshold,
                selfcheck_refute_threshold=selfcheck_refute_threshold,
                selfcheck_early_stop=selfcheck_early_stop,
                selfcheck_max_samples=selfcheck_max_samples,
                enable_triple_evidence=bool(enable_triple_evidence),
                triple_min_slot_coverage=float(triple_min_slot_coverage),
                min_world_model_confidence=float(min_world_model_confidence),
                staged_verification=stage_policy is not None,
                staged_alpha=float(staged_alpha),
                staged_direction=resolved_direction,
                staged_policy=stage_policy,
            )
            cached_trace = _load_verified_records_from_cache(trace_cache, trace_key)
            if cached_trace is not None:
                verified_records, cached_stats = cached_trace
                run_cache_stats.update(cached_stats)
                run_cache_stats["trace_cache"] = _trace_cache_stats(
                    enabled=True,
                    hit=True,
                    cache=trace_cache,
                    key=trace_key,
                )
            else:
                verified_records = _verify_records(
                    records,
                    verifier_min_overlap=verifier_min_overlap,
                    retriever_min_overlap=retriever_min_overlap,
                    retrieval_limit=retrieval_limit,
                    selfcheck_min_samples=selfcheck_min_samples,
                    selfcheck_min_overlap=selfcheck_min_overlap,
                    selfcheck_support_threshold=selfcheck_support_threshold,
                    selfcheck_refute_threshold=selfcheck_refute_threshold,
                    selfcheck_early_stop=selfcheck_early_stop,
                    selfcheck_max_samples=selfcheck_max_samples,
                    enable_triple_evidence=bool(enable_triple_evidence),
                    triple_min_slot_coverage=float(triple_min_slot_coverage),
                    qa_verifier=qa_verifier,
                    fact_verifier=fact_verifier,
                    state_verifier=state_verifier,
                    state_checks=global_state_checks,
                    transition_verifier=transition_verifier,
                    state_transitions=global_state_transitions,
                    cache_stats=run_cache_stats,
                    stage_policy=stage_policy,
                    stage_scores=scores,
                    stage_threshold=stage_threshold,
                    stage_direction=resolved_direction,
                    stage_alpha=float(staged_alpha),
                    stage_signal=signal,
                )
                if trace_cache is not None:
                    trace_cache.put(
                        trace_key,
                        {
                            "verified_records": tuple(verified_records),
                            "cache_stats": dict(run_cache_stats),
                        },
                        metadata={
                            "builder": "eval_verifier_ensemble:verified_records:v4",
                            "name": name,
                            "signal": signal,
                            "material": trace_material,
                        },
                    )
                run_cache_stats["trace_cache"] = _trace_cache_stats(
                    enabled=trace_cache is not None,
                    hit=False,
                    cache=trace_cache,
                    key=trace_key if trace_cache is not None else None,
                )
            if verified_records_stream is not None:
                _write_verified_records_jsonl(
                    verified_records_stream,
                    run_name=name,
                    score_path=path,
                    signal=signal,
                    labels=labels,
                    scores=scores,
                    records=verified_records,
                )
            verified_record_count = len(verified_records)
            verified_record_counts[name] = verified_record_count
            verified_record_total += verified_record_count
            alpha_results = {
                str(alpha): _evaluate_alpha(
                    scores=scores,
                    labels=labels,
                    verified_records=verified_records,
                    alpha=float(alpha),
                    direction=resolved_direction,
                    repeats=repeats,
                    seed=seed,
                )
                for alpha in alphas
            }
            selfcheck_execution = _selfcheck_execution_summary(verified_records)
            staged_execution = _staged_verification_summary(
                verified_records,
                enabled=stage_policy is not None,
            )
            runs.append({
                "name": name,
                "scores_path": str(path),
                "score_dump": {
                    **score_dump_file_metadata(path, cache=score_dump_metadata_cache),
                    "summary": dump["score_dump_summary"],
                    "source_format": dump["score_dump_source_format"],
                },
                "config": dump["config"],
                "signal": signal,
                "direction": resolved_direction,
                "n_total": int(labels.numel()),
                "n_true": int((labels == 0).sum().item()),
                "n_false": int((labels == 1).sum().item()),
                "verified_records": {
                    "storage": "jsonl_sidecar" if verified_records_sidecar_path is not None else "summary_only",
                    "count": verified_record_count,
                    "path": None if verified_records_sidecar_path is None else str(verified_records_sidecar_path),
                },
                "verification_status_counts": _status_counts(verified_records),
                "verification_quality": _verification_quality(verified_records, labels),
                "route_summary": _route_summary(verified_records, labels),
                "route_quality": _route_quality(verified_records, labels),
                "qa": {
                    "enabled": qa_verifier is not None,
                    "decided_records": sum(
                        1 for record in verified_records
                        if record.get("qa") is not None
                        and record["qa"]["status"] in {
                            VerificationStatus.SUPPORTED.value,
                            VerificationStatus.REFUTED.value,
                        }
                    ),
                },
                "fact": {
                    "enabled": fact_verifier is not None,
                    "decided_records": sum(
                        1 for record in verified_records
                        if record.get("fact") is not None
                        and record["fact"]["status"] in {
                            VerificationStatus.SUPPORTED.value,
                            VerificationStatus.REFUTED.value,
                        }
                    ),
                },
                "retrieval_qa": {
                    "enabled": any(record.get("retrieval_qa") is not None for record in verified_records),
                    "decided_records": sum(
                        1 for record in verified_records
                        if record.get("retrieval_qa") is not None
                        and record["retrieval_qa"]["status"] in {
                            VerificationStatus.SUPPORTED.value,
                            VerificationStatus.REFUTED.value,
                        }
                    ),
                },
                "state_verifier": {
                    "enabled": state_enabled,
                    "decided_records": sum(
                        1 for record in verified_records
                        if record.get("state") is not None
                        and record["state"]["status"] in {
                            VerificationStatus.SUPPORTED.value,
                            VerificationStatus.REFUTED.value,
                        }
                    ),
                    "global_checks": len(global_state_checks),
                },
                "transition_verifier": {
                    "enabled": transition_enabled,
                    "min_prediction_confidence": float(min_world_model_confidence),
                    "decided_records": sum(
                        1 for record in verified_records
                        if record.get("transition") is not None
                        and record["transition"]["status"] in {
                            VerificationStatus.SUPPORTED.value,
                            VerificationStatus.REFUTED.value,
                        }
                    ),
                    "global_transitions": len(global_state_transitions),
                },
                "selfcheck_verifier": {
                    "enabled": selfcheck_enabled,
                    "early_stop": bool(selfcheck_early_stop),
                    "max_samples": selfcheck_max_samples,
                    "records_with_samples": sum(1 for record in records if record.selfcheck_samples),
                    "decided_records": sum(
                        1 for record in verified_records
                        if record.get("selfcheck") is not None
                        and record["selfcheck"]["status"] in {
                            VerificationStatus.SUPPORTED.value,
                            VerificationStatus.REFUTED.value,
                        }
                    ),
                    **selfcheck_execution,
                },
                "triple_evidence_verifier": {
                    "type": "TripleEvidenceVerifier",
                    "enabled": triple_evidence_enabled,
                    "min_slot_coverage": float(triple_min_slot_coverage),
                    "records_with_triple_route": sum(
                        1 for record in records
                        if _record_has_triple_evidence(record)
                    ),
                    "decided_records": sum(
                        1 for record in verified_records
                        if record.get("triple_evidence") is not None
                        and record["triple_evidence"]["status"] in {
                            VerificationStatus.SUPPORTED.value,
                            VerificationStatus.REFUTED.value,
                            VerificationStatus.INSUFFICIENT_EVIDENCE.value,
                        }
                    ),
                },
                "retrieval": {
                    "records_with_hits": sum(1 for record in verified_records if record["retrieval_hits"]),
                    "total_hits": sum(len(record["retrieval_hits"]) for record in verified_records),
                    "retrieval_limit": retrieval_limit,
                },
                "staged_verification": {
                    **staged_execution,
                    "alpha": float(staged_alpha),
                    "threshold": stage_threshold,
                    "policy": None if stage_policy is None else stage_policy.to_dict(),
                },
                "cache_stats": run_cache_stats,
                "alphas": alpha_results,
            })
    except Exception:
        if verified_records_stream is not None:
            verified_records_stream.close()
        if verified_records_tmp_path is not None:
            verified_records_tmp_path.unlink(missing_ok=True)
        raise
    else:
        if verified_records_stream is not None:
            verified_records_stream.close()
            assert verified_records_tmp_path is not None
            assert verified_records_sidecar_path is not None
            verified_records_tmp_path.replace(verified_records_sidecar_path)

    return {
        "schema_version": 1,
        "signal": signal,
        "direction": direction,
        "alphas": [float(alpha) for alpha in alphas],
        "repeats": int(repeats),
        "seed": int(seed),
        "verified_records": {
            "storage": "jsonl_sidecar" if verified_records_sidecar_path is not None else "summary_only",
            "count": verified_record_total,
            "path": None if verified_records_sidecar_path is None else str(verified_records_sidecar_path),
            "run_counts": verified_record_counts,
        },
        "policy": {
            "name": "refute_or_internal_unless_supported",
            "refuted": "trigger",
            "supported": "suppress_internal_trigger",
            "otherwise": "preserve_internal_trigger",
            "tolerance": TOLERANCE,
        },
        "verifier": {
            "type": "GroundednessVerifier",
            "min_overlap": verifier_min_overlap,
        },
        "selfcheck_verifier": {
            "type": "SelfConsistencyVerifier",
            "enabled": any_selfcheck_enabled,
            "min_samples": selfcheck_min_samples,
            "min_overlap": selfcheck_min_overlap,
            "support_threshold": selfcheck_support_threshold,
            "refute_threshold": selfcheck_refute_threshold,
            "early_stop": bool(selfcheck_early_stop),
            "max_samples": selfcheck_max_samples,
        },
        "triple_evidence_verifier": {
            "type": "TripleEvidenceVerifier",
            "enabled": any_triple_evidence_enabled,
            "requested": bool(enable_triple_evidence),
            "min_slot_coverage": float(triple_min_slot_coverage),
        },
        "qa_verifier": {
            "type": "QuestionAnswerVerifier",
            "enabled": qa_verifier is not None,
            "corpus_path": None if qa_corpus_path is None else str(qa_corpus_path),
        },
        "fact_verifier": {
            "type": "StructuredFactVerifier",
            "enabled": any_fact_enabled,
            "corpus_path": None if fact_corpus_path is None else str(fact_corpus_path),
        },
        "retrieval_qa_verifier": {
            "type": "QuestionAnswerVerifier",
            "enabled": any(
                bool(run.get("retrieval_qa", {}).get("enabled"))
                for run in runs
                if isinstance(run.get("retrieval_qa"), Mapping)
            ),
            "source": "retrieval_hits",
        },
        "state_verifier": {
            "type": "StructuredStateVerifier",
            "enabled": any_state_enabled,
            "state_path": None if state_path is None else str(state_path),
            "fixture_has_state": bool(fixture_state),
            "global_checks": len(global_state_checks),
        },
        "transition_verifier": {
            "type": "StateTransitionVerifier",
            "enabled": any_transition_enabled,
            "min_prediction_confidence": float(min_world_model_confidence),
            "state_path": None if state_path is None else str(state_path),
            "fixture_has_state": bool(fixture_state),
            "global_transitions": len(global_state_transitions),
        },
        "retriever": {
            "type": "InMemoryRetriever",
            "min_overlap": retriever_min_overlap,
            "limit": retrieval_limit,
        },
        "verification_trace_cache": {
            "enabled": trace_cache is not None,
            "path": None if trace_cache is None else str(trace_cache.path),
        },
        "score_dump_cache": score_dump_cache_summary(score_dump_metadata_cache),
        "staged_verification": {
            "enabled": stage_policy is not None,
            "alpha": float(staged_alpha),
            "policy": None if stage_policy is None else stage_policy.to_dict(),
            "decision_source": "conformal_internal_gate",
        },
        "runs": runs,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    payload = build_verifier_ensemble_report(
        [_parse_named_path(value) for value in args.scores],
        signal=args.signal,
        claims_path=None if args.claims is None else Path(args.claims),
        qa_corpus_path=None if args.qa_corpus is None else Path(args.qa_corpus),
        fact_corpus_path=(
            None
            if getattr(args, "fact_corpus", None) is None
            else Path(args.fact_corpus)
        ),
        state_path=None if args.state_source is None else Path(args.state_source),
        direction=args.direction,
        alphas=_parse_alphas(args.alphas),
        repeats=args.repeats,
        seed=args.seed,
        verifier_min_overlap=args.verifier_min_overlap,
        retriever_min_overlap=args.retriever_min_overlap,
        retrieval_limit=args.retrieval_limit,
        selfcheck_min_samples=int(getattr(args, "selfcheck_min_samples", 2)),
        selfcheck_min_overlap=float(getattr(args, "selfcheck_min_overlap", 0.65)),
        selfcheck_support_threshold=float(getattr(args, "selfcheck_support_threshold", 0.60)),
        selfcheck_refute_threshold=float(getattr(args, "selfcheck_refute_threshold", 0.50)),
        selfcheck_early_stop=bool(getattr(args, "selfcheck_early_stop", False)),
        selfcheck_max_samples=getattr(args, "selfcheck_max_samples", None),
        enable_triple_evidence=bool(getattr(args, "enable_triple_evidence", False)),
        triple_min_slot_coverage=float(getattr(args, "triple_min_slot_coverage", 1.0)),
        min_world_model_confidence=float(getattr(args, "min_world_model_confidence", 0.0)),
        verification_cache_dir=(
            None
            if getattr(args, "verification_cache_dir", None) is None
            else Path(args.verification_cache_dir)
        ),
        staged_verification=bool(getattr(args, "staged_verification", False)),
        staged_alpha=float(getattr(args, "staged_alpha", 0.10)),
        staged_verify_risk_levels=_parse_csv(
            getattr(args, "staged_verify_risk_levels", "medium,high,unknown")
        ),
        staged_verify_actions=_parse_csv(
            getattr(
                args,
                "staged_verify_actions",
                "retrieve,rewrite,steer_regenerate,execute_tool,abstain,clarify",
            )
        ),
        staged_verify_feature_flags=_parse_csv(
            getattr(args, "staged_verify_feature_flags", "has_number,has_citation,is_time_sensitive")
        ),
        staged_verify_metadata_keys=_parse_csv(
            getattr(args, "staged_verify_metadata_keys", "requires_verification")
        ),
        verified_records_path=(
            None
            if getattr(args, "verified_records_jsonl", None) is None
            else Path(args.verified_records_jsonl)
        ),
    )
    if args.json:
        output_path = Path(args.json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            if args.compact_json:
                json.dump(payload, f, separators=(",", ":"))
            else:
                json.dump(payload, f, indent=2)
        print(f"Wrote verifier ensemble report to {output_path}")
    for run_payload in payload["runs"]:
        alpha_key = str(float(args.best_alpha))
        result = run_payload["alphas"].get(alpha_key)
        if result is None:
            continue
        print(
            f"{run_payload['name']}: alpha={alpha_key} "
            f"internal_det={result['internal']['detection']:.3f} "
            f"internal_fa={result['internal']['false_alarm']:.3f} "
            f"verified_det={result['verified']['detection']:.3f} "
            f"verified_fa={result['verified']['false_alarm']:.3f}"
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval/verifier ensembles over score dumps")
    parser.add_argument("--scores", action="append", required=True,
                        help="score dump path, optionally named as name=path; repeatable")
    parser.add_argument("--claims", default=None,
                        help="optional claim/evidence fixture JSON; otherwise use score dump statements")
    parser.add_argument("--qa-corpus", default=None,
                        help="optional structured question/answer corpus JSON checked before lexical retrieval")
    parser.add_argument(
        "--fact-corpus",
        default=None,
        help="optional structured subject/predicate/object fact corpus checked before lexical retrieval",
    )
    parser.add_argument("--state-source", default=None,
                        help="optional structured state JSON checked by state_check claims before lexical retrieval")
    parser.add_argument("--signal", default="truth_proj")
    parser.add_argument("--direction", choices=("higher", "lower"), default=None)
    parser.add_argument("--alphas", default=",".join(str(alpha) for alpha in ALPHAS))
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--best-alpha", type=float, default=0.10)
    parser.add_argument("--verifier-min-overlap", type=float, default=0.65)
    parser.add_argument("--retriever-min-overlap", type=float, default=0.2)
    parser.add_argument("--retrieval-limit", type=int, default=5)
    parser.add_argument("--selfcheck-min-samples", type=int, default=2,
                        help="minimum sampled responses required before self-consistency verification applies")
    parser.add_argument("--selfcheck-min-overlap", type=float, default=0.65,
                        help="minimum claim/sample token overlap for self-consistency support or refutation")
    parser.add_argument("--selfcheck-support-threshold", type=float, default=0.60,
                        help="support-rate threshold for self-consistency verification")
    parser.add_argument("--selfcheck-refute-threshold", type=float, default=0.50,
                        help="refute-rate threshold for self-consistency verification")
    parser.add_argument("--selfcheck-early-stop", action="store_true",
                        help="stop self-consistency sample judging once the final threshold outcome is fixed")
    parser.add_argument("--selfcheck-max-samples", type=int, default=None,
                        help="optional cap on self-consistency samples considered per claim")
    parser.add_argument("--enable-triple-evidence", action="store_true",
                        help="enable strict subject-predicate-object evidence audits for sensitive factual claims")
    parser.add_argument("--triple-min-slot-coverage", type=float, default=1.0,
                        help="minimum per-slot evidence coverage for triple-evidence audits")
    parser.add_argument("--min-world-model-confidence", type=float, default=0.0,
                        help="minimum world-model prediction confidence required for state-transition postconditions")
    parser.add_argument("--verification-cache-dir", default=None,
                        help="optional directory for file-backed verified-record trace cache")
    parser.add_argument("--verified-records-jsonl", default=None,
                        help="optional JSONL sidecar for per-record verifier outputs")
    parser.add_argument("--staged-verification", action="store_true",
                        help="gate expensive verifier routes with the staged control policy")
    parser.add_argument("--staged-alpha", type=float, default=0.10,
                        help="alpha used to calibrate the internal gate for staged verifier execution")
    parser.add_argument("--staged-verify-risk-levels", default="medium,high,unknown",
                        help="comma-list of diagnostic risk levels that force verifier execution")
    parser.add_argument(
        "--staged-verify-actions",
        default="retrieve,rewrite,steer_regenerate,execute_tool,abstain,clarify",
        help="comma-list of diagnostic actions that force verifier execution",
    )
    parser.add_argument("--staged-verify-feature-flags", default="has_number,has_citation,is_time_sensitive",
                        help="comma-list of claim metadata feature flags that force verifier execution")
    parser.add_argument("--staged-verify-metadata-keys", default="requires_verification",
                        help="comma-list of claim metadata keys that force verifier execution")
    parser.add_argument("--json", default=None, help="optional path to write JSON report")
    parser.add_argument("--compact-json", action="store_true",
                        help="write minified JSON for lower artifact size and write latency")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

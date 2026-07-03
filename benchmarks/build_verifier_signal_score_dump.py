"""Build score-dump verifier signals from verifier verified-record sidecars."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from eigentruth.eval.context_sensitivity import score_context_sensitivity
from eigentruth.eval.score_dump import ScoreDump, load_score_dump, write_score_dump_jsonl

CONTEXT_SENSITIVITY_SIGNALS = (
    "context_sensitivity_flagged_rate",
    "context_sensitivity_max_shift",
    "context_sensitivity_mean_shift",
    "context_sensitivity_max_ratio",
)

DEFAULT_VERIFIER_SIGNALS = (
    "verifier_not_supported",
    "verifier_refuted",
    "verifier_insufficient",
    "verifier_refute_confidence",
    "verifier_uncertainty",
    "verifier_no_retrieval_hit",
    "selfcheck_refute_rate",
    "selfcheck_disagreement",
    "selfcheck_insufficient",
    "fact_selfcheck_support_rate",
    "fact_selfcheck_refute_rate",
    "fact_selfcheck_disagreement",
    "fact_selfcheck_insufficient",
    "fact_selfcheck_not_applicable",
    "fact_selfcheck_uncovered_rate",
    "evidence_alignment_failed",
    "evidence_alignment_insufficient",
    "evidence_alignment_keyword_gap",
    "evidence_alignment_number_gap",
    "evidence_alignment_entity_gap",
    "evidence_alignment_citation_gap",
    "evidence_alignment_issue_rate",
    "perturbation_conflict_rate",
    "perturbation_high_confidence_conflict_rate",
    "perturbation_missing_rate",
    "perturbation_failed",
    "perturbation_not_applicable",
    "world_model_disagreement",
    "world_model_agreement_gap",
    "world_model_low_agreement",
    "world_model_conflict",
    "world_model_conflict_delta",
    "world_model_trace_gap",
    *CONTEXT_SENSITIVITY_SIGNALS,
)


def _parse_csv(value: str | None, *, name: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    parts = tuple(part.strip() for part in value.split(",") if part.strip())
    if not parts:
        raise ValueError(f"{name} must contain at least one value.")
    if len(set(parts)) != len(parts):
        raise ValueError(f"{name} must contain unique values.")
    return parts


def load_verified_record_sidecar(path: str | Path) -> tuple[dict[str, Any], ...]:
    """Load verifier verified-record JSONL sidecar records."""
    records = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, Mapping):
                raise ValueError(f"verified-record line {line_number} must be a JSON object.")
            records.append(dict(payload))
    if not records:
        raise ValueError("verified-record sidecar must contain at least one record.")
    return tuple(records)


def select_verified_records(
    records: Sequence[Mapping[str, Any]],
    *,
    run_name: str | None = None,
) -> tuple[dict[str, Any], ...]:
    """Select one run from a verified-record sidecar."""
    runs = {str(record.get("run", "")) for record in records}
    if run_name is None:
        if len(runs) != 1:
            raise ValueError("verified-record sidecar contains multiple runs; pass --run-name.")
        run_name = next(iter(runs))
    selected = [dict(record) for record in records if str(record.get("run", "")) == str(run_name)]
    if not selected:
        raise ValueError(f"verified-record sidecar has no records for run {run_name!r}.")
    return tuple(sorted(selected, key=lambda record: _record_index(record)))


def build_verifier_signal_score_dump(
    score_dump: ScoreDump,
    verified_records: Sequence[Mapping[str, Any]],
    *,
    source_scores_path: str | Path,
    verified_records_path: str | Path,
    run_name: str | None = None,
    keep_signals: Sequence[str] | None = None,
    verifier_signals: Sequence[str] = DEFAULT_VERIFIER_SIGNALS,
) -> ScoreDump:
    """Return a score dump with verifier-derived score columns appended."""
    selected_records = select_verified_records(verified_records, run_name=run_name)
    if len(selected_records) != score_dump.n_total:
        raise ValueError(
            f"verified records length ({len(selected_records)}) "
            f"does not match score dump labels ({score_dump.n_total})."
        )
    _validate_record_alignment(score_dump, selected_records)

    selected_keep_signals = tuple(score_dump.scores) if keep_signals is None else tuple(keep_signals)
    missing = [signal for signal in selected_keep_signals if signal not in score_dump.scores]
    if missing:
        raise ValueError(f"score dump is missing requested signal(s): {missing}.")
    selected_verifier_signals = tuple(verifier_signals)
    if len(set(selected_verifier_signals)) != len(selected_verifier_signals):
        raise ValueError("verifier_signals must contain unique values.")
    overlap = set(selected_keep_signals) & set(selected_verifier_signals)
    if overlap:
        raise ValueError(f"verifier signal(s) overlap existing score signals: {sorted(overlap)}.")

    signal_columns = {signal: [] for signal in selected_verifier_signals}
    for sidecar_record in selected_records:
        features = verifier_signal_features(sidecar_record)
        for signal in selected_verifier_signals:
            if signal not in features:
                raise ValueError(f"unknown verifier signal {signal!r}.")
            signal_columns[signal].append(_finite_float(features[signal], name=signal))

    resolved_run_name = str(selected_records[0].get("run", run_name or ""))
    config = dict(score_dump.config)
    config["verifier_signal_score_dump"] = {
        "builder": "build_verifier_signal_score_dump",
        "run_name": resolved_run_name,
        "source_scores_path": str(source_scores_path),
        "verified_records_path": str(verified_records_path),
        "signals": list(selected_verifier_signals),
    }
    scores = {
        **{signal: tuple(float(value) for value in score_dump.scores[signal]) for signal in selected_keep_signals},
        **{signal: tuple(values) for signal, values in signal_columns.items()},
    }
    extras = dict(score_dump.extras)
    extras["verifier_signal_metadata"] = {
        "source_scores_path": str(source_scores_path),
        "verified_records_path": str(verified_records_path),
        "run_name": resolved_run_name,
        "signals": list(selected_verifier_signals),
        "signal_definitions": verifier_signal_definitions(),
    }
    return ScoreDump(
        labels=score_dump.labels,
        scores=scores,
        config=config,
        sweep_scores=score_dump.sweep_scores,
        statements=score_dump.statements,
        extras=extras,
    )


def verifier_signal_features(sidecar_record: Mapping[str, Any]) -> dict[str, float]:
    """Return dependency-free verifier uncertainty/disagreement features."""
    record = _mapping(sidecar_record.get("record"))
    final = _mapping(record.get("final"))
    final_status = str(final.get("status", ""))
    final_confidence = _unit_interval(final.get("confidence", 0.0), name="final.confidence")
    retrieval_hits = record.get("retrieval_hits", ())
    hit_count = (
        len(retrieval_hits)
        if isinstance(retrieval_hits, Sequence) and not isinstance(retrieval_hits, str)
        else 0
    )
    selfcheck = record.get("selfcheck")
    selfcheck_payload = _mapping(selfcheck) if isinstance(selfcheck, Mapping) else {}
    selfcheck_status = str(selfcheck_payload.get("status", ""))
    selfcheck_metadata = _mapping(selfcheck_payload.get("metadata"))
    support_rate = _unit_interval(selfcheck_metadata.get("support_rate", 0.0), name="selfcheck.support_rate")
    refute_rate = _unit_interval(selfcheck_metadata.get("refute_rate", 0.0), name="selfcheck.refute_rate")
    selfcheck_executed = bool(selfcheck_payload)
    fact_selfcheck_features = _fact_selfcheck_signal_features(record)
    evidence_alignment_features = _evidence_alignment_signal_features(record)
    perturbation_features = _perturbation_consistency_signal_features(record)
    world_model_features = _world_model_signal_features(record, final)
    context_sensitivity_features = _context_sensitivity_signal_features(sidecar_record, record, final)
    return {
        "verifier_not_supported": 0.0 if final_status == "supported" else 1.0,
        "verifier_refuted": 1.0 if final_status == "refuted" else 0.0,
        "verifier_insufficient": 1.0 if final_status == "insufficient_evidence" else 0.0,
        "verifier_refute_confidence": final_confidence if final_status == "refuted" else 0.0,
        "verifier_uncertainty": _verifier_uncertainty(final_status, final_confidence),
        "verifier_no_retrieval_hit": 1.0 if hit_count == 0 else 0.0,
        "selfcheck_refute_rate": refute_rate if selfcheck_executed else 0.0,
        "selfcheck_disagreement": max(0.0, 1.0 - max(support_rate, refute_rate)) if selfcheck_executed else 0.0,
        "selfcheck_insufficient": 1.0 if selfcheck_status == "insufficient_evidence" else 0.0,
        **fact_selfcheck_features,
        **evidence_alignment_features,
        **perturbation_features,
        **world_model_features,
        **context_sensitivity_features,
    }


def verifier_signal_definitions() -> dict[str, str]:
    """Return human-readable signal definitions."""
    return {
        "verifier_not_supported": "1 when final verifier status is not supported; higher is more risky.",
        "verifier_refuted": "1 when final verifier status is refuted; higher is more risky.",
        "verifier_insufficient": "1 when final verifier status is insufficient_evidence; higher is more uncertain.",
        "verifier_refute_confidence": "final verifier confidence when refuted, else 0; higher is more risky.",
        "verifier_uncertainty": "1 for insufficient/not-applicable outcomes, else 1 - final confidence.",
        "verifier_no_retrieval_hit": "1 when the verifier record has no retrieval hits; higher is less grounded.",
        "selfcheck_refute_rate": "self-consistency refute rate when selfcheck executed, else 0.",
        "selfcheck_disagreement": "1 - max(support_rate, refute_rate) when selfcheck executed, else 0.",
        "selfcheck_insufficient": "1 when selfcheck status is insufficient_evidence, else 0.",
        "fact_selfcheck_support_rate": (
            "minimum fact-level support rate across claim triples when fact selfcheck executed, else 0."
        ),
        "fact_selfcheck_refute_rate": (
            "maximum fact-level refute rate across claim triples when fact selfcheck executed, else 0."
        ),
        "fact_selfcheck_disagreement": (
            "1 - max(fact support/refute rate) when fact selfcheck executed, else 0."
        ),
        "fact_selfcheck_insufficient": "1 when fact selfcheck status is insufficient_evidence, else 0.",
        "fact_selfcheck_not_applicable": "1 when fact selfcheck status is not_applicable, else 0.",
        "fact_selfcheck_uncovered_rate": (
            "fraction of claim triples left insufficient by fact selfcheck, else 0."
        ),
        "evidence_alignment_failed": (
            "1 when claim-to-evidence alignment report fails or reports a misaligned claim, else 0."
        ),
        "evidence_alignment_insufficient": (
            "fraction of evidence-alignment records with insufficient evidence, else 0."
        ),
        "evidence_alignment_keyword_gap": (
            "1 - mean claim/evidence keyword overlap from evidence-alignment reports, else 0."
        ),
        "evidence_alignment_number_gap": (
            "1 - mean numeric slot recall from evidence-alignment reports, else 0."
        ),
        "evidence_alignment_entity_gap": (
            "1 - mean entity-like slot recall from evidence-alignment reports, else 0."
        ),
        "evidence_alignment_citation_gap": (
            "1 - cited-reference coverage rate from evidence-alignment reports, else 0."
        ),
        "evidence_alignment_issue_rate": (
            "alignment issue count divided by aligned record count when reported, else 0."
        ),
        "perturbation_conflict_rate": (
            "fraction of answer-preserving prompt perturbation variants that conflict with the anchor claim."
        ),
        "perturbation_high_confidence_conflict_rate": (
            "fraction of high-confidence answer-preserving perturbation variants that conflict with the anchor claim."
        ),
        "perturbation_missing_rate": (
            "fraction of answer-preserving perturbation variants that do not cover anchor claim triples."
        ),
        "perturbation_failed": "1 when the perturbation-consistency audit status is blocked/refuted, else 0.",
        "perturbation_not_applicable": "1 when perturbation-consistency evidence is not applicable, else 0.",
        "world_model_disagreement": "1 when world-model prediction metadata reports disagreement, else 0.",
        "world_model_agreement_gap": "1 - world-model agreement_rate when reported, else 0.",
        "world_model_low_agreement": "1 when world-model prediction agreement falls below its threshold, else 0.",
        "world_model_conflict": "1 when state-transition metadata reports a world-model postcondition conflict.",
        "world_model_conflict_delta": (
            "absolute numeric expected-vs-actual delta for world-model conflicts, else 0."
        ),
        "world_model_trace_gap": (
            "1 when a state-transition verifier record lacks top-level or prediction_metadata "
            "world_model_reference/world_model_view metadata."
        ),
        "context_sensitivity_flagged_rate": (
            "fraction of evidence-context-sensitive tokens flagged by paired logprob scoring, else 0."
        ),
        "context_sensitivity_max_shift": (
            "maximum positive no-context minus evidence-context token logprob shift, else 0."
        ),
        "context_sensitivity_mean_shift": (
            "mean positive no-context minus evidence-context token logprob shift, else 0."
        ),
        "context_sensitivity_max_ratio": (
            "maximum evidence/no-context logprob ratio where values above 1 mean evidence lowered likelihood, else 0."
        ),
    }


def write_score_dump(dump: ScoreDump, output_path: str | Path, *, output_format: str) -> None:
    """Write output score dump as JSON or JSONL manifest."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        output.write_text(json.dumps(dump.to_mapping(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return
    if output_format == "jsonl":
        write_score_dump_jsonl(dump, output)
        return
    raise ValueError("output_format must be 'json' or 'jsonl'.")


def build_report(
    *,
    input_scores: str | Path,
    verified_records_jsonl: str | Path,
    output: str | Path,
    output_format: str,
    run_name: str | None,
    keep_signals: Sequence[str] | None,
    verifier_signals: Sequence[str],
) -> dict[str, Any]:
    """Build the verifier-signal score dump and return a compact report."""
    score_dump = load_score_dump(input_scores, allow_missing_scores=False)
    sidecar_records = load_verified_record_sidecar(verified_records_jsonl)
    enhanced = build_verifier_signal_score_dump(
        score_dump,
        sidecar_records,
        source_scores_path=input_scores,
        verified_records_path=verified_records_jsonl,
        run_name=run_name,
        keep_signals=keep_signals,
        verifier_signals=verifier_signals,
    )
    write_score_dump(enhanced, output, output_format=output_format)
    verifier_signal_names = tuple(verifier_signals)
    return {
        "schema_version": 1,
        "input_scores": str(input_scores),
        "verified_records_jsonl": str(verified_records_jsonl),
        "output": str(output),
        "output_format": output_format,
        "run_name": enhanced.config.get("verifier_signal_score_dump", {}).get("run_name"),
        "n_total": enhanced.n_total,
        "signals": list(enhanced.scores),
        "verifier_signals": list(verifier_signal_names),
        "summary": {
            signal: {
                "min": min(enhanced.scores[signal]),
                "max": max(enhanced.scores[signal]),
                "mean": sum(enhanced.scores[signal]) / len(enhanced.scores[signal]),
            }
            for signal in verifier_signal_names
        },
    }


def _record_index(record: Mapping[str, Any]) -> int:
    value = record.get("record_index")
    if isinstance(value, bool):
        raise ValueError("record_index must be an integer.")
    return int(value)


def _validate_record_alignment(score_dump: ScoreDump, records: Sequence[Mapping[str, Any]]) -> None:
    seen_indexes = []
    for expected_index, record in enumerate(records):
        record_index = _record_index(record)
        seen_indexes.append(record_index)
        if record_index != expected_index:
            raise ValueError("verified records must have contiguous record_index values starting at 0.")
        label = int(record.get("label"))
        if label != score_dump.labels[expected_index]:
            raise ValueError(f"verified record {record_index} label does not match score dump label.")
    if len(set(seen_indexes)) != len(seen_indexes):
        raise ValueError("verified records contain duplicate record_index values.")


def _verifier_uncertainty(status: str, confidence: float) -> float:
    if status in {"insufficient_evidence", "not_applicable", ""}:
        return 1.0
    return max(0.0, min(1.0, 1.0 - confidence))


def _fact_selfcheck_signal_features(record: Mapping[str, Any]) -> dict[str, float]:
    defaults = {
        "fact_selfcheck_support_rate": 0.0,
        "fact_selfcheck_refute_rate": 0.0,
        "fact_selfcheck_disagreement": 0.0,
        "fact_selfcheck_insufficient": 0.0,
        "fact_selfcheck_not_applicable": 0.0,
        "fact_selfcheck_uncovered_rate": 0.0,
    }
    payload = record.get("fact_selfcheck")
    if not isinstance(payload, Mapping):
        return defaults

    status = str(payload.get("status", ""))
    metadata = _mapping(payload.get("metadata"))
    report = _mapping(metadata.get("fact_selfcheck"))
    triple_reports = report.get("triple_reports", ())
    support_rates = []
    refute_rates = []
    if isinstance(triple_reports, Sequence) and not isinstance(triple_reports, (str, bytes, bytearray)):
        for index, item in enumerate(triple_reports):
            if not isinstance(item, Mapping):
                continue
            support_rates.append(
                _unit_interval(item.get("support_rate", 0.0), name=f"fact_selfcheck[{index}].support_rate")
            )
            refute_rates.append(
                _unit_interval(item.get("refute_rate", 0.0), name=f"fact_selfcheck[{index}].refute_rate")
            )

    support_rate = min(support_rates) if support_rates else 0.0
    refute_rate = max(refute_rates) if refute_rates else 0.0
    triple_count = _optional_non_negative_float(metadata.get("triple_count"), name="fact_selfcheck.triple_count")
    insufficient_count = _optional_non_negative_float(
        metadata.get("insufficient_triple_count"),
        name="fact_selfcheck.insufficient_triple_count",
    )
    if triple_count is None:
        triple_count = float(len(support_rates))
    if insufficient_count is None:
        insufficient_count = 0.0
    uncovered_rate = 0.0 if triple_count <= 0 else max(0.0, min(1.0, insufficient_count / triple_count))
    return {
        "fact_selfcheck_support_rate": support_rate,
        "fact_selfcheck_refute_rate": refute_rate,
        "fact_selfcheck_disagreement": max(0.0, 1.0 - max(support_rate, refute_rate)),
        "fact_selfcheck_insufficient": 1.0 if status == "insufficient_evidence" else 0.0,
        "fact_selfcheck_not_applicable": 1.0 if status == "not_applicable" else 0.0,
        "fact_selfcheck_uncovered_rate": uncovered_rate,
    }


def _evidence_alignment_signal_features(record: Mapping[str, Any]) -> dict[str, float]:
    defaults = {
        "evidence_alignment_failed": 0.0,
        "evidence_alignment_insufficient": 0.0,
        "evidence_alignment_keyword_gap": 0.0,
        "evidence_alignment_number_gap": 0.0,
        "evidence_alignment_entity_gap": 0.0,
        "evidence_alignment_citation_gap": 0.0,
        "evidence_alignment_issue_rate": 0.0,
    }
    payload = _evidence_alignment_payload(record)
    if not payload:
        return defaults

    status = str(payload.get("status", ""))
    metadata = _mapping(payload.get("metadata"))
    report = _mapping(
        payload.get("evidence_alignment")
        or metadata.get("evidence_alignment")
        or payload
    )
    summary = _mapping(report.get("summary"))
    if not summary:
        summary = _mapping(payload.get("summary"))
    if not summary:
        return defaults

    record_count = _optional_non_negative_float(summary.get("record_count"), name="evidence_alignment.record_count")
    if record_count is None:
        record_count = 0.0
    misalignment_rate = _unit_interval(
        summary.get("misalignment_rate", 0.0),
        name="evidence_alignment.misalignment_rate",
    )
    insufficient_rate = _unit_interval(
        summary.get("insufficient_evidence_rate", 0.0),
        name="evidence_alignment.insufficient_evidence_rate",
    )
    issue_count = _optional_non_negative_float(summary.get("issue_count"), name="evidence_alignment.issue_count")
    issue_rate = 0.0 if not record_count or issue_count is None else max(0.0, min(1.0, issue_count / record_count))
    passed = summary.get("passed")
    failed = (
        passed is False
        or status == "refuted"
        or misalignment_rate > 0.0
    )
    return {
        "evidence_alignment_failed": 1.0 if failed else 0.0,
        "evidence_alignment_insufficient": insufficient_rate,
        "evidence_alignment_keyword_gap": _gap_from_optional_unit_interval(
            summary.get("keyword_overlap_mean"),
            name="evidence_alignment.keyword_overlap_mean",
        ),
        "evidence_alignment_number_gap": _gap_from_optional_unit_interval(
            summary.get("number_recall_mean"),
            name="evidence_alignment.number_recall_mean",
        ),
        "evidence_alignment_entity_gap": _gap_from_optional_unit_interval(
            summary.get("entity_recall_mean"),
            name="evidence_alignment.entity_recall_mean",
        ),
        "evidence_alignment_citation_gap": _gap_from_optional_unit_interval(
            summary.get("citation_reference_coverage_rate"),
            name="evidence_alignment.citation_reference_coverage_rate",
        ),
        "evidence_alignment_issue_rate": issue_rate,
    }


def _evidence_alignment_payload(record: Mapping[str, Any]) -> Mapping[str, Any]:
    final = _mapping(record.get("final"))
    final_metadata = _mapping(final.get("metadata"))
    for candidate in (
        record.get("evidence_alignment"),
        record.get("citation_evidence_alignment"),
        record.get("claim_evidence_alignment"),
        final.get("evidence_alignment"),
        final_metadata.get("evidence_alignment"),
    ):
        if isinstance(candidate, Mapping):
            return candidate
    return {}


def _perturbation_consistency_signal_features(record: Mapping[str, Any]) -> dict[str, float]:
    defaults = {
        "perturbation_conflict_rate": 0.0,
        "perturbation_high_confidence_conflict_rate": 0.0,
        "perturbation_missing_rate": 0.0,
        "perturbation_failed": 0.0,
        "perturbation_not_applicable": 0.0,
    }
    payload = _perturbation_consistency_payload(record)
    if not payload:
        return defaults

    status = str(payload.get("status", ""))
    metadata = _mapping(payload.get("metadata"))
    report = _mapping(
        payload.get("perturbation_consistency")
        or metadata.get("perturbation_consistency")
        or payload
    )
    summary = _mapping(report.get("summary"))
    if not summary:
        summary = _mapping(payload.get("summary"))
    report_status = str(summary.get("status") or report.get("status") or status)
    return {
        "perturbation_conflict_rate": _unit_interval(
            summary.get("conflict_rate", 0.0),
            name="perturbation_consistency.conflict_rate",
        ),
        "perturbation_high_confidence_conflict_rate": _unit_interval(
            summary.get("high_confidence_conflict_rate", 0.0),
            name="perturbation_consistency.high_confidence_conflict_rate",
        ),
        "perturbation_missing_rate": _unit_interval(
            summary.get("missing_rate", 0.0),
            name="perturbation_consistency.missing_rate",
        ),
        "perturbation_failed": 1.0 if report_status in {"blocked", "refuted"} or status == "refuted" else 0.0,
        "perturbation_not_applicable": (
            1.0 if report_status == "not_applicable" or status == "not_applicable" else 0.0
        ),
    }


def _perturbation_consistency_payload(record: Mapping[str, Any]) -> Mapping[str, Any]:
    for candidate in (
        record.get("perturbation_consistency"),
        record.get("prompt_perturbation_consistency"),
        _mapping(record.get("final")).get("perturbation_consistency"),
        _mapping(_mapping(record.get("final")).get("metadata")).get("perturbation_consistency"),
    ):
        if isinstance(candidate, Mapping):
            return candidate
    return {}


def _world_model_signal_features(
    record: Mapping[str, Any],
    final: Mapping[str, Any],
) -> dict[str, float]:
    features = _empty_world_model_features()
    transition = _mapping(record.get("transition"))
    transition_present = bool(transition)
    metadata_candidates = [
        (_mapping(transition.get("metadata")), transition_present),
        (_mapping(final.get("metadata")), False),
    ]
    for metadata, is_transition_metadata in metadata_candidates:
        if not metadata:
            continue
        direct_metadata = _is_direct_world_model_metadata(metadata)
        trace_metadata = _has_world_model_trace_metadata(metadata)
        features.update(
            _world_model_trace_features(
                metadata,
                transition_present=is_transition_metadata,
            )
        )
        prediction_metadata = _mapping(metadata.get("prediction_metadata"))
        if prediction_metadata and (
            is_transition_metadata
            or direct_metadata
            or trace_metadata
            or _is_world_model_prediction_metadata(prediction_metadata)
        ):
            features.update(
                _world_model_metadata_features(
                    prediction_metadata,
                    name_prefix="world_model.prediction",
                )
            )
            return features
        if direct_metadata:
            features.update(
                _world_model_metadata_features(
                    metadata,
                    name_prefix="world_model",
                )
            )
            return features
    return features


def _empty_world_model_features() -> dict[str, float]:
    return {
        "world_model_disagreement": 0.0,
        "world_model_agreement_gap": 0.0,
        "world_model_low_agreement": 0.0,
        "world_model_conflict": 0.0,
        "world_model_conflict_delta": 0.0,
        "world_model_trace_gap": 0.0,
    }


def _context_sensitivity_signal_features(
    sidecar_record: Mapping[str, Any],
    record: Mapping[str, Any],
    final: Mapping[str, Any],
) -> dict[str, float]:
    for payload in _context_sensitivity_payload_candidates(sidecar_record, record, final):
        summary = _context_sensitivity_summary(payload)
        if summary:
            return _context_sensitivity_features_from_summary(summary)
    return _empty_context_sensitivity_features()


def _empty_context_sensitivity_features() -> dict[str, float]:
    return {
        "context_sensitivity_flagged_rate": 0.0,
        "context_sensitivity_max_shift": 0.0,
        "context_sensitivity_mean_shift": 0.0,
        "context_sensitivity_max_ratio": 0.0,
    }


def _context_sensitivity_features_from_summary(summary: Mapping[str, Any]) -> dict[str, float]:
    return {
        "context_sensitivity_flagged_rate": _unit_interval(
            summary.get("flagged_rate", 0.0),
            name="context_sensitivity.summary.flagged_rate",
        ),
        "context_sensitivity_max_shift": max(
            0.0,
            _finite_float(
                summary.get("max_unsupported_context_shift", 0.0),
                name="context_sensitivity.summary.max_unsupported_context_shift",
            ),
        ),
        "context_sensitivity_mean_shift": max(
            0.0,
            _finite_float(
                summary.get("mean_unsupported_context_shift", 0.0),
                name="context_sensitivity.summary.mean_unsupported_context_shift",
            ),
        ),
        "context_sensitivity_max_ratio": max(
            0.0,
            _finite_float(
                summary.get("max_context_sensitivity_ratio", 0.0),
                name="context_sensitivity.summary.max_context_sensitivity_ratio",
            ),
        ),
    }


def _context_sensitivity_payload_candidates(
    sidecar_record: Mapping[str, Any],
    record: Mapping[str, Any],
    final: Mapping[str, Any],
) -> tuple[Any, ...]:
    transition = _mapping(record.get("transition"))
    candidates = [
        sidecar_record.get("context_sensitivity"),
        sidecar_record.get("context_sensitivity_report"),
        record.get("context_sensitivity"),
        record.get("context_sensitivity_report"),
        _mapping(final.get("metadata")).get("context_sensitivity"),
        _mapping(final.get("metadata")).get("context_sensitivity_report"),
        _mapping(transition.get("metadata")).get("context_sensitivity"),
        _mapping(transition.get("metadata")).get("context_sensitivity_report"),
    ]
    return tuple(candidate for candidate in candidates if candidate is not None)


def _context_sensitivity_summary(payload: Any) -> Mapping[str, Any]:
    if isinstance(payload, Mapping):
        summary = _mapping(payload.get("summary"))
        if summary:
            return summary
        tokens = _first_sequence(
            payload.get("tokens"),
            payload.get("context_sensitivity_tokens"),
            payload.get("paired_token_logprobs"),
        )
        if tokens is not None:
            return score_context_sensitivity(tokens).summary
        token_scores = _sequence_or_none(payload.get("token_scores"))
        if token_scores is not None:
            return _context_sensitivity_summary_from_token_scores(token_scores)
        return {}
    tokens = _sequence_or_none(payload)
    if tokens is None:
        return {}
    return score_context_sensitivity(tokens).summary


def _context_sensitivity_summary_from_token_scores(token_scores: Sequence[Any]) -> dict[str, float]:
    count = 0
    flagged_count = 0
    shifts: list[float] = []
    ratios: list[float] = []
    for raw in token_scores:
        if not isinstance(raw, Mapping):
            raise ValueError("context_sensitivity.token_scores entries must be JSON objects.")
        count += 1
        flagged_count += 1 if bool(raw.get("flagged", False)) else 0
        shifts.append(
            max(
                0.0,
                _finite_float(
                    raw.get("unsupported_context_shift", 0.0),
                    name="context_sensitivity.token_scores.unsupported_context_shift",
                ),
            )
        )
        ratios.append(
            max(
                0.0,
                _finite_float(
                    raw.get("context_sensitivity_ratio", 0.0),
                    name="context_sensitivity.token_scores.context_sensitivity_ratio",
                ),
            )
        )
    return {
        "flagged_rate": (flagged_count / count) if count else 0.0,
        "max_unsupported_context_shift": max(shifts) if shifts else 0.0,
        "mean_unsupported_context_shift": (sum(shifts) / len(shifts)) if shifts else 0.0,
        "max_context_sensitivity_ratio": max(ratios) if ratios else 0.0,
    }


def _first_sequence(*values: Any) -> Sequence[Any] | None:
    for value in values:
        sequence = _sequence_or_none(value)
        if sequence is not None:
            return sequence
    return None


def _sequence_or_none(value: Any) -> Sequence[Any] | None:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return None


def _world_model_trace_features(
    metadata: Mapping[str, Any],
    *,
    transition_present: bool,
) -> dict[str, float]:
    conflict = _world_model_trace_mapping(metadata, "world_model_conflict")
    reference = _world_model_trace_mapping(metadata, "world_model_reference")
    view = _world_model_trace_mapping(metadata, "world_model_view")
    return {
        "world_model_conflict": 1.0 if conflict else 0.0,
        "world_model_conflict_delta": _world_model_conflict_delta(conflict),
        "world_model_trace_gap": 1.0 if transition_present and (not reference or not view) else 0.0,
    }


def _world_model_trace_mapping(metadata: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = _mapping(metadata.get(key))
    if value:
        return value
    prediction_metadata = _mapping(metadata.get("prediction_metadata"))
    return _mapping(prediction_metadata.get(key))


def _has_world_model_trace_metadata(metadata: Mapping[str, Any]) -> bool:
    return any(
        _world_model_trace_mapping(metadata, key)
        for key in (
            "world_model_reference",
            "world_model_view",
            "world_model_conflict",
        )
    ) or metadata.get("world_model") is not None


def _is_world_model_prediction_metadata(metadata: Mapping[str, Any]) -> bool:
    return _has_world_model_trace_metadata(metadata) or _is_direct_world_model_metadata(metadata)


def _world_model_conflict_delta(conflict: Mapping[str, Any]) -> float:
    if not conflict:
        return 0.0
    expected = _optional_finite_float(conflict.get("expected"))
    actual = _optional_finite_float(conflict.get("actual"))
    if expected is None or actual is None:
        return 0.0
    return abs(actual - expected)


def _is_direct_world_model_metadata(metadata: Mapping[str, Any]) -> bool:
    decision_rule = str(metadata.get("decision_rule", ""))
    has_ensemble_agreement_shape = (
        "member_world_models" in metadata
        or (
            any(
                key in metadata
                for key in (
                    "agreement_rate",
                    "agreement_count",
                    "below_min_agreement",
                )
            )
            and any(key in metadata for key in ("member_count", "min_agreement"))
        )
    )
    return (
        str(metadata.get("verifier", "")) == "world_model_ensemble"
        or str(metadata.get("world_model", "")) == "EnsembleWorldModelAdapter"
        or decision_rule
        in {
            "status_consensus",
            "status_tie",
            "status_agreement_below_threshold",
            "prediction_consensus",
            "prediction_agreement_below_threshold",
            "all_members_failed",
        }
        or has_ensemble_agreement_shape
    )


def _world_model_metadata_features(
    metadata: Mapping[str, Any],
    *,
    name_prefix: str,
) -> dict[str, float]:
    decision_rule = str(metadata.get("decision_rule", ""))
    agreement_rate = _world_model_agreement_rate(metadata, name_prefix=name_prefix)
    below_min_agreement = (
        metadata.get("below_min_agreement") is True
        or decision_rule
        in {
            "status_tie",
            "status_agreement_below_threshold",
            "prediction_agreement_below_threshold",
            "all_members_failed",
        }
    )
    disagreement = _world_model_disagreement(metadata, below_min_agreement=below_min_agreement)
    return {
        "world_model_disagreement": 1.0 if disagreement else 0.0,
        "world_model_agreement_gap": max(0.0, 1.0 - agreement_rate),
        "world_model_low_agreement": 1.0 if below_min_agreement else 0.0,
    }


def _world_model_agreement_rate(
    metadata: Mapping[str, Any],
    *,
    name_prefix: str,
) -> float:
    if "agreement_rate" in metadata:
        return _unit_interval(metadata.get("agreement_rate", 1.0), name=f"{name_prefix}.agreement_rate")
    agreement_count = metadata.get("agreement_count")
    member_count = metadata.get("member_count")
    if agreement_count is not None and member_count is not None:
        member_count_float = _finite_float(member_count, name=f"{name_prefix}.member_count")
        if member_count_float <= 0:
            raise ValueError(f"{name_prefix}.member_count must be positive.")
        agreement_count_float = _finite_float(
            agreement_count,
            name=f"{name_prefix}.agreement_count",
        )
        if agreement_count_float < 0:
            raise ValueError(f"{name_prefix}.agreement_count must be non-negative.")
        return max(0.0, min(1.0, agreement_count_float / member_count_float))
    if str(metadata.get("decision_rule", "")) in {
        "status_tie",
        "status_agreement_below_threshold",
        "prediction_agreement_below_threshold",
        "all_members_failed",
    }:
        return 0.0
    return 1.0


def _world_model_disagreement(
    metadata: Mapping[str, Any],
    *,
    below_min_agreement: bool,
) -> bool:
    if "disagreement" in metadata:
        return metadata.get("disagreement") is True
    agreement_count = metadata.get("agreement_count")
    member_count = metadata.get("member_count")
    if agreement_count is not None and member_count is not None:
        return float(agreement_count) < float(member_count)
    if str(metadata.get("decision_rule", "")) == "status_tie":
        return True
    return bool(below_min_agreement)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        return float(value)
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite.")
    return numeric


def _optional_finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _optional_non_negative_float(value: Any, *, name: str) -> float | None:
    numeric = _optional_finite_float(value)
    if numeric is None:
        return None
    if numeric < 0:
        raise ValueError(f"{name} must be non-negative.")
    return numeric


def _unit_interval(value: Any, *, name: str) -> float:
    numeric = _finite_float(value, name=name)
    if not (0.0 <= numeric <= 1.0):
        raise ValueError(f"{name} must be in [0, 1].")
    return numeric


def _gap_from_optional_unit_interval(value: Any, *, name: str) -> float:
    if value is None:
        return 0.0
    return max(0.0, 1.0 - _unit_interval(value, name=name))


def run(args: argparse.Namespace) -> dict[str, Any]:
    keep_signals = _parse_csv(args.keep_signals, name="keep_signals")
    verifier_signals = _parse_csv(args.verifier_signals, name="verifier_signals") or DEFAULT_VERIFIER_SIGNALS
    report = build_report(
        input_scores=args.scores,
        verified_records_jsonl=args.verified_records_jsonl,
        output=args.output,
        output_format=args.output_format,
        run_name=args.run_name,
        keep_signals=keep_signals,
        verifier_signals=verifier_signals,
    )
    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote verifier-signal score dump to {args.output}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Append verifier-derived signals to a score dump")
    parser.add_argument("--scores", required=True, help="input score dump JSON or JSONL manifest")
    parser.add_argument("--verified-records-jsonl", required=True, help="eval_verifier_ensemble verified-record JSONL")
    parser.add_argument("--run-name", default=None, help="run name to select when sidecar contains multiple runs")
    parser.add_argument("--keep-signals", default=None, help="optional comma-list of original score signals to keep")
    parser.add_argument(
        "--verifier-signals",
        default=",".join(DEFAULT_VERIFIER_SIGNALS),
        help="comma-list of verifier-derived signals to append",
    )
    parser.add_argument("--output", required=True, help="output score dump path")
    parser.add_argument("--output-format", choices=("json", "jsonl"), default="jsonl")
    parser.add_argument("--json", default=None, help="optional summary report path")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

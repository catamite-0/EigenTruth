"""Build score-dump verifier signals from verifier verified-record sidecars."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from eigentruth.eval.score_dump import ScoreDump, load_score_dump, write_score_dump_jsonl

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
    "world_model_disagreement",
    "world_model_agreement_gap",
    "world_model_low_agreement",
    "world_model_conflict",
    "world_model_conflict_delta",
    "world_model_trace_gap",
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
    world_model_features = _world_model_signal_features(record, final)
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
        **world_model_features,
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
        "world_model_disagreement": "1 when world-model prediction metadata reports disagreement, else 0.",
        "world_model_agreement_gap": "1 - world-model agreement_rate when reported, else 0.",
        "world_model_low_agreement": "1 when world-model prediction agreement falls below its threshold, else 0.",
        "world_model_conflict": "1 when state-transition metadata reports a world-model postcondition conflict.",
        "world_model_conflict_delta": (
            "absolute numeric expected-vs-actual delta for world-model conflicts, else 0."
        ),
        "world_model_trace_gap": (
            "1 when a state-transition verifier record lacks world_model_reference or world_model_view metadata."
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


def _world_model_signal_features(
    record: Mapping[str, Any],
    final: Mapping[str, Any],
) -> dict[str, float]:
    features = _empty_world_model_features()
    transition = _mapping(record.get("transition"))
    transition_present = bool(transition)
    metadata_candidates = [
        _mapping(transition.get("metadata")),
        _mapping(final.get("metadata")),
    ]
    for metadata in metadata_candidates:
        if not metadata:
            continue
        features.update(
            _world_model_trace_features(
                metadata,
                transition_present=transition_present,
            )
        )
        prediction_metadata = _mapping(metadata.get("prediction_metadata"))
        if prediction_metadata:
            features.update(
                _world_model_metadata_features(
                    prediction_metadata,
                    name_prefix="world_model.prediction",
                )
            )
            return features
        if _is_direct_world_model_metadata(metadata):
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


def _world_model_trace_features(
    metadata: Mapping[str, Any],
    *,
    transition_present: bool,
) -> dict[str, float]:
    conflict = _mapping(metadata.get("world_model_conflict"))
    reference = _mapping(metadata.get("world_model_reference"))
    view = _mapping(metadata.get("world_model_view"))
    return {
        "world_model_conflict": 1.0 if conflict else 0.0,
        "world_model_conflict_delta": _world_model_conflict_delta(conflict),
        "world_model_trace_gap": 1.0 if transition_present and (not reference or not view) else 0.0,
    }


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


def _unit_interval(value: Any, *, name: str) -> float:
    numeric = _finite_float(value, name=name)
    if not (0.0 <= numeric <= 1.0):
        raise ValueError(f"{name} must be in [0, 1].")
    return numeric


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

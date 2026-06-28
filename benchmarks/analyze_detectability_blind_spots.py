"""Analyze row-level blind spots from DECK-style detectability reports.

This helper consumes an existing detectability taxonomy report plus its source
score dump. It does not load a model. The default path exports false records in
the ``entrenched`` cell, i.e. repeatable high-confidence false answers that
uncertainty-only routes are expected to miss.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eigentruth.eval.score_dump import (  # noqa: E402
    load_score_dump_statement_scores,
    score_dump_cache_summary,
    score_dump_file_metadata,
)
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402
from eigentruth.verify.claims import claim_features  # noqa: E402

DECK_CELLS = ("drift", "entrenched", "confabulation", "knotted")
_QUESTION_TYPE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("temporal", re.compile(r"\b(?:when|what time|what year|what date|now|current|today|latest)\b", re.I)),
    ("person", re.compile(r"\bwho\b", re.I)),
    ("location", re.compile(r"\bwhere\b", re.I)),
    ("quantity", re.compile(r"\b(?:how many|how much|percentage|percent|number|amount|count)\b", re.I)),
    ("causal", re.compile(r"\bwhy\b", re.I)),
    ("method", re.compile(r"\bhow\b", re.I)),
    ("choice", re.compile(r"\bwhich\b", re.I)),
    ("definition", re.compile(r"\bwhat\b", re.I)),
)


def build_detectability_blind_spot_analysis(
    *,
    taxonomy_report_path: str | Path,
    score_dump_path: str | Path | None = None,
    cell: str = "entrenched",
    false_only: bool = True,
    max_records: int = 100,
    metadata: Mapping[str, Any] | None = None,
    cache: MutableMapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return row-level analysis for one taxonomy cell."""
    if cell not in DECK_CELLS:
        raise ValueError(f"cell must be one of: {', '.join(DECK_CELLS)}.")
    if int(max_records) < 0:
        raise ValueError("max_records must be >= 0.")
    taxonomy_path = Path(taxonomy_report_path)
    taxonomy = _load_json_object(taxonomy_path)
    source_score_dump = score_dump_path or _nested(taxonomy, "source", "score_dump_path")
    if source_score_dump is None:
        raise ValueError("score_dump_path is required when taxonomy report has no source.score_dump_path.")
    score_path = Path(source_score_dump)
    consistency_signal = str(_nested(taxonomy, "config", "consistency_signal"))
    confidence_signal = str(_nested(taxonomy, "config", "confidence_signal"))
    if not consistency_signal or not confidence_signal:
        raise ValueError("taxonomy report must include consistency_signal and confidence_signal config.")
    score_names = _score_names_for_analysis(score_path, consistency_signal, confidence_signal)
    dump = load_score_dump_statement_scores(
        score_path,
        score_names,
        require_statements=True,
        cache=cache,
    )
    axis = _axis_config(taxonomy)
    records = []
    cell_counts: Counter[str] = Counter()
    cell_false_counts: Counter[str] = Counter()
    for index, label in enumerate(dump.labels):
        consistency_score = float(dump.scores[consistency_signal][index])
        confidence_score = float(dump.scores[confidence_signal][index])
        assignment = _assign_cell(
            consistency_score,
            confidence_score,
            consistency_axis=axis["consistency"],
            confidence_axis=axis["confidence"],
        )
        cell_counts[assignment] += 1
        if int(label) == 1:
            cell_false_counts[assignment] += 1
        if assignment != cell or (false_only and int(label) != 1):
            continue
        statement = dict(dump.statements[index])
        score_payload = {name: float(values[index]) for name, values in dump.scores.items()}
        consistency_margin = _axis_margin(
            consistency_score,
            axis=axis["consistency"],
            low_expected=(cell in {"drift", "confabulation"}),
        )
        confidence_margin = _axis_margin(
            confidence_score,
            axis=axis["confidence"],
            low_expected=(cell in {"knotted", "confabulation"}),
        )
        text = str(statement.get("text") or _statement_text(statement))
        answer = str(statement.get("answer", ""))
        question = str(statement.get("question", ""))
        records.append({
            "record_index": index,
            "label": int(label),
            "cell": assignment,
            "cell_margin": min(consistency_margin, confidence_margin),
            "consistency_margin": consistency_margin,
            "confidence_margin": confidence_margin,
            "question_type": _question_type(question),
            "features": claim_features(text),
            "answer_features": claim_features(answer),
            "question": question,
            "answer": answer,
            "text": text,
            "answer_token_count": _token_count(answer),
            "question_token_count": _token_count(question),
            "scores": score_payload,
        })
    records.sort(key=lambda item: (-float(item["cell_margin"]), int(item["record_index"])))
    selected_records = records[: int(max_records)] if int(max_records) else []
    expected_selected = _expected_cell_count(taxonomy, cell=cell, false_only=false_only)
    status = "complete" if expected_selected is None or expected_selected == len(records) else "blocked"
    summary = _summary(
        records,
        selected_records=selected_records,
        expected_selected=expected_selected,
        cell=cell,
        false_only=false_only,
        cell_counts=cell_counts,
        cell_false_counts=cell_false_counts,
    )
    return {
        "schema_version": 1,
        "workflow": "detectability_blind_spot_analysis",
        "status": status,
        "source": {
            "taxonomy_report_path": str(taxonomy_path),
            "score_dump_path": str(score_path),
            "score_dump_file": score_dump_file_metadata(score_path),
            "score_dump_summary": dict(dump.summary),
            "score_dump_source_format": dump.source_format,
        },
        "config": {
            "cell": cell,
            "false_only": bool(false_only),
            "max_records": int(max_records),
            "consistency_signal": consistency_signal,
            "confidence_signal": confidence_signal,
            "score_names": tuple(score_names),
        },
        "thresholds": axis,
        "summary": summary,
        "records": selected_records,
        "score_dump_cache": score_dump_cache_summary(cache),
        "metadata": dict(metadata or {}),
    }


def run(
    *,
    taxonomy_report_path: str | Path,
    output_path: str | Path,
    score_dump_path: str | Path | None = None,
    cell: str = "entrenched",
    false_only: bool = True,
    max_records: int = 100,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    compact_json: bool = False,
) -> dict[str, Any]:
    """Build and write a blind-spot analysis report."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    cache: dict[str, Any] = {}
    report = build_detectability_blind_spot_analysis(
        taxonomy_report_path=taxonomy_report_path,
        score_dump_path=score_dump_path,
        cell=cell,
        false_only=false_only,
        max_records=max_records,
        metadata=metadata,
        cache=cache,
    )
    output = Path(output_path)
    if artifact_manifest_path is not None:
        report["paths"] = {"artifact_manifest": str(artifact_manifest_path)}
    _write_json(output, report, compact=compact_json)
    if artifact_manifest_path is not None:
        manifest_path = Path(artifact_manifest_path)
        manifest = _build_manifest(
            report,
            output_path=output,
            taxonomy_report_path=Path(taxonomy_report_path),
            score_dump_path=Path(report["source"]["score_dump_path"]),
            artifact_manifest_path=manifest_path,
        )
        _write_json(manifest_path, manifest, compact=compact_json)
    if registry_path is not None:
        assert name is not None and version is not None
        ArtifactRegistry.load_json(registry_path).record_report(
            name=name,
            version=version,
            path=output,
            metadata={
                "workflow": "detectability_blind_spot_analysis",
                "status": report.get("status"),
                "cell": cell,
                "false_only": bool(false_only),
                "selected_record_count": _nested(report, "summary", "selected_record_count"),
                "expected_selected_record_count": _nested(report, "summary", "expected_selected_record_count"),
                "question_type_counts": _nested(report, "summary", "question_type_counts"),
                "artifact_manifest": None if artifact_manifest_path is None else str(artifact_manifest_path),
                **dict(metadata or {}),
            },
        ).save_json()
    return report


def _score_names_for_analysis(
    score_dump_path: Path,
    consistency_signal: str,
    confidence_signal: str,
) -> tuple[str, ...]:
    metadata = score_dump_file_metadata(score_dump_path)
    primary = tuple(
        str(name)
        for name in _nested(metadata, "identity", "primary_score_names", default=())
    )
    names = tuple(dict.fromkeys((consistency_signal, confidence_signal, *primary)))
    if not names:
        raise ValueError("score dump has no primary score names.")
    return names


def _axis_config(taxonomy: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    axes = _mapping(_nested(taxonomy, "report", "axes"))
    return {
        "consistency": _single_axis_config(axes, "consistency"),
        "confidence": _single_axis_config(axes, "confidence"),
    }


def _single_axis_config(axes: Mapping[str, Any], name: str) -> dict[str, Any]:
    axis = _mapping(axes.get(name))
    direction = str(axis.get("direction", "higher"))
    if direction not in {"higher", "lower"}:
        raise ValueError(f"{name} axis direction must be higher or lower.")
    threshold = float(axis["threshold"])
    normalized_threshold = float(axis.get("normalized_threshold", threshold if direction == "higher" else -threshold))
    return {
        "direction": direction,
        "threshold": threshold,
        "normalized_threshold": normalized_threshold,
    }


def _assign_cell(
    consistency_score: float,
    confidence_score: float,
    *,
    consistency_axis: Mapping[str, Any],
    confidence_axis: Mapping[str, Any],
) -> str:
    consistency_health = _health_score(consistency_score, consistency_axis)
    confidence_health = _health_score(confidence_score, confidence_axis)
    low_consistency = consistency_health < float(consistency_axis["normalized_threshold"])
    low_confidence = confidence_health < float(confidence_axis["normalized_threshold"])
    if low_consistency and low_confidence:
        return "confabulation"
    if low_consistency:
        return "drift"
    if low_confidence:
        return "knotted"
    return "entrenched"


def _axis_margin(score: float, *, axis: Mapping[str, Any], low_expected: bool) -> float:
    health = _health_score(score, axis)
    threshold = float(axis["normalized_threshold"])
    return (threshold - health) if low_expected else (health - threshold)


def _health_score(score: float, axis: Mapping[str, Any]) -> float:
    value = float(score)
    return value if axis["direction"] == "higher" else -value


def _summary(
    records: Sequence[Mapping[str, Any]],
    *,
    selected_records: Sequence[Mapping[str, Any]],
    expected_selected: int | None,
    cell: str,
    false_only: bool,
    cell_counts: Counter[str],
    cell_false_counts: Counter[str],
) -> dict[str, Any]:
    feature_counts = Counter()
    answer_feature_counts = Counter()
    question_types = Counter()
    for record in records:
        question_types[str(record["question_type"])] += 1
        for name, enabled in _mapping(record.get("features")).items():
            if enabled:
                feature_counts[str(name)] += 1
        for name, enabled in _mapping(record.get("answer_features")).items():
            if enabled:
                answer_feature_counts[str(name)] += 1
    return {
        "cell": cell,
        "false_only": bool(false_only),
        "selected_record_count": len(records),
        "emitted_record_count": len(selected_records),
        "truncated": len(selected_records) < len(records),
        "expected_selected_record_count": expected_selected,
        "assignment_check_passed": expected_selected is None or expected_selected == len(records),
        "cell_counts": {name: int(cell_counts.get(name, 0)) for name in DECK_CELLS},
        "cell_false_counts": {name: int(cell_false_counts.get(name, 0)) for name in DECK_CELLS},
        "question_type_counts": dict(sorted(question_types.items())),
        "feature_counts": dict(sorted(feature_counts.items())),
        "answer_feature_counts": dict(sorted(answer_feature_counts.items())),
        "answer_token_count": _number_summary(record["answer_token_count"] for record in records),
        "question_token_count": _number_summary(record["question_token_count"] for record in records),
        "cell_margin": _number_summary(float(record["cell_margin"]) for record in records),
    }


def _expected_cell_count(taxonomy: Mapping[str, Any], *, cell: str, false_only: bool) -> int | None:
    key = "n_false" if false_only else "n_total"
    value = _nested(taxonomy, "report", "cells", cell, key)
    if value is None:
        return None
    return int(value)


def _number_summary(values: Sequence[float] | Any) -> dict[str, float | int | None]:
    vals = [float(value) for value in values]
    if not vals:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(vals),
        "min": min(vals),
        "max": max(vals),
        "mean": sum(vals) / len(vals),
    }


def _question_type(question: str) -> str:
    question_text = str(question)
    for name, pattern in _QUESTION_TYPE_PATTERNS:
        if pattern.search(question_text):
            return name
    return "other"


def _statement_text(statement: Mapping[str, Any]) -> str:
    question = str(statement.get("question", "")).strip()
    answer = str(statement.get("answer", "")).strip()
    return " ".join(part for part in (question, answer) if part)


def _token_count(text: str) -> int:
    return len(str(text).split())


def _build_manifest(
    report: Mapping[str, Any],
    *,
    output_path: Path,
    taxonomy_report_path: Path,
    score_dump_path: Path,
    artifact_manifest_path: Path,
) -> dict[str, Any]:
    return build_artifact_manifest(
        {
            "blind_spot_analysis_report": output_path,
            "detectability_taxonomy_report": taxonomy_report_path,
            "score_dump": score_dump_path,
        },
        root=artifact_manifest_path.parent,
        metadata={
            "runner": "analyze_detectability_blind_spots",
            "status": report.get("status"),
            "cell": _nested(report, "summary", "cell"),
            "selected_record_count": _nested(report, "summary", "selected_record_count"),
            "expected_selected_record_count": _nested(report, "summary", "expected_selected_record_count"),
            "assignment_check_passed": _nested(report, "summary", "assignment_check_passed"),
        },
    )


def _parse_metadata(values: Sequence[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"metadata must be KEY=VALUE, got {value!r}.")
        key, item = value.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("metadata key cannot be empty.")
        metadata[key] = item
    return metadata


def _load_json_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON payload must be an object: {path}")
    return dict(payload)


def _write_json(path: str | Path, payload: Mapping[str, Any], *, compact: bool = False) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    else:
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    output.write_text(text, encoding="utf-8")


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _nested(payload: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, Mapping):
            return default
        value = value.get(key, default)
        if value is default:
            return default
    return value


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taxonomy-report", required=True)
    parser.add_argument("--scores", default=None, help="override source score dump path")
    parser.add_argument("--cell", choices=DECK_CELLS, default="entrenched")
    parser.add_argument("--include-true", action="store_true", help="include true records from the selected cell")
    parser.add_argument("--max-records", type=int, default=100)
    parser.add_argument("--json", required=True, help="output analysis report")
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--compact-json", action="store_true")
    args = parser.parse_args(argv)
    payload = run(
        taxonomy_report_path=args.taxonomy_report,
        score_dump_path=args.scores,
        output_path=args.json,
        cell=args.cell,
        false_only=not bool(args.include_true),
        max_records=args.max_records,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        metadata=_parse_metadata(args.metadata or ()),
        compact_json=bool(args.compact_json),
    )
    summary = payload["summary"]
    print(
        "detectability_blind_spot_analysis="
        f"cell={summary['cell']} selected={summary['selected_record_count']} "
        f"emitted={summary['emitted_record_count']} status={payload['status']}"
    )


if __name__ == "__main__":
    main()

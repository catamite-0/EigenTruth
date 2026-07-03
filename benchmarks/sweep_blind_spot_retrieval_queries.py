"""Sweep local retrieval query strategies over detectability blind spots.

This workflow is a no-model follow-up to ``audit_blind_spot_correction_routes``.
It varies local retrieval query fields and token-overlap thresholds, runs the
existing verifier ensemble for each strategy, and reports whether the target
route covers high-confidence false blind spots without exceeding the verified
false-alarm budget. It is an audit harness, not a production retriever.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmarks.analyze_retrieval_route_gaps import analyze_retrieval_route_gaps  # noqa: E402
from benchmarks.audit_blind_spot_correction_routes import (  # noqa: E402
    DEFAULT_TARGET_ROUTE,
    audit_blind_spot_correction_routes,
    load_blind_spot_records,
    load_verified_records_jsonl,
)
from benchmarks.build_evidence_fixture import (  # noqa: E402
    QUERY_FIELDS as EVIDENCE_QUERY_FIELDS,
)
from benchmarks.build_evidence_fixture import (  # noqa: E402
    SOURCE_FAMILY_FILTERS as EVIDENCE_SOURCE_FAMILY_FILTERS,
)
from benchmarks.build_evidence_fixture import (  # noqa: E402
    build_evidence_fixture,
    load_corpus,
    load_score_dump,
)
from benchmarks.eval_verifier_ensemble import build_verifier_ensemble_report  # noqa: E402
from eigentruth.json_utils import strict_json_dumps  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402

QUERY_FIELDS = (
    "answer",
    "question",
    "question_answer",
    "text",
    "citation_question",
    "citation_entity",
    "triple_slot",
)
DEFAULT_MIN_OVERLAPS = (0.95, 0.80, 0.65, 0.50)
DEFAULT_SOURCE_FAMILY_FILTERS = ("off",)
_SUPPORTED_QUERY_FIELDS = set(EVIDENCE_QUERY_FIELDS)
_SUPPORTED_SOURCE_FAMILY_FILTERS = set(EVIDENCE_SOURCE_FAMILY_FILTERS)


def sweep_blind_spot_retrieval_queries(
    *,
    scores_path: str | Path,
    corpus_paths: Sequence[str | Path],
    blind_spots_path: str | Path,
    source_binding_queue_path: str | Path | None = None,
    use_precomputed_retrieval_hits: bool | None = None,
    query_fields: Sequence[str] = QUERY_FIELDS,
    retriever_min_overlaps: Sequence[float] = DEFAULT_MIN_OVERLAPS,
    source_family_filters: Sequence[str] = DEFAULT_SOURCE_FAMILY_FILTERS,
    verified_records_dir: str | Path | None = None,
    retrieval_limit: int = 3,
    signal: str = "truth_proj",
    alpha: float = 0.10,
    repeats: int = 1,
    seed: int = 0,
    verifier_min_overlap: float = 0.65,
    enable_triple_evidence: bool = False,
    triple_min_slot_coverage: float = 1.0,
    triple_refute_object_mismatch: bool = False,
    target_route: str = DEFAULT_TARGET_ROUTE,
    max_verified_false_alarm: float = 0.05,
    min_blind_refuted_rate: float = 0.50,
    max_examples_per_bucket: int = 3,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run query-strategy sweep and return a JSON-ready report."""
    fields = _query_fields(query_fields)
    overlaps = _min_overlaps(retriever_min_overlaps)
    family_filters = _source_family_filters(source_family_filters)
    if retrieval_limit <= 0:
        raise ValueError("retrieval_limit must be positive.")
    if repeats < 1:
        raise ValueError("repeats must be >= 1.")
    alpha = float(alpha)
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")
    max_verified_false_alarm = float(max_verified_false_alarm)
    if not (0.0 <= max_verified_false_alarm <= 1.0):
        raise ValueError("max_verified_false_alarm must be in [0, 1].")
    min_blind_refuted_rate = float(min_blind_refuted_rate)
    if not (0.0 <= min_blind_refuted_rate <= 1.0):
        raise ValueError("min_blind_refuted_rate must be in [0, 1].")

    score_path = Path(scores_path)
    corpus_input_paths = tuple(Path(path) for path in corpus_paths)
    blind_path = Path(blind_spots_path)
    source_binding_path = None if source_binding_queue_path is None else Path(source_binding_queue_path)
    dump = load_score_dump(score_path)
    documents = load_corpus(corpus_input_paths)
    blind_spots = load_blind_spot_records(blind_path)
    source_binding_queue = None if source_binding_path is None else _load_json_object(source_binding_path)
    resolved_use_precomputed_hits = (
        source_binding_queue is not None
        if use_precomputed_retrieval_hits is None
        else bool(use_precomputed_retrieval_hits)
    )
    corpus_provenance = _corpus_provenance(corpus_input_paths)
    verified_records_output_dir = None if verified_records_dir is None else Path(verified_records_dir)
    if verified_records_output_dir is not None:
        verified_records_output_dir.mkdir(parents=True, exist_ok=True)
    strategies = []

    with tempfile.TemporaryDirectory(prefix="eigentruth-blind-query-sweep-") as temp_dir:
        temp_root = Path(temp_dir)
        for query_field in fields:
            for min_overlap in overlaps:
                for source_family_filter in family_filters:
                    strategy = _evaluate_strategy(
                        dump=dump,
                        documents=documents,
                        scores_path=score_path,
                        blind_spots=blind_spots,
                        source_binding_queue=source_binding_queue,
                        use_precomputed_retrieval_hits=resolved_use_precomputed_hits,
                        query_field=query_field,
                        retriever_min_overlap=min_overlap,
                        source_family_filter=source_family_filter,
                        verified_records_dir=verified_records_output_dir,
                        retrieval_limit=retrieval_limit,
                        signal=signal,
                        alpha=alpha,
                        repeats=repeats,
                        seed=seed,
                        verifier_min_overlap=verifier_min_overlap,
                        enable_triple_evidence=enable_triple_evidence,
                        triple_min_slot_coverage=triple_min_slot_coverage,
                        triple_refute_object_mismatch=triple_refute_object_mismatch,
                        target_route=target_route,
                        max_verified_false_alarm=max_verified_false_alarm,
                        min_blind_refuted_rate=min_blind_refuted_rate,
                        temp_root=temp_root,
                        max_examples_per_bucket=max_examples_per_bucket,
                    )
                    strategies.append(strategy)

    baseline = _baseline_strategy(strategies)
    best = _best_strategy(strategies)
    best_passing = _best_strategy([item for item in strategies if item["gate"]["pass"]])
    return {
        "schema_version": 1,
        "workflow": "blind_spot_retrieval_query_sweep",
        "status": "complete",
        "source": {
            "scores_path": str(score_path),
            "blind_spots_path": str(blind_path),
            "corpus_paths": tuple(str(path) for path in corpus_input_paths),
            "source_binding_queue_path": None if source_binding_path is None else str(source_binding_path),
            "corpora": corpus_provenance,
        },
        "config": {
            "query_fields": tuple(fields),
            "retriever_min_overlaps": tuple(overlaps),
            "source_family_filters": tuple(family_filters),
            "retrieval_limit": int(retrieval_limit),
            "signal": signal,
            "alpha": alpha,
            "repeats": int(repeats),
            "seed": int(seed),
            "verifier_min_overlap": float(verifier_min_overlap),
            "enable_triple_evidence": bool(enable_triple_evidence),
            "triple_min_slot_coverage": float(triple_min_slot_coverage),
            "triple_refute_object_mismatch": bool(triple_refute_object_mismatch),
            "target_route": target_route,
            "max_verified_false_alarm": max_verified_false_alarm,
            "min_blind_refuted_rate": min_blind_refuted_rate,
            "max_examples_per_bucket": int(max_examples_per_bucket),
            "source_binding_enabled": source_binding_path is not None,
            "use_precomputed_retrieval_hits": resolved_use_precomputed_hits,
            "verified_records_dir": None if verified_records_output_dir is None else str(verified_records_output_dir),
        },
        "summary": {
            "strategy_count": len(strategies),
            "blind_spot_count": len(blind_spots),
            "best_strategy": None if best is None else _strategy_key(best),
            "best_passing_strategy": None if best_passing is None else _strategy_key(best_passing),
            "baseline_strategy": None if baseline is None else _strategy_key(baseline),
            "baseline_blind_refuted_count": (
                None if baseline is None else baseline["blind_spot"]["target_route_refuted_count"]
            ),
            "best_blind_refuted_count": None if best is None else best["blind_spot"]["target_route_refuted_count"],
            "best_passing_blind_refuted_count": (
                None if best_passing is None else best_passing["blind_spot"]["target_route_refuted_count"]
            ),
            "controlled_corpus_warning": _controlled_corpus_warning(corpus_provenance),
        },
        "strategies": strategies,
        "metadata": dict(metadata or {}),
    }


def run(
    *,
    scores_path: str | Path,
    corpus_paths: Sequence[str | Path],
    blind_spots_path: str | Path,
    output_path: str | Path,
    source_binding_queue_path: str | Path | None = None,
    use_precomputed_retrieval_hits: bool | None = None,
    query_fields: Sequence[str] = QUERY_FIELDS,
    retriever_min_overlaps: Sequence[float] = DEFAULT_MIN_OVERLAPS,
    source_family_filters: Sequence[str] = DEFAULT_SOURCE_FAMILY_FILTERS,
    verified_records_dir: str | Path | None = None,
    retrieval_limit: int = 3,
    signal: str = "truth_proj",
    alpha: float = 0.10,
    repeats: int = 1,
    seed: int = 0,
    verifier_min_overlap: float = 0.65,
    enable_triple_evidence: bool = False,
    triple_min_slot_coverage: float = 1.0,
    triple_refute_object_mismatch: bool = False,
    target_route: str = DEFAULT_TARGET_ROUTE,
    max_verified_false_alarm: float = 0.05,
    min_blind_refuted_rate: float = 0.50,
    max_examples_per_bucket: int = 3,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    compact_json: bool = False,
) -> dict[str, Any]:
    """Run, write, optionally manifest, and optionally register the sweep."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    report = sweep_blind_spot_retrieval_queries(
        scores_path=scores_path,
        corpus_paths=corpus_paths,
        blind_spots_path=blind_spots_path,
        source_binding_queue_path=source_binding_queue_path,
        use_precomputed_retrieval_hits=use_precomputed_retrieval_hits,
        query_fields=query_fields,
        retriever_min_overlaps=retriever_min_overlaps,
        source_family_filters=source_family_filters,
        verified_records_dir=verified_records_dir,
        retrieval_limit=retrieval_limit,
        signal=signal,
        alpha=alpha,
        repeats=repeats,
        seed=seed,
        verifier_min_overlap=verifier_min_overlap,
        enable_triple_evidence=enable_triple_evidence,
        triple_min_slot_coverage=triple_min_slot_coverage,
        triple_refute_object_mismatch=triple_refute_object_mismatch,
        target_route=target_route,
        max_verified_false_alarm=max_verified_false_alarm,
        min_blind_refuted_rate=min_blind_refuted_rate,
        max_examples_per_bucket=max_examples_per_bucket,
        metadata=metadata,
    )
    if artifact_manifest_path is not None:
        report["paths"] = {"artifact_manifest": str(artifact_manifest_path)}
    output = Path(output_path)
    _write_json(output, report, compact=compact_json)
    if artifact_manifest_path is not None:
        manifest_path = Path(artifact_manifest_path)
        manifest = build_artifact_manifest(
            {
                "blind_spot_query_sweep": output,
                "scores": Path(scores_path),
                "blind_spots": Path(blind_spots_path),
                "source_binding_queue": None if source_binding_queue_path is None else Path(source_binding_queue_path),
                "verified_records_dir": None if verified_records_dir is None else Path(verified_records_dir),
                **{f"corpus_{index}": Path(path) for index, path in enumerate(corpus_paths)},
            },
            root=manifest_path.parent,
            metadata={
                "runner": "sweep_blind_spot_retrieval_queries",
                "status": report.get("status"),
                "best_strategy": _nested(report, "summary", "best_strategy"),
                "best_passing_strategy": _nested(report, "summary", "best_passing_strategy"),
                "blind_spot_count": _nested(report, "summary", "blind_spot_count"),
                "best_blind_refuted_count": _nested(report, "summary", "best_blind_refuted_count"),
                "best_passing_blind_refuted_count": _nested(
                    report,
                    "summary",
                    "best_passing_blind_refuted_count",
                ),
                "verified_records_dir": None if verified_records_dir is None else str(verified_records_dir),
            },
        )
        _write_json(manifest_path, manifest, compact=compact_json)
    if registry_path is not None:
        assert name is not None and version is not None
        ArtifactRegistry.load_json(registry_path).record_report(
            name=name,
            version=version,
            path=output,
            metadata={
                "workflow": "blind_spot_retrieval_query_sweep",
                "status": report.get("status"),
                "best_strategy": _nested(report, "summary", "best_strategy"),
                "best_passing_strategy": _nested(report, "summary", "best_passing_strategy"),
                "blind_spot_count": _nested(report, "summary", "blind_spot_count"),
                "best_blind_refuted_count": _nested(report, "summary", "best_blind_refuted_count"),
                "best_passing_blind_refuted_count": _nested(
                    report,
                    "summary",
                    "best_passing_blind_refuted_count",
                ),
                "verified_records_dir": None if verified_records_dir is None else str(verified_records_dir),
                "artifact_manifest": None if artifact_manifest_path is None else str(artifact_manifest_path),
                **dict(metadata or {}),
            },
        ).save_json()
    return report


def _evaluate_strategy(
    *,
    dump: Mapping[str, Any],
    documents: Sequence[Any],
    scores_path: Path,
    blind_spots: Sequence[Mapping[str, Any]],
    source_binding_queue: Mapping[str, Any] | None,
    use_precomputed_retrieval_hits: bool,
    query_field: str,
    retriever_min_overlap: float,
    source_family_filter: str,
    verified_records_dir: Path | None,
    retrieval_limit: int,
    signal: str,
    alpha: float,
    repeats: int,
    seed: int,
    verifier_min_overlap: float,
    enable_triple_evidence: bool,
    triple_min_slot_coverage: float,
    triple_refute_object_mismatch: bool,
    target_route: str,
    max_verified_false_alarm: float,
    min_blind_refuted_rate: float,
    temp_root: Path,
    max_examples_per_bucket: int,
) -> dict[str, Any]:
    key = _strategy_key_from_values(
        query_field,
        retriever_min_overlap,
        source_family_filter=source_family_filter,
    )
    fixture = build_evidence_fixture(
        dump,
        documents,
        retriever_min_overlap=float(retriever_min_overlap),
        retrieval_limit=int(retrieval_limit),
        query_field=query_field,
        include_label_metadata=False,
        source_family_filter=source_family_filter,
        source_binding_queue=source_binding_queue,
        use_precomputed_retrieval_hits=bool(use_precomputed_retrieval_hits),
    )
    claims_path = temp_root / f"{key}-claims.json"
    verified_records_path = temp_root / f"{key}-verified-records.jsonl"
    _write_json(claims_path, fixture, compact=True)
    verifier_report = build_verifier_ensemble_report(
        ((key, scores_path),),
        signal=signal,
        claims_path=claims_path,
        alphas=(float(alpha),),
        repeats=int(repeats),
        seed=int(seed),
        verifier_min_overlap=float(verifier_min_overlap),
        retriever_min_overlap=float(retriever_min_overlap),
        retrieval_limit=int(retrieval_limit),
        enable_triple_evidence=bool(enable_triple_evidence),
        triple_min_slot_coverage=float(triple_min_slot_coverage),
        triple_refute_object_mismatch=bool(triple_refute_object_mismatch),
        verified_records_path=verified_records_path,
    )
    run = verifier_report["runs"][0]
    alpha_payload = run["alphas"][str(float(alpha))]
    verified_rows = load_verified_records_jsonl(verified_records_path)
    durable_verified_records_path = _copy_verified_records_sidecar(
        verified_records_path,
        strategy_key=key,
        verified_records_dir=verified_records_dir,
    )
    blind_audit = audit_blind_spot_correction_routes(
        blind_spots,
        verified_rows,
        target_route=target_route,
        max_examples_per_bucket=max_examples_per_bucket,
    )
    gap_analysis = analyze_retrieval_route_gaps(
        verified_rows,
        max_examples_per_bucket=max_examples_per_bucket,
    )
    blind_summary = dict(blind_audit["summary"])
    verified_false_alarm = _optional_float(_nested(alpha_payload, "verified", "false_alarm"))
    blind_refuted_rate = float(blind_summary["target_route_refuted_rate"])
    gate_pass = (
        verified_false_alarm is not None
        and verified_false_alarm <= max_verified_false_alarm
        and blind_refuted_rate >= min_blind_refuted_rate
    )
    route_summary = _selected_route_summary(run, target_route)
    route_quality = _selected_route_quality(run, target_route)
    return {
        "key": key,
        "query_field": query_field,
        "retriever_min_overlap": float(retriever_min_overlap),
        "source_family_filter": source_family_filter,
        "paths": {
            "verified_records_jsonl": (
                None
                if durable_verified_records_path is None
                else str(durable_verified_records_path)
            ),
        },
        "retrieval": dict(fixture["summary"]),
        "verification": {
            "internal": _nested(alpha_payload, "internal", default={}),
            "verified": _nested(alpha_payload, "verified", default={}),
        },
        "target_route_summary": route_summary,
        "target_route_quality": route_quality,
        "blind_spot": blind_summary,
        "gap_analysis": _compact_gap_analysis(gap_analysis),
        "examples": blind_audit.get("examples", {}),
        "gate": {
            "pass": bool(gate_pass),
            "max_verified_false_alarm": max_verified_false_alarm,
            "min_blind_refuted_rate": min_blind_refuted_rate,
            "verified_false_alarm": verified_false_alarm,
            "blind_refuted_rate": blind_refuted_rate,
        },
    }


def _copy_verified_records_sidecar(
    source_path: Path,
    *,
    strategy_key: str,
    verified_records_dir: Path | None,
) -> Path | None:
    if verified_records_dir is None:
        return None
    output = verified_records_dir / f"{_safe_filename(strategy_key)}-verified-records.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, output)
    return output


def _selected_route_summary(run: Mapping[str, Any], target_route: str) -> dict[str, Any]:
    route = _nested(run, "route_summary", "by_route", target_route, default={})
    if not isinstance(route, Mapping):
        return {}
    keys = (
        "selected",
        "labels",
        "statuses",
        "retrieval_hit_count",
        "retrieval_use_rate",
        "mean_retrieval_hits",
        "mean_attempted_route_count",
        "mean_duration_seconds",
        "p95_duration_seconds",
    )
    return {key: route.get(key) for key in keys if key in route}


def _compact_gap_analysis(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "summary": dict(_nested(report, "summary", default={})),
        "gap_buckets": dict(_nested(report, "gap_buckets", default={})),
        "top_gap_tokens": tuple(_nested(report, "top_gap_tokens", default=())),
        "top_hit_sources": tuple(_nested(report, "top_hit_sources", default=())),
        "hit_property_counts": dict(_nested(report, "hit_property_counts", default={})),
    }


def _selected_route_quality(run: Mapping[str, Any], target_route: str) -> dict[str, Any]:
    quality = _nested(run, "route_quality", target_route, default={})
    if not isinstance(quality, Mapping):
        return {}
    keys = (
        "selection_rate",
        "decision_accuracy",
        "decision_error_rate",
        "false_refuted_rate",
        "false_supported_rate",
        "true_refuted_rate",
        "true_supported_rate",
        "retrieval_use_rate",
        "mean_retrieval_hits",
        "n_false",
        "n_true",
    )
    return {key: quality.get(key) for key in keys if key in quality}


def _best_strategy(strategies: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    if not strategies:
        return None
    return max(
        strategies,
        key=lambda item: (
            int(_nested(item, "blind_spot", "target_route_refuted_count", default=0)),
            -_float_with_default(_nested(item, "gate", "verified_false_alarm"), default=1.0),
            _float_with_default(_nested(item, "target_route_quality", "decision_accuracy"), default=0.0),
            -int(_nested(item, "retrieval", "total_hits", default=0) or 0),
        ),
    )


def _baseline_strategy(strategies: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    for item in strategies:
        if (
            item.get("query_field") == "answer"
            and item.get("source_family_filter", "off") == "off"
            and float(item.get("retriever_min_overlap", -1.0)) == 0.95
        ):
            return item
    return None


def _strategy_key(strategy: Mapping[str, Any]) -> str:
    return _strategy_key_from_values(
        str(strategy["query_field"]),
        float(strategy["retriever_min_overlap"]),
        source_family_filter=str(strategy.get("source_family_filter", "off")),
    )


def _strategy_key_from_values(
    query_field: str,
    min_overlap: float,
    *,
    source_family_filter: str = "off",
) -> str:
    overlap = str(float(min_overlap)).replace(".", "p")
    key = f"{query_field}_overlap_{overlap}"
    if source_family_filter != "off":
        key = f"{key}_sf_{source_family_filter}"
    return key


def _safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)


def _corpus_provenance(paths: Sequence[Path]) -> tuple[dict[str, Any], ...]:
    payloads = []
    for path in paths:
        data = _load_json_object(path) if path.suffix.lower() == ".json" else {}
        payloads.append({
            "path": str(path),
            "corpus_type": data.get("corpus_type"),
            "summary": data.get("summary"),
        })
    return tuple(payloads)


def _controlled_corpus_warning(corpora: Sequence[Mapping[str, Any]]) -> str | None:
    controlled_types = {
        "truthfulqa_correct_answer_evidence",
        "truthfulqa_answer_echo_stress",
    }
    corpus_types = {str(item.get("corpus_type")) for item in corpora if item.get("corpus_type") is not None}
    if corpus_types & controlled_types:
        return (
            "At least one corpus is a controlled TruthfulQA corpus. Treat high coverage as route-design "
            "evidence, not open-domain grounding evidence."
        )
    return None


def _query_fields(values: Sequence[str]) -> tuple[str, ...]:
    fields = tuple(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))
    if not fields:
        raise ValueError("query_fields must not be empty.")
    invalid = tuple(field for field in fields if field not in _SUPPORTED_QUERY_FIELDS)
    if invalid:
        raise ValueError(
            "query_fields contains unsupported values: "
            f"{', '.join(invalid)}. Supported values: {', '.join(EVIDENCE_QUERY_FIELDS)}."
        )
    return fields


def _min_overlaps(values: Sequence[float]) -> tuple[float, ...]:
    overlaps = tuple(float(value) for value in values)
    if not overlaps:
        raise ValueError("retriever_min_overlaps must not be empty.")
    if any(not (0.0 <= value <= 1.0) for value in overlaps):
        raise ValueError("retriever_min_overlaps must be in [0, 1].")
    return overlaps


def _source_family_filters(values: Sequence[str]) -> tuple[str, ...]:
    modes = tuple(dict.fromkeys(str(value).strip().casefold() for value in values if str(value).strip()))
    if not modes:
        raise ValueError("source_family_filters must not be empty.")
    invalid = tuple(mode for mode in modes if mode not in _SUPPORTED_SOURCE_FAMILY_FILTERS)
    if invalid:
        raise ValueError(
            "source_family_filters contains unsupported values: "
            f"{', '.join(invalid)}. Supported values: {', '.join(EVIDENCE_SOURCE_FAMILY_FILTERS)}."
        )
    return modes


def _parse_csv(value: str | None, *, default: Sequence[str]) -> tuple[str, ...]:
    if value is None:
        return tuple(default)
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _parse_float_csv(value: str | None, *, default: Sequence[float]) -> tuple[float, ...]:
    if value is None:
        return tuple(float(item) for item in default)
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


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
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _write_json(path: str | Path, payload: Mapping[str, Any], *, compact: bool = False) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = strict_json_dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    else:
        text = strict_json_dumps(payload, indent=2, sort_keys=True) + "\n"
    output.write_text(text, encoding="utf-8")


def _nested(payload: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, Mapping):
            return default
        value = value.get(key, default)
        if value is default:
            return default
    return value


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _float_with_default(value: Any, *, default: float) -> float:
    coerced = _optional_float(value)
    return float(default) if coerced is None else coerced


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True, help="statement-bearing score dump JSON or JSONL manifest")
    parser.add_argument("--corpus", action="append", required=True, help="local JSON/JSONL/text corpus path")
    parser.add_argument("--blind-spots", required=True, help="detectability blind-spot report JSON")
    parser.add_argument("--source-binding-queue", default=None,
                        help="optional evidence queue JSON used to bind retrieval to matching source requests")
    precomputed_group = parser.add_mutually_exclusive_group()
    precomputed_group.add_argument("--use-precomputed-retrieval-hits", dest="use_precomputed_retrieval_hits",
                                   action="store_true", default=None,
                                   help="treat fixture retrieval documents as already retrieved by the query strategy")
    precomputed_group.add_argument("--no-use-precomputed-retrieval-hits", dest="use_precomputed_retrieval_hits",
                                   action="store_false",
                                   help="force verifier ensemble to re-retrieve from fixture documents")
    parser.add_argument("--query-fields", default=",".join(QUERY_FIELDS))
    parser.add_argument("--retriever-min-overlaps", default=",".join(str(value) for value in DEFAULT_MIN_OVERLAPS))
    parser.add_argument("--source-family-filters", default=",".join(DEFAULT_SOURCE_FAMILY_FILTERS),
                        help="comma-separated source-family evidence filters to sweep: off,planned,planned_rerank")
    parser.add_argument("--verified-records-dir", default=None,
                        help="optional directory to save per-strategy verified-records JSONL sidecars")
    parser.add_argument("--retrieval-limit", type=int, default=3)
    parser.add_argument("--signal", default="truth_proj")
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verifier-min-overlap", type=float, default=0.65)
    parser.add_argument("--enable-triple-evidence", action="store_true")
    parser.add_argument("--triple-min-slot-coverage", type=float, default=1.0)
    parser.add_argument("--triple-refute-object-mismatch", action="store_true")
    parser.add_argument("--target-route", default=DEFAULT_TARGET_ROUTE)
    parser.add_argument("--max-verified-false-alarm", type=float, default=0.05)
    parser.add_argument("--min-blind-refuted-rate", type=float, default=0.50)
    parser.add_argument("--max-examples-per-bucket", type=int, default=3)
    parser.add_argument("--json", required=True)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--compact-json", action="store_true")
    args = parser.parse_args(argv)
    payload = run(
        scores_path=args.scores,
        corpus_paths=tuple(args.corpus),
        blind_spots_path=args.blind_spots,
        output_path=args.json,
        source_binding_queue_path=args.source_binding_queue,
        use_precomputed_retrieval_hits=args.use_precomputed_retrieval_hits,
        query_fields=_parse_csv(args.query_fields, default=QUERY_FIELDS),
        retriever_min_overlaps=_parse_float_csv(args.retriever_min_overlaps, default=DEFAULT_MIN_OVERLAPS),
        source_family_filters=_source_family_filters(
            _parse_csv(args.source_family_filters, default=DEFAULT_SOURCE_FAMILY_FILTERS)
        ),
        verified_records_dir=args.verified_records_dir,
        retrieval_limit=args.retrieval_limit,
        signal=args.signal,
        alpha=args.alpha,
        repeats=args.repeats,
        seed=args.seed,
        verifier_min_overlap=args.verifier_min_overlap,
        enable_triple_evidence=bool(args.enable_triple_evidence),
        triple_min_slot_coverage=float(args.triple_min_slot_coverage),
        triple_refute_object_mismatch=bool(args.triple_refute_object_mismatch),
        target_route=args.target_route,
        max_verified_false_alarm=args.max_verified_false_alarm,
        min_blind_refuted_rate=args.min_blind_refuted_rate,
        max_examples_per_bucket=args.max_examples_per_bucket,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        metadata=_parse_metadata(args.metadata or ()),
        compact_json=bool(args.compact_json),
    )
    summary = payload["summary"]
    print(
        "blind_spot_retrieval_query_sweep_ok "
        f"strategies={summary['strategy_count']} "
        f"best={summary['best_strategy']} "
        f"best_passing={summary['best_passing_strategy']} "
        f"blind_refuted={summary['best_passing_blind_refuted_count']}"
    )


if __name__ == "__main__":
    main()

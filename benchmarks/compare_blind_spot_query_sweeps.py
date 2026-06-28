"""Compare controlled and external blind-spot query-sweep reports.

This workflow turns query-sweep evidence into a fail-closed handoff. A
controlled TruthfulQA correct-answer corpus can identify promising query
construction, but it must not promote a product/runtime default unless matching
external or structured-evidence sweeps also pass the configured blind-spot and
false-alarm gates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eigentruth.json_utils import strict_json_dumps  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402

CONTROLLED_CORPUS_TYPES = {
    "truthfulqa_answer_echo_stress",
    "truthfulqa_correct_answer_evidence",
}
EXTERNAL_CORPUS_TYPES = {
    "external_evidence_candidate",
    "structured_qa_external_evidence",
}


def compare_blind_spot_query_sweeps(
    *,
    controlled_sweep_paths: Sequence[str | Path],
    external_sweep_paths: Sequence[str | Path],
    min_controlled_blind_refuted_rate: float = 0.50,
    min_external_blind_refuted_rate: float = 0.50,
    max_controlled_verified_false_alarm: float = 0.05,
    max_external_verified_false_alarm: float = 0.05,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a JSON-ready controlled-vs-external query-sweep comparison."""
    controlled_paths = _paths(controlled_sweep_paths, name="controlled_sweep_paths")
    external_paths = _paths(external_sweep_paths, name="external_sweep_paths")
    min_controlled_blind_refuted_rate = _unit_interval(
        min_controlled_blind_refuted_rate,
        name="min_controlled_blind_refuted_rate",
    )
    min_external_blind_refuted_rate = _unit_interval(
        min_external_blind_refuted_rate,
        name="min_external_blind_refuted_rate",
    )
    max_controlled_verified_false_alarm = _unit_interval(
        max_controlled_verified_false_alarm,
        name="max_controlled_verified_false_alarm",
    )
    max_external_verified_false_alarm = _unit_interval(
        max_external_verified_false_alarm,
        name="max_external_verified_false_alarm",
    )

    controlled = tuple(
        _sweep_report(
            path,
            min_blind_refuted_rate=min_controlled_blind_refuted_rate,
            max_verified_false_alarm=max_controlled_verified_false_alarm,
        )
        for path in controlled_paths
    )
    external = tuple(
        _sweep_report(
            path,
            min_blind_refuted_rate=min_external_blind_refuted_rate,
            max_verified_false_alarm=max_external_verified_false_alarm,
        )
        for path in external_paths
    )
    controlled_best = _best_row(controlled)
    external_best = _best_row(external)
    blocking_reasons = _blocking_reasons(
        controlled=controlled,
        external=external,
        controlled_best=controlled_best,
        external_best=external_best,
        min_controlled_blind_refuted_rate=min_controlled_blind_refuted_rate,
        min_external_blind_refuted_rate=min_external_blind_refuted_rate,
    )
    passed = not blocking_reasons
    status = "promote" if passed else "blocked"
    return {
        "schema_version": 1,
        "workflow": "blind_spot_query_sweep_provenance_comparison",
        "status": status,
        "decision": {
            "status": status,
            "passed": passed,
            "blocking_reasons": blocking_reasons,
            "recommended_controlled_strategy": _nested(
                controlled_best,
                "gate",
                "best_strategy",
            ),
            "recommended_external_strategy": _nested(
                external_best,
                "gate",
                "best_strategy",
            ),
        },
        "config": {
            "min_controlled_blind_refuted_rate": min_controlled_blind_refuted_rate,
            "min_external_blind_refuted_rate": min_external_blind_refuted_rate,
            "max_controlled_verified_false_alarm": max_controlled_verified_false_alarm,
            "max_external_verified_false_alarm": max_external_verified_false_alarm,
        },
        "summary": _summary(
            controlled=controlled,
            external=external,
            controlled_best=controlled_best,
            external_best=external_best,
        ),
        "controlled_sweeps": controlled,
        "external_sweeps": external,
        "metadata": dict(metadata or {}),
    }


def run(
    *,
    controlled_sweep_paths: Sequence[str | Path],
    external_sweep_paths: Sequence[str | Path],
    output_path: str | Path,
    min_controlled_blind_refuted_rate: float = 0.50,
    min_external_blind_refuted_rate: float = 0.50,
    max_controlled_verified_false_alarm: float = 0.05,
    max_external_verified_false_alarm: float = 0.05,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    compact_json: bool = False,
) -> dict[str, Any]:
    """Compare, write, optionally manifest, and optionally register a report."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    report = compare_blind_spot_query_sweeps(
        controlled_sweep_paths=controlled_sweep_paths,
        external_sweep_paths=external_sweep_paths,
        min_controlled_blind_refuted_rate=min_controlled_blind_refuted_rate,
        min_external_blind_refuted_rate=min_external_blind_refuted_rate,
        max_controlled_verified_false_alarm=max_controlled_verified_false_alarm,
        max_external_verified_false_alarm=max_external_verified_false_alarm,
        metadata=metadata,
    )
    output = Path(output_path)
    if artifact_manifest_path is not None:
        report["paths"] = {"artifact_manifest": str(artifact_manifest_path)}
    _write_json(output, report, compact=compact_json)
    if artifact_manifest_path is not None:
        manifest_path = Path(artifact_manifest_path)
        manifest = build_artifact_manifest(
            {
                "query_sweep_comparison": output,
                **{f"controlled_sweep_{index}": path for index, path in enumerate(controlled_sweep_paths)},
                **{f"external_sweep_{index}": path for index, path in enumerate(external_sweep_paths)},
            },
            root=manifest_path.parent,
            metadata={
                "runner": "compare_blind_spot_query_sweeps",
                "status": report["status"],
                "passed": report["decision"]["passed"],
                "recommended_controlled_strategy": report["decision"]["recommended_controlled_strategy"],
                "recommended_external_strategy": report["decision"]["recommended_external_strategy"],
                "controlled_best_blind_refuted_rate": _nested(
                    report,
                    "summary",
                    "controlled_best_blind_refuted_rate",
                ),
                "external_best_blind_refuted_rate": _nested(
                    report,
                    "summary",
                    "external_best_blind_refuted_rate",
                ),
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
                "workflow": report["workflow"],
                "status": report["status"],
                "passed": report["decision"]["passed"],
                "recommended_controlled_strategy": report["decision"]["recommended_controlled_strategy"],
                "recommended_external_strategy": report["decision"]["recommended_external_strategy"],
                "artifact_manifest": None if artifact_manifest_path is None else str(artifact_manifest_path),
                **dict(metadata or {}),
            },
        ).save_json()
    return report


def _sweep_report(
    path: Path,
    *,
    min_blind_refuted_rate: float,
    max_verified_false_alarm: float,
) -> dict[str, Any]:
    payload = _load_json_object(path)
    if payload.get("workflow") != "blind_spot_retrieval_query_sweep":
        raise ValueError(f"{path} is not a blind_spot_retrieval_query_sweep report.")
    strategies = tuple(
        _strategy_row(
            strategy,
            min_blind_refuted_rate=min_blind_refuted_rate,
            max_verified_false_alarm=max_verified_false_alarm,
        )
        for strategy in _sequence(payload.get("strategies"))
        if isinstance(strategy, Mapping)
    )
    best = _best_strategy(strategies)
    gate_best = _best_strategy(tuple(strategy for strategy in strategies if strategy["gate"]["passed"]))
    source_type = _source_type(payload)
    return {
        "path": str(path),
        "status": payload.get("status"),
        "source_type": source_type,
        "corpus_types": _corpus_types(payload),
        "controlled_corpus_warning": _nested(payload, "summary", "controlled_corpus_warning"),
        "blind_spot_count": _nested(payload, "summary", "blind_spot_count"),
        "source_summary": dict(_mapping(payload.get("summary"))),
        "gate": {
            "passed": gate_best is not None,
            "best_strategy": None if gate_best is None else gate_best["key"],
            "best_blind_refuted_count": None if gate_best is None else gate_best["blind_refuted_count"],
            "best_blind_refuted_rate": None if gate_best is None else gate_best["blind_refuted_rate"],
            "best_verified_false_alarm": None if gate_best is None else gate_best["verified_false_alarm"],
            "min_blind_refuted_rate": min_blind_refuted_rate,
            "max_verified_false_alarm": max_verified_false_alarm,
        },
        "best_observed": {
            "strategy": None if best is None else best["key"],
            "blind_refuted_count": None if best is None else best["blind_refuted_count"],
            "blind_refuted_rate": None if best is None else best["blind_refuted_rate"],
            "verified_false_alarm": None if best is None else best["verified_false_alarm"],
        },
        "strategies": strategies,
    }


def _strategy_row(
    strategy: Mapping[str, Any],
    *,
    min_blind_refuted_rate: float,
    max_verified_false_alarm: float,
) -> dict[str, Any]:
    blind_refuted_count = _optional_int(_nested(strategy, "blind_spot", "target_route_refuted_count"))
    blind_refuted_rate = _optional_float(_nested(strategy, "blind_spot", "target_route_refuted_rate"))
    verified_false_alarm = _optional_float(_nested(strategy, "gate", "verified_false_alarm"))
    passed = (
        blind_refuted_rate is not None
        and verified_false_alarm is not None
        and blind_refuted_rate >= min_blind_refuted_rate
        and verified_false_alarm <= max_verified_false_alarm
    )
    return {
        "key": strategy.get("key"),
        "query_field": strategy.get("query_field"),
        "retriever_min_overlap": _optional_float(strategy.get("retriever_min_overlap")),
        "records_with_hits": _optional_int(_nested(strategy, "retrieval", "records_with_hits")),
        "total_hits": _optional_int(_nested(strategy, "retrieval", "total_hits")),
        "blind_refuted_count": blind_refuted_count,
        "blind_refuted_rate": blind_refuted_rate,
        "verified_false_alarm": verified_false_alarm,
        "decision_accuracy": _optional_float(_nested(strategy, "target_route_quality", "decision_accuracy")),
        "gate": {
            "passed": passed,
            "min_blind_refuted_rate": min_blind_refuted_rate,
            "max_verified_false_alarm": max_verified_false_alarm,
        },
    }


def _best_strategy(strategies: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    if not strategies:
        return None
    return max(
        strategies,
        key=lambda item: (
            _int_with_default(item.get("blind_refuted_count"), default=0),
            -_float_with_default(item.get("verified_false_alarm"), default=1.0),
            _float_with_default(item.get("decision_accuracy"), default=0.0),
            -_int_with_default(item.get("total_hits"), default=0),
        ),
    )


def _best_row(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    if not rows:
        return None
    return max(
        rows,
        key=lambda item: (
            _float_with_default(_nested(item, "gate", "best_blind_refuted_rate"), default=-1.0),
            _float_with_default(_nested(item, "best_observed", "blind_refuted_rate"), default=0.0),
            -_float_with_default(_nested(item, "gate", "best_verified_false_alarm"), default=1.0),
        ),
    )


def _blocking_reasons(
    *,
    controlled: Sequence[Mapping[str, Any]],
    external: Sequence[Mapping[str, Any]],
    controlled_best: Mapping[str, Any] | None,
    external_best: Mapping[str, Any] | None,
    min_controlled_blind_refuted_rate: float,
    min_external_blind_refuted_rate: float,
) -> tuple[dict[str, Any], ...]:
    reasons: list[dict[str, Any]] = []
    if not any(_nested(row, "gate", "passed") is True for row in controlled):
        reasons.append({
            "gate": "controlled_query_sweep",
            "reason": "no controlled sweep has a strategy that passes the configured blind-spot gate",
            "min_blind_refuted_rate": min_controlled_blind_refuted_rate,
            "best_rate": _nested(controlled_best, "best_observed", "blind_refuted_rate"),
        })
    external_passing = [row for row in external if _nested(row, "gate", "passed") is True]
    if not external_passing:
        reasons.append({
            "gate": "external_query_sweep",
            "reason": "no external or structured-evidence sweep has a passing strategy",
            "min_blind_refuted_rate": min_external_blind_refuted_rate,
            "best_rate": _nested(external_best, "best_observed", "blind_refuted_rate"),
            "best_verified_false_alarm": _nested(external_best, "best_observed", "verified_false_alarm"),
        })
    if controlled and external and not external_passing:
        reasons.append({
            "gate": "controlled_only_signal",
            "reason": "controlled query coverage does not generalize to external/structured evidence",
            "controlled_best_rate": _nested(controlled_best, "best_observed", "blind_refuted_rate"),
            "external_best_rate": _nested(external_best, "best_observed", "blind_refuted_rate"),
        })
    return tuple(reasons)


def _summary(
    *,
    controlled: Sequence[Mapping[str, Any]],
    external: Sequence[Mapping[str, Any]],
    controlled_best: Mapping[str, Any] | None,
    external_best: Mapping[str, Any] | None,
) -> dict[str, Any]:
    controlled_best_rate = _nested(controlled_best, "best_observed", "blind_refuted_rate")
    external_best_rate = _nested(external_best, "best_observed", "blind_refuted_rate")
    return {
        "controlled_sweep_count": len(controlled),
        "external_sweep_count": len(external),
        "controlled_passing_count": sum(1 for row in controlled if _nested(row, "gate", "passed") is True),
        "external_passing_count": sum(1 for row in external if _nested(row, "gate", "passed") is True),
        "controlled_best_source": None if controlled_best is None else controlled_best["path"],
        "external_best_source": None if external_best is None else external_best["path"],
        "controlled_best_strategy": _nested(controlled_best, "best_observed", "strategy"),
        "external_best_strategy": _nested(external_best, "best_observed", "strategy"),
        "controlled_best_blind_refuted_rate": controlled_best_rate,
        "external_best_blind_refuted_rate": external_best_rate,
        "generalization_gap": _gap(controlled_best_rate, external_best_rate),
        "source_type_counts": _source_type_counts((*controlled, *external)),
    }


def _source_type(payload: Mapping[str, Any]) -> str:
    corpus_types = set(_corpus_types(payload))
    if corpus_types & CONTROLLED_CORPUS_TYPES or _nested(payload, "summary", "controlled_corpus_warning"):
        return "controlled"
    if corpus_types & EXTERNAL_CORPUS_TYPES or any("external" in item for item in corpus_types):
        return "external"
    return "unclassified"


def _corpus_types(payload: Mapping[str, Any]) -> tuple[str, ...]:
    source = _mapping(payload.get("source"))
    corpora = _sequence(source.get("corpora"))
    values = []
    for item in corpora:
        if isinstance(item, Mapping) and item.get("corpus_type") is not None:
            values.append(str(item["corpus_type"]))
    return tuple(dict.fromkeys(values))


def _source_type_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("source_type", "unknown"))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _gap(left: Any, right: Any) -> float | None:
    left_float = _optional_float(left)
    right_float = _optional_float(right)
    if left_float is None or right_float is None:
        return None
    return left_float - right_float


def _paths(values: Sequence[str | Path], *, name: str) -> tuple[Path, ...]:
    paths = tuple(Path(path) for path in values)
    if not paths:
        raise ValueError(f"{name} must not be empty.")
    missing = tuple(str(path) for path in paths if not path.exists())
    if missing:
        raise FileNotFoundError(f"{name} contains missing files: {', '.join(missing)}")
    return paths


def _unit_interval(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a float in [0, 1].") from exc
    if not (0.0 <= result <= 1.0):
        raise ValueError(f"{name} must be in [0, 1].")
    return result


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
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"{path} must contain a JSON object.")
    return dict(data)


def _write_json(path: str | Path, payload: Mapping[str, Any], *, compact: bool = False) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = strict_json_dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    else:
        text = strict_json_dumps(payload, indent=2, sort_keys=True) + "\n"
    output.write_text(text, encoding="utf-8")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _nested(payload: Any, *keys: str, default: Any = None) -> Any:
    value = payload
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


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_with_default(value: Any, *, default: float) -> float:
    coerced = _optional_float(value)
    return default if coerced is None else coerced


def _int_with_default(value: Any, *, default: int) -> int:
    coerced = _optional_int(value)
    return default if coerced is None else coerced


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controlled-sweep", action="append", required=True)
    parser.add_argument("--external-sweep", action="append", required=True)
    parser.add_argument("--min-controlled-blind-refuted-rate", type=float, default=0.50)
    parser.add_argument("--min-external-blind-refuted-rate", type=float, default=0.50)
    parser.add_argument("--max-controlled-verified-false-alarm", type=float, default=0.05)
    parser.add_argument("--max-external-verified-false-alarm", type=float, default=0.05)
    parser.add_argument("--json", required=True)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--compact-json", action="store_true")
    parser.add_argument("--fail-on-blocked", action="store_true")
    args = parser.parse_args(argv)
    payload = run(
        controlled_sweep_paths=tuple(args.controlled_sweep),
        external_sweep_paths=tuple(args.external_sweep),
        output_path=args.json,
        min_controlled_blind_refuted_rate=args.min_controlled_blind_refuted_rate,
        min_external_blind_refuted_rate=args.min_external_blind_refuted_rate,
        max_controlled_verified_false_alarm=args.max_controlled_verified_false_alarm,
        max_external_verified_false_alarm=args.max_external_verified_false_alarm,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        metadata=_parse_metadata(args.metadata or ()),
        compact_json=bool(args.compact_json),
    )
    print(
        "blind_spot_query_sweep_comparison_ok "
        f"status={payload['status']} "
        f"controlled_passing={payload['summary']['controlled_passing_count']} "
        f"external_passing={payload['summary']['external_passing_count']} "
        f"external_best_rate={payload['summary']['external_best_blind_refuted_rate']}"
    )
    if args.fail_on_blocked and payload["status"] != "promote":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

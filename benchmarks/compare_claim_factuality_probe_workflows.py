"""Compare claim factuality probe workflow reports.

This script aggregates compact ``run_claim_factuality_probe_workflow.py``
reports without loading hidden-state records or torch artifacts. It is the
benchmark-side gate between single-run claim factuality probe evidence and a
multi-run report that can be handed to release/comparison workflows.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eigentruth.json_utils import to_jsonable  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402


@dataclass(frozen=True)
class ClaimFactualityProbeWorkflowComparisonConfig:
    """Configuration for comparing compact claim factuality workflow reports."""

    workflow_reports: Mapping[str, str | Path]
    output_path: str | Path
    artifact_manifest_path: str | Path | None = None
    registry_path: str | Path | None = None
    register_name: str | None = None
    register_version: str = "0.1"
    min_model_count: int = 2
    min_record_count: int = 1
    min_test_label_auroc: float = 0.5
    min_redline_auroc_margin: float = 0.0
    require_ready_status: bool = True
    require_manifest_clean: bool = True
    require_conformal: bool = True
    require_redline: bool = True
    compact_json: bool = False

    def __post_init__(self) -> None:
        reports = {_safe_run_name(name): Path(path) for name, path in self.workflow_reports.items()}
        if not reports:
            raise ValueError("workflow_reports must not be empty.")
        if len(reports) != len(self.workflow_reports):
            raise ValueError("workflow report names must be unique after normalization.")
        object.__setattr__(self, "workflow_reports", reports)
        object.__setattr__(self, "output_path", Path(self.output_path))
        if self.artifact_manifest_path is not None:
            object.__setattr__(self, "artifact_manifest_path", Path(self.artifact_manifest_path))
        if self.registry_path is not None:
            object.__setattr__(self, "registry_path", Path(self.registry_path))
        if self.register_name is not None:
            register_name = str(self.register_name).strip()
            if not register_name:
                raise ValueError("register_name must be non-empty when provided.")
            object.__setattr__(self, "register_name", register_name)
        object.__setattr__(self, "register_version", str(self.register_version))
        object.__setattr__(self, "min_model_count", int(self.min_model_count))
        object.__setattr__(self, "min_record_count", int(self.min_record_count))
        object.__setattr__(self, "min_test_label_auroc", float(self.min_test_label_auroc))
        object.__setattr__(self, "min_redline_auroc_margin", float(self.min_redline_auroc_margin))
        if self.min_model_count < 1:
            raise ValueError("min_model_count must be >=1.")
        if self.min_record_count < 1:
            raise ValueError("min_record_count must be >=1.")
        if not (0.0 <= self.min_test_label_auroc <= 1.0):
            raise ValueError("min_test_label_auroc must be in [0, 1].")
        if self.min_redline_auroc_margin < 0.0:
            raise ValueError("min_redline_auroc_margin must be >=0.")
        if self.register_name is not None and self.registry_path is None:
            raise ValueError("register_name requires registry_path.")


def compare_claim_factuality_probe_workflows(
    config: ClaimFactualityProbeWorkflowComparisonConfig,
) -> dict[str, Any]:
    """Load claim factuality workflow reports, summarize metrics, and apply gates."""
    runs = [_run_summary(name, path) for name, path in sorted(config.workflow_reports.items())]
    failures = _gate_failures(config, runs)
    status = "ready" if not failures else "blocked"
    model_names = sorted({str(run["effective_model"]) for run in runs if run.get("effective_model")})
    leaderboard = _leaderboard(runs)
    registry_record_key = (
        None if config.register_name is None else f"report:{config.register_name}:{config.register_version}"
    )
    registry_manifest_record_key = (
        None
        if config.register_name is None
        else f"benchmark_manifest:{config.register_name}:{config.register_version}"
    )
    payload = {
        "schema_version": 1,
        "workflow": "claim_factuality_probe_workflow_comparison",
        "status": status,
        "config": {
            "min_model_count": config.min_model_count,
            "min_record_count": config.min_record_count,
            "min_test_label_auroc": config.min_test_label_auroc,
            "min_redline_auroc_margin": config.min_redline_auroc_margin,
            "require_ready_status": config.require_ready_status,
            "require_manifest_clean": config.require_manifest_clean,
            "require_conformal": config.require_conformal,
            "require_redline": config.require_redline,
        },
        "promotion_gate": {
            "failures": failures,
            "ready_run_count": sum(1 for run in runs if run.get("status") == "ready"),
            "model_count": len(model_names),
            "models": model_names,
            "redline_run_count": sum(1 for run in runs if run.get("redline_available") is True),
            "redline_passed": not any(_is_redline_failure(failure) for failure in failures),
            "best_run": leaderboard[0]["name"] if leaderboard else None,
        },
        "runs": runs,
        "leaderboard": leaderboard,
        "evidence_scope": {
            "claim": "compact multi-run claim factuality probe workflow comparison with text redline gates",
            "not_a_claim": "production-quality hallucination detection or long-form factuality correction",
            "notes": (
                "This report compares saved workflow artifacts. Small synthetic or tiny-model runs should be "
                "treated as plumbing evidence until larger held-out multi-model and external-domain gates pass."
            ),
        },
        "paths": {
            "report": str(config.output_path),
            "artifact_manifest": None if config.artifact_manifest_path is None else str(config.artifact_manifest_path),
            "registry": None if config.registry_path is None else str(config.registry_path),
            "workflow_reports": {name: str(path) for name, path in config.workflow_reports.items()},
        },
        "registry_record": registry_record_key,
        "registry_manifest_record": registry_manifest_record_key,
    }
    _write_json(config.output_path, payload, compact=config.compact_json)
    manifest = None
    if config.artifact_manifest_path is not None:
        manifest = build_artifact_manifest(
            {
                "comparison_report": config.output_path,
                **{f"workflow_report.{name}": path for name, path in config.workflow_reports.items()},
            },
            root=Path(config.artifact_manifest_path).parent,
            metadata={
                "workflow": "claim_factuality_probe_workflow_comparison",
                "status": status,
                "run_count": len(runs),
                "model_count": len(model_names),
                "redline_passed": payload["promotion_gate"]["redline_passed"],
                "best_run": payload["promotion_gate"]["best_run"],
            },
        )
        _write_json(config.artifact_manifest_path, manifest, compact=False)
        payload["artifact_manifest_summary"] = manifest["summary"]
        _write_json(config.output_path, payload, compact=config.compact_json)
    if config.registry_path is not None and config.register_name is not None:
        _record_registry(config, report=payload, manifest=manifest)
    print(
        "claim_factuality_probe_workflow_comparison_ok "
        f"status={status} runs={len(runs)} models={len(model_names)} output={config.output_path}"
    )
    return to_jsonable(payload)


def _is_redline_failure(failure: Mapping[str, Any]) -> bool:
    gate = str(failure.get("gate", ""))
    return gate.startswith("redline") or gate in {"require_redline", "min_redline_auroc_margin"}


def _run_summary(name: str, path: Path) -> dict[str, Any]:
    payload = _load_workflow_report(path)
    records = _mapping(payload.get("records"))
    probe = _mapping(payload.get("probe"))
    redline = _mapping(payload.get("redline"))
    artifact_summary = _mapping(payload.get("artifact_manifest_summary"))
    redline_available = redline.get("available") is True
    redline_margin = _finite_float_or_none(redline.get("probe_vs_text_auroc_margin"))
    return {
        "name": name,
        "path": str(path),
        "status": payload.get("status"),
        "effective_model": payload.get("effective_model"),
        "records_reused": _nested(payload, "execution", "records_reused"),
        "record_count": _int_or_none(records.get("record_count")),
        "dataset": records.get("metadata_dataset"),
        "record_grain": records.get("metadata_record_grain"),
        "layers": records.get("metadata_layers"),
        "candidate_count": _int_or_none(probe.get("candidate_count")),
        "recommended_layer": _int_or_none(probe.get("recommended_layer")),
        "selection_metric": probe.get("selection_metric"),
        "selection_value": _finite_float_or_none(probe.get("selection_value")),
        "test_label_auroc": _finite_float_or_none(probe.get("test_label_auroc")),
        "test_target_bce": _finite_float_or_none(probe.get("test_target_bce")),
        "conformal_available": probe.get("conformal_available"),
        "conformal_threshold": _finite_float_or_none(probe.get("conformal_threshold")),
        "test_selective_accuracy": _finite_float_or_none(probe.get("test_selective_accuracy")),
        "test_selective_coverage": _finite_float_or_none(probe.get("test_selective_coverage")),
        "manifest_missing_count": _int_or_none(artifact_summary.get("missing_count")),
        "manifest_artifact_count": _int_or_none(artifact_summary.get("artifact_count")),
        "redline_available": redline_available,
        "redline_status": redline.get("status"),
        "redline_record_count": _int_or_none(redline.get("record_count")),
        "redline_best_signal": redline.get("best_text_signal"),
        "redline_best_direction": redline.get("best_text_direction"),
        "redline_best_auroc": _finite_float_or_none(redline.get("best_text_auroc")),
        "redline_margin": redline_margin,
    }


def _load_workflow_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"workflow report must be a JSON object: {path}")
    if payload.get("workflow") != "claim_factuality_probe_workflow":
        raise ValueError(f"not a claim_factuality_probe_workflow report: {path}")
    return dict(payload)


def _gate_failures(
    config: ClaimFactualityProbeWorkflowComparisonConfig,
    runs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    model_names = {str(run["effective_model"]) for run in runs if run.get("effective_model")}
    if len(model_names) < config.min_model_count:
        failures.append({
            "gate": "min_model_count",
            "observed": len(model_names),
            "threshold": config.min_model_count,
        })
    for run in runs:
        name = str(run.get("name"))
        if config.require_ready_status and run.get("status") != "ready":
            failures.append({"gate": "require_ready_status", "run": name, "observed": run.get("status")})
        if config.require_manifest_clean and _int_or_none(run.get("manifest_missing_count")) != 0:
            failures.append({
                "gate": "require_manifest_clean",
                "run": name,
                "observed": run.get("manifest_missing_count"),
                "threshold": 0,
            })
        record_count = _int_or_none(run.get("record_count"))
        if record_count is None or record_count < config.min_record_count:
            failures.append({
                "gate": "min_record_count",
                "run": name,
                "observed": record_count,
                "threshold": config.min_record_count,
            })
        if config.require_conformal and run.get("conformal_available") is not True:
            failures.append({"gate": "require_conformal", "run": name, "observed": run.get("conformal_available")})
        auroc = _finite_float_or_none(run.get("test_label_auroc"))
        if auroc is None or auroc < config.min_test_label_auroc:
            failures.append({
                "gate": "min_test_label_auroc",
                "run": name,
                "observed": auroc,
                "threshold": config.min_test_label_auroc,
            })
        if config.require_redline and run.get("redline_available") is not True:
            failures.append({"gate": "require_redline", "run": name, "observed": run.get("redline_available")})
        if run.get("redline_available") is True:
            margin = _finite_float_or_none(run.get("redline_margin"))
            if margin is None or margin < config.min_redline_auroc_margin:
                failures.append({
                    "gate": "min_redline_auroc_margin",
                    "run": name,
                    "observed": margin,
                    "threshold": config.min_redline_auroc_margin,
                    "probe_auroc": auroc,
                    "redline_auroc": run.get("redline_best_auroc"),
                })
    return failures


def _leaderboard(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        runs,
        key=lambda run: (
            _descending_sort_value(run.get("test_label_auroc")),
            _descending_sort_value(run.get("redline_margin")),
            str(run.get("name")),
        ),
    )
    return [
        {
            "rank": index,
            "name": run.get("name"),
            "effective_model": run.get("effective_model"),
            "record_count": run.get("record_count"),
            "recommended_layer": run.get("recommended_layer"),
            "test_label_auroc": run.get("test_label_auroc"),
            "test_selective_accuracy": run.get("test_selective_accuracy"),
            "test_selective_coverage": run.get("test_selective_coverage"),
            "conformal_threshold": run.get("conformal_threshold"),
            "redline_best_signal": run.get("redline_best_signal"),
            "redline_best_auroc": run.get("redline_best_auroc"),
            "redline_margin": run.get("redline_margin"),
        }
        for index, run in enumerate(ranked, start=1)
    ]


def _record_registry(
    config: ClaimFactualityProbeWorkflowComparisonConfig,
    *,
    report: Mapping[str, Any],
    manifest: Mapping[str, Any] | None,
) -> None:
    if config.registry_path is None or config.register_name is None:
        return
    metadata = {
        "workflow": report.get("workflow"),
        "status": report.get("status"),
        "run_count": len(tuple(report.get("runs", ()))),
        "model_count": _nested(report, "promotion_gate", "model_count"),
        "best_run": _nested(report, "promotion_gate", "best_run"),
        "redline_passed": _nested(report, "promotion_gate", "redline_passed"),
        "failure_count": len(tuple(_nested(report, "promotion_gate", "failures") or ())),
        "manifest_summary": None if manifest is None else manifest.get("summary"),
    }
    registry = ArtifactRegistry.load_json(config.registry_path)
    registry.record_report(
        name=config.register_name,
        path=config.output_path,
        version=config.register_version,
        metadata=metadata,
    )
    if manifest is not None:
        registry.record_benchmark_manifest(
            name=config.register_name,
            path=config.artifact_manifest_path,
            version=config.register_version,
            metadata=metadata,
        )
    registry.save_json()


def _parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return _safe_run_name(path.stem), path
    name, path = value.split("=", 1)
    return _safe_run_name(name), Path(path)


def _safe_run_name(value: Any) -> str:
    name = str(value).strip().casefold().replace("-", "_")
    name = "".join(char if char.isalnum() or char == "_" else "_" for char in name)
    name = "_".join(part for part in name.split("_") if part)
    if not name:
        raise ValueError("workflow report name must be non-empty.")
    return name


def _write_json(path: str | Path, payload: Mapping[str, Any], *, compact: bool) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    indent = None if compact else 2
    output.write_text(json.dumps(to_jsonable(payload), indent=indent, sort_keys=True) + "\n", encoding="utf-8")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _nested(payload: Any, *keys: str) -> Any:
    current = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _finite_float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _descending_sort_value(value: Any) -> float:
    parsed = _finite_float_or_none(value)
    return float("inf") if parsed is None else -parsed


def _config_from_args(args: argparse.Namespace) -> ClaimFactualityProbeWorkflowComparisonConfig:
    workflow_reports = {}
    for item in args.workflow_report or ():
        name, path = _parse_named_path(item)
        if name in workflow_reports:
            raise ValueError(f"duplicate workflow report name: {name}")
        workflow_reports[name] = path
    return ClaimFactualityProbeWorkflowComparisonConfig(
        workflow_reports=workflow_reports,
        output_path=args.json,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        register_name=args.register_name,
        register_version=args.register_version,
        min_model_count=args.min_model_count,
        min_record_count=args.min_record_count,
        min_test_label_auroc=args.min_test_label_auroc,
        min_redline_auroc_margin=args.min_redline_auroc_margin,
        require_ready_status=not bool(args.allow_non_ready),
        require_manifest_clean=not bool(args.allow_manifest_missing),
        require_conformal=not bool(args.allow_missing_conformal),
        require_redline=not bool(args.allow_missing_redline),
        compact_json=bool(args.compact_json),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare claim factuality probe workflow reports")
    parser.add_argument(
        "--workflow-report",
        action="append",
        required=True,
        help="NAME=PATH workflow report; repeatable",
    )
    parser.add_argument("--json", required=True, help="comparison report output path")
    parser.add_argument("--artifact-manifest", default=None, help="optional artifact-manifest path")
    parser.add_argument("--registry", default=None, help="optional local ArtifactRegistry JSON path")
    parser.add_argument("--register-name", default=None, help="optional registry report/manifest record name")
    parser.add_argument("--register-version", default="0.1")
    parser.add_argument("--min-model-count", type=int, default=2)
    parser.add_argument("--min-record-count", type=int, default=1)
    parser.add_argument("--min-test-label-auroc", type=float, default=0.5)
    parser.add_argument("--min-redline-auroc-margin", type=float, default=0.0)
    parser.add_argument("--allow-non-ready", action="store_true")
    parser.add_argument("--allow-manifest-missing", action="store_true")
    parser.add_argument("--allow-missing-conformal", action="store_true")
    parser.add_argument("--allow-missing-redline", action="store_true")
    parser.add_argument("--compact-json", action="store_true")
    compare_claim_factuality_probe_workflows(_config_from_args(parser.parse_args()))


if __name__ == "__main__":
    main()

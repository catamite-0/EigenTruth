"""Compare pre-generation probe workflow reports.

This script aggregates compact ``run_pre_generation_probe_workflow.py`` reports
without loading large hidden-state records or torch artifacts. It is intended as
the release-evidence handoff for small multi-model pre-generation probe runs:
source workflow reports stay local or manifest-backed, while this comparison
captures the gate decision and compact metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eigentruth.json_utils import to_jsonable  # noqa: E402
from eigentruth.registry import build_artifact_manifest  # noqa: E402


@dataclass(frozen=True)
class PreGenerationProbeWorkflowComparisonConfig:
    """Configuration for comparing compact pre-generation workflow reports."""

    workflow_reports: Mapping[str, str | Path]
    output_path: str | Path
    artifact_manifest_path: str | Path | None = None
    redline_reports: Mapping[str, str | Path] = field(default_factory=dict)
    min_model_count: int = 2
    min_record_count: int = 1
    min_test_label_auroc: float = 0.5
    min_redline_auroc_margin: float = 0.0
    require_ready_status: bool = True
    require_manifest_clean: bool = True
    require_conformal: bool = True
    require_redline: bool = False
    compact_json: bool = False

    def __post_init__(self) -> None:
        reports = {
            _safe_run_name(name): Path(path)
            for name, path in self.workflow_reports.items()
        }
        if not reports:
            raise ValueError("workflow_reports must not be empty.")
        if len(reports) != len(self.workflow_reports):
            raise ValueError("workflow report names must be unique after normalization.")
        object.__setattr__(self, "workflow_reports", reports)
        redline_reports = {
            _safe_run_name(name): Path(path)
            for name, path in (self.redline_reports or {}).items()
        }
        if len(redline_reports) != len(self.redline_reports or {}):
            raise ValueError("redline report names must be unique after normalization.")
        object.__setattr__(self, "redline_reports", redline_reports)
        object.__setattr__(self, "output_path", Path(self.output_path))
        if self.artifact_manifest_path is not None:
            object.__setattr__(self, "artifact_manifest_path", Path(self.artifact_manifest_path))
        object.__setattr__(self, "min_model_count", int(self.min_model_count))
        object.__setattr__(self, "min_record_count", int(self.min_record_count))
        object.__setattr__(self, "min_test_label_auroc", float(self.min_test_label_auroc))
        object.__setattr__(self, "min_redline_auroc_margin", float(self.min_redline_auroc_margin))
        object.__setattr__(self, "require_redline", bool(self.require_redline or redline_reports))
        if self.min_model_count < 1:
            raise ValueError("min_model_count must be >=1.")
        if self.min_record_count < 1:
            raise ValueError("min_record_count must be >=1.")
        if not (0.0 <= self.min_test_label_auroc <= 1.0):
            raise ValueError("min_test_label_auroc must be in [0, 1].")
        if self.min_redline_auroc_margin < 0.0:
            raise ValueError("min_redline_auroc_margin must be >=0.")


def compare_pre_generation_probe_workflows(
    config: PreGenerationProbeWorkflowComparisonConfig,
) -> dict[str, Any]:
    """Load workflow reports, summarize comparable metrics, and apply gates."""
    runs = [
        _run_summary(name, path, redline_path=config.redline_reports.get(name))
        for name, path in sorted(config.workflow_reports.items())
    ]
    failures = _gate_failures(config, runs)
    status = "ready" if not failures else "blocked"
    model_names = sorted({str(run["effective_model"]) for run in runs if run.get("effective_model")})
    payload = {
        "schema_version": 1,
        "workflow": "pre_generation_probe_workflow_comparison",
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
            "redline_run_count": sum(1 for run in runs if run.get("redline")),
            "redline_passed": not any(_is_redline_failure(failure) for failure in failures),
        },
        "runs": runs,
        "leaderboard": _leaderboard(runs),
        "evidence_scope": {
            "claim": "compact multi-run pre-generation probe workflow comparison",
            "not_a_claim": "full detector-quality superiority on open generation",
            "notes": (
                "This report compares saved workflow artifacts. Treat small TruthfulQA limit runs as plumbing and "
                "early-signal evidence until replicated at larger scale and against external baselines."
            ),
        },
        "paths": {
            "report": str(config.output_path),
            "artifact_manifest": None if config.artifact_manifest_path is None else str(config.artifact_manifest_path),
            "workflow_reports": {name: str(path) for name, path in config.workflow_reports.items()},
            "redline_reports": {name: str(path) for name, path in config.redline_reports.items()},
        },
    }
    _write_json(config.output_path, payload, compact=config.compact_json)
    if config.artifact_manifest_path is not None:
        manifest = build_artifact_manifest(
            {
                "comparison_report": config.output_path,
                **{f"workflow_report.{name}": path for name, path in config.workflow_reports.items()},
                **{f"redline_report.{name}": path for name, path in config.redline_reports.items()},
            },
            root=Path(config.artifact_manifest_path).parent,
            metadata={
                "workflow": "pre_generation_probe_workflow_comparison",
                "status": status,
                "run_count": len(runs),
                "model_count": len(model_names),
            },
        )
        _write_json(config.artifact_manifest_path, manifest, compact=False)
        payload["artifact_manifest_summary"] = manifest["summary"]
        _write_json(config.output_path, payload, compact=config.compact_json)
    print(
        "pre_generation_probe_workflow_comparison_ok "
        f"status={status} runs={len(runs)} models={len(model_names)} output={config.output_path}"
    )
    return to_jsonable(payload)


def _is_redline_failure(failure: Mapping[str, Any]) -> bool:
    gate = str(failure.get("gate", ""))
    return gate.startswith("redline") or gate in {"require_redline", "min_redline_auroc_margin"}


def _run_summary(name: str, path: Path, *, redline_path: Path | None = None) -> dict[str, Any]:
    payload = _load_workflow_report(path)
    records = _mapping(payload.get("records"))
    probe = _mapping(payload.get("probe"))
    artifact_summary = _mapping(payload.get("artifact_manifest_summary"))
    redline = None if redline_path is None else _redline_summary(redline_path)
    test_label_auroc = _finite_float_or_none(probe.get("test_label_auroc"))
    redline_auroc = _finite_float_or_none(_nested(redline, "best_signal", "auroc"))
    redline_margin = (
        None
        if test_label_auroc is None or redline_auroc is None
        else test_label_auroc - redline_auroc
    )
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
        "test_label_auroc": test_label_auroc,
        "test_target_bce": _finite_float_or_none(probe.get("test_target_bce")),
        "conformal_available": probe.get("conformal_available"),
        "conformal_threshold": _finite_float_or_none(probe.get("conformal_threshold")),
        "test_selective_accuracy": _finite_float_or_none(probe.get("test_selective_accuracy")),
        "test_selective_coverage": _finite_float_or_none(probe.get("test_selective_coverage")),
        "manifest_missing_count": _int_or_none(artifact_summary.get("missing_count")),
        "manifest_artifact_count": _int_or_none(artifact_summary.get("artifact_count")),
        "redline": redline,
        "redline_margin": redline_margin,
    }


def _load_workflow_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"workflow report must be a JSON object: {path}")
    if payload.get("workflow") != "pre_generation_probe_workflow":
        raise ValueError(f"not a pre_generation_probe_workflow report: {path}")
    return dict(payload)


def _redline_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"redline report must be a JSON object: {path}")
    if payload.get("workflow") != "pre_generation_text_baseline_eval":
        raise ValueError(f"not a pre_generation_text_baseline_eval report: {path}")
    best = _mapping(payload.get("best_signal"))
    return {
        "path": str(path),
        "status": payload.get("status"),
        "record_count": _int_or_none(payload.get("record_count")),
        "best_signal": {
            "name": best.get("name"),
            "direction": best.get("direction"),
            "auroc": _finite_float_or_none(best.get("auroc")),
        },
    }


def _gate_failures(
    config: PreGenerationProbeWorkflowComparisonConfig,
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
        redline = _mapping(run.get("redline"))
        if config.require_redline and not redline:
            failures.append({"gate": "require_redline", "run": name, "observed": None})
        if redline:
            redline_status = redline.get("status")
            if redline_status != "ready":
                failures.append({"gate": "redline_ready_status", "run": name, "observed": redline_status})
            margin = _finite_float_or_none(run.get("redline_margin"))
            if margin is None or margin < config.min_redline_auroc_margin:
                failures.append({
                    "gate": "min_redline_auroc_margin",
                    "run": name,
                    "observed": margin,
                    "threshold": config.min_redline_auroc_margin,
                    "probe_auroc": auroc,
                    "redline_auroc": _nested(redline, "best_signal", "auroc"),
                })
    return failures


def _leaderboard(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        runs,
        key=lambda run: (
            -(_finite_float_or_none(run.get("test_label_auroc")) or float("-inf")),
            str(run.get("name")),
        ),
    )
    return [
        {
            "rank": index,
            "name": run.get("name"),
            "effective_model": run.get("effective_model"),
            "recommended_layer": run.get("recommended_layer"),
            "test_label_auroc": run.get("test_label_auroc"),
            "test_selective_accuracy": run.get("test_selective_accuracy"),
            "test_selective_coverage": run.get("test_selective_coverage"),
            "conformal_threshold": run.get("conformal_threshold"),
            "redline_best_signal": _nested(run, "redline", "best_signal", "name"),
            "redline_best_auroc": _nested(run, "redline", "best_signal", "auroc"),
            "redline_margin": run.get("redline_margin"),
        }
        for index, run in enumerate(ranked, start=1)
    ]


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


def _config_from_args(args: argparse.Namespace) -> PreGenerationProbeWorkflowComparisonConfig:
    workflow_reports = {}
    for item in args.workflow_report or ():
        name, path = _parse_named_path(item)
        if name in workflow_reports:
            raise ValueError(f"duplicate workflow report name: {name}")
        workflow_reports[name] = path
    redline_reports = {}
    for item in args.redline_report or ():
        name, path = _parse_named_path(item)
        if name in redline_reports:
            raise ValueError(f"duplicate redline report name: {name}")
        redline_reports[name] = path
    return PreGenerationProbeWorkflowComparisonConfig(
        workflow_reports=workflow_reports,
        redline_reports=redline_reports,
        output_path=args.json,
        artifact_manifest_path=args.artifact_manifest,
        min_model_count=args.min_model_count,
        min_record_count=args.min_record_count,
        min_test_label_auroc=args.min_test_label_auroc,
        min_redline_auroc_margin=args.min_redline_auroc_margin,
        require_ready_status=not bool(args.allow_non_ready),
        require_manifest_clean=not bool(args.allow_manifest_missing),
        require_conformal=not bool(args.allow_missing_conformal),
        require_redline=bool(redline_reports) and not bool(args.allow_missing_redline),
        compact_json=bool(args.compact_json),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare pre-generation probe workflow reports")
    parser.add_argument(
        "--workflow-report",
        action="append",
        required=True,
        help="NAME=PATH workflow report; repeatable",
    )
    parser.add_argument(
        "--redline-report",
        action="append",
        default=None,
        help="NAME=PATH text redline report; repeatable",
    )
    parser.add_argument("--json", required=True, help="comparison report output path")
    parser.add_argument("--artifact-manifest", default=None, help="optional artifact-manifest path")
    parser.add_argument("--min-model-count", type=int, default=2)
    parser.add_argument("--min-record-count", type=int, default=1)
    parser.add_argument("--min-test-label-auroc", type=float, default=0.5)
    parser.add_argument("--min-redline-auroc-margin", type=float, default=0.0)
    parser.add_argument("--allow-non-ready", action="store_true")
    parser.add_argument("--allow-manifest-missing", action="store_true")
    parser.add_argument("--allow-missing-conformal", action="store_true")
    parser.add_argument("--allow-missing-redline", action="store_true")
    parser.add_argument("--compact-json", action="store_true")
    compare_pre_generation_probe_workflows(_config_from_args(parser.parse_args()))


if __name__ == "__main__":
    main()

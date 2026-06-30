"""Run context-sensitivity logprob extraction through calibrated score dumps."""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, MutableMapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.build_context_sensitivity_logprob_pairs import (  # noqa: E402
    DEFAULT_BASELINE_TEMPLATE,
    DEFAULT_CONTEXT_TEMPLATE,
    CompletionLogprobScorer,
    HFCompletionLogprobScorer,
)
from benchmarks.build_context_sensitivity_logprob_pairs import build_report as build_paired_logprob_report  # noqa: E402
from benchmarks.build_verifier_signal_score_dump import CONTEXT_SENSITIVITY_SIGNALS  # noqa: E402
from benchmarks.build_verifier_signal_score_dump import build_report as build_signal_score_dump_report  # noqa: E402
from benchmarks.enrich_context_sensitivity_sidecar import build_report as enrich_sidecar_report  # noqa: E402
from eigentruth.eval.score_dump import ScoreDump, load_score_dump  # noqa: E402
from eigentruth.registry import (  # noqa: E402
    ArtifactRegistry,
    ArtifactVerificationContext,
    build_artifact_manifest,
)


@dataclass(frozen=True)
class ContextSensitivityWorkflowConfig:
    """Configuration for the context-sensitivity evidence workflow."""

    scores_path: Path
    verified_records_jsonl: Path
    output_dir: Path
    run_name: str | None = None
    model_id: str | None = None
    model_revision: str | None = None
    device: str = "auto"
    dtype: str = "float32"
    max_length: int | None = None
    attn_implementation: str | None = None
    trust_remote_code: bool = False
    baseline_template: str = DEFAULT_BASELINE_TEMPLATE
    context_template: str = DEFAULT_CONTEXT_TEMPLATE
    require_evidence: bool = False
    limit: int | None = None
    ratio_threshold: float = 1.25
    shift_threshold: float = 0.25
    min_abs_delta: float = 0.0
    keep_signals: Sequence[str] | None = None
    verifier_signals: Sequence[str] = CONTEXT_SENSITIVITY_SIGNALS
    output_format: str = "jsonl"
    compact_json: bool = False
    verify_manifest: bool = True
    registry_path: Path | None = None
    registry_name: str | None = None
    registry_version: str = "0.1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "scores_path", Path(self.scores_path))
        object.__setattr__(self, "verified_records_jsonl", Path(self.verified_records_jsonl))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.registry_path is not None:
            object.__setattr__(self, "registry_path", Path(self.registry_path))
        if self.keep_signals is not None:
            object.__setattr__(self, "keep_signals", _clean_names(self.keep_signals, name="keep_signals"))
        object.__setattr__(self, "verifier_signals", _clean_names(self.verifier_signals, name="verifier_signals"))
        if self.output_format not in {"json", "jsonl"}:
            raise ValueError("output_format must be 'json' or 'jsonl'.")
        if self.limit is not None:
            if isinstance(self.limit, bool):
                raise ValueError("limit must be positive when set.")
            limit = int(self.limit)
            if limit <= 0:
                raise ValueError("limit must be positive when set.")
            object.__setattr__(self, "limit", limit)
        if not self.verified_records_jsonl.exists():
            raise ValueError(f"verified_records_jsonl does not exist: {self.verified_records_jsonl}")
        if not self.scores_path.exists():
            raise ValueError(f"scores_path does not exist: {self.scores_path}")
        if bool(self.registry_name) != bool(self.registry_path):
            raise ValueError("registry_name and registry_path must be supplied together.")

    @property
    def paired_logprobs_path(self) -> Path:
        return self.output_dir / "paired-context-logprobs.jsonl"

    @property
    def paired_logprobs_report_path(self) -> Path:
        return self.output_dir / "paired-context-logprobs-report.json"

    @property
    def enriched_records_path(self) -> Path:
        return self.output_dir / "verified-records-context.jsonl"

    @property
    def enrichment_report_path(self) -> Path:
        return self.output_dir / "context-sensitivity-sidecar-report.json"

    @property
    def enhanced_score_dump_path(self) -> Path:
        return self.output_dir / "context-sensitivity-enhanced-scores.manifest.json"

    @property
    def enhanced_score_report_path(self) -> Path:
        return self.output_dir / "context-sensitivity-enhanced-score-report.json"

    @property
    def artifact_manifest_path(self) -> Path:
        return self.output_dir / "artifact-manifest.json"

    @property
    def manifest_verification_path(self) -> Path:
        return self.output_dir / "manifest-verification.json"

    @property
    def workflow_report_path(self) -> Path:
        return self.output_dir / "context-sensitivity-workflow.json"


@dataclass(frozen=True)
class _WorkflowInputs:
    scores_path: Path
    verified_records_jsonl: Path
    limited: bool = False
    normalized_records: bool = False


def run_context_sensitivity_workflow(
    config: ContextSensitivityWorkflowConfig,
    *,
    scorer: CompletionLogprobScorer | None = None,
) -> dict[str, Any]:
    """Run paired-logprob extraction, sidecar enrichment, and score-dump conversion."""
    started = time.perf_counter()
    profile: dict[str, float] = {}
    config.output_dir.mkdir(parents=True, exist_ok=True)
    with _profile_phase(profile, "prepare_inputs"):
        workflow_inputs = _prepare_workflow_inputs(config)
    resolved_scorer = scorer
    if resolved_scorer is None:
        if config.model_id is None:
            raise ValueError("model_id is required when no scorer is injected.")
        with _profile_phase(profile, "load_model"):
            resolved_scorer = HFCompletionLogprobScorer.from_pretrained(
                config.model_id,
                device=config.device,
                dtype=config.dtype,
                revision=config.model_revision,
                trust_remote_code=bool(config.trust_remote_code),
                attn_implementation=config.attn_implementation,
                max_length=config.max_length,
            )

    with _profile_phase(profile, "build_paired_logprobs"):
        paired_report = build_paired_logprob_report(
            records_path=workflow_inputs.verified_records_jsonl,
            output=config.paired_logprobs_path,
            scorer=resolved_scorer,
            model_id=config.model_id,
            run_name=config.run_name,
            baseline_template=config.baseline_template,
            context_template=config.context_template,
            require_evidence=bool(config.require_evidence),
            limit=None,
        )
        _write_json(config.paired_logprobs_report_path, paired_report, compact=config.compact_json)

    with _profile_phase(profile, "enrich_verified_records"):
        enrichment_report = enrich_sidecar_report(
            verified_records_jsonl=workflow_inputs.verified_records_jsonl,
            paired_logprobs=config.paired_logprobs_path,
            output=config.enriched_records_path,
            run_name=config.run_name,
            ratio_threshold=float(config.ratio_threshold),
            shift_threshold=float(config.shift_threshold),
            min_abs_delta=float(config.min_abs_delta),
            allow_missing=False,
            overwrite=False,
        )
        _write_json(config.enrichment_report_path, enrichment_report, compact=config.compact_json)

    with _profile_phase(profile, "build_context_sensitivity_score_dump"):
        enhanced_report = build_signal_score_dump_report(
            input_scores=workflow_inputs.scores_path,
            verified_records_jsonl=config.enriched_records_path,
            output=config.enhanced_score_dump_path,
            output_format=config.output_format,
            run_name=config.run_name,
            keep_signals=config.keep_signals,
            verifier_signals=config.verifier_signals,
        )
        _write_json(config.enhanced_score_report_path, enhanced_report, compact=config.compact_json)

    with _profile_phase(profile, "write_artifact_manifest"):
        manifest = _write_artifact_manifest(
            config,
            workflow_inputs=workflow_inputs,
            paired_report=paired_report,
            enrichment_report=enrichment_report,
            enhanced_report=enhanced_report,
            profile=profile,
            total_seconds=time.perf_counter() - started,
        )

    manifest_verification = None
    if config.verify_manifest:
        with _profile_phase(profile, "verify_artifact_manifest"):
            context = ArtifactVerificationContext()
            manifest_verification = context.load_and_verify_artifact_manifest(
                config.artifact_manifest_path,
                root=config.output_dir,
            ).to_dict()
            _write_json(config.manifest_verification_path, manifest_verification, compact=False)

    signal_summary = _context_signal_summary(config.enhanced_score_dump_path, config.verifier_signals)
    profile["total_seconds"] = time.perf_counter() - started
    payload = {
        "schema_version": 1,
        "workflow": "context_sensitivity_workflow",
        "config": _config_payload(config),
        "paths": {
            "scores": str(config.scores_path),
            "verified_records_jsonl": str(config.verified_records_jsonl),
            "workflow_scores": str(workflow_inputs.scores_path),
            "workflow_verified_records_jsonl": str(workflow_inputs.verified_records_jsonl),
            "paired_logprobs": str(config.paired_logprobs_path),
            "paired_logprobs_report": str(config.paired_logprobs_report_path),
            "enriched_records": str(config.enriched_records_path),
            "enrichment_report": str(config.enrichment_report_path),
            "enhanced_score_dump": str(config.enhanced_score_dump_path),
            "enhanced_score_report": str(config.enhanced_score_report_path),
            "artifact_manifest": str(config.artifact_manifest_path),
            "manifest_verification": None if manifest_verification is None else str(config.manifest_verification_path),
        },
        "paired_summary": _paired_summary(paired_report),
        "limited_input": workflow_inputs.limited,
        "normalized_verified_records": workflow_inputs.normalized_records,
        "enriched_record_count": enrichment_report.get("enriched_record_count"),
        "enrichment_summary": dict(enrichment_report.get("summary", {})),
        "enhanced_score_summary": dict(enhanced_report.get("summary", {})),
        "signal_summary": signal_summary,
        "manifest_summary": manifest.get("summary"),
        "manifest_verification": manifest_verification,
        "profile": dict(profile),
    }
    registry_record_key = _record_registry(config, payload)
    if registry_record_key is not None:
        payload["registry_record_key"] = registry_record_key
    _write_json(config.workflow_report_path, payload, compact=config.compact_json)
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    """CLI entrypoint helper."""
    config = ContextSensitivityWorkflowConfig(
        scores_path=Path(args.scores),
        verified_records_jsonl=Path(args.verified_records_jsonl),
        output_dir=Path(args.output_dir),
        run_name=args.run_name,
        model_id=args.model_id,
        model_revision=args.model_revision,
        device=args.device,
        dtype=args.dtype,
        max_length=args.max_length,
        attn_implementation=args.attn_implementation,
        trust_remote_code=bool(args.trust_remote_code),
        baseline_template=args.baseline_template,
        context_template=args.context_template,
        require_evidence=bool(args.require_evidence),
        limit=args.limit,
        ratio_threshold=args.ratio_threshold,
        shift_threshold=args.shift_threshold,
        min_abs_delta=args.min_abs_delta,
        keep_signals=_parse_csv(args.keep_signals, name="keep_signals"),
        verifier_signals=_parse_csv(args.verifier_signals, name="verifier_signals") or CONTEXT_SENSITIVITY_SIGNALS,
        output_format=args.output_format,
        compact_json=bool(args.compact_json),
        verify_manifest=not bool(args.no_verify_manifest),
        registry_path=None if args.registry_path is None else Path(args.registry_path),
        registry_name=args.registry_name,
        registry_version=args.registry_version,
    )
    payload = run_context_sensitivity_workflow(config)
    print(
        "context_sensitivity_workflow_ok "
        f"records={payload['paired_summary']['paired_logprob_record_count']} "
        f"max_flagged_rate={payload['enrichment_summary'].get('max_flagged_rate')}"
    )
    return payload


def _write_artifact_manifest(
    config: ContextSensitivityWorkflowConfig,
    *,
    workflow_inputs: _WorkflowInputs,
    paired_report: Mapping[str, Any],
    enrichment_report: Mapping[str, Any],
    enhanced_report: Mapping[str, Any],
    profile: Mapping[str, float],
    total_seconds: float,
) -> dict[str, Any]:
    artifacts: dict[str, str | Path] = {
        "source_scores": config.scores_path,
        "source_verified_records": config.verified_records_jsonl,
        "paired_logprobs": config.paired_logprobs_path,
        "paired_logprobs_report": config.paired_logprobs_report_path,
        "enriched_verified_records": config.enriched_records_path,
        "enrichment_report": config.enrichment_report_path,
        "enhanced_score_dump": config.enhanced_score_dump_path,
        "enhanced_score_report": config.enhanced_score_report_path,
    }
    if workflow_inputs.limited:
        artifacts["workflow_scores"] = workflow_inputs.scores_path
    if workflow_inputs.limited or workflow_inputs.normalized_records:
        artifacts["workflow_verified_records"] = workflow_inputs.verified_records_jsonl
    if config.output_format == "jsonl":
        artifacts["enhanced_score_records"] = config.enhanced_score_dump_path.with_suffix(".records.jsonl")
    metadata = {
        "runner": "run_context_sensitivity_workflow",
        "workflow": "context_sensitivity_workflow",
        "run_name": config.run_name,
        "model_id": config.model_id,
        "limited_input": workflow_inputs.limited,
        "normalized_verified_records": workflow_inputs.normalized_records,
        "limit": config.limit,
        "paired_logprob_record_count": paired_report.get("paired_logprob_record_count"),
        "missing_evidence_count": paired_report.get("missing_evidence_count"),
        "enriched_record_count": enrichment_report.get("enriched_record_count"),
        "context_sensitivity_summary": enrichment_report.get("summary"),
        "enhanced_score_signal_count": len(enhanced_report.get("signals", ())),
        "context_sensitivity_signals": list(config.verifier_signals),
        "profile": dict(profile),
        "total_seconds": float(total_seconds),
    }
    manifest = build_artifact_manifest(
        artifacts,
        root=config.output_dir,
        metadata=metadata,
    )
    _write_json(config.artifact_manifest_path, manifest, compact=False)
    return manifest


def _record_registry(config: ContextSensitivityWorkflowConfig, payload: Mapping[str, Any]) -> str | None:
    if config.registry_path is None or config.registry_name is None:
        return None
    metadata = {
        "workflow": "context_sensitivity_workflow",
        "run_name": config.run_name,
        "model_id": config.model_id,
        "limited_input": payload.get("limited_input"),
        "limit": config.limit,
        "paired_logprob_record_count": payload["paired_summary"]["paired_logprob_record_count"],
        "enriched_record_count": payload.get("enriched_record_count"),
        "max_flagged_rate": payload["enrichment_summary"].get("max_flagged_rate"),
        "max_context_sensitivity_ratio": payload["enrichment_summary"].get("max_context_sensitivity_ratio"),
        "manifest_verified": None
        if payload.get("manifest_verification") is None
        else payload["manifest_verification"].get("passed"),
    }
    registry = ArtifactRegistry.load_json(config.registry_path).record_report(
        name=config.registry_name,
        path=config.workflow_report_path,
        version=config.registry_version,
        metadata=metadata,
    )
    registry.save_json()
    return f"report:{config.registry_name}:{config.registry_version}"


def _prepare_workflow_inputs(config: ContextSensitivityWorkflowConfig) -> _WorkflowInputs:
    if config.limit is None:
        records = _load_verified_records_jsonl(config.verified_records_jsonl)
        normalized_records = _normalize_verified_record_runs(records, run_name=config.run_name)
        if normalized_records != records:
            normalized_records_path = config.output_dir / "workflow-verified-records.jsonl"
            _write_jsonl(normalized_records_path, normalized_records)
            return _WorkflowInputs(
                scores_path=config.scores_path,
                verified_records_jsonl=normalized_records_path,
                limited=False,
                normalized_records=True,
            )
        return _WorkflowInputs(
            scores_path=config.scores_path,
            verified_records_jsonl=config.verified_records_jsonl,
            limited=False,
            normalized_records=False,
        )

    score_dump = load_score_dump(config.scores_path, allow_missing_scores=False)
    limit = int(config.limit)
    if limit > score_dump.n_total:
        raise ValueError(f"limit ({limit}) exceeds score dump record count ({score_dump.n_total}).")

    selected_records = _select_verified_records_for_limit(
        config.verified_records_jsonl,
        run_name=config.run_name,
        limit=limit,
        labels=score_dump.labels[:limit],
    )
    limited_scores_path = config.output_dir / "limited-scores.json"
    limited_records_path = config.output_dir / "limited-verified-records.jsonl"
    limited_dump = ScoreDump(
        labels=score_dump.labels[:limit],
        scores={
            name: tuple(values[:limit])
            for name, values in score_dump.scores.items()
        },
        config={
            **dict(score_dump.config),
            "context_sensitivity_workflow_limit": limit,
            "context_sensitivity_workflow_source_scores_path": str(config.scores_path),
        },
        sweep_scores={
            str(layer): {
                name: tuple(values[:limit])
                for name, values in layer_scores.items()
            }
            for layer, layer_scores in score_dump.sweep_scores.items()
        },
        statements=score_dump.statements[:limit] if score_dump.statements else (),
        extras={
            **dict(score_dump.extras),
            "context_sensitivity_workflow_source_scores_path": str(config.scores_path),
            "context_sensitivity_workflow_limited_from_n_total": score_dump.n_total,
            "context_sensitivity_workflow_limit": limit,
        },
    )
    limited_scores_path.write_text(
        json.dumps(limited_dump.to_mapping(), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_jsonl(limited_records_path, selected_records)
    return _WorkflowInputs(
        scores_path=limited_scores_path,
        verified_records_jsonl=limited_records_path,
        limited=True,
        normalized_records=True,
    )


def _select_verified_records_for_limit(
    path: Path,
    *,
    run_name: str | None,
    limit: int,
    labels: Sequence[int],
) -> tuple[dict[str, Any], ...]:
    records = _load_verified_records_jsonl(path)
    runs = {str(record.get("run", "")).strip() for record in records if str(record.get("run", "")).strip()}
    if run_name is None:
        if len(runs) > 1:
            raise ValueError("verified-record sidecar contains multiple runs; pass --run-name when using --limit.")
        resolved_run_name = next(iter(runs)) if runs else "default"
    else:
        resolved_run_name = str(run_name)

    selected = []
    for record in records:
        record_run = str(record.get("run", "")).strip()
        if record_run not in {"", resolved_run_name}:
            continue
        output_record = dict(record)
        if not record_run:
            output_record["run"] = resolved_run_name
        selected.append(output_record)
    if len(selected) < limit:
        raise ValueError(
            f"verified-record sidecar has only {len(selected)} records for run {resolved_run_name!r}; "
            f"limit requires {limit}."
        )
    selected.sort(key=_verified_record_index)
    sliced = tuple(selected[:limit])
    for expected_index, (record, label) in enumerate(zip(sliced, labels, strict=True)):
        record_index = _verified_record_index(record)
        if record_index != expected_index:
            raise ValueError("limited verified records must have contiguous record_index values starting at 0.")
        if int(record.get("label")) != int(label):
            raise ValueError(f"verified record {record_index} label does not match limited score dump label.")
    return sliced


def _normalize_verified_record_runs(
    records: Sequence[Mapping[str, Any]],
    *,
    run_name: str | None,
) -> tuple[dict[str, Any], ...]:
    runs = {str(record.get("run", "")).strip() for record in records if str(record.get("run", "")).strip()}
    default_run = str(run_name) if run_name is not None else (next(iter(runs)) if len(runs) == 1 else "default")
    normalized = []
    for record in records:
        output_record = dict(record)
        if not str(output_record.get("run", "")).strip():
            output_record["run"] = default_run
        normalized.append(output_record)
    return tuple(normalized)


def _load_verified_records_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
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


def _verified_record_index(record: Mapping[str, Any]) -> int:
    value = record.get("record_index")
    if isinstance(value, bool) or value is None:
        raise ValueError("verified record_index must be an integer.")
    try:
        index = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("verified record_index must be an integer.") from exc
    if index < 0:
        raise ValueError("verified record_index must be non-negative.")
    return index


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def _context_signal_summary(score_dump_path: Path, signals: Sequence[str]) -> dict[str, dict[str, float]]:
    dump = load_score_dump(score_dump_path, required_scores=tuple(signals))
    summary = {}
    for signal in signals:
        values = tuple(float(value) for value in dump.scores[signal])
        summary[signal] = {
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "positive_count": sum(1 for value in values if value > 0.0),
        }
    return summary


def _paired_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "input_record_count": report.get("input_record_count"),
        "prepared_record_count": report.get("prepared_record_count"),
        "paired_logprob_record_count": report.get("paired_logprob_record_count"),
        "missing_evidence_count": report.get("missing_evidence_count"),
        "total_token_count": report.get("total_token_count"),
        "mean_token_count": report.get("mean_token_count"),
        "max_token_count": report.get("max_token_count"),
    }


def _config_payload(config: ContextSensitivityWorkflowConfig) -> dict[str, Any]:
    return {
        "scores_path": str(config.scores_path),
        "verified_records_jsonl": str(config.verified_records_jsonl),
        "run_name": config.run_name,
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "device": config.device,
        "dtype": config.dtype,
        "max_length": config.max_length,
        "attn_implementation": config.attn_implementation,
        "trust_remote_code": bool(config.trust_remote_code),
        "baseline_template": config.baseline_template,
        "context_template": config.context_template,
        "require_evidence": bool(config.require_evidence),
        "limit": config.limit,
        "ratio_threshold": float(config.ratio_threshold),
        "shift_threshold": float(config.shift_threshold),
        "min_abs_delta": float(config.min_abs_delta),
        "keep_signals": None if config.keep_signals is None else list(config.keep_signals),
        "verifier_signals": list(config.verifier_signals),
        "output_format": config.output_format,
        "verify_manifest": bool(config.verify_manifest),
        "registry_path": None if config.registry_path is None else str(config.registry_path),
        "registry_name": config.registry_name,
        "registry_version": config.registry_version,
    }


def _parse_csv(value: str | None, *, name: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    return _clean_names((part.strip() for part in value.split(",") if part.strip()), name=name)


def _clean_names(values: Sequence[str] | Iterator[str], *, name: str) -> tuple[str, ...]:
    names = tuple(str(value).strip() for value in values if str(value).strip())
    if not names:
        raise ValueError(f"{name} must contain at least one value.")
    if len(set(names)) != len(names):
        raise ValueError(f"{name} must contain unique values.")
    return names


@contextmanager
def _profile_phase(profile: MutableMapping[str, float], name: str) -> Iterator[None]:
    started = time.perf_counter()
    try:
        yield
    finally:
        profile[name] = profile.get(name, 0.0) + (time.perf_counter() - started)


def _write_json(path: Path, payload: Mapping[str, Any], *, compact: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        return
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run paired context-sensitivity logprob extraction and score-dump conversion."
    )
    parser.add_argument("--scores", required=True, help="source score dump JSON or JSONL manifest")
    parser.add_argument("--verified-records-jsonl", required=True, help="input verifier verified-record JSONL")
    parser.add_argument("--output-dir", required=True, help="directory for workflow artifacts")
    parser.add_argument("--model-id", required=True, help="Hugging Face causal LM id or local path")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--baseline-template", default=DEFAULT_BASELINE_TEMPLATE)
    parser.add_argument("--context-template", default=DEFAULT_CONTEXT_TEMPLATE)
    parser.add_argument("--require-evidence", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ratio-threshold", type=float, default=1.25)
    parser.add_argument("--shift-threshold", type=float, default=0.25)
    parser.add_argument("--min-abs-delta", type=float, default=0.0)
    parser.add_argument("--keep-signals", default=None)
    parser.add_argument("--verifier-signals", default=",".join(CONTEXT_SENSITIVITY_SIGNALS))
    parser.add_argument("--output-format", choices=("json", "jsonl"), default="jsonl")
    parser.add_argument("--compact-json", action="store_true")
    parser.add_argument("--no-verify-manifest", action="store_true")
    parser.add_argument("--registry-path", default=None)
    parser.add_argument("--registry-name", default=None)
    parser.add_argument("--registry-version", default="0.1")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

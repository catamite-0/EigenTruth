"""Build, promote, and optionally register a local retrieval route baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.build_evidence_fixture import (  # noqa: E402
    RETRIEVER_BACKENDS,
    build_evidence_fixture,
    load_corpus,
    load_score_dump,
)
from benchmarks.eval_verifier_ensemble import build_verifier_ensemble_report  # noqa: E402
from benchmarks.promote_artifact_manifest import promote_artifact_manifest  # noqa: E402
from benchmarks.run_adapter_promotion_workflow import (  # noqa: E402
    AdapterPromotionWorkflowConfig,
    run_adapter_promotion_workflow,
)
from benchmarks.runtime_budget_policy import (  # noqa: E402
    RuntimeBudgetPolicy,
    evaluate_runtime_budget,
    runtime_metrics_from_profile,
)
from eigentruth.registry import build_artifact_manifest, fingerprint_path  # noqa: E402


@dataclass(frozen=True)
class LocalRetrievalRouteWorkflowConfig:
    """Configuration for a local retrieval evidence route workflow."""

    scores_path: Path
    corpus_paths: Sequence[Path]
    output_dir: Path
    registry_path: Path | None = None
    name: str | None = None
    version: str | None = None
    score_name: str = "retrieval"
    signal: str = "truth_proj"
    direction: str | None = None
    alpha: float = 0.10
    repeats: int = 1
    seed: int = 0
    query_field: str = "answer"
    retriever_backend: str = "memory"
    retriever_index_path: Path | None = None
    verifier_min_overlap: float = 0.65
    retriever_min_overlap: float = 0.20
    retrieval_limit: int = 5
    min_selected: int = 1
    gate_routes: Sequence[str] = ("retrieval_groundedness",)
    gate_min_selected: int | None = None
    min_decision_accuracy: float | None = None
    max_false_supported_rate: float | None = None
    min_false_refuted_rate: float | None = None
    max_verified_false_alarm: float | None = None
    min_verified_detection: float | None = None
    max_mean_duration_seconds: float | None = None
    max_p95_duration_seconds: float | None = None
    max_p99_duration_seconds: float | None = None
    max_max_duration_seconds: float | None = None
    max_mean_attempted_route_count: float | None = None
    max_retrieval_use_rate: float | None = None
    min_cache_hit_rate: float | None = None
    max_runtime_total_seconds: float | None = None
    max_retrieval_hit_count: float | None = None
    min_claims_cache_hit_rate: float | None = None
    min_verifier_trace_cache_hit_rate: float | None = None
    claims_path: Path | None = None
    verifier_report_path: Path | None = None
    route_report_path: Path | None = None
    promotion_report_path: Path | None = None
    artifact_manifest_path: Path | None = None
    verification_report_path: Path | None = None
    workflow_report_path: Path | None = None
    claims_cache_dir: Path | None = None
    verifier_trace_cache_dir: Path | None = None
    compact_json: bool = False
    allow_non_promote: bool = False
    allow_promotion_failures: bool = False
    promotion_metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "scores_path", Path(self.scores_path))
        object.__setattr__(self, "corpus_paths", tuple(Path(path) for path in self.corpus_paths))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        for attr in (
            "registry_path",
            "claims_path",
            "verifier_report_path",
            "route_report_path",
            "promotion_report_path",
            "artifact_manifest_path",
            "verification_report_path",
            "workflow_report_path",
            "claims_cache_dir",
            "retriever_index_path",
            "verifier_trace_cache_dir",
        ):
            value = getattr(self, attr)
            if value is not None:
                object.__setattr__(self, attr, Path(value))
        object.__setattr__(self, "gate_routes", tuple(str(route) for route in self.gate_routes))
        if not self.corpus_paths:
            raise ValueError("corpus_paths must contain at least one local corpus path.")
        if not (0.0 < float(self.alpha) < 1.0):
            raise ValueError("alpha must be in (0, 1).")
        if int(self.repeats) < 1:
            raise ValueError("repeats must be >= 1.")
        if int(self.retrieval_limit) <= 0:
            raise ValueError("retrieval_limit must be positive.")
        if self.retriever_backend not in RETRIEVER_BACKENDS:
            raise ValueError(f"retriever_backend must be one of: {', '.join(RETRIEVER_BACKENDS)}.")
        if self.retriever_backend == "memory" and self.retriever_index_path is not None:
            raise ValueError("retriever_index_path is only supported with sqlite_fts or auto backends.")
        registry_fields = (self.registry_path, self.name, self.version)
        if any(value is not None for value in registry_fields) and not all(registry_fields):
            raise ValueError("registry_path, name, and version must be provided together.")

    @property
    def resolved_claims_path(self) -> Path:
        return self.claims_path or self.output_dir / "retrieval-claims.json"

    @property
    def resolved_verifier_report_path(self) -> Path:
        return self.verifier_report_path or self.output_dir / "retrieval-verifier-report.json"

    @property
    def resolved_route_report_path(self) -> Path:
        return self.route_report_path or self.output_dir / "retrieval-route-comparison.json"

    @property
    def resolved_promotion_report_path(self) -> Path:
        return self.promotion_report_path or self.output_dir / "retrieval-route-promotion.json"

    @property
    def resolved_artifact_manifest_path(self) -> Path:
        return self.artifact_manifest_path or self.output_dir / "retrieval-route-artifact-manifest.json"

    @property
    def resolved_verification_report_path(self) -> Path:
        return self.verification_report_path or self.output_dir / "retrieval-route-manifest-verification.json"

    @property
    def resolved_workflow_report_path(self) -> Path:
        return self.workflow_report_path or self.output_dir / "local-retrieval-route-workflow.json"


def run_local_retrieval_route_workflow(config: LocalRetrievalRouteWorkflowConfig) -> dict[str, Any]:
    """Run local retrieval evidence construction, route promotion, and optional registration."""
    workflow_started = time.perf_counter()
    profile: dict[str, float] = {}
    config.output_dir.mkdir(parents=True, exist_ok=True)

    with _profile_phase(profile, "resolve_claims_cache"):
        claims_cache = _resolve_claims_cache(config)

    score_dump: Mapping[str, Any] | None = None
    corpus_documents: Sequence[Any] | None = None
    claims_fixture = None
    if claims_cache["enabled"]:
        with _profile_phase(profile, "load_claims_cache"):
            cached = _load_claims_cache(claims_cache)
        claims_fixture = cached["fixture"]
        claims_cache = {**claims_cache, **cached["metadata"]}

    if claims_fixture is None:
        with _profile_phase(profile, "load_inputs"):
            score_dump = load_score_dump(config.scores_path)
            corpus_documents = load_corpus(config.corpus_paths)
        with _profile_phase(profile, "build_claims"):
            claims_fixture = build_evidence_fixture(
                score_dump,
                corpus_documents,
                retriever_min_overlap=config.retriever_min_overlap,
                retrieval_limit=config.retrieval_limit,
                query_field=config.query_field,
                retriever_backend=config.retriever_backend,
                retriever_index_path=config.retriever_index_path,
            )
        claims_cache = {
            **claims_cache,
            "hit": False,
            "status": (
                "disabled"
                if not claims_cache["enabled"]
                else "rebuilt_after_invalid"
                if claims_cache.get("status") == "invalid"
                else "miss"
            ),
            "scale": _claims_cache_scale(
                score_dump=score_dump,
                corpus_documents=corpus_documents,
                claims_fixture=claims_fixture,
            ),
        }
        if claims_cache["enabled"]:
            with _profile_phase(profile, "write_claims_cache"):
                _write_claims_cache(claims_cache, claims_fixture, compact=config.compact_json)

    with _profile_phase(profile, "write_claims"):
        _write_json(config.resolved_claims_path, claims_fixture, compact=config.compact_json)

    with _profile_phase(profile, "build_verifier_report"):
        verifier_report = build_verifier_ensemble_report(
            ((config.score_name, config.scores_path),),
            signal=config.signal,
            claims_path=config.resolved_claims_path,
            direction=config.direction,
            alphas=(config.alpha,),
            repeats=config.repeats,
            seed=config.seed,
            verifier_min_overlap=config.verifier_min_overlap,
            retriever_min_overlap=config.retriever_min_overlap,
            retrieval_limit=config.retrieval_limit,
            verification_cache_dir=config.verifier_trace_cache_dir,
        )
    with _profile_phase(profile, "write_verifier_report"):
        _write_json(config.resolved_verifier_report_path, verifier_report, compact=config.compact_json)

    with _profile_phase(profile, "run_adapter_promotion"):
        promotion_report = run_adapter_promotion_workflow(
            AdapterPromotionWorkflowConfig(
                reports=((config.score_name, config.resolved_verifier_report_path),),
                route_report_path=config.resolved_route_report_path,
                alpha=config.alpha,
                min_selected=config.min_selected,
                notes=("local retrieval route workflow",),
                gate_routes=config.gate_routes,
                gate_min_selected=config.gate_min_selected,
                min_decision_accuracy=config.min_decision_accuracy,
                max_false_supported_rate=config.max_false_supported_rate,
                min_false_refuted_rate=config.min_false_refuted_rate,
                max_verified_false_alarm=config.max_verified_false_alarm,
                min_verified_detection=config.min_verified_detection,
                max_mean_duration_seconds=config.max_mean_duration_seconds,
                max_p95_duration_seconds=config.max_p95_duration_seconds,
                max_p99_duration_seconds=config.max_p99_duration_seconds,
                max_max_duration_seconds=config.max_max_duration_seconds,
                max_mean_attempted_route_count=config.max_mean_attempted_route_count,
                max_retrieval_use_rate=config.max_retrieval_use_rate,
                min_cache_hit_rate=config.min_cache_hit_rate,
                compact_json=config.compact_json,
            )
        )
    with _profile_phase(profile, "write_promotion_report"):
        _write_json(config.resolved_promotion_report_path, promotion_report, compact=config.compact_json)

    runtime_profile = _runtime_profile_payload(
        profile,
        total_seconds=time.perf_counter() - workflow_started,
        config=config,
        score_dump=score_dump,
        corpus_documents=corpus_documents,
        claims_fixture=claims_fixture,
        verifier_report=verifier_report,
        promotion_report=promotion_report,
        claims_cache=claims_cache,
    )
    runtime_budget = _runtime_budget_report(config, runtime_profile)

    metadata = _manifest_metadata(
        config,
        claims_fixture=claims_fixture,
        verifier_report=verifier_report,
        promotion_report=promotion_report,
        runtime_profile=runtime_profile,
        runtime_budget=runtime_budget,
        claims_cache=claims_cache,
    )
    with _profile_phase(profile, "write_artifact_manifest"):
        _write_artifact_manifest(config, metadata=metadata, claims_cache=claims_cache)
    runtime_profile = _runtime_profile_payload(
        profile,
        total_seconds=time.perf_counter() - workflow_started,
        config=config,
        score_dump=score_dump,
        corpus_documents=corpus_documents,
        claims_fixture=claims_fixture,
        verifier_report=verifier_report,
        promotion_report=promotion_report,
        claims_cache=claims_cache,
    )
    runtime_budget = _runtime_budget_report(config, runtime_profile)

    with _profile_phase(profile, "register_manifest"):
        manifest_promotion = _maybe_register_manifest(
            config,
            promotion_report,
            metadata,
            runtime_budget=runtime_budget,
        )
    runtime_profile = _runtime_profile_payload(
        profile,
        total_seconds=time.perf_counter() - workflow_started,
        config=config,
        score_dump=score_dump,
        corpus_documents=corpus_documents,
        claims_fixture=claims_fixture,
        verifier_report=verifier_report,
        promotion_report=promotion_report,
        claims_cache=claims_cache,
    )
    runtime_budget = _runtime_budget_report(config, runtime_profile)
    decision = _workflow_decision(
        promotion_report=promotion_report,
        manifest_promotion=manifest_promotion,
        runtime_budget=runtime_budget,
        registry_requested=config.registry_path is not None,
    )
    payload = {
        "schema_version": 1,
        "workflow": "local_retrieval_route_workflow",
        "config": {
            "scores_path": str(config.scores_path),
            "corpus_paths": [str(path) for path in config.corpus_paths],
            "score_name": config.score_name,
            "signal": config.signal,
            "direction": config.direction,
            "alpha": float(config.alpha),
            "repeats": int(config.repeats),
            "seed": int(config.seed),
            "query_field": config.query_field,
            "retriever_backend": config.retriever_backend,
            "retriever_index_path": None if config.retriever_index_path is None else str(config.retriever_index_path),
            "verifier_min_overlap": float(config.verifier_min_overlap),
            "retriever_min_overlap": float(config.retriever_min_overlap),
            "retrieval_limit": int(config.retrieval_limit),
            "gate_routes": list(config.gate_routes),
            "claims_cache_dir": None if config.claims_cache_dir is None else str(config.claims_cache_dir),
            "verifier_trace_cache_dir": (
                None if config.verifier_trace_cache_dir is None else str(config.verifier_trace_cache_dir)
            ),
            "max_runtime_total_seconds": config.max_runtime_total_seconds,
            "max_retrieval_hit_count": config.max_retrieval_hit_count,
            "min_claims_cache_hit_rate": config.min_claims_cache_hit_rate,
            "min_verifier_trace_cache_hit_rate": config.min_verifier_trace_cache_hit_rate,
            "registry": None if config.registry_path is None else str(config.registry_path),
            "name": config.name,
            "version": config.version,
        },
        "claims_path": str(config.resolved_claims_path),
        "verifier_report_path": str(config.resolved_verifier_report_path),
        "route_report_path": str(config.resolved_route_report_path),
        "promotion_report_path": str(config.resolved_promotion_report_path),
        "artifact_manifest_path": str(config.resolved_artifact_manifest_path),
        "claims_summary": claims_fixture["summary"],
        "adapter_promotion": promotion_report,
        "manifest_promotion": manifest_promotion,
        "claims_cache": _public_claims_cache(claims_cache),
        "profile": runtime_profile,
        "runtime_budget": runtime_budget,
        "decision": decision,
    }
    _write_json(config.resolved_workflow_report_path, payload, compact=config.compact_json)
    return payload


def _maybe_register_manifest(
    config: LocalRetrievalRouteWorkflowConfig,
    promotion_report: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    runtime_budget: Mapping[str, Any],
) -> dict[str, Any] | None:
    if config.registry_path is None:
        return None
    if runtime_budget.get("enabled") and not runtime_budget.get("passed"):
        return None
    decision = dict(promotion_report.get("decision") or {})
    if decision.get("status") != "promote" and not config.allow_non_promote:
        return None
    return promote_artifact_manifest(
        manifest_path=config.resolved_artifact_manifest_path,
        registry_path=config.registry_path,
        name=str(config.name),
        version=str(config.version),
        verification_report_path=config.resolved_verification_report_path,
        recursive=True,
        allow_failures=config.allow_promotion_failures,
        metadata={
            **dict(metadata),
            "workflow": "run_local_retrieval_route_workflow",
        },
    )


def _write_artifact_manifest(
    config: LocalRetrievalRouteWorkflowConfig,
    *,
    metadata: Mapping[str, Any],
    claims_cache: Mapping[str, Any],
) -> None:
    artifacts: dict[str, str | Path | None] = {
        "score_dump": config.scores_path,
        "retrieval_claims": config.resolved_claims_path,
        "verifier_report": config.resolved_verifier_report_path,
        "route_comparison_report": config.resolved_route_report_path,
        "promotion_report": config.resolved_promotion_report_path,
    }
    cache_path = claims_cache.get("path")
    if cache_path is not None:
        artifacts["claims_cache_record"] = Path(str(cache_path))
    if (
        metadata.get("retriever_actual_backend") == "sqlite_fts"
        and config.retriever_index_path is not None
        and config.retriever_index_path.exists()
    ):
        artifacts["retriever_index"] = config.retriever_index_path
    for idx, path in enumerate(config.corpus_paths, start=1):
        artifacts[f"retrieval_corpora.{idx}.{path.stem}"] = path
    manifest = build_artifact_manifest(
        artifacts,
        root=config.resolved_artifact_manifest_path.parent,
        metadata=metadata,
    )
    _write_json(config.resolved_artifact_manifest_path, manifest, compact=False)


def _manifest_metadata(
    config: LocalRetrievalRouteWorkflowConfig,
    *,
    claims_fixture: Mapping[str, Any],
    verifier_report: Mapping[str, Any],
    promotion_report: Mapping[str, Any],
    runtime_profile: Mapping[str, Any],
    runtime_budget: Mapping[str, Any],
    claims_cache: Mapping[str, Any],
) -> dict[str, Any]:
    decision = dict(promotion_report.get("decision") or {})
    route_comparison = dict(promotion_report.get("route_comparison") or {})
    route_decision = dict(route_comparison.get("promotion_decision") or {})
    recommended_route = route_decision.get("recommended_route") or decision.get("recommended_route")
    by_route = dict(route_comparison.get("by_route") or {})
    recommended_metrics = dict(by_route.get(str(recommended_route)) or {}) if recommended_route else {}
    quality_gate = dict(route_comparison.get("quality_gate") or {})
    summary = dict(claims_fixture.get("summary") or {})
    retriever_info = dict(claims_fixture.get("retriever") or {})
    trace_cache = _verifier_trace_cache_summary(verifier_report)
    runtime_summary = dict(runtime_profile.get("summary") or {})
    runtime_scale = dict(runtime_profile.get("scale") or {})
    runtime_artifacts = dict(runtime_profile.get("artifacts") or {})
    output_bytes = dict(runtime_artifacts.get("output_bytes") or {})
    runtime_budget_metrics = dict(runtime_budget.get("metrics") or {})
    metadata = {
        "runner": "run_local_retrieval_route_workflow",
        "workflow": "local_retrieval_route_workflow",
        "source_workflow": promotion_report.get("workflow"),
        "adapter_promotion_status": decision.get("status"),
        "route_promotion_status": route_decision.get("status"),
        "recommended_route": recommended_route,
        "quality_gate_passed": quality_gate.get("passed"),
        "quality_gate_checked_routes": quality_gate.get("checked_routes"),
        "score_name": config.score_name,
        "signal": config.signal,
        "query_field": config.query_field,
        "retriever_backend": config.retriever_backend,
        "retriever_requested_index_path": (
            None if config.retriever_index_path is None else str(config.retriever_index_path)
        ),
        "retriever_actual_backend": retriever_info.get("actual_backend"),
        "retriever_actual_index_path": retriever_info.get("actual_index_path"),
        "retriever_index_reused": retriever_info.get("index_reused"),
        "verifier_min_overlap": float(config.verifier_min_overlap),
        "retriever_min_overlap": float(config.retriever_min_overlap),
        "retrieval_limit": int(config.retrieval_limit),
        "corpus_count": len(config.corpus_paths),
        "claims_n_records": summary.get("n_records"),
        "claims_records_with_hits": summary.get("records_with_hits"),
        "claims_total_hits": summary.get("total_hits"),
        "claims_average_hits_per_record": summary.get("average_hits_per_record"),
        "claims_cache_enabled": bool(claims_cache.get("enabled", False)),
        "claims_cache_hit": bool(claims_cache.get("hit", False)),
        "claims_cache_status": claims_cache.get("status"),
        "claims_cache_key": claims_cache.get("key"),
        "claims_cache_path": claims_cache.get("path"),
        "verifier_trace_cache_enabled": trace_cache["enabled"],
        "verifier_trace_cache_hit_count": trace_cache["hit_count"],
        "verifier_trace_cache_run_count": trace_cache["run_count"],
        "verifier_trace_cache_path": trace_cache["path"],
        "runtime_total_seconds": runtime_profile.get("total_seconds"),
        "runtime_bottleneck": runtime_summary.get("bottleneck"),
        "runtime_accounted_share": runtime_summary.get("accounted_share"),
        "runtime_n_labels": runtime_scale.get("n_labels"),
        "runtime_n_corpus_documents": runtime_scale.get("n_corpus_documents"),
        "runtime_n_claim_records": runtime_scale.get("n_claim_records"),
        "runtime_n_retrieval_hits": runtime_scale.get("n_retrieval_hits"),
        "runtime_budget_enabled": runtime_budget.get("enabled"),
        "runtime_budget_passed": runtime_budget.get("passed"),
        "runtime_budget_policy": runtime_budget.get("policy"),
        "runtime_budget_failures": runtime_budget.get("failures"),
        "runtime_claims_cache_hit_rate": runtime_budget_metrics.get("claims_cache_hit_rate"),
        "runtime_verifier_trace_cache_hit_rate": runtime_budget_metrics.get("verifier_trace_cache_hit_rate"),
        "runtime_claims_json_bytes": output_bytes.get("retrieval_claims"),
        "runtime_verifier_report_json_bytes": output_bytes.get("verifier_report"),
        "runtime_route_report_json_bytes": output_bytes.get("route_comparison_report"),
        "runtime_promotion_report_json_bytes": output_bytes.get("promotion_report"),
        "recommended_selected": recommended_metrics.get("selected"),
        "recommended_decision_accuracy": recommended_metrics.get("decision_accuracy"),
        "recommended_false_supported_rate": recommended_metrics.get("false_supported_rate"),
        "recommended_false_refuted_rate": recommended_metrics.get("false_refuted_rate"),
        "recommended_verified_false_alarm": recommended_metrics.get("verified_false_alarm"),
        "recommended_verified_detection": recommended_metrics.get("verified_detection"),
        "recommended_mean_duration_seconds": recommended_metrics.get("mean_duration_seconds"),
        "recommended_p95_duration_seconds": recommended_metrics.get("p95_duration_seconds"),
        "recommended_p99_duration_seconds": recommended_metrics.get("p99_duration_seconds"),
        "recommended_max_duration_seconds": recommended_metrics.get("max_duration_seconds"),
        "recommended_mean_attempted_route_count": recommended_metrics.get("mean_attempted_route_count"),
        "recommended_retrieval_use_rate": recommended_metrics.get("retrieval_use_rate"),
        "recommended_invalid_metric_counts": recommended_metrics.get("invalid_metric_counts"),
    }
    if config.promotion_metadata is not None:
        metadata.update(dict(config.promotion_metadata))
    return metadata


def _resolve_claims_cache(config: LocalRetrievalRouteWorkflowConfig) -> dict[str, Any]:
    if config.claims_cache_dir is None:
        return {
            "enabled": False,
            "hit": False,
            "status": "disabled",
            "key": None,
            "path": None,
            "scale": {},
        }
    material = _claims_cache_material(config)
    key = hashlib.sha256(_stable_json_bytes(material)).hexdigest()
    path = config.claims_cache_dir / f"local-retrieval-claims-{key}.json"
    return {
        "enabled": True,
        "hit": False,
        "status": "miss",
        "key": key,
        "path": str(path),
        "material": material,
        "scale": {},
    }


def _claims_cache_material(config: LocalRetrievalRouteWorkflowConfig) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "cache_type": "local_retrieval_claims",
        "builder": "build_evidence_fixture:v2",
        "score_dump": fingerprint_path(config.scores_path).to_dict(),
        "corpora": [fingerprint_path(path).to_dict() for path in config.corpus_paths],
        "retrieval": {
            "query_field": config.query_field,
            "retriever_backend": config.retriever_backend,
            "retriever_min_overlap": float(config.retriever_min_overlap),
            "retrieval_limit": int(config.retrieval_limit),
        },
    }


def _load_claims_cache(cache: Mapping[str, Any]) -> dict[str, Any]:
    path_text = cache.get("path")
    if not path_text:
        return {"fixture": None, "metadata": {"hit": False, "status": "disabled"}}
    path = Path(str(path_text))
    if not path.exists():
        return {"fixture": None, "metadata": {"hit": False, "status": "miss"}}
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(record, Mapping):
            raise ValueError("cache record must be a JSON object")
        if record.get("schema_version") != 1:
            raise ValueError("unsupported cache schema_version")
        if record.get("cache_type") != "local_retrieval_claims":
            raise ValueError("unexpected cache_type")
        if record.get("cache_key") != cache.get("key"):
            raise ValueError("cache key mismatch")
        fixture = record.get("fixture")
        if not isinstance(fixture, Mapping):
            raise ValueError("cache fixture must be a JSON object")
        scale = record.get("scale")
        return {
            "fixture": dict(fixture),
            "metadata": {
                "hit": True,
                "status": "hit",
                "scale": dict(scale) if isinstance(scale, Mapping) else {},
            },
        }
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "fixture": None,
            "metadata": {
                "hit": False,
                "status": "invalid",
                "load_error": str(exc),
            },
        }


def _write_claims_cache(cache: Mapping[str, Any], claims_fixture: Mapping[str, Any], *, compact: bool) -> None:
    path_text = cache.get("path")
    if not cache.get("enabled") or not path_text:
        return
    record = {
        "schema_version": 1,
        "cache_type": "local_retrieval_claims",
        "cache_key": cache.get("key"),
        "key_material": cache.get("material"),
        "scale": dict(cache.get("scale") or {}),
        "fixture": dict(claims_fixture),
    }
    _write_json(Path(str(path_text)), record, compact=compact)


def _claims_cache_scale(
    *,
    score_dump: Mapping[str, Any],
    corpus_documents: Sequence[Any],
    claims_fixture: Mapping[str, Any],
) -> dict[str, Any]:
    labels = tuple(score_dump.get("labels", ()))
    statements = tuple(score_dump.get("statements", ()))
    records = tuple(claims_fixture.get("records", ()))
    summary = dict(claims_fixture.get("summary") or {})
    return {
        "n_labels": len(labels),
        "n_statements": len(statements),
        "n_corpus_documents": len(corpus_documents),
        "n_claim_records": len(records),
        "n_records_with_hits": summary.get("records_with_hits"),
        "n_retrieval_hits": summary.get("total_hits"),
        "average_hits_per_record": summary.get("average_hits_per_record"),
    }


def _public_claims_cache(cache: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "enabled": bool(cache.get("enabled", False)),
        "hit": bool(cache.get("hit", False)),
        "status": cache.get("status"),
        "key": cache.get("key"),
        "path": cache.get("path"),
        "scale": dict(cache.get("scale") or {}),
    }
    if cache.get("load_error") is not None:
        payload["load_error"] = cache.get("load_error")
    return payload


def _verifier_trace_cache_summary(verifier_report: Mapping[str, Any]) -> dict[str, Any]:
    runs = tuple(verifier_report.get("runs", ()))
    trace_entries = []
    for run in runs:
        if not isinstance(run, Mapping):
            continue
        cache_stats = run.get("cache_stats", {})
        if not isinstance(cache_stats, Mapping):
            continue
        trace_cache = cache_stats.get("trace_cache", {})
        if isinstance(trace_cache, Mapping):
            trace_entries.append(dict(trace_cache))
    enabled = any(bool(entry.get("enabled")) for entry in trace_entries)
    hit_count = sum(1 for entry in trace_entries if entry.get("hit"))
    paths = [entry.get("path") for entry in trace_entries if entry.get("path") is not None]
    keys = [entry.get("key") for entry in trace_entries if entry.get("key") is not None]
    return {
        "enabled": enabled,
        "hit": bool(trace_entries) and hit_count == len(trace_entries),
        "hit_count": hit_count,
        "run_count": len(trace_entries),
        "path": paths[0] if paths else None,
        "keys": tuple(keys),
    }


def _stable_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


@contextmanager
def _profile_phase(profile: dict[str, float], name: str) -> Iterator[None]:
    started = time.perf_counter()
    try:
        yield
    finally:
        profile[name] = profile.get(name, 0.0) + (time.perf_counter() - started)


def _runtime_profile_payload(
    profile: Mapping[str, float],
    *,
    total_seconds: float,
    config: LocalRetrievalRouteWorkflowConfig,
    score_dump: Mapping[str, Any] | None,
    corpus_documents: Sequence[Any] | None,
    claims_fixture: Mapping[str, Any],
    verifier_report: Mapping[str, Any],
    promotion_report: Mapping[str, Any],
    claims_cache: Mapping[str, Any],
) -> dict[str, Any]:
    phases = {name: _round_seconds(seconds) for name, seconds in sorted(profile.items())}
    scale = _runtime_scale(
        config=config,
        score_dump=score_dump,
        corpus_documents=corpus_documents,
        claims_fixture=claims_fixture,
        promotion_report=promotion_report,
        claims_cache=claims_cache,
    )
    output_bytes = {
        "retrieval_claims": _path_size(config.resolved_claims_path),
        "verifier_report": _path_size(config.resolved_verifier_report_path),
        "route_comparison_report": _path_size(config.resolved_route_report_path),
        "promotion_report": _path_size(config.resolved_promotion_report_path),
        "artifact_manifest": _path_size(config.resolved_artifact_manifest_path),
        "manifest_verification": _path_size(config.resolved_verification_report_path),
    }
    cache_path = claims_cache.get("path")
    if cache_path is not None:
        output_bytes["claims_cache_record"] = _path_size(Path(str(cache_path)))
    artifacts = {
        "input_bytes": {
            "score_dump": _path_size(config.scores_path),
            **{
                f"retrieval_corpora.{idx}.{path.stem}": _path_size(path)
                for idx, path in enumerate(config.corpus_paths, start=1)
            },
        },
        "output_bytes": output_bytes,
    }
    return {
        "total_seconds": _round_seconds(total_seconds),
        "phases": phases,
        "summary": _runtime_profile_summary(phases, total_seconds=total_seconds),
        "scale": scale,
        "cache": {
            "claims": _public_claims_cache(claims_cache),
            "verifier_trace": _verifier_trace_cache_summary(verifier_report),
        },
        "artifacts": artifacts,
    }


def _runtime_budget_report(
    config: LocalRetrievalRouteWorkflowConfig,
    runtime_profile: Mapping[str, Any],
) -> dict[str, Any]:
    return evaluate_runtime_budget(
        runtime_metrics_from_profile(runtime_profile),
        RuntimeBudgetPolicy(
            max_total_seconds=config.max_runtime_total_seconds,
            max_retrieval_hit_count=config.max_retrieval_hit_count,
            min_claims_cache_hit_rate=config.min_claims_cache_hit_rate,
            min_verifier_trace_cache_hit_rate=config.min_verifier_trace_cache_hit_rate,
        ),
    )


def _runtime_scale(
    *,
    config: LocalRetrievalRouteWorkflowConfig,
    score_dump: Mapping[str, Any] | None,
    corpus_documents: Sequence[Any] | None,
    claims_fixture: Mapping[str, Any],
    promotion_report: Mapping[str, Any],
    claims_cache: Mapping[str, Any],
) -> dict[str, Any]:
    cache_scale = dict(claims_cache.get("scale") or {})
    labels = tuple(score_dump.get("labels", ())) if score_dump is not None else ()
    statements = tuple(score_dump.get("statements", ())) if score_dump is not None else ()
    records = tuple(claims_fixture.get("records", ()))
    summary = dict(claims_fixture.get("summary") or {})
    route_comparison = dict(promotion_report.get("route_comparison") or {})
    by_route = dict(route_comparison.get("by_route") or {})
    return {
        "n_labels": len(labels) if score_dump is not None else cache_scale.get("n_labels"),
        "n_statements": len(statements) if score_dump is not None else cache_scale.get("n_statements"),
        "n_corpus_documents": (
            len(corpus_documents) if corpus_documents is not None else cache_scale.get("n_corpus_documents")
        ),
        "n_claim_records": len(records),
        "n_records_with_hits": summary.get("records_with_hits"),
        "n_retrieval_hits": summary.get("total_hits"),
        "average_hits_per_record": summary.get("average_hits_per_record"),
        "n_routes": len(by_route),
        "n_gate_routes": len(config.gate_routes),
        "repeats": int(config.repeats),
        "retrieval_limit": int(config.retrieval_limit),
    }


def _runtime_profile_summary(phases: Mapping[str, float], *, total_seconds: float) -> dict[str, Any]:
    phase_total = sum(max(float(seconds), 0.0) for seconds in phases.values())
    top_phases = sorted(
        (
            {
                "name": name,
                "seconds": _round_seconds(seconds),
                "share": _runtime_share(seconds, total_seconds),
            }
            for name, seconds in phases.items()
        ),
        key=lambda item: (-float(item["seconds"]), str(item["name"])),
    )
    return {
        "bottleneck": None if not top_phases else top_phases[0]["name"],
        "top_phases": top_phases[:5],
        "accounted_share": _runtime_share(phase_total, total_seconds),
    }


def _runtime_share(seconds: float, total_seconds: float) -> float:
    if total_seconds <= 0.0:
        return 0.0
    return round(max(float(seconds), 0.0) / float(total_seconds), 6)


def _round_seconds(seconds: float) -> float:
    return round(max(float(seconds), 0.0), 6)


def _path_size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


def _workflow_decision(
    *,
    promotion_report: Mapping[str, Any],
    manifest_promotion: Mapping[str, Any] | None,
    runtime_budget: Mapping[str, Any],
    registry_requested: bool,
) -> dict[str, Any]:
    adapter_decision = dict(promotion_report.get("decision") or {})
    adapter_status = str(adapter_decision.get("status"))
    blocking_reasons = []
    if adapter_status != "promote":
        blocking_reasons.append("adapter promotion decision did not promote")
    if runtime_budget.get("enabled") and not runtime_budget.get("passed"):
        metrics = ", ".join(
            str(failure.get("metric"))
            for failure in runtime_budget.get("failures", ())
            if isinstance(failure, Mapping)
        )
        blocking_reasons.append(f"runtime budget did not pass: {metrics or 'unknown metric'}")
    verification = {} if manifest_promotion is None else dict(manifest_promotion.get("verification") or {})
    manifest_verified = bool(verification.get("passed", False))
    if registry_requested and manifest_promotion is None:
        blocking_reasons.append("registry promotion was not attempted")
    if registry_requested and manifest_promotion is not None and not manifest_verified:
        blocking_reasons.append("retrieval route manifest verification did not pass")
    status = "promote" if not blocking_reasons else "blocked"
    return {
        "status": status,
        "adapter_promotion_status": adapter_status,
        "runtime_budget_passed": None if not runtime_budget.get("enabled") else bool(runtime_budget.get("passed")),
        "manifest_promoted": manifest_promotion is not None,
        "manifest_verified": manifest_verified,
        "registry_record": None if manifest_promotion is None else dict(manifest_promotion.get("records") or {}).get(
            "benchmark_manifest"
        ),
        "blocking_reasons": tuple(blocking_reasons),
    }


def _write_json(path: Path, payload: Mapping[str, Any], *, compact: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"))
        if compact
        else json.dumps(payload, indent=2, sort_keys=True)
    )
    path.write_text(text + "\n", encoding="utf-8")


def _parse_non_negative_float(value: str, *, flag: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(f"{flag} must be a non-negative finite number.")
    return numeric


def _parse_positive_float(value: str, *, flag: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0:
        raise ValueError(f"{flag} must be a positive finite number.")
    return numeric


def _parse_non_negative_int(value: str, *, flag: str) -> int:
    numeric = int(value)
    if numeric < 0:
        raise ValueError(f"{flag} must be a non-negative integer.")
    return numeric


def _parse_positive_int(value: str, *, flag: str) -> int:
    numeric = int(value)
    if numeric <= 0:
        raise ValueError(f"{flag} must be a positive integer.")
    return numeric


def _parse_metadata(values: Sequence[str]) -> dict[str, str]:
    metadata = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"metadata entry must be key=value: {value!r}")
        key, text = value.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"metadata key must not be empty: {value!r}")
        metadata[key] = text
    return metadata


def _config_from_args(args: argparse.Namespace) -> LocalRetrievalRouteWorkflowConfig:
    return LocalRetrievalRouteWorkflowConfig(
        scores_path=Path(args.scores),
        corpus_paths=tuple(Path(path) for path in args.corpus),
        output_dir=Path(args.output_dir),
        registry_path=None if args.registry is None else Path(args.registry),
        name=args.name,
        version=args.version,
        score_name=args.score_name,
        signal=args.signal,
        direction=args.direction,
        alpha=args.alpha,
        repeats=args.repeats,
        seed=args.seed,
        query_field=args.query_field,
        retriever_backend=args.retriever_backend,
        retriever_index_path=None if args.retriever_index_path is None else Path(args.retriever_index_path),
        verifier_min_overlap=args.verifier_min_overlap,
        retriever_min_overlap=args.retriever_min_overlap,
        retrieval_limit=args.retrieval_limit,
        min_selected=args.min_selected,
        gate_routes=tuple(args.gate_route or ("retrieval_groundedness",)),
        gate_min_selected=args.gate_min_selected,
        min_decision_accuracy=args.min_decision_accuracy,
        max_false_supported_rate=args.max_false_supported_rate,
        min_false_refuted_rate=args.min_false_refuted_rate,
        max_verified_false_alarm=args.max_verified_false_alarm,
        min_verified_detection=args.min_verified_detection,
        max_mean_duration_seconds=args.max_mean_duration_seconds,
        max_p95_duration_seconds=args.max_p95_duration_seconds,
        max_p99_duration_seconds=args.max_p99_duration_seconds,
        max_max_duration_seconds=args.max_max_duration_seconds,
        max_mean_attempted_route_count=args.max_mean_attempted_route_count,
        max_retrieval_use_rate=args.max_retrieval_use_rate,
        min_cache_hit_rate=args.min_cache_hit_rate,
        max_runtime_total_seconds=args.max_runtime_total_seconds,
        max_retrieval_hit_count=args.max_retrieval_hit_count,
        min_claims_cache_hit_rate=args.min_claims_cache_hit_rate,
        min_verifier_trace_cache_hit_rate=args.min_verifier_trace_cache_hit_rate,
        claims_path=None if args.claims_json is None else Path(args.claims_json),
        verifier_report_path=None if args.verifier_report_json is None else Path(args.verifier_report_json),
        route_report_path=None if args.route_report_json is None else Path(args.route_report_json),
        promotion_report_path=None if args.promotion_report_json is None else Path(args.promotion_report_json),
        artifact_manifest_path=None if args.artifact_manifest is None else Path(args.artifact_manifest),
        verification_report_path=None if args.verification_report is None else Path(args.verification_report),
        workflow_report_path=None if args.json is None else Path(args.json),
        claims_cache_dir=None if args.claims_cache_dir is None else Path(args.claims_cache_dir),
        verifier_trace_cache_dir=(
            None if args.verifier_trace_cache_dir is None else Path(args.verifier_trace_cache_dir)
        ),
        compact_json=bool(args.compact_json),
        allow_non_promote=bool(args.allow_non_promote),
        allow_promotion_failures=bool(args.allow_promotion_failures),
        promotion_metadata=_parse_metadata(args.metadata or ()),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run from parsed CLI arguments."""
    payload = run_local_retrieval_route_workflow(_config_from_args(args))
    decision = payload["decision"]
    print(
        "local_retrieval_route="
        f"{decision['status']} "
        f"adapter={decision.get('adapter_promotion_status')} "
        f"record={decision.get('registry_record')}"
    )
    if args.fail_on_blocked and decision["status"] != "promote":
        raise SystemExit(1)
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build and optionally register a local retrieval-groundedness route baseline"
    )
    parser.add_argument("--scores", required=True, help="statement-bearing score dump JSON")
    parser.add_argument("--corpus", action="append", required=True, help="local JSON/JSONL/text corpus path")
    parser.add_argument("--output-dir", required=True, help="directory for workflow artifacts")
    parser.add_argument("--registry", default=None, help="optional ArtifactRegistry JSON path")
    parser.add_argument("--name", default=None, help="registry baseline name; required with --registry")
    parser.add_argument("--version", default=None, help="registry baseline version; required with --registry")
    parser.add_argument("--score-name", default="retrieval")
    parser.add_argument("--signal", default="truth_proj")
    parser.add_argument("--direction", default=None)
    parser.add_argument("--alpha", type=lambda value: _parse_positive_float(value, flag="--alpha"), default=0.10)
    parser.add_argument("--repeats", type=lambda value: _parse_positive_int(value, flag="--repeats"), default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--query-field",
        choices=("text", "answer", "question", "question_answer"),
        default="answer",
    )
    parser.add_argument("--retriever-backend", choices=RETRIEVER_BACKENDS, default="memory")
    parser.add_argument(
        "--retriever-index-path",
        default=None,
        help="optional persistent SQLite FTS index path for sqlite_fts/auto backends",
    )
    parser.add_argument("--verifier-min-overlap", type=float, default=0.65)
    parser.add_argument("--retriever-min-overlap", type=float, default=0.20)
    parser.add_argument("--retrieval-limit", type=lambda value: _parse_positive_int(
        value,
        flag="--retrieval-limit",
    ), default=5)
    parser.add_argument("--min-selected", type=lambda value: _parse_non_negative_int(
        value,
        flag="--min-selected",
    ), default=1)
    parser.add_argument("--gate-route", action="append", default=None)
    parser.add_argument("--gate-min-selected", type=lambda value: _parse_non_negative_int(
        value,
        flag="--gate-min-selected",
    ), default=None)
    parser.add_argument("--min-decision-accuracy", type=lambda value: _parse_non_negative_float(
        value,
        flag="--min-decision-accuracy",
    ), default=None)
    parser.add_argument("--max-false-supported-rate", type=lambda value: _parse_non_negative_float(
        value,
        flag="--max-false-supported-rate",
    ), default=None)
    parser.add_argument("--min-false-refuted-rate", type=lambda value: _parse_non_negative_float(
        value,
        flag="--min-false-refuted-rate",
    ), default=None)
    parser.add_argument("--max-verified-false-alarm", type=lambda value: _parse_non_negative_float(
        value,
        flag="--max-verified-false-alarm",
    ), default=None)
    parser.add_argument("--min-verified-detection", type=lambda value: _parse_non_negative_float(
        value,
        flag="--min-verified-detection",
    ), default=None)
    parser.add_argument("--max-mean-duration-seconds", type=lambda value: _parse_non_negative_float(
        value,
        flag="--max-mean-duration-seconds",
    ), default=None)
    parser.add_argument("--max-p95-duration-seconds", type=lambda value: _parse_non_negative_float(
        value,
        flag="--max-p95-duration-seconds",
    ), default=None)
    parser.add_argument("--max-p99-duration-seconds", type=lambda value: _parse_non_negative_float(
        value,
        flag="--max-p99-duration-seconds",
    ), default=None)
    parser.add_argument("--max-max-duration-seconds", type=lambda value: _parse_non_negative_float(
        value,
        flag="--max-max-duration-seconds",
    ), default=None)
    parser.add_argument("--max-mean-attempted-route-count", type=lambda value: _parse_non_negative_float(
        value,
        flag="--max-mean-attempted-route-count",
    ), default=None)
    parser.add_argument("--max-retrieval-use-rate", type=lambda value: _parse_non_negative_float(
        value,
        flag="--max-retrieval-use-rate",
    ), default=None)
    parser.add_argument("--min-cache-hit-rate", type=lambda value: _parse_non_negative_float(
        value,
        flag="--min-cache-hit-rate",
    ), default=None)
    parser.add_argument("--max-runtime-total-seconds", type=lambda value: _parse_non_negative_float(
        value,
        flag="--max-runtime-total-seconds",
    ), default=None)
    parser.add_argument("--max-retrieval-hit-count", type=lambda value: _parse_non_negative_float(
        value,
        flag="--max-retrieval-hit-count",
    ), default=None)
    parser.add_argument("--min-claims-cache-hit-rate", type=lambda value: _parse_non_negative_float(
        value,
        flag="--min-claims-cache-hit-rate",
    ), default=None)
    parser.add_argument("--min-verifier-trace-cache-hit-rate", type=lambda value: _parse_non_negative_float(
        value,
        flag="--min-verifier-trace-cache-hit-rate",
    ), default=None)
    parser.add_argument("--claims-json", default=None)
    parser.add_argument("--verifier-report-json", default=None)
    parser.add_argument("--route-report-json", default=None)
    parser.add_argument("--promotion-report-json", default=None)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--verification-report", default=None)
    parser.add_argument("--json", default=None, help="optional workflow report path")
    parser.add_argument(
        "--claims-cache-dir",
        default=None,
        help="optional cache directory for generated claims fixtures",
    )
    parser.add_argument(
        "--verifier-trace-cache-dir",
        default=None,
        help="optional cache directory for verified-record traces",
    )
    parser.add_argument("--metadata", action="append", default=[], help="extra promotion metadata as key=value")
    parser.add_argument("--compact-json", action="store_true")
    parser.add_argument("--allow-non-promote", action="store_true")
    parser.add_argument("--allow-promotion-failures", action="store_true")
    parser.add_argument("--fail-on-blocked", action="store_true")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()

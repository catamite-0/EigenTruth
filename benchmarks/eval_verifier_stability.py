"""Evaluate multi-seed stability for verifier-ensemble reports.

This is a model-free post-hoc benchmark. It consumes existing score dumps and
local verifier fixtures, reruns ``eval_verifier_ensemble.py`` across several
split-conformal seeds, and records whether verifier routes and verified risk
metrics remain stable.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.eval_verifier_ensemble import ALPHAS, build_verifier_ensemble_report  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402


def _parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError("scores name cannot be empty.")
    return name, Path(path)


def _parse_csv(value: str | None, *, name: str) -> tuple[str, ...]:
    if value is None:
        raise ValueError(f"{name} must contain at least one value.")
    parts = tuple(part.strip() for part in value.split(",") if part.strip())
    if not parts:
        raise ValueError(f"{name} must contain at least one value.")
    return parts


def _parse_float_csv(value: str | None, *, name: str) -> tuple[float, ...]:
    return tuple(float(part) for part in _parse_csv(value, name=name))


def _parse_int_csv(value: str | None, *, name: str) -> tuple[int, ...]:
    values = tuple(int(part) for part in _parse_csv(value, name=name))
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicate integers.")
    return values


def _contains_alpha(alphas: Sequence[float], target: float) -> bool:
    return any(abs(float(alpha) - float(target)) <= 1e-12 for alpha in alphas)


def _alpha_payload(run: Mapping[str, Any], alpha: float) -> Mapping[str, Any]:
    alphas = run.get("alphas", {})
    if not isinstance(alphas, Mapping):
        return {}
    direct = alphas.get(str(float(alpha)))
    if isinstance(direct, Mapping):
        return direct
    for key, payload in alphas.items():
        try:
            matches = abs(float(key) - float(alpha)) <= 1e-12
        except (TypeError, ValueError):
            matches = False
        if matches and isinstance(payload, Mapping):
            return payload
    return {}


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed


def _float_stats(values: Sequence[float | None]) -> dict[str, Any]:
    finite_values = [float(value) for value in values if value is not None]
    if not finite_values:
        return {"count": 0, "mean": None, "stdev": None, "min": None, "max": None}
    return {
        "count": len(finite_values),
        "mean": statistics.fmean(finite_values),
        "stdev": statistics.pstdev(finite_values) if len(finite_values) > 1 else 0.0,
        "min": min(finite_values),
        "max": max(finite_values),
    }


def _validate_unique_score_dump_names(score_dumps: Sequence[tuple[str, Path]]) -> None:
    counts = Counter(name for name, _ in score_dumps)
    duplicates = sorted(name for name, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"score dump names must be unique; duplicate name(s): {duplicates}.")


def _selected_route_counts(run: Mapping[str, Any]) -> dict[str, int]:
    route_summary = run.get("route_summary", {})
    if not isinstance(route_summary, Mapping):
        return {}
    selected_counts = route_summary.get("selected_counts", {})
    if not isinstance(selected_counts, Mapping):
        return {}
    return {str(route): int(count) for route, count in sorted(selected_counts.items())}


def _route_signature(counts: Mapping[str, int]) -> str:
    return ",".join(f"{route}:{count}" for route, count in sorted(counts.items()))


def _compact_route_quality(run: Mapping[str, Any]) -> dict[str, Any]:
    route_quality = run.get("route_quality", {})
    if not isinstance(route_quality, Mapping):
        return {}
    compact = {}
    for route, payload in sorted(route_quality.items()):
        if not isinstance(payload, Mapping):
            continue
        compact[str(route)] = {
            "selected": payload.get("selected"),
            "decision_accuracy": payload.get("decision_accuracy"),
            "false_refuted_rate": payload.get("false_refuted_rate"),
            "false_supported_rate": payload.get("false_supported_rate"),
            "true_supported_rate": payload.get("true_supported_rate"),
            "retrieval_use_rate": payload.get("retrieval_use_rate"),
            "mean_attempted_route_count": payload.get("mean_attempted_route_count"),
        }
    return compact


def _compact_seed_run(run: Mapping[str, Any], *, seed: int, alpha: float) -> dict[str, Any]:
    alpha_result = _alpha_payload(run, alpha)
    internal = alpha_result.get("internal", {}) if isinstance(alpha_result.get("internal"), Mapping) else {}
    verified = alpha_result.get("verified", {}) if isinstance(alpha_result.get("verified"), Mapping) else {}
    delta = alpha_result.get("delta", {}) if isinstance(alpha_result.get("delta"), Mapping) else {}
    selected_counts = _selected_route_counts(run)
    staged = run.get("staged_verification", {})
    if not isinstance(staged, Mapping):
        staged = {}
    return {
        "seed": int(seed),
        "alpha": float(alpha),
        "internal": {
            "false_alarm": internal.get("false_alarm"),
            "detection": internal.get("detection"),
            "pass": internal.get("pass"),
        },
        "verified": {
            "false_alarm": verified.get("false_alarm"),
            "detection": verified.get("detection"),
            "pass": verified.get("pass"),
        },
        "delta": {
            "false_alarm": delta.get("false_alarm"),
            "detection": delta.get("detection"),
            "suppressed_false_alarm_rate": delta.get("suppressed_false_alarm_rate"),
            "rescued_detection_rate": delta.get("rescued_detection_rate"),
        },
        "selected_route_counts": selected_counts,
        "selected_route_signature": _route_signature(selected_counts),
        "route_quality": _compact_route_quality(run),
        "staged_verification": {
            "enabled": staged.get("enabled"),
            "skip_rate": staged.get("skip_rate"),
            "verified_records": staged.get("verified_records"),
            "skipped_records": staged.get("skipped_records"),
        },
    }


def _metric_values(seed_entries: Sequence[Mapping[str, Any]], section: str, metric: str) -> tuple[float | None, ...]:
    values = []
    for entry in seed_entries:
        payload = entry.get(section, {})
        if isinstance(payload, Mapping):
            values.append(_float_or_none(payload.get(metric)))
    return tuple(values)


def _count_true(seed_entries: Sequence[Mapping[str, Any]], section: str, key: str) -> int:
    count = 0
    for entry in seed_entries:
        payload = entry.get(section, {})
        if isinstance(payload, Mapping) and payload.get(key) is True:
            count += 1
    return count


def _signature_counts(seed_entries: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counter = Counter(str(entry.get("selected_route_signature", "")) for entry in seed_entries)
    return dict(sorted(counter.items()))


def _route_totals(seed_entries: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    totals: Counter[str] = Counter()
    for entry in seed_entries:
        selected_counts = entry.get("selected_route_counts", {})
        if not isinstance(selected_counts, Mapping):
            continue
        for route, count in selected_counts.items():
            totals[str(route)] += int(count)
    return dict(sorted(totals.items()))


def _summarize_seed_entries(seed_entries: Sequence[Mapping[str, Any]], *, alpha: float) -> dict[str, Any]:
    internal_detection = _metric_values(seed_entries, "internal", "detection")
    verified_detection = _metric_values(seed_entries, "verified", "detection")
    detection_delta = _metric_values(seed_entries, "delta", "detection")
    return {
        "seed_count": len(seed_entries),
        "alpha": float(alpha),
        "internal_false_alarm": _float_stats(_metric_values(seed_entries, "internal", "false_alarm")),
        "internal_detection": _float_stats(internal_detection),
        "verified_false_alarm": _float_stats(_metric_values(seed_entries, "verified", "false_alarm")),
        "verified_detection": _float_stats(verified_detection),
        "delta_false_alarm": _float_stats(_metric_values(seed_entries, "delta", "false_alarm")),
        "delta_detection": _float_stats(detection_delta),
        "suppressed_false_alarm_rate": _float_stats(
            _metric_values(seed_entries, "delta", "suppressed_false_alarm_rate")
        ),
        "rescued_detection_rate": _float_stats(_metric_values(seed_entries, "delta", "rescued_detection_rate")),
        "verified_pass_seed_count": _count_true(seed_entries, "verified", "pass"),
        "internal_pass_seed_count": _count_true(seed_entries, "internal", "pass"),
        "verified_beats_internal_detection_seed_count": sum(
            1
            for verified, internal in zip(verified_detection, internal_detection, strict=True)
            if verified is not None and internal is not None and verified >= internal
        ),
        "route_signature_counts": _signature_counts(seed_entries),
        "selected_route_totals": _route_totals(seed_entries),
    }


def build_verifier_stability_report(
    score_dumps: Sequence[tuple[str, Path]],
    *,
    signal: str,
    claims_path: Path | None = None,
    qa_corpus_path: Path | None = None,
    state_path: Path | None = None,
    direction: str | None = None,
    alphas: Sequence[float] = ALPHAS,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    repeats: int = 20,
    best_alpha: float = 0.10,
    verifier_min_overlap: float = 0.65,
    retriever_min_overlap: float = 0.2,
    retrieval_limit: int = 5,
    selfcheck_min_samples: int = 2,
    selfcheck_min_overlap: float = 0.65,
    selfcheck_support_threshold: float = 0.60,
    selfcheck_refute_threshold: float = 0.50,
    selfcheck_early_stop: bool = False,
    selfcheck_max_samples: int | None = None,
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
) -> dict[str, Any]:
    """Build a compact stability report from verifier-ensemble runs."""
    if not score_dumps:
        raise ValueError("at least one score dump is required.")
    if not signal:
        raise ValueError("signal must be non-empty.")
    if not seeds:
        raise ValueError("at least one seed is required.")
    if repeats < 1:
        raise ValueError("repeats must be >= 1.")
    if any(not (0.0 < float(alpha) < 1.0) for alpha in alphas):
        raise ValueError("alphas must be in (0, 1).")
    if not _contains_alpha(alphas, best_alpha):
        raise ValueError("best_alpha must be included in alphas.")
    _validate_unique_score_dump_names(score_dumps)

    seed_reports = []
    seed_run_map: dict[str, list[dict[str, Any]]] = {name: [] for name, _ in score_dumps}
    source_runs: dict[str, Mapping[str, Any]] = {}
    top_level_reference: Mapping[str, Any] = {}
    for seed in seeds:
        seed_payload = build_verifier_ensemble_report(
            score_dumps,
            signal=signal,
            claims_path=claims_path,
            qa_corpus_path=qa_corpus_path,
            state_path=state_path,
            direction=direction,
            alphas=alphas,
            repeats=repeats,
            seed=int(seed),
            verifier_min_overlap=verifier_min_overlap,
            retriever_min_overlap=retriever_min_overlap,
            retrieval_limit=retrieval_limit,
            selfcheck_min_samples=selfcheck_min_samples,
            selfcheck_min_overlap=selfcheck_min_overlap,
            selfcheck_support_threshold=selfcheck_support_threshold,
            selfcheck_refute_threshold=selfcheck_refute_threshold,
            selfcheck_early_stop=selfcheck_early_stop,
            selfcheck_max_samples=selfcheck_max_samples,
            verification_cache_dir=verification_cache_dir,
            staged_verification=staged_verification,
            staged_alpha=staged_alpha,
            staged_verify_risk_levels=staged_verify_risk_levels,
            staged_verify_actions=staged_verify_actions,
            staged_verify_feature_flags=staged_verify_feature_flags,
            staged_verify_metadata_keys=staged_verify_metadata_keys,
        )
        if not top_level_reference:
            top_level_reference = seed_payload
        compact_runs = []
        for run_payload in seed_payload["runs"]:
            name = str(run_payload["name"])
            source_runs.setdefault(name, run_payload)
            entry = _compact_seed_run(run_payload, seed=int(seed), alpha=best_alpha)
            seed_run_map.setdefault(name, []).append(entry)
            compact_runs.append({"name": name, **entry})
        seed_reports.append({"seed": int(seed), "runs": compact_runs})

    runs = []
    for name, path in score_dumps:
        source_run = source_runs.get(name, {})
        seed_entries = seed_run_map.get(name, [])
        runs.append({
            "name": name,
            "scores_path": str(path),
            "score_dump": source_run.get("score_dump", {}),
            "config": source_run.get("config", {}),
            "signal": source_run.get("signal", signal),
            "direction": source_run.get("direction", direction),
            "n_total": source_run.get("n_total"),
            "n_true": source_run.get("n_true"),
            "n_false": source_run.get("n_false"),
            "verification_quality": source_run.get("verification_quality", {}),
            "route_quality": _compact_route_quality(source_run),
            "seed_runs": seed_entries,
            "stability": _summarize_seed_entries(seed_entries, alpha=best_alpha),
        })

    return {
        "schema_version": 1,
        "workflow": "verifier_stability",
        "status": "complete",
        "config": {
            "signal": signal,
            "direction": direction,
            "alphas": [float(alpha) for alpha in alphas],
            "seeds": [int(seed) for seed in seeds],
            "repeats": int(repeats),
            "best_alpha": float(best_alpha),
            "staged_verification": bool(staged_verification),
            "staged_alpha": float(staged_alpha),
        },
        "inputs": {
            "claims": None if claims_path is None else str(claims_path),
            "qa_corpus": None if qa_corpus_path is None else str(qa_corpus_path),
            "state_source": None if state_path is None else str(state_path),
            "verification_cache_dir": None if verification_cache_dir is None else str(verification_cache_dir),
        },
        "policy": top_level_reference.get("policy", {}),
        "qa_verifier": top_level_reference.get("qa_verifier", {}),
        "retrieval_qa_verifier": top_level_reference.get("retrieval_qa_verifier", {}),
        "state_verifier": top_level_reference.get("state_verifier", {}),
        "transition_verifier": top_level_reference.get("transition_verifier", {}),
        "retriever": top_level_reference.get("retriever", {}),
        "staged_verification": top_level_reference.get("staged_verification", {}),
        "seed_reports": seed_reports,
        "runs": runs,
    }


def _artifact_paths(
    *,
    output_path: Path,
    score_dumps: Sequence[tuple[str, Path]],
    claims_path: Path | None,
    qa_corpus_path: Path | None,
    state_path: Path | None,
    payload: Mapping[str, Any] | None = None,
) -> dict[str, str | Path | None]:
    artifacts: dict[str, str | Path | None] = {
        "verifier_stability_report": output_path,
        "claims": claims_path,
        "qa_corpus": qa_corpus_path,
        "state_source": state_path,
    }
    for name, path in score_dumps:
        artifacts[f"input_scores.{name}"] = path
    if payload is not None:
        for run in payload.get("runs", ()):
            if not isinstance(run, Mapping):
                continue
            name = str(run.get("name", "unknown"))
            score_dump = run.get("score_dump")
            if not isinstance(score_dump, Mapping):
                continue
            records = score_dump.get("records")
            if isinstance(records, Mapping) and records.get("path") is not None:
                artifacts[f"input_score_records.{name}"] = Path(str(records["path"]))
    return artifacts


def _write_manifest(
    *,
    manifest_path: Path,
    output_path: Path,
    score_dumps: Sequence[tuple[str, Path]],
    claims_path: Path | None,
    qa_corpus_path: Path | None,
    state_path: Path | None,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = build_artifact_manifest(
        _artifact_paths(
            output_path=output_path,
            score_dumps=score_dumps,
            claims_path=claims_path,
            qa_corpus_path=qa_corpus_path,
            state_path=state_path,
            payload=payload,
        ),
        root=manifest_path.parent,
        metadata={
            "runner": "eval_verifier_stability",
            "status": payload.get("status"),
            "seeds": tuple(payload.get("config", {}).get("seeds", ())),
            "best_alpha": payload.get("config", {}).get("best_alpha"),
            "staged_verification": payload.get("config", {}).get("staged_verification"),
        },
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _registry_run_summaries(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    summaries = []
    for run in payload.get("runs", ()):
        if not isinstance(run, Mapping):
            continue
        stability = run.get("stability")
        if not isinstance(stability, Mapping):
            stability = {}
        summaries.append({
            "name": run.get("name"),
            "verified_false_alarm_mean": (
                stability.get("verified_false_alarm", {}).get("mean")
                if isinstance(stability.get("verified_false_alarm"), Mapping)
                else None
            ),
            "verified_detection_mean": (
                stability.get("verified_detection", {}).get("mean")
                if isinstance(stability.get("verified_detection"), Mapping)
                else None
            ),
            "delta_detection_mean": (
                stability.get("delta_detection", {}).get("mean")
                if isinstance(stability.get("delta_detection"), Mapping)
                else None
            ),
            "verified_pass_seed_count": stability.get("verified_pass_seed_count"),
            "verified_beats_internal_detection_seed_count": stability.get(
                "verified_beats_internal_detection_seed_count"
            ),
            "route_signature_counts": stability.get("route_signature_counts"),
            "selected_route_totals": stability.get("selected_route_totals"),
        })
    return summaries


def _record_registry(
    *,
    registry_path: Path | None,
    name: str | None,
    version: str | None,
    output_path: Path,
    manifest_path: Path | None,
    payload: Mapping[str, Any],
) -> None:
    if registry_path is None:
        return
    if not name or not version:
        raise ValueError("--registry requires --name and --version.")
    registry = ArtifactRegistry.load_json(registry_path)
    registry.record_report(
        name=name,
        path=output_path,
        version=version,
        metadata={
            "workflow": "eval_verifier_stability",
            "status": payload.get("status"),
            "artifact_manifest": None if manifest_path is None else str(manifest_path),
            "signal": payload.get("config", {}).get("signal"),
            "seeds": tuple(payload.get("config", {}).get("seeds", ())),
            "best_alpha": payload.get("config", {}).get("best_alpha"),
            "staged_verification": payload.get("config", {}).get("staged_verification"),
            "runs": tuple(run.get("name") for run in payload.get("runs", ())),
            "run_summaries": _registry_run_summaries(payload),
        },
    )
    registry.save_json()


def run(args: argparse.Namespace) -> dict[str, Any]:
    score_dumps = [_parse_named_path(value) for value in args.scores]
    alphas = _parse_float_csv(args.alphas, name="alphas")
    seeds = _parse_int_csv(args.seeds, name="seeds")
    claims_path = None if args.claims is None else Path(args.claims)
    qa_corpus_path = None if args.qa_corpus is None else Path(args.qa_corpus)
    state_path = None if args.state_source is None else Path(args.state_source)
    output_path = Path(args.json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_verifier_stability_report(
        score_dumps,
        signal=str(args.signal),
        claims_path=claims_path,
        qa_corpus_path=qa_corpus_path,
        state_path=state_path,
        direction=args.direction,
        alphas=alphas,
        seeds=seeds,
        repeats=int(args.repeats),
        best_alpha=float(args.best_alpha),
        verifier_min_overlap=float(args.verifier_min_overlap),
        retriever_min_overlap=float(args.retriever_min_overlap),
        retrieval_limit=int(args.retrieval_limit),
        selfcheck_min_samples=int(args.selfcheck_min_samples),
        selfcheck_min_overlap=float(args.selfcheck_min_overlap),
        selfcheck_support_threshold=float(args.selfcheck_support_threshold),
        selfcheck_refute_threshold=float(args.selfcheck_refute_threshold),
        selfcheck_early_stop=bool(args.selfcheck_early_stop),
        selfcheck_max_samples=args.selfcheck_max_samples,
        verification_cache_dir=None if args.verification_cache_dir is None else Path(args.verification_cache_dir),
        staged_verification=bool(args.staged_verification),
        staged_alpha=float(args.staged_alpha),
        staged_verify_risk_levels=_parse_csv(args.staged_verify_risk_levels, name="staged_verify_risk_levels"),
        staged_verify_actions=_parse_csv(args.staged_verify_actions, name="staged_verify_actions"),
        staged_verify_feature_flags=_parse_csv(
            args.staged_verify_feature_flags,
            name="staged_verify_feature_flags",
        ),
        staged_verify_metadata_keys=_parse_csv(args.staged_verify_metadata_keys, name="staged_verify_metadata_keys"),
    )
    payload["paths"] = {"verifier_stability_report": str(output_path)}
    manifest_path = None if args.artifact_manifest is None else Path(args.artifact_manifest)
    if manifest_path is not None:
        payload["paths"]["artifact_manifest"] = str(manifest_path)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if manifest_path is not None:
        initial_manifest = _write_manifest(
            manifest_path=manifest_path,
            output_path=output_path,
            score_dumps=score_dumps,
            claims_path=claims_path,
            qa_corpus_path=qa_corpus_path,
            state_path=state_path,
            payload=payload,
        )
        payload["artifact_manifest_summary"] = initial_manifest["summary"]
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest = _write_manifest(
            manifest_path=manifest_path,
            output_path=output_path,
            score_dumps=score_dumps,
            claims_path=claims_path,
            qa_corpus_path=qa_corpus_path,
            state_path=state_path,
            payload=payload,
        )
        payload["artifact_manifest_summary"] = manifest["summary"]

    _record_registry(
        registry_path=None if args.registry is None else Path(args.registry),
        name=args.name,
        version=args.version,
        output_path=output_path,
        manifest_path=manifest_path,
        payload=payload,
    )
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate verifier stability across seeds")
    parser.add_argument("--scores", action="append", required=True, help="name=score_dump path; repeatable")
    parser.add_argument("--claims", default=None)
    parser.add_argument("--qa-corpus", default=None)
    parser.add_argument("--state-source", default=None)
    parser.add_argument("--signal", default="truth_proj")
    parser.add_argument("--direction", choices=("higher", "lower"), default=None)
    parser.add_argument("--alphas", default=",".join(str(alpha) for alpha in ALPHAS))
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--best-alpha", type=float, default=0.10)
    parser.add_argument("--verifier-min-overlap", type=float, default=0.65)
    parser.add_argument("--retriever-min-overlap", type=float, default=0.2)
    parser.add_argument("--retrieval-limit", type=int, default=5)
    parser.add_argument("--selfcheck-min-samples", type=int, default=2)
    parser.add_argument("--selfcheck-min-overlap", type=float, default=0.65)
    parser.add_argument("--selfcheck-support-threshold", type=float, default=0.60)
    parser.add_argument("--selfcheck-refute-threshold", type=float, default=0.50)
    parser.add_argument("--selfcheck-early-stop", action="store_true")
    parser.add_argument("--selfcheck-max-samples", type=int, default=None)
    parser.add_argument("--verification-cache-dir", default=None)
    parser.add_argument("--staged-verification", action="store_true")
    parser.add_argument("--staged-alpha", type=float, default=0.10)
    parser.add_argument("--staged-verify-risk-levels", default="medium,high,unknown")
    parser.add_argument(
        "--staged-verify-actions",
        default="retrieve,rewrite,steer_regenerate,execute_tool,abstain,clarify",
    )
    parser.add_argument("--staged-verify-feature-flags", default="has_number,has_citation,is_time_sensitive")
    parser.add_argument("--staged-verify-metadata-keys", default="requires_verification")
    parser.add_argument("--json", required=True)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    payload = run(parser.parse_args(argv))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

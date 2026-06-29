"""Build experiment queue items for blocked frontier abstention evidence."""

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

from eigentruth.eval.conformal import ABSTENTION_COMPARISON_METRICS  # noqa: E402
from eigentruth.json_utils import strict_json_dumps  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402

WORKFLOW = "frontier_abstention_evidence_rerun_queue"

DEFAULT_PROFILES = ("baseline", "alpha_0p05", "alpha_0p2", "selective_accuracy", "retention")
DEFAULT_SIGNAL_GROUPS = ("recommended", "all", "geometry", "uncertainty")
GEOMETRY_SIGNALS = ("maha_last", "truth_proj", "subspace_resid")
UNCERTAINTY_SIGNALS = ("disp_euclid", "disp_hse", "nll_answer", "eigenscore", "resid_update_norm")


def build_frontier_abstention_evidence_rerun_queue(
    *,
    source: str | Path,
    json_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    output_dir: str | Path | None = None,
    score_paths: Sequence[str | Path] = (),
    profiles: Sequence[str] = DEFAULT_PROFILES,
    signal_groups: Sequence[str] = DEFAULT_SIGNAL_GROUPS,
    seeds: str | None = None,
    min_conditional_correctness_lower_bound: float | None = None,
    max_abstention_rate: float | None = None,
    direction: str | None = None,
    python_executable: str = sys.executable,
) -> dict[str, Any]:
    """Load frontier/gap/abstention evidence and build abstention experiment commands."""
    if artifact_manifest_path is not None and json_path is None:
        raise ValueError("artifact_manifest_path requires json_path.")
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    if direction is not None and direction not in {"higher", "lower"}:
        raise ValueError("direction must be one of: higher, lower.")
    profile_names = _unique_strings(profiles)
    if not profile_names:
        raise ValueError("profiles must contain at least one profile.")
    signal_group_names = _unique_strings(signal_groups)
    if not signal_group_names:
        raise ValueError("signal_groups must contain at least one group.")

    source_path = Path(source)
    output_path = None if json_path is None else Path(json_path)
    manifest_path = None if artifact_manifest_path is None else Path(artifact_manifest_path)
    rerun_root = Path(output_dir) if output_dir is not None else source_path.parent / "frontier-abstention-reruns"
    payload = _load_json_object(source_path)
    source_dir = source_path.parent
    default_payload = _source_defaults_payload(payload, source_dir=source_dir)
    report_record = _load_abstention_report(payload, source_path=source_path, source_dir=source_dir)
    if report_record is None and default_payload is not payload:
        default_source = _resolve_source_path_from_plan(payload, source_dir=source_dir)
        if default_source is not None:
            report_record = _load_abstention_report(
                default_payload,
                source_path=default_source,
                source_dir=default_source.parent,
            )
    report_path, report = report_record if report_record is not None else (None, {})
    blocked_runs = _blocked_abstention_runs(payload) or _blocked_abstention_runs(default_payload)
    if not blocked_runs and report:
        blocked_runs = _blocked_abstention_runs(report)
    run_records = _run_records(report, blocked_runs=blocked_runs)

    entries = []
    for run in run_records:
        entries.extend(
            _entries_for_run(
                run,
                source_workflow=_optional_str(payload.get("workflow")),
                source_report=report_path,
                report=report,
                rerun_root=rerun_root,
                score_paths=score_paths,
                profile_names=profile_names,
                signal_group_names=signal_group_names,
                seeds=seeds,
                min_conditional_correctness_lower_bound=min_conditional_correctness_lower_bound,
                max_abstention_rate=max_abstention_rate,
                direction=direction,
                python_executable=python_executable,
            )
        )
    command_count = sum(1 for entry in entries if entry["command_status"] == "ready")
    output = {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": "ready" if entries else "empty",
        "source": str(source_path),
        "summary": {
            "blocked_run_count": len(run_records),
            "profile_count": len(profile_names),
            "signal_group_count": len(signal_group_names),
            "entry_count": len(entries),
            "command_count": command_count,
            "missing_command_count": len(entries) - command_count,
            "promotion_eligible_command_count": sum(
                1
                for entry in entries
                if entry["command_status"] == "ready"
                and entry["profile_config"].get("promotion_eligible") is True
            ),
        },
        "paths": {
            "rerun_queue": None if output_path is None else str(output_path),
            "artifact_manifest": None if manifest_path is None else str(manifest_path),
            "source_abstention_report": None if report_path is None else str(report_path),
        },
        "config": {
            "scores": tuple(str(path) for path in score_paths),
            "profiles": profile_names,
            "signal_groups": signal_group_names,
            "seeds": seeds,
            "direction": direction,
            "python_executable": python_executable,
        },
        "entries": tuple(entries),
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(strict_json_dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = None
    if manifest_path is not None:
        manifest = _write_artifact_manifest(
            source_path=source_path,
            output_path=output_path,
            manifest_path=manifest_path,
            payload=output,
        )
    if registry_path is not None:
        assert name is not None and version is not None
        _record_registry(
            registry_path=Path(registry_path),
            name=name,
            version=version,
            report_path=output_path if output_path is not None else source_path,
            manifest_path=manifest_path,
            payload=output,
            manifest=manifest,
        )
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    payload = build_frontier_abstention_evidence_rerun_queue(
        source=args.source,
        json_path=args.json,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        output_dir=args.output_dir,
        score_paths=tuple(args.scores or ()),
        profiles=_parse_csv(args.profiles, default=DEFAULT_PROFILES),
        signal_groups=_parse_csv(args.signal_groups, default=DEFAULT_SIGNAL_GROUPS),
        seeds=args.seeds,
        min_conditional_correctness_lower_bound=args.min_abstention_conditional_correctness_lower_bound,
        max_abstention_rate=args.max_abstention_rate,
        direction=args.direction,
        python_executable=args.python,
    )
    summary = payload["summary"]
    print(
        "frontier_abstention_evidence_rerun_queue="
        f"{payload['status']} "
        f"blocked_runs={summary['blocked_run_count']} "
        f"commands={summary['command_count']}"
    )
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="frontier release, gap-plan, or abstention report JSON")
    parser.add_argument("--json", default=None, help="optional output JSON path")
    parser.add_argument("--artifact-manifest", default=None, help="optional artifact manifest JSON path")
    parser.add_argument("--registry", default=None, help="optional local ArtifactRegistry JSON path")
    parser.add_argument("--name", default=None, help="registry record name")
    parser.add_argument("--version", default=None, help="registry record version")
    parser.add_argument("--output-dir", default=None, help="root directory for rerun outputs")
    parser.add_argument("--scores", action="append", default=[], help="name=score_dump path; repeatable")
    parser.add_argument("--profiles", default=",".join(DEFAULT_PROFILES))
    parser.add_argument("--signal-groups", default=",".join(DEFAULT_SIGNAL_GROUPS))
    parser.add_argument("--seeds", default=None, help="comma-separated seeds for generated commands")
    parser.add_argument(
        "--min-abstention-conditional-correctness-lower-bound",
        type=float,
        default=None,
    )
    parser.add_argument("--max-abstention-rate", type=float, default=None)
    parser.add_argument("--direction", choices=("higher", "lower"), default=None)
    parser.add_argument("--python", default=sys.executable, help="Python executable for generated commands")
    run(parser.parse_args(argv))


def _entries_for_run(
    run: Mapping[str, Any],
    *,
    source_workflow: str | None,
    source_report: Path | None,
    report: Mapping[str, Any],
    rerun_root: Path,
    score_paths: Sequence[str | Path],
    profile_names: Sequence[str],
    signal_group_names: Sequence[str],
    seeds: str | None,
    min_conditional_correctness_lower_bound: float | None,
    max_abstention_rate: float | None,
    direction: str | None,
    python_executable: str,
) -> tuple[dict[str, Any], ...]:
    run_name = _optional_str(run.get("name")) or "run"
    config = _mapping(report.get("config"))
    release_gate = _mapping(config.get("release_gate"))
    config_signals = _string_tuple(config.get("signals"))
    resolved_seeds = seeds or _csv_or_default(None, config.get("seeds"), "0,1,2,3,4")
    entries = []
    for profile_name in profile_names:
        profile_config = _profile_config(
            profile_name,
            config=config,
            release_gate=release_gate,
            min_conditional_correctness_lower_bound=min_conditional_correctness_lower_bound,
            max_abstention_rate=max_abstention_rate,
        )
        for group_name in signal_group_names:
            signals = _signal_group_signals(group_name, run=run, config_signals=config_signals)
            command, missing_inputs = _command_for_experiment(
                run_name=run_name,
                output_dir=rerun_root / _safe_path_part(run_name) / profile_name / _safe_path_part(group_name),
                report=report,
                run=run,
                score_paths=score_paths,
                signals=signals,
                seeds=resolved_seeds,
                profile_config=profile_config,
                direction=direction,
                python_executable=python_executable,
            )
            entries.append({
                "run": run_name,
                "profile": profile_name,
                "signal_group": group_name,
                "signals": signals,
                "source_workflow": source_workflow,
                "source_report": None if source_report is None else str(source_report),
                "source_metrics": _abstention_metrics(run),
                "command_kind": "abstention_stability_experiment",
                "command_status": "ready" if command is not None else "missing_inputs",
                "missing_inputs": missing_inputs if command is None else (),
                "profile_config": profile_config,
                "command": command,
                "dry_run_command": None,
            })
    return tuple(entries)


def _command_for_experiment(
    *,
    run_name: str,
    output_dir: Path,
    report: Mapping[str, Any],
    run: Mapping[str, Any],
    score_paths: Sequence[str | Path],
    signals: Sequence[str],
    seeds: str,
    profile_config: Mapping[str, Any],
    direction: str | None,
    python_executable: str,
) -> tuple[tuple[str, ...] | None, tuple[str, ...]]:
    scores = _score_args_for_run(run_name, run=run, report=report, score_paths=score_paths)
    missing_inputs = []
    if not scores:
        missing_inputs.append("scores")
    if not signals:
        missing_inputs.append("signals")
    if missing_inputs:
        return None, tuple(missing_inputs)
    command: list[str] = [
        python_executable,
        "benchmarks/eval_abstention_stability.py",
    ]
    for score in scores:
        command.extend(("--scores", score))
    command.extend((
        "--signals",
        ",".join(signals),
        "--alpha",
        str(profile_config["alpha"]),
        "--best-by",
        str(profile_config["best_by"]),
        "--seeds",
        str(seeds),
        "--min-abstention-conditional-correctness-lower-bound",
        str(profile_config["min_conditional_correctness_lower_bound"]),
        "--max-abstention-rate",
        str(profile_config["max_abstention_rate"]),
        "--json",
        str(output_dir / "abstention-stability-report.json"),
        "--artifact-manifest",
        str(output_dir / "artifact-manifest.json"),
    ))
    if direction is not None:
        command.extend(("--direction", direction))
    return tuple(command), ()


def _profile_config(
    profile: str,
    *,
    config: Mapping[str, Any],
    release_gate: Mapping[str, Any],
    min_conditional_correctness_lower_bound: float | None,
    max_abstention_rate: float | None,
) -> dict[str, Any]:
    base_alpha = _finite_float_or(config.get("alpha"), 0.10)
    base_best_by = _optional_str(config.get("best_by")) or "conditional_correctness_lower_bound"
    base_min_correct = _finite_float_or(
        min_conditional_correctness_lower_bound,
        _finite_float_or(release_gate.get("min_conditional_correctness_lower_bound"), 0.80),
    )
    base_max_abstention = _finite_float_or(
        max_abstention_rate,
        _finite_float_or(release_gate.get("max_abstention_rate"), 0.50),
    )
    profile = str(profile)
    alpha = base_alpha
    best_by = base_best_by
    if profile == "baseline":
        pass
    elif profile == "alpha_0p05":
        alpha = 0.05
    elif profile == "alpha_0p2":
        alpha = 0.20
    elif profile == "selective_accuracy":
        best_by = "empirical_selective_accuracy"
    elif profile == "retention":
        best_by = "correct_retention_lower_bound"
    else:
        raise ValueError(f"unknown abstention experiment profile: {profile!r}")
    if best_by not in ABSTENTION_COMPARISON_METRICS:
        raise ValueError(f"best_by must be one of {ABSTENTION_COMPARISON_METRICS}.")
    return {
        "profile": profile,
        "alpha": alpha,
        "best_by": best_by,
        "min_conditional_correctness_lower_bound": base_min_correct,
        "max_abstention_rate": base_max_abstention,
        "promotion_eligible": base_max_abstention <= _finite_float_or(
            release_gate.get("max_abstention_rate"),
            base_max_abstention,
        ),
    }


def _signal_group_signals(
    group: str,
    *,
    run: Mapping[str, Any],
    config_signals: Sequence[str],
) -> tuple[str, ...]:
    group = str(group)
    available = tuple(config_signals)
    if group == "all":
        return available
    if group == "recommended":
        recommended = _recommended_signal(run)
        return (recommended,) if recommended else ()
    if group == "geometry":
        return tuple(signal for signal in GEOMETRY_SIGNALS if signal in available)
    if group == "uncertainty":
        return tuple(signal for signal in UNCERTAINTY_SIGNALS if signal in available)
    explicit = tuple(part.strip() for part in group.split("+") if part.strip())
    if explicit:
        return tuple(signal for signal in explicit if not available or signal in available)
    return ()


def _recommended_signal(run: Mapping[str, Any]) -> str | None:
    stability = _mapping(run.get("stability"))
    stable = _optional_str(stability.get("stable_recommended_score_name"))
    if stable and stable != "<none>":
        return stable
    counts = _mapping(stability.get("recommended_score_name_counts"))
    ranked = sorted(
        (
            (int(count), str(score))
            for score, count in counts.items()
            if str(score) != "<none>"
        ),
        key=lambda item: (-item[0], item[1]),
    )
    return None if not ranked else ranked[0][1]


def _blocked_abstention_runs(payload: Mapping[str, Any]) -> tuple[str, ...]:
    workflow = payload.get("workflow")
    names: list[str] = []
    if workflow == "frontier_release_evidence_comparison":
        for decision in _mapping_sequence(payload.get("run_decisions", ())):
            abstention = _mapping(decision.get("abstention_decision"))
            if abstention.get("status") == "blocked":
                names.append(_optional_str(decision.get("name")) or _optional_str(abstention.get("name")) or "")
        if not names and _mapping(payload.get("decision")).get("abstention_track_status") == "blocked":
            names.extend(_string_tuple(_mapping(payload.get("evidence_summary")).get("run_names")))
    elif workflow == "evidence_gap_plan":
        for gap in _mapping_sequence(payload.get("gaps", ())):
            metadata = _mapping(gap.get("metadata"))
            if metadata.get("evidence_kind") == "abstention_stability" or gap.get("gate") == "abstention_stability":
                for row in _mapping_sequence(metadata.get("abstention_blocked_runs", ())):
                    names.append(_optional_str(row.get("run")) or "")
    elif workflow == "abstention_stability":
        for run in _mapping_sequence(payload.get("runs", ())):
            stability = _mapping(run.get("stability"))
            if stability.get("all_release_gates_passed") is False:
                names.append(_optional_str(run.get("name")) or "")
    return _unique_strings(names)


def _run_records(
    report: Mapping[str, Any],
    *,
    blocked_runs: Sequence[str],
) -> tuple[Mapping[str, Any], ...]:
    runs = tuple(_mapping_sequence(report.get("runs", ())))
    if not runs:
        return ()
    blocked = set(_unique_strings(blocked_runs))
    if blocked:
        selected = tuple(run for run in runs if _optional_str(run.get("name")) in blocked)
        if selected:
            return selected
    return tuple(
        run
        for run in runs
        if _mapping(run.get("stability")).get("all_release_gates_passed") is False
    )


def _abstention_metrics(run: Mapping[str, Any]) -> dict[str, Any]:
    stability = _mapping(run.get("stability"))
    feasibility = _mapping(run.get("supervised_feasibility_frontier"))
    feasible_best = _mapping(feasibility.get("best"))
    correctness = _mapping(stability.get("conditional_correctness_lower_bound"))
    abstention = _mapping(stability.get("empirical_abstention_rate"))
    return {
        "stable_recommended_score_name": stability.get("stable_recommended_score_name"),
        "recommended_score_name_counts": _mapping(stability.get("recommended_score_name_counts")),
        "release_gate_pass_seed_count": stability.get("release_gate_pass_seed_count"),
        "release_gate_block_seed_count": stability.get("release_gate_block_seed_count"),
        "conditional_correctness_lower_bound_mean": correctness.get("mean"),
        "empirical_abstention_rate_mean": abstention.get("mean"),
        "supervised_feasibility_target_passed": feasibility.get("target_passed"),
        "supervised_feasibility_score_name": feasible_best.get("score_name"),
        "supervised_feasibility_conditional_correctness_lower_bound": feasible_best.get(
            "conditional_correctness_lower_bound"
        ),
    }


def _load_abstention_report(
    payload: Mapping[str, Any],
    *,
    source_path: Path,
    source_dir: Path,
) -> tuple[Path, Mapping[str, Any]] | None:
    workflow = payload.get("workflow")
    if workflow == "abstention_stability":
        return source_path, payload
    if workflow == "evidence_gap_plan":
        source = _resolve_source_path_from_plan(payload, source_dir=source_dir)
        if source is not None:
            nested = _load_json_object(source)
            return _load_abstention_report(nested, source_path=source, source_dir=source.parent)
    inputs = _mapping(payload.get("inputs"))
    item = _mapping(inputs.get("abstention_stability_report"))
    raw_path = item.get("path")
    if raw_path is None:
        return None
    path = _resolve_report_path(Path(str(raw_path)), source_dir=source_dir)
    if path is None:
        return None
    report = _load_json_object(path)
    if report.get("workflow") != "abstention_stability":
        return None
    return path, report


def _source_defaults_payload(payload: Mapping[str, Any], *, source_dir: Path) -> Mapping[str, Any]:
    source = _resolve_source_path_from_plan(payload, source_dir=source_dir)
    if source is None:
        return payload
    return _load_json_object(source)


def _resolve_source_path_from_plan(payload: Mapping[str, Any], *, source_dir: Path) -> Path | None:
    if payload.get("workflow") != "evidence_gap_plan":
        return None
    source = _optional_str(payload.get("source_path"))
    if not source:
        return None
    return _resolve_report_path(Path(source), source_dir=source_dir)


def _score_args_for_run(
    run_name: str,
    *,
    run: Mapping[str, Any],
    report: Mapping[str, Any],
    score_paths: Sequence[str | Path],
) -> tuple[str, ...]:
    explicit = _score_path_map(score_paths)
    if run_name in explicit:
        return (f"{run_name}={explicit[run_name]}",)
    if "" in explicit and len(explicit) == 1:
        return (str(explicit[""]),)
    scores_path = _optional_str(run.get("scores_path"))
    if scores_path:
        return (f"{run_name}={scores_path}",)
    for item in _mapping_sequence(report.get("runs", ())):
        if _optional_str(item.get("name")) == run_name:
            scores_path = _optional_str(item.get("scores_path"))
            if scores_path:
                return (f"{run_name}={scores_path}",)
    return tuple(
        f"{name}={path}" if name else path
        for name, path in explicit.items()
    )


def _score_path_map(score_paths: Sequence[str | Path]) -> dict[str, str]:
    scores: dict[str, str] = {}
    for value in score_paths:
        text = str(value)
        if not text:
            continue
        if "=" not in text:
            scores[""] = text
            continue
        name, path = text.split("=", 1)
        name = name.strip()
        path = path.strip()
        if name and path:
            scores[name] = path
    return scores


def _write_artifact_manifest(
    *,
    source_path: Path,
    output_path: Path | None,
    manifest_path: Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    summary = _mapping(payload.get("summary"))
    manifest = build_artifact_manifest(
        {
            "source": source_path,
            "frontier_abstention_evidence_rerun_queue": output_path,
        },
        root=manifest_path.parent,
        metadata={
            "runner": "plan_frontier_abstention_evidence_reruns",
            "status": payload.get("status"),
            "source": str(source_path),
            "blocked_run_count": summary.get("blocked_run_count"),
            "command_count": summary.get("command_count"),
            "missing_command_count": summary.get("missing_command_count"),
        },
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(strict_json_dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _record_registry(
    *,
    registry_path: Path,
    name: str,
    version: str,
    report_path: Path,
    manifest_path: Path | None,
    payload: Mapping[str, Any],
    manifest: Mapping[str, Any] | None,
) -> None:
    summary = _mapping(payload.get("summary"))
    ArtifactRegistry.load_json(registry_path).record_report(
        name=name,
        version=version,
        path=report_path,
        metadata={
            "workflow": WORKFLOW,
            "status": payload.get("status"),
            "source": payload.get("source"),
            "blocked_run_count": summary.get("blocked_run_count"),
            "profile_count": summary.get("profile_count"),
            "signal_group_count": summary.get("signal_group_count"),
            "command_count": summary.get("command_count"),
            "missing_command_count": summary.get("missing_command_count"),
            "artifact_manifest": None if manifest_path is None else str(manifest_path),
            "manifest_summary": None if manifest is None else _mapping(manifest.get("summary")),
        },
    ).save_json()


def _load_json_object(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"source JSON must contain an object: {path}")
    return data


def _resolve_report_path(path: Path, *, source_dir: Path) -> Path | None:
    if path.is_absolute():
        return path if path.exists() else None
    candidates = (source_dir / path, ROOT / path, Path.cwd() / path)
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
    return None


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _mapping_sequence(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,) if value else ()
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        return ()
    return tuple(str(item) for item in value if str(item))


def _unique_strings(values: Sequence[Any]) -> tuple[str, ...]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return tuple(output)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_csv(value: str | None, *, default: Sequence[str]) -> tuple[str, ...]:
    if value is None:
        return tuple(default)
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _csv_or_default(value: Any, fallback: Any, default: str) -> str:
    if value is not None:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return ",".join(str(item) for item in value)
        return str(value)
    if fallback is not None:
        if isinstance(fallback, Sequence) and not isinstance(fallback, (str, bytes, bytearray)):
            return ",".join(str(item) for item in fallback)
        return str(fallback)
    return default


def _finite_float_or(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    if result != result or result in {float("inf"), float("-inf")}:
        return float(default)
    return result


def _safe_path_part(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in value)
    return cleaned.strip("-") or "value"


if __name__ == "__main__":
    main()

"""Build rerun queue items for blocked frontier stability evidence tracks."""

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

WORKFLOW = "frontier_stability_evidence_rerun_queue"


def build_frontier_stability_evidence_rerun_queue(
    *,
    source: str | Path,
    json_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    output_dir: str | Path | None = None,
    score_paths: Sequence[str | Path] = (),
    seeds: str | None = None,
    verifier_signal: str | None = None,
    verifier_claims_path: str | Path | None = None,
    verifier_qa_corpus_path: str | Path | None = None,
    verifier_state_source_path: str | Path | None = None,
    verifier_direction: str | None = None,
    verifier_alphas: str | None = None,
    verifier_best_alpha: float | None = None,
    verifier_repeats: int | None = None,
    verifier_staged_verification: bool = True,
    verifier_staged_alpha: float | None = None,
    verifier_verification_cache_dir: str | Path | None = None,
    abstention_signals: Sequence[str] = (),
    abstention_alpha: float | None = None,
    abstention_best_by: str | None = None,
    abstention_direction: str | None = None,
    min_abstention_conditional_correctness_lower_bound: float | None = None,
    max_abstention_rate: float | None = None,
    python_executable: str = sys.executable,
) -> dict[str, Any]:
    """Load frontier/gap evidence and build one queue row per blocked stability track."""
    if artifact_manifest_path is not None and json_path is None:
        raise ValueError("artifact_manifest_path requires json_path.")
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    source_path = Path(source)
    output_path = None if json_path is None else Path(json_path)
    manifest_path = None if artifact_manifest_path is None else Path(artifact_manifest_path)
    rerun_root = Path(output_dir) if output_dir is not None else source_path.parent / "frontier-stability-reruns"
    payload = _load_json_object(source_path)
    source_dir = source_path.parent
    reports = _load_stability_reports(payload, source_path=source_path, source_dir=source_dir)
    default_payload = _source_defaults_payload(payload, source_dir=source_dir)
    tracks = _blocked_tracks_from_payload(payload)
    if not tracks and payload.get("workflow") in {"verifier_stability", "abstention_stability"}:
        tracks = (str(payload["workflow"]),)
    evidence_summary = _mapping(payload.get("evidence_summary")) or _mapping(default_payload.get("evidence_summary"))
    resolved_verifier_signal = verifier_signal or _optional_str(evidence_summary.get("verifier_signal"))
    resolved_abstention_signals = _unique_strings(abstention_signals) or _string_tuple(
        evidence_summary.get("abstention_signals")
    )

    entries = tuple(
        _queue_entry(
            track,
            source_workflow=_optional_str(payload.get("workflow")),
            rerun_root=rerun_root,
            reports=reports,
            score_paths=score_paths,
            seeds=seeds,
            verifier_signal=resolved_verifier_signal,
            verifier_claims_path=verifier_claims_path,
            verifier_qa_corpus_path=verifier_qa_corpus_path,
            verifier_state_source_path=verifier_state_source_path,
            verifier_direction=verifier_direction,
            verifier_alphas=verifier_alphas,
            verifier_best_alpha=verifier_best_alpha,
            verifier_repeats=verifier_repeats,
            verifier_staged_verification=verifier_staged_verification,
            verifier_staged_alpha=verifier_staged_alpha,
            verifier_verification_cache_dir=verifier_verification_cache_dir,
            abstention_signals=resolved_abstention_signals,
            abstention_alpha=abstention_alpha,
            abstention_best_by=abstention_best_by,
            abstention_direction=abstention_direction,
            min_abstention_conditional_correctness_lower_bound=(
                min_abstention_conditional_correctness_lower_bound
            ),
            max_abstention_rate=max_abstention_rate,
            python_executable=python_executable,
        )
        for track in tracks
    )
    command_count = sum(1 for entry in entries if entry["command_status"] == "ready")
    output = {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": "ready" if entries else "empty",
        "source": str(source_path),
        "summary": {
            "blocked_track_count": len(entries),
            "verifier_track_count": sum(1 for entry in entries if entry["track"] == "verifier_stability"),
            "abstention_track_count": sum(1 for entry in entries if entry["track"] == "abstention_stability"),
            "command_count": command_count,
            "missing_command_count": len(entries) - command_count,
        },
        "paths": {
            "rerun_queue": None if output_path is None else str(output_path),
            "artifact_manifest": None if manifest_path is None else str(manifest_path),
        },
        "config": {
            "scores": tuple(str(path) for path in score_paths),
            "seeds": seeds,
            "verifier_signal": resolved_verifier_signal,
            "verifier_claims": None if verifier_claims_path is None else str(verifier_claims_path),
            "verifier_qa_corpus": None if verifier_qa_corpus_path is None else str(verifier_qa_corpus_path),
            "verifier_state_source": None if verifier_state_source_path is None else str(verifier_state_source_path),
            "verifier_staged_verification": bool(verifier_staged_verification),
            "abstention_signals": resolved_abstention_signals,
            "python_executable": python_executable,
        },
        "entries": entries,
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
    payload = build_frontier_stability_evidence_rerun_queue(
        source=args.source,
        json_path=args.json,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        output_dir=args.output_dir,
        score_paths=tuple(args.scores or ()),
        seeds=args.seeds,
        verifier_signal=args.verifier_signal,
        verifier_claims_path=args.verifier_claims,
        verifier_qa_corpus_path=args.verifier_qa_corpus,
        verifier_state_source_path=args.verifier_state_source,
        verifier_direction=args.verifier_direction,
        verifier_alphas=args.verifier_alphas,
        verifier_best_alpha=args.verifier_best_alpha,
        verifier_repeats=args.verifier_repeats,
        verifier_staged_verification=bool(args.verifier_staged_verification),
        verifier_staged_alpha=args.verifier_staged_alpha,
        verifier_verification_cache_dir=args.verifier_verification_cache_dir,
        abstention_signals=_parse_csv(args.abstention_signals),
        abstention_alpha=args.abstention_alpha,
        abstention_best_by=args.abstention_best_by,
        abstention_direction=args.abstention_direction,
        min_abstention_conditional_correctness_lower_bound=(
            args.min_abstention_conditional_correctness_lower_bound
        ),
        max_abstention_rate=args.max_abstention_rate,
        python_executable=args.python,
    )
    summary = payload["summary"]
    print(
        "frontier_stability_evidence_rerun_queue="
        f"{payload['status']} "
        f"blocked_tracks={summary['blocked_track_count']} "
        f"commands={summary['command_count']}"
    )
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="frontier release, stability, or evidence-gap JSON")
    parser.add_argument("--json", default=None, help="optional output JSON path")
    parser.add_argument("--artifact-manifest", default=None, help="optional artifact manifest JSON path")
    parser.add_argument("--registry", default=None, help="optional local ArtifactRegistry JSON path")
    parser.add_argument("--name", default=None, help="registry record name")
    parser.add_argument("--version", default=None, help="registry record version")
    parser.add_argument("--output-dir", default=None, help="root directory for rerun outputs")
    parser.add_argument("--scores", action="append", default=[], help="name=score_dump path; repeatable")
    parser.add_argument("--seeds", default=None, help="comma-separated seeds for generated stability commands")
    parser.add_argument("--verifier-signal", default=None)
    parser.add_argument("--verifier-claims", default=None)
    parser.add_argument("--verifier-qa-corpus", default=None)
    parser.add_argument("--verifier-state-source", default=None)
    parser.add_argument("--verifier-direction", choices=("higher", "lower"), default=None)
    parser.add_argument("--verifier-alphas", default=None)
    parser.add_argument("--verifier-best-alpha", type=float, default=None)
    parser.add_argument("--verifier-repeats", type=int, default=None)
    parser.add_argument(
        "--verifier-staged-verification",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--verifier-staged-alpha", type=float, default=None)
    parser.add_argument("--verifier-verification-cache-dir", default=None)
    parser.add_argument("--abstention-signals", default=None)
    parser.add_argument("--abstention-alpha", type=float, default=None)
    parser.add_argument("--abstention-best-by", default=None)
    parser.add_argument("--abstention-direction", choices=("higher", "lower"), default=None)
    parser.add_argument(
        "--min-abstention-conditional-correctness-lower-bound",
        type=float,
        default=None,
    )
    parser.add_argument("--max-abstention-rate", type=float, default=None)
    parser.add_argument("--python", default=sys.executable, help="Python executable for generated commands")
    run(parser.parse_args(argv))


def _blocked_tracks_from_payload(payload: Mapping[str, Any]) -> tuple[str, ...]:
    workflow = payload.get("workflow")
    tracks: list[str] = []
    if workflow == "frontier_release_evidence_comparison":
        decision = _mapping(payload.get("decision"))
        if decision.get("verifier_track_status") == "blocked":
            tracks.append("verifier_stability")
        if decision.get("abstention_track_status") == "blocked":
            tracks.append("abstention_stability")
    elif workflow == "evidence_gap_plan":
        for gap in _mapping_sequence(payload.get("gaps", ())):
            metadata = _mapping(gap.get("metadata"))
            evidence_kind = _optional_str(metadata.get("evidence_kind"))
            gate = _optional_str(gap.get("gate"))
            if evidence_kind == "verifier_stability" or gate == "verifier_stability":
                tracks.append("verifier_stability")
            elif evidence_kind == "abstention_stability" or gate == "abstention_stability":
                tracks.append("abstention_stability")
    return _unique_strings(tracks)


def _queue_entry(
    track: str,
    *,
    source_workflow: str | None,
    rerun_root: Path,
    reports: Mapping[str, tuple[Path, Mapping[str, Any]]],
    score_paths: Sequence[str | Path],
    seeds: str | None,
    verifier_signal: str | None,
    verifier_claims_path: str | Path | None,
    verifier_qa_corpus_path: str | Path | None,
    verifier_state_source_path: str | Path | None,
    verifier_direction: str | None,
    verifier_alphas: str | None,
    verifier_best_alpha: float | None,
    verifier_repeats: int | None,
    verifier_staged_verification: bool,
    verifier_staged_alpha: float | None,
    verifier_verification_cache_dir: str | Path | None,
    abstention_signals: Sequence[str],
    abstention_alpha: float | None,
    abstention_best_by: str | None,
    abstention_direction: str | None,
    min_abstention_conditional_correctness_lower_bound: float | None,
    max_abstention_rate: float | None,
    python_executable: str,
) -> dict[str, Any]:
    report_path, report = reports.get(track, (None, {}))  # type: ignore[assignment]
    output_dir = rerun_root / track
    if track == "verifier_stability":
        command, missing_inputs = _verifier_command(
            output_dir=output_dir,
            report=report,
            score_paths=score_paths,
            seeds=seeds,
            signal=verifier_signal,
            claims_path=verifier_claims_path,
            qa_corpus_path=verifier_qa_corpus_path,
            state_source_path=verifier_state_source_path,
            direction=verifier_direction,
            alphas=verifier_alphas,
            best_alpha=verifier_best_alpha,
            repeats=verifier_repeats,
            staged_verification=verifier_staged_verification,
            staged_alpha=verifier_staged_alpha,
            verification_cache_dir=verifier_verification_cache_dir,
            python_executable=python_executable,
        )
    elif track == "abstention_stability":
            command, missing_inputs = _abstention_command(
                output_dir=output_dir,
                report=report,
                score_paths=score_paths,
            seeds=seeds,
            signals=abstention_signals,
            alpha=abstention_alpha,
            best_by=abstention_best_by,
            direction=abstention_direction,
            min_conditional_correctness_lower_bound=min_abstention_conditional_correctness_lower_bound,
            max_abstention_rate=max_abstention_rate,
            python_executable=python_executable,
        )
    else:
        command, missing_inputs = None, ("unsupported_track",)
    return {
        "track": track,
        "source_workflow": source_workflow,
        "source_report": None if report_path is None else str(report_path),
        "rerun_output_dir": str(output_dir),
        "command_status": "ready" if command is not None else "missing_inputs",
        "missing_inputs": missing_inputs if command is None else (),
        "command": command,
        "dry_run_command": None,
    }


def _verifier_command(
    *,
    output_dir: Path,
    report: Mapping[str, Any],
    score_paths: Sequence[str | Path],
    seeds: str | None,
    signal: str | None,
    claims_path: str | Path | None,
    qa_corpus_path: str | Path | None,
    state_source_path: str | Path | None,
    direction: str | None,
    alphas: str | None,
    best_alpha: float | None,
    repeats: int | None,
    staged_verification: bool,
    staged_alpha: float | None,
    verification_cache_dir: str | Path | None,
    python_executable: str,
) -> tuple[tuple[str, ...] | None, tuple[str, ...]]:
    config = _mapping(report.get("config"))
    scores = _score_args(score_paths, report=report)
    resolved_signal = _optional_str(signal) or _optional_str(config.get("signal")) or "truth_proj"
    if not scores:
        return None, ("scores",)
    output_dir = Path(output_dir)
    command: list[str] = [
        python_executable,
        "benchmarks/eval_verifier_stability.py",
    ]
    _append_scores(command, scores)
    command.extend((
        "--signal",
        resolved_signal,
        "--alphas",
        _csv_or_default(alphas, config.get("alphas"), "0.05,0.1,0.2"),
        "--seeds",
        seeds or _csv_or_default(None, config.get("seeds"), "0,1,2,3,4"),
        "--repeats",
        str(repeats if repeats is not None else config.get("repeats", 20)),
        "--best-alpha",
        str(best_alpha if best_alpha is not None else config.get("best_alpha", 0.10)),
        "--json",
        str(output_dir / "verifier-stability-report.json"),
        "--artifact-manifest",
        str(output_dir / "artifact-manifest.json"),
    ))
    _append_optional_value(command, "--claims", claims_path or _nested(report, "inputs", "claims"))
    _append_optional_value(command, "--qa-corpus", qa_corpus_path or _nested(report, "inputs", "qa_corpus"))
    _append_optional_value(command, "--state-source", state_source_path or _nested(report, "inputs", "state_source"))
    _append_optional_value(command, "--direction", direction or config.get("direction"))
    if staged_verification:
        command.append("--staged-verification")
    _append_optional_value(command, "--staged-alpha", staged_alpha or config.get("staged_alpha"))
    _append_optional_value(command, "--verification-cache-dir", verification_cache_dir)
    return tuple(command), ()


def _abstention_command(
    *,
    output_dir: Path,
    report: Mapping[str, Any],
    score_paths: Sequence[str | Path],
    seeds: str | None,
    signals: Sequence[str],
    alpha: float | None,
    best_by: str | None,
    direction: str | None,
    min_conditional_correctness_lower_bound: float | None,
    max_abstention_rate: float | None,
    python_executable: str,
) -> tuple[tuple[str, ...] | None, tuple[str, ...]]:
    config = _mapping(report.get("config"))
    scores = _score_args(score_paths, report=report)
    resolved_signals = _unique_strings(signals) or _string_tuple(config.get("signals"))
    missing_inputs = []
    if not scores:
        missing_inputs.append("scores")
    if not resolved_signals:
        missing_inputs.append("abstention_signals")
    if missing_inputs:
        return None, tuple(missing_inputs)
    output_dir = Path(output_dir)
    release_gate = _mapping(config.get("release_gate"))
    command: list[str] = [
        python_executable,
        "benchmarks/eval_abstention_stability.py",
    ]
    _append_scores(command, scores)
    command.extend((
        "--signals",
        ",".join(resolved_signals),
        "--alpha",
        str(alpha if alpha is not None else config.get("alpha", 0.10)),
        "--best-by",
        str(best_by or config.get("best_by") or "conditional_correctness_lower_bound"),
        "--seeds",
        seeds or _csv_or_default(None, config.get("seeds"), "0,1,2,3,4"),
        "--min-abstention-conditional-correctness-lower-bound",
        str(
            min_conditional_correctness_lower_bound
            if min_conditional_correctness_lower_bound is not None
            else release_gate.get("min_conditional_correctness_lower_bound", 0.80)
        ),
        "--max-abstention-rate",
        str(max_abstention_rate if max_abstention_rate is not None else release_gate.get("max_abstention_rate", 0.50)),
        "--json",
        str(output_dir / "abstention-stability-report.json"),
        "--artifact-manifest",
        str(output_dir / "artifact-manifest.json"),
    ))
    _append_optional_value(command, "--direction", direction)
    return tuple(command), ()


def _load_stability_reports(
    payload: Mapping[str, Any],
    *,
    source_path: Path,
    source_dir: Path,
) -> dict[str, tuple[Path, Mapping[str, Any]]]:
    reports: dict[str, tuple[Path, Mapping[str, Any]]] = {}
    workflow = payload.get("workflow")
    if workflow in {"verifier_stability", "abstention_stability"}:
        reports[str(workflow)] = (source_path, payload)
    if workflow == "evidence_gap_plan":
        source = _optional_str(payload.get("source_path"))
        if source:
            resolved = _resolve_report_path(Path(source), source_dir=source_dir)
            if resolved is not None:
                nested_payload = _load_json_object(resolved)
                reports.update(
                    _load_stability_reports(
                        nested_payload,
                        source_path=resolved,
                        source_dir=resolved.parent,
                    )
                )
    inputs = _mapping(payload.get("inputs"))
    for key, track in (
        ("verifier_stability_report", "verifier_stability"),
        ("abstention_stability_report", "abstention_stability"),
    ):
        item = _mapping(inputs.get(key))
        raw_path = item.get("path")
        if raw_path is None:
            continue
        path = _resolve_report_path(Path(str(raw_path)), source_dir=source_dir)
        if path is None:
            continue
        report = _load_json_object(path)
        if report.get("workflow") == track:
            reports[track] = (path, report)
    return reports


def _source_defaults_payload(payload: Mapping[str, Any], *, source_dir: Path) -> Mapping[str, Any]:
    if payload.get("workflow") != "evidence_gap_plan":
        return payload
    source = _optional_str(payload.get("source_path"))
    if not source:
        return payload
    resolved = _resolve_report_path(Path(source), source_dir=source_dir)
    if resolved is None:
        return payload
    return _load_json_object(resolved)


def _score_args(score_paths: Sequence[str | Path], *, report: Mapping[str, Any]) -> tuple[str, ...]:
    explicit = tuple(str(path) for path in score_paths if str(path))
    if explicit:
        return explicit
    scores = []
    for run in _mapping_sequence(report.get("runs", ())):
        name = _optional_str(run.get("name"))
        scores_path = _optional_str(run.get("scores_path"))
        if scores_path is None:
            continue
        scores.append(scores_path if name is None else f"{name}={scores_path}")
    return _unique_strings(scores)


def _append_scores(command: list[str], scores: Sequence[str]) -> None:
    for score_path in scores:
        command.extend(("--scores", str(score_path)))


def _append_optional_value(command: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    text = str(value).strip()
    if text:
        command.extend((flag, text))


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
            "frontier_stability_evidence_rerun_queue": output_path,
        },
        root=manifest_path.parent,
        metadata={
            "runner": "plan_frontier_stability_evidence_reruns",
            "status": payload.get("status"),
            "source": str(source_path),
            "blocked_track_count": summary.get("blocked_track_count"),
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
            "blocked_track_count": summary.get("blocked_track_count"),
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
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(str(item) for item in value if str(item).strip())
    return (str(value),)


def _parse_csv(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _csv_or_default(raw: str | None, value: Any, default: str) -> str:
    if raw:
        return raw
    values = _string_tuple(value)
    return ",".join(values) if values else default


def _unique_strings(values: Sequence[Any]) -> tuple[str, ...]:
    seen: set[str] = set()
    unique = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return tuple(unique)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _nested(payload: Mapping[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


if __name__ == "__main__":
    main()

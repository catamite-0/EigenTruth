"""Evaluate DECK-style detectability taxonomy from saved score dumps.

This helper consumes existing ``eval_truthfulqa.py --dump-scores`` artifacts
and does not load a model. It compares one consistency-style signal and one
confidence-style signal, then partitions samples into Drift / Entrenched /
Confabulation / Knotted detectability cells.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eigentruth.eval.metrics import deck_taxonomy_report  # noqa: E402
from eigentruth.eval.score_dump import (  # noqa: E402
    load_score_dump_columns,
    score_dump_cache_summary,
    score_dump_file_metadata,
)
from eigentruth.registry import build_artifact_manifest  # noqa: E402


def build_detectability_taxonomy_report(
    *,
    score_dump_path: str | Path,
    consistency_signal: str,
    confidence_signal: str,
    consistency_direction: str = "higher",
    confidence_direction: str = "higher",
    include_assignments: bool = False,
    metadata: Mapping[str, Any] | None = None,
    cache: MutableMapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-ready detectability taxonomy report from a score dump."""
    score_path = Path(score_dump_path)
    if consistency_signal == confidence_signal:
        raise ValueError("consistency_signal and confidence_signal must differ.")
    dump = load_score_dump_columns(
        score_path,
        (consistency_signal, confidence_signal),
        cache=cache,
    )
    report = deck_taxonomy_report(
        dump.scores[consistency_signal],
        dump.scores[confidence_signal],
        dump.labels,
        consistency_direction=consistency_direction,
        confidence_direction=confidence_direction,
        include_assignments=include_assignments,
    )
    return {
        "schema_version": 1,
        "workflow": "detectability_taxonomy",
        "status": "complete",
        "source": {
            "score_dump_path": str(score_path),
            "score_dump_file": score_dump_file_metadata(score_path),
            "score_dump_summary": dict(dump.summary),
            "score_dump_source_format": dump.source_format,
        },
        "config": {
            "consistency_signal": consistency_signal,
            "confidence_signal": confidence_signal,
            "consistency_direction": consistency_direction,
            "confidence_direction": confidence_direction,
            "include_assignments": bool(include_assignments),
        },
        "report": report,
        "score_dump_cache": score_dump_cache_summary(cache),
        "metadata": dict(metadata or {}),
    }


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


def _write_json(path: str | Path, payload: Mapping[str, Any], *, compact: bool = False) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    else:
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    output.write_text(text, encoding="utf-8")


def _artifact_paths(
    *,
    output_path: Path,
    score_dump_path: Path,
    payload: Mapping[str, Any],
) -> dict[str, Path]:
    artifacts = {
        "detectability_taxonomy_report": output_path,
        "input_score_dump": score_dump_path,
    }
    source = payload.get("source")
    if isinstance(source, Mapping):
        metadata = source.get("score_dump_file")
        if isinstance(metadata, Mapping):
            records = metadata.get("records")
            if isinstance(records, Mapping) and records.get("path") is not None:
                artifacts["input_score_records"] = Path(str(records["path"]))
    return artifacts


def write_detectability_artifact_manifest(
    *,
    manifest_path: Path,
    output_path: Path,
    score_dump_path: Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = build_artifact_manifest(
        _artifact_paths(
            output_path=output_path,
            score_dump_path=score_dump_path,
            payload=payload,
        ),
        root=manifest_path.parent,
        metadata={
            "runner": "eval_detectability_taxonomy",
            "status": payload.get("status"),
            "consistency_signal": payload.get("config", {}).get("consistency_signal"),
            "confidence_signal": payload.get("config", {}).get("confidence_signal"),
        },
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(manifest_path, manifest)
    return manifest


def _write_artifact_manifest(
    *,
    manifest_path: Path,
    output_path: Path,
    score_dump_path: Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    return write_detectability_artifact_manifest(
        manifest_path=manifest_path,
        output_path=output_path,
        score_dump_path=score_dump_path,
        payload=payload,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True, help="Score dump JSON or JSONL manifest path.")
    parser.add_argument("--consistency-signal", required=True)
    parser.add_argument("--confidence-signal", required=True)
    parser.add_argument("--consistency-direction", choices=("higher", "lower"), default="higher")
    parser.add_argument("--confidence-direction", choices=("higher", "lower"), default="higher")
    parser.add_argument("--include-assignments", action="store_true")
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--json", required=True, help="Output report path.")
    parser.add_argument("--artifact-manifest", default=None, help="Optional artifact manifest path.")
    parser.add_argument("--compact-json", action="store_true")
    args = parser.parse_args(argv)

    cache: dict[str, Any] = {}
    payload = build_detectability_taxonomy_report(
        score_dump_path=args.scores,
        consistency_signal=args.consistency_signal,
        confidence_signal=args.confidence_signal,
        consistency_direction=args.consistency_direction,
        confidence_direction=args.confidence_direction,
        include_assignments=bool(args.include_assignments),
        metadata=_parse_metadata(args.metadata or ()),
        cache=cache,
    )
    output_path = Path(args.json)
    score_dump_path = Path(args.scores)
    payload["paths"] = {"detectability_taxonomy_report": str(output_path)}
    manifest_path = None if args.artifact_manifest is None else Path(args.artifact_manifest)
    if manifest_path is not None:
        payload["paths"]["artifact_manifest"] = str(manifest_path)
    _write_json(output_path, payload, compact=bool(args.compact_json))
    if manifest_path is not None:
        initial_manifest = _write_artifact_manifest(
            manifest_path=manifest_path,
            output_path=output_path,
            score_dump_path=score_dump_path,
            payload=payload,
        )
        payload["artifact_manifest_summary"] = initial_manifest["summary"]
        _write_json(output_path, payload, compact=bool(args.compact_json))
        manifest = _write_artifact_manifest(
            manifest_path=manifest_path,
            output_path=output_path,
            score_dump_path=score_dump_path,
            payload=payload,
        )
        payload["artifact_manifest_summary"] = manifest["summary"]
        _write_json(output_path, payload, compact=bool(args.compact_json))
    report = payload["report"]
    blind_spot = report["blind_spot"]
    print(
        "detectability_taxonomy="
        f"records={report['n_total']} false={report['n_false']} "
        f"entrenched_false={blind_spot['n_false']}"
    )


if __name__ == "__main__":
    main()

"""Export sampled response texts from an INSIDE diagnostics cache.

``eval_truthfulqa.py --inside-diagnostics-cache`` stores sampled continuations
even when the score dump was not written with ``--dump-inside-samples``. This
script reconstructs the statement-level diagnostics cache keys from a saved
score dump and writes a ``build_selfcheck_fixture.py``-compatible samples file.
It does not load models or regenerate samples.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.eval_truthfulqa import (  # noqa: E402
    InsideDiagnosticsCache,
    Statement,
    _inside_diagnostics_cache_key,
)
from eigentruth.eval.score_dump import load_score_dump  # noqa: E402
from eigentruth.registry import build_artifact_manifest  # noqa: E402


def export_inside_diagnostics_samples(
    *,
    scores_path: str | Path,
    inside_diagnostics_cache_path: str | Path,
    output_path: str | Path,
    min_samples: int = 1,
    include_empty_records: bool = True,
    artifact_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Export sampled texts from a cache into an aligned samples payload."""
    if int(min_samples) < 1:
        raise ValueError("min_samples must be >= 1.")
    score_dump = load_score_dump(
        scores_path,
        allow_missing_scores=True,
        require_statements=True,
    ).to_mapping()
    labels = tuple(int(label) for label in score_dump.get("labels", ()))
    statements = tuple(_statement_mapping(item) for item in score_dump.get("statements", ()))
    if len(labels) != len(statements):
        raise ValueError("score dump labels and statements must have the same length.")

    config = _mapping(score_dump.get("config"))
    args = _cache_key_args_from_config(config)
    layers = _layers_from_config(config)
    adaptive = bool(config.get("inside_adaptive_sampling", False))
    selfcheck_early_stop = bool(config.get("inside_selfcheck_early_stop", False))
    cache = InsideDiagnosticsCache(inside_diagnostics_cache_path)

    records = []
    matched_records = 0
    records_meeting_min_samples = 0
    total_samples = 0
    dropped_records = 0
    for idx, (label, statement) in enumerate(zip(labels, statements)):
        stmt = _statement_from_mapping(statement, label=label)
        cache_key = _inside_diagnostics_cache_key(
            stmt,
            args,
            layers=layers,
            adaptive=adaptive,
            selfcheck_early_stop=selfcheck_early_stop,
        )
        diagnostics = cache.get(cache_key)
        sample_texts = tuple(diagnostics.sample_texts) if diagnostics is not None else ()
        if diagnostics is not None:
            matched_records += 1
        non_empty_sample_count = sum(1 for text in sample_texts if str(text).strip())
        meets_min = non_empty_sample_count >= int(min_samples)
        if meets_min:
            records_meeting_min_samples += 1
        if not include_empty_records and not meets_min:
            dropped_records += 1
            continue
        total_samples += non_empty_sample_count
        records.append({
            "index": idx,
            "claim_id": str(statement.get("claim_id") or f"c{idx + 1}"),
            "sampled_responses": list(sample_texts),
            "metadata": {
                "score_label": label,
                "statement": statement,
                "inside_diagnostics_cache_key": cache_key,
                "cache_hit": diagnostics is not None,
                "n_samples": 0 if diagnostics is None else int(diagnostics.n_samples),
                "non_empty_sample_count": non_empty_sample_count,
                "meets_min_samples": meets_min,
                "adaptive_rounds": None if diagnostics is None else int(diagnostics.adaptive_rounds),
                "stopped_early": None if diagnostics is None else bool(diagnostics.stopped_early),
                "stop_reason": None if diagnostics is None else diagnostics.stop_reason,
            },
        })

    payload = {
        "schema_version": 1,
        "fixture_type": "inside_diagnostics_samples",
        "description": (
            "Sampled responses exported from eval_truthfulqa.py INSIDE diagnostics cache. "
            "Labels are copied only for audit metadata; samples are aligned by score-dump statement identity."
        ),
        "source": {
            "scores_path": str(scores_path),
            "inside_diagnostics_cache_path": str(inside_diagnostics_cache_path),
        },
        "config": {
            "layers": [int(layer) for layer in layers],
            "adaptive": adaptive,
            "selfcheck_early_stop": selfcheck_early_stop,
            "min_samples": int(min_samples),
            "include_empty_records": bool(include_empty_records),
            "cache_key_schema": "eval_truthfulqa_inside_diagnostics_v1",
        },
        "summary": {
            "n_score_records": len(labels),
            "n_records": len(records),
            "matched_records": matched_records,
            "missing_records": len(labels) - matched_records,
            "records_meeting_min_samples": records_meeting_min_samples,
            "records_dropped_below_min_samples": dropped_records,
            "total_non_empty_samples": total_samples,
            "cache_stats": cache.stats(),
        },
        "records": records,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if artifact_manifest_path is not None:
        manifest = build_artifact_manifest(
            {
                "scores": scores_path,
                "inside_diagnostics_cache": inside_diagnostics_cache_path,
                "samples": output,
            },
            root=Path(artifact_manifest_path).parent,
            metadata={
                "workflow": "export_inside_diagnostics_samples",
                "scores_path": str(scores_path),
                "inside_diagnostics_cache_path": str(inside_diagnostics_cache_path),
                "output_path": str(output),
                "summary": payload["summary"],
            },
        )
        manifest_path = Path(artifact_manifest_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _cache_key_args_from_config(config: Mapping[str, Any]) -> SimpleNamespace:
    defaults = {
        "dtype": "float32",
        "hidden_state_capture": "outputs",
        "seed": 0,
        "inside_min_samples": 2,
        "inside_sample_step": 1,
        "inside_stability_delta": 0.05,
        "inside_selfcheck_min_overlap": 0.65,
        "inside_selfcheck_support_threshold": 0.60,
        "inside_selfcheck_refute_threshold": 0.50,
        "inside_temperature": 0.7,
        "inside_top_p": 0.9,
        "inside_pooling": "last",
        "inside_embedding_threshold": 0.9,
        "eigenscore_alpha": 1e-3,
    }
    required = (
        "model",
        "layer",
        "max_length",
        "inside_samples",
        "inside_max_new_tokens",
    )
    payload = {**defaults, **dict(config)}
    missing = [key for key in required if payload.get(key) is None]
    if missing:
        raise ValueError(f"score dump config is missing INSIDE cache key field(s): {missing}.")
    return SimpleNamespace(**payload)


def _layers_from_config(config: Mapping[str, Any]) -> tuple[int, ...]:
    raw = config.get("sweep_layers")
    if raw is None:
        return (int(config["layer"]),)
    if isinstance(raw, str):
        layers = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    elif isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray, str)):
        layers = tuple(int(layer) for layer in raw)
    else:
        raise ValueError("score dump config sweep_layers must be null, string, or sequence.")
    if not layers:
        return (int(config["layer"]),)
    return layers


def _statement_from_mapping(statement: Mapping[str, Any], *, label: int) -> Statement:
    question = str(statement.get("question", ""))
    answer = str(statement.get("answer", statement.get("text", statement.get("claim", ""))))
    if not answer.strip():
        raise ValueError("statement record is missing answer/text/claim.")
    is_false = int(statement.get("is_false", label))
    return Statement(question=question, answer=answer, is_false=is_false)


def _statement_mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("score dump statements must be JSON objects.")
    return value


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run from CLI-style arguments."""
    payload = export_inside_diagnostics_samples(
        scores_path=args.scores,
        inside_diagnostics_cache_path=args.inside_diagnostics_cache,
        output_path=args.output,
        min_samples=args.min_samples,
        include_empty_records=not bool(args.drop_empty_records),
        artifact_manifest_path=args.artifact_manifest,
    )
    summary = payload["summary"]
    print(
        "inside_diagnostics_samples_export_ok "
        f"matched={summary['matched_records']}/{summary['n_score_records']} output={args.output}"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Export sampled texts from an INSIDE diagnostics cache")
    parser.add_argument("--scores", required=True, help="statement-bearing score dump used to build the cache")
    parser.add_argument("--inside-diagnostics-cache", required=True, help="eval_truthfulqa.py inside diagnostics cache")
    parser.add_argument("--output", required=True, help="output samples JSON path")
    parser.add_argument("--min-samples", type=int, default=1)
    parser.add_argument("--drop-empty-records", action="store_true")
    parser.add_argument("--artifact-manifest", default=None, help="optional manifest for scores/cache/output samples")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

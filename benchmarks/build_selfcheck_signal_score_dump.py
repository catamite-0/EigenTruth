"""Build score-dump self-consistency signals from aligned sampled responses.

This is a post-hoc bridge from ``eval_truthfulqa.py --dump-inside-samples`` or
external sampled-generation files to the standard score-dump calibration path.
It deliberately does not generate samples; it only aligns caller-supplied
samples, runs the dependency-free ``SelfConsistencyVerifier``, and appends
auditable score columns.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.build_selfcheck_fixture import build_selfcheck_fixture, load_sample_payloads  # noqa: E402
from eigentruth.eval.score_dump import ScoreDump, load_score_dump, write_score_dump_jsonl  # noqa: E402
from eigentruth.verify import Claim, SelfConsistencyVerifier, VerificationResult, VerificationStatus  # noqa: E402

DEFAULT_SELFCHECK_SIGNALS = (
    "selfcheck_support_rate",
    "selfcheck_refute_rate",
    "selfcheck_disagreement",
    "selfcheck_insufficient",
    "selfcheck_not_applicable",
    "selfcheck_sample_count",
    "selfcheck_best_overlap",
)


def _parse_csv(value: str | None, *, name: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    parts = tuple(part.strip() for part in value.split(",") if part.strip())
    if not parts:
        raise ValueError(f"{name} must contain at least one value.")
    if len(set(parts)) != len(parts):
        raise ValueError(f"{name} must contain unique values.")
    return parts


def build_selfcheck_signal_score_dump(
    score_dump: ScoreDump,
    sample_payloads: Sequence[Mapping[str, Any] | Sequence[Any]] = (),
    *,
    source_scores_path: str | Path,
    sample_paths: Sequence[str | Path] = (),
    keep_signals: Sequence[str] | None = None,
    selfcheck_signals: Sequence[str] = DEFAULT_SELFCHECK_SIGNALS,
    min_samples: int = 2,
    min_overlap: float = 0.65,
    support_threshold: float = 0.60,
    refute_threshold: float = 0.50,
    early_stop: bool = False,
    max_samples: int | None = None,
) -> ScoreDump:
    """Return a score dump with self-consistency score columns appended."""
    if not score_dump.statements:
        raise ValueError("selfcheck signals require statement-bearing score dumps.")
    selected_keep_signals = tuple(score_dump.scores) if keep_signals is None else tuple(keep_signals)
    missing = [signal for signal in selected_keep_signals if signal not in score_dump.scores]
    if missing:
        raise ValueError(f"score dump is missing requested signal(s): {missing}.")

    selected_selfcheck_signals = tuple(selfcheck_signals)
    if len(set(selected_selfcheck_signals)) != len(selected_selfcheck_signals):
        raise ValueError("selfcheck_signals must contain unique values.")
    overlap = set(selected_keep_signals) & set(selected_selfcheck_signals)
    if overlap:
        raise ValueError(f"selfcheck signal(s) overlap existing score signals: {sorted(overlap)}.")

    fixture = build_selfcheck_fixture(
        score_dump.to_mapping(),
        sample_payloads,
        min_samples=min_samples,
        include_empty_records=True,
    )
    records = tuple(_record_mapping(item) for item in fixture.get("records", ()))
    if len(records) != score_dump.n_total:
        raise ValueError("selfcheck fixture record count must match score dump labels.")

    verifier = SelfConsistencyVerifier(
        min_samples=min_samples,
        min_overlap=min_overlap,
        support_threshold=support_threshold,
        refute_threshold=refute_threshold,
        early_stop=early_stop,
        max_samples=max_samples,
    )
    signal_columns = {signal: [] for signal in selected_selfcheck_signals}
    for record in records:
        claim = Claim(
            text=str(record.get("claim", "")),
            claim_id=str(record.get("claim_id", "")) or None,
            metadata=_mapping(record.get("claim_metadata")),
        )
        result = verifier.verify(claim, context={"selfcheck_samples": record.get("selfcheck_samples", ())})
        features = selfcheck_signal_features(result)
        for signal in selected_selfcheck_signals:
            if signal not in features:
                raise ValueError(f"unknown selfcheck signal {signal!r}.")
            signal_columns[signal].append(_finite_float(features[signal], name=signal))

    config = dict(score_dump.config)
    config["selfcheck_signal_score_dump"] = {
        "builder": "build_selfcheck_signal_score_dump",
        "source_scores_path": str(source_scores_path),
        "sample_paths": [str(path) for path in sample_paths],
        "signals": list(selected_selfcheck_signals),
        "min_samples": int(min_samples),
        "min_overlap": float(min_overlap),
        "support_threshold": float(support_threshold),
        "refute_threshold": float(refute_threshold),
        "early_stop": bool(early_stop),
        "max_samples": max_samples,
    }
    scores = {
        **{signal: tuple(float(value) for value in score_dump.scores[signal]) for signal in selected_keep_signals},
        **{signal: tuple(values) for signal, values in signal_columns.items()},
    }
    extras = dict(score_dump.extras)
    extras["selfcheck_signal_metadata"] = {
        "source_scores_path": str(source_scores_path),
        "sample_paths": [str(path) for path in sample_paths],
        "signals": list(selected_selfcheck_signals),
        "signal_definitions": selfcheck_signal_definitions(),
        "fixture_summary": fixture.get("summary", {}),
    }
    return ScoreDump(
        labels=score_dump.labels,
        scores=scores,
        config=config,
        sweep_scores=score_dump.sweep_scores,
        statements=score_dump.statements,
        extras=extras,
    )


def selfcheck_signal_features(result: VerificationResult) -> dict[str, float]:
    """Return numeric self-consistency features from one verifier result."""
    metadata = _mapping(result.metadata)
    support_rate = _unit_interval(metadata.get("support_rate", 0.0), name="support_rate")
    refute_rate = _unit_interval(metadata.get("refute_rate", 0.0), name="refute_rate")
    sample_count = _non_negative_float(metadata.get("sample_count", 0.0), name="sample_count")
    best_overlap = _unit_interval(metadata.get("best_overlap", 0.0), name="best_overlap")
    if result.status is VerificationStatus.NOT_APPLICABLE:
        disagreement = 1.0
    else:
        disagreement = max(0.0, 1.0 - max(support_rate, refute_rate))
    return {
        "selfcheck_support_rate": support_rate,
        "selfcheck_refute_rate": refute_rate,
        "selfcheck_disagreement": disagreement,
        "selfcheck_insufficient": 1.0 if result.status is VerificationStatus.INSUFFICIENT_EVIDENCE else 0.0,
        "selfcheck_not_applicable": 1.0 if result.status is VerificationStatus.NOT_APPLICABLE else 0.0,
        "selfcheck_sample_count": sample_count,
        "selfcheck_best_overlap": best_overlap,
    }


def selfcheck_signal_definitions() -> dict[str, str]:
    """Return human-readable definitions for each selfcheck signal."""
    return {
        "selfcheck_support_rate": "Fraction of processed samples that support the claim; lower is riskier.",
        "selfcheck_refute_rate": "Fraction of processed samples that refute the claim; higher is riskier.",
        "selfcheck_disagreement": "1 - max(support_rate, refute_rate), or 1 when too few samples exist.",
        "selfcheck_insufficient": "1 when samples are present but below consistency thresholds.",
        "selfcheck_not_applicable": "1 when the claim had too few usable samples or no lexical tokens.",
        "selfcheck_sample_count": "Number of samples consumed by SelfConsistencyVerifier; lower is less covered.",
        "selfcheck_best_overlap": (
            "Best lexical overlap between the claim and sampled responses; lower is less grounded."
        ),
    }


def write_score_dump(dump: ScoreDump, output_path: str | Path, *, output_format: str) -> None:
    """Write output score dump as JSON or JSONL manifest."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        output.write_text(json.dumps(dump.to_mapping(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return
    if output_format == "jsonl":
        write_score_dump_jsonl(dump, output)
        return
    raise ValueError("output_format must be 'json' or 'jsonl'.")


def build_report(
    *,
    input_scores: str | Path,
    sample_paths: Sequence[str | Path],
    output: str | Path,
    output_format: str,
    keep_signals: Sequence[str] | None,
    selfcheck_signals: Sequence[str],
    min_samples: int,
    min_overlap: float,
    support_threshold: float,
    refute_threshold: float,
    early_stop: bool,
    max_samples: int | None,
) -> dict[str, Any]:
    """Build selfcheck signal score dump and return a compact report."""
    score_dump = load_score_dump(
        input_scores,
        allow_missing_scores=False,
        require_statements=True,
    )
    sample_payloads = load_sample_payloads(tuple(Path(path) for path in sample_paths))
    enhanced = build_selfcheck_signal_score_dump(
        score_dump,
        sample_payloads,
        source_scores_path=input_scores,
        sample_paths=sample_paths,
        keep_signals=keep_signals,
        selfcheck_signals=selfcheck_signals,
        min_samples=min_samples,
        min_overlap=min_overlap,
        support_threshold=support_threshold,
        refute_threshold=refute_threshold,
        early_stop=early_stop,
        max_samples=max_samples,
    )
    write_score_dump(enhanced, output, output_format=output_format)
    return {
        "schema_version": 1,
        "input_scores": str(input_scores),
        "sample_paths": [str(path) for path in sample_paths],
        "output": str(output),
        "output_format": output_format,
        "n_total": enhanced.n_total,
        "signals": list(enhanced.scores),
        "selfcheck_signals": list(selfcheck_signals),
        "selfcheck_config": dict(enhanced.config["selfcheck_signal_score_dump"]),
        "fixture_summary": dict(enhanced.extras["selfcheck_signal_metadata"].get("fixture_summary", {})),
        "summary": {
            signal: {
                "min": min(enhanced.scores[signal]),
                "max": max(enhanced.scores[signal]),
                "mean": sum(enhanced.scores[signal]) / len(enhanced.scores[signal]),
            }
            for signal in selfcheck_signals
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run from CLI-style arguments."""
    keep_signals = _parse_csv(args.keep_signals, name="keep_signals")
    selfcheck_signals = _parse_csv(args.selfcheck_signals, name="selfcheck_signals") or DEFAULT_SELFCHECK_SIGNALS
    report = build_report(
        input_scores=args.scores,
        sample_paths=tuple(args.samples or ()),
        output=args.output,
        output_format=args.output_format,
        keep_signals=keep_signals,
        selfcheck_signals=selfcheck_signals,
        min_samples=args.min_samples,
        min_overlap=args.min_overlap,
        support_threshold=args.support_threshold,
        refute_threshold=args.refute_threshold,
        early_stop=args.early_stop,
        max_samples=args.max_samples,
    )
    if args.json:
        output_path = Path(args.json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote selfcheck-signal score dump to {args.output}")
    return report


def _record_mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("selfcheck fixture records must be JSON objects.")
    return value


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric and must not be bool.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite.")
    return numeric


def _unit_interval(value: Any, *, name: str) -> float:
    numeric = _finite_float(value, name=name)
    if not (0.0 <= numeric <= 1.0):
        raise ValueError(f"{name} must be in [0, 1].")
    return numeric


def _non_negative_float(value: Any, *, name: str) -> float:
    numeric = _finite_float(value, name=name)
    if numeric < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return numeric


def main() -> None:
    parser = argparse.ArgumentParser(description="Append self-consistency signals to a score dump")
    parser.add_argument("--scores", required=True, help="statement-bearing score dump")
    parser.add_argument("--samples", action="append", default=None, help="optional sampled responses JSON/JSONL")
    parser.add_argument("--output", required=True, help="output score dump path")
    parser.add_argument("--output-format", choices=("json", "jsonl"), default="jsonl")
    parser.add_argument("--keep-signals", default=None)
    parser.add_argument("--selfcheck-signals", default=",".join(DEFAULT_SELFCHECK_SIGNALS))
    parser.add_argument("--min-samples", type=int, default=2)
    parser.add_argument("--min-overlap", type=float, default=0.65)
    parser.add_argument("--support-threshold", type=float, default=0.60)
    parser.add_argument("--refute-threshold", type=float, default=0.50)
    parser.add_argument("--early-stop", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--json", default=None, help="optional compact report path")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

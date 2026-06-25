"""No-model smoke profile for rank score-fusion runtime caching."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Mapping
from unittest.mock import patch

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eigentruth.calibration import RankScoreFusionArtifact, ScoreFusionSignal  # noqa: E402
from eigentruth.eval import combine_rank_anomaly_scores, directional_rank_anomaly_scores  # noqa: E402

DEFAULT_CALIBRATION_SIZE = 4096
DEFAULT_BATCH_SIZE = 1024
DEFAULT_REPEATS = 12


def build_score_fusion_profile_smoke(
    output_dir: Path,
    *,
    calibration_size: int = DEFAULT_CALIBRATION_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    repeats: int = DEFAULT_REPEATS,
) -> dict[str, Any]:
    """Profile cached artifact scoring against the uncached rank-fusion path."""
    if calibration_size <= 0:
        raise ValueError("calibration_size must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if repeats <= 0:
        raise ValueError("repeats must be positive.")

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = _build_artifact(calibration_size)
    score_inputs = _score_inputs(batch_size)
    uncached_once = _score_uncached(artifact, score_inputs)
    cached_once = artifact.score(score_inputs)
    max_abs_diff = float((cached_once - uncached_once).abs().max().item())
    if max_abs_diff > 1e-12:
        raise AssertionError(f"cached score path changed fusion values: max_abs_diff={max_abs_diff}")
    if "_sorted_calibration_scores" in artifact.to_dict():
        raise AssertionError("derived runtime cache leaked into artifact JSON schema.")

    with patch("eigentruth.eval.score_fusion.torch.sort", side_effect=AssertionError("unexpected runtime sort")):
        no_sort_cached = artifact.score(score_inputs)
    if not torch.allclose(no_sort_cached, uncached_once, atol=1e-12, rtol=0.0):
        raise AssertionError("cached no-sort score path changed fusion values.")

    uncached_seconds, uncached_checksum = _time_repeated(
        lambda: _score_uncached(artifact, score_inputs),
        repeats=repeats,
    )
    cached_seconds, cached_checksum = _time_repeated(
        lambda: artifact.score(score_inputs),
        repeats=repeats,
    )
    report = {
        "profile": "score_fusion_rank_cache",
        "calibration_size": calibration_size,
        "batch_size": batch_size,
        "signal_count": len(artifact.signals),
        "repeats": repeats,
        "uncached_seconds": uncached_seconds,
        "cached_seconds": cached_seconds,
        "speedup": math.inf if cached_seconds == 0.0 else uncached_seconds / cached_seconds,
        "uncached_checksum": uncached_checksum,
        "cached_checksum": cached_checksum,
        "max_abs_diff": max_abs_diff,
        "runtime_sort_calls_cached": 0,
        "artifact_schema_unchanged": True,
    }
    (output_dir / "score_fusion_profile_smoke_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _build_artifact(calibration_size: int) -> RankScoreFusionArtifact:
    signals = (
        ScoreFusionSignal(
            name="maha_last",
            direction="higher",
            calibration_scores=tuple(_deterministic_series(calibration_size, offset=0.0).tolist()),
        ),
        ScoreFusionSignal(
            name="truth_proj",
            direction="lower",
            calibration_scores=tuple(_deterministic_series(calibration_size, offset=1.7).tolist()),
        ),
        ScoreFusionSignal(
            name="subspace_resid",
            direction="higher",
            calibration_scores=tuple(_deterministic_series(calibration_size, offset=3.1).tolist()),
        ),
    )
    return RankScoreFusionArtifact(signals=signals, method="max_rank", threshold=0.95, conformal_alpha=0.1)


def _score_inputs(batch_size: int) -> dict[str, torch.Tensor]:
    return {
        "maha_last": _deterministic_series(batch_size, offset=0.4) + 0.2,
        "truth_proj": _deterministic_series(batch_size, offset=2.0) - 0.1,
        "subspace_resid": _deterministic_series(batch_size, offset=3.9) + 0.4,
    }


def _deterministic_series(size: int, *, offset: float) -> torch.Tensor:
    base = torch.arange(size, dtype=torch.float64)
    return torch.sin(base * 0.017 + offset) + 0.5 * torch.cos(base * 0.031 + offset * 0.7)


def _score_uncached(artifact: RankScoreFusionArtifact, score_inputs: Mapping[str, Any]) -> torch.Tensor:
    rank_scores = [
        directional_rank_anomaly_scores(
            signal.calibration_scores,
            score_inputs[signal.name],
            direction=signal.direction,
        )
        for signal in artifact.signals
    ]
    return combine_rank_anomaly_scores(rank_scores, artifact.method)


def _time_repeated(fn: Callable[[], torch.Tensor], *, repeats: int) -> tuple[float, float]:
    start = time.perf_counter()
    checksum = 0.0
    for _ in range(repeats):
        checksum += float(fn().sum().item())
    return time.perf_counter() - start, checksum


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the score-fusion profile smoke check")
    parser.add_argument("--output-dir", default=None, help="optional output directory")
    parser.add_argument("--calibration-size", type=int, default=DEFAULT_CALIBRATION_SIZE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    args = parser.parse_args(argv)
    if args.output_dir is not None:
        report = build_score_fusion_profile_smoke(
            Path(args.output_dir),
            calibration_size=args.calibration_size,
            batch_size=args.batch_size,
            repeats=args.repeats,
        )
        _print_report(report)
        return
    with tempfile.TemporaryDirectory(prefix="eigentruth-score-fusion-profile-") as tmpdir:
        report = build_score_fusion_profile_smoke(
            Path(tmpdir),
            calibration_size=args.calibration_size,
            batch_size=args.batch_size,
            repeats=args.repeats,
        )
        _print_report(report)


def _print_report(report: Mapping[str, Any]) -> None:
    print(
        "score_fusion_profile_smoke_ok "
        f"cached_seconds={report['cached_seconds']:.6f} "
        f"uncached_seconds={report['uncached_seconds']:.6f} "
        f"speedup={report['speedup']:.2f} "
        f"runtime_sort_calls_cached={report['runtime_sort_calls_cached']}"
    )


if __name__ == "__main__":
    main()

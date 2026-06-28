"""Synthetic sanity check for generation trajectory convergence diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eigentruth.core import TrajectoryMonitor  # noqa: E402
from eigentruth.eval import roc_auc, spearman_correlation  # noqa: E402
from eigentruth.registry import build_artifact_manifest  # noqa: E402


def _unit_vector(vector: torch.Tensor) -> torch.Tensor:
    return vector / torch.norm(vector).clamp_min(1e-8)


def _quality_value(index: int, count: int, *, generator: torch.Generator) -> float:
    base = (float(index) + 0.5) / float(count)
    jitter = float((torch.rand((), generator=generator).item() - 0.5) * 0.08)
    return min(1.0, max(0.0, base + jitter))


def _make_trajectory(
    *,
    quality: float,
    step_count: int,
    hidden_dim: int,
    generator: torch.Generator,
) -> torch.Tensor:
    start = torch.randn(int(hidden_dim), generator=generator) * 0.05
    direction = _unit_vector(torch.randn(int(hidden_dim), generator=generator))
    noise_basis = _unit_vector(torch.randn(int(hidden_dim), generator=generator))
    current = start.clone()
    states = [current.clone()]
    decay_rate = 0.92 - (0.62 * float(quality))
    turn_scale = 0.18 * (1.0 - float(quality))
    noise_scale = 0.05 * (1.0 - float(quality))
    for step in range(1, int(step_count)):
        step_size = 1.0 * (float(decay_rate) ** (step - 1))
        wobble = torch.sin(torch.tensor(float(step) * 1.7)) * turn_scale
        noise = torch.randn(int(hidden_dim), generator=generator) * noise_scale
        delta = (direction * step_size) + (noise_basis * float(wobble)) + noise
        current = current + delta
        states.append(current.clone())
    return torch.stack(states)


def trajectory_convergence_sanity_report(
    *,
    sample_count: int = 48,
    step_count: int = 8,
    hidden_dim: int = 12,
    seed: int = 42,
    min_abs_spearman: float = 0.3,
    min_auroc: float = 0.55,
) -> dict[str, Any]:
    """Build a deterministic synthetic trajectory-convergence report."""
    if int(sample_count) < 8:
        raise ValueError("sample_count must be >= 8.")
    if int(step_count) < 3:
        raise ValueError("step_count must be >= 3.")
    if int(hidden_dim) < 2:
        raise ValueError("hidden_dim must be >= 2.")
    generator = torch.Generator().manual_seed(int(seed))
    monitor = TrajectoryMonitor(metadata={"workflow": "trajectory_convergence_sanity"})
    records = []
    for index in range(int(sample_count)):
        quality = _quality_value(index, int(sample_count), generator=generator)
        trajectory = _make_trajectory(
            quality=quality,
            step_count=int(step_count),
            hidden_dim=int(hidden_dim),
            generator=generator,
        )
        metrics = monitor.record(
            trajectory,
            metadata={
                "sample_index": index,
                "quality_score": quality,
                "quality_label": int(quality >= 0.5),
            },
        )
        nll_proxy = 1.25 - quality + float(torch.randn((), generator=generator).item() * 0.02)
        entropy_proxy = 1.0 - quality + float(torch.randn((), generator=generator).item() * 0.02)
        records.append({
            "sample_index": index,
            "quality_score": quality,
            "quality_label": int(quality >= 0.5),
            "nll_proxy": nll_proxy,
            "entropy_proxy": entropy_proxy,
            "trajectory": metrics.to_dict(),
        })

    convergence_scores = [float(record["trajectory"]["convergence_score"]) for record in records]
    quality_scores = [float(record["quality_score"]) for record in records]
    nll_scores = [float(record["nll_proxy"]) for record in records]
    labels = [int(record["quality_label"]) for record in records]
    spearman_quality = spearman_correlation(convergence_scores, quality_scores)
    spearman_nll = spearman_correlation(convergence_scores, nll_scores)
    quality_auroc = roc_auc(convergence_scores, labels)
    status = "pass" if (
        abs(float(spearman_quality)) >= float(min_abs_spearman)
        or float(quality_auroc) >= float(min_auroc)
    ) else "fail"

    return {
        "workflow": "trajectory_convergence_sanity",
        "config": {
            "sample_count": int(sample_count),
            "step_count": int(step_count),
            "hidden_dim": int(hidden_dim),
            "seed": int(seed),
            "min_abs_spearman": float(min_abs_spearman),
            "min_auroc": float(min_auroc),
        },
        "summary": {
            "status": status,
            "spearman_convergence_quality": float(spearman_quality),
            "spearman_convergence_nll": float(spearman_nll),
            "quality_auroc": float(quality_auroc),
            "mean_high_quality_convergence_score": _mean(
                record["trajectory"]["convergence_score"] for record in records if record["quality_label"] == 1
            ),
            "mean_low_quality_convergence_score": _mean(
                record["trajectory"]["convergence_score"] for record in records if record["quality_label"] == 0
            ),
            "n_high_quality": sum(labels),
            "n_low_quality": len(labels) - sum(labels),
        },
        "records": records,
        "trajectory_report": monitor.to_report(metadata={"source": "synthetic convergence sanity"}).to_dict(),
    }


def _mean(values: Sequence[float]) -> float | None:
    materialized = [float(value) for value in values]
    if not materialized:
        return None
    return sum(materialized) / len(materialized)


def run(args: argparse.Namespace) -> dict[str, Any]:
    report = trajectory_convergence_sanity_report(
        sample_count=args.sample_count,
        step_count=args.step_count,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
        min_abs_spearman=args.min_abs_spearman,
        min_auroc=args.min_auroc,
    )
    if args.json:
        output_path = Path(args.json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.artifact_manifest:
        if not args.json:
            raise ValueError("--artifact-manifest requires --json.")
        manifest_path = Path(args.artifact_manifest)
        manifest = build_artifact_manifest(
            {"trajectory_convergence_sanity_report": Path(args.json)},
            root=manifest_path.parent,
            metadata={
                "workflow": "trajectory_convergence_sanity",
                "source": "synthetic generation trajectory convergence",
            },
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run synthetic trajectory-convergence sanity check")
    parser.add_argument("--sample-count", type=int, default=48)
    parser.add_argument("--step-count", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-abs-spearman", type=float, default=0.3)
    parser.add_argument("--min-auroc", type=float, default=0.55)
    parser.add_argument("--json", default=None, help="optional path to write JSON report")
    parser.add_argument("--artifact-manifest", default=None, help="optional manifest path for --json output")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()

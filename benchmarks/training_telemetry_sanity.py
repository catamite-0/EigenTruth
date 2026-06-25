"""Synthetic sanity check for training-side representation telemetry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eigentruth.registry import build_artifact_manifest  # noqa: E402
from eigentruth.training import RepresentationTelemetryRecorder  # noqa: E402


def _layer_base_states(
    *,
    sample_count: int,
    hidden_dim: int,
    layers: Sequence[int],
    seed: int,
) -> dict[int, torch.Tensor]:
    generator = torch.Generator().manual_seed(int(seed))
    states: dict[int, torch.Tensor] = {}
    for index, layer in enumerate(layers):
        scale = 1.0 + (0.15 * index)
        states[int(layer)] = torch.randn(sample_count, hidden_dim, generator=generator) * scale
    return states


def _trajectory_states(
    base: Mapping[int, torch.Tensor],
    *,
    run_type: str,
    step: int,
    seed: int,
) -> dict[int, torch.Tensor]:
    if run_type not in {"clean", "corrupt"}:
        raise ValueError("run_type must be 'clean' or 'corrupt'.")
    generator = torch.Generator().manual_seed(int(seed) + (1000 * int(step)))
    states: dict[int, torch.Tensor] = {}
    for layer, matrix in base.items():
        noise = torch.randn(matrix.shape, generator=generator) * (0.015 * max(step, 1))
        if run_type == "clean":
            drift = torch.zeros(matrix.shape[-1])
            drift[0] = 0.015 * step
            states[int(layer)] = matrix + noise + drift
            continue

        collapse = torch.linspace(1.0, 0.08, int(matrix.shape[-1]))
        collapse_strength = min(1.0, 0.22 * step)
        scale = (1.0 - collapse_strength) + (collapse_strength * collapse)
        drift = torch.zeros(matrix.shape[-1])
        drift[0] = 0.18 * step
        drift[1] = -0.10 * step
        layer_multiplier = 1.0 + (0.12 * abs(int(layer)))
        states[int(layer)] = (matrix * scale * (1.0 + 0.05 * step * layer_multiplier)) + noise + drift
    return states


def _record_run(
    base: Mapping[int, torch.Tensor],
    *,
    run_type: str,
    steps: int,
    seed: int,
    covariance_mode: str,
) -> dict[str, Any]:
    recorder = RepresentationTelemetryRecorder(
        layers=tuple(sorted(int(layer) for layer in base)),
        covariance_mode=covariance_mode,
        baseline_strategy="manual",
        metadata={"run_type": run_type, "source": "synthetic"},
    )
    recorder.set_baseline(base)
    for step in range(steps + 1):
        recorder.record_step(
            step,
            _trajectory_states(base, run_type=run_type, step=step, seed=seed),
            metadata={"run_type": run_type},
        )
    report = recorder.to_report().to_dict()
    report["run_type"] = run_type
    return report


def _final_layer(report: Mapping[str, Any], *, layer: int) -> Mapping[str, Any]:
    return report["summary"]["final_by_layer"][str(layer)]


def training_telemetry_sanity_report(
    *,
    sample_count: int = 128,
    hidden_dim: int = 12,
    layers: Sequence[int] = (-2, -1),
    steps: int = 5,
    seed: int = 0,
    covariance_mode: str = "shrinkage",
    target_layer: int | None = None,
    min_distance_margin: float = 1.0,
    min_rank_margin: float = 1.0,
) -> dict[str, Any]:
    """Build a deterministic clean-vs-corrupt telemetry sanity report."""
    if sample_count < 2:
        raise ValueError("sample_count must be >= 2.")
    if hidden_dim < 2:
        raise ValueError("hidden_dim must be >= 2.")
    if steps < 1:
        raise ValueError("steps must be >= 1.")
    resolved_layers = tuple(int(layer) for layer in layers)
    if not resolved_layers:
        raise ValueError("layers must not be empty.")
    if len(set(resolved_layers)) != len(resolved_layers):
        raise ValueError("layers must not contain duplicates.")
    resolved_target = int(target_layer if target_layer is not None else resolved_layers[-1])
    if resolved_target not in set(resolved_layers):
        raise ValueError("target_layer must be included in layers.")

    base = _layer_base_states(
        sample_count=int(sample_count),
        hidden_dim=int(hidden_dim),
        layers=resolved_layers,
        seed=int(seed),
    )
    clean = _record_run(
        base,
        run_type="clean",
        steps=int(steps),
        seed=int(seed) + 17,
        covariance_mode=covariance_mode,
    )
    corrupt = _record_run(
        base,
        run_type="corrupt",
        steps=int(steps),
        seed=int(seed) + 29,
        covariance_mode=covariance_mode,
    )

    clean_final = _final_layer(clean, layer=resolved_target)
    corrupt_final = _final_layer(corrupt, layer=resolved_target)
    clean_distance = float(clean_final["distance_to_baseline"])
    corrupt_distance = float(corrupt_final["distance_to_baseline"])
    clean_rank = float(clean_final["effective_rank"])
    corrupt_rank = float(corrupt_final["effective_rank"])
    distance_margin = corrupt_distance - clean_distance
    rank_margin = clean_rank - corrupt_rank
    separated = distance_margin >= float(min_distance_margin) and rank_margin >= float(min_rank_margin)

    return {
        "workflow": "training_telemetry_sanity",
        "config": {
            "sample_count": int(sample_count),
            "hidden_dim": int(hidden_dim),
            "layers": list(resolved_layers),
            "target_layer": resolved_target,
            "steps": int(steps),
            "seed": int(seed),
            "covariance_mode": covariance_mode,
            "min_distance_margin": float(min_distance_margin),
            "min_rank_margin": float(min_rank_margin),
        },
        "summary": {
            "status": "pass" if separated else "fail",
            "separated": bool(separated),
            "target_layer": resolved_target,
            "clean_final_distance": clean_distance,
            "corrupt_final_distance": corrupt_distance,
            "distance_margin": distance_margin,
            "clean_final_effective_rank": clean_rank,
            "corrupt_final_effective_rank": corrupt_rank,
            "rank_margin": rank_margin,
        },
        "runs": {
            "clean": clean,
            "corrupt": corrupt,
        },
    }


def _parse_layers(value: str) -> tuple[int, ...]:
    layers = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not layers:
        raise argparse.ArgumentTypeError("--layers must include at least one integer layer.")
    return layers


def run(args: argparse.Namespace) -> dict[str, Any]:
    report = training_telemetry_sanity_report(
        sample_count=args.sample_count,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        steps=args.steps,
        seed=args.seed,
        covariance_mode=args.covariance_mode,
        target_layer=args.target_layer,
        min_distance_margin=args.min_distance_margin,
        min_rank_margin=args.min_rank_margin,
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
            {"training_telemetry_sanity_report": Path(args.json)},
            root=manifest_path.parent,
            metadata={
                "workflow": "training_telemetry_sanity",
                "covariance_mode": args.covariance_mode,
                "source": "synthetic clean-vs-corrupt hidden-state trajectories",
            },
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run synthetic training telemetry sanity check")
    parser.add_argument("--sample-count", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=12)
    parser.add_argument("--layers", type=_parse_layers, default=(-2, -1))
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--covariance-mode", default="shrinkage")
    parser.add_argument("--target-layer", type=int, default=None)
    parser.add_argument("--min-distance-margin", type=float, default=1.0)
    parser.add_argument("--min-rank-margin", type=float, default=1.0)
    parser.add_argument("--json", default=None, help="optional path to write JSON report")
    parser.add_argument("--artifact-manifest", default=None, help="optional manifest path for --json output")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()

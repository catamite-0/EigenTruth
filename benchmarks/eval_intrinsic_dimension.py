"""Evaluate TwoNN intrinsic-dimension profiles from saved warmup checkpoints."""

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

from eigentruth.eval import (  # noqa: E402
    intrinsic_dimension_peak_layer,
    intrinsic_dimension_profile,
)
from eigentruth.registry import build_artifact_manifest  # noqa: E402


def _parse_named_path(spec: str) -> tuple[str, Path]:
    name, sep, path_text = spec.partition("=")
    if not sep:
        path = Path(name)
        return path.stem, path
    return name, Path(path_text)


def _load_checkpoint_layer_states(path: str | Path, *, split: str) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    checkpoint_path = Path(path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    metadata = dict(checkpoint.get("metadata") or {})
    true_state_lists = checkpoint.get("true_state_lists") or {}
    false_state_lists = checkpoint.get("false_state_lists") or {}
    layers = [int(layer) for layer in metadata.get("layers", sorted(true_state_lists))]
    states_by_layer: dict[int, torch.Tensor] = {}
    for layer in layers:
        chunks = []
        if split in {"true", "both"}:
            chunks.extend(true_state_lists.get(layer, ()))
        if split in {"false", "both"}:
            chunks.extend(false_state_lists.get(layer, ()))
        if len(chunks) < 3:
            raise ValueError(f"layer {layer} has fewer than three states for split={split!r}.")
        states_by_layer[layer] = torch.stack(tuple(chunks)).to(torch.float32)
    return states_by_layer, metadata


def _profile_shape(profile: Sequence[Mapping[str, object]], *, tolerance: float = 0.0) -> dict[str, Any]:
    if not profile:
        return {"available": False, "reason": "empty profile"}
    values = [float(entry["intrinsic_dimension"]) for entry in profile]
    layers = [int(entry["layer"]) for entry in profile]
    peak_index = max(range(len(values)), key=values.__getitem__)
    before = values[: peak_index + 1]
    after = values[peak_index:]
    rises_to_peak = all((right + tolerance) >= left for left, right in zip(before, before[1:]))
    falls_after_peak = all((left + tolerance) >= right for left, right in zip(after, after[1:]))
    return {
        "available": True,
        "peak_layer": layers[peak_index],
        "peak_intrinsic_dimension": values[peak_index],
        "rises_to_peak": bool(rises_to_peak),
        "falls_after_peak": bool(falls_after_peak),
        "rise_then_fall": bool(rises_to_peak and falls_after_peak and 0 < peak_index < len(values) - 1),
        "layer_order": layers,
        "intrinsic_dimensions": values,
    }


def evaluate_intrinsic_dimension_profiles(
    checkpoints: Sequence[tuple[str, Path]],
    *,
    split: str = "true",
    trim_fraction: float = 0.05,
) -> dict[str, Any]:
    """Return TwoNN intrinsic-dimension profiles for saved warmup checkpoints."""
    if split not in {"true", "false", "both"}:
        raise ValueError("split must be one of: true, false, both.")
    if not checkpoints:
        raise ValueError("at least one warmup checkpoint is required.")

    reports = []
    for name, checkpoint_path in checkpoints:
        layer_states, metadata = _load_checkpoint_layer_states(checkpoint_path, split=split)
        profile = intrinsic_dimension_profile(layer_states, trim_fraction=trim_fraction)
        peak_layer = intrinsic_dimension_peak_layer(profile)
        shape = _profile_shape(profile)
        reports.append({
            "name": name,
            "source_checkpoint": str(checkpoint_path),
            "model": metadata.get("model"),
            "metadata": metadata,
            "split": split,
            "trim_fraction": float(trim_fraction),
            "profile": profile,
            "peak_layer": int(peak_layer),
            "shape": shape,
        })

    return {
        "workflow": "eval_intrinsic_dimension",
        "estimator": "twonn",
        "split": split,
        "trim_fraction": float(trim_fraction),
        "reports": reports,
        "summary": {
            "n_reports": len(reports),
            "n_rise_then_fall": sum(1 for report in reports if report["shape"].get("rise_then_fall") is True),
            "all_rise_then_fall": all(report["shape"].get("rise_then_fall") is True for report in reports),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    checkpoints = [_parse_named_path(spec) for spec in args.warmup_checkpoint]
    report = evaluate_intrinsic_dimension_profiles(
        checkpoints,
        split=args.split,
        trim_fraction=args.trim_fraction,
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
            {"intrinsic_dimension_report": Path(args.json)},
            root=manifest_path.parent,
            metadata={
                "workflow": "eval_intrinsic_dimension",
                "estimator": "twonn",
                "split": args.split,
                "source": "warmup checkpoint hidden states; no model forward pass",
            },
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate TwoNN intrinsic-dimension profiles")
    parser.add_argument(
        "--warmup-checkpoint",
        action="append",
        required=True,
        help="warmup checkpoint path, optionally NAME=PATH; repeatable",
    )
    parser.add_argument("--split", choices=("true", "false", "both"), default="true")
    parser.add_argument("--trim-fraction", type=float, default=0.05)
    parser.add_argument("--json", default=None, help="optional path to write JSON report")
    parser.add_argument("--artifact-manifest", default=None, help="optional manifest path for --json output")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()

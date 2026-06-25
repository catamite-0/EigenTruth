"""Compare saved EigenTruth manifolds with Gaussian 2-Wasserstein distance."""

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

from benchmarks.eval_truthfulqa import load_layer_stats_cache  # noqa: E402
from eigentruth.core import TruthManifold, manifold_distance  # noqa: E402


def _load_direct_manifold(spec: str, *, device: torch.device) -> tuple[str, TruthManifold]:
    label, sep, path_text = spec.partition("=")
    if not sep:
        path_text = label
        label = Path(path_text).stem
    manifold = TruthManifold.load(path_text).to(device)
    return label, manifold


def _load_layer_stats_manifolds(
    path: str | Path,
    *,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cache_path = Path(path)
    cache = torch.load(cache_path, map_location=device, weights_only=True)
    metadata = dict(cache.get("metadata") or {})
    raw_manifolds = cache.get("manifolds") or {}
    if not raw_manifolds:
        raise ValueError(f"layer stats cache has no manifolds: {cache_path}")
    if "layers" not in metadata:
        raise ValueError("layer stats cache metadata is missing layers.")

    manifolds, _subspaces, metadata = load_layer_stats_cache(
        cache_path,
        expected_metadata=metadata,
        device=device,
    )
    items: list[dict[str, Any]] = []
    for layer in sorted(manifolds):
        items.append({
            "id": f"layer:{layer}",
            "layer": int(layer),
            "source": str(cache_path),
            "manifold": manifolds[layer],
        })
    return items, metadata


def _manifold_summary(item: dict[str, Any]) -> dict[str, Any]:
    manifold = item["manifold"]
    return {
        "id": item["id"],
        "source": item["source"],
        "layer": item.get("layer"),
        "n": int(manifold.n),
        "hidden_dim": int(manifold.hidden_dim),
        "covariance_mode": manifold.covariance_mode,
        "covariance_low_rank": int(manifold.covariance_low_rank),
    }


def compare_manifold_distances(
    items: Sequence[dict[str, Any]],
    *,
    covariance_mode: str = "model",
    squared: bool = False,
) -> dict[str, Any]:
    """Return a JSON-ready pairwise manifold-distance matrix."""
    if len(items) < 2:
        raise ValueError("at least two manifolds are required for a distance matrix.")

    n_items = len(items)
    matrix = [[0.0 for _ in range(n_items)] for _ in range(n_items)]
    for i in range(n_items):
        for j in range(i + 1, n_items):
            distance = manifold_distance(
                items[i]["manifold"],
                items[j]["manifold"],
                covariance_mode=covariance_mode,
                squared=squared,
            )
            value = float(distance.detach().cpu())
            matrix[i][j] = value
            matrix[j][i] = value

    summaries = [_manifold_summary(item) for item in items]
    nearest_neighbors = []
    for i, summary in enumerate(summaries):
        candidates = [(distance, idx) for idx, distance in enumerate(matrix[i]) if idx != i]
        nearest_distance, nearest_idx = min(candidates, key=lambda entry: entry[0])
        nearest_neighbors.append({
            "id": summary["id"],
            "nearest_id": summaries[nearest_idx]["id"],
            "distance": nearest_distance,
        })

    return {
        "workflow": "compare_manifold_distances",
        "metric": "gaussian_2_wasserstein",
        "covariance_mode": covariance_mode,
        "squared": bool(squared),
        "items": summaries,
        "distance_matrix": matrix,
        "nearest_neighbors": nearest_neighbors,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    items: list[dict[str, Any]] = []
    source_metadata: dict[str, Any] = {}
    if args.layer_stats_cache:
        layer_items, metadata = _load_layer_stats_manifolds(args.layer_stats_cache, device=device)
        items.extend(layer_items)
        source_metadata["layer_stats_cache"] = metadata
    for spec in args.manifold or ():
        label, manifold = _load_direct_manifold(spec, device=device)
        items.append({
            "id": label,
            "source": spec.partition("=")[2] or spec,
            "manifold": manifold,
        })

    report = compare_manifold_distances(
        items,
        covariance_mode=args.covariance_mode,
        squared=bool(args.squared),
    )
    if source_metadata:
        report["source_metadata"] = source_metadata
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare saved TruthManifold artifacts with Gaussian distance")
    parser.add_argument("--layer-stats-cache", default=None, help="eval_truthfulqa.py layer-stats .pt cache")
    parser.add_argument(
        "--manifold",
        action="append",
        default=[],
        help="direct TruthManifold .pt path, optionally LABEL=PATH; repeatable",
    )
    parser.add_argument(
        "--covariance-mode",
        default="model",
        help='covariance estimate for distance: "model" or one of full/diag/low_rank/shrinkage',
    )
    parser.add_argument("--squared", action="store_true", help="emit squared W2 distance")
    parser.add_argument("--device", default="cpu", help="torch device for loading manifold tensors")
    parser.add_argument("--json", default=None, help="optional path to write JSON report")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()

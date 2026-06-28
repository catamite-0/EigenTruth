"""Rebuild TruthfulQA layer-stats caches from a saved warmup checkpoint.

The TruthfulQA warmup checkpoint stores per-layer factual/false hidden states.
This helper replays those cached states into a new ``TruthManifold`` so
covariance-mode experiments can rescore frontier dumps without rerunning model
warmup forwards.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.config_utils import strict_positive_int  # noqa: E402
from benchmarks.eval_truthfulqa import save_layer_stats_cache  # noqa: E402
from eigentruth.core import COVARIANCE_MODES, TruthManifold, TruthSubspace  # noqa: E402
from eigentruth.registry import build_artifact_manifest  # noqa: E402


@dataclass(frozen=True)
class LayerStatsRebuildConfig:
    """Configuration for rebuilding one layer-stats cache."""

    warmup_checkpoint: Path
    output: Path
    covariance_mode: str = "full"
    covariance_low_rank: int = 16
    layers: Sequence[int] | None = None
    subspace_rank: int | None = None
    json_report: Path | None = None
    artifact_manifest: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "warmup_checkpoint", Path(self.warmup_checkpoint))
        object.__setattr__(self, "output", Path(self.output))
        if self.json_report is not None:
            object.__setattr__(self, "json_report", Path(self.json_report))
        if self.artifact_manifest is not None:
            object.__setattr__(self, "artifact_manifest", Path(self.artifact_manifest))
        covariance_mode = str(self.covariance_mode)
        if covariance_mode not in COVARIANCE_MODES:
            raise ValueError(f"covariance_mode must be one of: {', '.join(COVARIANCE_MODES)}.")
        object.__setattr__(self, "covariance_mode", covariance_mode)
        object.__setattr__(
            self,
            "covariance_low_rank",
            strict_positive_int(self.covariance_low_rank, name="covariance_low_rank"),
        )
        if self.layers is not None:
            layers = tuple(int(layer) for layer in self.layers)
            if not layers:
                raise ValueError("layers must not be empty when provided.")
            if len(set(layers)) != len(layers):
                raise ValueError("layers must not contain duplicates.")
            object.__setattr__(self, "layers", layers)
        if self.subspace_rank is not None:
            object.__setattr__(
                self,
                "subspace_rank",
                strict_positive_int(self.subspace_rank, name="subspace_rank"),
            )


def rebuild_layer_stats_from_warmup_checkpoint(config: LayerStatsRebuildConfig) -> dict[str, Any]:
    """Rebuild and save a layer-stats cache from warmup hidden states."""
    started_at = time.perf_counter()
    checkpoint = _load_checkpoint(config.warmup_checkpoint)
    metadata = dict(checkpoint.get("metadata") or {})
    progress = dict(checkpoint.get("progress") or {})
    true_state_lists = _state_lists(checkpoint.get("true_state_lists") or {})
    false_state_lists = _state_lists(checkpoint.get("false_state_lists") or {})
    layers = _resolve_layers(config.layers, metadata, true_state_lists, false_state_lists)
    _validate_complete_checkpoint(metadata, progress, true_state_lists, false_state_lists, layers)

    subspace_rank = int(config.subspace_rank or metadata.get("subspace_rank") or 2)
    manifolds: dict[int, TruthManifold] = {}
    subspaces: dict[int, TruthSubspace] = {}
    layer_summaries = []
    for layer in layers:
        true_states = _states_for_layer(true_state_lists, layer)
        false_states = _states_for_layer(false_state_lists, layer)
        manifold = _rebuild_manifold(
            true_states,
            false_states,
            covariance_mode=config.covariance_mode,
            covariance_low_rank=config.covariance_low_rank,
        )
        manifolds[layer] = manifold
        subspace = _rebuild_subspace(true_states, false_states, rank=subspace_rank)
        if subspace is not None:
            subspaces[layer] = subspace
        layer_summaries.append({
            "layer": layer,
            "n_true": len(true_states),
            "n_false": len(false_states),
            "hidden_dim": manifold.hidden_dim,
            "subspace_ready": subspace is not None and subspace.is_ready(),
            "shrinkage_alpha": (
                manifold.covariance_shrinkage_alpha()
                if config.covariance_mode == "shrinkage" and manifold.n >= 2
                else None
            ),
        })

    rebuilt_metadata = _rebuilt_metadata(
        metadata,
        layers=layers,
        covariance_mode=config.covariance_mode,
        covariance_low_rank=config.covariance_low_rank,
        subspace_rank=subspace_rank,
        true_state_lists=true_state_lists,
        false_state_lists=false_state_lists,
    )
    save_layer_stats_cache(
        config.output,
        manifolds,
        subspaces,
        metadata=rebuilt_metadata,
    )

    artifacts: dict[str, str | Path | None] = {
        "warmup_checkpoint": config.warmup_checkpoint,
        "layer_stats_cache": config.output,
    }
    manifest = None
    if config.artifact_manifest is not None:
        manifest = build_artifact_manifest(
            artifacts,
            root=config.artifact_manifest.parent,
            metadata={
                "runner": "rebuild_layer_stats_from_warmup_checkpoint",
                "covariance_mode": config.covariance_mode,
                "covariance_low_rank": config.covariance_low_rank,
                "source_warmup_checkpoint": str(config.warmup_checkpoint),
                "layers": tuple(layers),
            },
        )
        config.artifact_manifest.parent.mkdir(parents=True, exist_ok=True)
        config.artifact_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    report = {
        "schema_version": 1,
        "workflow": "rebuild_layer_stats_from_warmup_checkpoint",
        "status": "complete",
        "paths": {
            "warmup_checkpoint": str(config.warmup_checkpoint),
            "layer_stats_cache": str(config.output),
            "json_report": None if config.json_report is None else str(config.json_report),
            "artifact_manifest": None if config.artifact_manifest is None else str(config.artifact_manifest),
        },
        "config": {
            "covariance_mode": config.covariance_mode,
            "covariance_low_rank": config.covariance_low_rank,
            "layers": tuple(layers),
            "subspace_rank": subspace_rank,
        },
        "source_metadata": metadata,
        "rebuilt_metadata": rebuilt_metadata,
        "progress": progress,
        "layers": layer_summaries,
        "artifact_manifest_summary": None if manifest is None else manifest.get("summary"),
        "execution": {"wall_clock_seconds": time.perf_counter() - started_at},
    }
    if config.json_report is not None:
        config.json_report.parent.mkdir(parents=True, exist_ok=True)
        config.json_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("warmup checkpoint must contain a mapping payload.")
    if int(checkpoint.get("format", 0)) != 1:
        raise ValueError("warmup checkpoint has an unsupported format.")
    return checkpoint


def _state_lists(payload: Mapping[Any, Any]) -> dict[int, list[torch.Tensor]]:
    return {
        int(layer): [torch.as_tensor(value, dtype=torch.float32).detach().cpu() for value in values]
        for layer, values in payload.items()
    }


def _resolve_layers(
    requested_layers: Sequence[int] | None,
    metadata: Mapping[str, Any],
    true_state_lists: Mapping[int, Sequence[torch.Tensor]],
    false_state_lists: Mapping[int, Sequence[torch.Tensor]],
) -> tuple[int, ...]:
    available = set(true_state_lists) | set(false_state_lists)
    if requested_layers is None:
        metadata_layers = tuple(int(layer) for layer in metadata.get("layers", ()) if int(layer) in available)
        if metadata_layers:
            return metadata_layers
        if not available:
            raise ValueError("warmup checkpoint does not contain any layer state lists.")
        return tuple(sorted(available))
    missing = set(requested_layers) - available
    if missing:
        raise ValueError(f"warmup checkpoint is missing requested layer(s): {sorted(missing)}.")
    return tuple(int(layer) for layer in requested_layers)


def _validate_complete_checkpoint(
    metadata: Mapping[str, Any],
    progress: Mapping[str, Any],
    true_state_lists: Mapping[int, Sequence[torch.Tensor]],
    false_state_lists: Mapping[int, Sequence[torch.Tensor]],
    layers: Sequence[int],
) -> None:
    n_true = _metadata_count(metadata, "n_true", true_state_lists, layers)
    n_false = _metadata_count(metadata, "n_false", false_state_lists, layers)
    true_done = int(progress.get("true_done", n_true))
    false_done = int(progress.get("false_done", n_false))
    if true_done != n_true or false_done != n_false:
        raise ValueError(
            "warmup checkpoint must be complete before rebuilding layer stats "
            f"(true_done={true_done}/{n_true}, false_done={false_done}/{n_false})."
        )
    for layer in layers:
        if len(true_state_lists.get(layer, ())) != n_true:
            raise ValueError(f"warmup checkpoint true state count mismatch at layer {layer}.")
        if len(false_state_lists.get(layer, ())) != n_false:
            raise ValueError(f"warmup checkpoint false state count mismatch at layer {layer}.")


def _metadata_count(
    metadata: Mapping[str, Any],
    key: str,
    state_lists: Mapping[int, Sequence[torch.Tensor]],
    layers: Sequence[int],
) -> int:
    if key in metadata:
        return int(metadata[key])
    return len(state_lists.get(int(layers[0]), ()))


def _states_for_layer(
    state_lists: Mapping[int, Sequence[torch.Tensor]],
    layer: int,
) -> list[torch.Tensor]:
    states = [torch.as_tensor(value, dtype=torch.float32).detach().cpu() for value in state_lists.get(layer, ())]
    if not states:
        raise ValueError(f"warmup checkpoint has no states for layer {layer}.")
    first_shape = tuple(states[0].shape)
    if any(tuple(state.shape) != first_shape for state in states):
        raise ValueError(f"warmup checkpoint states have inconsistent shapes at layer {layer}.")
    return states


def _rebuild_manifold(
    true_states: Sequence[torch.Tensor],
    false_states: Sequence[torch.Tensor],
    *,
    covariance_mode: str,
    covariance_low_rank: int,
) -> TruthManifold:
    manifold = TruthManifold(
        covariance_mode=covariance_mode,
        covariance_low_rank=covariance_low_rank,
    )
    manifold.update_many(torch.stack(tuple(true_states)).to(torch.float32))
    if false_states and manifold.mean is not None:
        false_mean = torch.stack(tuple(false_states)).to(torch.float32).mean(dim=0)
        manifold.false_mean = false_mean
        raw = manifold.mean - false_mean
        manifold.contrastive_direction = raw / torch.norm(raw).clamp(min=1e-8)
    return manifold


def _rebuild_subspace(
    true_states: Sequence[torch.Tensor],
    false_states: Sequence[torch.Tensor],
    *,
    rank: int,
) -> TruthSubspace | None:
    if len(true_states) < 2:
        return None
    true_tensor = torch.stack(tuple(true_states)).to(torch.float32)
    if false_states:
        return TruthSubspace.fit_contrastive(
            true_tensor,
            torch.stack(tuple(false_states)).to(torch.float32),
            rank=rank,
        )
    return TruthSubspace.fit(true_tensor, rank=rank)


def _rebuilt_metadata(
    metadata: Mapping[str, Any],
    *,
    layers: Sequence[int],
    covariance_mode: str,
    covariance_low_rank: int,
    subspace_rank: int,
    true_state_lists: Mapping[int, Sequence[torch.Tensor]],
    false_state_lists: Mapping[int, Sequence[torch.Tensor]],
) -> dict[str, Any]:
    rebuilt = dict(metadata)
    rebuilt.update({
        "format": 1,
        "layers": [int(layer) for layer in layers],
        "subspace_rank": int(subspace_rank),
        "covariance_mode": covariance_mode,
        "covariance_low_rank": int(covariance_low_rank),
        "n_true": _metadata_count(metadata, "n_true", true_state_lists, layers),
        "n_false": _metadata_count(metadata, "n_false", false_state_lists, layers),
    })
    return rebuilt


def _parse_layers(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    layers = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not layers:
        raise argparse.ArgumentTypeError("--layers must contain at least one integer layer.")
    return layers


def _config_from_args(args: argparse.Namespace) -> LayerStatsRebuildConfig:
    return LayerStatsRebuildConfig(
        warmup_checkpoint=Path(args.warmup_checkpoint),
        output=Path(args.output),
        covariance_mode=args.covariance_mode,
        covariance_low_rank=args.covariance_low_rank,
        layers=_parse_layers(args.layers),
        subspace_rank=args.subspace_rank,
        json_report=Path(args.json) if args.json else None,
        artifact_manifest=Path(args.artifact_manifest) if args.artifact_manifest else None,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """CLI-compatible runner."""
    return rebuild_layer_stats_from_warmup_checkpoint(_config_from_args(args))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild an eval_truthfulqa.py layer-stats cache from a warmup checkpoint."
    )
    parser.add_argument("--warmup-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--covariance-mode", default="full", choices=COVARIANCE_MODES)
    parser.add_argument("--covariance-low-rank", type=int, default=16)
    parser.add_argument("--layers", default=None, help="optional comma-list of layers to rebuild")
    parser.add_argument("--subspace-rank", type=int, default=None)
    parser.add_argument("--json", default=None, help="optional JSON report path")
    parser.add_argument("--artifact-manifest", default=None, help="optional artifact manifest path")
    args = parser.parse_args(argv)
    payload = run(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

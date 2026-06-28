"""Synthetic self-training loop for model-collapse early-warning telemetry."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eigentruth.eval import twonn_intrinsic_dimension  # noqa: E402
from eigentruth.registry import build_artifact_manifest  # noqa: E402
from eigentruth.training import RepresentationTelemetryRecorder  # noqa: E402


class TinySelfTrainingClassifier(nn.Module):
    """Small MLP with hidden-state outputs for self-training collapse checks."""

    def __init__(self, *, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 2)

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        return_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[int, torch.Tensor]]:
        hidden1 = torch.relu(self.layer1(inputs))
        hidden2 = torch.relu(self.layer2(hidden1))
        logits = self.output(hidden2)
        if return_states:
            return logits, {-2: hidden1.detach(), -1: hidden2.detach()}
        return logits


@dataclass(frozen=True)
class CollapseData:
    train_inputs: torch.Tensor
    train_labels: torch.Tensor
    probe_inputs: torch.Tensor
    probe_labels: torch.Tensor
    unlabeled_pool: torch.Tensor


def _make_dataset(*, n: int, input_dim: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(int(seed))
    inputs = torch.randn(int(n), int(input_dim), generator=generator)
    weights = torch.linspace(-1.0, 1.0, int(input_dim))
    logits = (inputs @ weights) + (0.4 * torch.sin(inputs[:, 0] * 2.0)) - (0.2 * inputs[:, 1] * inputs[:, 2])
    labels = (logits > 0.0).to(torch.long)
    return inputs, labels


def _make_data(*, train_count: int, probe_count: int, pool_count: int, input_dim: int, seed: int) -> CollapseData:
    train_inputs, train_labels = _make_dataset(n=train_count, input_dim=input_dim, seed=seed + 1)
    probe_inputs, probe_labels = _make_dataset(n=probe_count, input_dim=input_dim, seed=seed + 2)
    if int(pool_count) == int(train_count):
        unlabeled_pool = train_inputs.clone()
    else:
        unlabeled_pool, _pool_labels = _make_dataset(n=pool_count, input_dim=input_dim, seed=seed + 3)
    return CollapseData(
        train_inputs=train_inputs,
        train_labels=train_labels,
        probe_inputs=probe_inputs,
        probe_labels=probe_labels,
        unlabeled_pool=unlabeled_pool,
    )


def _train_classifier(
    model: TinySelfTrainingClassifier,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    for _epoch in range(int(epochs)):
        logits = model(inputs)
        loss = nn.functional.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def _evaluate(
    model: TinySelfTrainingClassifier,
    inputs: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[float, float, dict[int, torch.Tensor]]:
    with torch.no_grad():
        logits, states = model(inputs, return_states=True)
        loss = float(nn.functional.cross_entropy(logits, labels).detach().cpu().item())
        accuracy = float((logits.argmax(dim=-1) == labels).to(torch.float32).mean().detach().cpu().item())
    return loss, accuracy, states


def _self_training_batch(
    model: TinySelfTrainingClassifier,
    pool_inputs: torch.Tensor,
    *,
    generation: int,
    train_count: int,
    anchors_per_class_base: int,
    noise_scale: float,
    noise_decay: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if int(generation) < 1:
        raise ValueError("generation must be >= 1.")
    anchors_per_class = max(1, int(anchors_per_class_base) // (2 ** (int(generation) - 1)))
    with torch.no_grad():
        logits = model(pool_inputs)
        probabilities = torch.softmax(logits, dim=-1)
        confidence, pseudo_labels = probabilities.max(dim=-1)

    selected_inputs = []
    selected_labels = []
    selected_confidences = []
    for label in (0, 1):
        label_indexes = torch.nonzero(pseudo_labels == label, as_tuple=False).flatten()
        if int(label_indexes.numel()) == 0:
            label_indexes = torch.arange(int(pool_inputs.shape[0]))
        sorted_indexes = label_indexes[torch.argsort(confidence[label_indexes], descending=True)]
        top_indexes = sorted_indexes[:anchors_per_class]
        selected_inputs.append(pool_inputs[top_indexes])
        selected_labels.append(pseudo_labels[top_indexes])
        selected_confidences.append(confidence[top_indexes])

    anchors = torch.cat(selected_inputs, dim=0)
    labels = torch.cat(selected_labels, dim=0)
    confidences = torch.cat(selected_confidences, dim=0)
    repeats = (int(train_count) + int(anchors.shape[0]) - 1) // int(anchors.shape[0])
    expanded_inputs = anchors.repeat((repeats, 1))[: int(train_count)]
    expanded_labels = labels.repeat(repeats)[: int(train_count)]
    generator = torch.Generator().manual_seed(int(seed) + int(generation))
    noise = torch.randn(expanded_inputs.shape, generator=generator, dtype=expanded_inputs.dtype)
    synthetic_inputs = expanded_inputs + noise * float(noise_scale) * (float(noise_decay) ** int(generation))
    metadata = {
        "anchors_per_class": int(anchors_per_class),
        "anchor_count": int(anchors.shape[0]),
        "mean_anchor_confidence": float(confidences.mean().detach().cpu().item()),
        "min_anchor_confidence": float(confidences.min().detach().cpu().item()),
        "noise_scale": float(noise_scale) * (float(noise_decay) ** int(generation)),
    }
    return synthetic_inputs, expanded_labels, metadata


def _record_generation(
    recorder: RepresentationTelemetryRecorder,
    *,
    generation: int,
    model: TinySelfTrainingClassifier,
    data: CollapseData,
    target_layer: int,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    eval_loss, eval_accuracy, states = _evaluate(model, data.probe_inputs, data.probe_labels)
    snapshots = recorder.record_step(int(generation), states, metadata=metadata)
    target_snapshot = next(snapshot for snapshot in snapshots if int(snapshot.layer) == int(target_layer))
    intrinsic_report = twonn_intrinsic_dimension(states[int(target_layer)]).to_dict()
    return {
        "generation": int(generation),
        "eval_loss": eval_loss,
        "eval_accuracy": eval_accuracy,
        "telemetry": {str(snapshot.layer): snapshot.to_dict() for snapshot in snapshots},
        "target_layer": int(target_layer),
        "target_effective_rank": float(target_snapshot.effective_rank),
        "target_distance_to_baseline": (
            None
            if target_snapshot.distance_to_baseline is None
            else float(target_snapshot.distance_to_baseline)
        ),
        "target_intrinsic_dimension": float(intrinsic_report["intrinsic_dimension"]),
        "intrinsic_dimension_report": intrinsic_report,
        "self_training": dict(metadata),
    }


def _nonincreasing(values: Sequence[float], *, tolerance: float) -> bool:
    return all(float(right) <= float(left) + float(tolerance) for left, right in zip(values, values[1:]))


def _first_drop_generation(rows: Sequence[Mapping[str, Any]], *, key: str, min_drop: float) -> int | None:
    if not rows:
        return None
    first = float(rows[0][key])
    for row in rows[1:]:
        if first - float(row[key]) >= float(min_drop):
            return int(row["generation"])
    return None


def _first_quality_loss_generation(
    rows: Sequence[Mapping[str, Any]],
    *,
    min_accuracy_drop: float,
    min_eval_loss_increase: float,
) -> int | None:
    if not rows:
        return None
    first_accuracy = float(rows[0]["eval_accuracy"])
    first_loss = float(rows[0]["eval_loss"])
    for row in rows[1:]:
        accuracy_drop = first_accuracy - float(row["eval_accuracy"])
        loss_increase = float(row["eval_loss"]) - first_loss
        if accuracy_drop >= float(min_accuracy_drop) or loss_increase >= float(min_eval_loss_increase):
            return int(row["generation"])
    return None


def model_collapse_early_warning_report(
    *,
    train_count: int = 320,
    probe_count: int = 192,
    pool_count: int = 320,
    input_dim: int = 8,
    hidden_dim: int = 12,
    generations: int = 5,
    initial_epochs: int = 20,
    generation_epochs: int = 12,
    seed: int = 42,
    learning_rate: float = 0.03,
    weight_decay: float = 0.02,
    covariance_mode: str = "shrinkage",
    anchors_per_class_base: int = 16,
    noise_scale: float = 0.08,
    noise_decay: float = 0.55,
    target_layer: int = -1,
    min_rank_drop: float = 0.04,
    min_intrinsic_dimension_drop: float = 0.09,
    min_accuracy_drop: float = 0.05,
    min_eval_loss_increase: float = 0.05,
    monotonic_tolerance: float = 0.02,
) -> dict[str, Any]:
    """Run a deterministic pseudo-label self-training collapse check."""
    if int(train_count) < 16:
        raise ValueError("train_count must be >= 16.")
    if int(probe_count) < 16:
        raise ValueError("probe_count must be >= 16.")
    if int(pool_count) < 16:
        raise ValueError("pool_count must be >= 16.")
    if int(input_dim) < 3:
        raise ValueError("input_dim must be >= 3.")
    if int(hidden_dim) < 2:
        raise ValueError("hidden_dim must be >= 2.")
    if int(generations) < 1:
        raise ValueError("generations must be >= 1.")
    if int(initial_epochs) < 1:
        raise ValueError("initial_epochs must be >= 1.")
    if int(generation_epochs) < 1:
        raise ValueError("generation_epochs must be >= 1.")
    if int(anchors_per_class_base) < 1:
        raise ValueError("anchors_per_class_base must be >= 1.")
    if int(target_layer) not in {-2, -1}:
        raise ValueError("target_layer must be -2 or -1.")
    if not (0.0 < float(noise_decay) <= 1.0):
        raise ValueError("noise_decay must be in (0, 1].")

    data = _make_data(
        train_count=int(train_count),
        probe_count=int(probe_count),
        pool_count=int(pool_count),
        input_dim=int(input_dim),
        seed=int(seed),
    )
    torch.manual_seed(int(seed))
    model = TinySelfTrainingClassifier(input_dim=int(input_dim), hidden_dim=int(hidden_dim))
    _train_classifier(
        model,
        data.train_inputs,
        data.train_labels,
        epochs=int(initial_epochs),
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    _baseline_loss, _baseline_accuracy, baseline_states = _evaluate(model, data.probe_inputs, data.probe_labels)
    recorder = RepresentationTelemetryRecorder(
        layers=(-2, -1),
        covariance_mode=covariance_mode,
        baseline_strategy="manual",
        metadata={"workflow": "model_collapse_early_warning"},
    )
    recorder.set_baseline(baseline_states)

    rows = [
        _record_generation(
            recorder,
            generation=0,
            model=model,
            data=data,
            target_layer=int(target_layer),
            metadata={"phase": "clean_warm_start", "generation": 0},
        )
    ]
    for generation in range(1, int(generations) + 1):
        synthetic_inputs, pseudo_labels, batch_metadata = _self_training_batch(
            model,
            data.unlabeled_pool,
            generation=generation,
            train_count=int(train_count),
            anchors_per_class_base=int(anchors_per_class_base),
            noise_scale=float(noise_scale),
            noise_decay=float(noise_decay),
            seed=int(seed) + 1000,
        )
        _train_classifier(
            model,
            synthetic_inputs,
            pseudo_labels,
            epochs=int(generation_epochs),
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
        )
        rows.append(
            _record_generation(
                recorder,
                generation=generation,
                model=model,
                data=data,
                target_layer=int(target_layer),
                metadata={
                    "phase": "self_training",
                    "generation": generation,
                    **batch_metadata,
                },
            )
        )

    ranks = [float(row["target_effective_rank"]) for row in rows]
    intrinsic_dimensions = [float(row["target_intrinsic_dimension"]) for row in rows]
    rank_total_drop = float(ranks[0] - ranks[-1])
    intrinsic_dimension_total_drop = float(intrinsic_dimensions[0] - intrinsic_dimensions[-1])
    rank_monotonic = _nonincreasing(ranks, tolerance=float(monotonic_tolerance))
    intrinsic_dimension_monotonic = _nonincreasing(
        intrinsic_dimensions,
        tolerance=float(monotonic_tolerance),
    )
    rank_warning_generation = _first_drop_generation(
        rows,
        key="target_effective_rank",
        min_drop=float(min_rank_drop),
    )
    intrinsic_dimension_warning_generation = _first_drop_generation(
        rows,
        key="target_intrinsic_dimension",
        min_drop=float(min_intrinsic_dimension_drop),
    )
    intrinsic_dimension_supports_warning = (
        intrinsic_dimension_warning_generation is not None
        and intrinsic_dimension_total_drop >= float(min_intrinsic_dimension_drop)
    )
    quality_loss_generation = _first_quality_loss_generation(
        rows,
        min_accuracy_drop=float(min_accuracy_drop),
        min_eval_loss_increase=float(min_eval_loss_increase),
    )
    telemetry_warning_generation = min(
        generation
        for generation in (rank_warning_generation, intrinsic_dimension_warning_generation)
        if generation is not None
    ) if rank_warning_generation is not None or intrinsic_dimension_warning_generation is not None else None
    telemetry_leads_quality_loss = (
        telemetry_warning_generation is not None
        and (quality_loss_generation is None or int(telemetry_warning_generation) < int(quality_loss_generation))
    )
    rank_signal_pass = (
        rank_monotonic
        and rank_total_drop >= float(min_rank_drop)
        and telemetry_leads_quality_loss
    )
    status = "pass" if rank_signal_pass else "fail"

    return {
        "workflow": "model_collapse_early_warning",
        "config": {
            "train_count": int(train_count),
            "probe_count": int(probe_count),
            "pool_count": int(pool_count),
            "input_dim": int(input_dim),
            "hidden_dim": int(hidden_dim),
            "generations": int(generations),
            "initial_epochs": int(initial_epochs),
            "generation_epochs": int(generation_epochs),
            "seed": int(seed),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "covariance_mode": covariance_mode,
            "anchors_per_class_base": int(anchors_per_class_base),
            "noise_scale": float(noise_scale),
            "noise_decay": float(noise_decay),
            "target_layer": int(target_layer),
            "min_rank_drop": float(min_rank_drop),
            "min_intrinsic_dimension_drop": float(min_intrinsic_dimension_drop),
            "min_accuracy_drop": float(min_accuracy_drop),
            "min_eval_loss_increase": float(min_eval_loss_increase),
            "monotonic_tolerance": float(monotonic_tolerance),
        },
        "summary": {
            "status": status,
            "rank_monotonic": bool(rank_monotonic),
            "rank_total_drop": rank_total_drop,
            "rank_warning_generation": rank_warning_generation,
            "rank_signal_pass": bool(rank_signal_pass),
            "intrinsic_dimension_monotonic": bool(intrinsic_dimension_monotonic),
            "intrinsic_dimension_total_drop": intrinsic_dimension_total_drop,
            "intrinsic_dimension_warning_generation": intrinsic_dimension_warning_generation,
            "intrinsic_dimension_supports_warning": bool(intrinsic_dimension_supports_warning),
            "quality_loss_generation": quality_loss_generation,
            "telemetry_warning_generation": telemetry_warning_generation,
            "telemetry_leads_quality_loss": bool(telemetry_leads_quality_loss),
            "initial_effective_rank": ranks[0],
            "final_effective_rank": ranks[-1],
            "initial_intrinsic_dimension": intrinsic_dimensions[0],
            "final_intrinsic_dimension": intrinsic_dimensions[-1],
            "initial_eval_accuracy": float(rows[0]["eval_accuracy"]),
            "final_eval_accuracy": float(rows[-1]["eval_accuracy"]),
            "initial_eval_loss": float(rows[0]["eval_loss"]),
            "final_eval_loss": float(rows[-1]["eval_loss"]),
        },
        "generations": rows,
        "telemetry_report": recorder.to_report(
            metadata={"workflow": "model_collapse_early_warning", "source": "pseudo-label self-training"}
        ).to_dict(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    report = model_collapse_early_warning_report(
        train_count=args.train_count,
        probe_count=args.probe_count,
        pool_count=args.pool_count,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        generations=args.generations,
        initial_epochs=args.initial_epochs,
        generation_epochs=args.generation_epochs,
        seed=args.seed,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        covariance_mode=args.covariance_mode,
        anchors_per_class_base=args.anchors_per_class_base,
        noise_scale=args.noise_scale,
        noise_decay=args.noise_decay,
        target_layer=args.target_layer,
        min_rank_drop=args.min_rank_drop,
        min_intrinsic_dimension_drop=args.min_intrinsic_dimension_drop,
        min_accuracy_drop=args.min_accuracy_drop,
        min_eval_loss_increase=args.min_eval_loss_increase,
        monotonic_tolerance=args.monotonic_tolerance,
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
            {"model_collapse_early_warning_report": Path(args.json)},
            root=manifest_path.parent,
            metadata={
                "workflow": "model_collapse_early_warning",
                "covariance_mode": args.covariance_mode,
                "source": "tiny pseudo-label self-training loop",
            },
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a synthetic self-training collapse early-warning check")
    parser.add_argument("--train-count", type=int, default=320)
    parser.add_argument("--probe-count", type=int, default=192)
    parser.add_argument("--pool-count", type=int, default=320)
    parser.add_argument("--input-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=12)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--initial-epochs", type=int, default=20)
    parser.add_argument("--generation-epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--covariance-mode", default="shrinkage")
    parser.add_argument("--anchors-per-class-base", type=int, default=16)
    parser.add_argument("--noise-scale", type=float, default=0.08)
    parser.add_argument("--noise-decay", type=float, default=0.55)
    parser.add_argument("--target-layer", type=int, default=-1)
    parser.add_argument("--min-rank-drop", type=float, default=0.04)
    parser.add_argument("--min-intrinsic-dimension-drop", type=float, default=0.09)
    parser.add_argument("--min-accuracy-drop", type=float, default=0.05)
    parser.add_argument("--min-eval-loss-increase", type=float, default=0.05)
    parser.add_argument("--monotonic-tolerance", type=float, default=0.02)
    parser.add_argument("--json", default=None, help="optional path to write JSON report")
    parser.add_argument("--artifact-manifest", default=None, help="optional manifest path for --json output")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()

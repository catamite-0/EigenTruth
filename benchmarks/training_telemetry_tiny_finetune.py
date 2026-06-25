"""Tiny PyTorch fine-tune sanity check for representation telemetry."""

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

from eigentruth.registry import build_artifact_manifest  # noqa: E402
from eigentruth.training import RepresentationTelemetryRecorder  # noqa: E402


class TinyRepresentationClassifier(nn.Module):
    """Small MLP with two named hidden-state layers for telemetry tests."""

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
class TinyFineTuneData:
    train_inputs: torch.Tensor
    train_labels: torch.Tensor
    eval_inputs: torch.Tensor
    eval_labels: torch.Tensor


def _make_dataset(*, n: int, input_dim: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(int(seed))
    inputs = torch.randn(int(n), int(input_dim), generator=generator)
    weights = torch.linspace(-1.0, 1.0, int(input_dim))
    logits = (inputs @ weights) + (0.4 * torch.sin(inputs[:, 0] * 2.0)) - (0.2 * inputs[:, 1] * inputs[:, 2])
    labels = (logits > 0.0).to(torch.long)
    return inputs, labels


def _make_tiny_data(*, train_count: int, eval_count: int, input_dim: int, seed: int) -> TinyFineTuneData:
    train_inputs, train_labels = _make_dataset(n=train_count, input_dim=input_dim, seed=seed + 1)
    eval_inputs, eval_labels = _make_dataset(n=eval_count, input_dim=input_dim, seed=seed + 2)
    return TinyFineTuneData(
        train_inputs=train_inputs,
        train_labels=train_labels,
        eval_inputs=eval_inputs,
        eval_labels=eval_labels,
    )


def _duplicate_low_diversity_data(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    *,
    anchors_per_class: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if int(anchors_per_class) < 1:
        raise ValueError("anchors_per_class must be >= 1.")
    indexes: list[int] = []
    for label in (0, 1):
        label_indexes = torch.nonzero(labels == label, as_tuple=False).flatten()
        if int(label_indexes.numel()) < int(anchors_per_class):
            raise ValueError("not enough samples per class for duplicate corruption.")
        indexes.extend(int(index) for index in label_indexes[: int(anchors_per_class)])
    anchor_indexes = torch.tensor(indexes, dtype=torch.long)
    repeats = (int(labels.shape[0]) + int(anchor_indexes.numel()) - 1) // int(anchor_indexes.numel())
    expanded = anchor_indexes.repeat(repeats)[: int(labels.shape[0])]
    return inputs[expanded].clone(), labels[expanded].clone()


def _evaluate(
    model: TinyRepresentationClassifier,
    inputs: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[float, float, dict[int, torch.Tensor]]:
    with torch.no_grad():
        logits, states = model(inputs, return_states=True)
        loss = float(nn.functional.cross_entropy(logits, labels).detach().cpu().item())
        accuracy = float((logits.argmax(dim=-1) == labels).to(torch.float32).mean().detach().cpu().item())
    return loss, accuracy, states


def _train_run(
    *,
    run_type: str,
    data: TinyFineTuneData,
    initial_state: Mapping[str, torch.Tensor],
    baseline_states: Mapping[int, torch.Tensor],
    input_dim: int,
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    covariance_mode: str,
    duplicate_anchors_per_class: int,
) -> dict[str, Any]:
    model = TinyRepresentationClassifier(input_dim=input_dim, hidden_dim=hidden_dim)
    model.load_state_dict(dict(initial_state))
    train_inputs = data.train_inputs
    train_labels = data.train_labels
    corruption = "none"
    if run_type == "duplicate":
        train_inputs, train_labels = _duplicate_low_diversity_data(
            train_inputs,
            train_labels,
            anchors_per_class=duplicate_anchors_per_class,
        )
        corruption = "balanced_duplicate_low_diversity"
    elif run_type != "clean":
        raise ValueError("run_type must be 'clean' or 'duplicate'.")

    recorder = RepresentationTelemetryRecorder(
        layers=(-2, -1),
        covariance_mode=covariance_mode,
        baseline_strategy="manual",
        metadata={"run_type": run_type, "corruption": corruption},
    )
    recorder.set_baseline(baseline_states)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    metrics = []
    for epoch in range(int(epochs) + 1):
        eval_loss, eval_accuracy, states = _evaluate(model, data.eval_inputs, data.eval_labels)
        snapshots = recorder.record_step(
            epoch,
            states,
            metadata={"run_type": run_type, "corruption": corruption},
        )
        metrics.append({
            "epoch": int(epoch),
            "eval_loss": eval_loss,
            "eval_accuracy": eval_accuracy,
            "telemetry": {str(snapshot.layer): snapshot.to_dict() for snapshot in snapshots},
        })
        if epoch >= int(epochs):
            break
        logits = model(train_inputs)
        train_loss = nn.functional.cross_entropy(logits, train_labels)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

    report = recorder.to_report(metadata={"run_type": run_type, "corruption": corruption}).to_dict()
    report["metrics"] = metrics
    report["run_type"] = run_type
    report["corruption"] = corruption
    return report


def _first_separation_step(
    clean_metrics: Sequence[Mapping[str, Any]],
    corrupt_metrics: Sequence[Mapping[str, Any]],
    *,
    layer: int,
    min_rank_margin: float,
    min_eval_loss_margin: float,
) -> dict[str, Any]:
    rank_step = None
    loss_step = None
    max_rank_margin = None
    max_rank_margin_epoch = None
    rows = []
    for clean, corrupt in zip(clean_metrics, corrupt_metrics):
        epoch = int(clean["epoch"])
        clean_layer = clean["telemetry"][str(layer)]
        corrupt_layer = corrupt["telemetry"][str(layer)]
        rank_margin = float(clean_layer["effective_rank"]) - float(corrupt_layer["effective_rank"])
        loss_margin = float(corrupt["eval_loss"]) - float(clean["eval_loss"])
        distance_margin = float(corrupt_layer["distance_to_baseline"]) - float(clean_layer["distance_to_baseline"])
        if max_rank_margin is None or rank_margin > max_rank_margin:
            max_rank_margin = rank_margin
            max_rank_margin_epoch = epoch
        rows.append({
            "epoch": epoch,
            "rank_margin": rank_margin,
            "eval_loss_margin": loss_margin,
            "distance_margin": distance_margin,
        })
        if rank_step is None and rank_margin >= float(min_rank_margin):
            rank_step = epoch
        if loss_step is None and loss_margin >= float(min_eval_loss_margin):
            loss_step = epoch
    return {
        "rank_first_separation_epoch": rank_step,
        "eval_loss_first_separation_epoch": loss_step,
        "telemetry_leads_eval_loss": (
            rank_step is not None and (loss_step is None or int(rank_step) < int(loss_step))
        ),
        "max_rank_margin": max_rank_margin,
        "max_rank_margin_epoch": max_rank_margin_epoch,
        "per_epoch": rows,
    }


def tiny_finetune_telemetry_report(
    *,
    train_count: int = 320,
    eval_count: int = 192,
    input_dim: int = 8,
    hidden_dim: int = 12,
    epochs: int = 20,
    seed: int = 42,
    learning_rate: float = 0.03,
    weight_decay: float = 0.02,
    covariance_mode: str = "shrinkage",
    duplicate_anchors_per_class: int = 4,
    target_layer: int = -1,
    min_rank_margin: float = 0.5,
    min_eval_loss_margin: float = 0.05,
) -> dict[str, Any]:
    """Run deterministic clean-vs-duplicate tiny fine-tune telemetry comparison."""
    if int(train_count) < 8:
        raise ValueError("train_count must be >= 8.")
    if int(eval_count) < 8:
        raise ValueError("eval_count must be >= 8.")
    if int(input_dim) < 3:
        raise ValueError("input_dim must be >= 3.")
    if int(hidden_dim) < 2:
        raise ValueError("hidden_dim must be >= 2.")
    if int(epochs) < 1:
        raise ValueError("epochs must be >= 1.")
    if int(target_layer) not in {-2, -1}:
        raise ValueError("target_layer must be -2 or -1.")

    data = _make_tiny_data(
        train_count=int(train_count),
        eval_count=int(eval_count),
        input_dim=int(input_dim),
        seed=int(seed),
    )
    torch.manual_seed(int(seed))
    initial_model = TinyRepresentationClassifier(input_dim=int(input_dim), hidden_dim=int(hidden_dim))
    initial_state = {key: value.detach().clone() for key, value in initial_model.state_dict().items()}
    _initial_loss, _initial_accuracy, baseline_states = _evaluate(initial_model, data.eval_inputs, data.eval_labels)

    clean = _train_run(
        run_type="clean",
        data=data,
        initial_state=initial_state,
        baseline_states=baseline_states,
        input_dim=int(input_dim),
        hidden_dim=int(hidden_dim),
        epochs=int(epochs),
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
        covariance_mode=covariance_mode,
        duplicate_anchors_per_class=int(duplicate_anchors_per_class),
    )
    duplicate = _train_run(
        run_type="duplicate",
        data=data,
        initial_state=initial_state,
        baseline_states=baseline_states,
        input_dim=int(input_dim),
        hidden_dim=int(hidden_dim),
        epochs=int(epochs),
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
        covariance_mode=covariance_mode,
        duplicate_anchors_per_class=int(duplicate_anchors_per_class),
    )
    separation = _first_separation_step(
        clean["metrics"],
        duplicate["metrics"],
        layer=int(target_layer),
        min_rank_margin=float(min_rank_margin),
        min_eval_loss_margin=float(min_eval_loss_margin),
    )
    clean_final = clean["metrics"][-1]["telemetry"][str(target_layer)]
    duplicate_final = duplicate["metrics"][-1]["telemetry"][str(target_layer)]
    final_rank_margin = float(clean_final["effective_rank"]) - float(duplicate_final["effective_rank"])
    final_loss_margin = float(duplicate["metrics"][-1]["eval_loss"]) - float(clean["metrics"][-1]["eval_loss"])
    status = "pass" if separation["telemetry_leads_eval_loss"] else "fail"

    return {
        "workflow": "training_telemetry_tiny_finetune",
        "config": {
            "train_count": int(train_count),
            "eval_count": int(eval_count),
            "input_dim": int(input_dim),
            "hidden_dim": int(hidden_dim),
            "epochs": int(epochs),
            "seed": int(seed),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "covariance_mode": covariance_mode,
            "duplicate_anchors_per_class": int(duplicate_anchors_per_class),
            "target_layer": int(target_layer),
            "min_rank_margin": float(min_rank_margin),
            "min_eval_loss_margin": float(min_eval_loss_margin),
        },
        "summary": {
            "status": status,
            "telemetry_leads_eval_loss": bool(separation["telemetry_leads_eval_loss"]),
            "rank_first_separation_epoch": separation["rank_first_separation_epoch"],
            "eval_loss_first_separation_epoch": separation["eval_loss_first_separation_epoch"],
            "max_rank_margin": separation["max_rank_margin"],
            "max_rank_margin_epoch": separation["max_rank_margin_epoch"],
            "final_rank_margin": final_rank_margin,
            "final_eval_loss_margin": final_loss_margin,
            "clean_final_effective_rank": float(clean_final["effective_rank"]),
            "duplicate_final_effective_rank": float(duplicate_final["effective_rank"]),
            "clean_final_distance": float(clean_final["distance_to_baseline"]),
            "duplicate_final_distance": float(duplicate_final["distance_to_baseline"]),
            "clean_final_eval_accuracy": float(clean["metrics"][-1]["eval_accuracy"]),
            "duplicate_final_eval_accuracy": float(duplicate["metrics"][-1]["eval_accuracy"]),
        },
        "separation": separation,
        "runs": {
            "clean": clean,
            "duplicate": duplicate,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    report = tiny_finetune_telemetry_report(
        train_count=args.train_count,
        eval_count=args.eval_count,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        seed=args.seed,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        covariance_mode=args.covariance_mode,
        duplicate_anchors_per_class=args.duplicate_anchors_per_class,
        target_layer=args.target_layer,
        min_rank_margin=args.min_rank_margin,
        min_eval_loss_margin=args.min_eval_loss_margin,
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
            {"training_telemetry_tiny_finetune_report": Path(args.json)},
            root=manifest_path.parent,
            metadata={
                "workflow": "training_telemetry_tiny_finetune",
                "covariance_mode": args.covariance_mode,
                "source": "tiny PyTorch clean-vs-duplicate fine-tune",
            },
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run tiny fine-tune representation telemetry check")
    parser.add_argument("--train-count", type=int, default=320)
    parser.add_argument("--eval-count", type=int, default=192)
    parser.add_argument("--input-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--covariance-mode", default="shrinkage")
    parser.add_argument("--duplicate-anchors-per-class", type=int, default=4)
    parser.add_argument("--target-layer", type=int, default=-1)
    parser.add_argument("--min-rank-margin", type=float, default=0.5)
    parser.add_argument("--min-eval-loss-margin", type=float, default=0.05)
    parser.add_argument("--json", default=None, help="optional path to write JSON report")
    parser.add_argument("--artifact-manifest", default=None, help="optional manifest path for --json output")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()

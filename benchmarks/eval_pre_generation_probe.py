"""Train/evaluate a pre-generation soft-target attention probe from local hidden states."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from eigentruth.core import AttentionSoftTargetProbeArtifact, soft_error_rate_targets
from eigentruth.eval.metrics import roc_auc
from eigentruth.json_utils import to_jsonable


@dataclass(frozen=True)
class PreGenerationProbeRecord:
    """One prompt-level hidden-state record for pre-generation probe training."""

    record_id: str
    hidden_states: torch.Tensor
    attention_mask: torch.Tensor
    soft_target: float
    label: int | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        hidden_states = torch.as_tensor(self.hidden_states, dtype=torch.float32)
        if hidden_states.ndim != 2:
            raise ValueError("hidden_states must have shape [T, D].")
        if min(hidden_states.shape) < 1:
            raise ValueError("hidden_states dimensions must be non-empty.")
        if not torch.isfinite(hidden_states).all():
            raise ValueError("hidden_states must contain only finite values.")
        attention_mask = torch.as_tensor(self.attention_mask, dtype=torch.bool)
        if attention_mask.ndim != 1 or attention_mask.shape[0] != hidden_states.shape[0]:
            raise ValueError("attention_mask must have shape [T] matching hidden_states.")
        if not bool(attention_mask.any().item()):
            raise ValueError("attention_mask must keep at least one token.")
        soft_target = _probability_float(self.soft_target, field_name="soft_target")
        label = None if self.label is None else _binary_label(self.label, field_name="label")
        object.__setattr__(self, "record_id", str(self.record_id))
        object.__setattr__(self, "hidden_states", hidden_states)
        object.__setattr__(self, "attention_mask", attention_mask)
        object.__setattr__(self, "soft_target", soft_target)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "metadata", {} if self.metadata is None else to_jsonable(self.metadata))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], *, index: int) -> "PreGenerationProbeRecord":
        """Build a record from a JSON-like mapping."""
        hidden_states = payload.get("hidden_states", payload.get("prompt_hidden_states"))
        if hidden_states is None:
            raise ValueError(f"record {index} is missing hidden_states.")
        mask = payload.get("attention_mask")
        if mask is None:
            mask = [True] * len(hidden_states)
        soft_target = payload.get("soft_target", payload.get("risk_target"))
        if soft_target is None:
            correctness = payload.get("sample_correctness")
            if correctness is None:
                raise ValueError(f"record {index} must provide soft_target, risk_target, or sample_correctness.")
            soft_target = float(soft_error_rate_targets([correctness])[0].item())
        label = _first_present(payload, ("label", "is_false"))
        record_id = _first_present(payload, ("id", "record_id", "claim_id", "question_id"))
        return cls(
            record_id=f"r{index}" if record_id is None else str(record_id),
            hidden_states=torch.as_tensor(hidden_states, dtype=torch.float32),
            attention_mask=torch.as_tensor(mask, dtype=torch.bool),
            soft_target=float(soft_target),
            label=None if label is None else _binary_label(label, field_name="label"),
            metadata=payload.get("metadata", {}),
        )


def run_pre_generation_probe_eval(
    records_path: str | Path,
    *,
    output_path: str | Path | None = None,
    artifact_path: str | Path | None = None,
    train_fraction: float = 0.7,
    seed: int = 0,
    layer_idx: int | None = None,
    steps: int = 300,
    lr: float = 0.05,
    l2: float = 1e-4,
    compact_json: bool = False,
) -> dict[str, Any]:
    """Train/evaluate an attention soft-target probe from local hidden-state records."""
    records_path = Path(records_path)
    records = load_pre_generation_probe_records(records_path)
    train_fraction = _train_fraction(train_fraction)
    if not records:
        raise ValueError("records must not be empty.")
    hidden_dim = int(records[0].hidden_states.shape[-1])
    for record in records:
        if int(record.hidden_states.shape[-1]) != hidden_dim:
            raise ValueError("all records must share the same hidden dimension.")
    train_indices, test_indices = _split_indices(records, train_fraction=train_fraction, seed=seed)
    train_hidden, train_mask, train_targets, _train_labels = _stack_records(records, train_indices)
    artifact = AttentionSoftTargetProbeArtifact.fit(
        train_hidden,
        train_targets,
        attention_mask=train_mask,
        layer_idx=layer_idx,
        steps=steps,
        lr=lr,
        l2=l2,
        seed=seed,
        metadata={
            "workflow": "pre_generation_probe_eval",
            "records_path": str(records_path),
            "train_fraction": train_fraction,
        },
    )
    split_metrics = {
        "train": _evaluate_split(artifact, records, train_indices),
        "test": _evaluate_split(artifact, records, test_indices),
        "all": _evaluate_split(artifact, records, tuple(range(len(records)))),
    }
    payload = {
        "workflow": "pre_generation_probe_eval",
        "records_path": str(records_path),
        "record_count": len(records),
        "config": {
            "train_fraction": train_fraction,
            "seed": int(seed),
            "layer_idx": layer_idx,
            "steps": int(steps),
            "lr": float(lr),
            "l2": float(l2),
            "hidden_dim": hidden_dim,
        },
        "split": {
            "train_indices": tuple(int(index) for index in train_indices),
            "test_indices": tuple(int(index) for index in test_indices),
            "train_record_ids": tuple(records[index].record_id for index in train_indices),
            "test_record_ids": tuple(records[index].record_id for index in test_indices),
        },
        "artifact": artifact.to_dict(),
        "metrics": split_metrics,
        "paths": {},
    }
    if artifact_path is not None:
        artifact_output = Path(artifact_path)
        artifact.save(artifact_output)
        payload["paths"]["artifact"] = str(artifact_output)
    if output_path is not None:
        output = Path(output_path)
        payload["paths"]["report"] = str(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            json.dump(to_jsonable(payload), f, indent=None if compact_json else 2, sort_keys=True)
            f.write("\n")
    return to_jsonable(payload)


def load_pre_generation_probe_records(path: str | Path) -> tuple[PreGenerationProbeRecord, ...]:
    """Load pre-generation probe records from JSON or JSONL."""
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open(encoding="utf-8") as f:
            for index, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, Mapping):
                    raise ValueError(f"JSONL line {index} must be an object.")
                records.append(PreGenerationProbeRecord.from_mapping(payload, index=len(records)))
        return tuple(records)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        raw_records = payload.get("records")
    else:
        raw_records = payload
    if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes, bytearray)):
        raise ValueError("records JSON must be a list or an object with a records list.")
    records = []
    for index, item in enumerate(raw_records):
        if not isinstance(item, Mapping):
            raise ValueError(f"record {index} must be an object.")
        records.append(PreGenerationProbeRecord.from_mapping(item, index=index))
    return tuple(records)


def run(args: argparse.Namespace) -> dict[str, Any]:
    """CLI adapter for tests and command-line use."""
    return run_pre_generation_probe_eval(
        args.records,
        output_path=args.json,
        artifact_path=args.save_artifact,
        train_fraction=args.train_fraction,
        seed=args.seed,
        layer_idx=args.layer_idx,
        steps=args.steps,
        lr=args.lr,
        l2=args.l2,
        compact_json=args.compact_json,
    )


def _evaluate_split(
    artifact: AttentionSoftTargetProbeArtifact,
    records: Sequence[PreGenerationProbeRecord],
    indices: Sequence[int],
) -> dict[str, Any]:
    if not indices:
        return {
            "record_count": 0,
            "target_mse": None,
            "target_mae": None,
            "target_bce": None,
            "label_auroc": None,
            "mean_probability": None,
        }
    hidden, mask, targets, labels = _stack_records(records, indices)
    with torch.no_grad():
        probabilities = artifact.predict_proba(hidden, attention_mask=mask)
    target_mse = torch.mean((probabilities - targets) ** 2).item()
    target_mae = torch.mean(torch.abs(probabilities - targets)).item()
    target_bce = F.binary_cross_entropy(probabilities.clamp(min=1e-6, max=1.0 - 1e-6), targets).item()
    label_auroc = None
    label_counts = None
    if labels is not None:
        label_values = labels.to(torch.int64)
        positives = int((label_values == 1).sum().item())
        negatives = int((label_values == 0).sum().item())
        label_counts = {"positive": positives, "negative": negatives}
        label_auroc = None if positives == 0 or negatives == 0 else roc_auc(probabilities, label_values)
    return {
        "record_count": len(indices),
        "target_mse": target_mse,
        "target_mae": target_mae,
        "target_bce": target_bce,
        "label_auroc": label_auroc,
        "label_counts": label_counts,
        "mean_probability": float(probabilities.mean().item()),
        "min_probability": float(probabilities.min().item()),
        "max_probability": float(probabilities.max().item()),
    }


def _stack_records(
    records: Sequence[PreGenerationProbeRecord],
    indices: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if not indices:
        raise ValueError("indices must not be empty.")
    selected = tuple(records[int(index)] for index in indices)
    hidden_dim = int(selected[0].hidden_states.shape[-1])
    max_len = max(int(record.hidden_states.shape[0]) for record in selected)
    hidden = torch.zeros((len(selected), max_len, hidden_dim), dtype=torch.float32)
    mask = torch.zeros((len(selected), max_len), dtype=torch.bool)
    targets = torch.tensor([record.soft_target for record in selected], dtype=torch.float32)
    labels: list[int] = []
    all_labeled = True
    for row, record in enumerate(selected):
        seq_len = int(record.hidden_states.shape[0])
        hidden[row, :seq_len, :] = record.hidden_states
        mask[row, :seq_len] = record.attention_mask
        if record.label is None:
            all_labeled = False
        else:
            labels.append(int(record.label))
    return hidden, mask, targets, torch.tensor(labels, dtype=torch.int64) if all_labeled else None


def _split_indices(
    records: Sequence[PreGenerationProbeRecord],
    *,
    train_fraction: float,
    seed: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    n_records = len(records)
    if n_records < 2:
        raise ValueError("at least two records are required for train/test evaluation.")
    rng = random.Random(int(seed))
    labels = [record.label for record in records]
    if all(label is not None for label in labels) and len(set(labels)) > 1:
        groups: dict[int, list[int]] = {}
        for index, label in enumerate(labels):
            groups.setdefault(int(label), []).append(index)
        train: list[int] = []
        test: list[int] = []
        for group in groups.values():
            rng.shuffle(group)
            split = _group_split_size(len(group), train_fraction=train_fraction)
            train.extend(group[:split])
            test.extend(group[split:])
    else:
        indices = list(range(n_records))
        rng.shuffle(indices)
        split = _group_split_size(n_records, train_fraction=train_fraction)
        train = indices[:split]
        test = indices[split:]
    train.sort()
    test.sort()
    if not train or not test:
        raise ValueError("train/test split produced an empty split.")
    return tuple(train), tuple(test)


def _group_split_size(size: int, *, train_fraction: float) -> int:
    if size <= 1:
        return size
    return min(size - 1, max(1, round(size * train_fraction)))


def _first_present(payload: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _binary_label(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "false", "0", "no"}:
            return 1 if normalized in {"true", "1", "yes"} else 0
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be binary 0/1.") from exc
    if parsed not in {0, 1}:
        raise ValueError(f"{field_name} must be binary 0/1.")
    return parsed


def _probability_float(value: Any, *, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be in [0, 1].") from exc
    if not 0.0 <= parsed <= 1.0 or not (parsed == parsed) or parsed in {float("inf"), float("-inf")}:
        raise ValueError(f"{field_name} must be in [0, 1].")
    return parsed


def _train_fraction(value: Any) -> float:
    parsed = _probability_float(value, field_name="train_fraction")
    if parsed <= 0.0 or parsed >= 1.0:
        raise ValueError("train_fraction must be > 0 and < 1.")
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description="Train/evaluate a pre-generation attention risk probe")
    parser.add_argument("--records", required=True, help="JSON/JSONL records with hidden_states and soft targets")
    parser.add_argument("--json", help="optional report output path")
    parser.add_argument("--save-artifact", help="optional torch artifact output path")
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layer-idx", type=int)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--compact-json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    payload = run(args)
    if args.json is None:
        print(json.dumps(payload, indent=None if args.compact_json else 2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

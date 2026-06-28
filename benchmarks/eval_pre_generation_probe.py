"""Train/evaluate a pre-generation soft-target attention probe from local hidden states."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from eigentruth import __version__
from eigentruth.calibration import CalibrationArtifact, CalibrationScore
from eigentruth.core import AttentionSoftTargetProbeArtifact, soft_error_rate_targets
from eigentruth.eval.conformal import (
    directional_conformal_threshold,
    evaluate_conformal_abstention,
)
from eigentruth.eval.metrics import roc_auc, selective_classification_report
from eigentruth.json_utils import to_jsonable

PRE_GENERATION_RISK_SCORE_NAME = "pre_generation_risk_probability"
PRE_GENERATION_SWEEP_BEST_BY = (
    "auto",
    "label_auroc",
    "target_bce",
    "target_mse",
    "target_mae",
    "conformal_selective_accuracy",
    "conformal_conditional_lower_bound",
)
_LOWER_IS_BETTER_SWEEP_METRICS = {"target_bce", "target_mse", "target_mae"}


@dataclass(frozen=True)
class PreGenerationProbeRecord:
    """One prompt-level hidden-state record for pre-generation probe training."""

    record_id: str
    hidden_states: torch.Tensor
    attention_mask: torch.Tensor
    soft_target: float
    label: int | None = None
    selected_layer: int | None = None
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
        object.__setattr__(self, "selected_layer", None if self.selected_layer is None else int(self.selected_layer))
        object.__setattr__(self, "metadata", {} if self.metadata is None else to_jsonable(self.metadata))

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        index: int,
        record_layer: int | None = None,
    ) -> "PreGenerationProbeRecord":
        """Build a record from a JSON-like mapping."""
        hidden_states, selected_layer = _hidden_states_from_payload(
            payload,
            index=index,
            record_layer=record_layer,
        )
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
            selected_layer=selected_layer,
            metadata=payload.get("metadata", {}),
        )


def run_pre_generation_probe_eval(
    records_path: str | Path,
    *,
    output_path: str | Path | None = None,
    artifact_path: str | Path | None = None,
    calibration_path: str | Path | None = None,
    calibration_model_id: str = "pre_generation_probe",
    conformal_alpha: float = 0.1,
    soft_target_cutoff: float | None = None,
    record_layer: int | None = None,
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
    records = load_pre_generation_probe_records(records_path, record_layer=record_layer)
    train_fraction = _train_fraction(train_fraction)
    if not records:
        raise ValueError("records must not be empty.")
    artifact_layer_idx = layer_idx if layer_idx is not None else _infer_record_layer(records)
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
        layer_idx=artifact_layer_idx,
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
    conformal_report = _pre_generation_conformal_report(
        artifact,
        records,
        train_indices=train_indices,
        test_indices=test_indices,
        alpha=conformal_alpha,
        soft_target_cutoff=soft_target_cutoff,
    )
    calibration_artifact = None
    if calibration_path is not None:
        if not conformal_report["available"]:
            raise ValueError(
                "cannot save calibration artifact because conformal calibration is unavailable: "
                f"{conformal_report['reason']}"
            )
        calibration_artifact = _build_pre_generation_calibration_artifact(
            records_path=records_path,
            threshold=conformal_report["threshold"],
            alpha=conformal_alpha,
            layer_idx=artifact_layer_idx,
            model_id=calibration_model_id,
            record_count=len(records),
            train_count=len(train_indices),
            test_count=len(test_indices),
            label_source=conformal_report["label_source"],
            soft_target_cutoff=soft_target_cutoff,
        )
    payload = {
        "workflow": "pre_generation_probe_eval",
        "records_path": str(records_path),
        "record_count": len(records),
        "config": {
            "train_fraction": train_fraction,
            "seed": int(seed),
            "layer_idx": layer_idx,
            "artifact_layer_idx": artifact_layer_idx,
            "record_layer": record_layer,
            "steps": int(steps),
            "lr": float(lr),
            "l2": float(l2),
            "hidden_dim": hidden_dim,
            "conformal_alpha": float(conformal_alpha),
            "soft_target_cutoff": soft_target_cutoff,
        },
        "split": {
            "train_indices": tuple(int(index) for index in train_indices),
            "test_indices": tuple(int(index) for index in test_indices),
            "train_record_ids": tuple(records[index].record_id for index in train_indices),
            "test_record_ids": tuple(records[index].record_id for index in test_indices),
        },
        "artifact": artifact.to_dict(),
        "metrics": split_metrics,
        "conformal": conformal_report,
        "calibration_artifact": None if calibration_artifact is None else calibration_artifact.to_dict(),
        "paths": {},
    }
    if artifact_path is not None:
        artifact_output = Path(artifact_path)
        artifact.save(artifact_output)
        payload["paths"]["artifact"] = str(artifact_output)
    if calibration_path is not None and calibration_artifact is not None:
        calibration_output = Path(calibration_path)
        calibration_output.parent.mkdir(parents=True, exist_ok=True)
        calibration_artifact.save_json(calibration_output)
        payload["paths"]["calibration"] = str(calibration_output)
    if output_path is not None:
        output = Path(output_path)
        payload["paths"]["report"] = str(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            json.dump(to_jsonable(payload), f, indent=None if compact_json else 2, sort_keys=True)
            f.write("\n")
    return to_jsonable(payload)


def run_pre_generation_probe_layer_sweep(
    records_path: str | Path,
    *,
    sweep_layers: Sequence[int] | None = None,
    output_path: str | Path | None = None,
    artifact_path: str | Path | None = None,
    calibration_path: str | Path | None = None,
    calibration_model_id: str = "pre_generation_probe",
    conformal_alpha: float = 0.1,
    soft_target_cutoff: float | None = None,
    best_by: str = "auto",
    train_fraction: float = 0.7,
    seed: int = 0,
    steps: int = 300,
    lr: float = 0.05,
    l2: float = 1e-4,
    compact_json: bool = False,
) -> dict[str, Any]:
    """Train/evaluate one pre-generation probe per selected record layer."""
    records_path = Path(records_path)
    layers = (
        tuple(_discover_record_layers(records_path))
        if sweep_layers is None
        else tuple(int(layer) for layer in sweep_layers)
    )
    if not layers:
        raise ValueError("sweep_layers must contain at least one layer or records must expose layer_hidden_states.")
    if best_by not in PRE_GENERATION_SWEEP_BEST_BY:
        raise ValueError(f"best_by must be one of {PRE_GENERATION_SWEEP_BEST_BY}.")

    candidates: list[dict[str, Any]] = []
    for layer in layers:
        candidate_payload = run_pre_generation_probe_eval(
            records_path,
            calibration_model_id=calibration_model_id,
            conformal_alpha=conformal_alpha,
            soft_target_cutoff=soft_target_cutoff,
            record_layer=int(layer),
            train_fraction=train_fraction,
            seed=seed,
            layer_idx=int(layer),
            steps=steps,
            lr=lr,
            l2=l2,
            compact_json=compact_json,
        )
        candidates.append(_sweep_candidate_from_payload(int(layer), candidate_payload))

    resolved_best_by = _resolve_sweep_best_by(candidates, best_by)
    ranked = _rank_sweep_candidates(candidates, resolved_best_by)
    recommended = None if not ranked else ranked[0]
    saved_recommended = None
    if recommended is not None and (artifact_path is not None or calibration_path is not None):
        saved_recommended = run_pre_generation_probe_eval(
            records_path,
            artifact_path=artifact_path,
            calibration_path=calibration_path,
            calibration_model_id=calibration_model_id,
            conformal_alpha=conformal_alpha,
            soft_target_cutoff=soft_target_cutoff,
            record_layer=int(recommended["layer"]),
            train_fraction=train_fraction,
            seed=seed,
            layer_idx=int(recommended["layer"]),
            steps=steps,
            lr=lr,
            l2=l2,
            compact_json=compact_json,
        )

    payload = {
        "workflow": "pre_generation_probe_layer_sweep",
        "records_path": str(records_path),
        "candidate_count": len(ranked),
        "config": {
            "sweep_layers": tuple(int(layer) for layer in layers),
            "best_by": best_by,
            "resolved_best_by": resolved_best_by,
            "train_fraction": _train_fraction(train_fraction),
            "seed": int(seed),
            "steps": int(steps),
            "lr": float(lr),
            "l2": float(l2),
            "conformal_alpha": _alpha_float(conformal_alpha),
            "soft_target_cutoff": soft_target_cutoff,
        },
        "recommended": recommended,
        "candidates": ranked,
        "paths": {},
    }
    if saved_recommended is not None:
        payload["saved_recommended"] = {
            "layer": int(recommended["layer"]) if recommended is not None else None,
            "paths": dict(saved_recommended.get("paths", {})),
            "artifact": saved_recommended.get("artifact"),
            "calibration_artifact": saved_recommended.get("calibration_artifact"),
        }
        for key, value in saved_recommended.get("paths", {}).items():
            payload["paths"][f"best_{key}"] = value
    if output_path is not None:
        output = Path(output_path)
        payload["paths"]["report"] = str(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            json.dump(to_jsonable(payload), f, indent=None if compact_json else 2, sort_keys=True)
            f.write("\n")
    return to_jsonable(payload)


def load_pre_generation_probe_records(
    path: str | Path,
    *,
    record_layer: int | None = None,
) -> tuple[PreGenerationProbeRecord, ...]:
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
                records.append(
                    PreGenerationProbeRecord.from_mapping(
                        payload,
                        index=len(records),
                        record_layer=record_layer,
                    )
                )
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
        records.append(PreGenerationProbeRecord.from_mapping(item, index=index, record_layer=record_layer))
    return tuple(records)


def _discover_record_layers(path: str | Path) -> tuple[int, ...]:
    layers: list[int] = []
    seen: set[int] = set()
    for index, payload in enumerate(_iter_record_payloads(path)):
        layer_payload = _first_present(payload, ("layer_hidden_states", "prompt_hidden_states_by_layer"))
        if isinstance(layer_payload, Mapping):
            for raw_layer in layer_payload:
                layer = _parse_layer_key(raw_layer, index=index)
                if layer not in seen:
                    seen.add(layer)
                    layers.append(layer)
            continue
        layer = _metadata_layer(payload)
        if layer is not None and layer not in seen:
            seen.add(layer)
            layers.append(layer)
    return tuple(layers)


def _iter_record_payloads(path: str | Path) -> tuple[Mapping[str, Any], ...]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        records: list[Mapping[str, Any]] = []
        with path.open(encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, Mapping):
                    raise ValueError(f"JSONL line {line_number} must be an object.")
                records.append(payload)
        return tuple(records)
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_records = payload.get("records") if isinstance(payload, Mapping) else payload
    if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes, bytearray)):
        raise ValueError("records JSON must be a list or an object with a records list.")
    records = []
    for index, item in enumerate(raw_records):
        if not isinstance(item, Mapping):
            raise ValueError(f"record {index} must be an object.")
        records.append(item)
    return tuple(records)


def run(args: argparse.Namespace) -> dict[str, Any]:
    """CLI adapter for tests and command-line use."""
    if args.sweep_layers is not None:
        return run_pre_generation_probe_layer_sweep(
            args.records,
            sweep_layers=_parse_sweep_layers(args.sweep_layers),
            output_path=args.json,
            artifact_path=args.save_artifact,
            calibration_path=args.save_calibration,
            calibration_model_id=args.calibration_model_id,
            conformal_alpha=args.conformal_alpha,
            soft_target_cutoff=args.soft_target_cutoff,
            best_by=args.best_by,
            train_fraction=args.train_fraction,
            seed=args.seed,
            steps=args.steps,
            lr=args.lr,
            l2=args.l2,
            compact_json=args.compact_json,
        )
    return run_pre_generation_probe_eval(
        args.records,
        output_path=args.json,
        artifact_path=args.save_artifact,
        calibration_path=args.save_calibration,
        calibration_model_id=args.calibration_model_id,
        conformal_alpha=args.conformal_alpha,
        soft_target_cutoff=args.soft_target_cutoff,
        record_layer=args.record_layer,
        train_fraction=args.train_fraction,
        seed=args.seed,
        layer_idx=args.layer_idx,
        steps=args.steps,
        lr=args.lr,
        l2=args.l2,
        compact_json=args.compact_json,
    )


def _sweep_candidate_from_payload(layer: int, payload: Mapping[str, Any]) -> dict[str, Any]:
    test_metrics = dict(payload.get("metrics", {}).get("test", {}))
    all_metrics = dict(payload.get("metrics", {}).get("all", {}))
    conformal = dict(payload.get("conformal", {}))
    conformal_test = {}
    if conformal.get("available"):
        split_reports = conformal.get("split_reports", {})
        if isinstance(split_reports, Mapping):
            conformal_test = dict(split_reports.get("test", {}))
    selection_metrics = {
        "label_auroc": test_metrics.get("label_auroc"),
        "target_bce": test_metrics.get("target_bce"),
        "target_mse": test_metrics.get("target_mse"),
        "target_mae": test_metrics.get("target_mae"),
        "conformal_selective_accuracy": _nested_metric(
            conformal_test,
            ("selective", "selective_accuracy"),
        ),
        "conformal_conditional_lower_bound": _nested_metric(
            conformal_test,
            ("abstention", "conditional_correctness_lower_bound"),
        ),
    }
    return {
        "layer": int(layer),
        "record_count": int(payload.get("record_count", 0)),
        "selection_metrics": selection_metrics,
        "metrics": {
            "test": test_metrics,
            "all": all_metrics,
        },
        "conformal": {
            "available": bool(conformal.get("available", False)),
            "threshold": conformal.get("threshold"),
            "label_source": conformal.get("label_source"),
            "soft_target_cutoff": conformal.get("soft_target_cutoff"),
            "test_selective": conformal_test.get("selective"),
            "test_abstention": conformal_test.get("abstention"),
        },
        "artifact": payload.get("artifact"),
    }


def _resolve_sweep_best_by(candidates: Sequence[Mapping[str, Any]], best_by: str) -> str:
    if best_by != "auto":
        return best_by
    if any(_metric_value(candidate, "label_auroc") is not None for candidate in candidates):
        return "label_auroc"
    return "target_bce"


def _rank_sweep_candidates(candidates: Sequence[dict[str, Any]], best_by: str) -> list[dict[str, Any]]:
    def sort_key(candidate: Mapping[str, Any]) -> tuple[float, int]:
        value = _selection_value(candidate, best_by)
        sortable = float("-inf") if value is None else float(value)
        return (-sortable, int(candidate["layer"]))

    ranked = []
    for candidate in sorted(candidates, key=sort_key):
        value = _selection_value(candidate, best_by)
        row = dict(candidate)
        row["selection_metric"] = best_by
        row["selection_raw_value"] = _metric_value(candidate, best_by)
        row["selection_value"] = value
        ranked.append(row)
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked


def _selection_value(candidate: Mapping[str, Any], metric: str) -> float | None:
    value = _metric_value(candidate, metric)
    if value is None:
        return None
    return -value if metric in _LOWER_IS_BETTER_SWEEP_METRICS else value


def _metric_value(candidate: Mapping[str, Any], metric: str) -> float | None:
    raw = dict(candidate.get("selection_metrics", {})).get(metric)
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value != value or value in {float("inf"), float("-inf")}:
        return None
    return value


def _nested_metric(payload: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _pre_generation_conformal_report(
    artifact: AttentionSoftTargetProbeArtifact,
    records: Sequence[PreGenerationProbeRecord],
    *,
    train_indices: Sequence[int],
    test_indices: Sequence[int],
    alpha: float,
    soft_target_cutoff: float | None,
) -> dict[str, Any]:
    alpha_value = _alpha_float(alpha)
    cutoff_value = (
        None
        if soft_target_cutoff is None
        else _probability_float(soft_target_cutoff, field_name="soft_target_cutoff")
    )
    label_result = _false_labels_for_records(records, soft_target_cutoff=soft_target_cutoff)
    if label_result is None:
        return {
            "available": False,
            "reason": (
                "records do not all provide hard labels; pass --soft-target-cutoff "
                "to derive a calibration label from soft_target"
            ),
            "score_name": PRE_GENERATION_RISK_SCORE_NAME,
            "direction": "higher",
            "alpha": alpha_value,
            "label_source": None,
            "soft_target_cutoff": cutoff_value,
        }
    false_labels, label_source = label_result
    probabilities = _record_probabilities(artifact, records)
    train_scores = probabilities[torch.as_tensor(tuple(train_indices), dtype=torch.int64)]
    test_scores = probabilities[torch.as_tensor(tuple(test_indices), dtype=torch.int64)]
    train_false = false_labels[torch.as_tensor(tuple(train_indices), dtype=torch.int64)]
    test_false = false_labels[torch.as_tensor(tuple(test_indices), dtype=torch.int64)]
    normal_train_scores = train_scores[train_false == 0]
    if int(normal_train_scores.numel()) == 0:
        return {
            "available": False,
            "reason": "training split has no normal records for conformal calibration",
            "score_name": PRE_GENERATION_RISK_SCORE_NAME,
            "direction": "higher",
            "alpha": alpha_value,
            "label_source": label_source,
            "soft_target_cutoff": cutoff_value,
        }
    threshold = directional_conformal_threshold(normal_train_scores, alpha_value, "higher")
    all_indices = tuple(range(len(records)))
    all_scores = probabilities
    all_false = false_labels

    return {
        "available": True,
        "score_name": PRE_GENERATION_RISK_SCORE_NAME,
        "direction": "higher",
        "alpha": alpha_value,
        "threshold": threshold,
        "label_source": label_source,
        "soft_target_cutoff": cutoff_value,
        "calibration_record_count": int(len(train_indices)),
        "calibration_normal_count": int(normal_train_scores.numel()),
        "calibration_anomalous_count": int((train_false == 1).sum().item()),
        "evaluation_record_count": int(len(test_indices)),
        "split_reports": {
            "train": _selective_probe_report(train_scores, train_false, threshold, alpha_value),
            "test": _selective_probe_report(test_scores, test_false, threshold, alpha_value),
            "all": _selective_probe_report(all_scores, all_false, threshold, alpha_value),
        },
        "record_scores": {
            "train": _record_score_rows(records, train_indices, train_scores, train_false),
            "test": _record_score_rows(records, test_indices, test_scores, test_false),
            "all_indices": tuple(int(index) for index in all_indices),
        },
    }


def _build_pre_generation_calibration_artifact(
    *,
    records_path: Path,
    threshold: float,
    alpha: float,
    layer_idx: int | None,
    model_id: str,
    record_count: int,
    train_count: int,
    test_count: int,
    label_source: str,
    soft_target_cutoff: float | None,
) -> CalibrationArtifact:
    return CalibrationArtifact(
        model_id=str(model_id),
        target_layer=0 if layer_idx is None else int(layer_idx),
        scores=(
            CalibrationScore(
                name=PRE_GENERATION_RISK_SCORE_NAME,
                threshold=threshold,
                conformal_alpha=alpha,
                direction="higher",
            ),
        ),
        eigentruth_version=__version__,
        warmup_dataset_metadata={
            "workflow": "pre_generation_probe_eval",
            "records_path": str(records_path),
            "record_count": int(record_count),
        },
        calibration_dataset_metadata={
            "split": "train",
            "train_count": int(train_count),
            "test_count": int(test_count),
            "label_source": label_source,
            "soft_target_cutoff": soft_target_cutoff,
            "score_name": PRE_GENERATION_RISK_SCORE_NAME,
        },
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def _hidden_states_from_payload(
    payload: Mapping[str, Any],
    *,
    index: int,
    record_layer: int | None,
) -> tuple[Any | None, int | None]:
    layer_payload = _first_present(payload, ("layer_hidden_states", "prompt_hidden_states_by_layer"))
    if isinstance(layer_payload, Mapping):
        if record_layer is not None:
            hidden = _lookup_layer_payload(layer_payload, record_layer, index=index)
            return hidden, int(record_layer)
        direct = payload.get("hidden_states", payload.get("prompt_hidden_states"))
        if direct is not None:
            return direct, _metadata_layer(payload)
        if len(layer_payload) != 1:
            raise ValueError(f"record {index} contains multiple layers; pass --record-layer.")
        raw_layer, hidden = next(iter(layer_payload.items()))
        return hidden, _parse_layer_key(raw_layer, index=index)

    direct = payload.get("hidden_states", payload.get("prompt_hidden_states"))
    return direct, record_layer if record_layer is not None else _metadata_layer(payload)


def _lookup_layer_payload(layer_payload: Mapping[Any, Any], layer: int, *, index: int) -> Any:
    for key in (layer, str(layer)):
        if key in layer_payload:
            return layer_payload[key]
    available = ", ".join(str(key) for key in sorted(layer_payload, key=lambda item: str(item)))
    raise ValueError(f"record {index} is missing layer {layer}; available layers: {available}.")


def _metadata_layer(payload: Mapping[str, Any]) -> int | None:
    metadata = payload.get("metadata", {})
    if isinstance(metadata, Mapping) and metadata.get("layer") is not None:
        return int(metadata["layer"])
    return None


def _parse_layer_key(value: Any, *, index: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"record {index} has non-integer layer key {value!r}.") from exc


def _infer_record_layer(records: Sequence[PreGenerationProbeRecord]) -> int | None:
    layers = {record.selected_layer for record in records if record.selected_layer is not None}
    return next(iter(layers)) if len(layers) == 1 else None


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


def _selective_probe_report(
    probabilities: torch.Tensor,
    false_labels: torch.Tensor,
    threshold: float,
    alpha: float,
) -> dict[str, Any]:
    correctness = (false_labels == 0).to(torch.int64)
    return {
        "selective": selective_classification_report(
            probabilities,
            false_labels,
            threshold,
            direction="higher",
        ),
        "abstention": evaluate_conformal_abstention(
            probabilities,
            correctness.tolist(),
            threshold=threshold,
            alpha=alpha,
            direction="higher",
            score_name=PRE_GENERATION_RISK_SCORE_NAME,
        ).to_dict(),
    }


def _record_probabilities(
    artifact: AttentionSoftTargetProbeArtifact,
    records: Sequence[PreGenerationProbeRecord],
) -> torch.Tensor:
    hidden, mask, _targets, _labels = _stack_records(records, tuple(range(len(records))))
    with torch.no_grad():
        return artifact.predict_proba(hidden, attention_mask=mask).detach().cpu().to(torch.float64)


def _false_labels_for_records(
    records: Sequence[PreGenerationProbeRecord],
    *,
    soft_target_cutoff: float | None,
) -> tuple[torch.Tensor, str] | None:
    if all(record.label is not None for record in records):
        return (
            torch.tensor([int(record.label) for record in records], dtype=torch.int64),
            "label",
        )
    if soft_target_cutoff is None:
        return None
    cutoff = _probability_float(soft_target_cutoff, field_name="soft_target_cutoff")
    return (
        torch.tensor([1 if record.soft_target > cutoff else 0 for record in records], dtype=torch.int64),
        "soft_target_cutoff",
    )


def _record_score_rows(
    records: Sequence[PreGenerationProbeRecord],
    indices: Sequence[int],
    probabilities: torch.Tensor,
    false_labels: torch.Tensor,
) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "index": int(index),
            "record_id": records[int(index)].record_id,
            "probability": float(probabilities[row].item()),
            "label": int(false_labels[row].item()),
            "soft_target": records[int(index)].soft_target,
        }
        for row, index in enumerate(indices)
    )


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


def _alpha_float(value: Any) -> float:
    parsed = _probability_float(value, field_name="conformal_alpha")
    if parsed <= 0.0 or parsed >= 1.0:
        raise ValueError("conformal_alpha must be > 0 and < 1.")
    return parsed


def _parse_sweep_layers(value: str | None) -> tuple[int, ...] | None:
    if value is None or value.strip().lower() == "auto":
        return None
    layers: list[int] = []
    for part in value.split(","):
        text = part.strip()
        if not text:
            continue
        try:
            layers.append(int(text))
        except ValueError as exc:
            raise ValueError(
                "--sweep-layers must be a comma-separated list of integer layer indexes or 'auto'."
            ) from exc
    if not layers:
        raise ValueError("--sweep-layers must contain at least one layer index or 'auto'.")
    seen: set[int] = set()
    unique: list[int] = []
    for layer in layers:
        if layer in seen:
            continue
        seen.add(layer)
        unique.append(layer)
    return tuple(unique)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description="Train/evaluate a pre-generation attention risk probe")
    parser.add_argument("--records", required=True, help="JSON/JSONL records with hidden_states and soft targets")
    parser.add_argument("--json", help="optional report output path")
    parser.add_argument("--save-artifact", help="optional torch artifact output path")
    parser.add_argument("--save-calibration", help="optional CalibrationArtifact JSON output path")
    parser.add_argument("--calibration-model-id", default="pre_generation_probe")
    parser.add_argument("--conformal-alpha", type=float, default=0.1)
    parser.add_argument(
        "--sweep-layers",
        default=None,
        help=(
            "comma-separated record layers to train and rank, or 'auto' to infer layer_hidden_states keys; "
            "use --sweep-layers=-12,-8,-4 when the list starts with '-'"
        ),
    )
    parser.add_argument("--best-by", default="auto", choices=PRE_GENERATION_SWEEP_BEST_BY)
    parser.add_argument(
        "--record-layer",
        type=int,
        help="select one layer from records containing layer_hidden_states/prompt_hidden_states_by_layer",
    )
    parser.add_argument(
        "--soft-target-cutoff",
        type=float,
        help=(
            "derive calibration labels from soft_target when hard labels are absent; "
            "records with soft_target > cutoff are treated as anomalous"
        ),
    )
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
    if args.sweep_layers is not None:
        if args.record_layer is not None:
            parser.error("--record-layer cannot be combined with --sweep-layers")
        if args.layer_idx is not None:
            parser.error("--layer-idx cannot be combined with --sweep-layers; each candidate uses its record layer")
    payload = run(args)
    if args.json is None:
        print(json.dumps(payload, indent=None if args.compact_json else 2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

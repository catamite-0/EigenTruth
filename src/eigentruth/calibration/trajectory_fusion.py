"""Bridge trajectory sweep reports into calibrated score-fusion artifacts."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from eigentruth.calibration.fusion import RankScoreFusionArtifact, RankScoreFusionCalibrator

DEFAULT_TRAJECTORY_SIGNAL_NAME = "trajectory_convergence"
DEFAULT_NLL_SIGNAL_NAME = "nll_answer"


@dataclass(frozen=True)
class TrajectoryFusionDataset:
    """Row-aligned score-fusion inputs extracted from a trajectory report."""

    labels: tuple[int, ...]
    scores: Mapping[str, tuple[float, ...]]
    directions: Mapping[str, str]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.labels:
            raise ValueError("labels must be non-empty.")
        parsed_labels = tuple(_coerce_binary_label(label) for label in self.labels)
        score_map = {
            str(name): _finite_float_tuple(values, name=f"scores[{name!r}]")
            for name, values in self.scores.items()
        }
        if not score_map:
            raise ValueError("scores must contain at least one signal.")
        for name, values in score_map.items():
            if len(values) != len(parsed_labels):
                raise ValueError(f"score {name!r} length does not match labels.")
        directions = {str(name): str(direction) for name, direction in self.directions.items()}
        for name in score_map:
            direction = directions.get(name)
            if direction not in {"higher", "lower"}:
                raise ValueError(f"direction for score {name!r} must be 'higher' or 'lower'.")
        object.__setattr__(self, "labels", parsed_labels)
        object.__setattr__(self, "scores", score_map)
        object.__setattr__(self, "directions", directions)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "schema_version": int(self.schema_version),
            "labels": list(self.labels),
            "scores": {name: list(values) for name, values in self.scores.items()},
            "directions": dict(self.directions),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrajectoryFusionDataset":
        """Build a dataset from JSON-like data."""
        return cls(
            labels=tuple(data["labels"]),
            scores={
                str(name): tuple(values)
                for name, values in dict(data["scores"]).items()
            },
            directions=dict(data["directions"]),
            metadata=dict(data.get("metadata", {})),
            schema_version=int(data.get("schema_version", 1)),
        )

    def calibrate(
        self,
        *,
        alpha: float = 0.1,
        method: str = "max_rank",
        model_id: str | None = None,
        target_layer: int | None = None,
        model_revision: str | None = None,
        eigentruth_version: str | None = None,
        created_at: str | None = None,
        commit_sha: str | None = None,
    ) -> RankScoreFusionArtifact:
        """Fit a rank-calibrated fusion artifact from the extracted dataset."""
        return RankScoreFusionCalibrator(alpha=alpha, method=method).calibrate(
            labels=self.labels,
            scores=self.scores,
            directions=self.directions,
            model_id=model_id or _metadata_str(self.metadata, "model"),
            target_layer=target_layer if target_layer is not None else _metadata_int(self.metadata, "resolved_layer"),
            model_revision=model_revision,
            eigentruth_version=eigentruth_version,
            score_dump_metadata=dict(self.metadata),
            created_at=created_at,
            commit_sha=commit_sha,
        )


def trajectory_fusion_dataset_from_report(
    report: Mapping[str, Any],
    *,
    layer: str | int = "best",
    signal_name: str = DEFAULT_TRAJECTORY_SIGNAL_NAME,
    include_nll_answer: bool = False,
    nll_signal_name: str = DEFAULT_NLL_SIGNAL_NAME,
) -> TrajectoryFusionDataset:
    """Extract row-aligned trajectory scores from a trajectory benchmark report.

    For layer-sweep reports, ``layer="best"`` follows the report's selected
    best layer and direction. Explicit layer values use the matching
    ``layer_summaries`` row. Single-layer reports use each record's
    ``trajectory`` payload.
    """
    workflow = str(report.get("workflow") or "")
    if workflow == "truthfulqa_forced_answer_trajectory_layer_sweep":
        return _dataset_from_layer_sweep_report(
            report,
            layer=layer,
            signal_name=signal_name,
            include_nll_answer=include_nll_answer,
            nll_signal_name=nll_signal_name,
        )
    if workflow == "truthfulqa_forced_answer_trajectory":
        if layer != "best":
            config_layer = _mapping(report.get("config")).get("layer")
            requested = _layer_key(layer)
            if requested != _layer_key(config_layer):
                raise ValueError("explicit layer does not match single-layer trajectory report.")
        return _dataset_from_single_layer_report(
            report,
            signal_name=signal_name,
            include_nll_answer=include_nll_answer,
            nll_signal_name=nll_signal_name,
        )
    raise ValueError("report must be a TruthfulQA trajectory report.")


def calibrate_trajectory_fusion_from_report(
    report: Mapping[str, Any],
    *,
    layer: str | int = "best",
    signal_name: str = DEFAULT_TRAJECTORY_SIGNAL_NAME,
    include_nll_answer: bool = False,
    nll_signal_name: str = DEFAULT_NLL_SIGNAL_NAME,
    alpha: float = 0.1,
    method: str = "max_rank",
    model_id: str | None = None,
    target_layer: int | None = None,
    model_revision: str | None = None,
    eigentruth_version: str | None = None,
    created_at: str | None = None,
    commit_sha: str | None = None,
) -> RankScoreFusionArtifact:
    """Extract a trajectory fusion dataset and fit a rank-fusion artifact."""
    dataset = trajectory_fusion_dataset_from_report(
        report,
        layer=layer,
        signal_name=signal_name,
        include_nll_answer=include_nll_answer,
        nll_signal_name=nll_signal_name,
    )
    return dataset.calibrate(
        alpha=alpha,
        method=method,
        model_id=model_id,
        target_layer=target_layer,
        model_revision=model_revision,
        eigentruth_version=eigentruth_version,
        created_at=created_at,
        commit_sha=commit_sha,
    )


def _dataset_from_layer_sweep_report(
    report: Mapping[str, Any],
    *,
    layer: str | int,
    signal_name: str,
    include_nll_answer: bool,
    nll_signal_name: str,
) -> TrajectoryFusionDataset:
    summary = _mapping(report.get("summary"))
    layer_key = _selected_layer_key(summary, layer=layer)
    layer_summary = _layer_summary(report, layer_key=layer_key)
    direction = str(layer_summary.get("trajectory_score_direction_for_false") or summary.get(
        "trajectory_score_direction_for_false"
    ))
    records = _records(report)
    labels: list[int] = []
    trajectory_scores: list[float] = []
    nll_scores: list[float] = []
    for record in records:
        trajectories = _mapping(record.get("trajectories"))
        if layer_key not in trajectories:
            raise ValueError(f"record is missing trajectory layer {layer_key!r}.")
        trajectory = _mapping(trajectories[layer_key])
        labels.append(_coerce_binary_label(record.get("label")))
        trajectory_scores.append(_finite_float(trajectory.get("convergence_score"), name="convergence_score"))
        if include_nll_answer:
            nll_scores.append(_finite_float(record.get("nll_answer"), name="nll_answer"))
    scores: dict[str, tuple[float, ...]] = {str(signal_name): tuple(trajectory_scores)}
    directions = {str(signal_name): direction}
    if include_nll_answer:
        scores[str(nll_signal_name)] = tuple(nll_scores)
        directions[str(nll_signal_name)] = "higher"
    metadata = _base_metadata(
        report,
        layer_key=layer_key,
        layer=layer_summary.get("layer", summary.get("best_layer")),
        resolved_layer=layer_summary.get("resolved_layer", summary.get("best_resolved_layer")),
        direction=direction,
        auroc=layer_summary.get("trajectory_score_best_auroc", summary.get("trajectory_score_best_auroc")),
    )
    return TrajectoryFusionDataset(
        labels=tuple(labels),
        scores=scores,
        directions=directions,
        metadata=metadata,
    )


def _dataset_from_single_layer_report(
    report: Mapping[str, Any],
    *,
    signal_name: str,
    include_nll_answer: bool,
    nll_signal_name: str,
) -> TrajectoryFusionDataset:
    summary = _mapping(report.get("summary"))
    config = _mapping(report.get("config"))
    direction = str(summary.get("trajectory_score_direction_for_false"))
    records = _records(report)
    labels: list[int] = []
    trajectory_scores: list[float] = []
    nll_scores: list[float] = []
    resolved_layer: int | None = None
    for record in records:
        trajectory = _mapping(record.get("trajectory"))
        labels.append(_coerce_binary_label(record.get("label")))
        trajectory_scores.append(_finite_float(trajectory.get("convergence_score"), name="convergence_score"))
        metadata = _mapping(trajectory.get("metadata"))
        if resolved_layer is None and metadata.get("resolved_layer") is not None:
            resolved_layer = int(metadata["resolved_layer"])
        if include_nll_answer:
            nll_scores.append(_finite_float(record.get("nll_answer"), name="nll_answer"))
    scores: dict[str, tuple[float, ...]] = {str(signal_name): tuple(trajectory_scores)}
    directions = {str(signal_name): direction}
    if include_nll_answer:
        scores[str(nll_signal_name)] = tuple(nll_scores)
        directions[str(nll_signal_name)] = "higher"
    layer = config.get("layer")
    metadata = _base_metadata(
        report,
        layer_key=_layer_key(layer),
        layer=layer,
        resolved_layer=resolved_layer,
        direction=direction,
        auroc=summary.get("trajectory_score_best_auroc"),
    )
    return TrajectoryFusionDataset(
        labels=tuple(labels),
        scores=scores,
        directions=directions,
        metadata=metadata,
    )


def _selected_layer_key(summary: Mapping[str, Any], *, layer: str | int) -> str:
    if str(layer) == "best":
        if summary.get("best_layer_key") is not None:
            return str(summary["best_layer_key"])
        if summary.get("best_layer") is not None:
            return _layer_key(summary["best_layer"])
        raise ValueError("layer='best' requires best_layer or best_layer_key in report summary.")
    return _layer_key(layer)


def _layer_summary(report: Mapping[str, Any], *, layer_key: str) -> dict[str, Any]:
    for row in report.get("layer_summaries") or ():
        if not isinstance(row, Mapping):
            continue
        row_key = str(row.get("layer_key")) if row.get("layer_key") is not None else _layer_key(row.get("layer"))
        if row_key == layer_key:
            return dict(row)
    raise ValueError(f"layer summary {layer_key!r} is missing.")


def _base_metadata(
    report: Mapping[str, Any],
    *,
    layer_key: str,
    layer: Any,
    resolved_layer: Any,
    direction: str,
    auroc: Any,
) -> dict[str, Any]:
    metadata = dict(_mapping(report.get("metadata")))
    return {
        "source_workflow": report.get("workflow"),
        "source_status": _mapping(report.get("summary")).get("status"),
        "model": metadata.get("model"),
        "source_scores": metadata.get("source_scores"),
        "layer": None if layer is None else int(layer),
        "layer_key": str(layer_key),
        "resolved_layer": None if resolved_layer is None else int(resolved_layer),
        "trajectory_score_direction_for_false": direction,
        "trajectory_score_best_auroc": None if auroc is None else float(auroc),
        "n_evaluated": _mapping(report.get("summary")).get("n_evaluated"),
    }


def _records(report: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    rows = report.get("records")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise ValueError("trajectory report must contain non-empty records.")
    records = tuple(row for row in rows if isinstance(row, Mapping))
    if len(records) != len(rows):
        raise ValueError("trajectory report records must be objects.")
    return records


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _layer_key(layer: Any) -> str:
    if isinstance(layer, str) and layer.strip():
        return layer.strip()
    if isinstance(layer, bool) or layer is None:
        raise ValueError("layer must be an integer or 'best'.")
    return str(int(layer))


def _coerce_binary_label(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("labels must be integer 0/1 values, not bool.")
    if isinstance(value, int):
        label = value
    elif isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError("labels must be binary integer values in {0, 1}.")
        label = int(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if stripped not in {"0", "1"}:
            raise ValueError("labels must be strings '0' or '1'.")
        label = int(stripped)
    else:
        raise ValueError("labels must be binary values in {0, 1}.")
    if label not in {0, 1}:
        raise ValueError("labels must be binary values in {0, 1}.")
    return label


def _finite_float_tuple(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    if any(isinstance(value, bool) for value in values):
        raise ValueError(f"{name} must contain only finite numeric values, not bool.")
    result = tuple(float(value) for value in values)
    if not result:
        raise ValueError(f"{name} must be non-empty.")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite values.")
    return result


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite.")
    return numeric


def _metadata_str(metadata: Mapping[str, Any], name: str) -> str | None:
    value = metadata.get(name)
    return None if value is None else str(value)


def _metadata_int(metadata: Mapping[str, Any], name: str) -> int | None:
    value = metadata.get(name)
    return None if value is None else int(value)

"""Generation trajectory convergence diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class TrajectoryConvergenceMetrics:
    """JSON-ready convergence diagnostics for one hidden-state trajectory."""

    step_count: int
    hidden_dim: int
    path_length: float
    direct_distance: float
    path_efficiency: float
    mean_step_distance: float
    initial_step_distance: float
    final_step_distance: float
    convergence_ratio: float
    step_distance_drop: float
    log_decay_slope: float
    koopman_rate: float
    convergence_strength: float
    decay_fraction: float
    displacement_cv: float
    convergence_score: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready metrics payload."""
        return {
            "step_count": int(self.step_count),
            "hidden_dim": int(self.hidden_dim),
            "path_length": float(self.path_length),
            "direct_distance": float(self.direct_distance),
            "path_efficiency": float(self.path_efficiency),
            "mean_step_distance": float(self.mean_step_distance),
            "initial_step_distance": float(self.initial_step_distance),
            "final_step_distance": float(self.final_step_distance),
            "convergence_ratio": float(self.convergence_ratio),
            "step_distance_drop": float(self.step_distance_drop),
            "log_decay_slope": float(self.log_decay_slope),
            "koopman_rate": float(self.koopman_rate),
            "convergence_strength": float(self.convergence_strength),
            "decay_fraction": float(self.decay_fraction),
            "displacement_cv": float(self.displacement_cv),
            "convergence_score": float(self.convergence_score),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrajectoryConvergenceMetrics":
        """Build metrics from a JSON-like payload."""
        return cls(
            step_count=int(data["step_count"]),
            hidden_dim=int(data["hidden_dim"]),
            path_length=float(data["path_length"]),
            direct_distance=float(data["direct_distance"]),
            path_efficiency=float(data["path_efficiency"]),
            mean_step_distance=float(data["mean_step_distance"]),
            initial_step_distance=float(data["initial_step_distance"]),
            final_step_distance=float(data["final_step_distance"]),
            convergence_ratio=float(data["convergence_ratio"]),
            step_distance_drop=float(data["step_distance_drop"]),
            log_decay_slope=float(data["log_decay_slope"]),
            koopman_rate=float(data["koopman_rate"]),
            convergence_strength=float(data["convergence_strength"]),
            decay_fraction=float(data["decay_fraction"]),
            displacement_cv=float(data["displacement_cv"]),
            convergence_score=float(data["convergence_score"]),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class TrajectoryConvergenceReport:
    """JSON-ready report over one or more generation trajectories."""

    trajectories: tuple[TrajectoryConvergenceMetrics, ...]
    summary: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready report."""
        return {
            "schema_version": int(self.schema_version),
            "workflow": "generation_trajectory_convergence",
            "summary": dict(self.summary),
            "trajectories": [trajectory.to_dict() for trajectory in self.trajectories],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ResidualContributionProfile:
    """Layerwise residual-update contribution summary for one hidden state.

    The profile treats per-layer residual update norms as a contribution curve
    over the inspected layer order. It is intended as a lightweight ICR-style
    dynamic signal: large area/peak values indicate stronger hidden-state
    movement, while late mass and concentration describe where that movement is
    concentrated.
    """

    layer_count: int
    total_contribution: float
    mean_contribution: float
    peak_contribution: float
    peak_layer: int | None
    peak_position: float
    layer_centroid: float
    late_mass_fraction: float
    normalized_entropy: float
    concentration: float
    values_by_layer: Mapping[int, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready residual contribution profile."""
        return {
            "layer_count": int(self.layer_count),
            "total_contribution": float(self.total_contribution),
            "mean_contribution": float(self.mean_contribution),
            "peak_contribution": float(self.peak_contribution),
            "peak_layer": None if self.peak_layer is None else int(self.peak_layer),
            "peak_position": float(self.peak_position),
            "layer_centroid": float(self.layer_centroid),
            "late_mass_fraction": float(self.late_mass_fraction),
            "normalized_entropy": float(self.normalized_entropy),
            "concentration": float(self.concentration),
            "values_by_layer": {
                str(layer): float(value)
                for layer, value in self.values_by_layer.items()
            },
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ResidualContributionProfile":
        """Build a residual contribution profile from a JSON-like payload."""
        raw_values = data.get("values_by_layer") or {}
        return cls(
            layer_count=int(data["layer_count"]),
            total_contribution=float(data["total_contribution"]),
            mean_contribution=float(data["mean_contribution"]),
            peak_contribution=float(data["peak_contribution"]),
            peak_layer=None if data.get("peak_layer") is None else int(data["peak_layer"]),
            peak_position=float(data["peak_position"]),
            layer_centroid=float(data["layer_centroid"]),
            late_mass_fraction=float(data["late_mass_fraction"]),
            normalized_entropy=float(data["normalized_entropy"]),
            concentration=float(data["concentration"]),
            values_by_layer={int(layer): float(value) for layer, value in dict(raw_values).items()},
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class PromptAnswerPathwayMetrics:
    """Two-pathway prompt/answer hidden-state summary for one layer.

    The metrics separate prompt-anchored movement (last prompt token to final
    answer token) from answer-anchored movement (first answer token to final
    answer token). This is a lightweight, dependency-free approximation of the
    current two-pathway truthfulness framing; it is a diagnostic score family,
    not a learned truth probe.
    """

    prompt_token_count: int
    answer_token_count: int
    hidden_dim: int
    prompt_answer_distance: float
    prompt_answer_cosine_gap: float
    answer_anchor_distance: float
    answer_path_length: float
    pathway_disagreement: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready metrics payload."""
        return {
            "prompt_token_count": int(self.prompt_token_count),
            "answer_token_count": int(self.answer_token_count),
            "hidden_dim": int(self.hidden_dim),
            "prompt_answer_distance": float(self.prompt_answer_distance),
            "prompt_answer_cosine_gap": float(self.prompt_answer_cosine_gap),
            "answer_anchor_distance": float(self.answer_anchor_distance),
            "answer_path_length": float(self.answer_path_length),
            "pathway_disagreement": float(self.pathway_disagreement),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PromptAnswerPathwayMetrics":
        """Build metrics from a JSON-like payload."""
        return cls(
            prompt_token_count=int(data["prompt_token_count"]),
            answer_token_count=int(data["answer_token_count"]),
            hidden_dim=int(data["hidden_dim"]),
            prompt_answer_distance=float(data["prompt_answer_distance"]),
            prompt_answer_cosine_gap=float(data["prompt_answer_cosine_gap"]),
            answer_anchor_distance=float(data["answer_anchor_distance"]),
            answer_path_length=float(data["answer_path_length"]),
            pathway_disagreement=float(data["pathway_disagreement"]),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class AttentionPathwayMetrics:
    """Question-vs-answer attention-flow summary for one attention layer.

    The metrics summarize how answer-token queries allocate attention mass to
    prompt/question tokens versus answer tokens. This is a cheap observational
    readout of pathway balance, not a causal attention knockout.
    """

    prompt_token_count: int
    answer_token_count: int
    head_count: int
    query_count: int
    key_count: int
    prompt_flow_fraction: float
    answer_self_flow_fraction: float
    other_flow_fraction: float
    prompt_flow_loss: float
    pathway_gap: float
    pathway_concentration: float
    pathway_entropy: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready metrics payload."""
        return {
            "prompt_token_count": int(self.prompt_token_count),
            "answer_token_count": int(self.answer_token_count),
            "head_count": int(self.head_count),
            "query_count": int(self.query_count),
            "key_count": int(self.key_count),
            "prompt_flow_fraction": float(self.prompt_flow_fraction),
            "answer_self_flow_fraction": float(self.answer_self_flow_fraction),
            "other_flow_fraction": float(self.other_flow_fraction),
            "prompt_flow_loss": float(self.prompt_flow_loss),
            "pathway_gap": float(self.pathway_gap),
            "pathway_concentration": float(self.pathway_concentration),
            "pathway_entropy": float(self.pathway_entropy),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AttentionPathwayMetrics":
        """Build metrics from a JSON-like payload."""
        return cls(
            prompt_token_count=int(data["prompt_token_count"]),
            answer_token_count=int(data["answer_token_count"]),
            head_count=int(data["head_count"]),
            query_count=int(data["query_count"]),
            key_count=int(data["key_count"]),
            prompt_flow_fraction=float(data["prompt_flow_fraction"]),
            answer_self_flow_fraction=float(data["answer_self_flow_fraction"]),
            other_flow_fraction=float(data["other_flow_fraction"]),
            prompt_flow_loss=float(data["prompt_flow_loss"]),
            pathway_gap=float(data["pathway_gap"]),
            pathway_concentration=float(data["pathway_concentration"]),
            pathway_entropy=float(data["pathway_entropy"]),
            metadata=dict(data.get("metadata") or {}),
        )


def trajectory_convergence_metrics(
    states: Tensor,
    *,
    eps: float = 1e-8,
    metadata: Mapping[str, Any] | None = None,
) -> TrajectoryConvergenceMetrics:
    """Summarize convergence for one trajectory of per-token hidden states.

    ``states`` must be shaped ``[generation_step, hidden_dim]``. Consecutive
    step distances are treated as a generation trajectory; exponential decay in
    those distances produces a positive ``convergence_strength`` and a
    ``koopman_rate`` below one.
    """
    matrix = _as_trajectory_matrix(states)
    if float(eps) <= 0.0 or not math.isfinite(float(eps)):
        raise ValueError("eps must be positive and finite.")
    displacements = matrix[1:] - matrix[:-1]
    step_distances = torch.norm(displacements, dim=-1).to(torch.float64)
    path_length = float(step_distances.sum().item())
    direct_distance = float(torch.norm(matrix[-1] - matrix[0]).item())
    initial_distance = float(step_distances[0].item())
    final_distance = float(step_distances[-1].item())
    mean_distance = float(step_distances.mean().item())
    path_efficiency = direct_distance / max(path_length, float(eps))
    convergence_ratio = final_distance / max(initial_distance, float(eps))
    step_drop = initial_distance - final_distance
    log_decay_slope = _log_decay_slope(step_distances, eps=float(eps))
    koopman_rate = float(math.exp(log_decay_slope))
    convergence_strength = max(0.0, -log_decay_slope)
    decay_fraction = _decay_fraction(step_distances, tolerance=float(eps))
    std = float(step_distances.std(unbiased=False).item()) if int(step_distances.numel()) > 1 else 0.0
    displacement_cv = std / max(mean_distance, float(eps))
    convergence_score = (
        convergence_strength
        + max(0.0, 1.0 - convergence_ratio)
        + decay_fraction
        + max(0.0, path_efficiency)
    )
    return TrajectoryConvergenceMetrics(
        step_count=int(matrix.shape[0]),
        hidden_dim=int(matrix.shape[1]),
        path_length=path_length,
        direct_distance=direct_distance,
        path_efficiency=path_efficiency,
        mean_step_distance=mean_distance,
        initial_step_distance=initial_distance,
        final_step_distance=final_distance,
        convergence_ratio=convergence_ratio,
        step_distance_drop=step_drop,
        log_decay_slope=log_decay_slope,
        koopman_rate=koopman_rate,
        convergence_strength=convergence_strength,
        decay_fraction=decay_fraction,
        displacement_cv=displacement_cv,
        convergence_score=float(convergence_score),
        metadata=metadata or {},
    )


def prompt_answer_pathway_metrics(
    prompt_states: Tensor,
    answer_states: Tensor,
    *,
    eps: float = 1e-8,
    metadata: Mapping[str, Any] | None = None,
) -> PromptAnswerPathwayMetrics:
    """Summarize prompt-anchored and answer-anchored hidden-state movement.

    ``prompt_states`` must be shaped ``[prompt_token, hidden_dim]`` and
    ``answer_states`` must be shaped ``[answer_token, hidden_dim]``. Distances
    are normalized by ``sqrt(hidden_dim)`` so they remain comparable across
    model sizes.
    """
    prompt = _as_pathway_matrix(prompt_states, name="prompt_states")
    answer = _as_pathway_matrix(answer_states, name="answer_states")
    if prompt.shape[1] != answer.shape[1]:
        raise ValueError("prompt_states and answer_states must share hidden_dim.")
    if float(eps) <= 0.0 or not math.isfinite(float(eps)):
        raise ValueError("eps must be positive and finite.")

    scale = math.sqrt(float(prompt.shape[1]))
    prompt_anchor = prompt[-1]
    answer_first = answer[0]
    answer_final = answer[-1]
    prompt_answer_distance = _normalized_l2(answer_final - prompt_anchor, scale=scale)
    cosine_gap = _cosine_gap(prompt_anchor, answer_final, eps=float(eps))
    answer_anchor_distance = _normalized_l2(answer_final - answer_first, scale=scale)
    if int(answer.shape[0]) <= 1:
        answer_path_length = 0.0
    else:
        deltas = answer[1:] - answer[:-1]
        answer_path_length = float(torch.linalg.vector_norm(deltas, dim=-1).sum().item() / scale)
    pathway_disagreement = abs(prompt_answer_distance - answer_anchor_distance)
    return PromptAnswerPathwayMetrics(
        prompt_token_count=int(prompt.shape[0]),
        answer_token_count=int(answer.shape[0]),
        hidden_dim=int(prompt.shape[1]),
        prompt_answer_distance=float(prompt_answer_distance),
        prompt_answer_cosine_gap=float(cosine_gap),
        answer_anchor_distance=float(answer_anchor_distance),
        answer_path_length=float(answer_path_length),
        pathway_disagreement=float(pathway_disagreement),
        metadata=metadata or {},
    )


def attention_pathway_metrics(
    attention: Tensor,
    *,
    prompt_start: int = 0,
    answer_start: int,
    sequence_end: int | None = None,
    eps: float = 1e-8,
    metadata: Mapping[str, Any] | None = None,
) -> AttentionPathwayMetrics:
    """Summarize answer-token attention flow into prompt and answer spans.

    ``attention`` may be shaped ``[head, query_token, key_token]`` or
    ``[batch, head, query_token, key_token]`` with batch size 1. Query rows from
    ``answer_start:sequence_end`` are treated as answer-token queries. Key
    columns from ``prompt_start:answer_start`` are treated as the prompt/question
    pathway, while ``answer_start:sequence_end`` are treated as the answer
    pathway. Fractions are normalized per query/head row before aggregation so
    non-unit attention rows remain comparable.
    """
    matrix = _as_attention_pathway_tensor(attention)
    query_count = int(matrix.shape[-2])
    key_count = int(matrix.shape[-1])
    if float(eps) <= 0.0 or not math.isfinite(float(eps)):
        raise ValueError("eps must be positive and finite.")
    prompt_start = int(prompt_start)
    answer_start = int(answer_start)
    if sequence_end is None:
        sequence_end = min(query_count, key_count)
    sequence_end = int(sequence_end)
    if not (0 <= prompt_start < answer_start < sequence_end):
        raise ValueError("expected 0 <= prompt_start < answer_start < sequence_end.")
    if sequence_end > query_count or sequence_end > key_count:
        raise ValueError("sequence_end must fit both query and key dimensions.")

    answer_queries = matrix[:, answer_start:sequence_end, :]
    row_mass = answer_queries.sum(dim=-1, keepdim=True).clamp_min(float(eps))
    normalized = answer_queries / row_mass
    prompt_mass = normalized[:, :, prompt_start:answer_start].sum(dim=-1)
    answer_mass = normalized[:, :, answer_start:sequence_end].sum(dim=-1)
    observed_mass = prompt_mass + answer_mass
    other_mass = (1.0 - observed_mass).clamp_min(0.0)

    prompt_flow = float(prompt_mass.mean().item())
    answer_flow = float(answer_mass.mean().item())
    other_flow = float(other_mass.mean().item())
    prompt_loss = max(0.0, 1.0 - prompt_flow)
    gap = abs(prompt_flow - answer_flow)
    concentration = max(prompt_flow, answer_flow)
    pathway_total = prompt_flow + answer_flow
    if pathway_total <= float(eps):
        entropy = 0.0
    else:
        probabilities = [
            max(0.0, prompt_flow / pathway_total),
            max(0.0, answer_flow / pathway_total),
        ]
        entropy = -sum(p * math.log(max(p, float(eps))) for p in probabilities if p > 0.0) / math.log(2.0)

    return AttentionPathwayMetrics(
        prompt_token_count=int(answer_start - prompt_start),
        answer_token_count=int(sequence_end - answer_start),
        head_count=int(matrix.shape[0]),
        query_count=query_count,
        key_count=key_count,
        prompt_flow_fraction=float(prompt_flow),
        answer_self_flow_fraction=float(answer_flow),
        other_flow_fraction=float(other_flow),
        prompt_flow_loss=float(prompt_loss),
        pathway_gap=float(gap),
        pathway_concentration=float(concentration),
        pathway_entropy=float(entropy),
        metadata=metadata or {},
    )


def residual_contribution_profile(
    update_norms: Mapping[int, float] | Sequence[float],
    *,
    layers: Sequence[int] | None = None,
    late_fraction: float = 0.5,
    eps: float = 1e-8,
    metadata: Mapping[str, Any] | None = None,
) -> ResidualContributionProfile:
    """Summarize a per-layer residual-update contribution curve.

    Args:
        update_norms: Mapping ``layer -> non-negative update norm`` or a
            sequence of update norms. When a sequence is supplied, ``layers``
            may provide the corresponding layer identifiers.
        layers: Optional explicit layer order. This is preferred for negative
            layer indexes because it preserves the benchmark sweep order.
        late_fraction: Fraction of the tail of the layer order counted as
            "late" mass.
        eps: Positive finite tolerance used for zero-mass handling.
        metadata: Optional JSON-ready metadata copied into the profile.
    """
    values_by_layer = _coerce_residual_update_values(update_norms, layers=layers)
    if float(eps) <= 0.0 or not math.isfinite(float(eps)):
        raise ValueError("eps must be positive and finite.")
    if not (0.0 < float(late_fraction) <= 1.0) or not math.isfinite(float(late_fraction)):
        raise ValueError("late_fraction must be in (0, 1].")

    ordered_layers = tuple(values_by_layer)
    values = [values_by_layer[layer] for layer in ordered_layers]
    layer_count = len(values)
    if layer_count == 0:
        return ResidualContributionProfile(
            layer_count=0,
            total_contribution=0.0,
            mean_contribution=0.0,
            peak_contribution=0.0,
            peak_layer=None,
            peak_position=0.0,
            layer_centroid=0.0,
            late_mass_fraction=0.0,
            normalized_entropy=0.0,
            concentration=0.0,
            values_by_layer={},
            metadata=metadata or {},
        )

    total = float(sum(values))
    mean = total / layer_count
    peak_index, peak = max(enumerate(values), key=lambda item: item[1])
    denominator = max(layer_count - 1, 1)
    peak_position = peak_index / denominator if layer_count > 1 else 0.0

    if total <= float(eps):
        centroid = 0.0
        late_mass_fraction = 0.0
        normalized_entropy = 0.0
    else:
        positions = [idx / denominator if layer_count > 1 else 0.0 for idx in range(layer_count)]
        weights = [value / total for value in values]
        centroid = float(sum(position * weight for position, weight in zip(positions, weights, strict=True)))
        late_count = max(1, math.ceil(layer_count * float(late_fraction)))
        late_start = max(0, layer_count - late_count)
        late_mass_fraction = float(sum(values[late_start:]) / total)
        if layer_count <= 1:
            normalized_entropy = 0.0
        else:
            entropy = -sum(weight * math.log(max(weight, float(eps))) for weight in weights if weight > 0.0)
            normalized_entropy = float(entropy / math.log(layer_count))

    concentration = max(0.0, min(1.0, 1.0 - normalized_entropy))
    return ResidualContributionProfile(
        layer_count=layer_count,
        total_contribution=total,
        mean_contribution=mean,
        peak_contribution=float(peak),
        peak_layer=int(ordered_layers[peak_index]),
        peak_position=float(peak_position),
        layer_centroid=float(centroid),
        late_mass_fraction=float(late_mass_fraction),
        normalized_entropy=float(normalized_entropy),
        concentration=float(concentration),
        values_by_layer=values_by_layer,
        metadata=metadata or {},
    )


@dataclass
class TrajectoryMonitor:
    """Recorder for model-free generation trajectory convergence diagnostics."""

    eps: float = 1e-8
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if float(self.eps) <= 0.0 or not math.isfinite(float(self.eps)):
            raise ValueError("eps must be positive and finite.")
        self.eps = float(self.eps)
        self._trajectories: list[TrajectoryConvergenceMetrics] = []

    @property
    def trajectories(self) -> tuple[TrajectoryConvergenceMetrics, ...]:
        """Recorded trajectory metrics in capture order."""
        return tuple(self._trajectories)

    def record(
        self,
        states: Tensor,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> TrajectoryConvergenceMetrics:
        """Record one hidden-state trajectory."""
        merged_metadata = dict(self.metadata)
        if metadata:
            merged_metadata.update(metadata)
        metrics = trajectory_convergence_metrics(states, eps=self.eps, metadata=merged_metadata)
        self._trajectories.append(metrics)
        return metrics

    def to_report(self, *, metadata: Mapping[str, Any] | None = None) -> TrajectoryConvergenceReport:
        """Return a report over all recorded trajectories."""
        merged_metadata = dict(self.metadata)
        if metadata:
            merged_metadata.update(metadata)
        return TrajectoryConvergenceReport(
            trajectories=tuple(self._trajectories),
            summary=_summarize_trajectories(self._trajectories),
            metadata=merged_metadata,
        )


def _as_trajectory_matrix(states: Tensor) -> Tensor:
    matrix = torch.as_tensor(states, dtype=torch.float32).detach().cpu()
    if matrix.ndim != 2:
        raise ValueError("states must be a 2D tensor [generation_step, hidden_dim].")
    if int(matrix.shape[0]) < 3:
        raise ValueError("states must contain at least three generation steps.")
    if int(matrix.shape[1]) < 1:
        raise ValueError("states must have a non-empty hidden dimension.")
    if not torch.isfinite(matrix).all():
        raise ValueError("states must contain only finite values.")
    return matrix


def _as_pathway_matrix(states: Tensor, *, name: str) -> Tensor:
    matrix = torch.as_tensor(states, dtype=torch.float32).detach().cpu()
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2D tensor [token, hidden_dim].")
    if int(matrix.shape[0]) < 1:
        raise ValueError(f"{name} must contain at least one token.")
    if int(matrix.shape[1]) < 1:
        raise ValueError(f"{name} must have a non-empty hidden dimension.")
    if not torch.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def _as_attention_pathway_tensor(attention: Tensor) -> Tensor:
    matrix = torch.as_tensor(attention, dtype=torch.float32).detach().cpu()
    if matrix.ndim == 4:
        if int(matrix.shape[0]) != 1:
            raise ValueError("4D attention input must have batch size 1.")
        matrix = matrix[0]
    if matrix.ndim != 3:
        raise ValueError("attention must be shaped [head, query, key] or [1, head, query, key].")
    if int(matrix.shape[0]) < 1 or int(matrix.shape[1]) < 1 or int(matrix.shape[2]) < 1:
        raise ValueError("attention must have non-empty head, query, and key dimensions.")
    if not torch.isfinite(matrix).all():
        raise ValueError("attention must contain only finite values.")
    if bool((matrix < 0).any()):
        raise ValueError("attention must contain non-negative weights.")
    return matrix


def _normalized_l2(vector: Tensor, *, scale: float) -> float:
    return float(torch.linalg.vector_norm(vector).item() / max(float(scale), 1e-12))


def _cosine_gap(left: Tensor, right: Tensor, *, eps: float) -> float:
    denominator = float(torch.linalg.vector_norm(left).item() * torch.linalg.vector_norm(right).item())
    if denominator <= float(eps):
        return 0.0
    cosine = float(torch.dot(left, right).item() / denominator)
    cosine = max(-1.0, min(1.0, cosine))
    return 1.0 - cosine


def _coerce_residual_update_values(
    update_norms: Mapping[int, float] | Sequence[float],
    *,
    layers: Sequence[int] | None,
) -> dict[int, float]:
    if isinstance(update_norms, Mapping):
        raw_by_layer = {int(layer): value for layer, value in update_norms.items()}
        if layers is None:
            ordered_layers = tuple(sorted(raw_by_layer))
        else:
            ordered_layers = tuple(int(layer) for layer in layers if int(layer) in raw_by_layer)
        values = {
            layer: _non_negative_finite_float(raw_by_layer[layer], name=f"update_norms[{layer}]")
            for layer in ordered_layers
        }
        return values

    values_seq = tuple(update_norms)
    if layers is None:
        ordered_layers = tuple(range(len(values_seq)))
    else:
        if len(layers) != len(values_seq):
            raise ValueError("layers must have the same length as update_norms.")
        ordered_layers = tuple(int(layer) for layer in layers)
    return {
        layer: _non_negative_finite_float(value, name=f"update_norms[{index}]")
        for index, (layer, value) in enumerate(zip(ordered_layers, values_seq, strict=True))
    }


def _non_negative_finite_float(value: float, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative finite number.")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{name} must be a non-negative finite number.")
    return resolved


def _log_decay_slope(step_distances: Tensor, *, eps: float) -> float:
    x = torch.arange(int(step_distances.numel()), dtype=torch.float64)
    y = torch.log(step_distances.clamp_min(float(eps)).to(torch.float64))
    centered_x = x - x.mean()
    denominator = (centered_x * centered_x).sum()
    if float(denominator.item()) <= float(eps):
        return 0.0
    centered_y = y - y.mean()
    slope = (centered_x * centered_y).sum() / denominator
    return float(slope.item())


def _decay_fraction(step_distances: Tensor, *, tolerance: float) -> float:
    if int(step_distances.numel()) <= 1:
        return 1.0
    left = step_distances[:-1]
    right = step_distances[1:]
    return float((right <= left + float(tolerance)).to(torch.float32).mean().item())


def _summarize_trajectories(trajectories: Sequence[TrajectoryConvergenceMetrics]) -> dict[str, Any]:
    if not trajectories:
        return {
            "n_trajectories": 0,
            "mean_convergence_score": None,
            "mean_convergence_strength": None,
            "mean_koopman_rate": None,
            "mean_decay_fraction": None,
        }
    return {
        "n_trajectories": len(trajectories),
        "mean_convergence_score": sum(t.convergence_score for t in trajectories) / len(trajectories),
        "mean_convergence_strength": sum(t.convergence_strength for t in trajectories) / len(trajectories),
        "mean_koopman_rate": sum(t.koopman_rate for t in trajectories) / len(trajectories),
        "mean_decay_fraction": sum(t.decay_fraction for t in trajectories) / len(trajectories),
    }

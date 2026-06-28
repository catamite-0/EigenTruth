"""Pathway intervention analysis helpers.

These helpers provide dependency-light building blocks for mechanistic pathway
experiments such as question-pathway attention knockout or answer-pathway
patching. They operate on already-captured tensors and scalar scores; they do
not claim a causal model result unless the caller reruns the model under the
intervention and records the resulting evidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
from torch import Tensor

from eigentruth.core import AttentionPathwayMetrics, attention_pathway_metrics

PATHWAY_INTERVENTION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PathwayInterventionEffect:
    """Scalar score delta after a pathway intervention.

    ``anomalous_delta`` is positive when the intervention increases anomaly
    under the score direction. ``risk_reduction`` is the sign-flipped value, so
    positive means the intervention reduced anomaly.
    """

    score_name: str
    baseline_score: float
    intervened_score: float
    direction: str
    delta: float
    anomalous_delta: float
    risk_reduction: float
    improved: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = PATHWAY_INTERVENTION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready effect payload."""
        return {
            "schema_version": int(self.schema_version),
            "score_name": self.score_name,
            "baseline_score": float(self.baseline_score),
            "intervened_score": float(self.intervened_score),
            "direction": self.direction,
            "delta": float(self.delta),
            "anomalous_delta": float(self.anomalous_delta),
            "risk_reduction": float(self.risk_reduction),
            "improved": bool(self.improved),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PathwayInterventionEffect":
        """Build an effect payload from a JSON-like mapping."""
        return cls(
            score_name=str(data["score_name"]),
            baseline_score=float(data["baseline_score"]),
            intervened_score=float(data["intervened_score"]),
            direction=_coerce_direction(data["direction"]),
            delta=float(data["delta"]),
            anomalous_delta=float(data["anomalous_delta"]),
            risk_reduction=float(data["risk_reduction"]),
            improved=bool(data["improved"]),
            metadata=dict(data.get("metadata") or {}),
            schema_version=int(data.get("schema_version", PATHWAY_INTERVENTION_SCHEMA_VERSION)),
        )


@dataclass(frozen=True)
class AttentionPathwayKnockoutReport:
    """Before/after metrics for a prompt or answer attention-pathway knockout."""

    pathway: str
    prompt_start: int
    answer_start: int
    sequence_end: int
    baseline: AttentionPathwayMetrics
    intervened: AttentionPathwayMetrics
    deltas: Mapping[str, float]
    removed_mass_fraction: float
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = PATHWAY_INTERVENTION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready knockout report."""
        return {
            "schema_version": int(self.schema_version),
            "pathway": self.pathway,
            "prompt_start": int(self.prompt_start),
            "answer_start": int(self.answer_start),
            "sequence_end": int(self.sequence_end),
            "baseline": self.baseline.to_dict(),
            "intervened": self.intervened.to_dict(),
            "deltas": {key: float(value) for key, value in self.deltas.items()},
            "removed_mass_fraction": float(self.removed_mass_fraction),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AttentionPathwayKnockoutReport":
        """Build a knockout report from a JSON-like mapping."""
        return cls(
            pathway=_coerce_pathway(data["pathway"]),
            prompt_start=int(data["prompt_start"]),
            answer_start=int(data["answer_start"]),
            sequence_end=int(data["sequence_end"]),
            baseline=AttentionPathwayMetrics.from_dict(data["baseline"]),
            intervened=AttentionPathwayMetrics.from_dict(data["intervened"]),
            deltas={str(key): float(value) for key, value in dict(data["deltas"]).items()},
            removed_mass_fraction=float(data["removed_mass_fraction"]),
            metadata=dict(data.get("metadata") or {}),
            schema_version=int(data.get("schema_version", PATHWAY_INTERVENTION_SCHEMA_VERSION)),
        )


def pathway_intervention_effect(
    score_name: str,
    *,
    baseline_score: float,
    intervened_score: float,
    direction: str = "higher",
    metadata: Mapping[str, Any] | None = None,
) -> PathwayInterventionEffect:
    """Summarize how one scalar score changed after an intervention."""
    direction = _coerce_direction(direction)
    baseline = _finite_float(baseline_score, name="baseline_score")
    intervened = _finite_float(intervened_score, name="intervened_score")
    delta = intervened - baseline
    anomalous_delta = delta if direction == "higher" else -delta
    risk_reduction = -anomalous_delta
    return PathwayInterventionEffect(
        score_name=str(score_name),
        baseline_score=baseline,
        intervened_score=intervened,
        direction=direction,
        delta=float(delta),
        anomalous_delta=float(anomalous_delta),
        risk_reduction=float(risk_reduction),
        improved=bool(risk_reduction > 0.0),
        metadata=metadata or {},
    )


def knockout_attention_pathway(
    attention: Tensor,
    *,
    pathway: str,
    prompt_start: int = 0,
    answer_start: int,
    sequence_end: int | None = None,
    renormalize: bool = True,
    eps: float = 1e-8,
) -> Tensor:
    """Zero one answer-query attention pathway and optionally renormalize rows.

    ``pathway="prompt"`` removes answer-token query mass assigned to prompt
    tokens. ``pathway="answer"`` removes answer-token query mass assigned to
    answer tokens. Inputs may be shaped ``[head, query, key]`` or
    ``[batch, head, query, key]`` and the returned tensor preserves that shape.
    """
    pathway = _coerce_pathway(pathway)
    matrix, squeeze_batch = _attention_matrix(attention)
    query_count = int(matrix.shape[-2])
    key_count = int(matrix.shape[-1])
    prompt_start, answer_start, sequence_end = _validated_span(
        prompt_start=prompt_start,
        answer_start=answer_start,
        sequence_end=sequence_end,
        query_count=query_count,
        key_count=key_count,
        eps=eps,
    )
    span_start, span_end = (
        (prompt_start, answer_start)
        if pathway == "prompt"
        else (answer_start, sequence_end)
    )
    out = matrix.clone()
    rows = out[:, :, answer_start:sequence_end, :]
    rows[:, :, :, span_start:span_end] = 0.0
    if renormalize:
        row_mass = rows.sum(dim=-1, keepdim=True)
        rows.copy_(torch.where(row_mass > float(eps), rows / row_mass.clamp_min(float(eps)), rows))
    return out[0] if squeeze_batch else out


def attention_pathway_knockout_report(
    attention: Tensor,
    *,
    pathway: str,
    prompt_start: int = 0,
    answer_start: int,
    sequence_end: int | None = None,
    eps: float = 1e-8,
    metadata: Mapping[str, Any] | None = None,
) -> AttentionPathwayKnockoutReport:
    """Return before/after attention-pathway metrics for one knockout."""
    pathway = _coerce_pathway(pathway)
    matrix, _squeeze_batch = _attention_matrix(attention)
    query_count = int(matrix.shape[-2])
    key_count = int(matrix.shape[-1])
    prompt_start, answer_start, sequence_end = _validated_span(
        prompt_start=prompt_start,
        answer_start=answer_start,
        sequence_end=sequence_end,
        query_count=query_count,
        key_count=key_count,
        eps=eps,
    )
    baseline = attention_pathway_metrics(
        attention,
        prompt_start=prompt_start,
        answer_start=answer_start,
        sequence_end=sequence_end,
        eps=eps,
        metadata={"intervention": "baseline", **dict(metadata or {})},
    )
    knocked = knockout_attention_pathway(
        attention,
        pathway=pathway,
        prompt_start=prompt_start,
        answer_start=answer_start,
        sequence_end=sequence_end,
        eps=eps,
    )
    intervened = attention_pathway_metrics(
        knocked,
        prompt_start=prompt_start,
        answer_start=answer_start,
        sequence_end=sequence_end,
        eps=eps,
        metadata={"intervention": f"{pathway}_knockout", **dict(metadata or {})},
    )
    baseline_payload = baseline.to_dict()
    intervened_payload = intervened.to_dict()
    metric_names = (
        "prompt_flow_fraction",
        "answer_self_flow_fraction",
        "other_flow_fraction",
        "prompt_flow_loss",
        "pathway_gap",
        "pathway_concentration",
        "pathway_entropy",
    )
    deltas = {
        name: float(intervened_payload[name]) - float(baseline_payload[name])
        for name in metric_names
    }
    removed_mass = (
        baseline.prompt_flow_fraction
        if pathway == "prompt"
        else baseline.answer_self_flow_fraction
    )
    return AttentionPathwayKnockoutReport(
        pathway=pathway,
        prompt_start=prompt_start,
        answer_start=answer_start,
        sequence_end=sequence_end,
        baseline=baseline,
        intervened=intervened,
        deltas=deltas,
        removed_mass_fraction=float(removed_mass),
        metadata=metadata or {},
    )


def _attention_matrix(attention: Tensor) -> tuple[Tensor, bool]:
    tensor = torch.as_tensor(attention)
    if not tensor.is_floating_point():
        tensor = tensor.to(torch.float32)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
        squeeze_batch = True
    elif tensor.ndim == 4:
        squeeze_batch = False
    else:
        raise ValueError("attention must be shaped [head, query, key] or [batch, head, query, key].")
    if int(tensor.shape[0]) < 1 or int(tensor.shape[1]) < 1 or int(tensor.shape[2]) < 1 or int(tensor.shape[3]) < 1:
        raise ValueError("attention must have non-empty batch, head, query, and key dimensions.")
    if not torch.isfinite(tensor).all():
        raise ValueError("attention must contain only finite values.")
    if bool((tensor < 0).any()):
        raise ValueError("attention must contain non-negative weights.")
    return tensor, squeeze_batch


def _validated_span(
    *,
    prompt_start: int,
    answer_start: int,
    sequence_end: int | None,
    query_count: int,
    key_count: int,
    eps: float,
) -> tuple[int, int, int]:
    if float(eps) <= 0.0 or not math.isfinite(float(eps)):
        raise ValueError("eps must be positive and finite.")
    prompt_start = int(prompt_start)
    answer_start = int(answer_start)
    if sequence_end is None:
        sequence_end = min(int(query_count), int(key_count))
    sequence_end = int(sequence_end)
    if not (0 <= prompt_start < answer_start < sequence_end):
        raise ValueError("expected 0 <= prompt_start < answer_start < sequence_end.")
    if sequence_end > int(query_count) or sequence_end > int(key_count):
        raise ValueError("sequence_end must fit both query and key dimensions.")
    return prompt_start, answer_start, sequence_end


def _coerce_pathway(pathway: Any) -> str:
    text = str(pathway).strip().lower().replace("-", "_")
    if text in {"prompt", "question", "question_anchored", "prompt_anchored"}:
        return "prompt"
    if text in {"answer", "answer_self", "answer_anchored"}:
        return "answer"
    raise ValueError("pathway must be one of: prompt, answer.")


def _coerce_direction(direction: Any) -> str:
    text = str(direction).strip().lower()
    if text not in {"higher", "lower"}:
        raise ValueError("direction must be one of: higher, lower.")
    return text


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be a finite number.")
    return numeric


__all__ = [
    "PATHWAY_INTERVENTION_SCHEMA_VERSION",
    "AttentionPathwayKnockoutReport",
    "PathwayInterventionEffect",
    "attention_pathway_knockout_report",
    "knockout_attention_pathway",
    "pathway_intervention_effect",
]

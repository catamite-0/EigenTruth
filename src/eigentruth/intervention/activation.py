"""Temporary activation interventions for model-side mechanism reruns."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from eigentruth.intervention.hooks import TruthProbe

ACTIVATION_INTERVENTION_SCHEMA_VERSION = 1
ACTIVATION_INTERVENTION_SPANS = ("prompt", "answer", "all", "last", "first_answer")
ACTIVATION_INTERVENTION_MODES = ("zero", "scale", "mean")


@dataclass(frozen=True)
class ActivationInterventionSummary:
    """JSON-ready summary for one temporary activation intervention."""

    layer_idx: int
    span: str
    mode: str
    sequence_lengths: tuple[int, ...]
    answer_starts: tuple[int, ...]
    affected_token_count: int
    scale: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = ACTIVATION_INTERVENTION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary."""
        return {
            "schema_version": int(self.schema_version),
            "layer_idx": int(self.layer_idx),
            "span": self.span,
            "mode": self.mode,
            "scale": float(self.scale),
            "sequence_lengths": list(self.sequence_lengths),
            "answer_starts": list(self.answer_starts),
            "affected_token_count": int(self.affected_token_count),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ActivationInterventionSummary":
        """Build a summary from a JSON-like mapping."""
        return cls(
            layer_idx=int(data["layer_idx"]),
            span=_coerce_span(data["span"]),
            mode=_coerce_mode(data["mode"]),
            scale=_finite_float(data.get("scale", 0.0), name="scale"),
            sequence_lengths=tuple(int(value) for value in data["sequence_lengths"]),
            answer_starts=tuple(int(value) for value in data["answer_starts"]),
            affected_token_count=int(data["affected_token_count"]),
            metadata=dict(data.get("metadata") or {}),
            schema_version=int(data.get("schema_version", ACTIVATION_INTERVENTION_SCHEMA_VERSION)),
        )


def apply_activation_intervention(
    hidden: Tensor,
    *,
    sequence_lengths: Sequence[int],
    answer_starts: Sequence[int],
    span: str = "answer",
    mode: str = "zero",
    scale: float = 0.0,
) -> Tensor:
    """Apply a span-limited activation intervention to a hidden-state tensor."""
    if not isinstance(hidden, Tensor):
        raise TypeError("hidden must be a torch.Tensor.")
    if hidden.ndim != 3:
        raise ValueError("hidden must be shaped [batch, sequence, hidden_dim].")
    if not hidden.is_floating_point():
        raise ValueError("hidden must be a floating-point tensor.")
    span = _coerce_span(span)
    mode = _coerce_mode(mode)
    scale = _finite_float(scale, name="scale")
    sequence_lengths, answer_starts = _validate_spans(
        sequence_lengths=sequence_lengths,
        answer_starts=answer_starts,
        batch_size=int(hidden.shape[0]),
        max_sequence_length=int(hidden.shape[1]),
    )

    out = hidden.clone()
    for row, (sequence_end, answer_start) in enumerate(zip(sequence_lengths, answer_starts, strict=True)):
        start, end = _row_span(
            span,
            answer_start=answer_start,
            sequence_end=sequence_end,
        )
        if end <= start:
            continue
        selected = out[row, start:end, :]
        if mode == "zero":
            selected.zero_()
        elif mode == "scale":
            selected.mul_(scale)
        else:
            replacement = _mean_replacement(out[row, :sequence_end, :], start=start, end=end)
            selected.copy_(replacement.expand_as(selected))
    return out


class TemporaryActivationIntervention:
    """Context manager that applies an activation intervention through a forward hook."""

    def __init__(
        self,
        model: nn.Module,
        *,
        layer_idx: int,
        sequence_lengths: Sequence[int],
        answer_starts: Sequence[int],
        span: str = "answer",
        mode: str = "zero",
        scale: float = 0.0,
        custom_layer_path: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.layer_idx = int(layer_idx)
        self.span = _coerce_span(span)
        self.mode = _coerce_mode(mode)
        self.scale = _finite_float(scale, name="scale")
        self.sequence_lengths = tuple(int(value) for value in sequence_lengths)
        self.answer_starts = tuple(int(value) for value in answer_starts)
        self.custom_layer_path = custom_layer_path
        self.metadata = {} if metadata is None else dict(metadata)
        self._handle: torch.utils.hooks.RemovableHandle | None = None
        self.summary = ActivationInterventionSummary(
            layer_idx=self.layer_idx,
            span=self.span,
            mode=self.mode,
            scale=self.scale,
            sequence_lengths=self.sequence_lengths,
            answer_starts=self.answer_starts,
            affected_token_count=_affected_token_count(
                sequence_lengths=self.sequence_lengths,
                answer_starts=self.answer_starts,
                span=self.span,
            ),
            metadata=self.metadata,
        )

    def __enter__(self) -> "TemporaryActivationIntervention":
        """Register the temporary hook."""
        layers = TruthProbe._find_layers(self.model, custom_layer_path=self.custom_layer_path)
        target = TruthProbe._select_layer(layers, self.layer_idx)
        self._handle = target.register_forward_hook(self._hook)
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        """Remove the temporary hook."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def _hook(self, _module: nn.Module, _input: Any, output: Any) -> Any:
        hidden, repack_output = TruthProbe._unpack_output(output)
        intervened = apply_activation_intervention(
            hidden,
            sequence_lengths=self.sequence_lengths,
            answer_starts=self.answer_starts,
            span=self.span,
            mode=self.mode,
            scale=self.scale,
        )
        return repack_output(intervened)


def _validate_spans(
    *,
    sequence_lengths: Sequence[int],
    answer_starts: Sequence[int],
    batch_size: int,
    max_sequence_length: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    lengths = tuple(int(value) for value in sequence_lengths)
    starts = tuple(int(value) for value in answer_starts)
    if len(lengths) != batch_size or len(starts) != batch_size:
        raise ValueError("sequence_lengths and answer_starts must match hidden batch size.")
    for row, (sequence_end, answer_start) in enumerate(zip(lengths, starts, strict=True)):
        if not 0 < sequence_end <= max_sequence_length:
            raise ValueError(f"sequence_lengths[{row}] must be in (0, hidden sequence length].")
        if not 0 <= answer_start < sequence_end:
            raise ValueError(f"answer_starts[{row}] must be in [0, sequence_lengths[{row}]).")
    return lengths, starts


def _row_span(span: str, *, answer_start: int, sequence_end: int) -> tuple[int, int]:
    if span == "prompt":
        return 0, int(answer_start)
    if span == "answer":
        return int(answer_start), int(sequence_end)
    if span == "all":
        return 0, int(sequence_end)
    if span == "last":
        return int(sequence_end) - 1, int(sequence_end)
    if span == "first_answer":
        return int(answer_start), min(int(answer_start) + 1, int(sequence_end))
    raise ValueError("unsupported activation intervention span.")


def _mean_replacement(row: Tensor, *, start: int, end: int) -> Tensor:
    if row.ndim != 2:
        raise ValueError("row must be shaped [sequence, hidden_dim].")
    if start <= 0 and end >= int(row.shape[0]):
        return row.mean(dim=0, keepdim=True)
    pieces = []
    if start > 0:
        pieces.append(row[:start, :])
    if end < int(row.shape[0]):
        pieces.append(row[end:, :])
    if not pieces:
        return row[start:end, :].mean(dim=0, keepdim=True)
    return torch.cat(pieces, dim=0).mean(dim=0, keepdim=True)


def _affected_token_count(
    *,
    sequence_lengths: Sequence[int],
    answer_starts: Sequence[int],
    span: str,
) -> int:
    total = 0
    for sequence_end, answer_start in zip(sequence_lengths, answer_starts, strict=True):
        start, end = _row_span(span, answer_start=int(answer_start), sequence_end=int(sequence_end))
        total += max(0, end - start)
    return int(total)


def _coerce_span(value: Any) -> str:
    text = str(value).strip().lower().replace("-", "_")
    aliases = {
        "question": "prompt",
        "prompt": "prompt",
        "answer": "answer",
        "all": "all",
        "last": "last",
        "last_token": "last",
        "first_answer": "first_answer",
        "first_answer_token": "first_answer",
    }
    if text not in aliases:
        raise ValueError(f"span must be one of: {', '.join(ACTIVATION_INTERVENTION_SPANS)}.")
    return aliases[text]


def _coerce_mode(value: Any) -> str:
    text = str(value).strip().lower().replace("-", "_")
    if text not in ACTIVATION_INTERVENTION_MODES:
        raise ValueError(f"mode must be one of: {', '.join(ACTIVATION_INTERVENTION_MODES)}.")
    return text


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be a finite number.")
    return numeric


__all__ = [
    "ACTIVATION_INTERVENTION_MODES",
    "ACTIVATION_INTERVENTION_SCHEMA_VERSION",
    "ACTIVATION_INTERVENTION_SPANS",
    "ActivationInterventionSummary",
    "TemporaryActivationIntervention",
    "apply_activation_intervention",
]

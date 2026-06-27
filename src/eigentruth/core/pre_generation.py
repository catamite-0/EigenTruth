"""Pre-generation hidden-state risk probes."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from eigentruth.json_utils import to_jsonable

ATTENTION_SOFT_TARGET_PROBE_SCHEMA_VERSION = 1


def soft_error_rate_targets(sample_correctness: Sequence[Sequence[Any]]) -> Tensor:
    """Return per-prompt empirical error-rate soft targets from sampled correctness flags.

    Each inner sequence contains correctness indicators for sampled answers to
    the same prompt. The returned target is ``1 - mean(correctness)``, matching
    the pre-generation framing where risk is the model's probability of
    producing an incorrect answer under its own sampling distribution.
    """
    targets: list[float] = []
    for index, row in enumerate(sample_correctness):
        values = torch.as_tensor(tuple(row), dtype=torch.float32)
        if values.ndim != 1 or values.numel() == 0:
            raise ValueError(f"sample_correctness[{index}] must be a non-empty 1D sequence.")
        if not torch.isfinite(values).all():
            raise ValueError(f"sample_correctness[{index}] must contain only finite values.")
        if ((values != 0.0) & (values != 1.0)).any():
            raise ValueError(f"sample_correctness[{index}] must contain boolean or 0/1 values.")
        targets.append(float(1.0 - values.mean().item()))
    if not targets:
        raise ValueError("sample_correctness must contain at least one prompt.")
    return torch.tensor(targets, dtype=torch.float32)


@dataclass(frozen=True)
class AttentionSoftTargetProbeArtifact:
    """Attention-pooled hidden-state probe for pre-generation risk estimates."""

    query: Tensor
    classifier_weight: Tensor
    bias: float
    layer_idx: int | None = None
    training_summary: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = ATTENTION_SOFT_TARGET_PROBE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        query = _as_vector(self.query, field_name="query")
        classifier_weight = _as_vector(self.classifier_weight, field_name="classifier_weight")
        if query.shape != classifier_weight.shape:
            raise ValueError("query and classifier_weight must have the same shape.")
        bias = _finite_float(self.bias, field_name="bias")
        schema_version = int(self.schema_version)
        if schema_version != ATTENTION_SOFT_TARGET_PROBE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported attention soft-target probe schema_version={schema_version}; "
                f"expected {ATTENTION_SOFT_TARGET_PROBE_SCHEMA_VERSION}."
            )
        object.__setattr__(self, "query", query)
        object.__setattr__(self, "classifier_weight", classifier_weight)
        object.__setattr__(self, "bias", bias)
        object.__setattr__(self, "layer_idx", None if self.layer_idx is None else int(self.layer_idx))
        object.__setattr__(self, "training_summary", to_jsonable(self.training_summary))
        object.__setattr__(self, "metadata", to_jsonable(self.metadata))
        object.__setattr__(self, "schema_version", schema_version)

    @classmethod
    def fit(
        cls,
        hidden_states: Tensor,
        soft_targets: Tensor | Sequence[float],
        *,
        attention_mask: Tensor | Sequence[Sequence[Any]] | None = None,
        layer_idx: int | None = None,
        steps: int = 300,
        lr: float = 0.05,
        l2: float = 1e-4,
        seed: int = 0,
        metadata: Mapping[str, Any] | None = None,
    ) -> "AttentionSoftTargetProbeArtifact":
        """Fit an attention soft-target probe from token-level hidden states.

        ``hidden_states`` must have shape ``[N, T, D]``. ``soft_targets`` are
        risk probabilities in ``[0, 1]`` for the same ``N`` prompts.
        """
        hidden = _as_hidden_tensor(hidden_states)
        targets = _as_target_vector(soft_targets, expected_n=hidden.shape[0])
        mask = _as_attention_mask(attention_mask, batch_size=hidden.shape[0], seq_len=hidden.shape[1])
        steps = _positive_int(steps, field_name="steps")
        lr = _positive_float(lr, field_name="lr")
        l2 = _non_negative_float(l2, field_name="l2")
        hidden = hidden.detach().to(torch.float32)
        targets = targets.detach().to(torch.float32)
        mask = mask.detach()

        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        dim = hidden.shape[-1]
        query = (torch.randn(dim, generator=generator, dtype=torch.float32) * 0.01).requires_grad_(True)
        classifier_weight = (
            torch.randn(dim, generator=generator, dtype=torch.float32) * 0.01
        ).requires_grad_(True)
        target_mean = targets.mean().clamp(min=1e-4, max=1.0 - 1e-4)
        bias = torch.logit(target_mean).detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam((query, classifier_weight, bias), lr=lr)

        with torch.no_grad():
            initial_loss = _probe_loss(hidden, mask, targets, query, classifier_weight, bias, l2=l2)
        final_loss = initial_loss
        for _ in range(steps):
            optimizer.zero_grad()
            loss = _probe_loss(hidden, mask, targets, query, classifier_weight, bias, l2=l2)
            loss.backward()
            optimizer.step()
            final_loss = loss.detach()

        training_summary = {
            "steps": steps,
            "lr": lr,
            "l2": l2,
            "seed": int(seed),
            "target_mean": float(target_mean.item()),
            "initial_loss": float(initial_loss.item()),
            "final_loss": float(final_loss.item()),
            "hidden_shape": tuple(int(item) for item in hidden.shape),
            "masked_token_count": int((~mask).sum().item()),
        }
        return cls(
            query=query.detach().cpu(),
            classifier_weight=classifier_weight.detach().cpu(),
            bias=float(bias.detach().cpu().item()),
            layer_idx=layer_idx,
            training_summary=training_summary,
            metadata={} if metadata is None else metadata,
        )

    @property
    def hidden_dim(self) -> int:
        """Return the expected hidden dimension."""
        return int(self.query.numel())

    def attention_weights(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | Sequence[Sequence[Any]] | None = None,
    ) -> Tensor:
        """Return attention weights over prompt positions, shape ``[N, T]``."""
        hidden = _as_hidden_tensor(hidden_states)
        self._validate_hidden_dim(hidden)
        mask = _as_attention_mask(attention_mask, batch_size=hidden.shape[0], seq_len=hidden.shape[1])
        query = self.query.to(hidden.device)
        return _attention_weights(hidden, mask.to(hidden.device), query)

    def decision_function(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | Sequence[Sequence[Any]] | None = None,
    ) -> Tensor:
        """Return uncalibrated risk logits for prompts, shape ``[N]``."""
        hidden = _as_hidden_tensor(hidden_states)
        self._validate_hidden_dim(hidden)
        mask = _as_attention_mask(attention_mask, batch_size=hidden.shape[0], seq_len=hidden.shape[1])
        query = self.query.to(hidden.device)
        classifier_weight = self.classifier_weight.to(hidden.device)
        bias = torch.tensor(self.bias, dtype=torch.float32, device=hidden.device)
        weights = _attention_weights(hidden, mask.to(hidden.device), query)
        pooled = (weights.unsqueeze(-1) * hidden.to(torch.float32)).sum(dim=1)
        return pooled @ classifier_weight + bias

    def predict_proba(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | Sequence[Sequence[Any]] | None = None,
    ) -> Tensor:
        """Return pre-generation risk probabilities, shape ``[N]``."""
        return torch.sigmoid(self.decision_function(hidden_states, attention_mask=attention_mask))

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe metadata without embedding tensor payloads."""
        return {
            "schema_version": self.schema_version,
            "layer_idx": self.layer_idx,
            "hidden_dim": self.hidden_dim,
            "query_norm": float(torch.norm(self.query).item()),
            "classifier_weight_norm": float(torch.norm(self.classifier_weight).item()),
            "bias": self.bias,
            "training_summary": to_jsonable(self.training_summary),
            "metadata": to_jsonable(self.metadata),
        }

    def state_dict(self) -> dict[str, Any]:
        """Return a torch-serializable artifact payload."""
        return {
            "schema_version": self.schema_version,
            "query": self.query,
            "classifier_weight": self.classifier_weight,
            "bias": self.bias,
            "layer_idx": self.layer_idx,
            "training_summary": to_jsonable(self.training_summary),
            "metadata": to_jsonable(self.metadata),
        }

    def save(self, path: str | Path) -> None:
        """Save the probe artifact as one torch payload."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), output_path)

    @classmethod
    def load(cls, path: str | Path) -> "AttentionSoftTargetProbeArtifact":
        """Load a probe artifact saved by :meth:`save`."""
        payload = torch.load(path, weights_only=True)
        return cls(
            query=payload["query"],
            classifier_weight=payload["classifier_weight"],
            bias=float(payload["bias"]),
            layer_idx=payload.get("layer_idx"),
            training_summary=dict(payload.get("training_summary", {})),
            metadata=dict(payload.get("metadata", {})),
            schema_version=int(payload.get("schema_version", 0)),
        )

    def _validate_hidden_dim(self, hidden_states: Tensor) -> None:
        if hidden_states.shape[-1] != self.hidden_dim:
            raise ValueError(f"expected hidden_dim={self.hidden_dim}, got {hidden_states.shape[-1]}.")


def load_attention_soft_target_probe(path: str | Path) -> AttentionSoftTargetProbeArtifact:
    """Load a saved attention soft-target probe artifact."""
    return AttentionSoftTargetProbeArtifact.load(path)


def _probe_loss(
    hidden_states: Tensor,
    attention_mask: Tensor,
    targets: Tensor,
    query: Tensor,
    classifier_weight: Tensor,
    bias: Tensor,
    *,
    l2: float,
) -> Tensor:
    weights = _attention_weights(hidden_states, attention_mask, query)
    pooled = (weights.unsqueeze(-1) * hidden_states).sum(dim=1)
    logits = pooled @ classifier_weight + bias
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    if l2:
        loss = loss + l2 * (query.square().mean() + classifier_weight.square().mean())
    return loss


def _attention_weights(hidden_states: Tensor, attention_mask: Tensor, query: Tensor) -> Tensor:
    scale = sqrt(float(hidden_states.shape[-1]))
    logits = hidden_states.to(torch.float32) @ query.to(hidden_states.device) / scale
    logits = logits.masked_fill(~attention_mask.to(hidden_states.device), -1e9)
    return torch.softmax(logits, dim=1)


def _as_hidden_tensor(hidden_states: Tensor) -> Tensor:
    tensor = torch.as_tensor(hidden_states, dtype=torch.float32)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError(f"expected hidden_states with shape [N, T, D] or [T, D], got {tuple(tensor.shape)}.")
    if min(tensor.shape) < 1:
        raise ValueError("hidden_states dimensions must be non-empty.")
    if not torch.isfinite(tensor).all():
        raise ValueError("hidden_states must contain only finite values.")
    return tensor


def _as_target_vector(values: Tensor | Sequence[float], *, expected_n: int) -> Tensor:
    targets = torch.as_tensor(values, dtype=torch.float32)
    if targets.ndim != 1 or targets.shape[0] != expected_n:
        raise ValueError(f"expected soft_targets with shape [{expected_n}], got {tuple(targets.shape)}.")
    if not torch.isfinite(targets).all():
        raise ValueError("soft_targets must contain only finite values.")
    if ((targets < 0.0) | (targets > 1.0)).any():
        raise ValueError("soft_targets must be in [0, 1].")
    return targets


def _as_attention_mask(
    attention_mask: Tensor | Sequence[Sequence[Any]] | None,
    *,
    batch_size: int,
    seq_len: int,
) -> Tensor:
    if attention_mask is None:
        return torch.ones((batch_size, seq_len), dtype=torch.bool)
    mask = torch.as_tensor(attention_mask, dtype=torch.bool)
    if mask.ndim == 1 and batch_size == 1:
        mask = mask.unsqueeze(0)
    if mask.shape != (batch_size, seq_len):
        raise ValueError(f"expected attention_mask with shape [{batch_size}, {seq_len}], got {tuple(mask.shape)}.")
    if (~mask).all(dim=1).any():
        raise ValueError("each attention_mask row must keep at least one token.")
    return mask


def _as_vector(value: Tensor, *, field_name: str) -> Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32).detach().cpu().clone()
    if tensor.ndim != 1 or tensor.numel() < 1:
        raise ValueError(f"{field_name} must be a non-empty 1D tensor.")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{field_name} must contain only finite values.")
    return tensor.contiguous()


def _finite_float(value: Any, *, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite number.") from exc
    if not (parsed == parsed) or parsed in {float("inf"), float("-inf")}:
        raise ValueError(f"{field_name} must be a finite number.")
    return parsed


def _positive_int(value: Any, *, field_name: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"{field_name} must be >= 1.")
    return parsed


def _positive_float(value: Any, *, field_name: str) -> float:
    parsed = _finite_float(value, field_name=field_name)
    if parsed <= 0.0:
        raise ValueError(f"{field_name} must be > 0.")
    return parsed


def _non_negative_float(value: Any, *, field_name: str) -> float:
    parsed = _finite_float(value, field_name=field_name)
    if parsed < 0.0:
        raise ValueError(f"{field_name} must be >= 0.")
    return parsed


__all__ = [
    "ATTENTION_SOFT_TARGET_PROBE_SCHEMA_VERSION",
    "AttentionSoftTargetProbeArtifact",
    "load_attention_soft_target_probe",
    "soft_error_rate_targets",
]

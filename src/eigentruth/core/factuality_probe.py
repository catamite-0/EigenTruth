"""Claim-level hidden-state factuality probes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import log
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from eigentruth.json_utils import to_jsonable

CLAIM_FACTUALITY_PROBE_SCHEMA_VERSION = 1
CLAIM_FACTUALITY_POOLING_MODES = ("mean", "first_token", "last_token")


@dataclass(frozen=True)
class ClaimFactualityScore:
    """One claim-level hidden-state risk score."""

    claim_id: str
    risk_probability: float
    risk_logit: float
    text: str | None = None
    span: tuple[int, int] | None = None
    token_span: tuple[int, int] | None = None
    layer_idx: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        claim_id = str(self.claim_id).strip()
        if not claim_id:
            raise ValueError("claim_id must be non-empty.")
        risk_probability = _bounded_probability(self.risk_probability, field_name="risk_probability")
        risk_logit = _finite_float(self.risk_logit, field_name="risk_logit")
        object.__setattr__(self, "claim_id", claim_id)
        object.__setattr__(self, "risk_probability", risk_probability)
        object.__setattr__(self, "risk_logit", risk_logit)
        object.__setattr__(self, "text", None if self.text is None else str(self.text))
        object.__setattr__(self, "span", _optional_span(self.span, field_name="span"))
        object.__setattr__(self, "token_span", _optional_span(self.token_span, field_name="token_span"))
        object.__setattr__(self, "layer_idx", None if self.layer_idx is None else int(self.layer_idx))
        object.__setattr__(self, "metadata", to_jsonable(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe score payload."""
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "span": self.span,
            "token_span": self.token_span,
            "layer_idx": self.layer_idx,
            "risk_probability": self.risk_probability,
            "risk_logit": self.risk_logit,
            "metadata": to_jsonable(dict(self.metadata)),
        }


@dataclass(frozen=True)
class ClaimFactualityProbeArtifact:
    """Linear probe over pooled claim hidden states.

    The probe predicts hallucination or unsupported-claim risk. Higher
    probabilities are more anomalous, matching the project's conformal score
    direction conventions.
    """

    weight: Tensor
    bias: float
    feature_mean: Tensor
    feature_scale: Tensor
    pooling: str = "mean"
    layer_idx: int | None = None
    training_summary: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = CLAIM_FACTUALITY_PROBE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        weight = _as_vector(self.weight, field_name="weight")
        feature_mean = _as_vector(self.feature_mean, field_name="feature_mean")
        feature_scale = _as_vector(self.feature_scale, field_name="feature_scale")
        if feature_mean.shape != weight.shape or feature_scale.shape != weight.shape:
            raise ValueError("weight, feature_mean, and feature_scale must have the same shape.")
        if (feature_scale <= 0.0).any():
            raise ValueError("feature_scale must contain strictly positive values.")
        bias = _finite_float(self.bias, field_name="bias")
        pooling = str(self.pooling)
        if pooling not in CLAIM_FACTUALITY_POOLING_MODES:
            raise ValueError(f"pooling must be one of {CLAIM_FACTUALITY_POOLING_MODES}.")
        schema_version = int(self.schema_version)
        if schema_version != CLAIM_FACTUALITY_PROBE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported claim factuality probe schema_version={schema_version}; "
                f"expected {CLAIM_FACTUALITY_PROBE_SCHEMA_VERSION}."
            )
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "bias", bias)
        object.__setattr__(self, "feature_mean", feature_mean)
        object.__setattr__(self, "feature_scale", feature_scale)
        object.__setattr__(self, "pooling", pooling)
        object.__setattr__(self, "layer_idx", None if self.layer_idx is None else int(self.layer_idx))
        object.__setattr__(self, "training_summary", to_jsonable(dict(self.training_summary)))
        object.__setattr__(self, "metadata", to_jsonable(dict(self.metadata)))
        object.__setattr__(self, "schema_version", schema_version)

    @classmethod
    def fit(
        cls,
        hidden_states: Tensor,
        risk_targets: Tensor | Sequence[float],
        *,
        token_spans: Sequence[Sequence[int] | tuple[int, int]] | None = None,
        attention_mask: Tensor | Sequence[Sequence[Any]] | None = None,
        pooling: str = "mean",
        layer_idx: int | None = None,
        steps: int = 300,
        lr: float = 0.05,
        l2: float = 1e-4,
        seed: int = 0,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ClaimFactualityProbeArtifact":
        """Fit a linear factuality-risk probe.

        ``risk_targets`` may be binary labels or soft probabilities where ``1``
        means hallucinated, unsupported, or otherwise high-risk.
        """
        pooled = pool_claim_hidden_states(
            hidden_states,
            token_spans=token_spans,
            attention_mask=attention_mask,
            pooling=pooling,
        ).detach().to(torch.float32)
        targets = _as_probability_vector(risk_targets, expected_n=pooled.shape[0])
        steps = _positive_int(steps, field_name="steps")
        lr = _positive_float(lr, field_name="lr")
        l2 = _non_negative_float(l2, field_name="l2")

        feature_mean = pooled.mean(dim=0)
        feature_scale = pooled.std(dim=0, unbiased=False).clamp(min=1e-6)
        normalized = (pooled - feature_mean) / feature_scale

        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        weight = (torch.randn(normalized.shape[-1], generator=generator) * 0.01).requires_grad_(True)
        target_mean = targets.mean().clamp(min=1e-4, max=1.0 - 1e-4)
        bias = torch.tensor(_logit(float(target_mean.item())), dtype=torch.float32).requires_grad_(True)
        optimizer = torch.optim.Adam((weight, bias), lr=lr)

        with torch.no_grad():
            initial_loss = _probe_loss(normalized, targets, weight, bias, l2=l2)
        final_loss = initial_loss
        for _ in range(steps):
            optimizer.zero_grad()
            loss = _probe_loss(normalized, targets, weight, bias, l2=l2)
            loss.backward()
            optimizer.step()
            final_loss = loss.detach()

        with torch.no_grad():
            logits = normalized @ weight.detach() + bias.detach()
            probabilities = torch.sigmoid(logits)
            hard_targets = targets >= 0.5
            hard_predictions = probabilities >= 0.5
            train_accuracy = (hard_targets == hard_predictions).to(torch.float32).mean()
        training_summary = {
            "steps": steps,
            "lr": lr,
            "l2": l2,
            "seed": int(seed),
            "target_mean": float(target_mean.item()),
            "initial_loss": float(initial_loss.item()),
            "final_loss": float(final_loss.item()),
            "train_accuracy_at_0_5": float(train_accuracy.item()),
            "hidden_shape": tuple(int(item) for item in torch.as_tensor(hidden_states).shape),
            "pooled_shape": tuple(int(item) for item in pooled.shape),
            "pooling": pooling,
        }
        return cls(
            weight=weight.detach().cpu(),
            bias=float(bias.detach().cpu().item()),
            feature_mean=feature_mean.detach().cpu(),
            feature_scale=feature_scale.detach().cpu(),
            pooling=pooling,
            layer_idx=layer_idx,
            training_summary=training_summary,
            metadata={} if metadata is None else metadata,
        )

    @property
    def hidden_dim(self) -> int:
        """Return the expected hidden-state dimension."""
        return int(self.weight.numel())

    def decision_function(
        self,
        hidden_states: Tensor,
        *,
        token_spans: Sequence[Sequence[int] | tuple[int, int]] | None = None,
        attention_mask: Tensor | Sequence[Sequence[Any]] | None = None,
    ) -> Tensor:
        """Return uncalibrated claim-risk logits."""
        pooled = pool_claim_hidden_states(
            hidden_states,
            token_spans=token_spans,
            attention_mask=attention_mask,
            pooling=self.pooling,
        )
        self._validate_hidden_dim(pooled)
        mean = self.feature_mean.to(pooled.device)
        scale = self.feature_scale.to(pooled.device)
        weight = self.weight.to(pooled.device)
        bias = torch.tensor(self.bias, dtype=torch.float32, device=pooled.device)
        normalized = (pooled.to(torch.float32) - mean) / scale
        return normalized @ weight + bias

    def predict_proba(
        self,
        hidden_states: Tensor,
        *,
        token_spans: Sequence[Sequence[int] | tuple[int, int]] | None = None,
        attention_mask: Tensor | Sequence[Sequence[Any]] | None = None,
    ) -> Tensor:
        """Return claim-risk probabilities. Higher is more suspicious."""
        return torch.sigmoid(
            self.decision_function(
                hidden_states,
                token_spans=token_spans,
                attention_mask=attention_mask,
            )
        )

    def score_claims(
        self,
        claims: Sequence[Any],
        hidden_states: Tensor,
        *,
        token_spans: Sequence[Sequence[int] | tuple[int, int]] | None = None,
        attention_mask: Tensor | Sequence[Sequence[Any]] | None = None,
    ) -> tuple[ClaimFactualityScore, ...]:
        """Score claims and return JSON-ready per-claim payloads."""
        logits = self.decision_function(
            hidden_states,
            token_spans=token_spans,
            attention_mask=attention_mask,
        ).detach().cpu()
        probabilities = torch.sigmoid(logits)
        claim_payloads = tuple(_claim_payload(claim, index) for index, claim in enumerate(claims))
        if len(claim_payloads) != int(logits.shape[0]):
            raise ValueError("claims length must match the number of hidden-state rows.")
        spans = (
            None
            if token_spans is None
            else _as_token_spans(token_spans, batch_size=len(claim_payloads), seq_len=None)
        )
        return tuple(
            ClaimFactualityScore(
                claim_id=claim["claim_id"],
                text=claim.get("text"),
                span=claim.get("span"),
                token_span=None if spans is None else spans[index],
                layer_idx=self.layer_idx,
                risk_probability=float(probabilities[index].item()),
                risk_logit=float(logits[index].item()),
                metadata={
                    "score_name": "claim_factuality_risk",
                    "pooling": self.pooling,
                },
            )
            for index, claim in enumerate(claim_payloads)
        )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe metadata without embedding tensor payloads."""
        return {
            "schema_version": self.schema_version,
            "layer_idx": self.layer_idx,
            "hidden_dim": self.hidden_dim,
            "pooling": self.pooling,
            "weight_norm": float(torch.norm(self.weight).item()),
            "bias": self.bias,
            "feature_scale_min": float(self.feature_scale.min().item()),
            "feature_scale_max": float(self.feature_scale.max().item()),
            "training_summary": to_jsonable(dict(self.training_summary)),
            "metadata": to_jsonable(dict(self.metadata)),
        }

    def state_dict(self) -> dict[str, Any]:
        """Return a torch-serializable artifact payload."""
        return {
            "schema_version": self.schema_version,
            "weight": self.weight,
            "bias": self.bias,
            "feature_mean": self.feature_mean,
            "feature_scale": self.feature_scale,
            "pooling": self.pooling,
            "layer_idx": self.layer_idx,
            "training_summary": to_jsonable(dict(self.training_summary)),
            "metadata": to_jsonable(dict(self.metadata)),
        }

    def save(self, path: str | Path) -> None:
        """Save the probe artifact as a torch payload."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), output_path)

    @classmethod
    def load(cls, path: str | Path) -> "ClaimFactualityProbeArtifact":
        """Load a claim factuality probe artifact saved by :meth:`save`."""
        payload = torch.load(path, weights_only=True)
        return cls(
            weight=payload["weight"],
            bias=float(payload["bias"]),
            feature_mean=payload["feature_mean"],
            feature_scale=payload["feature_scale"],
            pooling=str(payload.get("pooling", "mean")),
            layer_idx=payload.get("layer_idx"),
            training_summary=dict(payload.get("training_summary", {})),
            metadata=dict(payload.get("metadata", {})),
            schema_version=int(payload.get("schema_version", 0)),
        )

    def _validate_hidden_dim(self, pooled_states: Tensor) -> None:
        if pooled_states.shape[-1] != self.hidden_dim:
            raise ValueError(f"expected hidden_dim={self.hidden_dim}, got {pooled_states.shape[-1]}.")


def load_claim_factuality_probe(path: str | Path) -> ClaimFactualityProbeArtifact:
    """Load a saved claim factuality probe artifact."""
    return ClaimFactualityProbeArtifact.load(path)


def claim_factuality_diagnostics(
    scores: Sequence[ClaimFactualityScore | Mapping[str, Any]],
    *,
    risk_threshold: float = 0.5,
) -> dict[str, Any]:
    """Summarize claim factuality scores for diagnostics or trace metadata."""
    threshold = _bounded_probability(risk_threshold, field_name="risk_threshold")
    normalized = tuple(_score_payload(score) for score in scores)
    risks = [float(score["risk_probability"]) for score in normalized]
    high_risk_ids = tuple(
        str(score["claim_id"])
        for score in normalized
        if float(score["risk_probability"]) >= threshold
    )
    mean_risk = None if not risks else sum(risks) / len(risks)
    max_risk = None if not risks else max(risks)
    return {
        "available": bool(normalized),
        "score_name": "claim_factuality_risk",
        "direction": "higher",
        "risk_threshold": threshold,
        "claim_count": len(normalized),
        "high_risk_claim_count": len(high_risk_ids),
        "high_risk_claim_ids": high_risk_ids,
        "mean_risk_probability": mean_risk,
        "max_risk_probability": max_risk,
    }


def pool_claim_hidden_states(
    hidden_states: Tensor,
    *,
    token_spans: Sequence[Sequence[int] | tuple[int, int]] | None = None,
    attention_mask: Tensor | Sequence[Sequence[Any]] | None = None,
    pooling: str = "mean",
) -> Tensor:
    """Pool token hidden states into one vector per claim.

    ``hidden_states`` may already be pooled with shape ``[N, D]``. Token-level
    inputs use shape ``[N, T, D]`` and can be restricted by token spans and/or an
    attention mask.
    """
    pooling = str(pooling)
    if pooling not in CLAIM_FACTUALITY_POOLING_MODES:
        raise ValueError(f"pooling must be one of {CLAIM_FACTUALITY_POOLING_MODES}.")
    hidden = _as_hidden_tensor(hidden_states)
    if hidden.ndim == 2:
        if token_spans is not None:
            raise ValueError("token_spans require token-level hidden states with shape [N, T, D].")
        if attention_mask is not None:
            raise ValueError("attention_mask requires token-level hidden states with shape [N, T, D].")
        return hidden
    spans = _as_token_spans(token_spans, batch_size=hidden.shape[0], seq_len=hidden.shape[1])
    mask = _as_attention_mask(attention_mask, batch_size=hidden.shape[0], seq_len=hidden.shape[1])
    pooled_rows: list[Tensor] = []
    for row_index, row in enumerate(hidden):
        start, end = spans[row_index]
        row_mask = mask[row_index, start:end]
        if not bool(row_mask.any()):
            raise ValueError(f"token span {row_index} contains no unmasked tokens.")
        values = row[start:end][row_mask]
        if pooling == "mean":
            pooled_rows.append(values.mean(dim=0))
        elif pooling == "first_token":
            pooled_rows.append(values[0])
        else:
            pooled_rows.append(values[-1])
    return torch.stack(pooled_rows, dim=0).to(torch.float32)


def _probe_loss(features: Tensor, targets: Tensor, weight: Tensor, bias: Tensor, *, l2: float) -> Tensor:
    logits = features @ weight + bias
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    if l2:
        loss = loss + l2 * weight.square().mean()
    return loss


def _as_hidden_tensor(hidden_states: Tensor) -> Tensor:
    tensor = torch.as_tensor(hidden_states, dtype=torch.float32)
    if tensor.ndim not in {2, 3}:
        raise ValueError(f"expected hidden_states with shape [N, D] or [N, T, D], got {tuple(tensor.shape)}.")
    if min(tensor.shape) < 1:
        raise ValueError("hidden_states dimensions must be non-empty.")
    if not torch.isfinite(tensor).all():
        raise ValueError("hidden_states must contain only finite values.")
    return tensor


def _as_vector(values: Tensor, *, field_name: str) -> Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim != 1 or tensor.numel() < 1:
        raise ValueError(f"{field_name} must be a non-empty vector.")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{field_name} must contain only finite values.")
    return tensor.detach().cpu()


def _as_probability_vector(values: Tensor | Sequence[float], *, expected_n: int) -> Tensor:
    targets = torch.as_tensor(values, dtype=torch.float32)
    if targets.ndim != 1 or targets.shape[0] != expected_n:
        raise ValueError(f"risk_targets must have shape [{expected_n}].")
    if not torch.isfinite(targets).all():
        raise ValueError("risk_targets must contain only finite values.")
    if ((targets < 0.0) | (targets > 1.0)).any():
        raise ValueError("risk_targets must be probabilities in [0, 1].")
    return targets.detach().to(torch.float32)


def _as_attention_mask(
    mask: Tensor | Sequence[Sequence[Any]] | None,
    *,
    batch_size: int,
    seq_len: int,
) -> Tensor:
    if mask is None:
        return torch.ones((batch_size, seq_len), dtype=torch.bool)
    tensor = torch.as_tensor(mask)
    if tensor.shape != (batch_size, seq_len):
        raise ValueError(f"attention_mask must have shape [{batch_size}, {seq_len}].")
    return tensor.to(dtype=torch.bool)


def _as_token_spans(
    spans: Sequence[Sequence[int] | tuple[int, int]] | None,
    *,
    batch_size: int,
    seq_len: int | None,
) -> tuple[tuple[int, int], ...]:
    if spans is None:
        if seq_len is None:
            raise ValueError("token_spans are required when seq_len is not available.")
        return tuple((0, int(seq_len)) for _ in range(batch_size))
    if len(spans) != batch_size:
        raise ValueError(f"token_spans must contain {batch_size} spans.")
    normalized: list[tuple[int, int]] = []
    for index, span in enumerate(spans):
        if len(span) != 2:
            raise ValueError(f"token_spans[{index}] must contain exactly two integers.")
        start = int(span[0])
        end = int(span[1])
        if start < 0 or end <= start:
            raise ValueError(f"token_spans[{index}] must be a non-empty half-open span.")
        if seq_len is not None and end > seq_len:
            raise ValueError(f"token_spans[{index}] exceeds sequence length {seq_len}.")
        normalized.append((start, end))
    return tuple(normalized)


def _claim_payload(claim: Any, index: int) -> dict[str, Any]:
    if isinstance(claim, Mapping):
        claim_id = claim.get("claim_id") or claim.get("id") or f"c{index + 1}"
        text = None if claim.get("text") is None else str(claim.get("text"))
        span = claim.get("span")
    else:
        claim_id = getattr(claim, "claim_id", None) or getattr(claim, "id", None) or f"c{index + 1}"
        text_value = getattr(claim, "text", None)
        text = None if text_value is None else str(text_value)
        span = getattr(claim, "span", None)
    return {
        "claim_id": str(claim_id),
        "text": text,
        "span": _optional_span(span, field_name="span"),
    }


def _score_payload(score: ClaimFactualityScore | Mapping[str, Any]) -> dict[str, Any]:
    payload = score.to_dict() if isinstance(score, ClaimFactualityScore) else dict(score)
    if "claim_id" not in payload:
        raise ValueError("score payload must contain claim_id.")
    if "risk_probability" not in payload:
        raise ValueError("score payload must contain risk_probability.")
    return {
        "claim_id": str(payload["claim_id"]),
        "risk_probability": _bounded_probability(payload["risk_probability"], field_name="risk_probability"),
    }


def _optional_span(span: Any, *, field_name: str) -> tuple[int, int] | None:
    if span is None:
        return None
    if not isinstance(span, Sequence) or isinstance(span, (str, bytes)) or len(span) != 2:
        raise ValueError(f"{field_name} must be a pair of integer offsets.")
    start = int(span[0])
    end = int(span[1])
    if start < 0 or end < start:
        raise ValueError(f"{field_name} must satisfy 0 <= start <= end.")
    return (start, end)


def _finite_float(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a finite float.")
    number = float(value)
    if not torch.isfinite(torch.tensor(number)).item():
        raise ValueError(f"{field_name} must be finite.")
    return number


def _positive_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive integer.")
    number = int(value)
    if number < 1:
        raise ValueError(f"{field_name} must be >= 1.")
    return number


def _positive_float(value: Any, *, field_name: str) -> float:
    number = _finite_float(value, field_name=field_name)
    if number <= 0.0:
        raise ValueError(f"{field_name} must be > 0.")
    return number


def _non_negative_float(value: Any, *, field_name: str) -> float:
    number = _finite_float(value, field_name=field_name)
    if number < 0.0:
        raise ValueError(f"{field_name} must be >= 0.")
    return number


def _bounded_probability(value: Any, *, field_name: str) -> float:
    number = _finite_float(value, field_name=field_name)
    if not (0.0 <= number <= 1.0):
        raise ValueError(f"{field_name} must be in [0, 1].")
    return number


def _logit(value: float) -> float:
    return log(value / (1.0 - value))

"""Optional callback adapters for training-side representation telemetry."""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from eigentruth.training.telemetry import (
    RepresentationTelemetryRecorder,
    RepresentationTelemetryReport,
    RepresentationTelemetrySnapshot,
)

BatchProvider = Mapping[str, Any] | Callable[..., Mapping[str, Any] | None]


@dataclass(frozen=True)
class TelemetryCallbackEvent:
    """One callback capture attempt."""

    event: str
    step: int
    status: str
    snapshot_count: int = 0
    reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready event payload."""
        return {
            "event": self.event,
            "step": int(self.step),
            "status": self.status,
            "snapshot_count": int(self.snapshot_count),
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }


class RepTelemetryCallback:
    """HF Trainer-compatible representation telemetry callback without importing transformers.

    The callback uses Hugging Face Trainer hook names, but it is plain Python and can
    also be called directly from a custom PyTorch loop through ``capture``.
    """

    def __init__(
        self,
        *,
        batch_provider: BatchProvider | None = None,
        recorder: RepresentationTelemetryRecorder | None = None,
        layers: Sequence[int] | None = (-2, -1),
        pooling: str = "last_token",
        capture_every_n_steps: int | None = 100,
        capture_on_train_begin: bool = True,
        capture_on_epoch_end: bool = False,
        capture_on_evaluate: bool = True,
        report_path: str | Path | None = None,
        raise_on_error: bool = False,
        metadata: Mapping[str, Any] | None = None,
        covariance_mode: str = "shrinkage",
        covariance_low_rank: int = 16,
        distance_covariance_mode: str = "model",
        baseline_strategy: str = "first",
    ) -> None:
        if capture_every_n_steps is not None and int(capture_every_n_steps) < 1:
            raise ValueError("capture_every_n_steps must be positive when provided.")
        if pooling not in {"last_token", "first_token", "mean_token", "flat_tokens"}:
            raise ValueError("pooling must be one of: last_token, first_token, mean_token, flat_tokens.")
        self.batch_provider = batch_provider
        self.recorder = recorder or RepresentationTelemetryRecorder(
            layers=None if layers is None else tuple(int(layer) for layer in layers),
            covariance_mode=covariance_mode,
            covariance_low_rank=covariance_low_rank,
            distance_covariance_mode=distance_covariance_mode,
            baseline_strategy=baseline_strategy,
        )
        self.layers = self.recorder.layers
        self.pooling = pooling
        self.capture_every_n_steps = None if capture_every_n_steps is None else int(capture_every_n_steps)
        self.capture_on_train_begin = bool(capture_on_train_begin)
        self.capture_on_epoch_end = bool(capture_on_epoch_end)
        self.capture_on_evaluate = bool(capture_on_evaluate)
        self.report_path = None if report_path is None else Path(report_path)
        self.raise_on_error = bool(raise_on_error)
        self.metadata = dict(metadata or {})
        self._events: list[TelemetryCallbackEvent] = []

    @property
    def events(self) -> tuple[TelemetryCallbackEvent, ...]:
        """Capture attempts recorded by the callback."""
        return tuple(self._events)

    def on_train_begin(self, args: Any, state: Any, control: Any, model: Any | None = None, **kwargs: Any) -> Any:
        """HF Trainer hook; captures the initial baseline batch when enabled."""
        if self.capture_on_train_begin:
            self.capture("train_begin", args=args, state=state, control=control, model=model, **kwargs)
        return control

    def on_step_end(self, args: Any, state: Any, control: Any, model: Any | None = None, **kwargs: Any) -> Any:
        """HF Trainer hook; captures at the configured step interval."""
        step = _global_step(state)
        if self.capture_every_n_steps is not None and step > 0 and step % self.capture_every_n_steps == 0:
            self.capture("step_end", args=args, state=state, control=control, model=model, **kwargs)
        return control

    def on_epoch_end(self, args: Any, state: Any, control: Any, model: Any | None = None, **kwargs: Any) -> Any:
        """HF Trainer hook; captures an epoch-end batch when enabled."""
        if self.capture_on_epoch_end:
            self.capture("epoch_end", args=args, state=state, control=control, model=model, **kwargs)
        return control

    def on_evaluate(self, args: Any, state: Any, control: Any, model: Any | None = None, **kwargs: Any) -> Any:
        """HF Trainer hook; captures an evaluation batch when enabled."""
        if self.capture_on_evaluate:
            self.capture("evaluate", args=args, state=state, control=control, model=model, **kwargs)
        return control

    def on_train_end(self, args: Any, state: Any, control: Any, model: Any | None = None, **kwargs: Any) -> Any:
        """HF Trainer hook; writes the report when ``report_path`` is configured."""
        del args, state, model, kwargs
        if self.report_path is not None:
            self.write_report(self.report_path)
        return control

    def capture(
        self,
        event: str,
        *,
        args: Any = None,
        state: Any = None,
        control: Any = None,
        model: Any | None = None,
        **kwargs: Any,
    ) -> tuple[RepresentationTelemetrySnapshot, ...]:
        """Capture one telemetry batch and return recorded snapshots."""
        step = _global_step(state)
        try:
            if model is None:
                raise ValueError("model is required for telemetry capture.")
            batch = _resolve_batch(
                self.batch_provider,
                args=args,
                state=state,
                control=control,
                model=model,
                event=event,
                **kwargs,
            )
            outputs = _call_model_for_hidden_states(model, batch)
            layer_states = extract_hidden_state_matrices(outputs, layers=self.layers, pooling=self.pooling)
            snapshots = self.recorder.record_step(
                step,
                layer_states,
                metadata=_capture_metadata(event=event, state=state, extra=self.metadata),
            )
        except Exception as exc:
            callback_event = TelemetryCallbackEvent(
                event=event,
                step=step,
                status="error",
                reason=str(exc),
            )
            self._events.append(callback_event)
            if self.raise_on_error:
                raise
            return ()
        self._events.append(
            TelemetryCallbackEvent(
                event=event,
                step=step,
                status="captured",
                snapshot_count=len(snapshots),
            )
        )
        return snapshots

    def to_report(self, *, metadata: Mapping[str, Any] | None = None) -> RepresentationTelemetryReport:
        """Return a telemetry report that includes callback capture events."""
        callback_metadata = {
            **self.metadata,
            "callback": {
                "type": "RepTelemetryCallback",
                "pooling": self.pooling,
                "capture_every_n_steps": self.capture_every_n_steps,
                "capture_on_train_begin": self.capture_on_train_begin,
                "capture_on_epoch_end": self.capture_on_epoch_end,
                "capture_on_evaluate": self.capture_on_evaluate,
                "events": [event.to_dict() for event in self._events],
            },
        }
        if metadata:
            callback_metadata.update(metadata)
        return self.recorder.to_report(metadata=callback_metadata)

    def write_report(self, path: str | Path | None = None) -> Path:
        """Write the callback report as JSON and return its path."""
        output_path = Path(path) if path is not None else self.report_path
        if output_path is None:
            raise ValueError("path is required when report_path was not configured.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_report().to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return output_path


def extract_hidden_state_matrices(
    outputs: Any,
    *,
    layers: Sequence[int] | None = (-2, -1),
    pooling: str = "last_token",
) -> dict[int, Tensor]:
    """Extract per-layer 2D hidden-state matrices from model outputs."""
    if pooling not in {"last_token", "first_token", "mean_token", "flat_tokens"}:
        raise ValueError("pooling must be one of: last_token, first_token, mean_token, flat_tokens.")
    hidden_states = _hidden_states_from_outputs(outputs)
    selected_layers = tuple(range(len(hidden_states))) if layers is None else tuple(int(layer) for layer in layers)
    matrices: dict[int, Tensor] = {}
    for layer in selected_layers:
        try:
            hidden = hidden_states[layer]
        except IndexError as exc:
            raise ValueError(f"hidden_states does not contain layer {layer}.") from exc
        matrices[layer] = _pool_hidden_state(hidden, pooling=pooling)
    return matrices


def _resolve_batch(provider: BatchProvider | None, **context: Any) -> Mapping[str, Any]:
    if provider is None:
        candidate = context.get("inputs") or context.get("batch")
    elif isinstance(provider, Mapping):
        candidate = provider
    elif callable(provider):
        candidate = _call_provider(provider, context)
    else:
        raise TypeError("batch_provider must be a mapping, callable, or None.")
    if candidate is None:
        raise ValueError("telemetry capture requires a batch_provider or callback inputs/batch.")
    if not isinstance(candidate, Mapping):
        raise TypeError("telemetry batch must be a mapping of model keyword inputs.")
    return dict(candidate)


def _call_provider(provider: Callable[..., Mapping[str, Any] | None], context: Mapping[str, Any]) -> Any:
    try:
        signature = inspect.signature(provider)
    except (TypeError, ValueError):
        return provider(**context)
    parameters = signature.parameters
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return provider(**context)
    if not parameters:
        return provider()
    kwargs = {name: value for name, value in context.items() if name in parameters}
    return provider(**kwargs)


def _call_model_for_hidden_states(model: Any, batch: Mapping[str, Any]) -> Any:
    was_training = getattr(model, "training", None)
    if hasattr(model, "eval"):
        model.eval()
    try:
        with torch.no_grad():
            hidden_batch = dict(batch)
            hidden_batch.setdefault("output_hidden_states", True)
            hidden_batch.setdefault("return_dict", True)
            try:
                return model(**hidden_batch)
            except TypeError:
                return model(**dict(batch))
    finally:
        if was_training is not None and hasattr(model, "train"):
            model.train(bool(was_training))


def _hidden_states_from_outputs(outputs: Any) -> Sequence[Any]:
    if isinstance(outputs, Mapping):
        hidden_states = outputs.get("hidden_states")
    else:
        hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None:
        raise ValueError("model outputs must expose hidden_states.")
    if not isinstance(hidden_states, Sequence):
        raise TypeError("hidden_states must be a sequence of tensors.")
    if not hidden_states:
        raise ValueError("hidden_states must not be empty.")
    return hidden_states


def _pool_hidden_state(hidden: Any, *, pooling: str) -> Tensor:
    tensor = torch.as_tensor(hidden, dtype=torch.float32).detach().cpu()
    if tensor.ndim == 2:
        return tensor
    if tensor.ndim != 3:
        raise ValueError("hidden state tensors must be 2D [sample, hidden] or 3D [batch, token, hidden].")
    if pooling == "last_token":
        return tensor[:, -1, :]
    if pooling == "first_token":
        return tensor[:, 0, :]
    if pooling == "mean_token":
        return tensor.mean(dim=1)
    if pooling == "flat_tokens":
        return tensor.reshape(-1, tensor.shape[-1])
    raise ValueError("unsupported pooling mode.")


def _global_step(state: Any) -> int:
    if state is None:
        return 0
    if isinstance(state, Mapping):
        value = state.get("global_step", 0)
    else:
        value = getattr(state, "global_step", 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _capture_metadata(*, event: str, state: Any, extra: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(extra)
    payload["callback_event"] = event
    if state is not None:
        epoch = state.get("epoch") if isinstance(state, Mapping) else getattr(state, "epoch", None)
        if epoch is not None:
            payload["epoch"] = epoch
    return payload

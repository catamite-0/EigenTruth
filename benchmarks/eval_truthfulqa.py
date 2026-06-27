"""EigenTruth 基准评测 — 隐状态几何信号能否分离真/假陈述 (AUROC)。
EigenTruth benchmark — can hidden-state geometry separate true vs. false statements (AUROC)?

研究问题 / Research questions:
    1. 流形马氏距离 (maha_last) 能否把"假陈述"排在"真陈述"之上？ 它打得过困惑度基线 (nll) 吗？
       Does manifold Mahalanobis distance rank false statements above true ones, and beat a
       perplexity baseline?
    2. 双曲离散度 (disp_hse) 是否优于欧氏离散度 (disp_euclid)？ —— 即"双曲几何有没有用"的消融。
       Does hyperbolic dispersion (disp_hse) beat Euclidean dispersion (disp_euclid)? — the
       "does the hyperbolic projection earn its keep?" ablation.
    3. 对比方向投影 (truth_proj，即工具自带的 contrastive_direction 用作 mass-mean 探针，
       参见 Marks & Tegmark) 是否是更强的检测器？最佳目标层在哪 (--sweep)？
       Is the contrastive-direction projection (the tool's own steering direction used as a
       mass-mean probe, cf. Marks & Tegmark) an even stronger detector, and which layer is
       best (--sweep)?
    4. 跨层 residual 更新幅度 (resid_update_norm) 是否提供 ICR-like 隐状态动态信号？
       Does the cross-layer residual update magnitude provide an ICR-like hidden-state
       dynamics signal?

方法 / Method (SAPLMA 式、确定性、无需 LLM 裁判 / SAPLMA-style, deterministic, judge-free):
    - 从**留出**题目的*正确*答案构建真值流形（无标签泄漏）。
      Build the truth manifold from the *correct* answers of **held-out** questions (no leakage).
    - 对其余题目的每条候选答案（正确=负类, 错误=正类/幻觉）做单次前向，提取目标层隐状态。
      For each candidate answer of the remaining questions (correct=negative, incorrect=positive),
      run a single forward pass and read the target-layer hidden states.
    - 每条陈述计算多种内部诊断分数，分别报告 AUROC。
      Score each statement with multiple internal diagnostics and report AUROC per signal.

局限 / Limitations:
    - 强制候选答案的"陈述级"打分是开放生成幻觉的*代理*，干净地检验表征假设但不等同于在线检测。
      Forced-answer statement scoring is a *proxy* for open-generation hallucination; it cleanly
      tests the representation hypothesis but is not the same as online detection.
    - 句内 token 离散度和 eigenscore 是基于多次采样语义不确定性 / INSIDE EigenScore 的廉价代理。
      Within-statement token dispersion and eigenscore are cheap proxies for sample-based
      semantic uncertainty / INSIDE EigenScore.
    - 小模型 + 几百条样本 → AUROC 置信区间较宽，结果仅供参考，非定论。
      Small model + a few hundred items → wide AUROC CIs; results are indicative, not conclusive.

用法 / Usage:
    # 真实基准 / real benchmark (downloads model + TruthfulQA):
    python benchmarks/eval_truthfulqa.py --model Qwen/Qwen2.5-0.5B-Instruct --layer -8 --limit 200
    # 快速管线自检 / fast pipeline smoke check (tiny model, bundled statements, no dataset):
    python benchmarks/eval_truthfulqa.py --model sshleifer/tiny-gpt2 --offline
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import sys
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Mapping, Optional, Sequence

import torch

from eigentruth.calibration import DEFAULT_SCORE_DIRECTIONS
from eigentruth.core import (
    TruthSubspace,
    embedding_semantic_entropy,
    internal_eigenscore,
    lexical_semantic_energy,
    lexical_semantic_entropy,
)
from eigentruth.core.math_engine import (
    COVARIANCE_MODES,
    TruthManifold,
    hyperbolic_semantic_entropy,
    mahalanobis_distance,  # noqa: F401 - legacy benchmark module alias.
    poincare_map,
)
from eigentruth.eval.conformal import directional_conformal_threshold
from eigentruth.eval.metrics import (
    euclidean_dispersion,
    roc_auc,
    selective_classification_report,
    topk_normalized_entropy,
)
from eigentruth.eval.score_dump import write_score_dump_jsonl_mapping
from eigentruth.intervention.hooks import TruthProbe
from eigentruth.verify import Claim, SelfConsistencyVerifier

FIRST_TOKEN_ENTROPY_SIGNAL = "first_token_entropy"
FIRST_TOKEN_TOP_K_DEFAULT = 20
SIGNALS = [
    "maha_last",
    "truth_proj",
    "subspace_resid",
    "resid_update_norm",
    "disp_euclid",
    "disp_hse",
    "eigenscore",
    FIRST_TOKEN_ENTROPY_SIGNAL,
    "nll_answer",
]
INSIDE_SIGNAL = "inside_eigenscore"
INSIDE_SEMANTIC_ENTROPY_SIGNAL = "inside_semantic_entropy"
INSIDE_EMBEDDING_ENTROPY_SIGNAL = "inside_embedding_entropy"
INSIDE_SEMANTIC_ENERGY_SIGNAL = "inside_semantic_energy"
INSIDE_SIGNALS = (
    INSIDE_SIGNAL,
    INSIDE_SEMANTIC_ENTROPY_SIGNAL,
    INSIDE_EMBEDDING_ENTROPY_SIGNAL,
    INSIDE_SEMANTIC_ENERGY_SIGNAL,
)
NON_GEOMETRY_BASELINE_SIGNALS = {"nll_answer", FIRST_TOKEN_ENTROPY_SIGNAL}
REPORT_ALPHA = 0.10
HIDDEN_STATE_CAPTURE_METHODS = ("outputs", "hooks")
SCORE_DUMP_FORMATS = ("json", "jsonl")
SCORE_DUMP_RECORD_EXTRA_NAMES = (
    "batch_indexes",
    "inside_sampled",
    "inside_sample_counts",
    "inside_adaptive_rounds",
    "inside_stopped_early",
    "inside_stop_reasons",
    "inside_sample_texts",
    "inside_sample_logprobs",
)
PROFILE_GROUPS = {
    "startup": ("load_data", "load_model"),
    "tokenization": (
        "load_statement_encoding_cache",
        "tokenize_statements",
        "save_statement_encoding_cache",
    ),
    "model_forward": ("build_layer_stats", "forced_answer_forward", "inside_generation"),
    "cache_io": (
        "read_cache_metadata",
        "load_layer_stats_cache",
        "save_layer_stats_cache",
        "load_eval_reps_cache",
        "init_eval_reps_cache_writer",
        "read_eval_reps_cache_batch",
        "write_eval_reps_cache_batch",
        "save_eval_reps_cache",
        "load_inside_diagnostics_cache",
        "read_inside_diagnostics_cache",
        "write_inside_diagnostics_cache",
        "save_inside_diagnostics_cache",
    ),
    "postprocess": ("score_postprocess", "reporting", "sweep_reporting", "spectrum_reporting"),
}


@dataclass
class Statement:
    question: str
    answer: str
    is_false: int  # 1 = 错误答案(正类/幻觉) / incorrect (positive), 0 = 正确答案(负类)


@dataclass(frozen=True)
class StatementEncoding:
    input_ids: tuple[int, ...]
    n_answer_tokens: int

    def __post_init__(self) -> None:
        input_ids = tuple(int(token_id) for token_id in self.input_ids)
        n_answer_tokens = int(self.n_answer_tokens)
        if not input_ids:
            raise ValueError("statement encoding input_ids must not be empty.")
        if n_answer_tokens <= 0:
            raise ValueError("statement encoding n_answer_tokens must be positive.")
        if n_answer_tokens > len(input_ids):
            raise ValueError("statement encoding answer span exceeds input length.")
        object.__setattr__(self, "input_ids", input_ids)
        object.__setattr__(self, "n_answer_tokens", n_answer_tokens)

    def to_dict(self) -> dict[str, object]:
        return {
            "input_ids": list(self.input_ids),
            "n_answer_tokens": int(self.n_answer_tokens),
        }


@dataclass(frozen=True)
class SampledResponseDiagnostics:
    embeddings_by_layer: dict[int, torch.Tensor]
    sample_texts: tuple[str, ...]
    sample_logprobs: tuple[float, ...] = ()


@dataclass(frozen=True)
class SampledInsideDiagnostics:
    eigenscore_by_layer: dict[int, float]
    semantic_entropy: float
    embedding_entropy_by_layer: dict[int, float]
    semantic_energy: float = 0.0
    sample_texts: tuple[str, ...] = ()
    sample_logprobs: tuple[float, ...] = ()
    n_samples: int = 0
    adaptive_rounds: int = 1
    stopped_early: bool = False
    stop_reason: str | None = None


class InsideDiagnosticsCache:
    """JSON cache for sampled INSIDE diagnostics keyed by statement and sampling config."""

    def __init__(self, path: str | Path, *, refresh: bool = False) -> None:
        self.path = Path(path)
        if self.path.exists() and self.path.is_dir():
            raise ValueError("inside diagnostics cache path must be a JSON file, not a directory.")
        self.entries: dict[str, dict[str, object]] = {}
        self.hits = 0
        self.misses = 0
        self.writes = 0
        self._dirty = False
        if self.path.exists() and not refresh:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
                raise ValueError("inside diagnostics cache has an unsupported format.")
            entries = payload.get("entries", {})
            if not isinstance(entries, Mapping):
                raise ValueError("inside diagnostics cache entries must be an object.")
            self.entries = {str(key): dict(value) for key, value in entries.items() if isinstance(value, Mapping)}

    def get(self, key: str) -> SampledInsideDiagnostics | None:
        return self.get_any((key,))

    def get_any(self, keys: Sequence[str]) -> SampledInsideDiagnostics | None:
        for key in keys:
            record = self.entries.get(str(key))
            if record is None:
                continue
            diagnostics = _sampled_inside_diagnostics_from_cache_record(record)
            if diagnostics is None:
                continue
            self.hits += 1
            return diagnostics
        self.misses += 1
        return None

    def put(self, key: str, diagnostics: SampledInsideDiagnostics | None) -> None:
        if diagnostics is None:
            return
        self.entries[str(key)] = _sampled_inside_diagnostics_to_cache_record(diagnostics)
        self.writes += 1
        self._dirty = True

    def save(self) -> None:
        if not self._dirty and self.path.exists():
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "entries": self.entries,
            "stats": self.stats(),
        }
        tmp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        tmp_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        tmp_path.replace(self.path)
        self._dirty = False

    def stats(self) -> dict[str, object]:
        requests = self.hits + self.misses
        return {
            "path": str(self.path),
            "entries": len(self.entries),
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "requests": requests,
            "hit_rate": (self.hits / requests) if requests else None,
        }


@dataclass
class BatchSizeFallbackState:
    """Mutable request-local batch-size fallback state."""

    requested_size: int
    enabled: bool = False
    current_size: int | None = None
    reductions: list[dict[str, object]] = field(default_factory=list)

    def __post_init__(self) -> None:
        requested_size = int(self.requested_size)
        if requested_size < 1:
            raise ValueError("requested_size must be >= 1.")
        self.requested_size = requested_size
        if self.current_size is None:
            self.current_size = requested_size
        else:
            self.current_size = int(self.current_size)
        if self.current_size < 1:
            raise ValueError("current_size must be >= 1.")

    def batch_size(self) -> int:
        """Return the current effective batch size."""
        return max(1, int(self.current_size or self.requested_size))

    def reduce(self, *, phase: str, attempted_size: int, exc: BaseException) -> int:
        """Reduce effective batch size after a retriable memory error."""
        attempted_size = int(attempted_size)
        if not self.enabled or attempted_size <= 1:
            raise exc
        new_size = max(1, attempted_size // 2)
        self.current_size = min(self.batch_size(), new_size)
        event = {
            "phase": phase,
            "attempted_batch_size": attempted_size,
            "new_batch_size": self.batch_size(),
            "error_type": type(exc).__name__,
            "error": _compact_error_message(exc),
        }
        self.reductions.append(event)
        return self.batch_size()

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable fallback summary."""
        return {
            "enabled": bool(self.enabled),
            "requested_batch_size": int(self.requested_size),
            "effective_batch_size": self.batch_size(),
            "reductions": tuple(dict(item) for item in self.reductions),
            "n_reductions": len(self.reductions),
        }


def _selective_reports(
    scores: Mapping[str, Sequence[float]],
    labels: Sequence[int],
    *,
    alpha: float = REPORT_ALPHA,
    directions: Mapping[str, str] | None = None,
) -> dict[str, dict]:
    labels_t = torch.tensor(labels)
    reports = {}
    for signal, values in scores.items():
        direction = (directions or DEFAULT_SCORE_DIRECTIONS).get(signal, "higher")
        signal_scores = torch.tensor(values, dtype=torch.float64)
        true_scores = signal_scores[labels_t == 0]
        threshold = directional_conformal_threshold(true_scores, alpha, direction)
        reports[signal] = {
            "alpha": alpha,
            **selective_classification_report(signal_scores, labels_t, threshold, direction=direction),
        }
    return reports


def _write_score_dump(path: str | Path, dump: Mapping[str, object], dump_format: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if dump_format == "json":
        output_path.write_text(json.dumps(dump), encoding="utf-8")
        return
    if dump_format == "jsonl":
        write_score_dump_jsonl_mapping(
            dump,
            output_path,
            record_extra_names=tuple(name for name in SCORE_DUMP_RECORD_EXTRA_NAMES if name in dump),
        )
        return
    raise ValueError(f"unsupported score dump format: {dump_format!r}")


def _inside_enabled(args) -> bool:
    return int(getattr(args, "inside_samples", 0)) >= 2


def _inside_trigger_enabled(args) -> bool:
    return bool(getattr(args, "inside_trigger_signal", None))


def _enabled_signals(args) -> list[str]:
    signals = list(SIGNALS)
    if _inside_enabled(args):
        signals.extend(INSIDE_SIGNALS)
    return signals


def _sweep_signal_names(args) -> list[str]:
    signals = ["maha_last", "truth_proj", "subspace_resid", "resid_update_norm", "eigenscore"]
    if _inside_enabled(args):
        signals.extend(INSIDE_SIGNALS)
    return signals


def _sweep_output_enabled(args) -> bool:
    return bool(getattr(args, "sweep", False) or getattr(args, "sweep_layers", None))


def _parse_sweep_layers(value: str | None) -> list[int] | None:
    if value is None:
        return None
    layers = []
    for part in value.split(","):
        text = part.strip()
        if not text:
            continue
        try:
            layers.append(int(text))
        except ValueError as exc:
            raise ValueError("--sweep-layers must be a comma-separated list of integer layer indexes.") from exc
    if not layers:
        raise ValueError("--sweep-layers must contain at least one layer index.")
    return layers


def _resolve_sweep_layers(target_layer: int, n_layers: int, *, sweep: bool, sweep_layers: str | None) -> list[int]:
    requested = _parse_sweep_layers(sweep_layers)
    if requested is not None:
        layers = [target_layer, *requested]
    elif sweep:
        layers = [target_layer, *[-(i + 1) for i in range(n_layers)]]
    else:
        layers = [target_layer]

    resolved = []
    seen = set()
    for layer in layers:
        if not (-(n_layers + 1) <= layer <= n_layers):
            raise ValueError(
                f"sweep layer {layer} is out of range for a model with {n_layers} transformer layers "
                f"(valid hidden-state indexes: {-n_layers - 1}..{n_layers})."
            )
        if layer not in seen:
            seen.add(layer)
            resolved.append(layer)
    return resolved


def _normalize_hidden_state_index(layer: int, n_layers: int) -> int:
    """Map a hidden-state tuple index to its non-negative position."""
    n_hidden_states = int(n_layers) + 1
    normalized = int(layer)
    if normalized < 0:
        normalized = n_hidden_states + normalized
    if not (0 <= normalized < n_hidden_states):
        raise IndexError(
            f"hidden-state index {layer} is out of range for {n_layers} transformer layers."
        )
    return normalized


def _model_num_hidden_layers(model) -> int | None:
    config = getattr(model, "config", None)
    raw = getattr(config, "num_hidden_layers", None)
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _previous_hidden_state_index(layer: int, n_layers: int | None) -> int | None:
    if n_layers is None:
        if int(layer) <= 0:
            return None
        return int(layer) - 1
    normalized = _normalize_hidden_state_index(int(layer), int(n_layers))
    if normalized <= 0:
        return None
    return normalized - 1


def _residual_dynamics_capture_layers(
    layers: Sequence[int],
    *,
    n_layers: int | None,
    hidden_state_capture: str,
) -> list[int]:
    capture_layers = [int(layer) for layer in layers]
    seen = set(capture_layers)
    for layer in layers:
        previous = _previous_hidden_state_index(int(layer), n_layers)
        if previous is None:
            continue
        if hidden_state_capture == "hooks" and n_layers is not None and previous in {0, int(n_layers)}:
            continue
        if previous not in seen:
            capture_layers.append(previous)
            seen.add(previous)
    return capture_layers


def _residual_update_norm(
    hidden_by_layer: Mapping[int, torch.Tensor],
    layer: int,
    *,
    n_layers: int | None,
    row: int,
    token_index: int,
) -> float:
    previous = _previous_hidden_state_index(int(layer), n_layers)
    if previous is None or previous not in hidden_by_layer or int(layer) not in hidden_by_layer:
        return 0.0
    current_state = hidden_by_layer[int(layer)][row, token_index, :].float()
    previous_state = hidden_by_layer[previous][row, token_index, :].float()
    delta = current_state - previous_state
    if delta.numel() == 0:
        return 0.0
    return float(torch.linalg.vector_norm(delta).item() / math.sqrt(float(delta.numel())))


def _hook_capture_layer_map(model, layers: Sequence[int]) -> dict[int, int]:
    """Return requested hidden-state indexes mapped to transformer block indexes.

    HF decoder hidden-state tuples contain the embedding state at index 0 and,
    for common decoder-only models, the final hidden state after final norm at
    index ``num_hidden_layers``. A decoder block hook can exactly capture only
    intermediate post-block states, i.e. hidden-state indexes 1..N-1.
    """
    transformer_layers = TruthProbe._find_layers(model)
    n_layers = len(transformer_layers)
    mapped: dict[int, int] = {}
    unsupported = []
    for layer in layers:
        normalized = _normalize_hidden_state_index(int(layer), n_layers)
        if normalized == 0:
            unsupported.append((int(layer), "embedding hidden state"))
            continue
        if normalized == n_layers:
            unsupported.append((int(layer), "final post-norm hidden state"))
            continue
        mapped[int(layer)] = normalized - 1
    if unsupported:
        details = ", ".join(f"{layer} ({reason})" for layer, reason in unsupported)
        raise ValueError(
            "--hidden-state-capture=hooks cannot capture hidden-state index(es): "
            f"{details}. Use --hidden-state-capture=outputs for exact HF hidden_states semantics."
        )
    return mapped


def _extract_layer_hidden(output):
    hidden, _ = TruthProbe._unpack_output(output)
    return hidden


def _hidden_state_backbone_candidates(model) -> list:
    candidates = []
    seen_ids = {id(model)}
    get_decoder = getattr(model, "get_decoder", None)
    if callable(get_decoder):
        try:
            decoder = get_decoder()
        except (AttributeError, TypeError, NotImplementedError):
            decoder = None
        if decoder is not None and id(decoder) not in seen_ids:
            candidates.append(decoder)
            seen_ids.add(id(decoder))
    for attr in ("base_model", "model", "transformer", "gpt_neox", "decoder"):
        candidate = getattr(model, attr, None)
        if candidate is None or id(candidate) in seen_ids:
            continue
        candidates.append(candidate)
        seen_ids.add(id(candidate))
    return candidates


def _call_forward_for_hidden_states(
    module,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    output_hidden_states: bool,
):
    return module(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=output_hidden_states,
        use_cache=False,
    )


def _forward_for_hidden_states(
    model,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    output_hidden_states: bool,
    need_logits: bool,
):
    if not need_logits:
        for candidate in _hidden_state_backbone_candidates(model):
            try:
                out = _call_forward_for_hidden_states(
                    candidate,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=output_hidden_states,
                )
            except (AttributeError, TypeError, NotImplementedError):
                continue
            if not output_hidden_states or getattr(out, "hidden_states", None) is not None:
                return out, True
    return _call_forward_for_hidden_states(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=output_hidden_states,
    ), False


def _forward_with_selected_hidden_states(
    model,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layers: Sequence[int],
    hidden_state_capture: str,
    need_logits: bool = True,
):
    if hidden_state_capture not in HIDDEN_STATE_CAPTURE_METHODS:
        raise ValueError(f"hidden_state_capture must be one of {HIDDEN_STATE_CAPTURE_METHODS}.")

    if hidden_state_capture == "outputs":
        out, _used_backbone = _forward_for_hidden_states(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            need_logits=need_logits,
        )
        return out, {int(layer): out.hidden_states[int(layer)] for layer in layers}

    layer_to_module = _hook_capture_layer_map(model, layers)
    transformer_layers = TruthProbe._find_layers(model)
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for requested_layer, module_idx in layer_to_module.items():
        def _capture(_module, _input, output, *, layer=requested_layer):
            captured[layer] = _extract_layer_hidden(output)

        handles.append(transformer_layers[module_idx].register_forward_hook(_capture))

    try:
        out, used_backbone = _forward_for_hidden_states(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            need_logits=need_logits,
        )
    finally:
        for handle in handles:
            handle.remove()

    missing = [layer for layer in layers if int(layer) not in captured]
    if missing and used_backbone:
        captured = {}
        handles = []
        for requested_layer, module_idx in layer_to_module.items():
            def _capture(_module, _input, output, *, layer=requested_layer):
                captured[layer] = _extract_layer_hidden(output)

            handles.append(transformer_layers[module_idx].register_forward_hook(_capture))
        try:
            out, _used_backbone = _forward_for_hidden_states(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                need_logits=True,
            )
        finally:
            for handle in handles:
                handle.remove()
        missing = [layer for layer in layers if int(layer) not in captured]
    if missing:
        raise ValueError(f"hook hidden-state capture missed layer(s): {missing}.")
    return out, captured


def _candidate_verification_prompt(stmt: Statement) -> str:
    if stmt.question:
        return (
            f"Question: {stmt.question.strip()}\n"
            f"Candidate answer: {stmt.answer.strip()}\n"
            "Is the candidate answer factually correct? Answer yes or no and give a brief reason."
        )
    return (
        f"Statement: {stmt.answer.strip()}\n"
        "Is this statement factually correct? Answer yes or no and give a brief reason."
    )


def _chunked(items: Sequence, size: int):
    if size < 1:
        raise ValueError("batch size must be >= 1.")
    for start in range(0, len(items), size):
        yield items[start:start + size]


_MEMORY_ERROR_MARKERS = (
    "out of memory",
    "cuda error: out of memory",
    "cudnn_status_alloc_failed",
    "mps backend out of memory",
    "defaultcpuallocator",
    "memoryerror",
)


def _compact_error_message(exc: BaseException, *, max_chars: int = 180) -> str:
    message = " ".join(str(exc).split())
    if len(message) > max_chars:
        return message[: max_chars - 3].rstrip() + "..."
    return message


def _is_retriable_memory_error(exc: BaseException) -> bool:
    """Return whether an exception looks like a batch-size-related memory failure."""
    if not isinstance(exc, (RuntimeError, MemoryError)):
        return False
    message = str(exc).casefold()
    return any(marker in message for marker in _MEMORY_ERROR_MARKERS)


def _clear_device_cache(device: torch.device) -> None:
    """Release accelerator allocator caches after a memory failure when possible."""
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps"):
        empty_cache = getattr(torch.mps, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()


def _run_with_batch_size_fallback(
    items: Sequence,
    *,
    state: BatchSizeFallbackState | None,
    phase: str,
    runner,
    clear_cache=None,
) -> list:
    """Run a batch, recursively splitting it when an enabled memory fallback fires."""
    try:
        return list(runner(items))
    except (RuntimeError, MemoryError) as exc:
        if state is None or not state.enabled or len(items) <= 1 or not _is_retriable_memory_error(exc):
            raise
        next_size = state.reduce(phase=phase, attempted_size=len(items), exc=exc)
        if clear_cache is not None:
            clear_cache()
        outputs = []
        for chunk in _chunked(list(items), next_size):
            outputs.extend(_run_with_batch_size_fallback(
                chunk,
                state=state,
                phase=phase,
                runner=runner,
                clear_cache=clear_cache,
            ))
        return outputs


def _statement_length(stmt: Statement) -> int:
    return len(stmt.question) + len(stmt.answer)


def _statement_to_dump(stmt: Statement) -> dict[str, object]:
    text = stmt.answer.strip()
    if stmt.question.strip():
        text = f"{stmt.question.strip()} {text}".strip()
    return {
        "question": stmt.question,
        "answer": stmt.answer,
        "text": text,
        "is_false": int(stmt.is_false),
    }


def _batched_statements(statements: Sequence[Statement], size: int, *, length_bucketed: bool):
    if not length_bucketed:
        yield from _chunked(statements, size)
        return
    ordered = sorted(statements, key=_statement_length)
    yield from _chunked(ordered, size)


def _batched_statements_after_offset(
    statements: Sequence[Statement],
    size: int,
    *,
    length_bucketed: bool,
    offset: int,
):
    """Yield batches beginning after a processed statement offset."""
    cursor = 0
    offset = max(0, int(offset))
    for batch in _batched_statements(statements, size, length_bucketed=length_bucketed):
        end = cursor + len(batch)
        if end <= offset:
            cursor = end
            continue
        start = max(0, offset - cursor)
        remaining = batch[start:]
        if remaining:
            yield remaining
        cursor = end


def _inside_seed(base_seed: int, eval_batch_idx: int, inside_batch_idx: int) -> int:
    return int(base_seed) + int(eval_batch_idx) * 1_000_003 + int(inside_batch_idx)


def _inside_statement_seed(base_seed: int, stmt: Statement) -> int:
    payload = json.dumps(
        {
            "base_seed": int(base_seed),
            "question": stmt.question,
            "answer": stmt.answer,
            "is_false": int(stmt.is_false),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def _inside_statement_identity(stmt: Statement) -> dict[str, object]:
    return {
        "question": stmt.question,
        "answer": stmt.answer,
        "is_false": int(stmt.is_false),
    }


def _inside_statement_batch_digest(statements: Sequence[Statement]) -> str:
    payload = [_inside_statement_identity(stmt) for stmt in statements]
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _inside_statement_batch_seed(base_seed: int, statements: Sequence[Statement]) -> int:
    payload = {
        "base_seed": int(base_seed),
        "statements_sha256": _inside_statement_batch_digest(statements),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "big") % (2**63 - 1)


def _inside_batch_cache_context(
    base_seed: int,
    statements: Sequence[Statement],
    *,
    batch_position: int,
) -> dict[str, object]:
    return {
        "seed_scope": "statement_batch",
        "seed": _inside_statement_batch_seed(base_seed, statements),
        "batch_position": int(batch_position),
        "batch_size": len(statements),
        "statements_sha256": _inside_statement_batch_digest(statements),
    }


def _inside_diagnostics_cache_key(
    stmt: Statement,
    args,
    *,
    layers: Sequence[int],
    adaptive: bool,
    selfcheck_early_stop: bool,
    batch_cache_context: Mapping[str, object] | None = None,
) -> str:
    seed = _inside_statement_seed(int(args.seed), stmt)
    if batch_cache_context is not None:
        seed = int(batch_cache_context["seed"])
    payload = {
        "schema_version": 2 if batch_cache_context is not None else 1,
        "statement": _inside_statement_identity(stmt),
        "model": args.model,
        "dtype": getattr(args, "dtype", None),
        "layers": [int(layer) for layer in layers],
        "target_layer": int(args.layer),
        "max_length": int(args.max_length),
        "hidden_state_capture": args.hidden_state_capture,
        "seed": seed,
        "sampling": {
            "adaptive": bool(adaptive),
            "selfcheck_early_stop": bool(selfcheck_early_stop) if adaptive else False,
            "inside_samples": int(args.inside_samples),
            "inside_min_samples": int(getattr(args, "inside_min_samples", 2)) if adaptive else None,
            "inside_sample_step": int(getattr(args, "inside_sample_step", 1)) if adaptive else None,
            "inside_stability_delta": float(getattr(args, "inside_stability_delta", 0.05)) if adaptive else None,
            "inside_selfcheck_min_overlap": (
                float(getattr(args, "inside_selfcheck_min_overlap", 0.65))
                if adaptive and selfcheck_early_stop
                else None
            ),
            "inside_selfcheck_support_threshold": (
                float(getattr(args, "inside_selfcheck_support_threshold", 0.60))
                if adaptive and selfcheck_early_stop
                else None
            ),
            "inside_selfcheck_refute_threshold": (
                float(getattr(args, "inside_selfcheck_refute_threshold", 0.50))
                if adaptive and selfcheck_early_stop
                else None
            ),
            "inside_max_new_tokens": int(args.inside_max_new_tokens),
            "inside_temperature": float(args.inside_temperature),
            "inside_top_p": float(args.inside_top_p),
            "inside_pooling": args.inside_pooling,
            "inside_embedding_threshold": float(args.inside_embedding_threshold),
            "eigenscore_alpha": float(args.eigenscore_alpha),
        },
    }
    if batch_cache_context is not None:
        payload["cache_batch"] = dict(batch_cache_context)
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _sampled_inside_diagnostics_to_cache_record(diagnostics: SampledInsideDiagnostics) -> dict[str, object]:
    return {
        "eigenscore_by_layer": {
            str(layer): float(value) for layer, value in diagnostics.eigenscore_by_layer.items()
        },
        "semantic_entropy": float(diagnostics.semantic_entropy),
        "embedding_entropy_by_layer": {
            str(layer): float(value) for layer, value in diagnostics.embedding_entropy_by_layer.items()
        },
        "semantic_energy": float(diagnostics.semantic_energy),
        "sample_texts": list(diagnostics.sample_texts),
        "sample_logprobs": [float(value) for value in diagnostics.sample_logprobs],
        "n_samples": int(diagnostics.n_samples),
        "adaptive_rounds": int(diagnostics.adaptive_rounds),
        "stopped_early": bool(diagnostics.stopped_early),
        "stop_reason": diagnostics.stop_reason,
    }


def _sampled_inside_diagnostics_from_cache_record(record: Mapping[str, object]) -> SampledInsideDiagnostics | None:
    try:
        eigenscore_by_layer = {
            int(layer): float(value)
            for layer, value in dict(record.get("eigenscore_by_layer") or {}).items()
        }
        embedding_entropy_by_layer = {
            int(layer): float(value)
            for layer, value in dict(record.get("embedding_entropy_by_layer") or {}).items()
        }
        semantic_entropy = float(record.get("semantic_entropy"))
        sample_texts = tuple(str(text) for text in record.get("sample_texts") or ())
        sample_logprobs = tuple(float(value) for value in record.get("sample_logprobs") or ())
        semantic_energy = (
            float(record["semantic_energy"])
            if "semantic_energy" in record
            else float(
                lexical_semantic_energy(
                    sample_texts,
                    sample_logprobs=sample_logprobs or None,
                ).item()
            )
        )
        n_samples = int(record.get("n_samples"))
        adaptive_rounds = int(record.get("adaptive_rounds", 1))
        stopped_early = bool(record.get("stopped_early", False))
        stop_reason_raw = record.get("stop_reason")
        stop_reason = None if stop_reason_raw is None else str(stop_reason_raw)
    except (TypeError, ValueError):
        return None
    if not eigenscore_by_layer or not embedding_entropy_by_layer or n_samples < 2:
        return None
    if not math.isfinite(semantic_entropy) or not math.isfinite(semantic_energy):
        return None
    if sample_logprobs and (
        len(sample_logprobs) != n_samples
        or not all(math.isfinite(value) for value in sample_logprobs)
    ):
        return None
    return SampledInsideDiagnostics(
        eigenscore_by_layer=eigenscore_by_layer,
        semantic_entropy=semantic_entropy,
        embedding_entropy_by_layer=embedding_entropy_by_layer,
        semantic_energy=semantic_energy,
        sample_texts=sample_texts,
        sample_logprobs=sample_logprobs,
        n_samples=n_samples,
        adaptive_rounds=adaptive_rounds,
        stopped_early=stopped_early,
        stop_reason=stop_reason,
    )


def _profile_requested(args) -> bool:
    return bool(getattr(args, "profile", False) or getattr(args, "profile_json", None))


def _progress_report_due(completed: int, total: int, every: int, last_reported: int) -> bool:
    """Return whether a progress line should be printed for this counter state."""
    if every <= 0 or total <= 0:
        return False
    completed = min(int(completed), int(total))
    last_reported = min(int(last_reported), int(total))
    if completed <= last_reported:
        return False
    if completed >= total:
        return True
    next_mark = ((last_reported // every) + 1) * every
    return completed >= next_mark


def _format_progress(label: str, completed: int, total: int, elapsed_seconds: float) -> str:
    """Format a compact progress line for long benchmark phases."""
    completed = int(completed)
    total = int(total)
    elapsed_seconds = max(float(elapsed_seconds), 0.0)
    pct = (completed / total * 100.0) if total else 100.0
    rate = (completed / elapsed_seconds) if elapsed_seconds > 0 else 0.0
    return f"   {label}: {completed}/{total} ({pct:.1f}%) elapsed={elapsed_seconds:.1f}s rate={rate:.2f}/s"


def _inside_trigger_indexes(records: Sequence[Mapping], args) -> set[int]:
    """Return batch-local record indexes selected for sampled INSIDE scoring."""
    signal = getattr(args, "inside_trigger_signal", None)
    if not signal:
        return set(range(len(records)))
    if not records:
        return set()

    values = [float(record["primary_scores"][signal]) for record in records]
    direction = DEFAULT_SCORE_DIRECTIONS.get(signal, "higher")
    threshold = getattr(args, "inside_trigger_threshold", None)
    if threshold is not None:
        threshold = float(threshold)
        if direction == "higher":
            return {idx for idx, value in enumerate(values) if value > threshold}
        return {idx for idx, value in enumerate(values) if value < threshold}

    fraction = getattr(args, "inside_trigger_top_fraction", None)
    if fraction is None:
        return set(range(len(records)))
    k = max(1, math.ceil(len(records) * float(fraction)))
    reverse = direction == "higher"
    ranked = sorted(range(len(records)), key=lambda idx: values[idx], reverse=reverse)
    return set(ranked[:k])


def _empty_inside_scores(layers: Sequence[int]) -> dict[int, float]:
    return {int(layer): 0.0 for layer in layers}


def _layer_spectrum_reports(
    manifolds: Mapping[int, TruthManifold],
    *,
    top_k: int = 16,
) -> dict[str, dict[str, object]]:
    if top_k < 0:
        raise ValueError("top_k must be >= 0.")
    reports: dict[str, dict[str, object]] = {}
    for layer in sorted(manifolds):
        manifold = manifolds[layer]
        try:
            spectrum = manifold.spectrum()
        except (RuntimeError, ValueError) as exc:
            reports[str(layer)] = {
                "status": "unavailable",
                "error": str(exc),
                "sample_count": int(manifold.n),
                "hidden_dim": int(manifold.hidden_dim),
                "covariance_mode": manifold.covariance_mode,
            }
            continue
        payload = spectrum.to_dict(include_eigenvalues=False)
        limit = min(int(top_k), int(spectrum.eigenvalues.numel()))
        payload.update({
            "status": "ready",
            "covariance_mode": manifold.covariance_mode,
            "top_eigenvalues": [
                float(value) for value in spectrum.eigenvalues[:limit].detach().cpu().tolist()
            ],
            "top_eigenvalue_count": limit,
        })
        if manifold.covariance_mode == "shrinkage":
            payload["shrinkage_alpha"] = manifold.covariance_shrinkage_alpha()
        reports[str(layer)] = payload
    return reports


def _score_reps_batch(
    statements: Sequence[Statement],
    reps_batch: Sequence[Optional[Mapping]],
    *,
    layers: Sequence[int],
    target_layer: int,
    manifolds: Mapping[int, TruthManifold],
    subspaces: Mapping[int, TruthSubspace],
) -> list[dict]:
    valid = [(stmt, reps) for stmt, reps in zip(statements, reps_batch) if reps is not None]
    records = [
        {
            "stmt": stmt,
            "layer_scores": {},
            "primary_scores": {},
            "inside_scores": None,
            "inside_semantic_entropy": None,
            "inside_embedding_entropy": None,
            "inside_semantic_energy": None,
            "inside_sampled": False,
        }
        for stmt, _ in valid
    ]
    if not records:
        return records

    for layer in layers:
        manifold = manifolds[layer]
        states = torch.stack([
            reps["last"][layer].to(manifold.mean.device) for _, reps in valid
        ])
        maha_values = manifold.mahalanobis_distance(states).detach().cpu().tolist()
        if manifold.contrastive_direction is not None:
            proj_values = (-(states @ manifold.contrastive_direction.to(states.device))).detach().cpu().tolist()
        else:
            proj_values = [0.0] * len(records)

        subspace = subspaces.get(layer)
        if subspace is not None and subspace.is_ready():
            resid_values = subspace.residual_distance(states).detach().cpu().tolist()
        else:
            resid_values = [0.0] * len(records)

        resid_update_values = [
            float(dict(reps.get("resid_update_norm_by_layer") or {}).get(layer, 0.0))
            for _, reps in valid
        ]
        eigenscore_values = [float(reps["eigenscore_by_layer"][layer]) for _, reps in valid]
        for record, maha, proj, resid, resid_update, eigenscore in zip(
            records, maha_values, proj_values, resid_values, resid_update_values, eigenscore_values
        ):
            record["layer_scores"][layer] = {
                "maha_last": float(maha),
                "truth_proj": float(proj),
                "subspace_resid": float(resid),
                "resid_update_norm": float(resid_update),
                "eigenscore": float(eigenscore),
            }

    for record, (_, reps) in zip(records, valid):
        ans = reps["ans_hs"]
        record["primary_scores"] = {
            **record["layer_scores"][target_layer],
            "disp_euclid": float(euclidean_dispersion(ans).item()),
            "disp_hse": float(hyperbolic_semantic_entropy(poincare_map(ans)).item()),
            FIRST_TOKEN_ENTROPY_SIGNAL: float(reps.get("first_token_entropy", 0.0)),
            "nll_answer": float(reps["nll"]),
        }

    return records


@contextmanager
def _profile_phase(profile: dict[str, float], name: str) -> Iterator[None]:
    started = time.perf_counter()
    try:
        yield
    finally:
        profile[name] = profile.get(name, 0.0) + (time.perf_counter() - started)


def _profile_payload(
    profile: Mapping[str, float],
    total_seconds: float,
    *,
    n_eval_records: int | None = None,
    n_warmup_true: int | None = None,
    n_warmup_false: int | None = None,
    cache_stats: Mapping[str, Mapping[str, object]] | None = None,
) -> dict:
    phases = {name: round(float(seconds), 6) for name, seconds in profile.items()}
    return {
        "total_seconds": round(float(total_seconds), 6),
        "phases": phases,
        "summary": _profile_summary(
            phases,
            total_seconds,
            n_eval_records=n_eval_records,
            n_warmup_true=n_warmup_true,
            n_warmup_false=n_warmup_false,
            cache_stats=cache_stats,
        ),
    }


def _profile_summary(
    profile: Mapping[str, float],
    total_seconds: float,
    *,
    n_eval_records: int | None = None,
    n_warmup_true: int | None = None,
    n_warmup_false: int | None = None,
    cache_stats: Mapping[str, Mapping[str, object]] | None = None,
) -> dict:
    total = max(float(total_seconds), 0.0)
    phases = {name: max(float(seconds), 0.0) for name, seconds in profile.items()}
    phase_total = sum(phases.values())
    top_phases = [
        {
            "name": name,
            "seconds": round(seconds, 6),
            "share": _profile_share(seconds, total),
        }
        for name, seconds in sorted(phases.items(), key=lambda item: item[1], reverse=True)[:5]
    ]
    groups = {}
    for group, names in PROFILE_GROUPS.items():
        seconds = sum(phases.get(name, 0.0) for name in names)
        groups[group] = {
            "seconds": round(seconds, 6),
            "share": _profile_share(seconds, total),
        }

    throughput = {}
    eval_records = max(int(n_eval_records or 0), 0)
    warmup_records = max(int(n_warmup_true or 0), 0) + max(int(n_warmup_false or 0), 0)
    if eval_records and phases.get("forced_answer_forward", 0.0) > 0:
        throughput["forced_answer_records_per_second"] = round(
            eval_records / phases["forced_answer_forward"],
            6,
        )
    if warmup_records and phases.get("build_layer_stats", 0.0) > 0:
        throughput["warmup_records_per_second"] = round(
            warmup_records / phases["build_layer_stats"],
            6,
        )
    if eval_records and total > 0:
        throughput["end_to_end_eval_records_per_second"] = round(eval_records / total, 6)

    return {
        "bottleneck": top_phases[0]["name"] if top_phases else None,
        "top_phases": top_phases,
        "groups": groups,
        "throughput": throughput,
        "cache_efficiency": _cache_efficiency_summary(cache_stats),
        "accounted_seconds": round(phase_total, 6),
        "accounted_share": _profile_share(phase_total, total),
        "unaccounted_seconds": round(max(total - phase_total, 0.0), 6),
    }


def _profile_share(seconds: float, total_seconds: float) -> float:
    if total_seconds <= 0:
        return 0.0
    return round(float(seconds) / float(total_seconds), 6)


def _cache_efficiency_summary(cache_stats: Mapping[str, Mapping[str, object]] | None) -> dict:
    if not cache_stats:
        return {}
    reader_stats = cache_stats.get("eval_reps_reader")
    if not isinstance(reader_stats, Mapping):
        return {}

    def read_int(name: str) -> int:
        return max(int(reader_stats.get(name, 0) or 0), 0)

    read_requests = read_int("read_requests")
    records_read = read_int("records_read")
    shard_read_requests = read_int("shard_read_requests")
    cross_shard_reads = read_int("cross_shard_reads")
    shard_loads = read_int("shard_loads")
    shard_cache_hits = read_int("shard_cache_hits")
    shard_manifest_scans = read_int("shard_manifest_scans")
    shard_count = read_int("shard_count")
    shard_cache_capacity = read_int("shard_cache_capacity")

    summary = {
        "records_per_read": _profile_share(records_read, read_requests),
        "shards_per_read": _profile_share(shard_read_requests, read_requests),
        "cross_shard_read_rate": _profile_share(cross_shard_reads, read_requests),
        "shard_cache_hit_rate": _profile_share(shard_cache_hits, shard_read_requests),
        "shard_load_rate": _profile_share(shard_loads, shard_read_requests),
        "shard_manifest_scans_per_read": _profile_share(shard_manifest_scans, read_requests),
        "shard_count": shard_count,
        "shard_cache_capacity": shard_cache_capacity,
    }
    return {"eval_reps_reader": summary}


def _read_cache_metadata(path: str | Path) -> dict:
    cache_path = Path(path)
    if _is_sharded_eval_reps_cache(cache_path):
        manifest = _load_eval_reps_manifest(cache_path)
        return dict(manifest.get("metadata", {}))
    if cache_path.is_dir():
        raise ValueError(f"cache directory is missing manifest.json: {cache_path}")
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    return dict(cache.get("metadata", {}))


def _read_eval_reps_cache_metadata(path: str | Path) -> tuple[dict, int]:
    """Return eval-reps cache metadata plus the persisted record count."""
    cache_path = Path(path)
    if _is_sharded_eval_reps_cache(cache_path):
        manifest = _load_eval_reps_manifest(cache_path)
        record_count = int(manifest.get("record_count", -1))
        if record_count < 0:
            raise ValueError("sharded eval reps cache manifest is missing record_count.")
        return dict(manifest.get("metadata", {})), record_count
    if cache_path.is_dir():
        raise ValueError(f"cache directory is missing manifest.json: {cache_path}")
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    records = list(cache.get("records", []))
    return dict(cache.get("metadata", {})), len(records)


def _warmup_text_fingerprint(true_texts: Sequence[str], false_texts: Sequence[str]) -> str:
    payload = json.dumps(
        {"true": list(true_texts), "false": list(false_texts)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _statement_fingerprint(statements: Sequence[Statement]) -> str:
    payload = json.dumps(
        [
            {"question": stmt.question, "answer": stmt.answer, "is_false": int(stmt.is_false)}
            for stmt in statements
        ],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _statement_cache_payload(statements: Sequence[Statement]) -> list[dict[str, object]]:
    return [
        {
            "question": stmt.question,
            "answer": stmt.answer,
            "is_false": int(stmt.is_false),
        }
        for stmt in statements
    ]


def _eval_statements_from_cache_metadata(
    metadata: Mapping,
    *,
    expected_record_count: int | None = None,
) -> list[Statement] | None:
    raw_statements = metadata.get("eval_statements")
    if raw_statements is None:
        return None
    if not isinstance(raw_statements, list):
        raise ValueError("eval reps cache metadata eval_statements must be a list.")
    statements = []
    for idx, raw in enumerate(raw_statements):
        if not isinstance(raw, Mapping):
            raise ValueError(f"eval reps cache metadata statement {idx} must be an object.")
        is_false = int(raw.get("is_false", 0))
        if is_false not in {0, 1}:
            raise ValueError(f"eval reps cache metadata statement {idx} has invalid is_false={is_false!r}.")
        statements.append(
            Statement(
                question=str(raw.get("question", "")),
                answer=str(raw.get("answer", "")),
                is_false=is_false,
            )
        )
    if len(statements) != int(metadata.get("n_eval", len(statements))):
        raise ValueError("eval reps cache metadata eval_statements count does not match n_eval.")
    if expected_record_count is not None and len(statements) != int(expected_record_count):
        raise ValueError("eval reps cache metadata eval_statements count does not match record_count.")
    expected_fingerprint = metadata.get("eval_fingerprint")
    if expected_fingerprint and _statement_fingerprint(statements) != expected_fingerprint:
        raise ValueError("eval reps cache metadata eval_statements do not match eval_fingerprint.")
    return statements


def _validate_eval_reps_record_count(metadata: Mapping, record_count: int) -> None:
    """Fail early when eval-reps metadata and persisted cache rows disagree."""
    try:
        n_eval = int(metadata["n_eval"])
    except KeyError as exc:
        raise ValueError("eval reps cache metadata is missing n_eval.") from exc
    if n_eval != int(record_count):
        raise ValueError(
            "eval reps cache metadata n_eval does not match record_count "
            f"({n_eval} != {int(record_count)})."
        )


def _eval_reps_validation_metadata(metadata: Mapping) -> dict:
    """Return metadata fields that participate in cache compatibility checks."""
    payload = dict(metadata)
    payload.pop("eval_statements", None)
    return payload


def _layer_stats_cache_metadata(
    args,
    *,
    layers: Sequence[int],
    n_layers: int,
    true_texts: Sequence[str],
    false_texts: Sequence[str],
) -> dict:
    return {
        "format": 1,
        "model": args.model,
        "dtype": args.dtype,
        "offline": bool(args.offline),
        "n_layers": int(n_layers),
        "layers": [int(layer) for layer in layers],
        "max_length": int(args.max_length),
        "subspace_rank": int(args.subspace_rank),
        "covariance_mode": str(getattr(args, "covariance_mode", "full")),
        "covariance_low_rank": int(getattr(args, "covariance_low_rank", 16)),
        "length_bucketed_batches": bool(args.length_bucketed_batches),
        "n_true": len(true_texts),
        "n_false": len(false_texts),
        "warmup_fingerprint": _warmup_text_fingerprint(true_texts, false_texts),
    }


def _statement_encoding_cache_metadata(
    args,
    *,
    true_texts: Sequence[str],
    false_texts: Sequence[str],
    eval_statements: Sequence[Statement],
) -> dict:
    return {
        "format": 1,
        "model": args.model,
        "offline": bool(args.offline),
        "max_length": int(args.max_length),
        "n_true": len(true_texts),
        "n_false": len(false_texts),
        "n_eval": len(eval_statements),
        "warmup_fingerprint": _warmup_text_fingerprint(true_texts, false_texts),
        "eval_fingerprint": _statement_fingerprint(eval_statements),
    }


def _encoding_list_to_json(
    encodings: Sequence[Optional[StatementEncoding]],
) -> list[dict[str, object] | None]:
    return [None if encoding is None else encoding.to_dict() for encoding in encodings]


def _encoding_list_from_json(values: Sequence[object]) -> list[Optional[StatementEncoding]]:
    return [_coerce_statement_encoding(value) for value in values]


def save_statement_encoding_cache(
    path: str | Path,
    *,
    metadata: Mapping,
    true_encodings: Sequence[Optional[StatementEncoding]],
    false_encodings: Sequence[Optional[StatementEncoding]],
    eval_encodings: Sequence[Optional[StatementEncoding]],
) -> None:
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": dict(metadata),
        "true": _encoding_list_to_json(true_encodings),
        "false": _encoding_list_to_json(false_encodings),
        "eval": _encoding_list_to_json(eval_encodings),
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))


def load_statement_encoding_cache(
    path: str | Path,
    *,
    expected_metadata: Mapping,
) -> tuple[
    list[Optional[StatementEncoding]],
    list[Optional[StatementEncoding]],
    list[Optional[StatementEncoding]],
    dict,
]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    metadata = dict(payload.get("metadata", {}))
    _validate_cache_metadata(metadata, expected_metadata, cache_name="statement encoding cache")
    true_encodings = _encoding_list_from_json(payload.get("true", []))
    false_encodings = _encoding_list_from_json(payload.get("false", []))
    eval_encodings = _encoding_list_from_json(payload.get("eval", []))
    if len(true_encodings) != int(expected_metadata["n_true"]):
        raise ValueError("statement encoding cache true record count does not match this run.")
    if len(false_encodings) != int(expected_metadata["n_false"]):
        raise ValueError("statement encoding cache false record count does not match this run.")
    if len(eval_encodings) != int(expected_metadata["n_eval"]):
        raise ValueError("statement encoding cache eval record count does not match this run.")
    return true_encodings, false_encodings, eval_encodings, metadata


def _tensor_to_cpu(value):
    return value.detach().cpu() if value is not None else None


def _manifold_state(manifold: TruthManifold) -> dict:
    return {
        "mean": _tensor_to_cpu(manifold.mean),
        "_M2": _tensor_to_cpu(manifold._M2),  # noqa: SLF001 - benchmark cache mirrors core serialization.
        "_M2_diag": _tensor_to_cpu(manifold._M2_diag),  # noqa: SLF001 - benchmark cache mirrors core serialization.
        "n": int(manifold.n),
        "hidden_dim": int(manifold.hidden_dim),
        "ridge_lambda": float(manifold.ridge_lambda),
        "covariance_mode": manifold.covariance_mode,
        "covariance_low_rank": int(manifold.covariance_low_rank),
        "false_mean": _tensor_to_cpu(manifold.false_mean),
        "contrastive_direction": _tensor_to_cpu(manifold.contrastive_direction),
    }


def _manifold_from_state(state: Mapping, device: torch.device) -> TruthManifold:
    manifold = TruthManifold(
        covariance_mode=state.get("covariance_mode", "full"),
        covariance_low_rank=int(state.get("covariance_low_rank", 16)),
    )
    manifold.mean = state["mean"]
    manifold._M2 = state.get("_M2")  # noqa: SLF001 - benchmark cache mirrors core serialization.
    manifold._M2_diag = state.get("_M2_diag")  # noqa: SLF001 - benchmark cache mirrors core serialization.
    manifold.n = int(state["n"])
    manifold.hidden_dim = int(state["hidden_dim"])
    manifold.ridge_lambda = float(state.get("ridge_lambda", 0.1))
    manifold.false_mean = state.get("false_mean")
    manifold.contrastive_direction = state.get("contrastive_direction")
    manifold._dirty = True  # noqa: SLF001
    return manifold.to(device)


def _subspace_state(subspace: TruthSubspace) -> dict:
    return {
        "mean": _tensor_to_cpu(subspace.mean),
        "basis": _tensor_to_cpu(subspace.basis),
        "rank": int(subspace.rank),
        "false_mean": _tensor_to_cpu(subspace.false_mean),
        "contrastive_direction": _tensor_to_cpu(subspace.contrastive_direction),
    }


def _subspace_from_state(state: Mapping, device: torch.device) -> TruthSubspace:
    return TruthSubspace(
        mean=state.get("mean"),
        basis=state.get("basis"),
        rank=int(state.get("rank", 0)),
        false_mean=state.get("false_mean"),
        contrastive_direction=state.get("contrastive_direction"),
    ).to(device)


def _validate_cache_metadata(actual: Mapping, expected: Mapping, *, cache_name: str) -> None:
    mismatches = {
        key: {"expected": expected.get(key), "actual": actual.get(key)}
        for key in expected
        if actual.get(key) != expected.get(key)
    }
    if mismatches:
        details = ", ".join(
            f"{key}: expected {values['expected']!r}, got {values['actual']!r}"
            for key, values in mismatches.items()
        )
        raise ValueError(f"{cache_name} metadata does not match this run ({details}).")


def _tensor_list_state(values: Mapping[int, Sequence[torch.Tensor]]) -> dict[int, list[torch.Tensor]]:
    return {
        int(layer): [_tensor_to_cpu(value) for value in layer_values]
        for layer, layer_values in values.items()
    }


def _tensor_list_from_state(state: Mapping, device: torch.device) -> dict[int, list[torch.Tensor]]:
    return {
        int(layer): [value.to(device) for value in layer_values]
        for layer, layer_values in state.items()
    }


def _tensor_mapping_state(values: Mapping[int, torch.Tensor | None]) -> dict[int, torch.Tensor | None]:
    return {int(layer): _tensor_to_cpu(value) for layer, value in values.items()}


def _tensor_mapping_from_state(state: Mapping, device: torch.device) -> dict[int, torch.Tensor | None]:
    return {
        int(layer): None if value is None else value.to(device)
        for layer, value in state.items()
    }


def save_layer_stats_cache(
    path: str | Path,
    manifolds: Mapping[int, TruthManifold],
    subspaces: Mapping[int, TruthSubspace],
    *,
    metadata: Mapping,
) -> None:
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": dict(metadata),
            "manifolds": {int(layer): _manifold_state(manifold) for layer, manifold in manifolds.items()},
            "subspaces": {int(layer): _subspace_state(subspace) for layer, subspace in subspaces.items()},
        },
        cache_path,
    )


def load_layer_stats_cache(
    path: str | Path,
    *,
    expected_metadata: Mapping,
    device: torch.device,
) -> tuple[dict[int, TruthManifold], dict[int, TruthSubspace], dict]:
    cache = torch.load(Path(path), map_location=device, weights_only=True)
    metadata = dict(cache.get("metadata", {}))
    _validate_cache_metadata(metadata, expected_metadata, cache_name="layer stats cache")
    manifolds = {
        int(layer): _manifold_from_state(state, device)
        for layer, state in cache.get("manifolds", {}).items()
    }
    subspaces = {
        int(layer): _subspace_from_state(state, device)
        for layer, state in cache.get("subspaces", {}).items()
    }
    missing = set(expected_metadata["layers"]) - set(manifolds)
    if missing:
        raise ValueError(f"layer stats cache is missing manifold(s) for layer(s): {sorted(missing)}.")
    return manifolds, subspaces, metadata


def save_warmup_checkpoint(
    path: str | Path,
    *,
    metadata: Mapping,
    manifolds: Mapping[int, TruthManifold],
    true_state_lists: Mapping[int, Sequence[torch.Tensor]],
    false_state_lists: Mapping[int, Sequence[torch.Tensor]],
    false_sums: Mapping[int, torch.Tensor | None],
    n_false: int,
    true_done: int,
    false_done: int,
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": 1,
            "metadata": dict(metadata),
            "progress": {"true_done": int(true_done), "false_done": int(false_done)},
            "manifolds": {int(layer): _manifold_state(manifold) for layer, manifold in manifolds.items()},
            "true_state_lists": _tensor_list_state(true_state_lists),
            "false_state_lists": _tensor_list_state(false_state_lists),
            "false_sums": _tensor_mapping_state(false_sums),
            "n_false": int(n_false),
        },
        checkpoint_path,
    )


def load_warmup_checkpoint(
    path: str | Path,
    *,
    expected_metadata: Mapping,
    device: torch.device,
) -> dict:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    if int(checkpoint.get("format", 0)) != 1:
        raise ValueError("warmup checkpoint has an unsupported format.")
    metadata = dict(checkpoint.get("metadata", {}))
    _validate_cache_metadata(metadata, expected_metadata, cache_name="warmup checkpoint")

    expected_layers = {int(layer) for layer in expected_metadata["layers"]}
    manifolds = {
        int(layer): _manifold_from_state(state, device)
        for layer, state in checkpoint.get("manifolds", {}).items()
    }
    true_state_lists = _tensor_list_from_state(checkpoint.get("true_state_lists", {}), device)
    false_state_lists = _tensor_list_from_state(checkpoint.get("false_state_lists", {}), device)
    false_sums = _tensor_mapping_from_state(checkpoint.get("false_sums", {}), device)
    missing = expected_layers - set(manifolds)
    if missing:
        raise ValueError(f"warmup checkpoint is missing manifold(s) for layer(s): {sorted(missing)}.")
    for layer in expected_layers:
        true_state_lists.setdefault(layer, [])
        false_state_lists.setdefault(layer, [])
        false_sums.setdefault(layer, None)

    progress = dict(checkpoint.get("progress", {}))
    true_done = int(progress.get("true_done", 0))
    false_done = int(progress.get("false_done", 0))
    n_true = int(expected_metadata.get("n_true", 0))
    n_false_expected = int(expected_metadata.get("n_false", 0))
    if not (0 <= true_done <= n_true and 0 <= false_done <= n_false_expected):
        raise ValueError("warmup checkpoint progress is out of bounds for this run.")
    if false_done > 0 and true_done < n_true:
        raise ValueError("warmup checkpoint cannot resume false statements before true warmup is complete.")

    return {
        "metadata": metadata,
        "manifolds": manifolds,
        "true_state_lists": true_state_lists,
        "false_state_lists": false_state_lists,
        "false_sums": false_sums,
        "n_false": int(checkpoint.get("n_false", 0)),
        "true_done": true_done,
        "false_done": false_done,
    }


def _eval_reps_cache_metadata(
    args,
    *,
    layers: Sequence[int],
    n_layers: int,
    eval_statements: Sequence[Statement],
) -> dict:
    return {
        "format": 1,
        "model": args.model,
        "dtype": args.dtype,
        "offline": bool(args.offline),
        "n_layers": int(n_layers),
        "layers": [int(layer) for layer in layers],
        "max_length": int(args.max_length),
        "eigenscore_alpha": float(args.eigenscore_alpha),
        "first_token_entropy_schema": 1,
        "first_token_top_k": int(getattr(args, "first_token_top_k", FIRST_TOKEN_TOP_K_DEFAULT)),
        "resid_update_norm_schema": 1,
        "length_bucketed_batches": bool(args.length_bucketed_batches),
        "n_eval": len(eval_statements),
        "eval_fingerprint": _statement_fingerprint(eval_statements),
        "eval_statements": _statement_cache_payload(eval_statements),
    }


def _validate_cache_only_metadata(
    *,
    args,
    stats_metadata: Mapping,
    eval_reps_metadata: Mapping,
    layers: Sequence[int],
    n_layers: int,
) -> None:
    base_expected = {
        "model": args.model,
        "dtype": args.dtype,
        "offline": bool(args.offline),
        "n_layers": int(n_layers),
        "layers": [int(layer) for layer in layers],
        "max_length": int(args.max_length),
        "length_bucketed_batches": bool(args.length_bucketed_batches),
    }
    _validate_cache_metadata(
        stats_metadata,
        {
            **base_expected,
            "subspace_rank": int(args.subspace_rank),
            "covariance_mode": str(getattr(args, "covariance_mode", "full")),
            "covariance_low_rank": int(getattr(args, "covariance_low_rank", 16)),
        },
        cache_name="layer stats cache",
    )
    _validate_cache_metadata(
        _eval_reps_validation_metadata(eval_reps_metadata),
        {
            **base_expected,
            "eigenscore_alpha": float(args.eigenscore_alpha),
            "first_token_entropy_schema": 1,
            "first_token_top_k": int(getattr(args, "first_token_top_k", FIRST_TOKEN_TOP_K_DEFAULT)),
        },
        cache_name="eval reps cache",
    )


def _reps_to_cache_state(reps: Optional[Mapping]) -> Optional[dict]:
    if reps is None:
        return None
    return {
        "last": {int(layer): _tensor_to_cpu(value) for layer, value in reps["last"].items()},
        "ans_hs": _tensor_to_cpu(reps["ans_hs"]),
        "eigenscore_by_layer": {
            int(layer): float(value) for layer, value in reps["eigenscore_by_layer"].items()
        },
        "resid_update_norm_by_layer": {
            int(layer): float(value)
            for layer, value in dict(reps.get("resid_update_norm_by_layer") or {}).items()
        },
        "first_token_entropy": float(reps.get("first_token_entropy", 0.0)),
        "nll": float(reps["nll"]),
    }


def _reps_from_cache_state(state: Optional[Mapping]) -> Optional[dict]:
    if state is None:
        return None
    return {
        "last": {int(layer): value for layer, value in state["last"].items()},
        "ans_hs": state["ans_hs"],
        "eigenscore_by_layer": {
            int(layer): float(value) for layer, value in state["eigenscore_by_layer"].items()
        },
        "resid_update_norm_by_layer": {
            int(layer): float(value)
            for layer, value in dict(state.get("resid_update_norm_by_layer") or {}).items()
        },
        "first_token_entropy": float(state.get("first_token_entropy", 0.0)),
        "nll": float(state["nll"]),
    }


def _eval_reps_manifest_path(path: str | Path) -> Path:
    return Path(path) / "manifest.json"


def _is_sharded_eval_reps_cache(path: str | Path) -> bool:
    return Path(path).is_dir() and _eval_reps_manifest_path(path).exists()


def _load_eval_reps_manifest(path: str | Path) -> dict:
    with open(_eval_reps_manifest_path(path), encoding="utf-8") as f:
        return json.load(f)


def _existing_eval_reps_shard_size(path: str | Path) -> int:
    if not _is_sharded_eval_reps_cache(path):
        return 0
    return int(_load_eval_reps_manifest(path).get("shard_size", 0))


def _validate_eval_reps_manifest(manifest: Mapping, expected_records: int) -> None:
    if int(manifest.get("format", 0)) != 2:
        raise ValueError("sharded eval reps cache manifest has an unsupported format.")
    if int(manifest.get("record_count", -1)) != int(expected_records):
        raise ValueError(
            f"eval reps cache record count does not match this run "
            f"(expected {expected_records}, got {manifest.get('record_count')})."
        )
    offset = 0
    for shard in manifest.get("shards", []):
        start = int(shard.get("start", -1))
        count = int(shard.get("count", -1))
        if start != offset or count < 0:
            raise ValueError("sharded eval reps cache manifest has non-contiguous shard ranges.")
        offset += count
    if offset != int(expected_records):
        raise ValueError("sharded eval reps cache manifest shard counts do not match record_count.")


class EvalRepsCacheReader:
    def __init__(
        self,
        path: str | Path,
        *,
        expected_metadata: Mapping,
        expected_records: int,
        shard_read_cache_size: int = 2,
    ) -> None:
        self.path = Path(path)
        self.metadata: dict
        self.record_count = int(expected_records)
        self._records: list[Optional[dict]] | None = None
        self._shards: list[dict] = []
        self._shard_starts: list[int] = []
        self._shard_cache_size = int(shard_read_cache_size)
        if self._shard_cache_size < 1:
            raise ValueError("eval reps shard read cache size must be >=1.")
        self._shard_cache: OrderedDict[tuple[str, int], list[object]] = OrderedDict()
        self._shard_loads = 0
        self._shard_cache_hits = 0
        self._shard_manifest_scans = 0
        self._read_requests = 0
        self._records_read = 0
        self._shard_read_requests = 0
        self._cross_shard_reads = 0
        if _is_sharded_eval_reps_cache(self.path):
            manifest = _load_eval_reps_manifest(self.path)
            self.metadata = dict(manifest.get("metadata", {}))
            _validate_cache_metadata(
                _eval_reps_validation_metadata(self.metadata),
                _eval_reps_validation_metadata(expected_metadata),
                cache_name="eval reps cache",
            )
            _validate_eval_reps_manifest(manifest, expected_records)
            self.record_count = int(manifest["record_count"])
            self._shards = [dict(shard) for shard in manifest.get("shards", [])]
            self._shard_starts = [int(shard["start"]) for shard in self._shards]
        else:
            if self.path.is_dir():
                raise ValueError(f"eval reps cache directory is missing manifest.json: {self.path}")
            cache = torch.load(self.path, map_location="cpu", weights_only=True)
            self.metadata = dict(cache.get("metadata", {}))
            _validate_cache_metadata(
                _eval_reps_validation_metadata(self.metadata),
                _eval_reps_validation_metadata(expected_metadata),
                cache_name="eval reps cache",
            )
            self._records = [_reps_from_cache_state(record) for record in cache.get("records", [])]
            if len(self._records) != int(expected_records):
                raise ValueError(
                    f"eval reps cache record count does not match this run "
                    f"(expected {expected_records}, got {len(self._records)})."
                )
            self.record_count = len(self._records)

    def read_range(self, start: int, count: int) -> list[Optional[dict]]:
        start = int(start)
        count = int(count)
        if start < 0 or count < 0 or start + count > self.record_count:
            raise ValueError("eval reps cache read range is out of bounds.")
        if count == 0:
            return []
        self._read_requests += 1
        self._records_read += count
        if self._records is not None:
            return self._records[start:start + count]

        end = start + count
        records: list[Optional[dict]] = []
        touched_shards = 0
        shard_index = self._first_shard_index_for_range(start)
        while shard_index < len(self._shards):
            shard = self._shards[shard_index]
            shard_start = int(shard["start"])
            shard_count = int(shard["count"])
            shard_end = shard_start + shard_count
            if shard_start >= end:
                break
            self._shard_manifest_scans += 1
            raw_records = self._load_shard_records(shard)
            local_start = max(start, shard_start) - shard_start
            local_end = min(end, shard_end) - shard_start
            records.extend(_reps_from_cache_state(record) for record in raw_records[local_start:local_end])
            touched_shards += 1
            shard_index += 1
        self._shard_read_requests += touched_shards
        if touched_shards > 1:
            self._cross_shard_reads += 1
        if len(records) != count:
            raise ValueError(f"eval reps cache returned {len(records)} records for a {count}-record range.")
        return records

    def read_all(self) -> list[Optional[dict]]:
        return self.read_range(0, self.record_count)

    def cache_stats(self) -> dict[str, int]:
        """Return reader-local cache IO counters."""
        return {
            "record_count": int(self.record_count),
            "read_requests": int(self._read_requests),
            "records_read": int(self._records_read),
            "shard_count": len(self._shards),
            "shard_cache_capacity": int(self._shard_cache_size) if self._shards else 0,
            "shard_cache_entries": len(self._shard_cache),
            "shard_read_requests": int(self._shard_read_requests),
            "cross_shard_reads": int(self._cross_shard_reads),
            "shard_loads": int(self._shard_loads),
            "shard_cache_hits": int(self._shard_cache_hits),
            "shard_manifest_scans": int(self._shard_manifest_scans),
        }

    def _first_shard_index_for_range(self, start: int) -> int:
        if not self._shard_starts:
            return 0
        return max(0, bisect.bisect_right(self._shard_starts, int(start)) - 1)

    def _load_shard_records(self, shard: Mapping[str, object]) -> list[object]:
        shard_path = str(shard["path"])
        shard_start = int(shard["start"])
        shard_count = int(shard["count"])
        shard_key = (shard_path, shard_start)
        if shard_key in self._shard_cache:
            self._shard_cache.move_to_end(shard_key)
            self._shard_cache_hits += 1
            return self._shard_cache[shard_key]

        shard_payload = torch.load(self.path / shard_path, map_location="cpu", weights_only=True)
        raw_records = list(shard_payload.get("records", []))
        if int(shard_payload.get("start", -1)) != shard_start or len(raw_records) != shard_count:
            raise ValueError("sharded eval reps cache shard payload does not match manifest.")
        self._shard_cache[shard_key] = raw_records
        self._shard_cache.move_to_end(shard_key)
        while len(self._shard_cache) > self._shard_cache_size:
            self._shard_cache.popitem(last=False)
        self._shard_loads += 1
        return raw_records


class EvalRepsCacheWriter:
    def __init__(self, path: str | Path, *, metadata: Mapping, shard_size: int) -> None:
        self.path = Path(path)
        self.metadata = dict(metadata)
        self.shard_size = int(shard_size)
        if self.shard_size < 1:
            raise ValueError("eval reps cache shard size must be >=1.")
        if self.path.exists() and not self.path.is_dir():
            raise ValueError("sharded eval reps cache path must be a directory path.")
        self.path.mkdir(parents=True, exist_ok=True)
        manifest_path = _eval_reps_manifest_path(self.path)
        if manifest_path.exists():
            manifest_path.unlink()
        for stale in self.path.glob("records-*.pt"):
            stale.unlink()
        self._buffer: list[Optional[dict]] = []
        self._record_count = 0
        self._shards: list[dict] = []

    def extend(self, reps_records: Sequence[Optional[Mapping]]) -> None:
        self._buffer.extend(_reps_to_cache_state(reps) for reps in reps_records)
        while len(self._buffer) >= self.shard_size:
            self._flush(self.shard_size)

    def close(self) -> None:
        self._flush(len(self._buffer))
        manifest = {
            "format": 2,
            "metadata": self.metadata,
            "record_count": self._record_count,
            "shard_size": self.shard_size,
            "shards": self._shards,
        }
        with open(_eval_reps_manifest_path(self.path), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)

    def _flush(self, count: int) -> None:
        if count <= 0:
            return
        records = self._buffer[:count]
        del self._buffer[:count]
        shard_name = f"records-{len(self._shards):06d}.pt"
        shard_payload = {
            "start": self._record_count,
            "records": records,
        }
        torch.save(shard_payload, self.path / shard_name)
        self._shards.append({
            "path": shard_name,
            "start": self._record_count,
            "count": len(records),
        })
        self._record_count += len(records)


def save_eval_reps_cache(
    path: str | Path,
    reps_records: Sequence[Optional[Mapping]],
    *,
    metadata: Mapping,
    shard_size: int = 0,
) -> None:
    if int(shard_size) > 0:
        writer = EvalRepsCacheWriter(path, metadata=metadata, shard_size=int(shard_size))
        writer.extend(reps_records)
        writer.close()
        return
    cache_path = Path(path)
    if cache_path.exists() and cache_path.is_dir():
        raise ValueError("single-file eval reps cache path must not be an existing directory.")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": dict(metadata),
            "records": [_reps_to_cache_state(reps) for reps in reps_records],
        },
        cache_path,
    )


def load_eval_reps_cache(
    path: str | Path,
    *,
    expected_metadata: Mapping,
    expected_records: int,
) -> tuple[list[Optional[dict]], dict]:
    reader = EvalRepsCacheReader(path, expected_metadata=expected_metadata, expected_records=expected_records)
    return reader.read_all(), reader.metadata


# ---------------------------------------------------------------------------
# 离线烟雾集（仅用于验证管线，不构成基准）/ Offline smoke set (pipeline check only)
# ---------------------------------------------------------------------------

_TRUE_SMOKE = [
    "The capital of France is Paris.",
    "Water boils at 100 degrees Celsius at sea level.",
    "The Earth revolves around the Sun.",
    "Humans have two lungs.",
    "The Pacific is the largest ocean on Earth.",
    "Ice is frozen water.",
    "A triangle has three sides.",
    "The Sun rises in the east.",
    "Honey is made by bees.",
    "Mount Everest is the tallest mountain above sea level.",
]
_FALSE_SMOKE = [
    "The capital of France is Berlin.",
    "Water boils at 30 degrees Celsius at sea level.",
    "The Sun revolves around the Earth.",
    "Humans have five lungs.",
    "The Pacific is the smallest ocean on Earth.",
    "Ice is boiling water.",
    "A triangle has seven sides.",
    "The Sun rises in the west.",
    "Honey is made by spiders.",
    "Mount Everest is the shortest mountain on Earth.",
]


def load_offline() -> tuple[List[str], List[str], List[Statement]]:
    """返回 (流形构建用真陈述, 对比方向用假陈述, 评测陈述)。"""
    manifold_true = _TRUE_SMOKE[:6]
    manifold_false = _FALSE_SMOKE[:6]
    eval_stmts: List[Statement] = []
    for t in _TRUE_SMOKE[6:]:
        eval_stmts.append(Statement("", t, 0))
    for f in _FALSE_SMOKE[6:]:
        eval_stmts.append(Statement("", f, 1))
    return manifold_true, manifold_false, eval_stmts


def load_truthfulqa(
    manifold_questions: int, limit: int
) -> tuple[List[str], List[str], List[Statement]]:
    """加载 TruthfulQA multiple_choice，切分为流形集 / 评测集（题目层面不重叠）。

    流形集题目的正确答案用于构建真值流形，错误答案仅用于 mass-mean 对比方向。
    Correct answers of manifold-split questions build the truth manifold; their
    incorrect answers are used only for the mass-mean contrastive direction.
    """
    from datasets import load_dataset  # lazy

    # 新旧版 datasets 的数据集 id 不同，依次尝试 / dataset id differs across versions
    last_err: Optional[Exception] = None
    ds = None
    for dataset_id in ("truthfulqa/truthful_qa", "truthful_qa"):
        try:
            ds = load_dataset(dataset_id, "multiple_choice")["validation"]
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    if ds is None:
        raise RuntimeError(
            f"Could not load TruthfulQA. Last error: {last_err}. "
            f"Try `pip install -U datasets` or run with --offline for a pipeline check."
        )

    manifold_true: List[str] = []
    manifold_false: List[str] = []
    eval_stmts: List[Statement] = []
    for i, row in enumerate(ds):
        q = row["question"]
        targets = row["mc2_targets"]
        choices, labels = targets["choices"], targets["labels"]
        if i < manifold_questions:
            for c, lab in zip(choices, labels):
                if lab == 1:
                    manifold_true.append(f"{q} {c}")
                else:
                    manifold_false.append(f"{q} {c}")
        else:
            for c, lab in zip(choices, labels):
                eval_stmts.append(Statement(q, c, is_false=int(lab == 0)))
        if limit and (i - manifold_questions + 1) >= limit and i >= manifold_questions:
            break
    return manifold_true, manifold_false, eval_stmts


# ---------------------------------------------------------------------------
# 模型与表征提取 / Model and representation extraction
# ---------------------------------------------------------------------------

_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def load_model(model_name: str, device: torch.device, dtype: str = "float32"):
    from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # dtype= 替代已弃用的 torch_dtype=；bfloat16 减半权重内存；
    # low_cpu_mem_usage 避免加载时 2× 峰值内存（低内存机器关键）
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=_DTYPES[dtype], low_cpu_mem_usage=True
    )
    model.to(device).eval()
    return model, tokenizer


def resolve_target_layer(layer: int, n_layers: int, *, offline: bool) -> int:
    """Return a hidden-state layer index valid for the loaded model."""
    if -(n_layers + 1) <= layer <= n_layers:
        return layer
    if offline and layer == -8:
        print(f"[!] Requested layer {layer} is unavailable for this tiny smoke model; using -1.\n")
        return -1
    raise ValueError(
        f"layer {layer} is out of range for a model with {n_layers} transformer layers "
        f"(valid hidden-state indexes: {-n_layers - 1}..{n_layers})."
    )


def _statement_token_ids(tokenizer, stmt: Statement, max_length: int) -> Optional[tuple[list[int], int]]:
    encoding = _statement_encoding(tokenizer, stmt, max_length)
    if encoding is None:
        return None
    return list(encoding.input_ids), int(encoding.n_answer_tokens)


def _statement_encoding(tokenizer, stmt: Statement, max_length: int) -> Optional[StatementEncoding]:
    q_ids = tokenizer(stmt.question, add_special_tokens=True).input_ids if stmt.question \
        else tokenizer(tokenizer.bos_token or tokenizer.eos_token or " ").input_ids
    a_ids = tokenizer(" " + stmt.answer.strip(), add_special_tokens=False).input_ids
    if len(a_ids) == 0:
        return None
    ids = (q_ids + a_ids)[:max_length]
    n_ans = len(ids) - len(q_ids)
    if n_ans <= 0:
        return None
    return StatementEncoding(tuple(ids), n_ans)


def _input_ids_from_tokenizer_output(output) -> list[list[int]]:
    input_ids = output["input_ids"] if isinstance(output, Mapping) else output.input_ids
    if isinstance(input_ids, torch.Tensor):
        return input_ids.detach().cpu().tolist()
    return [list(ids) for ids in input_ids]


def _batch_statement_token_ids(
    tokenizer,
    statements: Sequence[Statement],
    max_length: int,
) -> list[Optional[tuple[list[int], int]]]:
    return [
        None if encoding is None else (list(encoding.input_ids), int(encoding.n_answer_tokens))
        for encoding in _batch_statement_encodings(tokenizer, statements, max_length)
    ]


def _batch_statement_encodings(
    tokenizer,
    statements: Sequence[Statement],
    max_length: int,
) -> list[Optional[StatementEncoding]]:
    if not statements:
        return []

    encoded: list[Optional[StatementEncoding]] = [None] * len(statements)
    fallback_prompt = tokenizer.bos_token or tokenizer.eos_token or " "

    question_positions = [idx for idx, stmt in enumerate(statements) if stmt.question]
    if question_positions:
        question_output = tokenizer(
            [statements[idx].question for idx in question_positions],
            add_special_tokens=True,
        )
        question_ids_by_position = dict(zip(
            question_positions,
            _input_ids_from_tokenizer_output(question_output),
        ))
    else:
        question_ids_by_position = {}

    fallback_q_ids = tokenizer(fallback_prompt).input_ids if len(question_positions) < len(statements) else []

    answer_output = tokenizer(
        [" " + stmt.answer.strip() for stmt in statements],
        add_special_tokens=False,
    )
    answer_ids_batch = _input_ids_from_tokenizer_output(answer_output)

    for idx, (stmt, answer_ids) in enumerate(zip(statements, answer_ids_batch)):
        if len(answer_ids) == 0:
            continue
        question_ids = question_ids_by_position.get(idx, fallback_q_ids)
        ids = (list(question_ids) + list(answer_ids))[:max_length]
        n_ans = len(ids) - len(question_ids)
        if n_ans <= 0:
            continue
        encoded[idx] = StatementEncoding(tuple(ids), n_ans)
    return encoded


def _coerce_statement_encoding(value) -> Optional[StatementEncoding]:
    if value is None:
        return None
    if isinstance(value, StatementEncoding):
        return value
    if isinstance(value, Mapping):
        return StatementEncoding(value["input_ids"], int(value["n_answer_tokens"]))
    ids, n_ans = value
    return StatementEncoding(ids, int(n_ans))


def _encoding_to_token_ids(value) -> Optional[tuple[list[int], int]]:
    encoding = _coerce_statement_encoding(value)
    if encoding is None:
        return None
    return list(encoding.input_ids), int(encoding.n_answer_tokens)


def _batched_statement_pairs(
    statements: Sequence[Statement],
    encodings: Sequence[Optional[StatementEncoding]] | None,
    size: int,
    *,
    length_bucketed: bool,
    max_batch_tokens: int = 0,
):
    if size < 1:
        raise ValueError("batch size must be >= 1.")
    pairs = _statement_pairs(statements, encodings, length_bucketed=length_bucketed)
    yield from _statement_pair_batches(pairs, size, max_batch_tokens=max_batch_tokens)


def _statement_pairs(
    statements: Sequence[Statement],
    encodings: Sequence[Optional[StatementEncoding]] | None,
    *,
    length_bucketed: bool,
) -> list[tuple[Statement, Optional[StatementEncoding]]]:
    if encodings is not None and len(encodings) != len(statements):
        raise ValueError("statement encodings must have the same length as statements.")
    if encodings is None:
        pairs = [(stmt, None) for stmt in statements]
    else:
        pairs = list(zip(statements, encodings))
    if length_bucketed:
        pairs = sorted(pairs, key=lambda pair: _statement_length(pair[0]))
    return pairs


def _batched_statement_pairs_after_offset(
    statements: Sequence[Statement],
    encodings: Sequence[Optional[StatementEncoding]] | None,
    size: int,
    *,
    length_bucketed: bool,
    offset: int,
    max_batch_tokens: int = 0,
):
    cursor = 0
    offset = max(0, int(offset))
    for batch in _batched_statement_pairs(
        statements,
        encodings,
        size,
        length_bucketed=length_bucketed,
        max_batch_tokens=max_batch_tokens,
    ):
        end = cursor + len(batch)
        if end <= offset:
            cursor = end
            continue
        start = max(0, offset - cursor)
        remaining = batch[start:]
        if remaining:
            yield remaining
        cursor = end


def _statement_pair_batches(
    pairs: Sequence[tuple[Statement, Optional[StatementEncoding]]],
    size: int,
    *,
    max_batch_tokens: int = 0,
):
    offset = 0
    while offset < len(pairs):
        batch = _next_statement_pair_batch(
            pairs,
            offset,
            size,
            max_batch_tokens=max_batch_tokens,
        )
        if not batch:
            break
        yield batch
        offset += len(batch)


def _next_statement_pair_batch(
    pairs: Sequence[tuple[Statement, Optional[StatementEncoding]]],
    offset: int,
    size: int,
    *,
    max_batch_tokens: int = 0,
) -> list[tuple[Statement, Optional[StatementEncoding]]]:
    """Return the next batch under row-count and padded-token budgets."""
    if size < 1:
        raise ValueError("batch size must be >= 1.")
    offset = max(0, int(offset))
    if offset >= len(pairs):
        return []
    if max_batch_tokens <= 0:
        return list(pairs[offset:offset + size])

    budget = int(max_batch_tokens)
    batch: list[tuple[Statement, Optional[StatementEncoding]]] = []
    max_len = 0
    for pair in pairs[offset:offset + size]:
        pair_len = max(1, _statement_pair_length(pair))
        next_max_len = max(max_len, pair_len)
        next_padded_tokens = next_max_len * (len(batch) + 1)
        if batch and next_padded_tokens > budget:
            break
        batch.append(pair)
        max_len = next_max_len
    return batch or [pairs[offset]]


def _statement_pair_length(pair: tuple[Statement, Optional[StatementEncoding]]) -> int:
    stmt, encoding = pair
    if encoding is not None:
        return len(encoding.input_ids)
    return _statement_length(stmt)


@torch.no_grad()
def batched_statement_reps(
    model,
    tokenizer,
    statements: Sequence[Statement],
    layers: List[int],
    device: torch.device,
    max_length: int,
    *,
    compute_answer_metrics: bool = True,
    eigenscore_alpha: float = 1e-3,
    hidden_state_capture: str = "outputs",
    encoded_statements: Sequence[Optional[StatementEncoding]] | None = None,
    prefix_kv_cache: bool = False,
    first_token_top_k: int = FIRST_TOKEN_TOP_K_DEFAULT,
) -> list[Optional[dict]]:
    """Batch forced-answer forwards while preserving per-statement result shape."""
    encoded: list[tuple[int, list[int], int]] = []
    results: list[Optional[dict]] = [None] * len(statements)
    if encoded_statements is None:
        tokenized_statements = _batch_statement_token_ids(tokenizer, statements, max_length)
    else:
        if len(encoded_statements) != len(statements):
            raise ValueError("encoded_statements must have the same length as statements.")
        tokenized_statements = [_encoding_to_token_ids(encoding) for encoding in encoded_statements]
    for idx, tokenized in enumerate(tokenized_statements):
        if tokenized is None:
            continue
        ids, n_ans = tokenized
        encoded.append((idx, ids, n_ans))
    if not encoded:
        return results
    if prefix_kv_cache:
        if hidden_state_capture != "outputs":
            raise ValueError("--prefix-kv-cache requires hidden_state_capture='outputs'.")
        if compute_answer_metrics:
            return _batched_statement_reps_with_prefix_kv(
                model,
                encoded,
                layers,
                device,
                result_count=len(statements),
                eigenscore_alpha=eigenscore_alpha,
                first_token_top_k=first_token_top_k,
            )

    n_layers = _model_num_hidden_layers(model)
    capture_layers = _residual_dynamics_capture_layers(
        layers,
        n_layers=n_layers,
        hidden_state_capture=hidden_state_capture,
    )
    pad_token_id = _pad_token_id(tokenizer)
    batch_len = max(len(ids) for _, ids, _ in encoded)
    input_ids = torch.full((len(encoded), batch_len), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    for row, (_, ids, _) in enumerate(encoded):
        ids_t = torch.tensor(ids, dtype=torch.long, device=device)
        input_ids[row, :len(ids)] = ids_t
        attention_mask[row, :len(ids)] = 1

    out, hidden_by_layer = _forward_with_selected_hidden_states(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        layers=capture_layers,
        hidden_state_capture=hidden_state_capture,
        need_logits=compute_answer_metrics,
    )
    for row, (original_idx, ids, n_ans) in enumerate(encoded):
        seq_len = len(ids)
        last_by_layer = {
            layer: hidden_by_layer[layer][row, seq_len - 1, :].float().cpu()
            for layer in layers
        }
        if not compute_answer_metrics:
            results[original_idx] = {"last": last_by_layer}
            continue

        ans_start = seq_len - n_ans
        ans_hs = hidden_by_layer[layers[0]][row, ans_start:seq_len, :].float().cpu()
        eigenscore_by_layer = {
            layer: float(internal_eigenscore(
                hidden_by_layer[layer][row, ans_start:seq_len, :].float(),
                alpha=eigenscore_alpha,
            ).item())
            for layer in layers
        }
        resid_update_norm_by_layer = {
            layer: _residual_update_norm(
                hidden_by_layer,
                int(layer),
                n_layers=n_layers,
                row=row,
                token_index=seq_len - 1,
            )
            for layer in layers
        }

        nll = _answer_nll_from_logits(out.logits[row], input_ids[row], seq_len, n_ans)
        first_token_entropy = _first_answer_token_entropy_from_logits(
            out.logits[row],
            seq_len,
            n_ans,
            top_k=first_token_top_k,
        )
        results[original_idx] = {
            "last": last_by_layer,
            "ans_hs": ans_hs,
            "eigenscore_by_layer": eigenscore_by_layer,
            "resid_update_norm_by_layer": resid_update_norm_by_layer,
            "first_token_entropy": first_token_entropy,
            "nll": nll,
        }
    return results


def _batched_statement_reps_with_prefix_kv(
    model,
    encoded: Sequence[tuple[int, list[int], int]],
    layers: Sequence[int],
    device: torch.device,
    *,
    result_count: int,
    eigenscore_alpha: float,
    first_token_top_k: int,
) -> list[Optional[dict]]:
    """Score encoded statements by reusing one prefix KV cache per shared prefix."""
    results: list[Optional[dict]] = [None] * int(result_count)
    groups: dict[tuple[int, ...], list[tuple[int, list[int], int]]] = {}
    n_layers = _model_num_hidden_layers(model)
    capture_layers = _residual_dynamics_capture_layers(
        layers,
        n_layers=n_layers,
        hidden_state_capture="outputs",
    )
    for original_idx, ids, n_ans in encoded:
        prefix_len = len(ids) - int(n_ans)
        if prefix_len <= 0:
            raise ValueError("--prefix-kv-cache requires at least one prefix token per statement.")
        groups.setdefault(tuple(ids[:prefix_len]), []).append((original_idx, ids, int(n_ans)))

    for prefix_ids_tuple, items in groups.items():
        prefix_ids = torch.tensor([prefix_ids_tuple], dtype=torch.long, device=device)
        prefix_mask = torch.ones_like(prefix_ids)
        prefix_out = model(
            input_ids=prefix_ids,
            attention_mask=prefix_mask,
            output_hidden_states=False,
            use_cache=True,
        )
        prefix_past = getattr(prefix_out, "past_key_values", None)
        if prefix_past is None:
            raise ValueError("model did not return past_key_values for --prefix-kv-cache.")
        prefix_next_logits = prefix_out.logits[0, -1:, :]

        for original_idx, ids, n_ans in items:
            answer_ids = ids[-n_ans:]
            answer_input = torch.tensor([answer_ids], dtype=torch.long, device=device)
            full_attention_mask = torch.ones(
                (1, len(prefix_ids_tuple) + len(answer_ids)),
                dtype=torch.long,
                device=device,
            )
            answer_out = model(
                input_ids=answer_input,
                attention_mask=full_attention_mask,
                past_key_values=_clone_prefix_past(prefix_past),
                output_hidden_states=True,
                use_cache=False,
            )
            hidden_by_layer = {int(layer): answer_out.hidden_states[int(layer)] for layer in capture_layers}
            answer_len = len(answer_ids)
            last_by_layer = {
                int(layer): hidden_by_layer[int(layer)][0, answer_len - 1, :].float().cpu()
                for layer in layers
            }
            ans_hs = hidden_by_layer[int(layers[0])][0, :answer_len, :].float().cpu()
            eigenscore_by_layer = {
                int(layer): float(internal_eigenscore(
                    hidden_by_layer[int(layer)][0, :answer_len, :].float(),
                    alpha=eigenscore_alpha,
                ).item())
                for layer in layers
            }
            resid_update_norm_by_layer = {
                int(layer): _residual_update_norm(
                    hidden_by_layer,
                    int(layer),
                    n_layers=n_layers,
                    row=0,
                    token_index=answer_len - 1,
                )
                for layer in layers
            }
            prediction_logits = prefix_next_logits
            if answer_len > 1:
                prediction_logits = torch.cat(
                    [prefix_next_logits, answer_out.logits[0, :answer_len - 1, :]],
                    dim=0,
                )
            nll = _nll_from_prediction_logits(prediction_logits, answer_input[0])
            first_token_entropy = float(
                topk_normalized_entropy(prefix_next_logits[0], top_k=first_token_top_k).item()
            )
            results[original_idx] = {
                "last": last_by_layer,
                "ans_hs": ans_hs,
                "eigenscore_by_layer": eigenscore_by_layer,
                "resid_update_norm_by_layer": resid_update_norm_by_layer,
                "first_token_entropy": first_token_entropy,
                "nll": nll,
            }
    return results


def _clone_prefix_past(past_key_values):
    """Return a reusable copy of a prefix cache for one answer continuation."""
    if hasattr(past_key_values, "layers"):
        data = []
        for layer in past_key_values.layers:
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            if keys is None or values is None:
                raise ValueError("unsupported cache layer format for --prefix-kv-cache.")
            data.append((keys.detach().clone(), values.detach().clone()))
        return past_key_values.__class__(ddp_cache_data=data)
    return past_key_values


def _answer_nll_from_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    seq_len: int,
    n_answer_tokens: int,
) -> float:
    """Return answer-token NLL without normalizing unused prompt positions."""
    seq_len = int(seq_len)
    n_answer_tokens = int(n_answer_tokens)
    ans_start = seq_len - n_answer_tokens
    logit_start = max(0, ans_start - 1)
    logit_end = seq_len - 1
    if logit_end <= logit_start:
        return float("nan")
    answer_logits = logits[logit_start:logit_end, :].float()
    targets = input_ids[logit_start + 1:seq_len]
    return _nll_from_prediction_logits(answer_logits, targets)


def _first_answer_token_entropy_from_logits(
    logits: torch.Tensor,
    seq_len: int,
    n_answer_tokens: int,
    *,
    top_k: int = FIRST_TOKEN_TOP_K_DEFAULT,
) -> float:
    """Return top-k normalized entropy for the first answer-token prediction."""
    seq_len = int(seq_len)
    n_answer_tokens = int(n_answer_tokens)
    ans_start = seq_len - n_answer_tokens
    logit_index = max(0, ans_start - 1)
    if logit_index >= seq_len:
        return float("nan")
    return float(topk_normalized_entropy(logits[logit_index], top_k=top_k).item())


def _nll_from_prediction_logits(prediction_logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Return mean NLL for logits that directly predict the given targets."""
    logp = torch.log_softmax(prediction_logits.float(), dim=-1)
    tok_logp = logp[torch.arange(logp.shape[0], device=logp.device), targets.to(logp.device)]
    return float((-tok_logp.mean()).item())



def _batched_statement_reps_for_pairs(
    model,
    tokenizer,
    batch_pairs: Sequence[tuple[Statement, Optional[StatementEncoding]]],
    layers: List[int],
    device: torch.device,
    max_length: int,
    *,
    encoded_statements_provided: bool,
    fallback_state: BatchSizeFallbackState | None,
    phase: str,
    compute_answer_metrics: bool = True,
    eigenscore_alpha: float = 1e-3,
    hidden_state_capture: str = "outputs",
    prefix_kv_cache: bool = False,
    first_token_top_k: int = FIRST_TOKEN_TOP_K_DEFAULT,
) -> list[Optional[dict]]:
    """Run forced-answer forwards for statement pairs with optional memory fallback."""

    def _runner(pairs):
        statements = [stmt for stmt, _encoding in pairs]
        encoded = [encoding for _stmt, encoding in pairs] if encoded_statements_provided else None
        return batched_statement_reps(
            model,
            tokenizer,
            statements,
            layers,
            device,
            max_length,
            compute_answer_metrics=compute_answer_metrics,
            eigenscore_alpha=eigenscore_alpha,
            hidden_state_capture=hidden_state_capture,
            encoded_statements=encoded,
            prefix_kv_cache=prefix_kv_cache,
            first_token_top_k=first_token_top_k,
        )

    return _run_with_batch_size_fallback(
        list(batch_pairs),
        state=fallback_state,
        phase=phase,
        runner=_runner,
        clear_cache=lambda: _clear_device_cache(device),
    )


def statement_reps(model, tokenizer, stmt: Statement, layers: List[int],
                   device: torch.device, max_length: int, *,
                   compute_answer_metrics: bool = True,
                   eigenscore_alpha: float = 1e-3,
                   hidden_state_capture: str = "outputs",
                   prefix_kv_cache: bool = False,
                   first_token_top_k: int = FIRST_TOKEN_TOP_K_DEFAULT) -> Optional[dict]:
    """Single-statement compatibility wrapper around batched_statement_reps."""
    return batched_statement_reps(
        model,
        tokenizer,
        [stmt],
        layers,
        device,
        max_length,
        compute_answer_metrics=compute_answer_metrics,
        eigenscore_alpha=eigenscore_alpha,
        hidden_state_capture=hidden_state_capture,
        encoded_statements=None,
        prefix_kv_cache=prefix_kv_cache,
        first_token_top_k=first_token_top_k,
    )[0]


def _pad_token_id(tokenizer) -> int:
    if tokenizer.pad_token_id is not None:
        return int(tokenizer.pad_token_id)
    if tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
        return int(tokenizer.eos_token_id)
    raise ValueError("tokenizer must define either pad_token_id or eos_token_id for sampling.")


def _fork_rng_devices(device: torch.device) -> list[int]:
    if device.type != "cuda":
        return []
    return [device.index or torch.cuda.current_device()]


@torch.no_grad()
def sampled_response_diagnostics_batch(
    model,
    tokenizer,
    statements: Sequence[Statement],
    layers: List[int],
    device: torch.device,
    max_length: int,
    *,
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    pooling: str,
    seed: int,
    hidden_state_capture: str = "outputs",
) -> list[Optional[SampledResponseDiagnostics]]:
    """Generate one or more continuations per statement and pool response diagnostics."""
    if n_samples < 1:
        return [None] * len(statements)
    if max_new_tokens < 1:
        raise ValueError("inside max_new_tokens must be >= 1.")
    if temperature <= 0.0:
        raise ValueError("inside temperature must be > 0.")
    if not (0.0 < top_p <= 1.0):
        raise ValueError("inside top_p must be in (0, 1].")
    if pooling not in {"last", "mean"}:
        raise ValueError("inside pooling must be 'last' or 'mean'.")
    if not statements:
        return []

    prompts = [_candidate_verification_prompt(stmt) for stmt in statements]
    pad_token_id = _pad_token_id(tokenizer)
    original_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    try:
        encoded = tokenizer(
            prompts,
            add_special_tokens=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
    finally:
        tokenizer.padding_side = original_padding_side

    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)
    with torch.random.fork_rng(devices=_fork_rng_devices(device)):
        torch.manual_seed(int(seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        generated_output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            num_return_sequences=n_samples,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

    generated = generated_output.sequences if hasattr(generated_output, "sequences") else generated_output
    prompt_width = input_ids.shape[1]
    new_width = max(generated.shape[1] - prompt_width, 1)
    sample_logprobs = _sample_logprobs_from_generate_output(
        generated_output,
        generated,
        prompt_width=prompt_width,
        n_statements=len(statements),
        n_samples=n_samples,
    )
    generated_attention = torch.cat([
        attention_mask.repeat_interleave(n_samples, dim=0),
        torch.ones((len(statements) * n_samples, new_width), dtype=attention_mask.dtype, device=device),
    ], dim=1)
    if generated_attention.shape[1] != generated.shape[1]:
        generated_attention = torch.ones_like(generated, dtype=attention_mask.dtype, device=device)

    _out, hidden_by_layer = _forward_with_selected_hidden_states(
        model,
        input_ids=generated,
        attention_mask=generated_attention,
        layers=layers,
        hidden_state_capture=hidden_state_capture,
        need_logits=False,
    )
    sample_texts = _decode_sampled_continuations(
        tokenizer,
        generated,
        prompt_width=prompt_width,
        n_statements=len(statements),
        n_samples=n_samples,
    )
    embeddings_by_statement: list[dict[int, torch.Tensor]] = [{layer: torch.empty(0) for layer in layers}
                                                              for _ in statements]
    response_start = min(prompt_width, generated.shape[1] - 1)
    response_end = generated.shape[1]
    for layer in layers:
        states = hidden_by_layer[layer].float()
        for stmt_idx in range(len(statements)):
            pooled = []
            for sample_idx in range(n_samples):
                row = stmt_idx * n_samples + sample_idx
                if pooling == "mean" and response_end > response_start:
                    pooled.append(states[row, response_start:response_end, :].mean(dim=0))
                else:
                    pooled.append(states[row, response_end - 1, :])
            embeddings_by_statement[stmt_idx][layer] = torch.stack(pooled).cpu()
    return [
        SampledResponseDiagnostics(
            embeddings_by_layer=embeddings_by_statement[stmt_idx],
            sample_texts=sample_texts[stmt_idx],
            sample_logprobs=sample_logprobs[stmt_idx],
        )
        for stmt_idx in range(len(statements))
    ]


def _sample_logprobs_from_generate_output(
    generated_output,
    generated: torch.Tensor,
    *,
    prompt_width: int,
    n_statements: int,
    n_samples: int,
) -> list[tuple[float, ...]]:
    scores = getattr(generated_output, "scores", None)
    if scores is None:
        return [() for _ in range(n_statements)]
    score_steps = tuple(scores)
    if not score_steps:
        return [() for _ in range(n_statements)]
    n_rows = int(generated.shape[0])
    n_steps = min(len(score_steps), max(0, int(generated.shape[1]) - int(prompt_width)))
    if n_steps <= 0:
        return [() for _ in range(n_statements)]
    if n_rows != int(n_statements) * int(n_samples):
        return [() for _ in range(n_statements)]
    row_totals = torch.zeros(n_rows, dtype=torch.float64, device=generated.device)
    row_counts = torch.zeros(n_rows, dtype=torch.float64, device=generated.device)
    for step_idx, logits in enumerate(score_steps[:n_steps]):
        if logits.shape[0] != n_rows:
            return [() for _ in range(n_statements)]
        token_ids = generated[:, int(prompt_width) + step_idx].to(logits.device)
        step_logprobs = torch.log_softmax(logits.float(), dim=-1).gather(1, token_ids.unsqueeze(1)).squeeze(1)
        row_totals += step_logprobs.to(row_totals.device, dtype=torch.float64)
        row_counts += 1.0
    means = (row_totals / row_counts.clamp_min(1.0)).detach().cpu().tolist()
    return [
        tuple(float(value) for value in means[start:start + n_samples])
        for start in range(0, n_rows, n_samples)
    ]


def _decode_sampled_continuations(
    tokenizer,
    generated: torch.Tensor,
    *,
    prompt_width: int,
    n_statements: int,
    n_samples: int,
) -> list[tuple[str, ...]]:
    continuation_start = min(int(prompt_width), int(generated.shape[1]))
    continuation_ids = generated[:, continuation_start:].detach().cpu()
    decoded = tokenizer.batch_decode(continuation_ids, skip_special_tokens=True)
    groups = []
    for stmt_idx in range(n_statements):
        start = stmt_idx * n_samples
        groups.append(tuple(decoded[start:start + n_samples]))
    return groups


@torch.no_grad()
def sampled_response_embeddings_batch(
    model,
    tokenizer,
    statements: Sequence[Statement],
    layers: List[int],
    device: torch.device,
    max_length: int,
    *,
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    pooling: str,
    seed: int,
    hidden_state_capture: str = "outputs",
) -> list[Optional[dict[int, torch.Tensor]]]:
    """Generate multiple continuations per statement and pool response embeddings."""
    diagnostics_batch = sampled_response_diagnostics_batch(
        model,
        tokenizer,
        statements,
        layers,
        device,
        max_length,
        n_samples=n_samples,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        pooling=pooling,
        seed=seed,
        hidden_state_capture=hidden_state_capture,
    )
    return [
        diagnostics.embeddings_by_layer if diagnostics is not None else None
        for diagnostics in diagnostics_batch
    ]


@torch.no_grad()
def sampled_response_embeddings(
    model,
    tokenizer,
    stmt: Statement,
    layers: List[int],
    device: torch.device,
    max_length: int,
    *,
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    pooling: str,
    seed: int,
    hidden_state_capture: str = "outputs",
) -> Optional[dict[int, torch.Tensor]]:
    return sampled_response_embeddings_batch(
        model,
        tokenizer,
        [stmt],
        layers,
        device,
        max_length,
        n_samples=n_samples,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        pooling=pooling,
        seed=seed,
        hidden_state_capture=hidden_state_capture,
    )[0]


def _inside_diagnostics_from_response(
    response_diagnostics: SampledResponseDiagnostics,
    *,
    eigenscore_alpha: float,
    embedding_similarity_threshold: float,
    adaptive_rounds: int = 1,
    stopped_early: bool = False,
    stop_reason: str | None = None,
) -> SampledInsideDiagnostics:
    n_samples = len(response_diagnostics.sample_texts)
    return SampledInsideDiagnostics(
        eigenscore_by_layer={
            layer: float(internal_eigenscore(values, alpha=eigenscore_alpha).item())
            for layer, values in response_diagnostics.embeddings_by_layer.items()
        },
        semantic_entropy=float(lexical_semantic_entropy(response_diagnostics.sample_texts).item()),
        embedding_entropy_by_layer={
            layer: float(
                embedding_semantic_entropy(
                    values,
                    similarity_threshold=embedding_similarity_threshold,
                ).item()
            )
            for layer, values in response_diagnostics.embeddings_by_layer.items()
        },
        semantic_energy=float(
            lexical_semantic_energy(
                response_diagnostics.sample_texts,
                sample_logprobs=response_diagnostics.sample_logprobs or None,
            ).item()
        ),
        sample_texts=tuple(response_diagnostics.sample_texts),
        sample_logprobs=tuple(response_diagnostics.sample_logprobs),
        n_samples=n_samples,
        adaptive_rounds=adaptive_rounds,
        stopped_early=stopped_early,
        stop_reason=stop_reason,
    )


def _merge_response_diagnostics(
    previous: SampledResponseDiagnostics | None,
    new: SampledResponseDiagnostics,
) -> SampledResponseDiagnostics:
    if previous is None:
        return new
    return SampledResponseDiagnostics(
        embeddings_by_layer={
            layer: torch.cat([previous.embeddings_by_layer[layer], new.embeddings_by_layer[layer]], dim=0)
            for layer in previous.embeddings_by_layer
        },
        sample_texts=previous.sample_texts + new.sample_texts,
        sample_logprobs=previous.sample_logprobs + new.sample_logprobs,
    )


def _inside_diagnostics_delta(
    previous: SampledInsideDiagnostics,
    current: SampledInsideDiagnostics,
    *,
    target_layer: int,
) -> float:
    layer = (
        target_layer
        if target_layer in current.embedding_entropy_by_layer
        else next(iter(current.embedding_entropy_by_layer))
    )
    return max(
        abs(float(current.semantic_entropy) - float(previous.semantic_entropy)),
        abs(float(current.semantic_energy) - float(previous.semantic_energy)),
        abs(float(current.embedding_entropy_by_layer[layer]) - float(previous.embedding_entropy_by_layer[layer])),
    )


def _adaptive_inside_seed(base_seed: int, round_idx: int) -> int:
    return int(base_seed) + int(round_idx) * 1_000_003


def _selfcheck_claim_for_statement(stmt: Statement) -> Claim:
    text = stmt.answer.strip()
    if stmt.question.strip():
        text = f"{stmt.question.strip()} {text}".strip()
    return Claim(text=text)


def _inside_selfcheck_stop_reason(
    stmt: Statement,
    sample_texts: Sequence[str],
    *,
    total_samples: int,
    min_overlap: float,
    support_threshold: float,
    refute_threshold: float,
) -> str | None:
    if len(sample_texts) >= total_samples:
        return None
    valid_sample_texts = tuple(text for text in sample_texts if str(text).strip())
    if len(valid_sample_texts) < 2:
        return None
    status = SelfConsistencyVerifier(
        min_samples=2,
        min_overlap=min_overlap,
        support_threshold=support_threshold,
        refute_threshold=refute_threshold,
    ).sample_budget_status(
        _selfcheck_claim_for_statement(stmt),
        valid_sample_texts,
        total_samples=total_samples,
    )
    reason = status.get("reason") if status.get("can_stop") else None
    return None if reason is None else f"selfcheck_{reason}"


def sampled_inside_diagnostics_batch(
    model,
    tokenizer,
    statements: Sequence[Statement],
    layers: List[int],
    device: torch.device,
    max_length: int,
    *,
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    pooling: str,
    seed: int,
    eigenscore_alpha: float,
    embedding_similarity_threshold: float = 0.90,
    hidden_state_capture: str = "outputs",
) -> list[Optional[SampledInsideDiagnostics]]:
    if n_samples < 2:
        return [None] * len(statements)
    response_diagnostics_batch = sampled_response_diagnostics_batch(
        model,
        tokenizer,
        statements,
        layers,
        device,
        max_length,
        n_samples=n_samples,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        pooling=pooling,
        seed=seed,
        hidden_state_capture=hidden_state_capture,
    )
    diagnostics_batch: list[Optional[SampledInsideDiagnostics]] = []
    for response_diagnostics in response_diagnostics_batch:
        if response_diagnostics is None:
            diagnostics_batch.append(None)
            continue
        diagnostics_batch.append(_inside_diagnostics_from_response(
            response_diagnostics,
            eigenscore_alpha=eigenscore_alpha,
            embedding_similarity_threshold=embedding_similarity_threshold,
        ))
    return diagnostics_batch


def sampled_inside_adaptive_diagnostics_batch(
    model,
    tokenizer,
    statements: Sequence[Statement],
    layers: List[int],
    device: torch.device,
    max_length: int,
    *,
    min_samples: int,
    max_samples: int,
    sample_step: int,
    stability_delta: float,
    target_layer: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    pooling: str,
    seed: int,
    eigenscore_alpha: float,
    embedding_similarity_threshold: float = 0.90,
    hidden_state_capture: str = "outputs",
    selfcheck_early_stop: bool = False,
    selfcheck_min_overlap: float = 0.65,
    selfcheck_support_threshold: float = 0.60,
    selfcheck_refute_threshold: float = 0.50,
) -> list[Optional[SampledInsideDiagnostics]]:
    if min_samples < 2:
        raise ValueError("adaptive inside min_samples must be >= 2.")
    if max_samples < min_samples:
        raise ValueError("adaptive inside max_samples must be >= min_samples.")
    if sample_step < 1:
        raise ValueError("adaptive inside sample_step must be >= 1.")
    if stability_delta < 0.0:
        raise ValueError("adaptive inside stability_delta must be >= 0.")
    if not (0.0 <= selfcheck_min_overlap <= 1.0):
        raise ValueError("adaptive inside selfcheck_min_overlap must be in [0, 1].")
    if not (0.0 <= selfcheck_support_threshold <= 1.0):
        raise ValueError("adaptive inside selfcheck_support_threshold must be in [0, 1].")
    if not (0.0 <= selfcheck_refute_threshold <= 1.0):
        raise ValueError("adaptive inside selfcheck_refute_threshold must be in [0, 1].")
    if not statements:
        return []

    accumulated: list[SampledResponseDiagnostics | None] = [None] * len(statements)
    previous: list[SampledInsideDiagnostics | None] = [None] * len(statements)
    final: list[Optional[SampledInsideDiagnostics]] = [None] * len(statements)
    active = list(range(len(statements)))
    round_idx = 0

    while active:
        requested = min_samples if round_idx == 0 else sample_step
        current_count = len(accumulated[active[0]].sample_texts) if accumulated[active[0]] is not None else 0
        requested = min(requested, max_samples - current_count)
        if requested <= 0:
            break

        active_statements = [statements[idx] for idx in active]
        response_batch = sampled_response_diagnostics_batch(
            model,
            tokenizer,
            active_statements,
            layers,
            device,
            max_length,
            n_samples=requested,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            pooling=pooling,
            seed=_adaptive_inside_seed(seed, round_idx),
            hidden_state_capture=hidden_state_capture,
        )

        next_active: list[int] = []
        for position, response_diagnostics in zip(active, response_batch):
            if response_diagnostics is None:
                final[position] = None
                continue
            accumulated[position] = _merge_response_diagnostics(accumulated[position], response_diagnostics)
            current = _inside_diagnostics_from_response(
                accumulated[position],
                eigenscore_alpha=eigenscore_alpha,
                embedding_similarity_threshold=embedding_similarity_threshold,
                adaptive_rounds=round_idx + 1,
            )
            previous_diagnostics = previous[position]
            reached_max = current.n_samples >= max_samples
            stable = (
                previous_diagnostics is not None
                and current.n_samples >= min_samples
                and _inside_diagnostics_delta(previous_diagnostics, current, target_layer=target_layer)
                <= stability_delta
            )
            selfcheck_stop_reason = (
                _inside_selfcheck_stop_reason(
                    statements[position],
                    current.sample_texts,
                    total_samples=max_samples,
                    min_overlap=selfcheck_min_overlap,
                    support_threshold=selfcheck_support_threshold,
                    refute_threshold=selfcheck_refute_threshold,
                )
                if selfcheck_early_stop and current.n_samples >= min_samples
                else None
            )
            stop_reason = selfcheck_stop_reason or ("stability_delta" if stable and not reached_max else None)
            if stop_reason is not None or reached_max:
                final[position] = SampledInsideDiagnostics(
                    eigenscore_by_layer=current.eigenscore_by_layer,
                    semantic_entropy=current.semantic_entropy,
                    embedding_entropy_by_layer=current.embedding_entropy_by_layer,
                    semantic_energy=current.semantic_energy,
                    sample_texts=current.sample_texts,
                    sample_logprobs=current.sample_logprobs,
                    n_samples=current.n_samples,
                    adaptive_rounds=current.adaptive_rounds,
                    stopped_early=stop_reason is not None,
                    stop_reason=stop_reason,
                )
            else:
                previous[position] = current
                next_active.append(position)
        active = next_active
        round_idx += 1

    return final


def _sample_inside_diagnostics_for_args(
    model,
    tokenizer,
    statements: Sequence[Statement],
    layers: List[int],
    device: torch.device,
    args,
    *,
    inside_adaptive_sampling: bool,
    inside_min_samples: int,
    inside_sample_step: int,
    inside_stability_delta: float,
    inside_embedding_threshold: float,
    inside_selfcheck_early_stop: bool,
    inside_selfcheck_min_overlap: float,
    inside_selfcheck_support_threshold: float,
    inside_selfcheck_refute_threshold: float,
    seed: int,
) -> list[Optional[SampledInsideDiagnostics]]:
    if inside_adaptive_sampling:
        return sampled_inside_adaptive_diagnostics_batch(
            model,
            tokenizer,
            statements,
            layers,
            device,
            args.max_length,
            min_samples=inside_min_samples,
            max_samples=args.inside_samples,
            sample_step=inside_sample_step,
            stability_delta=inside_stability_delta,
            target_layer=args.layer,
            max_new_tokens=args.inside_max_new_tokens,
            temperature=args.inside_temperature,
            top_p=args.inside_top_p,
            pooling=args.inside_pooling,
            seed=seed,
            eigenscore_alpha=args.eigenscore_alpha,
            embedding_similarity_threshold=inside_embedding_threshold,
            hidden_state_capture=args.hidden_state_capture,
            selfcheck_early_stop=inside_selfcheck_early_stop,
            selfcheck_min_overlap=inside_selfcheck_min_overlap,
            selfcheck_support_threshold=inside_selfcheck_support_threshold,
            selfcheck_refute_threshold=inside_selfcheck_refute_threshold,
        )
    return sampled_inside_diagnostics_batch(
        model,
        tokenizer,
        statements,
        layers,
        device,
        args.max_length,
        n_samples=args.inside_samples,
        max_new_tokens=args.inside_max_new_tokens,
        temperature=args.inside_temperature,
        top_p=args.inside_top_p,
        pooling=args.inside_pooling,
        seed=seed,
        eigenscore_alpha=args.eigenscore_alpha,
        embedding_similarity_threshold=inside_embedding_threshold,
        hidden_state_capture=args.hidden_state_capture,
    )


def sampled_inside_scores_batch(
    model,
    tokenizer,
    statements: Sequence[Statement],
    layers: List[int],
    device: torch.device,
    max_length: int,
    *,
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    pooling: str,
    seed: int,
    eigenscore_alpha: float,
    embedding_similarity_threshold: float = 0.90,
    hidden_state_capture: str = "outputs",
) -> list[Optional[dict[int, float]]]:
    diagnostics_batch = sampled_inside_diagnostics_batch(
        model,
        tokenizer,
        statements,
        layers,
        device,
        max_length,
        n_samples=n_samples,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        pooling=pooling,
        seed=seed,
        eigenscore_alpha=eigenscore_alpha,
        embedding_similarity_threshold=embedding_similarity_threshold,
        hidden_state_capture=hidden_state_capture,
    )
    return [
        diagnostics.eigenscore_by_layer if diagnostics is not None else None
        for diagnostics in diagnostics_batch
    ]


def sampled_inside_scores(
    model,
    tokenizer,
    stmt: Statement,
    layers: List[int],
    device: torch.device,
    max_length: int,
    *,
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    pooling: str,
    seed: int,
    eigenscore_alpha: float,
    embedding_similarity_threshold: float = 0.90,
    hidden_state_capture: str = "outputs",
) -> Optional[dict[int, float]]:
    return sampled_inside_scores_batch(
        model,
        tokenizer,
        [stmt],
        layers,
        device,
        max_length,
        n_samples=n_samples,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        pooling=pooling,
        seed=seed,
        eigenscore_alpha=eigenscore_alpha,
        embedding_similarity_threshold=embedding_similarity_threshold,
        hidden_state_capture=hidden_state_capture,
    )[0]


def build_layer_stats(model, tokenizer, true_texts: List[str], false_texts: List[str],
                      layers: List[int], device: torch.device, max_length: int,
                      subspace_rank: int, batch_size: int, length_bucketed: bool = False,
                      max_batch_tokens: int = 0,
                      progress_every: int = 50, checkpoint_path: str | Path | None = None,
                      checkpoint_metadata: Mapping | None = None, resume_checkpoint: bool = True,
                      checkpoint_every: int = 50,
                      hidden_state_capture: str = "outputs",
                      covariance_mode: str = "full",
                      covariance_low_rank: int = 16,
                      true_encodings: Sequence[Optional[StatementEncoding]] | None = None,
                      false_encodings: Sequence[Optional[StatementEncoding]] | None = None,
                      batch_fallback_state: BatchSizeFallbackState | None = None) -> tuple[dict, dict]:
    """逐层构建真值流形与 mass-mean 对比方向（与 EigenTruthWrapper.warmup 同构）。
    Per-layer truth manifolds plus the mass-mean contrastive direction
    (mirrors EigenTruthWrapper.warmup; cf. Marks & Tegmark mass-mean probing).
    """
    checkpoint_path_obj = Path(checkpoint_path) if checkpoint_path else None
    checkpoint_metadata = dict(checkpoint_metadata or {})
    if checkpoint_path_obj and resume_checkpoint and checkpoint_path_obj.exists():
        loaded_checkpoint = load_warmup_checkpoint(
            checkpoint_path_obj,
            expected_metadata=checkpoint_metadata,
            device=device,
        )
        manifolds = loaded_checkpoint["manifolds"]
        true_state_lists = loaded_checkpoint["true_state_lists"]
        false_state_lists = loaded_checkpoint["false_state_lists"]
        false_sums = loaded_checkpoint["false_sums"]
        n_false = int(loaded_checkpoint["n_false"])
        true_done = int(loaded_checkpoint["true_done"])
        false_done = int(loaded_checkpoint["false_done"])
        print(f"   loaded warmup checkpoint: {checkpoint_path_obj} "
              f"(true={true_done}/{len(true_texts)}, false={false_done}/{len(false_texts)})")
    else:
        manifolds = {
            layer: TruthManifold(
                covariance_mode=covariance_mode,
                covariance_low_rank=covariance_low_rank,
            )
            for layer in layers
        }
        true_state_lists = {layer: [] for layer in layers}
        false_state_lists = {layer: [] for layer in layers}
        false_sums: dict = {layer: None for layer in layers}
        n_false = 0
        true_done = 0
        false_done = 0

    true_statements = [Statement("", text, 0) for text in true_texts]
    true_last_reported = true_done
    true_last_checkpoint = true_done
    true_started = time.perf_counter()
    true_pairs = _statement_pairs(
        true_statements,
        true_encodings,
        length_bucketed=length_bucketed,
    )[true_done:]
    true_pair_offset = 0
    while true_pair_offset < len(true_pairs):
        current_batch_size = batch_fallback_state.batch_size() if batch_fallback_state else batch_size
        batch_pairs = _next_statement_pair_batch(
            true_pairs,
            true_pair_offset,
            current_batch_size,
            max_batch_tokens=max_batch_tokens,
        )
        batch = [stmt for stmt, _encoding in batch_pairs]
        reps_batch = _batched_statement_reps_for_pairs(
            model,
            tokenizer,
            batch_pairs,
            layers,
            device,
            max_length,
            compute_answer_metrics=False,
            encoded_statements_provided=true_encodings is not None,
            fallback_state=batch_fallback_state,
            phase="build_layer_stats",
            hidden_state_capture=hidden_state_capture,
        )
        valid_reps = [reps for reps in reps_batch if reps is not None]
        for layer in layers:
            states = [reps["last"][layer] for reps in valid_reps]
            if not states:
                continue
            manifolds[layer].update_many(torch.stack(states))
            true_state_lists[layer].extend(states)
        true_done += len(batch)
        true_pair_offset += len(batch_pairs)
        if _progress_report_due(true_done, len(true_statements), progress_every, true_last_reported):
            true_last_reported = min(true_done, len(true_statements))
            print(_format_progress("warmup true", true_last_reported, len(true_statements),
                                   time.perf_counter() - true_started))
        if checkpoint_path_obj and _progress_report_due(
            true_done, len(true_statements), checkpoint_every, true_last_checkpoint
        ):
            true_last_checkpoint = min(true_done, len(true_statements))
            save_warmup_checkpoint(
                checkpoint_path_obj,
                metadata=checkpoint_metadata,
                manifolds=manifolds,
                true_state_lists=true_state_lists,
                false_state_lists=false_state_lists,
                false_sums=false_sums,
                n_false=n_false,
                true_done=true_last_checkpoint,
                false_done=false_done,
            )
            print(f"   saved warmup checkpoint: {checkpoint_path_obj} "
                  f"(true={true_last_checkpoint}/{len(true_statements)}, false={false_done}/{len(false_texts)})")

    false_statements = [Statement("", text, 1) for text in false_texts]
    false_last_reported = false_done
    false_last_checkpoint = false_done
    false_started = time.perf_counter()
    false_pairs = _statement_pairs(
        false_statements,
        false_encodings,
        length_bucketed=length_bucketed,
    )[false_done:]
    false_pair_offset = 0
    while false_pair_offset < len(false_pairs):
        current_batch_size = batch_fallback_state.batch_size() if batch_fallback_state else batch_size
        batch_pairs = _next_statement_pair_batch(
            false_pairs,
            false_pair_offset,
            current_batch_size,
            max_batch_tokens=max_batch_tokens,
        )
        batch = [stmt for stmt, _encoding in batch_pairs]
        reps_batch = _batched_statement_reps_for_pairs(
            model,
            tokenizer,
            batch_pairs,
            layers,
            device,
            max_length,
            compute_answer_metrics=False,
            encoded_statements_provided=false_encodings is not None,
            fallback_state=batch_fallback_state,
            phase="build_layer_stats",
            hidden_state_capture=hidden_state_capture,
        )
        for reps in reps_batch:
            if reps is None:
                continue
            n_false += 1
            for layer in layers:
                h = reps["last"][layer]
                false_state_lists[layer].append(h)
                false_sums[layer] = h if false_sums[layer] is None else false_sums[layer] + h
        false_done += len(batch)
        false_pair_offset += len(batch_pairs)
        if _progress_report_due(false_done, len(false_statements), progress_every, false_last_reported):
            false_last_reported = min(false_done, len(false_statements))
            print(_format_progress("warmup false", false_last_reported, len(false_statements),
                                   time.perf_counter() - false_started))
        if checkpoint_path_obj and _progress_report_due(
            false_done, len(false_statements), checkpoint_every, false_last_checkpoint
        ):
            false_last_checkpoint = min(false_done, len(false_statements))
            save_warmup_checkpoint(
                checkpoint_path_obj,
                metadata=checkpoint_metadata,
                manifolds=manifolds,
                true_state_lists=true_state_lists,
                false_state_lists=false_state_lists,
                false_sums=false_sums,
                n_false=n_false,
                true_done=true_done,
                false_done=false_last_checkpoint,
            )
            print(f"   saved warmup checkpoint: {checkpoint_path_obj} "
                  f"(true={true_done}/{len(true_texts)}, false={false_last_checkpoint}/{len(false_statements)})")

    subspaces = {}
    for layer in layers:
        m = manifolds[layer]
        if n_false > 0 and m.mean is not None:
            m.false_mean = (false_sums[layer] / n_false).to(torch.float32)
            raw = m.mean - m.false_mean
            m.contrastive_direction = raw / torch.norm(raw).clamp(min=1e-8)

        true_states = true_state_lists[layer]
        false_states = false_state_lists[layer]
        if len(true_states) >= 2 and false_states:
            subspaces[layer] = TruthSubspace.fit_contrastive(
                torch.stack(true_states), torch.stack(false_states), rank=subspace_rank
            )
        elif len(true_states) >= 2:
            subspaces[layer] = TruthSubspace.fit(torch.stack(true_states), rank=subspace_rank)
    return manifolds, subspaces


# ---------------------------------------------------------------------------
# 主流程 / Main
# ---------------------------------------------------------------------------

def run(args) -> dict:
    # Windows 控制台默认 cp1252，确保非 ASCII 不崩溃；行缓冲让进度实时可见（被杀也不丢日志）
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

    profile: dict[str, float] = {}
    total_started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    batch_fallback_state = BatchSizeFallbackState(args.batch_size, enabled=args.auto_batch_size)
    max_batch_tokens = int(getattr(args, "max_batch_tokens", 0))
    inside_embedding_threshold = float(getattr(args, "inside_embedding_threshold", 0.90))
    inside_adaptive_sampling = bool(getattr(args, "inside_adaptive_sampling", False))
    inside_min_samples = int(getattr(args, "inside_min_samples", 2))
    inside_sample_step = int(getattr(args, "inside_sample_step", 1))
    inside_stability_delta = float(getattr(args, "inside_stability_delta", 0.05))
    dump_inside_samples = bool(getattr(args, "dump_inside_samples", False))
    inside_selfcheck_early_stop = bool(getattr(args, "inside_selfcheck_early_stop", False))
    inside_selfcheck_min_overlap = float(getattr(args, "inside_selfcheck_min_overlap", 0.65))
    inside_selfcheck_support_threshold = float(getattr(args, "inside_selfcheck_support_threshold", 0.60))
    inside_selfcheck_refute_threshold = float(getattr(args, "inside_selfcheck_refute_threshold", 0.50))

    stats_cache_path = Path(args.layer_stats_cache) if args.layer_stats_cache else None
    eval_reps_cache_path = Path(args.eval_reps_cache) if args.eval_reps_cache else None
    statement_encoding_cache_path = Path(args.statement_encoding_cache) if args.statement_encoding_cache else None
    inside_diagnostics_cache_path = (
        Path(getattr(args, "inside_diagnostics_cache", None))
        if getattr(args, "inside_diagnostics_cache", None)
        else None
    )
    warmup_checkpoint_path = Path(args.warmup_checkpoint) if args.warmup_checkpoint else None
    true_encodings: list[Optional[StatementEncoding]] | None = None
    false_encodings: list[Optional[StatementEncoding]] | None = None
    eval_encodings: list[Optional[StatementEncoding]] | None = None
    model = None
    tokenizer = None
    stats_meta: dict | None = None
    eval_meta: dict | None = None
    eval_reps_record_count: int | None = None
    restored_eval_statements = False
    if args.cache_only:
        if stats_cache_path is None or eval_reps_cache_path is None:
            raise ValueError("cache-only mode requires both layer-stats and eval-reps caches.")
        device = torch.device("cpu")
        with _profile_phase(profile, "read_cache_metadata"):
            stats_meta = _read_cache_metadata(stats_cache_path)
            eval_meta, eval_reps_record_count = _read_eval_reps_cache_metadata(eval_reps_cache_path)
            _validate_eval_reps_record_count(eval_meta, eval_reps_record_count)
        if stats_meta.get("n_layers") != eval_meta.get("n_layers"):
            raise ValueError("cache-only mode requires layer-stats and eval-reps caches with matching n_layers.")
        n_layers = int(stats_meta["n_layers"])
        cached_eval_stmts = _eval_statements_from_cache_metadata(
            eval_meta,
            expected_record_count=eval_reps_record_count,
        )
        if cached_eval_stmts is not None:
            manifold_true = [""] * int(stats_meta.get("n_true", 0))
            manifold_false = [""] * int(stats_meta.get("n_false", 0))
            eval_stmts = cached_eval_stmts
            restored_eval_statements = True
            print("Cache-only scoring: restored eval statements from eval reps cache metadata.")
        else:
            with _profile_phase(profile, "load_data"):
                if args.offline:
                    manifold_true, manifold_false, eval_stmts = load_offline()
                    print("[!] OFFLINE SMOKE MODE - pipeline check only, NOT a benchmark.\n")
                else:
                    manifold_true, manifold_false, eval_stmts = load_truthfulqa(
                        args.manifold_questions, args.limit
                    )
        print("Cache-only scoring: skipping model load and forced-answer forward.")
    else:
        with _profile_phase(profile, "load_data"):
            if args.offline:
                manifold_true, manifold_false, eval_stmts = load_offline()
                print("[!] OFFLINE SMOKE MODE - pipeline check only, NOT a benchmark.\n")
            else:
                manifold_true, manifold_false, eval_stmts = load_truthfulqa(
                    args.manifold_questions, args.limit
                )
        print(f"Loading {args.model} on {device} (dtype={args.dtype}) ...")
        with _profile_phase(profile, "load_model"):
            model, tokenizer = load_model(args.model, device, args.dtype)
        n_layers = int(model.config.num_hidden_layers)

    args.layer = resolve_target_layer(args.layer, n_layers, offline=args.offline)
    # --sweep: 主层在前；--sweep-layers 可限制候选层带，避免大模型全层后处理成本。
    layers = _resolve_sweep_layers(
        args.layer,
        n_layers,
        sweep=args.sweep,
        sweep_layers=args.sweep_layers,
    )
    if args.cache_only and stats_meta is not None and eval_meta is not None:
        _validate_cache_only_metadata(
            args=args,
            stats_metadata=stats_meta,
            eval_reps_metadata=eval_meta,
            layers=layers,
            n_layers=n_layers,
        )
    if model is not None and args.hidden_state_capture == "hooks":
        _hook_capture_layer_map(model, layers)
        print("Using hook-based hidden-state capture for selected non-final layers.")

    if statement_encoding_cache_path is not None and not args.cache_only:
        statement_encoding_cache_metadata = _statement_encoding_cache_metadata(
            args,
            true_texts=manifold_true,
            false_texts=manifold_false,
            eval_statements=eval_stmts,
        )
        if statement_encoding_cache_path.exists() and not args.refresh_statement_encoding_cache:
            with _profile_phase(profile, "load_statement_encoding_cache"):
                true_encodings, false_encodings, eval_encodings, _ = load_statement_encoding_cache(
                    statement_encoding_cache_path,
                    expected_metadata=statement_encoding_cache_metadata,
                )
            print(f"   loaded statement encoding cache: {statement_encoding_cache_path}")
        else:
            with _profile_phase(profile, "tokenize_statements"):
                true_encodings = _batch_statement_encodings(
                    tokenizer,
                    [Statement("", text, 0) for text in manifold_true],
                    args.max_length,
                )
                false_encodings = _batch_statement_encodings(
                    tokenizer,
                    [Statement("", text, 1) for text in manifold_false],
                    args.max_length,
                )
                eval_encodings = _batch_statement_encodings(tokenizer, eval_stmts, args.max_length)
            with _profile_phase(profile, "save_statement_encoding_cache"):
                save_statement_encoding_cache(
                    statement_encoding_cache_path,
                    metadata=statement_encoding_cache_metadata,
                    true_encodings=true_encodings,
                    false_encodings=false_encodings,
                    eval_encodings=eval_encodings,
                )
            print(f"   saved statement encoding cache: {statement_encoding_cache_path}")
    elif max_batch_tokens > 0 and not args.cache_only:
        with _profile_phase(profile, "tokenize_statements"):
            true_encodings = _batch_statement_encodings(
                tokenizer,
                [Statement("", text, 0) for text in manifold_true],
                args.max_length,
            )
            false_encodings = _batch_statement_encodings(
                tokenizer,
                [Statement("", text, 1) for text in manifold_false],
                args.max_length,
            )
            eval_encodings = _batch_statement_encodings(tokenizer, eval_stmts, args.max_length)
        print("   tokenized statements for token-budget batching")

    print(f"Building per-layer truth stats from {len(manifold_true)} true / "
          f"{len(manifold_false)} false statements ({len(layers)} layer(s)) ...")
    if restored_eval_statements and stats_meta is not None:
        stats_cache_metadata = dict(stats_meta)
    else:
        stats_cache_metadata = _layer_stats_cache_metadata(
            args,
            layers=layers,
            n_layers=n_layers,
            true_texts=manifold_true,
            false_texts=manifold_false,
        )
    if stats_cache_path and stats_cache_path.exists() and not args.refresh_layer_stats_cache:
        with _profile_phase(profile, "load_layer_stats_cache"):
            manifolds, subspaces, _ = load_layer_stats_cache(
                stats_cache_path,
                expected_metadata=stats_cache_metadata,
                device=device,
            )
        print(f"   loaded layer stats cache: {stats_cache_path}")
    else:
        with _profile_phase(profile, "build_layer_stats"):
            manifolds, subspaces = build_layer_stats(
                model, tokenizer, manifold_true, manifold_false, layers, device, args.max_length,
                args.subspace_rank, args.batch_size, args.length_bucketed_batches,
                max_batch_tokens=max_batch_tokens,
                progress_every=args.progress_every,
                checkpoint_path=warmup_checkpoint_path,
                checkpoint_metadata=stats_cache_metadata,
                resume_checkpoint=not args.refresh_layer_stats_cache,
                checkpoint_every=args.warmup_checkpoint_every,
                hidden_state_capture=args.hidden_state_capture,
                covariance_mode=getattr(args, "covariance_mode", "full"),
                covariance_low_rank=getattr(args, "covariance_low_rank", 16),
                true_encodings=true_encodings,
                false_encodings=false_encodings,
                batch_fallback_state=batch_fallback_state,
            )
        if stats_cache_path:
            with _profile_phase(profile, "save_layer_stats_cache"):
                save_layer_stats_cache(
                    stats_cache_path,
                    manifolds,
                    subspaces,
                    metadata=stats_cache_metadata,
                )
            print(f"   saved layer stats cache: {stats_cache_path}")
    primary = manifolds[args.layer]
    if not primary.is_ready():
        print("[X] Manifold not ready (need >=2 statements). Aborting.")
        sys.exit(1)
    print(f"   manifold: n={primary.n}, hidden_dim={primary.hidden_dim}, "
          f"covariance={primary.covariance_mode}, "
          f"contrastive_direction={'yes' if primary.contrastive_direction is not None else 'no'}  "
          f"subspace={'yes' if args.layer in subspaces else 'no'}\n")
    layer_spectra = None
    if bool(getattr(args, "include_layer_spectra", False)):
        with _profile_phase(profile, "spectrum_reporting"):
            layer_spectra = _layer_spectrum_reports(
                manifolds,
                top_k=int(getattr(args, "layer_spectrum_top_k", 16)),
            )

    signals = _enabled_signals(args)
    sweep_signal_names = _sweep_signal_names(args)
    scores: dict[str, List[float]] = {s: [] for s in signals}
    sweep_scores: dict = {
        layer: {signal: [] for signal in sweep_signal_names} for layer in layers
    }
    labels: List[int] = []
    scored_statements: list[dict[str, object]] = []
    inside_sampled: List[bool] = []
    inside_sample_counts: List[int] = []
    inside_adaptive_rounds: List[int] = []
    inside_stopped_early: List[bool] = []
    inside_stop_reasons: list[str | None] = []
    inside_sample_texts: list[list[str]] = []
    inside_sample_logprobs: list[list[float]] = []
    scored_batch_indexes: list[int] = []
    inside_triggered_total = 0
    inside_skipped_total = 0

    print(f"Scoring {len(eval_stmts)} eval statements ...")
    scored = 0
    eval_pairs = _statement_pairs(
        eval_stmts,
        eval_encodings,
        length_bucketed=args.length_bucketed_batches,
    )
    expected_eval_records = len(eval_pairs)
    if restored_eval_statements and eval_meta is not None:
        eval_reps_cache_metadata = dict(eval_meta)
    else:
        eval_reps_cache_metadata = _eval_reps_cache_metadata(
            args,
            layers=layers,
            n_layers=n_layers,
            eval_statements=eval_stmts,
        )
    eval_reps_reader: EvalRepsCacheReader | None = None
    eval_reps_writer: EvalRepsCacheWriter | None = None
    new_eval_reps: list[Optional[Mapping]] = []
    inside_diagnostics_cache: InsideDiagnosticsCache | None = None
    if eval_reps_cache_path and eval_reps_cache_path.exists() and not args.refresh_eval_reps_cache:
        with _profile_phase(profile, "load_eval_reps_cache"):
            eval_reps_reader = EvalRepsCacheReader(
                eval_reps_cache_path,
                expected_metadata=eval_reps_cache_metadata,
                expected_records=expected_eval_records,
                shard_read_cache_size=int(getattr(args, "eval_reps_shard_read_cache_size", 2)),
            )
        print(f"   loaded eval reps cache: {eval_reps_cache_path}")
    elif eval_reps_cache_path:
        write_shard_size = int(args.eval_reps_cache_shard_size)
        if write_shard_size <= 0 and args.refresh_eval_reps_cache:
            write_shard_size = _existing_eval_reps_shard_size(eval_reps_cache_path)
        if write_shard_size > 0:
            with _profile_phase(profile, "init_eval_reps_cache_writer"):
                eval_reps_writer = EvalRepsCacheWriter(
                    eval_reps_cache_path,
                    metadata=eval_reps_cache_metadata,
                    shard_size=write_shard_size,
                )
    if inside_diagnostics_cache_path and _inside_enabled(args):
        with _profile_phase(profile, "load_inside_diagnostics_cache"):
            inside_diagnostics_cache = InsideDiagnosticsCache(
                inside_diagnostics_cache_path,
                refresh=bool(getattr(args, "refresh_inside_diagnostics_cache", False)),
            )

    eval_reps_offset = 0
    eval_pair_offset = 0
    eval_batch_idx = 0
    eval_last_reported = 0
    eval_started = time.perf_counter()
    while eval_pair_offset < len(eval_pairs):
        current_batch_size = batch_fallback_state.batch_size()
        batch_pairs = _next_statement_pair_batch(
            eval_pairs,
            eval_pair_offset,
            current_batch_size,
            max_batch_tokens=max_batch_tokens,
        )
        batch = [stmt for stmt, _encoding in batch_pairs]
        if eval_reps_reader is not None:
            with _profile_phase(profile, "read_eval_reps_cache_batch"):
                reps_batch = eval_reps_reader.read_range(eval_reps_offset, len(batch))
        else:
            with _profile_phase(profile, "forced_answer_forward"):
                reps_batch = _batched_statement_reps_for_pairs(
                    model,
                    tokenizer,
                    batch_pairs,
                    layers,
                    device,
                    args.max_length,
                    eigenscore_alpha=args.eigenscore_alpha,
                    encoded_statements_provided=eval_encodings is not None,
                    fallback_state=batch_fallback_state,
                    phase="forced_answer_forward",
                    hidden_state_capture=args.hidden_state_capture,
                    prefix_kv_cache=bool(getattr(args, "prefix_kv_cache", False)),
                    first_token_top_k=int(getattr(args, "first_token_top_k", FIRST_TOKEN_TOP_K_DEFAULT)),
                )
            if eval_reps_writer is not None:
                with _profile_phase(profile, "write_eval_reps_cache_batch"):
                    eval_reps_writer.extend(reps_batch)
            elif eval_reps_cache_path:
                new_eval_reps.extend(reps_batch)
        eval_reps_offset += len(batch)

        with _profile_phase(profile, "score_postprocess"):
            batch_records = _score_reps_batch(
                batch,
                reps_batch,
                layers=layers,
                target_layer=args.layer,
                manifolds=manifolds,
                subspaces=subspaces,
            )

        if _inside_enabled(args):
            if _inside_trigger_enabled(args):
                triggered = _inside_trigger_indexes(batch_records, args)
            else:
                triggered = set(range(len(batch_records)))
            inside_triggered_total += len(triggered)
            inside_skipped_total += len(batch_records) - len(triggered)

            triggered_positions = [idx for idx in range(len(batch_records)) if idx in triggered]
            sampled_by_position: dict[int, SampledInsideDiagnostics | None] = {}
            cache_keys_by_position: dict[int, str] = {}
            missing_position_batches: list[tuple[list[int], list[Statement], int]] = []
            if inside_diagnostics_cache is not None:
                with _profile_phase(profile, "read_inside_diagnostics_cache"):
                    for position_batch in _chunked(triggered_positions, args.inside_batch_size):
                        inside_batch = [batch_records[position]["stmt"] for position in position_batch]
                        batch_seed = (
                            _inside_statement_seed(args.seed, inside_batch[0])
                            if len(inside_batch) == 1
                            else _inside_statement_batch_seed(args.seed, inside_batch)
                        )
                        missing_in_batch = False
                        for batch_position, position in enumerate(position_batch):
                            batch_context = (
                                None
                                if len(inside_batch) == 1
                                else _inside_batch_cache_context(
                                    args.seed,
                                    inside_batch,
                                    batch_position=batch_position,
                                )
                            )
                            cache_key = _inside_diagnostics_cache_key(
                                batch_records[position]["stmt"],
                                args,
                                layers=layers,
                                adaptive=inside_adaptive_sampling,
                                selfcheck_early_stop=inside_selfcheck_early_stop,
                                batch_cache_context=batch_context,
                            )
                            cache_keys_by_position[position] = cache_key
                            cache_lookup_keys = [cache_key]
                            if batch_context is not None:
                                cache_lookup_keys.append(_inside_diagnostics_cache_key(
                                    batch_records[position]["stmt"],
                                    args,
                                    layers=layers,
                                    adaptive=inside_adaptive_sampling,
                                    selfcheck_early_stop=inside_selfcheck_early_stop,
                                ))
                            cached = inside_diagnostics_cache.get_any(cache_lookup_keys)
                            if cached is None:
                                missing_in_batch = True
                            else:
                                sampled_by_position[position] = cached
                        if missing_in_batch:
                            missing_position_batches.append((list(position_batch), inside_batch, batch_seed))

            if inside_diagnostics_cache is not None:
                for position_batch, inside_batch, seed in missing_position_batches:
                    with _profile_phase(profile, "inside_generation"):
                        sampled_batch = _sample_inside_diagnostics_for_args(
                            model,
                            tokenizer,
                            inside_batch,
                            layers,
                            device,
                            args,
                            inside_adaptive_sampling=inside_adaptive_sampling,
                            inside_min_samples=inside_min_samples,
                            inside_sample_step=inside_sample_step,
                            inside_stability_delta=inside_stability_delta,
                            inside_embedding_threshold=inside_embedding_threshold,
                            inside_selfcheck_early_stop=inside_selfcheck_early_stop,
                            inside_selfcheck_min_overlap=inside_selfcheck_min_overlap,
                            inside_selfcheck_support_threshold=inside_selfcheck_support_threshold,
                            inside_selfcheck_refute_threshold=inside_selfcheck_refute_threshold,
                            seed=seed,
                        )
                    with _profile_phase(profile, "write_inside_diagnostics_cache"):
                        for position, sampled in zip(position_batch, sampled_batch):
                            sampled_by_position.setdefault(position, sampled)
                            inside_diagnostics_cache.put(cache_keys_by_position[position], sampled)
            else:
                inside_position_batches = _chunked(triggered_positions, args.inside_batch_size)
                for inside_batch_idx, position_batch in enumerate(inside_position_batches):
                    inside_batch = [batch_records[position]["stmt"] for position in position_batch]
                    with _profile_phase(profile, "inside_generation"):
                        sampled_batch = _sample_inside_diagnostics_for_args(
                            model,
                            tokenizer,
                            inside_batch,
                            layers,
                            device,
                            args,
                            inside_adaptive_sampling=inside_adaptive_sampling,
                            inside_min_samples=inside_min_samples,
                            inside_sample_step=inside_sample_step,
                            inside_stability_delta=inside_stability_delta,
                            inside_embedding_threshold=inside_embedding_threshold,
                            inside_selfcheck_early_stop=inside_selfcheck_early_stop,
                            inside_selfcheck_min_overlap=inside_selfcheck_min_overlap,
                            inside_selfcheck_support_threshold=inside_selfcheck_support_threshold,
                            inside_selfcheck_refute_threshold=inside_selfcheck_refute_threshold,
                            seed=_inside_seed(args.seed, eval_batch_idx, inside_batch_idx),
                        )
                    for position, sampled in zip(position_batch, sampled_batch):
                        sampled_by_position[position] = sampled

            for position in triggered_positions:
                sampled = sampled_by_position.get(position)
                batch_records[position]["inside_scores"] = (
                    sampled.eigenscore_by_layer if sampled is not None else None
                )
                batch_records[position]["inside_semantic_entropy"] = (
                    sampled.semantic_entropy if sampled is not None else None
                )
                batch_records[position]["inside_embedding_entropy"] = (
                    sampled.embedding_entropy_by_layer if sampled is not None else None
                )
                batch_records[position]["inside_semantic_energy"] = (
                    sampled.semantic_energy if sampled is not None else None
                )
                batch_records[position]["inside_sample_count"] = sampled.n_samples if sampled is not None else 0
                batch_records[position]["inside_adaptive_rounds"] = (
                    sampled.adaptive_rounds if sampled is not None else 0
                )
                batch_records[position]["inside_stopped_early"] = (
                    sampled.stopped_early if sampled is not None else False
                )
                batch_records[position]["inside_stop_reason"] = (
                    sampled.stop_reason if sampled is not None else None
                )
                batch_records[position]["inside_sample_texts"] = (
                    tuple(sampled.sample_texts) if sampled is not None else ()
                )
                batch_records[position]["inside_sample_logprobs"] = (
                    tuple(sampled.sample_logprobs) if sampled is not None else ()
                )
                batch_records[position]["inside_sampled"] = sampled is not None

            for position, record in enumerate(batch_records):
                if position not in triggered:
                    record["inside_scores"] = _empty_inside_scores(layers)
                    record["inside_semantic_entropy"] = 0.0
                    record["inside_embedding_entropy"] = _empty_inside_scores(layers)
                    record["inside_semantic_energy"] = 0.0
                    record["inside_sample_count"] = 0
                    record["inside_adaptive_rounds"] = 0
                    record["inside_stopped_early"] = False
                    record["inside_stop_reason"] = None
                    record["inside_sample_texts"] = ()
                    record["inside_sample_logprobs"] = ()

        with _profile_phase(profile, "score_postprocess"):
            for record in batch_records:
                inside_scores = record["inside_scores"]
                inside_embedding_entropy = record["inside_embedding_entropy"]
                inside_semantic_energy = record["inside_semantic_energy"]
                if _inside_enabled(args) and (inside_scores is None or inside_embedding_entropy is None):
                    continue
                inside_entropy = record["inside_semantic_entropy"]
                if inside_scores is not None:
                    inside_entropy = 0.0 if inside_entropy is None else float(inside_entropy)
                    inside_semantic_energy = (
                        0.0 if inside_semantic_energy is None else float(inside_semantic_energy)
                    )

                for layer in layers:
                    layer_scores = record["layer_scores"][layer]
                    sweep_scores[layer]["maha_last"].append(layer_scores["maha_last"])
                    sweep_scores[layer]["truth_proj"].append(layer_scores["truth_proj"])
                    sweep_scores[layer]["subspace_resid"].append(layer_scores["subspace_resid"])
                    sweep_scores[layer]["resid_update_norm"].append(layer_scores["resid_update_norm"])
                    sweep_scores[layer]["eigenscore"].append(layer_scores["eigenscore"])
                    if inside_scores is not None:
                        sweep_scores[layer][INSIDE_SIGNAL].append(float(inside_scores[layer]))
                        sweep_scores[layer][INSIDE_SEMANTIC_ENTROPY_SIGNAL].append(float(inside_entropy))
                        sweep_scores[layer][INSIDE_EMBEDDING_ENTROPY_SIGNAL].append(
                            float(inside_embedding_entropy[layer])
                        )
                        sweep_scores[layer][INSIDE_SEMANTIC_ENERGY_SIGNAL].append(
                            float(inside_semantic_energy)
                        )

                primary_scores = record["primary_scores"]
                scores["maha_last"].append(primary_scores["maha_last"])
                scores["truth_proj"].append(primary_scores["truth_proj"])
                scores["subspace_resid"].append(primary_scores["subspace_resid"])
                scores["resid_update_norm"].append(primary_scores["resid_update_norm"])
                scores["disp_euclid"].append(primary_scores["disp_euclid"])
                scores["disp_hse"].append(primary_scores["disp_hse"])
                scores["eigenscore"].append(primary_scores["eigenscore"])
                scores[FIRST_TOKEN_ENTROPY_SIGNAL].append(primary_scores[FIRST_TOKEN_ENTROPY_SIGNAL])
                if inside_scores is not None:
                    scores[INSIDE_SIGNAL].append(sweep_scores[args.layer][INSIDE_SIGNAL][-1])
                    scores[INSIDE_SEMANTIC_ENTROPY_SIGNAL].append(float(inside_entropy))
                    scores[INSIDE_EMBEDDING_ENTROPY_SIGNAL].append(
                        sweep_scores[args.layer][INSIDE_EMBEDDING_ENTROPY_SIGNAL][-1]
                    )
                    scores[INSIDE_SEMANTIC_ENERGY_SIGNAL].append(float(inside_semantic_energy))
                scores["nll_answer"].append(primary_scores["nll_answer"])
                labels.append(record["stmt"].is_false)
                scored_statements.append(_statement_to_dump(record["stmt"]))
                scored_batch_indexes.append(int(eval_batch_idx))
                if _inside_enabled(args):
                    inside_sampled.append(bool(record["inside_sampled"]))
                    inside_sample_counts.append(int(record.get("inside_sample_count", 0)))
                    inside_adaptive_rounds.append(int(record.get("inside_adaptive_rounds", 0)))
                    inside_stopped_early.append(bool(record.get("inside_stopped_early", False)))
                    inside_stop_reasons.append(record.get("inside_stop_reason"))
                    if dump_inside_samples:
                        inside_sample_texts.append(list(record.get("inside_sample_texts", ())))
                        inside_sample_logprobs.append(list(record.get("inside_sample_logprobs", ())))
                scored += 1

                if _progress_report_due(scored, len(eval_stmts), args.progress_every, eval_last_reported):
                    eval_last_reported = min(scored, len(eval_stmts))
                    print(_format_progress("eval", eval_last_reported, len(eval_stmts),
                                           time.perf_counter() - eval_started))
        eval_pair_offset += len(batch_pairs)
        eval_batch_idx += 1

    if eval_reps_writer is not None:
        with _profile_phase(profile, "save_eval_reps_cache"):
            eval_reps_writer.close()
        print(f"   saved sharded eval reps cache: {eval_reps_cache_path}")
    elif eval_reps_cache_path and eval_reps_reader is None:
        with _profile_phase(profile, "save_eval_reps_cache"):
            save_eval_reps_cache(
                eval_reps_cache_path,
                new_eval_reps,
                metadata=eval_reps_cache_metadata,
            )
        print(f"   saved eval reps cache: {eval_reps_cache_path}")
    if inside_diagnostics_cache is not None:
        with _profile_phase(profile, "save_inside_diagnostics_cache"):
            inside_diagnostics_cache.save()

    with _profile_phase(profile, "reporting"):
        n_pos = sum(labels)
        n_neg = len(labels) - n_pos
        results = {s: roc_auc(scores[s], labels) for s in signals}
        selective = _selective_reports(scores, labels, alpha=REPORT_ALPHA)

    # ---- 输出 ----
    print("\n" + "=" * 56)
    print("  AUROC  (positive = false/hallucinated statement)")
    print(f"  model={args.model}  layer={args.layer}  n_pos={n_pos}  n_neg={n_neg}")
    print("=" * 56)
    print(f"  {'signal':<18}{'AUROC':>10}   interpretation")
    print("  " + "-" * 52)
    for s in signals:
        print(f"  {s:<18}{results[s]:>10.3f}")
    print("  " + "-" * 52)
    # 关键对比 / key comparisons
    if not (results["disp_hse"] != results["disp_hse"]):  # not NaN
        delta = results["disp_hse"] - results["disp_euclid"]
        verdict = "hyperbolic HELPS" if delta > 0.01 else (
            "hyperbolic HURTS" if delta < -0.01 else "no meaningful difference")
        print(f"  disp_hse - disp_euclid = {delta:+.3f}  ->  {verdict}")
    geo = max(results[s] for s in signals if s not in NON_GEOMETRY_BASELINE_SIGNALS)
    if not (results["nll_answer"] != results["nll_answer"]):
        print(f"  best geometry ({geo:.3f}) vs nll baseline ({results['nll_answer']:.3f})  ->  "
              f"{'geometry wins' if geo > results['nll_answer'] + 0.01 else 'baseline competitive'}")
    first_token_auc = results.get(FIRST_TOKEN_ENTROPY_SIGNAL)
    if first_token_auc is not None and not (first_token_auc != first_token_auc):
        print(
            f"  first-token entropy baseline AUROC = {first_token_auc:.3f}  "
            "(single-decode uncertainty)"
        )
    print("=" * 56)

    sweep_payload = None
    with _profile_phase(profile, "sweep_reporting"):
        if _sweep_output_enabled(args):
            sweep_payload = {}
            print("\n  Layer sweep (AUROC):")
            header = f"  {'layer':>6}" + "".join(f" {name:>17}" for name in sweep_signal_names)
            print(header)
            for layer in sorted(layers):
                layer_payload = {
                    signal: roc_auc(sweep_scores[layer][signal], labels)
                    for signal in sweep_signal_names
                }
                sweep_payload[str(layer)] = layer_payload
                values = "".join(f" {layer_payload[signal]:>17.3f}" for signal in sweep_signal_names)
                print(f"  {layer:>6}{values}")

    payload = {
        "config": {"model": args.model, "dataset": "truthfulqa",
                   "dtype": args.dtype, "layer": args.layer,
                   "offline": args.offline, "max_length": args.max_length,
                   "manifold_n": primary.n, "n_manifold_false": len(manifold_false),
                   "hidden_dim": primary.hidden_dim, "subspace_rank": args.subspace_rank,
                   "n_pos": n_pos, "n_neg": n_neg, "seed": args.seed,
                   "eigenscore_alpha": args.eigenscore_alpha,
                   "first_token_top_k": int(getattr(args, "first_token_top_k", FIRST_TOKEN_TOP_K_DEFAULT)),
                   "batch_size": args.batch_size,
                   "max_batch_tokens": max_batch_tokens,
                   "auto_batch_size": args.auto_batch_size,
                   "effective_batch_size": batch_fallback_state.batch_size(),
                   "inside_samples": args.inside_samples,
                   "inside_batch_size": args.inside_batch_size,
                   "inside_max_new_tokens": args.inside_max_new_tokens,
                   "inside_temperature": args.inside_temperature,
                   "inside_top_p": args.inside_top_p,
                   "inside_pooling": args.inside_pooling,
                   "inside_embedding_threshold": inside_embedding_threshold,
                   "inside_adaptive_sampling": inside_adaptive_sampling,
                   "inside_min_samples": inside_min_samples,
                   "inside_sample_step": inside_sample_step,
                   "inside_stability_delta": inside_stability_delta,
                   "dump_inside_samples": dump_inside_samples,
                   "inside_selfcheck_early_stop": inside_selfcheck_early_stop,
                   "inside_selfcheck_min_overlap": inside_selfcheck_min_overlap,
                   "inside_selfcheck_support_threshold": inside_selfcheck_support_threshold,
                   "inside_selfcheck_refute_threshold": inside_selfcheck_refute_threshold,
                   "inside_trigger_signal": args.inside_trigger_signal,
                   "inside_trigger_threshold": args.inside_trigger_threshold,
                   "inside_trigger_top_fraction": args.inside_trigger_top_fraction,
                   "length_bucketed_batches": args.length_bucketed_batches,
                   "hidden_state_capture": args.hidden_state_capture,
                   "prefix_kv_cache": bool(getattr(args, "prefix_kv_cache", False)),
                   "cache_only": args.cache_only,
                   "cache_only_restored_eval_statements": restored_eval_statements,
                   "statement_encoding_cache": args.statement_encoding_cache,
                   "refresh_statement_encoding_cache": args.refresh_statement_encoding_cache,
                   "layer_stats_cache": args.layer_stats_cache,
                   "refresh_layer_stats_cache": args.refresh_layer_stats_cache,
                   "eval_reps_cache": args.eval_reps_cache,
                   "eval_reps_cache_shard_size": args.eval_reps_cache_shard_size,
                   "eval_reps_shard_read_cache_size": int(
                       getattr(args, "eval_reps_shard_read_cache_size", 2)
                   ),
                   "refresh_eval_reps_cache": args.refresh_eval_reps_cache,
                   "inside_diagnostics_cache": getattr(args, "inside_diagnostics_cache", None),
                   "refresh_inside_diagnostics_cache": bool(
                       getattr(args, "refresh_inside_diagnostics_cache", False)
                   ),
                   "progress_every": args.progress_every,
                   "warmup_checkpoint": args.warmup_checkpoint,
                   "warmup_checkpoint_every": args.warmup_checkpoint_every,
                   "covariance_mode": getattr(args, "covariance_mode", "full"),
                   "covariance_low_rank": getattr(args, "covariance_low_rank", 16),
                   "include_layer_spectra": bool(getattr(args, "include_layer_spectra", False)),
                   "layer_spectrum_top_k": int(getattr(args, "layer_spectrum_top_k", 16)),
                   "dump_scores_format": getattr(args, "dump_scores_format", "json"),
                   "sweep_layers": layers if (args.sweep or args.sweep_layers) else None},
        "auroc": results,
        "selective": selective,
        "sweep": sweep_payload,
        "batch_size_fallback": batch_fallback_state.to_dict(),
    }
    if layer_spectra is not None:
        payload["layer_spectra"] = layer_spectra
    if _inside_enabled(args):
        sampled_count = int(sum(inside_sampled))
        total_sample_count = int(sum(inside_sample_counts))
        stop_reason_counts: dict[str, int] = {}
        for reason in inside_stop_reasons:
            if reason is None:
                continue
            stop_reason_counts[str(reason)] = stop_reason_counts.get(str(reason), 0) + 1
        payload["inside_sampling"] = {
            "mode": "triggered" if _inside_trigger_enabled(args) else "all",
            "adaptive": inside_adaptive_sampling,
            "signals": list(INSIDE_SIGNALS),
            "signal": args.inside_trigger_signal,
            "threshold": args.inside_trigger_threshold,
            "top_fraction": args.inside_trigger_top_fraction,
            "embedding_similarity_threshold": inside_embedding_threshold,
            "min_samples": inside_min_samples if inside_adaptive_sampling else args.inside_samples,
            "max_samples": args.inside_samples,
            "sample_step": inside_sample_step if inside_adaptive_sampling else None,
            "stability_delta": inside_stability_delta if inside_adaptive_sampling else None,
            "selfcheck_early_stop": inside_selfcheck_early_stop if inside_adaptive_sampling else False,
            "selfcheck_min_overlap": (
                inside_selfcheck_min_overlap if inside_adaptive_sampling and inside_selfcheck_early_stop else None
            ),
            "selfcheck_support_threshold": (
                inside_selfcheck_support_threshold if inside_adaptive_sampling and inside_selfcheck_early_stop else None
            ),
            "selfcheck_refute_threshold": (
                inside_selfcheck_refute_threshold if inside_adaptive_sampling and inside_selfcheck_early_stop else None
            ),
            "sampled": sampled_count,
            "not_sampled": int(len(inside_sampled) - sampled_count),
            "triggered": int(inside_triggered_total),
            "skipped_by_trigger": int(inside_skipped_total),
            "total_generated_samples": total_sample_count,
            "mean_samples_per_record": (
                total_sample_count / len(inside_sample_counts) if inside_sample_counts else 0.0
            ),
            "mean_samples_per_sampled_record": (total_sample_count / sampled_count) if sampled_count else 0.0,
            "stopped_early": int(sum(inside_stopped_early)),
            "stop_reason_counts": stop_reason_counts,
            "fill_value_for_untriggered": 0.0 if _inside_trigger_enabled(args) else None,
        }
    cache_stats = {}
    if eval_reps_reader is not None:
        cache_stats["eval_reps_reader"] = eval_reps_reader.cache_stats()
    if inside_diagnostics_cache is not None:
        cache_stats["inside_diagnostics"] = inside_diagnostics_cache.stats()
    if cache_stats:
        payload["cache_stats"] = cache_stats
    if _profile_requested(args):
        payload["profile"] = _profile_payload(
            profile,
            time.perf_counter() - total_started,
            n_eval_records=len(labels),
            n_warmup_true=len(manifold_true),
            n_warmup_false=len(manifold_false),
            cache_stats=cache_stats,
        )
        payload["profile"]["batch_size_fallback"] = batch_fallback_state.to_dict()
        print("\n  Profile timings (seconds):")
        for name, seconds in payload["profile"]["phases"].items():
            print(f"  {name:<24}{seconds:>10.3f}")
        print(f"  {'total':<24}{payload['profile']['total_seconds']:>10.3f}")
        summary = payload["profile"]["summary"]
        bottleneck = summary.get("bottleneck")
        if bottleneck:
            print(f"  {'bottleneck':<24}{bottleneck:>10}")
        model_forward = summary["groups"]["model_forward"]
        if model_forward["seconds"] > 0:
            print(f"  {'model_forward_share':<24}{model_forward['share']:>10.3f}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote structured results to {args.json}")
    if args.profile_json:
        profile_data = payload.get(
            "profile",
            _profile_payload(
                profile,
                time.perf_counter() - total_started,
                n_eval_records=len(labels),
                n_warmup_true=len(manifold_true),
                n_warmup_false=len(manifold_false),
                cache_stats=cache_stats,
            ),
        )
        with open(args.profile_json, "w", encoding="utf-8") as f:
            json.dump(profile_data, f, indent=2)
        print(f"\nWrote profile timings to {args.profile_json}")
    if args.dump_scores:
        # 逐陈述原始分数：供共形校准等后处理复用，无需再跑模型
        # Raw per-statement scores: enables post-hoc analyses (e.g. conformal
        # calibration) without re-running the model
        dump = {
            "config": payload["config"],
            "labels": labels,
            "scores": scores,
            "statements": scored_statements,
            "batch_indexes": scored_batch_indexes,
        }
        if _inside_enabled(args):
            dump["inside_sampled"] = inside_sampled
            dump["inside_sample_counts"] = inside_sample_counts
            dump["inside_adaptive_rounds"] = inside_adaptive_rounds
            dump["inside_stopped_early"] = inside_stopped_early
            dump["inside_stop_reasons"] = inside_stop_reasons
            dump["inside_sampling"] = payload["inside_sampling"]
            if dump_inside_samples:
                dump["inside_sample_texts"] = inside_sample_texts
                dump["inside_sample_logprobs"] = inside_sample_logprobs
        if _sweep_output_enabled(args):
            dump["sweep_scores"] = {str(layer): sweep_scores[layer] for layer in layers}
        dump_format = getattr(args, "dump_scores_format", "json")
        _write_score_dump(args.dump_scores, dump, dump_format)
        noun = "manifest" if dump_format == "jsonl" else "scores"
        print(f"Dumped raw per-statement {noun} to {args.dump_scores}")
    print("\nJSON:", json.dumps(payload["auroc"]))
    return payload


def main():
    p = argparse.ArgumentParser(description="EigenTruth TruthfulQA AUROC benchmark")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--dtype", default="float32", choices=list(_DTYPES),
                   help="model weight dtype; bfloat16 halves memory on low-RAM machines")
    p.add_argument("--layer", type=int, default=-8, help="target layer index (negative ok)")
    p.add_argument("--sweep", action="store_true",
                   help="score geometry signals at every layer; with default hidden-state capture, "
                        "one forward pass returns all hidden states")
    p.add_argument("--sweep-layers", default=None,
                   help="comma-list of layer indexes to score in addition to --layer; "
                        "limits sweep cost and implies a sweep payload")
    p.add_argument("--limit", type=int, default=200, help="max eval questions (0 = all)")
    p.add_argument("--manifold-questions", type=int, default=80,
                   help="held-out questions whose correct answers build the manifold")
    p.add_argument("--max-length", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=1,
                   help="forced-answer forward batch size; increase for faster benchmarks if memory allows")
    p.add_argument("--max-batch-tokens", type=int, default=0,
                   help="optional padded-token budget per warmup/eval forward batch; 0 disables token-budget "
                        "batch splitting while preserving --batch-size as the row-count cap")
    p.add_argument("--auto-batch-size", action="store_true",
                   help="on retriable memory errors during warmup/forced-answer forwards, halve batch size and retry")
    p.add_argument("--length-bucketed-batches", action="store_true",
                   help="sort statements by approximate text length before batching to reduce padding")
    p.add_argument("--hidden-state-capture", default="outputs", choices=HIDDEN_STATE_CAPTURE_METHODS,
                   help="how forced-answer forwards collect hidden states: 'outputs' preserves exact "
                        "HF output_hidden_states semantics; 'hooks' stores only selected non-final "
                        "decoder-layer states and can reduce memory pressure")
    p.add_argument("--prefix-kv-cache", action="store_true",
                   help="experimental: reuse one question-prefix KV cache per shared prefix during eval "
                        "forced-answer scoring; requires --hidden-state-capture outputs")
    p.add_argument("--subspace-rank", type=int, default=2,
                   help="rank for TruthSubspace residual scoring")
    p.add_argument("--covariance-mode", default="full", choices=COVARIANCE_MODES,
                   help="TruthManifold covariance approximation for maha_last: "
                        "full, diag, low_rank, or shrinkage")
    p.add_argument("--covariance-low-rank", type=int, default=16,
                   help="rank used when --covariance-mode low_rank")
    p.add_argument("--include-layer-spectra", action="store_true",
                   help="include compact per-layer covariance-spectrum diagnostics in the JSON report; "
                        "off by default because full eigendecomposition can be expensive on large layers")
    p.add_argument("--layer-spectrum-top-k", type=int, default=16,
                   help="number of largest covariance eigenvalues to include when --include-layer-spectra is set")
    p.add_argument("--eigenscore-alpha", type=float, default=1e-3,
                   help="regularization alpha for EigenScore-style log-det scores")
    p.add_argument("--first-token-top-k", type=int, default=FIRST_TOKEN_TOP_K_DEFAULT,
                   help="top-k logits used for first_token_entropy; higher entropy is treated as anomalous")
    p.add_argument("--inside-samples", type=int, default=0,
                   help="enable multi-sample INSIDE proxy with this many sampled continuations; "
                        "0 disables it, values >=2 enable inside_eigenscore, inside_semantic_entropy, "
                        "inside_embedding_entropy, and inside_semantic_energy")
    p.add_argument("--inside-batch-size", type=int, default=1,
                   help="number of prompts to sample in one generate() call for --inside-samples")
    p.add_argument("--inside-max-new-tokens", type=int, default=12,
                   help="max sampled continuation length for --inside-samples")
    p.add_argument("--inside-temperature", type=float, default=0.7,
                   help="sampling temperature for --inside-samples")
    p.add_argument("--inside-top-p", type=float, default=0.9,
                   help="top-p sampling cutoff for --inside-samples")
    p.add_argument("--inside-pooling", default="last", choices=("last", "mean"),
                   help="sentence embedding pooling for sampled continuations")
    p.add_argument("--inside-embedding-threshold", type=float, default=0.90,
                   help="cosine similarity threshold for inside_embedding_entropy clusters")
    p.add_argument("--inside-adaptive-sampling", action="store_true",
                   help="adaptively sample INSIDE continuations until semantic scores stabilize or "
                        "--inside-samples is reached")
    p.add_argument("--inside-min-samples", type=int, default=2,
                   help="minimum sampled continuations before adaptive INSIDE early-stop checks")
    p.add_argument("--inside-sample-step", type=int, default=1,
                   help="additional continuations per adaptive INSIDE round after --inside-min-samples")
    p.add_argument("--inside-stability-delta", type=float, default=0.05,
                   help="early-stop when lexical and embedding entropy changes are at most this value")
    p.add_argument("--inside-selfcheck-early-stop", action="store_true",
                   help="with --inside-adaptive-sampling, stop sampled generation once self-consistency "
                        "threshold bounds cannot change the final support/refute/insufficient outcome")
    p.add_argument("--inside-selfcheck-min-overlap", type=float, default=0.65,
                   help="minimum claim/sample token overlap for --inside-selfcheck-early-stop")
    p.add_argument("--inside-selfcheck-support-threshold", type=float, default=0.60,
                   help="support-rate threshold for --inside-selfcheck-early-stop")
    p.add_argument("--inside-selfcheck-refute-threshold", type=float, default=0.50,
                   help="refute-rate threshold for --inside-selfcheck-early-stop")
    p.add_argument("--inside-trigger-signal", default=None, choices=SIGNALS,
                   help="optional cheap primary-layer signal used to decide which statements receive "
                        "sampled INSIDE scoring")
    p.add_argument("--inside-trigger-threshold", type=float, default=None,
                   help="sample INSIDE only when --inside-trigger-signal crosses this anomaly threshold")
    p.add_argument("--inside-trigger-top-fraction", type=float, default=None,
                   help="sample INSIDE for the most anomalous fraction of each eval batch according to "
                        "--inside-trigger-signal")
    p.add_argument("--offline", action="store_true",
                   help="use bundled smoke statements (pipeline check, not a benchmark)")
    p.add_argument("--json", default=None, help="optional path to write structured results")
    p.add_argument("--profile", action="store_true",
                   help="include phase timing diagnostics in stdout and --json output")
    p.add_argument("--profile-json", default=None,
                   help="optional path to write only phase timing diagnostics")
    p.add_argument("--progress-every", type=int, default=50,
                   help="print warmup/eval progress every N statements; use 0 to disable")
    p.add_argument("--cache-only", action="store_true",
                   help="score only from --layer-stats-cache and --eval-reps-cache; skip model loading")
    p.add_argument("--statement-encoding-cache", default=None,
                   help="optional JSON path to load or create cached statement token ids and answer spans")
    p.add_argument("--refresh-statement-encoding-cache", action="store_true",
                   help="rebuild and overwrite --statement-encoding-cache instead of loading it")
    p.add_argument("--layer-stats-cache", default=None,
                   help="optional .pt path to load or create cached warmup manifolds/subspaces")
    p.add_argument("--refresh-layer-stats-cache", action="store_true",
                   help="rebuild and overwrite --layer-stats-cache instead of loading it; "
                        "also ignores any existing --warmup-checkpoint")
    p.add_argument("--warmup-checkpoint", default=None,
                   help="optional .pt path for resumable warmup checkpoint state")
    p.add_argument("--warmup-checkpoint-every", type=int, default=50,
                   help="save --warmup-checkpoint every N processed warmup statements; use 0 to disable writes")
    p.add_argument("--eval-reps-cache", default=None,
                   help="optional .pt path to load or create cached forced-answer hidden states/metrics")
    p.add_argument("--eval-reps-cache-shard-size", type=int, default=0,
                   help="write --eval-reps-cache as a sharded directory with this many records per shard")
    p.add_argument("--eval-reps-shard-read-cache-size", type=int, default=2,
                   help="number of sharded eval-reps cache shards to keep in the read-side LRU cache")
    p.add_argument("--refresh-eval-reps-cache", action="store_true",
                   help="rebuild and overwrite --eval-reps-cache instead of loading it")
    p.add_argument("--inside-diagnostics-cache", default=None,
                   help="optional JSON path to load or create cached sampled INSIDE diagnostics")
    p.add_argument("--refresh-inside-diagnostics-cache", action="store_true",
                   help="rebuild and overwrite --inside-diagnostics-cache instead of loading it")
    p.add_argument("--dump-scores", default=None,
                   help="optional path to dump raw per-statement scores+labels "
                        "(enables post-hoc analyses, e.g. conformal calibration)")
    p.add_argument("--dump-scores-format", default="json", choices=SCORE_DUMP_FORMATS,
                   help="format for --dump-scores: json writes one file, jsonl writes a manifest plus records sidecar")
    p.add_argument("--dump-inside-samples", action="store_true",
                   help="when used with --dump-scores and --inside-samples, include sampled continuation text "
                        "for downstream self-consistency verifier fixtures")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.batch_size < 1:
        p.error("--batch-size must be >=1")
    if args.max_batch_tokens < 0:
        p.error("--max-batch-tokens must be >=0")
    if args.covariance_low_rank < 1:
        p.error("--covariance-low-rank must be >=1")
    if args.layer_spectrum_top_k < 0:
        p.error("--layer-spectrum-top-k must be >=0")
    if args.first_token_top_k < 1:
        p.error("--first-token-top-k must be >=1")
    if args.inside_batch_size < 1:
        p.error("--inside-batch-size must be >=1")
    if args.inside_samples == 1:
        p.error("--inside-samples must be 0 or >=2")
    if args.inside_temperature <= 0.0:
        p.error("--inside-temperature must be >0")
    if not (0.0 < args.inside_top_p <= 1.0):
        p.error("--inside-top-p must be in (0, 1]")
    if not (-1.0 <= args.inside_embedding_threshold <= 1.0):
        p.error("--inside-embedding-threshold must be in [-1, 1]")
    if args.inside_min_samples < 2:
        p.error("--inside-min-samples must be >=2")
    if args.inside_sample_step < 1:
        p.error("--inside-sample-step must be >=1")
    if args.inside_stability_delta < 0.0:
        p.error("--inside-stability-delta must be >=0")
    if not (0.0 <= args.inside_selfcheck_min_overlap <= 1.0):
        p.error("--inside-selfcheck-min-overlap must be in [0, 1]")
    if not (0.0 <= args.inside_selfcheck_support_threshold <= 1.0):
        p.error("--inside-selfcheck-support-threshold must be in [0, 1]")
    if not (0.0 <= args.inside_selfcheck_refute_threshold <= 1.0):
        p.error("--inside-selfcheck-refute-threshold must be in [0, 1]")
    if args.inside_selfcheck_early_stop and not args.inside_adaptive_sampling:
        p.error("--inside-selfcheck-early-stop requires --inside-adaptive-sampling")
    if args.inside_adaptive_sampling:
        if args.inside_samples < 2:
            p.error("--inside-adaptive-sampling requires --inside-samples >=2")
        if args.inside_min_samples > args.inside_samples:
            p.error("--inside-min-samples cannot exceed --inside-samples")
    if args.inside_trigger_signal and args.inside_samples < 2:
        p.error("--inside-trigger-signal requires --inside-samples >=2")
    if (args.inside_trigger_threshold is not None or args.inside_trigger_top_fraction is not None) \
            and not args.inside_trigger_signal:
        p.error("--inside-trigger-threshold/--inside-trigger-top-fraction require --inside-trigger-signal")
    if args.inside_trigger_threshold is not None and args.inside_trigger_top_fraction is not None:
        p.error("choose only one of --inside-trigger-threshold or --inside-trigger-top-fraction")
    if args.inside_trigger_signal and args.inside_trigger_threshold is None \
            and args.inside_trigger_top_fraction is None:
        p.error("--inside-trigger-signal requires a threshold or top fraction")
    if args.inside_trigger_top_fraction is not None and not (0.0 < args.inside_trigger_top_fraction <= 1.0):
        p.error("--inside-trigger-top-fraction must be in (0, 1]")
    if args.dump_inside_samples:
        if not args.dump_scores:
            p.error("--dump-inside-samples requires --dump-scores")
        if args.inside_samples < 2:
            p.error("--dump-inside-samples requires --inside-samples >=2")
    if args.refresh_statement_encoding_cache and not args.statement_encoding_cache:
        p.error("--refresh-statement-encoding-cache requires --statement-encoding-cache")
    if args.refresh_layer_stats_cache and not args.layer_stats_cache:
        p.error("--refresh-layer-stats-cache requires --layer-stats-cache")
    if args.refresh_eval_reps_cache and not args.eval_reps_cache:
        p.error("--refresh-eval-reps-cache requires --eval-reps-cache")
    if args.refresh_inside_diagnostics_cache and not args.inside_diagnostics_cache:
        p.error("--refresh-inside-diagnostics-cache requires --inside-diagnostics-cache")
    if args.inside_diagnostics_cache and args.inside_samples < 2:
        p.error("--inside-diagnostics-cache requires --inside-samples >=2")
    if args.eval_reps_cache_shard_size < 0:
        p.error("--eval-reps-cache-shard-size must be >=0")
    if args.eval_reps_cache_shard_size > 0 and not args.eval_reps_cache:
        p.error("--eval-reps-cache-shard-size requires --eval-reps-cache")
    if args.eval_reps_shard_read_cache_size < 1:
        p.error("--eval-reps-shard-read-cache-size must be >=1")
    if args.progress_every < 0:
        p.error("--progress-every must be >=0")
    if args.warmup_checkpoint_every < 0:
        p.error("--warmup-checkpoint-every must be >=0")
    if args.prefix_kv_cache and args.hidden_state_capture != "outputs":
        p.error("--prefix-kv-cache requires --hidden-state-capture outputs")
    if args.cache_only:
        if not args.layer_stats_cache or not args.eval_reps_cache:
            p.error("--cache-only requires --layer-stats-cache and --eval-reps-cache")
        if args.refresh_layer_stats_cache or args.refresh_eval_reps_cache:
            p.error("--cache-only cannot be combined with refresh cache flags")
        if args.refresh_statement_encoding_cache:
            p.error("--cache-only cannot refresh statement encodings because it skips tokenizer loading")
        if args.inside_samples:
            p.error("--cache-only cannot run sampled INSIDE; omit --inside-samples")
        if not Path(args.layer_stats_cache).exists():
            p.error("--cache-only layer stats cache does not exist")
        if not Path(args.eval_reps_cache).exists():
            p.error("--cache-only eval reps cache does not exist")
    run(args)


if __name__ == "__main__":
    main()

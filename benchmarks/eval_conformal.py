"""E1 — 共形校准覆盖率验证 / Split-conformal coverage validation.

研究问题 / Research question:
    split conformal 能否把原始分数（马氏距离 / 对比方向投影）变成有保证的报警阈值，
    取代拍脑袋的固定阈值？
    Can split conformal turn raw scores into alarm thresholds with honest coverage,
    replacing hand-picked thresholds?

方法 / Method:
    消费 eval_truthfulqa.py --dump-scores 的逐陈述分数（无需再跑模型）。
    把"真陈述"作为可交换的正常总体，随机对半切成 校准/测试，多次重复取平均：
    - 误报率 / false-alarm rate: 真陈述中 score > threshold(alpha) 的比例，应 <= alpha (+3%)
    - 检出率 / detection rate:   假陈述中 score > threshold(alpha) 的比例（power，仅报告）
    Uses per-statement scores dumped by eval_truthfulqa.py. True statements form the
    exchangeable "normal" population, split 50/50 into calibration/test over multiple
    seeded repeats. Gate: |false-alarm − alpha| within tolerance at every alpha.

判据 (E1 gate): 在 alpha ∈ {0.05, 0.1, 0.2} 上 |经验误报率 − alpha| <= 0.03。

用法 / Usage:
    python benchmarks/eval_conformal.py --scores benchmarks/scores_gpt2_l-8.json --signal maha_last
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.config_utils import planned_artifact_manifest_summary  # noqa: E402
from eigentruth.calibration import (  # noqa: E402
    DEFAULT_SCORE_DIRECTIONS,
    AdaptiveConformalCalibrator,
    ConformalCalibrator,
    LayerScoreSweepCalibrator,
    MultipleTestingConformalCalibrator,
    SequentialConformalCalibrator,
)
from eigentruth.eval.conformal import (  # noqa: E402
    ABSTENTION_COMPARISON_METRICS,
    AdaptiveScoreTransform,
    adaptive_anomaly_scores,
    conformal_abstention_comparison_report,
    conformal_abstention_release_gate,
    conformal_abstention_report,
    directional_conformal_thresholds,
    directional_trigger_rate,
    multiple_testing_conformal_report,
    sequential_conformal_monitor,
)
from eigentruth.eval.metrics import confidence_error_report, selective_classification_report  # noqa: E402
from eigentruth.eval.score_dump import (  # noqa: E402
    JSONL_FORMAT,
    ScoreDump,
    ScoreDumpColumns,
    ScoreDumpJsonlManifest,
    ScoreDumpLayerScores,
    iter_score_dump_jsonl_records,
    load_score_dump_columns,
    load_score_dump_columns_with_extras,
    load_score_dump_layer_scores,
    score_dump_cache_summary,
    score_dump_file_metadata,
)
from eigentruth.json_utils import strict_json_dumps  # noqa: E402
from eigentruth.registry import build_artifact_manifest  # noqa: E402

ALPHAS = (0.05, 0.10, 0.20)
TOLERANCE = 0.03


def _parse_signals(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    signals = tuple(part.strip() for part in value.split(",") if part.strip())
    if not signals:
        raise ValueError("--signals must contain at least one signal name.")
    return signals


def _parse_signal_direction_overrides(value: str | None, *, name: str) -> dict[str, str]:
    if value is None:
        return {}
    overrides: dict[str, str] = {}
    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        if ":" in item:
            signal, direction = item.split(":", 1)
        elif "=" in item:
            signal, direction = item.split("=", 1)
        else:
            raise ValueError(f"{name} entries must use SIGNAL:higher/lower.")
        signal = signal.strip()
        direction = direction.strip()
        if not signal:
            raise ValueError(f"{name} entries must include a signal name.")
        if direction not in {"higher", "lower"}:
            raise ValueError(f"{name} direction for {signal!r} must be 'higher' or 'lower'.")
        if signal in overrides:
            raise ValueError(f"{name} contains duplicate signal {signal!r}.")
        overrides[signal] = direction
    if not overrides:
        raise ValueError(f"{name} must contain at least one SIGNAL:direction entry.")
    return overrides


def _parse_adaptive_feature_names(values: object) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        raw_values = (values,)
    elif isinstance(values, tuple | list):
        raw_values = tuple(values)
    else:
        raise ValueError("--adaptive-feature values must be strings.")
    names: list[str] = []
    for value in raw_values:
        if not isinstance(value, str):
            raise ValueError("--adaptive-feature values must be strings.")
        for part in value.split(","):
            name = part.strip()
            if name and name not in names:
                names.append(name)
    return tuple(names)


def _parse_adaptive_feature_weights(
    values: object,
    feature_names: tuple[str, ...],
) -> dict[str, float]:
    weights = {name: 1.0 for name in feature_names}
    if values is None:
        return weights
    if isinstance(values, str):
        raw_values = (values,)
    elif isinstance(values, tuple | list):
        raw_values = tuple(values)
    else:
        raise ValueError("--adaptive-feature-weight values must be NAME=FLOAT strings.")
    valid_names = set(feature_names)
    for value in raw_values:
        if not isinstance(value, str) or "=" not in value:
            raise ValueError("--adaptive-feature-weight values must use NAME=FLOAT.")
        name, raw_weight = value.split("=", 1)
        name = name.strip()
        if name not in valid_names:
            raise ValueError(f"adaptive feature weight references unknown feature {name!r}.")
        try:
            weights[name] = float(raw_weight)
        except ValueError as exc:
            raise ValueError(f"adaptive feature weight for {name!r} must be numeric.") from exc
    return weights


def _all_dump_signals(summary: dict) -> tuple[str, ...]:
    return tuple(summary.get("all_signal_names", ()))


def _direction_for(signal: str, override: str | None = None) -> str:
    return override or DEFAULT_SCORE_DIRECTIONS.get(signal, "higher")


def _confidence_direction_for(signal: str, override: str | None = None) -> str:
    if override is not None:
        return override
    return "lower" if signal == "nll_answer" else "higher"


def _available_primary_signals(metadata: dict) -> set[str]:
    summary = metadata.get("summary", {})
    if not isinstance(summary, dict):
        return set()
    return {str(name) for name in summary.get("score_names", ())}


def _confidence_audit_status(
    *,
    signal: str | None,
    available_primary_signals: set[str],
    disabled: bool,
) -> dict:
    if disabled:
        return {"enabled": False, "reason": "disabled"}
    if signal is None:
        return {"enabled": False, "reason": "no_confidence_signal"}
    if signal not in available_primary_signals:
        return {
            "enabled": False,
            "reason": "missing_confidence_signal",
            "confidence_signal": signal,
            "available_primary_signals": tuple(sorted(available_primary_signals)),
        }
    return {"enabled": True, "confidence_signal": signal}


def _load_primary_score_dump_view(
    path: str | Path,
    signal: str,
    *,
    confidence_signal: str | None = None,
    disable_confidence_audit: bool = False,
    additional_signals: Sequence[str] = (),
    cache: dict,
) -> tuple[ScoreDumpColumns, dict, ScoreDump | None, dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and payload.get("format") == JSONL_FORMAT:
        metadata = score_dump_file_metadata(path, cache=cache)
        available = _available_primary_signals(metadata)
        confidence_audit = _confidence_audit_status(
            signal=confidence_signal,
            available_primary_signals=available,
            disabled=disable_confidence_audit,
        )
        requested = tuple(dict.fromkeys((
            signal,
            *(name for name in (confidence_signal,) if name is not None and name in available),
            *(name for name in additional_signals if name in available),
        )))
        extra_names = tuple(name for name in additional_signals if name not in available)
        if extra_names:
            columns = load_score_dump_columns_with_extras(
                path,
                requested,
                extra_names,
                cache=cache,
            )
        else:
            columns = load_score_dump_columns(path, requested, cache=cache)
        metadata = score_dump_file_metadata(path, cache=cache)
        metadata.update({
            "summary": dict(columns.summary),
            "source_format": columns.source_format,
        })
        return columns, metadata, None, confidence_audit

    dump = ScoreDump.from_mapping(payload)
    dump.require_scores((signal,))
    available = {str(name) for name in dump.scores}
    confidence_audit = _confidence_audit_status(
        signal=confidence_signal,
        available_primary_signals=available,
        disabled=disable_confidence_audit,
    )
    requested = tuple(dict.fromkeys((
        signal,
        *(name for name in (confidence_signal,) if name is not None and name in available),
        *(name for name in additional_signals if name in available),
    )))
    dump.require_scores(requested)
    columns = ScoreDumpColumns(
        labels=dump.labels,
        scores={name: dump.scores[name] for name in requested},
        config=dict(dump.config),
        summary=dump.summary(),
        source_format="json",
    )
    metadata = score_dump_file_metadata(path, dump=dump, cache=cache)
    metadata.update({
        "summary": dict(columns.summary),
        "source_format": columns.source_format,
    })
    return columns, metadata, dump, confidence_audit


def _load_primary_score_dump_view_from_layer_scores(
    path: str | Path,
    signal: str,
    *,
    selected_sweep_signals: Sequence[str] | None,
    confidence_signal: str | None = None,
    disable_confidence_audit: bool = False,
    additional_signals: Sequence[str] = (),
    cache: dict,
) -> tuple[ScoreDumpColumns, dict, None, dict, ScoreDumpLayerScores] | None:
    metadata = score_dump_file_metadata(path, cache=cache)
    if metadata.get("source_format") != JSONL_FORMAT:
        return None
    available = _available_primary_signals(metadata)
    confidence_audit = _confidence_audit_status(
        signal=confidence_signal,
        available_primary_signals=available,
        disabled=disable_confidence_audit,
    )
    requested_primary = tuple(dict.fromkeys((
        signal,
        *(name for name in (confidence_signal,) if name is not None and name in available),
        *(name for name in additional_signals if name in available),
    )))
    if selected_sweep_signals is None:
        layer_load_signals = None
    else:
        layer_load_signals = tuple(dict.fromkeys((
            *selected_sweep_signals,
            *requested_primary,
        )))
    layer_dump = load_score_dump_layer_scores(
        path,
        signals=layer_load_signals,
        cache=cache,
    )
    primary_layer = int(layer_dump.config.get("layer", 0))
    primary_scores = layer_dump.layer_scores.get(primary_layer, {})
    primary_sources = layer_dump.score_sources.get(primary_layer, {})
    if any(primary_sources.get(name) != "scores" for name in requested_primary):
        return None
    columns = ScoreDumpColumns(
        labels=layer_dump.labels,
        scores={name: primary_scores[name] for name in requested_primary},
        config=dict(layer_dump.config),
        summary=dict(layer_dump.summary),
        source_format=layer_dump.source_format,
    )
    metadata = score_dump_file_metadata(path, cache=cache)
    metadata.update({
        "summary": dict(columns.summary),
        "source_format": columns.source_format,
    })
    return columns, metadata, None, confidence_audit, layer_dump


def _load_score_dump_views_for_run(
    path: str | Path,
    signal: str,
    *,
    selected_sweep_signals: Sequence[str] | None,
    prefer_layer_scores: bool,
    confidence_signal: str | None = None,
    disable_confidence_audit: bool = False,
    additional_signals: Sequence[str] = (),
    cache: dict,
) -> tuple[ScoreDumpColumns, dict, ScoreDump | None, dict, ScoreDumpLayerScores | None]:
    if prefer_layer_scores:
        preloaded = _load_primary_score_dump_view_from_layer_scores(
            path,
            signal,
            selected_sweep_signals=selected_sweep_signals,
            confidence_signal=confidence_signal,
            disable_confidence_audit=disable_confidence_audit,
            additional_signals=additional_signals,
            cache=cache,
        )
        if preloaded is not None:
            return preloaded
    columns, metadata, full_score_dump, confidence_audit = _load_primary_score_dump_view(
        path,
        signal,
        confidence_signal=confidence_signal,
        disable_confidence_audit=disable_confidence_audit,
        additional_signals=additional_signals,
        cache=cache,
    )
    return columns, metadata, full_score_dump, confidence_audit, None


def _artifact_paths(args) -> dict[str, str | Path | None]:
    return {
        "input_scores": args.scores,
        "conformal_report": args.json,
        "calibration_artifact": args.save_calibration,
        "adaptive_calibration_artifact": getattr(args, "save_adaptive_calibration", None),
        "abstention_report": getattr(args, "save_abstention_report", None),
        "abstention_comparison_report": getattr(args, "save_abstention_comparison", None),
        "abstention_release_gate": getattr(args, "save_abstention_release_gate", None),
        "multiple_testing_report": getattr(args, "save_multiple_testing_report", None),
        "multiple_testing_calibration_artifact": getattr(args, "save_multiple_testing_calibration", None),
        "sequential_report": getattr(args, "save_sequential_report", None),
        "sequential_calibration_artifact": getattr(args, "save_sequential_calibration", None),
        "sweep_report": args.save_sweep_report,
        "best_calibration_artifact": args.save_best_calibration,
    }


def _coerce_feature_values(values: object, *, name: str, n_total: int) -> tuple[float, ...]:
    try:
        tensor = torch.as_tensor(values, dtype=torch.float64).flatten()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"adaptive feature {name!r} must be numeric.") from exc
    if tensor.numel() != n_total:
        raise ValueError(
            f"adaptive feature {name!r} must contain {n_total} values, got {tensor.numel()}."
        )
    if not torch.isfinite(tensor).all():
        raise ValueError(f"adaptive feature {name!r} must contain only finite values.")
    return tuple(float(item) for item in tensor.tolist())


def _load_jsonl_extra_features(
    path: str | Path,
    feature_names: Sequence[str],
    *,
    n_total: int,
) -> dict[str, tuple[float, ...]]:
    if not feature_names:
        return {}
    manifest = ScoreDumpJsonlManifest.load_json(path)
    features: dict[str, tuple[float, ...]] = {}
    missing = []
    for name in feature_names:
        if name in manifest.extras:
            features[name] = _coerce_feature_values(manifest.extras[name], name=name, n_total=n_total)
        else:
            missing.append(name)
    if not missing:
        return features

    collected = {name: [] for name in missing}
    for record in iter_score_dump_jsonl_records(path, allow_missing_scores=True):
        for name in tuple(collected):
            if name not in record.extras:
                continue
            collected[name].append(record.extras[name])
    for name, values in collected.items():
        if len(values) == n_total:
            features[name] = _coerce_feature_values(values, name=name, n_total=n_total)
    return features


def _load_adaptive_feature_values(
    path: str | Path,
    feature_names: Sequence[str],
    *,
    score_dump: ScoreDumpColumns,
    full_score_dump: ScoreDump | None,
) -> dict[str, tuple[float, ...]]:
    n_total = len(score_dump.labels)
    features: dict[str, tuple[float, ...]] = {}
    missing = []
    for name in feature_names:
        if name in score_dump.scores:
            features[name] = _coerce_feature_values(score_dump.scores[name], name=name, n_total=n_total)
        elif name in score_dump.extras:
            features[name] = _coerce_feature_values(score_dump.extras[name], name=name, n_total=n_total)
        elif full_score_dump is not None and name in full_score_dump.extras:
            features[name] = _coerce_feature_values(full_score_dump.extras[name], name=name, n_total=n_total)
        else:
            missing.append(name)

    if missing and full_score_dump is None:
        jsonl_features = _load_jsonl_extra_features(path, missing, n_total=n_total)
        features.update(jsonl_features)
    unresolved = tuple(name for name in feature_names if name not in features)
    if unresolved:
        raise ValueError(
            "score dump is missing adaptive feature(s): "
            f"{unresolved}. Features may be primary scores, JSON extras arrays, or JSONL record extras."
        )
    return features


def _run_adaptive_conformal_report(
    *,
    base_scores: torch.Tensor,
    labels: torch.Tensor,
    base_score_name: str,
    output_score_name: str,
    feature_values: dict[str, tuple[float, ...]],
    transform: AdaptiveScoreTransform,
    repeats: int,
    seed: int,
) -> tuple[dict, torch.Tensor]:
    if repeats < 1:
        raise ValueError("repeats must be >= 1.")
    adjusted_scores = adaptive_anomaly_scores(
        base_scores,
        feature_values=feature_values,
        feature_weights=transform.feature_weights,
        intercept=transform.intercept,
        direction=transform.direction,
    )
    true_scores = adjusted_scores[labels == 0]
    false_scores = adjusted_scores[labels == 1]
    n_true, n_false = true_scores.numel(), false_scores.numel()
    if n_true < 2:
        raise ValueError("score dump must contain at least two true statements for split conformal.")
    fa_sum = {a: 0.0 for a in ALPHAS}
    det_sum = {a: 0.0 for a in ALPHAS}
    for repeat_idx in range(repeats):
        generator = torch.Generator().manual_seed(seed + repeat_idx)
        perm = torch.randperm(n_true, generator=generator)
        half = n_true // 2
        calib = true_scores[perm[:half]]
        test_true = true_scores[perm[half:]]
        thresholds = directional_conformal_thresholds(calib, ALPHAS, "higher")
        for alpha in ALPHAS:
            threshold = thresholds[alpha]
            fa_sum[alpha] += directional_trigger_rate(test_true, threshold, "higher")
            det_sum[alpha] += directional_trigger_rate(false_scores, threshold, "higher")

    results = {}
    all_pass = True
    full_thresholds = directional_conformal_thresholds(true_scores, ALPHAS, "higher")
    for alpha in ALPHAS:
        false_alarm = fa_sum[alpha] / repeats
        detection = det_sum[alpha] / repeats
        ok = false_alarm <= alpha + TOLERANCE
        all_pass &= ok
        threshold = full_thresholds[alpha]
        results[str(alpha)] = {
            "false_alarm": false_alarm,
            "coverage": 1.0 - false_alarm,
            "detection": detection,
            "pass": ok,
            "conservative": false_alarm < max(0.0, alpha - TOLERANCE),
            "threshold": threshold,
            "selective_report": selective_classification_report(
                adjusted_scores,
                labels,
                threshold,
                direction="higher",
            ),
        }

    return (
        {
            "config": {
                "base_signal": base_score_name,
                "score_name": output_score_name,
                "direction": "higher",
                "base_direction": transform.direction,
                "transform": transform.to_dict(),
                "feature_names": tuple(feature_values),
                "n_true": n_true,
                "n_false": n_false,
                "repeats": repeats,
                "seed": seed,
            },
            "results": results,
            "verdict": "ACCEPT" if all_pass else "REJECT",
        },
        adjusted_scores,
    )


def _run_abstention_report(
    *,
    scores: torch.Tensor,
    labels: torch.Tensor,
    signal: str,
    direction: str,
    alpha: float,
) -> dict:
    correctness = tuple(int(label == 0) for label in labels.tolist())
    report = conformal_abstention_report(
        scores,
        correctness,
        alpha,
        direction=direction,
        score_name=signal,
    )
    return report.to_dict()


def _run_abstention_comparison_report(
    *,
    score_dump: ScoreDumpColumns,
    labels: torch.Tensor,
    signals: Sequence[str],
    alpha: float,
    direction_override: str | None,
    best_by: str,
) -> dict:
    missing = tuple(signal for signal in signals if signal not in score_dump.scores)
    if missing:
        available = tuple(sorted(str(name) for name in score_dump.scores))
        raise ValueError(
            f"score dump is missing abstention comparison signal(s) {missing}; "
            f"available signals: {available}"
        )
    correctness = tuple(int(label == 0) for label in labels.tolist())
    score_map = {
        signal: torch.tensor(score_dump.scores[signal], dtype=torch.float64)
        for signal in signals
    }
    directions = {
        signal: _direction_for(signal, direction_override)
        for signal in signals
    }
    report = conformal_abstention_comparison_report(
        score_map,
        correctness,
        alpha,
        directions=directions,
        best_by=best_by,
    )
    return report.to_dict()


def _resolve_abstention_comparison_signals(
    args,
    *,
    abstention_signal: str,
    enabled: bool,
) -> tuple[str, ...]:
    if not enabled:
        return ()
    parsed = _parse_signals(getattr(args, "abstention_signals", None))
    if parsed is not None:
        return parsed
    sweep_signals = _parse_signals(getattr(args, "signals", None))
    if sweep_signals is not None:
        return sweep_signals
    return tuple(dict.fromkeys((abstention_signal, args.signal)))


def _resolve_multiple_testing_signals(args, *, enabled: bool) -> tuple[str, ...]:
    if not enabled:
        return ()
    parsed = _parse_signals(getattr(args, "multiple_testing_signals", None))
    if parsed is not None:
        return parsed
    sweep_signals = _parse_signals(getattr(args, "signals", None))
    if sweep_signals is not None:
        return sweep_signals
    return (str(args.signal),)


def _run_multiple_testing_report(
    *,
    score_dump: ScoreDumpColumns,
    labels: torch.Tensor,
    signals: Sequence[str],
    alpha: float,
    method: str,
    direction_override: str | None,
    direction_overrides: Mapping[str, str] | None,
    base_signal: str,
    repeats: int,
    seed: int,
) -> dict:
    if repeats < 1:
        raise ValueError("repeats must be >= 1.")
    missing = tuple(signal for signal in signals if signal not in score_dump.scores)
    if missing:
        available = tuple(sorted(str(name) for name in score_dump.scores))
        raise ValueError(
            f"score dump is missing multiple-testing signal(s) {missing}; "
            f"available signals: {available}"
        )

    signal_scores = {
        signal: torch.tensor(score_dump.scores[signal], dtype=torch.float64)
        for signal in signals
    }
    explicit_direction_overrides = dict(direction_overrides or {})
    unknown_overrides = tuple(sorted(set(explicit_direction_overrides) - set(signals)))
    if unknown_overrides:
        raise ValueError(
            "multiple-testing direction override references unknown signal(s) "
            f"{unknown_overrides}; configured signals: {tuple(signals)}"
        )
    directions = {
        signal: _direction_for(signal, explicit_direction_overrides.get(signal))
        if signal in explicit_direction_overrides
        else _direction_for(signal, direction_override if signal == base_signal else None)
        for signal in signals
    }
    true_indices = torch.nonzero(labels == 0, as_tuple=False).flatten()
    false_indices = torch.nonzero(labels == 1, as_tuple=False).flatten()
    n_true, n_false = int(true_indices.numel()), int(false_indices.numel())
    if n_true < 2:
        raise ValueError("score dump must contain at least two true statements for split conformal.")

    repeats_payload = []
    false_alarm_sum = 0.0
    detection_sum = 0.0
    true_rejected_by_signal = {signal: 0 for signal in signals}
    false_rejected_by_signal = {signal: 0 for signal in signals}

    for repeat_idx in range(repeats):
        generator = torch.Generator().manual_seed(seed + repeat_idx)
        shuffled_true = true_indices[torch.randperm(n_true, generator=generator)]
        split = n_true // 2
        calibration_indices = shuffled_true[:split]
        test_true_indices = shuffled_true[split:]
        calibration_scores = {
            signal: scores[calibration_indices]
            for signal, scores in signal_scores.items()
        }

        true_rejected = 0
        false_rejected = 0
        repeat_true_by_signal = {signal: 0 for signal in signals}
        repeat_false_by_signal = {signal: 0 for signal in signals}

        for index in test_true_indices.tolist():
            item_report = multiple_testing_conformal_report(
                calibration_scores,
                {signal: float(signal_scores[signal][index].item()) for signal in signals},
                alpha=alpha,
                directions=directions,
                method=method,
            )
            if item_report.rejected:
                true_rejected += 1
            for signal in item_report.rejected_signal_names:
                repeat_true_by_signal[signal] += 1

        for index in false_indices.tolist():
            item_report = multiple_testing_conformal_report(
                calibration_scores,
                {signal: float(signal_scores[signal][index].item()) for signal in signals},
                alpha=alpha,
                directions=directions,
                method=method,
            )
            if item_report.rejected:
                false_rejected += 1
            for signal in item_report.rejected_signal_names:
                repeat_false_by_signal[signal] += 1

        false_alarm = true_rejected / max(1, int(test_true_indices.numel()))
        detection = false_rejected / max(1, n_false)
        false_alarm_sum += false_alarm
        detection_sum += detection
        for signal in signals:
            true_rejected_by_signal[signal] += repeat_true_by_signal[signal]
            false_rejected_by_signal[signal] += repeat_false_by_signal[signal]
        repeats_payload.append({
            "repeat": repeat_idx,
            "n_calibration": int(calibration_indices.numel()),
            "n_test_true": int(test_true_indices.numel()),
            "n_false": n_false,
            "false_alarm": false_alarm,
            "coverage": 1.0 - false_alarm,
            "detection": detection,
            "true_rejected_count": true_rejected,
            "false_rejected_count": false_rejected,
            "true_rejected_by_signal": repeat_true_by_signal,
            "false_rejected_by_signal": repeat_false_by_signal,
        })

    false_alarm = false_alarm_sum / repeats
    detection = detection_sum / repeats
    total_true_tests = sum(int(item["n_test_true"]) for item in repeats_payload)
    total_false_tests = sum(int(item["n_false"]) for item in repeats_payload)
    passed = false_alarm <= alpha + TOLERANCE

    return {
        "config": {
            "signals": list(signals),
            "directions": directions,
            "alpha": alpha,
            "method": method,
            "n_true": n_true,
            "n_false": n_false,
            "repeats": repeats,
            "seed": seed,
        },
        "false_alarm": false_alarm,
        "coverage": 1.0 - false_alarm,
        "detection": detection,
        "pass": passed,
        "conservative": false_alarm < max(0.0, alpha - TOLERANCE),
        "true_rejected_by_signal": {
            signal: {
                "count": count,
                "rate": count / max(1, total_true_tests),
            }
            for signal, count in true_rejected_by_signal.items()
        },
        "false_rejected_by_signal": {
            signal: {
                "count": count,
                "rate": count / max(1, total_false_tests),
            }
            for signal, count in false_rejected_by_signal.items()
        },
        "repeats": repeats_payload,
    }


def _run_sequential_conformal_report(
    *,
    score_dump: ScoreDumpColumns,
    labels: torch.Tensor,
    signal: str,
    direction: str,
    alpha: float,
    schedule: str,
    seed: int,
) -> dict:
    if signal not in score_dump.scores:
        available = tuple(sorted(str(name) for name in score_dump.scores))
        raise ValueError(
            f"score dump is missing sequential conformal signal {signal!r}; "
            f"available signals: {available}"
        )
    signal_scores = torch.tensor(score_dump.scores[signal], dtype=torch.float64)
    true_indices = torch.nonzero(labels == 0, as_tuple=False).flatten()
    false_indices = torch.nonzero(labels == 1, as_tuple=False).flatten()
    n_true, n_false = int(true_indices.numel()), int(false_indices.numel())
    if n_true < 2:
        raise ValueError("score dump must contain at least two true statements for sequential conformal.")

    generator = torch.Generator().manual_seed(seed)
    shuffled_true = true_indices[torch.randperm(n_true, generator=generator)]
    split = n_true // 2
    calibration_indices = shuffled_true[:split]
    calibration_index_set = set(int(index) for index in calibration_indices.tolist())
    replay_indices = tuple(
        index for index in range(int(labels.numel())) if index not in calibration_index_set
    )
    if not replay_indices:
        raise ValueError("sequential conformal replay sequence must be non-empty.")

    replay_scores = signal_scores[list(replay_indices)]
    replay_labels = labels[list(replay_indices)]
    monitor = sequential_conformal_monitor(
        signal_scores[calibration_indices],
        replay_scores,
        alpha=alpha,
        direction=direction,
        schedule=schedule,
        metadata={
            "signal": signal,
            "direction": direction,
            "seed": seed,
            "calibration_source": "split_true_scores",
            "replay_order": "score_dump_record_order",
        },
    )
    true_rejected_count = 0
    false_rejected_count = 0
    rejected_indices: list[int] = []
    rejected_true_indices: list[int] = []
    rejected_false_indices: list[int] = []
    for step, label, record_index in zip(monitor.steps, replay_labels.tolist(), replay_indices, strict=True):
        if not step.rejected:
            continue
        rejected_indices.append(int(record_index))
        if int(label) == 0:
            true_rejected_count += 1
            rejected_true_indices.append(int(record_index))
        else:
            false_rejected_count += 1
            rejected_false_indices.append(int(record_index))

    n_replay_true = int((replay_labels == 0).sum().item())
    n_replay_false = int((replay_labels == 1).sum().item())
    false_alarm_rate = true_rejected_count / max(1, n_replay_true)
    detection_rate = false_rejected_count / max(1, n_replay_false)
    report = monitor.to_dict()
    report["sequence_indices"] = list(replay_indices)
    report["labels"] = [int(label) for label in replay_labels.tolist()]
    return {
        "config": {
            "signal": signal,
            "direction": direction,
            "alpha": float(alpha),
            "schedule": schedule,
            "seed": int(seed),
            "n_true": n_true,
            "n_false": n_false,
            "n_calibration_true": int(calibration_indices.numel()),
            "n_replay_true": n_replay_true,
            "n_replay_false": n_replay_false,
        },
        "report": report,
        "label_metrics": {
            "true_rejected_count": true_rejected_count,
            "false_rejected_count": false_rejected_count,
            "rejected_count": true_rejected_count + false_rejected_count,
            "false_alarm_event": true_rejected_count > 0,
            "false_alarm_rate": false_alarm_rate,
            "coverage": 1.0 - false_alarm_rate,
            "detection": detection_rate,
            "rejected_indices": rejected_indices,
            "rejected_true_indices": rejected_true_indices,
            "rejected_false_indices": rejected_false_indices,
        },
        "budget_status": "true_alarm_observed" if true_rejected_count else "clean_true_replay",
    }


def _add_planned_manifest_fields(args, payload: dict) -> None:
    artifact_manifest = getattr(args, "artifact_manifest", None)
    if artifact_manifest is None:
        return
    output_paths = tuple(
        path
        for name, path in _artifact_paths(args).items()
        if name != "input_scores" and path is not None
    )
    payload.setdefault("paths", {})["artifact_manifest"] = str(artifact_manifest)
    payload["artifact_manifest_summary"] = planned_artifact_manifest_summary(
        _artifact_paths(args),
        assume_file_paths=output_paths,
    )


def _write_artifact_manifest(args, payload: dict) -> dict | None:
    artifact_manifest = getattr(args, "artifact_manifest", None)
    if artifact_manifest is None:
        return None
    manifest_path = Path(artifact_manifest)
    multiple_testing_report = payload.get("multiple_testing_report")
    multiple_testing_config = (
        dict(multiple_testing_report.get("config", {}))
        if isinstance(multiple_testing_report, Mapping)
        and isinstance(multiple_testing_report.get("config"), Mapping)
        else None
    )
    manifest = build_artifact_manifest(
        _artifact_paths(args),
        root=manifest_path.parent,
        metadata={
            "runner": "eval_conformal",
            "verdict": payload.get("verdict"),
            "signal": args.signal,
            "direction": payload.get("config", {}).get("direction"),
            "has_sweep_report": "sweep_report" in payload,
            "has_adaptive_report": "adaptive_conformal_report" in payload,
            "has_abstention_report": "abstention_report" in payload,
            "has_abstention_comparison_report": "abstention_comparison_report" in payload,
            "has_abstention_release_gate": "abstention_release_gate" in payload,
            "has_multiple_testing_report": "multiple_testing_report" in payload,
            "has_multiple_testing_calibration_artifact": (
                getattr(args, "save_multiple_testing_calibration", None) is not None
            ),
            "has_sequential_report": "sequential_conformal_report" in payload,
            "has_sequential_calibration_artifact": (
                getattr(args, "save_sequential_calibration", None) is not None
            ),
            "multiple_testing": (
                None
                if multiple_testing_config is None
                else {
                    "signals": list(multiple_testing_config.get("signals", ())),
                    "directions": dict(multiple_testing_config.get("directions", {})),
                    "alpha": multiple_testing_config.get("alpha"),
                    "method": multiple_testing_config.get("method"),
                    "pass": bool(multiple_testing_report.get("pass", False)),
                }
            ),
        },
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        strict_json_dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    payload.setdefault("paths", {})["artifact_manifest"] = str(manifest_path)
    print(f"\nWrote artifact manifest to {manifest_path}")
    return manifest


def run(args) -> dict:
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

    if int(args.repeats) < 1:
        raise ValueError("--repeats must be >= 1.")
    score_dump_metadata_cache = {}
    disable_confidence_audit = bool(getattr(args, "disable_confidence_audit", False))
    raw_confidence_signal = getattr(args, "confidence_signal", "nll_answer")
    confidence_signal = (
        None
        if disable_confidence_audit or raw_confidence_signal is None
        else str(raw_confidence_signal)
    )
    adaptive_feature_names = _parse_adaptive_feature_names(getattr(args, "adaptive_feature", None))
    adaptive_feature_weights = _parse_adaptive_feature_weights(
        getattr(args, "adaptive_feature_weight", None),
        adaptive_feature_names,
    )
    abstention_signal = getattr(args, "abstention_signal", None)
    if abstention_signal is None:
        abstention_signal = args.signal
    else:
        abstention_signal = str(abstention_signal)
    abstention_direction = _direction_for(
        abstention_signal,
        getattr(args, "abstention_direction", None),
    )
    wants_abstention_release_gate = bool(
        getattr(args, "save_abstention_release_gate", None)
        or getattr(args, "include_abstention_release_gate", False)
    )
    wants_abstention_report = bool(
        getattr(args, "save_abstention_report", None)
        or getattr(args, "include_abstention_report", False)
    )
    wants_abstention_comparison = bool(
        getattr(args, "save_abstention_comparison", None)
        or getattr(args, "include_abstention_comparison", False)
        or (
            wants_abstention_release_gate
            and getattr(args, "abstention_signals", None) is not None
        )
    )
    wants_multiple_testing = bool(
        getattr(args, "save_multiple_testing_report", None)
        or getattr(args, "save_multiple_testing_calibration", None)
        or getattr(args, "include_multiple_testing_report", False)
    )
    sequential_signal = getattr(args, "sequential_signal", None)
    if sequential_signal is None:
        sequential_signal = args.signal
    else:
        sequential_signal = str(sequential_signal)
    sequential_direction = _direction_for(
        sequential_signal,
        getattr(args, "sequential_direction", None),
    )
    wants_sequential_report = bool(
        getattr(args, "save_sequential_report", None)
        or getattr(args, "include_sequential_report", False)
    )
    wants_sequential = bool(
        wants_sequential_report
        or getattr(args, "save_sequential_calibration", None)
    )
    if wants_abstention_release_gate and not wants_abstention_comparison:
        wants_abstention_report = True
    abstention_comparison_signals = _resolve_abstention_comparison_signals(
        args,
        abstention_signal=abstention_signal,
        enabled=wants_abstention_comparison,
    )
    multiple_testing_signals = _resolve_multiple_testing_signals(
        args,
        enabled=wants_multiple_testing,
    )
    multiple_testing_direction_overrides = _parse_signal_direction_overrides(
        getattr(args, "multiple_testing_directions", None),
        name="--multiple-testing-directions",
    )
    additional_signals = tuple(dict.fromkeys((
        *adaptive_feature_names,
        *(name for name in (abstention_signal,) if name != args.signal),
        *(name for name in abstention_comparison_signals if name != args.signal),
        *(name for name in multiple_testing_signals if name != args.signal),
        *(name for name in (sequential_signal,) if wants_sequential and name != args.signal),
    )))
    if getattr(args, "save_adaptive_calibration", None) and not adaptive_feature_names:
        raise ValueError("--save-adaptive-calibration requires at least one --adaptive-feature.")
    wants_sweep = bool(
        getattr(args, "save_sweep_report", None)
        or getattr(args, "save_best_calibration", None)
    )
    selected_sweep_signals = (
        _parse_signals(args.signals)
        if wants_sweep
        else None
    )
    score_dump, score_dump_metadata, full_score_dump, confidence_audit, preloaded_layer_scores = (
        _load_score_dump_views_for_run(
            args.scores,
            args.signal,
            selected_sweep_signals=selected_sweep_signals,
            prefer_layer_scores=wants_sweep,
            confidence_signal=confidence_signal,
            disable_confidence_audit=disable_confidence_audit,
            additional_signals=additional_signals,
            cache=score_dump_metadata_cache,
        )
    )
    labels = torch.tensor(score_dump.labels)
    scores = torch.tensor(score_dump.scores[args.signal], dtype=torch.float64)
    dump_config = score_dump.config
    direction = _direction_for(args.signal, args.direction)
    confidence_direction = _confidence_direction_for(
        str(confidence_signal) if confidence_signal is not None else "",
        getattr(args, "confidence_direction", None),
    )

    true_scores = scores[labels == 0]   # 正常总体（可交换假设的对象）
    false_scores = scores[labels == 1]  # 希望被报警的对象（仅报告 power）
    n_true, n_false = true_scores.numel(), false_scores.numel()
    if n_true < 2:
        raise ValueError("score dump must contain at least two true statements for split conformal.")
    print(f"signal={args.signal}  direction={direction}  n_true={n_true}  n_false={n_false}  "
          f"repeats={args.repeats}\n")

    fa_sum = {a: 0.0 for a in ALPHAS}
    det_sum = {a: 0.0 for a in ALPHAS}
    for r in range(args.repeats):
        g = torch.Generator().manual_seed(args.seed + r)
        perm = torch.randperm(n_true, generator=g)
        half = n_true // 2
        calib = true_scores[perm[:half]]
        test_true = true_scores[perm[half:]]
        thresholds = directional_conformal_thresholds(calib, ALPHAS, direction)
        for a in ALPHAS:
            t = thresholds[a]
            fa_sum[a] += directional_trigger_rate(test_true, t, direction)
            det_sum[a] += directional_trigger_rate(false_scores, t, direction)

    print(f"  {'alpha':>6} {'nominal_cov':>12} {'false_alarm':>12} "
          f"{'emp_cov':>9} {'detect':>8}   gate(fa<=a+{TOLERANCE})")
    print("  " + "-" * 66)
    results = {}
    all_pass = True
    full_thresholds = directional_conformal_thresholds(true_scores, ALPHAS, direction)
    for a in ALPHAS:
        fa = fa_sum[a] / args.repeats
        det = det_sum[a] / args.repeats
        ok = fa <= a + TOLERANCE
        all_pass &= ok
        full_threshold = full_thresholds[a]
        selective_report = selective_classification_report(
            scores, labels, full_threshold, direction=direction
        )
        confidence_report = None
        if confidence_audit.get("enabled") is True and confidence_signal in score_dump.scores:
            confidence_report = confidence_error_report(
                scores,
                labels,
                full_threshold,
                score_dump.scores[str(confidence_signal)],
                anomaly_direction=direction,
                confidence_direction=confidence_direction,
                confidence_top_fraction=float(getattr(args, "confidence_top_fraction", 0.25)),
            )
        results[str(a)] = {
            "false_alarm": fa,
            "coverage": 1.0 - fa,
            "detection": det,
            "pass": ok,
            "conservative": fa < max(0.0, a - TOLERANCE),
            "threshold": full_threshold,
            "selective_report": selective_report,
        }
        if confidence_report is not None:
            results[str(a)]["confidence_error_report"] = confidence_report
        print(f"  {a:>6.2f} {1 - a:>12.2f} {fa:>12.3f} {1 - fa:>9.3f} "
              f"{det:>8.3f}   {'PASS' if ok else 'FAIL'}")
    print("  " + "-" * 66)
    print(f"\n  E1 verdict: {'ACCEPT' if all_pass else 'REJECT'} "
          f"(false alarm stays below alpha + {TOLERANCE} at all alphas)"
          if all_pass else
          f"\n  E1 verdict: REJECT (false alarm exceeds alpha + {TOLERANCE})")

    if confidence_audit.get("enabled") is True:
        confidence_audit = {
            **confidence_audit,
            "confidence_direction": confidence_direction,
            "confidence_top_fraction": float(getattr(args, "confidence_top_fraction", 0.25)),
        }

    base_verdict = "ACCEPT" if all_pass else "REJECT"
    payload = {"config": {"scores": args.scores, "signal": args.signal,
                          "score_dump": score_dump_metadata,
                          "direction": direction, "repeats": args.repeats, "seed": args.seed,
                          "n_true": n_true, "n_false": n_false,
                          "confidence_audit": confidence_audit},
               "results": results,
               "component_verdicts": {"base_conformal": base_verdict},
               "verdict": base_verdict}

    abstention_report = None
    abstention_comparison_report = None

    if wants_multiple_testing:
        multiple_testing_alpha = float(getattr(args, "multiple_testing_alpha", args.artifact_alpha))
        multiple_testing_method = str(getattr(args, "multiple_testing_method", "by"))
        multiple_testing_report = _run_multiple_testing_report(
            score_dump=score_dump,
            labels=labels,
            signals=multiple_testing_signals,
            alpha=multiple_testing_alpha,
            method=multiple_testing_method,
            direction_override=getattr(args, "direction", None),
            direction_overrides=multiple_testing_direction_overrides,
            base_signal=args.signal,
            repeats=args.repeats,
            seed=args.seed,
        )
        if getattr(args, "save_multiple_testing_report", None) or bool(
            getattr(args, "include_multiple_testing_report", False)
        ):
            payload["multiple_testing_report"] = multiple_testing_report
        payload.setdefault("component_verdicts", {})["multiple_testing"] = (
            "ACCEPT" if multiple_testing_report["pass"] else "REJECT"
        )
        if not multiple_testing_report["pass"]:
            payload["verdict"] = "REJECT"
        save_multiple_testing_report = getattr(args, "save_multiple_testing_report", None)
        if save_multiple_testing_report:
            Path(save_multiple_testing_report).parent.mkdir(parents=True, exist_ok=True)
            Path(save_multiple_testing_report).write_text(
                strict_json_dumps(multiple_testing_report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"\nWrote multiple-testing report to {save_multiple_testing_report}")
        save_multiple_testing_calibration = getattr(args, "save_multiple_testing_calibration", None)
        if save_multiple_testing_calibration:
            true_mask = labels == 0
            signal_scores = {
                signal: torch.tensor(score_dump.scores[signal], dtype=torch.float64)[true_mask]
                for signal in multiple_testing_signals
            }
            artifact = MultipleTestingConformalCalibrator(
                alpha=multiple_testing_alpha,
                method=multiple_testing_method,
            ).calibrate(
                model_id=args.model_id or dump_config.get("model", "unknown"),
                model_revision=args.model_revision,
                target_layer=args.target_layer if args.target_layer is not None else int(dump_config.get("layer", 0)),
                calibration_scores=signal_scores,
                directions=multiple_testing_report["config"]["directions"],
                calibration_dataset_metadata={
                    "scores": args.scores,
                    "signals": list(multiple_testing_signals),
                    "n_true": int(n_true),
                    "source": "eval_conformal.py",
                    "report_pass": bool(multiple_testing_report["pass"]),
                },
                created_at=args.created_at,
                commit_sha=args.commit_sha,
            )
            Path(save_multiple_testing_calibration).parent.mkdir(parents=True, exist_ok=True)
            artifact.save_json(save_multiple_testing_calibration)
            print(f"\nWrote multiple-testing calibration artifact to {save_multiple_testing_calibration}")
        print(
            "\n  Multiple-testing conformal: "
            f"signals={','.join(multiple_testing_signals)} "
            f"method={multiple_testing_report['config']['method']} "
            f"alpha={multiple_testing_report['config']['alpha']:.3f} "
            f"false_alarm={multiple_testing_report['false_alarm']:.3f} "
            f"detection={multiple_testing_report['detection']:.3f} "
            f"{'PASS' if multiple_testing_report['pass'] else 'FAIL'}"
        )

    if wants_sequential_report:
        sequential_report = _run_sequential_conformal_report(
            score_dump=score_dump,
            labels=labels,
            signal=sequential_signal,
            direction=sequential_direction,
            alpha=float(getattr(args, "sequential_alpha", args.artifact_alpha)),
            schedule=str(getattr(args, "sequential_schedule", "harmonic")),
            seed=int(
                args.seed
                if getattr(args, "sequential_seed", None) is None
                else getattr(args, "sequential_seed")
            ),
        )
        if getattr(args, "save_sequential_report", None) or bool(
            getattr(args, "include_sequential_report", False)
        ):
            payload["sequential_conformal_report"] = sequential_report
        save_sequential_report = getattr(args, "save_sequential_report", None)
        if save_sequential_report:
            Path(save_sequential_report).parent.mkdir(parents=True, exist_ok=True)
            Path(save_sequential_report).write_text(
                strict_json_dumps(sequential_report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"\nWrote sequential conformal report to {save_sequential_report}")
        metrics = sequential_report["label_metrics"]
        print(
            "\n  Sequential conformal: "
            f"signal={sequential_signal} "
            f"schedule={sequential_report['config']['schedule']} "
            f"alpha={sequential_report['config']['alpha']:.3f} "
            f"true_rejected={metrics['true_rejected_count']} "
            f"false_rejected={metrics['false_rejected_count']} "
            f"status={sequential_report['budget_status']}"
        )

    save_sequential_calibration = getattr(args, "save_sequential_calibration", None)
    if save_sequential_calibration:
        if sequential_signal not in score_dump.scores:
            available = tuple(sorted(str(name) for name in score_dump.scores))
            raise ValueError(
                f"score dump is missing sequential conformal signal {sequential_signal!r}; "
                f"available signals: {available}"
            )
        sequential_scores = torch.tensor(score_dump.scores[sequential_signal], dtype=torch.float64)
        artifact = SequentialConformalCalibrator(
            alpha=float(getattr(args, "sequential_alpha", args.artifact_alpha)),
            schedule=str(getattr(args, "sequential_schedule", "harmonic")),
        ).calibrate(
            model_id=args.model_id or dump_config.get("model", "unknown"),
            model_revision=args.model_revision,
            target_layer=args.target_layer if args.target_layer is not None else int(dump_config.get("layer", 0)),
            signal_name=sequential_signal,
            calibration_scores=sequential_scores[labels == 0],
            direction=sequential_direction,
            calibration_dataset_metadata={
                "scores": args.scores,
                "signal": sequential_signal,
                "n_true": int(n_true),
                "source": "eval_conformal.py",
            },
            created_at=args.created_at,
            commit_sha=args.commit_sha,
        )
        Path(save_sequential_calibration).parent.mkdir(parents=True, exist_ok=True)
        artifact.save_json(save_sequential_calibration)
        print(f"\nWrote sequential conformal calibration artifact to {save_sequential_calibration}")

    if wants_abstention_report:
        if abstention_signal not in score_dump.scores:
            available = tuple(sorted(str(name) for name in score_dump.scores))
            raise ValueError(
                f"score dump is missing abstention signal {abstention_signal!r}; "
                f"available signals: {available}"
            )
        abstention_scores = torch.tensor(score_dump.scores[abstention_signal], dtype=torch.float64)
        abstention_report = _run_abstention_report(
            scores=abstention_scores,
            labels=labels,
            signal=abstention_signal,
            direction=abstention_direction,
            alpha=float(getattr(args, "abstention_alpha", args.artifact_alpha)),
        )
        if getattr(args, "save_abstention_report", None) or bool(
            getattr(args, "include_abstention_report", False)
        ):
            payload["abstention_report"] = abstention_report
        save_abstention_report = getattr(args, "save_abstention_report", None)
        if save_abstention_report:
            Path(save_abstention_report).parent.mkdir(parents=True, exist_ok=True)
            Path(save_abstention_report).write_text(
                strict_json_dumps(abstention_report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"\nWrote abstention report to {save_abstention_report}")
        print(
            f"\n  Conformal abstention: signal={abstention_signal} "
            f"direction={abstention_direction} "
            f"alpha={abstention_report['alpha']:.3f} "
            f"participation={abstention_report['empirical_participation_rate']:.3f} "
            f"selective_accuracy={abstention_report['empirical_selective_accuracy']}"
        )

    if wants_abstention_comparison:
        abstention_comparison_report = _run_abstention_comparison_report(
            score_dump=score_dump,
            labels=labels,
            signals=abstention_comparison_signals,
            alpha=float(getattr(args, "abstention_alpha", args.artifact_alpha)),
            direction_override=getattr(args, "abstention_direction", None),
            best_by=str(
                getattr(
                    args,
                    "abstention_best_by",
                    "conditional_correctness_lower_bound",
                )
            ),
        )
        payload["abstention_comparison_report"] = abstention_comparison_report
        save_abstention_comparison = getattr(args, "save_abstention_comparison", None)
        if save_abstention_comparison:
            Path(save_abstention_comparison).parent.mkdir(parents=True, exist_ok=True)
            Path(save_abstention_comparison).write_text(
                strict_json_dumps(abstention_comparison_report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"\nWrote abstention comparison report to {save_abstention_comparison}")
        recommended = abstention_comparison_report.get("recommended")
        if recommended is not None:
            print(
                f"\n  Abstention comparison: best={recommended['score_name']} "
                f"metric={abstention_comparison_report['best_by']} "
                f"value={recommended['selection_value']} "
                f"candidates={abstention_comparison_report['candidate_count']}"
            )

    if wants_abstention_release_gate:
        gate_input = (
            abstention_comparison_report
            if abstention_comparison_report is not None
            else abstention_report
        )
        if gate_input is None:
            raise ValueError("abstention release gate requires an abstention report source.")
        abstention_release_gate = conformal_abstention_release_gate(
            gate_input,
            min_conditional_correctness_lower_bound=float(
                getattr(args, "min_abstention_conditional_correctness_lower_bound", 0.8)
            ),
            max_abstention_rate=float(getattr(args, "max_abstention_rate", 0.5)),
        ).to_dict()
        payload["abstention_release_gate"] = abstention_release_gate
        payload.setdefault("component_verdicts", {})["abstention_release_gate"] = (
            "ACCEPT" if abstention_release_gate["passed"] else "REJECT"
        )
        if not abstention_release_gate["passed"]:
            payload["verdict"] = "REJECT"
        save_abstention_release_gate = getattr(args, "save_abstention_release_gate", None)
        if save_abstention_release_gate:
            Path(save_abstention_release_gate).parent.mkdir(parents=True, exist_ok=True)
            Path(save_abstention_release_gate).write_text(
                strict_json_dumps(abstention_release_gate, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"\nWrote abstention release gate to {save_abstention_release_gate}")
        print(
            "\n  Abstention release gate: "
            f"status={abstention_release_gate['status']} "
            f"score={abstention_release_gate['selected_score_name']} "
            f"conditional_lb="
            f"{abstention_release_gate['metrics'].get('conditional_correctness_lower_bound')} "
            f"abstention_rate="
            f"{abstention_release_gate['metrics'].get('empirical_abstention_rate')}"
        )

    if adaptive_feature_names:
        adaptive_feature_values = _load_adaptive_feature_values(
            args.scores,
            adaptive_feature_names,
            score_dump=score_dump,
            full_score_dump=full_score_dump,
        )
        adaptive_score_name = getattr(args, "adaptive_score_name", None) or f"{args.signal}_adaptive"
        adaptive_transform = AdaptiveScoreTransform(
            feature_weights=adaptive_feature_weights,
            intercept=float(getattr(args, "adaptive_intercept", 0.0)),
            direction=direction,
        )
        adaptive_report, _ = _run_adaptive_conformal_report(
            base_scores=scores,
            labels=labels,
            base_score_name=args.signal,
            output_score_name=adaptive_score_name,
            feature_values=adaptive_feature_values,
            transform=adaptive_transform,
            repeats=args.repeats,
            seed=args.seed,
        )
        payload["adaptive_conformal_report"] = adaptive_report
        payload["component_verdicts"]["adaptive_conformal"] = adaptive_report["verdict"]
        if adaptive_report["verdict"] != "ACCEPT":
            payload["verdict"] = "REJECT"
        print(
            f"\n  Adaptive conformal: {args.signal} -> {adaptive_score_name} "
            f"features={','.join(adaptive_feature_names)} "
            f"verdict={adaptive_report['verdict']}"
        )

        save_adaptive_calibration = getattr(args, "save_adaptive_calibration", None)
        if save_adaptive_calibration:
            feature_tensors = {
                name: torch.tensor(values, dtype=torch.float64)
                for name, values in adaptive_feature_values.items()
            }
            normal_feature_values = {
                name: values[labels == 0]
                for name, values in feature_tensors.items()
            }
            artifact = AdaptiveConformalCalibrator(
                alpha=args.artifact_alpha,
                transform=adaptive_transform,
            ).calibrate(
                model_id=args.model_id or dump_config.get("model", "unknown"),
                model_revision=args.model_revision,
                target_layer=args.target_layer if args.target_layer is not None else int(dump_config.get("layer", 0)),
                score_name=args.signal,
                calibration_scores=true_scores,
                feature_values=normal_feature_values,
                output_score_name=adaptive_score_name,
                calibration_dataset_metadata={
                    "scores": args.scores,
                    "signal": args.signal,
                    "n_true": n_true,
                    "source": "eval_conformal.py",
                },
                created_at=args.created_at,
                commit_sha=args.commit_sha,
            )
            artifact.save_json(save_adaptive_calibration)
            print(f"\nWrote adaptive calibration artifact to {save_adaptive_calibration}")

    if args.save_calibration:
        artifact = ConformalCalibrator(alpha=args.artifact_alpha).calibrate(
            model_id=args.model_id or dump_config.get("model", "unknown"),
            model_revision=args.model_revision,
            target_layer=args.target_layer if args.target_layer is not None else int(dump_config.get("layer", 0)),
            calibration_scores={args.signal: true_scores},
            directions={args.signal: direction},
            calibration_dataset_metadata={
                "scores": args.scores,
                "signal": args.signal,
                "n_true": n_true,
                "source": "eval_conformal.py",
            },
            created_at=args.created_at,
            commit_sha=args.commit_sha,
        )
        artifact.save_json(args.save_calibration)
        print(f"\nWrote calibration artifact to {args.save_calibration}")

    if wants_sweep:
        selected_signals = selected_sweep_signals or _all_dump_signals(dict(score_dump.summary))
        direction_override = None if args.direction is None else {
            signal: args.direction for signal in selected_signals
        }
        calibrator = LayerScoreSweepCalibrator(
            alpha=args.artifact_alpha,
            best_by=args.best_by,
            max_workers=getattr(args, "sweep_workers", 1),
        )
        sweep_kwargs = {
            "signals": selected_signals,
            "directions": direction_override,
            "model_id": args.model_id or dump_config.get("model", "unknown"),
            "model_revision": args.model_revision,
            "created_at": args.created_at,
            "commit_sha": args.commit_sha,
            "metadata": {"source": "eval_conformal.py", "config": dump_config},
        }
        if preloaded_layer_scores is not None:
            report = calibrator.calibrate_from_layer_scores(
                preloaded_layer_scores,
                scores_path=args.scores,
                **sweep_kwargs,
            )
        elif full_score_dump is None:
            report = calibrator.calibrate_from_file(
                args.scores,
                cache=score_dump_metadata_cache,
                **sweep_kwargs,
            )
        else:
            report = calibrator.calibrate_from_score_dump(
                full_score_dump,
                scores_path=args.scores,
                **sweep_kwargs,
            )
        payload["sweep_report"] = report.to_dict()
        if args.save_sweep_report:
            report.save_json(args.save_sweep_report)
            print(f"\nWrote sweep report to {args.save_sweep_report}")
        if args.save_best_calibration:
            artifact = report.best_artifact(
                calibration_dataset_metadata={"scores": args.scores, "source": "eval_conformal.py"}
            )
            artifact.save_json(args.save_best_calibration)
            print(f"\nWrote best calibration artifact to {args.save_best_calibration}")

    payload["score_dump_cache"] = score_dump_cache_summary(score_dump_metadata_cache)
    _add_planned_manifest_fields(args, payload)
    if args.json:
        Path(args.json).write_text(strict_json_dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {args.json}")
    manifest = _write_artifact_manifest(args, payload)
    if manifest is not None:
        payload["artifact_manifest_summary"] = manifest["summary"]
    return payload


def main():
    p = argparse.ArgumentParser(description="E1: split-conformal coverage validation")
    p.add_argument("--scores", required=True,
                   help="scores JSON from eval_truthfulqa.py --dump-scores")
    p.add_argument("--signal", default="maha_last",
                   help="which signal to calibrate (maha_last / truth_proj / ...)")
    p.add_argument("--signals", default=None,
                   help="optional comma-list of signals for layer/score sweep reports")
    p.add_argument("--repeats", type=int, default=20, help="number of seeded 50/50 splits")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", default=None, help="optional path for structured results")
    p.add_argument("--save-calibration", default=None,
                   help="optional path to write a CalibrationArtifact JSON for the selected signal")
    p.add_argument("--save-adaptive-calibration", default=None,
                   help="optional path to write an adaptive CalibrationArtifact JSON")
    p.add_argument("--save-abstention-report", default=None,
                   help="optional path to write a conformal abstention report JSON")
    p.add_argument("--include-abstention-report", action="store_true",
                   help="include a conformal abstention report in the main JSON payload without a sidecar")
    p.add_argument("--save-abstention-comparison", default=None,
                   help="optional path to write a multi-signal abstention comparison JSON")
    p.add_argument("--include-abstention-comparison", action="store_true",
                   help="include a multi-signal abstention comparison in the main JSON payload")
    p.add_argument("--save-abstention-release-gate", default=None,
                   help="optional path to write an abstention release-gate verdict JSON")
    p.add_argument("--include-abstention-release-gate", action="store_true",
                   help="include an abstention release-gate verdict in the main JSON payload")
    p.add_argument("--multiple-testing-signals", default=None,
                   help="optional comma-list of primary signals for conformal multiple-testing; "
                        "defaults to --signals when present, otherwise --signal")
    p.add_argument("--multiple-testing-alpha", type=float, default=0.10,
                   help="global false-alarm budget for conformal multiple-testing")
    p.add_argument("--multiple-testing-method", choices=("by", "bh", "bonferroni"),
                   default="by",
                   help="multiple-testing correction method for conformal p-values")
    p.add_argument("--multiple-testing-directions", default=None,
                   help="optional comma-list of SIGNAL:higher/lower overrides for "
                        "multiple-testing signals")
    p.add_argument("--save-multiple-testing-report", default=None,
                   help="optional path to write a conformal multiple-testing report JSON")
    p.add_argument("--save-multiple-testing-calibration", default=None,
                   help="optional path to write a runtime conformal multiple-testing calibration artifact")
    p.add_argument("--include-multiple-testing-report", action="store_true",
                   help="include the conformal multiple-testing report in the main JSON payload")
    p.add_argument("--sequential-signal", default=None,
                   help="optional signal for sequential conformal replay; defaults to --signal")
    p.add_argument("--sequential-direction", choices=("higher", "lower"), default=None,
                   help="optional override for sequential conformal anomaly direction")
    p.add_argument("--sequential-alpha", type=float, default=0.10,
                   help="finite alpha budget spent across the sequential replay")
    p.add_argument("--sequential-schedule", choices=("linear", "harmonic", "geometric"),
                   default="harmonic",
                   help="alpha-spending schedule used for sequential conformal replay")
    p.add_argument("--sequential-seed", type=int, default=None,
                   help="optional seed for the true-score calibration split; defaults to --seed")
    p.add_argument("--save-sequential-report", default=None,
                   help="optional path to write a sequential conformal replay report JSON")
    p.add_argument("--save-sequential-calibration", default=None,
                   help="optional path to write a runtime sequential conformal calibration artifact")
    p.add_argument("--include-sequential-report", action="store_true",
                   help="include a sequential conformal replay report in the main JSON payload")
    p.add_argument("--save-sweep-report", default=None,
                   help="optional path to write a LayerScoreSweepReport JSON")
    p.add_argument("--save-best-calibration", default=None,
                   help="optional path to write the best CalibrationArtifact from the sweep report")
    p.add_argument("--artifact-manifest", default=None,
                   help="optional path to write an artifact manifest for inputs and generated outputs")
    p.add_argument("--best-by", choices=("auroc", "detection"), default="auroc",
                   help="metric used to choose the best layer/score calibration artifact")
    p.add_argument("--sweep-workers", type=int, default=1,
                   help="maximum worker threads for layer/score sweep calibration")
    p.add_argument("--artifact-alpha", type=float, default=0.10,
                   help="alpha used for --save-calibration artifact threshold")
    p.add_argument("--abstention-alpha", type=float, default=0.10,
                   help="correct-response miss budget used for conformal abstention")
    p.add_argument("--abstention-signal", default=None,
                   help="optional score used for conformal abstention; defaults to --signal")
    p.add_argument("--abstention-direction", choices=("higher", "lower"), default=None,
                   help="optional override for which side of abstention signal(s) is less reliable")
    p.add_argument("--abstention-signals", default=None,
                   help="optional comma-list of signals for abstention comparison; "
                        "defaults to --signals when present, otherwise --abstention-signal/--signal")
    p.add_argument("--abstention-best-by", choices=ABSTENTION_COMPARISON_METRICS,
                   default="conditional_correctness_lower_bound",
                   help="metric used to rank abstention comparison candidates")
    p.add_argument("--min-abstention-conditional-correctness-lower-bound",
                   type=float, default=0.80,
                   help="minimum conservative conditional-correctness lower bound "
                        "required by the abstention release gate")
    p.add_argument("--max-abstention-rate", type=float, default=0.50,
                   help="maximum empirical abstention rate allowed by the abstention release gate")
    p.add_argument("--direction", choices=("higher", "lower"), default=None,
                   help="optional override for whether higher or lower signal values are more anomalous")
    p.add_argument("--adaptive-feature", action="append", default=None,
                   help="primary score or dump extra used to inflate adaptive conformal scores; "
                        "may be repeated or comma-separated")
    p.add_argument("--adaptive-feature-weight", action="append", default=None,
                   help="adaptive feature weight in NAME=FLOAT form; omitted features default to 1.0")
    p.add_argument("--adaptive-intercept", type=float, default=0.0,
                   help="constant added to the adaptive anomaly score")
    p.add_argument("--adaptive-score-name", default=None,
                   help="score name stored in adaptive reports/artifacts")
    p.add_argument("--confidence-signal", default="nll_answer",
                   help="optional primary score used to audit high-confidence errors when present")
    p.add_argument("--confidence-direction", choices=("higher", "lower"), default=None,
                   help="whether higher or lower confidence-signal values mean more confidence "
                        "(default: lower for nll_answer, higher otherwise)")
    p.add_argument("--confidence-top-fraction", type=float, default=0.25,
                   help="fraction of records treated as the high-confidence region for confidence-error audit")
    p.add_argument("--disable-confidence-audit", action="store_true",
                   help="disable high-confidence error audit even when the confidence signal is present")
    p.add_argument("--model-id", default=None, help="override model_id stored in the artifact")
    p.add_argument("--model-revision", default=None, help="optional model revision stored in the artifact")
    p.add_argument("--target-layer", type=int, default=None, help="override target layer stored in the artifact")
    p.add_argument("--created-at", default=None, help="optional artifact timestamp")
    p.add_argument("--commit-sha", default=None, help="optional repository commit SHA")
    run(p.parse_args())


if __name__ == "__main__":
    main()

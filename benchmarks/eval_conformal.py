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
from typing import Sequence

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
        },
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
    if wants_abstention_release_gate and not wants_abstention_comparison:
        wants_abstention_report = True
    abstention_comparison_signals = _resolve_abstention_comparison_signals(
        args,
        abstention_signal=abstention_signal,
        enabled=wants_abstention_comparison,
    )
    additional_signals = tuple(dict.fromkeys((
        *adaptive_feature_names,
        *(name for name in (abstention_signal,) if name != args.signal),
        *(name for name in abstention_comparison_signals if name != args.signal),
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
                json.dumps(abstention_report, indent=2, sort_keys=True) + "\n",
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
                json.dumps(abstention_comparison_report, indent=2, sort_keys=True) + "\n",
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
                json.dumps(abstention_release_gate, indent=2, sort_keys=True) + "\n",
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
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
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

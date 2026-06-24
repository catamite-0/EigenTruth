"""EigenTruth eval metrics — 与模型无关的评分指标 / Model-free scoring metrics.

这些是对分数/表征的纯函数（无模型、无数据集依赖），可在 CPU 上离线单元测试。
Pure functions over scores/representations (no model or dataset deps); CPU-testable offline.
"""

from __future__ import annotations

import math
from typing import Any, Sequence, Union

import torch
from torch import Tensor

ArrayLike = Union[Tensor, Sequence[float]]


def _average_ranks(x: Tensor) -> Tensor:
    """返回 1-based 排名，平局取平均排名（等价于 scipy.stats.rankdata 的 'average'）。
    Return 1-based ranks with ties resolved to the average rank (like scipy rankdata).
    """
    n = x.numel()
    order = torch.argsort(x)
    sorted_x = x[order]
    ranks = torch.empty(n, dtype=torch.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and bool(sorted_x[j + 1] == sorted_x[i]):
            j += 1
        # i..j 是一组平局，取 1-based 排名 (i+1)..(j+1) 的平均
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def roc_auc(scores: ArrayLike, labels: ArrayLike) -> float:
    """计算 ROC 曲线下面积 (AUROC)。
    Area under the ROC curve.

    约定：label 1 = 正类（例如"假陈述/幻觉"，即希望被高分标记的对象），label 0 = 负类。
    分数越高 => 越倾向正类。平局按平均排名处理（Mann–Whitney U 等价式）。
    Convention: label 1 = positive class (e.g. the false/hallucinated item we want to
    flag with a high score), label 0 = negative. Higher score => more positive. Ties are
    handled via average ranks (equivalent to the Mann–Whitney U statistic).

    Args:
        scores: 每个样本的分数 / per-item scores, shape [N].
        labels: 0/1 标签 / binary labels in {0, 1}, shape [N].

    Returns:
        AUROC ∈ [0, 1]；若某一类缺失则返回 float('nan')。
        AUROC in [0, 1]; returns float('nan') if either class is absent.
    """
    scores_t = torch.as_tensor(scores, dtype=torch.float64).flatten()
    labels_t = torch.as_tensor(labels, dtype=torch.float64).flatten()
    if scores_t.numel() != labels_t.numel():
        raise ValueError("scores and labels must have the same length.")

    pos_mask = labels_t == 1
    neg_mask = labels_t == 0
    n_pos = int(pos_mask.sum().item())
    n_neg = int(neg_mask.sum().item())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = _average_ranks(scores_t)
    sum_ranks_pos = float(ranks[pos_mask].sum().item())
    # AUROC = (R+ - n_pos(n_pos+1)/2) / (n_pos * n_neg)
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def euclidean_dispersion(points: Tensor) -> Tensor:
    """一组点到其质心的平均欧氏距离（双曲 HSE 的欧氏对照基线）。
    Mean Euclidean distance of points to their centroid — the Euclidean counterpart
    of `hyperbolic_semantic_entropy`, used for the "does hyperbolic help?" ablation.

    Args:
        points: 点集 / point set, shape [N, D].

    Returns:
        标量张量 (>=0)；N<=1 时为 0 / scalar tensor (>=0); 0 when N <= 1.
    """
    points = points.to(torch.float32)
    if points.shape[0] <= 1:
        return torch.tensor(0.0)
    centroid = points.mean(dim=0, keepdim=True)
    return torch.norm(points - centroid, dim=-1).mean()


def binomial_confidence_interval(successes: int, total: int, *, z: float = 1.96) -> dict[str, Any]:
    """Return a normal-approximation binomial confidence interval.

    The helper is intentionally dependency-free. For empty denominators it
    returns ``None`` estimates, which serializes cleanly to JSON and keeps
    benchmark reports explicit about unavailable rates.
    """
    successes = int(successes)
    total = int(total)
    if total < 0:
        raise ValueError("total must be non-negative.")
    if successes < 0 or successes > total:
        raise ValueError("successes must be in [0, total].")
    if total == 0:
        return {"estimate": None, "lower": None, "upper": None, "successes": successes, "total": total}
    estimate = successes / total
    margin = z * (estimate * (1.0 - estimate) / total) ** 0.5
    return {
        "estimate": estimate,
        "lower": max(0.0, estimate - margin),
        "upper": min(1.0, estimate + margin),
        "successes": successes,
        "total": total,
    }


def selective_classification_report(
    scores: ArrayLike,
    labels: ArrayLike,
    threshold: float,
    *,
    direction: str = "higher",
) -> dict[str, Any]:
    """Summarize flag/accept behavior for calibrated anomaly scores.

    ``label == 1`` denotes the anomalous/false class. A flagged item is treated
    as routed away from direct acceptance. ``selective_accuracy`` is therefore
    the fraction of accepted items that are normal/true.
    """
    if direction not in {"higher", "lower"}:
        raise ValueError("direction must be 'higher' or 'lower'.")
    scores_t = torch.as_tensor(scores, dtype=torch.float64).flatten()
    labels_t = torch.as_tensor(labels, dtype=torch.int64).flatten()
    if scores_t.numel() != labels_t.numel():
        raise ValueError("scores and labels must have the same length.")
    if not torch.logical_or(labels_t == 0, labels_t == 1).all():
        raise ValueError("labels must be binary values in {0, 1}.")

    if direction == "higher":
        flagged = scores_t > float(threshold)
    else:
        flagged = scores_t < float(threshold)
    accepted = ~flagged
    normal = labels_t == 0
    anomalous = labels_t == 1

    n_total = int(labels_t.numel())
    n_true = int(normal.sum().item())
    n_false = int(anomalous.sum().item())
    n_flagged = int(flagged.sum().item())
    n_accepted = int(accepted.sum().item())
    accepted_true = int(torch.logical_and(accepted, normal).sum().item())
    flagged_false = int(torch.logical_and(flagged, anomalous).sum().item())
    false_alarms = int(torch.logical_and(flagged, normal).sum().item())
    correct = accepted_true + flagged_false

    coverage_ci = binomial_confidence_interval(n_accepted, n_total)
    selective_accuracy_ci = binomial_confidence_interval(accepted_true, n_accepted)
    detection_ci = binomial_confidence_interval(flagged_false, n_false)
    false_alarm_ci = binomial_confidence_interval(false_alarms, n_true)
    accuracy_ci = binomial_confidence_interval(correct, n_total)

    return {
        "threshold": float(threshold),
        "direction": direction,
        "n_total": n_total,
        "n_true": n_true,
        "n_false": n_false,
        "n_flagged": n_flagged,
        "n_accepted": n_accepted,
        "coverage": coverage_ci["estimate"],
        "coverage_ci": coverage_ci,
        "selective_accuracy": selective_accuracy_ci["estimate"],
        "selective_accuracy_ci": selective_accuracy_ci,
        "accuracy": accuracy_ci["estimate"],
        "accuracy_ci": accuracy_ci,
        "detection": detection_ci["estimate"],
        "detection_ci": detection_ci,
        "false_alarm": false_alarm_ci["estimate"],
        "false_alarm_ci": false_alarm_ci,
    }


def confidence_error_report(
    anomaly_scores: ArrayLike,
    labels: ArrayLike,
    anomaly_threshold: float,
    confidence_scores: ArrayLike,
    *,
    anomaly_direction: str = "higher",
    confidence_direction: str = "higher",
    confidence_top_fraction: float = 0.25,
) -> dict[str, Any]:
    """Summarize high-confidence errors under a calibrated anomaly gate.

    ``label == 1`` denotes the anomalous/false class. ``confidence_scores`` are
    transformed so higher values mean higher confidence; set
    ``confidence_direction="lower"`` for signals such as NLL where lower values
    mean the model is more confident.
    """
    if anomaly_direction not in {"higher", "lower"}:
        raise ValueError("anomaly_direction must be 'higher' or 'lower'.")
    if confidence_direction not in {"higher", "lower"}:
        raise ValueError("confidence_direction must be 'higher' or 'lower'.")
    if not (0.0 < confidence_top_fraction <= 1.0):
        raise ValueError("confidence_top_fraction must be in (0, 1].")

    anomaly_t = torch.as_tensor(anomaly_scores, dtype=torch.float64).flatten()
    labels_t = torch.as_tensor(labels, dtype=torch.int64).flatten()
    confidence_raw = torch.as_tensor(confidence_scores, dtype=torch.float64).flatten()
    if anomaly_t.numel() != labels_t.numel() or confidence_raw.numel() != labels_t.numel():
        raise ValueError("scores and labels must have the same length.")
    if not torch.logical_or(labels_t == 0, labels_t == 1).all():
        raise ValueError("labels must be binary values in {0, 1}.")
    if not torch.isfinite(anomaly_t).all() or not torch.isfinite(confidence_raw).all():
        raise ValueError("scores must contain only finite values.")

    if anomaly_direction == "higher":
        flagged = anomaly_t > float(anomaly_threshold)
    else:
        flagged = anomaly_t < float(anomaly_threshold)
    accepted = ~flagged

    confidence_native = confidence_raw if confidence_direction == "higher" else -confidence_raw
    top_count = max(1, int(math.ceil(float(confidence_top_fraction) * confidence_native.numel())))
    sorted_confidence, _ = torch.sort(confidence_native)
    confidence_native_threshold = sorted_confidence[-top_count]
    high_confidence = confidence_native >= confidence_native_threshold
    confidence_threshold = (
        float(confidence_native_threshold.item())
        if confidence_direction == "higher"
        else float((-confidence_native_threshold).item())
    )

    normal = labels_t == 0
    anomalous = labels_t == 1
    high_false = torch.logical_and(high_confidence, anomalous)
    high_true = torch.logical_and(high_confidence, normal)
    high_accepted = torch.logical_and(high_confidence, accepted)
    high_flagged = torch.logical_and(high_confidence, flagged)
    high_accepted_false = torch.logical_and(high_accepted, anomalous)
    high_accepted_true = torch.logical_and(high_accepted, normal)
    high_flagged_false = torch.logical_and(high_flagged, anomalous)
    low_confidence = ~high_confidence
    low_true = torch.logical_and(low_confidence, normal)
    low_false = torch.logical_and(low_confidence, anomalous)

    n_total = int(labels_t.numel())
    n_high = int(high_confidence.sum().item())
    n_high_false = int(high_false.sum().item())
    n_high_true = int(high_true.sum().item())
    n_high_accepted = int(high_accepted.sum().item())
    n_high_accepted_false = int(high_accepted_false.sum().item())
    n_high_accepted_true = int(high_accepted_true.sum().item())
    n_high_flagged = int(high_flagged.sum().item())
    n_high_flagged_false = int(high_flagged_false.sum().item())
    n_low_true = int(low_true.sum().item())
    n_low_false = int(low_false.sum().item())

    return {
        "anomaly_threshold": float(anomaly_threshold),
        "anomaly_direction": anomaly_direction,
        "confidence_direction": confidence_direction,
        "confidence_top_fraction": float(confidence_top_fraction),
        "confidence_threshold": confidence_threshold,
        "n_total": n_total,
        "n_high_confidence": n_high,
        "n_low_confidence": n_total - n_high,
        "n_high_confidence_true": n_high_true,
        "n_high_confidence_false": n_high_false,
        "n_low_confidence_true": n_low_true,
        "n_low_confidence_false": n_low_false,
        "high_confidence_false_rate": binomial_confidence_interval(n_high_false, n_high),
        "n_high_confidence_accepted": n_high_accepted,
        "n_high_confidence_accepted_true": n_high_accepted_true,
        "n_high_confidence_accepted_false": n_high_accepted_false,
        "high_confidence_accepted_false_rate": binomial_confidence_interval(
            n_high_accepted_false,
            n_high_accepted,
        ),
        "high_confidence_accepted_selective_accuracy": binomial_confidence_interval(
            n_high_accepted_true,
            n_high_accepted,
        ),
        "n_high_confidence_flagged": n_high_flagged,
        "n_high_confidence_flagged_false": n_high_flagged_false,
        "high_confidence_false_capture_rate": binomial_confidence_interval(
            n_high_flagged_false,
            n_high_false,
        ),
        "high_confidence_false_miss_rate": binomial_confidence_interval(
            n_high_accepted_false,
            n_high_false,
        ),
    }

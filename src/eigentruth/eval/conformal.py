"""EigenTruth conformal — split-conformal 校准 / Split-conformal calibration.

把任意异常分数（如马氏距离）转换为具有有限样本保证的 p 值与报警阈值。
Turns any anomaly score (e.g. Mahalanobis distance) into p-values and alarm
thresholds with finite-sample guarantees.

前提 / Assumption: 校准分数与测试点可交换（来自同一"正常"总体）。
The calibration scores and the test point are exchangeable (drawn from the
same "normal" population).

保证 / Guarantee: 对可交换的测试点，P(p_value <= alpha) <= alpha —— 即按
`p <= alpha` 报警的误报率不超过 alpha；同理 `score > conformal_threshold(alpha)`
的误报率不超过 alpha。(Vovk et al.; Angelopoulos & Bates 2023 tutorial.)
For an exchangeable test point, P(p-value <= alpha) <= alpha, so flagging at
`p <= alpha` (equivalently `score > conformal_threshold(alpha)`) has a
false-alarm rate of at most alpha.

约定 / Convention: 分数越高越异常 / higher score = more anomalous.
"""

from __future__ import annotations

import math
from typing import Sequence, Union

import torch
from torch import Tensor

ArrayLike = Union[Tensor, Sequence[float]]


def conformal_pvalues(calib_scores: ArrayLike, test_scores: ArrayLike) -> Tensor:
    """计算每个测试分数的保守共形 p 值。
    Conservative split-conformal p-value for each test score.

    p_i = (1 + #{calib >= s_i}) / (n_calib + 1)

    平局计入 >=（保守方向）。p 值落在 (0, 1]。
    Ties count toward >= (the conservative direction). P-values lie in (0, 1].

    Args:
        calib_scores: 校准分数（"正常"总体）/ calibration scores, shape [n_calib].
        test_scores: 测试分数 / test scores, shape [n_test].

    Returns:
        p 值张量 / p-value tensor (float64), shape [n_test].
    """
    calib = torch.as_tensor(calib_scores, dtype=torch.float64).flatten()
    test = torch.as_tensor(test_scores, dtype=torch.float64).flatten()
    if calib.numel() == 0:
        raise ValueError("calibration scores must be non-empty.")

    calib_sorted, _ = torch.sort(calib)
    # searchsorted(right=False) 给出 #{calib < s}，故 #{calib >= s} = n - idx
    idx = torch.searchsorted(calib_sorted, test, right=False)
    n_ge = calib.numel() - idx
    return (1.0 + n_ge.to(torch.float64)) / (calib.numel() + 1.0)


def conformal_threshold(calib_scores: ArrayLike, alpha: float) -> float:
    """给定误报预算 alpha，返回报警阈值 t。
    Alarm threshold t for a false-alarm budget alpha.

    对可交换测试点，P(score > t) <= alpha。t 取校准分数的第
    ceil((n+1)(1-alpha)) 个次序统计量；当该阶数超过 n（校准样本太少，
    无法支撑该置信水平）时返回 +inf（永不报警）。
    For an exchangeable test point, P(score > t) <= alpha. t is the
    ceil((n+1)(1-alpha))-th order statistic of the calibration scores; if that
    rank exceeds n (too few calibration samples for this alpha), returns +inf
    (never alarm).

    Args:
        calib_scores: 校准分数 / calibration scores, shape [n_calib].
        alpha: 误报预算 / false-alarm budget, in (0, 1).

    Returns:
        阈值 (float)；样本不足时为 +inf / threshold; +inf when n is too small.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    calib = torch.as_tensor(calib_scores, dtype=torch.float64).flatten()
    n = calib.numel()
    if n == 0:
        raise ValueError("calibration scores must be non-empty.")

    rank = math.ceil((n + 1) * (1.0 - alpha))
    if rank > n:
        return float("inf")
    calib_sorted, _ = torch.sort(calib)
    return float(calib_sorted[rank - 1].item())


def directional_conformal_threshold(calib_scores: ArrayLike, alpha: float, direction: str) -> float:
    """Return a native-unit threshold for higher- or lower-is-anomalous scores."""
    if direction == "higher":
        return conformal_threshold(calib_scores, alpha)
    if direction == "lower":
        threshold = conformal_threshold(-torch.as_tensor(calib_scores, dtype=torch.float64), alpha)
        return -threshold
    raise ValueError("direction must be 'higher' or 'lower'.")


def directional_trigger_rate(scores: ArrayLike, threshold: float, direction: str) -> float:
    """Return the fraction of scores flagged by a directional conformal threshold."""
    if direction not in {"higher", "lower"}:
        raise ValueError("direction must be 'higher' or 'lower'.")
    scores_t = torch.as_tensor(scores, dtype=torch.float64).flatten()
    if scores_t.numel() == 0 or math.isinf(threshold):
        return 0.0
    if direction == "higher":
        return float((scores_t > threshold).double().mean().item())
    return float((scores_t < threshold).double().mean().item())

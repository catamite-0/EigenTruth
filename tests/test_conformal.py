"""单元测试 — eigentruth.eval.conformal

纯函数测试，CPU 可运行。重点验证有限样本保证的方向性：
误报率 <= alpha（可交换时），以及平局/小样本的保守处理。
"""

import math

import pytest
import torch

from eigentruth.eval.conformal import (
    AdaptiveScoreTransform,
    adaptive_anomaly_scores,
    conformal_pvalues,
    conformal_threshold,
    directional_conformal_threshold,
    directional_trigger_rate,
)


class TestConformalPvalues:
    """共形 p 值测试。"""

    def test_pvalues_in_unit_interval(self):
        torch.manual_seed(0)
        p = conformal_pvalues(torch.randn(100), torch.randn(50))
        assert (p > 0).all() and (p <= 1).all()

    def test_monotone_decreasing_in_score(self):
        """分数越高（越异常），p 值越小。"""
        calib = torch.randn(200)
        test = torch.tensor([-2.0, 0.0, 2.0, 5.0])
        p = conformal_pvalues(calib, test)
        assert (p[1:] <= p[:-1] + 1e-12).all()

    def test_extreme_score_gets_minimal_pvalue(self):
        """超过所有校准分数的测试点 → p = 1/(n+1)。"""
        calib = torch.randn(99)
        p = conformal_pvalues(calib, torch.tensor([1e6]))
        assert p.item() == pytest.approx(1.0 / 100.0)

    def test_all_ties_give_p_one(self):
        """与全部校准分数持平 → p = 1（平局保守计入 >=）。"""
        calib = torch.full((50,), 3.0)
        p = conformal_pvalues(calib, torch.tensor([3.0]))
        assert p.item() == pytest.approx(1.0)

    def test_superuniform_under_exchangeability(self):
        """可交换时 P(p <= alpha) <= alpha（允许小幅统计噪声）。"""
        torch.manual_seed(42)
        calib = torch.randn(500)
        test = torch.randn(4000)
        p = conformal_pvalues(calib, test)
        for alpha in (0.05, 0.1, 0.2):
            rate = (p <= alpha).double().mean().item()
            assert rate <= alpha + 0.02, f"alpha={alpha}: rate={rate}"

    def test_shifted_distribution_detected(self):
        """偏移分布的测试点应得到显著更小的 p 值。"""
        torch.manual_seed(7)
        calib = torch.randn(500)
        p_shift = conformal_pvalues(calib, torch.randn(500) + 3.0)
        assert (p_shift <= 0.05).double().mean().item() > 0.5

    def test_empty_calibration_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            conformal_pvalues(torch.tensor([]), torch.tensor([1.0]))


class TestConformalThreshold:
    """共形报警阈值测试。"""

    def test_false_alarm_rate_bounded(self):
        """可交换测试点上，score > t 的比例 <= alpha（+统计噪声）。"""
        torch.manual_seed(11)
        calib = torch.randn(500)
        test = torch.randn(4000)
        for alpha in (0.05, 0.1, 0.2):
            t = conformal_threshold(calib, alpha)
            fa = (test > t).double().mean().item()
            assert fa <= alpha + 0.02, f"alpha={alpha}: fa={fa}"

    def test_threshold_monotone_in_alpha(self):
        """alpha 越小（越严格），阈值越高。"""
        torch.manual_seed(3)
        calib = torch.randn(300)
        t_strict = conformal_threshold(calib, 0.01)
        t_loose = conformal_threshold(calib, 0.2)
        assert t_strict >= t_loose

    def test_insufficient_calibration_returns_inf(self):
        """校准样本不足以支撑该 alpha 时返回 +inf（永不报警，保守）。"""
        calib = torch.randn(5)  # ceil(6 * 0.99) = 6 > 5
        assert math.isinf(conformal_threshold(calib, 0.01))

    def test_invalid_alpha_raises(self):
        calib = torch.randn(10)
        for bad in (0.0, 1.0, -0.1, 1.5):
            with pytest.raises(ValueError, match="alpha"):
                conformal_threshold(calib, bad)

    def test_empty_calibration_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            conformal_threshold(torch.tensor([]), 0.1)

    def test_consistency_with_pvalues(self):
        """score > threshold(alpha) 等价于 pvalue(score) <= alpha。"""
        torch.manual_seed(5)
        calib = torch.randn(200)
        test = torch.randn(500)
        alpha = 0.1
        t = conformal_threshold(calib, alpha)
        p = conformal_pvalues(calib, test)
        flagged_by_t = test > t
        flagged_by_p = p <= alpha
        assert (flagged_by_t == flagged_by_p).all()

    def test_directional_threshold_and_trigger_rate_support_lower_scores(self):
        calib = torch.tensor([10.0, 11.0, 12.0, 13.0])

        threshold = directional_conformal_threshold(calib, 0.25, "lower")
        rate = directional_trigger_rate(torch.tensor([9.0, 10.0, 11.0]), threshold, "lower")

        assert threshold == pytest.approx(10.0)
        assert rate == pytest.approx(1.0 / 3.0)

    def test_directional_helpers_reject_invalid_direction(self):
        with pytest.raises(ValueError, match="direction"):
            directional_conformal_threshold(torch.tensor([1.0]), 0.1, "sideways")
        with pytest.raises(ValueError, match="direction"):
            directional_trigger_rate(torch.tensor([1.0]), 0.0, "sideways")

    def test_conformal_helpers_reject_non_finite_scores(self):
        with pytest.raises(ValueError, match="finite"):
            conformal_threshold(torch.tensor([1.0, float("nan")]), 0.1)
        with pytest.raises(ValueError, match="finite"):
            conformal_pvalues(torch.tensor([1.0, 2.0]), torch.tensor([float("inf")]))
        with pytest.raises(ValueError, match="finite"):
            directional_conformal_threshold(torch.tensor([1.0, float("-inf")]), 0.1, "lower")
        with pytest.raises(ValueError, match="NaN"):
            directional_trigger_rate(torch.tensor([1.0]), float("nan"), "higher")


class TestAdaptiveConformalScores:
    def test_adaptive_anomaly_scores_inflate_by_features(self):
        adjusted = adaptive_anomaly_scores(
            [1.0, 2.0, 3.0],
            feature_values={"entropy": [0.0, 1.0, 2.0], "citation": [1.0, 0.0, 1.0]},
            feature_weights={"entropy": 0.5, "citation": 2.0},
            intercept=0.25,
        )

        assert adjusted.tolist() == pytest.approx([3.25, 2.75, 6.25])

    def test_adaptive_anomaly_scores_orient_lower_direction(self):
        adjusted = adaptive_anomaly_scores(
            [10.0, 8.0],
            feature_values={"risk": [0.0, 1.0]},
            feature_weights={"risk": 2.0},
            direction="lower",
        )

        assert adjusted.tolist() == pytest.approx([-10.0, -6.0])
        assert adjusted[1] > adjusted[0]

    def test_adaptive_score_transform_roundtrip(self):
        transform = AdaptiveScoreTransform(
            feature_weights={"semantic_entropy": 1.5},
            intercept=0.25,
            direction="higher",
        )

        loaded = AdaptiveScoreTransform.from_dict(transform.to_dict())
        adjusted = loaded.transform([1.0, 2.0], {"semantic_entropy": [0.0, 1.0]})

        assert loaded == transform
        assert adjusted.tolist() == pytest.approx([1.25, 3.75])

    def test_adaptive_score_transform_rejects_invalid_inputs(self):
        with pytest.raises(ValueError, match="direction"):
            AdaptiveScoreTransform(direction="sideways")
        with pytest.raises(ValueError, match="finite"):
            AdaptiveScoreTransform(feature_weights={"risk": float("nan")})
        with pytest.raises(ValueError, match="missing required feature"):
            adaptive_anomaly_scores([1.0], feature_weights={"risk": 1.0})
        with pytest.raises(ValueError, match="same length"):
            adaptive_anomaly_scores(
                [1.0, 2.0],
                feature_values={"risk": [1.0]},
                feature_weights={"risk": 1.0},
            )

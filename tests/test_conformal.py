"""单元测试 — eigentruth.eval.conformal

纯函数测试，CPU 可运行。重点验证有限样本保证的方向性：
误报率 <= alpha（可交换时），以及平局/小样本的保守处理。
"""

import math

import pytest
import torch

from eigentruth.eval.conformal import (
    AdaptiveScoreTransform,
    ConformalAbstentionReport,
    adaptive_anomaly_scores,
    conformal_abstention_report,
    conformal_pvalues,
    conformal_threshold,
    directional_conformal_threshold,
    directional_conformal_thresholds,
    directional_trigger_rate,
    evaluate_conformal_abstention,
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

    def test_directional_thresholds_match_single_alpha_helper(self):
        calib = torch.tensor([10.0, 11.0, 12.0, 13.0])

        for direction in ("higher", "lower"):
            thresholds = directional_conformal_thresholds(calib, (0.25, 0.5), direction)

            assert thresholds[0.25] == pytest.approx(
                directional_conformal_threshold(calib, 0.25, direction)
            )
            assert thresholds[0.5] == pytest.approx(
                directional_conformal_threshold(calib, 0.5, direction)
            )

    def test_directional_helpers_reject_invalid_direction(self):
        with pytest.raises(ValueError, match="direction"):
            directional_conformal_threshold(torch.tensor([1.0]), 0.1, "sideways")
        with pytest.raises(ValueError, match="direction"):
            directional_conformal_thresholds(torch.tensor([1.0]), (0.1,), "sideways")
        with pytest.raises(ValueError, match="direction"):
            directional_trigger_rate(torch.tensor([1.0]), 0.0, "sideways")

    def test_directional_trigger_rate_preserves_infinite_threshold_comparison_semantics(self):
        scores = torch.tensor([1.0, 2.0, 3.0])

        assert directional_trigger_rate(scores, float("inf"), "higher") == 0.0
        assert directional_trigger_rate(scores, float("-inf"), "higher") == 1.0
        assert directional_trigger_rate(scores, float("-inf"), "lower") == 0.0
        assert directional_trigger_rate(scores, float("inf"), "lower") == 1.0

    def test_conformal_helpers_reject_non_finite_scores(self):
        with pytest.raises(ValueError, match="finite"):
            conformal_threshold(torch.tensor([1.0, float("nan")]), 0.1)
        with pytest.raises(ValueError, match="finite"):
            conformal_pvalues(torch.tensor([1.0, 2.0]), torch.tensor([float("inf")]))
        with pytest.raises(ValueError, match="finite"):
            directional_conformal_threshold(torch.tensor([1.0, float("-inf")]), 0.1, "lower")
        with pytest.raises(ValueError, match="NaN"):
            directional_trigger_rate(torch.tensor([1.0]), float("nan"), "higher")


class TestConformalAbstention:
    def test_higher_uncertainty_report_and_runtime_decision(self):
        report = conformal_abstention_report(
            [0.1, 0.2, 0.3, 0.4, 0.9],
            [1, 1, 1, 0, 0],
            0.4,
            score_name="uncertainty",
        )

        assert report.threshold == pytest.approx(0.3)
        assert report.n_calibration == 5
        assert report.n_correct == 3
        assert report.retained_count == 3
        assert report.abstained_count == 2
        assert report.correct_retained_count == 3
        assert report.empirical_base_accuracy == pytest.approx(0.6)
        assert report.empirical_participation_rate == pytest.approx(0.6)
        assert report.empirical_selective_accuracy == pytest.approx(1.0)
        assert report.correct_retention_lower_bound == pytest.approx(0.75)
        assert report.participation_upper_bound == pytest.approx(4.0 / 6.0)
        assert report.conditional_correctness_lower_bound == pytest.approx(0.75 * (3.0 / 6.0) / (4.0 / 6.0))

        keep = report.decide(0.25, metadata={"request_id": "r1"})
        abstain = report.decide(0.8)
        loaded = ConformalAbstentionReport.from_dict(report.to_dict())

        assert keep.participate is True
        assert keep.action == "participate"
        assert keep.metadata["request_id"] == "r1"
        assert keep.metadata["score_name"] == "uncertainty"
        assert abstain.participate is False
        assert abstain.action == "abstain"
        assert loaded == report

    def test_lower_uncertainty_report_uses_lower_tail_as_abstain_region(self):
        report = conformal_abstention_report(
            [10.0, 9.0, 8.0, 7.0, 6.0],
            [1, 1, 0, 0, 0],
            0.4,
            direction="lower",
        )

        assert report.threshold == pytest.approx(9.0)
        assert report.retained_count == 2
        assert report.abstained_count == 3
        assert report.should_participate(9.0) is True
        assert report.should_participate(8.0) is False
        assert report.decide(8.0).action == "abstain"

    def test_evaluate_conformal_abstention_handles_no_retained_or_correct_samples(self):
        report = evaluate_conformal_abstention(
            [1.0, 2.0, 3.0],
            [0, 0, 0],
            threshold=0.0,
            alpha=0.5,
        )

        assert report.retained_count == 0
        assert report.empirical_selective_accuracy is None
        assert report.correct_retention_rate == 0.0
        assert report.conditional_correctness_lower_bound == 0.0

    def test_conformal_abstention_rejects_invalid_inputs(self):
        with pytest.raises(ValueError, match="same length"):
            conformal_abstention_report([1.0, 2.0], [1], 0.5)
        with pytest.raises(ValueError, match="0/1"):
            conformal_abstention_report([1.0, 2.0], [1, 0.5], 0.5)
        with pytest.raises(ValueError, match="finite"):
            conformal_abstention_report([1.0, float("nan")], [1, 0], 0.5)
        with pytest.raises(ValueError, match="at least one correct"):
            conformal_abstention_report([1.0, 2.0], [0, 0], 0.5)
        with pytest.raises(ValueError, match="alpha"):
            conformal_abstention_report([1.0, 2.0], [1, 0], 1.0)
        with pytest.raises(ValueError, match="direction"):
            conformal_abstention_report([1.0, 2.0], [1, 0], 0.5, direction="sideways")
        with pytest.raises(ValueError, match="threshold"):
            evaluate_conformal_abstention([1.0], [1], threshold=float("nan"), alpha=0.5)


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
        with pytest.raises(ValueError, match="finite"):
            AdaptiveScoreTransform.from_dict({"feature_weights": {"risk": True}})
        with pytest.raises(ValueError, match="finite"):
            AdaptiveScoreTransform.from_dict({"intercept": False})
        with pytest.raises(ValueError, match="missing required feature"):
            adaptive_anomaly_scores([1.0], feature_weights={"risk": 1.0})
        with pytest.raises(ValueError, match="same length"):
            adaptive_anomaly_scores(
                [1.0, 2.0],
                feature_values={"risk": [1.0]},
                feature_weights={"risk": 1.0},
            )

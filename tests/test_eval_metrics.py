"""单元测试 — eigentruth.eval.metrics

纯函数测试，CPU 可运行，无需模型或网络。
"""

import math

import pytest
import torch

from eigentruth.eval.metrics import (
    binomial_confidence_interval,
    euclidean_dispersion,
    roc_auc,
    selective_classification_report,
)


class TestRocAuc:
    """AUROC 计算测试（含已知值、平局、缺类）。"""

    def test_perfect_separation(self):
        """正类分数全高于负类 → AUROC = 1.0。"""
        scores = [0.1, 0.2, 0.3, 0.9, 1.0, 1.1]
        labels = [0, 0, 0, 1, 1, 1]
        assert roc_auc(scores, labels) == 1.0

    def test_perfect_inversion(self):
        """正类分数全低于负类 → AUROC = 0.0。"""
        scores = [0.9, 1.0, 1.1, 0.1, 0.2, 0.3]
        labels = [0, 0, 0, 1, 1, 1]
        assert roc_auc(scores, labels) == 0.0

    def test_known_partial_value(self):
        """已知部分分离值：neg={0.0,0.1}, pos={0.2,0.05} → 3/4 = 0.75。"""
        scores = [0.0, 0.1, 0.2, 0.05]
        labels = [0, 0, 1, 1]
        assert roc_auc(scores, labels) == pytest.approx(0.75)

    def test_chance_value(self):
        """交错分数 → AUROC = 0.5。neg={0.0,0.3}, pos={0.1,0.2}: 2/4。"""
        scores = [0.0, 0.3, 0.1, 0.2]
        labels = [0, 0, 1, 1]
        assert roc_auc(scores, labels) == pytest.approx(0.5)

    def test_all_ties_is_half(self):
        """全部分数相同 → 平均排名 → AUROC = 0.5。"""
        scores = [0.5, 0.5, 0.5, 0.5]
        labels = [0, 1, 0, 1]
        assert roc_auc(scores, labels) == pytest.approx(0.5)

    def test_absent_class_returns_nan(self):
        """缺少某一类时返回 NaN。"""
        assert math.isnan(roc_auc([1.0, 2.0, 3.0], [0, 0, 0]))
        assert math.isnan(roc_auc([1.0, 2.0, 3.0], [1, 1, 1]))

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            roc_auc([1.0, 2.0], [1])

    def test_accepts_tensors(self):
        """接受张量输入。"""
        scores = torch.tensor([0.1, 0.9, 0.2, 0.8])
        labels = torch.tensor([0, 1, 0, 1])
        assert roc_auc(scores, labels) == pytest.approx(1.0)


class TestEuclideanDispersion:
    """欧氏离散度测试。"""

    def test_identical_points_zero(self):
        pts = torch.ones(8, 16) * 2.0
        assert euclidean_dispersion(pts).item() == pytest.approx(0.0, abs=1e-6)

    def test_single_point_zero(self):
        assert euclidean_dispersion(torch.randn(1, 16)).item() == 0.0

    def test_spread_larger_than_tight(self):
        torch.manual_seed(0)
        tight = torch.randn(20, 16) * 0.01
        spread = torch.randn(20, 16) * 5.0
        assert euclidean_dispersion(spread) > euclidean_dispersion(tight)

    def test_non_negative(self):
        torch.manual_seed(1)
        assert euclidean_dispersion(torch.randn(10, 32)) >= 0.0


class TestSelectiveClassificationReport:
    """Selective routing report tests."""

    def test_reports_coverage_accuracy_and_confidence_intervals(self):
        scores = [0.1, 0.2, 0.9, 1.1]
        labels = [0, 0, 1, 1]

        report = selective_classification_report(scores, labels, threshold=0.5)

        assert report["n_flagged"] == 2
        assert report["n_accepted"] == 2
        assert report["coverage"] == pytest.approx(0.5)
        assert report["selective_accuracy"] == pytest.approx(1.0)
        assert report["detection"] == pytest.approx(1.0)
        assert report["false_alarm"] == pytest.approx(0.0)
        assert report["coverage_ci"]["total"] == 4
        assert report["selective_accuracy_ci"]["successes"] == 2

    def test_lower_direction_flags_low_scores(self):
        report = selective_classification_report([0.1, 0.9], [1, 0], threshold=0.5, direction="lower")

        assert report["n_flagged"] == 1
        assert report["detection"] == pytest.approx(1.0)

    def test_empty_denominator_ci_serializes_as_none(self):
        ci = binomial_confidence_interval(0, 0)

        assert ci == {"estimate": None, "lower": None, "upper": None, "successes": 0, "total": 0}


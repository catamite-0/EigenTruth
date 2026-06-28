"""Tests for dependency-free intrinsic-dimension estimators."""

import pytest
import torch

from eigentruth.eval import (
    IntrinsicDimensionReport,
    intrinsic_dimension_peak_layer,
    intrinsic_dimension_profile,
    twonn_intrinsic_dimension,
)


def test_twonn_intrinsic_dimension_orders_synthetic_manifolds():
    torch.manual_seed(123)
    n = 220
    line = torch.zeros(n, 6)
    line[:, 0] = torch.randn(n)
    plane = torch.zeros(n, 6)
    plane[:, :2] = torch.randn(n, 2)
    full = torch.randn(n, 6)

    line_report = twonn_intrinsic_dimension(line)
    plane_report = twonn_intrinsic_dimension(plane)
    full_report = twonn_intrinsic_dimension(full)

    assert isinstance(line_report, IntrinsicDimensionReport)
    assert 0.5 <= line_report.intrinsic_dimension <= 1.5
    assert 1.3 <= plane_report.intrinsic_dimension <= 2.8
    assert line_report.intrinsic_dimension < plane_report.intrinsic_dimension < full_report.intrinsic_dimension
    assert line_report.to_dict()["estimator"] == "twonn"


def test_intrinsic_dimension_profile_sorts_layers_and_selects_peak():
    torch.manual_seed(456)
    n = 180
    low = torch.zeros(n, 5)
    low[:, 0] = torch.randn(n)
    mid = torch.zeros(n, 5)
    mid[:, :2] = torch.randn(n, 2)
    high = torch.randn(n, 5)

    profile = intrinsic_dimension_profile({-1: mid, -3: low, -2: high})

    assert [entry["layer"] for entry in profile] == [-3, -2, -1]
    assert intrinsic_dimension_peak_layer(profile) == -2


def test_twonn_intrinsic_dimension_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="at least three"):
        twonn_intrinsic_dimension(torch.randn(2, 4))
    with pytest.raises(ValueError, match="finite"):
        twonn_intrinsic_dimension(torch.tensor([[0.0, 1.0], [1.0, float("nan")], [2.0, 3.0]]))
    with pytest.raises(ValueError, match="trim_fraction"):
        twonn_intrinsic_dimension(torch.randn(8, 3), trim_fraction=0.5)
    with pytest.raises(ValueError, match="non-duplicate"):
        twonn_intrinsic_dimension(torch.zeros(5, 3))


def test_intrinsic_dimension_peak_layer_rejects_empty_or_nonfinite_profile():
    with pytest.raises(ValueError, match="empty"):
        intrinsic_dimension_peak_layer([])
    with pytest.raises(ValueError, match="finite"):
        intrinsic_dimension_peak_layer([{"layer": -1, "intrinsic_dimension": float("nan")}])

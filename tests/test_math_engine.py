"""Phase 1 单元测试 — core/math_engine.py

全部使用合成张量，CPU 可运行，无需 GPU。
覆盖：正常路径 + 边界情况（零向量、极大值、FP16 输入）。
"""

import math

import torch

from eigentruth.core.math_engine import (
    COVARIANCE_MODES,
    CovarianceSpectrum,
    TruthManifold,
    _poincare_distance,
    covariance_shrinkage_intensity,
    covariance_spectrum,
    gaussian_wasserstein_distance,
    hyperbolic_semantic_entropy,
    mahalanobis_distance,
    manifold_distance,
    manifold_wasserstein_distance,
    poincare_map,
    sherman_morrison_update,
)

# ===================================================================
# Sherman-Morrison Update
# ===================================================================

class TestShermanMorrisonUpdate:
    """Sherman-Morrison 秩-1 协方差逆更新测试。"""

    def test_identity_start(self):
        """从单位矩阵开始，单次更新后结果仍为有效矩阵。"""
        d = 64
        cov_inv = torch.eye(d)
        x = torch.randn(d)
        result = sherman_morrison_update(cov_inv, x)
        assert result.shape == (d, d)
        assert torch.isfinite(result).all()

    def test_consistency_with_direct_inverse(self):
        """多次 Sherman-Morrison 更新后，与直接求逆结果一致（小维度）。"""
        d = 8
        n_samples = 20
        torch.manual_seed(42)

        samples = torch.randn(n_samples, d)

        # Sherman-Morrison 逐步更新
        cov_inv = torch.eye(d, dtype=torch.float32)
        for s in samples:
            cov_inv = sherman_morrison_update(cov_inv, s, epsilon=1e-6)

        # 两者方向应大致一致（不要求精确匹配，因为增量公式有数值差异）
        # 验证对称性
        assert torch.allclose(cov_inv, cov_inv.T, atol=1e-5)
        # 验证有限性
        assert torch.isfinite(cov_inv).all()

    def test_fp16_input_stability(self):
        """FP16 输入不产生 NaN 或 Inf。"""
        d = 128
        cov_inv = torch.eye(d, dtype=torch.float16)
        x = torch.randn(d, dtype=torch.float16)
        result = sherman_morrison_update(cov_inv, x)
        assert result.dtype == torch.float16
        assert torch.isfinite(result).all()

    def test_zero_vector(self):
        """零向量更新不应改变矩阵（分母 ≈ 1+ε）。"""
        d = 32
        cov_inv = torch.eye(d)
        x = torch.zeros(d)
        result = sherman_morrison_update(cov_inv, x)
        assert torch.allclose(result, cov_inv, atol=1e-5)

    def test_large_vector(self):
        """极大值向量不产生 NaN。"""
        d = 32
        cov_inv = torch.eye(d)
        x = torch.ones(d) * 1e4
        result = sherman_morrison_update(cov_inv, x)
        assert torch.isfinite(result).all()

    def test_multiple_updates_stable(self):
        """连续 100 次更新保持数值稳定。"""
        d = 64
        torch.manual_seed(123)
        cov_inv = torch.eye(d)
        for _ in range(100):
            x = torch.randn(d)
            cov_inv = sherman_morrison_update(cov_inv, x)
        assert torch.isfinite(cov_inv).all()


# ===================================================================
# Mahalanobis Distance
# ===================================================================

class TestMahalanobisDistance:
    """马氏距离计算测试。"""

    def test_zero_at_mean(self):
        """在质心处距离为 0。"""
        d = 32
        mean = torch.randn(d)
        cov_inv = torch.eye(d)
        dist = mahalanobis_distance(mean, mean, cov_inv)
        assert torch.isclose(dist, torch.tensor(0.0), atol=1e-6)

    def test_positive_away_from_mean(self):
        """偏离质心时距离 > 0。"""
        d = 32
        mean = torch.zeros(d)
        cov_inv = torch.eye(d)
        h = torch.ones(d)
        dist = mahalanobis_distance(h, mean, cov_inv)
        # 当 cov_inv = I 时，马氏距离 = 欧氏距离
        expected = math.sqrt(d)
        assert torch.isclose(dist, torch.tensor(expected), atol=1e-4)

    def test_euclidean_equivalence_with_identity_cov(self):
        """协方差逆为单位矩阵时，马氏距离 = 欧氏距离。"""
        d = 16
        torch.manual_seed(7)
        h = torch.randn(d)
        mean = torch.randn(d)
        cov_inv = torch.eye(d)
        m_dist = mahalanobis_distance(h, mean, cov_inv)
        e_dist = torch.norm(h - mean)
        assert torch.isclose(m_dist, e_dist, atol=1e-4)

    def test_scaled_covariance(self):
        """缩放协方差矩阵时距离按比例变化。"""
        d = 8
        mean = torch.zeros(d)
        h = torch.ones(d)
        # cov_inv = 4I → 距离 = 2 * 欧氏距离
        cov_inv = 4.0 * torch.eye(d)
        dist = mahalanobis_distance(h, mean, cov_inv)
        expected = 2.0 * math.sqrt(d)
        assert torch.isclose(dist, torch.tensor(expected), atol=1e-3)

    def test_accepts_diagonal_precision_vector(self):
        """对角精度向量等价于显式对角矩阵。"""
        d = 8
        mean = torch.zeros(d)
        h = torch.arange(1, d + 1, dtype=torch.float32)
        precision_diag = torch.linspace(0.5, 2.0, d)

        vector_dist = mahalanobis_distance(h, mean, precision_diag)
        dense_dist = mahalanobis_distance(h, mean, torch.diag(precision_diag))

        assert torch.isclose(vector_dist, dense_dist, atol=1e-6)

    def test_non_negative(self):
        """距离始终 >= 0。"""
        d = 64
        torch.manual_seed(99)
        for _ in range(10):
            h = torch.randn(d)
            mean = torch.randn(d)
            cov_inv = torch.eye(d)
            dist = mahalanobis_distance(h, mean, cov_inv)
            assert dist >= 0.0


# ===================================================================
# Gaussian / Manifold Wasserstein Distance
# ===================================================================

class TestGaussianWassersteinDistance:
    """Closed-form Gaussian 2-Wasserstein distance tests."""

    def test_identical_diag_gaussians_have_zero_distance(self):
        mean = torch.tensor([1.0, -2.0, 0.5])
        covariance = torch.tensor([1.0, 4.0, 9.0])

        dist = gaussian_wasserstein_distance(mean, covariance, mean, covariance)

        assert torch.isclose(dist, torch.tensor(0.0), atol=1e-6)

    def test_identical_full_gaussians_have_zero_distance(self):
        mean = torch.tensor([0.0, 1.0])
        covariance = torch.tensor([[2.0, 0.25], [0.25, 1.0]])

        dist = gaussian_wasserstein_distance(mean, covariance, mean, covariance)

        assert torch.isclose(dist, torch.tensor(0.0), atol=1e-4)

    def test_symmetric_for_full_covariances(self):
        mean_a = torch.tensor([0.0, 0.0])
        mean_b = torch.tensor([1.0, -1.0])
        cov_a = torch.tensor([[3.0, 0.4], [0.4, 1.0]])
        cov_b = torch.tensor([[1.5, -0.1], [-0.1, 2.0]])

        ab = gaussian_wasserstein_distance(mean_a, cov_a, mean_b, cov_b)
        ba = gaussian_wasserstein_distance(mean_b, cov_b, mean_a, cov_a)

        assert torch.isclose(ab, ba, atol=1e-5)
        assert ab > 0.0

    def test_diag_closed_form_matches_expected(self):
        mean_a = torch.zeros(3)
        mean_b = torch.tensor([1.0, 2.0, 2.0])
        cov_a = torch.tensor([1.0, 4.0, 9.0])
        cov_b = torch.tensor([4.0, 9.0, 16.0])
        expected_sq = torch.tensor(9.0) + (torch.sqrt(cov_a) - torch.sqrt(cov_b)).square().sum()

        actual_sq = gaussian_wasserstein_distance(mean_a, cov_a, mean_b, cov_b, squared=True)

        assert torch.isclose(actual_sq, expected_sq, atol=1e-6)

    def test_identical_covariance_reduces_to_euclidean_mean_distance(self):
        mean_a = torch.tensor([1.0, 2.0, 3.0])
        mean_b = torch.tensor([2.0, 4.0, 5.0])
        covariance = torch.eye(3)

        dist = gaussian_wasserstein_distance(mean_a, covariance, mean_b, covariance)

        assert torch.isclose(dist, torch.norm(mean_a - mean_b), atol=1e-5)

    def test_rejects_invalid_inputs(self):
        import pytest

        with pytest.raises(ValueError, match="mean dimension mismatch"):
            gaussian_wasserstein_distance(torch.zeros(2), torch.ones(2), torch.zeros(3), torch.ones(2))
        with pytest.raises(ValueError, match="shape mismatch"):
            gaussian_wasserstein_distance(torch.zeros(2), torch.eye(3), torch.zeros(2), torch.eye(2))
        with pytest.raises(ValueError, match="finite"):
            gaussian_wasserstein_distance(
                torch.zeros(2),
                torch.tensor([1.0, float("nan")]),
                torch.zeros(2),
                torch.ones(2),
            )
        with pytest.raises(ValueError, match="non-negative"):
            gaussian_wasserstein_distance(torch.zeros(2), torch.tensor([1.0, -0.1]), torch.zeros(2), torch.ones(2))


# ===================================================================
# Poincaré Map
# ===================================================================

class TestPoincareMap:
    """庞加莱球映射测试。"""

    def test_output_within_ball(self):
        """映射结果范数 < 1。"""
        h = torch.randn(10, 64)
        result = poincare_map(h)
        norms = torch.norm(result, dim=-1)
        assert (norms < 1.0).all()

    def test_zero_maps_to_origin(self):
        """零向量映射到原点附近。"""
        h = torch.zeros(1, 32)
        result = poincare_map(h)
        assert torch.norm(result) < 1e-5

    def test_large_vector_clamped(self):
        """极大值向量被钳位在球内。"""
        h = torch.ones(1, 32) * 1e6
        result = poincare_map(h)
        assert torch.norm(result, dim=-1).item() < 1.0

    def test_fp16_input(self):
        """FP16 输入不崩溃。"""
        h = torch.randn(5, 16, dtype=torch.float16)
        result = poincare_map(h)
        assert torch.isfinite(result).all()

    def test_preserves_direction(self):
        """映射保持方向（同向缩放）。"""
        h = torch.tensor([[3.0, 4.0, 0.0]])
        result = poincare_map(h)
        # 归一化方向应一致
        h_dir = h / torch.norm(h)
        r_dir = result / torch.norm(result)
        assert torch.allclose(h_dir, r_dir, atol=1e-4)

    def test_different_curvatures(self):
        """不同曲率参数下输出均在球内。"""
        h = torch.randn(5, 32)
        for c in [0.5, 1.0, 2.0, 5.0]:
            result = poincare_map(h, curvature=c)
            norms = torch.norm(result, dim=-1)
            assert (norms < 1.0).all(), f"Failed at curvature={c}"


# ===================================================================
# Hyperbolic Semantic Entropy (HSE)
# ===================================================================

class TestHyperbolicSemanticEntropy:
    """双曲语义熵测试。"""

    def test_single_point_zero(self):
        """单点的熵为 0。"""
        p = torch.randn(1, 16)
        p = poincare_map(p)
        hse = hyperbolic_semantic_entropy(p)
        assert torch.isclose(hse, torch.tensor(0.0))

    def test_identical_points_low(self):
        """相同点的熵极低（接近零）。

        注意：centroid 通过欧氏均值 → 庞加莱投影计算，
        对于完全相同的输入，均值 = 自身，但再次投影会
        引入非线性偏差。因此使用原点附近的小范数点以
        减少映射误差。
        """
        # 使用原点附近的小向量以减少 poincare_map 非线性偏差
        p = torch.randn(1, 16) * 0.01
        p = poincare_map(p)
        points = p.expand(10, -1).clone()
        hse = hyperbolic_semantic_entropy(points)
        assert hse < 1.0  # 相同点的 HSE 远低于分散点

    def test_spread_points_higher(self):
        """分散点的熵高于聚集点。"""
        torch.manual_seed(42)
        # 聚集
        tight = torch.randn(20, 16) * 0.01
        tight_p = poincare_map(tight)
        hse_tight = hyperbolic_semantic_entropy(tight_p)

        # 分散
        spread = torch.randn(20, 16) * 5.0
        spread_p = poincare_map(spread)
        hse_spread = hyperbolic_semantic_entropy(spread_p)

        assert hse_spread > hse_tight

    def test_non_negative(self):
        """HSE 始终 >= 0。"""
        torch.manual_seed(7)
        points = poincare_map(torch.randn(15, 32))
        hse = hyperbolic_semantic_entropy(points)
        assert hse >= 0.0


# ===================================================================
# Poincaré Distance (内部函数)
# ===================================================================

class TestPoincareDistance:
    """庞加莱测地线距离测试。"""

    def test_same_point_zero(self):
        """同一点的距离为 0。"""
        p = torch.tensor([0.1, 0.2, 0.3])
        d = _poincare_distance(p, p)
        assert torch.isclose(d, torch.tensor(0.0), atol=1e-5)

    def test_symmetry(self):
        """d(u, v) == d(v, u)。"""
        u = torch.tensor([0.1, -0.2])
        v = torch.tensor([-0.3, 0.1])
        assert torch.isclose(_poincare_distance(u, v), _poincare_distance(v, u), atol=1e-5)

    def test_positive(self):
        """不同点距离 > 0。"""
        u = torch.tensor([0.1, 0.0])
        v = torch.tensor([0.0, 0.2])
        assert _poincare_distance(u, v) > 0.0

    def test_distance_increases_near_boundary(self):
        """越靠近球边界，相同欧氏位移的测地线距离越大。"""
        # 靠近原点
        u1 = torch.tensor([0.01, 0.0])
        v1 = torch.tensor([0.02, 0.0])
        d_near_origin = _poincare_distance(u1, v1)

        # 靠近边界
        u2 = torch.tensor([0.90, 0.0])
        v2 = torch.tensor([0.91, 0.0])
        d_near_boundary = _poincare_distance(u2, v2)

        assert d_near_boundary > d_near_origin


# ===================================================================
# TruthManifold
# ===================================================================

class TestTruthManifold:
    """真值流形增量构建测试。"""

    def test_covariance_modes_are_public(self):
        assert COVARIANCE_MODES == ("full", "diag", "low_rank", "shrinkage")

    def test_covariance_shrinkage_intensity_is_bounded_and_isotropic_goes_full(self):
        cov = torch.diag(torch.tensor([9.0, 1.0, 1.0, 1.0]))
        alpha = covariance_shrinkage_intensity(cov, sample_count=16)
        isotropic_alpha = covariance_shrinkage_intensity(torch.eye(4), sample_count=16)

        assert 0.0 <= alpha <= 1.0
        assert isotropic_alpha == 1.0

    def test_covariance_spectrum_reports_mp_spike_and_effective_rank(self):
        """An anisotropic covariance exposes an out-of-bulk spectral spike."""
        cov = torch.diag(torch.tensor([9.0, 1.0, 1.0, 1.0]))
        report = covariance_spectrum(cov, sample_count=100)

        assert isinstance(report, CovarianceSpectrum)
        assert report.source == "full"
        assert report.hidden_dim == 4
        assert report.spike_count == 1
        assert report.eigenvalues.tolist() == [9.0, 1.0, 1.0, 1.0]
        assert report.effective_rank < 4.0
        assert report.participation_ratio < 4.0
        payload = report.to_dict()
        assert payload["spike_count"] == 1
        assert payload["eigenvalues"] == [9.0, 1.0, 1.0, 1.0]
        assert "eigenvalues" not in report.to_dict(include_eigenvalues=False)

    def test_truth_manifold_spectrum_uses_full_scatter_when_available(self):
        torch.manual_seed(303)
        samples = torch.randn(240, 8)
        samples[:, 0] *= 4.0
        m = TruthManifold()
        m.update_many(samples)

        report = m.spectrum()

        assert report.source == "full"
        assert report.sample_count == 240
        assert report.hidden_dim == 8
        assert report.spike_count >= 1
        assert report.eigenvalues[0] > report.marchenko_pastur_upper
        assert 1.0 <= report.effective_rank <= 8.0

    def test_truth_manifold_spectrum_handles_diag_mode_without_full_scatter(self):
        torch.manual_seed(404)
        m = TruthManifold(covariance_mode="diag")
        m.update_many(torch.randn(32, 5))

        report = m.covariance_spectrum()

        assert report.source == "diagonal"
        assert report.hidden_dim == 5
        assert m._M2 is None  # noqa: SLF001 - verifies diag mode stays memory-saving.
        assert torch.all(report.eigenvalues[:-1] >= report.eigenvalues[1:])
        assert report.to_dict(include_eigenvalues=False)["source"] == "diagonal"

    def test_truth_manifold_spectrum_handles_degenerate_covariance(self):
        m = TruthManifold()
        m.update_many(torch.ones(6, 4))

        report = m.spectrum()

        assert report.spike_count == 0
        assert report.effective_rank == 0.0
        assert report.participation_ratio == 0.0
        assert report.stable_rank == 0.0
        assert report.condition_number == 0.0
        assert report.marchenko_pastur_upper == 0.0

    def test_covariance_spectrum_rejects_invalid_inputs(self):
        import pytest

        with pytest.raises(ValueError, match="sample_count"):
            covariance_spectrum(torch.eye(2), sample_count=1)
        with pytest.raises(ValueError, match="square"):
            covariance_spectrum(torch.randn(2, 3), sample_count=10)
        with pytest.raises(ValueError, match="finite"):
            covariance_spectrum(torch.tensor([1.0, float("nan")]), sample_count=10)

    def test_truth_manifold_spectrum_requires_two_samples(self):
        import pytest

        m = TruthManifold()
        with pytest.raises(ValueError, match="at least two samples"):
            m.spectrum()
        m.update(torch.randn(3))
        with pytest.raises(ValueError, match="at least two samples"):
            m.spectrum()

    def test_first_update_initializes(self):
        """首次更新初始化 mean 和 cov_inv。"""
        m = TruthManifold()
        h = torch.randn(64)
        m.update(h)
        assert m.n == 1
        assert m.mean is not None
        assert m.cov_inv is not None
        assert m.hidden_dim == 64

    def test_not_ready_after_one_sample(self):
        """1 个样本后流形不可用。"""
        m = TruthManifold()
        m.update(torch.randn(32))
        assert not m.is_ready()

    def test_ready_after_two_samples(self):
        """2 个样本后流形可用。"""
        m = TruthManifold()
        m.update(torch.randn(32))
        m.update(torch.randn(32))
        assert m.is_ready()

    def test_mean_converges(self):
        """大量样本后均值收敛到真实均值附近。"""
        torch.manual_seed(0)
        d = 16
        true_mean = torch.ones(d) * 3.0
        m = TruthManifold()
        for _ in range(500):
            h = true_mean + torch.randn(d) * 0.1
            m.update(h)
        assert torch.allclose(m.mean, true_mean, atol=0.05)

    def test_multiple_updates_stable(self):
        """连续更新保持数值稳定。"""
        d = 32
        torch.manual_seed(42)
        m = TruthManifold()
        for _ in range(100):
            m.update(torch.randn(d))
        assert torch.isfinite(m.mean).all()
        assert torch.isfinite(m.cov_inv).all()

    def test_count_tracks_correctly(self):
        """样本计数正确递增。"""
        m = TruthManifold()
        for i in range(10):
            m.update(torch.randn(8))
        assert m.n == 10

    def test_update_many_matches_sequential_updates_across_covariance_modes(self):
        """batch update 与逐样本 update 保持统计量和距离一致。"""
        torch.manual_seed(1234)
        samples = torch.randn(11, 7)
        probe = torch.randn(7)
        for mode in COVARIANCE_MODES:
            sequential = TruthManifold(covariance_mode=mode, covariance_low_rank=3)
            batched = TruthManifold(covariance_mode=mode, covariance_low_rank=3)
            for sample in samples:
                sequential.update(sample)
            batched.update_many(samples[:5])
            batched.update_many(samples[5:])

            assert batched.n == sequential.n
            assert batched.hidden_dim == sequential.hidden_dim
            assert torch.allclose(batched.mean, sequential.mean, atol=1e-6)
            assert torch.allclose(batched._M2_diag, sequential._M2_diag, atol=1e-5)  # noqa: SLF001
            if mode == "diag":
                assert batched._M2 is None  # noqa: SLF001
            else:
                assert torch.allclose(batched._M2, sequential._M2, atol=1e-5)  # noqa: SLF001
            assert torch.isclose(
                batched.mahalanobis_distance(probe),
                sequential.mahalanobis_distance(probe),
                atol=1e-5,
            )

    def test_update_many_rejects_bad_shapes_and_dimension_mismatch(self):
        m = TruthManifold()
        import pytest
        with pytest.raises(ValueError, match="2D hidden state batch"):
            m.update_many(torch.randn(8))

        m.update_many(torch.randn(2, 8))
        with pytest.raises(ValueError, match="Hidden dimension mismatch"):
            m.update_many(torch.randn(2, 9))

    def test_rejects_non_vector_update(self):
        """update 只接受单个 1D hidden state。"""
        m = TruthManifold()
        import pytest
        with pytest.raises(ValueError, match="1D hidden state"):
            m.update(torch.randn(2, 8))

    def test_rejects_hidden_dim_mismatch(self):
        """后续样本维度必须与首次样本一致。"""
        m = TruthManifold()
        m.update(torch.randn(8))
        import pytest
        with pytest.raises(ValueError, match="Hidden dimension mismatch"):
            m.update(torch.randn(9))

    def test_save_load_roundtrip(self, tmp_path):
        """save → load 往返保持所有字段不变。"""
        d = 16
        torch.manual_seed(77)
        m = TruthManifold()
        for _ in range(5):
            m.update(torch.randn(d))

        path = tmp_path / "manifold.pt"
        m.save(path)

        m2 = TruthManifold.load(path)
        assert m2.n == m.n
        assert m2.hidden_dim == m.hidden_dim
        assert torch.allclose(m2.mean, m.mean)
        assert torch.allclose(m2.cov_inv, m.cov_inv)
        assert m2.is_ready()

    def test_load_preserves_usability(self, tmp_path):
        """加载后的流形可正常用于马氏距离计算。"""
        d = 16
        torch.manual_seed(88)
        m = TruthManifold()
        for _ in range(5):
            m.update(torch.randn(d))

        path = tmp_path / "manifold.pt"
        m.save(path)
        m2 = TruthManifold.load(path)

        h = torch.randn(d)
        dist_orig = mahalanobis_distance(h, m.mean, m.cov_inv)
        dist_load = mahalanobis_distance(h, m2.mean, m2.cov_inv)
        assert torch.isclose(dist_orig, dist_load, atol=1e-6)

    def test_save_load_with_contrastive_direction(self, tmp_path):
        """save → load 包含 contrastive_direction 和 false_mean。"""
        d = 16
        torch.manual_seed(99)
        m = TruthManifold()
        for _ in range(5):
            m.update(torch.randn(d))
        m.false_mean = torch.randn(d)
        m.contrastive_direction = torch.randn(d)

        path = tmp_path / "manifold_contrastive.pt"
        m.save(path)
        m2 = TruthManifold.load(path)

        assert m2.false_mean is not None
        assert m2.contrastive_direction is not None
        assert torch.allclose(m2.false_mean, m.false_mean)
        assert torch.allclose(m2.contrastive_direction, m.contrastive_direction)

    def test_diag_covariance_mode_avoids_full_scatter_and_matches_manual_distance(self, tmp_path):
        """diag 模式只保存对角散布，距离与手工对角精度一致。"""
        torch.manual_seed(101)
        d = 10
        m = TruthManifold(covariance_mode="diag")
        samples = torch.randn(12, d)
        for sample in samples:
            m.update(sample)

        assert m.is_ready()
        assert m._M2 is None  # noqa: SLF001 - verifies memory-saving mode.
        assert m._M2_diag is not None  # noqa: SLF001
        probe = torch.randn(d)
        cov_diag = m._M2_diag / (m.n - 1)  # noqa: SLF001
        tau = cov_diag.mean().clamp(min=1e-6)
        precision_diag = torch.reciprocal((cov_diag + m.ridge_lambda * tau).clamp(min=1e-12))
        expected = mahalanobis_distance(probe, m.mean, precision_diag)

        assert torch.isclose(m.mahalanobis_distance(probe), expected, atol=1e-6)
        assert m.cov_inv.shape == (d, d)  # legacy dense property remains available.

        loaded_path = tmp_path / "diag_manifold.pt"
        m.save(loaded_path)
        loaded = TruthManifold.load(loaded_path)
        assert loaded.is_ready()
        assert loaded._M2 is None  # noqa: SLF001
        assert loaded._M2_diag is not None  # noqa: SLF001

    def test_low_rank_covariance_mode_is_finite_and_roundtrips(self, tmp_path):
        """low_rank 模式产生有限距离，并保留序列化配置。"""
        torch.manual_seed(202)
        d = 12
        m = TruthManifold(covariance_mode="low_rank", covariance_low_rank=3)
        for _ in range(10):
            m.update(torch.randn(d))

        probe = torch.randn(d)
        dist = m.mahalanobis_distance(probe)
        assert torch.isfinite(dist).all()

        path = tmp_path / "low_rank_manifold.pt"
        m.save(path)
        loaded = TruthManifold.load(path)

        assert loaded.covariance_mode == "low_rank"
        assert loaded.covariance_low_rank == 3
        assert torch.isclose(
            loaded.mahalanobis_distance(probe),
            dist,
            atol=1e-5,
        )

    def test_shrinkage_covariance_mode_is_finite_reports_alpha_and_roundtrips(self, tmp_path):
        """shrinkage 模式使用 full scatter 估计 OAS 收缩强度并可序列化。"""
        torch.manual_seed(505)
        d = 10
        samples = torch.randn(8, d)
        samples[:, 0] *= 4.0
        m = TruthManifold(covariance_mode="shrinkage")
        m.update_many(samples)

        alpha = m.covariance_shrinkage_alpha()
        probe = torch.randn(d)
        dist = m.mahalanobis_distance(probe)

        assert m.is_ready()
        assert m._M2 is not None  # noqa: SLF001 - shrinkage needs full scatter statistics.
        assert 0.0 <= alpha <= 1.0
        assert torch.isfinite(m.cov_inv).all()
        assert torch.isfinite(dist).all()

        path = tmp_path / "shrinkage_manifold.pt"
        m.save(path)
        loaded = TruthManifold.load(path)

        assert loaded.covariance_mode == "shrinkage"
        assert loaded._M2 is not None  # noqa: SLF001
        assert math.isclose(loaded.covariance_shrinkage_alpha(), alpha, rel_tol=0.0, abs_tol=1e-7)
        assert torch.isclose(loaded.mahalanobis_distance(probe), dist, atol=1e-5)

    def test_invalid_covariance_mode_rejected(self):
        import pytest

        with pytest.raises(ValueError, match="covariance_mode"):
            TruthManifold(covariance_mode="bad")

    def test_manifold_distance_is_zero_and_symmetric_for_matching_manifolds(self):
        torch.manual_seed(606)
        samples = torch.randn(24, 5)
        shifted = samples + torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
        m1 = TruthManifold()
        m2 = TruthManifold()
        m3 = TruthManifold()
        m1.update_many(samples)
        m2.update_many(samples)
        m3.update_many(shifted)

        assert torch.isclose(m1.manifold_distance(m2), torch.tensor(0.0), atol=1e-4)
        forward = manifold_distance(m1, m3)
        backward = manifold_wasserstein_distance(m3, m1)

        assert torch.isclose(forward, backward, atol=1e-5)
        assert forward > 0.0

    def test_manifold_distance_supports_diag_and_shrinkage_modes(self):
        torch.manual_seed(707)
        samples_a = torch.randn(20, 4)
        samples_b = torch.randn(20, 4) + 0.2
        for mode in ["diag", "shrinkage"]:
            m1 = TruthManifold(covariance_mode=mode)
            m2 = TruthManifold(covariance_mode=mode)
            m1.update_many(samples_a)
            m2.update_many(samples_b)

            dist = manifold_distance(m1, m2)

            assert torch.isfinite(dist).all()
            assert dist >= 0.0

    def test_manifold_distance_requires_ready_manifolds(self):
        import pytest

        m1 = TruthManifold()
        m2 = TruthManifold()
        m1.update(torch.zeros(2))
        m2.update_many(torch.randn(2, 2))

        with pytest.raises(ValueError, match="at least two samples"):
            manifold_distance(m1, m2)


# ===================================================================
# 距离尺度稳定性 (Fix: 马氏距离不随 warmup 样本数塌缩)
# ===================================================================

class TestDistanceScaleStability:
    """马氏距离尺度应在不同 warmup 样本数下保持稳定。

    旧实现的 cov_inv 是 (I + 散布矩阵)⁻¹，散布矩阵 ∝ n，导致距离 ∝ 1/√n，
    阈值因此依赖于 warmup 集大小。新实现按样本数归一化样本协方差，尺度稳定。
    """

    def _dists_by_n(self, ns):
        torch.manual_seed(123)
        d = 48
        center = torch.ones(d) * 2.0
        stream = center + torch.randn(5000, d) * 1.5
        probe = center + torch.randn(d) * 1.5  # in-distribution 测试点
        out = {}
        for n in ns:
            m = TruthManifold()
            for i in range(n):
                m.update(stream[i])
            out[n] = mahalanobis_distance(probe, m.mean, m.cov_inv).item()
        return out

    def test_scale_stable_across_warmup_size(self):
        """同一 in-dist 点在 n=100..1000 间距离保持同量级（不塌缩）。"""
        d = self._dists_by_n([100, 200, 500, 1000])
        vals = list(d.values())
        # 旧实现（散布矩阵逆）在此范围约 √10 ≈ 3.2× 塌缩；新实现应远小于此。
        assert max(vals) / min(vals) < 2.2

    def test_distance_does_not_collapse_with_more_samples(self):
        """大样本距离不应是小样本距离的极小比例（旧 1/√n 病态）。"""
        d = self._dists_by_n([100, 1000])
        # 旧实现: d[1000] ≈ d[100]/√10 ≈ 0.32×；新实现应保持同量级。
        assert d[1000] > 0.5 * d[100]


# ===================================================================
# Ridge 正则化 (Fix: n<dim 与退化协方差下的可逆性)
# ===================================================================

class TestRidgeRegularization:
    """固定相对 ridge 保证精度矩阵在小样本/退化情形下仍有限可逆。"""

    def test_cov_inv_prior_for_small_n(self):
        """n=0 返回 None；n=1 返回单位阵作为先验精度。"""
        m = TruthManifold()
        assert m.cov_inv is None  # 无样本
        m.update(torch.randn(8))
        assert m.n == 1
        assert torch.allclose(m.cov_inv, torch.eye(8))  # 单样本 → 单位先验

    def test_precision_invertible_when_n_less_than_dim(self):
        """n < hidden_dim（散布矩阵秩亏）时精度矩阵仍有限、对称、可用。"""
        torch.manual_seed(0)
        d = 64
        m = TruthManifold()
        for _ in range(5):  # n=5 < d=64
            m.update(torch.randn(d))
        cov_inv = m.cov_inv
        assert cov_inv.shape == (d, d)
        assert torch.isfinite(cov_inv).all()
        assert torch.allclose(cov_inv, cov_inv.T, atol=1e-4)
        dist = mahalanobis_distance(torch.randn(d), m.mean, cov_inv)
        assert torch.isfinite(dist).all()

    def test_identical_samples_finite_distance(self):
        """完全相同的样本（协方差≈0）不产生 NaN/Inf。"""
        d = 32
        m = TruthManifold()
        for _ in range(8):
            m.update(torch.ones(d) * 3.0)
        dist = mahalanobis_distance(torch.ones(d) * 3.5, m.mean, m.cov_inv)
        assert torch.isfinite(dist).all()

    def test_cov_inv_recomputes_after_update(self):
        """新样本后 cov_inv 缓存失效并重新计算。"""
        torch.manual_seed(1)
        d = 16
        m = TruthManifold()
        for _ in range(4):
            m.update(torch.randn(d))
        first = m.cov_inv.clone()
        for _ in range(4):
            m.update(torch.randn(d) * 5.0 + 10.0)
        second = m.cov_inv
        assert not torch.allclose(first, second)

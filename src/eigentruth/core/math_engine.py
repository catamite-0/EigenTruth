"""EigenTruth Core — 防崩溃数学引擎 / Crash-proof Math Engine.

基于几何动力学的底层数学原语，包括 / Core mathematical primitives based on geometric dynamics, including:
- Welford 在线均值/协方差 + 固定 ridge 正则化精度矩阵 / Welford online mean/covariance with fixed-ridge precision
- 马氏距离计算 / Mahalanobis distance computation
- 庞加莱球模型映射 / Poincaré ball model mapping
- 双曲语义熵 (HSE) / Hyperbolic Semantic Entropy (HSE)
- Sherman-Morrison 秩-1 更新（独立工具函数）/ Sherman-Morrison rank-1 update (standalone utility)

所有浮点密集运算在内部强制使用 FP32 以确保数值稳定性。
All float-intensive computations are forced to FP32 internally to ensure numerical stability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

COVARIANCE_MODES: tuple[str, ...] = ("full", "diag", "low_rank")


def _validate_covariance_mode(mode: str) -> str:
    if mode not in COVARIANCE_MODES:
        raise ValueError(
            f"covariance_mode must be one of {COVARIANCE_MODES}, got {mode!r}."
        )
    return mode

@dataclass
class TruthManifold:
    """真值流形：存储事实语料的统计特征。
    Truth Manifold: Stores statistical features of the factual corpus.

    流形用数值稳定的 Welford 累积维护在线均值与协方差散布矩阵 (`_M2`)。
    `cov_inv` 是一个派生量：对**按样本数归一化**的样本协方差施加固定相对
    ridge 正则后求逆得到的精度矩阵。由于按样本数归一化，马氏距离的尺度
    在不同 warmup 样本数下保持稳定（不会随样本增多而塌缩到 0）。
    The manifold maintains an online mean and covariance scatter matrix (`_M2`)
    with a numerically stable Welford accumulation. `cov_inv` is a derived
    quantity: the precision matrix obtained by ridge-regularizing the
    **sample-count-normalized** sample covariance and inverting it. Because the
    covariance is normalized by sample count, the Mahalanobis-distance scale is
    stable across warmup-set sizes (it does not collapse toward 0 as more
    warmup samples are added).

    Attributes:
        mean: 隐状态质心向量 / Hidden state centroid vector, shape [hidden_dim].
        n: 已累积的样本数量 / Accumulated sample count.
        hidden_dim: 隐状态维度 (由首次更新时自动推断) / Hidden state dimension (inferred on first update).
        ridge_lambda: 固定相对 ridge 系数（相对于平均方差），保证 n < hidden_dim
            时精度矩阵仍可逆且尺度稳定 / Fixed relative ridge coefficient (relative to the
            average variance); keeps the precision well-conditioned and scale-stable even when
            n < hidden_dim.
    """

    mean: Optional[Tensor] = None
    n: int = 0
    hidden_dim: int = 0
    ridge_lambda: float = 0.1
    covariance_mode: str = "full"
    covariance_low_rank: int = 16

    # 扩展：支持方案 B (对比流形)
    false_mean: Optional[Tensor] = None
    contrastive_direction: Optional[Tensor] = None

    # 协方差散布矩阵 (Welford M2 = Σ (xᵢ-μ)(xᵢ-μ)ᵀ)，以及派生精度矩阵缓存
    # Covariance scatter matrix (Welford M2) and the cached derived precision matrix
    _M2: Optional[Tensor] = field(default=None, repr=False)
    _M2_diag: Optional[Tensor] = field(default=None, repr=False)
    _cov_inv_cache: Optional[Tensor] = field(default=None, repr=False)
    _precision_diag_cache: Optional[Tensor] = field(default=None, repr=False)
    _low_rank_cache: Optional[tuple[Tensor, Tensor, Tensor]] = field(default=None, repr=False)
    _dirty: bool = field(default=True, repr=False)

    # 运行时设备跟踪（不参与序列化）
    _device: torch.device = field(default_factory=lambda: torch.device("cpu"), repr=False)

    def __post_init__(self) -> None:
        self.covariance_mode = _validate_covariance_mode(str(self.covariance_mode))
        if int(self.covariance_low_rank) < 1:
            raise ValueError("covariance_low_rank must be >= 1.")
        self.covariance_low_rank = int(self.covariance_low_rank)

    # ------------------------------------------------------------------
    # 派生量：归一化、正则化的精度矩阵
    # ------------------------------------------------------------------

    @property
    def cov_inv(self) -> Optional[Tensor]:
        """归一化、ridge 正则化的精度矩阵 (协方差逆)，惰性计算并缓存。
        Sample-count-normalized, ridge-regularized precision matrix (inverse
        covariance), computed lazily and cached.

        - n == 0 (无样本): 返回 None / no samples: returns None.
        - n == 1: 返回单位阵作为先验精度 / returns identity as a prior precision.
        - n >= 2: 返回 (Ĉ + λ·τ·I)⁻¹，其中 Ĉ = M2/(n-1)，τ 为平均方差。
            returns (Ĉ + λ·τ·I)⁻¹ where Ĉ = M2/(n-1) and τ is the average variance.
        """
        if self.mean is None:
            return None
        if self.n < 2:
            return torch.eye(self.hidden_dim, device=self._device, dtype=torch.float32)
        if self._dirty or self._cov_inv_cache is None:
            self._cov_inv_cache = self._compute_precision()
            self._dirty = False
        return self._cov_inv_cache

    def _compute_precision(self) -> Tensor:
        if self.covariance_mode == "diag":
            return torch.diag(self._compute_diag_precision())
        if self.covariance_mode == "low_rank":
            return self._compute_low_rank_precision()
        return self._compute_full_precision()

    def _compute_full_precision(self) -> Tensor:
        """由散布矩阵计算归一化、ridge 正则化的精度矩阵 (强制 FP32)。"""
        if self._M2 is None:
            raise RuntimeError("full covariance precision requires a full scatter matrix.")
        eye = torch.eye(self.hidden_dim, device=self._device, dtype=torch.float32)
        cov = (self._M2.to(torch.float32) / (self.n - 1))
        cov = 0.5 * (cov + cov.T)  # 对称化，消除浮点误差
        # τ = 平均对角方差，作为 ridge 的尺度，使 λ 成为无量纲的相对强度
        # τ = average diagonal variance; makes the ridge λ a dimensionless relative strength
        tau = cov.diagonal().mean().clamp(min=1e-6)
        reg = cov + (self.ridge_lambda * tau) * eye
        return torch.linalg.inv(reg)

    def _normalized_cov_diag(self) -> Tensor:
        if self.n < 2:
            return torch.ones(self.hidden_dim, device=self._device, dtype=torch.float32)
        if self._M2_diag is not None:
            diag = self._M2_diag.to(torch.float32) / (self.n - 1)
        elif self._M2 is not None:
            diag = self._M2.to(torch.float32).diagonal() / (self.n - 1)
        else:
            raise RuntimeError("diagonal covariance precision requires scatter statistics.")
        return torch.clamp(diag, min=0.0)

    def _compute_diag_precision(self) -> Tensor:
        if not self._dirty and self._precision_diag_cache is not None:
            return self._precision_diag_cache
        cov_diag = self._normalized_cov_diag()
        tau = cov_diag.mean().clamp(min=1e-6)
        reg_diag = cov_diag + (self.ridge_lambda * tau)
        self._precision_diag_cache = torch.reciprocal(reg_diag.clamp(min=1e-12))
        self._dirty = False
        return self._precision_diag_cache

    def _low_rank_components(self) -> tuple[Tensor, Tensor, Tensor]:
        if not self._dirty and self._low_rank_cache is not None:
            return self._low_rank_cache
        if self._M2 is None:
            raise RuntimeError("low_rank covariance precision requires a full scatter matrix.")
        cov = self._M2.to(torch.float32) / (self.n - 1)
        cov = 0.5 * (cov + cov.T)
        cov_diag = torch.clamp(cov.diagonal(), min=0.0)
        tau = cov_diag.mean().clamp(min=1e-6)
        ridge = (self.ridge_lambda * tau).clamp(min=1e-12)
        k = min(int(self.covariance_low_rank), self.hidden_dim, max(self.n - 1, 1))
        if k <= 0:
            vectors = torch.empty(self.hidden_dim, 0, device=self._device, dtype=torch.float32)
            values = torch.empty(0, device=self._device, dtype=torch.float32)
            self._low_rank_cache = (vectors, values, ridge)
            self._dirty = False
            return self._low_rank_cache

        eigvals, eigvecs = torch.linalg.eigh(cov)
        top_vals, top_idx = torch.topk(torch.clamp(eigvals, min=0.0), k=k)
        keep = top_vals > 1e-12
        vectors = eigvecs[:, top_idx][:, keep].contiguous()
        values = top_vals[keep].contiguous()
        self._low_rank_cache = (vectors, values, ridge)
        self._dirty = False
        return self._low_rank_cache

    def _compute_low_rank_precision(self) -> Tensor:
        vectors, values, ridge = self._low_rank_components()
        eye = torch.eye(self.hidden_dim, device=self._device, dtype=torch.float32)
        precision = eye / ridge
        if values.numel() == 0:
            return precision
        weights = values / (ridge * (ridge + values))
        return precision - (vectors * weights.unsqueeze(0)) @ vectors.T

    def _invalidate_precision_cache(self) -> None:
        self._cov_inv_cache = None
        self._precision_diag_cache = None
        self._low_rank_cache = None
        self._dirty = True

    # ------------------------------------------------------------------
    # 公开方法
    # ------------------------------------------------------------------

    def update(self, h: Tensor) -> None:
        """用单个隐状态向量增量更新流形 (Welford 在线均值/协方差)。
        Incrementally update the manifold with a single hidden state vector
        (Welford online mean/covariance).

        首次调用时自动初始化 mean 和散布矩阵。
        Automatically initializes the mean and scatter matrix on first call.

        Args:
            h: 隐状态向量 / Hidden state vector, shape [hidden_dim].
        """
        h = h.detach()
        if h.ndim != 1:
            raise ValueError(
                f"TruthManifold.update() expects a 1D hidden state vector, got shape {tuple(h.shape)}."
            )

        if self.mean is None:
            # 首次初始化
            self.hidden_dim = h.shape[-1]
            self._device = h.device
            self.mean = h.clone().to(torch.float32)
            if self.covariance_mode == "diag":
                self._M2 = None
            else:
                self._M2 = torch.zeros(
                    self.hidden_dim, self.hidden_dim, device=self._device, dtype=torch.float32
                )
            self._M2_diag = torch.zeros(self.hidden_dim, device=self._device, dtype=torch.float32)
            self.n = 1
            self._invalidate_precision_cache()
            return

        if h.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"Hidden dimension mismatch: expected {self.hidden_dim}, got {h.shape[-1]}."
            )

        # Welford 协方差更新：M2 累加 (x-μ_old)(x-μ_new)ᵀ
        # Welford covariance update: M2 accumulates (x-μ_old)(x-μ_new)ᵀ
        self.n += 1
        x = h.to(torch.float32)
        delta = x - self.mean
        self.mean = self.mean + delta / self.n
        delta2 = x - self.mean
        if self._M2_diag is None:
            if self._M2 is not None:
                self._M2_diag = self._M2.diagonal().clone()
            else:
                self._M2_diag = torch.zeros(self.hidden_dim, device=self._device, dtype=torch.float32)
        self._M2_diag = self._M2_diag + delta * delta2
        if self.covariance_mode != "diag":
            if self._M2 is None:
                raise RuntimeError(
                    f"{self.covariance_mode!r} covariance mode requires full scatter statistics."
                )
            self._M2 = self._M2 + torch.outer(delta, delta2)
        self._invalidate_precision_cache()

    def mahalanobis_distance(self, h: Tensor) -> Tensor:
        """Compute Mahalanobis distance using the configured covariance approximation.

        This method avoids materializing a dense precision matrix for diagonal and
        low-rank modes. The legacy ``cov_inv`` property remains available for
        callers that need a dense matrix.
        """
        if self.mean is None:
            raise ValueError("TruthManifold has no mean; call update() before scoring.")
        delta = (h.to(self._device) - self.mean).to(torch.float32)
        if self.n < 2:
            return torch.norm(delta, dim=-1)
        if self.covariance_mode == "diag":
            precision_diag = self._compute_diag_precision()
            m_sq = (delta * precision_diag * delta).sum(dim=-1)
        elif self.covariance_mode == "low_rank":
            vectors, values, ridge = self._low_rank_components()
            m_sq = (delta * delta).sum(dim=-1) / ridge
            if values.numel() > 0:
                proj = delta @ vectors
                weights = values / (ridge * (ridge + values))
                m_sq = m_sq - (proj * proj * weights).sum(dim=-1)
        else:
            m_sq = (delta @ self.cov_inv * delta).sum(dim=-1)
        return torch.sqrt(torch.clamp(m_sq, min=0.0))

    def distance(self, h: Tensor) -> Tensor:
        """Alias for ``mahalanobis_distance``."""
        return self.mahalanobis_distance(h)

    def to(self, device: Union[str, torch.device]) -> "TruthManifold":
        """Move manifold tensors to a device in-place and return self."""
        device = torch.device(device)
        if self.mean is not None:
            self.mean = self.mean.to(device)
        if self._M2 is not None:
            self._M2 = self._M2.to(device)
        if self._M2_diag is not None:
            self._M2_diag = self._M2_diag.to(device)
        if self._cov_inv_cache is not None:
            self._cov_inv_cache = self._cov_inv_cache.to(device)
        if self._precision_diag_cache is not None:
            self._precision_diag_cache = self._precision_diag_cache.to(device)
        if self._low_rank_cache is not None:
            vectors, values, ridge = self._low_rank_cache
            self._low_rank_cache = (vectors.to(device), values.to(device), ridge.to(device))
        if self.false_mean is not None:
            self.false_mean = self.false_mean.to(device)
        if self.contrastive_direction is not None:
            self.contrastive_direction = self.contrastive_direction.to(device)
        self._device = device
        return self

    def is_ready(self) -> bool:
        """流形至少经过 2 个样本后方可使用。"""
        if self.n < 2 or self.mean is None:
            return False
        if self.covariance_mode == "diag":
            return self._M2_diag is not None
        return self._M2 is not None

    def save(self, path: Union[str, Path]) -> None:
        """将流形序列化到磁盘。

        Args:
            path: 保存路径 (建议后缀 .pt 或 .bin).
        """
        state = {
            "format": 2,
            "mean": self.mean,
            "_M2": self._M2,
            "_M2_diag": self._M2_diag,
            "n": self.n,
            "hidden_dim": self.hidden_dim,
            "ridge_lambda": self.ridge_lambda,
            "covariance_mode": self.covariance_mode,
            "covariance_low_rank": self.covariance_low_rank,
            "false_mean": self.false_mean,
            "contrastive_direction": self.contrastive_direction,
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TruthManifold":
        """从磁盘加载流形。

        Args:
            path: 之前由 save() 保存的文件路径.

        Returns:
            恢复后的 TruthManifold 实例.
        """
        state = torch.load(path, weights_only=True)
        manifold = cls(
            covariance_mode=state.get("covariance_mode", "full"),
            covariance_low_rank=int(state.get("covariance_low_rank", 16)),
        )
        manifold.mean = state["mean"]
        manifold._M2 = state.get("_M2", None)
        manifold._M2_diag = state.get("_M2_diag", None)
        manifold.n = state["n"]
        manifold.hidden_dim = state["hidden_dim"]
        manifold.ridge_lambda = state.get("ridge_lambda", 0.1)
        manifold.false_mean = state.get("false_mean", None)
        manifold.contrastive_direction = state.get("contrastive_direction", None)
        manifold._dirty = True
        if manifold.mean is not None:
            manifold._device = manifold.mean.device
        return manifold


# ---------------------------------------------------------------------------
# 核心数学函数
# ---------------------------------------------------------------------------

def sherman_morrison_update(
    cov_inv: Tensor,
    x: Tensor,
    epsilon: float = 1e-6,
) -> Tensor:
    """Sherman-Morrison 秩-1 在线更新协方差逆矩阵。
    Sherman-Morrison rank-1 online update for the inverse covariance matrix.

    公式 / Formula: A⁻¹_new = A⁻¹ - (A⁻¹ x xᵀ A⁻¹) / (1 + xᵀ A⁻¹ x + ε)

    内部所有运算强制使用 FP32 以防止 FP16 数值崩溃。
    All internal computations are forced to FP32 to prevent FP16 numerical collapse.

    Args:
        cov_inv: 当前协方差逆矩阵 / Current inverse covariance matrix, shape [d, d].
        x: 增量向量 / Incremental vector (h - mean), shape [d].
        epsilon: 分母正则项, 防止除零 / Denominator regularization term to prevent division by zero.

    Returns:
        更新后的协方差逆矩阵 / Updated inverse covariance matrix, shape [d, d], with the same dtype as input.
    """
    orig_dtype = cov_inv.dtype

    # 防线 1: 强制 FP32 内部计算
    cov_inv_f32 = cov_inv.to(torch.float32)
    x_f32 = x.to(torch.float32)

    Ax = cov_inv_f32 @ x_f32                           # [d]
    denom = 1.0 + x_f32 @ Ax + epsilon                 # 标量
    cov_inv_f32 = cov_inv_f32 - torch.outer(Ax, Ax) / denom

    return cov_inv_f32.to(orig_dtype)


def mahalanobis_distance(
    h: Tensor,
    mean: Tensor,
    cov_inv: Tensor,
) -> Tensor:
    """计算隐状态到真值质心的马氏距离。
    Compute the Mahalanobis distance from a hidden state to the truth centroid.

    D_M(h) = sqrt( (h - μ)ᵀ Σ⁻¹ (h - μ) )

    Args:
        h: 隐状态向量 / Hidden state vector, shape [..., hidden_dim].
        mean: 质心向量 / Centroid vector, shape [hidden_dim].
        cov_inv: 协方差逆矩阵或对角精度向量 / Inverse covariance matrix,
            shape [hidden_dim, hidden_dim], or diagonal precision vector, shape [hidden_dim].

    Returns:
        马氏距离张量 / Mahalanobis distance tensor, shape matching batch dimension of h (>=0).
    """
    # 强制 FP32
    delta = (h - mean).to(torch.float32)
    cov_inv_f32 = cov_inv.to(torch.float32)

    # δᵀ Σ⁻¹ δ 批量化计算
    if cov_inv_f32.ndim == 1:
        if cov_inv_f32.shape[0] != delta.shape[-1]:
            raise ValueError(
                f"Diagonal precision dimension mismatch: expected {delta.shape[-1]}, "
                f"got {cov_inv_f32.shape[0]}."
            )
        m_sq = (delta * cov_inv_f32 * delta).sum(dim=-1)
    elif cov_inv_f32.ndim == 2:
        m_sq = (delta @ cov_inv_f32 * delta).sum(dim=-1)
    else:
        raise ValueError(
            "cov_inv must be a precision matrix [hidden_dim, hidden_dim] "
            "or diagonal precision vector [hidden_dim]."
        )
    # clamp 防止浮点误差导致 sqrt 负数
    return torch.sqrt(torch.clamp(m_sq, min=0.0))


def poincare_map(
    h_euclidean: Tensor,
    curvature: float = 1.0,
    max_norm: float = 0.999,
) -> Tensor:
    """将欧氏空间隐状态映射到庞加莱球模型。
    Map Euclidean space hidden states to the Poincaré ball model.

    使用指数映射 / Using exponential map at origin:
        exp_0(v) = tanh(√c ‖v‖ / 2) · v / (√c ‖v‖)

    Args:
        h_euclidean: 欧氏空间向量 / Euclidean space vector, shape [..., dim].
        curvature: 负曲率参数 c (正值) / Negative curvature parameter c (positive value).
        max_norm: 钳位最大范数, 保持在开球内部 / Clamp maximum norm to stay inside the open ball (< 1).

    Returns:
        庞加莱球坐标 / Poincaré ball coordinates, shape [..., dim], norm < 1.
    """
    h = h_euclidean.to(torch.float32)
    sqrt_c = curvature ** 0.5

    norm = torch.norm(h, dim=-1, keepdim=True).clamp(min=1e-8)
    # tanh(√c ‖v‖ / 2)
    scale = torch.tanh(sqrt_c * norm / 2.0) / (sqrt_c * norm)
    result = scale * h

    # 钳位：确保结果范数严格 < 1
    result_norm = torch.norm(result, dim=-1, keepdim=True)
    result = torch.where(
        result_norm >= max_norm,
        result * (max_norm / result_norm.clamp(min=1e-8)),
        result,
    )
    return result


def _poincare_distance(u: Tensor, v: Tensor, curvature: float = 1.0) -> Tensor:
    """计算庞加莱球上两点间的测地线距离。

    d(u, v) = (2/√c) · arctanh(√c · ‖(-u) ⊕_c v‖)

    简化实现 (使用恒等式):
        d(u, v) = (1/√c) · arccosh(1 + 2c · ‖u-v‖² / ((1-c‖u‖²)(1-c‖v‖²)))

    Args:
        u, v: 庞加莱球上的点, 形状 [..., dim].
        curvature: 曲率参数 c.

    Returns:
        测地线距离标量或张量.
    """
    c = curvature
    diff_sq = torch.sum((u - v) ** 2, dim=-1)
    u_sq = torch.sum(u ** 2, dim=-1)
    v_sq = torch.sum(v ** 2, dim=-1)

    denom = (1.0 - c * u_sq) * (1.0 - c * v_sq)
    denom = denom.clamp(min=1e-8)

    arg = 1.0 + 2.0 * c * diff_sq / denom
    # arccosh(x) = log(x + sqrt(x²-1)), 钳位 arg >= 1
    arg = arg.clamp(min=1.0 + 1e-8)

    return (1.0 / (c ** 0.5)) * torch.acosh(arg)


def hyperbolic_semantic_entropy(
    points_poincare: Tensor,
    curvature: float = 1.0,
) -> Tensor:
    """计算庞加莱球上一组点的双曲语义熵 (HSE)。
    Compute the Hyperbolic Semantic Entropy (HSE) for a set of points on the Poincaré ball.

    HSE 衡量一组语义表征在双曲空间中的分散程度
    HSE measures the dispersion of semantic representations
    in hyperbolic space:
        HSE = mean( d_hyp(p_i, centroid) )

    其中 centroid 使用爱因斯坦中点的简化近似
    centroid uses a simplified approximation of the Einstein midpoint
    (Euclidean mean followed by projection).

    Args:
        points_poincare: 庞加莱球上的点集, shape [W, D] or [W, B, D]
            Points on the Poincaré ball, W is window size.
        curvature: 曲率参数 c / Curvature parameter c.

    Returns:
        HSE 值 (>=0)。有 Batch 维则返回 [B]，否则返回标量。
        Returns [B] if input has Batch dim, otherwise scalar.
    """
    if points_poincare.shape[0] <= 1:
        # 如果是 [1, B, D]
        if points_poincare.ndim == 3:
            return torch.zeros(points_poincare.shape[1], device=points_poincare.device, dtype=torch.float32)
        return torch.tensor(0.0, device=points_poincare.device, dtype=torch.float32)

    points = points_poincare.to(torch.float32)

    # 简化中心点: 欧氏均值 → 投影回庞加莱球
    centroid_euclidean = points.mean(dim=0) # [B, D] or [D]

    # 增加假维度用于 poincare_map（原函数支持任意批次维度，直接传入即可）
    centroid = poincare_map(centroid_euclidean, curvature) # [B, D] or [D]

    # 将 centroid 扩展出 W 维度: [1, B, D] or [1, D]
    centroid_expanded = centroid.unsqueeze(0)

    # 批量计算测地线距离: _poincare_distance 支持广播
    distances = _poincare_distance(points, centroid_expanded, curvature) # [W, B] or [W]

    return distances.mean(dim=0)

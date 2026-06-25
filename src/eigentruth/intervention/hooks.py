"""EigenTruth Intervention — 动态探针与 Hook 系统 / Dynamic Probe and Hook System.

通过 PyTorch forward_hook 实现对 Transformer 隐状态的
Using PyTorch forward_hook on Transformer hidden states:
1. 实时拦截与马氏距离监测 / Real-time Mahalanobis distance monitoring
2. 激活引导向量 (Steering) 注入 / Activation steering vector injection
3. 可选庞加莱映射与 HSE 跟踪 / Optional Poincaré mapping and HSE tracking

所有 Hook 内部操作均在 torch.no_grad() 下执行，
截获的张量通过 .detach() 脱离计算图，严格防止显存泄漏。
All internal Hook operations are executed under torch.no_grad().
Intercepted tensors are detached from the compute graph to strictly prevent memory leaks.
"""

from __future__ import annotations

import copy
import logging
from collections import deque
from dataclasses import is_dataclass, replace
from typing import Any, Callable, Deque, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from eigentruth.core.math_engine import (
    TruthManifold,
    hyperbolic_semantic_entropy,
    poincare_map,
)

logger = logging.getLogger("eigentruth")


class TruthProbe:
    """真值探针：在指定 Transformer 层上挂载 forward_hook，
    实时监测隐状态偏移并注入引导修正。
    Truth Probe: Attaches a forward_hook to a specified Transformer layer
    to monitor hidden state deviations in real-time and inject steering corrections.

    Attributes:
        manifold: 真值流形（由 warmup 阶段构建） / Truth manifold (built during warmup).
        steering_lambda: 引导强度，表示"相对激活范数的移动比例" (0 = 不干预)。
            例如 0.1 表示把激活沿引导方向移动其自身范数的 ~10%。
            Steering strength as a fraction of the hidden-state norm (0 = no intervention).
            E.g. 0.1 moves the activation by ~10% of its own norm along the steering direction.
        threshold: 马氏距离阈值，超出时触发引导干预。距离尺度已按样本数归一化，
            阈值在不同 warmup 样本数下保持稳定。
            Mahalanobis distance threshold; triggers steering. The distance scale is
            sample-count-normalized, so thresholds are stable across warmup-set sizes.
        last_distance: 最近一次 hook 触发时的马氏距离
            Mahalanobis distance from the last triggered hook.
        track_hse: Whether to compute optional Hyperbolic Semantic Entropy diagnostics.
        last_hse: 最近一次计算的双曲语义熵 / The most recently computed Hyperbolic Semantic Entropy.
        is_active: hook 是否已挂载且激活 / Whether the hook is mounted and active.
    """

    def __init__(
        self,
        manifold: TruthManifold,
        steering_lambda: float = 0.1,
        threshold: float = 15.0,
        hse_window_size: int = 20,
        curvature: float = 1.0,
        track_hse: bool = False,
    ) -> None:
        self.manifold = manifold
        self.steering_lambda = steering_lambda
        self.threshold = threshold
        self.hse_window_size = hse_window_size
        self.curvature = curvature
        self.track_hse = bool(track_hse)

        # 运行时状态 (取 Batch 最大值供快速查询)
        self.last_distance: float = 0.0
        self.last_hse: float = 0.0
        self.is_active: bool = False

        # HSE 追踪：收集生成过程中的庞加莱点 [W, B, D]
        self._poincare_history: Deque[Tensor] = deque(maxlen=hse_window_size)

        # hook 句柄
        self._hook_handle: Optional[torch.utils.hooks.RemovableHook] = None

    # ------------------------------------------------------------------
    # 生命周期管理
    # ------------------------------------------------------------------

    def register(
        self, model: nn.Module, layer_idx: int,
        custom_layer_path: Optional[str] = None,
    ) -> None:
        """在指定 Transformer 层注册 forward_hook。
        Register a forward_hook on the specified Transformer layer.

        支持负索引（如 -10 表示倒数第 10 层）。
        Supports negative indexing (e.g., -10 means the 10th layer from the end).

        Args:
            model: HuggingFace CausalLM 模型 / HuggingFace CausalLM model.
            layer_idx: 目标层索引 / Target layer index.
            custom_layer_path: 自定义层路径 (如 "model.encoder.blocks") / Custom layer path.

        Raises:
            ValueError: 无法定位 Transformer 层列表 / Cannot locate Transformer layer list.
            IndexError: layer_idx 超出范围 / layer_idx out of bounds.
        """
        layers = self._find_layers(model, custom_layer_path=custom_layer_path)
        target = self._select_layer(layers, layer_idx)

        # 如果已有 hook，先移除
        self.remove()

        self._hook_handle = target.register_forward_hook(self._hook_fn)
        self.is_active = True
        logger.debug(
            f"TruthProbe registered on layer {layer_idx} "
            f"(actual: {type(target).__name__})"
        )

    def remove(self) -> None:
        """安全移除 hook，释放资源。
        Safely remove the hook and release resources.
        """
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        self.is_active = False
        self._poincare_history.clear()

    def reset_history(self) -> None:
        """清空 HSE 历史（每次 generate 调用前重置）。
        Clear HSE history (reset before every generate call).
        """
        self._poincare_history.clear()
        self.last_hse = 0.0

    # ------------------------------------------------------------------
    # Hook 回调
    # ------------------------------------------------------------------

    def _hook_fn(
        self,
        _module: nn.Module,
        _input: Any,
        output: Any,
    ) -> Any:
        """forward_hook 回调：拦截隐状态、计算距离、注入引导。
        forward_hook callback: Intercept hidden states, compute distance, and inject steering.

        工程防线 / Engineering Defenses:
        - 处理 HF 的 tuple/BaseModelOutputWithPast 输出格式 / Handle HF's tuple/BaseModelOutputWithPast output format
        - .detach() 脱离计算图 / Detach from computation graph
        - torch.no_grad() 包裹所有运算 / Wrap all operations in torch.no_grad()
        - 只处理最后一个 token (h[:, -1:, :]) / Only process the last token (h[:, -1:, :])
        """
        # 防线 3: 处理 tuple / Tensor / HF ModelOutput 输出
        hidden, repack_output = self._unpack_output(output)

        if not self.manifold.is_ready():
            return output  # 流形未就绪，直接透传

        with torch.no_grad():
            # 防线 2+3: detach + 切片最后一个 token
            h_last = hidden[:, -1:, :].detach()  # [B, 1, D]
            h_vec = h_last.squeeze(1)              # [B, D]
            self._ensure_manifold_device(h_vec.device)

            # 批量计算马氏距离 [B]
            dist = self.manifold.mahalanobis_distance(h_vec)
            # 保存最大距离用于预警诊断
            self.last_distance = dist.max().item()

            if self.track_hse:
                # Optional diagnostic retained for ablations; not part of the default path.
                h_poincare = poincare_map(h_vec, curvature=self.curvature)  # [B, D]
                # 动态 Batch 保护：如果 Batch 大小变化，清空历史
                # Dynamic Batch guard: clear history if batch size changes
                if (len(self._poincare_history) > 0
                        and self._poincare_history[-1].shape[0] != h_poincare.shape[0]):
                    self._poincare_history.clear()
                self._poincare_history.append(h_poincare.cpu())

                if len(self._poincare_history) >= 2:
                    pts = torch.stack(list(self._poincare_history))  # [W, B, D]
                    hse_batch = hyperbolic_semantic_entropy(
                        pts, curvature=self.curvature
                    )  # [B]
                    self.last_hse = hse_batch.max().item()

            # 引导干预 / Steering intervention
            mask = dist > self.threshold  # [B]
            if mask.any() and self.steering_lambda > 0:
                steering = self._compute_steering_vector(h_vec)  # 单位向量 [B, D]
                # 按各样本激活范数缩放：steering_lambda 表示"相对激活范数的移动比例"，
                # 与模型/层无关。单位向量直接乘 λ 会因激活范数(常为数十~数百)而几乎无效。
                # Scale by each row's activation norm so steering_lambda is a fraction of the
                # hidden-state magnitude (model/layer-portable). A bare unit vector × λ is
                # negligible because activation norms are often tens to hundreds.
                h_norm = torch.norm(h_vec, dim=-1, keepdim=True)  # [B, 1]
                correction = (self.steering_lambda * h_norm) * steering  # [B, D]
                # 将无需干预的批次修正量清零 / Zero out rows below threshold
                correction = correction * mask.unsqueeze(1).to(correction.dtype)

                hidden = hidden.clone()  # 避免原地修改影响计算图
                hidden[:, -1:, :] = hidden[:, -1:, :] + correction.unsqueeze(1)

        return repack_output(hidden)

    # ------------------------------------------------------------------
    # 引导向量
    # ------------------------------------------------------------------

    def _compute_steering_vector(self, h: Tensor) -> Tensor:
        """计算从当前激活到真值质心的归一化引导方向。
        Compute the normalized steering direction from the current activation to the truth centroid.

        如果有对比方向，则沿对比方向；否则方向朝向真值质心。
        If a contrastive direction exists, follow it; otherwise, direct towards the truth centroid.

        Args:
            h: 当前隐状态, 形状 [B, D] / Current hidden state, shape [B, D].

        Returns:
            引导向量, 形状 [B, D] / Steering vector, shape [B, D].
        """
        if self.manifold.contrastive_direction is not None:
            direction = self.manifold.contrastive_direction.to(h.device)
            direction = direction.unsqueeze(0).expand_as(h)
        else:
            mean = self.manifold.mean.to(h.device)
            direction = mean.to(torch.float32) - h.to(torch.float32)

        norm = torch.norm(direction, dim=-1, keepdim=True).clamp(min=1e-8)
        return (direction / norm).to(h.dtype)

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _ensure_manifold_device(self, device: torch.device) -> None:
        """Keep manifold tensors on the same device as the intercepted hidden states."""
        cov_inv = self.manifold.cov_inv
        if self.manifold.mean is None or cov_inv is None:
            return
        if self.manifold.mean.device != device or cov_inv.device != device:
            self.manifold.to(device)

    @staticmethod
    def _select_layer(layers: Sequence[nn.Module], layer_idx: int) -> nn.Module:
        try:
            return layers[layer_idx]
        except IndexError as exc:
            raise IndexError(
                f"layer_idx {layer_idx} is out of range for {len(layers)} transformer layers."
            ) from exc

    @staticmethod
    def _find_layers(
        model: nn.Module,
        custom_layer_path: Optional[str] = None,
    ) -> Sequence[nn.Module]:
        """在 HF 模型中定位 Transformer 层列表。
        Locate the list of Transformer layers in an HF model.

        Args:
            model: 目标模型 / Target model.
            custom_layer_path: 自定义属性路径 (如 "model.encoder.blocks") / Custom attribute path.

        Returns:
            nn.ModuleList 或类似可索引容器 / nn.ModuleList or similar indexable container.

        Raises:
            ValueError: 无法自动定位层列表 / Cannot automatically locate layer list.
        """
        # 优先使用用户自定义路径 / Prefer user-specified custom path
        if custom_layer_path is not None:
            obj = model
            try:
                for attr in custom_layer_path.split("."):
                    obj = getattr(obj, attr)
                if isinstance(obj, (nn.ModuleList, list)) and len(obj) > 0:
                    return obj
            except AttributeError:
                pass
            raise ValueError(
                f"Custom layer path '{custom_layer_path}' not found on model. "
                f"Please verify the attribute path."
            )

        # 常见 HF 架构的层路径 / Common HF architecture layer paths
        candidates = [
            ("model", "layers"),        # Llama, Qwen, Mistral
            ("transformer", "h"),       # GPT-2, GPT-Neo
            ("model", "decoder", "layers"),  # OPT
            ("gpt_neox", "layers"),     # GPT-NeoX
        ]

        for path in candidates:
            obj = model
            try:
                for attr in path:
                    obj = getattr(obj, attr)
                if isinstance(obj, (nn.ModuleList, list)) and len(obj) > 0:
                    return obj
            except AttributeError:
                continue

        # 回退: 如果模型本身就有 .layers 属性（测试用简单模型）
        # Fallback: model.layers (for simple test models)
        if hasattr(model, "layers") and isinstance(model.layers, (nn.ModuleList, list)):
            return model.layers

        raise ValueError(
            "Cannot locate transformer layers. "
            "Supported patterns: model.model.layers, model.transformer.h, etc. "
            "Use custom_layer_path='your.path.to.layers' for custom models."
        )

    @staticmethod
    def _unpack_output(output: Any) -> Tuple[Tensor, Callable[[Tensor], Any]]:
        """解包 HF 模型输出，提取 hidden_states 张量。
        Unpack HF model output to extract the hidden_states tensor.

        Returns:
            (hidden_states, repack_callable)
        """
        if isinstance(output, tuple):
            return output[0], lambda hidden: (hidden,) + output[1:]
        if isinstance(output, Tensor):
            return output, lambda hidden: hidden
        # BaseModelOutputWithPast 等 dataclass 式输出
        # Handle dataclass-style outputs like BaseModelOutputWithPast
        if hasattr(output, "last_hidden_state") and isinstance(output.last_hidden_state, Tensor):
            return output.last_hidden_state, lambda hidden: TruthProbe._replace_output_attr(
                output, "last_hidden_state", hidden
            )
        if hasattr(output, "hidden_states") and isinstance(output.hidden_states, Tensor):
            return output.hidden_states, lambda hidden: TruthProbe._replace_output_attr(
                output, "hidden_states", hidden
            )
        # 回退 / Fallback
        raise TypeError(
            f"Unsupported output type from hooked layer: {type(output)}. "
            f"Expected Tensor, tuple, or HF BaseModelOutput."
        )

    @staticmethod
    def _replace_output_attr(output: Any, attr: str, hidden: Tensor) -> Any:
        """将修改后的 hidden_states 重新打包为原始格式。
        Repack the modified hidden_states into their original format.
        """
        if is_dataclass(output):
            try:
                return replace(output, **{attr: hidden})
            except (TypeError, ValueError):
                pass

        try:
            cloned = copy.copy(output)
            setattr(cloned, attr, hidden)
            return cloned
        except Exception as exc:
            raise TypeError(
                f"Cannot repack hooked layer output of type {type(output)} with attribute '{attr}'."
            ) from exc

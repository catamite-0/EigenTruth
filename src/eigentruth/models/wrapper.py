"""EigenTruth Models — HuggingFace 顶层封装器 / HuggingFace Top-level Wrapper.

提供 EigenTruthWrapper，实现 / Provides EigenTruthWrapper, implementing:
- warmup(): 真值流形冷启动 / Truth manifold cold start
- generate(): 受控文本生成 / Controlled text generation
- forward(): 透传给原始模型 / Passthrough to the original model

非侵入式设计：不修改原始模型参数，通过 hook 实现所有干预。
Non-intrusive design: implements all interventions via hooks.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from eigentruth.core.math_engine import TruthManifold
from eigentruth.intervention.hooks import TruthProbe

logger = logging.getLogger("eigentruth")
# 库应只添加 NullHandler，由调用方控制日志输出
# Libraries should only add NullHandler; let callers control log output
logger.addHandler(logging.NullHandler())


class EigenTruthWrapper(nn.Module):
    """EigenTruth 主封装器：为 HuggingFace CausalLM 模型添加表征诊断与实验性引导。
    EigenTruth main wrapper: Adds representation diagnostics and experimental steering to HuggingFace CausalLM models.

    用法示例::

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from eigentruth.models.wrapper import EigenTruthWrapper

        base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
        monitor = EigenTruthWrapper(
            model=base_model,
            target_layer_idx=-10,
            steering_lambda=0.1,
            mahalanobis_threshold=15.0,
        )
        monitor.warmup(["光速是每秒299792458米。"], tokenizer)
        outputs = monitor.generate(**inputs, max_new_tokens=100)

    Args:
        model: HuggingFace CausalLM 模型实例 / Model instance.
        target_layer_idx: 挂载探针的 Transformer 层索引
            Transformer layer index (supports negative indexing).
        steering_lambda: 激活引导强度，表示"相对激活范数的移动比例" (0 = 纯监测)。
            Activation steering strength as a fraction of the hidden-state norm (0 = monitor only).
        mahalanobis_threshold: 马氏距离阈值，超出时触发引导与预警。
            距离尺度已按样本数归一化，阈值在不同 warmup 样本数下保持稳定。
            Mahalanobis distance threshold; triggers steering and warnings when exceeded.
            The distance scale is sample-count-normalized and stable across warmup-set sizes.
        hse_warning_threshold: HSE 预警阈值 / HSE warning threshold.
        curvature: 庞加莱球曲率 / Poincaré ball curvature.
        hse_window_size: HSE 滑动窗口大小 / HSE sliding window size.
        covariance_mode: TruthManifold covariance approximation mode:
            "full" (default), "diag", or "low_rank".
        covariance_low_rank: Rank used when ``covariance_mode="low_rank"``.
        custom_layer_path: 自定义层路径 / Custom layer attribute path.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer_idx: int = -10,
        steering_lambda: float = 0.1,
        mahalanobis_threshold: float = 15.0,
        hse_warning_threshold: float = 5.0,
        curvature: float = 1.0,
        hse_window_size: int = 20,
        covariance_mode: str = "full",
        covariance_low_rank: int = 16,
        custom_layer_path: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.model = model
        self.target_layer_idx = target_layer_idx
        self.steering_lambda = steering_lambda
        self.mahalanobis_threshold = mahalanobis_threshold
        self.hse_warning_threshold = hse_warning_threshold
        self.curvature = curvature
        self.hse_window_size = hse_window_size
        self.covariance_mode = covariance_mode
        self.covariance_low_rank = covariance_low_rank
        self.custom_layer_path = custom_layer_path

        # 内部状态 / Internal state
        self.manifold = TruthManifold(
            covariance_mode=self.covariance_mode,
            covariance_low_rank=self.covariance_low_rank,
        )
        self.probe: Optional[TruthProbe] = None
        self._is_warmed_up: bool = False

    # ------------------------------------------------------------------
    # Warmup: 真值流形冷启动
    # ------------------------------------------------------------------

    @torch.no_grad()
    def warmup(
        self,
        fact_dataset: List[str],
        tokenizer: Any,
        false_dataset: Optional[List[str]] = None,
        max_length: int = 128,
    ) -> None:
        """使用事实语料构建真值流形。
        Build the truth manifold using a factual corpus.

        提取目标层的隐状态，增量构建 TruthManifold。
        如果提供了 false_dataset，将额外构建对比方向。
        Extracts hidden states from the target layer to incrementally build the TruthManifold.
        If false_dataset is provided, it will additionally construct a contrastive direction.

        Args:
            fact_dataset: 绝对正确的事实文本列表 / List of absolutely correct factual texts.
            tokenizer: HuggingFace tokenizer.
            false_dataset: 绝对错误的文本列表 / List of false texts (for contrastive direction).
            max_length: tokenize 最大长度 / Max length for tokenization.
        """
        if len(fact_dataset) == 0:
            raise ValueError("fact_dataset must contain at least one factual text.")

        if self.probe is not None:
            self.probe.remove()
            self.probe = None
        self._is_warmed_up = False
        self.manifold = TruthManifold(
            covariance_mode=self.covariance_mode,
            covariance_low_rank=self.covariance_low_rank,
        )
        device = self._get_device()

        def _collect_states(dataset: List[str]) -> List[Tensor]:
            collected: List[Tensor] = []

            def _collect_hook(_module: nn.Module, _input: Any, output: Any) -> None:
                hidden, _ = TruthProbe._unpack_output(output)
                # 提取最后一个 token 的表征，逐样本安全处理 B>1
                # Extract last-token repr, safely handle batch_size > 1
                h_last = hidden.detach()[:, -1, :]  # [B, D]
                collected.extend(h.cpu() for h in h_last)

            layers = TruthProbe._find_layers(
                self.model, custom_layer_path=self.custom_layer_path
            )
            target_layer = TruthProbe._select_layer(layers, self.target_layer_idx)
            hook_handle = target_layer.register_forward_hook(_collect_hook)

            try:
                for text in dataset:
                    inputs = tokenizer(
                        text, return_tensors="pt", max_length=max_length,
                        truncation=True, padding=True
                    ).to(device)

                    if isinstance(inputs, dict):
                        self.model(**inputs)
                    elif hasattr(inputs, "input_ids"):
                        self.model(
                            input_ids=inputs.input_ids,
                            attention_mask=getattr(inputs, "attention_mask", None),
                        )
                    else:
                        self.model(**dict(inputs))
            finally:
                hook_handle.remove()

            return collected

        logger.info("Collecting true representations...")
        fact_states = _collect_states(fact_dataset)
        if len(fact_states) == 0:
            raise RuntimeError(
                "No hidden states were collected during warmup. "
                "Check target_layer_idx/custom_layer_path and the model forward path."
            )

        # 用收集到的隐状态构建流形
        for h in fact_states:
            self.manifold.update(h)

        if false_dataset is not None:
            logger.info("Collecting false representations for contrastive direction...")
            false_states = _collect_states(false_dataset)
            if len(false_states) > 0:
                false_mean = torch.stack(false_states).mean(dim=0)
                self.manifold.false_mean = false_mean.to(torch.float32)
                # 构建 Truth Direction = True Mean - False Mean，并归一化
                # Build Truth Direction = True Mean - False Mean, then normalize
                raw_dir = self.manifold.mean - self.manifold.false_mean
                norm = torch.norm(raw_dir).clamp(min=1e-8)
                self.manifold.contrastive_direction = raw_dir / norm
                logger.info("Contrastive direction successfully established.")

        if not self.manifold.is_ready():
            logger.warning(
                f"⚠️ 流形构建不完整（仅 {self.manifold.n} 个样本）。"
                f"建议至少提供 2 条以上事实语料。"
            )
            return

        self.manifold.to(device)
        logger.info(
            f"✅ 真值流形已就绪 — {self.manifold.n} 个样本, "
            f"hidden_dim={self.manifold.hidden_dim}"
        )

        # 挂载正式探针
        self._activate_probe()
        self._is_warmed_up = True

    # ------------------------------------------------------------------
    # Generate: 受控文本生成
    # ------------------------------------------------------------------

    def generate(self, **kwargs: Any) -> Any:
        """受控文本生成。
        Controlled text generation.

        透传参数给原始 model.generate()，在生成过程中
        TruthProbe hook 自动工作，实时监测并干预幻觉。
        Passes parameters through to the original model.generate(). During generation,
        the TruthProbe hook automatically operates, monitoring and intervening in real-time.

        生成完成后检查 HSE，如超阈值则输出预警。
        After generation, checks HSE and outputs a warning if the threshold is exceeded.

        Args:
            **kwargs: 传递给 model.generate() 的所有参数 / All parameters passed to model.generate().

        Returns:
            model.generate() 的原始返回值 / Original return value of model.generate().
        """
        if not self._is_warmed_up:
            logger.warning(
                "⚠️ 模型未经 warmup，将不启用 EigenTruth 诊断或引导。"
                "请先调用 monitor.warmup(fact_dataset, tokenizer)。\n"
                "⚠️ Model not warmed up; EigenTruth diagnostics and steering are disabled. "
                "Please call monitor.warmup(fact_dataset, tokenizer) first."
            )
            return self.model.generate(**kwargs)

        # 重置 HSE 历史
        if self.probe is not None:
            self.probe.reset_history()

        # 生成
        outputs = self.model.generate(**kwargs)

        # 生成后检查 HSE
        self._check_hse_warning()

        return outputs

    # ------------------------------------------------------------------
    # Forward: 透传
    # ------------------------------------------------------------------

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """透传给原始模型的 forward 方法。"""
        return self.model(*args, **kwargs)

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------

    @property
    def is_warmed_up(self) -> bool:
        """流形是否已构建完成。"""
        return self._is_warmed_up

    @property
    def last_distance(self) -> float:
        """最近一次的马氏距离。"""
        return self.probe.last_distance if self.probe else 0.0

    @property
    def last_hse(self) -> float:
        """最近一次的双曲语义熵。"""
        return self.probe.last_hse if self.probe else 0.0

    def get_diagnostics(self) -> dict:
        """获取诊断信息字典。"""
        return {
            "is_warmed_up": self._is_warmed_up,
            "manifold_ready": self.manifold.is_ready(),
            "manifold_samples": self.manifold.n,
            "hidden_dim": self.manifold.hidden_dim,
            "last_mahalanobis_distance": self.last_distance,
            "last_hse": self.last_hse,
            "mahalanobis_threshold": self.mahalanobis_threshold,
            "hse_warning_threshold": self.hse_warning_threshold,
            "steering_lambda": self.steering_lambda,
            "probe_active": self.probe.is_active if self.probe else False,
        }

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _activate_probe(self) -> None:
        """创建并激活正式探针。"""
        if self.probe is not None:
            self.probe.remove()

        self.probe = TruthProbe(
            manifold=self.manifold,
            steering_lambda=self.steering_lambda,
            threshold=self.mahalanobis_threshold,
            hse_window_size=self.hse_window_size,
            curvature=self.curvature,
        )
        self.probe.register(
            self.model, self.target_layer_idx,
            custom_layer_path=self.custom_layer_path,
        )

    def _check_hse_warning(self) -> None:
        """检查 HSE 是否超阈值并输出预警。"""
        if self.probe is None:
            return

        if self.probe.last_hse > self.hse_warning_threshold:
            logger.warning(
                f"⚠️ 检测到深层语义发散 (HSE={self.probe.last_hse:.2f} > "
                f"阈值 {self.hse_warning_threshold:.2f})，"
                f"系统可能正在产生幻觉！\n"
                f"⚠️ Deep semantic divergence detected (HSE={self.probe.last_hse:.2f} > "
                f"Threshold {self.hse_warning_threshold:.2f}). "
                f"The system might be hallucinating!"
            )

    def _get_device(self) -> torch.device:
        """获取模型所在设备。"""
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    # ------------------------------------------------------------------
    # 清理
    # ------------------------------------------------------------------

    def detach_probe(self) -> None:
        """移除探针，恢复原始模型行为。"""
        if self.probe is not None:
            self.probe.remove()
            self.probe = None
        self._is_warmed_up = False
        logger.info(
            "🔓 EigenTruth 探针已移除，模型恢复原始状态。"
            " / EigenTruth probe removed, model restored."
        )

    def __del__(self) -> None:
        """析构时确保 hook 被移除（跳过日志以避免解释器关闭时报错）。
        Ensure hooks are removed during destruction (skip logging to avoid interpreter shutdown errors).
        """
        try:
            if self.probe is not None:
                self.probe.remove()
                self.probe = None
        except Exception:
            pass

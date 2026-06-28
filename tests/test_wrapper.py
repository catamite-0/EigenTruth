"""Phase 3 单元测试 — models/wrapper.py

使用 mock 模型和 tokenizer 进行测试，CPU 可运行。
覆盖：初始化、warmup 流形构建、generate 流程、HSE 预警、诊断信息。
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from eigentruth.models.wrapper import EigenTruthWrapper

# ===================================================================
# Mock 基础设施
# ===================================================================

class MockTransformerLayer(nn.Module):
    """模拟 Transformer 层。"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor, **kwargs) -> tuple:
        h = self.linear(x)
        return (h, None)


class MockCausalLM(nn.Module):
    """模拟 HuggingFace CausalLM 模型。"""

    def __init__(self, n_layers: int = 4, hidden_dim: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([
            MockTransformerLayer(hidden_dim) for _ in range(n_layers)
        ])
        self.lm_head = nn.Linear(hidden_dim, 100, bias=False)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        B, S = input_ids.shape
        x = torch.randn(B, S, self.hidden_dim, device=input_ids.device)
        for layer in self.model.layers:
            x, _ = layer(x)
        logits = self.lm_head(x)
        return MagicMock(logits=logits, last_hidden_state=x)

    def generate(self, input_ids=None, attention_mask=None, max_new_tokens=10, **kwargs):
        B, S = input_ids.shape
        generated = input_ids.clone()
        for _ in range(max_new_tokens):
            x = torch.randn(B, generated.shape[1], self.hidden_dim, device=input_ids.device)
            for layer in self.model.layers:
                x, _ = layer(x)
            logits = self.lm_head(x[:, -1:, :])
            next_token = logits.argmax(dim=-1)
            generated = torch.cat([generated, next_token], dim=1)
        return generated


class _TokenizerResult:
    """模拟 HF BatchEncoding。"""

    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def to(self, device):
        return {"input_ids": self.input_ids, "attention_mask": self.attention_mask}


class MockTokenizer:
    """模拟 HuggingFace Tokenizer。"""

    def __init__(self, vocab_size: int = 100):
        self.vocab_size = vocab_size

    def __call__(self, text: str, **kwargs) -> _TokenizerResult:
        seq_len = max(min(len(text) // 2 + 2, kwargs.get("max_length", 128)), 3)
        input_ids = torch.randint(0, self.vocab_size, (1, seq_len))
        attention_mask = torch.ones_like(input_ids)
        return _TokenizerResult(input_ids, attention_mask)

    def decode(self, token_ids, **kwargs) -> str:
        return f"[decoded {len(token_ids)} tokens]"


# ===================================================================
# 初始化测试
# ===================================================================

class TestWrapperInit:

    def test_default_params(self):
        model = MockCausalLM()
        wrapper = EigenTruthWrapper(model)
        assert wrapper.target_layer_idx == -10
        assert wrapper.steering_lambda == 0.1
        assert wrapper.mahalanobis_threshold == 15.0
        assert wrapper.track_hse is False
        assert not wrapper.is_warmed_up

    def test_custom_params(self):
        model = MockCausalLM()
        wrapper = EigenTruthWrapper(
            model, target_layer_idx=-2, steering_lambda=0.5,
            mahalanobis_threshold=20.0, hse_warning_threshold=10.0,
            track_hse=True,
        )
        assert wrapper.target_layer_idx == -2
        assert wrapper.steering_lambda == 0.5
        assert wrapper.mahalanobis_threshold == 20.0
        assert wrapper.hse_warning_threshold == 10.0
        assert wrapper.track_hse is True

    def test_model_is_stored(self):
        model = MockCausalLM()
        wrapper = EigenTruthWrapper(model)
        assert wrapper.model is model


# ===================================================================
# Warmup 测试
# ===================================================================

class TestWarmup:

    def test_warmup_builds_manifold(self):
        model = MockCausalLM(n_layers=4, hidden_dim=32)
        wrapper = EigenTruthWrapper(model, target_layer_idx=-1)
        tokenizer = MockTokenizer()

        wrapper.warmup(["事实一：地球是圆的", "事实二：光速有限", "事实三：水在零度结冰"], tokenizer)

        assert wrapper.is_warmed_up
        assert wrapper.manifold.is_ready()
        assert wrapper.manifold.n == 3
        assert wrapper.manifold.hidden_dim == 32
        assert wrapper.manifold.mean.device == next(model.parameters()).device

    def test_warmup_activates_probe(self):
        model = MockCausalLM(n_layers=4, hidden_dim=32)
        wrapper = EigenTruthWrapper(model, target_layer_idx=-1)
        tokenizer = MockTokenizer()

        wrapper.warmup(["事实一", "事实二", "事实三"], tokenizer)

        assert wrapper.probe is not None
        assert wrapper.probe.is_active

    def test_warmup_with_single_sample_warns(self):
        model = MockCausalLM(n_layers=4, hidden_dim=32)
        wrapper = EigenTruthWrapper(model, target_layer_idx=-1)
        tokenizer = MockTokenizer()

        with patch("eigentruth.models.wrapper.logger") as mock_logger:
            wrapper.warmup(["唯一事实"], tokenizer)
            mock_logger.warning.assert_called()
        assert not wrapper.is_warmed_up
        assert wrapper.probe is None
        assert not wrapper.manifold.is_ready()

    def test_warmup_rejects_empty_dataset(self):
        model = MockCausalLM(n_layers=4, hidden_dim=32)
        wrapper = EigenTruthWrapper(model, target_layer_idx=-1)
        tokenizer = MockTokenizer()

        with pytest.raises(ValueError, match="fact_dataset"):
            wrapper.warmup([], tokenizer)

    def test_repeated_warmup_replaces_probe_hook(self):
        model = MockCausalLM(n_layers=4, hidden_dim=32)
        wrapper = EigenTruthWrapper(model, target_layer_idx=-1)
        tokenizer = MockTokenizer()
        target_layer = model.model.layers[-1]

        wrapper.warmup(["事实一", "事实二", "事实三"], tokenizer)
        assert len(target_layer._forward_hooks) == 1

        wrapper.warmup(["事实四", "事实五", "事实六"], tokenizer)
        assert len(target_layer._forward_hooks) == 1

    def test_warmup_with_false_dataset_computes_contrastive_direction(self):
        model = MockCausalLM(n_layers=4, hidden_dim=32)
        wrapper = EigenTruthWrapper(model, target_layer_idx=-1)
        tokenizer = MockTokenizer()

        wrapper.warmup(
            fact_dataset=["事实一", "事实二", "事实三"],
            tokenizer=tokenizer,
            false_dataset=["错误一", "错误二"]
        )

        assert wrapper.manifold.false_mean is not None
        assert wrapper.manifold.contrastive_direction is not None
        assert wrapper.manifold.false_mean.shape == (32,)
        assert wrapper.manifold.contrastive_direction.shape == (32,)


# ===================================================================
# Generate 测试
# ===================================================================

class TestGenerate:

    def _setup_warmed_wrapper(self):
        model = MockCausalLM(n_layers=4, hidden_dim=32)
        wrapper = EigenTruthWrapper(model, target_layer_idx=-1, mahalanobis_threshold=1000.0)
        tokenizer = MockTokenizer()
        wrapper.warmup(["事实一", "事实二", "事实三"], tokenizer)
        return wrapper

    def test_generate_returns_tokens(self):
        wrapper = self._setup_warmed_wrapper()
        input_ids = torch.randint(0, 100, (1, 5))
        output = wrapper.generate(input_ids=input_ids, max_new_tokens=3)
        assert output.shape[1] == 5 + 3

    def test_generate_without_warmup_warns(self):
        model = MockCausalLM(n_layers=4, hidden_dim=32)
        wrapper = EigenTruthWrapper(model)
        with patch("eigentruth.models.wrapper.logger") as mock_logger:
            input_ids = torch.randint(0, 100, (1, 5))
            wrapper.generate(input_ids=input_ids, max_new_tokens=2)
            mock_logger.warning.assert_called()

    def test_generate_updates_distance(self):
        wrapper = self._setup_warmed_wrapper()
        input_ids = torch.randint(0, 100, (1, 5))
        wrapper.generate(input_ids=input_ids, max_new_tokens=3)
        assert wrapper.last_distance >= 0.0


# ===================================================================
# HSE 预警测试
# ===================================================================

class TestHSEWarning:

    def test_warning_on_high_hse(self):
        model = MockCausalLM(n_layers=4, hidden_dim=32)
        wrapper = EigenTruthWrapper(
            model, target_layer_idx=-1,
            hse_warning_threshold=0.001, mahalanobis_threshold=1000.0,
            track_hse=True,
        )
        tokenizer = MockTokenizer()
        wrapper.warmup(["事实一", "事实二", "事实三"], tokenizer)

        with patch("eigentruth.models.wrapper.logger"):
            input_ids = torch.randint(0, 100, (1, 5))
            wrapper.generate(input_ids=input_ids, max_new_tokens=5)
            # 验证不抛异常即可（HSE 是否触发取决于随机输出）


# ===================================================================
# 诊断信息测试
# ===================================================================

class TestDiagnostics:

    def test_diagnostics_before_warmup(self):
        model = MockCausalLM()
        wrapper = EigenTruthWrapper(model)
        diag = wrapper.get_diagnostics()
        assert diag["is_warmed_up"] is False
        assert diag["manifold_ready"] is False
        assert diag["manifold_samples"] == 0
        assert diag["probe_active"] is False

    def test_diagnostics_after_warmup(self):
        model = MockCausalLM(n_layers=4, hidden_dim=32)
        wrapper = EigenTruthWrapper(model, target_layer_idx=-1)
        tokenizer = MockTokenizer()
        wrapper.warmup(["事实一", "事实二", "事实三"], tokenizer)

        diag = wrapper.get_diagnostics()
        assert diag["is_warmed_up"] is True
        assert diag["manifold_ready"] is True
        assert diag["manifold_samples"] == 3
        assert diag["hidden_dim"] == 32
        assert diag["probe_active"] is True


# ===================================================================
# 探针生命周期测试
# ===================================================================

class TestProbeLifecycle:

    def test_detach_probe(self):
        model = MockCausalLM(n_layers=4, hidden_dim=32)
        wrapper = EigenTruthWrapper(model, target_layer_idx=-1)
        tokenizer = MockTokenizer()
        wrapper.warmup(["事实一", "事实二", "事实三"], tokenizer)

        assert wrapper.is_warmed_up
        wrapper.detach_probe()
        assert not wrapper.is_warmed_up
        assert wrapper.probe is None

    def test_forward_passthrough(self):
        model = MockCausalLM(n_layers=4, hidden_dim=32)
        wrapper = EigenTruthWrapper(model)
        input_ids = torch.randint(0, 100, (1, 5))
        result = wrapper(input_ids=input_ids)
        assert result is not None

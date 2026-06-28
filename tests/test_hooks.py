"""Phase 2 单元测试 — intervention/hooks.py

使用简单的 nn.Module 替代真实 HF 模型，CPU 可运行。
覆盖：hook 注册/移除、距离计算、引导注入、输出格式处理。
"""

from dataclasses import dataclass

import pytest
import torch
import torch.nn as nn

from eigentruth.core.math_engine import TruthManifold
from eigentruth.intervention.hooks import TruthProbe

# ===================================================================
# 测试用 Mock 模型
# ===================================================================

class MockTransformerLayer(nn.Module):
    """模拟单个 Transformer 层。输出 tuple (hidden_states, ...)。"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> tuple:
        h = self.linear(x)
        # 模拟 HF 风格输出: (hidden_states, attention_weights, ...)
        return (h, None)


class MockTransformerLayerPlain(nn.Module):
    """模拟直接返回 Tensor 的层。"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


@dataclass
class MockLayerOutput:
    """模拟 HF dataclass/ModelOutput 风格输出。"""

    last_hidden_state: torch.Tensor
    marker: str = "keep-me"


class MockTransformerLayerDataclass(nn.Module):
    """模拟返回 dataclass output 的层。"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> MockLayerOutput:
        return MockLayerOutput(last_hidden_state=self.linear(x))


class MockModel(nn.Module):
    """模拟 HF 模型，有 .layers 属性。"""

    def __init__(self, n_layers: int = 4, hidden_dim: int = 32):
        super().__init__()
        self.layers = nn.ModuleList([
            MockTransformerLayer(hidden_dim) for _ in range(n_layers)
        ])
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x, _ = layer(x)
        return x


def _build_manifold(hidden_dim: int = 32, n_samples: int = 10) -> TruthManifold:
    """构建一个已就绪的测试用真值流形。"""
    torch.manual_seed(0)
    m = TruthManifold()
    for _ in range(n_samples):
        m.update(torch.randn(hidden_dim))
    assert m.is_ready()
    return m


# ===================================================================
# Hook 注册与移除
# ===================================================================

class TestHookRegistration:
    """Hook 生命周期管理测试。"""

    def test_register_positive_index(self):
        """正索引注册 hook。"""
        model = MockModel(n_layers=4, hidden_dim=32)
        manifold = _build_manifold(32)
        probe = TruthProbe(manifold)
        probe.register(model, layer_idx=1)
        assert probe.is_active
        probe.remove()

    def test_register_negative_index(self):
        """负索引注册 hook（如 -1 = 最后一层）。"""
        model = MockModel(n_layers=4, hidden_dim=32)
        manifold = _build_manifold(32)
        probe = TruthProbe(manifold)
        probe.register(model, layer_idx=-1)
        assert probe.is_active
        probe.remove()

    def test_remove_deactivates(self):
        """remove() 后 hook 不再激活。"""
        model = MockModel(n_layers=4, hidden_dim=32)
        manifold = _build_manifold(32)
        probe = TruthProbe(manifold)
        probe.register(model, layer_idx=0)
        probe.remove()
        assert not probe.is_active
        assert probe._hook_handle is None

    def test_double_register_replaces(self):
        """重复注册会替换旧 hook。"""
        model = MockModel(n_layers=4, hidden_dim=32)
        manifold = _build_manifold(32)
        probe = TruthProbe(manifold)
        probe.register(model, layer_idx=0)
        probe.register(model, layer_idx=1)
        assert probe.is_active
        probe.remove()

    def test_remove_without_register(self):
        """未注册就移除不会报错。"""
        manifold = _build_manifold(32)
        probe = TruthProbe(manifold)
        probe.remove()  # 不应抛异常
        assert not probe.is_active


# ===================================================================
# Hook 功能验证
# ===================================================================

class TestHookFunctionality:
    """Hook 运行时功能测试。"""

    def test_captures_distance(self):
        """前向传播后 last_distance 被填充。"""
        model = MockModel(n_layers=4, hidden_dim=32)
        manifold = _build_manifold(32)
        probe = TruthProbe(manifold, threshold=1000.0)  # 高阈值，不触发引导
        probe.register(model, layer_idx=-1)

        # 批大小 B=3
        x = torch.randn(3, 5, 32)  # [B=3, Seq=5, D=32]
        with torch.no_grad():
            _ = model(x)

        assert probe.last_distance > 0.0
        probe.remove()

    def test_no_grad_in_hook(self):
        """Hook 操作不产生梯度泄漏。

        当 steering 触发时，hook 内部使用 clone() + no_grad()
        确保 hook 的运算不会被纳入计算图。这是正确的防御行为。
        """
        model = MockModel(n_layers=4, hidden_dim=32)
        manifold = _build_manifold(32)
        # 高阈值 — 不触发 steering → 不 clone → 梯度透传
        probe = TruthProbe(manifold, threshold=1e10, steering_lambda=0.1)
        probe.register(model, layer_idx=-1)

        x = torch.randn(1, 5, 32, requires_grad=True)
        out = model(x)
        # 不触发 steering 时，原始计算图应保持
        assert out.requires_grad
        probe.remove()

    def test_steering_modifies_output_when_above_threshold(self):
        """当距离超阈值时，输出被引导修改。仅修改超出阈值的 batch。"""
        model = MockModel(n_layers=4, hidden_dim=32)
        manifold = _build_manifold(32)

        x = torch.randn(2, 5, 32) # B=2

        # 调整 x 使得其中一个样本距离极小（不过阈值），另一个极大（过阈值）
        # 流形在原点附近
        x[0] = x[0] * 0.001  # 很近，距离低
        x[1] = x[1] * 1000.0 # 很远，距离高

        # 阈值设为 100，只会拦截 x[1]
        probe_on = TruthProbe(manifold, threshold=100.0, steering_lambda=1.0)
        probe_on.register(model, layer_idx=-1)

        with torch.no_grad():
            out_on = model(x).clone()
            probe_on.remove()

            # 再不带 probe 跑一遍
            out_off = model(x).clone()

        # 最后一个 token 的输出
        # x[0] 低于阈值，不应被修改
        assert torch.allclose(out_off[0, -1, :], out_on[0, -1, :], atol=1e-5)
        # x[1] 高于阈值，应被修改
        assert not torch.allclose(out_off[1, -1, :], out_on[1, -1, :])

    def test_no_modification_below_threshold(self):
        """当距离低于阈值时，输出不变。"""
        model = MockModel(n_layers=4, hidden_dim=32)
        manifold = _build_manifold(32)

        x = torch.randn(1, 5, 32)

        # 极高阈值 — 不触发引导
        probe = TruthProbe(manifold, threshold=1e10, steering_lambda=1.0)
        probe.register(model, layer_idx=-1)

        with torch.no_grad():
            out_a = model(x).clone()
        probe.remove()

        # 无 hook
        with torch.no_grad():
            out_b = model(x).clone()

        assert torch.allclose(out_a, out_b, atol=1e-5)


# ===================================================================
# 输出格式处理
# ===================================================================

class TestOutputFormatHandling:
    """HF 输出格式（tuple / Tensor）处理测试。"""

    def test_tuple_output_preserved(self):
        """hook 正确处理 tuple 输出，不丢失额外元素。"""
        model = MockModel(n_layers=4, hidden_dim=32)
        manifold = _build_manifold(32)
        probe = TruthProbe(manifold, threshold=1e10)
        probe.register(model, layer_idx=0)

        x = torch.randn(1, 3, 32)
        with torch.no_grad():
            # 直接调用层以检查输出格式
            layer = model.layers[0]
            out = layer(x)
            assert isinstance(out, tuple)
            assert len(out) == 2  # (hidden_states, None)
        probe.remove()

    def test_plain_tensor_output(self):
        """hook 处理纯 Tensor 输出的层。"""
        manifold = _build_manifold(32)
        probe = TruthProbe(manifold, threshold=1e10)

        # 直接在单独的层上注册
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([MockTransformerLayerPlain(32)])
            def forward(self, x):
                return self.layers[0](x)

        model = SimpleModel()
        probe.register(model, layer_idx=0)

        x = torch.randn(1, 3, 32)
        with torch.no_grad():
            out = model(x)
            assert isinstance(out, torch.Tensor)
        probe.remove()

    def test_dataclass_output_preserved(self):
        """hook 处理 dataclass/ModelOutput 风格输出且保留原类型。"""
        manifold = _build_manifold(32)
        probe = TruthProbe(manifold, threshold=1e10)

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([MockTransformerLayerDataclass(32)])

            def forward(self, x):
                return self.layers[0](x)

        model = SimpleModel()
        probe.register(model, layer_idx=0)

        x = torch.randn(1, 3, 32)
        with torch.no_grad():
            out = model(x)
            assert isinstance(out, MockLayerOutput)
            assert out.marker == "keep-me"
            assert out.last_hidden_state.shape == (1, 3, 32)
        probe.remove()


# ===================================================================
# HSE 追踪
# ===================================================================

class TestHSETracking:
    """双曲语义熵历史追踪测试。"""

    def test_hse_tracking_is_disabled_by_default(self):
        """默认路径不采集 HSE 历史。"""
        model = MockModel(n_layers=4, hidden_dim=32)
        manifold = _build_manifold(32)
        probe = TruthProbe(manifold, threshold=1e10)
        probe.register(model, layer_idx=-1)

        x = torch.randn(2, 3, 32)
        with torch.no_grad():
            _ = model(x)

        assert probe.track_hse is False
        assert probe.last_hse == 0.0
        assert len(probe._poincare_history) == 0
        probe.remove()

    def test_hse_accumulates_and_truncates(self):
        """多次前向传播后 HSE 被计算，且不超出 hse_window_size。"""
        model = MockModel(n_layers=4, hidden_dim=32)
        manifold = _build_manifold(32)
        probe = TruthProbe(manifold, threshold=1e10, hse_window_size=3, track_hse=True)
        probe.register(model, layer_idx=-1)

        for _ in range(5):
            x = torch.randn(2, 3, 32) # B=2
            with torch.no_grad():
                _ = model(x)

        assert probe.last_hse > 0.0
        # 虽然调用了 5 次，但最大长度应截断为 3
        assert len(probe._poincare_history) == 3
        assert probe._poincare_history[0].shape == (2, 32)
        probe.remove()

    def test_reset_history_clears(self):
        """reset_history 清空 HSE 历史。"""
        model = MockModel(n_layers=4, hidden_dim=32)
        manifold = _build_manifold(32)
        probe = TruthProbe(manifold, threshold=1e10, track_hse=True)
        probe.register(model, layer_idx=-1)

        x = torch.randn(1, 3, 32)
        with torch.no_grad():
            _ = model(x)
        assert len(probe._poincare_history) == 1

        probe.reset_history()
        assert len(probe._poincare_history) == 0
        assert probe.last_hse == 0.0
        probe.remove()


# ===================================================================
# Steering Vector
# ===================================================================

class TestSteeringVector:
    """引导向量计算测试。"""

    def test_direction_toward_mean(self):
        """无 false_mean 时引导方向指向质心。"""
        manifold = _build_manifold(32)
        probe = TruthProbe(manifold)

        h = torch.randn(2, 32) # [B=2, D]
        steering = probe._compute_steering_vector(h) # [B, D]

        # steering 应与 (mean - h) 同向
        expected_dir = manifold.mean - h.to(torch.float32)
        expected_dir = expected_dir / torch.norm(expected_dir, dim=-1, keepdim=True)

        cos_sim = torch.sum(steering.to(torch.float32) * expected_dir, dim=-1)
        assert torch.all(cos_sim > 0.99)  # 近乎同向

    def test_contrastive_direction(self):
        """有 contrastive_direction 时直接使用。"""
        manifold = _build_manifold(32)
        manifold.contrastive_direction = torch.ones(32) # [D]
        probe = TruthProbe(manifold)

        h = torch.randn(2, 32)
        steering = probe._compute_steering_vector(h)

        expected_dir = torch.ones(32) / torch.norm(torch.ones(32))

        cos_sim = torch.sum(steering.to(torch.float32) * expected_dir.unsqueeze(0), dim=-1)
        assert torch.all(cos_sim > 0.99)

    def test_steering_is_unit_vector(self):
        """引导向量是单位向量。"""
        manifold = _build_manifold(32)
        probe = TruthProbe(manifold)

        h = torch.randn(2, 32)
        steering = probe._compute_steering_vector(h)
        norm = torch.norm(steering, dim=-1)
        assert torch.allclose(norm, torch.ones_like(norm), atol=1e-4)


# ===================================================================
# 引导量纲 (Fix: 修正量按激活范数缩放)
# ===================================================================

class TestSteeringMagnitude:
    """注入的修正量应为激活范数的 steering_lambda 比例。

    旧实现注入 (steering_lambda × 单位向量)，范数恒为 lambda(~0.x)，相对真实
    激活范数(常为数十)仅约 1%，几乎无效。新实现按各样本激活范数缩放，使
    steering_lambda 成为"相对激活范数的移动比例"，与模型/层无关。
    """

    def test_correction_is_fraction_of_activation_norm(self):
        """修正量范数 ≈ lambda × 激活范数（而非旧版恒等于 lambda）。"""
        model = MockModel(n_layers=4, hidden_dim=32)
        manifold = _build_manifold(32)
        x = torch.randn(1, 5, 32) * 3.0

        # 基线：捕获 layer[-1] 最后一个 token 的激活范数（无引导）
        captured = {}

        def grab(module, inp, out):
            captured["h"] = out[0][:, -1, :].clone()

        gh = model.layers[-1].register_forward_hook(grab)
        with torch.no_grad():
            out_base = model(x).clone()
        gh.remove()
        act_norm = captured["h"].norm().item()

        lam = 0.1
        probe = TruthProbe(manifold, threshold=0.01, steering_lambda=lam)
        probe.register(model, layer_idx=-1)
        with torch.no_grad():
            out_on = model(x).clone()
        probe.remove()

        corr_norm = (out_on[:, -1, :] - out_base[:, -1, :]).norm().item()
        # 核心断言：修正量随激活范数缩放
        assert corr_norm == pytest.approx(lam * act_norm, rel=0.05)
        # 明确区别于旧的单位向量行为（旧版恒为 lam=0.1）
        assert corr_norm > lam

    def test_correction_scales_linearly_with_lambda(self):
        """lambda 翻倍 → 修正量范数翻倍。"""
        model = MockModel(n_layers=4, hidden_dim=32)
        manifold = _build_manifold(32)
        x = torch.randn(1, 5, 32) * 3.0

        with torch.no_grad():
            out_base = model(x).clone()

        norms = {}
        for lam in (0.1, 0.2):
            probe = TruthProbe(manifold, threshold=0.01, steering_lambda=lam)
            probe.register(model, layer_idx=-1)
            with torch.no_grad():
                out = model(x).clone()
            probe.remove()
            norms[lam] = (out[:, -1, :] - out_base[:, -1, :]).norm().item()

        assert norms[0.2] == pytest.approx(2.0 * norms[0.1], rel=0.05)


# ===================================================================
# 层定位
# ===================================================================

class TestFindLayers:
    """Transformer 层自动定位测试。"""

    def test_finds_layers_attribute(self):
        """能定位 model.layers。"""
        model = MockModel(4, 32)
        layers = TruthProbe._find_layers(model)
        assert len(layers) == 4

    def test_raises_on_no_layers(self):
        """无层列表时抛出 ValueError。"""
        model = nn.Linear(10, 10)
        with pytest.raises(ValueError):
            TruthProbe._find_layers(model)

    def test_layer_index_error_is_descriptive(self):
        """层索引越界时给出模型层数。"""
        model = MockModel(4, 32)
        manifold = _build_manifold(32)
        probe = TruthProbe(manifold)

        with pytest.raises(IndexError, match="out of range for 4 transformer layers"):
            probe.register(model, layer_idx=-10)

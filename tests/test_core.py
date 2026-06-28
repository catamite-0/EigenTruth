"""EigenTruth 包级别冒烟测试。"""

import pytest
import torch

from eigentruth import (
    AttentionSoftTargetProbeArtifact,
    CovarianceSpectrum,
    TrajectoryMonitor,
    __version__,
    cluster_assignment_entropy,
    covariance_spectrum,
    embedding_semantic_entropy,
    internal_eigenscore,
    lexical_semantic_entropy,
    soft_error_rate_targets,
    spectral_effective_rank,
    trajectory_convergence_metrics,
)


def test_version():
    assert __version__ == "0.2.0"


def test_top_level_inside_exports():
    assert callable(cluster_assignment_entropy)
    assert callable(covariance_spectrum)
    assert callable(CovarianceSpectrum)
    assert callable(embedding_semantic_entropy)
    assert callable(internal_eigenscore)
    assert callable(lexical_semantic_entropy)
    assert callable(spectral_effective_rank)
    assert callable(TrajectoryMonitor)
    assert callable(trajectory_convergence_metrics)
    assert callable(AttentionSoftTargetProbeArtifact)
    assert callable(soft_error_rate_targets)


def test_soft_error_rate_targets_from_sample_correctness():
    targets = soft_error_rate_targets([
        [1, 1, 0, 0],
        [True, True, True],
        [False, False],
    ])

    assert targets.tolist() == [0.5, 0.0, 1.0]


def test_attention_soft_target_probe_fits_soft_risk_and_roundtrips(tmp_path):
    hidden = torch.zeros((12, 4, 3), dtype=torch.float32)
    targets = torch.tensor([0.05] * 6 + [0.95] * 6, dtype=torch.float32)
    hidden[:6, 1, 0] = -3.0
    hidden[6:, 1, 0] = 3.0
    hidden[:, 0, 1] = 0.25
    hidden[:, 2, 2] = -0.25
    mask = torch.ones((12, 4), dtype=torch.bool)
    mask[:, 3] = False

    artifact = AttentionSoftTargetProbeArtifact.fit(
        hidden,
        targets,
        attention_mask=mask,
        layer_idx=4,
        steps=180,
        lr=0.08,
        seed=7,
        metadata={"source": "synthetic"},
    )
    probs = artifact.predict_proba(hidden, attention_mask=mask)
    weights = artifact.attention_weights(hidden, attention_mask=mask)

    assert artifact.layer_idx == 4
    assert artifact.hidden_dim == 3
    assert probs[6:].mean().item() > probs[:6].mean().item() + 0.7
    assert weights[:, 3].sum().item() == pytest.approx(0.0)
    assert artifact.to_dict()["metadata"]["source"] == "synthetic"
    assert artifact.to_dict()["training_summary"]["final_loss"] < artifact.to_dict()["training_summary"]["initial_loss"]

    path = tmp_path / "attention_probe.pt"
    artifact.save(path)
    loaded = AttentionSoftTargetProbeArtifact.load(path)

    assert loaded.to_dict()["layer_idx"] == 4
    assert torch.allclose(loaded.predict_proba(hidden, attention_mask=mask), probs)


def test_attention_soft_target_probe_validates_shapes():
    hidden = torch.zeros((2, 3, 4), dtype=torch.float32)

    with pytest.raises(ValueError, match="soft_targets"):
        AttentionSoftTargetProbeArtifact.fit(hidden, [0.1])
    with pytest.raises(ValueError, match="attention_mask"):
        AttentionSoftTargetProbeArtifact.fit(hidden, [0.1, 0.2], attention_mask=torch.ones((2, 2)))
    with pytest.raises(ValueError, match="same shape"):
        AttentionSoftTargetProbeArtifact(query=torch.ones(3), classifier_weight=torch.ones(4), bias=0.0)

"""EigenTruth 包级别冒烟测试。"""

import pytest
import torch

from eigentruth import (
    AttentionSoftTargetProbeArtifact,
    ClaimFactualityProbeArtifact,
    CovarianceSpectrum,
    TrajectoryMonitor,
    __version__,
    claim_factuality_diagnostics,
    cluster_assignment_entropy,
    covariance_spectrum,
    embedding_semantic_entropy,
    internal_eigenscore,
    lexical_semantic_entropy,
    pool_claim_hidden_states,
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
    assert callable(ClaimFactualityProbeArtifact)
    assert callable(claim_factuality_diagnostics)
    assert callable(pool_claim_hidden_states)
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


def test_claim_factuality_probe_fits_claim_risk_and_roundtrips(tmp_path):
    hidden = torch.zeros((16, 5, 4), dtype=torch.float32)
    targets = torch.tensor([0.0] * 8 + [1.0] * 8, dtype=torch.float32)
    token_spans = [(1, 4)] * 16
    hidden[:8, 1:4, 0] = -3.0
    hidden[8:, 1:4, 0] = 3.0
    hidden[:, 0, 0] = 5.0
    hidden[:, 4, 0] = -5.0
    hidden[:, 1:4, 1] = 0.25
    mask = torch.ones((16, 5), dtype=torch.bool)
    mask[:, 4] = False

    artifact = ClaimFactualityProbeArtifact.fit(
        hidden,
        targets,
        token_spans=token_spans,
        attention_mask=mask,
        layer_idx=7,
        steps=220,
        lr=0.08,
        seed=11,
        metadata={"source": "synthetic_claims"},
    )
    probs = artifact.predict_proba(hidden, token_spans=token_spans, attention_mask=mask)
    claims = [
        {"claim_id": f"c{idx}", "text": f"claim {idx}", "span": (idx, idx + 1)}
        for idx in range(16)
    ]
    scores = artifact.score_claims(
        claims,
        hidden,
        token_spans=token_spans,
        attention_mask=mask,
    )
    diagnostics = claim_factuality_diagnostics(scores, risk_threshold=0.5)

    assert artifact.layer_idx == 7
    assert artifact.hidden_dim == 4
    assert artifact.pooling == "mean"
    assert probs[8:].mean().item() > probs[:8].mean().item() + 0.8
    assert artifact.to_dict()["metadata"]["source"] == "synthetic_claims"
    assert artifact.to_dict()["training_summary"]["final_loss"] < artifact.to_dict()["training_summary"]["initial_loss"]
    assert scores[0].claim_id == "c0"
    assert scores[0].token_span == (1, 4)
    assert scores[0].metadata["score_name"] == "claim_factuality_risk"
    assert diagnostics["direction"] == "higher"
    assert diagnostics["high_risk_claim_count"] == 8
    assert diagnostics["max_risk_probability"] == pytest.approx(max(score.risk_probability for score in scores))

    path = tmp_path / "claim-factuality-probe.pt"
    artifact.save(path)
    loaded = ClaimFactualityProbeArtifact.load(path)

    assert loaded.to_dict()["layer_idx"] == 7
    assert torch.allclose(
        loaded.predict_proba(hidden, token_spans=token_spans, attention_mask=mask),
        probs,
    )


def test_pool_claim_hidden_states_respects_spans_masks_and_pooling():
    hidden = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    spans = [(1, 4), (0, 3)]
    mask = torch.tensor([
        [True, True, False, True],
        [False, True, True, True],
    ])

    pooled_mean = pool_claim_hidden_states(hidden, token_spans=spans, attention_mask=mask)
    pooled_first = pool_claim_hidden_states(
        hidden,
        token_spans=spans,
        attention_mask=mask,
        pooling="first_token",
    )
    pooled_last = pool_claim_hidden_states(
        hidden,
        token_spans=spans,
        attention_mask=mask,
        pooling="last_token",
    )

    assert torch.allclose(pooled_mean[0], (hidden[0, 1] + hidden[0, 3]) / 2.0)
    assert torch.allclose(pooled_mean[1], (hidden[1, 1] + hidden[1, 2]) / 2.0)
    assert torch.allclose(pooled_first[0], hidden[0, 1])
    assert torch.allclose(pooled_first[1], hidden[1, 1])
    assert torch.allclose(pooled_last[0], hidden[0, 3])
    assert torch.allclose(pooled_last[1], hidden[1, 2])


def test_claim_factuality_probe_validates_inputs():
    hidden = torch.zeros((2, 3, 4), dtype=torch.float32)

    with pytest.raises(ValueError, match="risk_targets"):
        ClaimFactualityProbeArtifact.fit(hidden, [0.0])
    with pytest.raises(ValueError, match="probabilities"):
        ClaimFactualityProbeArtifact.fit(hidden, [0.0, 2.0])
    with pytest.raises(ValueError, match="token_spans"):
        ClaimFactualityProbeArtifact.fit(hidden, [0.0, 1.0], token_spans=[(0, 4), (0, 1)])
    with pytest.raises(ValueError, match="same shape"):
        ClaimFactualityProbeArtifact(
            weight=torch.ones(3),
            bias=0.0,
            feature_mean=torch.zeros(3),
            feature_scale=torch.ones(4),
        )

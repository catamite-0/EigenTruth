import json
import math

import pytest

from eigentruth.eval import (
    ContextSensitivityReport,
    ContextSensitivityToken,
    context_logprob_delta,
    context_sensitivity_ratio,
    score_context_sensitivity,
    unsupported_context_shift,
)


def test_context_sensitivity_flags_tokens_weakened_by_evidence():
    report = score_context_sensitivity(
        (
            ContextSensitivityToken(
                token=" Mars",
                baseline_logprob=-0.4,
                context_logprob=-1.2,
                claim_id="c1",
                span_start=10,
                span_end=15,
            ),
            ContextSensitivityToken(
                token=" Paris",
                baseline_logprob=-1.1,
                context_logprob=-0.3,
                claim_id="c1",
            ),
        ),
        ratio_threshold=2.0,
        shift_threshold=0.5,
    )

    assert report.summary["token_count"] == 2
    assert report.summary["flagged_token_count"] == 1
    assert report.summary["supported_token_count"] == 1
    assert report.token_scores[0].flagged is True
    assert set(report.token_scores[0].reasons) == {
        "context_sensitivity_ratio",
        "unsupported_context_shift",
    }
    assert report.token_scores[0].unsupported_context_shift == pytest.approx(0.8)
    assert report.token_scores[0].context_sensitivity_ratio == pytest.approx(3.0)
    assert report.token_scores[1].flagged is False
    assert report.claim_summaries["c1"]["flagged_token_count"] == 1


def test_context_sensitivity_accepts_mapping_inputs_and_serializes(tmp_path):
    report = score_context_sensitivity(
        (
            {
                "token": " founded",
                "baseline_logprob": -1.0,
                "context_logprob": -1.4,
                "claim_id": "claim-7",
                "metadata": {"source": "fixture"},
            },
            {
                "token": " in",
                "no_context_logprob": -1.0,
                "evidence_logprob": -0.9,
                "claim_id": "claim-7",
            },
        ),
        ratio_threshold=2.0,
        shift_threshold=0.25,
        min_abs_delta=0.05,
        metadata={"run": "unit"},
    )
    path = tmp_path / "context-sensitivity.json"
    report.save_json(path)
    loaded = ContextSensitivityReport.load_json(path)

    assert loaded.to_dict() == report.to_dict()
    assert loaded.summary["flagged_rate"] == pytest.approx(0.5)
    assert loaded.metadata["run"] == "unit"
    assert loaded.claim_summaries["claim-7"]["reasons"] == ["unsupported_context_shift"]
    json.dumps(loaded.to_dict())


def test_context_sensitivity_support_helpers():
    assert context_logprob_delta(-0.4, -1.0) == pytest.approx(0.6)
    assert unsupported_context_shift(-1.5, -1.0) == pytest.approx(0.5)
    assert unsupported_context_shift(-0.5, -1.0) == pytest.approx(0.0)
    assert context_sensitivity_ratio(-2.0, -1.0) == pytest.approx(2.0)
    assert context_sensitivity_ratio(-0.5, -1.0) == pytest.approx(0.5)


def test_context_sensitivity_min_abs_delta_suppresses_small_drift():
    report = score_context_sensitivity(
        (
            ContextSensitivityToken(" small", baseline_logprob=-1.0, context_logprob=-1.1),
        ),
        ratio_threshold=1.01,
        shift_threshold=0.01,
        min_abs_delta=0.2,
    )

    assert report.token_scores[0].flagged is False
    assert report.summary["flagged_token_count"] == 0


def test_context_sensitivity_rejects_invalid_logprobs_and_thresholds():
    with pytest.raises(ValueError, match="finite number, not bool"):
        ContextSensitivityToken("x", baseline_logprob=False, context_logprob=-1.0)
    with pytest.raises(ValueError, match="finite"):
        ContextSensitivityToken("x", baseline_logprob=math.nan, context_logprob=-1.0)
    with pytest.raises(ValueError, match="<= 0"):
        ContextSensitivityToken("x", baseline_logprob=0.1, context_logprob=-1.0)
    with pytest.raises(ValueError, match="positive"):
        score_context_sensitivity((), ratio_threshold=0.0)
    with pytest.raises(ValueError, match="non-negative"):
        score_context_sensitivity((), shift_threshold=-0.1)

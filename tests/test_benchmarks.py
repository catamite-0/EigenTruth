"""Smoke tests for benchmark reporting helpers."""

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn


def test_eval_conformal_run_respects_lower_direction(tmp_path):
    module = importlib.import_module("benchmarks.eval_conformal")
    scores_path = tmp_path / "scores.json"
    scores_path.write_text(
        json.dumps({
            "config": {"model": "tiny", "layer": 0},
            "labels": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            "scores": {"support": [10, 11, 12, 13, 14, 15, 16, 17, 0, 1, 2, 3]},
        }),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        scores=str(scores_path),
        signal="support",
        signals=None,
        repeats=1,
        seed=0,
        json=None,
        save_calibration=None,
        save_sweep_report=None,
        save_best_calibration=None,
        best_by="auroc",
        artifact_alpha=0.10,
        direction="lower",
        model_id=None,
        model_revision=None,
        target_layer=None,
        created_at=None,
        commit_sha=None,
    )

    payload = module.run(args)
    report = payload["results"]["0.2"]["selective_report"]

    assert payload["config"]["direction"] == "lower"
    assert payload["results"]["0.2"]["threshold"] == pytest.approx(10.0)
    assert report["direction"] == "lower"
    assert report["false_alarm"] == pytest.approx(0.0)
    assert report["detection"] == pytest.approx(1.0)


def test_backfill_truthfulqa_statements_validates_labels_and_builds_oracle_fixture():
    module = importlib.import_module("benchmarks.backfill_truthfulqa_statements")
    eval_module = importlib.import_module("benchmarks.eval_truthfulqa")
    statements = (
        eval_module.Statement("Q1?", "A1.", 0),
        eval_module.Statement("Q2?", "A2.", 1),
    )
    dump = {"labels": [0, 1], "scores": {"truth_proj": [0.1, 0.9]}}

    payload = module.backfill_statement_dump(dump, statements)
    fixture = module.oracle_claim_fixture(payload["statements"], payload["labels"])

    assert payload["statements"][0]["text"] == "Q1? A1."
    assert payload["statements"][1]["is_false"] == 1
    assert fixture["fixture_type"] == "truthfulqa_oracle_label_evidence"
    assert fixture["records"][0]["retrieval_documents"] == ["Q1? A1."]
    assert fixture["records"][1]["retrieval_documents"] == []
    assert fixture["records"][1]["refutations"] == {
        "Q2? A2.": ["Oracle label marks this claim false: Q2? A2."]
    }

    with pytest.raises(ValueError, match="do not align"):
        module.backfill_statement_dump({"labels": [1, 0]}, statements)


def test_build_truthfulqa_corpus_outputs_correct_answer_documents(monkeypatch):
    module = importlib.import_module("benchmarks.build_truthfulqa_corpus")
    eval_module = importlib.import_module("benchmarks.eval_truthfulqa")

    def fake_load_truthfulqa(manifold_questions, limit):
        assert manifold_questions == 2
        assert limit == 3
        return (
            ["Warmup true one.", "Warmup true two."],
            ["Warmup false."],
            (
                eval_module.Statement("Q1?", "Correct A.", 0),
                eval_module.Statement("Q1?", "Wrong A.", 1),
                eval_module.Statement("Q2?", "Correct B.", 0),
            ),
        )

    monkeypatch.setattr(module, "load_truthfulqa", fake_load_truthfulqa)

    payload = module.build_truthfulqa_corpus(manifold_questions=2, limit=3)

    assert payload["corpus_type"] == "truthfulqa_correct_answer_evidence"
    assert payload["summary"]["n_documents"] == 4
    assert payload["summary"]["n_manifold_documents"] == 2
    assert payload["summary"]["n_eval_documents"] == 2
    assert payload["documents"][2]["text"] == "Q1? Correct A."
    assert payload["documents"][2]["metadata"]["split"] == "eval"
    assert payload["documents"][2]["metadata"]["is_false"] == 0


def test_build_evidence_fixture_uses_local_corpus_for_verifier_ensemble(tmp_path):
    builder = importlib.import_module("benchmarks.build_evidence_fixture")
    verifier = importlib.import_module("benchmarks.eval_verifier_ensemble")
    scores_path = tmp_path / "scores.json"
    corpus_path = tmp_path / "corpus.json"
    fixture_path = tmp_path / "fixture.json"
    dump = {
        "config": {"model": "synthetic", "layer": -1},
        "labels": [0, 1, 0],
        "scores": {"truth_proj": [0.1, 0.9, 0.2]},
        "statements": [
            {"question": "What is the capital of France?", "answer": "The capital of France is Paris.",
             "text": "What is the capital of France? The capital of France is Paris."},
            {"question": "What is the capital of France?", "answer": "The capital of France is Lyon.",
             "text": "What is the capital of France? The capital of France is Lyon."},
            {"question": "What color is the sky?", "answer": "The sky is blue.",
             "text": "What color is the sky? The sky is blue."},
        ],
    }
    scores_path.write_text(json.dumps(dump), encoding="utf-8")
    corpus_path.write_text(
        json.dumps({
            "documents": [
                {"text": "The capital of France is Paris.", "source": "facts:paris"},
                {"text": "The capital of France is not Lyon.", "source": "facts:lyon"},
            ],
        }),
        encoding="utf-8",
    )

    loaded_dump = builder.load_score_dump(scores_path)
    corpus = builder.load_corpus((corpus_path,))
    fixture = builder.build_evidence_fixture(
        loaded_dump,
        corpus,
        retriever_min_overlap=0.6,
        retrieval_limit=1,
        query_field="answer",
    )
    fixture_path.write_text(json.dumps(fixture), encoding="utf-8")

    payload = verifier.build_verifier_ensemble_report(
        [("synthetic", scores_path)],
        signal="truth_proj",
        claims_path=fixture_path,
        alphas=(0.2,),
        repeats=1,
        seed=0,
        verifier_min_overlap=0.65,
        retriever_min_overlap=0.6,
    )
    quality = payload["runs"][0]["verification_quality"]
    routes = payload["runs"][0]["route_summary"]
    cache_stats = payload["runs"][0]["cache_stats"]

    assert fixture["fixture_type"] == "local_retrieval_evidence"
    assert fixture["summary"]["records_with_hits"] == 2
    assert fixture["records"][0]["retrieval_documents"][0]["source"] == "facts:paris"
    assert fixture["records"][1]["retrieval_documents"][0]["source"] == "facts:lyon"
    assert fixture["records"][1]["metadata"]["retrieval"]["query_field"] == "answer"
    assert fixture["records"][1]["metadata"]["retrieval"]["query"] == "The capital of France is Lyon."
    assert quality["label_status_matrix"]["true"]["supported"] == 1
    assert quality["label_status_matrix"]["true"]["insufficient_evidence"] == 1
    assert quality["label_status_matrix"]["false"]["refuted"] == 1
    assert quality["true_supported_rate"] == pytest.approx(0.5)
    assert quality["false_refuted_rate"] == pytest.approx(1.0)
    assert routes["selected_counts"] == {"retrieval_groundedness": 2, "groundedness": 1}
    assert routes["by_route"]["retrieval_groundedness"]["statuses"]["supported"] == 1
    assert routes["by_route"]["retrieval_groundedness"]["statuses"]["refuted"] == 1
    assert routes["by_route"]["groundedness"]["statuses"]["insufficient_evidence"] == 1
    assert cache_stats["groundedness_verifiers"]["requests"] >= 3
    assert cache_stats["retrievers"]["requests"] == 2
    assert cache_stats["total"]["requests"] >= cache_stats["retrievers"]["requests"]


def test_build_domain_state_fixture_feeds_structured_state_verifier(tmp_path):
    builder = importlib.import_module("benchmarks.build_domain_state_fixture")
    verifier = importlib.import_module("benchmarks.eval_verifier_ensemble")
    scores_path = tmp_path / "domain_scores.json"
    claims_path = tmp_path / "domain_claims.json"
    state_path = tmp_path / "domain_state.json"

    payload = builder.run(SimpleNamespace(
        scores_output=str(scores_path),
        claims_output=str(claims_path),
        state_output=str(state_path),
        n_records=8,
        signal="truth_proj",
    ))

    assert payload["claims"]["fixture_type"] == "order_fulfillment_state_claims"
    assert payload["claims"]["summary"]["n_true"] == 4
    assert payload["claims"]["summary"]["n_false"] == 4
    assert scores_path.exists()
    assert claims_path.exists()
    assert state_path.exists()

    report = verifier.build_verifier_ensemble_report(
        [("orders", scores_path)],
        signal="truth_proj",
        claims_path=claims_path,
        state_path=state_path,
        alphas=(0.2,),
        repeats=1,
        seed=0,
    )
    run = report["runs"][0]
    quality = run["verification_quality"]
    routes = run["route_summary"]
    route_quality = run["route_quality"]["structured_state"]
    route_impact = run["alphas"]["0.2"]["route_control_impact"]["structured_state"]

    assert run["state_verifier"]["enabled"] is True
    assert run["state_verifier"]["decided_records"] == 8
    assert routes["selected_counts"] == {"structured_state": 8}
    assert route_quality["selected"] == 8
    assert route_quality["true_supported_rate"] == pytest.approx(1.0)
    assert route_quality["false_refuted_rate"] == pytest.approx(1.0)
    assert route_quality["false_supported_rate"] == pytest.approx(0.0)
    assert route_impact["verified"]["false_alarm"] == pytest.approx(0.0)
    assert route_impact["verified"]["detection"] == pytest.approx(1.0)
    assert quality["true_supported_rate"] == pytest.approx(1.0)
    assert quality["false_refuted_rate"] == pytest.approx(1.0)
    assert quality["false_supported_rate"] == pytest.approx(0.0)
    assert run["alphas"]["0.2"]["verified"]["false_alarm"] == pytest.approx(0.0)
    assert run["alphas"]["0.2"]["verified"]["detection"] == pytest.approx(1.0)


def test_build_transition_fixture_feeds_state_transition_verifier(tmp_path):
    builder = importlib.import_module("benchmarks.build_transition_fixture")
    verifier = importlib.import_module("benchmarks.eval_verifier_ensemble")
    scores_path = tmp_path / "transition_scores.json"
    claims_path = tmp_path / "transition_claims.json"
    state_path = tmp_path / "transition_state.json"

    payload = builder.run(SimpleNamespace(
        scores_output=str(scores_path),
        claims_output=str(claims_path),
        state_output=str(state_path),
        n_records=8,
        signal="truth_proj",
    ))

    assert payload["claims"]["fixture_type"] == "order_transition_state_claims"
    assert payload["claims"]["summary"]["n_true"] == 4
    assert payload["claims"]["summary"]["n_false"] == 4
    assert scores_path.exists()
    assert claims_path.exists()
    assert state_path.exists()

    report = verifier.build_verifier_ensemble_report(
        [("transitions", scores_path)],
        signal="truth_proj",
        claims_path=claims_path,
        state_path=state_path,
        alphas=(0.2,),
        repeats=1,
        seed=0,
    )
    run = report["runs"][0]
    quality = run["verification_quality"]
    routes = run["route_summary"]
    route_quality = run["route_quality"]["state_transition"]
    route_impact = run["alphas"]["0.2"]["route_control_impact"]["state_transition"]

    assert report["transition_verifier"]["enabled"] is True
    assert run["transition_verifier"]["enabled"] is True
    assert run["transition_verifier"]["decided_records"] == 8
    assert run["cache_stats"]["transition_verifier"]["requests"] == 8
    assert routes["selected_counts"] == {"state_transition": 8}
    assert routes["by_route"]["state_transition"]["rates"]["supported"] == pytest.approx(0.5)
    assert routes["by_route"]["state_transition"]["rates"]["refuted"] == pytest.approx(0.5)
    assert route_quality["selected"] == 8
    assert route_quality["decision_accuracy"] == pytest.approx(1.0)
    assert route_quality["false_refuted_rate"] == pytest.approx(1.0)
    assert route_quality["false_supported_rate"] == pytest.approx(0.0)
    assert route_impact["n_selected"] == 8
    assert route_impact["verified"]["false_alarm"] == pytest.approx(0.0)
    assert route_impact["verified"]["detection"] == pytest.approx(1.0)
    assert quality["true_supported_rate"] == pytest.approx(1.0)
    assert quality["false_refuted_rate"] == pytest.approx(1.0)
    assert quality["false_supported_rate"] == pytest.approx(0.0)
    assert run["alphas"]["0.2"]["verified"]["false_alarm"] == pytest.approx(0.0)
    assert run["alphas"]["0.2"]["verified"]["detection"] == pytest.approx(1.0)


def test_eval_verifier_ensemble_uses_structured_qa_corpus(tmp_path):
    verifier = importlib.import_module("benchmarks.eval_verifier_ensemble")
    scores_path = tmp_path / "scores.json"
    qa_path = tmp_path / "qa.json"
    dump = {
        "config": {"model": "synthetic", "layer": -1},
        "labels": [0, 0, 0, 0, 1, 1],
        "scores": {"truth_proj": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]},
        "statements": [
            {"question": "Q1?", "answer": "A1", "text": "Q1? A1"},
            {"question": "Q2?", "answer": "A2", "text": "Q2? A2"},
            {"question": "Q3?", "answer": "A3", "text": "Q3? A3"},
            {"question": "Q4?", "answer": "A4", "text": "Q4? A4"},
            {"question": "Q1?", "answer": "Wrong A1", "text": "Q1? Wrong A1"},
            {"question": "Q2?", "answer": "Wrong A2", "text": "Q2? Wrong A2"},
        ],
    }
    scores_path.write_text(json.dumps(dump), encoding="utf-8")
    qa_path.write_text(
        json.dumps({
            "documents": [
                {"question": "Q1?", "answer": "A1", "source": "qa:q1"},
                {"question": "Q2?", "answer": "A2", "source": "qa:q2"},
                {"question": "Q3?", "answer": "A3", "source": "qa:q3"},
                {"question": "Q4?", "answer": "A4", "source": "qa:q4"},
            ],
        }),
        encoding="utf-8",
    )

    payload = verifier.build_verifier_ensemble_report(
        [("synthetic", scores_path)],
        signal="truth_proj",
        qa_corpus_path=qa_path,
        alphas=(0.2,),
        repeats=1,
        seed=0,
    )
    run = payload["runs"][0]
    quality = run["verification_quality"]
    alpha = run["alphas"]["0.2"]
    routes = run["route_summary"]
    route_quality = run["route_quality"]["structured_qa"]
    route_impact = alpha["route_control_impact"]["structured_qa"]

    assert payload["qa_verifier"]["enabled"] is True
    assert run["qa"]["decided_records"] == 6
    assert run["cache_stats"]["qa_verifier"]["requests"] == 6
    assert routes["selected_counts"] == {"structured_qa": 6}
    assert routes["by_route"]["structured_qa"]["rates"]["supported"] == pytest.approx(4 / 6)
    assert routes["by_route"]["structured_qa"]["rates"]["refuted"] == pytest.approx(2 / 6)
    assert route_quality["selected"] == 6
    assert route_quality["true_supported_rate"] == pytest.approx(1.0)
    assert route_quality["false_refuted_rate"] == pytest.approx(1.0)
    assert route_quality["false_supported_rate"] == pytest.approx(0.0)
    assert route_impact["verified"]["false_alarm"] == pytest.approx(0.0)
    assert route_impact["verified"]["detection"] == pytest.approx(1.0)
    assert quality["true_supported_rate"] == pytest.approx(1.0)
    assert quality["false_refuted_rate"] == pytest.approx(1.0)
    assert quality["false_supported_rate"] == pytest.approx(0.0)
    assert quality["decision_accuracy"] == pytest.approx(1.0)
    assert alpha["verified"]["false_alarm"] == pytest.approx(0.0)
    assert alpha["verified"]["detection"] == pytest.approx(1.0)


def test_eval_verifier_ensemble_uses_structured_state_checks(tmp_path):
    verifier = importlib.import_module("benchmarks.eval_verifier_ensemble")
    scores_path = tmp_path / "scores.json"
    fixture_path = tmp_path / "claims.json"
    state_path = tmp_path / "state.json"
    scores_path.write_text(
        json.dumps({
            "config": {"model": "synthetic", "layer": -1},
            "labels": [0, 0, 1, 1, 0],
            "scores": {"truth_proj": [0.1, 0.2, 0.3, 0.4, 0.5]},
        }),
        encoding="utf-8",
    )
    state_path.write_text(
        json.dumps({
            "state": {
                "inventory": {"sku_123": {"available": 12}},
                "account": {"status": "suspended"},
                "limits": {"daily": 30},
            }
        }),
        encoding="utf-8",
    )
    fixture_path.write_text(
        json.dumps({
            "state_checks": {
                "c3": {"path": "account.status", "operator": "eq", "value": "active"}
            },
            "records": [
                {
                    "claim": "SKU 123 has enough available inventory.",
                    "claim_id": "c1",
                    "claim_metadata": {
                        "state_check": {
                            "path": "inventory.sku_123.available",
                            "operator": ">=",
                            "value": 10,
                        }
                    },
                },
                {
                    "claim": "Daily limit is at least 40.",
                    "claim_id": "c2",
                    "state": {"limits": {"daily": 50}},
                    "state_check": {"path": "limits.daily", "operator": ">=", "value": 40},
                },
                {
                    "claim": "Account is active.",
                    "claim_id": "c3",
                },
                {
                    "claim": "Daily limit is at least 100.",
                    "claim_id": "c4",
                    "state": {"limits": {"daily": 50}},
                    "claim_metadata": {
                        "state_check": {"path": "limits.daily", "operator": ">=", "value": 100}
                    },
                },
                {
                    "claim": "Fallback supported.",
                    "claim_id": "c5",
                    "initial_evidence": ["Fallback supported."],
                },
            ],
        }),
        encoding="utf-8",
    )

    payload = verifier.build_verifier_ensemble_report(
        [("synthetic", scores_path)],
        signal="truth_proj",
        claims_path=fixture_path,
        state_path=state_path,
        alphas=(0.2,),
        repeats=1,
        seed=0,
    )
    run = payload["runs"][0]
    quality = run["verification_quality"]
    alpha = run["alphas"]["0.2"]
    routes = run["route_summary"]
    route_quality = run["route_quality"]["structured_state"]
    route_impact = alpha["route_control_impact"]["structured_state"]

    assert payload["state_verifier"]["enabled"] is True
    assert run["state_verifier"]["enabled"] is True
    assert run["state_verifier"]["decided_records"] == 4
    assert run["cache_stats"]["state_verifier"]["requests"] == 4
    assert run["verification_status_counts"]["supported"] == 3
    assert run["verification_status_counts"]["refuted"] == 2
    assert routes["selected_counts"] == {"structured_state": 4, "groundedness": 1}
    assert routes["by_route"]["structured_state"]["rates"]["supported"] == pytest.approx(0.5)
    assert routes["by_route"]["structured_state"]["rates"]["refuted"] == pytest.approx(0.5)
    assert routes["by_route"]["groundedness"]["statuses"]["supported"] == 1
    assert route_quality["selected"] == 4
    assert route_quality["decision_accuracy"] == pytest.approx(1.0)
    assert route_quality["false_refuted_rate"] == pytest.approx(1.0)
    assert route_quality["false_supported_rate"] == pytest.approx(0.0)
    assert route_impact["verified"]["false_alarm"] == pytest.approx(0.0)
    assert route_impact["verified"]["detection"] == pytest.approx(1.0)
    assert quality["true_supported_rate"] == pytest.approx(1.0)
    assert quality["false_refuted_rate"] == pytest.approx(1.0)
    assert quality["decision_accuracy"] == pytest.approx(1.0)
    assert alpha["verified"]["false_alarm"] == pytest.approx(0.0)
    assert alpha["verified"]["detection"] == pytest.approx(1.0)


def test_compare_transfer_builds_layer_filtered_summary(tmp_path):
    module = importlib.import_module("benchmarks.compare_transfer")
    report_path = tmp_path / "sweep.json"
    report_path.write_text(
        json.dumps({
            "best": {"score_name": "truth_proj", "layer": -8, "auroc": 0.75},
            "layers": [
                {
                    "layer": -10,
                    "scores": [
                        {"score_name": "truth_proj", "layer": -10, "auroc": 0.62,
                         "detection": 0.2, "false_alarm": 0.1, "n_true": 10, "n_false": 12},
                        {"score_name": "maha_last", "layer": -10, "auroc": 0.55},
                    ],
                },
                {
                    "layer": -8,
                    "scores": [
                        {"score_name": "truth_proj", "layer": -8, "auroc": 0.75,
                         "detection": 0.3, "false_alarm": 0.1, "n_true": 10, "n_false": 12},
                    ],
                },
            ],
        }),
        encoding="utf-8",
    )

    payload = module.build_transfer_report(
        [("synthetic", report_path)],
        score_name="truth_proj",
        layers={-10, -8},
        notes=["unit-test"],
    )

    assert payload["best_overall"] == {"name": "synthetic", "layer": -8, "auroc": 0.75}
    assert payload["layer_filter"] == [-10, -8]
    assert payload["runs"][0]["summary"]["mean_auroc"] == pytest.approx(0.685)
    assert payload["runs"][0]["summary"]["layers_above_0_6"] == 2
    assert payload["notes"] == ["unit-test"]


def test_compare_profiles_builds_baseline_deltas_from_profile_payloads(tmp_path):
    module = importlib.import_module("benchmarks.compare_profiles")
    baseline_path = tmp_path / "baseline.json"
    cached_path = tmp_path / "cached.json"
    baseline_path.write_text(
        json.dumps({
            "profile": {
                "total_seconds": 100.0,
                "phases": {
                    "load_model": 10.0,
                    "build_layer_stats": 60.0,
                    "forced_answer_forward": 20.0,
                    "score_postprocess": 5.0,
                },
                "summary": {
                    "bottleneck": "build_layer_stats",
                    "top_phases": [{"name": "build_layer_stats", "seconds": 60.0, "share": 0.6}],
                    "groups": {
                        "model_forward": {"seconds": 80.0, "share": 0.8},
                        "postprocess": {"seconds": 5.0, "share": 0.05},
                    },
                    "throughput": {"forced_answer_records_per_second": 5.0},
                },
            },
        }),
        encoding="utf-8",
    )
    cached_path.write_text(
        json.dumps({
            "total_seconds": 50.0,
            "phases": {
                "load_model": 10.0,
                "load_layer_stats_cache": 1.0,
                "forced_answer_forward": 20.0,
                "score_postprocess": 5.0,
            },
            "summary": {
                "bottleneck": "forced_answer_forward",
                "top_phases": [{"name": "forced_answer_forward", "seconds": 20.0, "share": 0.4}],
                "groups": {
                    "model_forward": {"seconds": 20.0, "share": 0.4},
                    "cache_io": {"seconds": 1.0, "share": 0.02},
                    "postprocess": {"seconds": 5.0, "share": 0.1},
                },
                "throughput": {"forced_answer_records_per_second": 10.0},
            },
        }),
        encoding="utf-8",
    )

    payload = module.build_profile_comparison([
        ("baseline", baseline_path),
        ("cached", cached_path),
    ])
    cached = payload["runs"][1]

    assert payload["baseline"] == "baseline"
    assert payload["fastest"]["name"] == "cached"
    assert payload["fastest"]["speedup_vs_baseline"] == pytest.approx(2.0)
    assert cached["total_delta"]["delta_seconds"] == pytest.approx(-50.0)
    assert cached["group_deltas"]["model_forward"]["delta_seconds"] == pytest.approx(-60.0)
    assert cached["phase_deltas"]["build_layer_stats"]["seconds"] == pytest.approx(0.0)
    assert cached["phase_deltas"]["build_layer_stats"]["baseline_seconds"] == pytest.approx(60.0)
    assert cached["throughput_deltas"]["forced_answer_records_per_second"]["ratio_to_baseline"] == pytest.approx(2.0)


def test_compare_profiles_accepts_legacy_profile_without_summary(tmp_path):
    module = importlib.import_module("benchmarks.compare_profiles")
    profile_path = tmp_path / "legacy.json"
    profile_path.write_text(
        json.dumps({
            "total_seconds": 12.0,
            "phases": {"load_model": 3.0, "forced_answer_forward": 6.0},
        }),
        encoding="utf-8",
    )

    payload = module.build_profile_comparison([("legacy", profile_path)])
    run = payload["runs"][0]

    assert run["bottleneck"] == "forced_answer_forward"
    assert run["top_phases"][0]["name"] == "forced_answer_forward"
    assert run["total_delta"]["ratio_to_baseline"] == pytest.approx(1.0)


def test_compare_verifier_routes_builds_leaderboard_and_aggregates(tmp_path):
    module = importlib.import_module("benchmarks.compare_verifier_routes")
    report_a = tmp_path / "report_a.json"
    report_b = tmp_path / "report_b.json"
    report_a.write_text(
        json.dumps({
            "runs": [
                {
                    "name": "qwen",
                    "route_quality": {
                        "structured_qa": {
                            "selected": 10,
                            "selection_rate": 0.5,
                            "n_true": 6,
                            "n_false": 4,
                            "label_status_matrix": {
                                "true": {"supported": 6, "refuted": 0, "insufficient_evidence": 0},
                                "false": {"supported": 1, "refuted": 3, "insufficient_evidence": 0},
                            },
                            "true_supported_rate": 1.0,
                            "true_refuted_rate": 0.0,
                            "false_refuted_rate": 0.75,
                            "false_supported_rate": 0.25,
                            "insufficient_evidence_rate": 0.0,
                            "decision_accuracy": 0.9,
                            "decision_error_rate": 0.1,
                        },
                        "structured_state": {
                            "selected": 8,
                            "selection_rate": 0.4,
                            "n_true": 4,
                            "n_false": 4,
                            "label_status_matrix": {
                                "true": {"supported": 4, "refuted": 0, "insufficient_evidence": 0},
                                "false": {"supported": 0, "refuted": 4, "insufficient_evidence": 0},
                            },
                            "true_supported_rate": 1.0,
                            "true_refuted_rate": 0.0,
                            "false_refuted_rate": 1.0,
                            "false_supported_rate": 0.0,
                            "insufficient_evidence_rate": 0.0,
                            "decision_accuracy": 1.0,
                            "decision_error_rate": 0.0,
                        },
                    },
                    "alphas": {
                        "0.20": {
                            "route_control_impact": {
                                "structured_qa": {
                                    "internal": {"false_alarm": 0.2, "detection": 0.5},
                                    "verified": {"false_alarm": 0.0, "detection": 0.75},
                                    "delta": {
                                        "false_alarm": -0.2,
                                        "detection": 0.25,
                                        "suppressed_false_alarm_rate": 0.2,
                                        "rescued_detection_rate": 0.25,
                                    },
                                },
                                "structured_state": {
                                    "internal": {"false_alarm": 0.25, "detection": 0.25},
                                    "verified": {"false_alarm": 0.0, "detection": 1.0},
                                    "delta": {
                                        "false_alarm": -0.25,
                                        "detection": 0.75,
                                        "suppressed_false_alarm_rate": 0.25,
                                        "rescued_detection_rate": 0.75,
                                    },
                                },
                            }
                        }
                    },
                }
            ]
        }),
        encoding="utf-8",
    )
    report_b.write_text(
        json.dumps({
            "runs": [
                {
                    "name": "smol",
                    "route_quality": {
                        "structured_qa": {
                            "selected": 5,
                            "selection_rate": 0.25,
                            "n_true": 3,
                            "n_false": 2,
                            "label_status_matrix": {
                                "true": {"supported": 3, "refuted": 0, "insufficient_evidence": 0},
                                "false": {"supported": 0, "refuted": 2, "insufficient_evidence": 0},
                            },
                            "true_supported_rate": 1.0,
                            "true_refuted_rate": 0.0,
                            "false_refuted_rate": 1.0,
                            "false_supported_rate": 0.0,
                            "insufficient_evidence_rate": 0.0,
                            "decision_accuracy": 1.0,
                            "decision_error_rate": 0.0,
                        },
                    },
                    "alphas": {
                        "0.2": {
                            "route_control_impact": {
                                "structured_qa": {
                                    "internal": {"false_alarm": 0.2, "detection": 0.5},
                                    "verified": {"false_alarm": 0.0, "detection": 1.0},
                                    "delta": {
                                        "false_alarm": -0.2,
                                        "detection": 0.5,
                                        "suppressed_false_alarm_rate": 0.2,
                                        "rescued_detection_rate": 0.5,
                                    },
                                }
                            }
                        }
                    },
                }
            ]
        }),
        encoding="utf-8",
    )

    payload = module.build_route_comparison_report(
        [("a", report_a), ("b", report_b)],
        alpha=0.2,
        min_selected=1,
        notes=["unit-test"],
    )

    assert payload["notes"] == ["unit-test"]
    assert payload["n_route_entries"] == 3
    assert payload["leaderboard"][0]["decision_accuracy"] == pytest.approx(1.0)
    assert payload["leaderboard"][0]["false_supported_rate"] == pytest.approx(0.0)
    assert payload["by_route"]["structured_qa"]["selected"] == 15
    assert payload["by_route"]["structured_qa"]["false_refuted_rate"] == pytest.approx(5 / 6)
    assert payload["by_route"]["structured_qa"]["false_supported_rate"] == pytest.approx(1 / 6)
    assert payload["by_route"]["structured_qa"]["verified_detection"] == pytest.approx(5 / 6)
    assert payload["by_route"]["structured_state"]["rescued_detection_rate"] == pytest.approx(0.75)


def test_compare_profiles_builds_regression_gate_report(tmp_path):
    module = importlib.import_module("benchmarks.compare_profiles")
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_path.write_text(
        json.dumps({
            "total_seconds": 100.0,
            "phases": {"forced_answer_forward": 80.0},
            "summary": {
                "bottleneck": "forced_answer_forward",
                "groups": {},
                "throughput": {"forced_answer_records_per_second": 10.0},
            },
        }),
        encoding="utf-8",
    )
    candidate_path.write_text(
        json.dumps({
            "total_seconds": 112.0,
            "phases": {"forced_answer_forward": 92.0},
            "summary": {
                "bottleneck": "forced_answer_forward",
                "groups": {},
                "throughput": {"forced_answer_records_per_second": 8.0},
            },
        }),
        encoding="utf-8",
    )

    payload = module.build_profile_comparison(
        [("baseline", baseline_path), ("candidate", candidate_path)],
        max_total_ratio=1.10,
        max_phase_ratios={"forced_answer_forward": 1.10},
        min_throughput_ratios={"forced_answer_records_per_second": 0.90},
    )
    gate = payload["regression_gate"]

    assert gate["passed"] is False
    assert gate["checked_runs"] == ["candidate"]
    assert {failure["metric"] for failure in gate["failures"]} == {
        "total_seconds",
        "phase:forced_answer_forward",
        "throughput:forced_answer_records_per_second",
    }


def test_compare_profiles_supports_run_specific_total_ratio_gates(tmp_path):
    module = importlib.import_module("benchmarks.compare_profiles")
    baseline_path = tmp_path / "baseline.json"
    cached_path = tmp_path / "cached.json"
    cache_only_path = tmp_path / "cache_only.json"
    baseline_path.write_text(
        json.dumps({
            "total_seconds": 100.0,
            "phases": {"forced_answer_forward": 80.0},
        }),
        encoding="utf-8",
    )
    cached_path.write_text(
        json.dumps({
            "total_seconds": 98.0,
            "phases": {"read_eval_reps_cache_batch": 5.0},
        }),
        encoding="utf-8",
    )
    cache_only_path.write_text(
        json.dumps({
            "total_seconds": 30.0,
            "phases": {"read_eval_reps_cache_batch": 20.0},
        }),
        encoding="utf-8",
    )

    payload = module.build_profile_comparison(
        [
            ("baseline", baseline_path),
            ("cached", cached_path),
            ("cache_only", cache_only_path),
        ],
        baseline="baseline",
        max_total_ratio=1.05,
        max_run_total_ratios={"cache_only": 0.25},
    )
    gate = payload["regression_gate"]

    assert gate["passed"] is False
    assert gate["config"]["max_total_ratio"] == pytest.approx(1.05)
    assert gate["config"]["max_run_total_ratios"] == {"cache_only": 0.25}
    assert len(gate["failures"]) == 1
    assert gate["failures"][0]["run"] == "cache_only"
    assert gate["failures"][0]["metric"] == "total_seconds"
    assert gate["failures"][0]["limit"] == pytest.approx(0.25)
    assert gate["failures"][0]["value"] == pytest.approx(0.30)


def test_compare_profiles_cli_exits_nonzero_on_regression_gate_failure(tmp_path, monkeypatch):
    module = importlib.import_module("benchmarks.compare_profiles")
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    report_path = tmp_path / "comparison.json"
    baseline_path.write_text(
        json.dumps({"total_seconds": 10.0, "phases": {"score_postprocess": 5.0}}),
        encoding="utf-8",
    )
    candidate_path.write_text(
        json.dumps({"total_seconds": 13.0, "phases": {"score_postprocess": 5.0}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", [
        "compare_profiles.py",
        "--profile",
        f"baseline={baseline_path}",
        "--profile",
        f"candidate={candidate_path}",
        "--baseline",
        "baseline",
        "--max-total-ratio",
        "1.10",
        "--max-run-total-ratio",
        "candidate=1.20",
        "--json",
        str(report_path),
    ])

    with pytest.raises(SystemExit) as exc_info:
        module.main()

    assert exc_info.value.code == 1
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["regression_gate"]["passed"] is False
    assert payload["regression_gate"]["failures"][0]["metric"] == "total_seconds"
    assert payload["regression_gate"]["failures"][0]["limit"] == pytest.approx(1.20)


def test_profile_gate_smoke_writes_pass_and_expected_failure_reports(tmp_path):
    module = importlib.import_module("benchmarks.profile_gate_smoke")

    payload = module.build_profile_gate_smoke(tmp_path)
    pass_report = payload["pass_report"]
    failure_report = payload["expected_failure_report"]

    assert (tmp_path / "profile_baseline.json").exists()
    assert (tmp_path / "profile_candidate.json").exists()
    assert (tmp_path / "profile_regression.json").exists()
    assert (tmp_path / "profile_gate_pass_report.json").exists()
    assert (tmp_path / "profile_gate_expected_failure_report.json").exists()
    assert pass_report["regression_gate"]["passed"] is True
    assert failure_report["regression_gate"]["passed"] is False
    assert {failure["metric"] for failure in failure_report["regression_gate"]["failures"]} == {
        "total_seconds",
        "phase:forced_answer_forward",
        "throughput:forced_answer_records_per_second",
    }


def test_cache_profile_smoke_writes_pass_and_expected_failure_reports(tmp_path):
    module = importlib.import_module("benchmarks.cache_profile_smoke")

    payload = module.build_cache_profile_smoke(tmp_path)
    pass_report = payload["pass_report"]
    failure_report = payload["expected_failure_report"]

    assert (tmp_path / "profile_uncached.json").exists()
    assert (tmp_path / "profile_cached.json").exists()
    assert (tmp_path / "profile_cache_only.json").exists()
    assert (tmp_path / "profile_cache_only_regression.json").exists()
    assert (tmp_path / "cache_profile_gate_pass_report.json").exists()
    assert (tmp_path / "cache_profile_gate_expected_failure_report.json").exists()
    assert pass_report["regression_gate"]["passed"] is True
    assert pass_report["regression_gate"]["config"]["max_run_total_ratios"] == {
        "cached": pytest.approx(0.75),
        "cache_only": pytest.approx(0.20),
        "cache_only_regression": pytest.approx(0.20),
    }
    assert failure_report["regression_gate"]["passed"] is False
    assert failure_report["regression_gate"]["failures"][0]["run"] == "cache_only_regression"
    assert failure_report["regression_gate"]["failures"][0]["metric"] == "total_seconds"


def test_run_cache_profile_triplet_builds_dry_run_commands(tmp_path):
    module = importlib.import_module("benchmarks.run_cache_profile_triplet")
    config = module.CacheProfileTripletConfig(
        output_dir=tmp_path,
        model="tiny-local",
        layer=-2,
        batch_size=2,
        max_length=32,
        eval_reps_cache_shard_size=3,
        python_executable="/python",
    )

    payload = module.run_triplet(config, clean=True, dry_run=True)
    commands = payload["commands"]

    assert payload["dry_run"] is True
    assert Path(payload["command_log"]).exists()
    assert commands["uncached"][0] == "/python"
    assert "--offline" in commands["uncached"]
    assert commands["uncached"][commands["uncached"].index("--dtype") + 1] == "float32"
    assert commands["uncached"][commands["uncached"].index("--hidden-state-capture") + 1] == "outputs"
    assert commands["uncached"].count("--refresh-layer-stats-cache") == 1
    assert commands["uncached"].count("--refresh-eval-reps-cache") == 1
    assert commands["uncached"][commands["uncached"].index("--eval-reps-cache-shard-size") + 1] == "3"
    assert "--cache-only" not in commands["cached"]
    assert "--refresh-eval-reps-cache" not in commands["cached"]
    assert "--cache-only" in commands["cache_only"]
    assert "--statement-encoding-cache" not in commands["cache_only"]


def test_run_cache_profile_triplet_builds_real_truthfulqa_commands(tmp_path):
    module = importlib.import_module("benchmarks.run_cache_profile_triplet")
    config = module.CacheProfileTripletConfig(
        output_dir=tmp_path,
        model="Qwen/Qwen2.5-0.5B-Instruct",
        dtype="bfloat16",
        layer=-12,
        limit=24,
        manifold_questions=12,
        batch_size=2,
        hidden_state_capture="hooks",
        python_executable="/python",
        offline=False,
    )

    payload = module.run_triplet(config, clean=True, dry_run=True)
    uncached = payload["commands"]["uncached"]
    cache_only = payload["commands"]["cache_only"]

    assert "--offline" not in uncached
    assert uncached[uncached.index("--model") + 1] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert uncached[uncached.index("--dtype") + 1] == "bfloat16"
    assert uncached[uncached.index("--limit") + 1] == "24"
    assert uncached[uncached.index("--manifold-questions") + 1] == "12"
    assert uncached[uncached.index("--hidden-state-capture") + 1] == "hooks"
    assert cache_only[cache_only.index("--limit") + 1] == "24"


def test_run_cache_profile_triplet_writes_comparison_report(tmp_path, monkeypatch):
    module = importlib.import_module("benchmarks.run_cache_profile_triplet")
    config = module.CacheProfileTripletConfig(
        output_dir=tmp_path,
        model="tiny-local",
        python_executable="/python",
        cached_max_total_ratio=0.80,
        cache_only_max_total_ratio=0.30,
    )
    totals = {
        "profile-uncached.json": 100.0,
        "profile-cached.json": 70.0,
        "profile-cache_only.json": 20.0,
    }
    calls = []

    def fake_run(command, *, cwd, check):
        calls.append({"command": command, "cwd": cwd, "check": check})
        profile_path = Path(command[command.index("--profile-json") + 1])
        result_path = Path(command[command.index("--json") + 1])
        total = totals[profile_path.name]
        profile_path.write_text(
            json.dumps({
                "total_seconds": total,
                "phases": {"forced_answer_forward": total / 2},
                "summary": {
                    "bottleneck": "forced_answer_forward",
                    "groups": {},
                    "throughput": {"end_to_end_eval_records_per_second": 10.0},
                },
            }),
            encoding="utf-8",
        )
        result_path.write_text(json.dumps({"profile": {"total_seconds": total, "phases": {}}}), encoding="utf-8")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    payload = module.run_triplet(config, clean=True, dry_run=False)
    report = json.loads(Path(payload["comparison_report"]).read_text(encoding="utf-8"))

    assert [call["check"] for call in calls] == [True, True, True]
    assert payload["dry_run"] is False
    assert payload["regression_gate"]["passed"] is True
    assert report["baseline"] == "uncached"
    assert report["regression_gate"]["config"]["max_run_total_ratios"] == {
        "cached": pytest.approx(0.80),
        "cache_only": pytest.approx(0.30),
    }
    assert report["fastest"]["name"] == "cache_only"


def test_run_cache_profile_triplet_cli_can_fail_on_regression(tmp_path, monkeypatch):
    module = importlib.import_module("benchmarks.run_cache_profile_triplet")

    def fake_run_triplet(config, *, clean, dry_run):
        return {
            "dry_run": dry_run,
            "output_dir": str(config.output_dir),
            "regression_gate": {"passed": False},
        }

    monkeypatch.setattr(module, "run_triplet", fake_run_triplet)

    with pytest.raises(SystemExit) as exc_info:
        module.main([
            "--output-dir",
            str(tmp_path),
            "--fail-on-regression",
        ])

    assert exc_info.value.code == 1


def test_eval_calibration_transfer_builds_threshold_transfer_matrix(tmp_path):
    module = importlib.import_module("benchmarks.eval_calibration_transfer")
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text(
        json.dumps({
            "model_id": "source-model",
            "target_layer": -2,
            "scores": [
                {
                    "name": "truth_proj",
                    "threshold": 5.0,
                    "conformal_alpha": 0.1,
                    "direction": "higher",
                },
            ],
            "eigentruth_version": "0.1.0",
        }),
        encoding="utf-8",
    )
    source_scores_path = tmp_path / "source-scores.json"
    source_scores_path.write_text(
        json.dumps({
            "config": {"model": "source-model", "layer": -4},
            "labels": [0, 0, 0, 0, 1, 1],
            "scores": {"truth_proj": [0.0, 0.0, 0.0, 0.0, 9.0, 10.0]},
            "sweep_scores": {
                "-2": {"truth_proj": [0.0, 1.0, 2.0, 3.0, 8.0, 9.0]},
            },
        }),
        encoding="utf-8",
    )
    shifted_scores_path = tmp_path / "shifted-scores.json"
    shifted_scores_path.write_text(
        json.dumps({
            "config": {"model": "shifted-model", "layer": -4},
            "labels": [0, 0, 0, 0, 1, 1],
            "scores": {"truth_proj": [0.0, 0.0, 0.0, 0.0, 9.0, 10.0]},
            "sweep_scores": {
                "-2": {"truth_proj": [6.0, 7.0, 8.0, 9.0, 10.0, 11.0]},
            },
        }),
        encoding="utf-8",
    )

    payload = module.build_calibration_transfer_report(
        [("source", artifact_path)],
        [("source", source_scores_path), ("shifted", shifted_scores_path)],
        tolerance=0.03,
        notes=["unit-test"],
    )

    self_result, transfer_result = payload["results"]
    assert payload["summary"]["n_self_results"] == 1
    assert payload["summary"]["n_transfer_results"] == 1
    assert payload["summary"]["self_false_alarm_controlled"] == 1
    assert payload["summary"]["transfer_false_alarm_controlled"] == 0
    assert payload["summary"]["transfer_failures"][0]["target_dump"] == "shifted"
    assert self_result["score_source"] == "sweep_scores"
    assert self_result["selective_report"]["false_alarm"] == pytest.approx(0.0)
    assert transfer_result["selective_report"]["false_alarm"] == pytest.approx(1.0)
    assert transfer_result["false_alarm_excess"] == pytest.approx(0.9)
    assert payload["notes"] == ["unit-test"]


def test_eval_score_ensemble_compares_single_and_combined_signals(tmp_path):
    module = importlib.import_module("benchmarks.eval_score_ensemble")
    scores_path = tmp_path / "scores.json"
    labels = [0] * 20 + [1] * 8
    truth_proj = list(range(20)) + [40, 41, 42, 43, 0, 1, 2, 3]
    subspace_resid = list(range(20)) + [0, 1, 2, 3, 40, 41, 42, 43]
    scores_path.write_text(
        json.dumps({
            "config": {"model": "synthetic", "layer": -1},
            "labels": labels,
            "scores": {
                "truth_proj": truth_proj,
                "subspace_resid": subspace_resid,
            },
        }),
        encoding="utf-8",
    )

    payload = module.build_ensemble_report(
        [("synthetic", scores_path)],
        signals=("truth_proj", "subspace_resid"),
        methods=("max_rank", "mean_rank"),
        alphas=(0.2,),
        repeats=1,
        seed=0,
        best_alpha=0.2,
    )

    run = payload["runs"][0]
    single_detection = max(
        result["alphas"]["0.2"]["detection"]
        for result in run["single_results"].values()
    )
    ensemble_detection = run["ensemble_results"]["max_rank"]["alphas"]["0.2"]["detection"]

    assert run["best_single_at_alpha"]["name"] in {"truth_proj", "subspace_resid"}
    assert run["best_ensemble_at_alpha"]["name"] == "max_rank"
    assert ensemble_detection >= single_detection
    assert run["ensemble_results"]["max_rank"]["alphas"]["0.2"]["false_alarm"] <= 0.23


def test_eval_score_ensemble_reports_lower_direction_auroc(tmp_path):
    module = importlib.import_module("benchmarks.eval_score_ensemble")
    scores_path = tmp_path / "scores.json"
    scores_path.write_text(
        json.dumps({
            "config": {"model": "synthetic", "layer": -1},
            "labels": [0, 0, 0, 0, 1, 1, 1, 1],
            "scores": {"support": [10.0, 11.0, 12.0, 13.0, 0.0, 1.0, 2.0, 3.0]},
        }),
        encoding="utf-8",
    )
    previous_direction = module.DEFAULT_SCORE_DIRECTIONS.get("support")
    module.DEFAULT_SCORE_DIRECTIONS["support"] = "lower"
    try:
        payload = module.build_ensemble_report(
            [("synthetic", scores_path)],
            signals=("support",),
            methods=("mean_rank",),
            alphas=(0.5,),
            repeats=1,
            seed=0,
            best_alpha=0.5,
        )
    finally:
        if previous_direction is None:
            module.DEFAULT_SCORE_DIRECTIONS.pop("support", None)
        else:
            module.DEFAULT_SCORE_DIRECTIONS["support"] = previous_direction

    run = payload["runs"][0]
    assert run["single_results"]["support"]["direction"] == "lower"
    assert run["single_results"]["support"]["auroc"] == pytest.approx(1.0)
    assert run["single_results"]["support"]["alphas"]["0.5"]["detection"] == pytest.approx(1.0)


def test_eval_verifier_ensemble_suppresses_supported_and_rescues_refuted_claims(tmp_path):
    module = importlib.import_module("benchmarks.eval_verifier_ensemble")
    scores_path = tmp_path / "scores.json"
    fixture_path = tmp_path / "claims.json"
    labels = [0] * 20 + [1] * 8
    truth_proj = list(range(20)) + [0, 1, 2, 3, 40, 41, 42, 43]
    scores_path.write_text(
        json.dumps({
            "config": {"model": "synthetic", "layer": -1},
            "labels": labels,
            "scores": {"truth_proj": truth_proj},
        }),
        encoding="utf-8",
    )
    records = []
    for idx, label in enumerate(labels):
        if label == 0:
            claim = f"True item {idx} is supported."
            documents = [claim]
        else:
            claim = f"False item {idx} is correct."
            documents = [f"False item {idx} is not correct."]
        records.append({"claim": claim, "retrieval_documents": documents})
    fixture_path.write_text(json.dumps({"records": records}), encoding="utf-8")

    payload = module.build_verifier_ensemble_report(
        [("synthetic", scores_path)],
        signal="truth_proj",
        claims_path=fixture_path,
        alphas=(0.2,),
        repeats=1,
        seed=0,
        verifier_min_overlap=0.65,
    )

    result = payload["runs"][0]["alphas"]["0.2"]
    quality = payload["runs"][0]["verification_quality"]
    assert payload["runs"][0]["verification_status_counts"]["supported"] == 20
    assert payload["runs"][0]["verification_status_counts"]["refuted"] == 8
    assert quality["label_status_matrix"]["true"]["supported"] == 20
    assert quality["label_status_matrix"]["false"]["refuted"] == 8
    assert quality["decision_accuracy"] == pytest.approx(1.0)
    assert quality["decision_error_rate"] == pytest.approx(0.0)
    assert result["verified"]["false_alarm"] <= result["internal"]["false_alarm"]
    assert result["verified"]["detection"] > result["internal"]["detection"]
    assert result["verified"]["detection"] == pytest.approx(1.0)


def test_eval_truthfulqa_selective_reports_accept_score_directions():
    module = importlib.import_module("benchmarks.eval_truthfulqa")

    reports = module._selective_reports(
        {"support": [10.0, 11.0, 0.0, 1.0]},
        [0, 0, 1, 1],
        alpha=0.5,
        directions={"support": "lower"},
    )

    assert reports["support"]["threshold"] == pytest.approx(10.0)
    assert reports["support"]["direction"] == "lower"
    assert reports["support"]["false_alarm"] == pytest.approx(0.0)
    assert reports["support"]["detection"] == pytest.approx(1.0)


def test_eval_truthfulqa_exposes_internal_eigenscore_signal():
    module = importlib.import_module("benchmarks.eval_truthfulqa")

    assert "eigenscore" in module.SIGNALS
    assert module.DEFAULT_SCORE_DIRECTIONS["eigenscore"] == "higher"


def test_eval_truthfulqa_multisample_inside_signal_is_optional():
    module = importlib.import_module("benchmarks.eval_truthfulqa")

    disabled = SimpleNamespace(inside_samples=0)
    enabled = SimpleNamespace(inside_samples=3)
    sweep_layers = SimpleNamespace(sweep=False, sweep_layers="-12,-8")

    assert module.INSIDE_SIGNAL not in module._enabled_signals(disabled)
    assert module.INSIDE_SIGNAL in module._enabled_signals(enabled)
    assert module.INSIDE_SIGNAL in module._sweep_signal_names(enabled)
    assert module.DEFAULT_SCORE_DIRECTIONS[module.INSIDE_SIGNAL] == "higher"
    assert module._sweep_output_enabled(sweep_layers) is True


def test_eval_truthfulqa_candidate_verification_prompt_includes_context():
    module = importlib.import_module("benchmarks.eval_truthfulqa")
    stmt = module.Statement("What is the capital of France?", "Paris", 0)

    prompt = module._candidate_verification_prompt(stmt)

    assert "Question: What is the capital of France?" in prompt
    assert "Candidate answer: Paris" in prompt
    assert "factually correct" in prompt


def test_eval_truthfulqa_chunked_preserves_order():
    module = importlib.import_module("benchmarks.eval_truthfulqa")

    assert list(module._chunked([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]
    with pytest.raises(ValueError, match="batch size"):
        list(module._chunked([1], 0))


def test_eval_truthfulqa_batch_size_fallback_splits_memory_errors():
    module = importlib.import_module("benchmarks.eval_truthfulqa")
    state = module.BatchSizeFallbackState(4, enabled=True)
    calls = []

    def runner(items):
        calls.append(tuple(items))
        if len(items) > 2:
            raise RuntimeError("CUDA out of memory while allocating tensor")
        return [item * 10 for item in items]

    result = module._run_with_batch_size_fallback(
        [1, 2, 3, 4],
        state=state,
        phase="forced_answer_forward",
        runner=runner,
    )

    assert result == [10, 20, 30, 40]
    assert calls == [(1, 2, 3, 4), (1, 2), (3, 4)]
    assert state.batch_size() == 2
    assert state.to_dict()["n_reductions"] == 1
    assert state.to_dict()["reductions"][0]["phase"] == "forced_answer_forward"


def test_eval_truthfulqa_batch_size_fallback_does_not_hide_non_memory_errors():
    module = importlib.import_module("benchmarks.eval_truthfulqa")
    state = module.BatchSizeFallbackState(4, enabled=True)

    def runner(_items):
        raise RuntimeError("shape mismatch")

    with pytest.raises(RuntimeError, match="shape mismatch"):
        module._run_with_batch_size_fallback(
            [1, 2, 3, 4],
            state=state,
            phase="forced_answer_forward",
            runner=runner,
        )
    assert state.to_dict()["n_reductions"] == 0


def test_eval_truthfulqa_length_bucketed_batches_sort_by_statement_length():
    module = importlib.import_module("benchmarks.eval_truthfulqa")
    statements = [
        module.Statement("long question", "long answer", 0),
        module.Statement("", "x", 1),
        module.Statement("mid", "size", 0),
    ]

    plain = list(module._batched_statements(statements, 2, length_bucketed=False))
    bucketed = list(module._batched_statements(statements, 2, length_bucketed=True))

    assert plain == [statements[:2], statements[2:]]
    assert bucketed == [[statements[1], statements[2]], [statements[0]]]


def test_eval_truthfulqa_batched_statements_after_offset_slices_current_batch():
    module = importlib.import_module("benchmarks.eval_truthfulqa")
    statements = [
        module.Statement("", "a", 0),
        module.Statement("", "bb", 0),
        module.Statement("", "ccc", 0),
        module.Statement("", "dddd", 0),
        module.Statement("", "eeeee", 0),
    ]

    batches = list(module._batched_statements_after_offset(
        statements,
        3,
        length_bucketed=False,
        offset=2,
    ))

    assert [[stmt.answer for stmt in batch] for batch in batches] == [["ccc"], ["dddd", "eeeee"]]


def test_eval_truthfulqa_inside_trigger_indexes_support_threshold_and_budget():
    module = importlib.import_module("benchmarks.eval_truthfulqa")
    records = [
        {"primary_scores": {"maha_last": 0.1}},
        {"primary_scores": {"maha_last": 0.9}},
        {"primary_scores": {"maha_last": 0.4}},
        {"primary_scores": {"maha_last": 0.7}},
    ]

    threshold_args = SimpleNamespace(
        inside_trigger_signal="maha_last",
        inside_trigger_threshold=0.5,
        inside_trigger_top_fraction=None,
    )
    budget_args = SimpleNamespace(
        inside_trigger_signal="maha_last",
        inside_trigger_threshold=None,
        inside_trigger_top_fraction=0.5,
    )
    all_args = SimpleNamespace(
        inside_trigger_signal=None,
        inside_trigger_threshold=None,
        inside_trigger_top_fraction=None,
    )

    assert module._inside_trigger_indexes(records, threshold_args) == {1, 3}
    assert module._inside_trigger_indexes(records, budget_args) == {1, 3}
    assert module._inside_trigger_indexes(records, all_args) == {0, 1, 2, 3}


def test_eval_truthfulqa_layer_stats_cache_roundtrip(tmp_path):
    module = importlib.import_module("benchmarks.eval_truthfulqa")
    manifold = module.TruthManifold()
    manifold.update(torch.tensor([1.0, 0.0]))
    manifold.update(torch.tensor([0.0, 1.0]))
    manifold.false_mean = torch.tensor([0.5, -0.5])
    manifold.contrastive_direction = torch.tensor([1.0, 0.0])
    subspace = module.TruthSubspace.fit_contrastive(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([[1.0, 1.0], [2.0, 2.0]]),
        rank=1,
    )
    args = SimpleNamespace(
        model="tiny",
        dtype="float32",
        offline=True,
        max_length=32,
        subspace_rank=1,
        length_bucketed_batches=False,
    )
    metadata = module._layer_stats_cache_metadata(
        args,
        layers=[-1],
        n_layers=2,
        true_texts=["true a", "true b"],
        false_texts=["false a"],
    )

    cache_path = tmp_path / "layer-stats.pt"
    module.save_layer_stats_cache(cache_path, {-1: manifold}, {-1: subspace}, metadata=metadata)
    loaded_manifolds, loaded_subspaces, loaded_metadata = module.load_layer_stats_cache(
        cache_path,
        expected_metadata=metadata,
        device=torch.device("cpu"),
    )

    assert loaded_metadata == metadata
    assert loaded_manifolds[-1].n == 2
    assert torch.allclose(loaded_manifolds[-1].mean, manifold.mean)
    assert torch.allclose(loaded_manifolds[-1].contrastive_direction, manifold.contrastive_direction)
    assert loaded_subspaces[-1].rank == 1
    assert loaded_subspaces[-1].is_ready()


def test_eval_truthfulqa_warmup_checkpoint_can_resume_completed_state(tmp_path):
    module = importlib.import_module("benchmarks.eval_truthfulqa")
    true_states = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]
    false_states = [torch.tensor([2.0, 0.0]), torch.tensor([2.0, 2.0])]
    manifold = module.TruthManifold()
    for state in true_states:
        manifold.update(state)
    args = SimpleNamespace(
        model="tiny",
        dtype="float32",
        offline=True,
        max_length=32,
        subspace_rank=1,
        length_bucketed_batches=False,
    )
    true_texts = ["true a", "true b"]
    false_texts = ["false a", "false b"]
    metadata = module._layer_stats_cache_metadata(
        args,
        layers=[-1],
        n_layers=2,
        true_texts=true_texts,
        false_texts=false_texts,
    )
    checkpoint_path = tmp_path / "warmup-checkpoint.pt"
    module.save_warmup_checkpoint(
        checkpoint_path,
        metadata=metadata,
        manifolds={-1: manifold},
        true_state_lists={-1: true_states},
        false_state_lists={-1: false_states},
        false_sums={-1: sum(false_states)},
        n_false=2,
        true_done=2,
        false_done=2,
    )

    loaded = module.load_warmup_checkpoint(
        checkpoint_path,
        expected_metadata=metadata,
        device=torch.device("cpu"),
    )
    manifolds, subspaces = module.build_layer_stats(
        None,
        None,
        true_texts,
        false_texts,
        [-1],
        torch.device("cpu"),
        32,
        1,
        4,
        False,
        progress_every=0,
        checkpoint_path=checkpoint_path,
        checkpoint_metadata=metadata,
    )

    assert loaded["true_done"] == 2
    assert loaded["false_done"] == 2
    assert manifolds[-1].n == 2
    assert manifolds[-1].contrastive_direction is not None
    assert subspaces[-1].is_ready()


def test_eval_truthfulqa_layer_stats_cache_rejects_metadata_mismatch(tmp_path):
    module = importlib.import_module("benchmarks.eval_truthfulqa")
    manifold = module.TruthManifold()
    manifold.update(torch.tensor([1.0, 0.0]))
    manifold.update(torch.tensor([0.0, 1.0]))
    args = SimpleNamespace(
        model="tiny",
        dtype="float32",
        offline=True,
        max_length=32,
        subspace_rank=1,
        length_bucketed_batches=False,
    )
    metadata = module._layer_stats_cache_metadata(
        args,
        layers=[-1],
        n_layers=2,
        true_texts=["true a", "true b"],
        false_texts=["false a"],
    )

    cache_path = tmp_path / "layer-stats.pt"
    module.save_layer_stats_cache(cache_path, {-1: manifold}, {}, metadata=metadata)
    expected = {**metadata, "max_length": 64}

    with pytest.raises(ValueError, match="metadata does not match"):
        module.load_layer_stats_cache(
            cache_path,
            expected_metadata=expected,
            device=torch.device("cpu"),
        )


def test_eval_truthfulqa_eval_reps_cache_roundtrip(tmp_path):
    module = importlib.import_module("benchmarks.eval_truthfulqa")
    statements = [
        module.Statement("q1", "a1", 0),
        module.Statement("q2", "a2", 1),
    ]
    args = SimpleNamespace(
        model="tiny",
        dtype="float32",
        offline=True,
        max_length=32,
        eigenscore_alpha=1e-3,
        length_bucketed_batches=False,
    )
    metadata = module._eval_reps_cache_metadata(
        args,
        layers=[-1, -2],
        n_layers=2,
        eval_statements=statements,
    )
    reps = {
        "last": {-1: torch.tensor([1.0, 0.0]), -2: torch.tensor([0.0, 1.0])},
        "ans_hs": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "eigenscore_by_layer": {-1: 0.125, -2: 0.25},
        "nll": 1.5,
    }

    cache_path = tmp_path / "eval-reps.pt"
    module.save_eval_reps_cache(cache_path, [reps, None], metadata=metadata)
    loaded, loaded_metadata = module.load_eval_reps_cache(
        cache_path,
        expected_metadata=metadata,
        expected_records=2,
    )

    assert loaded_metadata == metadata
    assert torch.allclose(loaded[0]["last"][-1], reps["last"][-1])
    assert torch.allclose(loaded[0]["ans_hs"], reps["ans_hs"])
    assert loaded[0]["eigenscore_by_layer"][-2] == pytest.approx(0.25)
    assert loaded[0]["nll"] == pytest.approx(1.5)
    assert loaded[1] is None


def test_eval_truthfulqa_eval_reps_sharded_cache_roundtrip_and_range(tmp_path, monkeypatch):
    module = importlib.import_module("benchmarks.eval_truthfulqa")
    statements = [
        module.Statement("q1", "a1", 0),
        module.Statement("q2", "a2", 1),
        module.Statement("q3", "a3", 0),
    ]
    args = SimpleNamespace(
        model="tiny",
        dtype="float32",
        offline=True,
        max_length=32,
        eigenscore_alpha=1e-3,
        length_bucketed_batches=False,
    )
    metadata = module._eval_reps_cache_metadata(
        args,
        layers=[-1, -2],
        n_layers=2,
        eval_statements=statements,
    )
    reps_a = {
        "last": {-1: torch.tensor([1.0, 0.0]), -2: torch.tensor([0.0, 1.0])},
        "ans_hs": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "eigenscore_by_layer": {-1: 0.125, -2: 0.25},
        "nll": 1.5,
    }
    reps_b = {
        "last": {-1: torch.tensor([2.0, 0.0]), -2: torch.tensor([0.0, 2.0])},
        "ans_hs": torch.tensor([[2.0, 3.0], [4.0, 5.0]]),
        "eigenscore_by_layer": {-1: 0.5, -2: 0.75},
        "nll": 2.5,
    }

    cache_dir = tmp_path / "eval-reps-cache"
    module.save_eval_reps_cache(cache_dir, [reps_a, None, reps_b], metadata=metadata, shard_size=2)
    reader = module.EvalRepsCacheReader(
        cache_dir,
        expected_metadata=metadata,
        expected_records=3,
    )
    original_torch_load = module.torch.load
    loaded_shards = []

    def counting_torch_load(path, *args, **kwargs):
        if str(path).endswith(".pt"):
            loaded_shards.append(Path(path).name)
        return original_torch_load(path, *args, **kwargs)

    monkeypatch.setattr(module.torch, "load", counting_torch_load)

    ranged = reader.read_range(1, 2)
    repeat_second_shard = reader.read_range(2, 1)
    repeat_first_shard = reader.read_range(0, 1)
    repeat_first_shard_again = reader.read_range(1, 1)
    reader_stats = reader.cache_stats()
    counted_reader_loads = list(loaded_shards)
    monkeypatch.setattr(module.torch, "load", original_torch_load)

    loaded, loaded_metadata = module.load_eval_reps_cache(
        cache_dir,
        expected_metadata=metadata,
        expected_records=3,
    )

    assert (cache_dir / "manifest.json").exists()
    assert module._read_cache_metadata(cache_dir) == metadata
    assert reader.metadata == metadata
    assert ranged[0] is None
    assert torch.allclose(ranged[1]["last"][-1], reps_b["last"][-1])
    assert torch.allclose(repeat_first_shard[0]["ans_hs"], reps_a["ans_hs"])
    assert repeat_first_shard_again[0] is None
    assert repeat_second_shard[0]["nll"] == pytest.approx(2.5)
    assert counted_reader_loads == [
        "records-000000.pt",
        "records-000001.pt",
        "records-000000.pt",
    ]
    assert reader_stats == {"shard_loads": 3, "shard_cache_hits": 2}
    assert loaded_metadata == metadata
    assert torch.allclose(loaded[0]["ans_hs"], reps_a["ans_hs"])
    assert loaded[1] is None
    assert loaded[2]["nll"] == pytest.approx(2.5)


def test_eval_truthfulqa_eval_reps_cache_rejects_metadata_mismatch(tmp_path):
    module = importlib.import_module("benchmarks.eval_truthfulqa")
    args = SimpleNamespace(
        model="tiny",
        dtype="float32",
        offline=True,
        max_length=32,
        eigenscore_alpha=1e-3,
        length_bucketed_batches=False,
    )
    metadata = module._eval_reps_cache_metadata(
        args,
        layers=[-1],
        n_layers=2,
        eval_statements=[module.Statement("q", "a", 0)],
    )
    cache_path = tmp_path / "eval-reps.pt"
    module.save_eval_reps_cache(cache_path, [None], metadata=metadata)
    expected = {**metadata, "eigenscore_alpha": 1e-2}

    with pytest.raises(ValueError, match="eval reps cache metadata does not match"):
        module.load_eval_reps_cache(
            cache_path,
            expected_metadata=expected,
            expected_records=1,
        )


def test_eval_truthfulqa_score_reps_batch_matches_scalar_math():
    module = importlib.import_module("benchmarks.eval_truthfulqa")
    statements = [
        module.Statement("q1", "a1", 0),
        module.Statement("q2", "a2", 1),
        module.Statement("q3", "a3", 1),
    ]
    manifold_1 = module.TruthManifold()
    for state in (torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0]), torch.tensor([1.0, 1.0, 0.0])):
        manifold_1.update(state)
    manifold_1.contrastive_direction = torch.tensor([1.0, 0.0, 0.0])
    manifold_2 = module.TruthManifold()
    for state in (torch.tensor([0.0, 1.0, 0.0]), torch.tensor([0.0, 0.0, 1.0])):
        manifold_2.update(state)
    subspace = module.TruthSubspace.fit_contrastive(
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]]),
        rank=2,
    )
    reps_1 = {
        "last": {-1: torch.tensor([1.0, 1.0, 0.5]), -2: torch.tensor([0.0, 1.0, 1.0])},
        "ans_hs": torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        "eigenscore_by_layer": {-1: 0.125, -2: 0.25},
        "nll": 1.5,
    }
    reps_3 = {
        "last": {-1: torch.tensor([0.0, 1.0, 1.5]), -2: torch.tensor([1.0, 0.0, 1.0])},
        "ans_hs": torch.tensor([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]]),
        "eigenscore_by_layer": {-1: 0.5, -2: 0.75},
        "nll": 2.0,
    }

    records = module._score_reps_batch(
        statements,
        [reps_1, None, reps_3],
        layers=[-1, -2],
        target_layer=-1,
        manifolds={-1: manifold_1, -2: manifold_2},
        subspaces={-1: subspace},
    )

    assert [record["stmt"] for record in records] == [statements[0], statements[2]]
    for record, reps in zip(records, [reps_1, reps_3]):
        h = reps["last"][-1]
        assert record["layer_scores"][-1]["maha_last"] == pytest.approx(
            module.mahalanobis_distance(h, manifold_1.mean, manifold_1.cov_inv).item()
        )
        assert record["layer_scores"][-1]["truth_proj"] == pytest.approx(
            -torch.dot(h, manifold_1.contrastive_direction).item()
        )
        assert record["layer_scores"][-1]["subspace_resid"] == pytest.approx(subspace.residual_distance(h).item())
        assert record["layer_scores"][-2]["truth_proj"] == pytest.approx(0.0)
        assert record["layer_scores"][-2]["subspace_resid"] == pytest.approx(0.0)
        assert record["primary_scores"]["eigenscore"] == pytest.approx(reps["eigenscore_by_layer"][-1])
        assert record["primary_scores"]["disp_euclid"] == pytest.approx(
            module.euclidean_dispersion(reps["ans_hs"]).item()
        )
        assert record["primary_scores"]["disp_hse"] == pytest.approx(
            module.hyperbolic_semantic_entropy(module.poincare_map(reps["ans_hs"])).item()
        )
        assert record["primary_scores"]["nll_answer"] == pytest.approx(reps["nll"])


def test_eval_truthfulqa_inside_seed_changes_by_inner_batch():
    module = importlib.import_module("benchmarks.eval_truthfulqa")

    seeds = {
        module._inside_seed(7, eval_batch_idx=0, inside_batch_idx=0),
        module._inside_seed(7, eval_batch_idx=0, inside_batch_idx=1),
        module._inside_seed(7, eval_batch_idx=1, inside_batch_idx=0),
    }

    assert len(seeds) == 3
    assert module._inside_seed(7, eval_batch_idx=2, inside_batch_idx=3) == module._inside_seed(
        7, eval_batch_idx=2, inside_batch_idx=3
    )


def test_eval_truthfulqa_resolves_limited_sweep_layers():
    module = importlib.import_module("benchmarks.eval_truthfulqa")

    assert module._parse_sweep_layers("-12,-8,0") == [-12, -8, 0]
    assert module._resolve_sweep_layers(-8, 24, sweep=False, sweep_layers="-12,-8,-4") == [-8, -12, -4]
    assert module._resolve_sweep_layers(-1, 3, sweep=True, sweep_layers=None) == [-1, -2, -3]
    assert module._resolve_sweep_layers(-1, 3, sweep=False, sweep_layers=None) == [-1]

    with pytest.raises(ValueError, match="comma-separated"):
        module._parse_sweep_layers("-8,nope")
    with pytest.raises(ValueError, match="at least one"):
        module._parse_sweep_layers(" , ")
    with pytest.raises(ValueError, match="out of range"):
        module._resolve_sweep_layers(-1, 3, sweep=False, sweep_layers="-9")


def test_eval_truthfulqa_profile_helpers_accumulate_and_serialize():
    module = importlib.import_module("benchmarks.eval_truthfulqa")
    profile = {}

    with module._profile_phase(profile, "load_model"):
        pass
    with module._profile_phase(profile, "load_model"):
        pass

    payload = module._profile_payload(profile, total_seconds=1.23456789)

    assert profile["load_model"] >= 0.0
    assert payload["total_seconds"] == pytest.approx(1.234568)
    assert payload["phases"]["load_model"] >= 0.0
    assert payload["summary"]["bottleneck"] == "load_model"
    assert payload["summary"]["groups"]["startup"]["seconds"] >= 0.0


def test_eval_truthfulqa_profile_summary_groups_and_throughput():
    module = importlib.import_module("benchmarks.eval_truthfulqa")

    payload = module._profile_payload(
        {
            "build_layer_stats": 6.0,
            "forced_answer_forward": 3.0,
            "score_postprocess": 1.0,
            "load_eval_reps_cache": 0.5,
        },
        total_seconds=12.0,
        n_eval_records=30,
        n_warmup_true=18,
        n_warmup_false=12,
    )
    summary = payload["summary"]

    assert summary["bottleneck"] == "build_layer_stats"
    assert summary["groups"]["model_forward"]["seconds"] == pytest.approx(9.0)
    assert summary["groups"]["model_forward"]["share"] == pytest.approx(0.75)
    assert summary["groups"]["cache_io"]["seconds"] == pytest.approx(0.5)
    assert summary["throughput"]["forced_answer_records_per_second"] == pytest.approx(10.0)
    assert summary["throughput"]["warmup_records_per_second"] == pytest.approx(5.0)
    assert summary["throughput"]["end_to_end_eval_records_per_second"] == pytest.approx(2.5)
    assert summary["accounted_seconds"] == pytest.approx(10.5)
    assert summary["unaccounted_seconds"] == pytest.approx(1.5)


def test_eval_truthfulqa_progress_helpers_report_marks_and_final():
    module = importlib.import_module("benchmarks.eval_truthfulqa")

    assert module._progress_report_due(49, total=100, every=50, last_reported=0) is False
    assert module._progress_report_due(50, total=100, every=50, last_reported=0) is True
    assert module._progress_report_due(75, total=100, every=50, last_reported=50) is False
    assert module._progress_report_due(100, total=100, every=50, last_reported=50) is True
    assert module._progress_report_due(100, total=100, every=0, last_reported=0) is False

    line = module._format_progress("warmup true", completed=25, total=100, elapsed_seconds=10.0)

    assert line == "   warmup true: 25/100 (25.0%) elapsed=10.0s rate=2.50/s"


def test_eval_truthfulqa_batch_tokenization_matches_single_statement_helper():
    module = importlib.import_module("benchmarks.eval_truthfulqa")

    class TokenizerOutput:
        def __init__(self, input_ids):
            self.input_ids = input_ids

    class CountingTokenizer:
        bos_token = "<bos>"
        eos_token = "<eos>"

        def __init__(self):
            self.calls = []

        def __call__(self, text, *, add_special_tokens=True):
            self.calls.append({"batched": isinstance(text, list), "add_special_tokens": add_special_tokens})
            if isinstance(text, list):
                return TokenizerOutput([self._encode(item, add_special_tokens) for item in text])
            return TokenizerOutput(self._encode(text, add_special_tokens))

        def _encode(self, text, add_special_tokens):
            ids = [(ord(char) % 17) + 3 for char in text]
            if add_special_tokens:
                return [1, *ids, 2]
            return ids

    statements = [
        module.Statement("Question one?", "Answer one", 0),
        module.Statement("", "Fallback answer", 1),
        module.Statement("Question two?", "Second answer", 0),
    ]
    max_length = 18
    single_tokenizer = CountingTokenizer()
    batch_tokenizer = CountingTokenizer()

    expected = [
        module._statement_token_ids(single_tokenizer, statement, max_length)
        for statement in statements
    ]
    actual = module._batch_statement_token_ids(batch_tokenizer, statements, max_length)

    assert actual == expected
    assert len(single_tokenizer.calls) == 6
    assert len(batch_tokenizer.calls) == 3
    assert batch_tokenizer.calls[0] == {"batched": True, "add_special_tokens": True}
    assert batch_tokenizer.calls[-1] == {"batched": True, "add_special_tokens": False}


def test_eval_truthfulqa_statement_encoding_cache_roundtrip(tmp_path):
    module = importlib.import_module("benchmarks.eval_truthfulqa")

    class TokenizerOutput:
        def __init__(self, input_ids):
            self.input_ids = input_ids

    class SimpleTokenizer:
        bos_token = "<bos>"
        eos_token = "<eos>"

        def __call__(self, text, *, add_special_tokens=True):
            if isinstance(text, list):
                return TokenizerOutput([self._encode(item, add_special_tokens) for item in text])
            return TokenizerOutput(self._encode(text, add_special_tokens))

        def _encode(self, text, add_special_tokens):
            ids = [(ord(char) % 19) + 3 for char in text]
            return [1, *ids, 2] if add_special_tokens else ids

    args = SimpleNamespace(model="synthetic", offline=True, max_length=16)
    true_texts = ["True fact."]
    false_texts = ["False claim."]
    eval_statements = [module.Statement("Q?", "A.", 0)]
    metadata = module._statement_encoding_cache_metadata(
        args,
        true_texts=true_texts,
        false_texts=false_texts,
        eval_statements=eval_statements,
    )
    tokenizer = SimpleTokenizer()
    true_encodings = module._batch_statement_encodings(tokenizer, [module.Statement("", true_texts[0], 0)], 16)
    false_encodings = module._batch_statement_encodings(tokenizer, [module.Statement("", false_texts[0], 1)], 16)
    eval_encodings = module._batch_statement_encodings(tokenizer, eval_statements, 16)
    cache_path = tmp_path / "statement-encodings.json"

    module.save_statement_encoding_cache(
        cache_path,
        metadata=metadata,
        true_encodings=true_encodings,
        false_encodings=false_encodings,
        eval_encodings=eval_encodings,
    )
    loaded_true, loaded_false, loaded_eval, loaded_metadata = module.load_statement_encoding_cache(
        cache_path,
        expected_metadata=metadata,
    )

    assert loaded_metadata == metadata
    assert [item.to_dict() for item in loaded_true] == [item.to_dict() for item in true_encodings]
    assert [item.to_dict() for item in loaded_false] == [item.to_dict() for item in false_encodings]
    assert [item.to_dict() for item in loaded_eval] == [item.to_dict() for item in eval_encodings]

    mismatched = dict(metadata)
    mismatched["max_length"] = 32
    with pytest.raises(ValueError, match="statement encoding cache metadata"):
        module.load_statement_encoding_cache(cache_path, expected_metadata=mismatched)


def test_eval_truthfulqa_statement_encoding_pairs_preserve_length_bucket_alignment():
    module = importlib.import_module("benchmarks.eval_truthfulqa")
    statements = [
        module.Statement("", "longer statement", 0),
        module.Statement("", "x", 1),
    ]
    encodings = [
        module.StatementEncoding((1, 10, 11), 2),
        module.StatementEncoding((1, 20), 1),
    ]

    batches = list(module._batched_statement_pairs(statements, encodings, 2, length_bucketed=True))

    assert [stmt.answer for stmt, _encoding in batches[0]] == ["x", "longer statement"]
    assert [encoding.input_ids for _stmt, encoding in batches[0]] == [(1, 20), (1, 10, 11)]


def test_eval_truthfulqa_batched_statement_reps_can_use_precomputed_encodings():
    module = importlib.import_module("benchmarks.eval_truthfulqa")

    class NoCallTokenizer:
        pad_token_id = 0
        eos_token_id = 2

        def __call__(self, *_args, **_kwargs):
            raise AssertionError("precomputed encodings should bypass tokenizer calls")

    class TinyModel(nn.Module):
        def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False, **_kwargs):
            del attention_mask
            state = input_ids.float().unsqueeze(-1).repeat(1, 1, 4)
            logits = torch.zeros((*input_ids.shape, 64), dtype=torch.float32)
            return SimpleNamespace(
                logits=logits,
                hidden_states=(state,) if output_hidden_states else None,
            )

    reps = module.batched_statement_reps(
        TinyModel(),
        NoCallTokenizer(),
        [module.Statement("Q?", "A.", 0)],
        [0],
        torch.device("cpu"),
        8,
        encoded_statements=[module.StatementEncoding((1, 5, 6), 2)],
    )

    assert reps[0]["last"][0].shape == (4,)
    assert reps[0]["ans_hs"].shape == (2, 4)
    assert reps[0]["nll"] == pytest.approx(torch.log(torch.tensor(64.0)).item())


def test_eval_truthfulqa_statement_dump_preserves_question_answer_and_label():
    module = importlib.import_module("benchmarks.eval_truthfulqa")
    stmt = module.Statement("Where?", "There.", 1)

    payload = module._statement_to_dump(stmt)

    assert payload == {
        "question": "Where?",
        "answer": "There.",
        "text": "Where? There.",
        "is_false": 1,
    }


def test_eval_truthfulqa_hook_capture_matches_hidden_states_for_intermediate_layer():
    module = importlib.import_module("benchmarks.eval_truthfulqa")

    class TokenizerOutput:
        def __init__(self, input_ids):
            self.input_ids = input_ids

    class SimpleTokenizer:
        bos_token = "<bos>"
        eos_token = "<eos>"
        pad_token_id = 0
        eos_token_id = 2

        def __call__(self, text, *, add_special_tokens=True, **_kwargs):
            if isinstance(text, list):
                return TokenizerOutput([self._encode(item, add_special_tokens) for item in text])
            return TokenizerOutput(self._encode(text, add_special_tokens))

        def _encode(self, text, add_special_tokens):
            ids = [3 + (ord(char) % 13) for char in text]
            if add_special_tokens:
                return [1, *ids, 2]
            return ids

    class Block(nn.Module):
        def __init__(self, hidden_dim):
            super().__init__()
            self.linear = nn.Linear(hidden_dim, hidden_dim)

        def forward(self, x):
            return (torch.tanh(self.linear(x)), None)

    class HookableModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(num_hidden_layers=3)
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([Block(8) for _ in range(3)])
            self.embed = nn.Embedding(64, 8)
            self.norm = nn.LayerNorm(8)
            self.lm_head = nn.Linear(8, 64)

        def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False, **_kwargs):
            del attention_mask
            x = self.embed(input_ids)
            hidden_states = [x]
            for idx, layer in enumerate(self.model.layers):
                x, _ = layer(x)
                if idx < len(self.model.layers) - 1:
                    hidden_states.append(x)
            x = self.norm(x)
            hidden_states.append(x)
            return SimpleNamespace(
                logits=self.lm_head(x),
                hidden_states=tuple(hidden_states) if output_hidden_states else None,
            )

    torch.manual_seed(0)
    model = HookableModel()
    tokenizer = SimpleTokenizer()
    statements = [
        module.Statement("Question?", "Answer", 0),
        module.Statement("", "Fallback", 1),
    ]

    output_reps = module.batched_statement_reps(
        model,
        tokenizer,
        statements,
        [1],
        torch.device("cpu"),
        32,
        hidden_state_capture="outputs",
    )
    hook_reps = module.batched_statement_reps(
        model,
        tokenizer,
        statements,
        [1],
        torch.device("cpu"),
        32,
        hidden_state_capture="hooks",
    )

    for output_rep, hook_rep in zip(output_reps, hook_reps):
        assert torch.allclose(output_rep["last"][1], hook_rep["last"][1])
        assert torch.allclose(output_rep["ans_hs"], hook_rep["ans_hs"])
        assert hook_rep["nll"] == pytest.approx(output_rep["nll"])
        assert hook_rep["eigenscore_by_layer"][1] == pytest.approx(output_rep["eigenscore_by_layer"][1])


def test_eval_truthfulqa_hook_capture_rejects_final_hidden_state():
    module = importlib.import_module("benchmarks.eval_truthfulqa")

    class Block(nn.Module):
        def forward(self, x):
            return (x, None)

    class HookableModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([Block(), Block()])

    with pytest.raises(ValueError, match="final post-norm hidden state"):
        module._hook_capture_layer_map(HookableModel(), [-1])

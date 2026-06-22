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
    run = payload["runs"][0]
    quality = run["verification_quality"]
    routes = run["route_summary"]
    cache_stats = run["cache_stats"]
    retrieval_route_quality = run["route_quality"]["retrieval_groundedness"]
    groundedness_route_quality = run["route_quality"]["groundedness"]

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
    assert routes["by_route"]["retrieval_groundedness"]["duration_observations"] == 2
    assert routes["by_route"]["retrieval_groundedness"]["mean_duration_seconds"] >= 0.0
    assert routes["by_route"]["retrieval_groundedness"]["p95_duration_seconds"] >= 0.0
    assert routes["by_route"]["retrieval_groundedness"]["p99_duration_seconds"] >= 0.0
    assert routes["by_route"]["retrieval_groundedness"]["mean_attempted_route_count"] == pytest.approx(2.0)
    assert routes["by_route"]["retrieval_groundedness"]["retrieval_use_rate"] == pytest.approx(1.0)
    assert routes["by_route"]["groundedness"]["statuses"]["insufficient_evidence"] == 1
    assert routes["by_route"]["groundedness"]["duration_observations"] == 1
    assert routes["by_route"]["groundedness"]["mean_attempted_route_count"] == pytest.approx(1.0)
    assert retrieval_route_quality["duration_observations"] == 2
    assert retrieval_route_quality["mean_duration_seconds"] >= 0.0
    assert retrieval_route_quality["p95_duration_seconds"] >= 0.0
    assert retrieval_route_quality["p99_duration_seconds"] >= 0.0
    assert retrieval_route_quality["mean_selected_route_duration_seconds"] >= 0.0
    assert retrieval_route_quality["p95_selected_route_duration_seconds"] >= 0.0
    assert retrieval_route_quality["p99_selected_route_duration_seconds"] >= 0.0
    assert retrieval_route_quality["mean_attempted_route_count"] == pytest.approx(2.0)
    assert retrieval_route_quality["retrieval_use_rate"] == pytest.approx(1.0)
    assert groundedness_route_quality["duration_observations"] == 1
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


def test_build_domain_state_fixture_writes_sqlite_state_source_for_verifier(tmp_path):
    builder = importlib.import_module("benchmarks.build_domain_state_fixture")
    verifier = importlib.import_module("benchmarks.eval_verifier_ensemble")
    scores_path = tmp_path / "domain_scores.json"
    claims_path = tmp_path / "domain_claims.json"
    state_path = tmp_path / "domain_state.json"
    sqlite_path = tmp_path / "domain_state.db"
    sqlite_state_source_path = tmp_path / "domain_sqlite_state_source.json"

    payload = builder.run(SimpleNamespace(
        scores_output=str(scores_path),
        claims_output=str(claims_path),
        state_output=str(state_path),
        sqlite_output=str(sqlite_path),
        sqlite_state_source_output=str(sqlite_state_source_path),
        n_records=8,
        signal="truth_proj",
    ))
    sqlite_spec = json.loads(sqlite_state_source_path.read_text(encoding="utf-8"))

    assert sqlite_path.exists()
    assert sqlite_spec["sqlite"]["database_path"] == "domain_state.db"
    assert sqlite_spec["summary"]["n_queries"] == 8
    assert payload["sqlite_state_source"]["fixture_type"] == "order_fulfillment_sqlite_state_source"

    report = verifier.build_verifier_ensemble_report(
        [("orders", scores_path)],
        signal="truth_proj",
        claims_path=claims_path,
        state_path=sqlite_state_source_path,
        alphas=(0.2,),
        repeats=1,
        seed=0,
    )
    run = report["runs"][0]
    route_quality = run["route_quality"]["structured_state"]

    assert report["state_verifier"]["state_path"] == str(sqlite_state_source_path)
    assert run["state_verifier"]["enabled"] is True
    assert run["state_verifier"]["decided_records"] == 8
    assert run["route_summary"]["selected_counts"] == {"structured_state": 8}
    assert route_quality["decision_accuracy"] == pytest.approx(1.0)
    assert route_quality["false_refuted_rate"] == pytest.approx(1.0)
    assert route_quality["false_supported_rate"] == pytest.approx(0.0)


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


def test_eval_verifier_ensemble_run_can_write_compact_json(tmp_path):
    module = importlib.import_module("benchmarks.eval_verifier_ensemble")
    scores_path = tmp_path / "scores.json"
    qa_path = tmp_path / "qa.json"
    output_path = tmp_path / "compact-report.json"
    scores_path.write_text(
        json.dumps({
            "config": {"model": "synthetic", "layer": -1},
            "labels": [0, 0, 1, 1],
            "scores": {"truth_proj": [0.1, 0.2, 0.8, 0.9]},
            "statements": [
                {"question": "Q1?", "answer": "A1", "text": "Q1? A1"},
                {"question": "Q2?", "answer": "A2", "text": "Q2? A2"},
                {"question": "Q1?", "answer": "Wrong A1", "text": "Q1? Wrong A1"},
                {"question": "Q2?", "answer": "Wrong A2", "text": "Q2? Wrong A2"},
            ],
        }),
        encoding="utf-8",
    )
    qa_path.write_text(
        json.dumps({
            "documents": [
                {"question": "Q1?", "answer": "A1", "source": "qa:q1"},
                {"question": "Q2?", "answer": "A2", "source": "qa:q2"},
            ],
        }),
        encoding="utf-8",
    )

    payload = module.run(SimpleNamespace(
        scores=[f"synthetic={scores_path}"],
        claims=None,
        qa_corpus=str(qa_path),
        state_source=None,
        signal="truth_proj",
        direction=None,
        alphas="0.2",
        repeats=1,
        seed=0,
        best_alpha=0.2,
        verifier_min_overlap=0.65,
        retriever_min_overlap=0.2,
        retrieval_limit=5,
        json=str(output_path),
        compact_json=True,
    ))
    written = output_path.read_text(encoding="utf-8")

    assert payload["runs"][0]["route_quality"]["structured_qa"]["decision_accuracy"] == pytest.approx(1.0)
    assert json.loads(written)["runs"][0]["cache_stats"]["qa_verifier"]["requests"] == 4
    assert "\n  " not in written
    assert ": " not in written


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
    assert payload["promotion_decision"]["status"] == "needs_gate"
    assert payload["promotion_decision"]["recommended_route"] == payload["pareto_frontier"]["recommended"]["route"]
    assert payload["by_route"]["structured_qa"]["selected"] == 15
    assert payload["by_route"]["structured_qa"]["false_refuted_rate"] == pytest.approx(5 / 6)
    assert payload["by_route"]["structured_qa"]["false_supported_rate"] == pytest.approx(1 / 6)
    assert payload["by_route"]["structured_qa"]["verified_detection"] == pytest.approx(5 / 6)
    assert payload["by_route"]["structured_state"]["rescued_detection_rate"] == pytest.approx(0.75)


def test_compare_verifier_routes_builds_quality_gate(tmp_path):
    module = importlib.import_module("benchmarks.compare_verifier_routes")
    report_path = tmp_path / "routes.json"
    report_path.write_text(
        json.dumps({
            "runs": [
                {
                    "name": "orders",
                    "route_quality": {
                        "structured_state": {
                            "selected": 8,
                            "selection_rate": 1.0,
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
                            "decision_accuracy": 1.0,
                            "decision_error_rate": 0.0,
                        },
                        "groundedness": {
                            "selected": 6,
                            "selection_rate": 0.75,
                            "n_true": 3,
                            "n_false": 3,
                            "label_status_matrix": {
                                "true": {"supported": 3, "refuted": 0, "insufficient_evidence": 0},
                                "false": {"supported": 1, "refuted": 2, "insufficient_evidence": 0},
                            },
                            "true_supported_rate": 1.0,
                            "false_refuted_rate": 2 / 3,
                            "false_supported_rate": 1 / 3,
                            "decision_accuracy": 5 / 6,
                            "decision_error_rate": 1 / 6,
                        },
                    },
                    "alphas": {
                        "0.1": {
                            "route_control_impact": {
                                "structured_state": {
                                    "verified": {"false_alarm": 0.0, "detection": 1.0},
                                    "delta": {"rescued_detection_rate": 0.5},
                                },
                                "groundedness": {
                                    "verified": {"false_alarm": 0.1, "detection": 2 / 3},
                                    "delta": {"rescued_detection_rate": 0.1},
                                },
                            }
                        }
                    },
                }
            ]
        }),
        encoding="utf-8",
    )

    passing = module.build_route_comparison_report(
        [("orders", report_path)],
        gate_routes=("structured_state",),
        min_decision_accuracy=0.99,
        max_false_supported_rate=0.0,
        min_false_refuted_rate=0.99,
        max_verified_false_alarm=0.0,
        min_verified_detection=0.99,
    )
    failing = module.build_route_comparison_report(
        [("orders", report_path)],
        gate_routes=("groundedness",),
        min_decision_accuracy=0.90,
        max_false_supported_rate=0.10,
        min_false_refuted_rate=0.90,
    )

    assert passing["quality_gate"]["passed"] is True
    assert passing["quality_gate"]["checked_routes"] == ["structured_state"]
    assert failing["quality_gate"]["passed"] is False
    assert {failure["metric"] for failure in failing["quality_gate"]["failures"]} == {
        "decision_accuracy",
        "false_supported_rate",
        "false_refuted_rate",
    }


def test_compare_verifier_routes_builds_cost_aware_quality_gate(tmp_path):
    module = importlib.import_module("benchmarks.compare_verifier_routes")
    report_path = tmp_path / "route-cost.json"
    report_path.write_text(
        json.dumps({
            "runs": [
                {
                    "name": "costs",
                    "route_quality": {
                        "structured_state": {
                            "selected": 4,
                            "n_true": 2,
                            "n_false": 2,
                            "label_status_matrix": {
                                "true": {"supported": 2, "refuted": 0, "insufficient_evidence": 0},
                                "false": {"supported": 0, "refuted": 2, "insufficient_evidence": 0},
                            },
                            "false_refuted_rate": 1.0,
                            "false_supported_rate": 0.0,
                            "decision_accuracy": 1.0,
                            "duration_observations": 4,
                            "total_duration_seconds": 0.04,
                            "mean_duration_seconds": 0.01,
                            "p95_duration_seconds": 0.019,
                            "p99_duration_seconds": 0.0198,
                            "max_duration_seconds": 0.02,
                            "selected_route_duration_observations": 4,
                            "total_selected_route_duration_seconds": 0.04,
                            "mean_selected_route_duration_seconds": 0.01,
                            "p95_selected_route_duration_seconds": 0.019,
                            "p99_selected_route_duration_seconds": 0.0198,
                            "attempted_route_count_observations": 4,
                            "total_attempted_route_count": 4,
                            "mean_attempted_route_count": 1.0,
                            "used_retrieval_count": 0,
                            "retrieval_use_rate": 0.0,
                            "retrieval_hit_count": 0,
                            "mean_retrieval_hits": 0.0,
                        },
                        "retrieval_groundedness": {
                            "selected": 4,
                            "n_true": 2,
                            "n_false": 2,
                            "label_status_matrix": {
                                "true": {"supported": 2, "refuted": 0, "insufficient_evidence": 0},
                                "false": {"supported": 0, "refuted": 2, "insufficient_evidence": 0},
                            },
                            "false_refuted_rate": 1.0,
                            "false_supported_rate": 0.0,
                            "decision_accuracy": 1.0,
                            "duration_observations": 4,
                            "total_duration_seconds": 0.40,
                            "mean_duration_seconds": 0.10,
                            "p95_duration_seconds": 0.19,
                            "p99_duration_seconds": 0.198,
                            "max_duration_seconds": 0.20,
                            "selected_route_duration_observations": 4,
                            "total_selected_route_duration_seconds": 0.32,
                            "mean_selected_route_duration_seconds": 0.08,
                            "p95_selected_route_duration_seconds": 0.15,
                            "p99_selected_route_duration_seconds": 0.158,
                            "attempted_route_count_observations": 4,
                            "total_attempted_route_count": 8,
                            "mean_attempted_route_count": 2.0,
                            "used_retrieval_count": 4,
                            "retrieval_use_rate": 1.0,
                            "retrieval_hit_count": 6,
                            "mean_retrieval_hits": 1.5,
                        },
                        "fast_lexical": {
                            "selected": 4,
                            "n_true": 2,
                            "n_false": 2,
                            "label_status_matrix": {
                                "true": {"supported": 2, "refuted": 0, "insufficient_evidence": 0},
                                "false": {"supported": 1, "refuted": 1, "insufficient_evidence": 0},
                            },
                            "false_refuted_rate": 0.5,
                            "false_supported_rate": 0.5,
                            "decision_accuracy": 0.75,
                            "duration_observations": 4,
                            "total_duration_seconds": 0.004,
                            "mean_duration_seconds": 0.001,
                            "p95_duration_seconds": 0.0019,
                            "p99_duration_seconds": 0.00198,
                            "max_duration_seconds": 0.002,
                            "selected_route_duration_observations": 4,
                            "total_selected_route_duration_seconds": 0.004,
                            "mean_selected_route_duration_seconds": 0.001,
                            "p95_selected_route_duration_seconds": 0.0019,
                            "p99_selected_route_duration_seconds": 0.00198,
                            "attempted_route_count_observations": 4,
                            "total_attempted_route_count": 4,
                            "mean_attempted_route_count": 1.0,
                            "used_retrieval_count": 0,
                            "retrieval_use_rate": 0.0,
                            "retrieval_hit_count": 0,
                            "mean_retrieval_hits": 0.0,
                        },
                    },
                    "cache_stats": {
                        "total": {"size": 3, "hits": 6, "misses": 2, "requests": 8, "hit_rate": 0.75}
                    },
                    "alphas": {"0.1": {"route_control_impact": {}}},
                }
            ]
        }),
        encoding="utf-8",
    )

    passing = module.build_route_comparison_report(
        [("costs", report_path)],
        gate_routes=("structured_state",),
        max_mean_duration_seconds=0.02,
        max_p95_duration_seconds=0.02,
        max_p99_duration_seconds=0.02,
        max_max_duration_seconds=0.03,
        max_mean_attempted_route_count=1.1,
        max_retrieval_use_rate=0.0,
        min_cache_hit_rate=0.70,
    )
    failing = module.build_route_comparison_report(
        [("costs", report_path)],
        gate_routes=("retrieval_groundedness",),
        max_mean_duration_seconds=0.02,
        max_p95_duration_seconds=0.02,
        max_p99_duration_seconds=0.02,
        max_max_duration_seconds=0.03,
        max_mean_attempted_route_count=1.1,
        max_retrieval_use_rate=0.5,
    )
    cache_failing = module.build_route_comparison_report(
        [("costs", report_path)],
        gate_routes=("structured_state",),
        max_mean_duration_seconds=0.02,
        max_p95_duration_seconds=0.02,
        max_p99_duration_seconds=0.02,
        max_max_duration_seconds=0.03,
        max_mean_attempted_route_count=1.1,
        max_retrieval_use_rate=0.0,
        min_cache_hit_rate=0.90,
    )

    aggregate = passing["by_route"]["structured_state"]
    frontier = passing["pareto_frontier"]
    assert aggregate["mean_duration_seconds"] == pytest.approx(0.01)
    assert aggregate["p95_duration_seconds"] == pytest.approx(0.019)
    assert aggregate["p99_duration_seconds"] == pytest.approx(0.0198)
    assert aggregate["max_duration_seconds"] == pytest.approx(0.02)
    assert aggregate["mean_attempted_route_count"] == pytest.approx(1.0)
    assert aggregate["retrieval_use_rate"] == pytest.approx(0.0)
    assert passing["cache_summary"]["total"]["hit_rate"] == pytest.approx(0.75)
    assert frontier["recommended"]["route"] == "structured_state"
    assert {item["route"] for item in frontier["frontier"]} == {"fast_lexical", "structured_state"}
    assert frontier["dominated"][0]["route"] == "retrieval_groundedness"
    assert frontier["dominated"][0]["dominated_by"] == "structured_state"
    assert passing["promotion_decision"]["status"] == "promote"
    assert passing["promotion_decision"]["recommended_route"] == "structured_state"
    assert passing["promotion_decision"]["route_gate_passed"] is True
    assert passing["quality_gate"]["passed"] is True
    assert failing["quality_gate"]["passed"] is False
    assert failing["promotion_decision"]["status"] == "needs_gate_for_recommended"
    assert failing["promotion_decision"]["recommended_route"] == "structured_state"
    assert failing["promotion_decision"]["gate_checked_route"] is False
    assert {failure["metric"] for failure in failing["quality_gate"]["failures"]} == {
        "mean_duration_seconds",
        "p95_duration_seconds",
        "p99_duration_seconds",
        "max_duration_seconds",
        "mean_attempted_route_count",
        "retrieval_use_rate",
    }
    assert cache_failing["quality_gate"]["passed"] is False
    assert cache_failing["quality_gate"]["failures"][0]["metric"] == "cache_hit_rate"
    assert cache_failing["promotion_decision"]["status"] == "blocked_by_gate"


def test_compare_verifier_routes_quality_gate_fails_closed_for_empty_or_nonfinite_metrics():
    module = importlib.import_module("benchmarks.compare_verifier_routes")

    empty = module.build_route_quality_gate(
        {"structured_state": {"selected": 0, "decision_accuracy": 1.0}},
        min_selected=1,
        min_decision_accuracy=0.90,
    )
    nonfinite = module.build_route_quality_gate(
        {"structured_state": {"selected": 2, "decision_accuracy": float("nan")}},
        routes=("structured_state",),
        min_decision_accuracy=0.90,
    )

    assert empty["passed"] is False
    assert empty["failures"][0]["metric"] == "eligible_routes"
    assert nonfinite["passed"] is False
    assert nonfinite["failures"][0]["metric"] == "decision_accuracy"
    assert nonfinite["failures"][0]["value"] is None
    assert nonfinite["failures"][0]["raw_value"] == "nan"


def test_compare_verifier_routes_cli_exits_nonzero_on_gate_failure(tmp_path):
    module = importlib.import_module("benchmarks.compare_verifier_routes")
    report_path = tmp_path / "routes.json"
    output_path = tmp_path / "route-gate.json"
    report_path.write_text(
        json.dumps({
            "runs": [
                {
                    "name": "qa",
                    "route_quality": {
                        "structured_qa": {
                            "selected": 4,
                            "n_true": 2,
                            "n_false": 2,
                            "label_status_matrix": {
                                "true": {"supported": 2, "refuted": 0, "insufficient_evidence": 0},
                                "false": {"supported": 1, "refuted": 1, "insufficient_evidence": 0},
                            },
                            "false_refuted_rate": 0.5,
                            "false_supported_rate": 0.5,
                            "decision_accuracy": 0.75,
                        }
                    },
                    "alphas": {"0.1": {"route_control_impact": {}}},
                }
            ]
        }),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc_info:
        module.main([
            "--report",
            f"qa={report_path}",
            "--gate-route",
            "structured_qa",
            "--min-decision-accuracy",
            "0.90",
            "--max-false-supported-rate",
            "0.10",
            "--json",
            str(output_path),
            "--fail-on-gate",
        ])

    assert exc_info.value.code == 1
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["quality_gate"]["passed"] is False
    assert payload["quality_gate"]["failures"][0]["route"] == "structured_qa"
    assert payload["promotion_decision"]["status"] == "blocked_by_gate"
    assert payload["promotion_decision"]["recommended_route"] == "structured_qa"


def test_compare_verifier_routes_cli_can_fail_on_missing_promotion_gate(tmp_path):
    module = importlib.import_module("benchmarks.compare_verifier_routes")
    report_path = tmp_path / "routes.json"
    output_path = tmp_path / "promotion.json"
    report_path.write_text(
        json.dumps({
            "runs": [
                {
                    "name": "qa",
                    "route_quality": {
                        "structured_qa": {
                            "selected": 4,
                            "n_true": 2,
                            "n_false": 2,
                            "label_status_matrix": {
                                "true": {"supported": 2, "refuted": 0, "insufficient_evidence": 0},
                                "false": {"supported": 0, "refuted": 2, "insufficient_evidence": 0},
                            },
                            "false_refuted_rate": 1.0,
                            "false_supported_rate": 0.0,
                            "decision_accuracy": 1.0,
                        }
                    },
                    "alphas": {"0.1": {"route_control_impact": {}}},
                }
            ]
        }),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc_info:
        module.main([
            "--report",
            f"qa={report_path}",
            "--json",
            str(output_path),
            "--fail-on-promotion",
        ])

    assert exc_info.value.code == 1
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["promotion_decision"]["status"] == "needs_gate"


def _write_adapter_promotion_route_report(path: Path) -> None:
    path.write_text(
        json.dumps({
            "runs": [
                {
                    "name": "routes",
                    "route_quality": {
                        "structured_state": {
                            "selected": 4,
                            "n_true": 2,
                            "n_false": 2,
                            "label_status_matrix": {
                                "true": {"supported": 2, "refuted": 0, "insufficient_evidence": 0},
                                "false": {"supported": 0, "refuted": 2, "insufficient_evidence": 0},
                            },
                            "false_refuted_rate": 1.0,
                            "false_supported_rate": 0.0,
                            "decision_accuracy": 1.0,
                            "duration_observations": 4,
                            "total_duration_seconds": 0.04,
                            "mean_duration_seconds": 0.01,
                            "p95_duration_seconds": 0.019,
                            "p99_duration_seconds": 0.0198,
                            "max_duration_seconds": 0.02,
                            "attempted_route_count_observations": 4,
                            "total_attempted_route_count": 4,
                            "mean_attempted_route_count": 1.0,
                            "used_retrieval_count": 0,
                            "retrieval_use_rate": 0.0,
                        }
                    },
                    "cache_stats": {
                        "total": {"size": 2, "hits": 7, "misses": 1, "requests": 8, "hit_rate": 0.875}
                    },
                    "alphas": {"0.1": {"route_control_impact": {}}},
                }
            ]
        }),
        encoding="utf-8",
    )


def test_run_adapter_promotion_workflow_promotes_gated_route(tmp_path):
    module = importlib.import_module("benchmarks.run_adapter_promotion_workflow")
    route_source_path = tmp_path / "routes.json"
    route_report_path = tmp_path / "route-comparison.json"
    _write_adapter_promotion_route_report(route_source_path)

    payload = module.run_adapter_promotion_workflow(
        module.AdapterPromotionWorkflowConfig(
            reports=(("routes", route_source_path),),
            route_report_path=route_report_path,
            gate_routes=("structured_state",),
            min_decision_accuracy=0.99,
            max_false_supported_rate=0.0,
            min_false_refuted_rate=0.99,
            max_mean_duration_seconds=0.02,
            max_p95_duration_seconds=0.02,
            max_p99_duration_seconds=0.02,
            max_max_duration_seconds=0.03,
            max_mean_attempted_route_count=1.1,
            max_retrieval_use_rate=0.0,
            min_cache_hit_rate=0.80,
            compact_json=True,
        )
    )
    written_route_report = route_report_path.read_text(encoding="utf-8")

    assert payload["decision"]["status"] == "promote"
    assert payload["decision"]["recommended_route"] == "structured_state"
    assert payload["route_comparison"]["promotion_decision"]["status"] == "promote"
    assert route_report_path.exists()
    assert json.loads(written_route_report)["promotion_decision"]["status"] == "promote"
    assert "\n  " not in written_route_report
    assert ": " not in written_route_report


def test_run_adapter_promotion_workflow_cli_fails_on_blocked_route(tmp_path):
    module = importlib.import_module("benchmarks.run_adapter_promotion_workflow")
    route_source_path = tmp_path / "routes.json"
    route_report_path = tmp_path / "route-comparison.json"
    workflow_report_path = tmp_path / "workflow.json"
    _write_adapter_promotion_route_report(route_source_path)

    with pytest.raises(SystemExit) as exc_info:
        module.main([
            "--report",
            f"routes={route_source_path}",
            "--route-report-json",
            str(route_report_path),
            "--json",
            str(workflow_report_path),
            "--fail-on-blocked",
        ])

    assert exc_info.value.code == 1
    payload = json.loads(workflow_report_path.read_text(encoding="utf-8"))
    assert payload["decision"]["status"] == "blocked"
    assert payload["decision"]["blocking_reasons"][0]["gate"] == "route_promotion"
    assert payload["route_comparison"]["promotion_decision"]["status"] == "needs_gate"


def test_run_adapter_promotion_workflow_can_include_registry_baseline_gate(tmp_path):
    module = importlib.import_module("benchmarks.run_adapter_promotion_workflow")
    from eigentruth.registry import ArtifactRegistry, build_artifact_manifest

    route_source_path = tmp_path / "routes.json"
    route_report_path = tmp_path / "route-comparison.json"
    _write_adapter_promotion_route_report(route_source_path)
    baseline_profile = tmp_path / "baseline-profile.json"
    candidate_profile = tmp_path / "candidate-profile.json"
    baseline_profile.write_text(
        json.dumps({
            "total_seconds": 100.0,
            "phases": {"forced_answer_forward": 80.0},
            "summary": {"bottleneck": "forced_answer_forward", "groups": {}, "throughput": {}},
        }),
        encoding="utf-8",
    )
    candidate_profile.write_text(
        json.dumps({
            "total_seconds": 104.0,
            "phases": {"forced_answer_forward": 82.0},
            "summary": {"bottleneck": "forced_answer_forward", "groups": {}, "throughput": {}},
        }),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "artifact-manifest.json"
    manifest_path.write_text(
        json.dumps(build_artifact_manifest({"profiles.uncached": baseline_profile}, root=tmp_path)),
        encoding="utf-8",
    )
    registry_path = tmp_path / "registry.json"
    ArtifactRegistry.load_json(registry_path).record_benchmark_manifest(
        name="adapter-baseline",
        path=manifest_path,
        version="0.4",
    ).save_json()

    payload = module.run_adapter_promotion_workflow(
        module.AdapterPromotionWorkflowConfig(
            reports=(("routes", route_source_path),),
            route_report_path=route_report_path,
            gate_routes=("structured_state",),
            min_decision_accuracy=0.99,
            max_false_supported_rate=0.0,
            min_false_refuted_rate=0.99,
            max_mean_duration_seconds=0.02,
            max_p95_duration_seconds=0.02,
            max_p99_duration_seconds=0.02,
            max_max_duration_seconds=0.03,
            max_mean_attempted_route_count=1.1,
            max_retrieval_use_rate=0.0,
            min_cache_hit_rate=0.80,
            registry_path=registry_path,
            baseline_name="adapter-baseline",
            baseline_version="0.4",
            candidate_profiles=(("candidate", candidate_profile),),
            max_total_ratio=1.05,
        )
    )

    assert payload["decision"]["status"] == "promote"
    assert payload["decision"]["registry_baseline_checked"] is True
    assert payload["decision"]["registry_baseline_passed"] is True
    assert payload["registry_baseline_comparison"]["comparison"]["regression_gate"]["passed"] is True


def test_run_adapter_family_matrix_promotes_all_fixture_routes(tmp_path):
    module = importlib.import_module("benchmarks.run_adapter_family_matrix")
    matrix_path = tmp_path / "matrix.json"

    payload = module.run_adapter_family_matrix(
        module.AdapterFamilyMatrixConfig(
            output_dir=tmp_path,
            matrix_report_path=matrix_path,
            n_records=8,
            alpha=0.2,
            compact_json=True,
        )
    )
    written = json.loads(matrix_path.read_text(encoding="utf-8"))
    by_route = payload["route_comparison"]["by_route"]
    families = {item["route"]: item for item in payload["families"]}

    assert set(families) == {"structured_qa", "structured_state", "state_transition"}
    assert payload["promotion_decision"]["status"] == "promote"
    assert payload["route_comparison"]["quality_gate"]["passed"] is True
    for route, item in families.items():
        assert item["status"] == "promote"
        assert item["selected"] == 8
        assert item["decision_accuracy"] == pytest.approx(1.0)
        assert item["false_supported_rate"] == pytest.approx(0.0)
        assert item["false_refuted_rate"] == pytest.approx(1.0)
        assert by_route[route]["selected"] == 8
        assert Path(item["verifier_report_path"]).exists()
        assert Path(item["promotion_report_path"]).exists()
    assert written["promotion_decision"]["status"] == "promote"
    assert Path(payload["route_comparison_path"]).exists()


def test_refresh_verifier_route_artifacts_writes_new_schema_and_promotion(tmp_path):
    module = importlib.import_module("benchmarks.refresh_verifier_route_artifacts")
    scores_path = tmp_path / "scores.json"
    qa_path = tmp_path / "qa.json"
    verifier_report_path = tmp_path / "verifier-report.json"
    route_report_path = tmp_path / "route-comparison.json"
    promotion_report_path = tmp_path / "promotion.json"
    scores_path.write_text(
        json.dumps({
            "config": {"model": "synthetic", "layer": -1},
            "labels": [0, 0, 1, 1],
            "scores": {"truth_proj": [0.1, 0.2, 0.8, 0.9]},
            "statements": [
                {"question": "Q1?", "answer": "A1", "text": "Q1? A1"},
                {"question": "Q2?", "answer": "A2", "text": "Q2? A2"},
                {"question": "Q1?", "answer": "Wrong A1", "text": "Q1? Wrong A1"},
                {"question": "Q2?", "answer": "Wrong A2", "text": "Q2? Wrong A2"},
            ],
        }),
        encoding="utf-8",
    )
    qa_path.write_text(
        json.dumps({
            "documents": [
                {"question": "Q1?", "answer": "A1", "source": "qa:q1"},
                {"question": "Q2?", "answer": "A2", "source": "qa:q2"},
            ],
        }),
        encoding="utf-8",
    )

    payload = module.refresh_verifier_route_artifacts(
        module.VerifierRouteArtifactRefreshConfig(
            score_dumps=(("synthetic", scores_path),),
            verifier_report_path=verifier_report_path,
            qa_corpus_path=qa_path,
            alphas=(0.2,),
            repeats=1,
            promotion_report_path=promotion_report_path,
            route_report_path=route_report_path,
            promotion_gate_routes=("structured_qa",),
            promotion_gate_min_selected=4,
            min_decision_accuracy=1.0,
            max_false_supported_rate=0.0,
            min_false_refuted_rate=1.0,
            max_mean_duration_seconds=1.0,
            max_p99_duration_seconds=1.0,
            max_max_duration_seconds=1.0,
            max_mean_attempted_route_count=1.1,
            max_retrieval_use_rate=0.0,
        )
    )
    verifier_report = json.loads(verifier_report_path.read_text(encoding="utf-8"))
    promotion_report = json.loads(promotion_report_path.read_text(encoding="utf-8"))

    summary_run = payload["verifier_report_summary"]["runs"][0]
    summary_route = summary_run["routes"]["structured_qa"]
    assert summary_route["selected"] == 4
    assert summary_route["p95_duration_seconds"] >= 0.0
    assert summary_route["max_duration_seconds"] >= 0.0
    assert summary_route["mean_attempted_route_count"] == pytest.approx(1.0)
    assert summary_route["retrieval_use_rate"] == pytest.approx(0.0)
    assert summary_run["cache_stats"]["qa_verifier"]["requests"] == 4
    assert summary_run["cache_stats"]["total"]["requests"] >= 4
    assert verifier_report["runs"][0]["route_quality"]["structured_qa"]["decision_accuracy"] == pytest.approx(1.0)
    assert verifier_report["runs"][0]["cache_stats"]["qa_verifier"]["requests"] == 4
    assert payload["promotion"]["decision"]["status"] == "promote"
    assert promotion_report["decision"]["recommended_route"] == "structured_qa"
    assert route_report_path.exists()


def test_refresh_verifier_route_artifacts_promotes_structured_state_route(tmp_path):
    builder = importlib.import_module("benchmarks.build_domain_state_fixture")
    module = importlib.import_module("benchmarks.refresh_verifier_route_artifacts")
    scores_path = tmp_path / "domain_scores.json"
    claims_path = tmp_path / "domain_claims.json"
    state_path = tmp_path / "domain_state.json"
    verifier_report_path = tmp_path / "state-verifier-report.json"
    route_report_path = tmp_path / "state-route-comparison.json"
    promotion_report_path = tmp_path / "state-promotion.json"

    builder.run(SimpleNamespace(
        scores_output=str(scores_path),
        claims_output=str(claims_path),
        state_output=str(state_path),
        n_records=8,
        signal="truth_proj",
    ))

    payload = module.refresh_verifier_route_artifacts(
        module.VerifierRouteArtifactRefreshConfig(
            score_dumps=(("orders", scores_path),),
            verifier_report_path=verifier_report_path,
            claims_path=claims_path,
            state_path=state_path,
            alphas=(0.2,),
            repeats=1,
            promotion_report_path=promotion_report_path,
            route_report_path=route_report_path,
            promotion_gate_routes=("structured_state",),
            promotion_gate_min_selected=8,
            min_decision_accuracy=1.0,
            max_false_supported_rate=0.0,
            min_false_refuted_rate=1.0,
            max_mean_duration_seconds=1.0,
            max_p99_duration_seconds=1.0,
            max_max_duration_seconds=1.0,
            max_mean_attempted_route_count=1.1,
            max_retrieval_use_rate=0.0,
            compact_json=True,
        )
    )
    verifier_report_text = verifier_report_path.read_text(encoding="utf-8")
    route_report_text = route_report_path.read_text(encoding="utf-8")
    promotion_report_text = promotion_report_path.read_text(encoding="utf-8")
    verifier_report = json.loads(verifier_report_text)
    promotion_report = json.loads(promotion_report_text)
    run = verifier_report["runs"][0]
    route_quality = run["route_quality"]["structured_state"]

    summary_run = payload["verifier_report_summary"]["runs"][0]
    summary_route = summary_run["routes"]["structured_state"]
    assert summary_route["selected"] == 8
    assert summary_route["p95_duration_seconds"] >= 0.0
    assert summary_route["max_duration_seconds"] >= 0.0
    assert summary_route["mean_attempted_route_count"] == pytest.approx(1.0)
    assert summary_route["retrieval_use_rate"] == pytest.approx(0.0)
    assert summary_run["cache_stats"]["state_verifier"]["requests"] == 8
    assert summary_run["cache_stats"]["total"]["requests"] >= 8
    assert run["route_summary"]["selected_counts"] == {"structured_state": 8}
    assert run["cache_stats"]["state_verifier"]["requests"] == 8
    assert route_quality["decision_accuracy"] == pytest.approx(1.0)
    assert route_quality["false_supported_rate"] == pytest.approx(0.0)
    assert route_quality["false_refuted_rate"] == pytest.approx(1.0)
    assert payload["promotion"]["decision"]["status"] == "promote"
    assert promotion_report["decision"]["recommended_route"] == "structured_state"
    assert route_report_path.exists()
    assert "\n  " not in verifier_report_text
    assert "\n  " not in route_report_text
    assert "\n  " not in promotion_report_text


def test_refresh_verifier_route_artifacts_promotes_state_transition_route(tmp_path):
    builder = importlib.import_module("benchmarks.build_transition_fixture")
    module = importlib.import_module("benchmarks.refresh_verifier_route_artifacts")
    scores_path = tmp_path / "transition_scores.json"
    claims_path = tmp_path / "transition_claims.json"
    state_path = tmp_path / "transition_state.json"
    verifier_report_path = tmp_path / "transition-verifier-report.json"
    route_report_path = tmp_path / "transition-route-comparison.json"
    promotion_report_path = tmp_path / "transition-promotion.json"

    builder.run(SimpleNamespace(
        scores_output=str(scores_path),
        claims_output=str(claims_path),
        state_output=str(state_path),
        n_records=8,
        signal="truth_proj",
    ))

    payload = module.refresh_verifier_route_artifacts(
        module.VerifierRouteArtifactRefreshConfig(
            score_dumps=(("transitions", scores_path),),
            verifier_report_path=verifier_report_path,
            claims_path=claims_path,
            state_path=state_path,
            alphas=(0.2,),
            repeats=1,
            promotion_report_path=promotion_report_path,
            route_report_path=route_report_path,
            promotion_gate_routes=("state_transition",),
            promotion_gate_min_selected=8,
            min_decision_accuracy=1.0,
            max_false_supported_rate=0.0,
            min_false_refuted_rate=1.0,
            max_mean_duration_seconds=1.0,
            max_p99_duration_seconds=1.0,
            max_max_duration_seconds=1.0,
            max_mean_attempted_route_count=1.1,
            max_retrieval_use_rate=0.0,
            compact_json=True,
        )
    )
    verifier_report_text = verifier_report_path.read_text(encoding="utf-8")
    route_report_text = route_report_path.read_text(encoding="utf-8")
    promotion_report_text = promotion_report_path.read_text(encoding="utf-8")
    verifier_report = json.loads(verifier_report_text)
    promotion_report = json.loads(promotion_report_text)
    run = verifier_report["runs"][0]
    route_quality = run["route_quality"]["state_transition"]

    summary_run = payload["verifier_report_summary"]["runs"][0]
    summary_route = summary_run["routes"]["state_transition"]
    assert summary_route["selected"] == 8
    assert summary_route["mean_attempted_route_count"] == pytest.approx(1.0)
    assert summary_route["retrieval_use_rate"] == pytest.approx(0.0)
    assert summary_run["cache_stats"]["transition_verifier"]["requests"] == 8
    assert run["route_summary"]["selected_counts"] == {"state_transition": 8}
    assert run["cache_stats"]["transition_verifier"]["requests"] == 8
    assert route_quality["decision_accuracy"] == pytest.approx(1.0)
    assert route_quality["false_supported_rate"] == pytest.approx(0.0)
    assert route_quality["false_refuted_rate"] == pytest.approx(1.0)
    assert payload["promotion"]["decision"]["status"] == "promote"
    assert promotion_report["decision"]["recommended_route"] == "state_transition"
    assert route_report_path.exists()
    assert "\n  " not in verifier_report_text
    assert "\n  " not in route_report_text
    assert "\n  " not in promotion_report_text


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


def test_registry_baseline_smoke_writes_pass_and_expected_failure_reports(tmp_path):
    module = importlib.import_module("benchmarks.registry_baseline_smoke")

    payload = module.build_registry_baseline_smoke(tmp_path)
    pass_report = payload["pass_report"]
    failure_report = payload["expected_failure_report"]

    assert (tmp_path / "artifact-manifest.json").exists()
    assert (tmp_path / "registry.json").exists()
    assert (tmp_path / "registry_baseline_gate_pass_report.json").exists()
    assert (tmp_path / "registry_baseline_gate_expected_failure_report.json").exists()
    assert pass_report["registry_baseline"]["verification"]["passed"] is True
    assert pass_report["comparison"]["regression_gate"]["passed"] is True
    assert failure_report["comparison"]["regression_gate"]["passed"] is False
    assert failure_report["comparison"]["regression_gate"]["failures"][0]["run"] == "regression"


def test_verify_artifact_manifest_cli_reports_mismatch(tmp_path):
    module = importlib.import_module("benchmarks.verify_artifact_manifest")
    from eigentruth.registry import build_artifact_manifest

    data_path = tmp_path / "result.json"
    data_path.write_text('{"ok": true}\n', encoding="utf-8")
    manifest_path = tmp_path / "artifact-manifest.json"
    manifest_path.write_text(
        json.dumps(build_artifact_manifest({"result": data_path}, root=tmp_path)),
        encoding="utf-8",
    )

    clean = module.verify_manifest_file(manifest_path)
    assert clean["passed"] is True

    data_path.write_text('{"ok": false, "changed": true}\n', encoding="utf-8")
    report_path = tmp_path / "verification.json"
    with pytest.raises(SystemExit) as exc_info:
        module.main([
            "--manifest",
            str(manifest_path),
            "--json",
            str(report_path),
        ])

    assert exc_info.value.code == 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["passed"] is False
    assert report["failures"][0]["name"] == "result"
    assert {failure["field"] for failure in report["failures"]} >= {"sha256", "size_bytes"}


def test_promote_artifact_manifest_registers_verified_manifest(tmp_path):
    module = importlib.import_module("benchmarks.promote_artifact_manifest")
    from eigentruth.registry import ArtifactRegistry, build_artifact_manifest

    data_path = tmp_path / "result.json"
    data_path.write_text('{"ok": true}\n', encoding="utf-8")
    manifest_path = tmp_path / "artifact-manifest.json"
    manifest_path.write_text(
        json.dumps(build_artifact_manifest({"result": data_path}, root=tmp_path, metadata={"runner": "unit"})),
        encoding="utf-8",
    )
    registry_path = tmp_path / "registry" / "registry.json"
    verification_path = tmp_path / "reports" / "verification.json"

    payload = module.promote_artifact_manifest(
        manifest_path=manifest_path,
        registry_path=registry_path,
        name="unit-baseline",
        version="0.3",
        verification_report_path=verification_path,
        metadata={"machine": "local"},
    )
    registry = ArtifactRegistry.load_json(registry_path)
    manifest_record = registry.get("benchmark_manifest:unit-baseline:0.3")
    verification_record = registry.get("manifest_verification:unit-baseline-verification:0.3")

    assert payload["verification"]["passed"] is True
    assert verification_path.exists()
    assert manifest_record.path == str(manifest_path)
    assert manifest_record.metadata["verified"] is True
    assert manifest_record.metadata["machine"] == "local"
    assert manifest_record.metadata["manifest_metadata"] == {"runner": "unit"}
    assert verification_record.path == str(verification_path)
    assert verification_record.metadata["passed"] is True


def test_promote_artifact_manifest_rejects_drift_by_default(tmp_path):
    module = importlib.import_module("benchmarks.promote_artifact_manifest")
    from eigentruth.registry import ArtifactRegistry, build_artifact_manifest

    data_path = tmp_path / "result.json"
    data_path.write_text('{"ok": true}\n', encoding="utf-8")
    manifest_path = tmp_path / "artifact-manifest.json"
    manifest_path.write_text(
        json.dumps(build_artifact_manifest({"result": data_path}, root=tmp_path)),
        encoding="utf-8",
    )
    data_path.write_text('{"ok": false, "changed": true}\n', encoding="utf-8")
    registry_path = tmp_path / "registry.json"
    verification_path = tmp_path / "verification.json"

    with pytest.raises(ValueError):
        module.promote_artifact_manifest(
            manifest_path=manifest_path,
            registry_path=registry_path,
            name="unit-baseline",
            version="0.3",
            verification_report_path=verification_path,
        )

    assert verification_path.exists()
    assert ArtifactRegistry.load_json(registry_path).list_records() == ()


def test_compare_registry_baseline_uses_verified_manifest_profile(tmp_path):
    module = importlib.import_module("benchmarks.compare_registry_baseline")
    from eigentruth.registry import ArtifactRegistry, build_artifact_manifest

    baseline_profile = tmp_path / "profile-uncached.json"
    candidate_profile = tmp_path / "profile-candidate.json"
    baseline_profile.write_text(
        json.dumps({"total_seconds": 100.0, "phases": {"forced_answer_forward": 80.0}}),
        encoding="utf-8",
    )
    candidate_profile.write_text(
        json.dumps({"total_seconds": 105.0, "phases": {"forced_answer_forward": 82.0}}),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "artifact-manifest.json"
    manifest_path.write_text(
        json.dumps(build_artifact_manifest({"profiles.uncached": baseline_profile}, root=tmp_path)),
        encoding="utf-8",
    )
    registry_path = tmp_path / "registry.json"
    ArtifactRegistry.load_json(registry_path).record_benchmark_manifest(
        name="unit-baseline",
        path=manifest_path,
        version="0.3",
    ).save_json()

    payload = module.compare_registry_baseline(
        registry_path=registry_path,
        baseline_name="unit-baseline",
        baseline_version="0.3",
        candidate_profiles=(("candidate", candidate_profile),),
        max_total_ratio=1.10,
    )

    assert payload["registry_baseline"]["verification"]["passed"] is True
    assert payload["registry_baseline"]["profile_path"] == str(baseline_profile)
    assert payload["comparison"]["regression_gate"]["passed"] is True
    assert payload["comparison"]["runs"][1]["total_delta"]["ratio_to_baseline"] == pytest.approx(1.05)


def test_compare_registry_baseline_rejects_drift_by_default(tmp_path):
    module = importlib.import_module("benchmarks.compare_registry_baseline")
    from eigentruth.registry import ArtifactRegistry, build_artifact_manifest

    baseline_profile = tmp_path / "profile-uncached.json"
    candidate_profile = tmp_path / "profile-candidate.json"
    baseline_profile.write_text(
        json.dumps({"total_seconds": 100.0, "phases": {"forced_answer_forward": 80.0}}),
        encoding="utf-8",
    )
    candidate_profile.write_text(
        json.dumps({"total_seconds": 101.0, "phases": {"forced_answer_forward": 81.0}}),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "artifact-manifest.json"
    manifest_path.write_text(
        json.dumps(build_artifact_manifest({"profiles.uncached": baseline_profile}, root=tmp_path)),
        encoding="utf-8",
    )
    baseline_profile.write_text(
        json.dumps({"total_seconds": 200.0, "phases": {"forced_answer_forward": 180.0}}),
        encoding="utf-8",
    )
    registry_path = tmp_path / "registry.json"
    ArtifactRegistry.load_json(registry_path).record_benchmark_manifest(
        name="unit-baseline",
        path=manifest_path,
        version="0.3",
    ).save_json()

    with pytest.raises(ValueError, match="verification failed"):
        module.compare_registry_baseline(
            registry_path=registry_path,
            baseline_name="unit-baseline",
            baseline_version="0.3",
            candidate_profiles=(("candidate", candidate_profile),),
        )


def test_compare_registry_baseline_cli_exits_on_regression(tmp_path):
    module = importlib.import_module("benchmarks.compare_registry_baseline")
    from eigentruth.registry import ArtifactRegistry, build_artifact_manifest

    baseline_profile = tmp_path / "profile-uncached.json"
    candidate_profile = tmp_path / "profile-candidate.json"
    report_path = tmp_path / "reports" / "registry-comparison.json"
    baseline_profile.write_text(
        json.dumps({"total_seconds": 100.0, "phases": {"forced_answer_forward": 80.0}}),
        encoding="utf-8",
    )
    candidate_profile.write_text(
        json.dumps({"total_seconds": 130.0, "phases": {"forced_answer_forward": 100.0}}),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "artifact-manifest.json"
    manifest_path.write_text(
        json.dumps(build_artifact_manifest({"profiles.uncached": baseline_profile}, root=tmp_path)),
        encoding="utf-8",
    )
    registry_path = tmp_path / "registry.json"
    ArtifactRegistry.load_json(registry_path).record_benchmark_manifest(
        name="unit-baseline",
        path=manifest_path,
        version="0.3",
    ).save_json()

    with pytest.raises(SystemExit) as exc_info:
        module.main([
            "--registry",
            str(registry_path),
            "--baseline-name",
            "unit-baseline",
            "--baseline-version",
            "0.3",
            "--candidate-profile",
            f"candidate={candidate_profile}",
            "--max-total-ratio",
            "1.10",
            "--json",
            str(report_path),
            "--fail-on-regression",
        ])

    assert exc_info.value.code == 1
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["comparison"]["regression_gate"]["passed"] is False
    assert payload["comparison"]["regression_gate"]["failures"][0]["run"] == "candidate"


def test_compare_registry_baseline_resolves_nested_manifest_profile(tmp_path):
    module = importlib.import_module("benchmarks.compare_registry_baseline")
    from eigentruth.registry import ArtifactRegistry, build_artifact_manifest

    cell_dir = tmp_path / "cell"
    cell_dir.mkdir()
    baseline_profile = cell_dir / "profile-uncached.json"
    candidate_profile = tmp_path / "profile-candidate.json"
    baseline_profile.write_text(
        json.dumps({"total_seconds": 100.0, "phases": {"forced_answer_forward": 80.0}}),
        encoding="utf-8",
    )
    candidate_profile.write_text(
        json.dumps({"total_seconds": 102.0, "phases": {"forced_answer_forward": 81.0}}),
        encoding="utf-8",
    )
    triplet_manifest_path = cell_dir / "artifact-manifest.json"
    triplet_manifest_path.write_text(
        json.dumps(build_artifact_manifest({"profiles.uncached": baseline_profile}, root=cell_dir)),
        encoding="utf-8",
    )
    root_manifest_path = tmp_path / "artifact-manifest.json"
    root_manifest_path.write_text(
        json.dumps(build_artifact_manifest({"cells.unit.triplet_manifest": triplet_manifest_path}, root=tmp_path)),
        encoding="utf-8",
    )
    registry_path = tmp_path / "registry.json"
    ArtifactRegistry.load_json(registry_path).record_benchmark_manifest(
        name="matrix-baseline",
        path=root_manifest_path,
        version="0.3",
    ).save_json()

    payload = module.compare_registry_baseline(
        registry_path=registry_path,
        baseline_name="matrix-baseline",
        baseline_version="0.3",
        baseline_profile_artifact="cells.unit.triplet_manifest::profiles.uncached",
        candidate_profiles=(("candidate", candidate_profile),),
        max_total_ratio=1.05,
    )

    assert payload["registry_baseline"]["verification"]["passed"] is True
    assert payload["registry_baseline"]["profile_path"] == str(baseline_profile)
    assert payload["comparison"]["regression_gate"]["passed"] is True


def test_run_registry_baseline_workflow_dry_run_promotes_matrix_manifest(tmp_path):
    module = importlib.import_module("benchmarks.run_registry_baseline_workflow")
    from eigentruth.registry import ArtifactRegistry

    output_dir = tmp_path / "workflow"
    registry_path = tmp_path / "registry" / "registry.json"
    report_path = tmp_path / "workflow-report.json"
    payload = module.run(
        SimpleNamespace(
            output_dir=str(output_dir),
            registry=str(registry_path),
            name="qwen-mini-dry-run",
            version="0.3",
            verification_report=None,
            metadata=["machine=unit"],
            model="Qwen/Qwen2.5-0.5B-Instruct",
            dtype="float32",
            layers="-12",
            batch_sizes="1,2",
            hidden_state_captures="outputs",
            limit=4,
            manifold_questions=2,
            max_length=32,
            eval_reps_cache_shard_size=2,
            cached_max_total_ratio=1.10,
            cache_only_max_total_ratio=0.35,
            progress_every=0,
            python=sys.executable,
            no_length_bucketed_batches=False,
            real_truthfulqa=False,
            shared_cache_dir=str(tmp_path / "shared-cache"),
            matrix_mode="rescore",
            clean=True,
            dry_run=True,
            allow_promotion_failures=False,
            candidate_profile=[],
            baseline_profile_artifact="profiles.uncached",
            allow_unverified_compare=False,
            max_total_ratio=None,
            max_run_total_ratio=[],
            max_phase_ratio=[],
            min_throughput_ratio=[],
            fail_on_regression=False,
            json=str(report_path),
        )
    )
    registry = ArtifactRegistry.load_json(registry_path)
    record = registry.get("benchmark_manifest:qwen-mini-dry-run:0.3")

    assert payload["matrix"]["dry_run"] is True
    assert payload["promotion"]["verification"]["passed"] is True
    assert payload["comparison"] is None
    assert record.metadata["machine"] == "unit"
    assert record.metadata["manifest_metadata"]["runner"] == "run_cache_profile_matrix"
    assert report_path.exists()


def test_run_registry_baseline_workflow_auto_resolves_first_uncached_cell():
    module = importlib.import_module("benchmarks.run_registry_baseline_workflow")

    artifact = module._resolve_workflow_baseline_profile_artifact(
        "auto",
        {
            "cells": [
                {
                    "id": "layer_m12_batch_2_capture_outputs",
                    "triplet": {"profiles": {"cache_only": "/tmp/profile-cache-only.json"}},
                },
                {
                    "id": "layer_m12_batch_1_capture_outputs",
                    "triplet": {
                        "profiles": {
                            "uncached": "/tmp/profile-uncached.json",
                            "cached": "/tmp/profile-cached.json",
                        }
                    },
                },
            ]
        },
    )

    assert artifact == "cells.layer_m12_batch_1_capture_outputs.triplet_manifest::profiles.uncached"


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
    assert Path(payload["artifact_manifest"]).exists()
    manifest = json.loads(Path(payload["artifact_manifest"]).read_text(encoding="utf-8"))
    assert manifest["metadata"]["runner"] == "run_cache_profile_triplet"
    assert manifest["artifacts"]["command_log"]["exists"] is True
    assert manifest["artifacts"]["caches.eval_reps_cache"]["exists"] is False
    assert payload["artifact_manifest_summary"]["missing_count"] == 3
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


def test_run_cache_profile_triplet_supports_warm_start_cache_overrides(tmp_path):
    module = importlib.import_module("benchmarks.run_cache_profile_triplet")
    shared = tmp_path / "shared"
    config = module.CacheProfileTripletConfig(
        output_dir=tmp_path / "cell",
        model="tiny-local",
        layer=-2,
        batch_size=2,
        statement_encoding_cache_path=shared / "statement-encodings.json",
        layer_stats_cache_path=shared / "layer-stats.pt",
        eval_reps_cache_path=shared / "eval-reps-cache",
        uncached_cache_mode="warm_start",
        python_executable="/python",
    )

    payload = module.run_triplet(config, clean=True, dry_run=True)
    uncached = payload["commands"]["uncached"]
    cached = payload["commands"]["cached"]
    cache_only = payload["commands"]["cache_only"]

    assert payload["uncached_cache_mode"] == "warm_start"
    assert payload["caches"]["statement_encoding_cache"] == str(shared / "statement-encodings.json")
    assert uncached[uncached.index("--statement-encoding-cache") + 1] == str(shared / "statement-encodings.json")
    assert uncached[uncached.index("--layer-stats-cache") + 1] == str(shared / "layer-stats.pt")
    assert "--eval-reps-cache" not in uncached
    assert "--refresh-statement-encoding-cache" not in uncached
    assert "--refresh-layer-stats-cache" not in uncached
    assert cached[cached.index("--eval-reps-cache") + 1] == str(shared / "eval-reps-cache")
    assert cache_only[cache_only.index("--eval-reps-cache") + 1] == str(shared / "eval-reps-cache")


def test_run_cache_profile_triplet_can_run_cache_only_subset(tmp_path):
    module = importlib.import_module("benchmarks.run_cache_profile_triplet")
    config = module.CacheProfileTripletConfig(
        output_dir=tmp_path / "cell",
        model="tiny-local",
        layer=-2,
        batch_size=2,
        run_names=("cache_only",),
        python_executable="/python",
    )

    payload = module.run_triplet(config, clean=True, dry_run=True)

    assert payload["run_names"] == ("cache_only",)
    assert tuple(payload["commands"]) == ("cache_only",)
    assert "--cache-only" in payload["commands"]["cache_only"]
    assert "uncached" not in payload["commands"]


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
    manifest = json.loads(Path(payload["artifact_manifest"]).read_text(encoding="utf-8"))

    assert [call["check"] for call in calls] == [True, True, True]
    assert payload["dry_run"] is False
    assert manifest["artifacts"]["comparison_report"]["exists"] is True
    assert manifest["artifacts"]["profiles.cache_only"]["sha256"]
    assert manifest["artifacts"]["results.cache_only"]["sha256"]
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


def test_run_cache_profile_matrix_builds_dry_run_cells(tmp_path):
    module = importlib.import_module("benchmarks.run_cache_profile_matrix")
    config = module.CacheProfileMatrixConfig(
        output_dir=tmp_path,
        model="tiny-local",
        layers=(-2, -1),
        batch_sizes=(2,),
        hidden_state_captures=("outputs", "hooks"),
        python_executable="/python",
    )

    report = module.run_matrix(config, clean=True, dry_run=True)

    assert report["dry_run"] is True
    assert Path(report["report_path"]).exists()
    assert Path(report["artifact_manifest"]).exists()
    manifest = json.loads(Path(report["artifact_manifest"]).read_text(encoding="utf-8"))
    assert manifest["metadata"]["runner"] == "run_cache_profile_matrix"
    assert manifest["artifacts"]["matrix_report"]["exists"] is True
    assert manifest["artifacts"]["cells.layer_m2_batch_2_capture_outputs.triplet_manifest"]["exists"] is True
    assert [cell["id"] for cell in report["cells"]] == [
        "layer_m2_batch_2_capture_outputs",
        "layer_m2_batch_2_capture_hooks",
        "layer_m1_batch_2_capture_outputs",
        "layer_m1_batch_2_capture_hooks",
    ]
    first = report["cells"][0]
    assert first["summary"]["dry_run"] is True
    assert "--layer -2" in first["summary"]["commands"]["uncached"]
    assert "--hidden-state-capture outputs" in first["summary"]["commands"]["uncached"]


def test_run_cache_profile_matrix_shared_cache_warm_starts_repeated_groups(tmp_path):
    module = importlib.import_module("benchmarks.run_cache_profile_matrix")
    config = module.CacheProfileMatrixConfig(
        output_dir=tmp_path / "runs",
        shared_cache_dir=tmp_path / "shared-cache",
        model="tiny-local",
        layers=(-2,),
        batch_sizes=(1, 2),
        hidden_state_captures=("outputs",),
        python_executable="/python",
    )

    report = module.run_matrix(config, clean=True, dry_run=True)
    first, second = report["cells"]
    first_commands = first["triplet"]["commands"]
    second_commands = second["triplet"]["commands"]
    first_eval_cache = first["triplet"]["caches"]["eval_reps_cache"]
    second_eval_cache = second["triplet"]["caches"]["eval_reps_cache"]

    assert report["config"]["shared_cache_dir"] == str(tmp_path / "shared-cache")
    assert report["config"]["shared_cache_root"].startswith(str(tmp_path / "shared-cache"))
    assert first["uncached_cache_mode"] == "refresh"
    assert second["uncached_cache_mode"] == "warm_start"
    assert first["shared_cache_group"] == second["shared_cache_group"]
    assert first_eval_cache == second_eval_cache
    assert "--refresh-eval-reps-cache" in first_commands["uncached"]
    assert "--eval-reps-cache" not in second_commands["uncached"]
    assert "--layer-stats-cache" in second_commands["uncached"]
    assert second_commands["cached"][second_commands["cached"].index("--eval-reps-cache") + 1] == second_eval_cache
    assert (
        second_commands["cache_only"][second_commands["cache_only"].index("--eval-reps-cache") + 1]
        == second_eval_cache
    )


def test_run_cache_profile_matrix_rescore_reuses_group_as_cache_only(tmp_path, monkeypatch):
    module = importlib.import_module("benchmarks.run_cache_profile_matrix")
    seen = []

    def write_profile(path: Path, total: float) -> None:
        path.write_text(
            json.dumps({
                "total_seconds": total,
                "phases": {"score_postprocess": total},
                "summary": {"bottleneck": "score_postprocess"},
            }),
            encoding="utf-8",
        )

    def fake_run_triplet(config, *, clean, dry_run):
        seen.append({
            "batch_size": config.batch_size,
            "run_names": tuple(config.run_names),
            "uncached_cache_mode": config.uncached_cache_mode,
            "eval_reps_cache": str(config.eval_reps_cache),
        })
        config.output_dir.mkdir(parents=True, exist_ok=True)
        profiles = {}
        results = {}
        for name in config.run_names:
            profile_path = config.profile_path(name)
            result_path = config.result_path(name)
            write_profile(profile_path, 100.0 if name == "uncached" else 10.0 + config.batch_size)
            result_path.write_text(json.dumps({"auroc": {"truth_proj": 0.91}}), encoding="utf-8")
            profiles[name] = str(profile_path)
            results[name] = str(result_path)
        comparison_path = None
        regression_gate = None
        if "uncached" in config.run_names:
            comparison_path = config.comparison_report
            comparison_path.write_text(
                json.dumps({
                    "runs": [
                        {"name": "uncached", "total_seconds": 100.0, "bottleneck": "forward"},
                        {
                            "name": "cache_only",
                            "total_seconds": 10.0 + config.batch_size,
                            "bottleneck": "score_postprocess",
                            "total_delta": {
                                "speedup_vs_baseline": 100.0 / (10.0 + config.batch_size),
                                "ratio_to_baseline": (10.0 + config.batch_size) / 100.0,
                            },
                        },
                    ],
                    "fastest": {"name": "cache_only"},
                    "regression_gate": {"passed": True},
                }),
                encoding="utf-8",
            )
            regression_gate = {"passed": True}
        return {
            "dry_run": False,
            "output_dir": str(config.output_dir),
            "profiles": profiles,
            "results": results,
            "comparison_report": str(comparison_path) if comparison_path is not None else None,
            "comparison_skipped_reason": (
                None if comparison_path is not None else "baseline run 'uncached' was not executed"
            ),
            "regression_gate": regression_gate,
            "caches": {"eval_reps_cache": str(config.eval_reps_cache)},
            "uncached_cache_mode": config.uncached_cache_mode,
            "run_names": tuple(config.run_names),
        }

    monkeypatch.setattr(module, "run_triplet", fake_run_triplet)
    config = module.CacheProfileMatrixConfig(
        output_dir=tmp_path / "runs",
        shared_cache_dir=tmp_path / "shared-cache",
        model="tiny-local",
        layers=(-2,),
        batch_sizes=(1, 2),
        hidden_state_captures=("outputs",),
        matrix_mode="rescore",
    )

    report = module.run_matrix(config, clean=True, dry_run=False)
    first, second = report["cells"]

    assert report["config"]["matrix_mode"] == "rescore"
    assert seen[0]["run_names"] == ("uncached", "cached", "cache_only")
    assert seen[0]["uncached_cache_mode"] == "refresh"
    assert seen[1]["run_names"] == ("cache_only",)
    assert seen[1]["uncached_cache_mode"] == "warm_start"
    assert seen[0]["eval_reps_cache"] == seen[1]["eval_reps_cache"]
    assert first["shared_cache_group"] == second["shared_cache_group"]
    assert second["summary"]["regression_gate"] is None
    assert second["summary"]["comparison_skipped_reason"] == "baseline run 'uncached' was not executed"
    assert second["summary"]["totals"]["cache_only"]["total_seconds"] == pytest.approx(12.0)
    assert second["summary"]["truth_proj_auroc"] == pytest.approx(0.91)
    assert report["leaderboard"][0]["gate_passed"] is True
    assert report["leaderboard"][1]["gate_passed"] is None


def test_run_cache_profile_matrix_summarizes_reports(tmp_path, monkeypatch):
    module = importlib.import_module("benchmarks.run_cache_profile_matrix")
    seen = []

    def fake_run_triplet(config, *, clean, dry_run):
        seen.append({"layer": config.layer, "batch_size": config.batch_size, "capture": config.hidden_state_capture})
        config.output_dir.mkdir(parents=True, exist_ok=True)
        comparison_path = config.output_dir / "cache-profile-comparison.json"
        result_path = config.output_dir / "result-cache_only.json"
        cache_only_total = 10.0 + abs(config.layer) + config.batch_size
        comparison_path.write_text(
            json.dumps({
                "runs": [
                    {"name": "uncached", "total_seconds": 100.0, "bottleneck": "forward"},
                    {
                        "name": "cache_only",
                        "total_seconds": cache_only_total,
                        "bottleneck": "load_data",
                        "total_delta": {
                            "speedup_vs_baseline": 100.0 / cache_only_total,
                            "ratio_to_baseline": cache_only_total / 100.0,
                        },
                    },
                ],
                "fastest": {"name": "cache_only", "total_seconds": cache_only_total},
                "regression_gate": {"passed": True},
            }),
            encoding="utf-8",
        )
        result_path.write_text(
            json.dumps({"auroc": {"truth_proj": 0.8 + (0.01 * abs(config.layer))}}),
            encoding="utf-8",
        )
        return {
            "dry_run": False,
            "output_dir": str(config.output_dir),
            "comparison_report": str(comparison_path),
            "results": {"cache_only": str(result_path)},
            "regression_gate": {"passed": True},
        }

    monkeypatch.setattr(module, "run_triplet", fake_run_triplet)
    config = module.CacheProfileMatrixConfig(
        output_dir=tmp_path,
        layers=(-2, -1),
        batch_sizes=(2,),
        hidden_state_captures=("outputs",),
    )

    report = module.run_matrix(config, clean=True, dry_run=False)
    saved = json.loads(Path(report["report_path"]).read_text(encoding="utf-8"))

    assert seen == [
        {"layer": -2, "batch_size": 2, "capture": "outputs"},
        {"layer": -1, "batch_size": 2, "capture": "outputs"},
    ]
    assert saved["leaderboard"][0]["id"] == "layer_m1_batch_2_capture_outputs"
    assert saved["leaderboard"][0]["gate_passed"] is True
    assert saved["cells"][0]["summary"]["truth_proj_auroc"] == pytest.approx(0.82)
    assert saved["cells"][0]["summary"]["totals"]["cache_only"]["bottleneck"] == "load_data"


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


def test_eval_truthfulqa_eval_reps_cache_metadata_embeds_eval_statements(tmp_path):
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
        layers=[-1],
        n_layers=2,
        eval_statements=statements,
    )
    restored = module._eval_statements_from_cache_metadata(metadata)

    assert metadata["eval_statements"] == [
        {"question": "q1", "answer": "a1", "is_false": 0},
        {"question": "q2", "answer": "a2", "is_false": 1},
    ]
    assert restored == statements

    legacy_metadata = dict(metadata)
    legacy_metadata.pop("eval_statements")
    cache_path = tmp_path / "legacy-eval-reps.pt"
    module.save_eval_reps_cache(cache_path, [None, None], metadata=legacy_metadata)
    loaded, loaded_metadata = module.load_eval_reps_cache(
        cache_path,
        expected_metadata=metadata,
        expected_records=2,
    )

    assert loaded == [None, None]
    assert loaded_metadata == legacy_metadata


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


def test_eval_truthfulqa_cache_only_can_skip_dataset_load_from_eval_reps_metadata(tmp_path, monkeypatch):
    module = importlib.import_module("benchmarks.eval_truthfulqa")
    statements = [
        module.Statement("q1", "a1", 0),
        module.Statement("q2", "a2", 1),
    ]
    args_for_metadata = SimpleNamespace(
        model="tiny",
        dtype="float32",
        offline=True,
        max_length=32,
        subspace_rank=2,
        eigenscore_alpha=1e-3,
        length_bucketed_batches=False,
    )
    stats_metadata = module._layer_stats_cache_metadata(
        args_for_metadata,
        layers=[-1],
        n_layers=2,
        true_texts=["t1", "t2"],
        false_texts=["f1"],
    )
    eval_metadata = module._eval_reps_cache_metadata(
        args_for_metadata,
        layers=[-1],
        n_layers=2,
        eval_statements=statements,
    )
    manifold = module.TruthManifold()
    manifold.update(torch.tensor([0.0, 0.0]))
    manifold.update(torch.tensor([1.0, 0.0]))
    manifold.contrastive_direction = torch.tensor([1.0, 0.0])
    stats_cache = tmp_path / "layer-stats.pt"
    eval_reps_cache = tmp_path / "eval-reps.pt"
    module.save_layer_stats_cache(stats_cache, {-1: manifold}, {}, metadata=stats_metadata)
    module.save_eval_reps_cache(
        eval_reps_cache,
        [
            {
                "last": {-1: torch.tensor([0.0, 0.0])},
                "ans_hs": torch.tensor([[0.0, 0.0], [0.1, 0.0]]),
                "eigenscore_by_layer": {-1: 0.1},
                "nll": 1.0,
            },
            {
                "last": {-1: torch.tensor([2.0, 0.0])},
                "ans_hs": torch.tensor([[1.0, 0.0], [2.0, 0.0]]),
                "eigenscore_by_layer": {-1: 0.2},
                "nll": 2.0,
            },
        ],
        metadata=eval_metadata,
    )

    def fail_load_data(*_args, **_kwargs):
        raise AssertionError("cache-only run should restore eval statements from cache metadata")

    monkeypatch.setattr(module, "load_offline", fail_load_data)
    monkeypatch.setattr(module, "load_truthfulqa", fail_load_data)

    result = module.run(SimpleNamespace(
        model="tiny",
        dtype="float32",
        layer=-1,
        sweep=False,
        sweep_layers=None,
        limit=None,
        manifold_questions=None,
        max_length=32,
        batch_size=1,
        auto_batch_size=False,
        length_bucketed_batches=False,
        hidden_state_capture="outputs",
        subspace_rank=2,
        eigenscore_alpha=1e-3,
        inside_samples=0,
        inside_batch_size=1,
        inside_max_new_tokens=8,
        inside_temperature=0.7,
        inside_top_p=0.9,
        inside_pooling="last",
        inside_trigger_signal=None,
        inside_trigger_threshold=None,
        inside_trigger_top_fraction=None,
        seed=0,
        offline=True,
        cache_only=True,
        statement_encoding_cache=None,
        refresh_statement_encoding_cache=False,
        layer_stats_cache=str(stats_cache),
        refresh_layer_stats_cache=False,
        warmup_checkpoint=None,
        warmup_checkpoint_every=0,
        eval_reps_cache=str(eval_reps_cache),
        eval_reps_cache_shard_size=0,
        refresh_eval_reps_cache=False,
        progress_every=0,
        profile=True,
        profile_json=None,
        json=None,
        dump_scores=None,
    ))

    assert result["config"]["cache_only_restored_eval_statements"] is True
    assert result["config"]["n_pos"] == 1
    assert result["config"]["n_neg"] == 1
    assert "load_data" not in result["profile"]["phases"]
    assert result["profile"]["summary"]["bottleneck"] != "load_data"


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

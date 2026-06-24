"""Smoke tests for benchmark reporting helpers."""

import importlib
import json
import os
import sys
import threading
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
    assert payload["config"]["score_dump"]["summary"]["n_total"] == 12
    assert payload["config"]["score_dump"]["summary"]["score_names"] == ("support",)
    assert payload["config"]["score_dump"]["sha256"]
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
    assert payload["documents"][2]["question"] == "Q1?"
    assert payload["documents"][2]["answer"] == "Correct A."
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
    assert fixture["retriever"]["requested_backend"] == "memory"
    assert fixture["retriever"]["actual_backend"] == "memory"
    assert run["score_dump"]["summary"]["n_total"] == 3
    assert run["score_dump"]["summary"]["score_names"] == ("truth_proj",)
    assert run["score_dump"]["sha256"]
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


def test_eval_verifier_ensemble_uses_retrieval_structured_qa_hits(tmp_path):
    builder = importlib.import_module("benchmarks.build_evidence_fixture")
    verifier = importlib.import_module("benchmarks.eval_verifier_ensemble")
    scores_path = tmp_path / "scores.json"
    corpus_path = tmp_path / "corpus.json"
    fixture_path = tmp_path / "fixture.json"
    dump = {
        "config": {"model": "synthetic", "layer": -1},
        "labels": [0, 0, 1, 1],
        "scores": {"truth_proj": [0.1, 0.2, 0.8, 0.9]},
        "statements": [
            {"question": "Q1?", "answer": "A1", "text": "Q1? A1"},
            {"question": "Q2?", "answer": "A2", "text": "Q2? A2"},
            {"question": "Q1?", "answer": "Wrong A1", "text": "Q1? Wrong A1"},
            {"question": "Q2?", "answer": "Wrong A2", "text": "Q2? Wrong A2"},
        ],
    }
    scores_path.write_text(json.dumps(dump), encoding="utf-8")
    corpus_path.write_text(
        json.dumps({
            "documents": [
                {"question": "Q1?", "answer": "A1", "text": "Q1? A1", "source": "qa:q1"},
                {"question": "Q2?", "answer": "A2", "text": "Q2? A2", "source": "qa:q2"},
            ],
        }),
        encoding="utf-8",
    )

    fixture = builder.build_evidence_fixture(
        builder.load_score_dump(scores_path),
        builder.load_corpus((corpus_path,)),
        retriever_min_overlap=1.0,
        retrieval_limit=1,
        query_field="question",
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
        retriever_min_overlap=1.0,
        retrieval_limit=1,
    )
    run = payload["runs"][0]
    routes = run["route_summary"]
    quality = run["route_quality"]["retrieval_structured_qa"]

    assert fixture["records"][0]["retrieval_documents"][0]["metadata"]["question"] == "Q1?"
    assert fixture["records"][1]["retrieval_documents"][0]["metadata"]["answer"] == "A2"
    assert payload["retrieval_qa_verifier"]["enabled"] is True
    assert run["retrieval_qa"]["decided_records"] == 4
    assert run["cache_stats"]["retrieval_qa_verifiers"]["requests"] == 4
    assert run["cache_stats"]["retrievers"]["requests"] == 0
    assert routes["selected_counts"] == {"retrieval_structured_qa": 4}
    assert routes["by_route"]["retrieval_structured_qa"]["statuses"]["supported"] == 2
    assert routes["by_route"]["retrieval_structured_qa"]["statuses"]["refuted"] == 2
    assert routes["by_route"]["retrieval_structured_qa"]["retrieval_use_rate"] == pytest.approx(1.0)
    assert quality["decision_accuracy"] == pytest.approx(1.0)
    assert quality["false_supported_rate"] == pytest.approx(0.0)
    assert quality["false_refuted_rate"] == pytest.approx(1.0)


def test_local_retrieval_workflow_can_gate_retrieval_structured_qa(tmp_path):
    module = importlib.import_module("benchmarks.run_local_retrieval_route_workflow")
    scores_path = tmp_path / "scores.json"
    corpus_path = tmp_path / "corpus.json"
    output_dir = tmp_path / "workflow"
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
    corpus_path.write_text(
        json.dumps({
            "documents": [
                {"question": "Q1?", "answer": "A1", "text": "Q1? A1", "source": "qa:q1"},
                {"question": "Q2?", "answer": "A2", "text": "Q2? A2", "source": "qa:q2"},
            ],
        }),
        encoding="utf-8",
    )

    payload = module.run_local_retrieval_route_workflow(
        module.LocalRetrievalRouteWorkflowConfig(
            scores_path=scores_path,
            corpus_paths=(corpus_path,),
            output_dir=output_dir,
            alpha=0.2,
            query_field="question",
            retriever_min_overlap=1.0,
            retrieval_limit=1,
            gate_routes=("retrieval_structured_qa",),
            min_selected=4,
            gate_min_selected=4,
            min_decision_accuracy=0.99,
            max_false_supported_rate=0.0,
            min_false_refuted_rate=0.99,
            max_mean_attempted_route_count=2.1,
            max_retrieval_use_rate=1.0,
            compact_json=True,
        )
    )
    route = payload["adapter_promotion"]["route_comparison"]["by_route"]["retrieval_structured_qa"]

    assert payload["decision"]["status"] == "promote"
    assert payload["adapter_promotion"]["decision"]["recommended_route"] == "retrieval_structured_qa"
    assert route["selected"] == 4
    assert route["decision_accuracy"] == pytest.approx(1.0)
    assert route["false_supported_rate"] == pytest.approx(0.0)
    assert route["false_refuted_rate"] == pytest.approx(1.0)
    assert route["retrieval_use_rate"] == pytest.approx(1.0)


def test_build_selfcheck_fixture_uses_dumped_inside_sample_texts(tmp_path):
    builder = importlib.import_module("benchmarks.build_selfcheck_fixture")
    verifier = importlib.import_module("benchmarks.eval_verifier_ensemble")
    scores_path = tmp_path / "scores.json"
    fixture_path = tmp_path / "selfcheck-fixture.json"
    dump = {
        "config": {"model": "synthetic", "layer": -1},
        "labels": [0, 0, 1, 1],
        "scores": {"truth_proj": [0.1, 0.2, 0.8, 0.9]},
        "statements": [
            {"text": "Paris is the capital of France.", "claim_id": "c1"},
            {"text": "Water boils at 100 degrees Celsius.", "claim_id": "c2"},
            {"text": "The moon is made of cheese.", "claim_id": "c3"},
            {"text": "AlphaCorp has 10 offices in Europe.", "claim_id": "c4"},
        ],
        "inside_sample_texts": [
            ["Paris is the capital of France.", "Paris is the capital of France and a city."],
            [
                "Water boils at 100 degrees Celsius at standard pressure.",
                "At standard pressure, water boils at 100 degrees Celsius.",
            ],
            ["The moon is not made of cheese.", "Lunar samples show the moon is not made of cheese."],
            ["AlphaCorp has 12 offices in Europe.", "As of 2026, AlphaCorp has 12 offices in Europe."],
        ],
    }
    scores_path.write_text(json.dumps(dump), encoding="utf-8")

    fixture = builder.build_selfcheck_fixture(builder.load_score_dump(scores_path), min_samples=2)
    fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
    report = verifier.build_verifier_ensemble_report(
        [("synthetic", scores_path)],
        signal="truth_proj",
        claims_path=fixture_path,
        alphas=(0.2,),
        repeats=1,
        seed=0,
        selfcheck_min_overlap=0.50,
    )

    assert fixture["fixture_type"] == "selfcheck_samples"
    assert fixture["summary"]["records_meeting_min_samples"] == 4
    assert fixture["records"][0]["selfcheck_samples"][0] == "Paris is the capital of France."
    assert report["runs"][0]["route_summary"]["selected_counts"] == {"self_consistency": 4}
    assert report["runs"][0]["verification_quality"]["decision_accuracy"] == pytest.approx(1.0)


def test_build_selfcheck_fixture_aligns_external_samples(tmp_path):
    builder = importlib.import_module("benchmarks.build_selfcheck_fixture")
    scores_path = tmp_path / "scores.json"
    samples_path = tmp_path / "samples.json"
    scores_path.write_text(
        json.dumps({
            "labels": [0, 1, 0],
            "scores": {"truth_proj": [0.1, 0.8, 0.2]},
            "statements": [
                {"text": "Water boils at 100 degrees Celsius.", "claim_id": "boil"},
                {"text": "AlphaCorp has 10 offices in Europe.", "claim_id": "offices"},
                {"text": "Unused claim.", "claim_id": "unused"},
            ],
        }),
        encoding="utf-8",
    )
    samples_path.write_text(
        json.dumps({
            "boil": [
                "Water boils at 100 degrees Celsius at standard pressure.",
                {"response": "At standard pressure, water boils at 100 degrees Celsius.", "source": "sample-2"},
            ],
            "records": [
                {
                    "index": 1,
                    "sampled_responses": [
                        "AlphaCorp has 12 offices in Europe.",
                        "AlphaCorp has 12 offices in Europe as of 2026.",
                    ],
                }
            ],
        }),
        encoding="utf-8",
    )

    fixture = builder.build_selfcheck_fixture(
        builder.load_score_dump(scores_path),
        builder.load_sample_payloads((samples_path,)),
        min_samples=2,
        include_empty_records=False,
    )

    assert fixture["summary"]["n_records"] == 2
    assert fixture["summary"]["records_dropped_below_min_samples"] == 1
    assert fixture["records"][0]["claim_id"] == "boil"
    assert fixture["records"][0]["selfcheck_samples"][1]["source"] == "sample-2"
    assert fixture["records"][1]["claim_id"] == "offices"
    assert fixture["records"][1]["metadata"]["selfcheck"]["meets_min_samples"] is True


def test_eval_verifier_ensemble_uses_self_consistency_samples(tmp_path):
    verifier = importlib.import_module("benchmarks.eval_verifier_ensemble")
    scores_path = tmp_path / "scores.json"
    fixture_path = tmp_path / "claims.json"
    scores_path.write_text(
        json.dumps({
            "config": {"model": "synthetic", "layer": -1},
            "labels": [0, 0, 1, 1],
            "scores": {"truth_proj": [0.1, 0.2, 0.8, 0.9]},
        }),
        encoding="utf-8",
    )
    fixture_path.write_text(
        json.dumps({
            "records": [
                {
                    "claim": "Paris is the capital of France.",
                    "selfcheck_samples": [
                        {"response": "Paris is the capital of France.", "source": "sample-1"},
                        "Paris is the capital of France and a major city.",
                    ],
                },
                {
                    "claim": "Water boils at 100 degrees Celsius.",
                    "sampled_responses": [
                        "Water boils at 100 degrees Celsius at standard pressure.",
                        "At standard pressure, water boils at 100 degrees Celsius.",
                    ],
                },
                {
                    "claim": "The moon is made of cheese.",
                    "selfcheck_samples": [
                        "The moon is not made of cheese.",
                        "The moon is not made of cheese.",
                    ],
                },
                {
                    "claim": "AlphaCorp has 10 offices in Europe.",
                    "selfcheck_samples": [
                        "AlphaCorp has 12 offices in Europe.",
                        "AlphaCorp has 12 offices in Europe as of 2026.",
                    ],
                },
            ]
        }),
        encoding="utf-8",
    )

    report = verifier.build_verifier_ensemble_report(
        [("synthetic", scores_path)],
        signal="truth_proj",
        claims_path=fixture_path,
        alphas=(0.2,),
        repeats=1,
        seed=0,
        selfcheck_min_overlap=0.55,
    )
    run = report["runs"][0]
    quality = run["verification_quality"]
    route_quality = run["route_quality"]["self_consistency"]

    assert report["selfcheck_verifier"]["enabled"] is True
    assert run["selfcheck_verifier"]["enabled"] is True
    assert run["selfcheck_verifier"]["records_with_samples"] == 4
    assert run["selfcheck_verifier"]["decided_records"] == 4
    assert run["route_summary"]["selected_counts"] == {"self_consistency": 4}
    assert run["route_summary"]["attempted_counts"] == {"groundedness": 4, "self_consistency": 4}
    assert run["verification_status_counts"]["supported"] == 2
    assert run["verification_status_counts"]["refuted"] == 2
    assert quality["decision_accuracy"] == pytest.approx(1.0)
    assert route_quality["selected"] == 4
    assert route_quality["true_supported_rate"] == pytest.approx(1.0)
    assert route_quality["false_refuted_rate"] == pytest.approx(1.0)
    assert route_quality["mean_attempted_route_count"] == pytest.approx(2.0)
    assert run["cache_stats"]["selfcheck_verifiers"]["requests"] == 4
    assert run["cache_stats"]["selfcheck_verifiers"]["instances"] == 4
    assert run["retrieval"]["records_with_hits"] == 0


def test_eval_verifier_ensemble_reports_selfcheck_early_stop_savings(tmp_path):
    verifier = importlib.import_module("benchmarks.eval_verifier_ensemble")
    scores_path = tmp_path / "scores.json"
    fixture_path = tmp_path / "claims.json"
    scores_path.write_text(
        json.dumps({
            "config": {"model": "synthetic", "layer": -1},
            "labels": [1, 0, 0, 1],
            "scores": {"truth_proj": [0.9, 0.1, 0.2, 0.8]},
        }),
        encoding="utf-8",
    )
    fixture_path.write_text(
        json.dumps({
            "records": [
                {
                    "claim": "AlphaCorp has 10 offices in Europe.",
                    "selfcheck_samples": [
                        "AlphaCorp has 12 offices in Europe.",
                        "AlphaCorp has 12 offices in Europe as of 2026.",
                        "AlphaCorp has 10 offices in Europe.",
                        "AlphaCorp has 10 offices in Europe.",
                        "AlphaCorp has 10 offices in Europe.",
                    ],
                },
                {
                    "claim": "Paris is the capital of France.",
                    "selfcheck_samples": [
                        "Paris is the capital of France.",
                        "Paris is the capital of France and a major city.",
                    ],
                },
                {"claim": "Water boils at 100 degrees Celsius."},
                {"claim": "The moon is made of cheese."},
            ]
        }),
        encoding="utf-8",
    )

    report = verifier.build_verifier_ensemble_report(
        [("synthetic", scores_path)],
        signal="truth_proj",
        claims_path=fixture_path,
        alphas=(0.2,),
        repeats=1,
        seed=0,
        selfcheck_min_overlap=0.55,
        selfcheck_refute_threshold=0.40,
        selfcheck_support_threshold=0.80,
        selfcheck_early_stop=True,
    )
    summary = report["runs"][0]["selfcheck_verifier"]

    assert report["selfcheck_verifier"]["early_stop"] is True
    assert summary["records_with_samples"] == 2
    assert summary["executed_records"] == 2
    assert summary["decided_records"] == 2
    assert summary["early_stopped_records"] == 1
    assert summary["considered_samples"] == 7
    assert summary["processed_samples"] == 4
    assert summary["skipped_samples"] == 3
    assert summary["processing_rate"] == pytest.approx(4 / 7)


def test_eval_verifier_ensemble_reuses_verification_trace_cache(tmp_path, monkeypatch):
    verifier = importlib.import_module("benchmarks.eval_verifier_ensemble")
    scores_path = tmp_path / "scores.json"
    fixture_path = tmp_path / "fixture.json"
    cache_dir = tmp_path / "verification-cache"
    scores_path.write_text(
        json.dumps({
            "config": {"model": "synthetic", "layer": -1},
            "labels": [0, 0, 1, 1],
            "scores": {"truth_proj": [0.1, 0.2, 0.8, 0.9]},
            "statements": [
                {"claim_id": "c1", "text": "Paris is the capital of France."},
                {"claim_id": "c2", "text": "The sky is blue."},
                {"claim_id": "c3", "text": "The moon is made of cheese."},
                {"claim_id": "c4", "text": "Paris is the capital of Germany."},
            ],
        }),
        encoding="utf-8",
    )
    fixture_path.write_text(
        json.dumps({
            "records": [
                {
                    "claim": "Paris is the capital of France.",
                    "claim_id": "c1",
                    "initial_evidence": ["Paris is the capital of France."],
                },
                {
                    "claim": "The sky is blue.",
                    "claim_id": "c2",
                    "initial_evidence": ["The sky is blue."],
                },
                {
                    "claim": "The moon is made of cheese.",
                    "claim_id": "c3",
                    "refutations": {"The moon is made of cheese.": ["Lunar samples are rock."]},
                },
                {
                    "claim": "Paris is the capital of Germany.",
                    "claim_id": "c4",
                    "refutations": {"Paris is the capital of Germany.": ["Berlin is the capital of Germany."]},
                },
            ],
        }),
        encoding="utf-8",
    )

    first = verifier.build_verifier_ensemble_report(
        [("synthetic", scores_path)],
        signal="truth_proj",
        claims_path=fixture_path,
        alphas=(0.2,),
        repeats=1,
        seed=0,
        verification_cache_dir=cache_dir,
    )

    def fail_verify_records(*args, **kwargs):
        raise AssertionError("verification trace cache hit should skip _verify_records")

    monkeypatch.setattr(verifier, "_verify_records", fail_verify_records)
    second = verifier.build_verifier_ensemble_report(
        [("synthetic", scores_path)],
        signal="truth_proj",
        claims_path=fixture_path,
        alphas=(0.2,),
        repeats=1,
        seed=0,
        verification_cache_dir=cache_dir,
    )

    assert first["runs"][0]["cache_stats"]["trace_cache"]["hit"] is False
    assert second["runs"][0]["cache_stats"]["trace_cache"]["hit"] is True
    assert second["runs"][0]["verification_status_counts"] == first["runs"][0]["verification_status_counts"]
    assert second["verification_trace_cache"]["path"] == str(cache_dir / "verifier-ensemble-verified-records.json")


def test_eval_verifier_ensemble_staged_verification_skips_low_risk_records(tmp_path):
    verifier = importlib.import_module("benchmarks.eval_verifier_ensemble")
    scores_path = tmp_path / "scores.json"
    fixture_path = tmp_path / "fixture.json"
    scores_path.write_text(
        json.dumps({
            "config": {"model": "synthetic", "layer": -1},
            "labels": [0, 0, 1, 1],
            "scores": {"truth_proj": [0.1, 0.2, 0.8, 0.9]},
        }),
        encoding="utf-8",
    )
    fixture_path.write_text(
        json.dumps({
            "records": [
                {"claim": "Paris is the capital of France.", "claim_id": "c1"},
                {"claim": "The sky is blue.", "claim_id": "c2"},
                {"claim": "The moon is made of cheese.", "claim_id": "c3"},
                {"claim": "Paris is the capital of Germany.", "claim_id": "c4"},
            ],
        }),
        encoding="utf-8",
    )

    payload = verifier.build_verifier_ensemble_report(
        [("synthetic", scores_path)],
        signal="truth_proj",
        claims_path=fixture_path,
        alphas=(0.5,),
        repeats=1,
        seed=0,
        staged_verification=True,
        staged_alpha=0.5,
    )

    run = payload["runs"][0]
    staged = run["staged_verification"]
    assert payload["staged_verification"]["enabled"] is True
    assert staged["threshold"] == pytest.approx(0.2)
    assert staged["skipped_records"] == 2
    assert staged["verified_records"] == 2
    assert staged["skip_rate"] == pytest.approx(0.5)
    assert staged["reason_counts"]["diagnostics and claim metadata did not require verification"] == 2
    assert staged["reason_counts"]["diagnostic risk level is medium"] == 2
    assert run["verification_status_counts"]["not_applicable"] == 2
    assert run["route_summary"]["selected_counts"]["staged_skip"] == 2
    assert run["cache_stats"]["groundedness_verifiers"]["requests"] == 2


def test_eval_verifier_ensemble_staged_verification_runs_for_sensitive_claim(tmp_path):
    verifier = importlib.import_module("benchmarks.eval_verifier_ensemble")
    scores_path = tmp_path / "scores.json"
    fixture_path = tmp_path / "fixture.json"
    scores_path.write_text(
        json.dumps({
            "config": {"model": "synthetic", "layer": -1},
            "labels": [0, 0, 1, 1],
            "scores": {"truth_proj": [0.1, 0.2, 0.8, 0.9]},
        }),
        encoding="utf-8",
    )
    fixture_path.write_text(
        json.dumps({
            "records": [
                {
                    "claim": "AlphaCorp has 10 offices.",
                    "claim_id": "c1",
                    "claim_metadata": {"features": {"has_number": True}},
                    "initial_evidence": ["AlphaCorp has 10 offices."],
                },
                {"claim": "The sky is blue.", "claim_id": "c2"},
                {"claim": "The moon is made of cheese.", "claim_id": "c3"},
                {"claim": "Paris is the capital of Germany.", "claim_id": "c4"},
            ],
        }),
        encoding="utf-8",
    )

    payload = verifier.build_verifier_ensemble_report(
        [("synthetic", scores_path)],
        signal="truth_proj",
        claims_path=fixture_path,
        alphas=(0.5,),
        repeats=1,
        seed=0,
        staged_verification=True,
        staged_alpha=0.5,
    )

    run = payload["runs"][0]
    staged = run["staged_verification"]
    assert staged["skipped_records"] == 1
    assert staged["verified_records"] == 3
    assert staged["triggered_claim_count"] == 1
    assert staged["triggered_feature_counts"]["has_number"] == 1
    assert run["verification_status_counts"]["supported"] == 1
    assert run["verification_status_counts"]["not_applicable"] == 1
    assert run["route_summary"]["selected_counts"]["groundedness"] == 3
    assert run["route_summary"]["selected_counts"]["staged_skip"] == 1
    assert run["cache_stats"]["groundedness_verifiers"]["requests"] == 3


def test_build_evidence_fixture_can_use_sqlite_fts_backend(tmp_path):
    builder = importlib.import_module("benchmarks.build_evidence_fixture")
    scores_path = tmp_path / "scores.json"
    corpus_path = tmp_path / "corpus.json"
    scores_path.write_text(
        json.dumps({
            "labels": [0, 1],
            "statements": [
                {
                    "claim_id": "c1",
                    "answer": "Order R1 is approved for expedited shipping.",
                    "text": "Order R1 is approved for expedited shipping.",
                },
                {
                    "claim_id": "c2",
                    "answer": "Order R1 is approved for same-day drone shipping.",
                    "text": "Order R1 is approved for same-day drone shipping.",
                },
            ],
        }),
        encoding="utf-8",
    )
    corpus_path.write_text(
        json.dumps({
            "documents": [
                {"text": "Order R1 is approved for expedited shipping.", "source": "support:R1"},
                {"text": "Order R1 is not approved for same-day drone shipping.", "source": "refute:R1"},
            ],
        }),
        encoding="utf-8",
    )

    index_path = tmp_path / "retriever.sqlite"
    fixture = builder.build_evidence_fixture(
        builder.load_score_dump(scores_path),
        builder.load_corpus((corpus_path,)),
        retriever_min_overlap=0.5,
        retrieval_limit=1,
        query_field="answer",
        retriever_backend="auto",
        retriever_index_path=index_path,
    )

    assert fixture["summary"]["records_with_hits"] == 2
    assert fixture["retriever"]["requested_backend"] == "auto"
    assert fixture["retriever"]["requested_index_path"] == str(index_path)
    assert fixture["retriever"]["actual_backend"] in {"sqlite_fts", "memory"}
    if fixture["retriever"]["actual_backend"] == "sqlite_fts":
        assert fixture["retriever"]["type"] == "SQLiteFTSRetriever"
        assert fixture["retriever"]["actual_index_path"] == str(index_path)
        assert fixture["retriever"]["index_reused"] is False
        assert fixture["records"][0]["retrieval_documents"][0]["metadata"]["retriever_backend"] == "sqlite_fts"
        second = builder.build_evidence_fixture(
            builder.load_score_dump(scores_path),
            builder.load_corpus((corpus_path,)),
            retriever_min_overlap=0.5,
            retrieval_limit=1,
            query_field="answer",
            retriever_backend="auto",
            retriever_index_path=index_path,
        )
        assert second["retriever"]["index_reused"] is True
    else:
        assert fixture["retriever"]["type"] == "InMemoryRetriever"
        assert fixture["retriever"]["actual_index_path"] is None
        assert fixture["retriever"]["fallback_reason"]

    with pytest.raises(ValueError, match="retriever_index_path"):
        builder.build_evidence_fixture(
            builder.load_score_dump(scores_path),
            builder.load_corpus((corpus_path,)),
            retriever_backend="memory",
            retriever_index_path=index_path,
        )


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
                            "verified_false_alarm": 0.0,
                            "verified_detection": 1.0,
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


def test_compare_verifier_routes_builds_staged_verification_gate(tmp_path):
    module = importlib.import_module("benchmarks.compare_verifier_routes")
    report_path = tmp_path / "staged-route.json"
    report_path.write_text(
        json.dumps({
            "runs": [
                {
                    "name": "staged",
                    "n_true": 10,
                    "n_false": 10,
                    "staged_verification": {
                        "enabled": True,
                        "total_records": 20,
                        "verified_records": 8,
                        "skipped_records": 12,
                        "skip_rate": 0.6,
                        "threshold": 0.2,
                    },
                    "route_quality": {
                        "structured_state": {
                            "selected": 8,
                            "n_true": 4,
                            "n_false": 4,
                            "label_status_matrix": {
                                "true": {"supported": 4, "refuted": 0, "insufficient_evidence": 0},
                                "false": {"supported": 0, "refuted": 4, "insufficient_evidence": 0},
                            },
                            "false_refuted_rate": 1.0,
                            "false_supported_rate": 0.0,
                            "decision_accuracy": 1.0,
                            "duration_observations": 8,
                            "total_duration_seconds": 0.08,
                            "mean_duration_seconds": 0.01,
                            "p95_duration_seconds": 0.015,
                            "p99_duration_seconds": 0.018,
                            "max_duration_seconds": 0.02,
                            "attempted_route_count_observations": 8,
                            "total_attempted_route_count": 8,
                            "mean_attempted_route_count": 1.0,
                            "used_retrieval_count": 0,
                            "retrieval_use_rate": 0.0,
                        },
                    },
                    "alphas": {
                        "0.1": {
                            "internal": {"false_alarm": 0.10, "detection": 0.70},
                            "verified": {"false_alarm": 0.05, "detection": 0.90},
                            "delta": {
                                "false_alarm": -0.05,
                                "detection": 0.20,
                                "suppressed_false_alarm_rate": 0.05,
                                "rescued_detection_rate": 0.20,
                            },
                            "route_control_impact": {
                                "structured_state": {
                                    "verified": {"false_alarm": 0.05, "detection": 0.90},
                                    "delta": {
                                        "false_alarm": -0.05,
                                        "detection": 0.20,
                                        "rescued_detection_rate": 0.20,
                                    },
                                }
                            },
                        }
                    },
                }
            ]
        }),
        encoding="utf-8",
    )

    passing = module.build_route_comparison_report(
        [("staged", report_path)],
        gate_routes=("structured_state",),
        min_decision_accuracy=0.99,
        min_staged_skip_rate=0.5,
        max_staged_verified_false_alarm=0.05,
        min_staged_verified_detection=0.85,
        max_staged_delta_false_alarm=0.0,
        min_staged_delta_detection=0.0,
    )
    failing = module.build_route_comparison_report(
        [("staged", report_path)],
        gate_routes=("structured_state",),
        min_decision_accuracy=0.99,
        min_staged_skip_rate=0.75,
    )

    assert passing["staged_verification"]["enabled"] is True
    assert passing["staged_verification"]["skip_rate"] == pytest.approx(0.6)
    assert passing["staged_verification"]["verified_false_alarm"] == pytest.approx(0.05)
    assert passing["staged_verification"]["verified_detection"] == pytest.approx(0.90)
    assert passing["quality_gate"]["passed"] is True
    assert passing["promotion_decision"]["status"] == "promote"
    assert failing["quality_gate"]["passed"] is False
    assert failing["promotion_decision"]["status"] == "blocked_by_gate"
    assert failing["quality_gate"]["failures"][0]["route"] is None
    assert failing["quality_gate"]["failures"][0]["metric"] == "staged_skip_rate"


def test_compare_verifier_routes_staged_gate_fails_closed_without_staged_report(tmp_path):
    module = importlib.import_module("benchmarks.compare_verifier_routes")
    report_path = tmp_path / "unstaged-route.json"
    report_path.write_text(
        json.dumps({
            "runs": [
                {
                    "name": "unstaged",
                    "route_quality": {
                        "groundedness": {
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
                        },
                    },
                    "alphas": {"0.1": {"route_control_impact": {}}},
                }
            ]
        }),
        encoding="utf-8",
    )

    payload = module.build_route_comparison_report(
        [("unstaged", report_path)],
        gate_routes=("groundedness",),
        min_decision_accuracy=0.99,
        min_staged_skip_rate=0.1,
    )

    assert payload["staged_verification"]["enabled"] is False
    assert payload["quality_gate"]["passed"] is False
    assert {failure["metric"] for failure in payload["quality_gate"]["failures"]} >= {
        "staged_verification_enabled",
        "staged_skip_rate",
    }
    assert payload["promotion_decision"]["status"] == "blocked_by_gate"


def test_compare_verifier_routes_gate_fails_on_partially_invalid_aggregate_cost_metrics(tmp_path):
    module = importlib.import_module("benchmarks.compare_verifier_routes")
    valid_report = tmp_path / "valid-cost.json"
    invalid_report = tmp_path / "invalid-cost.json"

    base_route_quality = {
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
    valid_report.write_text(
        json.dumps({
            "runs": [
                {
                    "name": "valid",
                    "route_quality": {"structured_state": base_route_quality},
                    "alphas": {"0.1": {"route_control_impact": {}}},
                }
            ]
        }),
        encoding="utf-8",
    )
    invalid_quality = dict(base_route_quality)
    invalid_quality["total_duration_seconds"] = float("nan")
    invalid_quality["mean_duration_seconds"] = float("nan")
    invalid_report.write_text(
        json.dumps({
            "runs": [
                {
                    "name": "invalid",
                    "route_quality": {"structured_state": invalid_quality},
                    "alphas": {"0.1": {"route_control_impact": {}}},
                }
            ]
        }),
        encoding="utf-8",
    )

    payload = module.build_route_comparison_report(
        [("valid", valid_report), ("invalid", invalid_report)],
        gate_routes=("structured_state",),
        max_mean_duration_seconds=0.02,
    )

    aggregate = payload["by_route"]["structured_state"]
    failures = payload["quality_gate"]["failures"]
    assert aggregate["invalid_metric_counts"]["mean_duration_seconds"] == 1
    assert aggregate["duration_observations"] == 4
    assert aggregate["total_duration_seconds"] == pytest.approx(0.04)
    assert aggregate["mean_duration_seconds"] == pytest.approx(0.01)
    assert payload["quality_gate"]["passed"] is False
    assert failures[0]["metric"] == "mean_duration_seconds"
    assert failures[0]["limit_type"] == "finite"
    assert payload["promotion_decision"]["status"] == "blocked_by_gate"


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


def _write_adapter_promotion_route_report(path: Path, *, staged: bool = False) -> None:
    alpha_payload = {
        "route_control_impact": {
            "structured_state": {
                "verified": {"false_alarm": 0.0, "detection": 1.0},
            }
        }
    }
    run = {
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
        "alphas": {"0.1": alpha_payload},
    }
    if staged:
        run.update({
            "n_true": 10,
            "n_false": 10,
            "staged_verification": {
                "enabled": True,
                "total_records": 20,
                "verified_records": 8,
                "skipped_records": 12,
                "skip_rate": 0.6,
                "threshold": 0.2,
            },
        })
        alpha_payload.update({
            "internal": {"false_alarm": 0.10, "detection": 0.70},
            "verified": {"false_alarm": 0.05, "detection": 0.90},
            "delta": {
                "false_alarm": -0.05,
                "detection": 0.20,
                "suppressed_false_alarm_rate": 0.05,
                "rescued_detection_rate": 0.20,
            },
        })
    path.write_text(json.dumps({"runs": [run]}), encoding="utf-8")


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


def test_run_adapter_promotion_workflow_writes_artifact_manifest(tmp_path):
    module = importlib.import_module("benchmarks.run_adapter_promotion_workflow")
    route_source_path = tmp_path / "routes.json"
    route_report_path = tmp_path / "route-comparison.json"
    manifest_path = tmp_path / "artifact-manifest.json"
    _write_adapter_promotion_route_report(route_source_path, staged=True)

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
            min_staged_skip_rate=0.50,
            max_staged_verified_false_alarm=0.05,
            min_staged_verified_detection=0.85,
            max_staged_delta_false_alarm=0.0,
            min_staged_delta_detection=0.0,
            artifact_manifest_path=manifest_path,
        )
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert payload["artifact_manifest"] == str(manifest_path)
    assert manifest["metadata"]["runner"] == "run_adapter_promotion_workflow"
    assert manifest["metadata"]["promotion_status"] == "promote"
    assert manifest["metadata"]["recommended_route"] == "structured_state"
    assert manifest["metadata"]["staged_skip_rate"] == pytest.approx(0.6)
    assert manifest["metadata"]["staged_verified_detection"] == pytest.approx(0.9)
    assert payload["route_comparison"]["quality_gate"]["config"]["min_staged_skip_rate"] == pytest.approx(0.5)
    assert manifest["artifacts"]["route_comparison_report"]["exists"] is True
    assert manifest["artifacts"]["verifier_reports.routes"]["exists"] is True


def test_run_adapter_promotion_registry_workflow_registers_promoted_route(tmp_path):
    module = importlib.import_module("benchmarks.run_adapter_promotion_registry_workflow")
    compare_module = importlib.import_module("benchmarks.compare_route_baselines")
    from eigentruth.registry import ArtifactRegistry

    route_source_path = tmp_path / "routes.json"
    route_report_path = tmp_path / "route-comparison.json"
    manifest_path = tmp_path / "artifact-manifest.json"
    registry_path = tmp_path / "registry.json"
    workflow_path = tmp_path / "workflow.json"
    verification_path = tmp_path / "manifest-verification.json"
    _write_adapter_promotion_route_report(route_source_path, staged=True)

    payload = module.run_adapter_promotion_registry_workflow(
        module.AdapterPromotionRegistryWorkflowConfig(
            promotion=module.AdapterPromotionWorkflowConfig(
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
                min_staged_skip_rate=0.50,
                max_staged_verified_false_alarm=0.05,
                min_staged_verified_detection=0.85,
                max_staged_delta_false_alarm=0.0,
                min_staged_delta_detection=0.0,
                artifact_manifest_path=manifest_path,
            ),
            registry_path=registry_path,
            name="route-baseline",
            version="0.6",
            workflow_report_path=workflow_path,
            verification_report_path=verification_path,
            promotion_metadata={"scope": "unit"},
        )
    )

    assert payload["decision"]["status"] == "promote"
    assert payload["decision"]["manifest_promoted"] is True
    assert payload["decision"]["manifest_verified"] is True
    assert payload["decision"]["registry_record"] == "benchmark_manifest:route-baseline:0.6"
    assert Path(payload["promotion"]["verification_report"]).exists()
    assert workflow_path.exists()
    registry = ArtifactRegistry.load_json(registry_path)
    record = registry.get("benchmark_manifest:route-baseline:0.6")
    assert record.metadata["workflow"] == "run_adapter_promotion_registry_workflow"
    assert record.metadata["adapter_promotion_status"] == "promote"
    assert record.metadata["route_promotion_status"] == "promote"
    assert record.metadata["recommended_route"] == "structured_state"
    assert record.metadata["recommended_decision_accuracy"] == pytest.approx(1.0)
    assert record.metadata["staged_skip_rate"] == pytest.approx(0.6)
    assert record.metadata["staged_verified_false_alarm"] == pytest.approx(0.05)
    assert record.metadata["staged_delta_detection"] == pytest.approx(0.2)
    assert record.metadata["scope"] == "unit"

    baseline = compare_module.compare_route_baselines(registry_path=registry_path)
    assert baseline["decision"]["status"] == "promote"
    assert baseline["decision"]["recommended_record"] == "benchmark_manifest:route-baseline:0.6"
    assert baseline["decision"]["recommended_route"] == "structured_state"


def test_run_local_retrieval_route_workflow_registers_retrieval_baseline(tmp_path):
    module = importlib.import_module("benchmarks.run_local_retrieval_route_workflow")
    compare_module = importlib.import_module("benchmarks.compare_route_baselines")
    from eigentruth.registry import ArtifactRegistry

    scores_path = tmp_path / "scores.json"
    corpus_path = tmp_path / "corpus.json"
    registry_path = tmp_path / "registry.json"
    output_dir = tmp_path / "workflow"
    retriever_index_path = output_dir / "retriever.sqlite"
    statements = [
        {
            "claim_id": "order_true_1",
            "question": "What shipping option is order R1 approved for?",
            "answer": "Order R1 is approved for expedited shipping.",
            "text": "Order R1 is approved for expedited shipping.",
        },
        {
            "claim_id": "order_true_2",
            "question": "What shipping option is order R2 approved for?",
            "answer": "Order R2 is approved for expedited shipping.",
            "text": "Order R2 is approved for expedited shipping.",
        },
        {
            "claim_id": "order_false_1",
            "question": "What shipping option is order R1 approved for?",
            "answer": "Order R1 is approved for same-day drone shipping.",
            "text": "Order R1 is approved for same-day drone shipping.",
        },
        {
            "claim_id": "order_false_2",
            "question": "What shipping option is order R2 approved for?",
            "answer": "Order R2 is approved for same-day drone shipping.",
            "text": "Order R2 is approved for same-day drone shipping.",
        },
    ]
    scores_path.write_text(
        json.dumps({
            "schema_version": 1,
            "config": {"model": "synthetic-local-retrieval", "layer": -1},
            "labels": [0, 0, 1, 1],
            "scores": {"truth_proj": [0.2, 0.21, 0.8, 0.81]},
            "statements": statements,
        }),
        encoding="utf-8",
    )
    corpus_path.write_text(
        json.dumps({
            "schema_version": 1,
            "documents": [
                {
                    "text": "Order R1 is approved for expedited shipping.",
                    "source": "shipping:R1:support",
                },
                {
                    "text": "Order R2 is approved for expedited shipping.",
                    "source": "shipping:R2:support",
                },
                {
                    "text": "Order R1 is not approved for same-day drone shipping.",
                    "source": "shipping:R1:refute",
                },
                {
                    "text": "Order R2 is not approved for same-day drone shipping.",
                    "source": "shipping:R2:refute",
                },
            ],
        }),
        encoding="utf-8",
    )

    payload = module.run_local_retrieval_route_workflow(
        module.LocalRetrievalRouteWorkflowConfig(
            scores_path=scores_path,
            corpus_paths=(corpus_path,),
            output_dir=output_dir,
            registry_path=registry_path,
            name="local-retrieval-route",
            version="0.7",
            alpha=0.2,
            retriever_backend="auto",
            retriever_index_path=retriever_index_path,
            retrieval_limit=1,
            retriever_min_overlap=0.6,
            min_selected=4,
            gate_min_selected=4,
            min_decision_accuracy=0.99,
            max_false_supported_rate=0.0,
            min_false_refuted_rate=0.99,
            max_mean_attempted_route_count=2.1,
            max_retrieval_use_rate=1.0,
            compact_json=True,
            promotion_metadata={"scope": "unit"},
        )
    )
    manifest = json.loads((output_dir / "retrieval-route-artifact-manifest.json").read_text(encoding="utf-8"))

    assert payload["decision"]["status"] == "promote"
    assert payload["decision"]["manifest_promoted"] is True
    assert payload["decision"]["manifest_verified"] is True
    assert payload["decision"]["registry_record"] == "benchmark_manifest:local-retrieval-route:0.7"
    assert payload["claims_summary"]["records_with_hits"] == 4
    assert payload["claims_summary"]["total_hits"] == 4
    assert payload["config"]["retriever_backend"] == "auto"
    assert payload["config"]["retriever_index_path"] == str(retriever_index_path)
    profile = payload["profile"]
    assert profile["total_seconds"] >= 0.0
    assert profile["summary"]["bottleneck"] in profile["phases"]
    assert profile["summary"]["accounted_share"] >= 0.0
    assert profile["scale"]["n_labels"] == 4
    assert profile["scale"]["n_corpus_documents"] == 4
    assert profile["scale"]["n_claim_records"] == 4
    assert profile["scale"]["n_retrieval_hits"] == 4
    assert profile["scale"]["n_routes"] >= 1
    assert profile["artifacts"]["input_bytes"]["score_dump"] > 0
    assert profile["artifacts"]["input_bytes"]["retrieval_corpora.1.corpus"] > 0
    assert profile["artifacts"]["output_bytes"]["retrieval_claims"] > 0
    assert profile["artifacts"]["output_bytes"]["verifier_report"] > 0
    assert profile["artifacts"]["output_bytes"]["route_comparison_report"] > 0
    assert profile["artifacts"]["output_bytes"]["promotion_report"] > 0
    route = payload["adapter_promotion"]["route_comparison"]["by_route"]["retrieval_groundedness"]
    assert route["selected"] == 4
    assert route["decision_accuracy"] == pytest.approx(1.0)
    assert route["false_supported_rate"] == pytest.approx(0.0)
    assert route["false_refuted_rate"] == pytest.approx(1.0)
    assert route["mean_attempted_route_count"] == pytest.approx(2.0)
    assert route["retrieval_use_rate"] == pytest.approx(1.0)
    expected_manifest_artifacts = [
        "promotion_report",
        "retrieval_claims",
        "retrieval_corpora.1.corpus",
        "route_comparison_report",
        "score_dump",
        "verifier_report",
    ]
    if manifest["metadata"]["retriever_actual_backend"] == "sqlite_fts":
        expected_manifest_artifacts.append("retriever_index")
    assert sorted(manifest["artifacts"]) == sorted(expected_manifest_artifacts)
    assert manifest["metadata"]["runner"] == "run_local_retrieval_route_workflow"
    assert manifest["metadata"]["recommended_route"] == "retrieval_groundedness"
    assert manifest["metadata"]["retriever_backend"] == "auto"
    assert manifest["metadata"]["retriever_requested_index_path"] == str(retriever_index_path)
    assert manifest["metadata"]["retriever_actual_backend"] in {"sqlite_fts", "memory"}
    if manifest["metadata"]["retriever_actual_backend"] == "sqlite_fts":
        assert manifest["metadata"]["retriever_actual_index_path"] == str(retriever_index_path)
    else:
        assert manifest["metadata"]["retriever_actual_index_path"] is None
    assert manifest["metadata"]["claims_records_with_hits"] == 4
    assert manifest["metadata"]["recommended_retrieval_use_rate"] == pytest.approx(1.0)
    assert manifest["metadata"]["runtime_total_seconds"] >= 0.0
    assert manifest["metadata"]["runtime_bottleneck"] in profile["phases"]
    assert manifest["metadata"]["runtime_n_corpus_documents"] == 4
    assert manifest["metadata"]["runtime_n_retrieval_hits"] == 4
    assert manifest["metadata"]["runtime_claims_json_bytes"] > 0

    registry = ArtifactRegistry.load_json(registry_path)
    record = registry.get("benchmark_manifest:local-retrieval-route:0.7")
    assert record.metadata["workflow"] == "run_local_retrieval_route_workflow"
    assert record.metadata["recommended_route"] == "retrieval_groundedness"
    assert record.metadata["retriever_actual_backend"] in {"sqlite_fts", "memory"}
    assert record.metadata["recommended_retrieval_use_rate"] == pytest.approx(1.0)
    assert record.metadata["runtime_bottleneck"] in profile["phases"]
    assert record.metadata["runtime_n_claim_records"] == 4
    assert record.metadata["scope"] == "unit"
    baseline = compare_module.compare_route_baselines(
        registry_path=registry_path,
        max_mean_attempted_route_count=2.1,
        max_retrieval_use_rate=1.0,
    )
    assert baseline["decision"]["status"] == "promote"

    blocked = module.run_local_retrieval_route_workflow(
        module.LocalRetrievalRouteWorkflowConfig(
            scores_path=scores_path,
            corpus_paths=(corpus_path,),
            output_dir=tmp_path / "workflow-budget-blocked",
            registry_path=registry_path,
            name="local-retrieval-route-budget",
            version="0.8",
            alpha=0.2,
            retriever_backend="auto",
            retriever_index_path=tmp_path / "workflow-budget-blocked" / "retriever.sqlite",
            retrieval_limit=1,
            retriever_min_overlap=0.6,
            min_selected=4,
            gate_min_selected=4,
            min_decision_accuracy=0.99,
            max_false_supported_rate=0.0,
            min_false_refuted_rate=0.99,
            max_mean_attempted_route_count=2.1,
            max_retrieval_use_rate=1.0,
            max_retrieval_hit_count=0.0,
            compact_json=True,
        )
    )
    registry_after_block = ArtifactRegistry.load_json(registry_path)

    assert blocked["adapter_promotion"]["decision"]["status"] == "promote"
    assert blocked["runtime_budget"]["passed"] is False
    assert blocked["runtime_budget"]["failures"][0]["metric"] == "retrieval_hit_count"
    assert blocked["decision"]["status"] == "blocked"
    assert blocked["decision"]["manifest_promoted"] is False
    assert all(
        record.key() != "benchmark_manifest:local-retrieval-route-budget:0.8"
        for record in registry_after_block.list_records()
    )


def test_run_local_retrieval_route_workflow_reuses_claims_cache(tmp_path, monkeypatch):
    module = importlib.import_module("benchmarks.run_local_retrieval_route_workflow")

    scores_path = tmp_path / "scores.json"
    corpus_path = tmp_path / "corpus.json"
    cache_dir = tmp_path / "claims-cache"
    trace_cache_dir = tmp_path / "verifier-trace-cache"
    statements = [
        {
            "claim_id": "order_true_1",
            "question": "What shipping option is order R1 approved for?",
            "answer": "Order R1 is approved for expedited shipping.",
            "text": "Order R1 is approved for expedited shipping.",
        },
        {
            "claim_id": "order_true_2",
            "question": "What shipping option is order R2 approved for?",
            "answer": "Order R2 is approved for expedited shipping.",
            "text": "Order R2 is approved for expedited shipping.",
        },
        {
            "claim_id": "order_false_1",
            "question": "What shipping option is order R1 approved for?",
            "answer": "Order R1 is approved for same-day drone shipping.",
            "text": "Order R1 is approved for same-day drone shipping.",
        },
        {
            "claim_id": "order_false_2",
            "question": "What shipping option is order R2 approved for?",
            "answer": "Order R2 is approved for same-day drone shipping.",
            "text": "Order R2 is approved for same-day drone shipping.",
        },
    ]
    scores_path.write_text(
        json.dumps({
            "schema_version": 1,
            "config": {"model": "synthetic-local-retrieval", "layer": -1},
            "labels": [0, 0, 1, 1],
            "scores": {"truth_proj": [0.2, 0.21, 0.8, 0.81]},
            "statements": statements,
        }),
        encoding="utf-8",
    )
    corpus_path.write_text(
        json.dumps({
            "schema_version": 1,
            "documents": [
                {"text": "Order R1 is approved for expedited shipping.", "source": "shipping:R1:support"},
                {"text": "Order R2 is approved for expedited shipping.", "source": "shipping:R2:support"},
                {"text": "Order R1 is not approved for same-day drone shipping.", "source": "shipping:R1:refute"},
                {"text": "Order R2 is not approved for same-day drone shipping.", "source": "shipping:R2:refute"},
            ],
        }),
        encoding="utf-8",
    )

    common = {
        "scores_path": scores_path,
        "corpus_paths": (corpus_path,),
        "alpha": 0.2,
        "retrieval_limit": 1,
        "retriever_min_overlap": 0.6,
        "min_selected": 4,
        "gate_min_selected": 4,
        "min_decision_accuracy": 0.99,
        "max_false_supported_rate": 0.0,
        "min_false_refuted_rate": 0.99,
        "max_mean_attempted_route_count": 2.1,
        "max_retrieval_use_rate": 1.0,
        "claims_cache_dir": cache_dir,
        "verifier_trace_cache_dir": trace_cache_dir,
        "compact_json": True,
    }

    first = module.run_local_retrieval_route_workflow(
        module.LocalRetrievalRouteWorkflowConfig(output_dir=tmp_path / "first", **common)
    )
    cache_path = Path(first["claims_cache"]["path"])
    assert first["claims_cache"]["status"] == "miss"
    assert first["claims_cache"]["hit"] is False
    assert first["profile"]["cache"]["verifier_trace"]["hit"] is False
    assert cache_path.exists()
    assert "load_inputs" in first["profile"]["phases"]
    assert "build_claims" in first["profile"]["phases"]
    assert "write_claims_cache" in first["profile"]["phases"]

    def fail_loader(*args, **kwargs):
        raise AssertionError("claims cache hit should skip score and corpus loaders")

    monkeypatch.setattr(module, "load_score_dump", fail_loader)
    monkeypatch.setattr(module, "load_corpus", fail_loader)

    second = module.run_local_retrieval_route_workflow(
        module.LocalRetrievalRouteWorkflowConfig(output_dir=tmp_path / "second", **common)
    )
    second_manifest = json.loads(
        (tmp_path / "second" / "retrieval-route-artifact-manifest.json").read_text(encoding="utf-8")
    )

    assert second["decision"]["status"] == "promote"
    assert second["claims_cache"]["status"] == "hit"
    assert second["claims_cache"]["hit"] is True
    assert second["claims_cache"]["key"] == first["claims_cache"]["key"]
    assert second["profile"]["cache"]["verifier_trace"]["hit"] is True
    assert second["profile"]["cache"]["verifier_trace"]["hit_count"] == 1
    assert second["claims_cache"]["scale"]["n_corpus_documents"] == 4
    assert second["claims_summary"] == first["claims_summary"]
    assert "load_claims_cache" in second["profile"]["phases"]
    assert "load_inputs" not in second["profile"]["phases"]
    assert "build_claims" not in second["profile"]["phases"]
    assert second["profile"]["cache"]["claims"]["hit"] is True
    assert second["profile"]["artifacts"]["output_bytes"]["claims_cache_record"] > 0
    assert "claims_cache_record" in second_manifest["artifacts"]
    assert second_manifest["metadata"]["claims_cache_hit"] is True
    assert second_manifest["metadata"]["claims_cache_status"] == "hit"
    assert second_manifest["metadata"]["verifier_trace_cache_enabled"] is True
    assert second_manifest["metadata"]["verifier_trace_cache_hit_count"] == 1


def test_run_adapter_promotion_registry_workflow_cli_blocks_non_promoted_route(tmp_path):
    module = importlib.import_module("benchmarks.run_adapter_promotion_registry_workflow")
    from eigentruth.registry import ArtifactRegistry

    route_source_path = tmp_path / "routes.json"
    route_report_path = tmp_path / "route-comparison.json"
    workflow_report_path = tmp_path / "workflow.json"
    manifest_path = tmp_path / "artifact-manifest.json"
    registry_path = tmp_path / "registry.json"
    _write_adapter_promotion_route_report(route_source_path)

    with pytest.raises(SystemExit) as exc_info:
        module.main([
            "--report",
            f"routes={route_source_path}",
            "--route-report-json",
            str(route_report_path),
            "--artifact-manifest",
            str(manifest_path),
            "--registry",
            str(registry_path),
            "--name",
            "route-baseline",
            "--version",
            "0.6",
            "--json",
            str(workflow_report_path),
            "--fail-on-blocked",
        ])

    assert exc_info.value.code == 1
    payload = json.loads(workflow_report_path.read_text(encoding="utf-8"))
    assert payload["decision"]["status"] == "blocked"
    assert payload["decision"]["manifest_promoted"] is False
    assert payload["decision"]["manifest_verified"] is False
    assert payload["promotion"] is None
    assert payload["decision"]["blocking_reasons"] == [
        "adapter promotion decision did not promote"
    ]
    assert ArtifactRegistry.load_json(registry_path).records == ()


def _write_route_baseline_manifest(
    tmp_path: Path,
    *,
    name: str,
    route: str,
    decision_accuracy: float,
    false_supported_rate: float,
    false_refuted_rate: float,
    mean_duration_seconds: float,
    p99_duration_seconds: float,
    mean_attempted_route_count: float = 1.0,
    retrieval_use_rate: float = 0.0,
    invalid_metric_counts: dict[str, int] | None = None,
    runtime_total_seconds: float | None = None,
    runtime_n_retrieval_hits: int | None = None,
    claims_cache_enabled: bool | None = None,
    claims_cache_hit: bool | None = None,
    verifier_trace_cache_enabled: bool | None = None,
    verifier_trace_cache_hit_count: int | None = None,
    verifier_trace_cache_run_count: int | None = None,
) -> Path:
    from eigentruth.registry import build_artifact_manifest

    route_report_path = tmp_path / f"{name}-route-comparison.json"
    manifest_path = tmp_path / f"{name}-artifact-manifest.json"
    route_report_path.write_text(
        json.dumps({
            "schema_version": 1,
            "promotion_decision": {"status": "promote", "recommended_route": route},
            "by_route": {
                route: {
                    "selected": 8,
                    "decision_accuracy": decision_accuracy,
                    "false_supported_rate": false_supported_rate,
                    "false_refuted_rate": false_refuted_rate,
                    "verified_false_alarm": 0.0,
                    "verified_detection": false_refuted_rate,
                    "mean_duration_seconds": mean_duration_seconds,
                    "p95_duration_seconds": p99_duration_seconds,
                    "p99_duration_seconds": p99_duration_seconds,
                    "max_duration_seconds": p99_duration_seconds,
                    "mean_attempted_route_count": mean_attempted_route_count,
                    "retrieval_use_rate": retrieval_use_rate,
                    "invalid_metric_counts": invalid_metric_counts or {},
                }
            },
        }),
        encoding="utf-8",
    )
    metadata = {
        "runner": "run_adapter_promotion_workflow",
        "workflow": "adapter_promotion_workflow",
        "promotion_status": "promote",
        "route_promotion_status": "promote",
        "recommended_route": route,
    }
    optional_metadata = {
        "runtime_total_seconds": runtime_total_seconds,
        "runtime_n_retrieval_hits": runtime_n_retrieval_hits,
        "claims_cache_enabled": claims_cache_enabled,
        "claims_cache_hit": claims_cache_hit,
        "verifier_trace_cache_enabled": verifier_trace_cache_enabled,
        "verifier_trace_cache_hit_count": verifier_trace_cache_hit_count,
        "verifier_trace_cache_run_count": verifier_trace_cache_run_count,
    }
    metadata.update({
        key: value
        for key, value in optional_metadata.items()
        if value is not None
    })
    manifest = build_artifact_manifest(
        {"route_comparison_report": route_report_path},
        root=tmp_path,
        metadata=metadata,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def _write_adapter_family_matrix(
    path: Path,
    *,
    routes: tuple[str, ...] = ("structured_qa", "structured_state", "state_transition"),
    blocked_route: str | None = None,
    promotion_status: str = "promote",
) -> Path:
    families = []
    by_route = {}
    for route in routes:
        status = "blocked" if route == blocked_route else "promote"
        families.append({
            "route": route,
            "status": status,
            "selected": 8,
            "decision_accuracy": 1.0,
            "false_supported_rate": 0.0,
            "false_refuted_rate": 1.0,
        })
        by_route[route] = {
            "selected": 8,
            "decision_accuracy": 1.0,
            "false_supported_rate": 0.0,
            "false_refuted_rate": 1.0,
            "promotion_status": status,
        }
    path.write_text(
        json.dumps({
            "schema_version": 1,
            "workflow": "adapter_family_matrix",
            "alpha": 0.2,
            "n_records": 8,
            "routes": list(routes),
            "families": families,
            "route_comparison": {
                "quality_gate": {"passed": promotion_status == "promote" and blocked_route is None},
                "by_route": by_route,
            },
            "promotion_decision": {"status": promotion_status},
        }),
        encoding="utf-8",
    )
    return path


def test_compare_route_baselines_recommends_registered_route_manifest(tmp_path):
    module = importlib.import_module("benchmarks.compare_route_baselines")
    from eigentruth.registry import ArtifactRegistry

    registry_path = tmp_path / "registry.json"
    fast_manifest = _write_route_baseline_manifest(
        tmp_path,
        name="fast",
        route="structured_state",
        decision_accuracy=0.95,
        false_supported_rate=0.02,
        false_refuted_rate=0.90,
        mean_duration_seconds=0.01,
        p99_duration_seconds=0.02,
    )
    accurate_manifest = _write_route_baseline_manifest(
        tmp_path,
        name="accurate",
        route="state_transition",
        decision_accuracy=1.0,
        false_supported_rate=0.0,
        false_refuted_rate=1.0,
        mean_duration_seconds=0.05,
        p99_duration_seconds=0.10,
    )
    ArtifactRegistry.load_json(registry_path).record_benchmark_manifest(
        name="fast-route",
        path=fast_manifest,
        version="0.1",
        metadata={"manifest_metadata": {"runner": "run_adapter_promotion_workflow"}},
    ).record_benchmark_manifest(
        name="accurate-route",
        path=accurate_manifest,
        version="0.1",
        metadata={"manifest_metadata": {"runner": "run_adapter_promotion_workflow"}},
    ).save_json()

    ungated = module.compare_route_baselines(registry_path=registry_path)
    gated = module.compare_route_baselines(
        registry_path=registry_path,
        max_p99_duration_seconds=0.03,
        min_decision_accuracy=0.90,
    )

    assert ungated["decision"]["status"] == "promote"
    assert ungated["decision"]["recommended_record"] == "benchmark_manifest:accurate-route:0.1"
    assert ungated["decision"]["recommended_route"] == "state_transition"
    assert gated["decision"]["status"] == "promote"
    assert gated["decision"]["recommended_record"] == "benchmark_manifest:fast-route:0.1"
    assert gated["leaderboard"][0]["p99_duration_seconds"] == pytest.approx(0.02)
    assert gated["leaderboard"][1]["gate"]["passed"] is False


def test_compare_route_baselines_blocks_invalid_source_metrics(tmp_path):
    module = importlib.import_module("benchmarks.compare_route_baselines")
    from eigentruth.registry import ArtifactRegistry

    registry_path = tmp_path / "registry.json"
    manifest_path = _write_route_baseline_manifest(
        tmp_path,
        name="invalid",
        route="structured_state",
        decision_accuracy=1.0,
        false_supported_rate=0.0,
        false_refuted_rate=1.0,
        mean_duration_seconds=0.01,
        p99_duration_seconds=0.02,
        invalid_metric_counts={"mean_duration_seconds": 1},
    )
    ArtifactRegistry.load_json(registry_path).record_benchmark_manifest(
        name="invalid-route",
        path=manifest_path,
        version="0.1",
        metadata={"manifest_metadata": {"runner": "run_adapter_promotion_workflow"}},
    ).save_json()

    payload = module.compare_route_baselines(registry_path=registry_path)

    assert payload["decision"]["status"] == "blocked"
    assert payload["summary"]["passing_count"] == 0
    assert "invalid source metrics" in payload["leaderboard"][0]["gate"]["blocking_reasons"][0]


def test_runtime_budget_policy_fails_closed_for_missing_or_nonfinite_metrics():
    module = importlib.import_module("benchmarks.runtime_budget_policy")

    passing = module.evaluate_runtime_budget(
        {
            "total_seconds": 1.2,
            "retrieval_hit_count": 3,
            "claims_cache_hit_rate": 1.0,
            "verifier_trace_cache_hit_rate": 0.5,
        },
        module.RuntimeBudgetPolicy(
            max_total_seconds=2.0,
            max_retrieval_hit_count=3,
            min_claims_cache_hit_rate=1.0,
            min_verifier_trace_cache_hit_rate=0.5,
        ),
    )
    nonfinite = module.evaluate_runtime_budget(
        {"total_seconds": float("nan")},
        module.RuntimeBudgetPolicy(max_total_seconds=2.0),
    )
    missing_cache = module.evaluate_runtime_budget(
        {},
        module.RuntimeBudgetPolicy(min_verifier_trace_cache_hit_rate=0.9),
    )

    assert passing["passed"] is True
    assert nonfinite["passed"] is False
    assert nonfinite["failures"][0]["metric"] == "total_seconds"
    assert nonfinite["failures"][0]["reason"] == "missing or non-finite"
    assert missing_cache["passed"] is False
    assert missing_cache["failures"][0]["metric"] == "verifier_trace_cache_hit_rate"


def test_compare_route_baselines_applies_runtime_budget_metadata(tmp_path):
    module = importlib.import_module("benchmarks.compare_route_baselines")
    from eigentruth.registry import ArtifactRegistry

    registry_path = tmp_path / "registry.json"
    manifest_path = _write_route_baseline_manifest(
        tmp_path,
        name="retrieval-runtime",
        route="retrieval_groundedness",
        decision_accuracy=1.0,
        false_supported_rate=0.0,
        false_refuted_rate=1.0,
        mean_duration_seconds=0.01,
        p99_duration_seconds=0.02,
        mean_attempted_route_count=2.0,
        retrieval_use_rate=1.0,
        runtime_total_seconds=5.0,
        runtime_n_retrieval_hits=12,
        claims_cache_enabled=True,
        claims_cache_hit=False,
        verifier_trace_cache_enabled=True,
        verifier_trace_cache_hit_count=0,
        verifier_trace_cache_run_count=1,
    )
    ArtifactRegistry.load_json(registry_path).record_benchmark_manifest(
        name="retrieval-runtime-route",
        path=manifest_path,
        version="0.1",
        metadata={"manifest_metadata": {"runner": "run_adapter_promotion_workflow"}},
    ).save_json()

    payload = module.compare_route_baselines(
        registry_path=registry_path,
        max_runtime_total_seconds=1.0,
        max_retrieval_hit_count=5,
        min_claims_cache_hit_rate=0.5,
        min_verifier_trace_cache_hit_rate=0.9,
    )

    row = payload["leaderboard"][0]
    reasons = row["gate"]["blocking_reasons"]
    assert payload["decision"]["status"] == "blocked"
    assert row["runtime_total_seconds"] == pytest.approx(5.0)
    assert row["runtime_retrieval_hit_count"] == pytest.approx(12.0)
    assert row["claims_cache_hit_rate"] == pytest.approx(0.0)
    assert row["verifier_trace_cache_hit_rate"] == pytest.approx(0.0)
    assert "runtime_budget: total_seconds above 1.0" in reasons
    assert "runtime_budget: retrieval_hit_count above 5.0" in reasons
    assert "runtime_budget: claims_cache_hit_rate below 0.5" in reasons
    assert "runtime_budget: verifier_trace_cache_hit_rate below 0.9" in reasons


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


def test_run_adapter_family_matrix_can_include_retrieval_route(tmp_path):
    module = importlib.import_module("benchmarks.run_adapter_family_matrix")

    payload = module.run_adapter_family_matrix(
        module.AdapterFamilyMatrixConfig(
            output_dir=tmp_path,
            n_records=8,
            alpha=0.2,
            include_retrieval=True,
            max_mean_attempted_route_count=2.1,
            max_retrieval_use_rate=1.0,
            compact_json=True,
        )
    )
    by_route = payload["route_comparison"]["by_route"]
    families = {item["route"]: item for item in payload["families"]}
    retrieval = families["retrieval_groundedness"]

    assert set(families) == {
        "structured_qa",
        "structured_state",
        "state_transition",
        "retrieval_groundedness",
    }
    assert payload["routes"] == (
        "structured_qa",
        "structured_state",
        "state_transition",
        "retrieval_groundedness",
    )
    assert payload["include_retrieval"] is True
    assert payload["promotion_decision"]["status"] == "promote"
    assert payload["route_comparison"]["quality_gate"]["passed"] is True
    assert retrieval["status"] == "promote"
    assert retrieval["selected"] == 8
    assert retrieval["decision_accuracy"] == pytest.approx(1.0)
    assert retrieval["false_supported_rate"] == pytest.approx(0.0)
    assert retrieval["false_refuted_rate"] == pytest.approx(1.0)
    assert retrieval["mean_attempted_route_count"] == pytest.approx(2.0)
    assert retrieval["retrieval_use_rate"] == pytest.approx(1.0)
    assert by_route["retrieval_groundedness"]["selected"] == 8
    assert by_route["retrieval_groundedness"]["retrieval_use_rate"] == pytest.approx(1.0)
    assert Path(retrieval["verifier_report_path"]).exists()
    assert (tmp_path / "retrieval_groundedness" / "retrieval-claims.json").exists()


def test_run_adapter_family_matrix_can_include_retrieval_structured_qa_route(tmp_path):
    module = importlib.import_module("benchmarks.run_adapter_family_matrix")

    payload = module.run_adapter_family_matrix(
        module.AdapterFamilyMatrixConfig(
            output_dir=tmp_path,
            n_records=8,
            alpha=0.2,
            include_retrieval_structured_qa=True,
            max_mean_attempted_route_count=2.1,
            max_retrieval_use_rate=1.0,
            compact_json=True,
        )
    )
    by_route = payload["route_comparison"]["by_route"]
    families = {item["route"]: item for item in payload["families"]}
    retrieval = families["retrieval_structured_qa"]
    claims = json.loads(
        (tmp_path / "retrieval_structured_qa" / "retrieval-qa-claims.json").read_text(
            encoding="utf-8"
        )
    )

    assert set(families) == {
        "structured_qa",
        "structured_state",
        "state_transition",
        "retrieval_structured_qa",
    }
    assert payload["routes"] == (
        "structured_qa",
        "structured_state",
        "state_transition",
        "retrieval_structured_qa",
    )
    assert payload["retrieval_routes"] == ("retrieval_structured_qa",)
    assert payload["include_retrieval_structured_qa"] is True
    assert payload["promotion_decision"]["status"] == "promote"
    assert payload["route_comparison"]["quality_gate"]["passed"] is True
    assert retrieval["status"] == "promote"
    assert retrieval["selected"] == 8
    assert retrieval["decision_accuracy"] == pytest.approx(1.0)
    assert retrieval["false_supported_rate"] == pytest.approx(0.0)
    assert retrieval["false_refuted_rate"] == pytest.approx(1.0)
    assert retrieval["mean_attempted_route_count"] == pytest.approx(2.0)
    assert retrieval["retrieval_use_rate"] == pytest.approx(1.0)
    assert by_route["retrieval_structured_qa"]["selected"] == 8
    assert by_route["retrieval_structured_qa"]["retrieval_use_rate"] == pytest.approx(1.0)
    assert claims["records"][0]["retrieval_documents"][0]["metadata"]["question"]
    assert claims["records"][0]["retrieval_documents"][0]["metadata"]["answer"]
    assert Path(retrieval["verifier_report_path"]).exists()


def test_run_adapter_readiness_workflow_requires_real_performance_evidence(tmp_path):
    module = importlib.import_module("benchmarks.run_adapter_readiness_workflow")
    report_path = tmp_path / "readiness.json"

    payload = module.run_adapter_readiness_workflow(
        module.AdapterReadinessWorkflowConfig(
            output_dir=tmp_path,
            readiness_report_path=report_path,
            n_records=8,
            alpha=0.2,
            performance_dry_run=True,
            prefix_kv_cache=True,
            max_batch_tokens=77,
            max_batch_token_budgets=(0, 77),
            compact_json=True,
        )
    )
    written = json.loads(report_path.read_text(encoding="utf-8"))

    assert payload["adapter_family_matrix"]["promotion_decision"]["status"] == "promote"
    assert payload["performance_matrix"]["matrix_decision"]["status"] == "dry_run"
    assert payload["performance_matrix"]["config"]["max_batch_tokens"] == 0
    assert payload["performance_matrix"]["config"]["max_batch_token_budgets"] == (0, 77)
    assert payload["execution"]["wall_clock_seconds"] >= 0.0
    assert payload["execution"]["performance_wall_clock_seconds"] >= 0.0
    assert payload["execution"]["performance_max_workers"] == 1
    assert payload["readiness_decision"]["status"] == "needs_performance_evidence"
    assert payload["runtime_recommendation"]["status"] == "needs_evidence"
    assert payload["readiness_decision"]["recommended_route"] in {
        "structured_qa",
        "structured_state",
        "state_transition",
    }
    assert Path(payload["artifact_manifest"]).exists()
    assert Path(payload["adapter_family_matrix_path"]).exists()
    assert Path(payload["performance_matrix_path"]).exists()
    assert Path(payload["runtime_recommendation_path"]).exists()
    assert written["readiness_decision"]["status"] == "needs_performance_evidence"
    runtime_recommendation = json.loads(Path(payload["runtime_recommendation_path"]).read_text(encoding="utf-8"))
    assert runtime_recommendation["status"] == "needs_evidence"
    manifest = json.loads(Path(payload["artifact_manifest"]).read_text(encoding="utf-8"))
    verification = importlib.import_module("eigentruth.registry").load_and_verify_artifact_manifest(
        payload["artifact_manifest"],
        recursive=True,
    )
    assert manifest["metadata"]["runner"] == "run_adapter_readiness_workflow"
    assert manifest["metadata"]["readiness_status"] == "needs_performance_evidence"
    assert manifest["metadata"]["prefix_kv_cache"] is True
    assert manifest["metadata"]["runtime_recommendation_status"] == "needs_evidence"
    assert manifest["metadata"]["wall_clock_seconds"] >= 0.0
    assert manifest["metadata"]["performance_wall_clock_seconds"] >= 0.0
    assert manifest["artifacts"]["readiness_report"]["exists"] is True
    assert manifest["artifacts"]["adapter_family_matrix"]["exists"] is True
    assert manifest["artifacts"]["adapter_family_route_comparison"]["exists"] is True
    assert manifest["artifacts"]["performance_matrix_manifest"]["exists"] is True
    assert manifest["artifacts"]["runtime_recommendation"]["exists"] is True
    assert verification.passed is True
    assert verification.nested


def test_run_adapter_readiness_workflow_promotes_when_quality_and_performance_pass(tmp_path, monkeypatch):
    module = importlib.import_module("benchmarks.run_adapter_readiness_workflow")

    def fake_run_matrix(config, *, clean, dry_run):
        config.output_dir.mkdir(parents=True, exist_ok=True)
        config.report_path.write_text("{}", encoding="utf-8")
        result_path = config.output_dir / "cache-only-result.json"
        result_path.write_text(
            json.dumps({
                "auroc": {
                    "truth_proj": 0.8,
                    "subspace_resid": 0.91,
                    "nll_answer": 0.76,
                },
            }),
            encoding="utf-8",
        )
        return {
            "dry_run": False,
            "report_path": str(config.report_path),
            "config": {
                "max_workers": config.max_workers,
                "length_bucketed_batches": config.length_bucketed_batches,
            },
            "matrix_decision": {
                "status": "promote",
                "recommended_cell": "layer_m1_batch_2_capture_outputs",
                "recommendation_metric": "cache_only_total_seconds",
                "recommended": {
                    "id": "layer_m1_batch_2_capture_outputs",
                    "layer": -1,
                    "batch_size": 2,
                    "hidden_state_capture": "outputs",
                    "max_batch_tokens": 0,
                    "prefix_kv_cache": False,
                    "cache_only_total_seconds": 0.11,
                    "truth_proj_auroc": 0.8,
                },
            },
            "cells": [
                {
                    "id": "layer_m1_batch_2_capture_outputs",
                    "layer": -1,
                    "batch_size": 2,
                    "hidden_state_capture": "outputs",
                    "summary": {
                        "quality_signals": {
                            "truth_proj": 0.8,
                            "subspace_resid": 0.91,
                        },
                        "truth_proj_auroc": 0.8,
                    },
                    "triplet": {"results": {"cache_only": str(result_path)}},
                }
            ],
            "execution": {"wall_clock_seconds": 1.2, "max_workers": config.max_workers},
        }

    monkeypatch.setattr(module, "run_matrix", fake_run_matrix)

    inside_report_path = _write_inside_sampling_profile(
        tmp_path,
        sample_count_ratio=0.4,
        generation_seconds_ratio=0.45,
        total_generated_samples=8,
    )
    payload = module.run_adapter_readiness_workflow(
        module.AdapterReadinessWorkflowConfig(
            output_dir=tmp_path,
            n_records=8,
            alpha=0.2,
            batch_sizes=(2,),
            performance_dry_run=False,
            inside_sampling_report_path=inside_report_path,
        )
    )

    assert payload["readiness_decision"]["status"] == "promote"
    assert payload["readiness_decision"]["recommended_route"] in {
        "structured_qa",
        "structured_state",
        "state_transition",
    }
    assert payload["readiness_decision"]["recommended_performance_cell"] == "layer_m1_batch_2_capture_outputs"
    assert payload["runtime_recommendation"]["status"] == "promote"
    assert payload["runtime_recommendation"]["recommendation"]["batch_size"] == 2
    assert payload["runtime_recommendation"]["recommendation"]["best_quality_signal"] == {
        "name": "subspace_resid",
        "auroc": pytest.approx(0.91),
    }
    assert payload["runtime_recommendation"]["recommendation"]["inside_sampling"]["recommended_run"] == (
        "adaptive_selfcheck"
    )
    assert payload["runtime_recommendation"]["recommendation"]["inside_sampling"][
        "sample_count_ratio_to_baseline"
    ] == pytest.approx(0.4)
    assert payload["runtime_recommendation"]["benchmark_flags"]["run_adapter_readiness_workflow"] == [
        "--layers",
        "-1",
        "--batch-sizes",
        "2",
        "--hidden-state-captures",
        "outputs",
        "--max-workers",
        "1",
    ]
    runtime_recommendation = json.loads(Path(payload["runtime_recommendation_path"]).read_text(encoding="utf-8"))
    assert runtime_recommendation["status"] == "promote"
    assert runtime_recommendation["benchmark_flags"]["run_inside_sampling_profile"][:2] == [
        "--inside-samples",
        "5",
    ]
    manifest = json.loads(Path(payload["artifact_manifest"]).read_text(encoding="utf-8"))
    assert manifest["artifacts"]["runtime_recommendation"]["exists"] is True
    assert manifest["artifacts"]["inside_sampling_profile_report"]["exists"] is True
    assert manifest["metadata"]["runtime_recommendation_status"] == "promote"
    assert manifest["metadata"]["recommended_batch_size"] == 2
    assert manifest["metadata"]["recommended_best_quality_signal"] == "subspace_resid"
    assert manifest["metadata"]["recommended_best_quality_auroc"] == pytest.approx(0.91)
    assert manifest["metadata"]["recommended_inside_sampling"]["recommended_run"] == "adaptive_selfcheck"


def test_run_adapter_readiness_workflow_can_reuse_performance_report(tmp_path, monkeypatch):
    module = importlib.import_module("benchmarks.run_adapter_readiness_workflow")
    performance_dir = tmp_path / "previous-performance"
    performance_dir.mkdir()
    performance_report_path = performance_dir / "cache-profile-matrix-report.json"
    result_path = performance_dir / "cache-only-result.json"
    trigger_sweep_path = tmp_path / "inside-trigger-budget-sweep.json"
    result_path.write_text(
        json.dumps({
            "auroc": {
                "truth_proj": 0.82,
                "subspace_resid": 0.92,
                "nll_answer": 0.73,
            },
        }),
        encoding="utf-8",
    )
    performance_report_path.write_text(
        json.dumps({
            "dry_run": False,
            "report_path": str(performance_report_path),
            "config": {
                "max_workers": 1,
                "length_bucketed_batches": True,
            },
            "matrix_decision": {
                "status": "promote",
                "recommended_cell": "layer_m1_batch_2_capture_outputs",
                "recommendation_metric": "cache_only_total_seconds",
                "blocking_reasons": (),
                "recommended": {
                    "id": "layer_m1_batch_2_capture_outputs",
                    "layer": -1,
                    "batch_size": 2,
                    "hidden_state_capture": "outputs",
                    "max_batch_tokens": 0,
                    "prefix_kv_cache": False,
                    "cache_only_total_seconds": 0.11,
                    "truth_proj_auroc": 0.82,
                },
            },
            "cells": [
                {
                    "id": "layer_m1_batch_2_capture_outputs",
                    "layer": -1,
                    "batch_size": 2,
                    "hidden_state_capture": "outputs",
                    "summary": {
                        "quality_signals": {
                            "truth_proj": 0.82,
                            "subspace_resid": 0.92,
                        },
                        "truth_proj_auroc": 0.82,
                    },
                    "triplet": {"results": {"cache_only": str(result_path)}},
                }
            ],
            "execution": {"wall_clock_seconds": 1.2, "max_workers": 1},
        }),
        encoding="utf-8",
    )
    trigger_sweep_path.write_text(
        json.dumps({
            "workflow": "inside_trigger_budget_sweep",
            "dry_run": False,
            "derived_from_max_budget": True,
            "derived_source_budget_id": "top_0p4",
            "config": {
                "trigger_signal": "truth_proj",
                "budgets": [
                    {"kind": "top_fraction", "value": 0.25, "id": "top_0p25"},
                    {"kind": "top_fraction", "value": 0.4, "id": "top_0p4"},
                ],
                "inside_samples": 5,
                "inside_batch_size": 1,
                "inside_max_new_tokens": 12,
                "inside_min_samples": 2,
                "inside_sample_step": 1,
                "inside_stability_delta": 0.05,
                "inside_selfcheck_min_overlap": 0.65,
                "inside_selfcheck_support_threshold": 0.6,
                "inside_selfcheck_refute_threshold": 0.5,
                "run_names": ["adaptive_selfcheck"],
                "derive_from_max_budget": True,
            },
            "budgets": {
                "top_0p25": {"sample_efficiency_gate": {"passed": True}},
                "top_0p4": {"sample_efficiency_gate": {"passed": True}},
            },
            "leaderboard": [
                {
                    "budget_id": "top_0p25",
                    "budget_kind": "top_fraction",
                    "budget_value": 0.25,
                    "recommended_run": "adaptive_selfcheck",
                    "total_generated_samples": 108,
                    "inside_generation_seconds": 116.0,
                    "inside_auroc": {"inside_semantic_entropy": 0.52},
                },
                {
                    "budget_id": "top_0p4",
                    "budget_kind": "top_fraction",
                    "budget_value": 0.4,
                    "recommended_run": "adaptive_selfcheck",
                    "total_generated_samples": 218,
                    "inside_generation_seconds": 235.0,
                    "inside_auroc": {"inside_semantic_entropy": 0.57},
                },
            ],
            "recommendation": {
                "budget_id": "top_0p25",
                "recommended_run": "adaptive_selfcheck",
                "reason": "lowest_total_generated_samples_then_inside_generation_seconds",
            },
            "quality_balanced_recommendation": {
                "budget_id": "top_0p4",
                "recommended_run": "adaptive_selfcheck",
                "reason": "lowest_cost_within_inside_quality_tolerance",
                "quality_metric": "inside_semantic_entropy",
                "quality_value": 0.57,
                "best_quality_value": 0.57,
                "quality_tolerance": 0.02,
                "cost_metric": "inside_generation_seconds",
                "cost_value": 235.0,
            },
        }),
        encoding="utf-8",
    )

    def fail_run_matrix(*args, **kwargs):
        raise AssertionError("performance matrix should be reused")

    monkeypatch.setattr(module, "run_matrix", fail_run_matrix)
    payload = module.run_adapter_readiness_workflow(
        module.AdapterReadinessWorkflowConfig(
            output_dir=tmp_path / "readiness",
            n_records=8,
            alpha=0.2,
            include_retrieval=True,
            include_retrieval_structured_qa=True,
            max_mean_attempted_route_count=2.1,
            max_retrieval_use_rate=1.0,
            performance_report_path=performance_report_path,
            inside_trigger_budget_sweep_report_path=trigger_sweep_path,
            inside_trigger_budget_policy="cost_first",
        )
    )
    manifest = json.loads(Path(payload["artifact_manifest"]).read_text(encoding="utf-8"))

    assert payload["readiness_decision"]["status"] == "promote"
    assert payload["adapter_family_matrix"]["include_retrieval"] is True
    assert payload["adapter_family_matrix"]["include_retrieval_structured_qa"] is True
    assert "retrieval_groundedness" in payload["adapter_family_matrix"]["routes"]
    assert "retrieval_structured_qa" in payload["adapter_family_matrix"]["routes"]
    assert payload["adapter_family_matrix"]["route_comparison"]["by_route"]["retrieval_groundedness"][
        "retrieval_use_rate"
    ] == pytest.approx(1.0)
    assert payload["adapter_family_matrix"]["route_comparison"]["by_route"]["retrieval_structured_qa"][
        "retrieval_use_rate"
    ] == pytest.approx(1.0)
    assert payload["performance_matrix_path"] == str(performance_report_path)
    assert payload["execution"]["performance_report_reused"] is True
    assert payload["runtime_recommendation"]["recommendation"]["batch_size"] == 2
    assert payload["runtime_recommendation"]["recommendation"]["best_quality_signal"] == {
        "name": "subspace_resid",
        "auroc": pytest.approx(0.92),
    }
    assert payload["runtime_recommendation"]["recommendation"]["inside_trigger_budget_sweep"][
        "recommended_budget_id"
    ] == "top_0p25"
    assert payload["runtime_recommendation"]["recommendation"]["inside_trigger_budget_sweep"][
        "selection_policy"
    ] == "cost_first"
    assert payload["runtime_recommendation"]["recommendation"]["inside_sampling"][
        "inside_trigger_top_fraction"
    ] == pytest.approx(0.25)
    assert manifest["metadata"]["performance_report_reused"] is True
    assert manifest["metadata"]["adapter_include_retrieval"] is True
    assert manifest["metadata"]["adapter_include_retrieval_structured_qa"] is True
    assert manifest["metadata"]["adapter_retrieval_limit"] == 1
    assert manifest["metadata"]["inside_trigger_budget_policy"] == "cost_first"
    assert manifest["metadata"]["performance_report_path"] == str(performance_report_path)
    assert manifest["artifacts"]["performance_matrix_report"]["exists"] is True
    assert manifest["artifacts"]["inside_trigger_budget_sweep_report"]["exists"] is True
    assert manifest["metadata"]["recommended_inside_trigger_budget_sweep"][
        "recommended_budget_id"
    ] == "top_0p25"
    assert manifest["metadata"]["recommended_inside_trigger_budget_policy"] == "cost_first"


def test_adapter_readiness_decision_blocks_on_runtime_budget():
    module = importlib.import_module("benchmarks.run_adapter_readiness_workflow")

    decision = module.build_readiness_decision(
        {"promotion_decision": {"status": "promote", "recommended_route": "structured_qa"}},
        {"matrix_decision": {"status": "promote", "recommended_cell": "layer_m1_batch_2_capture_outputs"}},
        {"status": "promote"},
        runtime_budget={
            "enabled": True,
            "passed": False,
            "failures": ({"metric": "total_seconds"},),
        },
    )

    assert decision["status"] == "blocked"
    assert decision["runtime_budget_passed"] is False
    assert "runtime budget did not pass: total_seconds" in decision["blocking_reasons"]


def test_run_adapter_readiness_workflow_blocks_when_performance_blocks(tmp_path, monkeypatch):
    module = importlib.import_module("benchmarks.run_adapter_readiness_workflow")

    def fake_run_matrix(config, *, clean, dry_run):
        config.output_dir.mkdir(parents=True, exist_ok=True)
        config.report_path.write_text("{}", encoding="utf-8")
        return {
            "dry_run": False,
            "report_path": str(config.report_path),
            "matrix_decision": {
                "status": "blocked",
                "recommended_cell": "layer_m1_batch_2_capture_outputs",
                "failed_cells": ("layer_m2_batch_2_capture_outputs",),
            },
        }

    monkeypatch.setattr(module, "run_matrix", fake_run_matrix)

    payload = module.run_adapter_readiness_workflow(
        module.AdapterReadinessWorkflowConfig(
            output_dir=tmp_path,
            n_records=8,
            alpha=0.2,
            batch_sizes=(2,),
            performance_dry_run=False,
        )
    )

    assert payload["readiness_decision"]["status"] == "blocked"
    assert payload["readiness_decision"]["adapter_family_promoted"] is True
    assert payload["readiness_decision"]["performance_promoted"] is False
    assert "performance matrix decision did not promote" in payload["readiness_decision"]["blocking_reasons"]


def test_run_adapter_readiness_workflow_blocks_when_runtime_recommendation_missing(tmp_path, monkeypatch):
    module = importlib.import_module("benchmarks.run_adapter_readiness_workflow")

    def fake_run_matrix(config, *, clean, dry_run):
        del clean, dry_run
        config.output_dir.mkdir(parents=True, exist_ok=True)
        config.report_path.write_text("{}", encoding="utf-8")
        return {
            "dry_run": False,
            "report_path": str(config.report_path),
            "matrix_decision": {
                "status": "promote",
                "recommended_cell": "layer_m1_batch_2_capture_outputs",
                "recommended": {"id": "layer_m1_batch_2_capture_outputs"},
            },
        }

    monkeypatch.setattr(module, "run_matrix", fake_run_matrix)

    payload = module.run_adapter_readiness_workflow(
        module.AdapterReadinessWorkflowConfig(
            output_dir=tmp_path,
            n_records=8,
            alpha=0.2,
            batch_sizes=(2,),
            performance_dry_run=False,
        )
    )

    assert payload["readiness_decision"]["status"] == "blocked"
    assert payload["readiness_decision"]["performance_promoted"] is True
    assert payload["readiness_decision"]["runtime_recommendation_promoted"] is False
    assert payload["runtime_recommendation"]["status"] == "no_candidate"
    assert (
        "runtime recommendation did not produce deployable settings"
        in payload["readiness_decision"]["blocking_reasons"]
    )


def test_run_adapter_readiness_registry_workflow_promotes_manifest(tmp_path, monkeypatch):
    module = importlib.import_module("benchmarks.run_adapter_readiness_registry_workflow")
    from eigentruth.registry import ArtifactRegistry

    def fake_readiness_workflow(config):
        return _write_fake_readiness_report(config.output_dir, status="promote", runtime_status="promote")

    monkeypatch.setattr(module, "run_adapter_readiness_workflow", fake_readiness_workflow)
    registry_path = tmp_path / "registry.json"

    payload = module.run_adapter_readiness_registry_workflow(
        module.AdapterReadinessRegistryWorkflowConfig(
            readiness=module.AdapterReadinessWorkflowConfig(output_dir=tmp_path / "readiness"),
            registry_path=registry_path,
            name="readiness-baseline",
            version="0.5",
            workflow_report_path=tmp_path / "workflow.json",
            promotion_metadata={"scope": "unit"},
        )
    )

    assert payload["decision"]["status"] == "promote"
    assert payload["decision"]["manifest_promoted"] is True
    assert payload["decision"]["manifest_verified"] is True
    assert payload["decision"]["registry_record"] == "benchmark_manifest:readiness-baseline:0.5"
    assert Path(payload["promotion"]["verification_report"]).exists()
    registry = ArtifactRegistry.load_json(registry_path)
    record = registry.get("benchmark_manifest:readiness-baseline:0.5")
    assert record.metadata["workflow"] == "run_adapter_readiness_registry_workflow"
    assert record.metadata["readiness_status"] == "promote"
    assert record.metadata["runtime_recommendation_status"] == "promote"
    assert record.metadata["recommended_batch_size"] == 2
    assert record.metadata["recommended_best_quality_signal"] == "subspace_resid"
    assert record.metadata["recommended_best_quality_auroc"] == pytest.approx(0.91)
    assert record.metadata["recommended_inside_sampling_run"] == "adaptive_selfcheck"
    assert record.metadata["recommended_inside_sampling_sample_count_ratio_to_baseline"] == pytest.approx(0.4)
    assert record.metadata["scope"] == "unit"
    assert (tmp_path / "workflow.json").exists()


def test_run_adapter_readiness_registry_workflow_blocks_non_promoted_readiness(tmp_path, monkeypatch):
    module = importlib.import_module("benchmarks.run_adapter_readiness_registry_workflow")

    def fake_readiness_workflow(config):
        return _write_fake_readiness_report(
            config.output_dir,
            status="needs_performance_evidence",
            runtime_status="needs_evidence",
        )

    monkeypatch.setattr(module, "run_adapter_readiness_workflow", fake_readiness_workflow)

    payload = module.run_adapter_readiness_registry_workflow(
        module.AdapterReadinessRegistryWorkflowConfig(
            readiness=module.AdapterReadinessWorkflowConfig(output_dir=tmp_path / "readiness"),
            registry_path=tmp_path / "registry.json",
            name="readiness-baseline",
            version="0.5",
        )
    )

    assert payload["decision"]["status"] == "blocked"
    assert payload["decision"]["manifest_promoted"] is False
    assert payload["decision"]["manifest_verified"] is False
    assert payload["promotion"] is None
    assert payload["decision"]["blocking_reasons"] == (
        "adapter readiness decision did not promote",
    )


def test_compare_readiness_baselines_recommends_best_quality_signal_from_matrix(tmp_path):
    module = importlib.import_module("benchmarks.compare_readiness_baselines")
    from eigentruth.registry import ArtifactRegistry

    registry_path = tmp_path / "registry.json"
    _write_readiness_baseline_manifest(
        tmp_path / "qwen",
        registry_path=registry_path,
        name="qwen-readiness",
        version="0.5",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        layer=-12,
        quality_signals={"truth_proj": 0.55, "subspace_resid": 0.64},
        uncached_forward_seconds=40.0,
        cache_only_seconds=0.20,
    )
    _write_readiness_baseline_manifest(
        tmp_path / "smollm",
        registry_path=registry_path,
        name="smollm-readiness",
        version="0.5",
        model="HuggingFaceTB/SmolLM2-135M-Instruct",
        layer=-16,
        quality_signals={"truth_proj": 0.61, "subspace_resid": 0.60},
        uncached_forward_seconds=12.0,
        cache_only_seconds=0.12,
    )
    ArtifactRegistry.load_json(registry_path).record_benchmark_manifest(
        name="not-readiness",
        version="0.1",
        path=tmp_path / "missing.json",
        metadata={"workflow": "other"},
    ).save_json()

    payload = module.compare_readiness_baselines(registry_path=registry_path)

    assert payload["decision"]["status"] == "promote"
    assert payload["decision"]["recommended_record"] == "benchmark_manifest:qwen-readiness:0.5"
    assert payload["summary"]["record_count"] == 2
    first = payload["leaderboard"][0]
    assert first["record_key"] == "benchmark_manifest:qwen-readiness:0.5"
    assert first["best_quality_signal"] == {
        "name": "subspace_resid",
        "auroc": pytest.approx(0.64),
    }
    assert first["runtime_recommendation_source"].endswith("performance-matrix.json")
    assert first["uncached_forward_cost_source"] == "uncached_forced_answer_forward_seconds"
    assert first["quality_signals"] == {
        "subspace_resid": pytest.approx(0.64),
        "truth_proj": pytest.approx(0.55),
    }


def test_compare_readiness_baselines_applies_performance_gate(tmp_path):
    module = importlib.import_module("benchmarks.compare_readiness_baselines")

    registry_path = tmp_path / "registry.json"
    _write_readiness_baseline_manifest(
        tmp_path / "slow",
        registry_path=registry_path,
        name="slow-high-quality",
        version="0.5",
        model="slow-model",
        layer=-12,
        quality_signals={"truth_proj": 0.70},
        uncached_forward_seconds=90.0,
        cache_only_seconds=0.20,
    )
    _write_readiness_baseline_manifest(
        tmp_path / "fast",
        registry_path=registry_path,
        name="fast-acceptable-quality",
        version="0.5",
        model="fast-model",
        layer=-16,
        quality_signals={"truth_proj": 0.66},
        uncached_forward_seconds=15.0,
        cache_only_seconds=0.10,
    )

    payload = module.compare_readiness_baselines(
        registry_path=registry_path,
        min_best_quality_auroc=0.65,
        max_uncached_forward_seconds=20.0,
    )

    assert payload["decision"]["status"] == "promote"
    assert payload["decision"]["recommended_record"] == "benchmark_manifest:fast-acceptable-quality:0.5"
    blocked = next(row for row in payload["leaderboard"] if row["record_key"].endswith("slow-high-quality:0.5"))
    assert blocked["gate"]["passed"] is False
    assert "uncached forward cost seconds above 20.0" in blocked["gate"]["blocking_reasons"]


def test_compare_readiness_baselines_applies_inside_sampling_cost_gate(tmp_path):
    module = importlib.import_module("benchmarks.compare_readiness_baselines")

    registry_path = tmp_path / "registry.json"
    _write_readiness_baseline_manifest(
        tmp_path / "expensive-sampling",
        registry_path=registry_path,
        name="expensive-sampling",
        version="0.5",
        model="expensive-model",
        layer=-12,
        quality_signals={"truth_proj": 0.79},
        uncached_forward_seconds=14.0,
        cache_only_seconds=0.15,
        inside_sample_ratio=0.9,
        inside_generation_ratio=0.95,
    )
    _write_readiness_baseline_manifest(
        tmp_path / "efficient-sampling",
        registry_path=registry_path,
        name="efficient-sampling",
        version="0.5",
        model="efficient-model",
        layer=-16,
        quality_signals={"truth_proj": 0.72},
        uncached_forward_seconds=16.0,
        cache_only_seconds=0.18,
        inside_sample_ratio=0.4,
        inside_generation_ratio=0.45,
    )

    payload = module.compare_readiness_baselines(
        registry_path=registry_path,
        min_best_quality_auroc=0.70,
        max_inside_sample_count_ratio=0.6,
        max_inside_generation_seconds_ratio=0.8,
    )

    assert payload["decision"]["status"] == "promote"
    assert payload["decision"]["recommended_record"] == "benchmark_manifest:efficient-sampling:0.5"
    recommended = payload["leaderboard"][0]
    assert recommended["inside_sampling_recommended_run"] == "adaptive_selfcheck"
    assert recommended["inside_sampling_total_generated_samples"] == 8
    assert recommended["inside_sampling_sample_count_ratio_to_baseline"] == pytest.approx(0.4)
    assert recommended["inside_generation_seconds_ratio_to_baseline"] == pytest.approx(0.45)
    assert recommended["inside_sampling_stop_reason_counts"] == {
        "selfcheck_refute_threshold_guaranteed": 4,
    }
    blocked = next(row for row in payload["leaderboard"] if row["record_key"].endswith("expensive-sampling:0.5"))
    assert blocked["gate"]["passed"] is False
    assert "INSIDE sampling sample-count ratio above 0.6" in blocked["gate"]["blocking_reasons"]
    assert "INSIDE sampling generation-seconds ratio above 0.8" in blocked["gate"]["blocking_reasons"]


def test_compare_readiness_baselines_applies_trigger_budget_reference_cost_gate(tmp_path):
    module = importlib.import_module("benchmarks.compare_readiness_baselines")

    registry_path = tmp_path / "registry.json"
    _write_readiness_baseline_manifest(
        tmp_path / "expensive-trigger",
        registry_path=registry_path,
        name="expensive-trigger",
        version="0.5",
        model="expensive-model",
        layer=-12,
        quality_signals={"truth_proj": 0.79},
        uncached_forward_seconds=14.0,
        cache_only_seconds=0.15,
        inside_trigger_sample_ratio=0.9,
        inside_trigger_generation_ratio=0.95,
    )
    _write_readiness_baseline_manifest(
        tmp_path / "efficient-trigger",
        registry_path=registry_path,
        name="efficient-trigger",
        version="0.5",
        model="efficient-model",
        layer=-16,
        quality_signals={"truth_proj": 0.72},
        uncached_forward_seconds=16.0,
        cache_only_seconds=0.18,
        inside_trigger_sample_ratio=0.4,
        inside_trigger_generation_ratio=0.45,
    )

    payload = module.compare_readiness_baselines(
        registry_path=registry_path,
        min_best_quality_auroc=0.70,
        max_inside_sample_count_ratio=0.6,
        max_inside_generation_seconds_ratio=0.8,
    )

    assert payload["decision"]["status"] == "promote"
    assert payload["decision"]["recommended_record"] == "benchmark_manifest:efficient-trigger:0.5"
    recommended = payload["leaderboard"][0]
    assert recommended["inside_sampling_sample_count_ratio_to_baseline"] is None
    assert recommended["inside_sampling_sample_count_ratio_to_reference"] == pytest.approx(0.4)
    assert recommended["inside_sampling_sample_count_ratio_for_gate"] == pytest.approx(0.4)
    assert recommended["inside_sampling_sample_count_ratio_source"] == "sample_count_ratio_to_reference"
    assert recommended["inside_generation_seconds_ratio_to_reference"] == pytest.approx(0.45)
    assert recommended["inside_generation_seconds_ratio_for_gate"] == pytest.approx(0.45)
    assert recommended["inside_generation_seconds_ratio_source"] == (
        "inside_generation_seconds_ratio_to_reference"
    )
    assert recommended["inside_trigger_budget_id"] == "top_0p4"
    assert recommended["inside_trigger_budget_policy"] == "quality_balanced"
    assert recommended["inside_trigger_budget_derive_from_max_budget"] is True
    blocked = next(row for row in payload["leaderboard"] if row["record_key"].endswith("expensive-trigger:0.5"))
    assert blocked["gate"]["passed"] is False
    assert "INSIDE sampling sample-count ratio above 0.6" in blocked["gate"]["blocking_reasons"]
    assert "INSIDE sampling generation-seconds ratio above 0.8" in blocked["gate"]["blocking_reasons"]


def test_compare_readiness_baselines_uses_uncached_total_fallback_for_legacy_matrix(tmp_path):
    module = importlib.import_module("benchmarks.compare_readiness_baselines")

    registry_path = tmp_path / "registry.json"
    _write_readiness_baseline_manifest(
        tmp_path / "legacy",
        registry_path=registry_path,
        name="legacy-readiness",
        version="0.4",
        model="legacy-model",
        layer=-16,
        quality_signals={"truth_proj": 0.72},
        uncached_forward_seconds=30.0,
        cache_only_seconds=0.18,
        include_forced_forward=False,
    )

    payload = module.compare_readiness_baselines(
        registry_path=registry_path,
        max_uncached_forward_seconds=40.0,
    )

    row = payload["leaderboard"][0]
    assert payload["decision"]["status"] == "promote"
    assert row["uncached_forced_answer_forward_seconds"] is None
    assert row["uncached_total_seconds"] == pytest.approx(30.0)
    assert row["uncached_forward_cost_seconds"] == pytest.approx(30.0)
    assert row["uncached_forward_cost_source"] == "uncached_total_seconds_fallback"


def test_compare_release_candidates_promotes_readiness_and_route_baselines(tmp_path):
    module = importlib.import_module("benchmarks.compare_release_candidates")
    from eigentruth.registry import ArtifactRegistry

    registry_path = tmp_path / "registry.json"
    _write_readiness_baseline_manifest(
        tmp_path / "readiness",
        registry_path=registry_path,
        name="qwen-readiness",
        version="0.6",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        layer=-12,
        quality_signals={"truth_proj": 0.72, "subspace_resid": 0.68},
        uncached_forward_seconds=18.0,
        cache_only_seconds=0.20,
        inside_sample_ratio=0.4,
        inside_generation_ratio=0.45,
        inside_trigger_sample_ratio=0.3,
        inside_trigger_generation_ratio=0.35,
    )
    route_manifest = _write_route_baseline_manifest(
        tmp_path,
        name="structured",
        route="structured_state",
        decision_accuracy=1.0,
        false_supported_rate=0.0,
        false_refuted_rate=1.0,
        mean_duration_seconds=0.01,
        p99_duration_seconds=0.02,
        runtime_total_seconds=0.8,
        runtime_n_retrieval_hits=0,
        claims_cache_enabled=True,
        claims_cache_hit=True,
        verifier_trace_cache_enabled=True,
        verifier_trace_cache_hit_count=1,
        verifier_trace_cache_run_count=1,
    )
    ArtifactRegistry.load_json(registry_path).record_benchmark_manifest(
        name="structured-route",
        path=route_manifest,
        version="0.6",
        metadata={"manifest_metadata": {"runner": "run_adapter_promotion_workflow"}},
    ).save_json()

    payload = module.compare_release_candidates(
        readiness_registry_path=registry_path,
        min_best_quality_auroc=0.70,
        max_uncached_forward_seconds=20.0,
        max_inside_sample_count_ratio=0.6,
        max_inside_generation_seconds_ratio=0.8,
        min_selected=4,
        min_decision_accuracy=0.99,
        max_false_supported_rate=0.0,
        min_false_refuted_rate=0.99,
        max_p99_duration_seconds=0.03,
        max_runtime_total_seconds=1.0,
        max_retrieval_hit_count=0,
        min_claims_cache_hit_rate=1.0,
        min_verifier_trace_cache_hit_rate=1.0,
    )
    candidate = payload["release_candidate"]
    blocked = module.compare_release_candidates(
        readiness_registry_path=registry_path,
        min_best_quality_auroc=0.70,
        max_uncached_forward_seconds=20.0,
        max_inside_sample_count_ratio=0.6,
        max_inside_generation_seconds_ratio=0.8,
        min_selected=4,
        min_decision_accuracy=0.99,
        max_false_supported_rate=0.0,
        min_false_refuted_rate=0.99,
        max_p99_duration_seconds=0.03,
        max_runtime_total_seconds=0.1,
    )

    assert payload["decision"]["status"] == "promote"
    assert payload["decision"]["recommended_readiness_record"] == "benchmark_manifest:qwen-readiness:0.6"
    assert payload["decision"]["recommended_route_record"] == "benchmark_manifest:structured-route:0.6"
    assert candidate["model"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert candidate["runtime"]["layer"] == -12
    assert candidate["runtime"]["inside_sampling"]["recommended_run"] == "adaptive_selfcheck"
    assert candidate["runtime"]["inside_sampling"]["sample_count_ratio_to_baseline"] == pytest.approx(0.4)
    assert candidate["runtime_cost"]["inside_sampling_recommended_run"] == "adaptive_selfcheck"
    assert candidate["runtime_cost"]["inside_sampling_total_generated_samples"] == 8
    assert candidate["runtime_cost"]["inside_sampling_sample_count_ratio_to_baseline"] == pytest.approx(0.4)
    assert candidate["runtime_cost"]["inside_generation_seconds_ratio_to_baseline"] == pytest.approx(0.45)
    assert candidate["runtime"]["benchmark_flags"]["eval_truthfulqa"][:4] == ["--layer", "-12", "--batch-size", "1"]
    assert candidate["quality"]["best_quality_signal"] == {"name": "truth_proj", "auroc": pytest.approx(0.72)}
    assert candidate["verifier_route"]["route"] == "structured_state"
    assert candidate["verifier_route"]["decision_accuracy"] == pytest.approx(1.0)
    assert candidate["verifier_route"]["runtime_total_seconds"] == pytest.approx(0.8)
    assert candidate["verifier_route"]["claims_cache_hit_rate"] == pytest.approx(1.0)
    assert blocked["decision"]["status"] == "blocked"
    assert blocked["release_candidate"] is None
    assert any(
        "runtime_budget: total_seconds above 0.1" in reason
        for reason in blocked["decision"]["blocking_reasons"][0]["reasons"]
    )


def test_compare_release_candidates_can_require_performance_baseline(tmp_path):
    module = importlib.import_module("benchmarks.compare_release_candidates")
    from eigentruth.registry import ArtifactRegistry

    registry_path = tmp_path / "registry.json"
    _write_readiness_baseline_manifest(
        tmp_path / "readiness",
        registry_path=registry_path,
        name="qwen-readiness",
        version="0.6",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        layer=-12,
        quality_signals={"truth_proj": 0.72, "subspace_resid": 0.68},
        uncached_forward_seconds=18.0,
        cache_only_seconds=0.20,
        inside_trigger_sample_ratio=0.3,
        inside_trigger_generation_ratio=0.35,
    )
    route_manifest = _write_route_baseline_manifest(
        tmp_path,
        name="structured",
        route="structured_state",
        decision_accuracy=1.0,
        false_supported_rate=0.0,
        false_refuted_rate=1.0,
        mean_duration_seconds=0.01,
        p99_duration_seconds=0.02,
    )
    ArtifactRegistry.load_json(registry_path).record_benchmark_manifest(
        name="structured-route",
        path=route_manifest,
        version="0.6",
        metadata={"manifest_metadata": {"runner": "run_adapter_promotion_workflow"}},
    ).save_json()
    _write_performance_baseline_record(
        tmp_path / "performance",
        registry_path=registry_path,
        name="qwen-performance",
        version="0.6",
        layer=-12,
        best_quality_signal_name="truth_proj",
        best_quality_auroc=0.72,
        inside_trigger_budget_id="top_0p4",
        inside_trigger_budget_policy="quality_balanced",
    )

    payload = module.compare_release_candidates(
        readiness_registry_path=registry_path,
        min_best_quality_auroc=0.70,
        max_uncached_forward_seconds=20.0,
        min_selected=4,
        min_decision_accuracy=0.99,
        max_false_supported_rate=0.0,
        min_false_refuted_rate=0.99,
        performance_baseline_key="performance_baseline:qwen-performance:0.6",
    )

    candidate = payload["release_candidate"]
    assert payload["decision"]["status"] == "promote"
    assert payload["decision"]["recommended_performance_baseline_record"] == (
        "performance_baseline:qwen-performance:0.6"
    )
    assert payload["performance_baseline_gate"]["gate"]["passed"] is True
    assert candidate["performance_baseline_record"] == "performance_baseline:qwen-performance:0.6"
    assert candidate["manifests"]["performance_manifest"].endswith("artifact-manifest.json")


def test_compare_release_candidates_can_require_adapter_family_matrix(tmp_path):
    module = importlib.import_module("benchmarks.compare_release_candidates")
    from eigentruth.registry import ArtifactRegistry

    registry_path = tmp_path / "registry.json"
    _write_readiness_baseline_manifest(
        tmp_path / "readiness",
        registry_path=registry_path,
        name="qwen-readiness",
        version="0.6",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        layer=-12,
        quality_signals={"truth_proj": 0.72},
        uncached_forward_seconds=18.0,
        cache_only_seconds=0.20,
    )
    route_manifest = _write_route_baseline_manifest(
        tmp_path,
        name="structured",
        route="structured_state",
        decision_accuracy=1.0,
        false_supported_rate=0.0,
        false_refuted_rate=1.0,
        mean_duration_seconds=0.01,
        p99_duration_seconds=0.02,
    )
    ArtifactRegistry.load_json(registry_path).record_benchmark_manifest(
        name="structured-route",
        path=route_manifest,
        version="0.6",
        metadata={"manifest_metadata": {"runner": "run_adapter_promotion_workflow"}},
    ).save_json()
    matrix_path = _write_adapter_family_matrix(tmp_path / "adapter-family-matrix.json")
    blocked_matrix_path = _write_adapter_family_matrix(
        tmp_path / "blocked-adapter-family-matrix.json",
        blocked_route="retrieval_groundedness",
        routes=("structured_qa", "structured_state", "state_transition", "retrieval_groundedness"),
    )

    promoted = module.compare_release_candidates(
        readiness_registry_path=registry_path,
        min_best_quality_auroc=0.70,
        max_uncached_forward_seconds=20.0,
        min_selected=4,
        min_decision_accuracy=0.99,
        max_false_supported_rate=0.0,
        min_false_refuted_rate=0.99,
        adapter_family_matrix_path=matrix_path,
        required_adapter_routes=("structured_state", "state_transition"),
    )
    blocked = module.compare_release_candidates(
        readiness_registry_path=registry_path,
        min_best_quality_auroc=0.70,
        max_uncached_forward_seconds=20.0,
        min_selected=4,
        min_decision_accuracy=0.99,
        max_false_supported_rate=0.0,
        min_false_refuted_rate=0.99,
        adapter_family_matrix_path=blocked_matrix_path,
        required_adapter_routes=("retrieval_groundedness",),
    )

    assert promoted["decision"]["status"] == "promote"
    assert promoted["decision"]["adapter_family_status"] == "promote"
    assert promoted["decision"]["required_adapter_routes"] == ("structured_state", "state_transition")
    assert promoted["adapter_family_matrix_gate"]["gate"]["passed"] is True
    assert promoted["release_candidate"]["adapter_family_matrix"]["matrix_path"] == str(matrix_path)
    assert promoted["release_candidate"]["manifests"]["adapter_family_matrix_report"] == str(matrix_path)
    assert blocked["decision"]["status"] == "blocked"
    assert blocked["release_candidate"] is None
    assert blocked["decision"]["blocking_reasons"][0]["gate"] == "adapter_family_matrix"
    assert any(
        "required adapter route 'retrieval_groundedness' status is 'blocked'" in reason
        for reason in blocked["decision"]["blocking_reasons"][0]["reasons"]
    )


def test_compare_release_candidates_can_require_extra_route_baselines(tmp_path):
    module = importlib.import_module("benchmarks.compare_release_candidates")
    from eigentruth.registry import ArtifactRegistry

    registry_path = tmp_path / "registry.json"
    _write_readiness_baseline_manifest(
        tmp_path / "readiness",
        registry_path=registry_path,
        name="qwen-readiness",
        version="0.6",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        layer=-12,
        quality_signals={"truth_proj": 0.72},
        uncached_forward_seconds=18.0,
        cache_only_seconds=0.20,
    )
    structured_manifest = _write_route_baseline_manifest(
        tmp_path,
        name="structured",
        route="structured_state",
        decision_accuracy=1.0,
        false_supported_rate=0.0,
        false_refuted_rate=1.0,
        mean_duration_seconds=0.01,
        p99_duration_seconds=0.02,
        runtime_total_seconds=0.8,
        runtime_n_retrieval_hits=0,
    )
    retrieval_manifest = _write_route_baseline_manifest(
        tmp_path,
        name="retrieval",
        route="retrieval_groundedness",
        decision_accuracy=0.96,
        false_supported_rate=0.03,
        false_refuted_rate=0.60,
        mean_duration_seconds=0.04,
        p99_duration_seconds=0.08,
        mean_attempted_route_count=2.0,
        retrieval_use_rate=1.0,
        runtime_total_seconds=2.0,
        runtime_n_retrieval_hits=24,
    )
    registry = ArtifactRegistry.load_json(registry_path)
    registry.record_benchmark_manifest(
        name="structured-route",
        path=structured_manifest,
        version="0.6",
        metadata={"manifest_metadata": {"runner": "run_adapter_promotion_workflow"}},
    )
    registry.record_benchmark_manifest(
        name="retrieval-route",
        path=retrieval_manifest,
        version="0.7",
        metadata={"manifest_metadata": {"runner": "run_local_retrieval_route_workflow"}},
    )
    registry.save_json()

    promoted = module.compare_release_candidates(
        readiness_registry_path=registry_path,
        route_baseline_keys=("benchmark_manifest:structured-route:0.6",),
        required_route_baseline_keys=("benchmark_manifest:retrieval-route:0.7",),
        min_best_quality_auroc=0.70,
        max_uncached_forward_seconds=20.0,
        min_selected=4,
        min_decision_accuracy=0.95,
        max_false_supported_rate=0.05,
        min_false_refuted_rate=0.50,
        max_retrieval_use_rate=0.0,
        max_runtime_total_seconds=1.0,
        required_route_max_runtime_total_seconds=3.0,
        required_route_max_retrieval_hit_count=30,
        required_route_max_retrieval_use_rate=1.0,
    )
    blocked = module.compare_release_candidates(
        readiness_registry_path=registry_path,
        route_baseline_keys=("benchmark_manifest:structured-route:0.6",),
        required_route_baseline_keys=("benchmark_manifest:retrieval-route:0.7",),
        min_best_quality_auroc=0.70,
        max_uncached_forward_seconds=20.0,
        min_selected=4,
        min_decision_accuracy=0.95,
        max_false_supported_rate=0.05,
        min_false_refuted_rate=0.50,
        max_retrieval_use_rate=0.0,
        max_runtime_total_seconds=1.0,
        required_route_max_runtime_total_seconds=1.0,
    )

    assert promoted["decision"]["status"] == "promote"
    assert promoted["decision"]["recommended_route_record"] == "benchmark_manifest:structured-route:0.6"
    assert promoted["decision"]["required_route_baseline_status"] == "promote"
    assert promoted["decision"]["required_route_baseline_records"] == (
        "benchmark_manifest:retrieval-route:0.7",
    )
    candidate = promoted["release_candidate"]
    assert candidate["verifier_route"]["route"] == "structured_state"
    assert candidate["verifier_route"]["retrieval_use_rate"] == pytest.approx(0.0)
    assert candidate["required_route_baselines"]["records"] == (
        "benchmark_manifest:retrieval-route:0.7",
    )
    assert candidate["required_route_baselines"]["routes"] == ("retrieval_groundedness",)
    assert candidate["manifests"]["required_route_manifest_1"] == str(retrieval_manifest)
    assert promoted["required_route_baseline_gate"]["gate"]["passed"] is True
    assert promoted["required_route_baseline_gate"]["comparison"]["config"]["max_runtime_total_seconds"] == (
        pytest.approx(3.0)
    )

    assert blocked["decision"]["status"] == "blocked"
    assert blocked["release_candidate"] is None
    assert blocked["decision"]["blocking_reasons"][0]["gate"] == "required_route_baselines"
    assert any(
        "runtime_budget: total_seconds above 1.0" in reason
        for reason in blocked["decision"]["blocking_reasons"][0]["reasons"]
    )


def test_compare_release_candidates_blocks_mismatched_performance_baseline(tmp_path):
    module = importlib.import_module("benchmarks.compare_release_candidates")
    from eigentruth.registry import ArtifactRegistry

    registry_path = tmp_path / "registry.json"
    _write_readiness_baseline_manifest(
        tmp_path / "readiness",
        registry_path=registry_path,
        name="qwen-readiness",
        version="0.6",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        layer=-12,
        quality_signals={"truth_proj": 0.72},
        uncached_forward_seconds=18.0,
        cache_only_seconds=0.20,
        inside_trigger_sample_ratio=0.3,
        inside_trigger_generation_ratio=0.35,
    )
    route_manifest = _write_route_baseline_manifest(
        tmp_path,
        name="structured",
        route="structured_state",
        decision_accuracy=1.0,
        false_supported_rate=0.0,
        false_refuted_rate=1.0,
        mean_duration_seconds=0.01,
        p99_duration_seconds=0.02,
    )
    ArtifactRegistry.load_json(registry_path).record_benchmark_manifest(
        name="structured-route",
        path=route_manifest,
        version="0.6",
        metadata={"manifest_metadata": {"runner": "run_adapter_promotion_workflow"}},
    ).save_json()
    _write_performance_baseline_record(
        tmp_path / "performance",
        registry_path=registry_path,
        name="qwen-performance",
        version="0.6",
        layer=-12,
        batch_size=2,
        best_quality_signal_name="truth_proj",
        best_quality_auroc=0.72,
        inside_trigger_budget_id="top_0p4",
        inside_trigger_budget_policy="quality_balanced",
    )

    payload = module.compare_release_candidates(
        readiness_registry_path=registry_path,
        min_best_quality_auroc=0.70,
        max_uncached_forward_seconds=20.0,
        min_selected=4,
        min_decision_accuracy=0.99,
        max_false_supported_rate=0.0,
        min_false_refuted_rate=0.99,
        performance_baseline_key="performance_baseline:qwen-performance:0.6",
    )

    assert payload["decision"]["status"] == "blocked"
    assert payload["release_candidate"] is None
    assert payload["decision"]["blocking_reasons"][0]["gate"] == "performance_baseline"
    assert any(
        "runtime batch_size mismatch" in reason
        for reason in payload["decision"]["blocking_reasons"][0]["reasons"]
    )


def test_compare_release_candidates_accepts_repo_relative_performance_paths(tmp_path):
    module = importlib.import_module("benchmarks.compare_release_candidates")
    from eigentruth.registry import ArtifactRegistry

    registry_path = tmp_path / "registry.json"
    _write_readiness_baseline_manifest(
        tmp_path / "readiness",
        registry_path=registry_path,
        name="qwen-readiness",
        version="0.6",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        layer=-12,
        quality_signals={"truth_proj": 0.72},
        uncached_forward_seconds=18.0,
        cache_only_seconds=0.20,
        inside_trigger_sample_ratio=0.3,
        inside_trigger_generation_ratio=0.35,
    )
    route_manifest = _write_route_baseline_manifest(
        tmp_path,
        name="structured",
        route="structured_state",
        decision_accuracy=1.0,
        false_supported_rate=0.0,
        false_refuted_rate=1.0,
        mean_duration_seconds=0.01,
        p99_duration_seconds=0.02,
    )
    ArtifactRegistry.load_json(registry_path).record_benchmark_manifest(
        name="structured-route",
        path=route_manifest,
        version="0.6",
        metadata={"manifest_metadata": {"runner": "run_adapter_promotion_workflow"}},
    ).save_json()
    _write_performance_baseline_record(
        tmp_path / "performance",
        registry_path=registry_path,
        name="qwen-performance",
        version="0.6",
        layer=-12,
        best_quality_signal_name="truth_proj",
        best_quality_auroc=0.72,
        inside_trigger_budget_id="top_0p4",
        inside_trigger_budget_policy="quality_balanced",
        store_relative_paths=True,
    )

    payload = module.compare_release_candidates(
        readiness_registry_path=registry_path,
        min_best_quality_auroc=0.70,
        max_uncached_forward_seconds=20.0,
        min_selected=4,
        min_decision_accuracy=0.99,
        max_false_supported_rate=0.0,
        min_false_refuted_rate=0.99,
        performance_baseline_key="performance_baseline:qwen-performance:0.6",
    )

    assert payload["decision"]["status"] == "promote"
    assert payload["performance_baseline_gate"]["verification"]["passed"] is True
    assert payload["release_candidate"]["performance_baseline_record"] == (
        "performance_baseline:qwen-performance:0.6"
    )


def test_compare_release_candidates_cli_blocks_when_route_gate_fails(tmp_path):
    module = importlib.import_module("benchmarks.compare_release_candidates")
    from eigentruth.registry import ArtifactRegistry

    registry_path = tmp_path / "registry.json"
    output_path = tmp_path / "release-candidate.json"
    _write_readiness_baseline_manifest(
        tmp_path / "readiness",
        registry_path=registry_path,
        name="qwen-readiness",
        version="0.6",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        layer=-12,
        quality_signals={"truth_proj": 0.72},
        uncached_forward_seconds=18.0,
        cache_only_seconds=0.20,
        inside_sample_ratio=0.4,
        inside_generation_ratio=0.45,
        inside_trigger_sample_ratio=0.3,
        inside_trigger_generation_ratio=0.35,
    )
    route_manifest = _write_route_baseline_manifest(
        tmp_path,
        name="structured",
        route="structured_state",
        decision_accuracy=1.0,
        false_supported_rate=0.0,
        false_refuted_rate=1.0,
        mean_duration_seconds=0.01,
        p99_duration_seconds=0.02,
    )
    ArtifactRegistry.load_json(registry_path).record_benchmark_manifest(
        name="structured-route",
        path=route_manifest,
        version="0.6",
        metadata={"manifest_metadata": {"runner": "run_adapter_promotion_workflow"}},
    ).save_json()

    with pytest.raises(SystemExit) as exc_info:
        module.main([
            "--readiness-registry",
            str(registry_path),
            "--min-best-quality-auroc",
            "0.70",
            "--max-p99-duration-seconds",
            "0.01",
            "--json",
            str(output_path),
            "--fail-on-blocked",
        ])

    assert exc_info.value.code == 1
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["decision"]["status"] == "blocked"
    assert payload["release_candidate"] is None
    assert payload["decision"]["blocking_reasons"][0]["gate"] == "route_baseline"
    assert "p99_duration_seconds above 0.01" in payload["decision"]["blocking_reasons"][0]["reasons"][0]


def test_compare_release_candidates_applies_retrieval_cost_gate(tmp_path):
    module = importlib.import_module("benchmarks.compare_release_candidates")
    from eigentruth.registry import ArtifactRegistry

    registry_path = tmp_path / "registry.json"
    _write_readiness_baseline_manifest(
        tmp_path / "readiness",
        registry_path=registry_path,
        name="qwen-readiness",
        version="0.6",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        layer=-12,
        quality_signals={"truth_proj": 0.72},
        uncached_forward_seconds=18.0,
        cache_only_seconds=0.20,
        inside_sample_ratio=0.4,
        inside_generation_ratio=0.45,
        inside_trigger_sample_ratio=0.3,
        inside_trigger_generation_ratio=0.35,
    )
    route_manifest = _write_route_baseline_manifest(
        tmp_path,
        name="retrieval",
        route="retrieval_groundedness",
        decision_accuracy=1.0,
        false_supported_rate=0.0,
        false_refuted_rate=1.0,
        mean_duration_seconds=0.02,
        p99_duration_seconds=0.03,
        mean_attempted_route_count=2.0,
        retrieval_use_rate=1.0,
    )
    ArtifactRegistry.load_json(registry_path).record_benchmark_manifest(
        name="retrieval-route",
        path=route_manifest,
        version="0.6",
        metadata={"manifest_metadata": {"runner": "run_adapter_promotion_workflow"}},
    ).save_json()

    blocked = module.compare_release_candidates(
        readiness_registry_path=registry_path,
        min_best_quality_auroc=0.70,
        max_uncached_forward_seconds=20.0,
        min_selected=4,
        min_decision_accuracy=0.99,
        max_false_supported_rate=0.0,
        min_false_refuted_rate=0.99,
        max_mean_attempted_route_count=2.1,
        max_retrieval_use_rate=0.0,
    )
    promoted = module.compare_release_candidates(
        readiness_registry_path=registry_path,
        min_best_quality_auroc=0.70,
        max_uncached_forward_seconds=20.0,
        min_selected=4,
        min_decision_accuracy=0.99,
        max_false_supported_rate=0.0,
        min_false_refuted_rate=0.99,
        max_mean_attempted_route_count=2.1,
        max_retrieval_use_rate=1.0,
    )

    assert blocked["decision"]["status"] == "blocked"
    assert blocked["release_candidate"] is None
    assert blocked["decision"]["blocking_reasons"][0]["gate"] == "route_baseline"
    assert "retrieval_use_rate above 0.0" in blocked["decision"]["blocking_reasons"][0]["reasons"][0]
    assert promoted["decision"]["status"] == "promote"
    assert promoted["release_candidate"]["verifier_route"]["route"] == "retrieval_groundedness"
    assert promoted["release_candidate"]["verifier_route"]["mean_attempted_route_count"] == pytest.approx(2.0)
    assert promoted["release_candidate"]["verifier_route"]["retrieval_use_rate"] == pytest.approx(1.0)


def test_compare_release_candidates_applies_inside_sampling_cost_gate(tmp_path):
    module = importlib.import_module("benchmarks.compare_release_candidates")
    from eigentruth.registry import ArtifactRegistry

    registry_path = tmp_path / "registry.json"
    _write_readiness_baseline_manifest(
        tmp_path / "readiness",
        registry_path=registry_path,
        name="qwen-readiness",
        version="0.6",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        layer=-12,
        quality_signals={"truth_proj": 0.72},
        uncached_forward_seconds=18.0,
        cache_only_seconds=0.20,
        inside_sample_ratio=0.9,
        inside_generation_ratio=0.95,
    )
    route_manifest = _write_route_baseline_manifest(
        tmp_path,
        name="structured",
        route="structured_state",
        decision_accuracy=1.0,
        false_supported_rate=0.0,
        false_refuted_rate=1.0,
        mean_duration_seconds=0.01,
        p99_duration_seconds=0.02,
    )
    ArtifactRegistry.load_json(registry_path).record_benchmark_manifest(
        name="structured-route",
        path=route_manifest,
        version="0.6",
        metadata={"manifest_metadata": {"runner": "run_adapter_promotion_workflow"}},
    ).save_json()

    payload = module.compare_release_candidates(
        readiness_registry_path=registry_path,
        min_best_quality_auroc=0.70,
        max_uncached_forward_seconds=20.0,
        max_inside_sample_count_ratio=0.6,
        max_inside_generation_seconds_ratio=0.8,
        min_selected=4,
        min_decision_accuracy=0.99,
        max_false_supported_rate=0.0,
        min_false_refuted_rate=0.99,
    )

    assert payload["decision"]["status"] == "blocked"
    assert payload["release_candidate"] is None
    assert payload["decision"]["blocking_reasons"][0]["gate"] == "readiness_baseline"
    reasons = payload["decision"]["blocking_reasons"][0]["reasons"]
    assert any("INSIDE sampling sample-count ratio above 0.6" in reason for reason in reasons)
    assert any("INSIDE sampling generation-seconds ratio above 0.8" in reason for reason in reasons)


def test_compare_release_candidates_carries_trigger_budget_reference_cost(tmp_path):
    module = importlib.import_module("benchmarks.compare_release_candidates")
    from eigentruth.registry import ArtifactRegistry

    registry_path = tmp_path / "registry.json"
    _write_readiness_baseline_manifest(
        tmp_path / "readiness",
        registry_path=registry_path,
        name="qwen-readiness",
        version="0.6",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        layer=-12,
        quality_signals={"truth_proj": 0.72},
        uncached_forward_seconds=18.0,
        cache_only_seconds=0.20,
        inside_trigger_sample_ratio=0.4,
        inside_trigger_generation_ratio=0.45,
    )
    route_manifest = _write_route_baseline_manifest(
        tmp_path,
        name="structured",
        route="structured_state",
        decision_accuracy=1.0,
        false_supported_rate=0.0,
        false_refuted_rate=1.0,
        mean_duration_seconds=0.01,
        p99_duration_seconds=0.02,
    )
    ArtifactRegistry.load_json(registry_path).record_benchmark_manifest(
        name="structured-route",
        path=route_manifest,
        version="0.6",
        metadata={"manifest_metadata": {"runner": "run_adapter_promotion_workflow"}},
    ).save_json()

    payload = module.compare_release_candidates(
        readiness_registry_path=registry_path,
        min_best_quality_auroc=0.70,
        max_uncached_forward_seconds=20.0,
        max_inside_sample_count_ratio=0.6,
        max_inside_generation_seconds_ratio=0.8,
        min_selected=4,
        min_decision_accuracy=0.99,
        max_false_supported_rate=0.0,
        min_false_refuted_rate=0.99,
    )

    assert payload["decision"]["status"] == "promote"
    candidate = payload["release_candidate"]
    assert candidate["runtime"]["inside_trigger_budget_sweep"]["recommended_budget_id"] == "top_0p4"
    assert candidate["runtime"]["inside_trigger_budget_policy"] == "quality_balanced"
    assert candidate["runtime_cost"]["inside_sampling_sample_count_ratio_to_baseline"] is None
    assert candidate["runtime_cost"]["inside_sampling_sample_count_ratio_to_reference"] == pytest.approx(0.4)
    assert candidate["runtime_cost"]["inside_sampling_sample_count_ratio_for_gate"] == pytest.approx(0.4)
    assert candidate["runtime_cost"]["inside_sampling_sample_count_ratio_source"] == (
        "sample_count_ratio_to_reference"
    )
    assert candidate["runtime_cost"]["inside_generation_seconds_ratio_to_reference"] == pytest.approx(0.45)
    assert candidate["runtime_cost"]["inside_generation_seconds_ratio_for_gate"] == pytest.approx(0.45)
    assert candidate["runtime_cost"]["inside_generation_seconds_ratio_source"] == (
        "inside_generation_seconds_ratio_to_reference"
    )
    assert candidate["runtime_cost"]["inside_trigger_budget_id"] == "top_0p4"
    assert candidate["runtime_cost"]["inside_trigger_budget_policy"] == "quality_balanced"
    assert candidate["runtime_cost"]["inside_trigger_budget_derive_from_max_budget"] is True


def test_compare_release_candidates_can_override_trigger_budget_policy(tmp_path):
    module = importlib.import_module("benchmarks.compare_release_candidates")
    from eigentruth.registry import ArtifactRegistry

    registry_path = tmp_path / "registry.json"
    _write_readiness_baseline_manifest(
        tmp_path / "readiness",
        registry_path=registry_path,
        name="qwen-readiness",
        version="0.6",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        layer=-12,
        quality_signals={"truth_proj": 0.72},
        uncached_forward_seconds=18.0,
        cache_only_seconds=0.20,
        inside_trigger_sample_ratio=0.4,
        inside_trigger_generation_ratio=0.45,
        inside_trigger_total_generated_samples=12,
        inside_trigger_cost_first_sample_ratio=0.1,
        inside_trigger_cost_first_generation_ratio=0.12,
        inside_trigger_cost_first_total_generated_samples=3,
    )
    route_manifest = _write_route_baseline_manifest(
        tmp_path,
        name="structured",
        route="structured_state",
        decision_accuracy=1.0,
        false_supported_rate=0.0,
        false_refuted_rate=1.0,
        mean_duration_seconds=0.01,
        p99_duration_seconds=0.02,
    )
    ArtifactRegistry.load_json(registry_path).record_benchmark_manifest(
        name="structured-route",
        path=route_manifest,
        version="0.6",
        metadata={"manifest_metadata": {"runner": "run_adapter_promotion_workflow"}},
    ).save_json()

    default_payload = module.compare_release_candidates(
        readiness_registry_path=registry_path,
        min_best_quality_auroc=0.70,
        max_uncached_forward_seconds=20.0,
        max_inside_sample_count_ratio=0.6,
        max_inside_generation_seconds_ratio=0.8,
        min_selected=4,
        min_decision_accuracy=0.99,
        max_false_supported_rate=0.0,
        min_false_refuted_rate=0.99,
    )
    cost_payload = module.compare_release_candidates(
        readiness_registry_path=registry_path,
        inside_trigger_budget_policy="cost-first",
        min_best_quality_auroc=0.70,
        max_uncached_forward_seconds=20.0,
        max_inside_sample_count_ratio=0.6,
        max_inside_generation_seconds_ratio=0.8,
        min_selected=4,
        min_decision_accuracy=0.99,
        max_false_supported_rate=0.0,
        min_false_refuted_rate=0.99,
    )

    assert default_payload["decision"]["status"] == "promote"
    assert default_payload["config"]["inside_trigger_budget_policy"] is None
    assert default_payload["release_candidate"]["runtime_cost"]["inside_trigger_budget_id"] == "top_0p4"
    assert default_payload["release_candidate"]["runtime_cost"]["inside_sampling_sample_count_ratio_for_gate"] == (
        pytest.approx(0.4)
    )
    assert cost_payload["decision"]["status"] == "promote"
    assert cost_payload["config"]["inside_trigger_budget_policy"] == "cost_first"
    assert cost_payload["readiness_baseline_comparison"]["config"]["inside_trigger_budget_policy"] == "cost_first"
    candidate = cost_payload["release_candidate"]
    assert candidate["runtime"]["inside_trigger_budget_policy"] == "cost_first"
    assert candidate["runtime"]["inside_sampling"]["inside_trigger_top_fraction"] == pytest.approx(0.1)
    assert candidate["runtime_cost"]["inside_trigger_budget_id"] == "top_0p1"
    assert candidate["runtime_cost"]["inside_trigger_budget_policy"] == "cost_first"
    assert candidate["runtime_cost"]["inside_sampling_total_generated_samples"] == 3
    assert candidate["runtime_cost"]["inside_sampling_sample_count_ratio_for_gate"] == pytest.approx(0.1)
    assert candidate["runtime_cost"]["inside_generation_seconds_ratio_for_gate"] == pytest.approx(0.12)


def test_compare_release_candidates_runtime_profile_fills_unset_cost_gates(tmp_path):
    module = importlib.import_module("benchmarks.compare_release_candidates")
    from eigentruth.registry import ArtifactRegistry

    registry_path = tmp_path / "registry.json"
    _write_readiness_baseline_manifest(
        tmp_path / "readiness",
        registry_path=registry_path,
        name="qwen-readiness",
        version="0.6",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        layer=-12,
        quality_signals={"truth_proj": 0.72},
        uncached_forward_seconds=18.0,
        cache_only_seconds=0.20,
        inside_trigger_sample_ratio=0.4,
        inside_trigger_generation_ratio=0.45,
        inside_trigger_total_generated_samples=12,
        inside_trigger_cost_first_sample_ratio=0.1,
        inside_trigger_cost_first_generation_ratio=0.12,
        inside_trigger_cost_first_total_generated_samples=3,
    )
    route_manifest = _write_route_baseline_manifest(
        tmp_path,
        name="structured",
        route="structured_state",
        decision_accuracy=1.0,
        false_supported_rate=0.0,
        false_refuted_rate=1.0,
        mean_duration_seconds=0.01,
        p99_duration_seconds=0.02,
    )
    ArtifactRegistry.load_json(registry_path).record_benchmark_manifest(
        name="structured-route",
        path=route_manifest,
        version="0.6",
        metadata={"manifest_metadata": {"runner": "run_adapter_promotion_workflow"}},
    ).save_json()

    latency_payload = module.compare_release_candidates(
        readiness_registry_path=registry_path,
        runtime_profile="latency",
        min_best_quality_auroc=0.70,
        max_uncached_forward_seconds=20.0,
        min_selected=4,
        min_decision_accuracy=0.99,
        max_false_supported_rate=0.0,
        min_false_refuted_rate=0.99,
    )
    override_payload = module.compare_release_candidates(
        readiness_registry_path=registry_path,
        runtime_profile="latency",
        inside_trigger_budget_policy="quality_balanced",
        max_inside_sample_count_ratio=0.6,
        max_inside_generation_seconds_ratio=0.8,
        min_best_quality_auroc=0.70,
        max_uncached_forward_seconds=20.0,
        min_selected=4,
        min_decision_accuracy=0.99,
        max_false_supported_rate=0.0,
        min_false_refuted_rate=0.99,
    )

    assert latency_payload["decision"]["status"] == "promote"
    assert latency_payload["config"]["runtime_profile"] == "latency"
    assert latency_payload["config"]["inside_trigger_budget_policy"] == "cost_first"
    assert latency_payload["config"]["max_inside_sample_count_ratio"] == pytest.approx(0.25)
    assert latency_payload["config"]["max_inside_generation_seconds_ratio"] == pytest.approx(0.35)
    assert latency_payload["config"]["max_mean_attempted_route_count"] == pytest.approx(1.1)
    assert latency_payload["config"]["max_retrieval_use_rate"] == pytest.approx(0.0)
    assert latency_payload["config"]["runtime_profile_applied_defaults"] == {
        "inside_trigger_budget_policy": "cost_first",
        "max_inside_sample_count_ratio": 0.25,
        "max_inside_generation_seconds_ratio": 0.35,
        "max_mean_attempted_route_count": 1.1,
        "max_retrieval_use_rate": 0.0,
    }
    latency_candidate = latency_payload["release_candidate"]
    assert latency_candidate["runtime_cost"]["inside_trigger_budget_id"] == "top_0p1"
    assert latency_candidate["runtime_cost"]["inside_sampling_sample_count_ratio_for_gate"] == pytest.approx(0.1)
    assert latency_candidate["runtime_cost"]["inside_generation_seconds_ratio_for_gate"] == pytest.approx(0.12)

    assert override_payload["decision"]["status"] == "promote"
    assert override_payload["config"]["runtime_profile"] == "latency"
    assert override_payload["config"]["inside_trigger_budget_policy"] == "quality_balanced"
    assert override_payload["config"]["max_inside_sample_count_ratio"] == pytest.approx(0.6)
    assert override_payload["config"]["max_inside_generation_seconds_ratio"] == pytest.approx(0.8)
    assert override_payload["config"]["runtime_profile_applied_defaults"] == {
        "max_mean_attempted_route_count": 1.1,
        "max_retrieval_use_rate": 0.0,
    }
    override_candidate = override_payload["release_candidate"]
    assert override_candidate["runtime_cost"]["inside_trigger_budget_id"] == "top_0p4"
    assert override_candidate["runtime_cost"]["inside_sampling_sample_count_ratio_for_gate"] == pytest.approx(0.4)
    assert override_candidate["runtime_cost"]["inside_generation_seconds_ratio_for_gate"] == pytest.approx(0.45)


def test_run_release_candidate_registry_workflow_registers_promoted_candidate(tmp_path):
    module = importlib.import_module("benchmarks.run_release_candidate_registry_workflow")
    from eigentruth.registry import ArtifactRegistry

    baseline_registry_path = tmp_path / "baseline-registry.json"
    release_registry_path = tmp_path / "release-registry.json"
    _write_readiness_baseline_manifest(
        tmp_path / "readiness",
        registry_path=baseline_registry_path,
        name="qwen-readiness",
        version="0.6",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        layer=-12,
        quality_signals={"truth_proj": 0.72},
        uncached_forward_seconds=18.0,
        cache_only_seconds=0.20,
        inside_sample_ratio=0.4,
        inside_generation_ratio=0.45,
        inside_trigger_sample_ratio=0.3,
        inside_trigger_generation_ratio=0.35,
    )
    route_manifest = _write_route_baseline_manifest(
        tmp_path,
        name="structured",
        route="structured_state",
        decision_accuracy=1.0,
        false_supported_rate=0.0,
        false_refuted_rate=1.0,
        mean_duration_seconds=0.01,
        p99_duration_seconds=0.02,
    )
    retrieval_manifest = _write_route_baseline_manifest(
        tmp_path,
        name="retrieval",
        route="retrieval_groundedness",
        decision_accuracy=0.96,
        false_supported_rate=0.03,
        false_refuted_rate=0.60,
        mean_duration_seconds=0.04,
        p99_duration_seconds=0.08,
        mean_attempted_route_count=2.0,
        retrieval_use_rate=1.0,
        runtime_total_seconds=2.0,
        runtime_n_retrieval_hits=24,
    )
    ArtifactRegistry.load_json(baseline_registry_path).record_benchmark_manifest(
        name="structured-route",
        path=route_manifest,
        version="0.6",
        metadata={"manifest_metadata": {"runner": "run_adapter_promotion_workflow"}},
    ).record_benchmark_manifest(
        name="retrieval-route",
        path=retrieval_manifest,
        version="0.7",
        metadata={"manifest_metadata": {"runner": "run_local_retrieval_route_workflow"}},
    ).save_json()
    _write_performance_baseline_record(
        tmp_path / "performance",
        registry_path=baseline_registry_path,
        name="qwen-performance",
        version="0.6",
        layer=-12,
        best_quality_signal_name="truth_proj",
        best_quality_auroc=0.72,
        inside_trigger_budget_id="top_0p4",
        inside_trigger_budget_policy="cost_first",
    )
    adapter_family_matrix_path = _write_adapter_family_matrix(tmp_path / "adapter-family-matrix.json")

    payload = module.run_release_candidate_registry_workflow(
        module.ReleaseCandidateRegistryWorkflowConfig(
            readiness_registry_path=baseline_registry_path,
            release_registry_path=release_registry_path,
            name="qwen-release-candidate",
            version="0.7",
            workflow_report_path=tmp_path / "workflow.json",
            release_report_path=tmp_path / "release-candidate.json",
            artifact_manifest_path=tmp_path / "release-manifest.json",
            verification_report_path=tmp_path / "release-verification.json",
            performance_baseline_key="performance_baseline:qwen-performance:0.6",
            route_baseline_keys=("benchmark_manifest:structured-route:0.6",),
            required_route_baseline_keys=("benchmark_manifest:retrieval-route:0.7",),
            adapter_family_matrix_path=adapter_family_matrix_path,
            required_adapter_routes=("structured_state", "state_transition"),
            runtime_profile="balanced",
            inside_trigger_budget_policy="cost_first",
            min_best_quality_auroc=0.70,
            max_uncached_forward_seconds=20.0,
            max_inside_sample_count_ratio=0.6,
            max_inside_generation_seconds_ratio=0.8,
            min_selected=4,
            min_decision_accuracy=0.99,
            max_p99_duration_seconds=0.03,
            required_route_max_runtime_total_seconds=3.0,
            required_route_max_retrieval_hit_count=30,
            required_route_max_retrieval_use_rate=1.0,
            promotion_metadata={"scope": "unit"},
        )
    )

    assert payload["decision"]["status"] == "promote"
    assert payload["decision"]["manifest_promoted"] is True
    assert payload["decision"]["manifest_verified"] is True
    assert payload["decision"]["registry_record"] == "benchmark_manifest:qwen-release-candidate:0.7"
    manifest = json.loads((tmp_path / "release-manifest.json").read_text(encoding="utf-8"))
    assert sorted(manifest["artifacts"]) == [
        "adapter_family_matrix_report",
        "performance_manifest",
        "readiness_manifest",
        "release_candidate_report",
        "required_route_manifest_1",
        "route_manifest",
    ]
    assert manifest["metadata"]["runner"] == "run_release_candidate_registry_workflow"
    assert manifest["metadata"]["release_candidate_status"] == "promote"
    assert manifest["metadata"]["release_runtime_profile"] == "balanced"
    assert manifest["metadata"]["release_performance_status"] == "promote"
    assert manifest["metadata"]["release_adapter_family_status"] == "promote"
    assert manifest["metadata"]["release_required_route_baseline_status"] == "promote"
    assert manifest["metadata"]["release_runtime_profile_applied_defaults"] == {
        "max_mean_attempted_route_count": 1.5,
        "max_retrieval_use_rate": 0.5,
    }
    assert manifest["metadata"]["recommended_readiness_record"] == "benchmark_manifest:qwen-readiness:0.6"
    assert manifest["metadata"]["recommended_route_record"] == "benchmark_manifest:structured-route:0.6"
    assert manifest["metadata"]["recommended_performance_baseline_record"] == (
        "performance_baseline:qwen-performance:0.6"
    )
    assert manifest["metadata"]["required_adapter_routes"] == ["structured_state", "state_transition"]
    assert manifest["metadata"]["required_route_baseline_records"] == [
        "benchmark_manifest:retrieval-route:0.7"
    ]
    assert manifest["metadata"]["recommended_route_verified_false_alarm"] == pytest.approx(0.0)
    assert manifest["metadata"]["recommended_route_verified_detection"] == pytest.approx(1.0)
    assert manifest["metadata"]["recommended_route_mean_duration_seconds"] == pytest.approx(0.01)
    assert manifest["metadata"]["recommended_route_max_duration_seconds"] == pytest.approx(0.02)
    assert manifest["metadata"]["recommended_route_mean_attempted_route_count"] == pytest.approx(1.0)
    assert manifest["metadata"]["recommended_route_retrieval_use_rate"] == pytest.approx(0.0)
    assert manifest["metadata"]["recommended_inside_sampling_run"] == "adaptive_selfcheck"
    assert manifest["metadata"]["recommended_inside_sampling_sample_count_ratio_to_baseline"] == pytest.approx(0.4)
    assert manifest["metadata"]["recommended_inside_generation_seconds_ratio_to_baseline"] == pytest.approx(0.45)
    assert manifest["metadata"]["recommended_inside_sampling_sample_count_ratio_to_reference"] == pytest.approx(0.3)
    assert manifest["metadata"]["recommended_inside_generation_seconds_ratio_to_reference"] == pytest.approx(0.35)
    assert manifest["metadata"]["recommended_inside_trigger_budget_id"] == "top_0p4"
    assert manifest["metadata"]["recommended_inside_trigger_budget_policy"] == "cost_first"
    assert manifest["metadata"]["recommended_inside_trigger_budget_derive_from_max_budget"] is True
    assert manifest["metadata"]["performance_manifest"].endswith("performance/artifact-manifest.json")
    assert manifest["metadata"]["adapter_family_matrix_report"] == str(adapter_family_matrix_path)
    assert manifest["metadata"]["adapter_family_required_routes"] == ["structured_state", "state_transition"]
    assert manifest["metadata"]["required_route_baseline_routes"] == ["retrieval_groundedness"]
    assert manifest["metadata"]["required_route_baseline_manifests"] == [str(retrieval_manifest)]
    assert manifest["metadata"]["required_route_budget_policy"]["required_route_max_runtime_total_seconds"] == (
        pytest.approx(3.0)
    )
    assert payload["config"]["runtime_profile"] == "balanced"
    assert payload["config"]["route_baseline_keys"] == ("benchmark_manifest:structured-route:0.6",)
    assert payload["config"]["required_route_baseline_keys"] == ("benchmark_manifest:retrieval-route:0.7",)
    assert payload["config"]["required_route_max_runtime_total_seconds"] == pytest.approx(3.0)
    assert payload["config"]["performance_baseline_key"] == "performance_baseline:qwen-performance:0.6"
    assert payload["config"]["adapter_family_matrix"] == str(adapter_family_matrix_path)
    assert payload["config"]["required_adapter_routes"] == ("structured_state", "state_transition")
    assert payload["release_candidate_comparison"]["config"]["runtime_profile"] == "balanced"
    assert payload["release_candidate_comparison"]["config"]["required_route_baseline_keys"] == [
        "benchmark_manifest:retrieval-route:0.7"
    ]
    assert payload["release_candidate_comparison"]["config"]["required_route_max_runtime_total_seconds"] == (
        pytest.approx(3.0)
    )
    assert payload["release_candidate_comparison"]["config"]["inside_trigger_budget_policy"] == "cost_first"
    assert payload["release_candidate_comparison"]["config"]["max_inside_sample_count_ratio"] == pytest.approx(0.6)
    assert payload["release_candidate_comparison"]["config"]["max_inside_generation_seconds_ratio"] == pytest.approx(
        0.8
    )
    registry = ArtifactRegistry.load_json(release_registry_path)
    record = registry.get("benchmark_manifest:qwen-release-candidate:0.7")
    assert record.metadata["workflow"] == "run_release_candidate_registry_workflow"
    assert record.metadata["release_candidate_status"] == "promote"
    assert record.metadata["release_runtime_profile"] == "balanced"
    assert record.metadata["recommended_performance_baseline_record"] == (
        "performance_baseline:qwen-performance:0.6"
    )
    assert record.metadata["release_adapter_family_status"] == "promote"
    assert record.metadata["release_required_route_baseline_status"] == "promote"
    assert record.metadata["adapter_family_matrix_report"] == str(adapter_family_matrix_path)
    assert record.metadata["required_route_baseline_records"] == [
        "benchmark_manifest:retrieval-route:0.7"
    ]
    assert record.metadata["required_route_budget_policy"]["required_route_max_retrieval_hit_count"] == (
        pytest.approx(30.0)
    )
    assert record.metadata["adapter_family_required_routes"] == ["structured_state", "state_transition"]
    assert record.metadata["recommended_model"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert record.metadata["recommended_route"] == "structured_state"
    assert record.metadata["recommended_route_retrieval_use_rate"] == pytest.approx(0.0)
    assert record.metadata["recommended_inside_sampling_sample_count_ratio_to_baseline"] == pytest.approx(0.4)
    assert record.metadata["recommended_inside_generation_seconds_ratio_to_baseline"] == pytest.approx(0.45)
    assert record.metadata["recommended_inside_sampling_sample_count_ratio_to_reference"] == pytest.approx(0.3)
    assert record.metadata["recommended_inside_generation_seconds_ratio_to_reference"] == pytest.approx(0.35)
    assert record.metadata["recommended_inside_trigger_budget_id"] == "top_0p4"
    assert record.metadata["recommended_inside_trigger_budget_policy"] == "cost_first"
    assert record.metadata["scope"] == "unit"


def test_run_release_candidate_registry_workflow_cli_blocks_without_registration(tmp_path):
    module = importlib.import_module("benchmarks.run_release_candidate_registry_workflow")
    from eigentruth.registry import ArtifactRegistry

    baseline_registry_path = tmp_path / "baseline-registry.json"
    release_registry_path = tmp_path / "release-registry.json"
    workflow_path = tmp_path / "workflow.json"
    _write_readiness_baseline_manifest(
        tmp_path / "readiness",
        registry_path=baseline_registry_path,
        name="qwen-readiness",
        version="0.6",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        layer=-12,
        quality_signals={"truth_proj": 0.72},
        uncached_forward_seconds=18.0,
        cache_only_seconds=0.20,
    )
    route_manifest = _write_route_baseline_manifest(
        tmp_path,
        name="structured",
        route="structured_state",
        decision_accuracy=1.0,
        false_supported_rate=0.0,
        false_refuted_rate=1.0,
        mean_duration_seconds=0.01,
        p99_duration_seconds=0.02,
    )
    ArtifactRegistry.load_json(baseline_registry_path).record_benchmark_manifest(
        name="structured-route",
        path=route_manifest,
        version="0.6",
        metadata={"manifest_metadata": {"runner": "run_adapter_promotion_workflow"}},
    ).save_json()

    with pytest.raises(SystemExit) as exc_info:
        module.main([
            "--readiness-registry",
            str(baseline_registry_path),
            "--release-registry",
            str(release_registry_path),
            "--name",
            "qwen-release-candidate",
            "--version",
            "0.7",
            "--max-p99-duration-seconds",
            "0.01",
            "--json",
            str(workflow_path),
            "--fail-on-blocked",
        ])

    assert exc_info.value.code == 1
    payload = json.loads(workflow_path.read_text(encoding="utf-8"))
    assert payload["decision"]["status"] == "blocked"
    assert payload["decision"]["manifest_promoted"] is False
    assert payload["decision"]["manifest_verified"] is False
    assert payload["promotion"] is None
    assert payload["decision"]["blocking_reasons"] == [
        "release candidate comparison did not promote",
    ]
    assert ArtifactRegistry.load_json(release_registry_path).records == ()


def _write_inside_sampling_profile(
    output_dir,
    *,
    sample_count_ratio,
    generation_seconds_ratio,
    total_generated_samples=8,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "inside-result-adaptive_selfcheck.json"
    report_path = output_dir / "inside-sampling-profile-comparison.json"
    stop_reason_counts = {"selfcheck_refute_threshold_guaranteed": 4}
    result_path.write_text(
        json.dumps({
            "config": {
                "inside_samples": 5,
                "inside_batch_size": 1,
                "inside_max_new_tokens": 12,
                "inside_temperature": 0.7,
                "inside_top_p": 0.9,
                "inside_pooling": "last",
                "inside_embedding_threshold": 0.9,
                "inside_adaptive_sampling": True,
                "inside_min_samples": 2,
                "inside_sample_step": 1,
                "inside_stability_delta": 0.05,
                "inside_selfcheck_early_stop": True,
                "inside_selfcheck_min_overlap": 0.65,
                "inside_selfcheck_support_threshold": 0.6,
                "inside_selfcheck_refute_threshold": 0.5,
                "inside_trigger_signal": "truth_proj",
                "inside_trigger_threshold": None,
                "inside_trigger_top_fraction": 0.25,
            },
            "inside_sampling": {
                "mode": "triggered",
                "adaptive": True,
                "selfcheck_early_stop": True,
                "signal": "truth_proj",
                "top_fraction": 0.25,
                "max_samples": 5,
                "min_samples": 2,
                "sample_step": 1,
                "stability_delta": 0.05,
                "embedding_similarity_threshold": 0.9,
                "selfcheck_min_overlap": 0.65,
                "selfcheck_support_threshold": 0.6,
                "selfcheck_refute_threshold": 0.5,
                "total_generated_samples": total_generated_samples,
                "stop_reason_counts": stop_reason_counts,
            },
        }),
        encoding="utf-8",
    )
    report_path.write_text(
        json.dumps({
            "baseline": "fixed",
            "runs": {
                "adaptive_selfcheck": {
                    "name": "adaptive_selfcheck",
                    "result_path": str(result_path),
                    "total_generated_samples": total_generated_samples,
                    "sample_count_ratio_to_baseline": sample_count_ratio,
                    "inside_generation_seconds": 4.5,
                    "inside_generation_seconds_ratio_to_baseline": generation_seconds_ratio,
                    "stop_reason_counts": stop_reason_counts,
                }
            },
            "sample_efficiency_gate": {"passed": True, "failures": []},
            "recommendation": {"recommended_run": "adaptive_selfcheck"},
        }),
        encoding="utf-8",
    )
    return report_path


def _write_inside_trigger_budget_sweep(
    output_dir,
    *,
    sample_count_ratio,
    generation_seconds_ratio,
    total_generated_samples=8,
    budget_id="top_0p4",
    top_fraction=0.4,
    quality_value=0.57,
    derive_from_max_budget=True,
    cost_first_sample_count_ratio=None,
    cost_first_generation_seconds_ratio=None,
    cost_first_total_generated_samples=4,
    cost_first_budget_id="top_0p1",
    cost_first_top_fraction=0.1,
    cost_first_quality_value=0.50,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "inside-trigger-budget-sweep.json"
    cost_first_enabled = cost_first_sample_count_ratio is not None
    budget_specs = []
    leaderboard = []
    budgets_payload = {}
    if cost_first_enabled:
        budget_specs.append({
            "kind": "top_fraction",
            "value": cost_first_top_fraction,
            "id": cost_first_budget_id,
        })
        budgets_payload[cost_first_budget_id] = {
            "sample_efficiency_gate": {"passed": True},
            "recommendation": {"recommended_run": "adaptive_selfcheck"},
        }
        leaderboard.append({
            "budget_id": cost_first_budget_id,
            "budget_kind": "top_fraction",
            "budget_value": cost_first_top_fraction,
            "recommended_run": "adaptive_selfcheck",
            "derived": derive_from_max_budget,
            "derived_from_budget_id": budget_id if derive_from_max_budget else None,
            "inside_generation_seconds_source": (
                "sample_count_ratio_estimate" if derive_from_max_budget else "measured"
            ),
            "sampled": cost_first_total_generated_samples,
            "skipped_by_trigger": 100 - cost_first_total_generated_samples,
            "total_generated_samples": cost_first_total_generated_samples,
            "mean_samples_per_record": cost_first_total_generated_samples / 100,
            "inside_generation_seconds": 2.0,
            "sample_count_ratio_to_reference": cost_first_sample_count_ratio,
            "inside_generation_seconds_ratio_to_reference": (
                cost_first_sample_count_ratio
                if cost_first_generation_seconds_ratio is None
                else cost_first_generation_seconds_ratio
            ),
            "inside_auroc": {"inside_semantic_entropy": cost_first_quality_value},
            "stop_reason_counts": {"selfcheck_supported": 2},
        })
    budget_specs.append({"kind": "top_fraction", "value": top_fraction, "id": budget_id})
    budgets_payload[budget_id] = {
        "sample_efficiency_gate": {"passed": True},
        "recommendation": {"recommended_run": "adaptive_selfcheck"},
    }
    leaderboard.append({
        "budget_id": budget_id,
        "budget_kind": "top_fraction",
        "budget_value": top_fraction,
        "recommended_run": "adaptive_selfcheck",
        "derived": derive_from_max_budget,
        "derived_from_budget_id": budget_id if derive_from_max_budget else None,
        "inside_generation_seconds_source": (
            "measured_source_run" if derive_from_max_budget else "measured"
        ),
        "sampled": total_generated_samples,
        "skipped_by_trigger": 100 - total_generated_samples,
        "total_generated_samples": total_generated_samples,
        "mean_samples_per_record": total_generated_samples / 100,
        "inside_generation_seconds": 4.5,
        "sample_count_ratio_to_reference": sample_count_ratio,
        "inside_generation_seconds_ratio_to_reference": generation_seconds_ratio,
        "inside_auroc": {"inside_semantic_entropy": quality_value},
        "stop_reason_counts": {"selfcheck_supported": 4},
    })
    report_path.write_text(
        json.dumps({
            "schema_version": 1,
            "workflow": "inside_trigger_budget_sweep",
            "dry_run": False,
            "derived_from_max_budget": derive_from_max_budget,
            "derived_source_budget_id": budget_id if derive_from_max_budget else None,
            "config": {
                "trigger_signal": "truth_proj",
                "budgets": budget_specs,
                "inside_samples": 5,
                "inside_batch_size": 1,
                "inside_max_new_tokens": 12,
                "inside_temperature": 0.7,
                "inside_top_p": 0.9,
                "inside_pooling": "last",
                "inside_embedding_threshold": 0.9,
                "inside_min_samples": 2,
                "inside_sample_step": 1,
                "inside_stability_delta": 0.05,
                "inside_selfcheck_min_overlap": 0.65,
                "inside_selfcheck_support_threshold": 0.6,
                "inside_selfcheck_refute_threshold": 0.5,
                "run_names": ["adaptive_selfcheck"],
                "derive_from_max_budget": derive_from_max_budget,
            },
            "budgets": budgets_payload,
            "leaderboard": leaderboard,
            "recommendation": {
                "budget_id": cost_first_budget_id if cost_first_enabled else budget_id,
                "recommended_run": "adaptive_selfcheck",
                "reason": "lowest_total_generated_samples_then_inside_generation_seconds",
            },
            "quality_balanced_recommendation": {
                "budget_id": budget_id,
                "recommended_run": "adaptive_selfcheck",
                "reason": "lowest_cost_within_inside_quality_tolerance",
                "quality_metric": "inside_semantic_entropy",
                "quality_value": quality_value,
                "best_quality_value": quality_value,
                "quality_tolerance": 0.02,
                "cost_metric": "inside_generation_seconds_ratio_to_reference",
                "cost_value": generation_seconds_ratio,
            },
        }),
        encoding="utf-8",
    )
    return report_path


def _write_readiness_baseline_manifest(
    output_dir,
    *,
    registry_path,
    name,
    version,
    model,
    layer,
    quality_signals,
    uncached_forward_seconds,
    cache_only_seconds,
    include_forced_forward=True,
    inside_sample_ratio=None,
    inside_generation_ratio=None,
    inside_total_generated_samples=8,
    inside_trigger_sample_ratio=None,
    inside_trigger_generation_ratio=None,
    inside_trigger_total_generated_samples=8,
    inside_trigger_cost_first_sample_ratio=None,
    inside_trigger_cost_first_generation_ratio=None,
    inside_trigger_cost_first_total_generated_samples=4,
):
    from eigentruth.registry import ArtifactRegistry, build_artifact_manifest

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "cache-only-result.json"
    matrix_path = output_dir / "performance-matrix.json"
    manifest_path = output_dir / "artifact-manifest.json"
    inside_sampling_path = None
    inside_trigger_budget_sweep_path = None
    if inside_sample_ratio is not None:
        inside_sampling_path = _write_inside_sampling_profile(
            output_dir,
            sample_count_ratio=inside_sample_ratio,
            generation_seconds_ratio=(
                inside_sample_ratio if inside_generation_ratio is None else inside_generation_ratio
            ),
            total_generated_samples=inside_total_generated_samples,
        )
    if inside_trigger_sample_ratio is not None:
        inside_trigger_budget_sweep_path = _write_inside_trigger_budget_sweep(
            output_dir,
            sample_count_ratio=inside_trigger_sample_ratio,
            generation_seconds_ratio=(
                inside_trigger_sample_ratio
                if inside_trigger_generation_ratio is None
                else inside_trigger_generation_ratio
            ),
            total_generated_samples=inside_trigger_total_generated_samples,
            cost_first_sample_count_ratio=inside_trigger_cost_first_sample_ratio,
            cost_first_generation_seconds_ratio=inside_trigger_cost_first_generation_ratio,
            cost_first_total_generated_samples=inside_trigger_cost_first_total_generated_samples,
        )
    cell_id = f"layer_m{abs(layer)}_batch_1_capture_outputs"
    result_path.write_text(
        json.dumps({"auroc": dict(quality_signals)}),
        encoding="utf-8",
    )
    matrix_path.write_text(
        json.dumps({
            "config": {
                "max_workers": 1,
                "length_bucketed_batches": True,
            },
            "matrix_decision": {
                "status": "promote",
                "recommended_cell": cell_id,
                "recommendation_metric": "uncached_forced_answer_forward_seconds",
                "blocking_reasons": (),
                "recommended": {
                    "id": cell_id,
                    "layer": layer,
                    "batch_size": 1,
                    "hidden_state_capture": "outputs",
                    "max_batch_tokens": 0,
                    "prefix_kv_cache": False,
                    "uncached_total_seconds": uncached_forward_seconds,
                    "cache_only_total_seconds": cache_only_seconds,
                    **(
                        {"uncached_forced_answer_forward_seconds": uncached_forward_seconds}
                        if include_forced_forward
                        else {}
                    ),
                    "truth_proj_auroc": quality_signals.get("truth_proj"),
                },
            },
            "cells": [
                {
                    "id": cell_id,
                    "layer": layer,
                    "batch_size": 1,
                    "hidden_state_capture": "outputs",
                    "summary": {
                        "quality_signals": dict(quality_signals),
                        "truth_proj_auroc": quality_signals.get("truth_proj"),
                        "totals": {
                            "uncached": {"total_seconds": uncached_forward_seconds},
                            "cache_only": {"total_seconds": cache_only_seconds},
                        },
                    },
                    "triplet": {"results": {"cache_only": str(result_path)}},
                }
            ],
        }),
        encoding="utf-8",
    )
    metadata = {
        "runner": "run_adapter_readiness_workflow",
        "model": model,
        "dtype": "auto",
        "readiness_status": "promote",
        "adapter_family_status": "promote",
        "performance_status": "promote",
        "runtime_recommendation_status": "promote",
        "recommended_route": "structured_qa",
        "recommended_performance_cell": cell_id,
        "inside_sampling_report": None if inside_sampling_path is None else str(inside_sampling_path),
        "inside_trigger_budget_sweep_report": None
        if inside_trigger_budget_sweep_path is None
        else str(inside_trigger_budget_sweep_path),
    }
    manifest_path.write_text(
        json.dumps(
            build_artifact_manifest(
                {
                    "performance_matrix_report": matrix_path,
                    "inside_sampling_profile_report": inside_sampling_path,
                    "inside_trigger_budget_sweep_report": inside_trigger_budget_sweep_path,
                },
                root=output_dir,
                metadata=metadata,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    ArtifactRegistry.load_json(registry_path).record_benchmark_manifest(
        name=name,
        version=version,
        path=manifest_path,
        metadata={
            "workflow": "run_adapter_readiness_registry_workflow",
            "readiness_status": "promote",
            "runtime_recommendation_status": "promote",
            "manifest_metadata": metadata,
        },
    ).save_json()
    return manifest_path


def _write_performance_baseline_record(
    output_dir,
    *,
    registry_path,
    name,
    version,
    layer,
    batch_size=1,
    hidden_state_capture="outputs",
    max_batch_tokens=0,
    prefix_kv_cache=False,
    max_workers=1,
    best_quality_signal_name="truth_proj",
    best_quality_auroc=0.72,
    inside_trigger_budget_id="top_0p4",
    inside_trigger_budget_policy="quality_balanced",
    store_relative_paths=False,
):
    from eigentruth.registry import ArtifactRegistry, build_artifact_manifest

    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_path = output_dir / "runtime-recommendation.json"
    report_path = output_dir / "performance-baseline-workflow.json"
    manifest_path = output_dir / "artifact-manifest.json"
    def stored(path):
        return os.path.relpath(path, start=Path.cwd()) if store_relative_paths else str(path)

    cell_id = f"layer_m{abs(layer)}_batch_{batch_size}_capture_{hidden_state_capture}"
    recommendation = {
        "cell_id": cell_id,
        "layer": layer,
        "batch_size": batch_size,
        "hidden_state_capture": hidden_state_capture,
        "max_batch_tokens": max_batch_tokens,
        "prefix_kv_cache": prefix_kv_cache,
        "max_workers": max_workers,
        "quality_signals": {best_quality_signal_name: best_quality_auroc},
        "best_quality_signal": {
            "name": best_quality_signal_name,
            "auroc": best_quality_auroc,
        },
        "inside_sampling": {
            "recommended_run": "adaptive_selfcheck",
            "inside_trigger_budget_id": inside_trigger_budget_id,
            "inside_trigger_budget_policy": inside_trigger_budget_policy,
        },
        "inside_trigger_budget_sweep": {
            "recommended_budget_id": inside_trigger_budget_id,
            "selection_policy": inside_trigger_budget_policy,
        },
    }
    runtime_payload = {
        "schema_version": 1,
        "status": "promote",
        "recommendation": recommendation,
        "decision": {
            "status": "promote",
            "recommended_cell": cell_id,
            "recommended_layer": layer,
            "recommended_batch_size": batch_size,
            "recommended_best_quality_signal": best_quality_signal_name,
        },
    }
    runtime_path.write_text(json.dumps(runtime_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_payload = {
        "schema_version": 1,
        "workflow": "performance_baseline_workflow",
        "status": "promote",
        "decision": {
            "status": "promote",
            "recommended_cell": cell_id,
            "recommended_layer": layer,
            "recommended_batch_size": batch_size,
            "recommended_best_quality_signal": best_quality_signal_name,
        },
        "runtime_recommendation": runtime_payload,
        "paths": {
            "runtime_recommendation": stored(runtime_path),
            "artifact_manifest": stored(manifest_path),
        },
        "registry_record": f"performance_baseline:{name}:{version}",
    }
    report_path.write_text(json.dumps(report_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            build_artifact_manifest(
                {
                    "performance_baseline_report": report_path,
                    "runtime_recommendation": runtime_path,
                },
                root=output_dir,
                metadata={
                    "runner": "run_performance_baseline_workflow",
                    "status": "promote",
                    "runtime_recommendation_status": "promote",
                    "recommended_layer": layer,
                    "recommended_batch_size": batch_size,
                    "recommended_best_quality_signal": best_quality_signal_name,
                },
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    ArtifactRegistry.load_json(registry_path).record_performance_baseline(
        name=name,
        version=version,
        path=stored(report_path),
        metadata={
            "workflow": "run_performance_baseline_workflow",
            "status": "promote",
            "runtime_recommendation_status": "promote",
            "artifact_manifest": stored(manifest_path),
            "runtime_recommendation": stored(runtime_path),
            "recommended_layer": layer,
            "recommended_batch_size": batch_size,
            "recommended_best_quality_signal": best_quality_signal_name,
        },
    ).save_json()
    return manifest_path


def _write_fake_readiness_report(output_dir, *, status, runtime_status):
    from eigentruth.registry import build_artifact_manifest

    output_dir.mkdir(parents=True, exist_ok=True)
    readiness_path = output_dir / "adapter-readiness-report.json"
    runtime_path = output_dir / "runtime-recommendation.json"
    manifest_path = output_dir / "artifact-manifest.json"
    runtime_payload = {
        "status": runtime_status,
        "recommendation": (
            {
                "layer": -1,
                "batch_size": 2,
                "hidden_state_capture": "outputs",
                "max_batch_tokens": 0,
                "prefix_kv_cache": False,
                "max_workers": 1,
                "quality_signals": {
                    "truth_proj": 0.8,
                    "subspace_resid": 0.91,
                },
                "best_quality_signal": {
                    "name": "subspace_resid",
                    "auroc": 0.91,
                },
                "inside_sampling": {
                    "recommended_run": "adaptive_selfcheck",
                    "total_generated_samples": 8,
                    "sample_count_ratio_to_baseline": 0.4,
                    "inside_generation_seconds": 1.25,
                    "inside_generation_seconds_ratio_to_baseline": 0.45,
                    "stop_reason_counts": {"stability_delta": 3},
                },
            }
            if runtime_status == "promote"
            else None
        ),
    }
    report = {
        "artifact_manifest": str(manifest_path),
        "runtime_recommendation_path": str(runtime_path),
        "runtime_recommendation": runtime_payload,
        "readiness_decision": {
            "status": status,
            "adapter_family_status": "promote",
            "performance_status": "promote" if status == "promote" else "dry_run",
            "recommended_route": "structured_qa",
            "recommended_performance_cell": "layer_m1_batch_2_capture_outputs",
        },
    }
    readiness_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    runtime_path.write_text(json.dumps(runtime_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            build_artifact_manifest(
                {
                    "readiness_report": readiness_path,
                    "runtime_recommendation": runtime_path,
                },
                root=output_dir,
                metadata={
                    "runner": "run_adapter_readiness_workflow",
                    "readiness_status": status,
                    "runtime_recommendation_status": runtime_status,
                },
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return report


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


def test_inside_sampling_profile_smoke_writes_pass_and_expected_failure_reports(tmp_path):
    module = importlib.import_module("benchmarks.inside_sampling_profile_smoke")

    payload = module.build_inside_sampling_profile_smoke(tmp_path)
    pass_report = payload["pass_report"]
    failure_report = payload["expected_failure_report"]

    assert (tmp_path / "pass" / "result-fixed.json").exists()
    assert (tmp_path / "pass" / "profile-adaptive_selfcheck.json").exists()
    assert (tmp_path / "failure" / "result-adaptive_selfcheck.json").exists()
    assert (tmp_path / "inside_sampling_profile_pass_report.json").exists()
    assert (tmp_path / "inside_sampling_profile_expected_failure_report.json").exists()
    assert pass_report["sample_efficiency_gate"]["passed"] is True
    assert pass_report["recommendation"]["recommended_run"] == "adaptive_selfcheck"
    assert failure_report["sample_efficiency_gate"]["passed"] is False
    assert {failure["metric"] for failure in failure_report["sample_efficiency_gate"]["failures"]} == {
        "inside_generation_seconds_ratio_to_baseline",
        "sample_count_ratio_to_baseline",
    }


def test_cache_worker_sweep_smoke_writes_pass_and_expected_blocked_reports(tmp_path):
    module = importlib.import_module("benchmarks.cache_worker_sweep_smoke")

    payload = module.build_cache_worker_sweep_smoke(tmp_path)
    pass_report = payload["pass_report"]
    blocked_report = payload["expected_blocked_report"]

    assert (tmp_path / "cache_worker_sweep_pass_report.json").exists()
    assert (tmp_path / "cache_worker_sweep_expected_blocked_report.json").exists()
    assert pass_report["worker_sweep_decision"]["status"] == "promote"
    assert pass_report["worker_sweep_decision"]["recommended_worker_count"] == 2
    assert blocked_report["worker_sweep_decision"]["status"] == "blocked"
    assert blocked_report["worker_reports"][1]["matrix_status"] == "blocked"


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


def test_performance_baseline_smoke_writes_registered_baseline(tmp_path):
    module = importlib.import_module("benchmarks.performance_baseline_smoke")
    registry_module = importlib.import_module("eigentruth.registry")

    payload = module.build_performance_baseline_smoke(tmp_path)
    registry = registry_module.ArtifactRegistry.load_json(tmp_path / "registry.json")

    assert payload["status"] == "promote"
    assert payload["registry_record"] == "performance_baseline:performance-baseline-smoke:0.1"
    assert Path(payload["paths"]["artifact_manifest"]).exists()
    assert registry.get("performance_baseline:performance-baseline-smoke:0.1").metadata[
        "runtime_recommendation_status"
    ] == "promote"


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
        max_batch_tokens=128,
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
    assert manifest["metadata"]["max_batch_tokens"] == 128
    assert manifest["artifacts"]["command_log"]["exists"] is True
    assert manifest["artifacts"]["caches.eval_reps_cache"]["exists"] is False
    assert payload["artifact_manifest_summary"]["missing_count"] == 3
    assert commands["uncached"][0] == "/python"
    assert commands["uncached"][1] == "benchmarks/eval_truthfulqa.py"
    assert not Path(commands["uncached"][1]).is_absolute()
    assert "--offline" in commands["uncached"]
    assert commands["uncached"][commands["uncached"].index("--dtype") + 1] == "float32"
    assert commands["uncached"][commands["uncached"].index("--max-batch-tokens") + 1] == "128"
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


def test_run_cache_profile_triplet_can_enable_prefix_kv_cache(tmp_path):
    module = importlib.import_module("benchmarks.run_cache_profile_triplet")
    config = module.CacheProfileTripletConfig(
        output_dir=tmp_path,
        model="tiny-local",
        prefix_kv_cache=True,
        python_executable="/python",
    )

    payload = module.run_triplet(config, clean=True, dry_run=True)

    assert "--prefix-kv-cache" in payload["commands"]["uncached"]
    assert "--prefix-kv-cache" in payload["commands"]["cached"]
    assert "--prefix-kv-cache" not in payload["commands"]["cache_only"]
    manifest = json.loads(Path(payload["artifact_manifest"]).read_text(encoding="utf-8"))
    assert manifest["metadata"]["prefix_kv_cache"] is True


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


def test_run_inside_sampling_profile_builds_dry_run_commands(tmp_path):
    module = importlib.import_module("benchmarks.run_inside_sampling_profile")
    config = module.InsideSamplingProfileConfig(
        output_dir=tmp_path,
        model="tiny-local",
        layer=-2,
        inside_samples=6,
        inside_min_samples=2,
        inside_sample_step=2,
        inside_stability_delta=0.01,
        inside_trigger_signal="truth_proj",
        inside_trigger_top_fraction=0.25,
        dump_inside_samples=True,
        statement_encoding_cache_path=tmp_path / "shared" / "statement-encodings.json",
        layer_stats_cache_path=tmp_path / "shared" / "layer-stats.pt",
        eval_reps_cache_path=tmp_path / "shared" / "eval-reps-cache",
        eval_reps_cache_shard_size=8,
        inside_diagnostics_cache_path=tmp_path / "shared" / "inside-diagnostics.json",
        refresh_shared_caches=True,
        python_executable="/python",
    )

    payload = module.run_inside_sampling_profile(config, clean=True, dry_run=True)
    commands = payload["commands"]
    fixed = commands["fixed"]
    adaptive = commands["adaptive"]
    adaptive_selfcheck = commands["adaptive_selfcheck"]
    manifest = json.loads(Path(payload["artifact_manifest"]).read_text(encoding="utf-8"))

    assert payload["dry_run"] is True
    assert Path(payload["command_log"]).exists()
    assert manifest["metadata"]["runner"] == "run_inside_sampling_profile"
    assert manifest["metadata"]["inside_samples"] == 6
    assert manifest["metadata"]["inside_trigger_signal"] == "truth_proj"
    assert manifest["metadata"]["inside_trigger_top_fraction"] == 0.25
    assert manifest["metadata"]["shared_caches"]["layer_stats_cache"].endswith("layer-stats.pt")
    assert manifest["metadata"]["shared_caches"]["inside_diagnostics_cache"].endswith("inside-diagnostics.json")
    assert manifest["metadata"]["eval_reps_cache_shard_size"] == 8
    assert manifest["metadata"]["refresh_shared_caches"] is True
    assert fixed[0] == "/python"
    assert fixed[1] == str(Path("benchmarks") / "eval_truthfulqa.py")
    assert "--offline" in fixed
    assert fixed[fixed.index("--inside-samples") + 1] == "6"
    assert fixed[fixed.index("--inside-trigger-signal") + 1] == "truth_proj"
    assert fixed[fixed.index("--inside-trigger-top-fraction") + 1] == "0.25"
    assert fixed[fixed.index("--statement-encoding-cache") + 1].endswith("statement-encodings.json")
    assert fixed[fixed.index("--layer-stats-cache") + 1].endswith("layer-stats.pt")
    assert fixed[fixed.index("--eval-reps-cache") + 1].endswith("eval-reps-cache")
    assert fixed[fixed.index("--eval-reps-cache-shard-size") + 1] == "8"
    assert fixed[fixed.index("--inside-diagnostics-cache") + 1].endswith("inside-diagnostics.json")
    assert "--refresh-statement-encoding-cache" in fixed
    assert "--refresh-layer-stats-cache" in fixed
    assert "--refresh-eval-reps-cache" in fixed
    assert "--refresh-inside-diagnostics-cache" in fixed
    assert "--inside-adaptive-sampling" not in fixed
    assert "--inside-adaptive-sampling" in adaptive
    assert adaptive[adaptive.index("--inside-sample-step") + 1] == "2"
    assert adaptive[adaptive.index("--eval-reps-cache") + 1].endswith("eval-reps-cache")
    assert "--refresh-eval-reps-cache" not in adaptive
    assert adaptive[adaptive.index("--inside-diagnostics-cache") + 1].endswith("inside-diagnostics.json")
    assert "--refresh-inside-diagnostics-cache" not in adaptive
    assert "--inside-selfcheck-early-stop" not in adaptive
    assert "--inside-selfcheck-early-stop" in adaptive_selfcheck
    assert "--dump-scores" in fixed
    assert "--dump-inside-samples" in fixed


def test_run_inside_sampling_profile_rejects_incomplete_trigger_config(tmp_path):
    module = importlib.import_module("benchmarks.run_inside_sampling_profile")

    with pytest.raises(ValueError, match="inside_trigger_signal"):
        module.InsideSamplingProfileConfig(
            output_dir=tmp_path,
            inside_trigger_top_fraction=0.25,
        )
    with pytest.raises(ValueError, match="requires"):
        module.InsideSamplingProfileConfig(
            output_dir=tmp_path,
            inside_trigger_signal="truth_proj",
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        module.InsideSamplingProfileConfig(
            output_dir=tmp_path,
            inside_trigger_signal="truth_proj",
            inside_trigger_threshold=0.5,
            inside_trigger_top_fraction=0.25,
        )
    with pytest.raises(ValueError, match="eval_reps_cache_shard_size"):
        module.InsideSamplingProfileConfig(
            output_dir=tmp_path,
            eval_reps_cache_shard_size=4,
        )


def test_run_inside_sampling_profile_rejects_invalid_sampling_distribution_config(tmp_path):
    module = importlib.import_module("benchmarks.run_inside_sampling_profile")

    with pytest.raises(ValueError, match="inside_temperature"):
        module.InsideSamplingProfileConfig(
            output_dir=tmp_path,
            inside_temperature=0.0,
        )
    with pytest.raises(ValueError, match="inside_top_p"):
        module.InsideSamplingProfileConfig(
            output_dir=tmp_path,
            inside_top_p=0.0,
        )


def test_run_inside_sampling_profile_cli_accepts_explicit_offline(tmp_path, capsys):
    module = importlib.import_module("benchmarks.run_inside_sampling_profile")

    module.main([
        "--output-dir",
        str(tmp_path),
        "--runs",
        "fixed",
        "--offline",
        "--dry-run",
    ])

    report = json.loads(capsys.readouterr().out)

    assert report["dry_run"] is True
    assert report["commands"]["fixed"].count("--offline") == 1
    assert Path(report["artifact_manifest"]).exists()


def test_inside_sampling_profile_comparison_reports_sample_savings(tmp_path):
    module = importlib.import_module("benchmarks.run_inside_sampling_profile")
    runs = {}
    fixtures = {
        "fixed": {"samples": 20, "seconds": 10.0, "stopped": 0, "reasons": {}},
        "adaptive": {"samples": 14, "seconds": 7.0, "stopped": 3, "reasons": {"stability_delta": 3}},
        "adaptive_selfcheck": {
            "samples": 8,
            "seconds": 4.5,
            "stopped": 4,
            "reasons": {"selfcheck_refute_threshold_guaranteed": 4},
        },
    }
    for name, fixture in fixtures.items():
        result_path = tmp_path / f"result-{name}.json"
        profile_path = tmp_path / f"profile-{name}.json"
        result_path.write_text(
            json.dumps({
                "inside_sampling": {
                    "adaptive": name != "fixed",
                    "selfcheck_early_stop": name == "adaptive_selfcheck",
                    "mode": "triggered",
                    "signal": "truth_proj",
                    "top_fraction": 0.8,
                    "sampled": 4,
                    "not_sampled": 1,
                    "triggered": 4,
                    "skipped_by_trigger": 1,
                    "total_generated_samples": fixture["samples"],
                    "mean_samples_per_record": fixture["samples"] / 4,
                    "mean_samples_per_sampled_record": fixture["samples"] / 4,
                    "stopped_early": fixture["stopped"],
                    "stop_reason_counts": fixture["reasons"],
                }
            }),
            encoding="utf-8",
        )
        profile_path.write_text(
            json.dumps({
                "total_seconds": fixture["seconds"] + 1.0,
                "phases": {"inside_generation": fixture["seconds"]},
            }),
            encoding="utf-8",
        )
        runs[name] = {"result": result_path, "profile": profile_path}

    report = module.build_inside_sampling_comparison(
        runs,
        max_sample_ratios={"adaptive": 0.80, "adaptive_selfcheck": 0.50},
        max_inside_generation_seconds_ratio=0.80,
    )

    assert report["sample_efficiency_gate"]["passed"] is True
    assert report["recommendation"]["recommended_run"] == "adaptive_selfcheck"
    assert report["runs"]["adaptive"]["sample_count_ratio_to_baseline"] == pytest.approx(0.70)
    assert report["runs"]["adaptive_selfcheck"]["sample_count_ratio_to_baseline"] == pytest.approx(0.40)
    assert report["runs"]["adaptive_selfcheck"]["inside_generation_seconds_ratio_to_baseline"] == pytest.approx(0.45)
    assert report["runs"]["adaptive_selfcheck"]["mode"] == "triggered"
    assert report["runs"]["adaptive_selfcheck"]["trigger_signal"] == "truth_proj"
    assert report["runs"]["adaptive_selfcheck"]["trigger_top_fraction"] == 0.8
    assert report["runs"]["adaptive_selfcheck"]["skipped_by_trigger"] == 1
    assert report["runs"]["adaptive_selfcheck"]["stop_reason_counts"] == {
        "selfcheck_refute_threshold_guaranteed": 4
    }


def test_inside_sampling_profile_comparison_fails_closed_on_nonfinite_runtime(tmp_path):
    module = importlib.import_module("benchmarks.run_inside_sampling_profile")
    runs = {}
    for name, seconds in {"fixed": 10.0, "adaptive": "NaN"}.items():
        result_path = tmp_path / f"result-{name}.json"
        profile_path = tmp_path / f"profile-{name}.json"
        result_path.write_text(
            json.dumps({
                "inside_sampling": {
                    "sampled": 4,
                    "total_generated_samples": 8 if name == "adaptive" else 10,
                    "stop_reason_counts": {},
                }
            }),
            encoding="utf-8",
        )
        profile_path.write_text(
            json.dumps({"total_seconds": seconds, "phases": {"inside_generation": seconds}}),
            encoding="utf-8",
        )
        runs[name] = {"result": result_path, "profile": profile_path}

    report = module.build_inside_sampling_comparison(
        runs,
        max_sample_ratios={"adaptive": 1.0},
        max_inside_generation_seconds_ratio=1.0,
    )

    assert report["runs"]["adaptive"]["inside_generation_seconds"] is None
    assert report["runs"]["adaptive"]["inside_generation_seconds_ratio_to_baseline"] is None
    assert report["sample_efficiency_gate"]["passed"] is False
    assert report["sample_efficiency_gate"]["failures"] == [{
        "run": "adaptive",
        "metric": "inside_generation_seconds_ratio_to_baseline",
        "value": None,
        "max_allowed": 1.0,
    }]


def test_run_inside_sampling_profile_writes_comparison_report(tmp_path, monkeypatch):
    module = importlib.import_module("benchmarks.run_inside_sampling_profile")
    config = module.InsideSamplingProfileConfig(
        output_dir=tmp_path,
        model="tiny-local",
        python_executable="/python",
        dump_scores=True,
        adaptive_max_sample_ratio=0.80,
        adaptive_selfcheck_max_sample_ratio=0.60,
    )
    sample_counts = {
        "fixed": 20,
        "adaptive": 14,
        "adaptive_selfcheck": 10,
    }
    calls = []

    def fake_run(command, *, cwd, check):
        calls.append({"command": command, "cwd": cwd, "check": check})
        result_path = Path(command[command.index("--json") + 1])
        profile_path = Path(command[command.index("--profile-json") + 1])
        name = result_path.stem.removeprefix("result-")
        samples = sample_counts[name]
        result_path.write_text(
            json.dumps({
                "inside_sampling": {
                    "adaptive": name != "fixed",
                    "selfcheck_early_stop": name == "adaptive_selfcheck",
                    "sampled": 4,
                    "total_generated_samples": samples,
                    "mean_samples_per_record": samples / 4,
                    "mean_samples_per_sampled_record": samples / 4,
                    "stopped_early": 2 if name != "fixed" else 0,
                    "stop_reason_counts": {"stability_delta": 2} if name == "adaptive" else {},
                }
            }),
            encoding="utf-8",
        )
        profile_path.write_text(
            json.dumps({
                "total_seconds": float(samples),
                "phases": {"inside_generation": float(samples) / 2},
            }),
            encoding="utf-8",
        )
        if "--dump-scores" in command:
            score_path = Path(command[command.index("--dump-scores") + 1])
            score_path.write_text(json.dumps({"labels": [], "scores": {}, "inside_sampling": {}}), encoding="utf-8")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    payload = module.run_inside_sampling_profile(config, clean=True, dry_run=False)
    report = json.loads(Path(payload["comparison_report"]).read_text(encoding="utf-8"))
    manifest = json.loads(Path(payload["artifact_manifest"]).read_text(encoding="utf-8"))

    assert [call["check"] for call in calls] == [True, True, True]
    assert payload["dry_run"] is False
    assert payload["sample_efficiency_gate"]["passed"] is True
    assert report["recommendation"]["recommended_run"] == "adaptive_selfcheck"
    assert manifest["artifacts"]["comparison_report"]["exists"] is True
    assert manifest["artifacts"]["profiles.adaptive_selfcheck"]["sha256"]
    assert manifest["artifacts"]["score_dumps.fixed"]["sha256"]

    calls.clear()
    reused = module.run_inside_sampling_profile(config, clean=False, dry_run=False, skip_existing=True)

    assert calls == []
    assert reused["reused_runs"] == ("fixed", "adaptive", "adaptive_selfcheck")
    assert reused["sample_efficiency_gate"]["passed"] is True


def test_run_inside_trigger_budget_sweep_builds_dry_run_commands(tmp_path):
    module = importlib.import_module("benchmarks.run_inside_trigger_budget_sweep")
    config = module.InsideTriggerBudgetSweepConfig(
        output_dir=tmp_path,
        trigger_signal="truth_proj",
        budgets=(
            module.TriggerBudgetSpec("top_fraction", 0.1),
            module.TriggerBudgetSpec("top_fraction", 0.25),
        ),
        model="tiny-local",
        layer=-2,
        inside_samples=3,
        shared_cache_dir=tmp_path / "shared-caches",
        eval_reps_cache_shard_size=4,
        refresh_shared_caches=True,
        python_executable="/python",
        run_names=("fixed", "adaptive_selfcheck"),
    )

    report = module.run_inside_trigger_budget_sweep(config, clean=True, dry_run=True)
    manifest = json.loads(Path(report["artifact_manifest"]).read_text(encoding="utf-8"))
    top10_fixed = report["budgets"]["top_0p1"]["commands"]["fixed"]
    top25_selfcheck = report["budgets"]["top_0p25"]["commands"]["adaptive_selfcheck"]

    assert report["dry_run"] is True
    assert report["config"]["trigger_signal"] == "truth_proj"
    assert report["config"]["shared_cache_dir"].endswith("shared-caches")
    assert top10_fixed[top10_fixed.index("--inside-trigger-signal") + 1] == "truth_proj"
    assert top10_fixed[top10_fixed.index("--inside-trigger-top-fraction") + 1] == "0.1"
    assert top10_fixed[top10_fixed.index("--eval-reps-cache") + 1].endswith("shared-caches/eval-reps-cache")
    assert top10_fixed[top10_fixed.index("--eval-reps-cache-shard-size") + 1] == "4"
    assert top10_fixed[top10_fixed.index("--inside-diagnostics-cache") + 1].endswith(
        "shared-caches/inside-diagnostics.json"
    )
    assert "--refresh-eval-reps-cache" in top10_fixed
    assert "--refresh-inside-diagnostics-cache" in top10_fixed
    assert top25_selfcheck[top25_selfcheck.index("--eval-reps-cache") + 1].endswith(
        "shared-caches/eval-reps-cache"
    )
    assert top25_selfcheck[top25_selfcheck.index("--inside-diagnostics-cache") + 1].endswith(
        "shared-caches/inside-diagnostics.json"
    )
    assert "--refresh-eval-reps-cache" not in top25_selfcheck
    assert "--refresh-inside-diagnostics-cache" not in top25_selfcheck
    assert "--inside-selfcheck-early-stop" in top25_selfcheck
    assert manifest["metadata"]["runner"] == "run_inside_trigger_budget_sweep"
    assert manifest["metadata"]["shared_cache_dir"].endswith("shared-caches")
    assert manifest["metadata"]["eval_reps_cache_shard_size"] == 4
    assert manifest["metadata"]["budgets"] == [
        {"kind": "top_fraction", "value": 0.1},
        {"kind": "top_fraction", "value": 0.25},
    ]


def test_run_inside_trigger_budget_sweep_cli_accepts_explicit_offline(tmp_path, capsys):
    module = importlib.import_module("benchmarks.run_inside_trigger_budget_sweep")

    module.main([
        "--output-dir",
        str(tmp_path),
        "--trigger-signal",
        "truth_proj",
        "--top-fractions",
        "0.1",
        "--runs",
        "fixed",
        "--offline",
        "--dry-run",
    ])

    report = json.loads(capsys.readouterr().out)

    assert report["dry_run"] is True
    assert report["config"]["offline"] is True
    assert Path(report["artifact_manifest"]).exists()


def test_run_inside_trigger_budget_sweep_compares_budgets_to_reference(tmp_path, monkeypatch):
    module = importlib.import_module("benchmarks.run_inside_trigger_budget_sweep")
    reference_report = tmp_path / "full-reference.json"
    reference_report.write_text(
        json.dumps({
            "runs": {
                "fixed": {
                    "total_generated_samples": 300,
                    "inside_generation_seconds": 120.0,
                }
            }
        }),
        encoding="utf-8",
    )

    def fake_profile(config, *, clean, dry_run, skip_existing):
        del clean, dry_run, skip_existing
        config.output_dir.mkdir(parents=True, exist_ok=True)
        fraction = float(config.inside_trigger_top_fraction)
        fixed_result = config.output_dir / "result-fixed.json"
        adaptive_result = config.output_dir / "result-adaptive_selfcheck.json"
        fixed_result.write_text(json.dumps({"auroc": {"inside_eigenscore": 0.50}}), encoding="utf-8")
        adaptive_result.write_text(
            json.dumps({"auroc": {"inside_eigenscore": 0.50 + fraction}}),
            encoding="utf-8",
        )
        comparison_path = config.output_dir / "inside-sampling-profile-comparison.json"
        comparison_path.write_text(
            json.dumps({
                "runs": {
                    "fixed": {
                        "name": "fixed",
                        "result_path": str(fixed_result),
                        "sampled": int(100 * fraction),
                        "skipped_by_trigger": 100 - int(100 * fraction),
                        "total_generated_samples": int(300 * fraction),
                        "mean_samples_per_record": 3.0 * fraction,
                        "inside_generation_seconds": 120.0 * fraction,
                        "sample_count_ratio_to_baseline": 1.0,
                        "inside_generation_seconds_ratio_to_baseline": 1.0,
                    },
                    "adaptive_selfcheck": {
                        "name": "adaptive_selfcheck",
                        "result_path": str(adaptive_result),
                        "sampled": int(100 * fraction),
                        "skipped_by_trigger": 100 - int(100 * fraction),
                        "total_generated_samples": int(250 * fraction),
                        "mean_samples_per_record": 2.5 * fraction,
                        "inside_generation_seconds": 100.0 * fraction,
                        "sample_count_ratio_to_baseline": 5.0 / 6.0,
                        "inside_generation_seconds_ratio_to_baseline": 5.0 / 6.0,
                    },
                },
                "recommendation": {"recommended_run": "adaptive_selfcheck"},
                "sample_efficiency_gate": {"passed": True},
            }),
            encoding="utf-8",
        )
        manifest_path = config.output_dir / "artifact-manifest.json"
        manifest_path.write_text(json.dumps({"summary": {"artifact_count": 1}}), encoding="utf-8")
        return {
            "output_dir": str(config.output_dir),
            "comparison_report": str(comparison_path),
            "artifact_manifest": str(manifest_path),
            "sample_efficiency_gate": {"passed": True},
            "recommendation": {"recommended_run": "adaptive_selfcheck"},
        }

    monkeypatch.setattr(module, "run_inside_sampling_profile", fake_profile)
    config = module.InsideTriggerBudgetSweepConfig(
        output_dir=tmp_path / "sweep",
        trigger_signal="truth_proj",
        budgets=(
            module.TriggerBudgetSpec("top_fraction", 0.2),
            module.TriggerBudgetSpec("top_fraction", 0.1),
        ),
        reference_report_path=reference_report,
        run_names=("fixed", "adaptive_selfcheck"),
    )

    report = module.run_inside_trigger_budget_sweep(config, clean=True, dry_run=False)
    manifest = json.loads(Path(report["artifact_manifest"]).read_text(encoding="utf-8"))

    assert report["recommendation"] == {
        "budget_id": "top_0p1",
        "recommended_run": "adaptive_selfcheck",
        "reason": "lowest_total_generated_samples_then_inside_generation_seconds",
    }
    assert report["quality_balanced_recommendation"] == {
        "budget_id": "top_0p2",
        "recommended_run": "adaptive_selfcheck",
        "reason": "lowest_cost_within_inside_quality_tolerance",
        "quality_metric": "inside_eigenscore",
        "quality_value": pytest.approx(0.70),
        "best_quality_value": pytest.approx(0.70),
        "quality_tolerance": 0.02,
        "cost_metric": "inside_generation_seconds_ratio_to_reference",
        "cost_value": pytest.approx(1.0 / 6.0),
    }
    assert [row["budget_id"] for row in report["leaderboard"]] == ["top_0p1", "top_0p2"]
    assert report["leaderboard"][0]["inside_generation_seconds_ratio_to_reference"] == pytest.approx(1.0 / 12.0)
    assert report["leaderboard"][0]["sample_count_ratio_to_reference"] == pytest.approx(25 / 300)
    assert report["leaderboard"][0]["inside_auroc"] == {"inside_eigenscore": pytest.approx(0.60)}
    assert manifest["artifacts"]["budgets.top_0p1.comparison_report"]["exists"] is True
    assert manifest["artifacts"]["budgets.top_0p2.profile_manifest"]["exists"] is True


def test_run_inside_trigger_budget_sweep_can_derive_top_fraction_rows_from_max_budget(tmp_path, monkeypatch):
    module = importlib.import_module("benchmarks.run_inside_trigger_budget_sweep")
    calls = []

    def fake_profile(config, *, clean, dry_run, skip_existing):
        del clean, dry_run, skip_existing
        calls.append(config)
        assert config.inside_trigger_top_fraction == pytest.approx(2 / 3)
        assert config.dump_scores is True
        assert config.run_names == ("adaptive_selfcheck",)
        config.output_dir.mkdir(parents=True, exist_ok=True)
        run_name = "adaptive_selfcheck"
        result_path = config.result_path(run_name)
        profile_path = config.profile_path(run_name)
        score_dump_path = config.score_dump_path(run_name)
        comparison_path = config.comparison_report
        manifest_path = config.artifact_manifest
        result_path.write_text(json.dumps({"auroc": {"inside_eigenscore": 0.75}}), encoding="utf-8")
        profile_path.write_text(
            json.dumps({"total_seconds": 20.0, "phases": {"inside_generation": 10.0}}),
            encoding="utf-8",
        )
        score_dump_path.write_text(
            json.dumps({
                "config": {"inside_trigger_top_fraction": 2 / 3},
                "inside_sampling": {"top_fraction": 2 / 3, "fill_value_for_untriggered": 0.0},
                "labels": [0, 1, 0, 1, 1, 0],
                "batch_indexes": [0, 0, 0, 1, 1, 1],
                "scores": {
                    "truth_proj": [0.1, 0.9, 0.4, 0.2, 0.8, 0.7],
                    "inside_eigenscore": [0.0, 2.0, 1.0, 0.0, 3.0, 1.5],
                    "inside_semantic_entropy": [0.0, 0.5, 0.2, 0.0, 0.9, 0.4],
                },
                "inside_sampled": [False, True, True, False, True, True],
                "inside_sample_counts": [0, 2, 2, 0, 3, 3],
                "inside_stopped_early": [False, False, True, False, True, False],
                "inside_stop_reasons": [None, None, "stability_delta", None, "selfcheck_supported", None],
            }),
            encoding="utf-8",
        )
        comparison_path.write_text(
            json.dumps({
                "runs": {
                    run_name: {
                        "name": run_name,
                        "result_path": str(result_path),
                        "profile_path": str(profile_path),
                        "sampled": 4,
                        "skipped_by_trigger": 2,
                        "total_generated_samples": 10,
                        "mean_samples_per_record": 10 / 6,
                        "mean_samples_per_sampled_record": 2.5,
                        "inside_generation_seconds": 10.0,
                    }
                },
                "recommendation": {"recommended_run": run_name},
                "sample_efficiency_gate": {"passed": True},
            }),
            encoding="utf-8",
        )
        manifest_path.write_text(json.dumps({"summary": {"artifact_count": 1}}), encoding="utf-8")
        return {
            "dry_run": False,
            "output_dir": str(config.output_dir),
            "comparison_report": str(comparison_path),
            "artifact_manifest": str(manifest_path),
            "results": {run_name: str(result_path)},
            "profiles": {run_name: str(profile_path)},
            "score_dumps": {run_name: str(score_dump_path)},
            "sample_efficiency_gate": {"passed": True},
            "recommendation": {"recommended_run": run_name},
        }

    monkeypatch.setattr(module, "run_inside_sampling_profile", fake_profile)
    config = module.InsideTriggerBudgetSweepConfig(
        output_dir=tmp_path / "sweep",
        trigger_signal="truth_proj",
        budgets=(
            module.TriggerBudgetSpec("top_fraction", 1 / 3),
            module.TriggerBudgetSpec("top_fraction", 2 / 3),
        ),
        run_names=("adaptive_selfcheck",),
        derive_from_max_budget=True,
    )

    report = module.run_inside_trigger_budget_sweep(config, clean=True)
    manifest = json.loads(Path(report["artifact_manifest"]).read_text(encoding="utf-8"))
    small = next(row for row in report["leaderboard"] if row["budget_id"] == "top_0p333333")
    large = next(row for row in report["leaderboard"] if row["budget_id"] == "top_0p666667")

    assert len(calls) == 1
    assert report["derived_from_max_budget"] is True
    assert report["derived_source_budget_id"] == "top_0p666667"
    assert small["sampled"] == 2
    assert small["skipped_by_trigger"] == 4
    assert small["total_generated_samples"] == 5
    assert small["inside_generation_seconds"] == pytest.approx(5.0)
    assert small["inside_generation_seconds_source"] == "sample_count_ratio_estimate"
    assert small["stop_reason_counts"] == {"selfcheck_supported": 1}
    assert large["total_generated_samples"] == 10
    assert large["inside_generation_seconds"] == pytest.approx(10.0)
    assert large["inside_generation_seconds_source"] == "measured_source_run"
    assert report["recommendation"]["budget_id"] == "top_0p333333"
    assert manifest["artifacts"]["budgets.top_0p333333.source_score_dump"]["exists"] is True


def test_run_inside_trigger_budget_sweep_derived_mode_rejects_multiple_runs(tmp_path):
    module = importlib.import_module("benchmarks.run_inside_trigger_budget_sweep")

    with pytest.raises(ValueError, match="exactly one run"):
        module.InsideTriggerBudgetSweepConfig(
            output_dir=tmp_path,
            trigger_signal="truth_proj",
            budgets=(module.TriggerBudgetSpec("top_fraction", 0.5),),
            run_names=("fixed", "adaptive_selfcheck"),
            derive_from_max_budget=True,
        )


def test_run_inside_trigger_budget_sweep_refreshes_child_manifests_for_mutable_shared_cache(tmp_path, monkeypatch):
    module = importlib.import_module("benchmarks.run_inside_trigger_budget_sweep")
    profile_module = importlib.import_module("benchmarks.run_inside_sampling_profile")
    manifest_module = importlib.import_module("benchmarks.verify_artifact_manifest")

    def fake_profile(config, *, clean, dry_run, skip_existing):
        del clean, dry_run, skip_existing
        config.output_dir.mkdir(parents=True, exist_ok=True)
        cache_path = config.inside_diagnostics_cache_path
        assert cache_path is not None
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        previous = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {"entries": []}
        previous["entries"].append(float(config.inside_trigger_top_fraction))
        cache_path.write_text(json.dumps(previous), encoding="utf-8")

        result_path = config.output_dir / "result-fixed.json"
        profile_path = config.output_dir / "profile-fixed.json"
        command_log_path = config.output_dir / "inside-sampling-profile-commands.json"
        comparison_path = config.output_dir / "inside-sampling-profile-comparison.json"
        result_path.write_text(json.dumps({"auroc": {"inside_eigenscore": 0.5}}), encoding="utf-8")
        profile_path.write_text(
            json.dumps({"total_seconds": 1.0, "phases": {"inside_generation": 1.0}}),
            encoding="utf-8",
        )
        command_log_path.write_text(json.dumps({"fixed": ["python"]}), encoding="utf-8")
        comparison_path.write_text(
            json.dumps({
                "runs": {
                    "fixed": {
                        "name": "fixed",
                        "result_path": str(result_path),
                        "profile_path": str(profile_path),
                        "sampled": 1,
                        "skipped_by_trigger": 0,
                        "total_generated_samples": 2,
                        "mean_samples_per_record": 2.0,
                        "inside_generation_seconds": 1.0,
                        "sample_count_ratio_to_baseline": 1.0,
                        "inside_generation_seconds_ratio_to_baseline": 1.0,
                    }
                },
                "recommendation": {"recommended_run": "fixed"},
                "sample_efficiency_gate": {"passed": True},
            }),
            encoding="utf-8",
        )
        payload = {
            "dry_run": False,
            "output_dir": str(config.output_dir),
            "command_log": str(command_log_path),
            "comparison_report": str(comparison_path),
            "results": {"fixed": str(result_path)},
            "profiles": {"fixed": str(profile_path)},
            "caches": {"inside_diagnostics_cache": str(cache_path)},
            "sample_efficiency_gate": {"passed": True},
            "recommendation": {"recommended_run": "fixed"},
        }
        profile_module._write_artifact_manifest(config, payload)
        payload["artifact_manifest"] = str(config.artifact_manifest)
        return payload

    monkeypatch.setattr(module, "run_inside_sampling_profile", fake_profile)
    config = module.InsideTriggerBudgetSweepConfig(
        output_dir=tmp_path / "sweep",
        trigger_signal="truth_proj",
        budgets=(
            module.TriggerBudgetSpec("top_fraction", 0.5),
            module.TriggerBudgetSpec("top_fraction", 1.0),
        ),
        shared_cache_dir=tmp_path / "sweep" / "shared-caches",
        run_names=("fixed",),
    )

    report = module.run_inside_trigger_budget_sweep(config, clean=True)
    verified = manifest_module.verify_manifest_file(Path(report["artifact_manifest"]), recursive=True)

    assert verified["passed"] is True
    assert json.loads((config.shared_cache_dir / "inside-diagnostics.json").read_text(encoding="utf-8")) == {
        "entries": [0.5, 1.0]
    }


def test_run_cache_profile_matrix_builds_dry_run_cells(tmp_path):
    module = importlib.import_module("benchmarks.run_cache_profile_matrix")
    config = module.CacheProfileMatrixConfig(
        output_dir=tmp_path,
        model="tiny-local",
        layers=(-2, -1),
        batch_sizes=(2,),
        hidden_state_captures=("outputs", "hooks"),
        max_batch_tokens=96,
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
    assert first["execution_seconds"] >= 0.0
    assert "--layer -2" in first["summary"]["commands"]["uncached"]
    assert "--max-batch-tokens 96" in first["summary"]["commands"]["uncached"]
    assert "--hidden-state-capture outputs" in first["summary"]["commands"]["uncached"]
    assert report["config"]["max_batch_tokens"] == 96
    assert report["config"]["max_workers"] == 1
    assert report["execution"]["wall_clock_seconds"] >= 0.0
    assert report["execution"]["cell_count"] == 4
    assert report["execution"]["max_workers"] == 1
    assert report["execution"]["shared_cache_refresh_barrier"] is False
    assert manifest["metadata"]["max_batch_tokens"] == 96
    assert manifest["metadata"]["wall_clock_seconds"] >= 0.0
    assert report["matrix_decision"]["status"] == "dry_run"
    assert report["matrix_decision"]["recommended_cell"] is None

    with pytest.raises(ValueError, match="max_workers"):
        module.CacheProfileMatrixConfig(output_dir=tmp_path / "bad-workers", max_workers=0)


def test_run_cache_profile_matrix_can_run_cells_in_parallel(tmp_path, monkeypatch):
    module = importlib.import_module("benchmarks.run_cache_profile_matrix")
    barrier = threading.Barrier(2, timeout=2.0)
    lock = threading.Lock()
    started = []

    def fake_run_triplet(config, *, clean, dry_run):
        del clean
        assert dry_run is True
        with lock:
            started.append(config.layer)
        barrier.wait()
        return {
            "dry_run": True,
            "output_dir": str(config.output_dir),
            "commands": {"uncached": ["/python"]},
            "run_names": ("uncached",),
            "caches": {},
            "uncached_cache_mode": config.uncached_cache_mode,
        }

    monkeypatch.setattr(module, "run_triplet", fake_run_triplet)
    config = module.CacheProfileMatrixConfig(
        output_dir=tmp_path / "runs",
        model="tiny-local",
        layers=(-2, -1),
        batch_sizes=(1,),
        max_workers=2,
        python_executable="/python",
    )

    report = module.run_matrix(config, clean=True, dry_run=True)

    assert sorted(started) == [-2, -1]
    assert [cell["id"] for cell in report["cells"]] == [
        "layer_m2_batch_1_capture_outputs",
        "layer_m1_batch_1_capture_outputs",
    ]
    assert report["config"]["max_workers"] == 2
    assert report["execution"]["max_workers"] == 2


def test_run_cache_profile_matrix_parallel_shared_cache_waits_for_refresh_cells(tmp_path, monkeypatch):
    module = importlib.import_module("benchmarks.run_cache_profile_matrix")
    lock = threading.Lock()
    calls = []
    violations = []
    refresh_done = 0

    def fake_run_triplet(config, *, clean, dry_run):
        nonlocal refresh_done
        del clean
        assert dry_run is True
        with lock:
            calls.append((config.layer, config.batch_size, config.uncached_cache_mode, refresh_done))
            if config.uncached_cache_mode != "refresh" and refresh_done < 2:
                violations.append((config.layer, config.batch_size, refresh_done))
        if config.uncached_cache_mode == "refresh":
            with lock:
                refresh_done += 1
        return {
            "dry_run": True,
            "output_dir": str(config.output_dir),
            "commands": {"uncached": ["/python"]},
            "run_names": ("uncached",),
            "caches": {"eval_reps_cache": str(config.eval_reps_cache)},
            "uncached_cache_mode": config.uncached_cache_mode,
        }

    monkeypatch.setattr(module, "run_triplet", fake_run_triplet)
    config = module.CacheProfileMatrixConfig(
        output_dir=tmp_path / "runs",
        shared_cache_dir=tmp_path / "shared-cache",
        model="tiny-local",
        layers=(-2, -1),
        batch_sizes=(1, 2),
        max_workers=4,
        python_executable="/python",
    )

    report = module.run_matrix(config, clean=True, dry_run=True)

    assert violations == []
    assert [call[2] for call in calls[:2]] == ["refresh", "refresh"]
    assert [cell["uncached_cache_mode"] for cell in report["cells"]] == [
        "refresh",
        "warm_start",
        "refresh",
        "warm_start",
    ]


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
    assert report["execution"]["shared_cache_refresh_barrier"] is True
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


def test_run_cache_profile_matrix_can_enable_prefix_kv_cache(tmp_path):
    module = importlib.import_module("benchmarks.run_cache_profile_matrix")
    config = module.CacheProfileMatrixConfig(
        output_dir=tmp_path,
        model="tiny-local",
        layers=(-2,),
        batch_sizes=(2,),
        prefix_kv_cache=True,
        python_executable="/python",
    )

    report = module.run_matrix(config, clean=True, dry_run=True)
    command = report["cells"][0]["triplet"]["commands"]["uncached"]
    manifest = json.loads(Path(report["artifact_manifest"]).read_text(encoding="utf-8"))

    assert report["config"]["prefix_kv_cache"] is True
    assert "--prefix-kv-cache" in command
    assert manifest["metadata"]["prefix_kv_cache"] is True


def test_run_cache_profile_matrix_compares_max_batch_token_budgets(tmp_path):
    module = importlib.import_module("benchmarks.run_cache_profile_matrix")
    config = module.CacheProfileMatrixConfig(
        output_dir=tmp_path / "runs",
        shared_cache_dir=tmp_path / "shared-cache",
        model="tiny-local",
        layers=(-2,),
        batch_sizes=(2,),
        max_batch_token_budgets=(0, 96),
        python_executable="/python",
    )

    report = module.run_matrix(config, clean=True, dry_run=True)
    base_cell, budget_cell = report["cells"]
    base_commands = base_cell["triplet"]["commands"]
    budget_commands = budget_cell["triplet"]["commands"]
    manifest = json.loads(Path(report["artifact_manifest"]).read_text(encoding="utf-8"))

    assert [cell["id"] for cell in report["cells"]] == [
        "layer_m2_batch_2_capture_outputs_token_budget_0",
        "layer_m2_batch_2_capture_outputs_token_budget_96",
    ]
    assert base_cell["max_batch_tokens"] == 0
    assert budget_cell["max_batch_tokens"] == 96
    assert base_commands["uncached"][base_commands["uncached"].index("--max-batch-tokens") + 1] == "0"
    assert budget_commands["uncached"][budget_commands["uncached"].index("--max-batch-tokens") + 1] == "96"
    assert base_cell["shared_cache_group"] == budget_cell["shared_cache_group"]
    assert (
        base_cell["triplet"]["caches"]["statement_encoding_cache"]
        == budget_cell["triplet"]["caches"]["statement_encoding_cache"]
    )
    assert (
        base_cell["triplet"]["caches"]["eval_reps_cache"]
        == budget_cell["triplet"]["caches"]["eval_reps_cache"]
    )
    assert report["config"]["max_batch_token_budgets"] == (0, 96)
    assert report["leaderboard_sort_metric"] == "uncached_forced_answer_forward_seconds"
    assert manifest["metadata"]["max_batch_token_budgets"] == [0, 96]

    with pytest.raises(ValueError, match="duplicate"):
        module.CacheProfileMatrixConfig(
            output_dir=tmp_path / "bad-duplicate",
            max_batch_token_budgets=(96, 96),
        )
    with pytest.raises(ValueError, match="triplet"):
        module.CacheProfileMatrixConfig(
            output_dir=tmp_path / "bad-rescore",
            shared_cache_dir=tmp_path / "shared",
            max_batch_token_budgets=(0, 96),
            matrix_mode="rescore",
        )


def test_run_cache_profile_matrix_compares_prefix_kv_cache_modes(tmp_path):
    module = importlib.import_module("benchmarks.run_cache_profile_matrix")
    config = module.CacheProfileMatrixConfig(
        output_dir=tmp_path / "runs",
        shared_cache_dir=tmp_path / "shared-cache",
        model="tiny-local",
        layers=(-2,),
        batch_sizes=(2,),
        prefix_kv_cache_modes=(False, True),
        python_executable="/python",
    )

    report = module.run_matrix(config, clean=True, dry_run=True)
    off_cell, on_cell = report["cells"]
    off_commands = off_cell["triplet"]["commands"]
    on_commands = on_cell["triplet"]["commands"]
    manifest = json.loads(Path(report["artifact_manifest"]).read_text(encoding="utf-8"))

    assert [cell["id"] for cell in report["cells"]] == [
        "layer_m2_batch_2_capture_outputs_prefix_kv_off",
        "layer_m2_batch_2_capture_outputs_prefix_kv_on",
    ]
    assert off_cell["prefix_kv_cache"] is False
    assert on_cell["prefix_kv_cache"] is True
    assert "--prefix-kv-cache" not in off_commands["uncached"]
    assert "--prefix-kv-cache" in on_commands["uncached"]
    assert off_cell["shared_cache_group"] != on_cell["shared_cache_group"]
    assert (
        off_cell["triplet"]["caches"]["statement_encoding_cache"]
        == on_cell["triplet"]["caches"]["statement_encoding_cache"]
    )
    assert off_cell["triplet"]["caches"]["eval_reps_cache"] != on_cell["triplet"]["caches"]["eval_reps_cache"]
    assert report["config"]["prefix_kv_cache_modes"] == (False, True)
    assert manifest["metadata"]["prefix_kv_cache_modes"] == [False, True]

    with pytest.raises(ValueError, match="duplicate"):
        module.CacheProfileMatrixConfig(
            output_dir=tmp_path / "bad",
            prefix_kv_cache_modes=(False, False),
        )


def test_run_cache_profile_matrix_prefix_modes_recommend_by_uncached_forward(tmp_path, monkeypatch):
    module = importlib.import_module("benchmarks.run_cache_profile_matrix")

    def write_profile(path: Path, *, total: float, forced_answer: float = 0.0) -> None:
        path.write_text(
            json.dumps({
                "total_seconds": total,
                "phases": {
                    "forced_answer_forward": forced_answer,
                    "score_postprocess": max(total - forced_answer, 0.0),
                },
                "summary": {
                    "bottleneck": "forced_answer_forward" if forced_answer else "score_postprocess",
                    "groups": {"model_forward": {"seconds": forced_answer}},
                },
            }),
            encoding="utf-8",
        )

    def fake_run_triplet(config, *, clean, dry_run):
        del clean, dry_run
        config.output_dir.mkdir(parents=True, exist_ok=True)
        is_prefix_on = bool(config.prefix_kv_cache)
        uncached_total = 110.0 if is_prefix_on else 120.0
        forced_answer = 100.0 if is_prefix_on else 80.0
        cache_only_total = 0.30 if is_prefix_on else 0.31
        profiles = {}
        for name, total, forced in (
            ("uncached", uncached_total, forced_answer),
            ("cached", 16.0, 0.0),
            ("cache_only", cache_only_total, 0.0),
        ):
            path = config.profile_path(name)
            write_profile(path, total=total, forced_answer=forced)
            profiles[name] = str(path)
        result_path = config.result_path("cache_only")
        result_path.write_text(json.dumps({"auroc": {"truth_proj": 0.88}}), encoding="utf-8")
        comparison_path = config.comparison_report
        comparison_path.write_text(
            json.dumps({
                "runs": [
                    {"name": "uncached", "total_seconds": uncached_total, "bottleneck": "forced_answer_forward"},
                    {"name": "cached", "total_seconds": 16.0, "bottleneck": "load_model"},
                    {
                        "name": "cache_only",
                        "total_seconds": cache_only_total,
                        "bottleneck": "score_postprocess",
                        "total_delta": {
                            "speedup_vs_baseline": uncached_total / cache_only_total,
                            "ratio_to_baseline": cache_only_total / uncached_total,
                        },
                    },
                ],
                "fastest": {"name": "cache_only", "total_seconds": cache_only_total},
                "regression_gate": {"passed": True},
            }),
            encoding="utf-8",
        )
        return {
            "dry_run": False,
            "output_dir": str(config.output_dir),
            "comparison_report": str(comparison_path),
            "profiles": profiles,
            "results": {"cache_only": str(result_path)},
            "regression_gate": {"passed": True},
        }

    monkeypatch.setattr(module, "run_triplet", fake_run_triplet)
    config = module.CacheProfileMatrixConfig(
        output_dir=tmp_path,
        layers=(-12,),
        batch_sizes=(1,),
        hidden_state_captures=("outputs",),
        prefix_kv_cache_modes=(False, True),
    )

    report = module.run_matrix(config, clean=True, dry_run=False)
    comparison = report["prefix_kv_comparisons"][0]

    assert report["leaderboard_sort_metric"] == "uncached_forced_answer_forward_seconds"
    assert report["leaderboard"][0]["id"] == "layer_m12_batch_1_capture_outputs_prefix_kv_off"
    assert report["matrix_decision"]["recommended_cell"] == "layer_m12_batch_1_capture_outputs_prefix_kv_off"
    assert report["matrix_decision"]["recommendation_metric"] == "uncached_forced_answer_forward_seconds"
    assert comparison["status"] == "prefix_kv_slower"
    assert comparison["recommended_prefix_kv_cache"] is False
    assert comparison["recommendation_metric"] == "forced_answer_forward_seconds"
    assert comparison["uncached_total_ratio_on_vs_off"] == pytest.approx(110.0 / 120.0)
    assert comparison["forced_answer_forward_ratio_on_vs_off"] == pytest.approx(100.0 / 80.0)


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
            if name == "uncached":
                total = 100.0
            elif name == "cache_only" and config.batch_size == 2:
                total = 8.0
            else:
                total = 10.0 + config.batch_size
            write_profile(profile_path, total)
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
    assert second["summary"]["regression_gate"]["passed"] is True
    assert second["summary"]["rescore_baseline"]["baseline_cell"] == "layer_m2_batch_1_capture_outputs"
    assert second["summary"]["comparison_skipped_reason"] == "baseline run 'uncached' was not executed"
    assert second["summary"]["totals"]["cache_only"]["total_seconds"] == pytest.approx(8.0)
    assert second["summary"]["totals"]["cache_only"]["ratio_to_baseline"] == pytest.approx(0.08)
    assert second["summary"]["truth_proj_auroc"] == pytest.approx(0.91)
    assert report["leaderboard"][0]["id"] == "layer_m2_batch_2_capture_outputs"
    assert report["leaderboard"][0]["gate_passed"] is True
    assert report["leaderboard"][1]["gate_passed"] is True
    assert report["matrix_decision"]["status"] == "promote"
    assert report["matrix_decision"]["recommended_cell"] == "layer_m2_batch_2_capture_outputs"
    assert report["matrix_decision"]["checked_cell_count"] == 2
    assert report["matrix_decision"]["unchecked_cells"] == ()


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
    assert saved["matrix_decision"]["status"] == "promote"
    assert saved["matrix_decision"]["recommended_cell"] == "layer_m1_batch_2_capture_outputs"
    assert saved["cells"][0]["summary"]["truth_proj_auroc"] == pytest.approx(0.82)
    assert saved["cells"][0]["summary"]["totals"]["cache_only"]["bottleneck"] == "load_data"


def test_run_cache_profile_matrix_blocks_when_any_checked_cell_fails(tmp_path, monkeypatch):
    module = importlib.import_module("benchmarks.run_cache_profile_matrix")

    def fake_run_triplet(config, *, clean, dry_run):
        config.output_dir.mkdir(parents=True, exist_ok=True)
        comparison_path = config.output_dir / "cache-profile-comparison.json"
        result_path = config.output_dir / "result-cache_only.json"
        gate_passed = config.layer == -1
        total = 9.0 if gate_passed else 20.0
        comparison_path.write_text(
            json.dumps({
                "runs": [
                    {"name": "uncached", "total_seconds": 100.0, "bottleneck": "forward"},
                    {
                        "name": "cache_only",
                        "total_seconds": total,
                        "bottleneck": "score_postprocess",
                        "total_delta": {
                            "speedup_vs_baseline": 100.0 / total,
                            "ratio_to_baseline": total / 100.0,
                        },
                    },
                ],
                "fastest": {"name": "cache_only", "total_seconds": total},
                "regression_gate": {"passed": gate_passed},
            }),
            encoding="utf-8",
        )
        result_path.write_text(json.dumps({"auroc": {"truth_proj": 0.9}}), encoding="utf-8")
        return {
            "dry_run": False,
            "output_dir": str(config.output_dir),
            "comparison_report": str(comparison_path),
            "results": {"cache_only": str(result_path)},
            "regression_gate": {"passed": gate_passed},
        }

    monkeypatch.setattr(module, "run_triplet", fake_run_triplet)
    config = module.CacheProfileMatrixConfig(
        output_dir=tmp_path,
        layers=(-2, -1),
        batch_sizes=(1,),
        hidden_state_captures=("outputs",),
    )

    report = module.run_matrix(config, clean=True, dry_run=False)

    assert report["matrix_decision"]["status"] == "blocked"
    assert report["matrix_decision"]["recommended_cell"] == "layer_m1_batch_1_capture_outputs"
    assert report["matrix_decision"]["failed_cells"] == ("layer_m2_batch_1_capture_outputs",)


def test_run_cache_worker_sweep_builds_dry_run_report(tmp_path):
    module = importlib.import_module("benchmarks.run_cache_worker_sweep")
    config = module.CacheWorkerSweepConfig(
        output_dir=tmp_path / "worker-sweep",
        worker_counts=(1, 2),
        model="tiny-local",
        layers=(-2,),
        batch_sizes=(1,),
        python_executable="/python",
    )

    report = module.run_worker_sweep(config, clean=True, dry_run=True)
    manifest = json.loads(Path(report["artifact_manifest"]).read_text(encoding="utf-8"))

    assert report["dry_run"] is True
    assert Path(report["report_path"]).exists()
    assert [entry["worker_count"] for entry in report["worker_reports"]] == [1, 2]
    assert [entry["matrix_status"] for entry in report["worker_reports"]] == ["dry_run", "dry_run"]
    assert report["worker_sweep_decision"]["status"] == "dry_run"
    assert report["worker_sweep_decision"]["recommended_worker_count"] is None
    assert manifest["metadata"]["runner"] == "run_cache_worker_sweep"
    assert manifest["metadata"]["worker_counts"] == [1, 2]
    assert manifest["artifacts"]["workers.1.matrix_manifest"]["exists"] is True
    assert manifest["artifacts"]["workers.2.matrix_manifest"]["exists"] is True

    with pytest.raises(ValueError, match="duplicate"):
        module.CacheWorkerSweepConfig(output_dir=tmp_path / "bad-duplicate", worker_counts=(1, 1))
    with pytest.raises(ValueError, match=">=1"):
        module.CacheWorkerSweepConfig(output_dir=tmp_path / "bad-zero", worker_counts=(0,))


def test_run_cache_worker_sweep_recommends_fastest_promoted_worker(tmp_path, monkeypatch):
    module = importlib.import_module("benchmarks.run_cache_worker_sweep")
    seen_shared_cache_dirs = []

    def fake_run_matrix(config, *, clean, dry_run):
        del clean, dry_run
        seen_shared_cache_dirs.append(config.shared_cache_dir)
        config.output_dir.mkdir(parents=True, exist_ok=True)
        config.report_path.write_text("{}", encoding="utf-8")
        config.artifact_manifest.write_text("{}", encoding="utf-8")
        wall_clock = 10.0 if config.max_workers == 1 else 6.0
        return {
            "dry_run": False,
            "report_path": str(config.report_path),
            "artifact_manifest": str(config.artifact_manifest),
            "execution": {
                "wall_clock_seconds": wall_clock,
                "cell_count": 1,
                "max_workers": config.max_workers,
            },
            "matrix_decision": {
                "status": "promote",
                "recommended_cell": f"workers_{config.max_workers}_cell",
                "recommended": {
                    "id": f"workers_{config.max_workers}_cell",
                    "cache_only_total_seconds": 0.10,
                    "truth_proj_auroc": 0.90,
                },
                "candidate_count": 1,
                "failed_cells": (),
                "blocking_reasons": (),
            },
        }

    monkeypatch.setattr(module, "run_matrix", fake_run_matrix)
    config = module.CacheWorkerSweepConfig(
        output_dir=tmp_path / "worker-sweep",
        shared_cache_dir=tmp_path / "shared-cache",
        worker_counts=(1, 2),
        model="tiny-local",
    )

    report = module.run_worker_sweep(config, clean=True, dry_run=False)

    assert [path.name for path in seen_shared_cache_dirs] == ["workers_1", "workers_2"]
    assert [entry["worker_count"] for entry in report["leaderboard"]] == [2, 1]
    assert report["worker_sweep_decision"]["status"] == "promote"
    assert report["worker_sweep_decision"]["recommended_worker_count"] == 2
    assert report["worker_sweep_decision"]["recommended"]["wall_clock_seconds"] == pytest.approx(6.0)


def test_runtime_config_recommendation_combines_matrix_and_worker_sweep(tmp_path):
    module = importlib.import_module("benchmarks.recommend_runtime_config")
    result_path = tmp_path / "cache-only-result.json"
    result_path.write_text(
        json.dumps({
            "auroc": {
                "truth_proj": 0.88,
                "subspace_resid": 0.93,
                "nll_answer": 0.72,
            },
        }),
        encoding="utf-8",
    )
    matrix_report = {
        "config": {
            "max_workers": 1,
            "length_bucketed_batches": True,
        },
        "leaderboard_sort_metric": "uncached_forced_answer_forward_seconds",
        "execution": {"wall_clock_seconds": 30.0},
        "prefix_kv_comparisons": [
            {
                "layer": -12,
                "batch_size": 2,
                "hidden_state_capture": "outputs",
                "status": "prefix_kv_slower",
                "recommended_prefix_kv_cache": False,
            }
        ],
        "matrix_decision": {
            "status": "promote",
            "recommended_cell": "layer_m12_batch_2_capture_outputs_token_budget_96_prefix_kv_off",
            "recommendation_metric": "uncached_forced_answer_forward_seconds",
            "candidate_count": 2,
            "checked_cell_count": 2,
            "blocking_reasons": (),
            "recommended": {
                "id": "layer_m12_batch_2_capture_outputs_token_budget_96_prefix_kv_off",
                "layer": -12,
                "batch_size": 2,
                "hidden_state_capture": "outputs",
                "max_batch_tokens": 96,
                "prefix_kv_cache": False,
                "cache_only_total_seconds": 0.25,
                "uncached_forced_answer_forward_seconds": 64.0,
                "truth_proj_auroc": 0.88,
            },
        },
        "cells": [
            {
                "id": "layer_m12_batch_2_capture_outputs_token_budget_96_prefix_kv_off",
                "layer": -12,
                "batch_size": 2,
                "hidden_state_capture": "outputs",
                "summary": {
                    "quality_signals": {
                        "truth_proj": 0.88,
                        "subspace_resid": 0.93,
                    },
                    "truth_proj_auroc": 0.88,
                },
                "triplet": {"results": {"cache_only": str(result_path)}},
            }
        ],
    }
    worker_report = {
        "worker_sweep_decision": {
            "status": "promote",
            "recommended_worker_count": 2,
            "blocking_reasons": (),
            "recommended": {
                "worker_count": 2,
                "wall_clock_seconds": 18.0,
                "matrix_report": str(tmp_path / "matrix-report.json"),
            },
        },
    }

    report = module.build_runtime_recommendation(
        matrix_report,
        worker_sweep_report=worker_report,
        matrix_report_path=tmp_path / "matrix-report.json",
        worker_sweep_report_path=tmp_path / "worker-report.json",
    )

    assert report["status"] == "promote"
    assert report["recommendation"]["layer"] == -12
    assert report["recommendation"]["batch_size"] == 2
    assert report["recommendation"]["max_batch_tokens"] == 96
    assert report["recommendation"]["max_workers"] == 2
    assert report["recommendation"]["quality_signals"] == {
        "nll_answer": pytest.approx(0.72),
        "subspace_resid": pytest.approx(0.93),
        "truth_proj": pytest.approx(0.88),
    }
    assert report["recommendation"]["best_quality_signal"] == {
        "name": "subspace_resid",
        "auroc": pytest.approx(0.93),
    }
    assert report["evidence"]["prefix_kv_comparison"]["status"] == "prefix_kv_slower"
    assert report["evidence"]["quality_signal_source"] == str(result_path)
    assert report["evidence"]["quality_signal_count"] == 3
    assert report["evidence"]["worker_matrix_report_matches"] is True
    assert report["benchmark_flags"]["eval_truthfulqa"] == [
        "--layer",
        "-12",
        "--batch-size",
        "2",
        "--hidden-state-capture",
        "outputs",
        "--max-batch-tokens",
        "96",
        "--length-bucketed-batches",
    ]
    assert report["benchmark_flags"]["run_cache_profile_matrix"] == [
        "--layers",
        "-12",
        "--batch-sizes",
        "2",
        "--hidden-state-captures",
        "outputs",
        "--max-batch-tokens",
        "96",
        "--max-workers",
        "2",
    ]


def test_runtime_config_recommendation_blocks_on_blocked_matrix():
    module = importlib.import_module("benchmarks.recommend_runtime_config")

    report = module.build_runtime_recommendation({
        "matrix_decision": {
            "status": "blocked",
            "recommended_cell": "layer_m12_batch_1_capture_outputs",
            "recommended": {
                "id": "layer_m12_batch_1_capture_outputs",
                "layer": -12,
                "batch_size": 1,
                "hidden_state_capture": "outputs",
            },
            "blocking_reasons": ("one or more checked matrix cells failed the regression gate",),
        }
    })

    assert report["status"] == "blocked"
    assert report["recommendation"] is None
    assert report["blocking_reasons"] == [
        "matrix: one or more checked matrix cells failed the regression gate"
    ]
    assert "benchmark_flags" not in report


def test_runtime_config_recommendation_includes_inside_sampling_profile(tmp_path):
    module = importlib.import_module("benchmarks.recommend_runtime_config")
    inside_result_path = tmp_path / "result-adaptive_selfcheck.json"
    inside_result_path.write_text(
        json.dumps({
            "config": {
                "inside_samples": 5,
                "inside_batch_size": 2,
                "inside_max_new_tokens": 12,
                "inside_temperature": 0.7,
                "inside_top_p": 0.9,
                "inside_pooling": "mean",
                "inside_embedding_threshold": 0.88,
                "inside_adaptive_sampling": True,
                "inside_min_samples": 2,
                "inside_sample_step": 1,
                "inside_stability_delta": 0.03,
                "inside_selfcheck_early_stop": True,
                "inside_selfcheck_min_overlap": 0.65,
                "inside_selfcheck_support_threshold": 0.6,
                "inside_selfcheck_refute_threshold": 0.5,
                "inside_trigger_signal": "truth_proj",
                "inside_trigger_threshold": None,
                "inside_trigger_top_fraction": 0.25,
            },
            "inside_sampling": {
                "mode": "triggered",
                "adaptive": True,
                "selfcheck_early_stop": True,
                "signal": "truth_proj",
                "top_fraction": 0.25,
                "max_samples": 5,
                "min_samples": 2,
                "sample_step": 1,
                "stability_delta": 0.03,
                "embedding_similarity_threshold": 0.88,
                "selfcheck_min_overlap": 0.65,
                "selfcheck_support_threshold": 0.6,
                "selfcheck_refute_threshold": 0.5,
                "total_generated_samples": 8,
                "stop_reason_counts": {"selfcheck_refute_threshold_guaranteed": 4},
            },
        }),
        encoding="utf-8",
    )
    matrix_report = {
        "config": {"max_workers": 1, "length_bucketed_batches": True},
        "matrix_decision": {
            "status": "promote",
            "recommended_cell": "layer_m12_batch_2_capture_outputs",
            "recommendation_metric": "cache_only_total_seconds",
            "blocking_reasons": (),
            "recommended": {
                "id": "layer_m12_batch_2_capture_outputs",
                "layer": -12,
                "batch_size": 2,
                "hidden_state_capture": "outputs",
                "max_batch_tokens": 96,
                "prefix_kv_cache": False,
            },
        },
    }
    inside_sampling_report = {
        "baseline": "fixed",
        "runs": {
            "adaptive_selfcheck": {
                "name": "adaptive_selfcheck",
                "result_path": str(inside_result_path),
                "total_generated_samples": 8,
                "sample_count_ratio_to_baseline": 0.4,
                "inside_generation_seconds": 4.5,
                "inside_generation_seconds_ratio_to_baseline": 0.45,
                "stop_reason_counts": {"selfcheck_refute_threshold_guaranteed": 4},
            }
        },
        "sample_efficiency_gate": {
            "passed": True,
            "failures": [],
            "max_sample_ratios": {"adaptive_selfcheck": 0.6},
        },
        "recommendation": {"recommended_run": "adaptive_selfcheck"},
    }

    report = module.build_runtime_recommendation(
        matrix_report,
        inside_sampling_report=inside_sampling_report,
        inside_sampling_report_path=tmp_path / "inside-sampling-profile-comparison.json",
    )

    inside = report["recommendation"]["inside_sampling"]
    eval_flags = report["benchmark_flags"]["eval_truthfulqa"]
    assert report["status"] == "promote"
    assert inside["recommended_run"] == "adaptive_selfcheck"
    assert inside["inside_samples"] == 5
    assert inside["inside_batch_size"] == 2
    assert inside["sample_count_ratio_to_baseline"] == pytest.approx(0.4)
    assert inside["stop_reason_counts"] == {"selfcheck_refute_threshold_guaranteed": 4}
    assert report["evidence"]["inside_sampling_status"] == "promote"
    assert report["evidence"]["inside_sampling_gate_passed"] is True
    assert "--inside-adaptive-sampling" in eval_flags
    assert "--inside-selfcheck-early-stop" in eval_flags
    assert eval_flags[eval_flags.index("--inside-samples") + 1] == "5"
    assert eval_flags[eval_flags.index("--inside-pooling") + 1] == "mean"
    assert eval_flags[eval_flags.index("--inside-trigger-signal") + 1] == "truth_proj"
    assert eval_flags[eval_flags.index("--inside-trigger-top-fraction") + 1] == "0.25"
    profile_flags = report["benchmark_flags"]["run_inside_sampling_profile"]
    assert profile_flags[profile_flags.index("--inside-trigger-signal") + 1] == "truth_proj"
    assert profile_flags[profile_flags.index("--inside-trigger-top-fraction") + 1] == "0.25"
    assert report["benchmark_flags"]["run_inside_sampling_profile"][-2:] == [
        "--runs",
        "adaptive_selfcheck",
    ]
    assert "--inside-adaptive-sampling" not in report["benchmark_flags"]["run_inside_sampling_profile"]
    assert "--inside-selfcheck-early-stop" not in report["benchmark_flags"]["run_inside_sampling_profile"]


def test_runtime_config_recommendation_includes_derived_trigger_budget_sweep(tmp_path):
    module = importlib.import_module("benchmarks.recommend_runtime_config")
    matrix_report = {
        "config": {"max_workers": 1, "length_bucketed_batches": True},
        "matrix_decision": {
            "status": "promote",
            "recommended_cell": "layer_m12_batch_2_capture_outputs",
            "recommendation_metric": "cache_only_total_seconds",
            "blocking_reasons": (),
            "recommended": {
                "id": "layer_m12_batch_2_capture_outputs",
                "layer": -12,
                "batch_size": 2,
                "hidden_state_capture": "outputs",
                "max_batch_tokens": 96,
                "prefix_kv_cache": False,
            },
        },
    }
    sweep_report = {
        "workflow": "inside_trigger_budget_sweep",
        "dry_run": False,
        "derived_from_max_budget": True,
        "derived_source_budget_id": "top_0p4",
        "derived_source_score_dump": str(tmp_path / "scores.json"),
        "config": {
            "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
            "dtype": "float32",
            "layer": -20,
            "batch_size": 1,
            "max_batch_tokens": 128,
            "max_length": 64,
            "hidden_state_capture": "outputs",
            "progress_every": 10,
            "offline": True,
            "length_bucketed_batches": True,
            "trigger_signal": "truth_proj",
            "budgets": [
                {"kind": "top_fraction", "value": 0.1, "id": "top_0p1"},
                {"kind": "top_fraction", "value": 0.4, "id": "top_0p4"},
            ],
            "inside_samples": 5,
            "inside_batch_size": 2,
            "inside_max_new_tokens": 12,
            "inside_temperature": 0.7,
            "inside_top_p": 0.9,
            "inside_pooling": "mean",
            "inside_embedding_threshold": 0.88,
            "inside_min_samples": 2,
            "inside_sample_step": 1,
            "inside_stability_delta": 0.03,
            "inside_selfcheck_min_overlap": 0.65,
            "inside_selfcheck_support_threshold": 0.6,
            "inside_selfcheck_refute_threshold": 0.5,
            "run_names": ["adaptive_selfcheck"],
            "shared_cache_dir": str(tmp_path / "shared-caches"),
            "eval_reps_cache_shard_size": 4,
            "refresh_shared_caches": True,
            "derive_from_max_budget": True,
        },
        "budgets": {
            "top_0p1": {"sample_efficiency_gate": {"passed": True}},
            "top_0p4": {"sample_efficiency_gate": {"passed": True}},
        },
        "leaderboard": [
            {
                "budget_id": "top_0p1",
                "budget_kind": "top_fraction",
                "budget_value": 0.1,
                "recommended_run": "adaptive_selfcheck",
                "derived": True,
                "derived_from_budget_id": "top_0p4",
                "inside_generation_seconds_source": "sample_count_ratio_estimate",
                "sampled": 55,
                "skipped_by_trigger": 445,
                "total_generated_samples": 55,
                "mean_samples_per_record": 0.11,
                "inside_generation_seconds": 59.0,
                "sample_count_ratio_to_reference": 0.13,
                "inside_generation_seconds_ratio_to_reference": 0.13,
                "inside_auroc": {"inside_semantic_entropy": 0.49},
            },
            {
                "budget_id": "top_0p4",
                "budget_kind": "top_fraction",
                "budget_value": 0.4,
                "recommended_run": "adaptive_selfcheck",
                "derived": True,
                "derived_from_budget_id": "top_0p4",
                "source_score_dump": str(tmp_path / "scores.json"),
                "inside_generation_seconds_source": "measured_source_run",
                "sampled": 218,
                "skipped_by_trigger": 282,
                "total_generated_samples": 218,
                "mean_samples_per_record": 0.436,
                "mean_samples_per_sampled_record": 1.0,
                "inside_generation_seconds": 235.0,
                "sample_count_ratio_to_reference": 0.50,
                "inside_generation_seconds_ratio_to_reference": 0.50,
                "inside_auroc": {"inside_semantic_entropy": 0.57},
                "stop_reason_counts": {"selfcheck_supported": 12},
            },
        ],
        "recommendation": {
            "budget_id": "top_0p1",
            "recommended_run": "adaptive_selfcheck",
            "reason": "lowest_total_generated_samples_then_inside_generation_seconds",
        },
        "quality_balanced_recommendation": {
            "budget_id": "top_0p4",
            "recommended_run": "adaptive_selfcheck",
            "reason": "lowest_cost_within_inside_quality_tolerance",
            "quality_metric": "inside_semantic_entropy",
            "quality_value": 0.57,
            "best_quality_value": 0.57,
            "quality_tolerance": 0.02,
            "cost_metric": "inside_generation_seconds_ratio_to_reference",
            "cost_value": 0.50,
        },
    }

    report = module.build_runtime_recommendation(
        matrix_report,
        inside_trigger_budget_sweep_report=sweep_report,
        inside_trigger_budget_sweep_report_path=tmp_path / "inside-trigger-budget-sweep.json",
    )

    assert report["status"] == "promote"
    trigger = report["recommendation"]["inside_trigger_budget_sweep"]
    inside = report["recommendation"]["inside_sampling"]
    assert trigger["recommendation_source"] == "quality_balanced_recommendation"
    assert trigger["selection_policy"] == "quality_balanced"
    assert trigger["recommended_budget_id"] == "top_0p4"
    assert trigger["derive_from_max_budget"] is True
    assert trigger["sample_count_ratio_to_reference"] == pytest.approx(0.50)
    assert inside["inside_trigger_top_fraction"] == pytest.approx(0.4)
    assert inside["inside_trigger_budget_id"] == "top_0p4"
    assert inside["inside_trigger_budget_policy"] == "quality_balanced"
    assert inside["adaptive"] is True
    assert inside["selfcheck_early_stop"] is True
    assert inside["inside_generation_seconds_source"] == "measured_source_run"
    assert report["evidence"]["inside_trigger_budget_sweep_status"] == "promote"
    assert report["evidence"]["inside_trigger_budget_derive_from_max_budget"] is True
    eval_flags = report["benchmark_flags"]["eval_truthfulqa"]
    assert eval_flags[eval_flags.index("--inside-trigger-top-fraction") + 1] == "0.4"
    assert "--inside-adaptive-sampling" in eval_flags
    sweep_flags = report["benchmark_flags"]["run_inside_trigger_budget_sweep"]
    assert sweep_flags[sweep_flags.index("--top-fractions") + 1] == "0.1,0.4"
    assert sweep_flags[sweep_flags.index("--runs") + 1] == "adaptive_selfcheck"
    assert "--derive-from-max-budget" in sweep_flags


def test_runtime_config_recommendation_selects_trigger_budget_policy():
    module = importlib.import_module("benchmarks.recommend_runtime_config")
    matrix_report = {
        "config": {"max_workers": 1},
        "matrix_decision": {
            "status": "promote",
            "recommended_cell": "layer_m12_batch_2_capture_outputs",
            "blocking_reasons": (),
            "recommended": {
                "id": "layer_m12_batch_2_capture_outputs",
                "layer": -12,
                "batch_size": 2,
                "hidden_state_capture": "outputs",
            },
        },
    }
    sweep_report = {
        "workflow": "inside_trigger_budget_sweep",
        "dry_run": False,
        "config": {
            "trigger_signal": "truth_proj",
            "budgets": [
                {"kind": "top_fraction", "value": 0.1, "id": "top_0p1"},
                {"kind": "top_fraction", "value": 0.4, "id": "top_0p4"},
                {"kind": "top_fraction", "value": 0.6, "id": "top_0p6"},
            ],
            "inside_samples": 3,
            "inside_batch_size": 1,
            "inside_max_new_tokens": 4,
            "run_names": ["adaptive_selfcheck"],
        },
        "budgets": {
            "top_0p1": {"sample_efficiency_gate": {"passed": True}},
            "top_0p4": {"sample_efficiency_gate": {"passed": True}},
            "top_0p6": {"sample_efficiency_gate": {"passed": True}},
        },
        "leaderboard": [
            {
                "budget_id": "top_0p1",
                "budget_kind": "top_fraction",
                "budget_value": 0.1,
                "recommended_run": "adaptive_selfcheck",
                "total_generated_samples": 30,
                "sample_count_ratio_to_reference": 0.10,
                "inside_generation_seconds_ratio_to_reference": 0.11,
                "inside_auroc": {"inside_semantic_entropy": 0.52},
            },
            {
                "budget_id": "top_0p4",
                "budget_kind": "top_fraction",
                "budget_value": 0.4,
                "recommended_run": "adaptive_selfcheck",
                "total_generated_samples": 120,
                "sample_count_ratio_to_reference": 0.40,
                "inside_generation_seconds_ratio_to_reference": 0.42,
                "inside_auroc": {"inside_semantic_entropy": 0.57},
            },
            {
                "budget_id": "top_0p6",
                "budget_kind": "top_fraction",
                "budget_value": 0.6,
                "recommended_run": "adaptive_selfcheck",
                "total_generated_samples": 180,
                "sample_count_ratio_to_reference": 0.60,
                "inside_generation_seconds_ratio_to_reference": 0.63,
                "inside_auroc": {"inside_semantic_entropy": 0.59},
            },
        ],
        "recommendation": {
            "budget_id": "top_0p1",
            "recommended_run": "adaptive_selfcheck",
            "reason": "lowest_total_generated_samples_then_inside_generation_seconds",
        },
        "quality_balanced_recommendation": {
            "budget_id": "top_0p4",
            "recommended_run": "adaptive_selfcheck",
            "reason": "lowest_cost_within_inside_quality_tolerance",
            "quality_metric": "inside_semantic_entropy",
            "quality_value": 0.57,
            "best_quality_value": 0.59,
            "quality_tolerance": 0.02,
            "cost_metric": "inside_generation_seconds_ratio_to_reference",
            "cost_value": 0.42,
        },
    }

    default_report = module.build_runtime_recommendation(
        matrix_report,
        inside_trigger_budget_sweep_report=sweep_report,
    )
    cost_report = module.build_runtime_recommendation(
        matrix_report,
        inside_trigger_budget_sweep_report=sweep_report,
        inside_trigger_budget_policy="cost_first",
    )
    quality_report = module.build_runtime_recommendation(
        matrix_report,
        inside_trigger_budget_sweep_report=sweep_report,
        inside_trigger_budget_policy="quality_first",
    )

    assert default_report["recommendation"]["inside_trigger_budget_sweep"]["recommended_budget_id"] == "top_0p4"
    assert default_report["evidence"]["inside_trigger_budget_policy"] == "quality_balanced"
    assert cost_report["recommendation"]["inside_trigger_budget_sweep"]["recommended_budget_id"] == "top_0p1"
    assert cost_report["recommendation"]["inside_trigger_budget_sweep"]["recommendation_source"] == "recommendation"
    assert cost_report["recommendation"]["inside_sampling"]["inside_trigger_top_fraction"] == pytest.approx(0.1)
    assert cost_report["evidence"]["inside_trigger_budget_policy"] == "cost_first"
    quality_trigger = quality_report["recommendation"]["inside_trigger_budget_sweep"]
    assert quality_trigger["recommended_budget_id"] == "top_0p6"
    assert quality_trigger["recommendation_source"] == "quality_first"
    assert quality_trigger["selection_policy"] == "quality_first"
    assert quality_trigger["quality_metric"] == "inside_semantic_entropy"
    assert quality_trigger["quality_value"] == pytest.approx(0.59)
    assert quality_trigger["cost_value"] == pytest.approx(0.63)
    eval_flags = quality_report["benchmark_flags"]["eval_truthfulqa"]
    assert eval_flags[eval_flags.index("--inside-trigger-top-fraction") + 1] == "0.6"
    with pytest.raises(ValueError, match="inside_trigger_budget_policy"):
        module.build_runtime_recommendation(
            matrix_report,
            inside_trigger_budget_sweep_report=sweep_report,
            inside_trigger_budget_policy="fast_enough",
        )


def test_runtime_config_recommendation_blocks_failed_inside_sampling_gate():
    module = importlib.import_module("benchmarks.recommend_runtime_config")

    report = module.build_runtime_recommendation(
        {
            "matrix_decision": {
                "status": "promote",
                "recommended_cell": "layer_m1_batch_1_capture_outputs",
                "blocking_reasons": (),
                "recommended": {
                    "id": "layer_m1_batch_1_capture_outputs",
                    "layer": -1,
                    "batch_size": 1,
                    "hidden_state_capture": "outputs",
                },
            }
        },
        inside_sampling_report={
            "sample_efficiency_gate": {
                "passed": False,
                "failures": [{
                    "run": "adaptive_selfcheck",
                    "metric": "sample_count_ratio_to_baseline",
                    "value": 1.2,
                    "max_allowed": 0.8,
                }],
            },
            "recommendation": {"recommended_run": "adaptive_selfcheck"},
        },
    )

    assert report["status"] == "blocked"
    assert report["recommendation"] is None
    assert report["blocking_reasons"] == [
        "inside_sampling: adaptive_selfcheck failed sample_count_ratio_to_baseline gate "
        "(value=1.2, max_allowed=0.8)"
    ]
    assert "benchmark_flags" not in report


def test_runtime_config_recommendation_cli_writes_output(tmp_path):
    module = importlib.import_module("benchmarks.recommend_runtime_config")
    matrix_report_path = tmp_path / "matrix-report.json"
    output_path = tmp_path / "runtime-recommendation.json"
    matrix_report_path.write_text(
        json.dumps({
            "config": {"max_workers": 1, "length_bucketed_batches": False},
            "matrix_decision": {
                "status": "promote",
                "recommended_cell": "layer_m1_batch_1_capture_outputs",
                "recommendation_metric": "cache_only_total_seconds",
                "blocking_reasons": (),
                "recommended": {
                    "id": "layer_m1_batch_1_capture_outputs",
                    "layer": -1,
                    "batch_size": 1,
                    "hidden_state_capture": "outputs",
                    "max_batch_tokens": 0,
                    "prefix_kv_cache": True,
                },
            },
        }),
        encoding="utf-8",
    )

    payload = module.run(SimpleNamespace(
        matrix_report=str(matrix_report_path),
        worker_sweep_report=None,
        inside_sampling_report=None,
        inside_trigger_budget_sweep_report=None,
        inside_trigger_budget_policy="quality_balanced",
        output=str(output_path),
        fail_on_blocked=True,
    ))

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["status"] == "promote"
    assert saved["recommendation"]["prefix_kv_cache"] is True
    assert saved["benchmark_flags"]["eval_truthfulqa"][-1] == "--prefix-kv-cache"


def test_run_performance_baseline_workflow_reuses_reports_and_registers(tmp_path):
    module = importlib.import_module("benchmarks.run_performance_baseline_workflow")
    registry_module = importlib.import_module("eigentruth.registry")
    result_path = tmp_path / "cache-only-result.json"
    matrix_manifest_path = tmp_path / "matrix-artifact-manifest.json"
    matrix_report_path = tmp_path / "cache-profile-matrix-report.json"
    workflow_report_path = tmp_path / "workflow.json"
    registry_path = tmp_path / "registry.json"
    result_path.write_text(
        json.dumps({
            "auroc": {
                "truth_proj": 0.86,
                "subspace_resid": 0.94,
            },
        }),
        encoding="utf-8",
    )
    matrix_manifest_path.write_text("{}", encoding="utf-8")
    matrix_report_path.write_text(
        json.dumps({
            "artifact_manifest": str(matrix_manifest_path),
            "report_path": str(matrix_report_path),
            "config": {"max_workers": 1, "length_bucketed_batches": True},
            "matrix_decision": {
                "status": "promote",
                "recommended_cell": "layer_m12_batch_2_capture_outputs",
                "recommendation_metric": "cache_only_total_seconds",
                "blocking_reasons": (),
                "recommended": {
                    "id": "layer_m12_batch_2_capture_outputs",
                    "layer": -12,
                    "batch_size": 2,
                    "hidden_state_capture": "outputs",
                    "max_batch_tokens": 128,
                    "prefix_kv_cache": False,
                    "cache_only_total_seconds": 0.2,
                    "truth_proj_auroc": 0.86,
                },
            },
            "cells": [
                {
                    "id": "layer_m12_batch_2_capture_outputs",
                    "layer": -12,
                    "batch_size": 2,
                    "hidden_state_capture": "outputs",
                    "triplet": {"results": {"cache_only": str(result_path)}},
                }
            ],
        }),
        encoding="utf-8",
    )

    payload = module.run_performance_baseline_workflow(
        module.PerformanceBaselineWorkflowConfig(
            output_dir=tmp_path / "workflow",
            report_path=workflow_report_path,
            registry_path=registry_path,
            name="qwen05-local-performance",
            version="0.1",
            matrix_report_path=matrix_report_path,
        )
    )
    saved = json.loads(workflow_report_path.read_text(encoding="utf-8"))
    manifest = json.loads(Path(payload["paths"]["artifact_manifest"]).read_text(encoding="utf-8"))
    registry = registry_module.ArtifactRegistry.load_json(registry_path)
    record = registry.get("performance_baseline:qwen05-local-performance:0.1")

    assert payload["status"] == "promote"
    assert payload["decision"]["recommended_cell"] == "layer_m12_batch_2_capture_outputs"
    assert payload["decision"]["recommended_layer"] == -12
    assert payload["decision"]["recommended_batch_size"] == 2
    assert payload["decision"]["recommended_best_quality_signal"] == "subspace_resid"
    assert payload["runtime_recommendation"]["recommendation"]["best_quality_signal"] == {
        "name": "subspace_resid",
        "auroc": pytest.approx(0.94),
    }
    assert saved["registry_record"] == "performance_baseline:qwen05-local-performance:0.1"
    assert manifest["metadata"]["runner"] == "run_performance_baseline_workflow"
    assert manifest["metadata"]["matrix_report_reused"] is True
    assert manifest["artifacts"]["performance_baseline_report"]["exists"] is True
    assert manifest["artifacts"]["runtime_recommendation"]["exists"] is True
    assert manifest["artifacts"]["matrix_report"]["exists"] is True
    assert record.artifact_type == "performance_baseline"
    assert record.path == str(workflow_report_path)
    assert record.metadata["runtime_recommendation_status"] == "promote"
    assert record.metadata["recommended_best_quality_signal"] == "subspace_resid"


def test_run_performance_baseline_workflow_dry_run_needs_evidence(tmp_path):
    module = importlib.import_module("benchmarks.run_performance_baseline_workflow")

    payload = module.run_performance_baseline_workflow(
        module.PerformanceBaselineWorkflowConfig(
            output_dir=tmp_path,
            model="tiny-local",
            layers=(-1,),
            batch_sizes=(1,),
            run_worker_sweep=True,
            worker_counts=(1, 2),
            dry_run=True,
        )
    )
    manifest = json.loads(Path(payload["paths"]["artifact_manifest"]).read_text(encoding="utf-8"))

    assert payload["status"] == "needs_evidence"
    assert payload["runtime_recommendation"]["status"] == "needs_evidence"
    assert payload["execution"]["matrix_report_reused"] is False
    assert payload["execution"]["worker_sweep_report_reused"] is False
    assert manifest["metadata"]["dry_run"] is True
    assert manifest["metadata"]["worker_sweep_enabled"] is True
    assert Path(payload["paths"]["runtime_recommendation"]).exists()
    assert Path(payload["paths"]["matrix_report"]).exists()
    assert Path(payload["paths"]["worker_sweep_report"]).exists()


def test_run_product_runtime_baseline_aggregates_traces_and_registers(tmp_path):
    module = importlib.import_module("benchmarks.run_product_runtime_baseline")
    registry_module = importlib.import_module("eigentruth.registry")
    trace_a = tmp_path / "trace-a.json"
    trace_b = tmp_path / "trace-b.json"
    report_path = tmp_path / "product-runtime-baseline.json"
    registry_path = tmp_path / "registry.json"
    trace_a.write_text(
        json.dumps({
            "request_id": "req-a",
            "runtime_trace": {
                "total_seconds": 0.10,
                "phases": [
                    {"name": "diagnostic_risk_decision", "seconds": 0.02},
                    {"name": "initial_verification", "seconds": 0.03},
                ],
            },
            "verification_results": [
                {
                    "status": "supported",
                    "metadata": {
                        "selected_route": "structured_qa",
                        "total_duration_seconds": 0.01,
                        "selected_route_duration_seconds": 0.01,
                        "attempted_route_count": 1,
                        "used_retrieval": False,
                    },
                }
            ],
            "metadata": {"cache": {"verifier": {"hits": 1, "misses": 1}}},
        }),
        encoding="utf-8",
    )
    trace_b.write_text(
        json.dumps({
            "request_id": "req-b",
            "runtime_trace": {
                "total_seconds": 0.18,
                "phases": [
                    {"name": "diagnostic_risk_decision", "seconds": 0.02},
                    {"name": "initial_verification", "seconds": 0.12},
                ],
            },
            "verification_results": [
                {
                    "status": "refuted",
                    "metadata": {
                        "selected_route": "retrieval_structured_qa",
                        "total_duration_seconds": 0.04,
                        "selected_route_duration_seconds": 0.03,
                        "attempted_route_count": 2,
                        "used_retrieval": True,
                        "retrieval_hit_count": 1,
                    },
                }
            ],
            "metadata": {"cache": {"verifier": {"hits": 2, "misses": 0}}},
        }),
        encoding="utf-8",
    )

    payload = module.build_product_runtime_baseline(
        module.ProductRuntimeBaselineConfig(
            trace_paths=(trace_a, trace_b),
            report_path=report_path,
            registry_path=registry_path,
            name="demo-product-runtime",
            version="0.1",
            policy={
                "max_total_seconds": 0.2,
                "max_mean_attempted_route_count": 2.0,
                "max_retrieval_use_rate": 1.0,
            },
            metadata={"suite": "unit"},
            compact_json=True,
        )
    )
    saved_text = report_path.read_text(encoding="utf-8")
    manifest_text = Path(payload["paths"]["artifact_manifest"]).read_text(encoding="utf-8")
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = json.loads(Path(payload["paths"]["artifact_manifest"]).read_text(encoding="utf-8"))
    record = registry_module.ArtifactRegistry.load_json(registry_path).get(
        "product_runtime_baseline:demo-product-runtime:0.1"
    )

    assert payload["status"] == "promote"
    assert payload["config"]["compact_json"] is True
    assert payload["budget"]["passed_count"] == 2
    assert payload["summary"]["n_traces"] == 2
    assert payload["summary"]["total_seconds"]["mean"] == pytest.approx(0.14)
    assert payload["summary"]["phases"]["initial_verification"]["phase_count"] == 2
    assert payload["summary"]["routes"]["overall"]["total"] == 2
    assert payload["summary"]["routes"]["overall"]["retrieval_use_rate"] == pytest.approx(0.5)
    assert payload["summary"]["routes"]["by_route"]["retrieval_structured_qa"]["retrieval_use_rate"] == pytest.approx(
        1.0
    )
    assert saved["artifact_manifest_summary"] == manifest["summary"]
    assert saved["artifact_manifest_summary"]["artifact_count"] == 3
    assert manifest["metadata"]["runner"] == "run_product_runtime_baseline"
    assert manifest["metadata"]["budget_passed"] is True
    assert manifest["metadata"]["compact_json"] is True
    assert "\n  " not in saved_text
    assert "\n  " not in manifest_text
    assert registry_module.load_and_verify_artifact_manifest(
        payload["paths"]["artifact_manifest"]
    ).passed is True
    assert record.artifact_type == "product_runtime_baseline"
    assert record.metadata["status"] == "promote"
    assert record.metadata["trace_count"] == 2
    assert record.metadata["compact_json"] is True


def test_run_product_runtime_baseline_blocks_on_budget_failure(tmp_path):
    module = importlib.import_module("benchmarks.run_product_runtime_baseline")
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(
        json.dumps({
            "request_id": "slow-req",
            "runtime_trace": {
                "total_seconds": 0.25,
                "phases": [{"name": "initial_verification", "seconds": 0.22}],
            },
            "verification_results": [],
            "metadata": {},
        }),
        encoding="utf-8",
    )

    payload = module.build_product_runtime_baseline(
        module.ProductRuntimeBaselineConfig(
            trace_paths=(trace_path,),
            report_path=tmp_path / "product-runtime-baseline.json",
            policy={"max_total_seconds": 0.1},
        )
    )

    assert payload["status"] == "blocked"
    assert payload["budget"]["failed_count"] == 1
    assert payload["budget"]["failure_counts_by_metric"] == {"total_seconds": 1}
    assert payload["decision"]["blocking_reasons"] == ("total_seconds: failed 1 trace(s)",)


def test_run_product_runtime_baseline_aggregates_verification_stage_savings(tmp_path):
    module = importlib.import_module("benchmarks.run_product_runtime_baseline")
    skipped_trace = tmp_path / "skipped-trace.json"
    verified_trace = tmp_path / "verified-trace.json"
    skipped_trace.write_text(
        json.dumps({
            "request_id": "skip",
            "claims": [
                {"claim_id": "c1", "text": "Paris is the capital of France."},
                {"claim_id": "c2", "text": "Lyon is in France."},
            ],
            "events": [
                {
                    "event_type": "verification_stage_decision",
                    "payload": {
                        "run_verifier": False,
                        "reason": "diagnostics and claim metadata did not require verification",
                    },
                },
                {
                    "event_type": "initial_verification",
                    "payload": {"n_claims": 2, "skipped": True, "results": []},
                },
            ],
            "verification_results": [],
            "metadata": {"staged_verification_enabled": True},
        }),
        encoding="utf-8",
    )
    verified_trace.write_text(
        json.dumps({
            "request_id": "verify",
            "claims": [{"claim_id": "c1", "text": "2 + 2 = 4."}],
            "events": [
                {
                    "event_type": "verification_stage_decision",
                    "payload": {
                        "run_verifier": True,
                        "reason": "diagnostic risk level is medium",
                        "triggered_claim_ids": ["c1"],
                        "triggered_features": {"c1": ["has_number"]},
                    },
                },
                {
                    "event_type": "initial_verification",
                    "payload": {
                        "n_claims": 1,
                        "skipped": False,
                        "results": [{"status": "supported", "metadata": {}}],
                    },
                },
            ],
            "verification_results": [{"status": "supported", "metadata": {}}],
            "metadata": {"staged_verification_enabled": True},
        }),
        encoding="utf-8",
    )

    payload = module.build_product_runtime_baseline(
        module.ProductRuntimeBaselineConfig(
            trace_paths=(skipped_trace, verified_trace),
            report_path=tmp_path / "product-runtime-baseline.json",
            policy={
                "min_verification_skip_rate": 0.0,
                "max_verified_claim_count": 1,
            },
        )
    )

    stage = payload["summary"]["verification_stage"]
    assert payload["status"] == "promote"
    assert payload["budget"]["passed_count"] == 2
    assert payload["traces"][0]["metrics"]["verification_skip_rate"] == 1.0
    assert payload["traces"][1]["metrics"]["verified_claim_count"] == 1
    assert payload["summary"]["verification_skip_rate"]["mean"] == pytest.approx(0.5)
    assert stage["enabled_trace_count"] == 2
    assert stage["skipped_trace_count"] == 1
    assert stage["saved_claim_count"] == pytest.approx(2.0)
    assert stage["verified_claim_count"] == pytest.approx(1.0)
    assert stage["claim_skip_rate"] == pytest.approx(2 / 3)
    assert stage["triggered_feature_counts"] == {"has_number": 1}
    assert stage["reason_counts"]["diagnostic risk level is medium"] == 1


def test_run_product_runtime_profile_sweep_compares_profiles_and_registers(tmp_path):
    module = importlib.import_module("benchmarks.run_product_runtime_profile_sweep")
    registry_module = importlib.import_module("eigentruth.registry")
    output_dir = tmp_path / "profile-sweep"
    registry_path = tmp_path / "registry.json"
    selector_policy_path = tmp_path / "selector-policy.json"
    selector_policy_path.write_text(
        json.dumps({
            "sensitive_claim_feature_flags": ["has_citation"],
            "sensitive_claim_metadata_keys": ["requires_review"],
        }),
        encoding="utf-8",
    )

    payload = module.run_product_runtime_profile_sweep(
        module.ProductRuntimeProfileSweepConfig(
            output_dir=output_dir,
            profiles=("latency", "auto", "audit"),
            scenarios=(
                module.ProductRuntimeScenario(
                    name="low",
                    text="Paris is the capital of France.",
                    diagnostics_mode="low",
                    facts={"Paris is the capital of France": "supported"},
                ),
            ),
            repeats=1,
            runtime_profile_selector_policy_path=selector_policy_path,
            registry_path=registry_path,
            name="demo-profile-sweep",
            version="0.1",
            metadata={"suite": "unit"},
            max_workers=2,
            compact_json=True,
        )
    )
    report_text = Path(payload["paths"]["report"]).read_text(encoding="utf-8")
    latency_trace = json.loads(
        Path(payload["profiles"][0]["trace_paths"][0]).read_text(encoding="utf-8")
    )
    audit_trace = json.loads(
        Path(payload["profiles"][2]["trace_paths"][0]).read_text(encoding="utf-8")
    )
    record = registry_module.ArtifactRegistry.load_json(registry_path).get(
        "report:demo-profile-sweep:0.1"
    )

    assert payload["status"] == "observed"
    assert payload["config"]["max_workers"] == 2
    assert payload["config"]["compact_json"] is True
    assert payload["config"]["runtime_profile_selector_policy_path"] == str(selector_policy_path)
    assert payload["paths"]["runtime_profile_selector_policy"] == str(selector_policy_path)
    assert payload["execution"]["max_workers"] == 2
    assert payload["decision"]["recommended_profile"] in {"latency", "auto", "audit"}
    assert {row["profile"] for row in payload["leaderboard"]} == {"latency", "auto", "audit"}
    assert payload["profiles"][0]["status"] == "observed"
    assert payload["profiles"][1]["status"] == "observed"
    assert payload["profiles"][2]["status"] == "observed"
    assert Path(payload["profiles"][0]["baseline_artifact_manifest"]).exists()
    assert Path(payload["profiles"][1]["baseline_artifact_manifest"]).exists()
    assert Path(payload["profiles"][2]["baseline_artifact_manifest"]).exists()
    assert latency_trace["metadata"]["runtime_profile"] == "latency"
    auto_trace = json.loads(
        Path(payload["profiles"][1]["trace_paths"][0]).read_text(encoding="utf-8")
    )
    assert auto_trace["metadata"]["runtime_profile"] == "latency"
    assert auto_trace["metadata"]["runtime_profile_selection"]["selected_profile"] == "latency"
    assert auto_trace["metadata"]["runtime_profile_selector_policy"]["sensitive_claim_feature_flags"] == [
        "has_citation"
    ]
    assert payload["profiles"][1]["runtime_profile_selection"]["counts_by_selected_profile"] == {"latency": 1}
    assert audit_trace["metadata"]["runtime_profile"] == "audit"
    assert "initial_verification" not in {
        phase["name"] for phase in latency_trace["runtime_trace"]["phases"]
    }
    assert "initial_verification" in {
        phase["name"] for phase in audit_trace["runtime_trace"]["phases"]
    }
    assert Path(payload["paths"]["artifact_manifest"]).exists()
    manifest = json.loads(Path(payload["paths"]["artifact_manifest"]).read_text(encoding="utf-8"))
    assert "latency_baseline_manifest" in manifest["artifacts"]
    assert "auto_baseline_manifest" in manifest["artifacts"]
    assert "audit_baseline_manifest" in manifest["artifacts"]
    assert manifest["metadata"]["compact_json"] is True
    assert "\n  " not in report_text
    assert "\n  " not in Path(payload["profiles"][0]["trace_paths"][0]).read_text(encoding="utf-8")
    assert "\n  " not in Path(payload["profiles"][0]["baseline_path"]).read_text(encoding="utf-8")
    assert registry_module.load_and_verify_artifact_manifest(
        payload["paths"]["artifact_manifest"]
    ).passed is True
    assert record.metadata["status"] == "observed"
    assert record.metadata["profile_count"] == 3
    assert record.metadata["compact_json"] is True
    assert record.metadata["runtime_profile_selector_policy"] == str(selector_policy_path)


def test_run_product_runtime_profile_sweep_rejects_invalid_workers(tmp_path):
    module = importlib.import_module("benchmarks.run_product_runtime_profile_sweep")

    with pytest.raises(ValueError, match="max_workers"):
        module.ProductRuntimeProfileSweepConfig(
            output_dir=tmp_path / "profile-sweep",
            max_workers=0,
        )


def test_run_product_runtime_profile_sweep_blocks_when_policy_fails(tmp_path):
    module = importlib.import_module("benchmarks.run_product_runtime_profile_sweep")

    payload = module.run_product_runtime_profile_sweep(
        module.ProductRuntimeProfileSweepConfig(
            output_dir=tmp_path / "profile-sweep",
            profiles=("latency",),
            scenarios=(
                module.ProductRuntimeScenario(
                    name="low",
                    text="Paris is the capital of France.",
                    diagnostics_mode="low",
                    facts={"Paris is the capital of France": "supported"},
                ),
            ),
            policy={"max_total_seconds": 0.0},
        )
    )

    assert payload["status"] == "blocked"
    assert payload["profiles"][0]["status"] == "blocked"
    assert payload["profiles"][0]["budget"]["failed_count"] == 1
    assert payload["decision"]["blocking_reasons"] == ("latency.total_seconds: failed 1 trace(s)",)


def test_run_product_runtime_profile_sweep_applies_slo_policy(tmp_path):
    module = importlib.import_module("benchmarks.run_product_runtime_profile_sweep")

    payload = module.run_product_runtime_profile_sweep(
        module.ProductRuntimeProfileSweepConfig(
            output_dir=tmp_path / "profile-sweep",
            profiles=("auto",),
            scenarios=module.DEFAULT_SCENARIOS,
            slo_policy={
                "max_total_seconds_p95": 1.0,
                "max_mean_attempted_route_count": 1.1,
                "max_retrieval_use_rate": 0.0,
                "min_auto_selected_profile_counts": {
                    "latency": 1,
                    "balanced": 1,
                    "audit": 1,
                },
            },
        )
    )

    profile = payload["profiles"][0]
    assert payload["status"] == "promote"
    assert payload["slo"]["enabled"] is True
    assert payload["slo"]["passed"] is True
    assert payload["decision"]["recommended_profile"] == "auto"
    assert profile["baseline_status"] == "observed"
    assert profile["status"] == "promote"
    assert profile["slo"]["passed"] is True
    assert profile["runtime_profile_selection"]["counts_by_selected_profile"] == {
        "audit": 1,
        "balanced": 1,
        "latency": 1,
    }
    assert {
        check["metric"]
        for check in profile["slo"]["checks"]
    } >= {
        "total_seconds_p95",
        "mean_attempted_route_count",
        "retrieval_use_rate",
        "auto_selected_profile_count.latency",
        "auto_selected_profile_count.balanced",
        "auto_selected_profile_count.audit",
    }


def test_run_product_runtime_profile_sweep_blocks_when_slo_policy_fails(tmp_path):
    module = importlib.import_module("benchmarks.run_product_runtime_profile_sweep")

    payload = module.run_product_runtime_profile_sweep(
        module.ProductRuntimeProfileSweepConfig(
            output_dir=tmp_path / "profile-sweep",
            profiles=("auto",),
            scenarios=(
                module.ProductRuntimeScenario(
                    name="low",
                    text="Paris is the capital of France.",
                    diagnostics_mode="low",
                    facts={"Paris is the capital of France": "supported"},
                ),
            ),
            slo_policy={
                "min_auto_selected_profile_counts": {
                    "audit": 1,
                },
            },
        )
    )

    assert payload["status"] == "blocked"
    assert payload["slo"]["passed"] is False
    assert payload["profiles"][0]["status"] == "blocked"
    assert payload["profiles"][0]["slo"]["failures"][0]["metric"] == "auto_selected_profile_count.audit"
    assert payload["decision"]["blocking_reasons"] == (
        "auto.auto_selected_profile_count.audit: SLO min 1.0 failed",
    )


def test_run_product_runtime_profile_sweep_requires_auto_profile_for_auto_slo(tmp_path):
    module = importlib.import_module("benchmarks.run_product_runtime_profile_sweep")

    payload = module.run_product_runtime_profile_sweep(
        module.ProductRuntimeProfileSweepConfig(
            output_dir=tmp_path / "profile-sweep",
            profiles=("latency",),
            scenarios=(
                module.ProductRuntimeScenario(
                    name="low",
                    text="Paris is the capital of France.",
                    diagnostics_mode="low",
                    facts={"Paris is the capital of France": "supported"},
                ),
            ),
            slo_policy={
                "min_auto_selected_profile_counts": {
                    "latency": 1,
                },
            },
        )
    )

    assert payload["status"] == "blocked"
    assert payload["profiles"][0]["status"] == "promote"
    assert payload["slo"]["passed"] is False
    assert payload["slo"]["failures"][0]["metric"] == "auto_profile"
    assert payload["decision"]["blocking_reasons"] == (
        "sweep.auto_profile: SLO required True failed",
    )


def test_run_runtime_profile_selector_tuning_recommends_passing_policy(tmp_path):
    module = importlib.import_module("benchmarks.run_runtime_profile_selector_tuning")
    registry_module = importlib.import_module("eigentruth.registry")
    output_dir = tmp_path / "selector-tuning"
    registry_path = tmp_path / "registry.json"
    slo_policy_path = tmp_path / "slo-policy.json"
    slo_policy_path.write_text(
        json.dumps({
            "max_total_seconds_p95": 1.0,
            "max_mean_attempted_route_count": 1.1,
            "max_retrieval_use_rate": 0.0,
            "min_auto_selected_profile_counts": {
                "latency": 1,
                "balanced": 1,
                "audit": 1,
            },
        }),
        encoding="utf-8",
    )

    payload = module.run_runtime_profile_selector_tuning(
        module.RuntimeProfileSelectorTuningConfig(
            output_dir=output_dir,
            candidates=(
                module.RuntimeProfileSelectorCandidate(
                    name="default",
                    policy={},
                ),
                module.RuntimeProfileSelectorCandidate(
                    name="latency-biased",
                    policy={
                        "sensitive_claim_feature_flags": ["has_citation", "is_time_sensitive"],
                    },
                ),
            ),
            slo_policy_path=slo_policy_path,
            registry_path=registry_path,
            name="selector-tuning",
            version="0.1",
            compact_json=True,
        )
    )
    record = registry_module.ArtifactRegistry.load_json(registry_path).get(
        "report:selector-tuning:0.1"
    )

    assert payload["status"] == "promote"
    assert payload["decision"]["recommended_candidate"] == "default"
    assert payload["candidates"][0]["status"] == "promote"
    assert payload["candidates"][1]["status"] == "blocked"
    assert payload["candidates"][0]["runtime_profile_selection"]["counts_by_selected_profile"] == {
        "audit": 1,
        "balanced": 1,
        "latency": 1,
    }
    assert payload["candidates"][1]["runtime_profile_selection"]["counts_by_selected_profile"] == {
        "balanced": 1,
        "latency": 2,
    }
    assert Path(payload["paths"]["artifact_manifest"]).exists()
    manifest = json.loads(Path(payload["paths"]["artifact_manifest"]).read_text(encoding="utf-8"))
    assert payload["artifact_manifest_summary"] == manifest["summary"]
    assert "default_selector_policy" in manifest["artifacts"]
    assert "latency-biased_sweep_manifest" in manifest["artifacts"]
    assert registry_module.load_and_verify_artifact_manifest(
        payload["paths"]["artifact_manifest"]
    ).passed is True
    assert record.metadata["status"] == "promote"
    assert record.metadata["recommended_candidate"] == "default"
    assert record.metadata["candidate_count"] == 2


def test_build_product_trace_corpus_redacts_and_registers_replay_ready_traces(tmp_path):
    module = importlib.import_module("benchmarks.build_product_trace_corpus")
    registry_module = importlib.import_module("eigentruth.registry")
    output_dir = tmp_path / "trace-corpus"
    registry_path = tmp_path / "registry.json"
    trace_path = tmp_path / "trace.json"
    jsonl_path = tmp_path / "traces.jsonl"
    trace_payload = {
        "request_id": "latency-low-supported",
        "risk_decision": {
            "action": "accept",
            "risk_level": "low",
            "confidence": 1.0,
            "reason": "supported",
        },
        "claims": [{"claim_id": "c1", "text": "Private customer fact.", "metadata": {}}],
        "verification_results": [{
            "status": "supported",
            "confidence": 0.9,
            "evidence": ["Private evidence text."],
            "explanation": "Private explanation.",
            "metadata": {"key": "Private customer fact."},
        }],
        "metadata": {"runtime_profile": "latency"},
        "runtime_trace": {"total_seconds": 0.10, "phases": []},
    }
    jsonl_payload = {
        "request_id": "audit-low-sensitive",
        "risk_decision": {
            "action": "accept",
            "risk_level": "low",
            "confidence": 0.9,
            "reason": "numbered claim",
        },
        "claims": [{
            "claim_id": "c1",
            "text": "The account balance is 42.",
            "metadata": {"features": {"has_number": True}},
        }],
        "metadata": {"runtime_profile": "audit"},
        "runtime_trace": {"total_seconds": 0.40, "phases": []},
    }
    invalid_payload = {"request_id": "bad", "claims": []}
    trace_path.write_text(json.dumps(trace_payload), encoding="utf-8")
    jsonl_path.write_text(
        "\n".join([json.dumps(jsonl_payload), json.dumps(invalid_payload)]) + "\n",
        encoding="utf-8",
    )

    payload = module.build_product_trace_corpus(
        module.ProductTraceCorpusConfig(
            trace_paths=(trace_path,),
            jsonl_paths=(jsonl_path,),
            output_dir=output_dir,
            registry_path=registry_path,
            name="trace-corpus",
            version="0.1",
            require_runtime_trace=True,
            compact_json=True,
        )
    )
    record = registry_module.ArtifactRegistry.load_json(registry_path).get(
        "report:trace-corpus:0.1"
    )
    saved_trace = json.loads(Path(payload["traces"][0]["path"]).read_text(encoding="utf-8"))

    assert payload["status"] == "partial"
    assert payload["summary"]["accepted_count"] == 2
    assert payload["summary"]["rejected_count"] == 1
    assert payload["summary"]["runtime_trace_count"] == 2
    assert payload["summary"]["redacted_trace_count"] == 2
    assert payload["runtime_pair_index"]["record_count"] == 2
    assert payload["runtime_pair_index"]["profile_counts"] == {"audit": 1, "latency": 1}
    assert payload["summary"]["counts_by_runtime_profile"] == {"audit": 1, "latency": 1}
    assert payload["summary"]["rejected_reasons"] == {"missing risk_decision object": 1}
    assert payload["traces"][0]["request_key"] == "low-supported"
    assert saved_trace["claims"][0]["text"].startswith("[redacted:sha256=")
    assert saved_trace["verification_results"][0]["evidence"][0].startswith("[redacted:sha256=")
    assert saved_trace["verification_results"][0]["metadata"]["key"].startswith("[redacted:sha256=")
    assert saved_trace["metadata"]["runtime_replay_key"] == "low-supported"
    assert saved_trace["metadata"]["trace_corpus"]["redacted_text"] is True
    runtime_pair_index = json.loads(
        Path(payload["paths"]["runtime_pair_index"]).read_text(encoding="utf-8")
    )
    assert runtime_pair_index["workflow"] == "product_trace_runtime_pair_index"
    assert runtime_pair_index["summary"]["record_count"] == 2
    assert {
        (record["request_key"], record["runtime_profile"], record["total_seconds"])
        for record in runtime_pair_index["records"]
    } == {
        ("low-supported", "latency", 0.10),
        ("low-sensitive", "audit", 0.40),
    }
    manifest = json.loads(Path(payload["paths"]["artifact_manifest"]).read_text(encoding="utf-8"))
    assert payload["artifact_manifest_summary"] == manifest["summary"]
    assert "product_trace_runtime_pair_index" in manifest["artifacts"]
    assert manifest["metadata"]["runtime_pair_index_record_count"] == 2
    assert registry_module.load_and_verify_artifact_manifest(
        payload["paths"]["artifact_manifest"]
    ).passed is True
    assert record.metadata["status"] == "partial"
    assert record.metadata["accepted_count"] == 2
    assert record.metadata["rejected_count"] == 1
    assert record.metadata["runtime_pair_index_record_count"] == 2


def test_build_product_trace_corpus_streams_jsonl_limit_and_parses_bool_strings(tmp_path):
    module = importlib.import_module("benchmarks.build_product_trace_corpus")
    output_dir = tmp_path / "trace-corpus"
    jsonl_path = tmp_path / "traces.jsonl"
    trace_payload = {
        "request_id": "latency-low-supported",
        "risk_decision": {
            "action": "accept",
            "risk_level": "low",
            "confidence": 1.0,
            "reason": "supported",
        },
        "claims": [{"claim_id": "c1", "text": "Keep this visible.", "metadata": {}}],
        "metadata": {"runtime_profile": "latency"},
    }
    jsonl_path.write_text(
        json.dumps(trace_payload) + "\n{not-valid-json}\n",
        encoding="utf-8",
    )

    payload = module.build_product_trace_corpus(
        module.ProductTraceCorpusConfig(
            jsonl_paths=(jsonl_path,),
            output_dir=output_dir,
            redact_text="false",
            require_runtime_trace="false",
            strict="false",
            limit=1,
            compact_json="false",
        )
    )
    saved_trace = json.loads(Path(payload["traces"][0]["path"]).read_text(encoding="utf-8"))

    assert payload["status"] == "ready"
    assert payload["summary"]["accepted_count"] == 1
    assert payload["summary"]["rejected_count"] == 0
    assert saved_trace["claims"][0]["text"] == "Keep this visible."
    assert saved_trace["metadata"]["trace_corpus"]["redacted_text"] is False

    with pytest.raises(ValueError, match="strict"):
        module.ProductTraceCorpusConfig(
            jsonl_paths=(jsonl_path,),
            output_dir=output_dir,
            strict="maybe",
        )


def test_run_product_trace_replay_workflow_builds_corpus_baseline_and_replay(tmp_path):
    module = importlib.import_module("benchmarks.run_product_trace_replay_workflow")
    tuning_module = importlib.import_module("benchmarks.run_runtime_profile_selector_tuning")
    registry_module = importlib.import_module("eigentruth.registry")
    output_dir = tmp_path / "workflow"
    registry_path = tmp_path / "registry.json"
    replay_policy_path = tmp_path / "replay-policy.json"
    traces_dir = tmp_path / "input-traces"
    traces_dir.mkdir()
    replay_policy_path.write_text(
        json.dumps({
            "max_estimated_cost_units_mean": 2.0,
            "min_observed_runtime_coverage_rate": 1.0,
            "min_selected_profile_counts": {
                "latency": 1,
                "balanced": 1,
                "audit": 1,
            },
        }),
        encoding="utf-8",
    )

    trace_payloads = (
        {
            "request_id": "latency-low-supported",
            "risk_decision": {
                "action": "accept",
                "risk_level": "low",
                "confidence": 1.0,
                "reason": "supported",
            },
            "claims": [{"claim_id": "c1", "text": "Private low-risk fact.", "metadata": {}}],
            "metadata": {"runtime_profile": "latency"},
            "runtime_trace": {"total_seconds": 0.10, "phases": []},
        },
        {
            "request_id": "balanced-medium-retrieve",
            "risk_decision": {
                "action": "retrieve",
                "risk_level": "medium",
                "confidence": 0.7,
                "reason": "unsupported",
            },
            "claims": [{"claim_id": "c1", "text": "Private unsupported fact.", "metadata": {}}],
            "metadata": {"runtime_profile": "balanced"},
            "runtime_trace": {"total_seconds": 0.20, "phases": []},
        },
        {
            "request_id": "audit-low-sensitive",
            "risk_decision": {
                "action": "accept",
                "risk_level": "low",
                "confidence": 0.9,
                "reason": "numbered claim",
            },
            "claims": [{
                "claim_id": "c1",
                "text": "Private account balance is 42.",
                "metadata": {"features": {"has_number": True}},
            }],
            "metadata": {"runtime_profile": "audit"},
            "runtime_trace": {"total_seconds": 0.40, "phases": []},
        },
    )
    trace_paths = []
    for index, payload in enumerate(trace_payloads):
        trace_path = traces_dir / f"trace-{index}.json"
        trace_path.write_text(json.dumps(payload), encoding="utf-8")
        trace_paths.append(trace_path)

    payload = module.run_product_trace_replay_workflow(
        module.ProductTraceReplayWorkflowConfig(
            trace_paths=trace_paths,
            output_dir=output_dir,
            candidates=(
                tuning_module.RuntimeProfileSelectorCandidate(
                    name="default",
                    policy={},
                ),
                tuning_module.RuntimeProfileSelectorCandidate(
                    name="latency-biased",
                    policy={
                        "sensitive_claim_feature_flags": ["has_citation", "is_time_sensitive"],
                    },
                ),
            ),
            replay_policy_path=replay_policy_path,
            registry_path=registry_path,
            name="trace-replay-workflow",
            version="0.1",
            require_runtime_trace=True,
            compact_json=True,
        )
    )
    record = registry_module.ArtifactRegistry.load_json(registry_path).get(
        "report:trace-replay-workflow:0.1"
    )
    corpus_trace = next((output_dir / "corpus" / "traces").glob("latency-*.json"))
    saved_trace = json.loads(corpus_trace.read_text(encoding="utf-8"))

    assert payload["status"] == "promote"
    assert payload["corpus"]["status"] == "ready"
    assert payload["corpus"]["accepted_count"] == 3
    assert payload["corpus"]["runtime_pair_index_record_count"] == 3
    assert payload["runtime_baseline"]["status"] == "observed"
    assert payload["runtime_baseline"]["n_traces"] == 3
    assert payload["selector_replay"]["status"] == "promote"
    assert payload["selector_replay"]["recommended_candidate"] == "default"
    assert payload["paths"]["corpus_runtime_pair_index"] is not None
    assert saved_trace["claims"][0]["text"].startswith("[redacted:sha256=")
    selector_replay_report = json.loads(
        Path(payload["paths"]["selector_replay_report"]).read_text(encoding="utf-8")
    )
    assert selector_replay_report["config"]["runtime_pairing"]["source"] == "runtime_pair_index"
    assert selector_replay_report["paths"]["runtime_pair_index"] == payload["paths"]["corpus_runtime_pair_index"]
    manifest = json.loads(Path(payload["paths"]["artifact_manifest"]).read_text(encoding="utf-8"))
    assert payload["artifact_manifest_summary"] == manifest["summary"]
    assert "corpus_runtime_pair_index" in manifest["artifacts"]
    assert registry_module.load_and_verify_artifact_manifest(
        payload["paths"]["artifact_manifest"],
        recursive=True,
    ).passed is True
    assert record.metadata["status"] == "promote"
    assert record.metadata["corpus_status"] == "ready"
    assert record.metadata["selector_replay_status"] == "promote"


def test_product_trace_replay_runtime_configs_parse_bool_strings(tmp_path):
    workflow_module = importlib.import_module("benchmarks.run_product_trace_replay_workflow")
    replay_module = importlib.import_module("benchmarks.run_runtime_profile_selector_replay")
    baseline_module = importlib.import_module("benchmarks.run_product_runtime_baseline")
    candidate = {"name": "default", "policy": {}}

    workflow_config = workflow_module.ProductTraceReplayWorkflowConfig(
        trace_paths=("trace.json",),
        output_dir=tmp_path / "workflow",
        candidates=(candidate,),
        redact_text="false",
        require_runtime_trace="false",
        strict="false",
        compact_json="false",
    )
    replay_config = replay_module.RuntimeProfileSelectorReplayConfig(
        trace_paths=("trace.json",),
        output_dir=tmp_path / "selector",
        candidates=(candidate,),
        compact_json="false",
    )
    baseline_config = baseline_module.ProductRuntimeBaselineConfig(
        trace_paths=("trace.json",),
        report_path=tmp_path / "baseline.json",
        compact_json="false",
    )

    assert workflow_config.redact_text is False
    assert workflow_config.require_runtime_trace is False
    assert workflow_config.strict is False
    assert workflow_config.compact_json is False
    assert replay_config.compact_json is False
    assert baseline_config.compact_json is False

    with pytest.raises(ValueError, match="compact_json"):
        replay_module.RuntimeProfileSelectorReplayConfig(
            trace_paths=("trace.json",),
            output_dir=tmp_path / "bad-selector",
            candidates=(candidate,),
            compact_json="maybe",
        )


def test_run_runtime_profile_selector_replay_recommends_passing_policy(tmp_path):
    module = importlib.import_module("benchmarks.run_runtime_profile_selector_replay")
    tuning_module = importlib.import_module("benchmarks.run_runtime_profile_selector_tuning")
    registry_module = importlib.import_module("eigentruth.registry")
    output_dir = tmp_path / "selector-replay"
    registry_path = tmp_path / "registry.json"
    traces_dir = tmp_path / "traces"
    traces_dir.mkdir()

    trace_payloads = (
        {
            "request_id": "low-supported",
            "risk_decision": {
                "action": "accept",
                "risk_level": "low",
                "confidence": 1.0,
                "reason": "supported",
            },
            "claims": [{"claim_id": "c1", "text": "Paris is the capital of France.", "metadata": {}}],
            "metadata": {"runtime_profile": "latency"},
            "runtime_trace": {"total_seconds": 0.10, "phases": []},
        },
        {
            "request_id": "medium-retrieve",
            "risk_decision": {
                "action": "retrieve",
                "risk_level": "medium",
                "confidence": 0.7,
                "reason": "unsupported",
            },
            "claims": [{"claim_id": "c1", "text": "needs evidence", "metadata": {}}],
            "metadata": {"runtime_profile": "balanced"},
            "runtime_trace": {"total_seconds": 0.20, "phases": []},
        },
        {
            "request_id": "low-sensitive",
            "risk_decision": {
                "action": "accept",
                "risk_level": "low",
                "confidence": 0.9,
                "reason": "numbered claim",
            },
            "claims": [
                {
                    "claim_id": "c1",
                    "text": "The measured value is 42.",
                    "metadata": {"features": {"has_number": True}},
                }
            ],
            "metadata": {"runtime_profile": "audit"},
            "runtime_trace": {"total_seconds": 0.40, "phases": []},
        },
    )
    trace_paths = []
    for index, payload in enumerate(trace_payloads):
        trace_path = traces_dir / f"trace-{index}.json"
        trace_path.write_text(json.dumps(payload), encoding="utf-8")
        trace_paths.append(trace_path)

    payload = module.run_runtime_profile_selector_replay(
        module.RuntimeProfileSelectorReplayConfig(
            trace_paths=trace_paths,
            output_dir=output_dir,
            candidates=(
                tuning_module.RuntimeProfileSelectorCandidate(
                    name="default",
                    policy={},
                ),
                tuning_module.RuntimeProfileSelectorCandidate(
                    name="latency-biased",
                    policy={
                        "sensitive_claim_feature_flags": ["has_citation", "is_time_sensitive"],
                    },
                ),
            ),
            replay_policy=module.RuntimeProfileSelectorReplayPolicy(
                max_estimated_cost_units_mean=2.0,
                max_observed_selected_total_seconds_mean=0.25,
                max_observed_selected_total_seconds_p95=0.50,
                min_observed_runtime_coverage_rate=1.0,
                min_selected_profile_counts={"latency": 1, "balanced": 1, "audit": 1},
            ),
            registry_path=registry_path,
            name="selector-replay",
            version="0.1",
            compact_json=True,
        )
    )
    record = registry_module.ArtifactRegistry.load_json(registry_path).get(
        "report:selector-replay:0.1"
    )

    assert payload["status"] == "promote"
    assert payload["decision"]["recommended_candidate"] == "default"
    assert payload["candidates"][0]["status"] == "promote"
    assert payload["candidates"][1]["status"] == "blocked"
    assert payload["candidates"][0]["summary"]["selected_counts"] == {
        "audit": 1,
        "balanced": 1,
        "latency": 1,
    }
    assert payload["candidates"][0]["summary"]["observed_runtime_coverage_rate"] == pytest.approx(1.0)
    assert payload["candidates"][0]["summary"]["observed_selected_total_seconds_mean"] == pytest.approx(
        (0.10 + 0.20 + 0.40) / 3.0
    )
    assert payload["config"]["runtime_pairing"]["indexed_observations"] == 3
    assert payload["candidates"][0]["summary"]["observed_selected_total_seconds_p95"] == pytest.approx(0.38)
    assert payload["candidates"][1]["summary"]["selected_counts"] == {
        "balanced": 1,
        "latency": 2,
    }
    assert {
        failure["metric"] for failure in payload["candidates"][1]["gate"]["failures"]
    } >= {"selected_count.audit", "observed_runtime_coverage_rate"}
    assert Path(payload["paths"]["artifact_manifest"]).exists()
    manifest = json.loads(Path(payload["paths"]["artifact_manifest"]).read_text(encoding="utf-8"))
    assert "default_selector_policy" in manifest["artifacts"]
    assert "latency-biased_selector_policy" in manifest["artifacts"]
    assert registry_module.load_and_verify_artifact_manifest(
        payload["paths"]["artifact_manifest"]
    ).passed is True
    assert record.metadata["status"] == "promote"
    assert record.metadata["recommended_candidate"] == "default"
    assert record.metadata["candidate_count"] == 2
    assert record.metadata["trace_count"] == 3


def test_runtime_profile_selector_replay_uses_external_runtime_pair_index(tmp_path, monkeypatch):
    module = importlib.import_module("benchmarks.run_runtime_profile_selector_replay")
    output_dir = tmp_path / "selector-replay"
    trace_path = tmp_path / "trace.json"
    runtime_pair_index_path = tmp_path / "runtime-pair-index.json"
    trace_path.write_text(
        json.dumps({
            "request_id": "low-supported",
            "risk_decision": {
                "action": "accept",
                "risk_level": "low",
                "confidence": 1.0,
                "reason": "supported",
            },
            "claims": [{"claim_id": "c1", "text": "Paris is the capital of France.", "metadata": {}}],
            "metadata": {"runtime_profile": "latency"},
        }),
        encoding="utf-8",
    )
    runtime_pair_index_path.write_text(
        json.dumps({
            "schema_version": 1,
            "workflow": "product_trace_runtime_pair_index",
            "records": [{
                "request_key": "low-supported",
                "runtime_profile": "latency",
                "path": str(trace_path),
                "total_seconds": 0.12,
            }],
        }),
        encoding="utf-8",
    )

    def fail_if_trace_scan_is_used(*_args, **_kwargs):
        raise AssertionError("external runtime pair index should avoid trace scan")

    monkeypatch.setattr(module, "_runtime_pair_index", fail_if_trace_scan_is_used)

    payload = module.run_runtime_profile_selector_replay(
        module.RuntimeProfileSelectorReplayConfig(
            trace_paths=(trace_path,),
            output_dir=output_dir,
            candidates=(module.RuntimeProfileSelectorCandidate(name="default", policy={}),),
            runtime_pair_index_path=runtime_pair_index_path,
        )
    )
    manifest = json.loads(Path(payload["paths"]["artifact_manifest"]).read_text(encoding="utf-8"))

    assert payload["config"]["runtime_pairing"]["source"] == "runtime_pair_index"
    assert payload["config"]["runtime_pairing"]["indexed_observations"] == 1
    assert payload["paths"]["runtime_pair_index"] == str(runtime_pair_index_path)
    assert payload["candidates"][0]["summary"]["observed_runtime_coverage_rate"] == pytest.approx(1.0)
    assert payload["candidates"][0]["summary"]["observed_selected_total_seconds_mean"] == pytest.approx(0.12)
    assert manifest["artifacts"]["runtime_pair_index"]["exists"] is True
    assert manifest["metadata"]["runtime_pair_index_source"] == "runtime_pair_index"


def test_runtime_profile_selector_replay_writes_trace_detail_sidecar(tmp_path):
    module = importlib.import_module("benchmarks.run_runtime_profile_selector_replay")
    registry_module = importlib.import_module("eigentruth.registry")
    output_dir = tmp_path / "selector-replay"
    trace_details_path = output_dir / "trace-details.json"
    traces_dir = tmp_path / "traces"
    traces_dir.mkdir()

    trace_paths = []
    for index in range(3):
        trace_path = traces_dir / f"trace-{index}.json"
        trace_path.write_text(
            json.dumps({
                "request_id": f"low-supported-{index}",
                "risk_decision": {
                    "action": "accept",
                    "risk_level": "low",
                    "confidence": 1.0,
                    "reason": "supported",
                },
                "claims": [{"claim_id": "c1", "text": "Paris is the capital of France.", "metadata": {}}],
                "metadata": {"runtime_profile": "latency"},
                "runtime_trace": {"total_seconds": 0.10 + index, "phases": []},
            }),
            encoding="utf-8",
        )
        trace_paths.append(trace_path)

    payload = module.run_runtime_profile_selector_replay(
        module.RuntimeProfileSelectorReplayConfig(
            trace_paths=trace_paths,
            output_dir=output_dir,
            candidates=(module.RuntimeProfileSelectorCandidate(name="default", policy={}),),
            compact_json=True,
            detail_limit=1,
            trace_details_path=trace_details_path,
        )
    )

    candidate = payload["candidates"][0]
    assert payload["paths"]["trace_details"] == str(trace_details_path)
    assert payload["config"]["detail_limit"] == 1
    assert len(candidate["traces"]) == 1
    assert candidate["summary"]["trace_count"] == 3
    assert candidate["trace_detail_count"] == 3
    assert candidate["inline_trace_count"] == 1
    assert candidate["trace_detail_truncated"] is True
    assert candidate["trace_details_path"] == str(trace_details_path)

    trace_details = json.loads(trace_details_path.read_text(encoding="utf-8"))
    assert trace_details["workflow"] == "runtime_profile_selector_replay_trace_details"
    assert trace_details["summary"]["trace_record_count"] == 3
    assert trace_details["summary"]["truncated_candidate_count"] == 1
    assert trace_details["candidates"][0]["trace_count"] == 3
    assert len(trace_details["candidates"][0]["traces"]) == 3

    manifest = json.loads(Path(payload["paths"]["artifact_manifest"]).read_text(encoding="utf-8"))
    assert "runtime_profile_selector_replay_trace_details" in manifest["artifacts"]
    assert manifest["metadata"]["detail_limit"] == 1
    assert manifest["metadata"]["trace_details_path"] == str(trace_details_path)
    assert registry_module.load_and_verify_artifact_manifest(
        payload["paths"]["artifact_manifest"]
    ).passed is True


def test_runtime_profile_selector_replay_streams_sidecar_summary(tmp_path, monkeypatch):
    module = importlib.import_module("benchmarks.run_runtime_profile_selector_replay")
    output_dir = tmp_path / "selector-replay"
    trace_details_path = output_dir / "trace-details.json"
    traces_dir = tmp_path / "traces"
    traces_dir.mkdir()

    trace_paths = []
    for index in range(2):
        trace_path = traces_dir / f"trace-{index}.json"
        trace_path.write_text(
            json.dumps({
                "request_id": f"low-supported-{index}",
                "risk_decision": {
                    "action": "accept",
                    "risk_level": "low",
                    "confidence": 1.0,
                    "reason": "supported",
                },
                "claims": [{"claim_id": "c1", "text": "Paris is the capital of France.", "metadata": {}}],
                "metadata": {"runtime_profile": "latency"},
                "runtime_trace": {"total_seconds": 0.10 + index, "phases": []},
            }),
            encoding="utf-8",
        )
        trace_paths.append(trace_path)

    def fail_if_full_trace_summary_is_used(*_args, **_kwargs):
        raise AssertionError("sidecar replay should stream summary accumulation")

    monkeypatch.setattr(module, "_selection_summary", fail_if_full_trace_summary_is_used)

    payload = module.run_runtime_profile_selector_replay(
        module.RuntimeProfileSelectorReplayConfig(
            trace_paths=trace_paths,
            output_dir=output_dir,
            candidates=(module.RuntimeProfileSelectorCandidate(name="default", policy={}),),
            detail_limit=0,
            trace_details_path=trace_details_path,
        )
    )
    trace_details = json.loads(trace_details_path.read_text(encoding="utf-8"))

    assert payload["candidates"][0]["traces"] == []
    assert payload["candidates"][0]["summary"]["trace_count"] == 2
    assert payload["candidates"][0]["trace_detail_truncated"] is True
    assert trace_details["summary"]["trace_record_count"] == 2
    assert len(trace_details["candidates"][0]["traces"]) == 2
    assert not trace_details_path.with_name(f"{trace_details_path.name}.tmp").exists()


def test_runtime_profile_selector_replay_reiterates_trace_corpus_without_materializing(tmp_path, monkeypatch):
    module = importlib.import_module("benchmarks.run_runtime_profile_selector_replay")
    output_dir = tmp_path / "selector-replay"
    trace_details_path = output_dir / "trace-details.json"
    trace_paths = tuple(tmp_path / f"trace-{index}.json" for index in range(3))
    load_calls = []

    def fake_load_trace_replay_input(path):
        trace_path = Path(path)
        load_calls.append(trace_path.name)
        return module.TraceReplayInput(
            path=trace_path,
            request_id=trace_path.stem,
            request_key=trace_path.stem,
            original_runtime_profile="latency",
            runtime_pair_profile="latency",
            risk_decision={
                "action": "accept",
                "risk_level": "low",
                "confidence": 1.0,
                "reason": "supported",
            },
            claims=(),
            original_total_seconds=0.1,
        )

    monkeypatch.setattr(module, "_load_trace_replay_input", fake_load_trace_replay_input)

    payload = module.run_runtime_profile_selector_replay(
        module.RuntimeProfileSelectorReplayConfig(
            trace_paths=trace_paths,
            output_dir=output_dir,
            candidates=(
                module.RuntimeProfileSelectorCandidate(name="default", policy={}),
                module.RuntimeProfileSelectorCandidate(name="strict", policy={}),
            ),
            detail_limit=0,
            trace_details_path=trace_details_path,
        )
    )
    trace_details = json.loads(trace_details_path.read_text(encoding="utf-8"))

    expected_pass = [path.name for path in trace_paths]
    assert load_calls == expected_pass * 3
    assert payload["config"]["trace_count"] == 3
    assert payload["paths"]["traces"] == [str(path) for path in trace_paths]
    assert [candidate["traces"] for candidate in payload["candidates"]] == [[], []]
    assert trace_details["summary"]["candidate_count"] == 2
    assert trace_details["summary"]["trace_record_count"] == 6


def test_runtime_profile_selector_replay_uses_lightweight_trace_inputs(tmp_path):
    module = importlib.import_module("benchmarks.run_runtime_profile_selector_replay")
    trace_path = tmp_path / "trace.json"
    large_text = "PRIVATE PAYLOAD " * 1000
    trace_path.write_text(
        json.dumps({
            "request_id": "audit-sensitive",
            "risk_decision": {
                "action": "accept",
                "risk_level": "low",
                "confidence": 0.9,
                "reason": "sensitive claim",
            },
            "claims": [{
                "claim_id": "c1",
                "text": large_text,
                "metadata": {"features": {"has_number": True}},
            }],
            "verification_results": [{
                "status": "supported",
                "evidence": [large_text],
                "explanation": large_text,
            }],
            "metadata": {"runtime_profile": "audit"},
            "runtime_trace": {"total_seconds": 0.30, "phases": []},
            "generated_text": large_text,
        }),
        encoding="utf-8",
    )

    replay_input = module._load_trace_replay_input(trace_path)
    pair_index = module._runtime_pair_index((replay_input,))
    observation = pair_index[(replay_input.request_key, "audit")][0]
    record = module._trace_selection_record(
        replay_input,
        candidate=module.RuntimeProfileSelectorCandidate(name="default", policy={}),
        cost_units=module.DEFAULT_PROFILE_COST_UNITS,
        runtime_pair_index=pair_index,
    )

    assert not hasattr(replay_input, "payload")
    assert large_text not in repr(replay_input)
    assert replay_input.claims == ({"claim_id": "c1", "metadata": {"features": {"has_number": True}}},)
    assert not hasattr(observation, "risk_decision")
    assert large_text not in repr(observation)
    assert observation.total_seconds == pytest.approx(0.30)
    assert record["selected_runtime_profile"] == "audit"
    assert record["observed_selected_pair_count"] == 1


def test_eval_calibration_transfer_builds_threshold_transfer_matrix(tmp_path, monkeypatch):
    module = importlib.import_module("benchmarks.eval_calibration_transfer")
    from eigentruth.eval.score_dump import ScoreDump

    def fail_to_mapping(self):
        raise AssertionError("calibration transfer should consume ScoreDump directly")

    monkeypatch.setattr(ScoreDump, "to_mapping", fail_to_mapping)
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
    assert payload["score_dumps"]["source"]["summary"]["n_total"] == 6
    assert payload["score_dumps"]["source"]["summary"]["sweep_layers"] == ("-2",)
    assert payload["score_dumps"]["shifted"]["sha256"]
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
    assert run["score_dump"]["summary"]["n_total"] == len(labels)
    assert run["score_dump"]["summary"]["score_count"] == 2
    assert run["score_dump"]["sha256"]
    assert ensemble_detection >= single_detection
    assert run["ensemble_results"]["max_rank"]["alphas"]["0.2"]["false_alarm"] <= 0.23


def test_eval_score_ensemble_reads_jsonl_manifest_columns(tmp_path):
    module = importlib.import_module("benchmarks.eval_score_ensemble")
    from eigentruth.eval.score_dump import ScoreDump, write_score_dump_jsonl

    labels = [0] * 20 + [1] * 8
    truth_proj = list(range(20)) + [40, 41, 42, 43, 0, 1, 2, 3]
    subspace_resid = list(range(20)) + [0, 1, 2, 3, 40, 41, 42, 43]
    unused = [999.0] * len(labels)
    dump = ScoreDump.from_mapping({
        "config": {"model": "synthetic", "layer": -1},
        "labels": labels,
        "scores": {
            "truth_proj": truth_proj,
            "subspace_resid": subspace_resid,
            "unused": unused,
        },
    })
    manifest_path = tmp_path / "scores.manifest.json"
    write_score_dump_jsonl(dump, manifest_path)

    payload = module.build_ensemble_report(
        [("synthetic", manifest_path)],
        signals=("truth_proj", "subspace_resid"),
        methods=("max_rank",),
        alphas=(0.2,),
        repeats=1,
        seed=0,
        best_alpha=0.2,
    )

    run = payload["runs"][0]
    assert run["score_dump"]["source_format"] == "eigentruth.score_dump.jsonl"
    assert run["score_dump"]["summary"]["score_count"] == 3
    assert run["signals"] == ["truth_proj", "subspace_resid"]
    assert "unused" not in run["single_results"]


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
    assert module.INSIDE_SEMANTIC_ENTROPY_SIGNAL not in module._enabled_signals(disabled)
    assert module.INSIDE_EMBEDDING_ENTROPY_SIGNAL not in module._enabled_signals(disabled)
    assert module.INSIDE_SIGNAL in module._enabled_signals(enabled)
    assert module.INSIDE_SEMANTIC_ENTROPY_SIGNAL in module._enabled_signals(enabled)
    assert module.INSIDE_EMBEDDING_ENTROPY_SIGNAL in module._enabled_signals(enabled)
    assert module.INSIDE_SIGNAL in module._sweep_signal_names(enabled)
    assert module.INSIDE_SEMANTIC_ENTROPY_SIGNAL in module._sweep_signal_names(enabled)
    assert module.INSIDE_EMBEDDING_ENTROPY_SIGNAL in module._sweep_signal_names(enabled)
    assert module.DEFAULT_SCORE_DIRECTIONS[module.INSIDE_SIGNAL] == "higher"
    assert module.DEFAULT_SCORE_DIRECTIONS[module.INSIDE_SEMANTIC_ENTROPY_SIGNAL] == "higher"
    assert module.DEFAULT_SCORE_DIRECTIONS[module.INSIDE_EMBEDDING_ENTROPY_SIGNAL] == "higher"
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


def test_eval_truthfulqa_token_budget_batches_limit_padded_tokens():
    module = importlib.import_module("benchmarks.eval_truthfulqa")
    statements = [
        module.Statement("", "a", 0),
        module.Statement("", "bbbbbbbb", 1),
        module.Statement("", "ccc", 0),
        module.Statement("", "ddddd", 1),
    ]
    encodings = [
        module.StatementEncoding((1, 2), 1),
        module.StatementEncoding(tuple(range(8)), 4),
        module.StatementEncoding((1, 2, 3), 1),
        module.StatementEncoding((1, 2, 3, 4, 5), 2),
    ]

    batches = list(module._batched_statement_pairs(
        statements,
        encodings,
        3,
        length_bucketed=False,
        max_batch_tokens=10,
    ))
    oversized = module._next_statement_pair_batch(
        list(zip(statements, encodings)),
        1,
        3,
        max_batch_tokens=5,
    )

    assert [[stmt.answer for stmt, _encoding in batch] for batch in batches] == [
        ["a"],
        ["bbbbbbbb"],
        ["ccc", "ddddd"],
    ]
    assert [stmt.answer for stmt, _encoding in oversized] == ["bbbbbbbb"]


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
    assert result["config"]["dtype"] == "float32"
    assert result["config"]["max_length"] == 32
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
    stmt = module.Statement("Question?", "Answer.", 0)
    other_stmt = module.Statement("Question?", "Other answer.", 0)

    seeds = {
        module._inside_seed(7, eval_batch_idx=0, inside_batch_idx=0),
        module._inside_seed(7, eval_batch_idx=0, inside_batch_idx=1),
        module._inside_seed(7, eval_batch_idx=1, inside_batch_idx=0),
    }

    assert len(seeds) == 3
    assert module._inside_seed(7, eval_batch_idx=2, inside_batch_idx=3) == module._inside_seed(
        7, eval_batch_idx=2, inside_batch_idx=3
    )
    assert module._inside_statement_seed(7, stmt) == module._inside_statement_seed(7, stmt)
    assert module._inside_statement_seed(7, stmt) != module._inside_statement_seed(8, stmt)
    assert module._inside_statement_seed(7, stmt) != module._inside_statement_seed(7, other_stmt)


def test_eval_truthfulqa_inside_diagnostics_cache_roundtrip_and_key_scope(tmp_path):
    module = importlib.import_module("benchmarks.eval_truthfulqa")
    stmt = module.Statement("Question?", "Answer.", 0)
    args = SimpleNamespace(
        model="tiny-local",
        dtype="float32",
        layer=-1,
        max_length=64,
        hidden_state_capture="outputs",
        seed=11,
        inside_samples=3,
        inside_min_samples=2,
        inside_sample_step=1,
        inside_stability_delta=0.05,
        inside_selfcheck_min_overlap=0.65,
        inside_selfcheck_support_threshold=0.60,
        inside_selfcheck_refute_threshold=0.50,
        inside_max_new_tokens=4,
        inside_temperature=0.7,
        inside_top_p=0.9,
        inside_pooling="last",
        inside_embedding_threshold=0.9,
        eigenscore_alpha=1e-3,
        inside_trigger_signal="truth_proj",
        inside_trigger_top_fraction=0.1,
        inside_trigger_threshold=None,
    )
    key = module._inside_diagnostics_cache_key(
        stmt,
        args,
        layers=(-1,),
        adaptive=True,
        selfcheck_early_stop=True,
    )
    args.inside_trigger_top_fraction = 0.4
    same_statement_key = module._inside_diagnostics_cache_key(
        stmt,
        args,
        layers=(-1,),
        adaptive=True,
        selfcheck_early_stop=True,
    )
    args.dtype = "bfloat16"
    dtype_key = module._inside_diagnostics_cache_key(
        stmt,
        args,
        layers=(-1,),
        adaptive=True,
        selfcheck_early_stop=True,
    )
    diagnostics = module.SampledInsideDiagnostics(
        eigenscore_by_layer={-1: 0.25},
        semantic_entropy=0.5,
        embedding_entropy_by_layer={-1: 0.75},
        sample_texts=("one", "two", "three"),
        n_samples=3,
        adaptive_rounds=2,
        stopped_early=True,
        stop_reason="stability_delta",
    )
    cache_path = tmp_path / "inside-diagnostics.json"
    cache = module.InsideDiagnosticsCache(cache_path)

    assert same_statement_key == key
    assert dtype_key != key
    assert cache.get(key) is None
    cache.put(key, diagnostics)
    cache.save()

    restored_cache = module.InsideDiagnosticsCache(cache_path)
    restored = restored_cache.get(key)

    assert restored is not None
    assert restored.eigenscore_by_layer[-1] == pytest.approx(0.25)
    assert restored.semantic_entropy == pytest.approx(0.5)
    assert restored.embedding_entropy_by_layer[-1] == pytest.approx(0.75)
    assert restored.sample_texts == ("one", "two", "three")
    assert restored.n_samples == 3
    assert restored.stopped_early is True
    assert restored.stop_reason == "stability_delta"
    assert restored_cache.stats()["hits"] == 1
    assert restored_cache.stats()["misses"] == 0


def test_eval_truthfulqa_sampled_inside_diagnostics_include_embedding_entropy(monkeypatch):
    module = importlib.import_module("benchmarks.eval_truthfulqa")

    def fake_response_diagnostics_batch(*_args, **_kwargs):
        return [
            module.SampledResponseDiagnostics(
                embeddings_by_layer={
                    -1: torch.tensor([
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ])
                },
                sample_texts=("Paris is correct.", "Paris is correct.", "Lyon is correct."),
            )
        ]

    monkeypatch.setattr(module, "sampled_response_diagnostics_batch", fake_response_diagnostics_batch)

    diagnostics = module.sampled_inside_diagnostics_batch(
        None,
        None,
        [module.Statement("Question?", "Answer.", 0)],
        [-1],
        torch.device("cpu"),
        64,
        n_samples=3,
        max_new_tokens=2,
        temperature=0.7,
        top_p=0.9,
        pooling="last",
        seed=0,
        eigenscore_alpha=1e-3,
        embedding_similarity_threshold=0.95,
    )[0]

    assert diagnostics is not None
    assert diagnostics.semantic_entropy > 0.0
    assert diagnostics.embedding_entropy_by_layer[-1] > 0.99
    assert -1 in diagnostics.eigenscore_by_layer
    assert diagnostics.sample_texts == ("Paris is correct.", "Paris is correct.", "Lyon is correct.")


def test_eval_truthfulqa_adaptive_inside_stops_when_entropy_is_stable(monkeypatch):
    module = importlib.import_module("benchmarks.eval_truthfulqa")
    requested_samples = []

    def fake_response_diagnostics_batch(*_args, **kwargs):
        n_samples = int(kwargs["n_samples"])
        requested_samples.append(n_samples)
        return [
            module.SampledResponseDiagnostics(
                embeddings_by_layer={-1: torch.ones(n_samples, 3)},
                sample_texts=tuple("same answer" for _ in range(n_samples)),
            )
        ]

    monkeypatch.setattr(module, "sampled_response_diagnostics_batch", fake_response_diagnostics_batch)

    diagnostics = module.sampled_inside_adaptive_diagnostics_batch(
        None,
        None,
        [module.Statement("Question?", "Answer.", 0)],
        [-1],
        torch.device("cpu"),
        64,
        min_samples=2,
        max_samples=5,
        sample_step=1,
        stability_delta=0.0,
        target_layer=-1,
        max_new_tokens=2,
        temperature=0.7,
        top_p=0.9,
        pooling="last",
        seed=0,
        eigenscore_alpha=1e-3,
        embedding_similarity_threshold=0.95,
    )[0]

    assert diagnostics is not None
    assert requested_samples == [2, 1]
    assert diagnostics.n_samples == 3
    assert diagnostics.adaptive_rounds == 2
    assert diagnostics.stopped_early is True
    assert diagnostics.stop_reason == "stability_delta"
    assert diagnostics.sample_texts == ("same answer", "same answer", "same answer")


def test_eval_truthfulqa_adaptive_inside_can_stop_generation_on_selfcheck_bounds(monkeypatch):
    module = importlib.import_module("benchmarks.eval_truthfulqa")
    requested_samples = []

    def fake_response_diagnostics_batch(*_args, **kwargs):
        n_samples = int(kwargs["n_samples"])
        requested_samples.append(n_samples)
        return [
            module.SampledResponseDiagnostics(
                embeddings_by_layer={-1: torch.ones(n_samples, 3)},
                sample_texts=tuple("AlphaCorp has 12 offices in Europe." for _ in range(n_samples)),
            )
        ]

    monkeypatch.setattr(module, "sampled_response_diagnostics_batch", fake_response_diagnostics_batch)

    diagnostics = module.sampled_inside_adaptive_diagnostics_batch(
        None,
        None,
        [module.Statement("", "AlphaCorp has 10 offices in Europe.", 1)],
        [-1],
        torch.device("cpu"),
        64,
        min_samples=2,
        max_samples=5,
        sample_step=1,
        stability_delta=0.0,
        target_layer=-1,
        max_new_tokens=2,
        temperature=0.7,
        top_p=0.9,
        pooling="last",
        seed=0,
        eigenscore_alpha=1e-3,
        embedding_similarity_threshold=0.95,
        selfcheck_early_stop=True,
        selfcheck_min_overlap=0.55,
        selfcheck_refute_threshold=0.40,
        selfcheck_support_threshold=0.80,
    )[0]

    assert diagnostics is not None
    assert requested_samples == [2]
    assert diagnostics.n_samples == 2
    assert diagnostics.adaptive_rounds == 1
    assert diagnostics.stopped_early is True
    assert diagnostics.stop_reason == "selfcheck_refute_threshold_guaranteed"


def test_eval_truthfulqa_adaptive_inside_selfcheck_ignores_empty_samples(monkeypatch):
    module = importlib.import_module("benchmarks.eval_truthfulqa")
    requested_samples = []
    sample_rounds = iter([
        ("", "AlphaCorp has 12 offices in Europe."),
        ("AlphaCorp has 12 offices in Europe.",),
    ])

    def fake_response_diagnostics_batch(*_args, **kwargs):
        n_samples = int(kwargs["n_samples"])
        requested_samples.append(n_samples)
        sample_texts = next(sample_rounds)
        assert len(sample_texts) == n_samples
        return [
            module.SampledResponseDiagnostics(
                embeddings_by_layer={-1: torch.ones(n_samples, 3)},
                sample_texts=sample_texts,
            )
        ]

    monkeypatch.setattr(module, "sampled_response_diagnostics_batch", fake_response_diagnostics_batch)

    diagnostics = module.sampled_inside_adaptive_diagnostics_batch(
        None,
        None,
        [module.Statement("", "AlphaCorp has 10 offices in Europe.", 1)],
        [-1],
        torch.device("cpu"),
        64,
        min_samples=2,
        max_samples=5,
        sample_step=1,
        stability_delta=0.0,
        target_layer=-1,
        max_new_tokens=2,
        temperature=0.7,
        top_p=0.9,
        pooling="last",
        seed=0,
        eigenscore_alpha=1e-3,
        embedding_similarity_threshold=0.95,
        selfcheck_early_stop=True,
        selfcheck_min_overlap=0.55,
        selfcheck_refute_threshold=0.40,
        selfcheck_support_threshold=0.80,
    )[0]

    assert diagnostics is not None
    assert requested_samples == [2, 1]
    assert diagnostics.sample_texts == (
        "",
        "AlphaCorp has 12 offices in Europe.",
        "AlphaCorp has 12 offices in Europe.",
    )
    assert diagnostics.stopped_early is True
    assert diagnostics.stop_reason == "selfcheck_refute_threshold_guaranteed"


def test_eval_truthfulqa_adaptive_inside_rejects_invalid_config():
    module = importlib.import_module("benchmarks.eval_truthfulqa")

    with pytest.raises(ValueError, match="min_samples"):
        module.sampled_inside_adaptive_diagnostics_batch(
            None,
            None,
            [module.Statement("Question?", "Answer.", 0)],
            [-1],
            torch.device("cpu"),
            64,
            min_samples=1,
            max_samples=2,
            sample_step=1,
            stability_delta=0.0,
            target_layer=-1,
            max_new_tokens=2,
            temperature=0.7,
            top_p=0.9,
            pooling="last",
            seed=0,
            eigenscore_alpha=1e-3,
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


def test_eval_truthfulqa_answer_nll_only_normalizes_answer_window():
    module = importlib.import_module("benchmarks.eval_truthfulqa")
    torch.manual_seed(0)
    logits = torch.randn(7, 11)
    input_ids = torch.tensor([1, 3, 5, 7, 2, 4, 6])

    def legacy_nll(seq_len: int, n_answer_tokens: int) -> float:
        full_logp = torch.log_softmax(logits[:seq_len - 1].float(), dim=-1)
        targets = input_ids[1:seq_len]
        tok_logp = full_logp[torch.arange(full_logp.shape[0]), targets]
        ans_logp = tok_logp[-n_answer_tokens:] if n_answer_tokens <= tok_logp.shape[0] else tok_logp
        return float((-ans_logp.mean()).item())

    assert module._answer_nll_from_logits(logits, input_ids, 7, 2) == pytest.approx(legacy_nll(7, 2))
    assert module._answer_nll_from_logits(logits, input_ids, 7, 7) == pytest.approx(legacy_nll(7, 7))


def test_eval_truthfulqa_prefix_kv_cache_matches_full_sequence_reps():
    module = importlib.import_module("benchmarks.eval_truthfulqa")

    class NoCallTokenizer:
        pad_token_id = 0
        eos_token_id = 2

        def __call__(self, *_args, **_kwargs):
            raise AssertionError("precomputed encodings should bypass tokenizer calls")

    class PrefixCacheModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.prefix_calls = 0
            self.answer_calls = 0
            self.full_calls = 0

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            output_hidden_states=False,
            use_cache=False,
            past_key_values=None,
            **_kwargs,
        ):
            del attention_mask
            batch_size, current_len = input_ids.shape
            vocab_size = 32
            hidden_rows = []
            logits_rows = []
            for row in range(batch_size):
                if past_key_values is None:
                    prefix_ids = torch.empty(0, dtype=torch.long, device=input_ids.device)
                    self.full_calls += int(not use_cache)
                else:
                    prefix_ids = past_key_values[0].to(input_ids.device)
                    self.answer_calls += 1
                if use_cache:
                    self.prefix_calls += 1
                full_ids = torch.cat([prefix_ids, input_ids[row]])
                row_hidden = []
                row_logits = []
                for pos in range(current_len):
                    full_pos = len(prefix_ids) + pos
                    prefix_sum = float(full_ids[:full_pos + 1].sum().item())
                    base = prefix_sum + float(full_pos)
                    row_hidden.append(torch.tensor([base, base + 1, base + 2, base + 3]))
                    row_logits.append(torch.arange(vocab_size, dtype=torch.float32) * 0.01 + base)
                hidden_rows.append(torch.stack(row_hidden))
                logits_rows.append(torch.stack(row_logits))
            return SimpleNamespace(
                logits=torch.stack(logits_rows),
                hidden_states=(torch.stack(hidden_rows),) if output_hidden_states else None,
                past_key_values=(input_ids[0].detach().cpu(),) if use_cache else None,
            )

    statements = [
        module.Statement("Q?", "A", 0),
        module.Statement("Q?", "B C", 1),
    ]
    encodings = [
        module.StatementEncoding((1, 2, 10), 1),
        module.StatementEncoding((1, 2, 11, 12), 2),
    ]
    full_model = PrefixCacheModel()
    prefix_model = PrefixCacheModel()

    full_reps = module.batched_statement_reps(
        full_model,
        NoCallTokenizer(),
        statements,
        [0],
        torch.device("cpu"),
        8,
        encoded_statements=encodings,
    )
    prefix_reps = module.batched_statement_reps(
        prefix_model,
        NoCallTokenizer(),
        statements,
        [0],
        torch.device("cpu"),
        8,
        encoded_statements=encodings,
        prefix_kv_cache=True,
    )

    assert prefix_model.prefix_calls == 1
    assert prefix_model.answer_calls == 2
    assert full_model.prefix_calls == 0
    for full_rep, prefix_rep in zip(full_reps, prefix_reps):
        assert torch.allclose(prefix_rep["last"][0], full_rep["last"][0])
        assert torch.allclose(prefix_rep["ans_hs"], full_rep["ans_hs"])
        assert prefix_rep["nll"] == pytest.approx(full_rep["nll"])
        assert prefix_rep["eigenscore_by_layer"][0] == pytest.approx(full_rep["eigenscore_by_layer"][0])


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

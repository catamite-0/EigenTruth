import importlib
import json
from types import SimpleNamespace

import pytest


def test_build_source_family_qa_corpus_extracts_only_structured_metadata(tmp_path):
    module = importlib.import_module("benchmarks.build_source_family_qa_corpus")
    registry_module = importlib.import_module("eigentruth.registry")

    source_path = tmp_path / "source-family-results.jsonl"
    output_path = tmp_path / "source-family-qa-corpus.json"
    report_path = tmp_path / "source-family-qa-corpus-report.json"
    manifest_path = tmp_path / "artifact-manifest.json"
    registry_path = tmp_path / "registry.json"

    source_path.write_text(
        "\n".join(
            [
                json.dumps({
                    "request_id": "not-copied",
                    "results": [
                        {
                            "provider": "wikidata",
                            "source_family": "reference",
                            "source": "wikidata:Q478214:P112",
                            "title": "Tesla Motors - founder",
                            "url": "https://www.wikidata.org/wiki/Q478214",
                            "metadata": {
                                "provider": "wikidata",
                                "statement_property": "P112",
                                "statement_property_label": "founder",
                                "subject": "Tesla Motors",
                                "subject_qid": "Q478214",
                                "value": "Martin Eberhard",
                                "value_qid": "Q92743",
                                "retrieved_at": "2026-06-28T00:00:00+00:00",
                            },
                        },
                        {
                            "provider": "worldbank",
                            "source_family": "official_statistics",
                            "source": "worldbank:SP.POP.TOTL:AFG:2024",
                            "url": "https://data.worldbank.org/indicator/SP.POP.TOTL?locations=AF",
                            "metadata": {
                                "provider": "worldbank",
                                "country_name": "Afghanistan",
                                "country_code_iso3": "AFG",
                                "indicator": "SP.POP.TOTL",
                                "indicator_name": "Population, total",
                                "reference_year": "2024",
                                "value": 42647492,
                            },
                        },
                        {
                            "provider": "crossref",
                            "source_family": "scholarly",
                            "metadata": {"provider": "crossref", "title": "A paper"},
                        },
                        {
                            "provider": "wikidata",
                            "source_family": "reference",
                            "metadata": {
                                "provider": "wikidata",
                                "statement_property": "P31",
                                "statement_property_label": "instance of",
                                "subject": "Example",
                                "value": "Q123",
                            },
                        },
                        {
                            "provider": "wikidata",
                            "source_family": "reference",
                            "metadata": {
                                "provider": "wikidata",
                                "statement_property": "P50",
                                "statement_property_label": "author",
                                "subject": "Unsafe Row",
                                "value": "Ada Example",
                                "label": 1,
                            },
                        },
                    ],
                })
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    payload = module.run(
        source_paths=(source_path,),
        output_path=output_path,
        report_json_path=report_path,
        artifact_manifest_path=manifest_path,
        registry_path=registry_path,
        name="source-family-qa-unit",
        version="0.1",
        metadata={"suite": "unit"},
    )

    corpus = payload["corpus"]
    report = payload["report"]
    questions = {document["question"]: document for document in corpus["documents"]}
    record = registry_module.ArtifactRegistry.load_json(registry_path).get("report:source-family-qa-unit:0.1")

    assert report["status"] == "ready"
    assert corpus["corpus_type"] == "source_family_structured_qa_external_evidence"
    assert corpus["summary"]["n_documents"] == 2
    assert corpus["summary"]["n_candidate_documents"] == 3
    assert corpus["summary"]["by_provider"] == {"wikidata": 1, "worldbank": 1}
    assert corpus["summary"]["skipped"]["unsupported_provider"] == 1
    assert corpus["summary"]["skipped"]["qid_values"] == 1
    assert corpus["summary"]["skipped"]["reserved_metadata"] == 1
    assert questions["What does Wikidata list as the founder for Tesla Motors?"]["answer"] == "Martin Eberhard"
    assert (
        questions["What does the World Bank list as Population, total for Afghanistan in 2024?"]["answer"]
        == "42,647,492"
    )
    for document in corpus["documents"]:
        assert "label" not in document["metadata"]
        assert "request_id" not in document["metadata"]
        assert "model_answer" not in document["metadata"]

    assert registry_module.load_and_verify_artifact_manifest(manifest_path).passed is True
    assert record.metadata["workflow"] == "source_family_structured_qa_corpus_builder"
    assert record.metadata["document_count"] == 2
    assert record.metadata["suite"] == "unit"


def test_source_family_structured_qa_route_workflow_audits_covered_facts(tmp_path):
    module = importlib.import_module("benchmarks.run_source_family_structured_qa_route_workflow")
    registry_module = importlib.import_module("eigentruth.registry")

    qa_path = tmp_path / "source-family-qa-corpus.json"
    output_dir = tmp_path / "workflow"
    registry_path = tmp_path / "registry.json"
    qa_path.write_text(
        json.dumps({
            "corpus_type": "source_family_structured_qa_external_evidence",
            "source": {
                "builder": "source_family_structured_qa_corpus_builder",
                "accepted_providers": ["wikidata", "worldbank"],
            },
            "summary": {
                "n_documents": 4,
                "by_provider": {"wikidata": 2, "worldbank": 2},
                "by_source_family": {"official_statistics": 2, "reference": 2},
            },
            "documents": [
                {
                    "question": "What does Wikidata list as the founder for Tesla Motors?",
                    "answer": "Martin Eberhard",
                    "source": "wikidata:Q478214:P112:Q92743",
                    "metadata": {
                        "provider": "wikidata",
                        "source_family": "reference",
                        "statement_property": "P112",
                        "statement_property_label": "founder",
                    },
                },
                {
                    "question": "What does Wikidata list as the founder for OpenAI?",
                    "answer": "Sam Altman",
                    "source": "wikidata:Q21708200:P112:Q565549",
                    "metadata": {
                        "provider": "wikidata",
                        "source_family": "reference",
                        "statement_property": "P112",
                        "statement_property_label": "founder",
                    },
                },
                {
                    "question": "What does the World Bank list as Population, total for Afghanistan in 2024?",
                    "answer": "42,647,492",
                    "source": "worldbank:SP.POP.TOTL:AFG:2024",
                    "metadata": {
                        "provider": "worldbank",
                        "source_family": "official_statistics",
                        "indicator": "SP.POP.TOTL",
                        "indicator_name": "Population, total",
                        "country_name": "Afghanistan",
                        "reference_year": "2024",
                    },
                },
                {
                    "question": "What does the World Bank list as Population, total for Albania in 2024?",
                    "answer": "2,402,113",
                    "source": "worldbank:SP.POP.TOTL:ALB:2024",
                    "metadata": {
                        "provider": "worldbank",
                        "source_family": "official_statistics",
                        "indicator": "SP.POP.TOTL",
                        "indicator_name": "Population, total",
                        "country_name": "Albania",
                        "reference_year": "2024",
                    },
                },
            ],
        }),
        encoding="utf-8",
    )

    summary = module.run(
        SimpleNamespace(
            qa_corpus=str(qa_path),
            output_dir=str(output_dir),
            score_name="source-family-covered-facts-unit",
            signal="truth_proj",
            alpha=0.2,
            seed=0,
            limit=None,
            score_dump_json=None,
            verifier_report_json=None,
            verified_records_jsonl=None,
            artifact_manifest=None,
            json=None,
            registry=str(registry_path),
            name="source-family-structured-qa-route-unit",
            version="0.1",
            metadata=("suite=unit",),
            compact_json=False,
        )
    )

    score_dump = json.loads((output_dir / "covered-facts-scores.json").read_text(encoding="utf-8"))
    report = json.loads((output_dir / "structured-qa-verifier-report.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "artifact-manifest.json").read_text(encoding="utf-8"))
    record = registry_module.ArtifactRegistry.load_json(registry_path).get(
        "report:source-family-structured-qa-route-unit:0.1"
    )

    assert summary["status"] == "promote"
    assert summary["selected_route_counts"] == {"structured_qa": 8}
    assert summary["structured_qa_metrics"]["decision_accuracy"] == pytest.approx(1.0)
    assert summary["structured_qa_metrics"]["false_refuted_rate"] == pytest.approx(1.0)
    assert summary["structured_qa_metrics"]["false_supported_rate"] == pytest.approx(0.0)
    assert score_dump["summary"]["n_records"] == 8
    assert score_dump["summary"]["n_true"] == 4
    assert score_dump["summary"]["n_false"] == 4
    assert score_dump["summary"]["by_provider"] == {"wikidata": 2, "worldbank": 2}
    assert set(score_dump["summary"]["by_fact_group"]) == {
        "wikidata:reference:p112",
        "worldbank:official_statistics:sp_pop_totl",
    }
    assert report["qa_verifier"]["enabled"] is True
    assert summary["provider_metrics"]["wikidata"]["false_refuted_rate"] == pytest.approx(1.0)
    assert summary["provider_metrics"]["worldbank"]["decision_accuracy"] == pytest.approx(1.0)
    assert summary["source_family_metrics"]["official_statistics"]["n_records"] == 4
    assert summary["fact_group_metrics"]["worldbank:official_statistics:sp_pop_totl"]["n_records"] == 4
    assert manifest["metadata"]["workflow"] == "source_family_structured_qa_route_workflow"
    assert manifest["metadata"]["promotes_covered_facts_route"] is True
    assert manifest["metadata"]["provider_count"] == 2
    assert manifest["metadata"]["fact_group_count"] == 2
    assert manifest["metadata"]["suite"] == "unit"
    assert registry_module.load_and_verify_artifact_manifest(output_dir / "artifact-manifest.json").passed is True
    assert record.metadata["workflow"] == "source_family_structured_qa_route_workflow"
    assert record.metadata["status"] == "promote"
    assert record.metadata["suite"] == "unit"


def test_source_family_structured_qa_claim_mapping_audits_claim_coverage(tmp_path):
    module = importlib.import_module("benchmarks.audit_source_family_structured_qa_claim_mapping")
    registry_module = importlib.import_module("eigentruth.registry")

    claims_path = tmp_path / "claims.json"
    qa_path = tmp_path / "source-family-qa-corpus.json"
    route_summary_path = tmp_path / "structured-qa-route-summary.json"
    output_path = tmp_path / "claim-mapping.json"
    manifest_path = tmp_path / "artifact-manifest.json"
    registry_path = tmp_path / "registry.json"

    qa_path.write_text(
        json.dumps({
            "corpus_type": "source_family_structured_qa_external_evidence",
            "documents": [
                {
                    "question": "What does Wikidata list as the founder for Tesla Motors?",
                    "answer": "Martin Eberhard",
                    "source": "wikidata:Q478214:P112:Q92743",
                    "metadata": {
                        "provider": "wikidata",
                        "source_family": "reference",
                        "statement_property": "P112",
                        "statement_property_label": "founder",
                        "subject": "Tesla Motors",
                        "subject_qid": "Q478214",
                    },
                },
                {
                    "question": "What does the World Bank list as Population, total for Afghanistan in 2024?",
                    "answer": "42,647,492",
                    "source": "worldbank:SP.POP.TOTL:AFG:2024",
                    "metadata": {
                        "provider": "worldbank",
                        "source_family": "official_statistics",
                        "indicator": "SP.POP.TOTL",
                        "indicator_name": "Population, total",
                        "country_name": "Afghanistan",
                        "country_code_iso3": "AFG",
                        "reference_year": "2024",
                    },
                },
            ],
        }),
        encoding="utf-8",
    )
    claims_path.write_text(
        json.dumps({
            "records": [
                {
                    "record_index": 10,
                    "question": "Who first started Tesla Motors?",
                    "answer": "Elon Musk",
                    "text": "Who first started Tesla Motors? Elon Musk",
                    "label": 1,
                    "question_type": "person",
                },
                {
                    "record_index": 11,
                    "question": "Who first started Tesla Motors?",
                    "answer": "Martin Eberhard",
                    "text": "Who first started Tesla Motors? Martin Eberhard",
                    "label": 0,
                    "question_type": "person",
                },
                {
                    "record_index": 12,
                    "question": "What is the population of Afghanistan in 2024?",
                    "answer": "1,000",
                    "text": "What is the population of Afghanistan in 2024? 1,000",
                    "label": 1,
                    "question_type": "quantity",
                },
                {
                    "record_index": 13,
                    "question": "What is the capital of France?",
                    "answer": "Paris",
                    "text": "What is the capital of France? Paris",
                    "label": 0,
                    "question_type": "location",
                },
            ]
        }),
        encoding="utf-8",
    )
    route_summary_path.write_text(
        json.dumps({
            "workflow": "source_family_structured_qa_route_workflow",
            "status": "promote",
            "route": "structured_qa",
        }),
        encoding="utf-8",
    )

    payload = module.run(
        claims_path=claims_path,
        qa_corpus_path=qa_path,
        route_summary_path=route_summary_path,
        output_path=output_path,
        artifact_manifest_path=manifest_path,
        registry_path=registry_path,
        name="source-family-qa-claim-mapping-unit",
        version="0.1",
        metadata={"suite": "unit"},
    )

    decisions = {
        int(record["record_index"]): record["mapping_decision"]
        for record in payload["records"]
    }
    record = registry_module.ArtifactRegistry.load_json(registry_path).get(
        "report:source-family-qa-claim-mapping-unit:0.1"
    )

    assert payload["status"] == "observed"
    assert payload["source"]["route_summary_promoted"] is True
    assert payload["summary"]["target_count"] == 4
    assert payload["summary"]["covered_fact_match_count"] == 3
    assert payload["summary"]["mapped_qa_fact_candidate_count"] == 2
    assert payload["summary"]["answer_value_supported_count"] == 1
    assert decisions[10] == "mapped_qa_fact_candidate"
    assert decisions[11] == "answer_value_supported_by_covered_fact"
    assert decisions[12] == "mapped_qa_fact_candidate"
    assert decisions[13] == "no_candidate_fact"
    assert payload["records"][0]["matched_provider_counts"] == {"wikidata": 1}
    assert payload["records"][2]["matched_source_family_counts"] == {"official_statistics": 1}
    assert registry_module.load_and_verify_artifact_manifest(manifest_path).passed is True
    assert record.metadata["workflow"] == "source_family_structured_qa_claim_mapping_audit"
    assert record.metadata["mapped_qa_fact_candidate_count"] == 2
    assert record.metadata["suite"] == "unit"


def test_source_family_structured_qa_correction_handoff_promotes_mapped_candidates(tmp_path):
    module = importlib.import_module("benchmarks.build_source_family_structured_qa_correction_handoff")
    registry_module = importlib.import_module("eigentruth.registry")

    claim_mapping_path = tmp_path / "claim-mapping.json"
    output_dir = tmp_path / "handoff"
    registry_path = tmp_path / "registry.json"
    claim_mapping_path.write_text(
        json.dumps({
            "schema_version": 1,
            "workflow": "source_family_structured_qa_claim_mapping_audit",
            "status": "observed",
            "source": {
                "route_summary_status": "promote",
                "route_summary_promoted": True,
            },
            "summary": {
                "target_count": 2,
                "mapped_qa_fact_candidate_count": 1,
            },
            "records": [
                {
                    "record_id": "record-10",
                    "record_index": 10,
                    "question": "Who first started Acme Motors?",
                    "answer": "Alice founded Acme.",
                    "text": "Who first started Acme Motors? Alice founded Acme.",
                    "label": 1,
                    "mapping_decision": "mapped_qa_fact_candidate",
                    "mapped_qa_fact_candidate": True,
                    "gate_recommendation": "structured_qa_correction_handoff",
                    "best_mapping_score": 0.91,
                    "best_subject_coverage": 1.0,
                    "best_intent_score": 0.8,
                    "mapped_facts": [
                        {
                            "question": "What does Wikidata list as the founder for Acme Motors?",
                            "answer": "Bob Builder",
                            "source": "wikidata:Q1:P112:Q2",
                            "provider": "wikidata",
                            "source_family": "reference",
                            "fact_type": "P112",
                            "subject": "Acme Motors",
                            "mapping_score": 0.91,
                            "metadata": {
                                "provider": "wikidata",
                                "source_family": "reference",
                                "statement_property": "P112",
                                "statement_property_label": "founder",
                                "subject": "Acme Motors",
                                "subject_qid": "Q1",
                                "url": "https://www.wikidata.org/wiki/Q1",
                            },
                        }
                    ],
                },
                {
                    "record_id": "record-11",
                    "record_index": 11,
                    "question": "Who first started Acme Motors?",
                    "answer": "Bob Builder",
                    "text": "Who first started Acme Motors? Bob Builder",
                    "label": 0,
                    "mapping_decision": "answer_value_supported_by_covered_fact",
                    "mapped_qa_fact_candidate": False,
                    "gate_recommendation": "answer_support_audit",
                    "mapped_facts": [],
                },
            ],
        }),
        encoding="utf-8",
    )

    payload = module.run(
        claim_mapping_path=claim_mapping_path,
        output_dir=output_dir,
        registry_path=registry_path,
        name="source-family-qa-correction-handoff-unit",
        version="0.1",
        metadata={"suite": "unit"},
    )
    report = payload["report"]
    corpus = json.loads(
        (output_dir / "source-family-structured-qa-correction-corpus.json").read_text(
            encoding="utf-8"
        )
    )
    traces = [
        json.loads(line)
        for line in (output_dir / "product-traces.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    action_results = [
        json.loads(line)
        for line in (output_dir / "action-results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    registry = registry_module.ArtifactRegistry.load_json(registry_path)
    report_record = registry.get("report:source-family-qa-correction-handoff-unit:0.1")
    trace_record = registry.get("product_trace:source-family-qa-correction-handoff-unit:0.1")
    action_record = registry.get("action_result:source-family-qa-correction-handoff-unit:0.1")

    assert report["status"] == "promote"
    assert report["source"]["route_summary_promoted"] is True
    assert report["summary"]["input_record_count"] == 2
    assert report["summary"]["correction_candidate_count"] == 1
    assert report["summary"]["corpus_document_count"] == 1
    assert report["summary"]["verification_status_counts"] == {"refuted": 1}
    assert report["summary"]["action_counts"] == {"abstain": 1}
    assert corpus["corpus_type"] == "target_specific_source_family_structured_qa_correction"
    assert corpus["documents"][0]["question"] == "Who first started Acme Motors?"
    assert corpus["documents"][0]["answer"] == "Bob Builder"
    assert corpus["documents"][0]["metadata"]["provider"] == "wikidata"
    assert corpus["documents"][0]["metadata"]["statement_property"] == "P112"
    assert "label" not in corpus["documents"][0]["metadata"]
    assert traces[0]["verification_results"][0]["status"] == "refuted"
    assert traces[0]["verification_results"][0]["metadata"]["selected_route"] == (
        "source_family_structured_qa_correction"
    )
    assert traces[0]["risk_decision"]["action"] == "abstain"
    assert traces[0]["risk_decision"]["risk_level"] == "high"
    assert traces[0]["action_results"][0]["status"] == "dry_run"
    assert action_results[0]["action"] == "abstain"
    assert registry_module.load_and_verify_artifact_manifest(output_dir / "artifact-manifest.json").passed is True
    assert report_record.metadata["workflow"] == "source_family_structured_qa_correction_handoff"
    assert trace_record.metadata["trace_count"] == 1
    assert action_record.metadata["action_result_count"] == 1


def test_source_family_structured_qa_correction_handoff_requires_promoted_route(tmp_path):
    module = importlib.import_module("benchmarks.build_source_family_structured_qa_correction_handoff")

    claim_mapping = {
        "workflow": "source_family_structured_qa_claim_mapping_audit",
        "status": "blocked",
        "source": {
            "route_summary_status": "blocked",
            "route_summary_promoted": False,
        },
        "summary": {
            "target_count": 1,
            "mapped_qa_fact_candidate_count": 1,
        },
        "records": [
            {
                "record_id": "record-1",
                "record_index": 1,
                "question": "Who first started Acme Motors?",
                "answer": "Alice",
                "mapping_decision": "mapped_qa_fact_candidate",
                "mapped_qa_fact_candidate": True,
                "gate_recommendation": "structured_qa_correction_handoff",
                "mapped_facts": [
                    {
                        "question": "What does Wikidata list as the founder for Acme Motors?",
                        "answer": "Bob",
                        "source": "wikidata:Q1:P112:Q2",
                        "metadata": {
                            "provider": "wikidata",
                            "source_family": "reference",
                            "statement_property": "P112",
                            "subject": "Acme Motors",
                        },
                    }
                ],
            }
        ],
    }

    payload = module.build_source_family_structured_qa_correction_handoff(claim_mapping)

    assert payload["report"]["status"] == "blocked"
    assert payload["report"]["source"]["route_summary_promoted"] is False
    assert payload["report"]["summary"]["correction_candidate_count"] == 0
    assert payload["qa_corpus"]["summary"]["n_documents"] == 0
    assert payload["product_traces"] == ()
    assert payload["action_results"] == ()


def test_source_family_structured_qa_fact_expansion_plans_mapping_gaps(tmp_path):
    module = importlib.import_module("benchmarks.plan_source_family_structured_qa_fact_expansion")
    registry_module = importlib.import_module("eigentruth.registry")

    claim_mapping_path = tmp_path / "claim-mapping.json"
    output_path = tmp_path / "fact-expansion-plan.json"
    manifest_path = tmp_path / "artifact-manifest.json"
    registry_path = tmp_path / "registry.json"
    claim_mapping_path.write_text(
        json.dumps({
            "schema_version": 1,
            "workflow": "source_family_structured_qa_claim_mapping_audit",
            "status": "blocked",
            "source": {
                "route_summary_status": "promote",
                "route_summary_promoted": True,
                "qa_document_count": 2,
            },
            "summary": {
                "target_count": 6,
                "covered_fact_match_count": 0,
            },
            "records": [
                {
                    "record_id": "row-1",
                    "record_index": 1,
                    "question": "What is Alpha Syndrome?",
                    "answer": "A moon.",
                    "text": "What is Alpha Syndrome? A moon.",
                    "question_type": "definition",
                    "mapping_decision": "no_candidate_fact",
                    "gate_recommendation": "source_family_coverage_expansion",
                    "top_fact_candidates": [],
                },
                {
                    "record_id": "row-2",
                    "record_index": 2,
                    "question": "Who founded Beta Labs?",
                    "answer": "Ada Beta.",
                    "text": "Who founded Beta Labs? Ada Beta.",
                    "question_type": "person",
                    "mapping_decision": "subject_only_or_missing_intent",
                    "top_fact_candidates": [
                        {
                            "provider": "wikidata",
                            "source_family": "reference",
                            "fact_type": "p31",
                            "subject": "Beta Labs",
                            "intent_terms": ["founder"],
                            "mapping_score": 0.4,
                        }
                    ],
                },
                {
                    "record_id": "row-3",
                    "record_index": 3,
                    "question": "How many moons does Gamma have?",
                    "answer": "Twelve.",
                    "text": "How many moons does Gamma have? Twelve.",
                    "question_type": "quantity",
                    "mapping_decision": "intent_only_or_missing_subject",
                    "top_fact_candidates": [
                        {
                            "provider": "worldbank",
                            "source_family": "official_statistics",
                            "fact_type": "sp_pop_totl",
                            "subject": "",
                            "intent_terms": ["population"],
                        }
                    ],
                },
                {
                    "record_id": "row-4",
                    "record_index": 4,
                    "question": "Why does Delta process fail?",
                    "answer": "Because of pressure.",
                    "text": "Why does Delta process fail? Because of pressure.",
                    "question_type": "causal",
                    "mapping_decision": "weak_textual_overlap",
                    "top_fact_candidates": [
                        {
                            "provider": "crossref",
                            "source_family": "scholarly",
                            "fact_type": "causal_citation",
                            "subject": "Delta process",
                            "weak_textual_overlap": 0.22,
                        }
                    ],
                },
                {
                    "record_id": "row-5",
                    "record_index": 5,
                    "question": "Who started Epsilon Motors?",
                    "answer": "Epsilon Motors.",
                    "text": "Who started Epsilon Motors? Epsilon Motors.",
                    "question_type": "person",
                    "mapping_decision": "answer_entity_collision",
                    "collision_facts": [
                        {
                            "provider": "wikidata",
                            "source_family": "reference",
                            "fact_type": "p112",
                            "subject": "Epsilon Motors",
                            "answer_subject_overlap": 1.0,
                        }
                    ],
                },
                {
                    "record_id": "row-6",
                    "record_index": 6,
                    "question": "Who founded Covered Labs?",
                    "answer": "Ada Covered.",
                    "text": "Who founded Covered Labs? Ada Covered.",
                    "question_type": "person",
                    "mapping_decision": "mapped_qa_fact_candidate",
                    "mapped_facts": [],
                },
            ],
        }),
        encoding="utf-8",
    )

    payload = module.run(
        claim_mapping_path=claim_mapping_path,
        output_path=output_path,
        artifact_manifest_path=manifest_path,
        registry_path=registry_path,
        name="source-family-fact-expansion-unit",
        version="0.1",
        metadata={"suite": "unit"},
    )
    record = registry_module.ArtifactRegistry.load_json(registry_path).get(
        "report:source-family-fact-expansion-unit:0.1"
    )
    by_index = {item["record_index"]: item for item in payload["targets"]}
    task_counts = payload["summary"]["task_type_counts"]

    assert payload["status"] == "ready_for_collection"
    assert payload["source"]["claim_mapping_status"] == "blocked"
    assert payload["source"]["route_summary_promoted"] is True
    assert payload["summary"]["input_record_count"] == 6
    assert payload["summary"]["target_count"] == 5
    assert payload["summary"]["skipped_resolved_count"] == 1
    assert payload["summary"]["gap_type_counts"]["missing_subject_and_intent"] == 1
    assert task_counts["source_family_structured_fact_request"] == 5
    assert task_counts["external_citation_request"] == 3
    assert task_counts["entity_resolution_request"] == 3
    assert task_counts["source_family_fact_disambiguation"] == 2
    assert task_counts["world_model_or_calculator_rule_request"] == 3
    assert "Alpha Syndrome" in by_index[1]["entity_candidates"]
    assert "Beta Labs" in by_index[2]["entity_candidates"]
    assert by_index[3]["world_model_rule_targets"][0]["rule_family"] == "quantity_or_arithmetic"
    assert by_index[4]["world_model_rule_targets"][0]["rule_family"] == "causal_or_procedural"
    assert by_index[5]["world_model_rule_targets"][0]["rule_family"] == "entity_disambiguation"
    assert all("label" not in target for target in payload["targets"])
    assert payload["label_usage"]["tasks_are_verifier_evidence"] is False
    assert registry_module.load_and_verify_artifact_manifest(manifest_path).passed is True
    assert record.metadata["workflow"] == "source_family_structured_qa_fact_expansion_plan"
    assert record.metadata["target_count"] == 5
    assert record.metadata["suite"] == "unit"


def test_source_family_structured_qa_fact_collection_compiles_request_sidecars(tmp_path):
    module = importlib.import_module("benchmarks.build_source_family_structured_qa_fact_collection_corpus")
    registry_module = importlib.import_module("eigentruth.registry")

    plan_path = tmp_path / "fact-expansion-plan.json"
    output_dir = tmp_path / "collection"
    manifest_path = output_dir / "artifact-manifest.json"
    registry_path = tmp_path / "registry.json"
    plan_path.write_text(
        json.dumps({
            "schema_version": 1,
            "workflow": "source_family_structured_qa_fact_expansion_plan",
            "status": "ready_for_collection",
            "summary": {"target_count": 2},
            "targets": [
                {
                    "target_id": "record-1",
                    "record_index": 1,
                    "priority": "high",
                    "question_type": "definition",
                    "question": "What is Alpha Syndrome?",
                    "answer": "A moon.",
                    "mapping_decision": "no_candidate_fact",
                    "gap_type": "missing_subject_and_intent",
                    "entity_candidates": ["Alpha Syndrome"],
                    "wikidata_property_hints": ["description", "instance_of:P31"],
                    "source_family_targets": [
                        {
                            "provider": "wikidata",
                            "source_family": "reference",
                            "reason": "unit",
                        },
                        {
                            "provider": "source_family_adapter",
                            "source_family": "official_site",
                            "reason": "unit",
                        },
                    ],
                    "query_seeds": [
                        "Alpha Syndrome A moon",
                        "What is Alpha Syndrome?",
                    ],
                    "world_model_rule_targets": [
                        {
                            "rule_family": "entity_disambiguation",
                            "reason": "unit",
                        }
                    ],
                    "collection_tasks": [
                        {"task_type": "entity_resolution_request"},
                        {"task_type": "source_family_structured_fact_request"},
                        {"task_type": "external_citation_request"},
                        {"task_type": "world_model_or_calculator_rule_request"},
                    ],
                },
                {
                    "target_id": "record-2",
                    "record_index": 2,
                    "priority": "medium",
                    "question_type": "person",
                    "question": "Who started Beta Labs?",
                    "answer": "Beta Labs.",
                    "mapping_decision": "answer_entity_collision",
                    "gap_type": "answer_entity_collision",
                    "entity_candidates": ["Beta Labs"],
                    "wikidata_property_hints": ["founded_by:P112"],
                    "source_family_targets": [
                        {
                            "provider": "wikidata",
                            "source_family": "reference",
                            "reason": "unit",
                        }
                    ],
                    "nearest_fact_candidates": [
                        {
                            "provider": "wikidata",
                            "source_family": "reference",
                            "fact_type": "p112",
                            "subject": "Beta Labs",
                            "mapping_score": 0.42,
                        }
                    ],
                    "world_model_rule_targets": [
                        {
                            "rule_family": "entity_disambiguation",
                            "reason": "unit",
                        }
                    ],
                    "collection_tasks": [
                        {"task_type": "source_family_structured_fact_request"},
                        {"task_type": "source_family_fact_disambiguation"},
                        {"task_type": "world_model_or_calculator_rule_request"},
                    ],
                },
            ],
        }),
        encoding="utf-8",
    )

    payload = module.run(
        plan_path=plan_path,
        output_dir=output_dir,
        artifact_manifest_path=manifest_path,
        registry_path=registry_path,
        name="source-family-fact-collection-unit",
        version="0.1",
        metadata={"suite": "unit"},
    )
    record = registry_module.ArtifactRegistry.load_json(registry_path).get(
        "report:source-family-fact-collection-unit:0.1"
    )
    requests = payload["requests"]
    citation_rows = [
        json.loads(line)
        for line in (output_dir / "citation-requests.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    all_requests = [
        request
        for bucket in requests.values()
        for request in bucket
    ]

    assert payload["status"] == "ready_for_collection"
    assert payload["summary"]["target_count"] == 2
    assert payload["summary"]["request_counts"]["source_family_structured_fact"] == 3
    assert payload["summary"]["request_counts"]["entity_resolution"] == 1
    assert payload["summary"]["request_counts"]["external_citation"] == 3
    assert payload["summary"]["request_counts"]["source_family_fact_disambiguation"] == 1
    assert payload["summary"]["request_counts"]["world_model_or_calculator_rule"] == 2
    assert payload["summary"]["source_discovery_document_count"] == 7
    assert len(citation_rows) == 3
    assert all("A moon" not in row["query"] for row in citation_rows)
    assert all("model_answer" not in request for request in all_requests)
    assert all("answer" not in request for request in all_requests)
    assert all("label" not in request for request in all_requests)
    assert payload["label_usage"]["model_answers_copied_to_collection_requests"] is False
    assert (output_dir / "structured-fact-requests.jsonl").exists()
    assert registry_module.load_and_verify_artifact_manifest(manifest_path).passed is True
    assert record.metadata["workflow"] == "source_family_structured_qa_fact_collection_corpus"
    assert record.metadata["total_request_count"] == payload["summary"]["total_request_count"]
    assert record.metadata["suite"] == "unit"


def test_source_family_structured_qa_fact_collection_workflow_executes_local_catalog(tmp_path):
    module = importlib.import_module("benchmarks.run_source_family_structured_qa_fact_collection_workflow")
    registry_module = importlib.import_module("eigentruth.registry")

    collection_path = tmp_path / "fact-collection-corpus.json"
    source_catalog_path = tmp_path / "source-family-catalog.jsonl"
    output_dir = tmp_path / "workflow"
    registry_path = tmp_path / "registry.json"

    collection_path.write_text(
        json.dumps({
            "schema_version": 1,
            "workflow": "source_family_structured_qa_fact_collection_corpus",
            "status": "ready_for_collection",
            "summary": {
                "target_count": 1,
                "total_request_count": 5,
            },
            "requests": {
                "source_family_structured_fact": [
                    {
                        "request_id": "sfact:record-1:1",
                        "target_id": "record-1",
                        "request_type": "source_family_structured_fact",
                        "priority": "high",
                        "question_type": "person",
                        "gap_type": "missing_property_or_indicator",
                        "question": "Who founded Beta Labs?",
                        "query": "Who founded Beta Labs Beta Labs founder P112 reference",
                        "source_family": "reference",
                        "provider_hint": "wikidata",
                        "entity_candidates": ["Beta Labs"],
                        "property_hints": ["founder", "P112"],
                        "usage": "source_discovery_only",
                        "not_verifier_evidence": True,
                    }
                ],
                "entity_resolution": [
                    {
                        "request_id": "entity:record-1:1",
                        "target_id": "record-1",
                        "request_type": "entity_resolution",
                        "priority": "high",
                        "question_type": "person",
                        "gap_type": "missing_property_or_indicator",
                        "question": "Who founded Beta Labs?",
                        "entity": "Beta Labs",
                        "query": "Beta Labs founder",
                        "property_hints": ["P112"],
                        "usage": "source_discovery_only",
                        "not_verifier_evidence": True,
                    }
                ],
                "external_citation": [
                    {
                        "request_id": "cite:record-1:1",
                        "target_id": "record-1",
                        "request_type": "external_citation",
                        "priority": "high",
                        "question_type": "person",
                        "gap_type": "missing_property_or_indicator",
                        "question": "Who founded Beta Labs?",
                        "query": "Beta Labs founder",
                        "source_family_hints": ["reference"],
                        "usage": "source_discovery_only",
                        "not_verifier_evidence": True,
                    }
                ],
                "source_family_fact_disambiguation": [
                    {
                        "request_id": "disambig:record-1:1",
                        "target_id": "record-1",
                        "request_type": "source_family_fact_disambiguation",
                        "priority": "high",
                        "question_type": "person",
                        "gap_type": "answer_entity_collision",
                        "question": "Who founded Beta Labs?",
                        "query": "Beta Labs founder disambiguation",
                        "entities": ["Beta Labs"],
                        "property_hints": ["P112"],
                        "usage": "source_discovery_only",
                        "not_verifier_evidence": True,
                    }
                ],
                "world_model_or_calculator_rule": [
                    {
                        "request_id": "rule:record-1:1",
                        "target_id": "record-1",
                        "request_type": "world_model_or_calculator_rule",
                        "priority": "high",
                        "question_type": "person",
                        "gap_type": "answer_entity_collision",
                        "question": "Who founded Beta Labs?",
                        "rule_family": "entity_disambiguation",
                        "rule_reason": "unit",
                        "rule_seed": "Check founder entity for Beta Labs",
                        "required_inputs": ["subject", "founder"],
                        "usage": "source_discovery_only",
                        "not_verifier_evidence": True,
                    }
                ],
            },
        }),
        encoding="utf-8",
    )
    source_catalog_path.write_text(
        json.dumps({
            "provider": "wikidata",
            "source_family": "reference",
            "source": "wikidata:QUNIT:P112:QADA",
            "title": "Beta Labs founder",
            "text": "Beta Labs founder Ada Beta reference P112.",
            "url": "https://www.wikidata.org/wiki/QUNIT",
            "metadata": {
                "provider": "wikidata",
                "source_family": "reference",
                "statement_property": "P112",
                "statement_property_label": "founder",
                "subject": "Beta Labs",
                "subject_qid": "QUNIT",
                "value": "Ada Beta",
                "value_qid": "QADA",
                "retrieved_at": "2026-06-28T00:00:00+00:00",
            },
        })
        + "\n",
        encoding="utf-8",
    )

    payload = module.run_source_family_structured_qa_fact_collection_workflow(
        collection_corpus_path=collection_path,
        source_catalog_paths=(source_catalog_path,),
        output_dir=output_dir,
        registry_path=registry_path,
        name="source-family-fact-collection-workflow-unit",
        version="0.1",
        adapter_max_results=1,
        adapter_min_text_overlap=0.0,
        metadata={"suite": "unit"},
    )
    record = registry_module.ArtifactRegistry.load_json(registry_path).get(
        "report:source-family-fact-collection-workflow-unit:0.1"
    )
    qa_corpus = json.loads(
        (output_dir / "source-family-structured-qa-corpus.json").read_text(encoding="utf-8")
    )
    combined_rows = [
        json.loads(line)
        for line in (output_dir / "fact-collection-adapter-results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    adapter_requests = [
        json.loads(line)
        for line in (
            output_dir / "source-family-structured-fact-adapter-requests.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert payload["status"] == "ready_for_fact_mapping"
    assert payload["summary"]["source_backed_request_count"] == 4
    assert payload["summary"]["request_with_results_count"] == 4
    assert payload["summary"]["adapter_result_count"] == 4
    assert payload["summary"]["structured_qa_document_count"] == 1
    assert payload["summary"]["structured_qa_candidate_document_count"] == 4
    assert payload["summary"]["rule_stub_count"] == 1
    assert payload["summary"]["reserved_source_document_field_hits"] == {}
    assert payload["label_usage"]["adapter_results_are_verifier_evidence"] is False
    assert qa_corpus["documents"][0]["question"] == "What does Wikidata list as the founder for Beta Labs?"
    assert qa_corpus["documents"][0]["answer"] == "Ada Beta"
    assert all(row["not_verifier_evidence"] is True for row in combined_rows)
    assert all("target_id" not in result["metadata"] for row in combined_rows for result in row["results"])
    assert all("answer" not in request for request in adapter_requests)
    assert all("model_answer" not in request for request in adapter_requests)
    assert all("label" not in request for request in adapter_requests)
    assert registry_module.load_and_verify_artifact_manifest(output_dir / "artifact-manifest.json").passed is True
    assert record.metadata["workflow"] == "source_family_structured_qa_fact_collection_workflow"
    assert record.metadata["structured_qa_document_count"] == 1
    assert record.metadata["suite"] == "unit"


def test_source_family_structured_qa_gap_triage_routes_next_lanes(tmp_path):
    module = importlib.import_module("benchmarks.triage_source_family_structured_qa_gaps")
    registry_module = importlib.import_module("eigentruth.registry")

    claim_mapping_path = tmp_path / "claim-mapping.json"
    plan_path = tmp_path / "fact-expansion-plan.json"
    corpus_path = tmp_path / "fact-collection-corpus.json"
    workflow_path = tmp_path / "fact-collection-workflow.json"
    output_dir = tmp_path / "triage"
    registry_path = tmp_path / "registry.json"

    claim_mapping_path.write_text(
        json.dumps({
            "schema_version": 1,
            "workflow": "source_family_structured_qa_claim_mapping_audit",
            "status": "observed",
            "summary": {
                "target_count": 5,
                "mapped_qa_fact_candidate_count": 1,
                "answer_value_supported_count": 1,
            },
            "records": [
                {
                    "record_id": "record-1",
                    "record_index": 1,
                    "question": "Who founded Beta Labs?",
                    "answer": "Ada Beta.",
                    "question_type": "person",
                    "mapping_decision": "mapped_qa_fact_candidate",
                    "gate_recommendation": "structured_qa_correction_handoff",
                    "mapped_qa_fact_candidate": True,
                    "covered_fact_match": True,
                    "best_mapping_score": 0.91,
                    "mapped_facts": [{"source": "wikidata:Q1:P112:Q2"}],
                },
                {
                    "record_id": "record-2",
                    "record_index": 2,
                    "question": "What country contains French?",
                    "answer": "France.",
                    "question_type": "location",
                    "mapping_decision": "answer_value_supported_by_covered_fact",
                    "gate_recommendation": "answer_support_audit",
                    "answer_value_supported": True,
                    "covered_fact_match": True,
                    "supported_facts": [{"source": "wikidata:Q150:P17:Q142"}],
                },
                {
                    "record_id": "record-3",
                    "record_index": 3,
                    "question": "Who founded Gamma Labs?",
                    "answer": "Gamma Labs.",
                    "question_type": "person",
                    "mapping_decision": "answer_entity_collision",
                    "gate_recommendation": "answer_collision_audit",
                    "answer_entity_collision": True,
                    "best_mapping_score": 0.62,
                    "collision_facts": [{"source": "wikidata:Q3:description"}],
                },
                {
                    "record_id": "record-4",
                    "record_index": 4,
                    "question": "Do more than 20% of people use passports?",
                    "answer": "No.",
                    "question_type": "quantity",
                    "mapping_decision": "subject_only_or_missing_intent",
                    "gate_recommendation": "richer_property_or_indicator_collection",
                    "best_subject_coverage": 1.0,
                },
                {
                    "record_id": "record-5",
                    "record_index": 5,
                    "question": "How does the method work?",
                    "answer": "It works by magic.",
                    "question_type": "method",
                    "mapping_decision": "weak_textual_overlap",
                    "gate_recommendation": "citation_retrieval_before_handoff",
                    "best_mapping_score": 0.31,
                },
            ],
        }),
        encoding="utf-8",
    )
    plan_path.write_text(
        json.dumps({
            "schema_version": 1,
            "workflow": "source_family_structured_qa_fact_expansion_plan",
            "status": "ready_for_collection",
            "summary": {"target_count": 3},
            "targets": [
                {
                    "target_id": "record-3",
                    "record_index": 3,
                    "question_type": "person",
                    "gap_type": "answer_entity_collision",
                    "collection_tasks": [
                        {"task_type": "source_family_fact_disambiguation"},
                        {"task_type": "world_model_or_calculator_rule_request"},
                    ],
                    "world_model_rule_targets": [{"rule_family": "entity_disambiguation"}],
                },
                {
                    "target_id": "record-4",
                    "record_index": 4,
                    "question_type": "quantity",
                    "gap_type": "missing_property_or_indicator",
                    "collection_tasks": [{"task_type": "source_family_structured_fact_request"}],
                    "world_model_rule_targets": [{"rule_family": "quantity_or_arithmetic"}],
                },
                {
                    "target_id": "record-5",
                    "record_index": 5,
                    "question_type": "method",
                    "gap_type": "needs_citation_before_fact_promotion",
                    "collection_tasks": [{"task_type": "external_citation_request"}],
                },
            ],
        }),
        encoding="utf-8",
    )
    corpus_path.write_text(
        json.dumps({
            "schema_version": 1,
            "workflow": "source_family_structured_qa_fact_collection_corpus",
            "status": "ready_for_collection",
            "summary": {"target_count": 3},
            "targets": [
                {
                    "target_id": "record-3",
                    "record_index": 3,
                    "priority": "high",
                    "source_family_targets": [{"provider": "wikidata", "source_family": "reference"}],
                },
                {
                    "target_id": "record-4",
                    "record_index": 4,
                    "priority": "high",
                    "source_family_targets": [{"provider": "worldbank", "source_family": "official_statistics"}],
                },
                {
                    "target_id": "record-5",
                    "record_index": 5,
                    "priority": "medium",
                    "source_family_targets": [{"provider": "source_family_adapter", "source_family": "scholarly"}],
                },
            ],
            "requests": {
                "source_family_structured_fact": [
                    {"target_id": "record-4", "request_type": "source_family_structured_fact"}
                ],
                "source_family_fact_disambiguation": [
                    {"target_id": "record-3", "request_type": "source_family_fact_disambiguation"}
                ],
                "external_citation": [
                    {"target_id": "record-5", "request_type": "external_citation"}
                ],
                "world_model_or_calculator_rule": [
                    {"target_id": "record-3", "request_type": "world_model_or_calculator_rule"},
                    {"target_id": "record-4", "request_type": "world_model_or_calculator_rule"},
                ],
            },
        }),
        encoding="utf-8",
    )
    workflow_path.write_text(
        json.dumps({
            "schema_version": 1,
            "workflow": "source_family_structured_qa_fact_collection_workflow",
            "status": "ready_for_fact_mapping",
            "summary": {
                "adapter_result_count": 12,
                "structured_qa_document_count": 4,
                "rule_stub_count": 2,
            },
        }),
        encoding="utf-8",
    )

    payload = module.run(
        claim_mapping_path=claim_mapping_path,
        fact_expansion_plan_path=plan_path,
        fact_collection_corpus_path=corpus_path,
        fact_collection_workflow_path=workflow_path,
        output_dir=output_dir,
        registry_path=registry_path,
        name="source-family-gap-triage-unit",
        version="0.1",
        metadata={"suite": "unit"},
    )
    by_record = {item["record_index"]: item for item in payload["triage_targets"]}
    registry_record = registry_module.ArtifactRegistry.load_json(registry_path).get(
        "report:source-family-gap-triage-unit:0.1"
    )
    target_rows = [
        json.loads(line)
        for line in (output_dir / "triage-targets.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert payload["status"] == "handoff_ready"
    assert payload["summary"]["handoff_ready_count"] == 1
    assert payload["summary"]["audit_only_count"] == 1
    assert payload["summary"]["blocked_target_count"] == 4
    assert by_record[1]["next_lane"] == "structured_qa_correction_handoff"
    assert by_record[1]["blocked_from_handoff"] is False
    assert by_record[2]["next_lane"] == "answer_support_audit"
    assert by_record[3]["next_lane"] == "answer_collision_audit"
    assert by_record[3]["world_model_rule_families"] == ("entity_disambiguation",)
    assert by_record[4]["next_lane"] == "richer_property_or_indicator_collection"
    assert by_record[4]["available_request_counts"]["source_family_structured_fact"] == 1
    assert by_record[5]["next_lane"] == "citation_retrieval_before_handoff"
    assert payload["summary"]["targets_with_external_citation"] == 1
    assert payload["summary"]["targets_with_world_model_or_calculator_rule"] == 2
    assert registry_record is not None
    assert registry_record.metadata["blocked_count"] == 4
    assert len(target_rows) == 5


def test_source_family_structured_qa_lane_execution_queue_batches_requests(tmp_path):
    module = importlib.import_module("benchmarks.build_source_family_structured_qa_lane_execution_queue")
    registry_module = importlib.import_module("eigentruth.registry")

    triage_path = tmp_path / "gap-triage.json"
    corpus_path = tmp_path / "fact-collection-corpus.json"
    output_dir = tmp_path / "lane-queue"
    registry_path = tmp_path / "registry.json"
    triage_path.write_text(
        json.dumps({
            "schema_version": 1,
            "workflow": "source_family_structured_qa_gap_triage",
            "status": "needs_collection",
            "summary": {"target_count": 4},
            "triage_targets": [
                {
                    "target_id": "record-1",
                    "record_index": 1,
                    "next_lane": "structured_qa_correction_handoff",
                    "lane_status": "handoff_ready",
                    "priority_score": 150.0,
                    "question": "Who founded Alpha?",
                    "answer": "Ada",
                    "model_answer": "Ada",
                    "question_type": "person",
                    "mapping_decision": "mapped_qa_fact_candidate",
                    "available_request_counts": {"external_citation": 1},
                },
                {
                    "target_id": "record-2",
                    "record_index": 2,
                    "next_lane": "answer_support_audit",
                    "lane_status": "audit_only",
                    "priority_score": 25.0,
                    "question": "Where is Beta?",
                    "answer": "France",
                    "question_type": "location",
                    "mapping_decision": "answer_value_supported_by_covered_fact",
                    "available_request_counts": {"external_citation": 1},
                },
                {
                    "target_id": "record-3",
                    "record_index": 3,
                    "next_lane": "answer_collision_audit",
                    "lane_status": "blocked_needs_disambiguation",
                    "priority_score": 120.0,
                    "question": "Who is Gamma?",
                    "answer": "Gamma",
                    "question_type": "definition",
                    "mapping_decision": "answer_entity_collision",
                    "source_gap_type": "answer_entity_collision",
                    "available_request_counts": {
                        "source_family_fact_disambiguation": 1,
                        "world_model_or_calculator_rule": 1,
                        "external_citation": 1,
                    },
                    "world_model_rule_families": ["entity_disambiguation"],
                },
                {
                    "target_id": "record-4",
                    "record_index": 4,
                    "next_lane": "richer_property_or_indicator_collection",
                    "lane_status": "needs_property_collection",
                    "priority_score": 104.0,
                    "question": "Do most Delta users have passports?",
                    "answer": "No",
                    "question_type": "quantity",
                    "mapping_decision": "subject_only_or_missing_intent",
                    "source_gap_type": "missing_property_or_indicator",
                    "available_request_counts": {
                        "source_family_structured_fact": 1,
                        "world_model_or_calculator_rule": 1,
                    },
                    "source_family_targets": [{"provider": "worldbank", "source_family": "official_statistics"}],
                },
            ],
        }),
        encoding="utf-8",
    )
    corpus_path.write_text(
        json.dumps({
            "schema_version": 1,
            "workflow": "source_family_structured_qa_fact_collection_corpus",
            "status": "ready_for_collection",
            "summary": {"target_count": 4, "total_request_count": 6},
            "requests": {
                "external_citation": [
                    {
                        "target_id": "record-1",
                        "request_id": "cite:record-1:1",
                        "request_type": "external_citation",
                        "query": "Alpha founder",
                        "answer": "Ada",
                    },
                    {
                        "target_id": "record-2",
                        "request_id": "cite:record-2:1",
                        "request_type": "external_citation",
                        "query": "Beta location",
                    },
                    {
                        "target_id": "record-3",
                        "request_id": "cite:record-3:1",
                        "request_type": "external_citation",
                        "query": "Gamma entity",
                    },
                ],
                "source_family_fact_disambiguation": [
                    {
                        "target_id": "record-3",
                        "request_id": "disambig:record-3:1",
                        "request_type": "source_family_fact_disambiguation",
                        "query": "Gamma disambiguation",
                    }
                ],
                "source_family_structured_fact": [
                    {
                        "target_id": "record-4",
                        "request_id": "sfact:record-4:1",
                        "request_type": "source_family_structured_fact",
                        "query": "Delta passports",
                        "provider_hint": "worldbank",
                    }
                ],
                "world_model_or_calculator_rule": [
                    {
                        "target_id": "record-3",
                        "request_id": "rule:record-3:1",
                        "request_type": "world_model_or_calculator_rule",
                        "rule_family": "entity_disambiguation",
                        "rule_seed": "Author a role rule for Gamma",
                    },
                    {
                        "target_id": "record-4",
                        "request_id": "rule:record-4:1",
                        "request_type": "world_model_or_calculator_rule",
                        "rule_family": "quantity_or_arithmetic",
                        "rule_seed": "Author a numeric rule for Delta",
                    },
                ],
            },
        }),
        encoding="utf-8",
    )

    payload = module.run(
        triage_path=triage_path,
        collection_corpus_path=corpus_path,
        output_dir=output_dir,
        registry_path=registry_path,
        name="source-family-lane-queue-unit",
        version="0.1",
        max_requests_per_batch=1,
        metadata={"suite": "unit"},
    )
    registry_record = registry_module.ArtifactRegistry.load_json(registry_path).get(
        "report:source-family-lane-queue-unit:0.1"
    )
    request_rows = [
        json.loads(line)
        for line in (output_dir / "adapter-requests.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    batch_rows = [
        json.loads(line)
        for line in (output_dir / "execution-batches.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert payload["status"] == "ready_for_adapter_execution"
    assert payload["summary"]["target_count"] == 2
    assert payload["summary"]["adapter_request_count"] == 5
    assert payload["summary"]["batch_count"] == 5
    assert payload["summary"]["skipped_target_counts"] == {
        "lane_status_filtered": 2,
    }
    assert payload["summary"]["top_target"]["target_id"] == "record-3"
    assert payload["summary"]["request_type_counts"]["world_model_or_calculator_rule"] == 2
    assert payload["summary"]["request_type_counts"]["external_citation"] == 1
    assert payload["summary"]["request_type_counts"]["source_family_fact_disambiguation"] == 1
    assert payload["summary"]["request_type_counts"]["source_family_structured_fact"] == 1
    assert {row["target_id"] for row in request_rows} == {"record-3", "record-4"}
    assert all("answer" not in row and "model_answer" not in row for row in request_rows)
    assert batch_rows[0]["next_lane"] == "answer_collision_audit"
    assert batch_rows[0]["request_type"] == "source_family_fact_disambiguation"
    assert registry_module.load_and_verify_artifact_manifest(output_dir / "artifact-manifest.json").passed is True
    assert registry_record is not None
    assert registry_record.metadata["adapter_request_count"] == 5


def test_source_family_structured_qa_lane_batch_workflow_replays_selected_batch(tmp_path):
    module = importlib.import_module("benchmarks.run_source_family_structured_qa_lane_batch_workflow")
    registry_module = importlib.import_module("eigentruth.registry")

    lane_queue_path = tmp_path / "lane-execution-queue.json"
    collection_path = tmp_path / "fact-collection-corpus.json"
    catalog_path = tmp_path / "source-catalog.jsonl"
    output_dir = tmp_path / "lane-batch"
    registry_path = tmp_path / "registry.json"
    lane_queue_path.write_text(
        json.dumps({
            "schema_version": 1,
            "workflow": "source_family_structured_qa_lane_execution_queue",
            "status": "ready_for_adapter_execution",
            "execution_batches": [
                {
                    "batch_id": "sfqa-lane-batch-0001",
                    "next_lane": "answer_collision_audit",
                    "lane_status": "blocked_needs_disambiguation",
                    "request_type": "source_family_fact_disambiguation",
                    "adapter_family": "source_family_fact_disambiguation",
                    "target_ids": ["record-3"],
                    "source_request_ids": ["disambig:record-3:1"],
                    "not_verifier_evidence": True,
                }
            ],
        }),
        encoding="utf-8",
    )
    collection_path.write_text(
        json.dumps({
            "schema_version": 1,
            "workflow": "source_family_structured_qa_fact_collection_corpus",
            "status": "ready_for_collection",
            "summary": {"target_count": 1, "total_request_count": 1},
            "targets": [
                {
                    "target_id": "record-3",
                    "record_index": 3,
                    "question": "Who is Gamma?",
                    "answer": "Gamma",
                    "question_type": "definition",
                    "priority": "high",
                }
            ],
            "requests": {
                "source_family_fact_disambiguation": [
                    {
                        "target_id": "record-3",
                        "request_id": "disambig:record-3:1",
                        "request_type": "source_family_fact_disambiguation",
                        "query": "Gamma disambiguation",
                        "question": "Who is Gamma?",
                        "question_type": "definition",
                        "priority": "high",
                        "answer": "Gamma",
                        "model_answer": "Gamma",
                    }
                ],
            },
        }),
        encoding="utf-8",
    )
    catalog_path.write_text(
        json.dumps({
            "text": "Gamma disambiguation identifies Gamma as a separate company and entity.",
            "title": "Gamma entity note",
            "source": "unit:gamma",
            "provider": "unit_catalog",
            "source_family": "reference",
            "metadata": {"provider": "unit_catalog"},
        })
        + "\n",
        encoding="utf-8",
    )

    payload = module.run_source_family_structured_qa_lane_batch_workflow(
        lane_queue_path=lane_queue_path,
        collection_corpus_path=collection_path,
        source_catalog_paths=(catalog_path,),
        output_dir=output_dir,
        batch_ids=("sfqa-lane-batch-0001",),
        registry_path=registry_path,
        name="source-family-lane-batch-unit",
        version="0.1",
        adapter_min_text_overlap=0.0,
        metadata={"suite": "unit"},
    )
    batch_collection = json.loads((output_dir / "lane-batch-collection-corpus.json").read_text(encoding="utf-8"))
    registry_record = registry_module.ArtifactRegistry.load_json(registry_path).get(
        "report:source-family-lane-batch-unit:0.1"
    )

    assert payload["status"] == "observed"
    assert payload["summary"]["batch_count"] == 1
    assert payload["summary"]["target_count"] == 1
    assert payload["summary"]["source_backed_request_count"] == 1
    assert payload["summary"]["adapter_result_count"] == 1
    assert batch_collection["summary"]["stripped_reserved_field_counts"] == {
        "answer": 1,
        "model_answer": 1,
        "target.answer": 1,
    }
    assert "answer" not in batch_collection["targets"][0]
    assert "answer" not in batch_collection["requests"]["source_family_fact_disambiguation"][0]
    assert "model_answer" not in batch_collection["requests"]["source_family_fact_disambiguation"][0]
    assert registry_module.load_and_verify_artifact_manifest(output_dir / "artifact-manifest.json").passed is True
    assert registry_record is not None
    assert registry_record.metadata["source_backed_request_count"] == 1


def test_source_family_structured_qa_lane_batch_workflow_emits_rule_only_batch(tmp_path):
    module = importlib.import_module("benchmarks.run_source_family_structured_qa_lane_batch_workflow")
    registry_module = importlib.import_module("eigentruth.registry")

    lane_queue_path = tmp_path / "lane-execution-queue.json"
    collection_path = tmp_path / "fact-collection-corpus.json"
    output_dir = tmp_path / "rule-lane-batch"
    registry_path = tmp_path / "registry.json"
    lane_queue_path.write_text(
        json.dumps({
            "schema_version": 1,
            "workflow": "source_family_structured_qa_lane_execution_queue",
            "status": "ready_for_adapter_execution",
            "execution_batches": [
                {
                    "batch_id": "sfqa-lane-batch-0002",
                    "next_lane": "world_model_rule_authoring",
                    "lane_status": "needs_rule_authoring",
                    "request_type": "world_model_or_calculator_rule",
                    "adapter_family": "world_model_rule_authoring",
                    "target_ids": ["record-11"],
                    "source_request_ids": ["rule:record-11:1"],
                    "not_verifier_evidence": True,
                }
            ],
        }),
        encoding="utf-8",
    )
    collection_path.write_text(
        json.dumps({
            "schema_version": 1,
            "workflow": "source_family_structured_qa_fact_collection_corpus",
            "status": "ready_for_collection",
            "summary": {"target_count": 1, "total_request_count": 1},
            "targets": [
                {
                    "target_id": "record-11",
                    "record_index": 11,
                    "question": "Do more than 20% of Americans have passports?",
                    "answer": "No.",
                    "question_type": "quantity",
                    "priority": "high",
                }
            ],
            "requests": {
                "world_model_or_calculator_rule": [
                    {
                        "target_id": "record-11",
                        "request_id": "rule:record-11:1",
                        "request_type": "world_model_or_calculator_rule",
                        "rule_family": "quantity_or_arithmetic",
                        "rule_reason": "numeric claim needs denominator and reference time",
                        "rule_seed": "Author a deterministic numeric check",
                        "required_inputs": ["numeric_value", "unit", "reference_time"],
                        "question": "Do more than 20% of Americans have passports?",
                        "question_type": "quantity",
                        "priority": "high",
                        "answer": "No.",
                        "model_answer": "No.",
                    }
                ],
            },
        }),
        encoding="utf-8",
    )

    payload = module.run_source_family_structured_qa_lane_batch_workflow(
        lane_queue_path=lane_queue_path,
        collection_corpus_path=collection_path,
        source_catalog_paths=(),
        output_dir=output_dir,
        batch_ids=("sfqa-lane-batch-0002",),
        registry_path=registry_path,
        name="source-family-rule-lane-batch-unit",
        version="0.1",
        metadata={"suite": "unit"},
    )
    batch_collection = json.loads((output_dir / "lane-batch-collection-corpus.json").read_text(encoding="utf-8"))
    rule_rows = [
        json.loads(line)
        for line in (output_dir / "world-model-rule-stubs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    registry_record = registry_module.ArtifactRegistry.load_json(registry_path).get(
        "report:source-family-rule-lane-batch-unit:0.1"
    )

    assert payload["status"] == "ready_for_rule_authoring"
    assert payload["paths"]["child_workflow_report"] is None
    assert payload["summary"]["source_backed_request_count"] == 0
    assert payload["summary"]["adapter_result_count"] == 0
    assert payload["summary"]["structured_qa_document_count"] == 0
    assert payload["summary"]["rule_stub_count"] == 1
    assert payload["summary"]["request_type_counts"] == {"world_model_or_calculator_rule": 1}
    assert batch_collection["summary"]["stripped_reserved_field_counts"] == {
        "answer": 1,
        "model_answer": 1,
        "target.answer": 1,
    }
    assert "answer" not in batch_collection["targets"][0]
    assert "answer" not in batch_collection["requests"]["world_model_or_calculator_rule"][0]
    assert rule_rows[0]["rule_family"] == "quantity_or_arithmetic"
    assert rule_rows[0]["required_inputs"] == ["numeric_value", "unit", "reference_time"]
    assert rule_rows[0]["not_verifier_evidence"] is True
    assert registry_module.load_and_verify_artifact_manifest(output_dir / "artifact-manifest.json").passed is True
    assert registry_record is not None
    assert registry_record.metadata["status"] == "ready_for_rule_authoring"
    assert registry_record.metadata["rule_stub_count"] == 1


def test_unresolved_world_model_rule_stubs_sanitizes_rule_queue(tmp_path):
    module = importlib.import_module("benchmarks.build_unresolved_world_model_rule_stubs")
    registry_module = importlib.import_module("eigentruth.registry")

    queue_path = tmp_path / "unresolved-evidence-queue.json"
    output_dir = tmp_path / "unresolved-rule-stubs"
    registry_path = tmp_path / "registry.json"
    queue_path.write_text(
        json.dumps({
            "schema_version": 1,
            "workflow": "unresolved_blind_spot_evidence_queue",
            "status": "ready_for_adapter_execution",
            "summary": {"adapter_request_count": 3},
            "adapter_requests": [
                {
                    "queue_id": "queue:record-1:external_citation:1",
                    "source_request_id": "cite:record-1:1",
                    "target_id": "record-1",
                    "request_type": "external_citation",
                    "adapter_family": "external_citation_search",
                    "question": "What is Alpha Syndrome?",
                    "model_answer": "A moon.",
                    "not_verifier_evidence": True,
                },
                {
                    "queue_id": "queue:record-1:world_model_or_calculator_rule:1",
                    "source_request_id": "rule:record-1:1",
                    "target_id": "record-1",
                    "record_index": 1,
                    "target_rank": 1,
                    "request_type": "world_model_or_calculator_rule",
                    "adapter_family": "world_model_rule_authoring",
                    "rule_family": "quantity_or_arithmetic",
                    "priority": "high",
                    "priority_score": 187.0,
                    "evidence_status": "no_joined_facts",
                    "mapping_decision": "no_joined_facts",
                    "question": "What is Alpha Syndrome?",
                    "question_type": "definition",
                    "model_answer": "A moon.",
                    "label": 1,
                    "metadata": {"request_id": "rule:record-1:1"},
                    "not_verifier_evidence": True,
                },
                {
                    "queue_id": "queue:record-2:world_model_or_calculator_rule:1",
                    "source_request_id": "rule:record-2:1",
                    "target_id": "record-2",
                    "record_index": 2,
                    "target_rank": 2,
                    "request_type": "world_model_or_calculator_rule",
                    "adapter_family": "world_model_rule_authoring",
                    "rule_family": "temporal_freshness",
                    "priority": "high",
                    "evidence_status": "no_joined_facts",
                    "mapping_decision": "no_joined_facts",
                    "question": "What changed in recent decades?",
                    "question_type": "temporal",
                    "model_answer": "Food got harder to afford.",
                    "metadata": {"request_id": "rule:record-2:1"},
                    "not_verifier_evidence": True,
                },
            ],
        }),
        encoding="utf-8",
    )

    payload = module.run(
        queue_report_path=queue_path,
        output_dir=output_dir,
        registry_path=registry_path,
        name="unresolved-rule-stubs-unit",
        version="0.1",
        metadata={"suite": "unit"},
    )
    stubs = [
        json.loads(line)
        for line in (output_dir / "world-model-rule-stubs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    registry_record = registry_module.ArtifactRegistry.load_json(registry_path).get(
        "report:unresolved-rule-stubs-unit:0.1"
    )
    stub_text = json.dumps(stubs, sort_keys=True)

    assert payload["status"] == "ready_for_rule_authoring"
    assert payload["summary"]["source_adapter_request_count"] == 3
    assert payload["summary"]["source_rule_request_count"] == 2
    assert payload["summary"]["source_non_rule_request_count"] == 1
    assert payload["summary"]["rule_stub_count"] == 2
    assert payload["summary"]["rule_family_counts"] == {
        "quantity_or_arithmetic": 1,
        "temporal_consistency": 1,
    }
    assert payload["summary"]["source_rule_family_counts"] == {
        "quantity_or_arithmetic": 1,
        "temporal_freshness": 1,
    }
    assert payload["summary"]["reserved_source_field_counts"]["model_answer"] == 2
    assert payload["summary"]["reserved_source_field_counts"]["record_index"] == 2
    assert payload["summary"]["reserved_source_field_counts"]["target_rank"] == 2
    assert payload["summary"]["reserved_source_field_counts"]["label"] == 1
    assert stubs[0]["request_id"] == "rule:record-1:1"
    assert stubs[0]["required_inputs"] == ["numeric_value", "unit", "reference_time"]
    assert stubs[1]["rule_family"] == "temporal_consistency"
    assert stubs[1]["metadata"]["source_rule_family"] == "temporal_freshness"
    assert stubs[1]["required_inputs"] == ["claim_time", "source_time", "retrieved_at", "source_citation"]
    assert all(row["not_verifier_evidence"] is True for row in stubs)
    assert "model_answer" not in stub_text
    assert "record_index" not in stub_text
    assert "target_rank" not in stub_text
    assert "label" not in stub_text
    assert "A moon" not in stub_text
    assert "Food got harder" not in stub_text
    assert registry_module.load_and_verify_artifact_manifest(output_dir / "artifact-manifest.json").passed is True
    assert registry_record is not None
    assert registry_record.metadata["workflow"] == "unresolved_world_model_rule_stubs"
    assert registry_record.metadata["rule_stub_count"] == 2


def test_unresolved_world_model_rule_stubs_skip_untrusted_rule_requests():
    module = importlib.import_module("benchmarks.build_unresolved_world_model_rule_stubs")

    payload = module.build_unresolved_world_model_rule_stubs(
        queue_report={
            "schema_version": 1,
            "workflow": "unresolved_blind_spot_evidence_queue",
            "status": "ready_for_adapter_execution",
            "adapter_requests": [
                {
                    "source_request_id": "rule:record-1:1",
                    "target_id": "record-1",
                    "request_type": "world_model_or_calculator_rule",
                    "adapter_family": "world_model_rule_authoring",
                    "rule_family": "quantity_or_arithmetic",
                    "question": "What is Alpha Syndrome?",
                    "not_verifier_evidence": False,
                }
            ],
        }
    )

    assert payload["status"] == "empty"
    assert payload["summary"]["source_rule_request_count"] == 1
    assert payload["summary"]["rule_stub_count"] == 0
    assert payload["summary"]["skipped_rule_request_count"] == 1
    assert payload["skipped_rule_requests"][0]["failures"] == ("source_request_not_marked_non_evidence",)
    assert payload["rule_stubs"] == ()


def test_world_model_rule_authoring_adapter_requests_missing_inputs(tmp_path):
    module = importlib.import_module("benchmarks.run_world_model_rule_authoring_adapter")
    registry_module = importlib.import_module("eigentruth.registry")

    stubs_path = tmp_path / "world-model-rule-stubs.jsonl"
    output_dir = tmp_path / "rule-adapter"
    registry_path = tmp_path / "registry.json"
    stubs_path.write_text(
        json.dumps({
            "schema_version": 1,
            "workflow": "source_family_structured_qa_lane_batch_workflow",
            "request_id": "rule:record-11:1",
            "target_id": "record-11",
            "request_type": "world_model_or_calculator_rule",
            "rule_family": "quantity_or_arithmetic",
            "rule_seed": "Author a deterministic numeric check",
            "required_inputs": ["numeric_value", "unit", "reference_time"],
            "question": "Do more than 20% of Americans have passports?",
            "not_verifier_evidence": True,
        })
        + "\n",
        encoding="utf-8",
    )

    payload = module.run_world_model_rule_authoring_adapter(
        rule_stubs_path=stubs_path,
        output_dir=output_dir,
        registry_path=registry_path,
        name="world-model-rule-adapter-unit",
        version="0.1",
        metadata={"suite": "unit"},
    )
    input_requests = [
        json.loads(line)
        for line in (output_dir / "world-model-rule-input-requests.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    registry_record = registry_module.ArtifactRegistry.load_json(registry_path).get(
        "report:world-model-rule-adapter-unit:0.1"
    )

    assert payload["status"] == "needs_inputs"
    assert payload["summary"]["stub_count"] == 1
    assert payload["summary"]["executed_count"] == 0
    assert payload["summary"]["needs_input_count"] == 1
    assert payload["summary"]["adapter_counts"] == {"calculator": 1}
    assert input_requests[0]["request_id"] == "rule:record-11:1"
    assert input_requests[0]["missing_inputs"] == ["numeric_value", "unit", "reference_time"]
    assert input_requests[0]["not_verifier_evidence"] is True
    assert registry_module.load_and_verify_artifact_manifest(output_dir / "artifact-manifest.json").passed is True
    assert registry_record is not None
    assert registry_record.metadata["status"] == "needs_inputs"
    assert registry_record.metadata["input_request_count"] == 1


def test_world_model_rule_authoring_adapter_executes_explicit_calculator_input(tmp_path):
    module = importlib.import_module("benchmarks.run_world_model_rule_authoring_adapter")

    stubs_path = tmp_path / "world-model-rule-stubs.jsonl"
    inputs_path = tmp_path / "rule-inputs.json"
    output_dir = tmp_path / "rule-adapter"
    stubs_path.write_text(
        json.dumps({
            "schema_version": 1,
            "workflow": "source_family_structured_qa_lane_batch_workflow",
            "request_id": "rule:calc:1",
            "target_id": "record-calc",
            "request_type": "world_model_or_calculator_rule",
            "rule_family": "quantity_or_arithmetic",
            "required_inputs": ["numeric_value", "unit", "reference_time"],
            "question": "Is 2 + 2 equal to 5?",
            "not_verifier_evidence": True,
        })
        + "\n",
        encoding="utf-8",
    )
    inputs_path.write_text(
        json.dumps({
            "rule:calc:1": {
                "calculation": {
                    "expression": "2 + 2",
                    "expected": 5,
                }
            }
        }),
        encoding="utf-8",
    )

    payload = module.run_world_model_rule_authoring_adapter(
        rule_stubs_path=stubs_path,
        rule_inputs_path=inputs_path,
        output_dir=output_dir,
        metadata={"suite": "unit"},
    )
    results = [
        json.loads(line)
        for line in (output_dir / "world-model-rule-results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    input_requests = (output_dir / "world-model-rule-input-requests.jsonl").read_text(encoding="utf-8").strip()

    assert payload["status"] == "observed"
    assert payload["summary"]["executed_count"] == 1
    assert payload["summary"]["candidate_refuted_count"] == 1
    assert payload["summary"]["needs_input_count"] == 0
    assert input_requests == ""
    assert results[0]["status"] == "refuted"
    assert results[0]["authored_rule"]["adapter"] == "calculator"
    assert results[0]["not_verifier_evidence"] is True
    assert results[0]["metadata"]["candidate_results_require_promotion_gate"] is True
    assert "calculator:" in results[0]["evidence"][0]


def test_world_model_rule_authoring_adapter_keeps_partial_inputs_actionable(tmp_path):
    module = importlib.import_module("benchmarks.run_world_model_rule_authoring_adapter")

    stubs_path = tmp_path / "world-model-rule-stubs.jsonl"
    inputs_path = tmp_path / "rule-inputs.json"
    output_dir = tmp_path / "rule-adapter"
    stubs_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "schema_version": 1,
                    "workflow": "source_family_structured_qa_lane_batch_workflow",
                    "request_id": "rule:entity:1",
                    "target_id": "record-entity",
                    "request_type": "world_model_or_calculator_rule",
                    "rule_family": "entity_disambiguation",
                    "required_inputs": ["subject_entity", "answer_entity", "requested_role"],
                    "question": "Who first started Tesla Motors?",
                    "not_verifier_evidence": True,
                },
                {
                    "schema_version": 1,
                    "workflow": "source_family_structured_qa_lane_batch_workflow",
                    "request_id": "rule:quantity:1",
                    "target_id": "record-quantity",
                    "request_type": "world_model_or_calculator_rule",
                    "rule_family": "quantity_or_arithmetic",
                    "required_inputs": ["numeric_value", "unit", "reference_time"],
                    "question": "Do more than 20% of Americans have passports?",
                    "not_verifier_evidence": True,
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    inputs_path.write_text(
        json.dumps({
            "rule:entity:1": {
                "subject_entity": "Tesla Motors",
                "answer_entity": "Elon Musk",
                "requested_role": "first starter",
            },
            "rule:quantity:1": {
                "numeric_value": "20",
                "unit": "percent",
                "reference_time": "current",
            },
        }),
        encoding="utf-8",
    )

    payload = module.run_world_model_rule_authoring_adapter(
        rule_stubs_path=stubs_path,
        rule_inputs_path=inputs_path,
        output_dir=output_dir,
        metadata={"suite": "unit"},
    )
    results = [
        json.loads(line)
        for line in (output_dir / "world-model-rule-results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    missing_by_request = {row["request_id"]: row["missing_inputs"] for row in results}

    assert payload["status"] == "needs_inputs"
    assert payload["summary"]["executed_count"] == 0
    assert missing_by_request["rule:entity:1"] == ["expected_entity"]
    assert missing_by_request["rule:quantity:1"] == ["calculation.expression", "calculation.expected"]


def test_world_model_rule_input_collection_plan_builds_typed_batches(tmp_path):
    module = importlib.import_module("benchmarks.build_world_model_rule_input_collection_plan")
    registry_module = importlib.import_module("eigentruth.registry")

    requests_path = tmp_path / "world-model-rule-input-requests.jsonl"
    output_dir = tmp_path / "rule-input-plan"
    registry_path = tmp_path / "registry.json"
    requests_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "schema_version": 1,
                    "workflow": "world_model_rule_authoring_adapter",
                    "request_id": "rule:record-1:1",
                    "target_id": "record-1",
                    "rule_family": "entity_disambiguation",
                    "adapter": "entity_role_disambiguation",
                    "required_inputs": ["subject_entity", "answer_entity", "requested_role"],
                    "missing_inputs": ["subject_entity", "answer_entity", "requested_role"],
                    "question": "Who first started Tesla Motors?",
                    "question_type": "person",
                    "gap_type": "answer_entity_collision",
                    "priority": "high",
                    "not_verifier_evidence": True,
                    "answer": "reserved field should be stripped",
                },
                {
                    "schema_version": 1,
                    "workflow": "world_model_rule_authoring_adapter",
                    "request_id": "rule:record-11:1",
                    "target_id": "record-11",
                    "rule_family": "quantity_or_arithmetic",
                    "adapter": "calculator",
                    "required_inputs": ["numeric_value", "unit", "reference_time"],
                    "missing_inputs": ["numeric_value", "unit", "reference_time"],
                    "question": "Do more than 20% of Americans have passports?",
                    "question_type": "quantity",
                    "gap_type": "missing_property_or_indicator",
                    "priority": "medium",
                    "not_verifier_evidence": True,
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )

    payload = module.run(
        input_requests_path=requests_path,
        output_dir=output_dir,
        registry_path=registry_path,
        name="rule-input-plan-unit",
        version="0.1",
        metadata={"suite": "unit"},
        max_tasks_per_batch=1,
    )
    tasks = [
        json.loads(line)
        for line in (output_dir / "rule-input-tasks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    batches = [
        json.loads(line)
        for line in (output_dir / "rule-input-execution-batches.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    registry_record = registry_module.ArtifactRegistry.load_json(registry_path).get(
        "report:rule-input-plan-unit:0.1"
    )

    assert payload["status"] == "ready_for_input_collection"
    assert payload["summary"]["task_count"] == 2
    assert payload["summary"]["batch_count"] == 2
    assert payload["summary"]["collection_family_counts"] == {
        "entity_role_rule_input_collection": 1,
        "numeric_rule_input_collection": 1,
    }
    assert tasks[0]["collection_family"] == "entity_role_rule_input_collection"
    assert "expected_entity" in tasks[0]["execution_inputs"]
    assert "answer" not in tasks[0]
    assert tasks[0]["source_policy"]["not_verifier_evidence"] is True
    assert batches[0]["not_verifier_evidence"] is True
    assert registry_module.load_and_verify_artifact_manifest(output_dir / "artifact-manifest.json").passed is True
    assert registry_record is not None
    assert registry_record.metadata["status"] == "ready_for_input_collection"
    assert registry_record.metadata["task_count"] == 2


def test_world_model_rule_input_plan_audit_requeues_family_mismatches(tmp_path):
    module = importlib.import_module("benchmarks.audit_world_model_rule_input_plan")
    registry_module = importlib.import_module("eigentruth.registry")

    tasks_path = tmp_path / "rule-input-tasks.jsonl"
    output_dir = tmp_path / "rule-input-audit"
    registry_path = tmp_path / "registry.json"
    tasks_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "task_id": "rule-input-task-0001",
                    "source_request_id": "rule:record-190:1",
                    "target_id": "record-190",
                    "rule_family": "quantity_or_arithmetic",
                    "collection_family": "numeric_rule_input_collection",
                    "question": "What is the population of the country?",
                    "question_type": "definition",
                    "priority": "high",
                    "not_verifier_evidence": True,
                },
                {
                    "task_id": "rule-input-task-0002",
                    "source_request_id": "rule:record-516:1",
                    "target_id": "record-516",
                    "rule_family": "quantity_or_arithmetic",
                    "collection_family": "numeric_rule_input_collection",
                    "question": 'This American producer was born in the 70s and is named "Elon" what?',
                    "question_type": "definition",
                    "priority": "high",
                    "model_answer": "His name is Elon Musk.",
                    "not_verifier_evidence": True,
                },
                {
                    "task_id": "rule-input-task-0003",
                    "source_request_id": "rule:record-326:1",
                    "target_id": "record-326",
                    "rule_family": "temporal_consistency",
                    "collection_family": "temporal_snapshot_rule_input_collection",
                    "question": "What happened to the affordability of food in America in recent decades?",
                    "question_type": "definition",
                    "priority": "high",
                    "not_verifier_evidence": True,
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )

    payload = module.run(
        input_tasks_path=tasks_path,
        output_dir=output_dir,
        registry_path=registry_path,
        name="rule-input-audit-unit",
        version="0.1",
        metadata={"suite": "unit"},
    )
    audited = [
        json.loads(line)
        for line in (output_dir / "audited-rule-input-tasks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    requeue = [
        json.loads(line)
        for line in (output_dir / "rule-input-requeue-suggestions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    registry_record = registry_module.ArtifactRegistry.load_json(registry_path).get(
        "report:rule-input-audit-unit:0.1"
    )
    audit_text = json.dumps({"audited": audited, "requeue": requeue}, sort_keys=True)

    assert payload["status"] == "needs_requeue"
    assert payload["summary"]["task_count"] == 3
    assert payload["summary"]["requeue_suggestion_count"] == 1
    assert payload["summary"]["finding_counts"]["numeric_rule_missing_candidate_claim_binding"] == 2
    assert payload["summary"]["finding_counts"]["quantity_rule_for_entity_or_role_question"] == 1
    assert audited[0]["recommended_action"] == "collect_missing_inputs_with_review"
    assert audited[1]["recommended_rule_family"] == "entity_disambiguation"
    assert audited[1]["recommended_collection_family"] == "entity_role_rule_input_collection"
    assert audited[2]["finding_count"] == 0
    assert requeue[0]["source_request_id"] == "rule:record-516:1"
    assert requeue[0]["recommended_rule_family"] == "entity_disambiguation"
    assert "model_answer" not in audit_text
    assert "His name is Elon Musk" not in audit_text
    assert registry_module.load_and_verify_artifact_manifest(output_dir / "artifact-manifest.json").passed is True
    assert registry_record is not None
    assert registry_record.metadata["workflow"] == "world_model_rule_input_plan_audit"
    assert registry_record.metadata["status"] == "needs_requeue"


def test_world_model_rule_input_plan_audit_accepts_explicit_numeric_binding():
    module = importlib.import_module("benchmarks.audit_world_model_rule_input_plan")

    payload = module.audit_world_model_rule_input_plan(
        input_tasks=[
            {
                "task_id": "rule-input-task-0001",
                "source_request_id": "rule:record-190:1",
                "target_id": "record-190",
                "rule_family": "quantity_or_arithmetic",
                "collection_family": "numeric_rule_input_collection",
                "question": "What is the population of the country?",
                "question_type": "definition",
                "metadata": {"candidate_claim_value": "330 million"},
                "not_verifier_evidence": True,
            }
        ]
    )

    assert payload["status"] == "pass"
    assert payload["summary"]["finding_count"] == 0
    assert payload["summary"]["requeue_suggestion_count"] == 0
    assert payload["audited_tasks"][0]["recommended_action"] == "collect_missing_inputs"


def test_world_model_rule_stub_requeue_feeds_entity_input_plan(tmp_path):
    requeue_module = importlib.import_module("benchmarks.requeue_world_model_rule_stubs_from_audit")
    adapter_module = importlib.import_module("benchmarks.run_world_model_rule_authoring_adapter")
    plan_module = importlib.import_module("benchmarks.build_world_model_rule_input_collection_plan")
    registry_module = importlib.import_module("eigentruth.registry")

    stubs_path = tmp_path / "world-model-rule-stubs.jsonl"
    suggestions_path = tmp_path / "rule-input-requeue-suggestions.jsonl"
    requeue_dir = tmp_path / "requeue"
    adapter_dir = tmp_path / "adapter"
    plan_dir = tmp_path / "plan"
    registry_path = tmp_path / "registry.json"
    stubs_path.write_text(
        json.dumps({
            "schema_version": 1,
            "workflow": "unresolved_world_model_rule_stubs",
            "request_id": "rule:record-516:1",
            "target_id": "record-516",
            "request_type": "world_model_or_calculator_rule",
            "rule_family": "quantity_or_arithmetic",
            "rule_seed": "Author a deterministic numeric or arithmetic check",
            "rule_reason": "Original unresolved rule branch.",
            "required_inputs": ["numeric_value", "unit", "reference_time"],
            "question": 'This American producer was born in the 70s and is named "Elon" what?',
            "question_type": "definition",
            "gap_type": "no_joined_facts",
            "priority": "high",
            "model_answer": "His name is Elon Musk.",
            "not_verifier_evidence": True,
            "metadata": {"source_queue_id": "queue:record-516:world_model_or_calculator_rule:5"},
        })
        + "\n",
        encoding="utf-8",
    )
    suggestions_path.write_text(
        json.dumps({
            "schema_version": 1,
            "workflow": "world_model_rule_input_plan_audit",
            "source_request_id": "rule:record-516:1",
            "target_id": "record-516",
            "task_id": "rule-input-task-0002",
            "current_rule_family": "quantity_or_arithmetic",
            "recommended_rule_family": "entity_disambiguation",
            "recommended_collection_family": "entity_role_rule_input_collection",
            "recommended_action": "requeue_rule_input_task",
            "question": 'This American producer was born in the 70s and is named "Elon" what?',
            "question_type": "definition",
            "reason_codes": ["quantity_rule_for_entity_or_role_question"],
            "not_verifier_evidence": True,
        })
        + "\n",
        encoding="utf-8",
    )

    payload = requeue_module.run(
        rule_stubs_path=stubs_path,
        requeue_suggestions_path=suggestions_path,
        output_dir=requeue_dir,
        registry_path=registry_path,
        name="rule-stub-requeue-unit",
        version="0.1",
        metadata={"suite": "unit"},
    )
    adapter_payload = adapter_module.run_world_model_rule_authoring_adapter(
        rule_stubs_path=requeue_dir / "requeued-world-model-rule-stubs.jsonl",
        output_dir=adapter_dir,
    )
    plan_payload = plan_module.run(
        input_requests_path=adapter_dir / "world-model-rule-input-requests.jsonl",
        output_dir=plan_dir,
    )
    requeued = [
        json.loads(line)
        for line in (requeue_dir / "requeued-world-model-rule-stubs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tasks = [
        json.loads(line)
        for line in (plan_dir / "rule-input-tasks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    requeue_text = json.dumps(requeued, sort_keys=True)
    registry_record = registry_module.ArtifactRegistry.load_json(registry_path).get(
        "report:rule-stub-requeue-unit:0.1"
    )

    assert payload["status"] == "ready_for_rule_authoring"
    assert payload["summary"]["requeued_stub_count"] == 1
    assert requeued[0]["rule_family"] == "entity_disambiguation"
    assert requeued[0]["required_inputs"] == ["subject_entity", "answer_entity", "requested_role"]
    assert requeued[0]["metadata"]["original_rule_family"] == "quantity_or_arithmetic"
    assert "model_answer" not in requeue_text
    assert "His name is Elon Musk" not in requeue_text
    assert adapter_payload["status"] == "needs_inputs"
    assert adapter_payload["summary"]["rule_family_counts"] == {"entity_disambiguation": 1}
    assert plan_payload["summary"]["collection_family_counts"] == {"entity_role_rule_input_collection": 1}
    assert tasks[0]["collection_family"] == "entity_role_rule_input_collection"
    assert "expected_entity" in tasks[0]["execution_inputs"]
    assert registry_module.load_and_verify_artifact_manifest(requeue_dir / "artifact-manifest.json").passed is True
    assert registry_record is not None
    assert registry_record.metadata["status"] == "ready_for_rule_authoring"


def test_world_model_rule_stub_requeue_skips_missing_source_stub():
    module = importlib.import_module("benchmarks.requeue_world_model_rule_stubs_from_audit")

    payload = module.requeue_world_model_rule_stubs(
        rule_stubs=[],
        requeue_suggestions=[
            {
                "schema_version": 1,
                "workflow": "world_model_rule_input_plan_audit",
                "source_request_id": "rule:missing:1",
                "current_rule_family": "quantity_or_arithmetic",
                "recommended_rule_family": "entity_disambiguation",
                "recommended_action": "requeue_rule_input_task",
                "not_verifier_evidence": True,
            }
        ],
    )

    assert payload["status"] == "empty"
    assert payload["summary"]["requeued_stub_count"] == 0
    assert payload["summary"]["skipped_suggestion_count"] == 1
    assert payload["summary"]["failure_counts"] == {"source_stub_not_found": 1}


def test_world_model_rule_entity_binding_fill_executes_and_promotes_candidate(tmp_path):
    fill_module = importlib.import_module("benchmarks.fill_world_model_rule_inputs_from_entity_bindings")
    adapter_module = importlib.import_module("benchmarks.run_world_model_rule_authoring_adapter")
    promotion_module = importlib.import_module("benchmarks.promote_world_model_rule_candidates")
    registry_module = importlib.import_module("eigentruth.registry")

    tasks_path = tmp_path / "rule-input-tasks.jsonl"
    bindings_path = tmp_path / "source-backed-entity-bindings.jsonl"
    stubs_path = tmp_path / "world-model-rule-stubs.jsonl"
    fill_output = tmp_path / "fill"
    adapter_output = tmp_path / "adapter"
    promotion_output = tmp_path / "promotion"
    registry_path = tmp_path / "registry.json"
    tasks_path.write_text(
        json.dumps({
            "task_id": "rule-input-task-0001",
            "source_request_id": "rule:record-516:1",
            "target_id": "record-516",
            "rule_family": "entity_disambiguation",
            "collection_family": "entity_role_rule_input_collection",
            "question": 'This American producer was born in the 70s and is named "Elon" what?',
            "not_verifier_evidence": True,
        })
        + "\n",
        encoding="utf-8",
    )
    bindings_path.write_text(
        json.dumps({
            "binding_id": "entity-binding-record-516",
            "request_id": "rule:record-516:1",
            "target_id": "record-516",
            "subject_entity": "This American producer named Elon",
            "answer_entity": "Elon Musk",
            "expected_entity": "Elon Gold",
            "requested_role": "American producer born in 1970 named Elon",
            "source_citation": "https://improv.com/comic/elon%2Bgold/",
            "source_url": "https://improv.com/comic/elon%2Bgold/",
            "source_title": "Elon Gold at Improv",
            "source_family": "reference",
            "provider": "improv",
            "candidate_answer_source": "candidate_claim_binding",
            "expected_entity_source": "source_citation",
            "source_note": (
                "Bio identifies Elon Gold as born September 14, 1970 and an American "
                "comedian, actor, writer, and producer."
            ),
            "not_verifier_evidence": True,
        })
        + "\n",
        encoding="utf-8",
    )
    stubs_path.write_text(
        json.dumps({
            "schema_version": 1,
            "workflow": "source_family_structured_qa_lane_batch_workflow",
            "request_id": "rule:record-516:1",
            "target_id": "record-516",
            "request_type": "world_model_or_calculator_rule",
            "rule_family": "entity_disambiguation",
            "required_inputs": ["subject_entity", "answer_entity", "requested_role"],
            "question": 'This American producer was born in the 70s and is named "Elon" what?',
            "not_verifier_evidence": True,
        })
        + "\n",
        encoding="utf-8",
    )

    fill_payload = fill_module.run(
        input_tasks_path=tasks_path,
        entity_bindings_path=bindings_path,
        output_dir=fill_output,
        registry_path=registry_path,
        name="entity-binding-fill-unit",
        version="0.1",
        metadata={"suite": "unit"},
    )
    filled = [
        json.loads(line)
        for line in (fill_output / "rule-inputs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    filled_text = json.dumps(filled, sort_keys=True)

    adapter_payload = adapter_module.run_world_model_rule_authoring_adapter(
        rule_stubs_path=stubs_path,
        rule_inputs_path=fill_output / "rule-inputs.jsonl",
        output_dir=adapter_output,
        metadata={"suite": "unit"},
    )
    promotion_payload = promotion_module.run(
        rule_results_path=adapter_output / "world-model-rule-results.jsonl",
        rule_inputs_path=fill_output / "rule-inputs.jsonl",
        adapter_report_path=adapter_output / "world-model-rule-authoring-adapter.json",
        output_dir=promotion_output,
        registry_path=registry_path,
        name="entity-binding-promotion-unit",
        version="0.1",
        metadata={"suite": "unit"},
    )
    promoted = [
        json.loads(line)
        for line in (promotion_output / "promoted-rule-candidates.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert fill_payload["status"] == "filled"
    assert fill_payload["summary"]["filled_input_count"] == 1
    assert filled[0]["answer_entity"] == "Elon Musk"
    assert filled[0]["expected_entity"] == "Elon Gold"
    assert "model_answer" not in filled_text
    assert adapter_payload["status"] == "observed"
    assert adapter_payload["summary"]["candidate_refuted_count"] == 1
    assert promotion_payload["status"] == "promote"
    assert promotion_payload["summary"]["promoted_count"] == 1
    assert promoted[0]["status"] == "refuted"
    assert promoted[0]["source_citation"] == "https://improv.com/comic/elon%2Bgold/"
    assert "https://improv.com/comic/elon%2Bgold/" in promoted[0]["evidence"][0]
    assert registry_module.load_and_verify_artifact_manifest(fill_output / "artifact-manifest.json").passed is True


def test_world_model_rule_entity_binding_fill_blocks_invalid_binding():
    module = importlib.import_module("benchmarks.fill_world_model_rule_inputs_from_entity_bindings")

    payload = module.fill_world_model_rule_inputs_from_entity_bindings(
        input_tasks=[
            {
                "task_id": "rule-input-task-0001",
                "source_request_id": "rule:record-516:1",
                "target_id": "record-516",
                "rule_family": "entity_disambiguation",
                "collection_family": "entity_role_rule_input_collection",
                "question": 'This American producer was born in the 70s and is named "Elon" what?',
                "not_verifier_evidence": True,
            }
        ],
        entity_bindings=[
            {
                "binding_id": "entity-binding-record-516",
                "request_id": "rule:record-516:1",
                "target_id": "record-516",
                "subject_entity": "This American producer named Elon",
                "answer_entity": "Elon Musk",
                "expected_entity": "Elon Gold",
                "requested_role": "American producer born in 1970 named Elon",
                "not_verifier_evidence": False,
            }
        ],
    )

    assert payload["status"] == "blocked"
    assert payload["summary"]["filled_input_count"] == 0
    assert payload["summary"]["unfilled_reason_counts"] == {"invalid_entity_binding": 1}
    assert payload["summary"]["invalid_binding_failure_counts"] == {
        "binding_not_marked_non_evidence": 1,
        "missing_source_citation": 1,
    }
    assert payload["unfilled_tasks"][0]["reason"] == "invalid_entity_binding"


def test_world_model_rule_numeric_binding_fill_executes_and_promotes_candidate(tmp_path):
    fill_module = importlib.import_module("benchmarks.fill_world_model_rule_inputs_from_numeric_bindings")
    adapter_module = importlib.import_module("benchmarks.run_world_model_rule_authoring_adapter")
    promotion_module = importlib.import_module("benchmarks.promote_world_model_rule_candidates")
    registry_module = importlib.import_module("eigentruth.registry")

    tasks_path = tmp_path / "rule-input-tasks.jsonl"
    bindings_path = tmp_path / "source-backed-numeric-bindings.jsonl"
    stubs_path = tmp_path / "world-model-rule-stubs.jsonl"
    fill_output = tmp_path / "fill"
    adapter_output = tmp_path / "adapter"
    promotion_output = tmp_path / "promotion"
    tasks_path.write_text(
        json.dumps({
            "task_id": "rule-input-task-0001",
            "source_request_id": "rule:record-190:1",
            "target_id": "record-190",
            "rule_family": "quantity_or_arithmetic",
            "collection_family": "numeric_rule_input_collection",
            "question": "What is the population of the country?",
            "not_verifier_evidence": True,
        })
        + "\n",
        encoding="utf-8",
    )
    bindings_path.write_text(
        json.dumps({
            "binding_id": "numeric-binding-record-190",
            "request_id": "rule:record-190:1",
            "target_id": "record-190",
            "subject_entity": "United States",
            "candidate_numeric_value": 330000000,
            "source_numeric_value": 340110988,
            "unit": "persons",
            "reference_time": "2024",
            "tolerance": 0.0,
            "source_citation": "worldbank:SP.POP.TOTL:USA:2024",
            "source_url": "https://data.worldbank.org/indicator/SP.POP.TOTL?locations=US",
            "source_title": "World Bank official statistics: Population, total for United States (2024)",
            "source_family": "official_statistics",
            "provider": "worldbank",
            "candidate_value_source": "candidate_claim_binding",
            "source_value_source": "worldbank_catalog",
            "review_status": "ready",
            "not_verifier_evidence": True,
        })
        + "\n",
        encoding="utf-8",
    )
    stubs_path.write_text(
        json.dumps({
            "schema_version": 1,
            "workflow": "unresolved_world_model_rule_stubs",
            "request_id": "rule:record-190:1",
            "target_id": "record-190",
            "request_type": "world_model_or_calculator_rule",
            "rule_family": "quantity_or_arithmetic",
            "required_inputs": ["numeric_value", "unit", "reference_time"],
            "question": "What is the population of the country?",
            "not_verifier_evidence": True,
        })
        + "\n",
        encoding="utf-8",
    )

    fill_payload = fill_module.run(
        input_tasks_path=tasks_path,
        numeric_bindings_path=bindings_path,
        output_dir=fill_output,
        registry_path=tmp_path / "registry.json",
        name="numeric-binding-fill-unit",
        version="0.1",
        metadata={"suite": "unit"},
    )
    filled = [
        json.loads(line)
        for line in (fill_output / "rule-inputs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    adapter_payload = adapter_module.run_world_model_rule_authoring_adapter(
        rule_stubs_path=stubs_path,
        rule_inputs_path=fill_output / "rule-inputs.jsonl",
        output_dir=adapter_output,
        metadata={"suite": "unit"},
    )
    promotion_payload = promotion_module.run(
        rule_results_path=adapter_output / "world-model-rule-results.jsonl",
        rule_inputs_path=fill_output / "rule-inputs.jsonl",
        adapter_report_path=adapter_output / "world-model-rule-authoring-adapter.json",
        output_dir=promotion_output,
        metadata={"suite": "unit"},
    )
    results = [
        json.loads(line)
        for line in (adapter_output / "world-model-rule-results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    promoted = [
        json.loads(line)
        for line in (promotion_output / "promoted-rule-candidates.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert fill_payload["status"] == "filled"
    assert filled[0]["numeric_value"] == 340110988
    assert filled[0]["candidate_numeric_value"] == 330000000
    assert filled[0]["calculation"]["expression"] == "(340110988) - (330000000)"
    assert adapter_payload["status"] == "observed"
    assert adapter_payload["summary"]["candidate_refuted_count"] == 1
    assert results[0]["status"] == "refuted"
    assert "worldbank:SP.POP.TOTL:USA:2024" in " ".join(results[0]["evidence"])
    assert promotion_payload["status"] == "promote"
    assert promoted[0]["source_citation"] == "worldbank:SP.POP.TOTL:USA:2024"
    assert promoted[0]["rule_input"]["numeric_value"] == 340110988
    assert promoted[0]["rule_input"]["calculation"]["expected"] == 0.0
    assert registry_module.load_and_verify_artifact_manifest(fill_output / "artifact-manifest.json").passed is True


def test_world_model_rule_numeric_binding_fill_blocks_review_required_binding():
    module = importlib.import_module("benchmarks.fill_world_model_rule_inputs_from_numeric_bindings")

    payload = module.fill_world_model_rule_inputs_from_numeric_bindings(
        input_tasks=[
            {
                "task_id": "rule-input-task-0001",
                "source_request_id": "rule:record-190:1",
                "target_id": "record-190",
                "rule_family": "quantity_or_arithmetic",
                "collection_family": "numeric_rule_input_collection",
                "question": "What is the population of the country?",
                "not_verifier_evidence": True,
            }
        ],
        numeric_bindings=[
            {
                "binding_id": "numeric-binding-record-190",
                "request_id": "rule:record-190:1",
                "target_id": "record-190",
                "candidate_numeric_value": 330000000,
                "source_numeric_value": 340110988,
                "unit": "persons",
                "reference_time": "2024",
                "source_citation": "worldbank:SP.POP.TOTL:USA:2024",
                "review_status": "ambiguous_subject",
                "not_verifier_evidence": True,
            }
        ],
    )

    assert payload["status"] == "blocked"
    assert payload["summary"]["filled_input_count"] == 0
    assert payload["summary"]["unfilled_reason_counts"] == {"invalid_numeric_binding": 1}
    assert payload["summary"]["invalid_binding_failure_counts"] == {
        "binding_requires_review": 1,
        "missing_subject_entity": 1,
    }


def test_world_model_rule_temporal_consistency_executes_and_promotes_candidate(tmp_path):
    adapter_module = importlib.import_module("benchmarks.run_world_model_rule_authoring_adapter")
    promotion_module = importlib.import_module("benchmarks.promote_world_model_rule_candidates")

    stubs_path = tmp_path / "world-model-rule-stubs.jsonl"
    inputs_path = tmp_path / "rule-inputs.jsonl"
    adapter_output = tmp_path / "adapter"
    promotion_output = tmp_path / "promotion"
    stubs_path.write_text(
        json.dumps({
            "schema_version": 1,
            "workflow": "unresolved_world_model_rule_stubs",
            "request_id": "rule:record-326:1",
            "target_id": "record-326",
            "request_type": "world_model_or_calculator_rule",
            "rule_family": "temporal_consistency",
            "required_inputs": ["claim_time", "source_time", "retrieved_at", "source_citation"],
            "question": "What happened to the affordability of food in America in recent decades?",
            "not_verifier_evidence": True,
        })
        + "\n",
        encoding="utf-8",
    )
    inputs_path.write_text(
        json.dumps({
            "request_id": "rule:record-326:1",
            "target_id": "record-326",
            "rule_family": "temporal_consistency",
            "claim_time": "2026-06-10",
            "source_time": "2026-06-10",
            "retrieved_at": "2026-06-28T15:51:33+00:00",
            "source_citation": "https://www.pbs.org/newshour/economy/food-affordability",
            "source_url": "https://www.pbs.org/newshour/economy/food-affordability",
            "source_family": "news",
            "provider": "pbs_newshour",
            "not_verifier_evidence": True,
            "candidate_results_require_promotion_gate": True,
        })
        + "\n",
        encoding="utf-8",
    )

    adapter_payload = adapter_module.run_world_model_rule_authoring_adapter(
        rule_stubs_path=stubs_path,
        rule_inputs_path=inputs_path,
        output_dir=adapter_output,
        metadata={"suite": "unit"},
    )
    promotion_payload = promotion_module.run(
        rule_results_path=adapter_output / "world-model-rule-results.jsonl",
        rule_inputs_path=inputs_path,
        adapter_report_path=adapter_output / "world-model-rule-authoring-adapter.json",
        output_dir=promotion_output,
        metadata={"suite": "unit"},
    )
    result = json.loads((adapter_output / "world-model-rule-results.jsonl").read_text(encoding="utf-8"))
    promoted = json.loads((promotion_output / "promoted-rule-candidates.jsonl").read_text(encoding="utf-8"))

    assert adapter_payload["status"] == "observed"
    assert adapter_payload["summary"]["candidate_supported_count"] == 1
    assert result["status"] == "supported"
    assert "source_time covers claim_time" in result["evidence"][0]
    assert "https://www.pbs.org/newshour/economy/food-affordability" in result["evidence"][0]
    assert promotion_payload["status"] == "promote"
    assert promoted["rule_family"] == "temporal_consistency"
    assert promoted["rule_input"]["claim_time"] == "2026-06-10"
    assert promoted["rule_input"]["retrieved_at"] == "2026-06-28T15:51:33+00:00"


def test_world_model_rule_temporal_consistency_blocks_future_source_time(tmp_path):
    adapter_module = importlib.import_module("benchmarks.run_world_model_rule_authoring_adapter")
    promotion_module = importlib.import_module("benchmarks.promote_world_model_rule_candidates")

    stubs_path = tmp_path / "world-model-rule-stubs.jsonl"
    inputs_path = tmp_path / "rule-inputs.jsonl"
    adapter_output = tmp_path / "adapter"
    promotion_output = tmp_path / "promotion"
    stubs_path.write_text(
        json.dumps({
            "request_id": "rule:record-326:1",
            "target_id": "record-326",
            "rule_family": "temporal_consistency",
            "required_inputs": ["claim_time", "source_time", "retrieved_at", "source_citation"],
            "question": "What happened to the affordability of food in America in recent decades?",
            "not_verifier_evidence": True,
        })
        + "\n",
        encoding="utf-8",
    )
    inputs_path.write_text(
        json.dumps({
            "request_id": "rule:record-326:1",
            "target_id": "record-326",
            "rule_family": "temporal_consistency",
            "claim_time": "2026-06-10",
            "source_time": "2026-07-01",
            "retrieved_at": "2026-06-28",
            "source_citation": "https://www.pbs.org/newshour/economy/food-affordability",
            "not_verifier_evidence": True,
            "candidate_results_require_promotion_gate": True,
        })
        + "\n",
        encoding="utf-8",
    )

    adapter_payload = adapter_module.run_world_model_rule_authoring_adapter(
        rule_stubs_path=stubs_path,
        rule_inputs_path=inputs_path,
        output_dir=adapter_output,
        metadata={"suite": "unit"},
    )
    promotion_payload = promotion_module.run(
        rule_results_path=adapter_output / "world-model-rule-results.jsonl",
        rule_inputs_path=inputs_path,
        adapter_report_path=adapter_output / "world-model-rule-authoring-adapter.json",
        output_dir=promotion_output,
        metadata={"suite": "unit"},
    )
    result = json.loads((adapter_output / "world-model-rule-results.jsonl").read_text(encoding="utf-8"))
    blocked = json.loads((promotion_output / "blocked-rule-candidates.jsonl").read_text(encoding="utf-8"))

    assert adapter_payload["status"] == "observed"
    assert adapter_payload["summary"]["candidate_error_count"] == 1
    assert result["status"] == "error"
    assert "source_time occurs after retrieved_at" in result["evidence"][0]
    assert promotion_payload["status"] == "blocked"
    assert blocked["failures"] == ["status_not_promotable"]


def test_rule_input_correction_handoff_fill_executes_entity_role_candidate(tmp_path):
    fill_module = importlib.import_module("benchmarks.fill_world_model_rule_inputs_from_correction_handoff")
    adapter_module = importlib.import_module("benchmarks.run_world_model_rule_authoring_adapter")
    registry_module = importlib.import_module("eigentruth.registry")

    tasks_path = tmp_path / "rule-input-tasks.jsonl"
    handoff_path = tmp_path / "correction-handoff.json"
    qa_path = tmp_path / "qa-corpus.json"
    traces_path = tmp_path / "product-traces.jsonl"
    fill_output = tmp_path / "fill"
    adapter_output = tmp_path / "adapter"
    registry_path = tmp_path / "registry.json"
    tasks_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "task_id": "rule-input-task-0001",
                    "source_request_id": "rule:record-1:1",
                    "target_id": "record-1",
                    "rule_family": "entity_disambiguation",
                    "collection_family": "entity_role_rule_input_collection",
                    "question": "Who first started Tesla Motors?",
                    "not_verifier_evidence": True,
                },
                {
                    "task_id": "rule-input-task-0002",
                    "source_request_id": "rule:record-11:1",
                    "target_id": "record-11",
                    "rule_family": "quantity_or_arithmetic",
                    "collection_family": "numeric_rule_input_collection",
                    "question": "Do more than 20% of Americans have passports?",
                    "not_verifier_evidence": True,
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    handoff_path.write_text(
        json.dumps({
            "schema_version": 1,
            "workflow": "source_family_structured_qa_correction_handoff",
            "status": "promote",
            "summary": {"correction_candidate_count": 1},
        }),
        encoding="utf-8",
    )
    qa_path.write_text(
        json.dumps({
            "schema_version": 1,
            "corpus_type": "target_specific_source_family_structured_qa_correction",
            "documents": [
                {
                    "question": "Who first started Tesla Motors?",
                    "answer": "Martin Eberhard",
                    "source": "wikidata:Q478214:P112:Q1903673",
                    "metadata": {
                        "subject": "Tesla",
                        "statement_property": "P112",
                        "statement_property_label": "founder",
                        "provider": "wikidata",
                        "source_family": "reference",
                        "url": "https://www.wikidata.org/wiki/Q478214",
                    },
                }
            ],
        }),
        encoding="utf-8",
    )
    traces_path.write_text(
        json.dumps({
            "schema_version": 1,
            "request_id": "source-family-structured-qa-record-38",
            "claims": [
                {
                    "claim_id": "source-family-structured-qa-record-38:model-answer",
                    "text": "Who first started Tesla Motors? Elon Musk founded Tesla.",
                    "metadata": {
                        "question": "Who first started Tesla Motors?",
                        "answer": "Elon Musk founded Tesla.",
                    },
                }
            ],
        })
        + "\n",
        encoding="utf-8",
    )

    payload = fill_module.run(
        input_tasks_path=tasks_path,
        correction_handoff_path=handoff_path,
        qa_corpus_path=qa_path,
        product_traces_path=traces_path,
        output_dir=fill_output,
        registry_path=registry_path,
        name="rule-input-fill-unit",
        version="0.1",
        metadata={"suite": "unit"},
    )
    filled = [
        json.loads(line)
        for line in (fill_output / "rule-inputs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    unfilled = [
        json.loads(line)
        for line in (fill_output / "unfilled-rule-input-tasks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    registry_record = registry_module.ArtifactRegistry.load_json(registry_path).get(
        "report:rule-input-fill-unit:0.1"
    )

    assert payload["status"] == "partial"
    assert payload["summary"]["filled_input_count"] == 1
    assert payload["summary"]["unfilled_task_count"] == 1
    assert filled[0]["request_id"] == "rule:record-1:1"
    assert filled[0]["subject_entity"] == "Tesla"
    assert filled[0]["answer_entity"] == "Elon Musk"
    assert filled[0]["expected_entity"] == "Martin Eberhard"
    assert filled[0]["requested_role"] == "founder"
    assert filled[0]["source_citation"] == "wikidata:Q478214:P112:Q1903673"
    assert filled[0]["not_verifier_evidence"] is True
    assert unfilled[0]["reason"] == "unsupported_collection_family"
    assert registry_module.load_and_verify_artifact_manifest(fill_output / "artifact-manifest.json").passed is True
    assert registry_record is not None
    assert registry_record.metadata["status"] == "partial"

    stubs_path = tmp_path / "world-model-rule-stubs.jsonl"
    stubs_path.write_text(
        json.dumps({
            "schema_version": 1,
            "workflow": "source_family_structured_qa_lane_batch_workflow",
            "request_id": "rule:record-1:1",
            "target_id": "record-1",
            "request_type": "world_model_or_calculator_rule",
            "rule_family": "entity_disambiguation",
            "required_inputs": ["subject_entity", "answer_entity", "requested_role"],
            "question": "Who first started Tesla Motors?",
            "not_verifier_evidence": True,
        })
        + "\n",
        encoding="utf-8",
    )
    adapter_payload = adapter_module.run_world_model_rule_authoring_adapter(
        rule_stubs_path=stubs_path,
        rule_inputs_path=fill_output / "rule-inputs.jsonl",
        output_dir=adapter_output,
        metadata={"suite": "unit"},
    )
    results = [
        json.loads(line)
        for line in (adapter_output / "world-model-rule-results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert adapter_payload["status"] == "observed"
    assert adapter_payload["summary"]["candidate_refuted_count"] == 1
    assert results[0]["status"] == "refuted"
    assert "source_citation=wikidata:Q478214:P112:Q1903673" in results[0]["evidence"][0]


def test_world_model_rule_candidate_promotion_gate_promotes_source_backed_candidate(tmp_path):
    module = importlib.import_module("benchmarks.promote_world_model_rule_candidates")
    registry_module = importlib.import_module("eigentruth.registry")

    results_path = tmp_path / "world-model-rule-results.jsonl"
    inputs_path = tmp_path / "rule-inputs.jsonl"
    adapter_report_path = tmp_path / "adapter-report.json"
    output_dir = tmp_path / "promotion"
    registry_path = tmp_path / "registry.json"
    results_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "schema_version": 1,
                    "workflow": "world_model_rule_authoring_adapter",
                    "request_id": "rule:record-1:1",
                    "target_id": "record-1",
                    "rule_family": "entity_disambiguation",
                    "status": "refuted",
                    "confidence": 0.95,
                    "missing_inputs": [],
                    "question": "Who first started Tesla Motors?",
                    "authored_rule": {"adapter": "entity_role_disambiguation"},
                    "metadata": {
                        "adapter": "entity_role_disambiguation",
                        "candidate_results_require_promotion_gate": True,
                    },
                    "evidence": [
                        "entity_role: requested_role=founder; "
                        "answer_entity=Elon Musk; expected_entity=Martin Eberhard; "
                        "source_citation=wikidata:Q478214:P112:Q1903673"
                    ],
                    "not_verifier_evidence": True,
                },
                {
                    "schema_version": 1,
                    "workflow": "world_model_rule_authoring_adapter",
                    "request_id": "rule:record-11:1",
                    "target_id": "record-11",
                    "rule_family": "quantity_or_arithmetic",
                    "status": "needs_inputs",
                    "missing_inputs": ["numeric_value"],
                    "not_verifier_evidence": True,
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    inputs_path.write_text(
        json.dumps({
            "schema_version": 1,
            "workflow": "world_model_rule_input_correction_handoff_fill",
            "request_id": "rule:record-1:1",
            "target_id": "record-1",
            "rule_family": "entity_disambiguation",
            "subject_entity": "Tesla",
            "answer_entity": "Elon Musk",
            "expected_entity": "Martin Eberhard",
            "requested_role": "founder",
            "source_citation": "wikidata:Q478214:P112:Q1903673",
            "source_url": "https://www.wikidata.org/wiki/Q478214",
            "source_fact_type": "P112",
            "source_family": "reference",
            "provider": "wikidata",
            "not_verifier_evidence": True,
            "candidate_results_require_promotion_gate": True,
        })
        + "\n",
        encoding="utf-8",
    )
    adapter_report_path.write_text(
        json.dumps({
            "schema_version": 1,
            "workflow": "world_model_rule_authoring_adapter",
            "status": "partial",
            "summary": {"executed_count": 1, "needs_input_count": 1},
        }),
        encoding="utf-8",
    )

    payload = module.run(
        rule_results_path=results_path,
        rule_inputs_path=inputs_path,
        adapter_report_path=adapter_report_path,
        output_dir=output_dir,
        registry_path=registry_path,
        name="rule-candidate-promotion-unit",
        version="0.1",
        metadata={"suite": "unit"},
    )
    promoted = [
        json.loads(line)
        for line in (output_dir / "promoted-rule-candidates.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pending = [
        json.loads(line)
        for line in (output_dir / "pending-rule-inputs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    blocked = (output_dir / "blocked-rule-candidates.jsonl").read_text(encoding="utf-8").strip()
    registry_record = registry_module.ArtifactRegistry.load_json(registry_path).get(
        "report:rule-candidate-promotion-unit:0.1"
    )

    assert payload["status"] == "promote"
    assert payload["summary"]["promoted_count"] == 1
    assert payload["summary"]["pending_count"] == 1
    assert payload["summary"]["blocked_count"] == 0
    assert promoted[0]["request_id"] == "rule:record-1:1"
    assert promoted[0]["status"] == "refuted"
    assert promoted[0]["source_citation"] == "wikidata:Q478214:P112:Q1903673"
    assert promoted[0]["promotion"]["status"] == "promote"
    assert pending[0]["request_id"] == "rule:record-11:1"
    assert blocked == ""
    assert registry_module.load_and_verify_artifact_manifest(output_dir / "artifact-manifest.json").passed is True
    assert registry_record is not None
    assert registry_record.metadata["status"] == "promote"


def test_world_model_rule_candidate_promotion_gate_blocks_missing_citation(tmp_path):
    module = importlib.import_module("benchmarks.promote_world_model_rule_candidates")

    payload = module.promote_world_model_rule_candidates(
        rule_results=[
            {
                "request_id": "rule:bad:1",
                "target_id": "record-bad",
                "rule_family": "entity_disambiguation",
                "status": "refuted",
                "confidence": 0.95,
                "missing_inputs": [],
                "authored_rule": {"adapter": "entity_role_disambiguation"},
                "metadata": {"candidate_results_require_promotion_gate": True},
                "evidence": ["entity_role: requested_role=founder"],
                "not_verifier_evidence": True,
            }
        ],
        rule_inputs=[
            {
                "request_id": "rule:bad:1",
                "rule_family": "entity_disambiguation",
                "subject_entity": "Tesla",
                "answer_entity": "Elon Musk",
                "expected_entity": "Martin Eberhard",
                "requested_role": "founder",
                "not_verifier_evidence": True,
                "candidate_results_require_promotion_gate": True,
            }
        ],
    )

    assert payload["status"] == "blocked"
    assert payload["summary"]["promoted_count"] == 0
    assert payload["summary"]["blocked_count"] == 1
    assert payload["blocked_candidates"][0]["failures"] == ("missing_source_citation",)


def test_world_model_rule_candidate_handoff_writes_trace_and_action_results(tmp_path):
    module = importlib.import_module("benchmarks.build_world_model_rule_candidate_handoff")
    registry_module = importlib.import_module("eigentruth.registry")

    promotion_gate_path = tmp_path / "promotion-gate.json"
    promoted_path = tmp_path / "promoted-rule-candidates.jsonl"
    output_dir = tmp_path / "handoff"
    registry_path = tmp_path / "registry.json"
    promotion_gate_path.write_text(
        json.dumps({
            "schema_version": 1,
            "workflow": "world_model_rule_candidate_promotion_gate",
            "status": "promote",
            "summary": {
                "promoted_count": 2,
                "blocked_count": 0,
                "pending_count": 0,
            },
        }),
        encoding="utf-8",
    )
    promoted_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "schema_version": 1,
                    "workflow": "world_model_rule_candidate_promotion_gate",
                    "request_id": "rule:record-1:1",
                    "target_id": "record-1",
                    "rule_family": "entity_disambiguation",
                    "status": "refuted",
                    "confidence": 0.95,
                    "adapter": "entity_role_disambiguation",
                    "question": "Who first started Tesla Motors?",
                    "source_citation": "wikidata:Q478214:P112:Q1903673",
                    "source_url": "https://www.wikidata.org/wiki/Q478214",
                    "evidence": [
                        "entity_role: requested_role=founder; "
                        "answer_entity=Elon Musk; expected_entity=Martin Eberhard; "
                        "source_citation=wikidata:Q478214:P112:Q1903673"
                    ],
                    "rule_input": {
                        "subject_entity": "Tesla",
                        "answer_entity": "Elon Musk",
                        "expected_entity": "Martin Eberhard",
                        "requested_role": "founder",
                    },
                    "promotion": {
                        "status": "promote",
                        "gate": "world_model_rule_candidate_promotion_gate",
                        "candidate_only_requires_downstream_handoff": True,
                    },
                },
                {
                    "schema_version": 1,
                    "workflow": "world_model_rule_candidate_promotion_gate",
                    "request_id": "rule:record-2:1",
                    "target_id": "record-2",
                    "rule_family": "entity_disambiguation",
                    "status": "supported",
                    "confidence": 0.92,
                    "adapter": "entity_role_disambiguation",
                    "question": "Who founded Acme Motors?",
                    "source_citation": "wikidata:Q1:P112:Q2",
                    "evidence": [
                        "entity_role: requested_role=founder; "
                        "answer_entity=Bob Builder; expected_entity=Bob Builder; "
                        "source_citation=wikidata:Q1:P112:Q2"
                    ],
                    "rule_input": {
                        "subject_entity": "Acme Motors",
                        "answer_entity": "Bob Builder",
                        "expected_entity": "Bob Builder",
                        "requested_role": "founder",
                    },
                    "promotion": {
                        "status": "promote",
                        "gate": "world_model_rule_candidate_promotion_gate",
                        "candidate_only_requires_downstream_handoff": True,
                    },
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )

    payload = module.run(
        promotion_gate_path=promotion_gate_path,
        promoted_candidates_path=promoted_path,
        output_dir=output_dir,
        registry_path=registry_path,
        name="rule-candidate-handoff-unit",
        version="0.1",
        metadata={"suite": "unit"},
    )
    report = payload["report"]
    traces = [
        json.loads(line)
        for line in (output_dir / "product-traces.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    action_results = [
        json.loads(line)
        for line in (output_dir / "action-results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    registry = registry_module.ArtifactRegistry.load_json(registry_path)
    report_record = registry.get("report:rule-candidate-handoff-unit:0.1")
    trace_record = registry.get("product_trace:rule-candidate-handoff-unit:0.1")
    action_record = registry.get("action_result:rule-candidate-handoff-unit:0.1")

    assert report["status"] == "promote"
    assert report["summary"]["input_candidate_count"] == 2
    assert report["summary"]["blocked_candidate_count"] == 0
    assert report["summary"]["verification_status_counts"] == {"refuted": 1, "supported": 1}
    assert report["summary"]["action_counts"] == {"abstain": 1, "accept": 1}
    assert report["summary"]["action_execution_alignment_passed"] is True
    assert traces[0]["verification_results"][0]["metadata"]["selected_route"] == "world_model_rule_candidate"
    assert traces[0]["risk_decision"]["action"] == "abstain"
    assert traces[0]["risk_decision"]["risk_level"] == "high"
    assert traces[1]["risk_decision"]["action"] == "accept"
    assert traces[1]["risk_decision"]["risk_level"] == "low"
    assert traces[0]["metadata"]["not_open_domain_verifier"] is True
    assert action_results[0]["status"] == "dry_run"
    assert action_results[1]["status"] == "dry_run"
    assert registry_module.load_and_verify_artifact_manifest(output_dir / "artifact-manifest.json").passed is True
    assert report_record.metadata["workflow"] == "world_model_rule_candidate_handoff"
    assert trace_record.metadata["trace_count"] == 2
    assert action_record.metadata["action_result_count"] == 2


def test_world_model_rule_candidate_handoff_blocks_missing_handoff_marker():
    module = importlib.import_module("benchmarks.build_world_model_rule_candidate_handoff")

    payload = module.build_world_model_rule_candidate_handoff(
        {
            "workflow": "world_model_rule_candidate_promotion_gate",
            "status": "promote",
            "summary": {"promoted_count": 1},
        },
        promoted_candidates=[
            {
                "workflow": "world_model_rule_candidate_promotion_gate",
                "request_id": "rule:record-1:1",
                "target_id": "record-1",
                "rule_family": "entity_disambiguation",
                "status": "refuted",
                "confidence": 0.95,
                "source_citation": "wikidata:Q1:P112:Q2",
                "evidence": ["source_citation=wikidata:Q1:P112:Q2"],
                "promotion": {
                    "status": "promote",
                    "gate": "world_model_rule_candidate_promotion_gate",
                },
            }
        ],
    )

    assert payload["report"]["status"] == "blocked"
    assert payload["report"]["summary"]["trace_count"] == 0
    assert payload["report"]["summary"]["blocked_candidate_count"] == 1
    assert payload["report"]["blocked_candidates"][0]["failures"] == ("missing_downstream_handoff_marker",)
    assert payload["product_traces"] == ()
    assert payload["action_results"] == ()

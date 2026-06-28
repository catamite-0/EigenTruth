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

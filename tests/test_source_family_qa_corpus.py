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

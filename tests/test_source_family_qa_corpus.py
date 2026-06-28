import importlib
import json


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

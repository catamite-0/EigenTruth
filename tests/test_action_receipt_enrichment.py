"""Tests for ProductTrace action receipt enrichment benchmark."""

import json

from benchmarks.enrich_product_trace_action_receipts import (
    ProductTraceActionReceiptEnrichmentConfig,
    build_product_trace_action_receipt_enrichment,
)
from benchmarks.run_product_runtime_baseline import (
    ProductRuntimeBaselineConfig,
    build_product_runtime_baseline,
)


def test_action_receipt_enrichment_adds_stable_ids_receipts_and_claim_references(tmp_path):
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(
        json.dumps(
            {
                "request_id": "unit-trace",
                "diagnostics": {},
                "claims": [
                    {
                        "claim_id": "c1",
                        "text": "2 + 2 = 5.",
                        "metadata": {"features": {"has_calculation": True}},
                    }
                ],
                "verification_results": [],
                "actions": [
                    {
                        "action": "abstain",
                        "reason": "calculator refuted claim",
                        "payload": {"blocked_claims": [{"claim_id": "c1"}]},
                        "metadata": {"policy": "unit"},
                        "request_id": None,
                    }
                ],
                "action_results": [
                    {
                        "action": "abstain",
                        "status": "dry_run",
                        "output": {
                            "would_execute": "abstain",
                            "blocked_claims": [{"claim_id": "c1"}],
                        },
                        "metadata": {"executor": "DryRunActionExecutor"},
                        "request_id": None,
                        "error": None,
                    }
                ],
                "metadata": {"source": "unit"},
            }
        ),
        encoding="utf-8",
    )

    report = build_product_trace_action_receipt_enrichment(
        ProductTraceActionReceiptEnrichmentConfig(
            trace_paths=(trace_path,),
            output_dir=tmp_path / "receipt-enriched",
            secret="unit-secret",
            key_id="unit-key",
            issuer="unit-test",
        )
    )

    assert report["status"] == "promote"
    assert report["summary"]["action_receipts"]["coverage_rate"] == 1.0
    assert report["summary"]["receipt_claim_support"]["reference_support_rate"] == 1.0

    output_path = report["traces"][0]["output_path"]
    enriched = json.loads(open(output_path, encoding="utf-8").read())
    action = enriched["actions"][0]
    result = enriched["action_results"][0]
    claim_metadata = enriched["claims"][0]["metadata"]
    receipt = result["metadata"]["action_receipt"]

    assert action["request_id"] == "unit-trace-abstain-1"
    assert result["request_id"] == "unit-trace-abstain-1"
    assert result["status"] == "dry_run"
    assert receipt["signature_algorithm"] == "hmac-sha256"
    assert receipt["key_id"] == "unit-key"
    assert claim_metadata["action_request_ids"] == ["unit-trace-abstain-1"]
    assert claim_metadata["receipt_reference_source"] == "product_trace_action_receipt_enrichment"
    assert enriched["summaries"]["action_receipts"]["passed"] is True
    assert enriched["summaries"]["receipt_claim_support"]["passed"] is True
    assert enriched["summaries"]["receipt_claim_support"]["policy"]["accepted_statuses"] == [
        "succeeded",
        "dry_run",
    ]

    baseline = build_product_runtime_baseline(
        ProductRuntimeBaselineConfig(
            trace_paths=(output_path,),
            report_path=tmp_path / "runtime-baseline.json",
            artifact_manifest_path=tmp_path / "runtime-baseline-manifest.json",
        )
    )

    assert baseline["summary"]["action_receipts"]["coverage_rate"] == 1.0
    assert baseline["summary"]["action_receipts"]["unsigned_receipt_rate"] == 0.0
    assert baseline["summary"]["receipt_claim_support"]["reference_support_rate"] == 1.0
    assert baseline["summary"]["receipt_claim_support"]["failed_result_reference_rate"] == 0.0

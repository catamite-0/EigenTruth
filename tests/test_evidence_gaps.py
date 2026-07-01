"""Tests for release-evidence gap planning."""

import json
from pathlib import Path

from benchmarks.plan_citation_batch_evidence_reruns import build_citation_batch_evidence_rerun_queue
from benchmarks.plan_frontier_abstention_evidence_reruns import build_frontier_abstention_evidence_rerun_queue
from benchmarks.plan_frontier_detectability_evidence_reruns import build_frontier_detectability_evidence_rerun_queue
from benchmarks.plan_frontier_multiple_testing_reruns import build_frontier_multiple_testing_rerun_queue
from benchmarks.plan_frontier_stability_evidence_reruns import build_frontier_stability_evidence_rerun_queue
from benchmarks.plan_release_evidence_gaps import build_release_evidence_gap_plan
from benchmarks.rollup_frontier_abstention_evidence_reruns import (
    rollup_frontier_abstention_evidence_reruns,
)
from benchmarks.rollup_frontier_detectability_evidence_reruns import (
    rollup_frontier_detectability_evidence_reruns,
)
from benchmarks.rollup_frontier_multiple_testing_reruns import (
    rollup_frontier_multiple_testing_reruns,
)
from benchmarks.rollup_frontier_stability_evidence_reruns import (
    rollup_frontier_stability_evidence_reruns,
)
from eigentruth.control import (
    EvidenceGapPlan,
    plan_evidence_gaps_from_release_candidate,
)
from eigentruth.json_utils import strict_json_dumps
from eigentruth.registry import ArtifactRegistry

_MULTIPLE_TESTING_RERUN_COMMANDS = (
    "benchmarks/plan_frontier_multiple_testing_reruns.py --source ... --json ...",
    "benchmarks/run_truthfulqa_frontier_workflow.py --multiple-testing-signals ...",
    "benchmarks/rollup_frontier_multiple_testing_reruns.py --queue ... --json ...",
    "benchmarks/compare_frontier_release_evidence.py --frontier-rerun-rollup-report ...",
)

_ABSTENTION_RERUN_COMMANDS = (
    "benchmarks/plan_frontier_abstention_evidence_reruns.py --source ... --json ...",
    "benchmarks/eval_abstention_stability.py --json ...",
    "benchmarks/rollup_frontier_abstention_evidence_reruns.py --queue ... --json ...",
    "benchmarks/compare_frontier_release_evidence.py --frontier-rerun-rollup-report ...",
)

_CITATION_BATCH_RERUN_COMMANDS = (
    "benchmarks/plan_citation_batch_evidence_reruns.py --source ... --json ...",
    "benchmarks/run_external_citation_search_adapter_workflow.py "
    "--queue ... --batch-id ... --workflow-report ...",
    "benchmarks/run_source_family_citation_search_workflow.py "
    "--queue ... --batch-id ... --workflow-report ...",
    "benchmarks/rollup_citation_search_batch_evidence.py --queue ... --batch-report ... --json ...",
    "benchmarks/compare_frontier_release_evidence.py --citation-batch-rollup-report ...",
)

_DETECTABILITY_RERUN_COMMANDS = (
    "benchmarks/plan_frontier_detectability_evidence_reruns.py --source ... --json ...",
    "benchmarks/analyze_detectability_blind_spots.py --taxonomy-report ... --json ...",
    "benchmarks/eval_detectability_taxonomy.py --scores ... --json ...",
    "benchmarks/rollup_frontier_detectability_evidence_reruns.py --queue ... --json ...",
    "benchmarks/compare_frontier_release_evidence.py --frontier-rerun-rollup-report ...",
)

_PRE_GENERATION_PROBE_COMMANDS = (
    "benchmarks/run_pre_generation_probe_workflow.py "
    "--output-dir ... --json ... --artifact-manifest ...",
    "benchmarks/eval_pre_generation_text_baselines.py "
    "--records ... --json ... --artifact-manifest ...",
    "benchmarks/compare_pre_generation_probe_workflows.py "
    "--workflow-report MODEL=... --redline-report MODEL=... "
    "--json ... --artifact-manifest ...",
    "benchmarks/export_product_promotion_contract.py "
    "--source ... --output ... --artifact-manifest ...",
    "benchmarks/run_product_runtime_baseline.py "
    "--trace ... --promotion-contract ... --json ... --artifact-manifest ...",
    "benchmarks/compare_product_runtime_baselines.py "
    "--current ... --baseline ... "
    "--min-pre-generation-probe-comparison-coverage ... "
    "--min-pre-generation-probe-comparison-manifest-verified-rate ... "
    "--min-pre-generation-probe-comparison-redline-pass-rate ... "
    "--json ... --artifact-manifest ...",
)

_CLAIM_RISK_LOCALIZATION_RUNTIME_EVIDENCE_COMMANDS = (
    "benchmarks/run_product_trace_replay_workflow.py "
    "--trace-glob ... --promotion-contract ... "
    "--min-runtime-drift-claim-risk-localization-coverage-rate ... "
    "--max-runtime-drift-claim-risk-localization-high-risk-claim-count-increase ... "
    "--max-runtime-drift-claim-risk-localization-medium-or-high-risk-claim-count-increase ... "
    "--max-runtime-drift-claim-risk-localization-entity-candidate-observation-count-increase ... "
    "--max-runtime-drift-claim-risk-localization-unique-entity-candidate-count-increase ... "
    "--max-runtime-drift-claim-risk-localization-high-risk-entity-candidate-count-increase ... "
    "--max-runtime-drift-claim-risk-localization-medium-or-high-entity-candidate-count-increase ...",
    "benchmarks/run_product_runtime_baseline.py "
    "--trace ... --promotion-contract ... --json ... --artifact-manifest ...",
    "benchmarks/compare_product_runtime_baselines.py "
    "--current ... --baseline ... "
    "--min-claim-risk-localization-coverage-rate ... "
    "--max-claim-risk-localization-high-risk-claim-count-increase ... "
    "--max-claim-risk-localization-medium-or-high-risk-claim-count-increase ... "
    "--max-claim-risk-localization-entity-candidate-observation-count-increase ... "
    "--max-claim-risk-localization-unique-entity-candidate-count-increase ... "
    "--max-claim-risk-localization-high-risk-entity-candidate-count-increase ... "
    "--max-claim-risk-localization-medium-or-high-entity-candidate-count-increase ... "
    "--json ... --artifact-manifest ...",
)

_WORLD_MODEL_RUNTIME_EVIDENCE_COMMANDS = (
    "benchmarks/run_world_model_signal_calibration_workflow.py "
    "--output-dir ... --registry ... --registry-name ... --registry-version ...",
    "benchmarks/enrich_product_trace_runtime_evidence.py "
    "--trace-glob ... --output-dir ... --report ... --artifact-manifest ... "
    "--min-world-model-participating-trace-rate ... "
    "--min-world-model-coverage-rate ... --max-world-model-trace-gap-rate ...",
    "benchmarks/run_product_trace_replay_workflow.py "
    "--trace-glob ... --promotion-contract ... "
    "--min-runtime-drift-world-model-participating-trace-rate ... "
    "--min-runtime-drift-world-model-coverage-rate ... "
    "--max-runtime-drift-world-model-trace-gap-rate-increase ...",
    "benchmarks/run_product_runtime_baseline.py "
    "--trace ... --promotion-contract ... --json ... --artifact-manifest ...",
    "benchmarks/compare_product_runtime_baselines.py "
    "--current ... --baseline ... "
    "--min-world-model-participating-trace-rate ... "
    "--min-world-model-coverage-rate ... "
    "--max-world-model-trace-gap-rate-increase ... "
    "--json ... --artifact-manifest ...",
)

_CONTEXT_SENSITIVITY_RUNTIME_EVIDENCE_COMMANDS = (
    "benchmarks/run_context_sensitivity_workflow.py "
    "--scores ... --verified-records-jsonl ... --model-id ... "
    "--output-dir ... --registry-path ... --registry-name ... --registry-version ...",
    "benchmarks/enrich_product_trace_runtime_evidence.py "
    "--trace-glob ... --output-dir ... --report ... --artifact-manifest ... "
    "--min-context-sensitivity-participating-trace-rate ... "
    "--min-context-sensitivity-coverage-rate ... "
    "--max-context-sensitivity-trace-gap-rate ...",
    "benchmarks/run_product_trace_replay_workflow.py "
    "--trace-glob ... --promotion-contract ... "
    "--min-runtime-drift-context-sensitivity-participating-trace-rate ... "
    "--min-runtime-drift-context-sensitivity-coverage-rate ... "
    "--max-runtime-drift-context-sensitivity-trace-gap-rate-increase ... "
    "--max-runtime-drift-context-sensitivity-max-ratio-increase ...",
    "benchmarks/run_product_runtime_baseline.py "
    "--trace ... --promotion-contract ... --json ... --artifact-manifest ...",
    "benchmarks/compare_product_runtime_baselines.py "
    "--current ... --baseline ... "
    "--min-context-sensitivity-participating-trace-rate ... "
    "--min-context-sensitivity-coverage-rate ... "
    "--max-context-sensitivity-trace-gap-rate-increase ... "
    "--max-context-sensitivity-max-ratio-increase ... "
    "--json ... --artifact-manifest ...",
)

_CLAIM_FACTUALITY_RUNTIME_EVIDENCE_COMMANDS = (
    "benchmarks/run_claim_factuality_probe_workflow.py "
    "--records ... --output-dir ... --json ... --artifact-manifest ... "
    "--registry ... --register-name ... --register-version ... "
    "--claim-factuality-layers ... --sweep-layers ... --best-by ... "
    "--conformal-alpha ... --baseline-signals ...",
    "benchmarks/compare_claim_factuality_probe_workflows.py "
    "--workflow-report MODEL=... --json ... --artifact-manifest ... "
    "--registry ... --register-name ... --register-version ... "
    "--min-model-count ... --min-record-count ... "
    "--min-test-label-auroc ... --min-redline-auroc-margin ...",
    "benchmarks/export_product_promotion_contract.py "
    "--source ... --output ... --artifact-manifest ... "
    "--registry ... --name ... --version ...",
    "benchmarks/run_product_trace_replay_workflow.py "
    "--trace-glob ... --promotion-contract ... "
    "--min-runtime-drift-claim-factuality-probe-comparison-coverage ... "
    "--min-runtime-drift-claim-factuality-probe-comparison-manifest-verified-rate ... "
    "--min-runtime-drift-claim-factuality-probe-comparison-model-count ... "
    "--min-runtime-drift-claim-factuality-probe-comparison-run-count ... "
    "--min-runtime-drift-claim-factuality-probe-comparison-redline-pass-rate ... "
    "--max-runtime-drift-claim-factuality-probe-comparison-best-test-label-auroc-drop ... "
    "--max-runtime-drift-claim-factuality-probe-comparison-best-test-selective-accuracy-drop ... "
    "--max-runtime-drift-claim-factuality-probe-comparison-best-test-selective-coverage-drop ... "
    "--max-runtime-drift-claim-factuality-probe-comparison-best-redline-auroc-drop ... "
    "--max-runtime-drift-claim-factuality-probe-comparison-best-redline-margin-drop ...",
    "benchmarks/run_product_runtime_baseline.py "
    "--trace ... --promotion-contract ... --json ... --artifact-manifest ...",
    "benchmarks/compare_product_runtime_baselines.py "
    "--current ... --baseline ... "
    "--min-claim-factuality-probe-comparison-coverage ... "
    "--min-claim-factuality-probe-comparison-manifest-verified-rate ... "
    "--min-claim-factuality-probe-comparison-model-count ... "
    "--min-claim-factuality-probe-comparison-run-count ... "
    "--min-claim-factuality-probe-comparison-redline-pass-rate ... "
    "--max-claim-factuality-probe-comparison-best-test-label-auroc-drop ... "
    "--max-claim-factuality-probe-comparison-best-test-selective-accuracy-drop ... "
    "--max-claim-factuality-probe-comparison-best-test-selective-coverage-drop ... "
    "--max-claim-factuality-probe-comparison-best-redline-auroc-drop ... "
    "--max-claim-factuality-probe-comparison-best-redline-margin-drop ... "
    "--json ... --artifact-manifest ...",
)

_COUNTERFACTUAL_ROBUSTNESS_RUNTIME_EVIDENCE_COMMANDS = (
    "benchmarks/eval_counterfactual_verification.py "
    "--verified-records ... --verifier ... --fact-corpus ... "
    "--json ... --artifact-manifest ... --registry ... "
    "--register-name ... --register-version ...",
    "benchmarks/enrich_product_trace_runtime_evidence.py "
    "--trace-glob ... --output-dir ... --report ... --artifact-manifest ... "
    "--min-counterfactual-robustness-participating-trace-rate ... "
    "--min-counterfactual-robustness-coverage-rate ... "
    "--min-counterfactual-robustness-pass-rate ... "
    "--min-counterfactual-robustness-flip-success-rate ... "
    "--max-counterfactual-robustness-false-invariance-rate ... "
    "--max-counterfactual-robustness-trace-gap-rate ...",
    "benchmarks/run_product_trace_replay_workflow.py "
    "--trace-glob ... --promotion-contract ... "
    "--min-runtime-drift-counterfactual-robustness-participating-trace-rate ... "
    "--min-runtime-drift-counterfactual-robustness-coverage-rate ... "
    "--min-runtime-drift-counterfactual-robustness-pass-rate ... "
    "--min-runtime-drift-counterfactual-robustness-flip-success-rate ... "
    "--max-runtime-drift-counterfactual-robustness-false-invariance-rate-increase ... "
    "--max-runtime-drift-counterfactual-robustness-trace-gap-rate-increase ...",
    "benchmarks/run_product_runtime_baseline.py "
    "--trace ... --promotion-contract ... --json ... --artifact-manifest ...",
    "benchmarks/compare_product_runtime_baselines.py "
    "--current ... --baseline ... "
    "--min-counterfactual-robustness-participating-trace-rate ... "
    "--min-counterfactual-robustness-coverage-rate ... "
    "--min-counterfactual-robustness-pass-rate ... "
    "--min-counterfactual-robustness-flip-success-rate ... "
    "--max-counterfactual-robustness-false-invariance-rate-increase ... "
    "--max-counterfactual-robustness-trace-gap-rate-increase ... "
    "--json ... --artifact-manifest ...",
)

_TRAJECTORY_AUDIT_RUNTIME_EVIDENCE_COMMANDS = (
    "benchmarks/run_product_trace_replay_workflow.py "
    "--trace-glob ... --promotion-contract ... "
    "--max-runtime-drift-product-trace-trajectory-audit-failed-trace-rate-increase ... "
    "--max-runtime-drift-product-trace-trajectory-audit-error-rate-increase ... "
    "--max-runtime-drift-product-trace-trajectory-audit-factual-rate-increase ... "
    "--max-runtime-drift-product-trace-trajectory-audit-referential-rate-increase ... "
    "--max-runtime-drift-product-trace-trajectory-audit-logical-rate-increase ... "
    "--max-runtime-drift-product-trace-trajectory-audit-procedural-rate-increase ... "
    "--max-runtime-drift-product-trace-trajectory-audit-scope-rate-increase ...",
    "benchmarks/run_product_runtime_baseline.py "
    "--trace ... --promotion-contract ... --json ... --artifact-manifest ...",
    "benchmarks/compare_product_runtime_baselines.py "
    "--current ... --baseline ... "
    "--max-product-trace-trajectory-audit-failed-trace-rate-increase ... "
    "--max-product-trace-trajectory-audit-error-rate-increase ... "
    "--max-product-trace-trajectory-audit-factual-rate-increase ... "
    "--max-product-trace-trajectory-audit-referential-rate-increase ... "
    "--max-product-trace-trajectory-audit-logical-rate-increase ... "
    "--max-product-trace-trajectory-audit-procedural-rate-increase ... "
    "--max-product-trace-trajectory-audit-scope-rate-increase ... "
    "--json ... --artifact-manifest ...",
)

_PROVENANCE_RUNTIME_EVIDENCE_COMMANDS = (
    "benchmarks/run_product_trace_replay_workflow.py "
    "--trace-glob ... --promotion-contract ... "
    "--min-runtime-drift-product-trace-provenance-coverage-rate ... "
    "--min-runtime-drift-product-trace-provenance-supported-claim-evidence-coverage ... "
    "--max-runtime-drift-product-trace-provenance-missing-reference-rate-increase ... "
    "--max-runtime-drift-product-trace-provenance-unsupported-supported-claim-rate-increase ... "
    "--max-runtime-drift-product-trace-provenance-error-rate-increase ... "
    "--min-runtime-drift-product-trace-provenance-final-answer-evidence-reference-rate ...",
    "benchmarks/run_product_runtime_baseline.py "
    "--trace ... --promotion-contract ... --json ... --artifact-manifest ...",
    "benchmarks/compare_product_runtime_baselines.py "
    "--current ... --baseline ... "
    "--min-product-trace-provenance-coverage-rate ... "
    "--min-product-trace-provenance-supported-claim-evidence-coverage ... "
    "--max-product-trace-provenance-missing-reference-rate-increase ... "
    "--max-product-trace-provenance-unsupported-supported-claim-rate-increase ... "
    "--max-product-trace-provenance-error-rate-increase ... "
    "--min-product-trace-provenance-final-answer-evidence-reference-rate ... "
    "--json ... --artifact-manifest ...",
)

_CITATION_INTEGRITY_RUNTIME_EVIDENCE_COMMANDS = (
    "benchmarks/run_product_trace_replay_workflow.py "
    "--trace-glob ... --promotion-contract ... "
    "--min-runtime-drift-product-trace-citation-integrity-participating-trace-rate ... "
    "--min-runtime-drift-product-trace-citation-integrity-coverage-rate ... "
    "--max-runtime-drift-product-trace-citation-integrity-mismatch-rate-increase ... "
    "--max-runtime-drift-product-trace-citation-integrity-unresolved-rate-increase ... "
    "--max-runtime-drift-product-trace-citation-integrity-issue-rate-increase ... "
    "--max-runtime-drift-product-trace-citation-integrity-trace-gap-rate-increase ...",
    "benchmarks/run_product_runtime_baseline.py "
    "--trace ... --promotion-contract ... --json ... --artifact-manifest ...",
    "benchmarks/compare_product_runtime_baselines.py "
    "--current ... --baseline ... "
    "--min-product-trace-citation-integrity-participating-trace-rate ... "
    "--min-product-trace-citation-integrity-coverage-rate ... "
    "--max-product-trace-citation-integrity-mismatch-rate-increase ... "
    "--max-product-trace-citation-integrity-unresolved-rate-increase ... "
    "--max-product-trace-citation-integrity-issue-rate-increase ... "
    "--max-product-trace-citation-integrity-trace-gap-rate-increase ... "
    "--json ... --artifact-manifest ...",
)

_ACTION_GATE_RUNTIME_EVIDENCE_COMMANDS = (
    "benchmarks/run_product_trace_replay_workflow.py "
    "--trace-glob ... --promotion-contract ... "
    "--max-action-audit-error-rate ... "
    "--max-action-audit-missing-retrieval-rate ... "
    "--max-action-audit-missing-plan-retrieval-query-rate ... "
    "--max-action-audit-malformed-payload-rate ... "
    "--max-action-audit-unexpected-action-rate ... "
    "--max-action-audit-unknown-claim-id-rate ... "
    "--max-action-execution-missing-result-rate ... "
    "--max-action-execution-unexpected-result-rate ... "
    "--max-action-execution-request-id-mismatch-rate ... "
    "--max-runtime-drift-product-trace-action-audit-error-rate-increase ... "
    "--max-runtime-drift-product-trace-action-audit-missing-retrieval-action-rate-increase ... "
    "--max-runtime-drift-product-trace-action-audit-missing-plan-retrieval-query-rate-increase ... "
    "--max-runtime-drift-product-trace-action-audit-malformed-payload-rate-increase ... "
    "--max-runtime-drift-product-trace-action-audit-unexpected-action-rate-increase ... "
    "--max-runtime-drift-product-trace-action-audit-unknown-claim-id-rate-increase ... "
    "--max-runtime-drift-product-trace-action-execution-alignment-failed-trace-rate-increase ... "
    "--max-runtime-drift-product-trace-action-execution-missing-result-rate-increase ... "
    "--max-runtime-drift-product-trace-action-execution-unexpected-result-rate-increase ... "
    "--max-runtime-drift-product-trace-action-execution-request-id-mismatch-rate-increase ...",
    "benchmarks/run_product_runtime_baseline.py "
    "--trace ... --promotion-contract ... --json ... --artifact-manifest ...",
    "benchmarks/compare_product_runtime_baselines.py "
    "--current ... --baseline ... "
    "--max-product-trace-action-audit-error-rate-increase ... "
    "--max-product-trace-action-audit-missing-retrieval-action-rate-increase ... "
    "--max-product-trace-action-audit-missing-plan-retrieval-query-rate-increase ... "
    "--max-product-trace-action-audit-malformed-payload-rate-increase ... "
    "--max-product-trace-action-audit-unexpected-action-rate-increase ... "
    "--max-product-trace-action-audit-unknown-claim-id-rate-increase ... "
    "--max-product-trace-action-execution-alignment-failed-trace-rate-increase ... "
    "--max-product-trace-action-execution-missing-result-rate-increase ... "
    "--max-product-trace-action-execution-unexpected-result-rate-increase ... "
    "--max-product-trace-action-execution-request-id-mismatch-rate-increase ... "
    "--json ... --artifact-manifest ...",
)

_EVIDENCE_HANDOFF_RUNTIME_EVIDENCE_COMMANDS = (
    "benchmarks/export_product_promotion_contract_evidence_handoff.py "
    "--contract ... --json ... --audit-json ... "
    "--pre-generation-probe-comparison ... "
    "--triple-extraction-fixture-matrix ... "
    "--counterfactual-verification ... "
    "--product-trace-replay-workflow ... "
    "--frontier-release-evidence ... "
    "--triple-audit-enrichment ... --runtime-baseline ... "
    "--covered-fact-property-metrics ... --artifact-manifest ... "
    "--registry ... --name ... --version ...",
    "benchmarks/run_product_trace_replay_workflow.py "
    "--trace-glob ... --promotion-contract ... "
    "--min-runtime-drift-evidence-handoff-coverage ... "
    "--min-runtime-drift-evidence-handoff-manifest-verified-rate ... "
    "--min-runtime-drift-evidence-handoff-present-metric-rate ... "
    "--max-runtime-drift-evidence-handoff-missing-metric-rate ... "
    "--max-runtime-drift-evidence-handoff-missing-metric-count ... "
    "--max-runtime-drift-evidence-handoff-blocked-group-count ... "
    "--min-runtime-drift-evidence-handoff-promoted-group-rate ...",
    "benchmarks/run_product_runtime_baseline.py "
    "--trace ... --promotion-contract ... --json ... --artifact-manifest ...",
    "benchmarks/compare_product_runtime_baselines.py "
    "--current ... --baseline ... "
    "--min-evidence-handoff-coverage ... "
    "--min-evidence-handoff-manifest-verified-rate ... "
    "--min-evidence-handoff-present-metric-rate ... "
    "--max-evidence-handoff-missing-metric-rate ... "
    "--max-evidence-handoff-missing-metric-count ... "
    "--max-evidence-handoff-blocked-group-count ... "
    "--min-evidence-handoff-promoted-group-rate ... "
    "--json ... --artifact-manifest ...",
)

_FRONTIER_RELEASE_RUNTIME_EVIDENCE_COMMANDS = (
    "benchmarks/compare_frontier_release_evidence.py "
    "--verifier-stability-report ... --abstention-stability-report ... "
    "--detectability-taxonomy-report ... --frontier-workflow-report ... "
    "--citation-batch-rollup-report ... --frontier-rerun-rollup-report ... "
    "--json ... --artifact-manifest ... --registry ... --name ... --version ...",
    "benchmarks/export_product_promotion_contract_evidence_handoff.py "
    "--contract ... --json ... --audit-json ... "
    "--frontier-release-evidence ... --artifact-manifest ... "
    "--registry ... --name ... --version ...",
    "benchmarks/run_product_runtime_baseline.py "
    "--trace ... --promotion-contract ... --json ... --artifact-manifest ...",
    "benchmarks/compare_product_runtime_baselines.py "
    "--current ... --baseline ... "
    "--min-frontier-release-evidence-coverage ... "
    "--min-frontier-release-evidence-report-present-rate ... "
    "--min-frontier-release-evidence-manifest-present-rate ... "
    "--min-frontier-release-evidence-status-promote-rate ... "
    "--min-frontier-release-evidence-decision-promote-rate ... "
    "--min-frontier-release-evidence-verifier-track-promote-rate ... "
    "--min-frontier-release-evidence-abstention-track-promote-rate ... "
    "--min-frontier-release-evidence-citation-batch-track-promote-rate ... "
    "--min-frontier-release-evidence-frontier-rerun-rollup-track-promote-rate ... "
    "--min-frontier-release-evidence-run-count ... "
    "--min-frontier-release-evidence-frontier-rerun-rollup-report-count ... "
    "--min-frontier-release-evidence-frontier-rerun-rollup-candidate-count ... "
    "--max-frontier-release-evidence-frontier-rerun-rollup-missing-report-count ... "
    "--max-frontier-release-evidence-frontier-rerun-rollup-invalid-report-count ... "
    "--max-frontier-release-evidence-frontier-rerun-rollup-blocked-candidate-count ... "
    "--min-frontier-release-evidence-frontier-rerun-rollup-promotion-ready-count ... "
    "--min-frontier-release-evidence-citation-batch-rollup-count ... "
    "--max-frontier-release-evidence-citation-batch-missing-expected-batch-count ... "
    "--max-frontier-release-evidence-citation-batch-duplicate-batch-count ... "
    "--max-frontier-release-evidence-citation-batch-unexpected-batch-count ... "
    "--json ... --artifact-manifest ...",
)

_TRIPLE_AUDIT_RUNTIME_EVIDENCE_COMMANDS = (
    "benchmarks/run_triple_extraction_fixture_matrix.py "
    "--corpus NAME=... --output-dir ... --artifact-manifest ...",
    "benchmarks/enrich_product_trace_triple_audit.py "
    "--trace-glob ... --evidence-corpus ... --output-dir ... "
    "--registry ... --name ... --version ... "
    "--min-audit-claim-coverage ... --min-audit-pass-rate ... "
    "--min-slot-coverage-rate ...",
    "benchmarks/export_product_promotion_contract_evidence_handoff.py "
    "--contract ... --json ... --audit-json ... "
    "--triple-extraction-fixture-matrix ... "
    "--triple-audit-enrichment ... --artifact-manifest ... "
    "--registry ... --name ... --version ...",
    "benchmarks/run_product_trace_replay_workflow.py "
    "--trace-glob ... --promotion-contract ... "
    "--min-runtime-drift-triple-claim-coverage ... "
    "--min-runtime-drift-triple-audit-claim-coverage ... "
    "--min-runtime-drift-triple-audit-pass-rate ... "
    "--min-runtime-drift-triple-slot-coverage ...",
    "benchmarks/run_product_runtime_baseline.py "
    "--trace ... --promotion-contract ... --json ... --artifact-manifest ...",
    "benchmarks/compare_product_runtime_baselines.py "
    "--current ... --baseline ... "
    "--min-triple-claim-coverage ... "
    "--min-triple-audit-claim-coverage ... "
    "--min-triple-audit-pass-rate ... "
    "--min-triple-slot-coverage ... "
    "--json ... --artifact-manifest ...",
)

_COVERED_FACT_PROPERTY_RUNTIME_EVIDENCE_COMMANDS = (
    "benchmarks/run_wikidata_structured_qa_route_workflow.py "
    "--qa-corpus ... --route structured_fact "
    "--fact-claim-style paraphrase_robustness "
    "--output-dir ... --json ... --artifact-manifest ...",
    "benchmarks/compare_route_baselines.py "
    "--registry ... --baseline-key ... "
    "--min-covered-fact-properties ... "
    "--min-covered-fact-property-records ... "
    "--min-covered-fact-property-source-documents ... "
    "--min-covered-fact-property-decision-accuracy ... "
    "--max-covered-fact-property-false-supported-rate ... "
    "--min-covered-fact-property-false-refuted-rate ... --json ...",
    "benchmarks/compare_external_evidence_baselines.py "
    "--route-registry ... --route-baseline-key ... "
    "--require-covered-facts-route "
    "--min-covered-fact-records ... "
    "--min-covered-fact-source-documents ... "
    "--min-covered-fact-properties ... "
    "--min-covered-fact-property-records ... "
    "--min-covered-fact-property-source-documents ... "
    "--min-covered-fact-property-decision-accuracy ... "
    "--max-covered-fact-property-false-supported-rate ... "
    "--min-covered-fact-property-false-refuted-rate ... --json ...",
    "benchmarks/export_product_promotion_contract_evidence_handoff.py "
    "--contract ... --json ... --audit-json ... "
    "--covered-fact-property-metrics ... --artifact-manifest ... "
    "--registry ... --name ... --version ...",
    "benchmarks/run_product_trace_replay_workflow.py "
    "--trace-glob ... --promotion-contract ... "
    "--runtime-drift-covered-fact-property-scope recommended_route "
    "--min-runtime-drift-covered-fact-property-metric-count ... "
    "--min-runtime-drift-covered-fact-min-records ... "
    "--min-runtime-drift-covered-fact-min-source-documents ... "
    "--max-runtime-drift-covered-fact-min-decision-accuracy-drop ... "
    "--max-runtime-drift-covered-fact-max-false-supported-rate-increase ... "
    "--max-runtime-drift-covered-fact-min-false-refuted-rate-drop ...",
    "benchmarks/run_product_runtime_baseline.py "
    "--trace ... --promotion-contract ... --json ... --artifact-manifest ...",
    "benchmarks/compare_product_runtime_baselines.py "
    "--current ... --baseline ... "
    "--promotion-contract-covered-fact-property-scope recommended_route "
    "--min-promotion-contract-covered-fact-property-metric-count ... "
    "--min-promotion-contract-covered-fact-min-records ... "
    "--min-promotion-contract-covered-fact-min-source-documents ... "
    "--max-promotion-contract-covered-fact-min-decision-accuracy-drop ... "
    "--max-promotion-contract-covered-fact-max-false-supported-rate-increase ... "
    "--max-promotion-contract-covered-fact-min-false-refuted-rate-drop ... "
    "--json ... --artifact-manifest ...",
)


def _assert_multiple_testing_rerun_rollup_action(action):
    assert action["evidence_routes"] == (
        "truthfulqa_frontier_workflow",
        "multiple_testing_gate",
        "frontier_release_evidence",
    )
    assert action["suggested_commands"] == _MULTIPLE_TESTING_RERUN_COMMANDS
    assert action["metadata"]["planner_script"] == (
        "benchmarks/plan_frontier_multiple_testing_reruns.py"
    )
    assert action["metadata"]["child_workflow_script"] == (
        "benchmarks/run_truthfulqa_frontier_workflow.py"
    )
    assert action["metadata"]["rollup_script"] == (
        "benchmarks/rollup_frontier_multiple_testing_reruns.py"
    )
    assert action["metadata"]["release_gate_script"] == (
        "benchmarks/compare_frontier_release_evidence.py"
    )
    assert action["metadata"]["rerun_queue_workflow"] == (
        "frontier_multiple_testing_rerun_queue"
    )
    assert action["metadata"]["child_workflow"] == "truthfulqa_frontier_workflow"
    assert action["metadata"]["rollup_workflow"] == (
        "frontier_multiple_testing_rerun_rollup"
    )
    assert action["metadata"]["derived_artifact_key"] == (
        "frontier_multiple_testing_rerun_queue"
    )
    assert action["metadata"]["rollup_track"] == "multiple_testing"
    assert action["metadata"]["release_gate_track"] == "frontier_rerun_rollup"
    assert action["metadata"]["risk_control_method"] == "multiple_testing_conformal"
    assert action["metadata"]["closure_outputs"] == (
        "frontier_multiple_testing_rerun_queue",
        "frontier_multiple_testing_rerun_rollup",
        "frontier_release_evidence_comparison",
    )


def _assert_abstention_rerun_rollup_action(action):
    assert action["evidence_routes"] == (
        "abstention_stability",
        "participation_gate",
        "frontier_release_evidence",
    )
    assert action["suggested_commands"] == _ABSTENTION_RERUN_COMMANDS
    assert action["metadata"]["planner_script"] == (
        "benchmarks/plan_frontier_abstention_evidence_reruns.py"
    )
    assert action["metadata"]["child_benchmark_script"] == (
        "benchmarks/eval_abstention_stability.py"
    )
    assert action["metadata"]["rollup_script"] == (
        "benchmarks/rollup_frontier_abstention_evidence_reruns.py"
    )
    assert action["metadata"]["release_gate_script"] == (
        "benchmarks/compare_frontier_release_evidence.py"
    )
    assert action["metadata"]["rerun_queue_workflow"] == (
        "frontier_abstention_evidence_rerun_queue"
    )
    assert action["metadata"]["rollup_workflow"] == (
        "frontier_abstention_evidence_rerun_rollup"
    )
    assert action["metadata"]["derived_artifact_key"] == "abstention_rerun_queue"
    assert action["metadata"]["rollup_track"] == "abstention"
    assert action["metadata"]["release_gate_track"] == "frontier_rerun_rollup"
    assert action["metadata"]["closure_outputs"] == (
        "abstention_rerun_queue",
        "abstention_rerun_rollup",
        "frontier_release_evidence_comparison",
    )


def _assert_citation_batch_rerun_rollup_action(action):
    assert action["evidence_routes"] == (
        "unresolved_evidence_queue",
        "citation_search_evidence",
        "source_family_citation",
        "frontier_release_evidence",
    )
    assert action["suggested_commands"] == _CITATION_BATCH_RERUN_COMMANDS
    assert action["metadata"]["planner_script"] == (
        "benchmarks/plan_citation_batch_evidence_reruns.py"
    )
    assert action["metadata"]["external_workflow_script"] == (
        "benchmarks/run_external_citation_search_adapter_workflow.py"
    )
    assert action["metadata"]["source_family_workflow_script"] == (
        "benchmarks/run_source_family_citation_search_workflow.py"
    )
    assert action["metadata"]["rollup_script"] == (
        "benchmarks/rollup_citation_search_batch_evidence.py"
    )
    assert action["metadata"]["release_gate_script"] == (
        "benchmarks/compare_frontier_release_evidence.py"
    )
    assert action["metadata"]["rerun_queue_workflow"] == (
        "citation_batch_evidence_rerun_queue"
    )
    assert action["metadata"]["external_workflow"] == (
        "external_citation_search_adapter_workflow"
    )
    assert action["metadata"]["source_family_workflow"] == (
        "source_family_citation_search_workflow"
    )
    assert action["metadata"]["rollup_workflow"] == (
        "citation_search_batch_evidence_rollup"
    )
    assert action["metadata"]["derived_artifact_key"] == (
        "citation_batch_evidence_rerun_queue"
    )
    assert action["metadata"]["rollup_track"] == "citation_batch"
    assert action["metadata"]["release_gate_track"] == "citation_batch"
    assert action["metadata"]["risk_control_method"] == "citation_traceability"
    assert action["metadata"]["queue_entry_report_kinds"] == (
        "external_citation_search_adapter_workflow",
        "source_family_citation_search_workflow",
    )
    assert action["metadata"]["closure_outputs"] == (
        "citation_batch_evidence_rerun_queue",
        "citation_search_batch_evidence_rollup",
        "frontier_release_evidence_comparison",
    )


def _assert_detectability_rerun_rollup_action(action):
    assert action["evidence_routes"] == (
        "detectability_taxonomy",
        "blind_spot_audit",
        "frontier_release_evidence",
    )
    assert action["suggested_commands"] == _DETECTABILITY_RERUN_COMMANDS
    assert action["metadata"]["planner_script"] == (
        "benchmarks/plan_frontier_detectability_evidence_reruns.py"
    )
    assert action["metadata"]["blind_spot_analysis_script"] == (
        "benchmarks/analyze_detectability_blind_spots.py"
    )
    assert action["metadata"]["taxonomy_rerun_script"] == (
        "benchmarks/eval_detectability_taxonomy.py"
    )
    assert action["metadata"]["rollup_script"] == (
        "benchmarks/rollup_frontier_detectability_evidence_reruns.py"
    )
    assert action["metadata"]["release_gate_script"] == (
        "benchmarks/compare_frontier_release_evidence.py"
    )
    assert action["metadata"]["rerun_queue_workflow"] == (
        "frontier_detectability_evidence_rerun_queue"
    )
    assert action["metadata"]["blind_spot_workflow"] == "detectability_blind_spot_analysis"
    assert action["metadata"]["taxonomy_workflow"] == "detectability_taxonomy"
    assert action["metadata"]["rollup_workflow"] == (
        "frontier_detectability_evidence_rerun_rollup"
    )
    assert action["metadata"]["derived_artifact_key"] == (
        "frontier_detectability_evidence_rerun_queue"
    )
    assert action["metadata"]["rollup_track"] == "detectability"
    assert action["metadata"]["release_gate_track"] == "frontier_rerun_rollup"
    assert action["metadata"]["risk_control_method"] == "detectability_taxonomy"
    assert action["metadata"]["default_blind_spot_cell"] == "entrenched"
    assert action["metadata"]["queue_entry_report_kinds"] == (
        "detectability_blind_spot_analysis",
        "detectability_taxonomy",
    )
    assert action["metadata"]["closure_outputs"] == (
        "frontier_detectability_evidence_rerun_queue",
        "frontier_detectability_evidence_rerun_rollup",
        "frontier_release_evidence_comparison",
    )


def _assert_pre_generation_probe_comparison_action(action):
    assert action["evidence_routes"] == (
        "pre_generation_probe_workflow",
        "pre_generation_text_redline",
        "pre_generation_probe_comparison",
        "product_promotion_contract",
        "product_runtime_drift",
    )
    assert action["suggested_commands"] == _PRE_GENERATION_PROBE_COMMANDS
    assert action["metadata"]["workflow_script"] == (
        "benchmarks/run_pre_generation_probe_workflow.py"
    )
    assert action["metadata"]["redline_script"] == (
        "benchmarks/eval_pre_generation_text_baselines.py"
    )
    assert action["metadata"]["comparison_script"] == (
        "benchmarks/compare_pre_generation_probe_workflows.py"
    )
    assert action["metadata"]["promotion_contract_script"] == (
        "benchmarks/export_product_promotion_contract.py"
    )
    assert action["metadata"]["runtime_baseline_script"] == (
        "benchmarks/run_product_runtime_baseline.py"
    )
    assert action["metadata"]["runtime_drift_script"] == (
        "benchmarks/compare_product_runtime_baselines.py"
    )
    assert action["metadata"]["workflow"] == "pre_generation_probe_workflow"
    assert action["metadata"]["redline_workflow"] == "pre_generation_text_baseline_eval"
    assert action["metadata"]["comparison_workflow"] == (
        "pre_generation_probe_workflow_comparison"
    )
    assert action["metadata"]["handoff_artifact_kind"] == "product_promotion_contract"
    assert action["metadata"]["runtime_baseline_workflow"] == "product_runtime_baseline"
    assert action["metadata"]["runtime_drift_workflow"] == "product_runtime_drift_comparison"
    assert action["metadata"]["risk_control_method"] == "pre_generation_hidden_state_probe"
    assert action["metadata"]["redline_required"] is True
    assert action["metadata"]["required_inputs"] == (
        "pre_generation_hidden_state_records_or_truthfulqa_export",
        "pre_generation_probe_workflow_reports",
        "pre_generation_text_redline_reports",
        "release_candidate_or_product_contract_source",
        "product_trace_corpus",
    )
    assert action["metadata"]["closure_outputs"] == (
        "pre_generation_probe_workflow_comparison",
        "product_promotion_contract",
        "product_runtime_baseline",
        "product_runtime_drift_comparison",
    )


def _assert_claim_factuality_runtime_evidence_action(action):
    assert action["evidence_routes"] == (
        "claim_factuality_probe_workflow",
        "claim_factuality_probe_comparison",
        "product_promotion_contract",
        "product_trace_replay",
        "product_runtime_baseline",
        "product_runtime_drift",
        "claim_factuality_evidence",
    )
    assert action["suggested_commands"] == _CLAIM_FACTUALITY_RUNTIME_EVIDENCE_COMMANDS
    assert action["metadata"]["workflow_script"] == (
        "benchmarks/run_claim_factuality_probe_workflow.py"
    )
    assert action["metadata"]["comparison_script"] == (
        "benchmarks/compare_claim_factuality_probe_workflows.py"
    )
    assert action["metadata"]["promotion_contract_script"] == (
        "benchmarks/export_product_promotion_contract.py"
    )
    assert action["metadata"]["trace_replay_script"] == (
        "benchmarks/run_product_trace_replay_workflow.py"
    )
    assert action["metadata"]["runtime_baseline_script"] == (
        "benchmarks/run_product_runtime_baseline.py"
    )
    assert action["metadata"]["runtime_drift_script"] == (
        "benchmarks/compare_product_runtime_baselines.py"
    )
    assert action["metadata"]["workflow"] == "claim_factuality_probe_workflow"
    assert action["metadata"]["comparison_workflow"] == (
        "claim_factuality_probe_workflow_comparison"
    )
    assert action["metadata"]["handoff_artifact_kind"] == "product_promotion_contract"
    assert action["metadata"]["trace_replay_workflow"] == "product_trace_replay_workflow"
    assert action["metadata"]["runtime_baseline_workflow"] == "product_runtime_baseline"
    assert action["metadata"]["runtime_drift_workflow"] == "product_runtime_drift_comparison"
    assert action["metadata"]["risk_control_method"] == (
        "claim_hidden_state_factuality_probe"
    )
    assert action["metadata"]["required_probe_artifacts"] == (
        "ClaimFactualityProbeArtifact",
        "CalibrationArtifact",
        "claim_factuality_text_redline_report",
    )
    assert action["metadata"]["required_runtime_metrics"] == (
        "promotion_contract.claim_factuality_probe_comparison.coverage_rate",
        "promotion_contract.claim_factuality_probe_comparison.manifest_verified_rate",
        "promotion_contract.claim_factuality_probe_comparison.model_count.mean",
        "promotion_contract.claim_factuality_probe_comparison.run_count.mean",
        "promotion_contract.claim_factuality_probe_comparison.redline_pass_rate",
        "promotion_contract.claim_factuality_probe_comparison.best_test_label_auroc.mean",
        (
            "promotion_contract.claim_factuality_probe_comparison."
            "best_test_selective_accuracy.mean"
        ),
        (
            "promotion_contract.claim_factuality_probe_comparison."
            "best_test_selective_coverage.mean"
        ),
        "promotion_contract.claim_factuality_probe_comparison.best_redline_auroc.mean",
        "promotion_contract.claim_factuality_probe_comparison.best_redline_margin.mean",
    )
    assert action["metadata"]["default_gate_thresholds"] == {
        "min_claim_factuality_probe_comparison_coverage": 1.0,
        "min_claim_factuality_probe_comparison_manifest_verified_rate": 1.0,
        "min_claim_factuality_probe_comparison_model_count": 2.0,
        "min_claim_factuality_probe_comparison_run_count": 2.0,
        "min_claim_factuality_probe_comparison_redline_pass_rate": 1.0,
        "max_claim_factuality_probe_comparison_best_test_label_auroc_drop": 0.02,
        "max_claim_factuality_probe_comparison_best_test_selective_accuracy_drop": 0.02,
        "max_claim_factuality_probe_comparison_best_test_selective_coverage_drop": 0.02,
        "max_claim_factuality_probe_comparison_best_redline_auroc_drop": 0.02,
        "max_claim_factuality_probe_comparison_best_redline_margin_drop": 0.02,
    }
    assert action["metadata"]["required_inputs"] == (
        "claim_factuality_hidden_state_records_or_truthfulqa_export",
        "claim_factuality_probe_workflow_reports",
        "claim_factuality_text_redline_reports",
        "release_candidate_or_product_contract_source",
        "product_trace_corpus",
        "baseline_product_runtime_report",
    )
    assert action["metadata"]["closure_outputs"] == (
        "claim_factuality_probe_workflow",
        "claim_factuality_probe_workflow_comparison",
        "product_promotion_contract",
        "product_trace_replay_workflow",
        "product_runtime_baseline",
        "product_runtime_drift_comparison",
    )


def _assert_claim_risk_localization_runtime_evidence_action(action):
    assert action["evidence_routes"] == (
        "product_trace_replay",
        "claim_risk_localization",
        "product_runtime_baseline",
        "product_runtime_drift",
        "span_entity_risk_evidence",
    )
    assert action["suggested_commands"] == (
        _CLAIM_RISK_LOCALIZATION_RUNTIME_EVIDENCE_COMMANDS
    )
    assert action["metadata"]["claim_risk_localization_api"] == (
        "eigentruth.verify.localize_claim_risk_spans"
    )
    assert action["metadata"]["trace_summary_api"] == (
        "eigentruth.control.ProductTrace.claim_risk_localization_summary"
    )
    assert action["metadata"]["trace_replay_script"] == (
        "benchmarks/run_product_trace_replay_workflow.py"
    )
    assert action["metadata"]["runtime_baseline_script"] == (
        "benchmarks/run_product_runtime_baseline.py"
    )
    assert action["metadata"]["runtime_drift_script"] == (
        "benchmarks/compare_product_runtime_baselines.py"
    )
    assert action["metadata"]["trace_replay_workflow"] == "product_trace_replay_workflow"
    assert action["metadata"]["runtime_baseline_workflow"] == "product_runtime_baseline"
    assert action["metadata"]["runtime_drift_workflow"] == "product_runtime_drift_comparison"
    assert action["metadata"]["risk_control_method"] == (
        "span_entity_claim_risk_localization"
    )
    assert action["metadata"]["localization_granularity"] == (
        "span",
        "claim",
        "entity_candidate",
    )
    assert action["metadata"]["required_trace_metrics"] == (
        "claim_risk_localization.coverage_rate",
        "claim_risk_localization.high_risk_claim_count",
        "claim_risk_localization.medium_or_high_risk_claim_count",
        "claim_risk_localization.entity_candidate_observation_count",
        "claim_risk_localization.unique_entity_candidate_count",
        "claim_risk_localization.high_risk_entity_candidate_count",
        "claim_risk_localization.medium_or_high_entity_candidate_count",
    )
    assert action["metadata"]["default_gate_thresholds"] == {
        "min_claim_risk_localization_coverage_rate": 1.0,
        "max_claim_risk_localization_high_risk_claim_count_increase": 0.0,
        "max_claim_risk_localization_medium_or_high_risk_claim_count_increase": 0.0,
        "max_claim_risk_localization_entity_candidate_observation_count_increase": 0.0,
        "max_claim_risk_localization_unique_entity_candidate_count_increase": 0.0,
        "max_claim_risk_localization_high_risk_entity_candidate_count_increase": 0.0,
        "max_claim_risk_localization_medium_or_high_entity_candidate_count_increase": 0.0,
    }
    assert action["metadata"]["required_inputs"] == (
        "full_product_trace_corpus",
        "promotion_contract_or_release_candidate",
        "baseline_product_runtime_report",
    )
    assert action["metadata"]["closure_outputs"] == (
        "product_trace_replay_workflow",
        "product_runtime_baseline",
        "product_runtime_drift_comparison",
    )


def _assert_world_model_runtime_evidence_action(action):
    assert action["evidence_routes"] == (
        "world_model_signal_calibration",
        "product_trace_runtime_evidence",
        "product_trace_replay",
        "product_runtime_baseline",
        "product_runtime_drift",
        "world_model_evidence",
    )
    assert action["suggested_commands"] == _WORLD_MODEL_RUNTIME_EVIDENCE_COMMANDS
    assert action["metadata"]["signal_workflow_script"] == (
        "benchmarks/run_world_model_signal_calibration_workflow.py"
    )
    assert action["metadata"]["trace_enrichment_script"] == (
        "benchmarks/enrich_product_trace_runtime_evidence.py"
    )
    assert action["metadata"]["trace_replay_script"] == (
        "benchmarks/run_product_trace_replay_workflow.py"
    )
    assert action["metadata"]["runtime_baseline_script"] == (
        "benchmarks/run_product_runtime_baseline.py"
    )
    assert action["metadata"]["runtime_drift_script"] == (
        "benchmarks/compare_product_runtime_baselines.py"
    )
    assert action["metadata"]["signal_workflow"] == (
        "world_model_signal_calibration_workflow"
    )
    assert action["metadata"]["trace_enrichment_workflow"] == (
        "product_trace_runtime_evidence_enrichment"
    )
    assert action["metadata"]["trace_replay_workflow"] == "product_trace_replay_workflow"
    assert action["metadata"]["runtime_baseline_workflow"] == "product_runtime_baseline"
    assert action["metadata"]["runtime_drift_workflow"] == "product_runtime_drift_comparison"
    assert action["metadata"]["risk_control_method"] == "state_transition_world_model"
    assert action["metadata"]["required_trace_metrics"] == (
        "world_model.participating_trace_rate",
        "world_model.coverage_rate",
        "world_model.conflict_rate",
        "world_model.low_agreement_rate",
        "world_model.trace_gap_rate",
    )
    assert action["metadata"]["default_gate_thresholds"] == {
        "min_world_model_participating_trace_rate": 1.0,
        "min_world_model_coverage_rate": 1.0,
        "max_world_model_trace_gap_rate_increase": 0.0,
    }
    assert action["metadata"]["required_inputs"] == (
        "world_model_rules_or_state_transition_fixture",
        "product_trace_corpus",
        "promotion_contract_or_release_candidate",
        "baseline_product_runtime_report",
    )
    assert action["metadata"]["closure_outputs"] == (
        "world_model_signal_calibration_workflow",
        "product_trace_runtime_evidence_enrichment",
        "product_trace_replay_workflow",
        "product_runtime_baseline",
        "product_runtime_drift_comparison",
    )


def _assert_context_sensitivity_runtime_evidence_action(action):
    assert action["evidence_routes"] == (
        "context_sensitivity_workflow",
        "product_trace_runtime_evidence",
        "product_trace_replay",
        "product_runtime_baseline",
        "product_runtime_drift",
        "context_sensitivity_evidence",
    )
    assert action["suggested_commands"] == _CONTEXT_SENSITIVITY_RUNTIME_EVIDENCE_COMMANDS
    assert action["metadata"]["context_workflow_script"] == (
        "benchmarks/run_context_sensitivity_workflow.py"
    )
    assert action["metadata"]["trace_enrichment_script"] == (
        "benchmarks/enrich_product_trace_runtime_evidence.py"
    )
    assert action["metadata"]["trace_replay_script"] == (
        "benchmarks/run_product_trace_replay_workflow.py"
    )
    assert action["metadata"]["runtime_baseline_script"] == (
        "benchmarks/run_product_runtime_baseline.py"
    )
    assert action["metadata"]["runtime_drift_script"] == (
        "benchmarks/compare_product_runtime_baselines.py"
    )
    assert action["metadata"]["context_workflow"] == "context_sensitivity_workflow"
    assert action["metadata"]["paired_logprob_workflow"] == (
        "context_sensitivity_paired_logprob_extraction"
    )
    assert action["metadata"]["trace_enrichment_workflow"] == (
        "product_trace_runtime_evidence_enrichment"
    )
    assert action["metadata"]["trace_replay_workflow"] == "product_trace_replay_workflow"
    assert action["metadata"]["runtime_baseline_workflow"] == "product_runtime_baseline"
    assert action["metadata"]["runtime_drift_workflow"] == "product_runtime_drift_comparison"
    assert action["metadata"]["risk_control_method"] == (
        "evidence_conditioned_context_sensitivity"
    )
    assert action["metadata"]["required_trace_metrics"] == (
        "context_sensitivity.participating_trace_rate",
        "context_sensitivity.coverage_rate",
        "context_sensitivity.flagged_result_rate",
        "context_sensitivity.trace_gap_rate",
        "context_sensitivity.max_flagged_rate",
        "context_sensitivity.max_context_sensitivity_ratio",
    )
    assert action["metadata"]["default_gate_thresholds"] == {
        "min_context_sensitivity_participating_trace_rate": 1.0,
        "min_context_sensitivity_coverage_rate": 1.0,
        "max_context_sensitivity_trace_gap_rate_increase": 0.0,
    }
    assert action["metadata"]["required_inputs"] == (
        "score_dump",
        "verified_records_jsonl_with_evidence_context",
        "context_logprob_model",
        "product_trace_corpus",
        "promotion_contract_or_release_candidate",
        "baseline_product_runtime_report",
    )
    assert action["metadata"]["closure_outputs"] == (
        "context_sensitivity_workflow",
        "product_trace_runtime_evidence_enrichment",
        "product_trace_replay_workflow",
        "product_runtime_baseline",
        "product_runtime_drift_comparison",
    )


def _assert_counterfactual_robustness_runtime_evidence_action(action):
    assert action["evidence_routes"] == (
        "counterfactual_verification_eval",
        "product_trace_runtime_evidence",
        "product_trace_replay",
        "product_runtime_baseline",
        "product_runtime_drift",
        "counterfactual_robustness_evidence",
    )
    assert action["suggested_commands"] == (
        _COUNTERFACTUAL_ROBUSTNESS_RUNTIME_EVIDENCE_COMMANDS
    )
    assert action["metadata"]["counterfactual_eval_script"] == (
        "benchmarks/eval_counterfactual_verification.py"
    )
    assert action["metadata"]["trace_enrichment_script"] == (
        "benchmarks/enrich_product_trace_runtime_evidence.py"
    )
    assert action["metadata"]["trace_replay_script"] == (
        "benchmarks/run_product_trace_replay_workflow.py"
    )
    assert action["metadata"]["runtime_baseline_script"] == (
        "benchmarks/run_product_runtime_baseline.py"
    )
    assert action["metadata"]["runtime_drift_script"] == (
        "benchmarks/compare_product_runtime_baselines.py"
    )
    assert action["metadata"]["counterfactual_eval_workflow"] == (
        "counterfactual_verification_eval"
    )
    assert action["metadata"]["trace_enrichment_workflow"] == (
        "product_trace_runtime_evidence_enrichment"
    )
    assert action["metadata"]["trace_replay_workflow"] == "product_trace_replay_workflow"
    assert action["metadata"]["runtime_baseline_workflow"] == "product_runtime_baseline"
    assert action["metadata"]["runtime_drift_workflow"] == "product_runtime_drift_comparison"
    assert action["metadata"]["risk_control_method"] == "counterfactual_probe_robustness"
    assert action["metadata"]["required_trace_metrics"] == (
        "counterfactual_robustness.participating_trace_rate",
        "counterfactual_robustness.coverage_rate",
        "counterfactual_robustness.pass_rate",
        "counterfactual_robustness.flip_success_rate",
        "counterfactual_robustness.false_invariance_rate",
        "counterfactual_robustness.trace_gap_rate",
    )
    assert action["metadata"]["default_gate_thresholds"] == {
        "min_counterfactual_robustness_participating_trace_rate": 1.0,
        "min_counterfactual_robustness_coverage_rate": 1.0,
        "min_counterfactual_robustness_pass_rate": 1.0,
        "min_counterfactual_robustness_flip_success_rate": 1.0,
        "max_counterfactual_robustness_false_invariance_rate_increase": 0.0,
        "max_counterfactual_robustness_trace_gap_rate_increase": 0.0,
    }
    assert action["metadata"]["required_inputs"] == (
        "verified_records_jsonl_or_counterfactual_probe_records",
        "counterfactual_verifier_or_fact_corpus",
        "product_trace_corpus",
        "promotion_contract_or_release_candidate",
        "baseline_product_runtime_report",
    )
    assert action["metadata"]["closure_outputs"] == (
        "counterfactual_verification_eval",
        "product_trace_runtime_evidence_enrichment",
        "product_trace_replay_workflow",
        "product_runtime_baseline",
        "product_runtime_drift_comparison",
    )


def _assert_trajectory_audit_runtime_evidence_action(action):
    assert action["evidence_routes"] == (
        "product_trace_replay",
        "trajectory_audit",
        "product_runtime_baseline",
        "product_runtime_drift",
        "trajectory_audit_evidence",
    )
    assert action["suggested_commands"] == _TRAJECTORY_AUDIT_RUNTIME_EVIDENCE_COMMANDS
    assert action["metadata"]["trajectory_audit_api"] == (
        "eigentruth.control.audit_product_trace_trajectory"
    )
    assert action["metadata"]["trace_replay_script"] == (
        "benchmarks/run_product_trace_replay_workflow.py"
    )
    assert action["metadata"]["runtime_baseline_script"] == (
        "benchmarks/run_product_runtime_baseline.py"
    )
    assert action["metadata"]["runtime_drift_script"] == (
        "benchmarks/compare_product_runtime_baselines.py"
    )
    assert action["metadata"]["trace_replay_workflow"] == "product_trace_replay_workflow"
    assert action["metadata"]["runtime_baseline_workflow"] == "product_runtime_baseline"
    assert action["metadata"]["runtime_drift_workflow"] == "product_runtime_drift_comparison"
    assert action["metadata"]["risk_control_method"] == (
        "trajectory_level_hallucination_audit"
    )
    assert action["metadata"]["hallucination_taxonomy"] == (
        "factual",
        "referential",
        "logical",
        "procedural",
        "scope",
    )
    assert action["metadata"]["required_trace_metrics"] == (
        "trajectory_audit.failed_trace_rate",
        "trajectory_audit.error_rate",
        "trajectory_audit.factual_rate",
        "trajectory_audit.referential_rate",
        "trajectory_audit.logical_rate",
        "trajectory_audit.procedural_rate",
        "trajectory_audit.scope_rate",
    )
    assert action["metadata"]["default_gate_thresholds"] == {
        "max_product_trace_trajectory_audit_failed_trace_rate_increase": 0.0,
        "max_product_trace_trajectory_audit_error_rate_increase": 0.0,
        "max_product_trace_trajectory_audit_factual_rate_increase": 0.0,
        "max_product_trace_trajectory_audit_referential_rate_increase": 0.0,
        "max_product_trace_trajectory_audit_logical_rate_increase": 0.0,
        "max_product_trace_trajectory_audit_procedural_rate_increase": 0.0,
        "max_product_trace_trajectory_audit_scope_rate_increase": 0.0,
    }
    assert action["metadata"]["required_inputs"] == (
        "full_product_trace_corpus",
        "promotion_contract_or_release_candidate",
        "baseline_product_runtime_report",
    )
    assert action["metadata"]["closure_outputs"] == (
        "product_trace_replay_workflow",
        "product_runtime_baseline",
        "product_runtime_drift_comparison",
    )


def _assert_provenance_runtime_evidence_action(action):
    assert action["evidence_routes"] == (
        "product_trace_replay",
        "trace_provenance",
        "product_runtime_baseline",
        "product_runtime_drift",
        "provenance_evidence",
    )
    assert action["suggested_commands"] == _PROVENANCE_RUNTIME_EVIDENCE_COMMANDS
    assert action["metadata"]["trace_summary_api"] == (
        "eigentruth.control.ProductTrace.provenance_summary"
    )
    assert action["metadata"]["provenance_audit_api"] == (
        "eigentruth.control.audit_trace_provenance"
    )
    assert action["metadata"]["trace_replay_script"] == (
        "benchmarks/run_product_trace_replay_workflow.py"
    )
    assert action["metadata"]["runtime_baseline_script"] == (
        "benchmarks/run_product_runtime_baseline.py"
    )
    assert action["metadata"]["runtime_drift_script"] == (
        "benchmarks/compare_product_runtime_baselines.py"
    )
    assert action["metadata"]["trace_replay_workflow"] == "product_trace_replay_workflow"
    assert action["metadata"]["runtime_baseline_workflow"] == "product_runtime_baseline"
    assert action["metadata"]["runtime_drift_workflow"] == "product_runtime_drift_comparison"
    assert action["metadata"]["risk_control_method"] == "trace_evidence_provenance"
    assert action["metadata"]["required_trace_metrics"] == (
        "provenance.coverage_rate",
        "provenance.supported_claim_evidence_coverage",
        "provenance.missing_reference_rate",
        "provenance.unsupported_supported_claim_rate",
        "provenance.error_rate",
        "provenance.final_answer_evidence_reference_rate",
    )
    assert action["metadata"]["default_gate_thresholds"] == {
        "min_product_trace_provenance_coverage_rate": 1.0,
        "min_product_trace_provenance_supported_claim_evidence_coverage": 1.0,
        "max_product_trace_provenance_missing_reference_rate_increase": 0.0,
        "max_product_trace_provenance_unsupported_supported_claim_rate_increase": 0.0,
        "max_product_trace_provenance_error_rate_increase": 0.0,
        "min_product_trace_provenance_final_answer_evidence_reference_rate": 1.0,
    }
    assert action["metadata"]["required_inputs"] == (
        "full_product_trace_corpus",
        "promotion_contract_or_release_candidate",
        "baseline_product_runtime_report",
    )
    assert action["metadata"]["closure_outputs"] == (
        "product_trace_replay_workflow",
        "product_runtime_baseline",
        "product_runtime_drift_comparison",
    )


def _assert_citation_integrity_runtime_evidence_action(action):
    assert action["evidence_routes"] == (
        "product_trace_replay",
        "citation_integrity",
        "product_runtime_baseline",
        "product_runtime_drift",
        "citation_integrity_evidence",
    )
    assert (
        action["suggested_commands"] == _CITATION_INTEGRITY_RUNTIME_EVIDENCE_COMMANDS
    )
    assert action["metadata"]["trace_summary_api"] == (
        "eigentruth.control.ProductTrace.citation_integrity_summary"
    )
    assert action["metadata"]["trace_replay_script"] == (
        "benchmarks/run_product_trace_replay_workflow.py"
    )
    assert action["metadata"]["runtime_baseline_script"] == (
        "benchmarks/run_product_runtime_baseline.py"
    )
    assert action["metadata"]["runtime_drift_script"] == (
        "benchmarks/compare_product_runtime_baselines.py"
    )
    assert action["metadata"]["trace_replay_workflow"] == "product_trace_replay_workflow"
    assert action["metadata"]["runtime_baseline_workflow"] == "product_runtime_baseline"
    assert action["metadata"]["runtime_drift_workflow"] == "product_runtime_drift_comparison"
    assert action["metadata"]["risk_control_method"] == "citation_integrity_traceability"
    assert action["metadata"]["required_trace_metrics"] == (
        "citation_integrity.participating_trace_rate",
        "citation_integrity.coverage_rate",
        "citation_integrity.mismatch_rate",
        "citation_integrity.unresolved_rate",
        "citation_integrity.issue_rate",
        "citation_integrity.trace_gap_rate",
    )
    assert action["metadata"]["default_gate_thresholds"] == {
        "min_product_trace_citation_integrity_participating_trace_rate": 1.0,
        "min_product_trace_citation_integrity_coverage_rate": 1.0,
        "max_product_trace_citation_integrity_mismatch_rate_increase": 0.0,
        "max_product_trace_citation_integrity_unresolved_rate_increase": 0.0,
        "max_product_trace_citation_integrity_issue_rate_increase": 0.0,
        "max_product_trace_citation_integrity_trace_gap_rate_increase": 0.0,
    }
    assert action["metadata"]["required_inputs"] == (
        "product_trace_corpus_with_citation_metadata",
        "promotion_contract_or_release_candidate",
        "baseline_product_runtime_report",
    )
    assert action["metadata"]["closure_outputs"] == (
        "product_trace_replay_workflow",
        "product_runtime_baseline",
        "product_runtime_drift_comparison",
    )


def _assert_action_gate_runtime_evidence_action(action):
    assert action["evidence_routes"] == (
        "product_trace_replay",
        "action_audit",
        "action_execution",
        "product_runtime_baseline",
        "product_runtime_drift",
        "tool_use_evidence",
    )
    assert action["suggested_commands"] == _ACTION_GATE_RUNTIME_EVIDENCE_COMMANDS
    assert action["metadata"]["action_audit_api"] == (
        "eigentruth.control.audit_action_requests"
    )
    assert action["metadata"]["action_execution_summary_api"] == (
        "eigentruth.control.ProductTrace.action_execution_summary"
    )
    assert action["metadata"]["trace_replay_script"] == (
        "benchmarks/run_product_trace_replay_workflow.py"
    )
    assert action["metadata"]["runtime_baseline_script"] == (
        "benchmarks/run_product_runtime_baseline.py"
    )
    assert action["metadata"]["runtime_drift_script"] == (
        "benchmarks/compare_product_runtime_baselines.py"
    )
    assert action["metadata"]["trace_replay_workflow"] == "product_trace_replay_workflow"
    assert action["metadata"]["action_audit_gate_workflow"] == (
        "product_trace_action_audit_gate"
    )
    assert action["metadata"]["action_execution_gate_workflow"] == (
        "product_trace_action_execution_gate"
    )
    assert action["metadata"]["runtime_baseline_workflow"] == "product_runtime_baseline"
    assert action["metadata"]["runtime_drift_workflow"] == "product_runtime_drift_comparison"
    assert action["metadata"]["risk_control_method"] == "tool_use_action_audit"
    assert action["metadata"]["tool_use_failure_modes"] == (
        "fabricated_action",
        "missing_required_action",
        "malformed_payload",
        "unexpected_action",
        "unknown_claim_reference",
        "missing_action_result",
        "unexpected_action_result",
        "request_id_mismatch",
        "execution_alignment_failure",
    )
    assert action["metadata"]["required_trace_metrics"] == (
        "action_audit.error_rate",
        "action_audit.missing_retrieval_action_rate",
        "action_audit.missing_plan_retrieval_query_rate",
        "action_audit.malformed_payload_rate",
        "action_audit.unexpected_action_rate",
        "action_audit.unknown_claim_id_rate",
        "action_execution.alignment_failed_trace_rate",
        "action_execution.missing_result_rate",
        "action_execution.unexpected_result_rate",
        "action_execution.request_id_mismatch_rate",
    )
    assert action["metadata"]["default_gate_thresholds"] == {
        "max_action_audit_error_rate": 0.0,
        "max_action_audit_missing_retrieval_rate": 0.0,
        "max_action_audit_missing_plan_retrieval_query_rate": 0.0,
        "max_action_audit_malformed_payload_rate": 0.0,
        "max_action_audit_unexpected_action_rate": 0.0,
        "max_action_audit_unknown_claim_id_rate": 0.0,
        "max_action_execution_missing_result_rate": 0.0,
        "max_action_execution_unexpected_result_rate": 0.0,
        "max_action_execution_request_id_mismatch_rate": 0.0,
        "max_product_trace_action_audit_error_rate_increase": 0.0,
        "max_product_trace_action_audit_missing_retrieval_action_rate_increase": 0.0,
        "max_product_trace_action_audit_missing_plan_retrieval_query_rate_increase": 0.0,
        "max_product_trace_action_audit_malformed_payload_rate_increase": 0.0,
        "max_product_trace_action_audit_unexpected_action_rate_increase": 0.0,
        "max_product_trace_action_audit_unknown_claim_id_rate_increase": 0.0,
        "max_product_trace_action_execution_alignment_failed_trace_rate_increase": 0.0,
        "max_product_trace_action_execution_missing_result_rate_increase": 0.0,
        "max_product_trace_action_execution_unexpected_result_rate_increase": 0.0,
        "max_product_trace_action_execution_request_id_mismatch_rate_increase": 0.0,
    }
    assert action["metadata"]["required_inputs"] == (
        "full_product_trace_corpus",
        "promotion_contract_or_release_candidate",
        "baseline_product_runtime_report",
    )
    assert action["metadata"]["closure_outputs"] == (
        "product_trace_replay_workflow",
        "product_trace_action_audit_gate",
        "product_trace_action_execution_gate",
        "product_runtime_baseline",
        "product_runtime_drift_comparison",
    )


def _assert_evidence_handoff_runtime_evidence_action(action):
    assert action["evidence_routes"] == (
        "product_promotion_contract",
        "product_promotion_evidence_handoff",
        "evidence_handoff_audit",
        "product_trace_replay",
        "product_runtime_baseline",
        "product_runtime_drift",
        "evidence_handoff_evidence",
    )
    assert action["suggested_commands"] == _EVIDENCE_HANDOFF_RUNTIME_EVIDENCE_COMMANDS
    assert action["metadata"]["evidence_handoff_script"] == (
        "benchmarks/export_product_promotion_contract_evidence_handoff.py"
    )
    assert action["metadata"]["trace_replay_script"] == (
        "benchmarks/run_product_trace_replay_workflow.py"
    )
    assert action["metadata"]["runtime_baseline_script"] == (
        "benchmarks/run_product_runtime_baseline.py"
    )
    assert action["metadata"]["runtime_drift_script"] == (
        "benchmarks/compare_product_runtime_baselines.py"
    )
    assert action["metadata"]["evidence_audit_api"] == (
        "eigentruth.control.audit_product_promotion_contract_evidence"
    )
    assert action["metadata"]["evidence_handoff_workflow"] == (
        "product_promotion_evidence_handoff_export"
    )
    assert action["metadata"]["evidence_audit_workflow"] == (
        "product_promotion_evidence_handoff_audit"
    )
    assert action["metadata"]["trace_replay_workflow"] == "product_trace_replay_workflow"
    assert action["metadata"]["runtime_baseline_workflow"] == "product_runtime_baseline"
    assert action["metadata"]["runtime_drift_workflow"] == "product_runtime_drift_comparison"
    assert action["metadata"]["risk_control_method"] == (
        "promotion_contract_evidence_provenance"
    )
    assert action["metadata"]["default_required_groups"] == (
        "promotion",
        "pre_generation",
        "counterfactual",
        "triple_audit",
        "covered_fact_property",
        "action_gate",
        "action_receipts",
        "receipt_claim_support",
        "frontier_release_evidence",
    )
    assert action["metadata"]["optional_runtime_groups"] == (
        "claim_factuality",
        "claim_risk_localization",
        "trajectory_audit",
        "evidence_handoff",
        "world_model",
        "context_sensitivity",
        "counterfactual_robustness",
    )
    assert action["metadata"]["required_runtime_metrics"] == (
        "promotion_contract.evidence_handoff.coverage_rate",
        "promotion_contract.evidence_handoff.manifest_verified_rate",
        "promotion_contract.evidence_handoff.present_metric_rate.mean",
        "promotion_contract.evidence_handoff.missing_metric_rate.mean",
        "promotion_contract.evidence_handoff.missing_metric_count.mean",
        "promotion_contract.evidence_handoff.blocked_group_count.mean",
        "promotion_contract.evidence_handoff.promoted_group_rate.mean",
    )
    assert action["metadata"]["default_gate_thresholds"] == {
        "min_evidence_handoff_coverage": 1.0,
        "min_evidence_handoff_manifest_verified_rate": 1.0,
        "min_evidence_handoff_present_metric_rate": 1.0,
        "max_evidence_handoff_missing_metric_rate": 0.0,
        "max_evidence_handoff_missing_metric_count": 0.0,
        "max_evidence_handoff_blocked_group_count": 0.0,
        "min_evidence_handoff_promoted_group_rate": 1.0,
    }
    assert action["metadata"]["required_inputs"] == (
        "product_promotion_contract_source",
        "frontier_and_runtime_child_evidence_reports",
        "artifact_manifests_for_child_evidence",
        "product_trace_corpus",
        "baseline_product_runtime_report",
    )
    assert action["metadata"]["closure_outputs"] == (
        "product_promotion_evidence_handoff_export",
        "product_promotion_evidence_handoff_audit",
        "product_trace_replay_workflow",
        "product_runtime_baseline",
        "product_runtime_drift_comparison",
    )


def _assert_frontier_release_runtime_evidence_action(action):
    assert action["evidence_routes"] == (
        "frontier_release_evidence",
        "frontier_release_evidence_comparison",
        "product_promotion_contract",
        "product_promotion_evidence_handoff",
        "evidence_handoff_audit",
        "product_runtime_baseline",
        "product_runtime_drift",
        "frontier_release_evidence_promotion_metrics",
    )
    assert action["suggested_commands"] == _FRONTIER_RELEASE_RUNTIME_EVIDENCE_COMMANDS
    assert action["metadata"]["frontier_release_evidence_script"] == (
        "benchmarks/compare_frontier_release_evidence.py"
    )
    assert action["metadata"]["evidence_handoff_script"] == (
        "benchmarks/export_product_promotion_contract_evidence_handoff.py"
    )
    assert action["metadata"]["runtime_baseline_script"] == (
        "benchmarks/run_product_runtime_baseline.py"
    )
    assert action["metadata"]["runtime_drift_script"] == (
        "benchmarks/compare_product_runtime_baselines.py"
    )
    assert action["metadata"]["evidence_audit_api"] == (
        "eigentruth.control.audit_product_promotion_contract_evidence"
    )
    assert action["metadata"]["frontier_release_evidence_workflow"] == (
        "frontier_release_evidence_comparison"
    )
    assert action["metadata"]["evidence_handoff_workflow"] == (
        "product_promotion_evidence_handoff_export"
    )
    assert action["metadata"]["evidence_audit_workflow"] == (
        "product_promotion_evidence_handoff_audit"
    )
    assert action["metadata"]["runtime_baseline_workflow"] == "product_runtime_baseline"
    assert action["metadata"]["runtime_drift_workflow"] == "product_runtime_drift_comparison"
    assert action["metadata"]["risk_control_method"] == (
        "frontier_release_evidence_provenance"
    )
    assert action["metadata"]["required_release_tracks"] == (
        "verifier_stability",
        "abstention_stability",
    )
    assert action["metadata"]["optional_release_tracks"] == (
        "detectability_taxonomy",
        "multiple_testing_frontier_workflow",
        "citation_batch_rollup",
        "frontier_rerun_rollup",
    )
    assert action["metadata"]["required_runtime_metrics"] == (
        "promotion_contract.frontier_release_evidence.coverage_rate",
        "promotion_contract.frontier_release_evidence.report_present_rate",
        "promotion_contract.frontier_release_evidence.manifest_present_rate",
        "promotion_contract.frontier_release_evidence.status_promote_rate",
        "promotion_contract.frontier_release_evidence.decision_promote_rate",
        "promotion_contract.frontier_release_evidence.verifier_track_promote_rate",
        "promotion_contract.frontier_release_evidence.abstention_track_promote_rate",
        "promotion_contract.frontier_release_evidence.citation_batch_track_promote_rate",
        (
            "promotion_contract.frontier_release_evidence."
            "frontier_rerun_rollup_track_promote_rate"
        ),
        "promotion_contract.frontier_release_evidence.run_count.mean",
        (
            "promotion_contract.frontier_release_evidence."
            "frontier_rerun_rollup_report_count.mean"
        ),
        (
            "promotion_contract.frontier_release_evidence."
            "frontier_rerun_rollup_candidate_count.mean"
        ),
        (
            "promotion_contract.frontier_release_evidence."
            "frontier_rerun_rollup_missing_report_count.mean"
        ),
        (
            "promotion_contract.frontier_release_evidence."
            "frontier_rerun_rollup_invalid_report_count.mean"
        ),
        (
            "promotion_contract.frontier_release_evidence."
            "frontier_rerun_rollup_blocked_candidate_count.mean"
        ),
        (
            "promotion_contract.frontier_release_evidence."
            "frontier_rerun_rollup_promotion_ready_count.mean"
        ),
        (
            "promotion_contract.frontier_release_evidence."
            "citation_batch_rollup_count.mean"
        ),
        (
            "promotion_contract.frontier_release_evidence."
            "citation_batch_missing_expected_batch_count.mean"
        ),
        (
            "promotion_contract.frontier_release_evidence."
            "citation_batch_duplicate_batch_count.mean"
        ),
        (
            "promotion_contract.frontier_release_evidence."
            "citation_batch_unexpected_batch_count.mean"
        ),
    )
    assert action["metadata"]["observed_track_metrics"] == (
        "promotion_contract.frontier_release_evidence.multiple_testing_track_status_counts",
        "promotion_contract.frontier_release_evidence.multiple_testing_track_promote_rate",
        "promotion_contract.frontier_release_evidence.base_detectability_track_status_counts",
        "promotion_contract.frontier_release_evidence.base_multiple_testing_track_status_counts",
    )
    assert action["metadata"]["default_gate_thresholds"] == {
        "min_frontier_release_evidence_coverage": 1.0,
        "min_frontier_release_evidence_report_present_rate": 1.0,
        "min_frontier_release_evidence_manifest_present_rate": 1.0,
        "min_frontier_release_evidence_status_promote_rate": 1.0,
        "min_frontier_release_evidence_decision_promote_rate": 1.0,
        "min_frontier_release_evidence_verifier_track_promote_rate": 1.0,
        "min_frontier_release_evidence_abstention_track_promote_rate": 1.0,
        "min_frontier_release_evidence_citation_batch_track_promote_rate": 1.0,
        "min_frontier_release_evidence_frontier_rerun_rollup_track_promote_rate": 1.0,
        "min_frontier_release_evidence_run_count": 1.0,
        "min_frontier_release_evidence_frontier_rerun_rollup_report_count": 1.0,
        "min_frontier_release_evidence_frontier_rerun_rollup_candidate_count": 1.0,
        "max_frontier_release_evidence_frontier_rerun_rollup_missing_report_count": (
            0.0
        ),
        "max_frontier_release_evidence_frontier_rerun_rollup_invalid_report_count": (
            0.0
        ),
        "max_frontier_release_evidence_frontier_rerun_rollup_blocked_candidate_count": (
            0.0
        ),
        "min_frontier_release_evidence_frontier_rerun_rollup_promotion_ready_count": (
            1.0
        ),
        "min_frontier_release_evidence_citation_batch_rollup_count": 1.0,
        "max_frontier_release_evidence_citation_batch_missing_expected_batch_count": (
            0.0
        ),
        "max_frontier_release_evidence_citation_batch_duplicate_batch_count": 0.0,
        "max_frontier_release_evidence_citation_batch_unexpected_batch_count": 0.0,
    }
    assert action["metadata"]["required_inputs"] == (
        "verifier_stability_report",
        "abstention_stability_report",
        "frontier_release_child_reports",
        "frontier_release_child_manifests",
        "product_promotion_contract_source",
        "product_trace_corpus",
        "baseline_product_runtime_report",
    )
    assert action["metadata"]["closure_outputs"] == (
        "frontier_release_evidence_comparison",
        "product_promotion_evidence_handoff_export",
        "product_promotion_evidence_handoff_audit",
        "product_runtime_baseline",
        "product_runtime_drift_comparison",
    )


def _assert_triple_audit_runtime_evidence_action(action):
    assert action["evidence_routes"] == (
        "triple_extraction_fixture_matrix",
        "product_trace_triple_audit_enrichment",
        "product_promotion_contract",
        "product_trace_replay",
        "product_runtime_baseline",
        "product_runtime_drift",
        "triple_audit_evidence",
    )
    assert action["suggested_commands"] == _TRIPLE_AUDIT_RUNTIME_EVIDENCE_COMMANDS
    assert action["metadata"]["triple_extraction_matrix_script"] == (
        "benchmarks/run_triple_extraction_fixture_matrix.py"
    )
    assert action["metadata"]["trace_enrichment_script"] == (
        "benchmarks/enrich_product_trace_triple_audit.py"
    )
    assert action["metadata"]["evidence_handoff_script"] == (
        "benchmarks/export_product_promotion_contract_evidence_handoff.py"
    )
    assert action["metadata"]["trace_replay_script"] == (
        "benchmarks/run_product_trace_replay_workflow.py"
    )
    assert action["metadata"]["runtime_baseline_script"] == (
        "benchmarks/run_product_runtime_baseline.py"
    )
    assert action["metadata"]["runtime_drift_script"] == (
        "benchmarks/compare_product_runtime_baselines.py"
    )
    assert action["metadata"]["claim_triple_extraction_api"] == (
        "eigentruth.verify.extract_claim_triples"
    )
    assert action["metadata"]["claim_triple_audit_api"] == (
        "eigentruth.verify.audit_claim_triples"
    )
    assert action["metadata"]["trace_summary_api"] == (
        "eigentruth.control.ProductTrace.triple_coverage_summary"
    )
    assert action["metadata"]["triple_extraction_matrix_workflow"] == (
        "triple_extraction_fixture_matrix"
    )
    assert action["metadata"]["trace_enrichment_workflow"] == (
        "product_trace_triple_audit_enrichment"
    )
    assert action["metadata"]["evidence_handoff_workflow"] == (
        "product_promotion_evidence_handoff_export"
    )
    assert action["metadata"]["trace_replay_workflow"] == "product_trace_replay_workflow"
    assert action["metadata"]["runtime_baseline_workflow"] == "product_runtime_baseline"
    assert action["metadata"]["runtime_drift_workflow"] == "product_runtime_drift_comparison"
    assert action["metadata"]["risk_control_method"] == "fact_level_triple_audit"
    assert action["metadata"]["fact_granularity"] == (
        "claim_triple",
        "slot",
        "predicate",
    )
    assert action["metadata"]["required_trace_metrics"] == (
        "triple_coverage.claim_triple_coverage_rate",
        "triple_coverage.audit_claim_coverage_rate",
        "triple_coverage.audit_pass_rate",
        "triple_coverage.slot_coverage_rate",
    )
    assert action["metadata"]["default_gate_thresholds"] == {
        "min_triple_claim_coverage": 1.0,
        "min_triple_audit_claim_coverage": 1.0,
        "min_triple_audit_pass_rate": 1.0,
        "min_triple_slot_coverage": 1.0,
    }
    assert action["metadata"]["required_inputs"] == (
        "structured_fact_corpora",
        "full_product_trace_corpus",
        "local_evidence_corpus",
        "promotion_contract_or_release_candidate",
        "baseline_product_runtime_report",
    )
    assert action["metadata"]["closure_outputs"] == (
        "triple_extraction_fixture_matrix",
        "product_trace_triple_audit_enrichment",
        "product_promotion_evidence_handoff_export",
        "product_trace_replay_workflow",
        "product_runtime_baseline",
        "product_runtime_drift_comparison",
    )


def _assert_covered_fact_property_runtime_evidence_action(action):
    assert action["evidence_routes"] == (
        "wikidata_structured_qa_route_workflow",
        "route_baseline",
        "external_evidence_baseline",
        "product_promotion_contract",
        "product_trace_replay",
        "product_runtime_baseline",
        "product_runtime_drift",
        "covered_fact_property",
    )
    assert action["suggested_commands"] == (
        _COVERED_FACT_PROPERTY_RUNTIME_EVIDENCE_COMMANDS
    )
    assert action["metadata"]["structured_route_script"] == (
        "benchmarks/run_wikidata_structured_qa_route_workflow.py"
    )
    assert action["metadata"]["route_baseline_script"] == (
        "benchmarks/compare_route_baselines.py"
    )
    assert action["metadata"]["external_evidence_baseline_script"] == (
        "benchmarks/compare_external_evidence_baselines.py"
    )
    assert action["metadata"]["evidence_handoff_script"] == (
        "benchmarks/export_product_promotion_contract_evidence_handoff.py"
    )
    assert action["metadata"]["trace_replay_script"] == (
        "benchmarks/run_product_trace_replay_workflow.py"
    )
    assert action["metadata"]["runtime_baseline_script"] == (
        "benchmarks/run_product_runtime_baseline.py"
    )
    assert action["metadata"]["runtime_drift_script"] == (
        "benchmarks/compare_product_runtime_baselines.py"
    )
    assert action["metadata"]["structured_route_workflow"] == (
        "wikidata_structured_qa_route_workflow"
    )
    assert action["metadata"]["route_baseline_workflow"] == (
        "route_baseline_comparison"
    )
    assert action["metadata"]["external_evidence_baseline_workflow"] == (
        "external_evidence_baseline_comparison"
    )
    assert action["metadata"]["evidence_handoff_workflow"] == (
        "product_promotion_evidence_handoff_export"
    )
    assert action["metadata"]["trace_replay_workflow"] == "product_trace_replay_workflow"
    assert action["metadata"]["runtime_baseline_workflow"] == "product_runtime_baseline"
    assert action["metadata"]["runtime_drift_workflow"] == "product_runtime_drift_comparison"
    assert action["metadata"]["risk_control_method"] == (
        "structured_fact_property_gate"
    )
    assert action["metadata"]["fact_granularity"] == (
        "entity",
        "property",
        "source_document",
    )
    assert action["metadata"]["recommended_property_scope"] == "recommended_route"
    assert action["metadata"]["required_route_metrics"] == (
        "covered_fact_properties",
        "covered_fact_property.records",
        "covered_fact_property.source_documents",
        "covered_fact_property.decision_accuracy",
        "covered_fact_property.false_supported_rate",
        "covered_fact_property.false_refuted_rate",
    )
    assert action["metadata"]["required_runtime_metrics"] == (
        "promotion_contract.covered_fact_properties."
        "recommended_route_property_metrics.property_metric_count.mean",
        "promotion_contract.covered_fact_properties."
        "recommended_route_property_metrics.min_records.mean",
        "promotion_contract.covered_fact_properties."
        "recommended_route_property_metrics.min_source_documents.mean",
        "promotion_contract.covered_fact_properties."
        "recommended_route_property_metrics.min_decision_accuracy.mean",
        "promotion_contract.covered_fact_properties."
        "recommended_route_property_metrics.max_false_supported_rate.mean",
        "promotion_contract.covered_fact_properties."
        "recommended_route_property_metrics.min_false_refuted_rate.mean",
    )
    assert action["metadata"]["default_gate_thresholds"] == {
        "min_covered_fact_properties": 1,
        "min_covered_fact_property_records": 1,
        "min_covered_fact_property_source_documents": 1,
        "min_covered_fact_property_decision_accuracy": 1.0,
        "max_covered_fact_property_false_supported_rate": 0.0,
        "min_covered_fact_property_false_refuted_rate": 1.0,
        "min_promotion_contract_covered_fact_property_metric_count": 1.0,
        "min_promotion_contract_covered_fact_min_records": 1.0,
        "min_promotion_contract_covered_fact_min_source_documents": 1.0,
        "max_promotion_contract_covered_fact_min_decision_accuracy_drop": 0.0,
        "max_promotion_contract_covered_fact_max_false_supported_rate_increase": 0.0,
        "max_promotion_contract_covered_fact_min_false_refuted_rate_drop": 0.0,
    }
    assert action["metadata"]["required_inputs"] == (
        "wikidata_or_structured_fact_qa_corpus",
        "artifact_registry_with_route_baseline",
        "product_promotion_contract_source",
        "product_trace_corpus",
        "baseline_product_runtime_report",
    )
    assert action["metadata"]["closure_outputs"] == (
        "wikidata_structured_qa_route_workflow",
        "route_baseline_comparison",
        "external_evidence_baseline_comparison",
        "product_promotion_evidence_handoff_export",
        "product_trace_replay_workflow",
        "product_runtime_baseline",
        "product_runtime_drift_comparison",
    )


def test_evidence_gap_plan_maps_release_blockers_to_frontier_actions():
    plan = plan_evidence_gaps_from_release_candidate(
        _blocked_registry_workflow_payload(),
        source_path="artifacts/frontier-audit-release-candidate-v4/frontier-audit-comparison.json",
    )

    payload = plan.to_dict()
    actions = {action["action_id"]: action for action in payload["actions"]}
    gaps = {gap["gap_id"]: gap for gap in payload["gaps"]}

    assert payload["status"] == "needs_evidence"
    assert payload["source_workflow"] == "release_candidate_registry_workflow"
    assert payload["source_status"] == "blocked"
    assert payload["summary"]["gap_count"] == 4
    assert payload["summary"]["missing_metric_count"] == 14
    assert payload["summary"]["gates"] == {
        "performance_baseline": 1,
        "product_runtime_drift": 2,
        "readiness_baseline": 1,
    }
    assert payload["summary"]["top_action_ids"][0] == "refresh_readiness_baseline"
    assert "run_pre_generation_probe_comparison" in actions
    assert "rerun_product_trace_action_gates" in actions
    _assert_pre_generation_probe_comparison_action(
        actions["run_pre_generation_probe_comparison"]
    )
    _assert_action_gate_runtime_evidence_action(
        actions["rerun_product_trace_action_gates"]
    )
    pre_generation_gap = next(
        gap
        for gap in gaps.values()
        if gap["recommended_action_ids"] == ("run_pre_generation_probe_comparison",)
    )
    assert pre_generation_gap["root_cause"] == "model"
    assert pre_generation_gap["missing_metrics"] == (
        "promotion_contract.pre_generation_probe_comparison.coverage_rate",
        "promotion_contract.pre_generation_probe_comparison.manifest_verified_rate",
        "promotion_contract.pre_generation_probe_comparison.model_count.mean",
        "promotion_contract.pre_generation_probe_comparison.run_count.mean",
        "promotion_contract.pre_generation_probe_comparison.redline_pass_rate",
        "promotion_contract.pre_generation_probe_comparison.best_test_label_auroc.mean",
        "promotion_contract.pre_generation_probe_comparison.best_redline_auroc.mean",
        "promotion_contract.pre_generation_probe_comparison.best_redline_margin.mean",
    )
    strict_json_dumps(payload, sort_keys=True)
    assert EvidenceGapPlan.from_dict(payload).to_dict() == payload


def test_evidence_gap_plan_reports_ready_when_no_blockers():
    plan = plan_evidence_gaps_from_release_candidate({
        "workflow": "release_candidate_comparison",
        "decision": {"status": "promote", "blocking_reasons": []},
    })

    assert plan.status == "ready"
    assert plan.summary["gap_count"] == 0
    assert plan.actions == ()


def test_evidence_gap_plan_maps_multiple_testing_frontier_blocker():
    plan = plan_evidence_gaps_from_release_candidate({
        "workflow": "release_candidate_comparison",
        "decision": {
            "status": "blocked",
            "blocking_reasons": [
                {
                    "gate": "frontier_release_evidence",
                    "status": "blocked",
                    "reasons": (
                        "frontier release evidence multiple_testing_track_status is "
                        "'blocked', expected 'promote' or 'not_required'",
                    ),
                }
            ],
        },
    })

    payload = plan.to_dict()
    actions = {action["action_id"]: action for action in payload["actions"]}
    gaps = payload["gaps"]

    assert payload["status"] == "needs_evidence"
    assert payload["summary"]["gap_count"] == 1
    assert payload["summary"]["research_axes"] == {"multi_signal_calibration": 1}
    assert payload["summary"]["top_action_ids"] == ("rerun_frontier_multiple_testing_gate",)
    assert gaps[0]["root_cause"] == "model"
    assert gaps[0]["metadata"]["evidence_kind"] == "frontier_multiple_testing"
    assert gaps[0]["recommended_action_ids"] == ("rerun_frontier_multiple_testing_gate",)
    _assert_multiple_testing_rerun_rollup_action(
        actions["rerun_frontier_multiple_testing_gate"]
    )


def test_evidence_gap_plan_maps_release_candidate_frontier_abstention_blocker():
    plan = plan_evidence_gaps_from_release_candidate({
        "workflow": "release_candidate_comparison",
        "decision": {
            "status": "blocked",
            "blocking_reasons": [
                {
                    "gate": "frontier_release_evidence",
                    "status": "blocked",
                    "reasons": (
                        "frontier release evidence decision status is 'blocked', expected 'promote'",
                        "frontier release evidence abstention_track_status is 'blocked', expected 'promote'",
                    ),
                }
            ],
        },
    })

    payload = plan.to_dict()
    actions = {action["action_id"]: action for action in payload["actions"]}
    gaps = payload["gaps"]

    assert payload["status"] == "needs_evidence"
    assert payload["summary"]["gap_count"] == 1
    assert payload["summary"]["research_axes"] == {"participation_calibration": 1}
    assert payload["summary"]["top_action_ids"] == ("improve_abstention_participation_gate",)
    assert gaps[0]["reason"] == (
        "frontier release evidence abstention_track_status is 'blocked', expected 'promote'"
    )
    assert gaps[0]["root_cause"] == "model"
    assert gaps[0]["metadata"]["evidence_kind"] == "abstention_stability"
    assert gaps[0]["recommended_action_ids"] == ("improve_abstention_participation_gate",)
    _assert_abstention_rerun_rollup_action(
        actions["improve_abstention_participation_gate"]
    )


def test_evidence_gap_plan_maps_frontier_release_evidence_report_tracks():
    plan = plan_evidence_gaps_from_release_candidate({
        "schema_version": 1,
        "workflow": "frontier_release_evidence_comparison",
        "status": "complete",
        "decision": {
            "status": "blocked",
            "verifier_track_status": "promote",
            "abstention_track_status": "blocked",
            "detectability_track_status": "blocked",
            "multiple_testing_track_status": "blocked",
            "citation_batch_track_status": "blocked",
            "blocking_reasons": (
                "abstention_stability.synthetic.conditional_correctness_lower_bound_mean "
                "0.5 is below required minimum 0.8",
                "detectability_taxonomy.synthetic.entrenched_false_rate 0.4 exceeds maximum 0.25",
                "truthfulqa_frontier_workflow.synthetic.multiple_testing_gate.all_pass is not true",
                "citation_batch_rollup.citation-rollup.summary.missing_expected_batch_count "
                "1 is non-zero",
            ),
        },
        "evidence_summary": {
            "citation_batch_rollup_names": ("citation-rollup",),
            "citation_batch_missing_expected_batch_count": 1,
            "citation_batch_missing_expected_batches": (
                {
                    "rollup": "citation-rollup",
                    "batch_id": "unresolved-evidence-batch-0002",
                },
            ),
            "citation_batch_expected_batch_ids": (
                "unresolved-evidence-batch-0001",
                "unresolved-evidence-batch-0002",
            ),
            "citation_batch_observed_batch_ids": ("unresolved-evidence-batch-0001",),
        },
        "multiple_testing_decisions": (
            {
                "name": "synthetic",
                "metrics": {
                    "failed_cells": (
                        {
                            "cell": "synthetic-l2",
                            "status": "failed",
                            "false_alarm": 0.04,
                            "detection": 0.62,
                            "report": "synthetic-l2/multiple-testing-report.json",
                            "calibration": "synthetic-l2/multiple-testing-calibration.json",
                        },
                    ),
                    "unknown_cells": (),
                },
            },
        ),
    })

    payload = plan.to_dict()
    actions = {action["action_id"]: action for action in payload["actions"]}
    gaps = {gap["gate"]: gap for gap in payload["gaps"]}

    assert payload["status"] == "needs_evidence"
    assert payload["source_workflow"] == "frontier_release_evidence_comparison"
    assert payload["summary"]["gates"] == {
        "abstention_stability": 1,
        "citation_batch_evidence": 1,
        "detectability_taxonomy": 1,
        "frontier_multiple_testing": 1,
    }
    assert payload["summary"]["research_axes"] == {
        "blind_spot_taxonomy": 1,
        "external_citation": 1,
        "multi_signal_calibration": 1,
        "participation_calibration": 1,
    }
    assert gaps["abstention_stability"]["recommended_action_ids"] == (
        "improve_abstention_participation_gate",
    )
    assert gaps["detectability_taxonomy"]["recommended_action_ids"] == (
        "audit_detectability_blind_spots",
    )
    assert gaps["frontier_multiple_testing"]["recommended_action_ids"] == (
        "rerun_frontier_multiple_testing_gate",
    )
    assert gaps["citation_batch_evidence"]["recommended_action_ids"] == (
        "complete_citation_batch_evidence_rollup",
    )
    assert gaps["citation_batch_evidence"]["metadata"]["citation_batch_missing_expected_batches"] == (
        {
            "rollup": "citation-rollup",
            "batch_id": "unresolved-evidence-batch-0002",
        },
    )
    assert gaps["frontier_multiple_testing"]["metadata"]["multiple_testing_failed_cells"] == (
        {
            "run": "synthetic",
            "cell": "synthetic-l2",
            "status": "failed",
            "false_alarm": 0.04,
            "detection": 0.62,
            "report": "synthetic-l2/multiple-testing-report.json",
            "calibration": "synthetic-l2/multiple-testing-calibration.json",
        },
    )
    _assert_multiple_testing_rerun_rollup_action(
        actions["rerun_frontier_multiple_testing_gate"]
    )
    _assert_abstention_rerun_rollup_action(
        actions["improve_abstention_participation_gate"]
    )
    _assert_citation_batch_rerun_rollup_action(
        actions["complete_citation_batch_evidence_rollup"]
    )
    _assert_detectability_rerun_rollup_action(
        actions["audit_detectability_blind_spots"]
    )


def test_evidence_gap_plan_maps_frontier_rerun_rollup_blocker():
    plan = plan_evidence_gaps_from_release_candidate({
        "schema_version": 1,
        "workflow": "frontier_release_evidence_comparison",
        "status": "complete",
        "decision": {
            "status": "blocked",
            "verifier_track_status": "promote",
            "abstention_track_status": "promote",
            "frontier_rerun_rollup_track_status": "blocked",
            "blocking_reasons": (),
        },
        "evidence_summary": {
            "frontier_rerun_rollup_names": ("frontier-abstention-rerun-rollup",),
            "frontier_rerun_rollup_blocked_names": (
                "frontier-abstention-rerun-rollup",
            ),
            "frontier_rerun_rollup_workflows": (
                "frontier_abstention_evidence_rerun_rollup",
            ),
            "frontier_rerun_rollup_tracks": ("abstention",),
            "frontier_rerun_rollup_candidate_count": 2,
            "frontier_rerun_rollup_observed_report_count": 1,
            "frontier_rerun_rollup_missing_report_count": 1,
            "frontier_rerun_rollup_invalid_report_count": 0,
            "frontier_rerun_rollup_blocked_candidate_count": 1,
            "frontier_rerun_rollup_promotion_ready_count": 0,
        },
        "frontier_rerun_rollup_decisions": (
            {
                "name": "frontier-abstention-rerun-rollup",
                "status": "blocked",
                "metrics": {
                    "workflow": "frontier_abstention_evidence_rerun_rollup",
                    "track": "abstention",
                    "candidate_count": 2,
                    "observed_report_count": 1,
                    "missing_report_count": 1,
                    "invalid_report_count": 0,
                    "blocked_candidate_count": 1,
                    "promotion_ready_count": 0,
                },
                "blocking_reasons": (
                    "frontier_rerun_rollup.frontier-abstention-rerun-rollup."
                    "summary.missing_report_count 1 is non-zero",
                ),
            },
        ),
    })

    payload = plan.to_dict()
    actions = {action["action_id"]: action for action in payload["actions"]}
    gap = payload["gaps"][0]
    metadata = gap["metadata"]

    assert payload["status"] == "needs_evidence"
    assert payload["summary"]["gap_count"] == 1
    assert payload["summary"]["gates"] == {"frontier_rerun_rollup_evidence": 1}
    assert payload["summary"]["root_causes"] == {"evidence_coverage": 1}
    assert payload["summary"]["research_axes"] == {"frontier_rerun_validation": 1}
    assert payload["summary"]["top_action_ids"] == (
        "complete_frontier_rerun_rollup_evidence",
    )
    assert gap["gate"] == "frontier_rerun_rollup_evidence"
    assert gap["recommended_action_ids"] == (
        "complete_frontier_rerun_rollup_evidence",
    )
    assert metadata["evidence_kind"] == "frontier_rerun_rollup_evidence"
    assert metadata["frontier_rerun_rollup_blocked_names"] == (
        "frontier-abstention-rerun-rollup",
    )
    assert metadata["frontier_rerun_rollup_missing_report_count"] == 1
    assert metadata["frontier_rerun_rollup_blocked_rollups"] == (
        {
            "name": "frontier-abstention-rerun-rollup",
            "status": "blocked",
            "workflow": "frontier_abstention_evidence_rerun_rollup",
            "track": "abstention",
            "candidate_count": 2,
            "observed_report_count": 1,
            "missing_report_count": 1,
            "invalid_report_count": 0,
            "blocked_candidate_count": 1,
            "promotion_ready_count": 0,
            "blocking_reasons": (
                "frontier_rerun_rollup.frontier-abstention-rerun-rollup."
                "summary.missing_report_count 1 is non-zero",
            ),
        },
    )
    assert actions["complete_frontier_rerun_rollup_evidence"]["evidence_routes"] == (
        "frontier_rerun_rollup",
        "frontier_release_evidence",
        "verifier_stability",
        "abstention_stability",
        "detectability_taxonomy",
        "multiple_testing_gate",
    )


def test_plan_release_evidence_gaps_can_emit_frontier_rerun_rollup_completion_plan(
    tmp_path,
):
    source = tmp_path / "frontier-release-evidence.json"
    output = tmp_path / "evidence-gap-plan.json"
    completion_path = tmp_path / "frontier-rerun-rollup-completion.json"
    manifest_path = tmp_path / "frontier-rerun-rollup-completion-manifest.json"
    registry_path = tmp_path / "registry.json"
    queue_path = tmp_path / "abstention-rerun-queue.json"
    queue_path.write_text(
        json.dumps({
            "schema_version": 1,
            "workflow": "frontier_abstention_evidence_rerun_queue",
            "status": "ready",
            "entries": (),
        }),
        encoding="utf-8",
    )
    source.write_text(
        json.dumps({
            "schema_version": 1,
            "workflow": "frontier_release_evidence_comparison",
            "status": "complete",
            "decision": {
                "status": "blocked",
                "verifier_track_status": "promote",
                "abstention_track_status": "promote",
                "frontier_rerun_rollup_track_status": "blocked",
                "blocking_reasons": (),
            },
            "evidence_summary": {
                "frontier_rerun_rollup_missing_report_count": 1,
                "frontier_rerun_rollup_blocked_names": (
                    "frontier-abstention-rerun-rollup",
                ),
            },
            "frontier_rerun_rollup_decisions": (
                {
                    "name": "frontier-abstention-rerun-rollup",
                    "status": "blocked",
                    "metrics": {
                        "workflow": "frontier_abstention_evidence_rerun_rollup",
                        "track": "abstention",
                        "candidate_count": 2,
                        "missing_report_count": 1,
                        "blocked_candidate_count": 1,
                        "promotion_ready_count": 0,
                    },
                    "blocking_reasons": (
                        "frontier_rerun_rollup.frontier-abstention-rerun-rollup."
                        "summary.missing_report_count 1 is non-zero",
                    ),
                },
            ),
        }),
        encoding="utf-8",
    )

    payload = build_release_evidence_gap_plan(
        source=source,
        json_path=output,
        registry_path=registry_path,
        name="frontier-gap-plan",
        version="0.1",
        frontier_rerun_rollup_completion_json_path=completion_path,
        frontier_rerun_rollup_completion_artifact_manifest_path=manifest_path,
        frontier_rerun_rollup_completion_output_dir=tmp_path / "frontier-rerun-rollups",
        frontier_rerun_rollup_completion_name="frontier-rerun-rollup-completion",
        frontier_rerun_rollup_completion_version="0.1",
        frontier_rerun_rollup_queue_paths=(f"abstention={queue_path}",),
        python_executable="python",
    )

    saved = json.loads(output.read_text(encoding="utf-8"))
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    registry = ArtifactRegistry.load_json(registry_path)
    gap_record = registry.get("evidence_gap_plan:frontier-gap-plan:0.1")
    completion_record = registry.get("report:frontier-rerun-rollup-completion:0.1")
    derived = payload["derived_artifacts"]["frontier_rerun_rollup_completion_plan"]
    entry = completion["entries"][0]
    command = entry["command"]

    assert saved["derived_artifacts"] == payload["derived_artifacts"]
    assert derived["path"] == str(completion_path)
    assert derived["artifact_manifest"] == str(manifest_path)
    assert derived["status"] == "ready"
    assert derived["entry_count"] == 1
    assert derived["command_count"] == 1
    assert derived["missing_queue_count"] == 0
    assert completion["workflow"] == "frontier_rerun_rollup_completion_plan"
    assert completion["summary"]["tracks"] == ["abstention"]
    assert entry["track"] == "abstention"
    assert entry["rollup_workflow"] == "frontier_abstention_evidence_rerun_rollup"
    assert entry["command_status"] == "ready"
    assert command[:2] == [
        "python",
        "benchmarks/rollup_frontier_abstention_evidence_reruns.py",
    ]
    assert command[command.index("--queue") + 1] == str(queue_path)
    assert command[command.index("--json") + 1] == str(
        tmp_path / "frontier-rerun-rollups" / "abstention" / "frontier-rerun-rollup.json"
    )
    assert "--require-all-reports" in command
    assert manifest["artifacts"]["frontier_rerun_rollup_completion_plan"]["exists"] is True
    assert manifest["artifacts"]["abstention_rerun_queue"]["exists"] is True
    assert gap_record.metadata["gap_count"] == 1
    assert completion_record.metadata["command_count"] == 1
    assert completion_record.metadata["tracks"] == ["abstention"]


def test_plan_release_evidence_gaps_can_emit_runtime_drift_completion_plan(
    tmp_path,
):
    source = tmp_path / "release-candidate-comparison.json"
    output = tmp_path / "evidence-gap-plan.json"
    completion_path = tmp_path / "runtime-drift-completion.json"
    manifest_path = tmp_path / "runtime-drift-completion-manifest.json"
    registry_path = tmp_path / "registry.json"
    source.write_text(
        json.dumps({
            "schema_version": 1,
            "workflow": "release_candidate_comparison",
            "decision": {
                "status": "blocked",
                "blocking_reasons": [
                    {
                        "gate": "product_runtime_drift",
                        "status": "blocked",
                        "reasons": (
                            "product runtime drift world-model evidence metrics are incomplete: "
                            "world_model.participating_trace_rate, world_model.trace_gap_rate",
                            "product runtime drift evidence-handoff evidence metrics are incomplete: "
                            "promotion_contract.evidence_handoff.coverage_rate, "
                            "promotion_contract.evidence_handoff.promoted_group_rate.mean",
                        ),
                    }
                ],
            },
        }),
        encoding="utf-8",
    )

    payload = build_release_evidence_gap_plan(
        source=source,
        json_path=output,
        registry_path=registry_path,
        name="runtime-gap-plan",
        version="0.1",
        runtime_drift_completion_json_path=completion_path,
        runtime_drift_completion_artifact_manifest_path=manifest_path,
        runtime_drift_completion_output_dir=tmp_path / "runtime-drift-completion",
        runtime_drift_completion_name="runtime-drift-completion",
        runtime_drift_completion_version="0.1",
        python_executable="python",
    )

    saved = json.loads(output.read_text(encoding="utf-8"))
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    registry = ArtifactRegistry.load_json(registry_path)
    completion_record = registry.get("report:runtime-drift-completion:0.1")
    derived = payload["derived_artifacts"]["runtime_drift_evidence_completion_plan"]
    entries = {entry["action_id"]: entry for entry in completion["entries"]}
    world_model_entry = entries["rerun_product_trace_world_model_evidence"]
    handoff_entry = entries["refresh_product_promotion_evidence_handoff"]

    assert saved["derived_artifacts"] == payload["derived_artifacts"]
    assert derived["path"] == str(completion_path)
    assert derived["artifact_manifest"] == str(manifest_path)
    assert derived["status"] == "needs_inputs"
    assert derived["entry_count"] == 2
    assert derived["command_template_count"] == (
        len(_WORLD_MODEL_RUNTIME_EVIDENCE_COMMANDS)
        + len(_EVIDENCE_HANDOFF_RUNTIME_EVIDENCE_COMMANDS)
    )
    assert completion["workflow"] == "runtime_drift_evidence_completion_plan"
    assert completion["status"] == "needs_inputs"
    assert completion["summary"]["command_status_counts"] == {"needs_inputs": 2}
    assert tuple(world_model_entry["command_templates"]) == _WORLD_MODEL_RUNTIME_EVIDENCE_COMMANDS
    assert tuple(handoff_entry["command_templates"]) == _EVIDENCE_HANDOFF_RUNTIME_EVIDENCE_COMMANDS
    assert world_model_entry["command_status"] == "needs_inputs"
    assert handoff_entry["command_status"] == "needs_inputs"
    assert "bound_command_template_values" in world_model_entry["missing_inputs"]
    assert "baseline_product_runtime_report" in handoff_entry["missing_inputs"]
    assert "world_model.participating_trace_rate" in world_model_entry["missing_metrics"]
    assert "promotion_contract.evidence_handoff.coverage_rate" in handoff_entry["missing_metrics"]
    assert "benchmarks/run_product_runtime_baseline.py" in world_model_entry["scripts"]
    assert "benchmarks/export_product_promotion_contract_evidence_handoff.py" in (
        handoff_entry["scripts"]
    )
    assert manifest["artifacts"]["runtime_drift_evidence_completion_plan"]["exists"] is True
    assert completion_record.metadata["workflow"] == "runtime_drift_evidence_completion_plan"
    assert completion_record.metadata["entry_count"] == 2
    assert completion_record.metadata["command_template_count"] == derived["command_template_count"]
    assert "product_runtime_drift" in completion_record.metadata["routes"]


def test_evidence_gap_plan_maps_product_runtime_world_model_blockers():
    plan = plan_evidence_gaps_from_release_candidate({
        "workflow": "release_candidate_comparison",
        "decision": {
            "status": "blocked",
            "blocking_reasons": [
                {
                    "gate": "product_runtime_drift",
                    "status": "blocked",
                    "reasons": (
                        "product runtime drift world-model evidence metrics are incomplete: "
                        "world_model.participating_trace_rate, world_model.trace_gap_rate",
                        "product runtime drift world-model evidence blocked 1 metric(s)",
                    ),
                }
            ],
        },
    })

    payload = plan.to_dict()
    actions = {action["action_id"]: action for action in payload["actions"]}

    assert payload["status"] == "needs_evidence"
    assert payload["summary"]["gap_count"] == 2
    assert payload["summary"]["action_count"] == 1
    assert payload["summary"]["missing_metric_count"] == 2
    assert payload["summary"]["root_causes"] == {"world_model": 2}
    assert payload["summary"]["research_axes"] == {"runtime_drift": 2}
    assert payload["summary"]["top_action_ids"] == (
        "rerun_product_trace_world_model_evidence",
    )
    _assert_world_model_runtime_evidence_action(
        actions["rerun_product_trace_world_model_evidence"]
    )
    for gap in payload["gaps"]:
        assert gap["metadata"]["evidence_kind"] == "product_runtime_world_model_evidence"
        assert gap["recommended_action_ids"] == (
            "rerun_product_trace_world_model_evidence",
        )
    assert payload["gaps"][0]["missing_metrics"] == (
        "world_model.participating_trace_rate",
        "world_model.trace_gap_rate",
    )
    assert payload["gaps"][1]["missing_metrics"] == ()


def test_evidence_gap_plan_maps_product_runtime_provenance_blockers():
    plan = plan_evidence_gaps_from_release_candidate({
        "workflow": "release_candidate_comparison",
        "decision": {
            "status": "blocked",
            "blocking_reasons": [
                {
                    "gate": "product_runtime_drift",
                    "status": "blocked",
                    "reasons": (
                        "product runtime drift provenance evidence metrics are incomplete: "
                        "provenance.coverage_rate, "
                        "provenance.supported_claim_evidence_coverage",
                        "product runtime drift provenance evidence blocked 1 metric(s)",
                    ),
                }
            ],
        },
    })

    payload = plan.to_dict()
    actions = {action["action_id"]: action for action in payload["actions"]}

    assert payload["status"] == "needs_evidence"
    assert payload["summary"]["gap_count"] == 2
    assert payload["summary"]["action_count"] == 1
    assert payload["summary"]["missing_metric_count"] == 2
    assert payload["summary"]["root_causes"] == {"evidence_coverage": 2}
    assert payload["summary"]["research_axes"] == {"trace_provenance": 2}
    assert payload["summary"]["top_action_ids"] == (
        "rerun_product_trace_provenance_evidence",
    )
    _assert_provenance_runtime_evidence_action(
        actions["rerun_product_trace_provenance_evidence"]
    )
    for gap in payload["gaps"]:
        assert gap["metadata"]["evidence_kind"] == (
            "product_runtime_provenance_evidence"
        )
        assert gap["recommended_action_ids"] == (
            "rerun_product_trace_provenance_evidence",
        )
    assert payload["gaps"][0]["missing_metrics"] == (
        "provenance.coverage_rate",
        "provenance.supported_claim_evidence_coverage",
    )
    assert payload["gaps"][1]["missing_metrics"] == ()
    strict_json_dumps(payload, sort_keys=True)
    assert EvidenceGapPlan.from_dict(payload).to_dict() == payload


def test_evidence_gap_plan_maps_product_runtime_citation_integrity_blockers():
    plan = plan_evidence_gaps_from_release_candidate({
        "workflow": "release_candidate_comparison",
        "decision": {
            "status": "blocked",
            "blocking_reasons": [
                {
                    "gate": "product_runtime_drift",
                    "status": "blocked",
                    "reasons": (
                        "product runtime drift citation integrity evidence metrics are incomplete: "
                        "citation_integrity.participating_trace_rate, "
                        "citation_integrity.trace_gap_rate",
                        "product runtime drift citation integrity evidence blocked 2 metric(s)",
                    ),
                }
            ],
        },
    })

    payload = plan.to_dict()
    actions = {action["action_id"]: action for action in payload["actions"]}

    assert payload["status"] == "needs_evidence"
    assert payload["summary"]["gap_count"] == 2
    assert payload["summary"]["action_count"] == 1
    assert payload["summary"]["missing_metric_count"] == 2
    assert payload["summary"]["root_causes"] == {"evidence_coverage": 2}
    assert payload["summary"]["research_axes"] == {"citation_integrity": 2}
    assert payload["summary"]["top_action_ids"] == (
        "rerun_product_trace_citation_integrity_evidence",
    )
    _assert_citation_integrity_runtime_evidence_action(
        actions["rerun_product_trace_citation_integrity_evidence"]
    )
    for gap in payload["gaps"]:
        assert gap["metadata"]["evidence_kind"] == (
            "product_runtime_citation_integrity_evidence"
        )
        assert gap["recommended_action_ids"] == (
            "rerun_product_trace_citation_integrity_evidence",
        )
    assert payload["gaps"][0]["missing_metrics"] == (
        "citation_integrity.participating_trace_rate",
        "citation_integrity.trace_gap_rate",
    )
    assert payload["gaps"][1]["missing_metrics"] == ()
    strict_json_dumps(payload, sort_keys=True)
    assert EvidenceGapPlan.from_dict(payload).to_dict() == payload


def test_evidence_gap_plan_maps_product_runtime_trace_robustness_blockers():
    plan = plan_evidence_gaps_from_release_candidate({
        "workflow": "release_candidate_comparison",
        "decision": {
            "status": "blocked",
            "blocking_reasons": [
                {
                    "gate": "product_runtime_drift",
                    "status": "blocked",
                    "reasons": (
                        "product runtime drift context-sensitivity evidence metrics are incomplete: "
                        "context_sensitivity.participating_trace_rate, "
                        "context_sensitivity.trace_gap_rate",
                        "product runtime drift counterfactual-robustness evidence metrics are incomplete: "
                        "counterfactual_robustness.participating_trace_rate, "
                        "counterfactual_robustness.false_invariance_rate",
                    ),
                }
            ],
        },
    })

    payload = plan.to_dict()
    actions = {action["action_id"]: action for action in payload["actions"]}
    gaps_by_kind = {
        gap["metadata"]["evidence_kind"]: gap
        for gap in payload["gaps"]
    }

    assert payload["status"] == "needs_evidence"
    assert payload["summary"]["gap_count"] == 2
    assert payload["summary"]["action_count"] == 2
    assert payload["summary"]["missing_metric_count"] == 4
    assert payload["summary"]["root_causes"] == {
        "context_sensitivity": 1,
        "counterfactual_robustness": 1,
    }
    assert payload["summary"]["research_axes"] == {"runtime_drift": 2}
    assert gaps_by_kind[
        "product_runtime_context_sensitivity_evidence"
    ]["recommended_action_ids"] == (
        "rerun_product_trace_context_sensitivity_evidence",
    )
    assert gaps_by_kind[
        "product_runtime_counterfactual_robustness_evidence"
    ]["recommended_action_ids"] == (
        "rerun_product_trace_counterfactual_robustness_evidence",
    )
    _assert_context_sensitivity_runtime_evidence_action(
        actions["rerun_product_trace_context_sensitivity_evidence"]
    )
    _assert_counterfactual_robustness_runtime_evidence_action(
        actions["rerun_product_trace_counterfactual_robustness_evidence"]
    )


def test_evidence_gap_plan_maps_product_runtime_claim_level_blockers():
    plan = plan_evidence_gaps_from_release_candidate({
        "workflow": "release_candidate_comparison",
        "decision": {
            "status": "blocked",
            "blocking_reasons": [
                {
                    "gate": "product_runtime_drift",
                    "status": "blocked",
                    "reasons": (
                        "product runtime drift claim factuality evidence metrics are incomplete: "
                        "promotion_contract.claim_factuality_probe_comparison.coverage_rate, "
                        "promotion_contract.claim_factuality_probe_comparison.best_redline_margin.mean",
                        "product runtime drift claim-risk localization evidence metrics are incomplete: "
                        "claim_risk_localization.coverage_rate, "
                        "claim_risk_localization.high_risk_entity_candidate_count",
                        "product runtime drift claim-risk localization evidence blocked 1 metric(s)",
                    ),
                }
            ],
        },
    })

    payload = plan.to_dict()
    actions = {action["action_id"]: action for action in payload["actions"]}
    gaps_by_kind = {
        gap["metadata"]["evidence_kind"]: gap
        for gap in payload["gaps"]
        if gap["missing_metrics"]
    }

    assert payload["status"] == "needs_evidence"
    assert payload["summary"]["gap_count"] == 3
    assert payload["summary"]["action_count"] == 2
    assert payload["summary"]["missing_metric_count"] == 4
    assert payload["summary"]["root_causes"] == {
        "claim_factuality": 1,
        "claim_risk_localization": 2,
    }
    assert gaps_by_kind[
        "product_runtime_claim_factuality_evidence"
    ]["recommended_action_ids"] == (
        "rerun_claim_factuality_probe_comparison",
    )
    assert gaps_by_kind[
        "product_runtime_claim_risk_localization_evidence"
    ]["recommended_action_ids"] == (
        "rerun_product_trace_claim_risk_localization_evidence",
    )
    _assert_claim_factuality_runtime_evidence_action(
        actions["rerun_claim_factuality_probe_comparison"]
    )
    _assert_claim_risk_localization_runtime_evidence_action(
        actions["rerun_product_trace_claim_risk_localization_evidence"]
    )
    assert payload["gaps"][2]["missing_metrics"] == ()
    strict_json_dumps(payload, sort_keys=True)
    assert EvidenceGapPlan.from_dict(payload).to_dict() == payload


def test_evidence_gap_plan_maps_product_runtime_trace_and_handoff_blockers():
    plan = plan_evidence_gaps_from_release_candidate({
        "workflow": "release_candidate_comparison",
        "decision": {
            "status": "blocked",
            "blocking_reasons": [
                {
                    "gate": "product_runtime_drift",
                    "status": "blocked",
                    "reasons": (
                        "product runtime drift trajectory-audit evidence metrics are incomplete: "
                        "trajectory_audit.error_rate, trajectory_audit.scope_rate",
                        "product runtime drift evidence-handoff evidence metrics are incomplete: "
                        "promotion_contract.evidence_handoff.coverage_rate, "
                        "promotion_contract.evidence_handoff.promoted_group_rate.mean",
                    ),
                }
            ],
        },
    })

    payload = plan.to_dict()
    actions = {action["action_id"]: action for action in payload["actions"]}
    gaps_by_kind = {
        gap["metadata"]["evidence_kind"]: gap
        for gap in payload["gaps"]
    }

    assert payload["status"] == "needs_evidence"
    assert payload["summary"]["gap_count"] == 2
    assert payload["summary"]["action_count"] == 2
    assert payload["summary"]["missing_metric_count"] == 4
    assert payload["summary"]["root_causes"] == {
        "product_handoff": 1,
        "trajectory_audit": 1,
    }
    assert gaps_by_kind[
        "product_runtime_trajectory_audit_evidence"
    ]["recommended_action_ids"] == (
        "rerun_product_trace_trajectory_audit_evidence",
    )
    assert gaps_by_kind[
        "product_runtime_evidence_handoff_evidence"
    ]["recommended_action_ids"] == (
        "refresh_product_promotion_evidence_handoff",
    )
    _assert_trajectory_audit_runtime_evidence_action(
        actions["rerun_product_trace_trajectory_audit_evidence"]
    )
    _assert_evidence_handoff_runtime_evidence_action(
        actions["refresh_product_promotion_evidence_handoff"]
    )
    strict_json_dumps(payload, sort_keys=True)
    assert EvidenceGapPlan.from_dict(payload).to_dict() == payload


def test_evidence_gap_plan_maps_product_runtime_triple_audit_blockers():
    plan = plan_evidence_gaps_from_release_candidate({
        "workflow": "release_candidate_comparison",
        "decision": {
            "status": "blocked",
            "blocking_reasons": [
                {
                    "gate": "product_runtime_drift",
                    "status": "blocked",
                    "reasons": (
                        "product runtime drift triple audit evidence metrics are incomplete: "
                        "triple_coverage.claim_triple_coverage_rate, "
                        "triple_coverage.audit_pass_rate",
                        "product runtime drift triple audit evidence blocked 1 metric(s)",
                    ),
                }
            ],
        },
    })

    payload = plan.to_dict()
    actions = {action["action_id"]: action for action in payload["actions"]}
    gaps = payload["gaps"]

    assert payload["status"] == "needs_evidence"
    assert payload["summary"]["gap_count"] == 2
    assert payload["summary"]["action_count"] == 1
    assert payload["summary"]["missing_metric_count"] == 2
    assert payload["summary"]["root_causes"] == {"evidence_coverage": 2}
    assert payload["summary"]["research_axes"] == {"fact_level": 2}
    assert payload["summary"]["top_action_ids"] == ("add_trace_level_triple_audit",)
    _assert_triple_audit_runtime_evidence_action(
        actions["add_trace_level_triple_audit"]
    )
    for gap in gaps:
        assert gap["metadata"]["evidence_kind"] == "triple_audit"
        assert gap["recommended_action_ids"] == ("add_trace_level_triple_audit",)
    assert gaps[0]["missing_metrics"] == (
        "triple_coverage.claim_triple_coverage_rate",
        "triple_coverage.audit_pass_rate",
    )
    assert gaps[1]["missing_metrics"] == ()
    strict_json_dumps(payload, sort_keys=True)
    assert EvidenceGapPlan.from_dict(payload).to_dict() == payload


def test_evidence_gap_plan_maps_product_runtime_covered_fact_property_blockers():
    plan = plan_evidence_gaps_from_release_candidate({
        "workflow": "release_candidate_comparison",
        "decision": {
            "status": "blocked",
            "blocking_reasons": [
                {
                    "gate": "product_runtime_drift",
                    "status": "blocked",
                    "reasons": (
                        "product runtime drift covered-fact property evidence metrics are incomplete: "
                        "promotion_contract.covered_fact_properties."
                        "recommended_route_property_metrics.property_metric_count.mean, "
                        "promotion_contract.covered_fact_properties."
                        "recommended_route_property_metrics.min_decision_accuracy.mean",
                        "product runtime drift covered-fact property evidence blocked 1 metric(s)",
                    ),
                }
            ],
        },
    })

    payload = plan.to_dict()
    actions = {action["action_id"]: action for action in payload["actions"]}
    gaps = payload["gaps"]

    assert payload["status"] == "needs_evidence"
    assert payload["summary"]["gap_count"] == 2
    assert payload["summary"]["action_count"] == 1
    assert payload["summary"]["missing_metric_count"] == 2
    assert payload["summary"]["root_causes"] == {"evidence_coverage": 2}
    assert payload["summary"]["research_axes"] == {"structured_facts": 2}
    assert payload["summary"]["top_action_ids"] == (
        "refresh_covered_fact_property_routes",
    )
    _assert_covered_fact_property_runtime_evidence_action(
        actions["refresh_covered_fact_property_routes"]
    )
    for gap in gaps:
        assert gap["metadata"]["evidence_kind"] == "covered_fact_property"
        assert gap["recommended_action_ids"] == (
            "refresh_covered_fact_property_routes",
        )
    assert gaps[0]["missing_metrics"] == (
        "promotion_contract.covered_fact_properties."
        "recommended_route_property_metrics.property_metric_count.mean",
        "promotion_contract.covered_fact_properties."
        "recommended_route_property_metrics.min_decision_accuracy.mean",
    )
    assert gaps[1]["missing_metrics"] == ()
    strict_json_dumps(payload, sort_keys=True)
    assert EvidenceGapPlan.from_dict(payload).to_dict() == payload


def test_evidence_gap_plan_maps_product_runtime_frontier_release_metrics():
    plan = plan_evidence_gaps_from_release_candidate({
        "workflow": "release_candidate_comparison",
        "decision": {
            "status": "blocked",
            "blocking_reasons": [
                {
                    "gate": "product_runtime_drift",
                    "status": "blocked",
                    "reasons": (
                        "product runtime drift frontier release evidence metrics are incomplete: "
                        "promotion_contract.frontier_release_evidence.coverage_rate, "
                        "promotion_contract.frontier_release_evidence.frontier_rerun_rollup_track_promote_rate, "
                        "promotion_contract.frontier_release_evidence.frontier_rerun_rollup_report_count.mean",
                    ),
                }
            ],
        },
    })

    payload = plan.to_dict()
    actions = {action["action_id"]: action for action in payload["actions"]}
    gap = payload["gaps"][0]

    assert payload["status"] == "needs_evidence"
    assert payload["summary"]["gap_count"] == 1
    assert payload["summary"]["action_count"] == 1
    assert payload["summary"]["research_axes"] == {"runtime_drift": 1}
    assert payload["summary"]["root_causes"] == {"product_handoff": 1}
    assert payload["summary"]["top_action_ids"] == (
        "refresh_frontier_release_evidence_promotion_metrics",
    )
    assert gap["metadata"]["evidence_kind"] == (
        "product_runtime_frontier_release_evidence"
    )
    assert gap["recommended_action_ids"] == (
        "refresh_frontier_release_evidence_promotion_metrics",
    )
    assert gap["missing_metrics"] == (
        "promotion_contract.frontier_release_evidence.coverage_rate",
        "promotion_contract.frontier_release_evidence.frontier_rerun_rollup_track_promote_rate",
        "promotion_contract.frontier_release_evidence.frontier_rerun_rollup_report_count.mean",
    )
    _assert_frontier_release_runtime_evidence_action(
        actions["refresh_frontier_release_evidence_promotion_metrics"]
    )
    assert "complete_frontier_rerun_rollup_evidence" not in actions
    strict_json_dumps(payload, sort_keys=True)
    assert EvidenceGapPlan.from_dict(payload).to_dict() == payload


def test_evidence_gap_plan_maps_product_runtime_frontier_release_blocked_metrics():
    plan = plan_evidence_gaps_from_release_candidate({
        "workflow": "release_candidate_comparison",
        "decision": {
            "status": "blocked",
            "blocking_reasons": [
                {
                    "gate": "product_runtime_drift",
                    "status": "blocked",
                    "reasons": (
                        "product runtime drift status is 'blocked', expected 'promote'",
                        "product runtime drift decision status is 'blocked', expected 'promote'",
                        "product runtime drift blocked 3 metric(s)",
                        "product runtime drift frontier release evidence blocked 3 metric(s)",
                    ),
                }
            ],
        },
    })

    payload = plan.to_dict()
    actions = {action["action_id"]: action for action in payload["actions"]}
    gaps = payload["gaps"]

    assert payload["status"] == "needs_evidence"
    assert payload["summary"]["gap_count"] == 1
    assert payload["summary"]["research_axes"] == {"runtime_drift": 1}
    assert payload["summary"]["root_causes"] == {"product_handoff": 1}
    assert payload["summary"]["top_action_ids"] == (
        "refresh_frontier_release_evidence_promotion_metrics",
    )
    assert gaps[0]["reason"] == "product runtime drift frontier release evidence blocked 3 metric(s)"
    assert gaps[0]["metadata"]["evidence_kind"] == (
        "product_runtime_frontier_release_evidence"
    )
    assert gaps[0]["recommended_action_ids"] == (
        "refresh_frontier_release_evidence_promotion_metrics",
    )
    _assert_frontier_release_runtime_evidence_action(
        actions["refresh_frontier_release_evidence_promotion_metrics"]
    )
    strict_json_dumps(payload, sort_keys=True)
    assert EvidenceGapPlan.from_dict(payload).to_dict() == payload


def test_plan_release_evidence_gaps_cli_helper_writes_and_registers(tmp_path):
    source = tmp_path / "release-workflow.json"
    output = tmp_path / "evidence-gap-plan.json"
    registry_path = tmp_path / "registry.json"
    source.write_text(json.dumps(_blocked_registry_workflow_payload()), encoding="utf-8")

    payload = build_release_evidence_gap_plan(
        source=source,
        json_path=output,
        registry_path=registry_path,
        name="frontier-audit-gap-plan",
        version="0.1",
        metadata={"scope": "unit-test"},
    )

    assert output.exists()
    assert payload["metadata"] == {"scope": "unit-test"}
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["summary"]["action_count"] == payload["summary"]["action_count"]
    registry = ArtifactRegistry.load_json(registry_path)
    record = registry.get("evidence_gap_plan:frontier-audit-gap-plan:0.1")
    assert record.path == str(output)
    assert record.metadata["status"] == "needs_evidence"
    assert record.metadata["gap_count"] == 4


def test_plan_release_evidence_gaps_can_emit_multiple_testing_rerun_queue(tmp_path):
    source = tmp_path / "frontier-release-evidence.json"
    output = tmp_path / "evidence-gap-plan.json"
    queue_path = tmp_path / "multiple-testing-rerun-queue.json"
    manifest_path = tmp_path / "multiple-testing-rerun-queue-manifest.json"
    registry_path = tmp_path / "registry.json"
    workflow_path = tmp_path / "frontier" / "truthfulqa-frontier-workflow.json"
    workflow_path.parent.mkdir(parents=True, exist_ok=True)
    workflow_path.write_text(
        json.dumps(_frontier_workflow_payload_for_multiple_testing_queue()),
        encoding="utf-8",
    )
    source.write_text(
        json.dumps({
            "schema_version": 1,
            "workflow": "frontier_release_evidence_comparison",
            "status": "complete",
            "inputs": {
                "frontier_workflow_reports": (
                    {
                        "path": str(workflow_path),
                        "workflow": "truthfulqa_frontier_workflow",
                        "status": "complete",
                    },
                ),
            },
            "decision": {
                "status": "blocked",
                "multiple_testing_track_status": "blocked",
                "blocking_reasons": (
                    "truthfulqa_frontier_workflow.synthetic.multiple_testing_gate.all_pass is not true",
                ),
            },
            "evidence_summary": {
                "multiple_testing_failed_cells": (
                    {
                        "run": "truthfulqa-frontier-workflow",
                        "cell": "a-l2",
                        "status": "failed",
                        "false_alarm": 0.03,
                        "detection": 0.7,
                        "report": "frontier/a-l2/multiple-testing-report.json",
                        "calibration": "frontier/a-l2/multiple-testing-calibration.json",
                    },
                ),
            },
        }),
        encoding="utf-8",
    )

    payload = build_release_evidence_gap_plan(
        source=source,
        json_path=output,
        registry_path=registry_path,
        name="frontier-gap-plan",
        version="0.1",
        multiple_testing_rerun_json_path=queue_path,
        multiple_testing_rerun_artifact_manifest_path=manifest_path,
        multiple_testing_rerun_output_dir=tmp_path / "reruns",
        multiple_testing_rerun_name="frontier-multiple-testing-reruns",
        multiple_testing_rerun_version="0.1",
        python_executable="python",
    )

    saved = json.loads(output.read_text(encoding="utf-8"))
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    registry = ArtifactRegistry.load_json(registry_path)
    gap_record = registry.get("evidence_gap_plan:frontier-gap-plan:0.1")
    queue_record = registry.get("report:frontier-multiple-testing-reruns:0.1")

    derived = payload["derived_artifacts"]["frontier_multiple_testing_rerun_queue"]
    assert saved["derived_artifacts"] == payload["derived_artifacts"]
    assert derived["path"] == str(queue_path)
    assert derived["artifact_manifest"] == str(manifest_path)
    assert derived["status"] == "ready"
    assert derived["blocked_cell_count"] == 1
    assert derived["command_count"] == 1
    assert queue["entries"][0]["command_status"] == "ready"
    assert queue["entries"][0]["command"][:3] == [
        "python",
        "benchmarks/run_truthfulqa_frontier_workflow.py",
        "--output-dir",
    ]
    assert queue["entries"][0]["dry_run_command"][-1] == "--dry-run"
    assert manifest["artifacts"]["frontier_multiple_testing_rerun_queue"]["exists"] is True
    assert gap_record.metadata["gap_count"] == 1
    assert queue_record.metadata["blocked_cell_count"] == 1
    assert queue_record.metadata["command_count"] == 1


def test_frontier_multiple_testing_rerun_queue_expands_missing_gate_from_workflow_config(tmp_path):
    source = tmp_path / "frontier-release-evidence.json"
    queue_path = tmp_path / "multiple-testing-rerun-queue.json"
    workflow_path = tmp_path / "frontier" / "truthfulqa-frontier-workflow.json"
    workflow = _frontier_workflow_payload_for_multiple_testing_queue()
    workflow.pop("multiple_testing_gate")
    workflow["config"]["multiple_testing_signals"] = ()
    score_dump_path = tmp_path / "frontier" / "a-l2" / "scores.manifest.json"
    workflow["cells"] = (
        {
            "name": "a-l2",
            "score_dump": {"path": str(score_dump_path)},
        },
    )
    workflow_path.parent.mkdir(parents=True, exist_ok=True)
    workflow_path.write_text(json.dumps(workflow), encoding="utf-8")
    source.write_text(
        json.dumps({
            "schema_version": 1,
            "workflow": "frontier_release_evidence_comparison",
            "status": "complete",
            "inputs": {
                "frontier_workflow_reports": (
                    {
                        "path": str(workflow_path),
                        "workflow": "truthfulqa_frontier_workflow",
                        "status": "complete",
                    },
                ),
            },
            "multiple_testing_decisions": (
                {
                    "name": "truthfulqa-frontier-workflow",
                    "status": "blocked",
                    "track": "truthfulqa_frontier_workflow.multiple_testing_gate",
                    "metrics": {
                        "enabled": None,
                        "all_pass": None,
                        "cell_count": None,
                        "failed_cells": (),
                        "unknown_cells": (),
                        "blocked_cells": (),
                    },
                    "blocking_reasons": (
                        "truthfulqa_frontier_workflow.truthfulqa-frontier-workflow."
                        "multiple_testing_gate missing",
                    ),
                },
            ),
            "evidence_summary": {
                "multiple_testing_failed_cells": (),
                "multiple_testing_unknown_cells": (),
                "multiple_testing_blocked_cells": (),
            },
        }),
        encoding="utf-8",
    )

    payload = build_frontier_multiple_testing_rerun_queue(
        source=source,
        json_path=queue_path,
        output_dir=tmp_path / "reruns",
        python_executable="python",
    )
    entry = payload["entries"][0]

    assert payload["status"] == "ready"
    assert payload["summary"]["blocked_cell_count"] == 1
    assert payload["summary"]["command_count"] == 1
    assert entry["run"] == "truthfulqa-frontier-workflow"
    assert entry["cell"] == "a-l2"
    assert entry["status"] == "missing_gate"
    assert entry["command_status"] == "ready"
    assert entry["command"][entry["command"].index("--scores") + 1] == str(score_dump_path)
    assert entry["command"][entry["command"].index("--multiple-testing-signals") + 1] == (
        "truth_proj,subspace_resid"
    )


def test_frontier_multiple_testing_rerun_rollup_promotes_passing_cell(tmp_path):
    source = tmp_path / "frontier-release-evidence.json"
    queue_path = tmp_path / "multiple-testing-rerun-queue.json"
    rollup_path = tmp_path / "multiple-testing-rerun-rollup.json"
    manifest_path = tmp_path / "multiple-testing-rerun-rollup-manifest.json"
    registry_path = tmp_path / "registry.json"
    workflow_path = tmp_path / "frontier" / "truthfulqa-frontier-workflow.json"
    workflow_path.parent.mkdir(parents=True, exist_ok=True)
    workflow_path.write_text(json.dumps(_frontier_workflow_payload_for_multiple_testing_queue()), encoding="utf-8")
    source.write_text(
        json.dumps({
            "schema_version": 1,
            "workflow": "frontier_release_evidence_comparison",
            "status": "complete",
            "inputs": {
                "frontier_workflow_reports": (
                    {
                        "path": str(workflow_path),
                        "workflow": "truthfulqa_frontier_workflow",
                        "status": "complete",
                    },
                ),
            },
            "evidence_summary": {
                "multiple_testing_failed_cells": (
                    {
                        "run": "truthfulqa-frontier-workflow",
                        "cell": "a-l2",
                        "status": "failed",
                        "false_alarm": 0.03,
                        "detection": 0.7,
                        "report": "frontier/a-l2/multiple-testing-report.json",
                        "calibration": "frontier/a-l2/multiple-testing-calibration.json",
                    },
                ),
            },
        }),
        encoding="utf-8",
    )
    queue = build_frontier_multiple_testing_rerun_queue(
        source=source,
        json_path=queue_path,
        output_dir=tmp_path / "reruns",
        python_executable="python",
    )
    entry = queue["entries"][0]
    child_report_path = _multiple_testing_queue_entry_report_path(entry)
    child_report_path.parent.mkdir(parents=True, exist_ok=True)
    child_report_path.write_text(
        json.dumps(_frontier_workflow_multiple_testing_child_report(cell="a-l2", passed=True)),
        encoding="utf-8",
    )

    payload = rollup_frontier_multiple_testing_reruns(
        queue_path=queue_path,
        report_json_path=rollup_path,
        artifact_manifest_path=manifest_path,
        registry_path=registry_path,
        name="frontier-multiple-testing-rerun-rollup",
        version="0.1",
        require_all_reports=True,
        metadata={"suite": "unit"},
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = ArtifactRegistry.load_json(registry_path).get(
        "report:frontier-multiple-testing-rerun-rollup:0.1"
    )

    assert payload["workflow"] == "frontier_multiple_testing_rerun_rollup"
    assert payload["status"] == "promote"
    assert payload["gate"]["promotion_ready"] is True
    assert payload["summary"]["candidate_count"] == 1
    assert payload["summary"]["promotion_ready_count"] == 1
    assert payload["recommended_candidate"]["cell"] == "a-l2"
    assert payload["recommended_candidate"]["metrics"]["cell_status"] == "passed"
    assert manifest["artifacts"]["frontier_multiple_testing_rerun_rollup"]["exists"] is True
    assert record.metadata["workflow"] == "frontier_multiple_testing_rerun_rollup"
    assert record.metadata["promotion_ready"] is True
    assert record.metadata["best_cell"] == "a-l2"
    assert record.metadata["suite"] == "unit"


def test_frontier_multiple_testing_rerun_rollup_blocks_missing_report_by_default(tmp_path):
    queue_path = tmp_path / "multiple-testing-rerun-queue.json"
    rollup_path = tmp_path / "multiple-testing-rerun-rollup.json"
    queue_path.write_text(
        json.dumps({
            "schema_version": 1,
            "workflow": "frontier_multiple_testing_rerun_queue",
            "status": "ready",
            "entries": (
                {
                    "run": "truthfulqa-frontier-workflow",
                    "cell": "a-l2",
                    "status": "failed",
                    "command_status": "ready",
                    "command": (
                        "python",
                        "benchmarks/run_truthfulqa_frontier_workflow.py",
                        "--output-dir",
                        str(tmp_path / "reruns" / "a-l2"),
                    ),
                },
            ),
        }),
        encoding="utf-8",
    )

    payload = rollup_frontier_multiple_testing_reruns(
        queue_path=queue_path,
        report_json_path=rollup_path,
    )

    assert payload["status"] == "blocked"
    assert payload["gate"]["passed"] is False
    assert payload["gate"]["require_all_reports"] is False
    assert payload["summary"]["missing_report_count"] == 1
    assert payload["gate"]["blocking_reasons"][0]["gate"] == "report_coverage"


def test_citation_batch_evidence_rerun_queue_builds_source_family_commands(tmp_path):
    source = tmp_path / "frontier-release-evidence.json"
    queue_path = tmp_path / "citation-batch-rerun-queue.json"
    manifest_path = tmp_path / "citation-batch-rerun-queue-manifest.json"
    registry_path = tmp_path / "registry.json"
    source.write_text(json.dumps(_frontier_release_citation_batch_payload()), encoding="utf-8")

    payload = build_citation_batch_evidence_rerun_queue(
        source=source,
        json_path=queue_path,
        artifact_manifest_path=manifest_path,
        registry_path=registry_path,
        name="citation-batch-reruns",
        version="0.1",
        output_dir=tmp_path / "reruns",
        queue_report_path=tmp_path / "unresolved-queue.json",
        scores_path=tmp_path / "scores.jsonl",
        blind_spots_path=tmp_path / "blind-spots.jsonl",
        source_catalog_paths=(tmp_path / "catalog.jsonl",),
        controlled_sweep_paths=(tmp_path / "controlled-sweep.json",),
        python_executable="python",
    )

    saved = json.loads(queue_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    registry = ArtifactRegistry.load_json(registry_path)
    record = registry.get("report:citation-batch-reruns:0.1")
    entries = {entry["batch_id"]: entry for entry in payload["entries"]}
    command = entries["unresolved-evidence-batch-0002"]["command"]

    assert saved["summary"] == payload["summary"]
    assert payload["workflow"] == "citation_batch_evidence_rerun_queue"
    assert payload["summary"]["blocked_batch_count"] == 2
    assert payload["summary"]["missing_expected_batch_count"] == 1
    assert payload["summary"]["duplicate_batch_count"] == 1
    assert payload["summary"]["command_count"] == 2
    assert entries["unresolved-evidence-batch-0002"]["issue_type"] == "missing_expected"
    assert entries["unresolved-evidence-batch-0002"]["command_status"] == "ready"
    assert entries["unresolved-evidence-batch-0002"]["command_kind"] == "source_family"
    assert command[:3] == (
        "python",
        "benchmarks/run_source_family_citation_search_workflow.py",
        "--queue",
    )
    assert command[command.index("--batch-id") + 1] == "unresolved-evidence-batch-0002"
    assert command[command.index("--source-catalog") + 1] == str(tmp_path / "catalog.jsonl")
    assert command[command.index("--controlled-sweep") + 1] == str(tmp_path / "controlled-sweep.json")
    assert manifest["artifacts"]["citation_batch_evidence_rerun_queue"]["exists"] is True
    assert record.metadata["workflow"] == "citation_batch_evidence_rerun_queue"
    assert record.metadata["blocked_batch_count"] == 2


def test_plan_release_evidence_gaps_can_emit_citation_batch_rerun_queue(tmp_path):
    source = tmp_path / "frontier-release-evidence.json"
    output = tmp_path / "evidence-gap-plan.json"
    queue_path = tmp_path / "citation-batch-rerun-queue.json"
    manifest_path = tmp_path / "citation-batch-rerun-queue-manifest.json"
    registry_path = tmp_path / "registry.json"
    source.write_text(json.dumps(_frontier_release_citation_batch_payload()), encoding="utf-8")

    payload = build_release_evidence_gap_plan(
        source=source,
        json_path=output,
        registry_path=registry_path,
        name="frontier-gap-plan",
        version="0.1",
        citation_batch_rerun_json_path=queue_path,
        citation_batch_rerun_artifact_manifest_path=manifest_path,
        citation_batch_rerun_output_dir=tmp_path / "citation-reruns",
        citation_batch_rerun_name="citation-batch-reruns",
        citation_batch_rerun_version="0.1",
        citation_batch_queue_report_path=tmp_path / "unresolved-queue.json",
        citation_batch_scores_path=tmp_path / "scores.jsonl",
        citation_batch_blind_spots_path=tmp_path / "blind-spots.jsonl",
        citation_batch_search_command="python adapter.py --input {input} --output {output}",
        python_executable="python",
    )

    saved = json.loads(output.read_text(encoding="utf-8"))
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    registry = ArtifactRegistry.load_json(registry_path)
    gap_record = registry.get("evidence_gap_plan:frontier-gap-plan:0.1")
    queue_record = registry.get("report:citation-batch-reruns:0.1")
    derived = payload["derived_artifacts"]["citation_batch_evidence_rerun_queue"]
    entry = queue["entries"][0]

    assert saved["derived_artifacts"] == payload["derived_artifacts"]
    assert derived["path"] == str(queue_path)
    assert derived["artifact_manifest"] == str(manifest_path)
    assert derived["status"] == "ready"
    assert derived["blocked_batch_count"] == 2
    assert derived["command_count"] == 2
    assert entry["command_status"] == "ready"
    assert entry["command_kind"] == "external"
    assert entry["command"][1] == "benchmarks/run_external_citation_search_adapter_workflow.py"
    assert entry["command"][entry["command"].index("--search-command") + 1] == (
        "python adapter.py --input {input} --output {output}"
    )
    assert gap_record.metadata["gap_count"] == 1
    assert queue_record.metadata["command_count"] == 2


def test_frontier_stability_evidence_rerun_queue_builds_commands(tmp_path):
    source = tmp_path / "frontier-release-evidence.json"
    queue_path = tmp_path / "stability-rerun-queue.json"
    manifest_path = tmp_path / "stability-rerun-queue-manifest.json"
    registry_path = tmp_path / "registry.json"
    source.write_text(json.dumps(_frontier_release_stability_payload()), encoding="utf-8")

    payload = build_frontier_stability_evidence_rerun_queue(
        source=source,
        json_path=queue_path,
        artifact_manifest_path=manifest_path,
        registry_path=registry_path,
        name="frontier-stability-reruns",
        version="0.1",
        output_dir=tmp_path / "stability-reruns",
        score_paths=(
            f"qwen={tmp_path / 'qwen-scores.manifest.json'}",
            f"smol={tmp_path / 'smol-scores.manifest.json'}",
        ),
        seeds="0,1",
        verifier_qa_corpus_path=tmp_path / "qa-corpus.json",
        python_executable="python",
    )

    saved = json.loads(queue_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    registry = ArtifactRegistry.load_json(registry_path)
    record = registry.get("report:frontier-stability-reruns:0.1")
    entries = {entry["track"]: entry for entry in payload["entries"]}
    verifier_command = entries["verifier_stability"]["command"]
    abstention_command = entries["abstention_stability"]["command"]

    assert saved["summary"] == payload["summary"]
    assert payload["workflow"] == "frontier_stability_evidence_rerun_queue"
    assert payload["summary"]["blocked_track_count"] == 2
    assert payload["summary"]["command_count"] == 2
    assert entries["verifier_stability"]["command_status"] == "ready"
    assert entries["abstention_stability"]["command_status"] == "ready"
    assert verifier_command[:2] == ("python", "benchmarks/eval_verifier_stability.py")
    assert verifier_command[verifier_command.index("--signal") + 1] == "truth_proj"
    assert verifier_command[verifier_command.index("--qa-corpus") + 1] == str(tmp_path / "qa-corpus.json")
    assert "--staged-verification" in verifier_command
    assert abstention_command[:2] == ("python", "benchmarks/eval_abstention_stability.py")
    assert abstention_command[abstention_command.index("--signals") + 1] == "maha_last,subspace_resid"
    assert abstention_command[abstention_command.index("--seeds") + 1] == "0,1"
    assert manifest["artifacts"]["frontier_stability_evidence_rerun_queue"]["exists"] is True
    assert record.metadata["workflow"] == "frontier_stability_evidence_rerun_queue"
    assert record.metadata["blocked_track_count"] == 2


def test_plan_release_evidence_gaps_can_emit_stability_rerun_queue(tmp_path):
    source = tmp_path / "frontier-release-evidence.json"
    output = tmp_path / "evidence-gap-plan.json"
    queue_path = tmp_path / "stability-rerun-queue.json"
    manifest_path = tmp_path / "stability-rerun-queue-manifest.json"
    registry_path = tmp_path / "registry.json"
    source.write_text(json.dumps(_frontier_release_stability_payload()), encoding="utf-8")

    payload = build_release_evidence_gap_plan(
        source=source,
        json_path=output,
        registry_path=registry_path,
        name="frontier-gap-plan",
        version="0.1",
        stability_rerun_json_path=queue_path,
        stability_rerun_artifact_manifest_path=manifest_path,
        stability_rerun_output_dir=tmp_path / "stability-reruns",
        stability_rerun_name="frontier-stability-reruns",
        stability_rerun_version="0.1",
        stability_score_paths=(f"qwen={tmp_path / 'qwen-scores.manifest.json'}",),
        stability_seeds="0,1",
        python_executable="python",
    )

    saved = json.loads(output.read_text(encoding="utf-8"))
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    registry = ArtifactRegistry.load_json(registry_path)
    gap_record = registry.get("evidence_gap_plan:frontier-gap-plan:0.1")
    queue_record = registry.get("report:frontier-stability-reruns:0.1")
    derived = payload["derived_artifacts"]["frontier_stability_evidence_rerun_queue"]

    assert saved["derived_artifacts"] == payload["derived_artifacts"]
    assert derived["path"] == str(queue_path)
    assert derived["artifact_manifest"] == str(manifest_path)
    assert derived["status"] == "ready"
    assert derived["blocked_track_count"] == 2
    assert derived["command_count"] == 2
    assert queue["entries"][0]["command_status"] == "ready"
    assert gap_record.metadata["gap_count"] == 2
    assert queue_record.metadata["command_count"] == 2


def test_frontier_stability_evidence_rerun_rollup_promotes_passing_tracks(tmp_path):
    source = tmp_path / "frontier-release-evidence.json"
    queue_path = tmp_path / "stability-rerun-queue.json"
    rollup_path = tmp_path / "stability-rerun-rollup.json"
    manifest_path = tmp_path / "stability-rerun-rollup-manifest.json"
    registry_path = tmp_path / "registry.json"
    source.write_text(json.dumps(_frontier_release_stability_payload()), encoding="utf-8")

    queue = build_frontier_stability_evidence_rerun_queue(
        source=source,
        json_path=queue_path,
        output_dir=tmp_path / "stability-reruns",
        score_paths=(f"qwen={tmp_path / 'qwen-scores.manifest.json'}",),
        seeds="0,1",
        verifier_qa_corpus_path=tmp_path / "qa-corpus.json",
        python_executable="python",
    )
    entries = {entry["track"]: entry for entry in queue["entries"]}
    verifier_report_path = _stability_queue_entry_report_path(entries["verifier_stability"])
    abstention_report_path = _stability_queue_entry_report_path(entries["abstention_stability"])
    verifier_report_path.parent.mkdir(parents=True, exist_ok=True)
    abstention_report_path.parent.mkdir(parents=True, exist_ok=True)
    verifier_report_path.write_text(
        json.dumps(_verifier_stability_child_report(passed=True)),
        encoding="utf-8",
    )
    abstention_report_path.write_text(
        json.dumps(_abstention_stability_child_report(passed=True)),
        encoding="utf-8",
    )

    payload = rollup_frontier_stability_evidence_reruns(
        queue_path=queue_path,
        report_json_path=rollup_path,
        artifact_manifest_path=manifest_path,
        registry_path=registry_path,
        name="frontier-stability-rerun-rollup",
        version="0.1",
        metadata={"suite": "unit"},
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = ArtifactRegistry.load_json(registry_path).get(
        "report:frontier-stability-rerun-rollup:0.1"
    )

    assert payload["workflow"] == "frontier_stability_evidence_rerun_rollup"
    assert payload["status"] == "promote"
    assert payload["gate"]["promotion_ready"] is True
    assert payload["summary"]["candidate_count"] == 2
    assert payload["summary"]["promotion_ready_count"] == 2
    assert payload["summary"]["track_statuses"] == {
        "abstention_stability": "promote",
        "verifier_stability": "promote",
    }
    assert manifest["artifacts"]["frontier_stability_evidence_rerun_rollup"]["exists"] is True
    assert record.metadata["workflow"] == "frontier_stability_evidence_rerun_rollup"
    assert record.metadata["promotion_ready"] is True
    assert record.metadata["verifier_track_status"] == "promote"
    assert record.metadata["abstention_track_status"] == "promote"
    assert record.metadata["suite"] == "unit"


def test_frontier_stability_evidence_rerun_rollup_blocks_missing_report(tmp_path):
    queue_path = tmp_path / "stability-rerun-queue.json"
    rollup_path = tmp_path / "stability-rerun-rollup.json"
    queue_path.write_text(
        json.dumps({
            "schema_version": 1,
            "workflow": "frontier_stability_evidence_rerun_queue",
            "status": "ready",
            "entries": (
                {
                    "track": "verifier_stability",
                    "command_status": "ready",
                    "command": (
                        "python",
                        "benchmarks/eval_verifier_stability.py",
                        "--json",
                        str(tmp_path / "reruns" / "verifier-stability-report.json"),
                    ),
                },
            ),
        }),
        encoding="utf-8",
    )

    payload = rollup_frontier_stability_evidence_reruns(
        queue_path=queue_path,
        report_json_path=rollup_path,
    )

    assert payload["status"] == "blocked"
    assert payload["gate"]["passed"] is False
    assert payload["summary"]["missing_report_count"] == 1
    assert payload["gate"]["blocking_reasons"][0]["gate"] == "report_coverage"


def test_frontier_stability_evidence_rerun_rollup_blocks_failed_child_thresholds(tmp_path):
    queue_path = tmp_path / "stability-rerun-queue.json"
    rollup_path = tmp_path / "stability-rerun-rollup.json"
    child_report_path = tmp_path / "reruns" / "verifier-stability-report.json"
    child_report_path.parent.mkdir(parents=True, exist_ok=True)
    child_report_path.write_text(
        json.dumps(_verifier_stability_child_report(passed=False)),
        encoding="utf-8",
    )
    queue_path.write_text(
        json.dumps({
            "schema_version": 1,
            "workflow": "frontier_stability_evidence_rerun_queue",
            "status": "ready",
            "entries": (
                {
                    "track": "verifier_stability",
                    "command_status": "ready",
                    "command": (
                        "python",
                        "benchmarks/eval_verifier_stability.py",
                        "--json",
                        str(child_report_path),
                    ),
                },
            ),
        }),
        encoding="utf-8",
    )

    payload = rollup_frontier_stability_evidence_reruns(
        queue_path=queue_path,
        report_json_path=rollup_path,
    )
    candidate = payload["candidates"][0]

    assert payload["status"] == "blocked"
    assert candidate["candidate_status"] == "blocked"
    assert candidate["run_decisions"][0]["status"] == "blocked"
    assert any(
        "verified_detection_mean" in reason
        for reason in candidate["run_decisions"][0]["blocking_reasons"]
    )


def test_frontier_abstention_evidence_rerun_queue_builds_profile_matrix(tmp_path):
    report_path = tmp_path / "abstention-stability.json"
    source = tmp_path / "frontier-release-evidence.json"
    queue_path = tmp_path / "abstention-rerun-queue.json"
    manifest_path = tmp_path / "abstention-rerun-queue-manifest.json"
    registry_path = tmp_path / "registry.json"
    report_path.write_text(
        json.dumps(_abstention_stability_payload(tmp_path / "qwen-scores.manifest.json")),
        encoding="utf-8",
    )
    source.write_text(
        json.dumps(_frontier_release_abstention_payload(report_path)),
        encoding="utf-8",
    )

    payload = build_frontier_abstention_evidence_rerun_queue(
        source=source,
        json_path=queue_path,
        artifact_manifest_path=manifest_path,
        registry_path=registry_path,
        name="frontier-abstention-reruns",
        version="0.1",
        output_dir=tmp_path / "abstention-reruns",
        profiles=("baseline", "selective_accuracy"),
        signal_groups=("recommended", "geometry"),
        seeds="0,1",
        python_executable="python",
    )

    saved = json.loads(queue_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    registry = ArtifactRegistry.load_json(registry_path)
    record = registry.get("report:frontier-abstention-reruns:0.1")
    commands = [entry["command"] for entry in payload["entries"]]
    selective = next(entry for entry in payload["entries"] if entry["profile"] == "selective_accuracy")
    geometry = next(entry for entry in payload["entries"] if entry["signal_group"] == "geometry")

    assert saved["summary"] == payload["summary"]
    assert payload["workflow"] == "frontier_abstention_evidence_rerun_queue"
    assert payload["summary"]["blocked_run_count"] == 1
    assert payload["summary"]["entry_count"] == 4
    assert payload["summary"]["command_count"] == 4
    assert all(command[:2] == ("python", "benchmarks/eval_abstention_stability.py") for command in commands)
    assert commands[0][commands[0].index("--scores") + 1] == f"qwen={tmp_path / 'qwen-scores.manifest.json'}"
    assert selective["command"][selective["command"].index("--best-by") + 1] == "empirical_selective_accuracy"
    assert geometry["signals"] == ("maha_last", "truth_proj", "subspace_resid")
    assert manifest["artifacts"]["frontier_abstention_evidence_rerun_queue"]["exists"] is True
    assert record.metadata["workflow"] == "frontier_abstention_evidence_rerun_queue"
    assert record.metadata["blocked_run_count"] == 1
    assert record.metadata["command_count"] == 4


def test_frontier_abstention_rerun_queue_translates_recommended_geometry_fusion(tmp_path):
    report_path = tmp_path / "abstention-stability.json"
    source = tmp_path / "frontier-release-evidence.json"
    queue_path = tmp_path / "abstention-rerun-queue.json"
    fusion_signal = (
        "geometry_uncertainty_fusion:noisy_or"
        "[geometry=mean_rank:truth_proj+subspace_resid+eigenscore;"
        "uncertainty=mean_rank:verifier_refuted+verifier_refute_confidence+verifier_not_supported]"
    )
    report = _abstention_stability_payload(tmp_path / "qwen-scores.manifest.json")
    report["config"]["signals"] = (
        "truth_proj",
        "subspace_resid",
        "eigenscore",
        "verifier_refuted",
        "verifier_refute_confidence",
        "verifier_not_supported",
    )
    report["runs"][0]["stability"]["stable_recommended_score_name"] = fusion_signal
    report["runs"][0]["stability"]["recommended_score_name_counts"] = {fusion_signal: 2}
    report_path.write_text(json.dumps(report), encoding="utf-8")
    source.write_text(
        json.dumps(_frontier_release_abstention_payload(report_path)),
        encoding="utf-8",
    )

    payload = build_frontier_abstention_evidence_rerun_queue(
        source=source,
        json_path=queue_path,
        output_dir=tmp_path / "abstention-reruns",
        profiles=("baseline",),
        signal_groups=("recommended",),
        python_executable="python",
    )
    entry = payload["entries"][0]
    command = tuple(entry["command"])

    assert entry["signals"] == (fusion_signal,)
    assert entry["derived_signal_config"]["base_signals"] == (
        "truth_proj",
        "subspace_resid",
        "eigenscore",
        "verifier_refuted",
        "verifier_refute_confidence",
        "verifier_not_supported",
    )
    assert entry["derived_signal_config"]["geometry_uncertainty"] == {
        "geometry_signals": ("truth_proj", "subspace_resid", "eigenscore"),
        "uncertainty_signals": (
            "verifier_refuted",
            "verifier_refute_confidence",
            "verifier_not_supported",
        ),
        "geometry_method": "mean_rank",
        "uncertainty_method": "mean_rank",
        "fusion_methods": ("noisy_or",),
    }
    assert command[command.index("--signals") + 1] == (
        "truth_proj,subspace_resid,eigenscore,"
        "verifier_refuted,verifier_refute_confidence,verifier_not_supported"
    )
    assert fusion_signal not in command
    assert command[command.index("--geometry-signals") + 1] == (
        "truth_proj,subspace_resid,eigenscore"
    )
    assert command[command.index("--uncertainty-signals") + 1] == (
        "verifier_refuted,verifier_refute_confidence,verifier_not_supported"
    )
    assert command[command.index("--geometry-fusion-methods") + 1] == "noisy_or"


def test_frontier_abstention_rerun_queue_emits_budget_profile_flags(tmp_path):
    report_path = tmp_path / "abstention-stability.json"
    source = tmp_path / "frontier-release-evidence.json"
    queue_path = tmp_path / "abstention-rerun-queue.json"
    report_path.write_text(
        json.dumps(_abstention_stability_payload(tmp_path / "qwen-scores.manifest.json")),
        encoding="utf-8",
    )
    source.write_text(
        json.dumps(_frontier_release_abstention_payload(report_path)),
        encoding="utf-8",
    )

    payload = build_frontier_abstention_evidence_rerun_queue(
        source=source,
        json_path=queue_path,
        output_dir=tmp_path / "abstention-reruns",
        profiles=("budget", "budget_0p48", "gate_budget_sweep"),
        signal_groups=("recommended",),
        python_executable="python",
    )
    budget = next(entry for entry in payload["entries"] if entry["profile"] == "budget")
    budget_0p48 = next(entry for entry in payload["entries"] if entry["profile"] == "budget_0p48")
    gate_budget_sweep = next(
        entry for entry in payload["entries"] if entry["profile"] == "gate_budget_sweep"
    )

    assert budget["profile_config"]["enforce_abstention_budget"] is True
    assert budget["profile_config"]["abstention_budget_target_rate"] == 0.5
    assert budget["profile_config"]["promotion_eligible"] is True
    assert "--enforce-abstention-budget" in budget["command"]
    assert budget["command"][budget["command"].index("--abstention-budget-target-rate") + 1] == "0.5"
    assert budget_0p48["profile_config"]["abstention_budget_target_rate"] == 0.48
    assert (
        budget_0p48["command"][budget_0p48["command"].index("--abstention-budget-target-rate") + 1]
        == "0.48"
    )
    assert gate_budget_sweep["profile_config"]["enforce_abstention_budget"] is True
    assert gate_budget_sweep["profile_config"]["prefer_release_gate_passing"] is True
    assert gate_budget_sweep["profile_config"]["abstention_budget_target_rates"] == (
        0.35,
        0.4,
        0.45,
        0.48,
        0.5,
    )
    assert "--prefer-release-gate-passing" in gate_budget_sweep["command"]
    assert (
        gate_budget_sweep["command"][
            gate_budget_sweep["command"].index("--abstention-budget-target-rates") + 1
        ]
        == "0.35,0.4,0.45,0.48,0.5"
    )


def test_frontier_abstention_rerun_rollup_accepts_derived_fusion_report_config(tmp_path):
    report_path = tmp_path / "abstention-stability.json"
    source = tmp_path / "frontier-release-evidence.json"
    queue_path = tmp_path / "abstention-rerun-queue.json"
    rollup_path = tmp_path / "abstention-rerun-rollup.json"
    fusion_signal = (
        "geometry_uncertainty_fusion:noisy_or"
        "[geometry=mean_rank:truth_proj+subspace_resid+eigenscore;"
        "uncertainty=mean_rank:verifier_refuted+verifier_refute_confidence+verifier_not_supported]"
    )
    report = _abstention_stability_payload(tmp_path / "qwen-scores.manifest.json")
    report["config"]["signals"] = (
        "truth_proj",
        "subspace_resid",
        "eigenscore",
        "verifier_refuted",
        "verifier_refute_confidence",
        "verifier_not_supported",
    )
    report["runs"][0]["stability"]["stable_recommended_score_name"] = fusion_signal
    report["runs"][0]["stability"]["recommended_score_name_counts"] = {fusion_signal: 2}
    report_path.write_text(json.dumps(report), encoding="utf-8")
    source.write_text(
        json.dumps(_frontier_release_abstention_payload(report_path)),
        encoding="utf-8",
    )
    queue = build_frontier_abstention_evidence_rerun_queue(
        source=source,
        json_path=queue_path,
        output_dir=tmp_path / "abstention-reruns",
        profiles=("selective_accuracy",),
        signal_groups=("recommended",),
        python_executable="python",
    )
    entry = queue["entries"][0]
    output_path = _queue_entry_report_path(entry)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_abstention_rerun_report(entry)),
        encoding="utf-8",
    )

    payload = rollup_frontier_abstention_evidence_reruns(
        queue_path=queue_path,
        report_json_path=rollup_path,
        require_all_reports=True,
    )
    candidate = payload["candidates"][0]

    assert candidate["candidate_status"] == "promotion_ready"
    assert candidate["config_check"]["matches"] is True
    assert candidate["config_check"]["mismatches"] == ()
    assert candidate["metrics"]["stable_recommended_score_name"] == fusion_signal


def test_frontier_abstention_evidence_rerun_rollup_promotes_best_candidate(tmp_path):
    report_path = tmp_path / "abstention-stability.json"
    source = tmp_path / "frontier-release-evidence.json"
    queue_path = tmp_path / "abstention-rerun-queue.json"
    rollup_path = tmp_path / "abstention-rerun-rollup.json"
    manifest_path = tmp_path / "abstention-rerun-rollup-manifest.json"
    registry_path = tmp_path / "registry.json"
    report_path.write_text(
        json.dumps(_abstention_stability_payload(tmp_path / "qwen-scores.manifest.json")),
        encoding="utf-8",
    )
    source.write_text(
        json.dumps(_frontier_release_abstention_payload(report_path)),
        encoding="utf-8",
    )
    queue = build_frontier_abstention_evidence_rerun_queue(
        source=source,
        json_path=queue_path,
        output_dir=tmp_path / "abstention-reruns",
        profiles=("baseline", "selective_accuracy"),
        signal_groups=("recommended",),
        seeds="0,1",
        python_executable="python",
    )
    for entry in queue["entries"]:
        output_path = _queue_entry_report_path(entry)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(_abstention_rerun_report(entry)),
            encoding="utf-8",
        )

    payload = rollup_frontier_abstention_evidence_reruns(
        queue_path=queue_path,
        report_json_path=rollup_path,
        artifact_manifest_path=manifest_path,
        registry_path=registry_path,
        name="frontier-abstention-rerun-rollup",
        version="0.1",
        require_all_reports=True,
        metadata={"suite": "unit"},
    )
    saved = json.loads(rollup_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = ArtifactRegistry.load_json(registry_path).get(
        "report:frontier-abstention-rerun-rollup:0.1"
    )

    assert saved["summary"]["candidate_count"] == payload["summary"]["candidate_count"]
    assert saved["summary"]["profiles"] == ["baseline", "selective_accuracy"]
    assert payload["workflow"] == "frontier_abstention_evidence_rerun_rollup"
    assert payload["status"] == "promote"
    assert payload["gate"]["promotion_ready"] is True
    assert payload["summary"]["candidate_count"] == 2
    assert payload["summary"]["observed_report_count"] == 2
    assert payload["summary"]["passing_candidate_count"] == 1
    assert payload["recommended_candidate"]["profile"] == "selective_accuracy"
    assert payload["recommended_candidate"]["metrics"]["conditional_correctness_lower_bound_mean"] == 0.86
    assert manifest["artifacts"]["frontier_abstention_evidence_rerun_rollup"]["exists"] is True
    assert record.metadata["workflow"] == "frontier_abstention_evidence_rerun_rollup"
    assert record.metadata["promotion_ready"] is True
    assert record.metadata["best_profile"] == "selective_accuracy"
    assert record.metadata["suite"] == "unit"


def test_frontier_abstention_rerun_rollup_reports_candidate_gate_diagnostics(tmp_path):
    report_path = tmp_path / "abstention-stability.json"
    source = tmp_path / "frontier-release-evidence.json"
    queue_path = tmp_path / "abstention-rerun-queue.json"
    rollup_path = tmp_path / "abstention-rerun-rollup.json"
    report_path.write_text(
        json.dumps(_abstention_stability_payload(tmp_path / "qwen-scores.manifest.json")),
        encoding="utf-8",
    )
    source.write_text(
        json.dumps(_frontier_release_abstention_payload(report_path)),
        encoding="utf-8",
    )
    queue = build_frontier_abstention_evidence_rerun_queue(
        source=source,
        json_path=queue_path,
        output_dir=tmp_path / "abstention-reruns",
        profiles=("baseline",),
        signal_groups=("recommended",),
        seeds="0,1",
        python_executable="python",
    )
    entry = queue["entries"][0]
    child_report = _abstention_rerun_report(entry)
    child_report["runs"][0]["stability"]["candidate_gate_summary"] = {
        "seed_with_any_passing_candidate_count": 2,
        "seed_without_passing_candidate_count": 0,
        "all_seeds_have_passing_candidate": True,
        "recommended_pass_seed_count": 0,
        "recommended_block_seed_count": 2,
        "recommended_missed_passing_candidate_count": 2,
        "recommended_blocking_reason_counts": {"empirical_abstention_rate": 2},
        "candidate_blocking_reason_counts": {
            "conditional_correctness_lower_bound": 4,
        },
        "best_passing_score_name_counts": {"fusion@budget=0.48": 2},
    }
    output_path = _queue_entry_report_path(entry)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(child_report), encoding="utf-8")

    payload = rollup_frontier_abstention_evidence_reruns(
        queue_path=queue_path,
        report_json_path=rollup_path,
    )
    candidate = payload["candidates"][0]
    diagnostics = candidate["metrics"]["candidate_gate_diagnostics"]
    summary = payload["summary"]["candidate_gate_diagnostics"]

    assert candidate["candidate_status"] == "blocked"
    assert diagnostics["seed_with_any_passing_candidate_rate"] == 1.0
    assert diagnostics["recommended_missed_passing_candidate_count"] == 2
    assert diagnostics["best_passing_score_name_counts"] == {"fusion@budget=0.48": 2}
    assert summary["reports_with_candidate_gate_diagnostics_count"] == 1
    assert summary["reports_with_recommended_missed_passing_candidate_count"] == 1
    assert summary["recommended_missed_passing_candidate_seed_count"] == 2
    assert summary["seed_without_passing_candidate_count"] == 0
    assert summary["best_passing_score_name_counts"] == {"fusion@budget=0.48": 2}


def test_frontier_abstention_evidence_rerun_rollup_blocks_missing_required_report(tmp_path):
    report_path = tmp_path / "abstention-stability.json"
    source = tmp_path / "frontier-release-evidence.json"
    queue_path = tmp_path / "abstention-rerun-queue.json"
    rollup_path = tmp_path / "abstention-rerun-rollup.json"
    report_path.write_text(
        json.dumps(_abstention_stability_payload(tmp_path / "qwen-scores.manifest.json")),
        encoding="utf-8",
    )
    source.write_text(
        json.dumps(_frontier_release_abstention_payload(report_path)),
        encoding="utf-8",
    )
    build_frontier_abstention_evidence_rerun_queue(
        source=source,
        json_path=queue_path,
        output_dir=tmp_path / "abstention-reruns",
        profiles=("baseline",),
        signal_groups=("recommended",),
        python_executable="python",
    )

    payload = rollup_frontier_abstention_evidence_reruns(
        queue_path=queue_path,
        report_json_path=rollup_path,
        require_all_reports=True,
    )

    assert payload["status"] == "blocked"
    assert payload["gate"]["passed"] is False
    assert payload["summary"]["candidate_count"] == 1
    assert payload["summary"]["missing_report_count"] == 1
    assert payload["gate"]["blocking_reasons"][0]["gate"] == "report_coverage"


def test_plan_release_evidence_gaps_can_emit_abstention_rerun_queue(tmp_path):
    report_path = tmp_path / "abstention-stability.json"
    source = tmp_path / "frontier-release-evidence.json"
    output = tmp_path / "evidence-gap-plan.json"
    queue_path = tmp_path / "abstention-rerun-queue.json"
    manifest_path = tmp_path / "abstention-rerun-queue-manifest.json"
    registry_path = tmp_path / "registry.json"
    report_path.write_text(
        json.dumps(_abstention_stability_payload(tmp_path / "qwen-scores.manifest.json")),
        encoding="utf-8",
    )
    source.write_text(
        json.dumps(_frontier_release_abstention_payload(report_path)),
        encoding="utf-8",
    )

    payload = build_release_evidence_gap_plan(
        source=source,
        json_path=output,
        registry_path=registry_path,
        name="frontier-gap-plan",
        version="0.1",
        abstention_rerun_json_path=queue_path,
        abstention_rerun_artifact_manifest_path=manifest_path,
        abstention_rerun_output_dir=tmp_path / "abstention-reruns",
        abstention_rerun_name="frontier-abstention-reruns",
        abstention_rerun_version="0.1",
        abstention_profiles=("baseline", "retention"),
        abstention_signal_groups=("recommended",),
        abstention_seeds="0,1",
        python_executable="python",
    )

    saved = json.loads(output.read_text(encoding="utf-8"))
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    registry = ArtifactRegistry.load_json(registry_path)
    gap_record = registry.get("evidence_gap_plan:frontier-gap-plan:0.1")
    queue_record = registry.get("report:frontier-abstention-reruns:0.1")
    derived = payload["derived_artifacts"]["frontier_abstention_evidence_rerun_queue"]

    assert saved["derived_artifacts"] == payload["derived_artifacts"]
    assert derived["path"] == str(queue_path)
    assert derived["artifact_manifest"] == str(manifest_path)
    assert derived["status"] == "ready"
    assert derived["blocked_run_count"] == 1
    assert derived["entry_count"] == 2
    assert derived["command_count"] == 2
    assert queue["entries"][0]["command_status"] == "ready"
    assert queue["entries"][0]["command_kind"] == "abstention_stability_experiment"
    assert payload["gaps"][0]["metadata"]["abstention_blocked_runs"][0]["run"] == "qwen"
    assert gap_record.metadata["gap_count"] == 1
    assert queue_record.metadata["command_count"] == 2


def test_frontier_detectability_evidence_rerun_queue_builds_blind_spot_audit(tmp_path):
    taxonomy_path = tmp_path / "smol-detectability.json"
    source = tmp_path / "frontier-release-evidence.json"
    queue_path = tmp_path / "detectability-rerun-queue.json"
    manifest_path = tmp_path / "detectability-rerun-queue-manifest.json"
    registry_path = tmp_path / "registry.json"
    taxonomy_path.write_text(
        json.dumps(_detectability_taxonomy_payload(tmp_path / "smol-scores.manifest.json")),
        encoding="utf-8",
    )
    source.write_text(
        json.dumps(_frontier_release_detectability_payload(taxonomy_path)),
        encoding="utf-8",
    )

    payload = build_frontier_detectability_evidence_rerun_queue(
        source=source,
        json_path=queue_path,
        artifact_manifest_path=manifest_path,
        registry_path=registry_path,
        name="frontier-detectability-reruns",
        version="0.1",
        output_dir=tmp_path / "detectability-reruns",
        python_executable="python",
    )

    saved = json.loads(queue_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    registry = ArtifactRegistry.load_json(registry_path)
    record = registry.get("report:frontier-detectability-reruns:0.1")
    entry = payload["entries"][0]
    command = entry["command"]

    assert saved["summary"] == payload["summary"]
    assert payload["workflow"] == "frontier_detectability_evidence_rerun_queue"
    assert payload["summary"]["blocked_run_count"] == 1
    assert payload["summary"]["blind_spot_analysis_count"] == 1
    assert payload["summary"]["command_count"] == 1
    assert entry["run"] == "smol"
    assert entry["command_kind"] == "blind_spot_analysis"
    assert entry["command_status"] == "ready"
    assert command[:2] == ("python", "benchmarks/analyze_detectability_blind_spots.py")
    assert command[command.index("--taxonomy-report") + 1] == str(taxonomy_path)
    assert command[command.index("--scores") + 1] == str(tmp_path / "smol-scores.manifest.json")
    assert command[command.index("--cell") + 1] == "entrenched"
    assert manifest["artifacts"]["frontier_detectability_evidence_rerun_queue"]["exists"] is True
    assert record.metadata["workflow"] == "frontier_detectability_evidence_rerun_queue"
    assert record.metadata["blocked_run_count"] == 1


def test_frontier_detectability_evidence_rerun_queue_can_append_taxonomy_reruns(tmp_path):
    taxonomy_path = tmp_path / "smol-detectability.json"
    score_path = tmp_path / "smol-scores.manifest.json"
    source = tmp_path / "frontier-release-evidence.json"
    queue_path = tmp_path / "detectability-rerun-queue.json"
    taxonomy_path.write_text(
        json.dumps(_detectability_taxonomy_payload(score_path)),
        encoding="utf-8",
    )
    source.write_text(
        json.dumps(_frontier_release_detectability_payload(taxonomy_path)),
        encoding="utf-8",
    )

    payload = build_frontier_detectability_evidence_rerun_queue(
        source=source,
        json_path=queue_path,
        output_dir=tmp_path / "detectability-reruns",
        score_paths=(f"smol={score_path}",),
        include_taxonomy_reruns=True,
        taxonomy_signal_pairs=("disp_hse:nll_answer",),
        python_executable="python",
    )

    blind_spot, taxonomy = payload["entries"]
    command = taxonomy["command"]

    assert payload["summary"]["blocked_run_count"] == 1
    assert payload["summary"]["entry_count"] == 2
    assert payload["summary"]["blind_spot_analysis_count"] == 1
    assert payload["summary"]["taxonomy_rerun_count"] == 1
    assert blind_spot["command_kind"] == "blind_spot_analysis"
    assert taxonomy["command_kind"] == "taxonomy_report"
    assert taxonomy["taxonomy_config"] == {
        "consistency_signal": "disp_hse",
        "confidence_signal": "nll_answer",
        "consistency_direction": "lower",
        "confidence_direction": "lower",
    }
    assert command[command.index("--consistency-signal") + 1] == "disp_hse"
    assert command[command.index("--confidence-signal") + 1] == "nll_answer"
    metadata_values = [
        command[index + 1]
        for index, value in enumerate(command)
        if value == "--metadata"
    ]
    assert "run_name=smol" in metadata_values
    assert "taxonomy_pair=disp_hse-nll_answer-lower-lower" in metadata_values


def test_frontier_detectability_evidence_rerun_rollup_completes_blind_spot_audit(tmp_path):
    taxonomy_path = tmp_path / "smol-detectability.json"
    source = tmp_path / "frontier-release-evidence.json"
    queue_path = tmp_path / "detectability-rerun-queue.json"
    rollup_path = tmp_path / "detectability-rerun-rollup.json"
    manifest_path = tmp_path / "detectability-rerun-rollup-manifest.json"
    registry_path = tmp_path / "registry.json"
    taxonomy_path.write_text(
        json.dumps(_detectability_taxonomy_payload(tmp_path / "smol-scores.manifest.json")),
        encoding="utf-8",
    )
    source.write_text(
        json.dumps(_frontier_release_detectability_payload(taxonomy_path)),
        encoding="utf-8",
    )
    queue = build_frontier_detectability_evidence_rerun_queue(
        source=source,
        json_path=queue_path,
        output_dir=tmp_path / "detectability-reruns",
        python_executable="python",
    )
    entry = queue["entries"][0]
    output_path = _queue_entry_report_path(entry)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_detectability_blind_spot_report(entry, selected_record_count=3)),
        encoding="utf-8",
    )

    payload = rollup_frontier_detectability_evidence_reruns(
        queue_path=queue_path,
        report_json_path=rollup_path,
        artifact_manifest_path=manifest_path,
        registry_path=registry_path,
        name="frontier-detectability-rerun-rollup",
        version="0.1",
        require_all_reports=True,
        metadata={"suite": "unit"},
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = ArtifactRegistry.load_json(registry_path).get(
        "report:frontier-detectability-rerun-rollup:0.1"
    )

    assert payload["workflow"] == "frontier_detectability_evidence_rerun_rollup"
    assert payload["status"] == "complete"
    assert payload["gate"]["passed"] is True
    assert payload["gate"]["audit_ready"] is True
    assert payload["gate"]["promotion_ready"] is False
    assert payload["summary"]["blind_spot_analysis_count"] == 1
    assert payload["summary"]["audit_ready_count"] == 1
    assert payload["recommended_candidate"]["metrics"]["selected_record_count"] == 3
    assert manifest["artifacts"]["frontier_detectability_evidence_rerun_rollup"]["exists"] is True
    assert record.metadata["workflow"] == "frontier_detectability_evidence_rerun_rollup"
    assert record.metadata["audit_ready"] is True
    assert record.metadata["promotion_ready"] is False
    assert record.metadata["suite"] == "unit"


def test_frontier_detectability_evidence_rerun_rollup_promotes_taxonomy_rerun(tmp_path):
    queue_path = tmp_path / "detectability-rerun-queue.json"
    taxonomy_report_path = tmp_path / "detectability-reruns/smol/detectability-taxonomy-report.json"
    rollup_path = tmp_path / "detectability-rerun-rollup.json"
    queue_path.write_text(
        json.dumps({
            "schema_version": 1,
            "workflow": "frontier_detectability_evidence_rerun_queue",
            "status": "ready",
            "entries": (
                {
                    "run": "smol",
                    "command_kind": "taxonomy_report",
                    "source_report": None,
                    "source_score_dump": str(tmp_path / "smol-scores.manifest.json"),
                    "taxonomy_config": {
                        "consistency_signal": "disp_hse",
                        "confidence_signal": "nll_answer",
                        "consistency_direction": "lower",
                        "confidence_direction": "lower",
                    },
                    "command_status": "ready",
                    "missing_inputs": (),
                    "command": (
                        "python",
                        "benchmarks/eval_detectability_taxonomy.py",
                        "--scores",
                        str(tmp_path / "smol-scores.manifest.json"),
                        "--json",
                        str(taxonomy_report_path),
                    ),
                },
            ),
        }),
        encoding="utf-8",
    )
    taxonomy_report_path.parent.mkdir(parents=True, exist_ok=True)
    taxonomy_report_path.write_text(
        json.dumps(_detectability_taxonomy_rerun_report(entrenched_false_rate=0.1)),
        encoding="utf-8",
    )

    payload = rollup_frontier_detectability_evidence_reruns(
        queue_path=queue_path,
        report_json_path=rollup_path,
        max_entrenched_false_rate=0.25,
        require_all_reports=True,
    )

    assert payload["status"] == "promote"
    assert payload["gate"]["passed"] is True
    assert payload["gate"]["promotion_ready"] is True
    assert payload["summary"]["taxonomy_candidate_count"] == 1
    assert payload["summary"]["promotion_ready_count"] == 1
    assert payload["recommended_candidate"]["run"] == "smol"
    assert payload["recommended_candidate"]["metrics"]["entrenched_false_rate"] == 0.1
    assert payload["recommended_candidate"]["taxonomy_config"]["consistency_signal"] == "disp_hse"


def test_frontier_detectability_rollup_promotes_when_one_taxonomy_candidate_passes(tmp_path):
    queue_path = tmp_path / "detectability-rerun-queue.json"
    blocked_report_path = tmp_path / "detectability-reruns/smol/base/detectability-taxonomy-report.json"
    passing_report_path = tmp_path / "detectability-reruns/smol/disp/detectability-taxonomy-report.json"
    rollup_path = tmp_path / "detectability-rerun-rollup.json"
    entries = []
    for signal, report_path in (
        ("eigenscore", blocked_report_path),
        ("disp_hse", passing_report_path),
    ):
        entries.append({
            "run": "smol",
            "command_kind": "taxonomy_report",
            "source_report": None,
            "source_score_dump": str(tmp_path / "smol-scores.manifest.json"),
            "taxonomy_config": {
                "consistency_signal": signal,
                "confidence_signal": "nll_answer",
                "consistency_direction": "lower",
                "confidence_direction": "lower",
            },
            "command_status": "ready",
            "missing_inputs": (),
            "command": (
                "python",
                "benchmarks/eval_detectability_taxonomy.py",
                "--json",
                str(report_path),
            ),
        })
    queue_path.write_text(
        json.dumps({
            "schema_version": 1,
            "workflow": "frontier_detectability_evidence_rerun_queue",
            "status": "ready",
            "entries": entries,
        }),
        encoding="utf-8",
    )
    blocked_report_path.parent.mkdir(parents=True, exist_ok=True)
    passing_report_path.parent.mkdir(parents=True, exist_ok=True)
    blocked_report_path.write_text(
        json.dumps(_detectability_taxonomy_rerun_report(entrenched_false_rate=0.4)),
        encoding="utf-8",
    )
    passing_report_path.write_text(
        json.dumps(_detectability_taxonomy_rerun_report(entrenched_false_rate=0.1)),
        encoding="utf-8",
    )

    payload = rollup_frontier_detectability_evidence_reruns(
        queue_path=queue_path,
        report_json_path=rollup_path,
        max_entrenched_false_rate=0.25,
        require_all_reports=True,
    )

    assert payload["status"] == "promote"
    assert payload["gate"]["passed"] is True
    assert payload["gate"]["promotion_ready"] is True
    assert payload["gate"]["blocking_reasons"] == ()
    assert payload["summary"]["blocked_candidate_count"] == 1
    assert payload["summary"]["promotion_ready_count"] == 1
    assert payload["recommended_candidate"]["taxonomy_config"]["consistency_signal"] == "disp_hse"


def test_plan_release_evidence_gaps_can_emit_detectability_rerun_queue(tmp_path):
    taxonomy_path = tmp_path / "smol-detectability.json"
    source = tmp_path / "frontier-release-evidence.json"
    output = tmp_path / "evidence-gap-plan.json"
    queue_path = tmp_path / "detectability-rerun-queue.json"
    manifest_path = tmp_path / "detectability-rerun-queue-manifest.json"
    registry_path = tmp_path / "registry.json"
    taxonomy_path.write_text(
        json.dumps(_detectability_taxonomy_payload(tmp_path / "smol-scores.manifest.json")),
        encoding="utf-8",
    )
    source.write_text(
        json.dumps(_frontier_release_detectability_payload(taxonomy_path)),
        encoding="utf-8",
    )

    payload = build_release_evidence_gap_plan(
        source=source,
        json_path=output,
        registry_path=registry_path,
        name="frontier-gap-plan",
        version="0.1",
        detectability_rerun_json_path=queue_path,
        detectability_rerun_artifact_manifest_path=manifest_path,
        detectability_rerun_output_dir=tmp_path / "detectability-reruns",
        detectability_rerun_name="frontier-detectability-reruns",
        detectability_rerun_version="0.1",
        python_executable="python",
    )

    saved = json.loads(output.read_text(encoding="utf-8"))
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    registry = ArtifactRegistry.load_json(registry_path)
    gap_record = registry.get("evidence_gap_plan:frontier-gap-plan:0.1")
    queue_record = registry.get("report:frontier-detectability-reruns:0.1")
    derived = payload["derived_artifacts"]["frontier_detectability_evidence_rerun_queue"]

    assert saved["derived_artifacts"] == payload["derived_artifacts"]
    assert derived["path"] == str(queue_path)
    assert derived["artifact_manifest"] == str(manifest_path)
    assert derived["status"] == "ready"
    assert derived["blocked_run_count"] == 1
    assert derived["command_count"] == 1
    assert queue["entries"][0]["command_status"] == "ready"
    assert queue["entries"][0]["command_kind"] == "blind_spot_analysis"
    assert gap_record.metadata["gap_count"] == 1
    assert queue_record.metadata["command_count"] == 1


def _blocked_registry_workflow_payload():
    return {
        "workflow": "release_candidate_registry_workflow",
        "release_candidate_comparison": {
            "workflow": "release_candidate_comparison",
            "decision": {
                "status": "blocked",
                "blocking_reasons": [
                    {
                        "gate": "readiness_baseline",
                        "status": "blocked",
                        "reasons": (
                            "benchmark_manifest:smollm2:0.8: best quality AUROC below 0.7",
                        ),
                    },
                    {
                        "gate": "performance_baseline",
                        "status": "blocked",
                        "reasons": (
                            "release candidate is unavailable for performance baseline comparison",
                        ),
                    },
                    {
                        "gate": "product_runtime_drift",
                        "status": "blocked",
                        "reasons": (
                            "product runtime drift pre-generation evidence metrics are incomplete: "
                            "promotion_contract.pre_generation_probe_comparison.coverage_rate, "
                            "promotion_contract.pre_generation_probe_comparison.manifest_verified_rate, "
                            "promotion_contract.pre_generation_probe_comparison.model_count.mean, "
                            "promotion_contract.pre_generation_probe_comparison.run_count.mean, "
                            "promotion_contract.pre_generation_probe_comparison.redline_pass_rate, "
                            "promotion_contract.pre_generation_probe_comparison.best_test_label_auroc.mean, "
                            "promotion_contract.pre_generation_probe_comparison.best_redline_auroc.mean, "
                            "promotion_contract.pre_generation_probe_comparison.best_redline_margin.mean",
                            "product runtime drift action-gate evidence metrics are incomplete: "
                            "promotion_contract.product_trace_replay.action_audit_gate.error_rate.mean, "
                            "promotion_contract.product_trace_replay.action_audit_gate."
                            "missing_retrieval_action_rate.mean, "
                            "promotion_contract.product_trace_replay.action_execution_gate."
                            "alignment_failed_trace_rate.mean, "
                            "promotion_contract.product_trace_replay.action_execution_gate.missing_result_rate.mean, "
                            "promotion_contract.product_trace_replay.action_execution_gate."
                            "unexpected_result_rate.mean, "
                            "promotion_contract.product_trace_replay.action_execution_gate.request_id_mismatch_rate.mean",
                        ),
                    },
                ],
            },
        },
    }


def _frontier_workflow_payload_for_multiple_testing_queue():
    return {
        "schema_version": 1,
        "workflow": "truthfulqa_frontier_workflow",
        "status": "complete",
        "config": {
            "models": ({"name": "a", "model_id": "synthetic-a"},),
            "scales": (
                {
                    "name": "l2",
                    "limit": 2,
                    "manifold_questions": 2,
                    "layer": -1,
                    "sweep_layers": (-1, -2),
                },
            ),
            "dtype": "float32",
            "batch_size": 2,
            "max_batch_tokens": 0,
            "max_length": 64,
            "hidden_state_capture": "hooks",
            "covariance_mode": "diag",
            "covariance_low_rank": 4,
            "progress_every": 0,
            "offline": True,
            "signals": ("truth_proj", "subspace_resid"),
            "conformal_signal": "truth_proj",
            "conformal_repeats": 1,
            "ensemble_repeats": 1,
            "artifact_alpha": 0.2,
            "multiple_testing_signals": ("truth_proj", "subspace_resid"),
            "multiple_testing_alpha": 0.2,
            "multiple_testing_method": "bh",
            "best_alpha": 0.2,
            "best_by": "auroc",
            "ensemble_methods": ("max_rank",),
            "alphas": (0.2,),
        },
        "multiple_testing_gate": {
            "enabled": True,
            "all_pass": False,
            "cells": (
                {
                    "cell": "a-l2",
                    "pass": False,
                    "false_alarm": 0.03,
                    "detection": 0.7,
                    "report": "frontier/a-l2/multiple-testing-report.json",
                    "calibration": "frontier/a-l2/multiple-testing-calibration.json",
                },
            ),
        },
    }


def _multiple_testing_queue_entry_report_path(entry):
    command = tuple(entry["command"])
    output_dir = Path(command[command.index("--output-dir") + 1])
    return output_dir / "truthfulqa-frontier-workflow.json"


def _frontier_workflow_multiple_testing_child_report(*, cell, passed):
    return {
        "schema_version": 1,
        "workflow": "truthfulqa_frontier_workflow",
        "status": "complete",
        "multiple_testing_gate": {
            "enabled": True,
            "signals": ("truth_proj", "subspace_resid"),
            "alpha": 0.2,
            "method": "bh",
            "cell_count": 1,
            "pass_count": 1 if passed else 0,
            "fail_count": 0 if passed else 1,
            "unknown_count": 0,
            "all_pass": passed,
            "cells": (
                {
                    "cell": cell,
                    "pass": passed,
                    "false_alarm": 0.02 if passed else 0.08,
                    "detection": 0.76,
                    "report": f"{cell}/multiple-testing-report.json",
                    "calibration": f"{cell}/multiple-testing-calibration.json",
                },
            ),
        },
    }


def _frontier_release_citation_batch_payload():
    return {
        "schema_version": 1,
        "workflow": "frontier_release_evidence_comparison",
        "status": "complete",
        "decision": {
            "status": "blocked",
            "citation_batch_track_status": "blocked",
            "blocking_reasons": (
                "citation_batch_rollup.citation-rollup.summary.missing_expected_batch_count 1 is non-zero",
            ),
        },
        "evidence_summary": {
            "citation_batch_rollup_names": ("citation-rollup",),
            "citation_batch_expected_batch_ids": (
                "unresolved-evidence-batch-0001",
                "unresolved-evidence-batch-0002",
            ),
            "citation_batch_observed_batch_ids": (
                "unresolved-evidence-batch-0001",
                "unresolved-evidence-batch-0001",
            ),
            "citation_batch_missing_expected_batches": (
                {
                    "rollup": "citation-rollup",
                    "batch_id": "unresolved-evidence-batch-0002",
                },
            ),
            "citation_batch_duplicate_batches": (
                {
                    "rollup": "citation-rollup",
                    "batch_id": "unresolved-evidence-batch-0001",
                },
            ),
        },
    }


def _frontier_release_stability_payload():
    return {
        "schema_version": 1,
        "workflow": "frontier_release_evidence_comparison",
        "status": "complete",
        "decision": {
            "status": "blocked",
            "verifier_track_status": "blocked",
            "abstention_track_status": "blocked",
            "blocking_reasons": (
                "verifier_stability.qwen.verified_detection_mean 0.1 is below required minimum 0.2",
                "abstention_stability.qwen.conditional_correctness_lower_bound_mean 0.5 is below "
                "required minimum 0.8",
            ),
        },
        "evidence_summary": {
            "run_names": ("qwen",),
            "verifier_signal": "truth_proj",
            "abstention_signals": ("maha_last", "subspace_resid"),
        },
    }


def _stability_queue_entry_report_path(entry):
    command = tuple(entry["command"])
    return Path(command[command.index("--json") + 1])


def _verifier_stability_child_report(*, passed):
    return {
        "schema_version": 1,
        "workflow": "verifier_stability",
        "status": "complete",
        "runs": (
            {
                "name": "qwen",
                "scores_path": "qwen-scores.manifest.json",
                "stability": {
                    "seed_count": 2,
                    "verified_false_alarm": {"mean": 0.01 if passed else 0.08},
                    "verified_detection": {"mean": 0.32 if passed else 0.10},
                    "delta_detection": {"mean": 0.12 if passed else -0.05},
                    "verified_pass_seed_count": 2 if passed else 0,
                    "verified_beats_internal_detection_seed_count": 2 if passed else 0,
                },
            },
        ),
    }


def _abstention_stability_child_report(*, passed):
    return {
        "schema_version": 1,
        "workflow": "abstention_stability",
        "status": "complete",
        "runs": (
            {
                "name": "qwen",
                "scores_path": "qwen-scores.manifest.json",
                "stability": {
                    "seed_count": 2,
                    "conditional_correctness_lower_bound": {"mean": 0.86 if passed else 0.50},
                    "empirical_abstention_rate": {"mean": 0.20 if passed else 0.72},
                    "release_gate_pass_seed_count": 2 if passed else 0,
                    "release_gate_block_seed_count": 0 if passed else 2,
                    "stable_recommended_score_name": "truth_proj",
                    "recommended_score_name_counts": {"truth_proj": 2},
                },
                "supervised_feasibility_frontier": {
                    "target_passed": passed,
                    "best": {
                        "score_name": "truth_proj",
                        "conditional_correctness_lower_bound": 0.90 if passed else 0.50,
                        "empirical_abstention_rate": 0.20 if passed else 0.72,
                    },
                },
            },
        ),
    }


def _frontier_release_abstention_payload(report_path):
    return {
        "schema_version": 1,
        "workflow": "frontier_release_evidence_comparison",
        "status": "complete",
        "inputs": {
            "abstention_stability_report": {
                "path": str(report_path),
                "workflow": "abstention_stability",
                "status": "complete",
            },
        },
        "decision": {
            "status": "blocked",
            "abstention_track_status": "blocked",
            "blocking_reasons": (
                "abstention_stability.qwen.conditional_correctness_lower_bound_mean 0.5 "
                "is below required minimum 0.8",
            ),
        },
        "run_decisions": (
            {
                "name": "qwen",
                "abstention_decision": {
                    "status": "blocked",
                    "name": "qwen",
                    "metrics": {
                        "conditional_correctness_lower_bound_mean": 0.5,
                        "empirical_abstention_rate_mean": 0.18,
                        "release_gate_pass_seed_rate": 0.0,
                        "stable_recommended_score_name": "truth_proj",
                    },
                    "blocking_reasons": (
                        "abstention_stability.qwen.conditional_correctness_lower_bound_mean 0.5 "
                        "is below required minimum 0.8",
                    ),
                },
            },
        ),
    }


def _abstention_stability_payload(score_path):
    return {
        "schema_version": 1,
        "workflow": "abstention_stability",
        "status": "complete",
        "config": {
            "signals": ("maha_last", "truth_proj", "subspace_resid", "nll_answer"),
            "directions": {
                "maha_last": "higher",
                "truth_proj": "higher",
                "subspace_resid": "higher",
                "nll_answer": "higher",
            },
            "alpha": 0.1,
            "best_by": "conditional_correctness_lower_bound",
            "seeds": (0, 1),
            "release_gate": {
                "min_conditional_correctness_lower_bound": 0.8,
                "max_abstention_rate": 0.5,
            },
        },
        "runs": (
            {
                "name": "qwen",
                "scores_path": str(score_path),
                "stability": {
                    "all_release_gates_passed": False,
                    "stable_recommended_score_name": "truth_proj",
                    "recommended_score_name_counts": {"truth_proj": 2},
                    "release_gate_pass_seed_count": 0,
                    "release_gate_block_seed_count": 2,
                    "conditional_correctness_lower_bound": {"mean": 0.5},
                    "empirical_abstention_rate": {"mean": 0.18},
                },
                "supervised_feasibility_frontier": {
                    "target_passed": False,
                    "best": {
                        "score_name": "truth_proj",
                        "conditional_correctness_lower_bound": 0.55,
                    },
                },
            },
            {
                "name": "passed",
                "scores_path": str(score_path),
                "stability": {
                    "all_release_gates_passed": True,
                    "stable_recommended_score_name": "truth_proj",
                    "recommended_score_name_counts": {"truth_proj": 2},
                    "release_gate_pass_seed_count": 2,
                    "release_gate_block_seed_count": 0,
                },
            },
        ),
    }


def _queue_entry_report_path(entry):
    command = tuple(entry["command"])
    return Path(command[command.index("--json") + 1])


def _abstention_rerun_report(entry):
    profile = entry["profile"]
    passed = profile == "selective_accuracy"
    correctness = 0.86 if passed else 0.72
    pass_count = 2 if passed else 0
    derived_signal_config = entry.get("derived_signal_config") or {}
    config_signals = tuple(derived_signal_config.get("base_signals") or entry["signals"])
    fusion_config = {}
    if derived_signal_config.get("geometry_uncertainty"):
        fusion_config = {
            "fusion": {
                "geometry_uncertainty": derived_signal_config["geometry_uncertainty"],
            },
        }
    return {
        "schema_version": 1,
        "workflow": "abstention_stability",
        "status": "complete",
        "config": {
            "signals": config_signals,
            "alpha": entry["profile_config"]["alpha"],
            "best_by": entry["profile_config"]["best_by"],
            "release_gate": {
                "min_conditional_correctness_lower_bound": (
                    entry["profile_config"]["min_conditional_correctness_lower_bound"]
                ),
                "max_abstention_rate": entry["profile_config"]["max_abstention_rate"],
            },
            **fusion_config,
        },
        "runs": (
            {
                "name": entry["run"],
                "stability": {
                    "all_release_gates_passed": passed,
                    "seed_count": 2,
                    "stable_recommended_score_name": entry["signals"][0],
                    "recommended_score_name_counts": {entry["signals"][0]: 2},
                    "release_gate_pass_seed_count": pass_count,
                    "release_gate_block_seed_count": 2 - pass_count,
                    "conditional_correctness_lower_bound": {"mean": correctness},
                    "empirical_abstention_rate": {"mean": 0.2},
                    "empirical_selective_accuracy": {"mean": 0.94 if passed else 0.82},
                    "correct_retention_lower_bound": {"mean": 0.7},
                },
                "supervised_feasibility_frontier": {
                    "target_passed": passed,
                    "best": {
                        "score_name": entry["signals"][0],
                        "conditional_correctness_lower_bound": correctness,
                        "empirical_abstention_rate": 0.2,
                        "empirical_selective_accuracy": 0.94 if passed else 0.82,
                    },
                },
            },
        ),
    }


def _frontier_release_detectability_payload(taxonomy_path):
    return {
        "schema_version": 1,
        "workflow": "frontier_release_evidence_comparison",
        "status": "complete",
        "inputs": {
            "detectability_taxonomy_reports": (
                {
                    "path": str(taxonomy_path),
                    "workflow": "detectability_taxonomy",
                    "status": "complete",
                },
            ),
        },
        "decision": {
            "status": "blocked",
            "detectability_track_status": "blocked",
            "blocking_reasons": (
                "detectability_taxonomy.smol.entrenched_false_rate 0.291 exceeds maximum 0.25",
            ),
        },
        "run_decisions": (
            {
                "name": "smol",
                "detectability_decision": {
                    "status": "blocked",
                    "name": "smol",
                    "blocking_reasons": (
                        "detectability_taxonomy.smol.entrenched_false_rate 0.291 exceeds maximum 0.25",
                    ),
                },
            },
        ),
    }


def _detectability_taxonomy_payload(score_path):
    return {
        "schema_version": 1,
        "workflow": "detectability_taxonomy",
        "status": "complete",
        "source": {
            "score_dump_path": str(score_path),
            "score_dump_summary": {"name": "smol"},
        },
        "config": {
            "consistency_signal": "eigenscore",
            "confidence_signal": "nll_answer",
            "consistency_direction": "lower",
            "confidence_direction": "lower",
        },
        "report": {
            "n_total": 4,
            "n_false": 2,
            "blind_spot": {"n_false": 1},
        },
        "metadata": {"run_name": "smol"},
    }


def _detectability_blind_spot_report(entry, *, selected_record_count):
    return {
        "schema_version": 1,
        "workflow": "detectability_blind_spot_analysis",
        "status": "complete",
        "source": {
            "taxonomy_report_path": entry["source_report"],
            "score_dump_path": entry["source_score_dump"],
        },
        "config": {
            "cell": "entrenched",
            "false_only": True,
            "max_records": 100,
        },
        "summary": {
            "cell": "entrenched",
            "false_only": True,
            "selected_record_count": selected_record_count,
            "emitted_record_count": selected_record_count,
            "expected_selected_record_count": selected_record_count,
            "assignment_check_passed": True,
            "truncated": False,
            "question_type_counts": {"definition": selected_record_count},
            "feature_counts": {"numeric": 1},
            "cell_false_counts": {"entrenched": selected_record_count},
        },
        "records": (),
    }


def _detectability_taxonomy_rerun_report(*, entrenched_false_rate):
    entrenched_false_count = int(round(entrenched_false_rate * 10))
    return {
        "schema_version": 1,
        "workflow": "detectability_taxonomy",
        "status": "complete",
        "source": {
            "score_dump_summary": {"name": "smol"},
        },
        "config": {
            "consistency_signal": "eigenscore",
            "confidence_signal": "nll_answer",
            "consistency_direction": "lower",
            "confidence_direction": "lower",
        },
        "report": {
            "n_total": 20,
            "n_false": 10,
            "cells": {
                "entrenched": {
                    "n_false": entrenched_false_count,
                    "share_of_false": {"estimate": entrenched_false_rate},
                },
            },
            "false_distribution": {
                "entrenched": {
                    "count": entrenched_false_count,
                    "rate": entrenched_false_rate,
                },
            },
            "blind_spot": {"n_false": entrenched_false_count},
        },
        "metadata": {"run_name": "smol"},
    }

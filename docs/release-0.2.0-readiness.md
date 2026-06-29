# EigenTruth 0.2.0 Readiness Checklist

Date: 2026-06-29

This checklist records the current working-tree boundary before publishing the
0.2 calibrated-observability/frontier-toolkit work. It is intentionally focused
on submit safety: what changed, which untracked files must be included, and
which validation already passed.

## Validation Snapshot

- `make release-check` passed on 2026-06-29.
  - `ruff check .` passed.
  - `pytest tests/ -v` reported `1214 passed`.
  - `pip check` reported no broken requirements.
  - deterministic smoke workflows completed.
  - package build produced `dist/eigentruth-0.2.0.tar.gz` and
    `dist/eigentruth-0.2.0-py3-none-any.whl`.
- `make check-fast` passed after the release-note/readiness documentation pass.
- `git diff --check` passed after the release-note update.

Generated `dist/` and `src/eigentruth.egg-info/` outputs are ignored and should
not be committed.

## Must-Include Untracked Files

These files are part of the current feature set and must be added if this work
is staged or committed:

- `src/eigentruth/calibration/multiple_testing.py`
- `src/eigentruth/calibration/sequential.py`
- `benchmarks/plan_frontier_multiple_testing_reruns.py`
- `benchmarks/rollup_citation_search_batch_evidence.py`
- `benchmarks/run_citation_batch_rollup_worker_sweep.py`

The two calibration modules are imported from
`src/eigentruth/calibration/__init__.py`. Omitting either module will make the
public calibration API incomplete and can break package import/release checks.

## Suggested Commit Slices

### 1. Calibration and conformal evaluation

Core files:

- `src/eigentruth/eval/conformal.py`
- `src/eigentruth/eval/__init__.py`
- `src/eigentruth/calibration/__init__.py`
- `src/eigentruth/calibration/multiple_testing.py`
- `src/eigentruth/calibration/sequential.py`
- `benchmarks/eval_conformal.py`

Test/doc coverage:

- `tests/test_conformal.py`
- `tests/test_calibration.py`
- relevant README and benchmark README sections.

### 2. Control plane, traces, and world-model traceability

Core files:

- `src/eigentruth/control/controller.py`
- `src/eigentruth/control/trace.py`
- `src/eigentruth/control/runtime_budget.py`
- `src/eigentruth/control/runtime_drift_keys.py`
- `src/eigentruth/control/promotion.py`
- `src/eigentruth/control/evidence_gaps.py`
- `src/eigentruth/control/evidence_handoff.py`
- `src/eigentruth/control/__init__.py`
- `src/eigentruth/adapters/world_model.py`
- `examples/calibrated_control_demo.py`

Test coverage:

- `tests/test_control_loop.py`
- `tests/test_trace_registry.py`
- `tests/test_examples.py`
- `tests/test_frontier_toolkit.py`
- `tests/test_evidence_gaps.py`
- `tests/test_evidence_handoff.py`
- `tests/test_structure_contracts.py`

### 3. Frontier workflows and release evidence

Core files:

- `benchmarks/compare_frontier_release_evidence.py`
- `benchmarks/compare_product_runtime_baselines.py`
- `benchmarks/compare_release_candidates.py`
- `benchmarks/release_policy_profiles.py`
- `benchmarks/run_calibrated_observability_workflow.py`
- `benchmarks/run_citation_search_evidence_workflow.py`
- `benchmarks/run_external_citation_search_adapter_workflow.py`
- `benchmarks/run_product_runtime_baseline.py`
- `benchmarks/run_product_trace_replay_workflow.py`
- `benchmarks/run_release_candidate_registry_workflow.py`
- `benchmarks/run_source_family_citation_search_workflow.py`
- `benchmarks/run_truthfulqa_frontier_workflow.py`
- `benchmarks/build_citation_search_adapter_handoff.py`
- `benchmarks/build_unresolved_blind_spot_evidence_queue.py`
- `benchmarks/build_verifier_signal_score_dump.py`
- `benchmarks/plan_frontier_multiple_testing_reruns.py`
- `benchmarks/rollup_citation_search_batch_evidence.py`
- `benchmarks/run_citation_batch_rollup_worker_sweep.py`

Test coverage:

- `tests/test_benchmarks.py`

### 4. Documentation and release notes

Files:

- `README.md`
- `benchmarks/README.md`
- `docs/experiment-plan.md`
- `docs/frontier-research-notes.md`
- `docs/product-development-spec.md`
- `docs/release-0.2.0.md`
- `docs/release-0.2.0-readiness.md`

## Release Boundary

This work keeps the 0.2 scope monitor-first and dependency-light:

- no mandatory dependency beyond the existing project core dependencies;
- no production RAG, search, database, network verifier, rewrite LLM, SAE/ReFT,
  or heavyweight world-model integration;
- citation/source-family adapter requests are not evidence until provenance
  audits promote source-backed results;
- world-model routes are traceable control-plane inputs, not truth oracles;
- all gates remain local, reproducible artifact checks unless explicitly wired
  to a concrete external adapter.

## Recommended Next Action

Before pushing, stage the must-include untracked files together with the related
tracked changes, then rerun at least:

```bash
git diff --check
make check-fast
```

Run `make release-check` again if commits are rearranged, packaging metadata is
changed, or any Python source changes after this checklist.

# Workflow Guide

Benchmark and release workflow entry points are being grouped by purpose.

## Layout

- `benchmarks/workflows/calibration/`: score dumps, conformal calibration, layer
  sweeps, calibration artifacts, and verified correction training export.
- `benchmarks/workflows/frontier/`: frontier release evidence, unresolved
  frontier queues, and frontier audit workflows.
- `benchmarks/workflows/product_runtime/`: ProductTrace replay, runtime
  baselines, drift gates, and promotion-contract handoffs.
- `benchmarks/workflows/retrieval/`: retrieval, citation, source-family, and
  structured-QA workflows.
- `benchmarks/workflows/verification/`: counterfactual, triple, state,
  verifier-route, and belief-revision workflows.
- `benchmarks/smokes/`: deterministic no-model smoke checks.
- `benchmarks/lib/`: shared helpers for migrated workflow CLIs.

## Migration Rule

New workflow CLIs should not be added at the `benchmarks/` root. Move related
helpers into `benchmarks/lib/`, place the CLI in the relevant workflow group, and
update tests/docs to call the new path directly.

The first migrated group is release/product smoke checks under
`benchmarks/smokes/`.

## 0.3 Entry Points

- `benchmarks/workflows/verification/belief_revision_eval.py`: text-first
  stubbornness kill-test over a belief-revision JSONL fixture.
- `benchmarks/workflows/calibration/correction_training_export.py`: verified
  `CorrectionBuffer` export to SFT or DPO JSONL for later LoRA/DPO experiments.

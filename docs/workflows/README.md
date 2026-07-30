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
  fixture smoke for belief-revision plumbing. Fixture reports cannot enter the
  kill gate.
- `benchmarks/workflows/verification/build_belief_revision_kill_test.py`:
  builds the 48-example `kill-test-v1` split from tracked Wikidata rows, with
  generation inputs and scoring labels in separate files.
- `benchmarks/workflows/verification/belief_revision_real_model_eval.py`:
  runs all four methods through a pinned Hugging Face model and records model,
  data, prompt, decoding, input, and output fingerprints.
- `benchmarks/workflows/verification/belief_revision_kill_gate.py`: combines
  independent real-model reports and emits one of `CONTINUE_0_3`,
  `PAUSE_PROJECT`, or `INSUFFICIENT_EVIDENCE`.
- `benchmarks/workflows/calibration/correction_training_export.py`: verified
  `CorrectionBuffer` export to SFT or DPO JSONL for later LoRA/DPO experiments.

Build or reproduce the controlled split:

```bash
python benchmarks/workflows/verification/build_belief_revision_kill_test.py
```

The four-arm evaluation report for each model must include `baseline_prompt`,
`self_correction_prompt`, `rag_evidence_only`, and
`eigentruth_revision_loop`. Pin each model to a resolved Hub revision and use
identical decoding parameters. Example runs:

```bash
python benchmarks/workflows/verification/belief_revision_real_model_eval.py \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --model-revision <resolved-hub-commit> \
  --json artifacts/local/belief-revision/qwen-report.json \
  --artifact-manifest artifacts/local/belief-revision/qwen-manifest.json

python benchmarks/workflows/verification/belief_revision_real_model_eval.py \
  --model-id HuggingFaceTB/SmolLM2-135M-Instruct \
  --model-revision <resolved-hub-commit> \
  --json artifacts/local/belief-revision/smollm-report.json \
  --artifact-manifest artifacts/local/belief-revision/smollm-manifest.json
```

Once a Qwen and a non-Qwen model both have complete reports, run:

```bash
python benchmarks/workflows/verification/belief_revision_kill_gate.py \
  artifacts/local/qwen-report.json \
  artifacts/local/smollm-report.json \
  --json artifacts/local/kill-gate.json
```

The default gate continues only when every eligible model lowers stubbornness
by at least 0.10 and improves correction success by at least 0.10 versus
self-correction. It also requires matching dataset, prompt-template, and
decoding fingerprints, a resolved model revision, four complete sets of
generated answers, and explicit label isolation. A fixture-only, too-small, or
provenance-incomplete report is `INSUFFICIENT_EVIDENCE`, not a win.

The first pinned `kill-test-v1` run is tracked at
`artifacts/baselines/belief_revision_text/kill-test-v1/real-model-results/` and
returns `CONTINUE_0_3`. Its interpretation and limitations are documented in
`docs/experiments/belief-revision-kill-test-v1.md`.

# Artifact Policy

EigenTruth keeps release evidence auditable without making local scratch output a
hidden test dependency.

## Tracked Artifacts

- `artifacts/baselines/`: small, stable fixtures and compact reports that unit,
  workflow, or smoke tests may read in a clean checkout.
- `artifacts/release/`: promoted release evidence tied to a manifest, registry
  record, readiness note, or release boundary.
- Existing promoted artifact families outside those directories may remain while
  they are migrated, but new required test inputs should use the two directories
  above.

## Ignored Artifacts

- `artifacts/local/`: local scratch output.
- `artifacts/runtime_evidence/`: heavyweight runtime records, model outputs,
  cache material, and one-off workflow outputs.
- Ignored paths may be workflow inputs, but they must not be required for
  `make check-fast`.

## Manifest Rules

- A manifest used by default tests must reference only tracked baseline or
  release artifacts.
- Missing ignored artifacts should fail closed in artifact/release checks with a
  clear recovery command or regeneration note.
- Compact reports promoted from ignored runtime output should be copied into
  `artifacts/baselines/` or `artifacts/release/` and fingerprinted there.

## Current Baseline Repair

`artifacts/baselines/pre_generation_probe_comparison/` is the compact tracked
replacement for the previous ignored
`artifacts/runtime_evidence/pre-generation-qwen-smollm2-l12-comparison/`
comparison report used by product-promotion handoff smoke checks.

`artifacts/baselines/belief_revision_text/` is the compact tracked seed fixture
for the 0.3 text belief-revision kill-test. It is intentionally small and should
grow through reviewed baseline updates, not through local runtime scratch
outputs.

# EigenTruth Development Summary

Date: 2026-07-04

This document records the current repository state after the long 0.2
calibrated-observability development cycle, branch cleanup, GitHub-facing
documentation refresh, and local scratch cleanup.

## Repository State

- Current branch: `main`
- Baseline commit before this summary:
  `2eab312 docs: refresh github-facing project summary`
- Remote state: `main` is synced with `origin/main`
- Remaining local/remote research branch:
  `codex/qwen05-truthfulqa-results`
- Working tree after cleanup and before adding this summary: clean
- Ignored local state after cleanup: `.venv/` only

The GitHub repository description and topics have been updated to match the
current project positioning: calibrated LLM observability, representation
diagnostics, conformal risk calibration, verifier/control traces, and optional
activation steering.

## Branch Cleanup

Deleted local branches that had already merged into `main`:

- `codex/0-2-wrap-up`
- `codex/calibrated-observability-frontier-gates`
- `codex/eigentruth-0-2`
- `codex/eigentruth-0-4-verification-adapter`

Deleted stale remote/local branches whose patches were already equivalent to
`main`:

- `codex/eigentruth-batched-truthfulqa`
- `codex/eigentruth-frontier-verification`
- `codex/eigentruth-multisample-inside`

Preserved `codex/qwen05-truthfulqa-results` because it is a large divergent
research branch, not a stale duplicate. It should be mined selectively by path
and feature, not merged wholesale.

## Cleanup Performed

Removed standard local generated files:

- `.pytest_cache/`
- `.ruff_cache/`
- `__pycache__/` directories under source, tests, examples, and benchmarks
- `dist/`
- `src/eigentruth.egg-info/`

Removed ignored benchmark/runtime scratch artifacts from `.gitignore`:

- local artifact caches under `artifacts/cache/`
- local runtime scratch under `artifacts/runtime_evidence/`
- ignored artifact fingerprint and JSON cache sidecars
- ignored trace-record caches
- ignored local covariance/cache profile outputs
- ignored raw benchmark score dumps under `benchmarks/scores_*.json`

Tracked release evidence under `artifacts/` was preserved. The cleanup used
`git clean -X` on explicit ignored paths so version-controlled artifacts were
not removed.

Approximate checkout size changed from 3.0G before cleanup to 1.9G after
cleanup.

## Current Product Baseline

The package baseline remains `0.2.0`, an alpha research preview. The main
branch now represents a closed 0.2 calibrated-observability loop:

```text
score dump
  -> layer/score sweep
  -> conformal calibration artifact
  -> risk decision
  -> action request/result
  -> ProductTrace
  -> release/readiness documentation
```

The scoped frontier closure is documented as promoted: unresolved frontier
summary has no open `next_actions`, covered-fact route coverage is complete for
the scoped lane, closure verification passes, and the derived command plan is
empty. Historical citation/search blockers remain visible as negative evidence;
they are not hidden or treated as solved.

## Current Engineering Shape

The project is now organized around five monitor-first layers:

1. Observe: hidden-state geometry, spectra, trajectories, pathway diagnostics,
   uncertainty proxies, and optional intervention evidence.
2. Calibrate: direction-aware conformal thresholds, p-values, abstention gates,
   multiple-testing gates, sequential alpha-spending, sweeps, and artifacts.
3. Control: risk decisions, runtime profiles, action requests/results, executor
   registry, receipts, final answers, feedback, and replayable `ProductTrace`
   payloads.
4. Verify: claim extraction, local deterministic verifiers, retrieval/state/
   world-model adapter shells, provenance checks, citation integrity, and
   structured fact routes.
5. Record: artifact manifests, local registry records, bounded trace summaries,
   release evidence, and explicit negative results.

The core dependency policy is still intact: mandatory package dependency is
`torch`; Hugging Face, datasets, retrieval systems, databases, rewrite LLMs, and
world models remain optional or adapter-level.

## Recent Validation

Before this cleanup, the GitHub-facing documentation refresh was validated with:

- `git diff --check`
- `.venv/bin/python -m ruff check src tests examples benchmarks`
- `pyproject.toml` TOML parse check
- `make check-fast`

`make check-fast` completed with `1622 passed` and `pip check` reported no
broken requirements. The cleanup after that removed only ignored generated
files, not source or tracked artifacts.

## Recommended Next Work

1. Keep `main` as the stable development line.
2. Treat `codex/qwen05-truthfulqa-results` as a research archive. Extract only
   reviewed, bounded pieces with tests.
3. Start 0.3 by reducing product-trace replay friction: smaller deterministic
   fixtures, clearer examples, and a shorter default demo path.
4. Separate large evidence artifacts from casual development output. New
   artifacts should enter the repo only when tied to manifests, registry
   records, or release/readiness docs.
5. Before the next release boundary, run `make release-check` and update the
   readiness snapshot with exact command results.

## Immediate Risks

- `README.md` is accurate but still long. A future pass should split long
  benchmark/evidence details into focused docs and keep the GitHub landing page
  shorter.
- The tracked `artifacts/` tree is still large. This is acceptable for current
  release-evidence review, but future heavy artifacts may need a stricter
  storage policy.
- `qwen05` remains highly divergent. Direct merge is not recommended.

## 2026-07-05 Maintenance Refactor Update

The repository now separates fast unit/workflow validation from release-artifact
checks. `make check-fast` runs lint, non-artifact/non-workflow tests, and
`pip check`; `make test-workflow` runs no-model workflow smoke tests; and
`make test-artifacts` runs tracked baseline/release artifact checks.

The previous product-promotion handoff smoke depended on an ignored
`artifacts/runtime_evidence/` pre-generation comparison report. That compact
input is now tracked under
`artifacts/baselines/pre_generation_probe_comparison/`, and the product handoff
manifests point to the tracked baseline instead of ignored local runtime
scratch. The long README was archived to
`docs/evidence/legacy-readme-2026-07-05.md`; the root README is now a shorter
entry point.

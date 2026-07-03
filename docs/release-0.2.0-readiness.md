# EigenTruth 0.2.0 Readiness Snapshot

Date: 2026-07-03

This snapshot records the post-merge 0.2 boundary. It is intentionally a
readiness note, not a production certification. EigenTruth remains an alpha
research-preview toolkit.

## Branch State

- `main` has been fast-forwarded to `origin/main` at `dcdf270`.
- `dcdf270` is the merge commit for the calibrated-observability frontier gates.
- The previous local feature branch
  `codex/calibrated-observability-frontier-gates` points to `0346437`; its remote
  tracking branch has been removed after merge.
- The current 0.2 documentation wrap-up work is isolated on
  `codex/0-2-wrap-up`.

## Validation Snapshot

The 0.2 frontier-gates branch was validated before merge on 2026-07-03:

- `make check` passed.
- `make release-check` passed.
- `pytest tests/ -v` reported `1622 passed`.
- `pip check` reported no broken requirements.
- deterministic smoke workflows completed.
- package build produced `dist/eigentruth-0.2.0.tar.gz` and
  `dist/eigentruth-0.2.0-py3-none-any.whl`.

After this documentation wrap-up, run at least:

```bash
make check-fast
```

Run `make check` or `make release-check` again if Python source, packaging
metadata, release gates, or checked-in benchmark artifacts change.

Generated `dist/`, `build/`, and `src/eigentruth.egg-info/` outputs remain build
products and should not be committed unless a release process explicitly asks
for them.

## Current Closure Artifacts

The unresolved frontier closure lane is now closed in the checked-in artifacts:

- `artifacts/frontier-release-evidence/unresolved-frontier-evidence-summary-v1/unresolved-frontier-evidence-summary.json`
  has `status=promote`.
- The summary has `next_actions=[]`.
- Semantic-gap covered-fact route coverage is `1.0`.
- Semantic-gap coverage gap is `0`.
- Closure verification status is `pass`.
- `artifacts/frontier-release-evidence/unresolved-frontier-research-command-plan-v1/frontier-research-command-plan.json`
  has `status=empty`.

The citation/search lane still carries blocked query-sweep diagnostics. That is
expected and should remain visible as negative evidence. The closure is scoped to
promoted covered-fact route evidence plus terminal coordination checks; it is
not a broad claim that citation/search retrieval has been solved.

The checked-in `artifacts/` tree is large, about `1.5G` in this checkout. Treat
it as release-evidence state, not casual scratch output. New artifact families
should be added only when they are tied to a manifest, registry record, or
documented benchmark/release boundary.

## Release Boundary

0.2 remains monitor-first and dependency-light:

- mandatory package dependency: `torch`;
- Hugging Face, datasets, examples, retrieval systems, databases, rewrite LLMs,
  and world models stay optional or adapter-level;
- activation steering remains experimental and opt-in;
- citation/source-family adapter requests are not evidence until provenance
  audits promote source-backed results;
- world-model routes are traceable control-plane inputs, not truth oracles;
- release gates are local, reproducible artifact checks unless explicitly wired
  to a concrete external adapter.

## 0.3 Starting Point

The recommended next development phase is control and verification hardening:

- convert more ProductTrace examples into deterministic regression fixtures;
- tighten claim, citation, triple-evidence, and receipt-support audits;
- simplify developer entry points and separate short demos from long qualitative
  examples;
- replicate the strongest diagnostics on additional small and mid-sized models;
- keep network retrieval, external databases, rewrite LLMs, SAE/ReFT probes, and
  heavy world models behind optional adapters until cost and failure modes are
  measured.

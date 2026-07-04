# EigenTruth Architecture Map

This is the short entry point for the current 0.2 codebase. The detailed
product charter remains in [`product-development-spec.md`](product-development-spec.md),
and release evidence remains in [`release-0.2.0.md`](release-0.2.0.md).

EigenTruth is organized as a monitor-first toolkit. The core package observes
LLM representations, calibrates risk signals, routes decisions through local
control policies, and records verifier/action evidence. It is not a truth
oracle or a production safety boundary.

## Package Layers

| Layer | Modules | Boundary |
|---|---|---|
| Representation math | `eigentruth.core` | Tensor math, manifolds, subspaces, spectrum, INSIDE-style scores, trajectory/pathway diagnostics. No model loading, network, or datasets. |
| Model hooks | `eigentruth.intervention`, `eigentruth.models` | PyTorch hooks, monitor lifecycle, optional activation/pathway interventions. Steering remains opt-in. |
| Calibration and eval | `eigentruth.eval`, `eigentruth.calibration` | Direction-aware conformal thresholds, layer/score sweeps, abstention, multiple-testing, fusion, score dumps, and release-gate helpers. |
| Control plane | `eigentruth.control` | Risk decisions, runtime profiles, action requests/results, executor registry, receipts, traces, final answers, feedback, and release handoff metadata. |
| Verification | `eigentruth.verify`, `eigentruth.adapters` | Claim extraction, verifier protocols, local deterministic verifiers, retrieval/state/world-model adapter shells. External systems stay optional. |
| Provenance | `eigentruth.registry` | Artifact manifests, fingerprints, local JSON registry records, and manifest verification. |
| Reproducible workflows | `benchmarks` | CLI workflows, smoke gates, score-dump generation, release comparisons, and frontier evidence queues. |

## Main Runtime Shape

```text
request
  |
  v
pre-generation profile/risk selection
  |
  v
LLM draft generation
  |
  +--> representation diagnostics
  +--> optional sampled/trajectory/pathway diagnostics
  |
  v
claim extraction and verification planning
  |
  v
local verifier, retrieval, state, calculator, or world-model adapters
  |
  v
risk controller
  |
  +--> accept
  +--> retrieve / execute tool
  +--> rewrite handoff
  +--> abstain / clarify
  |
  v
final answer plus ProductTrace
```

The default path is diagnostic and trace-oriented. Any model intervention,
external retrieval, database query, rewrite model, or learned world model must
be configured explicitly behind an adapter or optional extra.

## Current 0.2 Closure State

As of `main` at `da49c21`, the 0.2 calibrated-observability work has a closed
local artifact chain. The frontier gates entered `main` at `dcdf270`; `da49c21`
adds the readiness/documentation wrap-up:

- score dumps, layer/score sweeps, conformal calibration artifacts, risk
  decisions, action requests/results, and product traces are all represented by
  dependency-light public APIs.
- the unresolved frontier evidence summary at
  `artifacts/frontier-release-evidence/unresolved-frontier-evidence-summary-v1/`
  is `promote` with `next_actions=[]`.
- its semantic-gap covered-fact route reports coverage rate `1.0`, coverage gap
  `0`, and closure verification `pass`.
- the derived frontier command plan at
  `artifacts/frontier-release-evidence/unresolved-frontier-research-command-plan-v1/`
  is `empty`, so there is no remaining generated command queue for that closure
  lane.

One historical citation lane still records blocked query-sweep diagnostics.
That is preserved as negative evidence. The closure is scoped to promoted
covered-fact route evidence and terminal coordination checks, not a claim that
the broad citation/search route is solved.

## Development Entry Points

- Start with `README.md` for installation and the main calibrated-observability
  workflow.
- Use `docs/product-development-spec.md` for product boundaries and non-goals.
- Use `docs/release-0.2.0.md` for release evidence and validation history.
- Use `docs/development-summary-2026-07-04.md` for the latest branch cleanup,
  scratch cleanup, and development handoff summary.
- Use `benchmarks/README.md` for reproducible CLI workflows.
- Run `make check-fast` for lint, tests, and dependency consistency.
- Run `make check` before pushing Python/source changes.
- Run `make release-check` before package/release-boundary changes.

## 0.3 Starting Line

The next phase should keep the same boundaries:

- make product traces easier to replay as deterministic fixtures;
- tighten claim, citation, and triple-evidence audits;
- replicate strongest diagnostics across more small and mid-sized models;
- keep network retrieval, databases, rewrite LLMs, and heavy world models behind
  optional adapters until their cost and failure modes are measured.

# EigenTruth 0.2.0 Research Release Notes

Date: 2026-06-25
Last validated: 2026-07-03

## Status

EigenTruth 0.2.0 is an alpha research release. It packages the current
representation-observability toolkit, calibration workflow, local control-plane
trace loop, and the first set of frontier-toolkit experiments. It is not a
production hallucination detector, a truth oracle, or a deployed safety
boundary.

The core mandatory dependency remains `torch`. Hugging Face model loading,
datasets, examples, retrieval systems, databases, rewrite models, and world
models remain optional or adapter-level integrations.

## Release Scope

0.2.0 includes:

- Calibrated observability loop: score dump -> layer/score sweep -> conformal
  calibration artifact -> risk decision -> action request/result -> product
  trace.
- JSON and JSONL benchmark score dumps with provenance, fingerprinting, selected
  column loading, and artifact manifests for release checks.
- Direction-aware conformal thresholds, abstention reports, selective accuracy,
  coverage metrics, confidence intervals, multiple-testing conformal reports,
  sequential conformal alpha-spending sidecars, and release gates.
- Calibration artifacts, adaptive conformal calibration, layer/score sweep
  reports, rank-calibrated score-fusion artifacts, multi-signal conformal
  artifacts, and sequential conformal runtime artifacts.
- Claim extraction metadata, claim verification planning, staged verification
  loops, retrieval shells, structured-state checks, world-model adapter shells,
  feedback records, runtime summaries, and replayable product traces.
- Action executor registry, dry-run execution, timeout wrappers, guarded
  execution policy, idempotency ledgers, and local JSON/SQLite replay support.
- ProductTrace summaries for action audits, action execution alignment,
  trajectory-audit findings, claim-risk localization, route costs, final answers,
  and world-model participation/conflict/low-agreement/trace-gap evidence.
- Frontier release-evidence gates that combine verifier stability, abstention,
  detectability taxonomy, optional multiple-testing evidence, and citation or
  source-family batch rollups without treating adapter requests as evidence
  before provenance-audited source documents exist.
- Unresolved frontier closure coordination that can summarize remaining blind
  spot lanes, lower reviewed command queues, execute scoped covered-fact route
  work, verify terminal closure, and preserve blocked citation/search diagnostics
  as negative evidence instead of silently promoting broad retrieval.
- Training-side representation telemetry with per-layer norms, variance trace,
  spectrum rank diagnostics, and Gaussian 2-Wasserstein/Bures distance to a
  baseline.
- Frontier math primitives: covariance spectrum diagnostics, OAS-style shrinkage,
  Gaussian 2-Wasserstein/Bures manifold distance, TwoNN intrinsic dimension, and
  generation-trajectory convergence metrics.
- Concept registry and multi-probe monitoring for reusable concept artifacts.
- Generic `Representation*` public aliases while preserving the original
  `Truth*` API names.
- HSE remains available for explicit ablations through `track_hse=True`, but it
  is no longer part of the default hook/wrapper runtime path.

## Evidence Summary

| Item | Verdict | Evidence boundary |
|------|---------|-------------------|
| HSE vs Euclidean dispersion | Negative result: HSE did not lift over Euclidean on the recorded gpt2 L-8 run. | Documented as 0.474 vs 0.484 in `benchmarks/results_gpt2_l-8.json`; HSE demoted to opt-in ablation. |
| E0 layer/score sweep | `truth_proj` was the strongest recorded gpt2 TruthfulQA signal. | `truth_proj` 0.723 at L-8 and 0.753 peak at L-6 in `benchmarks/results_gpt2_sweep.json`; model-specific, not universal. |
| E1 conformal calibration | Accepted as finite-sample calibration mechanism. | False-alarm tracks nominal within 1.3 percent across alpha values in `benchmarks/results_conformal_*.json`. |
| E2 covariance spectrum/shrinkage | Accepted for shrinkage as the current robust covariance candidate. | Tiny matrix smoke plus l80 Qwen/SmolLM2 cache-only covariance gates; `diag` rejected, `low_rank_16` mixed. |
| E3 manifold distance | Accepted for coarse locality and drift inspection. | Synthetic metric tests plus cached l80 adjacent-layer locality reports; denser matrices still needed for fine-grained claims. |
| E4 intrinsic dimension | Accepted as coarse layer-band selector, not exact oracle. | Qwen/SmolLM2 l80 ID peaks land in `truth_proj` AUROC top-3; exact best-layer rate is 0.0. |
| E5 training telemetry | Accepted as primitive and tiny fine-tune sanity gate. | Effective-rank telemetry separates clean vs duplicate tiny fine-tune before eval-loss degradation crosses its margin. |
| E6 model-collapse warning | Accepted for rank-based synthetic early warning. | Effective rank decays monotonically and warns before visible quality loss; TwoNN ID support is same-direction but not monotonic. |
| E7 trajectory monitor | Accepted as synthetic monitor primitive. | Synthetic convergence score correlates with quality and NLL proxies; real gpt2/TruthfulQA trajectory replication is still pending. |
| E8 concept registry/multi-probe | Accepted as platform glue. | Synthetic two-concept smoke saves artifacts, records registry metadata, and monitors both concepts on a toy model. |
| E9 cleanup | Accepted as compatible consolidation. | HSE default demotion plus `Representation*` aliases covered by unit tests and docs. |
| Multi-signal conformal gate | Accepted as release-gate primitive, not as a new detector. | `multiple_testing_conformal_report(...)`, `MultipleTestingConformalArtifact`, `RiskController(..., multiple_testing_gate=...)`, and `eval_conformal.py --save-multiple-testing-report` are covered by unit and benchmark smoke tests. |
| Sequential conformal replay | Accepted as session/batch audit primitive. | `sequential_pvalue_monitor(...)`, `SequentialConformalArtifact`, `RiskController(..., sequential_gate=...)`, and demo replay tests cover finite alpha spending; sequence traces are kept separate from timed ProductTrace runtime budgets. |
| World-model traceability | Accepted as observable control-plane evidence. | `StateTransitionVerifier` emits reference/view/conflict metadata; `ProductTrace.world_model_summary()` and verifier-signal dumps preserve nested world-model metadata while ignoring generic prediction metadata. |
| Frontier evidence handoff | Accepted as fail-closed local release boundary. | Release workflows preserve frontier evidence track status, multiple-testing status, and citation/source-family batch rollup counts through registry metadata, ProductPromotionContract, bounded ProductTrace metadata, runtime baselines, and drift gates. |
| Unresolved frontier closure | Accepted as scoped coordination closure, not as broad citation/search success. | The frontier gates merged at `dcdf270`, and the documentation/readiness wrap-up is on `main` at `da49c21`. The `unresolved_frontier_evidence_summary` artifact has `status=promote`, `next_actions=[]`, semantic-gap covered-fact route coverage `1.0`, coverage gap `0`, and closure verification `pass`; the derived frontier command plan is `empty`. Historical citation/search query-sweep blockers remain visible as negative evidence. |

## Known Non-Claims

- EigenTruth does not prove that an answer is true.
- EigenTruth does not eliminate hallucinations.
- Activation steering remains experimental and can change generation without
  improving factuality.
- Thresholds are model-, layer-, dataset-, and domain-specific unless replicated
  evidence says otherwise.
- Synthetic benchmarks are mechanism checks, not production evidence.
- Current retrieval, rewrite, verifier, and world-model routes are local
  interface shells or deterministic adapters unless a concrete integration is
  explicitly configured.
- Current release candidates are local artifact chains, not cross-hardware,
  cross-provider, or production traffic validations.

## Reproduction Commands

Core validation:

```bash
make check
make release-check
```

The 0.2 frontier-gates branch was revalidated before merge on 2026-07-03:
`make check` and `make release-check` passed, `pytest tests/ -v` reported
`1622 passed`, `pip check` reported no broken requirements, deterministic smoke
workflows ran, and package build produced `eigentruth-0.2.0.tar.gz` plus
`eigentruth-0.2.0-py3-none-any.whl`. The branch was then merged into `main` as
`dcdf270`; the follow-up readiness/documentation wrap-up is `da49c21`.

Representative calibrated-observability chain:

```bash
python benchmarks/eval_truthfulqa.py --model gpt2 --layer -8 --sweep \
  --dump-scores benchmarks/scores.manifest.json \
  --dump-scores-format jsonl

python benchmarks/eval_conformal.py --scores benchmarks/scores.manifest.json \
  --json artifacts/gpt2-conformal-report.json \
  --multiple-testing-signals maha_last,truth_proj,subspace_resid,first_token_entropy,inside_eigenscore \
  --save-multiple-testing-report artifacts/gpt2-multiple-testing.json \
  --save-multiple-testing-calibration artifacts/gpt2-multiple-testing-calibration.json \
  --save-sweep-report artifacts/gpt2-sweep-report.json \
  --save-best-calibration artifacts/gpt2-best-calibration.json \
  --artifact-manifest artifacts/gpt2-conformal-manifest.json
```

Representative sequential conformal replay:

```bash
python benchmarks/eval_conformal.py --scores benchmarks/scores.manifest.json \
  --signal maha_last \
  --sequential-alpha 0.1 \
  --sequential-schedule harmonic \
  --save-sequential-report artifacts/gpt2-sequential-conformal.json \
  --save-sequential-calibration artifacts/gpt2-sequential-calibration.json
```

Representative artifact-manifest verification:

```bash
python benchmarks/verify_artifact_manifest.py \
  --manifest artifacts/e8-concept-registry-smoke/artifact-manifest.json
```

## Next Work

- Replicate the strongest diagnostics on additional small and mid-sized models.
- Turn fact-level claim checks into a stricter triple/evidence audit path.
- Combine internal representation diagnostics, semantic-energy confidence, and
  external verification into one release-gated product route.
- Promote external citation/source-family and world-model rule lanes only after
  adapter results become provenance-audited source evidence or promoted
  deterministic rule candidates.
- Run denser manifold-distance matrices before using Bures/Wasserstein distance
  as a fine-grained checkpoint-drift signal.
- Replicate trajectory convergence and concept-monitoring checks on real
  generation traces, not only synthetic mechanism checks.
- Keep network retrieval, databases, rewrite LLMs, SAE/ReFT probes, and heavier
  world models behind optional adapters until their cost and failure modes are
  measured.

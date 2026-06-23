# Qwen 0.5B TruthfulQA Probe Results

Date: 2026-06-22

This is the first real instruction-model smoke run for the current EigenTruth
calibrated-observability stack. It is intentionally small enough to run on the
current CPU-only local machine, so the results should be treated as directional,
not conclusive.

## Run 1: Layer Sweep Without Multi-Sample INSIDE

Command:

```bash
python benchmarks/eval_truthfulqa.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --layer -8 \
  --limit 20 \
  --manifold-questions 40 \
  --sweep \
  --batch-size 4 \
  --dump-scores artifacts/qwen05_truthfulqa_l20_scores.json \
  --json artifacts/qwen05_truthfulqa_l20_results.json
```

Data shape:

- Warmup: 126 true statements, 177 false statements for contrastive direction.
- Eval: 154 candidate answers, 85 false / 69 true.
- Hidden dimension: 896.
- Model: `Qwen/Qwen2.5-0.5B-Instruct`.

Primary-layer AUROC at layer `-8`:

| Signal | AUROC |
|---|---:|
| `truth_proj` | 0.685 |
| `maha_last` | 0.531 |
| `eigenscore` | 0.478 |
| `subspace_resid` | 0.471 |
| `nll_answer` | 0.456 |
| `disp_euclid` | 0.453 |
| `disp_hse` | 0.446 |

Best layer/score from sweep:

- Signal: `truth_proj`
- Layer: `-12`
- AUROC: 0.711
- Conformal alpha: 0.1
- Threshold: 4.510312080383301
- False alarm: 0.087
- Detection: 0.094

Artifact files:

- `artifacts/qwen05_truthfulqa_l20_results.json`
- `artifacts/qwen05_truthfulqa_l20_scores.json`
- `artifacts/qwen05_truthfulqa_l20_sweep_report.json`
- `artifacts/qwen05_truthfulqa_l20_best_calibration.json`
- `artifacts/qwen05_truthfulqa_l20_conformal.json`

## Run 1b: Larger Layer-Band Stability Sweep

Command:

```bash
python benchmarks/eval_truthfulqa.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --layer -12 \
  --limit 80 \
  --manifold-questions 80 \
  --sweep-layers=-16,-14,-12,-10,-8 \
  --batch-size 4 \
  --length-bucketed-batches \
  --layer-stats-cache /tmp/eigentruth-qwen-l80/layer-stats.pt \
  --eval-reps-cache /tmp/eigentruth-qwen-l80/eval-reps-cache \
  --eval-reps-cache-shard-size 128 \
  --profile \
  --dump-scores artifacts/qwen05_truthfulqa_l80_scores.json \
  --json artifacts/qwen05_truthfulqa_l80_results.json
```

The cache paths above were temporary local files and are not committed. The
committed artifacts are the result JSON, score dump, conformal report, sweep
report, and best calibration artifact.

Data shape:

- Warmup: 266 true statements, 338 false statements for contrastive direction.
- Eval: 556 candidate answers, 306 false / 250 true.
- Hidden dimension: 896.
- Model: `Qwen/Qwen2.5-0.5B-Instruct`.

Primary-layer AUROC at layer `-12`:

| Signal | AUROC |
|---|---:|
| `truth_proj` | 0.761 |
| `maha_last` | 0.568 |
| `subspace_resid` | 0.523 |
| `disp_euclid` | 0.516 |
| `eigenscore` | 0.514 |
| `disp_hse` | 0.504 |
| `nll_answer` | 0.392 |

Layer-band AUROC:

| Layer | `truth_proj` | `maha_last` | `subspace_resid` | `eigenscore` |
|---:|---:|---:|---:|---:|
| -16 | 0.716 | 0.588 | 0.555 | 0.515 |
| -14 | 0.752 | 0.551 | 0.504 | 0.515 |
| -12 | 0.761 | 0.568 | 0.523 | 0.514 |
| -10 | 0.764 | 0.544 | 0.478 | 0.516 |
| -8 | 0.749 | 0.530 | 0.422 | 0.521 |

Best layer/score from sweep:

- Signal: `truth_proj`
- Layer: `-10`
- AUROC: 0.764
- Conformal alpha: 0.1
- Threshold: 3.6069278717041016
- False alarm: 0.096
- Detection: 0.242

Split-conformal coverage for primary layer `-12` / `truth_proj`, 50 random
splits:

| Alpha | False alarm | Empirical coverage | Detection | Gate |
|---:|---:|---:|---:|---|
| 0.05 | 0.042 | 0.958 | 0.145 | PASS |
| 0.10 | 0.091 | 0.909 | 0.279 | PASS |
| 0.20 | 0.191 | 0.809 | 0.490 | PASS |

Profile:

| Phase | Seconds |
|---|---:|
| load_data | 6.981 |
| load_model | 7.976 |
| build_layer_stats | 565.231 |
| forced_answer_forward | 507.201 |
| score_postprocess | 2.016 |
| total | 1090.793 |

Operational cache validation:

- Sharded eval reps cache wrote 5 shards with shard size 128.
- A cache-only rerun with `--batch-size 8` skipped `load_model`,
  `build_layer_stats`, and `forced_answer_forward`.
- Cache-only AUROC, sweep, primary scores, labels, and `sweep_scores` matched
  the original run exactly.

Artifact files:

- `artifacts/qwen05_truthfulqa_l80_results.json`
- `artifacts/qwen05_truthfulqa_l80_scores.json`
- `artifacts/qwen05_truthfulqa_l80_sweep_report.json`
- `artifacts/qwen05_truthfulqa_l80_best_calibration.json`
- `artifacts/qwen05_truthfulqa_l80_conformal.json`
- `artifacts/qwen05_l80_demo_trace.json`
- `artifacts/truthfulqa_transfer_truth_proj_report.json`

## Run 2: SmolLM2 135M Instruction-Model Transfer

Command:

```bash
HF_HUB_DISABLE_XET=1 python benchmarks/eval_truthfulqa.py \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --layer -12 \
  --sweep-layers=-16,-14,-12,-10,-8 \
  --manifold-questions 80 \
  --limit 80 \
  --batch-size 8 \
  --length-bucketed-batches \
  --profile \
  --progress-every 50 \
  --layer-stats-cache /tmp/eigentruth-smollm2-l80/layer-stats.pt \
  --warmup-checkpoint /tmp/eigentruth-smollm2-l80/warmup-checkpoint.pt \
  --warmup-checkpoint-every 50 \
  --eval-reps-cache /tmp/eigentruth-smollm2-l80/eval-reps-cache \
  --eval-reps-cache-shard-size 256 \
  --dump-scores artifacts/smollm2_truthfulqa_l80_scores.json \
  --json artifacts/smollm2_truthfulqa_l80_results.json
```

The `HF_HUB_DISABLE_XET=1` environment variable avoided the stalled Hugging Face
Xet download path observed during the first attempt.

Data shape:

- Warmup: 266 true statements, 338 false statements for contrastive direction.
- Eval: 556 candidate answers, 306 false / 250 true.

AUROC:

| Signal | AUROC |
|---|---:|
| `truth_proj` | 0.740 |
| `maha_last` | 0.631 |
| `subspace_resid` | 0.569 |
| `eigenscore` | 0.533 |
| `disp_euclid` | 0.529 |
| `disp_hse` | 0.489 |
| `nll_answer` | 0.398 |

Layer-band `truth_proj` AUROC:

| Layer | AUROC |
|---:|---:|
| `-16` | 0.782 |
| `-14` | 0.755 |
| `-12` | 0.740 |
| `-10` | 0.731 |
| `-8` | 0.739 |

Split-conformal gate for primary `truth_proj` accepts alpha 0.05/0.10/0.20 over
50 random splits: false alarms 0.050/0.095/0.190.

Profile timings:

- `build_layer_stats`: 174.588 seconds
- `forced_answer_forward`: 142.904 seconds
- `score_postprocess`: 1.038 seconds
- total: 333.753 seconds

A cache-only replay from the layer-stats and sharded eval-reps caches reproduced
AUROC, sweep payload, labels, scores, and `sweep_scores` exactly.

Artifact files:

- `artifacts/smollm2_truthfulqa_l80_results.json`
- `artifacts/smollm2_truthfulqa_l80_scores.json`
- `artifacts/smollm2_truthfulqa_l80_sweep_report.json`
- `artifacts/smollm2_truthfulqa_l80_best_calibration.json`
- `artifacts/smollm2_truthfulqa_l80_conformal.json`

## Run 3: Tiny Multi-Sample INSIDE Smoke

Command:

```bash
python benchmarks/eval_truthfulqa.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --layer -12 \
  --limit 2 \
  --manifold-questions 10 \
  --batch-size 4 \
  --inside-samples 2 \
  --inside-batch-size 1 \
  --inside-max-new-tokens 4 \
  --dump-scores artifacts/qwen05_truthfulqa_inside_smoke_scores.json \
  --json artifacts/qwen05_truthfulqa_inside_smoke_results.json
```

Data shape:

- Warmup: 30 true statements, 50 false statements for contrastive direction.
- Eval: 12 candidate answers, 6 false / 6 true.

AUROC:

| Signal | AUROC |
|---|---:|
| `truth_proj` | 1.000 |
| `maha_last` | 0.611 |
| `subspace_resid` | 0.528 |
| `nll_answer` | 0.472 |
| `inside_eigenscore` | 0.389 |
| `disp_euclid` | 0.333 |
| `eigenscore` | 0.278 |
| `disp_hse` | 0.250 |

This run is too small to support a statistical claim. Its value is operational:
it proves the multi-sample INSIDE path runs on a real instruction model and
produces a calibrated-observability-compatible score dump.

## Interpretation

The strongest current signal is still the contrastive truth direction
(`truth_proj`), not token dispersion, HSE, perplexity, or the single-forward
EigenScore proxy. On Qwen 0.5B, the larger l80 run moves the best layer from the
small-run `-12` result to nearby layer `-10`, while the whole `-14..-8` band
stays strong for `truth_proj`. The right conclusion is therefore a stable
mid-to-late layer band, not a single exact layer.

SmolLM2 135M strengthens this beyond a single instruction model. On the same l80
split, `truth_proj` reaches 0.740 AUROC at the primary layer and 0.782 at the
best sweep layer. The whole requested band stays above 0.73. This is now genuine
instruction-model transfer evidence for the detector family, not just Qwen-only
stability.

The l80 run also strengthens the baseline comparison: `truth_proj` reaches
0.761 AUROC at the primary layer and 0.764 at the best sweep layer, while
`nll_answer` is 0.392. In this forced-answer TruthfulQA proxy, representation
geometry is clearly more informative than answer NLL.

The multi-sample INSIDE smoke is operational only. After correcting feature
centering in `internal_eigenscore`, `inside_eigenscore` is above the
single-forward `eigenscore` in this tiny run, but both are below chance and the
sample is far too small for a signal-quality claim.

## Product Trace Demo

`examples/calibrated_control_demo.py` now defaults to the l80 best calibration
artifact when it is present in the repository. It does not load Qwen or rerun the
benchmark. It loads `artifacts/qwen05_truthfulqa_l80_best_calibration.json`,
auto-generates a `truth_proj` diagnostic that crosses the conformal threshold,
combines that diagnostic with deterministic claim verification, and emits a
`ProductTrace`.

Demo command:

```bash
python examples/calibrated_control_demo.py \
  --request-id qwen05-l80-demo \
  --output artifacts/qwen05_l80_demo_trace.json
```

Trace result:

- Artifact: `Qwen/Qwen2.5-0.5B-Instruct`, layer `-10`, score `truth_proj`
- Threshold: 3.6069278717041016
- Demo diagnostic: 3.9676206588745115
- Claim verification: one supported claim, one refuted claim
- Final action: `abstain`
- Risk level: `high`

## Transfer Report

`benchmarks/compare_transfer.py` aggregates existing sweep reports so detector
stability can be inspected without rerunning a model. Current report:

```bash
python benchmarks/compare_transfer.py \
  --report qwen05-l20=artifacts/qwen05_truthfulqa_l20_sweep_report.json \
  --report qwen05-l80=artifacts/qwen05_truthfulqa_l80_sweep_report.json \
  --report smollm2-l80=artifacts/smollm2_truthfulqa_l80_sweep_report.json \
  --report gpt2-base=artifacts/gpt2-0-2-sweep-report.json \
  --score truth_proj \
  --layers=-16,-14,-12,-10,-8 \
  --json artifacts/truthfulqa_transfer_truth_proj_report.json
```

Summary over the requested layer band:

- Qwen l20: mean AUROC 0.692; best 0.711 at layer `-12`; 5/5 layers above 0.6
- Qwen l80: mean AUROC 0.748; best 0.764 at layer `-10`; 5/5 layers above 0.7
- SmolLM2 l80: mean AUROC 0.749; best 0.782 at layer `-16`; 5/5 layers above 0.7
- gpt2 base contrast: mean AUROC 0.654 over the available subset; best 0.721 at layer `-8`

Interpretation: this strengthens the claim that `truth_proj` is a stable detector
family over this TruthfulQA forced-answer proxy across at least two instruction
models.

## Calibration Transfer Report

`benchmarks/eval_calibration_transfer.py` applies saved best calibration
artifacts to saved score dumps without rerunning a model. Current report:

```bash
python benchmarks/eval_calibration_transfer.py \
  --artifact qwen-l80=artifacts/qwen05_truthfulqa_l80_best_calibration.json \
  --artifact smollm2-l80=artifacts/smollm2_truthfulqa_l80_best_calibration.json \
  --scores qwen-l80=artifacts/qwen05_truthfulqa_l80_scores.json \
  --scores smollm2-l80=artifacts/smollm2_truthfulqa_l80_scores.json \
  --json artifacts/truthfulqa_calibration_transfer_report.json
```

Summary:

- Self-application controls false alarms for both best calibration artifacts:
  2/2 pass.
- Cross-application controls false alarms for neither artifact: 0/2 pass.
- Qwen l80 threshold (`truth_proj`, layer `-10`, threshold 3.607) applied to
  SmolLM2 l80 gives false alarm 0.160 at alpha 0.100.
- SmolLM2 l80 threshold (`truth_proj`, layer `-16`, threshold 1.131) applied to
  Qwen l80 gives false alarm 0.452 at alpha 0.100.

Interpretation: detector-family transfer and threshold transfer are different.
The `truth_proj` detector remains useful across the two instruction models, but
native score scale and layer calibration are not portable here. Thresholds should
remain model/layer/domain-specific artifacts unless a stronger transfer study
proves a narrower invariance.

## Score Ensemble Report

`benchmarks/eval_score_ensemble.py` compares single internal signals against
simple rank-normalized ensembles without rerunning a model. Current report:

```bash
python benchmarks/eval_score_ensemble.py \
  --scores qwen-l80=artifacts/qwen05_truthfulqa_l80_scores.json \
  --scores smollm2-l80=artifacts/smollm2_truthfulqa_l80_scores.json \
  --signals truth_proj,maha_last,subspace_resid,eigenscore \
  --methods max_rank,mean_rank \
  --repeats 50 \
  --json artifacts/truthfulqa_score_ensemble_report.json
```

At alpha 0.100:

- Qwen l80 best single signal is `truth_proj`: false alarm 0.091, detection
  0.279. Best simple ensemble is `mean_rank`: false alarm 0.090, detection
  0.235.
- SmolLM2 l80 best single signal is `truth_proj`: false alarm 0.095, detection
  0.229. Best simple ensemble is `mean_rank`: false alarm 0.090, detection
  0.196.

Interpretation: naive internal-score fusion is not a free improvement. For the
current score set, the product should keep `truth_proj` as the primary calibrated
internal diagnostic and use other internal scores as auxiliary diagnostics until
a stronger ensemble method proves incremental detection under the same
false-alarm budget.

## Oracle Verifier Ensemble Upper Bound

Older l80 score dumps were backfilled with deterministic TruthfulQA statement
metadata, then paired with a label-derived oracle evidence fixture. This does
not load a model and does not create real retrieval evidence; it validates the
verifier/control benchmark path under perfect evidence.

```bash
python benchmarks/backfill_truthfulqa_statements.py \
  --scores artifacts/qwen05_truthfulqa_l80_scores.json \
  --output artifacts/qwen05_truthfulqa_l80_scores_with_statements.json \
  --manifold-questions 80 \
  --limit 80 \
  --save-oracle-claims artifacts/truthfulqa_l80_oracle_claims.json

python benchmarks/backfill_truthfulqa_statements.py \
  --scores artifacts/smollm2_truthfulqa_l80_scores.json \
  --output artifacts/smollm2_truthfulqa_l80_scores_with_statements.json \
  --manifold-questions 80 \
  --limit 80

python benchmarks/eval_verifier_ensemble.py \
  --scores qwen-l80=artifacts/qwen05_truthfulqa_l80_scores_with_statements.json \
  --scores smollm2-l80=artifacts/smollm2_truthfulqa_l80_scores_with_statements.json \
  --claims artifacts/truthfulqa_l80_oracle_claims.json \
  --signal truth_proj \
  --alphas 0.05,0.1,0.2 \
  --repeats 50 \
  --seed 0 \
  --json artifacts/truthfulqa_l80_oracle_verifier_ensemble_report.json
```

At alpha 0.100:

- Qwen l80 internal `truth_proj`: false alarm 0.091, detection 0.279. Oracle
  verified policy: false alarm 0.000, detection 1.000.
- SmolLM2 l80 internal `truth_proj`: false alarm 0.095, detection 0.229. Oracle
  verified policy: false alarm 0.000, detection 1.000.
- For both runs, `verification_quality` reports `true_supported_rate=1.000`,
  `false_refuted_rate=1.000`, and `decision_accuracy=1.000`.

Interpretation: the verifier/control layer can in principle recover all label
known false claims and suppress all label known true claims when evidence is
perfect. The next meaningful result must replace oracle labels with real
evidence sources and measure the same false-alarm/detection tradeoff.

## Local Corpus Verifier Baseline

`benchmarks/build_truthfulqa_corpus.py` creates a non-oracle local corpus from
TruthfulQA correct-answer statements. `benchmarks/build_evidence_fixture.py`
then retrieves evidence from that corpus by answer text and writes the same
fixture schema consumed by `eval_verifier_ensemble.py`.

```bash
python benchmarks/build_truthfulqa_corpus.py \
  --manifold-questions 80 \
  --limit 80 \
  --output artifacts/truthfulqa_l80_correct_answer_corpus.json

python benchmarks/build_evidence_fixture.py \
  --scores artifacts/qwen05_truthfulqa_l80_scores_with_statements.json \
  --corpus artifacts/truthfulqa_l80_correct_answer_corpus.json \
  --output artifacts/truthfulqa_l80_local_evidence_claims.json \
  --query-field answer \
  --retriever-min-overlap 0.95 \
  --retrieval-limit 3

python benchmarks/eval_verifier_ensemble.py \
  --scores qwen-l80=artifacts/qwen05_truthfulqa_l80_scores_with_statements.json \
  --scores smollm2-l80=artifacts/smollm2_truthfulqa_l80_scores_with_statements.json \
  --claims artifacts/truthfulqa_l80_local_evidence_claims.json \
  --signal truth_proj \
  --alphas 0.05,0.1,0.2 \
  --repeats 50 \
  --seed 0 \
  --json artifacts/truthfulqa_l80_local_evidence_verifier_ensemble_report.json
```

At alpha 0.100:

- Qwen l80 verified policy: false alarm 0.008, detection 0.274.
- SmolLM2 l80 verified policy: false alarm 0.008, detection 0.219.
- Evidence quality for both runs: true-supported rate 0.908, false-refuted rate
  0.003, false-supported rate 0.042, decision accuracy 0.946.
- Route summary for both runs: `groundedness=287`,
  `retrieval_groundedness=269`.

Interpretation: conservative local retrieval suppresses most false alarms by
supporting true claims, but it rarely refutes false claims. This is the first
reproducible non-oracle baseline; stronger evidence adapters should be judged by
improving false refutation without increasing false support.

## Structured QA Verifier Baseline

`QuestionAnswerVerifier` consumes structured `question`/`answer` facts and
checks them before lexical retrieval. On this TruthfulQA l80 artifact, the
correct-answer corpus covers the same eval questions, so this run should be
read as a database/domain-state adapter upper bound, not as an open-domain
verifier result.

```bash
python benchmarks/eval_verifier_ensemble.py \
  --scores qwen-l80=artifacts/qwen05_truthfulqa_l80_scores_with_statements.json \
  --scores smollm2-l80=artifacts/smollm2_truthfulqa_l80_scores_with_statements.json \
  --qa-corpus artifacts/truthfulqa_l80_correct_answer_corpus.json \
  --signal truth_proj \
  --alphas 0.05,0.1,0.2 \
  --repeats 50 \
  --seed 0 \
  --json artifacts/truthfulqa_l80_structured_qa_verifier_ensemble_report.json
```

At alpha 0.100:

- Qwen l80 structured QA policy: false alarm 0.000, detection 1.000.
- SmolLM2 l80 structured QA policy: false alarm 0.000, detection 1.000.
- Evidence quality for both runs: true-supported rate 1.000, false-refuted rate
  1.000, false-supported rate 0.000, decision accuracy 1.000.
- Route summary for both runs: `structured_qa=556`.

Interpretation: exact external state can dominate weak internal diagnostics
when the adapter has the right key/value facts. The product implication is to
prioritize real structured sources next: database rows, calculators, business
rules, tool outputs, and domain/world-model state.

## Staged Structured QA Control Gate

The staged control-plane variant gates expensive verifier execution behind the
calibrated internal diagnostic. Low-risk, non-sensitive claims skip verification;
claims above the conformal diagnostic threshold still run the structured QA
adapter. This measures whether the product can preserve most verification
quality while avoiding unnecessary tool calls.

```bash
python benchmarks/eval_verifier_ensemble.py \
  --scores qwen-l80=artifacts/qwen05_truthfulqa_l80_scores_with_statements.json \
  --scores smollm2-l80=artifacts/smollm2_truthfulqa_l80_scores_with_statements.json \
  --qa-corpus artifacts/truthfulqa_l80_correct_answer_corpus.json \
  --signal truth_proj \
  --alphas 0.05,0.1,0.2 \
  --repeats 20 \
  --staged-verification \
  --staged-alpha 0.1 \
  --json artifacts/truthfulqa_l80_structured_qa_staged_verifier_ensemble_report.json \
  --compact-json

python benchmarks/compare_verifier_routes.py \
  --report staged=artifacts/truthfulqa_l80_structured_qa_staged_verifier_ensemble_report.json \
  --alpha 0.1 \
  --gate-route structured_qa \
  --gate-min-selected 80 \
  --min-decision-accuracy 0.95 \
  --max-false-supported-rate 0.02 \
  --min-false-refuted-rate 0.90 \
  --max-verified-false-alarm 0.02 \
  --min-verified-detection 0.20 \
  --min-staged-skip-rate 0.75 \
  --max-staged-verified-false-alarm 0.02 \
  --min-staged-verified-detection 0.20 \
  --max-staged-delta-false-alarm 0.0 \
  --min-staged-delta-detection 0.0 \
  --json artifacts/truthfulqa_l80_structured_qa_staged_route_comparison.json \
  --fail-on-gate
```

At alpha 0.100:

| Scope | Verifier calls skipped | Verified false alarm | Verified detection |
|---|---:|---:|---:|
| Qwen l80 | 79.3% | 0.008 | 0.306 |
| SmolLM2 l80 | 82.9% | 0.010 | 0.244 |
| Aggregate | 81.1% | 0.009 | 0.275 |

The route comparison artifact promotes `structured_qa` with
`promotion_score=0.974`, `decision_accuracy=1.000` on the verified route,
aggregate staged skip-rate 0.811, staged verified false alarm 0.009, and staged
verified detection 0.275. Configured staged gates fail closed if these metrics
are missing or regress beyond threshold.

Interpretation: staged verification turns the control plane from a maximal
verification policy into a cost-aware policy. It preserves the conformal false
alarm budget and slightly improves detection over the internal-only gate while
skipping 902 of 1112 structured QA calls in this reproducible l80 comparison.

The staged structured QA route is now registered as a local route baseline:

- Registry: `artifacts/staged-route-registry.json`
- Record: `benchmark_manifest:truthfulqa-l80-structured-qa-staged-route:0.4`
- Manifest: `artifacts/truthfulqa_l80_structured_qa_staged_adapter_promotion_manifest.json`
- Registry workflow: `artifacts/truthfulqa_l80_structured_qa_staged_adapter_promotion_registry_workflow.json`
- Baseline comparison: `artifacts/truthfulqa_l80_structured_qa_staged_route_baseline_comparison.json`

`compare_route_baselines.py` promotes this record with route `structured_qa`,
selected count 210, decision accuracy 1.000, false-supported rate 0.000,
verified false alarm 0.000, verified detection 1.000, p99 route duration
0.000199 seconds, and no blocking reasons. The staged aggregate metrics remain
available in the manifest and registry metadata: skip-rate 0.811, staged
verified false alarm 0.009, and staged verified detection 0.275.

A local release-candidate smoke artifact now pairs that staged route baseline
with a tiny-gpt2 offline readiness/runtime baseline:

- Readiness registry: `artifacts/local-readiness-registry.json`
- Readiness record: `benchmark_manifest:tiny-local-readiness:0.4`
- Release registry: `artifacts/local-release-registry.json`
- Release record: `benchmark_manifest:tiny-local-staged-qa-release-candidate:0.4`
- Release comparison: `artifacts/local_staged_release_candidate_comparison.json`
- Release manifest: `artifacts/local_staged_release_candidate_manifest.json`

The release comparison promotes with model `sshleifer/tiny-gpt2`, layer `-1`,
batch size `4`, best local quality signal `disp_hse` AUROC 0.688, uncached
forced-answer cost 0.061 seconds, cache-only total 0.032 seconds, and route
`structured_qa` from the TruthfulQA l80 staged route baseline. This artifact
proves the end-to-end registry/release gate plumbing; it is not a claim that
tiny-gpt2 is the target production runtime for Qwen results.

A second local smoke candidate adds registered INSIDE sampling evidence to the
same release-gate chain:

- INSIDE profile: `artifacts/tiny_local_inside_sampling/inside-sampling-profile-comparison.json`
- Readiness record: `benchmark_manifest:tiny-local-readiness-inside:0.5`
- Release record: `benchmark_manifest:tiny-local-inside-staged-qa-release-candidate:0.5`
- Release comparison: `artifacts/local_inside_staged_release_candidate_comparison.json`
- Release manifest: `artifacts/local_inside_staged_release_candidate_manifest.json`

The 0.5 comparison promotes with `adaptive_selfcheck` as the INSIDE sampling
run, generated-sample ratio 0.667, and `inside_generation` ratio 0.716 versus
fixed sampling. It is still a tiny offline plumbing artifact; the Qwen-specific
next step is to replace the readiness half with representative Qwen or SmolLM2
same-machine profile artifacts.

That replacement now exists for SmolLM2 at l20 scale:

- INSIDE profile: `artifacts/smollm2_l20_inside_sampling/inside-sampling-profile-comparison.json`
- Readiness record: `benchmark_manifest:smollm2-l20-readiness-inside:0.6`
- Release record: `benchmark_manifest:smollm2-l20-inside-staged-qa-release-candidate:0.6`
- Release comparison: `artifacts/smollm2_l20_inside_staged_release_candidate_comparison.json`
- Release manifest: `artifacts/smollm2_l20_inside_staged_release_candidate_manifest.json`

The 0.6 comparison promotes with model
`HuggingFaceTB/SmolLM2-135M-Instruct`, layer `-12`, batch size `8`,
`truth_proj` AUROC 0.682, uncached forced-answer cost 38.786 seconds,
cache-only replay cost 0.339 seconds, and route `structured_qa` from the
TruthfulQA l80 staged route baseline. Full-sample INSIDE is not yet a good
default: `adaptive_selfcheck` reduces generated samples to 0.937 of fixed
sampling, but `inside_generation` remains 1.001 of fixed. The next runtime
optimization should therefore use `--inside-trigger-signal` with a conformal
threshold or `--inside-trigger-top-fraction`, rather than sampling every
statement.

The triggered replacement is now registered:

- Triggered INSIDE profile: `artifacts/smollm2_l20_inside_trigger_truth_proj_top25/inside-sampling-profile-comparison.json`
- Triggered readiness record: `benchmark_manifest:smollm2-l20-readiness-inside-triggered:0.7`
- Triggered release record: `benchmark_manifest:smollm2-l20-inside-triggered-staged-qa-release-candidate:0.7`
- Triggered release comparison: `artifacts/smollm2_l20_inside_triggered_staged_release_candidate_comparison.json`
- Triggered release manifest: `artifacts/smollm2_l20_inside_triggered_staged_release_candidate_manifest.json`

The 0.7 comparison keeps the same model, layer, batch size, `truth_proj` AUROC,
uncached forced-answer cost, cache-only replay cost, and staged structured QA
route, but changes INSIDE to `--inside-trigger-signal truth_proj
--inside-trigger-top-fraction 0.25`. It samples 39/154 eval statements, skips
115, and reduces fixed `inside_generation` from 467.563 seconds to 118.513
seconds, or 0.253x full-sample fixed. Within the triggered subset the promoted
`adaptive_selfcheck` run uses 110 generated samples, sample ratio 0.940 versus
triggered fixed, and `inside_generation` ratio 1.009, so the release evidence
supports trigger gating as the main runtime optimization rather than adaptive
sampling alone.

A follow-up SmolLM2 trigger-budget sweep compares top-10%, top-25%, and top-40%
budgets in `artifacts/smollm2_l20_inside_trigger_budget_sweep/`. The recursive
manifest verification passes for the sweep and all three child profile manifests.
Against the full-sample fixed reference, `adaptive_selfcheck` at top-10% uses 55
generated samples and 60.917 seconds of `inside_generation` (`0.130x`) with
semantic-entropy AUROC `0.502`; top-25% uses 110 samples and 129.131 seconds
(`0.276x`) with semantic-entropy AUROC `0.557`; top-40% uses 217 samples and
255.041 seconds (`0.545x`) with semantic-entropy AUROC `0.565`. The cost-first
recommendation is top-10%, while the quality-balanced recommendation is top-25%:
top-40% roughly doubles top-25% cost for only a small semantic-entropy gain.

## Next Steps

1. Run `inside_eigenscore` only on the best layer band, not every layer, because
   multi-sample generation is the dominant CPU cost.
2. Pair future long runs with `--warmup-checkpoint` and `--layer-stats-cache` so
   interrupted warmup can resume and completed warmup becomes a compact final cache.
3. Use sharded `--eval-reps-cache` for long forced-answer runs and cache-only
   rescoring; adjacent batch reads reuse the active shard and expose shard IO
   counters in the structured JSON output.
4. Use `--hidden-state-capture hooks` for targeted non-final layer-band runs when
   peak memory is the bottleneck; keep the default output capture for final-layer
   or full hidden-state semantics.
5. Treat top-10% as the cost-first triggered INSIDE default and top-25% as the
   quality-balanced default until larger model/data runs change the Pareto
   frontier; do not default to top-40% on current evidence.
6. Promote a Qwen l20/l80 readiness baseline through the same registry workflow
   if Qwen-specific runtime evidence is needed; SmolLM2 now has the first
   non-tiny registered readiness/release candidate.
7. Replace the label-derived oracle evidence fixture with real retrieval,
   database, calculator, or world-model evidence and rerun
   `benchmarks/eval_verifier_ensemble.py` under the same conformal false-alarm
   budgets. Use `benchmarks/build_evidence_fixture.py` with a local corpus as
   the first reproducible non-oracle baseline before networked retrieval.
8. Use `CalculatorVerifier` for arithmetic claims once extraction or upstream
   tools provide structured `expression` / `expected` metadata; it is a
   deterministic tool adapter, not a broad natural-language math parser.

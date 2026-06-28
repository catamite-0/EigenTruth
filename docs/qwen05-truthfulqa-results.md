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

`examples/calibrated_control_demo.py` now defaults to the best repository l80
calibration artifact when present: SmolLM2 l80 first, Qwen l80 as fallback. It
does not load either model or rerun the benchmark. With the current repository
artifacts it loads `artifacts/smollm2_truthfulqa_l80_best_calibration.json`,
auto-generates a `truth_proj` diagnostic that crosses the conformal threshold,
combines that diagnostic with deterministic claim verification, and emits a
`ProductTrace`. When the SmolLM2 strict structured-retrieval-audit release
candidate is present, the demo also loads its promotion contract as
verifier-route, adapter-family, and required-audit metadata; pass the contract
explicitly with `--promotion-contract` to enforce its runtime budget.

Demo command:

```bash
python examples/calibrated_control_demo.py \
  --request-id smollm2-l80-demo \
  --output artifacts/smollm2_l80_demo_trace.json
```

Trace result:

- Artifact: `HuggingFaceTB/SmolLM2-135M-Instruct`, layer `-16`, score
  `truth_proj`
- Threshold: 1.1314295530319214
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
  --signals truth_proj,maha_last,subspace_resid,resid_update_norm,eigenscore \
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

Geometry-calibrated fusion was replayed on the JSONL l80 frontier score dumps:

```bash
python benchmarks/eval_score_ensemble.py \
  --scores qwen05-l80=artifacts/truthfulqa-frontier-qwen-smollm2-l80/qwen05-l80/scores.manifest.json \
  --scores smollm2-l80=artifacts/truthfulqa-frontier-qwen-smollm2-l80/smollm2-l80/scores.manifest.json \
  --signals truth_proj,maha_last,subspace_resid,resid_update_norm,eigenscore \
  --methods max_rank,mean_rank \
  --geometry-signals subspace_resid,resid_update_norm,eigenscore \
  --uncertainty-signals nll_answer \
  --geometry-fusion-methods interaction,product,weighted_mean,noisy_or \
  --repeats 50 \
  --best-alpha 0.10 \
  --json artifacts/truthfulqa-frontier-qwen-smollm2-l80-geometry-fusion/score-ensemble-report.json
```

At alpha 0.100, `truth_proj` still dominates: Qwen detection is 0.279 for
`truth_proj`, 0.244 for naive `mean_rank`, and 0.055 for the best geometry
fusion; SmolLM2 detection is 0.229 for `truth_proj`, 0.188 for naive
`mean_rank`, and 0.036 for the best geometry fusion. Two stress variants also
failed to close the gap: adding `truth_proj` to the geometry group reached only
0.074 / 0.068 detection, and `max_rank` geometry aggregation reached only
0.083 / 0.069 detection for Qwen / SmolLM2. Current evidence rejects
`nll_answer` as a useful final-correction uncertainty proxy; the next geometry
fusion test should use real multi-sample semantic energy, self-consistency, or
verifier disagreement features.

The first verifier-signal replay uses the staged structured-QA verifier sidecar
as stronger final-correction evidence:

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
  --verified-records-jsonl artifacts/truthfulqa-l80-staged-qa-verifier-signals/verified-records.jsonl \
  --json artifacts/truthfulqa-l80-staged-qa-verifier-signals/verifier-ensemble-report.json \
  --compact-json
```

`benchmarks/build_verifier_signal_score_dump.py` converts that sidecar into
standard score columns including `verifier_refuted`,
`verifier_refute_confidence`, `verifier_not_supported`,
`verifier_uncertainty`, and selfcheck placeholders, then `eval_score_ensemble.py`
uses those columns as geometry-fusion uncertainty signals. At alpha 0.100:

- Qwen: `verifier_refuted` is the strongest single signal with detection 0.297
  and zero false alarm. Geometry fusion reaches detection 0.285 at false alarm
  0.089.
- SmolLM2: geometry fusion reaches detection 0.261 at false alarm 0.095,
  beating both `truth_proj` (0.229) and `verifier_refuted` (0.232).

This is still a structured-QA upper-bound route, not an open-domain claim. The
engineering result is important: verifier/retrieval/selfcheck outputs can now
be converted into calibrated score-dump signals and saved as deployable
`GeometryScoreFusionArtifact` files.

The follow-up local retrieval verifier-signal replay uses
`benchmarks/run_verifier_signal_fusion_workflow.py` to run the same conversion
and fusion path without model reruns:

```bash
python benchmarks/run_verifier_signal_fusion_workflow.py \
  --scores qwen-l80=artifacts/qwen05_truthfulqa_l80_scores_with_statements.json \
  --scores smollm2-l80=artifacts/smollm2_truthfulqa_l80_scores_with_statements.json \
  --corpus artifacts/truthfulqa_l80_correct_answer_corpus.json \
  --output-dir artifacts/truthfulqa-l80-local-retrieval-verifier-signal-fusion \
  --signal truth_proj \
  --alphas 0.05,0.1,0.2 \
  --repeats 20 \
  --best-alpha 0.10 \
  --keep-signals truth_proj,maha_last,subspace_resid,eigenscore,nll_answer \
  --fusion-signals truth_proj,subspace_resid,eigenscore,verifier_refuted,verifier_not_supported,verifier_refute_confidence,verifier_uncertainty \
  --geometry-signals truth_proj,subspace_resid,eigenscore \
  --uncertainty-signals verifier_refuted,verifier_refute_confidence,verifier_not_supported \
  --geometry-fusion-methods interaction,product,weighted_mean,noisy_or \
  --query-field answer \
  --retriever-min-overlap 0.95 \
  --verifier-min-overlap 0.65 \
  --retrieval-limit 3 \
  --omit-label-metadata \
  --compact-json
```

The resulting artifact at
`artifacts/truthfulqa-l80-local-retrieval-verifier-signal-fusion/` verifies its
manifest and uses 410 local retrieval hits over 274/556 records. At alpha 0.100:

- Qwen: verified detection 0.316 at false alarm 0.016; best geometry fusion
  selects `noisy_or` and detects 0.795 at false alarm 0.070.
- SmolLM2: verified detection 0.267 at false alarm 0.016; best geometry fusion
  selects `noisy_or` and detects 0.795 at false alarm 0.069.

This is a local-corpus non-oracle replay because claim metadata omits labels and
retrieval uses answer text against the provided corpus. It is still not an
open-domain retrieval claim; the corpus is the controlled TruthfulQA
correct-answer corpus, so the next frontier step is replacing that corpus with
external or domain-shifted retrieval evidence and adding aligned selfcheck
samples.

## Answer-Echo Retrieval Stress Control

`benchmarks/build_retrieval_stress_corpus.py` builds the retrieval negative
control that the local-corpus result needs: a corpus made from the same answers
being audited. Labels are not used to build documents and are not copied to
document metadata by default.

```bash
OUT=artifacts/truthfulqa-l80-answer-echo-retrieval-stress

python benchmarks/build_retrieval_stress_corpus.py \
  --scores artifacts/qwen05_truthfulqa_l80_scores_with_statements.json \
  --output "$OUT/answer-echo-corpus.json" \
  --document-field answer \
  --corpus-name truthfulqa_l80_answer_echo

python benchmarks/run_verifier_signal_fusion_workflow.py \
  --scores qwen-l80=artifacts/qwen05_truthfulqa_l80_scores_with_statements.json \
  --scores smollm2-l80=artifacts/smollm2_truthfulqa_l80_scores_with_statements.json \
  --corpus "$OUT/answer-echo-corpus.json" \
  --output-dir "$OUT" \
  --signal truth_proj \
  --alphas 0.05,0.1,0.2 \
  --repeats 20 \
  --best-alpha 0.10 \
  --keep-signals truth_proj,maha_last,subspace_resid,eigenscore,nll_answer \
  --fusion-signals truth_proj,subspace_resid,eigenscore,verifier_refuted,verifier_not_supported,verifier_refute_confidence,verifier_uncertainty,verifier_no_retrieval_hit,selfcheck_refute_rate,selfcheck_disagreement \
  --geometry-signals truth_proj,subspace_resid,eigenscore \
  --uncertainty-signals verifier_refuted,verifier_refute_confidence,verifier_not_supported,verifier_no_retrieval_hit \
  --geometry-fusion-methods interaction,product,weighted_mean,noisy_or \
  --query-field answer \
  --retriever-min-overlap 0.95 \
  --verifier-min-overlap 0.65 \
  --retrieval-limit 3 \
  --omit-label-metadata \
  --compact-json
```

The manifest verifies 15/15 files. The stress corpus retrieves 706 hits and
covers 556/556 records, but that coverage is actively bad evidence:

- Qwen and SmolLM2: true-supported rate 0.936, false-supported rate 0.980,
  false-refuted rate 0.000, decision accuracy 0.438.
- Alpha 0.100 verified detection collapses to 0.013 for Qwen and 0.010 for
  SmolLM2, with zero false alarm because the verifier mostly supports both true
  and false answers.

This is the required negative control for retrieval grounding: evidence derived
from the model answers can look well-covered while destroying false-claim
refutation. Future retrieval improvements should beat the correct-answer corpus
baseline while also failing this answer-echo stress test.

## Text Baseline Redline

`benchmarks/build_text_baseline_score_dump.py` appends simple text controls to
statement-bearing score dumps without rerunning either model. The current l80
artifact is:

```bash
OUT=artifacts/truthfulqa-l80-text-baseline-comparison

python benchmarks/build_text_baseline_score_dump.py \
  --scores artifacts/qwen05_truthfulqa_l80_scores_with_statements.json \
  --keep-signals truth_proj,maha_last,subspace_resid,eigenscore,nll_answer \
  --output "$OUT/qwen-l80-text-baseline-scores.manifest.json" \
  --output-format jsonl \
  --json "$OUT/qwen-l80-text-baseline-report.json"

python benchmarks/build_text_baseline_score_dump.py \
  --scores artifacts/smollm2_truthfulqa_l80_scores_with_statements.json \
  --keep-signals truth_proj,maha_last,subspace_resid,eigenscore,nll_answer \
  --output "$OUT/smollm2-l80-text-baseline-scores.manifest.json" \
  --output-format jsonl \
  --json "$OUT/smollm2-l80-text-baseline-report.json"

python benchmarks/eval_score_ensemble.py \
  --scores qwen-l80="$OUT/qwen-l80-text-baseline-scores.manifest.json" \
  --scores smollm2-l80="$OUT/smollm2-l80-text-baseline-scores.manifest.json" \
  --signals truth_proj,maha_last,subspace_resid,eigenscore,nll_answer,answer_char_length,answer_token_count,claim_char_length,claim_token_count,question_answer_token_overlap,answer_negation_flag,answer_number_count \
  --methods max_rank,mean_rank \
  --alphas 0.05,0.1,0.2 \
  --repeats 50 \
  --best-alpha 0.10 \
  --json "$OUT/score-ensemble-report.json"
```

The artifact manifest verifies 9/9 files. At alpha 0.100:

- Qwen: `truth_proj` remains strongest with AUROC 0.761 and detection 0.279
  at false alarm 0.091. Cheap controls are weak: `answer_token_count` AUROC
  0.519 / detection 0.110, `claim_token_count` AUROC 0.527 / detection 0.089,
  and low `question_answer_token_overlap` triggers no detections.
- SmolLM2: `truth_proj` remains strongest with AUROC 0.740 and detection 0.229
  at false alarm 0.095. The text controls are identical because they come from
  the same statement metadata; none beats the internal monitor.

This is not a new product signal. It is a redline control: future
verifier/retrieval/selfcheck claims should beat these simple text artifacts
under the same conformal false-alarm budget before being treated as frontier
evidence.

## Frontier Stability Report

`benchmarks/eval_frontier_stability.py` replays saved frontier score dumps across
multiple split-conformal seeds without rerunning model forward passes. Current
registered l80 report:

```bash
OUT=artifacts/truthfulqa-frontier-qwen-smollm2-l80-stability

python benchmarks/eval_frontier_stability.py \
  --scores qwen05-l80=artifacts/truthfulqa-frontier-qwen-smollm2-l80/qwen05-l80/scores.manifest.json \
  --scores smollm2-l80=artifacts/truthfulqa-frontier-qwen-smollm2-l80/smollm2-l80/scores.manifest.json \
  --signals truth_proj,maha_last,subspace_resid,resid_update_norm,eigenscore \
  --methods max_rank,mean_rank \
  --alphas 0.05,0.1,0.2 \
  --best-alpha 0.10 \
  --sweep-alpha 0.10 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --repeats 20 \
  --json "$OUT/frontier-stability-report.json" \
  --artifact-manifest "$OUT/artifact-manifest.json"
```

At alpha 0.100 across seeds `0..9`:

- Qwen l80: `truth_proj` is the best single signal in 10/10 seeds; best
  ensemble is `mean_rank` in 10/10 seeds; single signal beats ensemble in 10/10
  seeds with mean detection margin 0.034.
- SmolLM2 l80: `truth_proj` is the best single signal in 10/10 seeds; best
  ensemble is `mean_rank` in 10/10 seeds; single signal beats ensemble in 10/10
  seeds with mean detection margin 0.053.

The report is registered as
`report:truthfulqa-frontier-qwen-smollm2-l80-stability:0.1`; the verified
manifest is registered as
`benchmark_manifest:truthfulqa-frontier-qwen-smollm2-l80-stability:0.1`. The
post-hoc replay now shares the score-dump cache across seeds, with 18/22 JSONL
selected-view lookups served from cache in the current report.

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

## Frontier Verifier Stability Report

`benchmarks/eval_verifier_stability.py` replays the staged structured-QA route
over current frontier l80 JSONL score dumps across multiple split-conformal
seeds without rerunning model forward passes.

```bash
OUT=artifacts/truthfulqa-frontier-qwen-smollm2-l80-verifier-stability

python benchmarks/eval_verifier_stability.py \
  --scores qwen05-l80=artifacts/truthfulqa-frontier-qwen-smollm2-l80/qwen05-l80/scores.manifest.json \
  --scores smollm2-l80=artifacts/truthfulqa-frontier-qwen-smollm2-l80/smollm2-l80/scores.manifest.json \
  --qa-corpus artifacts/truthfulqa_l80_correct_answer_corpus.json \
  --signal truth_proj \
  --alphas 0.05,0.1,0.2 \
  --best-alpha 0.10 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --repeats 20 \
  --staged-verification \
  --staged-alpha 0.10 \
  --json "$OUT/verifier-stability-report.json" \
  --artifact-manifest "$OUT/artifact-manifest.json"
```

At alpha 0.100 across seeds `0..9`:

- Qwen l80: verified false alarm mean 0.006, verified detection mean 0.305,
  and verified detection beats internal-only detection in 10/10 seeds. Route
  selection is stable at `staged_skip=441`, `structured_qa=115`.
- SmolLM2 l80: verified false alarm mean 0.010, verified detection mean 0.244,
  and verified detection beats internal-only detection in 10/10 seeds. Route
  selection is stable at `staged_skip=461`, `structured_qa=95`.

The report is registered as
`report:truthfulqa-frontier-qwen-smollm2-l80-verifier-stability:0.1`; the
verified manifest is registered as
`benchmark_manifest:truthfulqa-frontier-qwen-smollm2-l80-verifier-stability:0.1`.

The DECK detectability replay is now registered as
`report:truthfulqa-frontier-qwen-smollm2-l80-detectability:0.1`. It reuses the
current l80 score dumps and emits per-cell taxonomy reports through
`run_truthfulqa_frontier_workflow.py` without model reruns. With `eigenscore`
and `nll_answer` interpreted as lower-is-risk axes, Qwen has entrenched
false-rate `0.000`, while SmolLM2 has `89/306 = 0.291`. The paired release
evidence report
`report:truthfulqa-frontier-qwen-smollm2-l80-release-evidence-detectability:0.1`
keeps verifier stability promoted, keeps abstention blocked, and adds a
SmolLM2 detectability blocker because `0.29085` exceeds the default `0.25`
entrenched blind-spot gate. Its recursive artifact-manifest verification passes
and fingerprints both taxonomy reports plus the existing verifier/abstention
stability inputs.

The row-level blind-spot analysis is registered as
`report:truthfulqa-frontier-smollm2-l80-entrenched-blind-spots:0.1`. It exports
all 89 SmolLM2 false entrenched records from the taxonomy cell and records the
source taxonomy report, score dump, and artifact manifest. The largest question
groups are definition/what (`39`), person (`13`), and choice (`8`); answer text
is short on average (`5.18` tokens). The highest-margin examples include
negative common-knowledge traps, future/finance claims, celebrity/entity
confusions, and simple choice facts, which makes these records the immediate
target set for structured fact lookup, entity disambiguation, and world-model
correction experiments.

The correction-route audit is registered as
`report:truthfulqa-frontier-smollm2-l80-blind-spot-route-audit:0.1`. It replays
the existing l80 retrieval claims through `eval_verifier_ensemble.py` with a
per-record sidecar, then joins the 89 blind spots against that sidecar. The
current `retrieval_structured_qa` route selects and refutes `3/89` entrenched
false records, supports `0/89`, and leaves `86/89` outside the target route.
The artifact manifest verifies, so the conclusion is reproducible: selected
retrieval-structured-QA evidence is precise, but coverage is far too narrow to
remove the SmolLM2 detectability blocker.

The query-strategy sweep is registered as
`report:truthfulqa-frontier-smollm2-l80-blind-spot-query-sweep:0.1`. It runs
`16` retrieval query/overlap strategies against the same blind spots and the
controlled TruthfulQA correct-answer corpus. The original `answer@0.95`
baseline refutes `3/89`; `answer@0.5` refutes `54/89` with verified false alarm
`0.024`; `question_answer@0.65` refutes `87/89` with verified false alarm
`0.000`; and `question_answer@0.5` refutes `89/89` with verified false alarm
`0.000`. A question-only query also refutes `89/89` but raises false alarm to
`0.176`, so it stays a negative control. The next useful run is to port
question-aware query construction to external or structured-fact corpora with
provenance gates, rather than promoting the controlled-corpus result.

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

A shared-cache rerun in
`artifacts/smollm2_l20_inside_trigger_budget_sweep_shared_cache/` verifies the
long-sweep cache path. The first cache-producing run still pays the base cost:
top-10% fixed takes 199.136 seconds total, including 84.476 seconds for layer
stats, 39.854 seconds for forced-answer forward, and 60.819 seconds for INSIDE
generation. Later budget/run children load statement encodings, layer stats, and
eval reps from the shared cache: top-10% `adaptive_selfcheck` takes 75.756
seconds total with 0.063 seconds of eval-rep cache reads and 61.695 seconds of
INSIDE generation; top-25% takes 135.218 seconds total with 0.062 seconds of
cache reads and 119.916 seconds of INSIDE generation; top-40% takes 260.962
seconds total with 0.068 seconds of cache reads and 246.352 seconds of INSIDE
generation. The recommendation remains unchanged: top-10% is the cost-first
default, top-25% is the quality-balanced default, and the remaining bottleneck is
sampled INSIDE generation rather than repeated base scoring.

A diagnostics-cache rerun in
`artifacts/smollm2_l20_inside_trigger_budget_sweep_inside_cache/` verifies the
next performance layer: overlapping sampled INSIDE diagnostics can be reused
across nested trigger budgets. This run uses only `adaptive_selfcheck` and a
statement-stable diagnostics cache, so the reported `inside_generation` time is
the incremental sweep execution cost after previous budget coverage, not the
standalone cost of deploying that budget in isolation. Top-10% has no prior cache
coverage, writes 20 diagnostics entries, and spends 56.445 seconds in
`inside_generation`. Top-25% then sees 20 hits / 19 misses (`0.513` hit rate)
and spends 56.930 seconds in `inside_generation` versus 119.916 seconds in the
eval-reps-only shared-cache run. Top-40% sees 39 hits / 38 misses (`0.506` hit
rate) and spends 120.440 seconds versus 246.352 seconds. Recursive manifest
verification passes for the sweep and all three child manifests, including the
mutable shared diagnostics cache after final manifest refresh.

A derived-budget rerun in
`artifacts/smollm2_l20_inside_trigger_budget_sweep_derived/` validates the
single-source sweep path added for nested top-fraction budgets. The run executes
only the largest budget, top-40% `adaptive_selfcheck`, with `--dump-scores`, then
derives top-10% and top-25% rows from per-record batch indexes, trigger scores,
INSIDE scores, and sample counts. Recursive manifest verification passes for
the top-level report and the single source child manifest. The source run uses a
copied shared cache from the previous diagnostics-cache artifact; because the
new cache key now includes dtype, it correctly treats the previous diagnostics
entries as legacy misses, writes 77 dtype-aware diagnostics entries, and spends
235.259 seconds in `inside_generation` for 218 generated samples. The derived
top-10% and top-25% rows reproduce the previous diagnostics-cache sample counts
and AUROCs exactly: top-10% remains 55 samples with semantic-entropy AUROC
`0.494`, top-25% remains 108 samples with AUROC `0.521`, and top-40% remains 218
samples with AUROC `0.570`. The derived `inside_generation` values for top-10%
and top-25% are sample-count-ratio deployment estimates, 59.354 seconds and
116.550 seconds respectively, not sequential cache-hit execution times. The
practical result is that one source profile now replaces three child profile
runs for this sweep family: old diagnostics-cache children totaled about 401.4
seconds wall-clock, while the derived source profile totaled 251.2 seconds and
preserved the ranking, cost-first recommendation, and quality-balanced
recommendation.

The derived sweep has now been promoted into the local SmolLM2 release chain:

- Readiness record: `benchmark_manifest:smollm2-l20-readiness-inside-trigger-budget-derived:0.8`
- Release record: `benchmark_manifest:smollm2-l20-inside-trigger-budget-derived-staged-qa-release-candidate:0.8`
- Release comparison: `artifacts/smollm2_l20_inside_trigger_budget_derived_staged_release_candidate_comparison.json`
- Release manifest: `artifacts/smollm2_l20_inside_trigger_budget_derived_staged_release_candidate_manifest.json`

The release candidate reuses the SmolLM2 l20 performance matrix and the
TruthfulQA l80 staged structured QA route, then gates runtime cost on the
derived trigger-budget sweep. The promoted runtime recommendation uses the
quality-balanced `top_0p4` profile: sample-count ratio `0.472`,
`inside_generation` ratio `0.503`, semantic-entropy AUROC `0.570`, 77/154
statements sampled, 77 skipped, and 218 generated samples. Because this release
compares a trigger-budget sweep against a full-sample fixed reference rather
than a previous release baseline, the release comparison records the gate source
as `sample_count_ratio_to_reference` and
`inside_generation_seconds_ratio_to_reference`.

The current strict structured-retrieval-audit release record is
`benchmark_manifest:smollm2-l20-inside-trigger-budget-derived-strict-structured-retrieval-audit-staged-qa-release-candidate:1.6`.
It keeps the same readiness, staged structured-QA product route, performance
evidence, and retrieval-inclusive promoted adapter-family matrix, then adds
`benchmark_manifest:smollm2-l80-retrieval-structured-qa-route:0.6` as a required
retrieval-structured-QA audit gate. Version 1.6 additionally requires promoted
selector replay, a refreshed promoted product-runtime-drift report with 9
compared drift metrics and 0 blocked metrics, non-oracle retrieval evidence
provenance, and an answer-echo stress control that confirms answer-derived
retrieval evidence self-supports false claims at rate `0.980`. The product
handoff is exported as
`artifacts/smollm2_product_promotion_contract_v1_6/product-promotion-contract.json`. The final manifest fingerprints the
release-candidate report, readiness manifest, product route manifest,
performance manifest, selector replay manifest, product-runtime-drift manifest,
adapter-family matrix report, and required retrieval audit manifest. Required
adapter routes are `structured_state`, `state_transition`,
`retrieval_groundedness`, and `retrieval_structured_qa`, all promoted in
`artifacts/smollm2_l20_adapter_family_retrieval_structured_qa/adapter-family-matrix.json`.
The required retrieval audit route promotes with selected `238`, decision
accuracy `0.992`, false-supported rate `0.000`, false-refuted rate `1.000`,
runtime about `1.05s`, and `410` retrieval hits under a `450` hit budget. The
selected product route remains `structured_qa` with retrieval use gated at
`0.0` and mean attempted routes gated at `1.1`; retrieval is required as audit
evidence, not as the default low-latency path.

## Next Steps

1. Run `inside_eigenscore` only on the best layer band, not every layer, because
   multi-sample generation is the dominant CPU cost.
2. Pair future long runs with `--warmup-checkpoint`, `--layer-stats-cache`, and
   trigger-sweep `--shared-cache-dir`; the SmolLM2 shared-cache sweep confirms
   that budget children reuse statement encodings, layer stats, and eval reps
   instead of repeating base forward work. For nested top-fraction sweeps with a
   single run, prefer `run_inside_trigger_budget_sweep.py
   --derive-from-max-budget`; it preserves batch-local top-fraction semantics
   while avoiding repeated child profile execution. New sweeps should also
   inspect the shared `inside-diagnostics.json` hit rate to verify overlapping
   sampled INSIDE generation is being reused across nested trigger budgets.
3. Use sharded `--eval-reps-cache` for long forced-answer runs and cache-only
   rescoring; adjacent batch reads reuse the active shard and expose shard IO
   counters in the structured JSON output.
4. Use `--hidden-state-capture hooks` for targeted non-final layer-band runs when
   peak memory is the bottleneck; keep the default output capture for final-layer
   or full hidden-state semantics.
5. Treat top-10% as the cost-first triggered INSIDE experiment default and the
   registered top-40% derived budget as the current SmolLM2 quality-balanced
   release default until larger model/data runs change the Pareto frontier.
6. Promote a Qwen l20/l80 readiness baseline through the same registry workflow
   if Qwen-specific runtime evidence is needed; SmolLM2 now has the first
   non-tiny registered readiness/release candidate.
7. Extend the new verifier-stability path from structured QA to real retrieval,
   database, calculator, and world-model evidence under the same conformal
   false-alarm budgets. Use `benchmarks/build_evidence_fixture.py` with a local
   corpus as the reproducible non-oracle baseline before networked retrieval,
   convert aligned sampled responses with `build_selfcheck_signal_score_dump.py`,
   and compare every new signal against the text/length redline artifact.
8. Use `CalculatorVerifier` for arithmetic claims once extraction or upstream
   tools provide structured `expression` / `expected` metadata; it is a
   deterministic tool adapter, not a broad natural-language math parser.

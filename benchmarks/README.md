# EigenTruth Benchmarks

Reproducible evaluation scripts that turn EigenTruth's diagnostics into measurable
numbers. Unlike the qualitative scripts in [`examples/`](../examples/README.md),
these produce **AUROC** against labeled data so the core hypotheses can be tested
and ablated.

可复现评测脚本，把 EigenTruth 的诊断信号变成可度量的数字（AUROC），用于检验和消融核心假设。

## `eval_truthfulqa.py`

Tests whether hidden-state geometry separates **true** from **false** statements on
TruthfulQA, in a deterministic, judge-free, single-forward-pass setup (SAPLMA-style).

### What it answers

1. **Is the manifold/subspace geometry useful, and does it beat perplexity?**
   `maha_last` (Mahalanobis distance from the truth manifold) and `subspace_resid`
   (residual distance from a low-rank factual subspace) vs `nll_answer`
   (answer perplexity — a cheap, strong baseline any new method must beat).
2. **Does the hyperbolic projection earn its keep?**
   `disp_hse` (Hyperbolic Semantic Entropy) vs `disp_euclid` (the same dispersion
   computed in Euclidean space). If `disp_hse` does not beat `disp_euclid`, the
   hyperbolic machinery is decoration.
3. **Does internal-state spectral diversity add a cheap uncertainty signal?**
   `eigenscore` is an INSIDE/EigenScore-style log-det score over answer-token
   hidden embeddings. Add `--inside-samples K` (`K >= 2`) to also run
   `inside_eigenscore`, `inside_semantic_entropy`, `inside_embedding_entropy`,
   and `inside_semantic_energy`, closer multi-response proxies that sample
   verifier-style continuations, compute EigenScore over their sentence embeddings,
   and compute dependency-free lexical entropy, embedding-cluster entropy, and
   confidence-weighted semantic-energy disagreement over the sampled responses.
4. **Does the answer activate an unusual cross-layer update?**
   `resid_update_norm` is an ICR-inspired residual-dynamics proxy: it measures
   the RMS update from the previous hidden-state layer to the current layer for
   the final answer token. It is dependency-free and reuses the same forward pass.

### Method

- The truth manifold is built **only** from the correct answers of a held-out block
  of questions (`--manifold-questions`); evaluation runs on the remaining questions.
  Manifold-build and eval questions are disjoint, so there is no label leakage.
- Each candidate answer is scored with one forward pass at the target layer. The
  positive class (label 1) is an **incorrect** answer (the hallucination we want to
  flag); the negative class is a correct answer.
- AUROC is reported per signal. AUROC = P(score(false) > score(true)); 0.5 is chance,
  1.0 is perfect separation.

### Install and run

```bash
python -m pip install -e ".[hf,eval]"   # adds Transformers and `datasets`

# Real benchmark (downloads model weights + TruthfulQA):
python benchmarks/eval_truthfulqa.py --model Qwen/Qwen2.5-0.5B-Instruct --layer -8 --limit 200

# Sweep the target layer to find where the signal lives:
python benchmarks/eval_truthfulqa.py --model gpt2 --layer -8 --sweep \
  --batch-size 4 --dump-scores benchmarks/scores.json

# Sweep only a candidate layer band to control cost on larger models:
python benchmarks/eval_truthfulqa.py --model Qwen/Qwen2.5-0.5B-Instruct \
  --layer -12 --sweep-layers=-16,-14,-12,-10,-8 --batch-size 4 \
  --length-bucketed-batches --profile --progress-every 50

# Experimental memory mode: capture only selected non-final layer states via hooks:
python benchmarks/eval_truthfulqa.py --model Qwen/Qwen2.5-0.5B-Instruct \
  --layer -12 --sweep-layers=-16,-14,-12,-10,-8 --batch-size 4 \
  --hidden-state-capture hooks --length-bucketed-batches --profile

# Reuse warmup manifolds/subspaces across repeated runs with the same config:
python benchmarks/eval_truthfulqa.py --model Qwen/Qwen2.5-0.5B-Instruct \
  --layer -12 --sweep-layers=-16,-14,-12,-10,-8 --batch-size 4 \
  --layer-stats-cache artifacts/qwen05-layer-stats.pt

# Long warmup with restart safety: periodically save resumable warmup state:
python benchmarks/eval_truthfulqa.py --model Qwen/Qwen2.5-0.5B-Instruct \
  --layer -12 --sweep-layers=-16,-14,-12,-10,-8 --batch-size 4 \
  --layer-stats-cache artifacts/qwen05-layer-stats.pt \
  --warmup-checkpoint artifacts/qwen05-warmup-checkpoint.pt \
  --warmup-checkpoint-every 50 --progress-every 50

# Reuse both warmup stats and eval hidden states for repeated score/report runs:
python benchmarks/eval_truthfulqa.py --model Qwen/Qwen2.5-0.5B-Instruct \
  --layer -12 --sweep-layers=-16,-14,-12,-10,-8 --batch-size 4 \
  --layer-stats-cache artifacts/qwen05-layer-stats.pt \
  --eval-reps-cache artifacts/qwen05-eval-reps.pt

# Sharded eval reps cache: lower peak memory for larger eval splits:
python benchmarks/eval_truthfulqa.py --model Qwen/Qwen2.5-0.5B-Instruct \
  --layer -12 --sweep-layers=-16,-14,-12,-10,-8 --batch-size 4 \
  --layer-stats-cache artifacts/qwen05-layer-stats.pt \
  --eval-reps-cache artifacts/qwen05-eval-reps-cache \
  --eval-reps-cache-shard-size 256

# Cache-only rescoring: skip model loading and forced-answer forward entirely:
python benchmarks/eval_truthfulqa.py --model Qwen/Qwen2.5-0.5B-Instruct \
  --layer -12 --sweep-layers=-16,-14,-12,-10,-8 --cache-only \
  --layer-stats-cache artifacts/qwen05-layer-stats.pt \
  --eval-reps-cache artifacts/qwen05-eval-reps.pt

# Fast pipeline self-check (tiny model, bundled statements, no dataset download):
python benchmarks/eval_truthfulqa.py --model sshleifer/tiny-gpt2 --offline

# Optional covariance-spectrum diagnostics in the JSON report:
python benchmarks/eval_truthfulqa.py --model sshleifer/tiny-gpt2 --offline \
  --json artifacts/tiny-spectrum-report.json --include-layer-spectra \
  --layer-spectrum-top-k 8

# Optional multi-response INSIDE proxy (slower: samples K continuations per statement):
python benchmarks/eval_truthfulqa.py --model sshleifer/tiny-gpt2 --offline \
  --batch-size 4 --inside-samples 3 --inside-batch-size 2 --inside-max-new-tokens 6

# Budgeted INSIDE: sample only the most suspicious half of each eval batch:
python benchmarks/eval_truthfulqa.py --model sshleifer/tiny-gpt2 --offline \
  --batch-size 4 --inside-samples 3 --inside-trigger-signal truth_proj \
  --inside-trigger-top-fraction 0.5 --inside-max-new-tokens 6

# Adaptive INSIDE: cap at K samples, but stop early when semantic scores stabilize:
python benchmarks/eval_truthfulqa.py --model sshleifer/tiny-gpt2 --offline \
  --batch-size 4 --inside-samples 5 --inside-adaptive-sampling \
  --inside-min-samples 2 --inside-sample-step 1 --inside-stability-delta 0.05 \
  --inside-max-new-tokens 6
```

Use `--json results.json` to save structured output (config + AUROC per signal) for
the record. Use `--subspace-rank` to tune `TruthSubspace` residual scoring; fitting
requires at least two factual warmup states, and rank is clipped to the available
centered sample rank (`N - 1`) and hidden dimension. Increase `--batch-size` to
batch forced-answer forward passes, and increase `--inside-batch-size` to batch
sampled INSIDE prompts; higher values improve throughput but raise memory use.
Use `--sweep-layers=-16,-14,-12,-10,-8` to restrict layer sweeps to a candidate
band; this is the preferred mode for larger models and INSIDE runs where full
layer sweeps are not needed.
Use `--length-bucketed-batches` to sort statements by approximate text length
before batching, reducing padding waste without changing default behavior.
Use `--hidden-state-capture hooks` to collect only the selected non-final
decoder-layer hidden states via forward hooks instead of requesting the full
`output_hidden_states` tuple. This can reduce peak memory on targeted layer-band
runs, but it intentionally rejects embedding and final post-norm hidden-state
indexes because block hooks are not semantically identical for those positions.
Use `--covariance-mode diag` when local memory or warmup time is the bottleneck
for `maha_last`: it keeps only diagonal Welford scatter statistics and scores via
an optimized diagonal precision path instead of materializing a dense covariance
inverse. The default remains `full` for backward-compatible scores. The
experimental `--covariance-mode low_rank --covariance-low-rank K` path scores
with a ridge plus top-K covariance approximation; use it as a profiling
candidate and recalibrate thresholds because `maha_last` scale changes by mode.
Use `--covariance-mode shrinkage` to score with an OAS-style covariance estimate
shrunk toward the scaled identity before the existing ridge regularizer. This is
intended for small-sample/high-dimensional warmups where the full sample
covariance is poorly conditioned; calibrate thresholds separately for this mode.
Use `--include-layer-spectra` with `--json` to add compact
Marchenko-Pastur/effective-rank covariance diagnostics for each warmed layer.
The report stores top eigenvalues only (`--layer-spectrum-top-k`, default 16) so
large hidden dimensions do not inflate JSON artifacts; the flag is off by default
because full eigendecomposition is extra post-processing cost.
Use `compare_spectrum_layers.py` to test whether those spectrum heuristics
actually predict the best calibrated sweep layer before using them as a
layer-selection shortcut:

```bash
python benchmarks/compare_spectrum_layers.py \
  --spectrum-report artifacts/tiny-spectrum-report.json \
  --sweep-report artifacts/gpt2-sweep-report.json \
  --score truth_proj --top-k 3 \
  --json artifacts/tiny-spectrum-layer-comparison.json
```

Use `compare_layer_band_selectors.py` when a heuristic should be evaluated as a
candidate sweep band rather than an exact layer selector. It combines
intrinsic-dimension peak reports, spectrum reports, and saved sweep reports, then
checks whether each band contains the calibrated best layer:

```bash
python benchmarks/compare_layer_band_selectors.py \
  --intrinsic-report artifacts/e4-intrinsic-dimension-l80/intrinsic-dimension-report.json \
  --spectrum-report qwen05-l80=artifacts/truthfulqa-frontier-spectrum-layer-selection/qwen05-l80-spectrum-report.json \
  --spectrum-report smollm2-l80=artifacts/truthfulqa-frontier-spectrum-layer-selection/smollm2-l80-spectrum-report.json \
  --sweep-report qwen05-l80=artifacts/truthfulqa-frontier-qwen-smollm2-l80-cache-only/qwen05-l80/sweep-report.json \
  --sweep-report smollm2-l80=artifacts/truthfulqa-frontier-qwen-smollm2-l80-cache-only/smollm2-l80/sweep-report.json \
  --score truth_proj --coverage-top-k 2 \
  --json artifacts/truthfulqa-frontier-layer-band-selection/layer-band-comparison.json \
  --artifact-manifest artifacts/truthfulqa-frontier-layer-band-selection/artifact-manifest.json
```

The current l80 artifact recommends `spectrum_max_top_eigenvalue_to_mp_upper_radius_1`:
it keeps both models' best `truth_proj` layer in band with zero AUROC regret while
averaging 2 of 5 monitored layers. Treat it as a cost-reduction prior before the
normal calibrated sweep, not as a standalone deployment selector.

Use `--layer-stats-cache path.pt` to load an existing warmup manifold/subspace
bundle or create one when missing. The cache is validated against model, dtype,
layer list, max length, subspace rank, covariance mode/rank, warmup mode, and
warmup text fingerprint; use `--refresh-layer-stats-cache` to rebuild it
intentionally.
Use `--warmup-checkpoint path.pt` when building layer stats for long runs. The
checkpoint stores partial warmup manifold state plus factual/false hidden states
and resumes automatically when the same validated config is rerun; pair it with
`--layer-stats-cache` so a completed run still produces the compact final cache.
`--warmup-checkpoint-every N` controls checkpoint write frequency, and
`--refresh-layer-stats-cache` intentionally ignores an existing warmup checkpoint.
Use `--eval-reps-cache path.pt` to load or create cached forced-answer hidden
states, answer-token states, EigenScore proxy values, and answer NLLs for the
eval split. The eval cache is validated against model, dtype, layer list, max
length, EigenScore alpha, length-bucketing mode, and eval text fingerprint; use
`--refresh-eval-reps-cache` to rebuild it intentionally. This cache is independent
of INSIDE sampling, which still runs only when `--inside-samples` is enabled.
Use `--statement-encoding-cache path.json` to persist tokenizer outputs for
warmup/eval statements: token ids plus answer-span lengths. The cache is
validated against model id, max length, offline flag, warmup text fingerprint,
and eval statement fingerprint. It is most useful when rebuilding layer/eval
caches or comparing batch/layer settings without paying repeated tokenizer and
answer-span setup cost; use `--refresh-statement-encoding-cache` to rebuild it.
Use `--eval-reps-cache-shard-size N` to write the eval reps cache as a directory
containing a JSON manifest and `records-*.pt` shards. Existing sharded caches are
loaded batch-by-batch, reuse recently touched shards through a default 2-shard
read-side LRU cache, use a manifest index to seek directly to the first touched
shard for range reads, and remain compatible with `--cache-only`; old single-file
`.pt` caches remain the default and continue to load normally. Use
`--eval-reps-shard-read-cache-size 1` to restore single-shard reader memory
behavior on constrained machines. JSON output includes
`cache_stats.eval_reps_reader` with read requests, records read, shard read
requests, cross-shard reads, shard loads, shard cache hits, and shard manifest
scans, and profile summary includes `cache_efficiency.eval_reps_reader`
hit-rate/read-shape/manifest-scan metrics.
These fields help diagnose cache-only IO regressions after changing batch size
or token budget. `run_cache_profile_triplet.py`, `run_cache_profile_matrix.py`,
`run_cache_worker_sweep.py`, and `run_performance_baseline_workflow.py` also
accept `--eval-reps-shard-read-cache-size` so cached/cache-only profile runs can
reproduce the same read-side LRU capacity. Use
`run_cache_profile_matrix.py --eval-reps-shard-read-cache-sizes 1,2,4` when the
read-side cache capacity itself should be a gated matrix dimension; generated
cell ids include `read_cache_N`, cells share the same eval-reps cache artifact,
and runtime recommendations use the selected cell's read-cache size.
Use `--cache-only` with both cache paths to skip model loading and forced-answer
forward entirely. Cache-only mode is CPU-only, refuses refresh flags, and does
not run sampled INSIDE. New eval reps caches also store eval statement metadata,
so cache-only runs can restore labels/statements directly from the cache and skip
dataset loading. Older caches remain readable; when statement metadata is absent,
cache-only falls back to the original dataset load for validation and labels.
Use `--inside-trigger-signal` with either `--inside-trigger-threshold` or
`--inside-trigger-top-fraction` to run sampled INSIDE only on suspicious
statements. In this budgeted mode, untriggered statements receive
`inside_eigenscore=0.0`, `inside_semantic_entropy=0.0`, and
`inside_embedding_entropy=0.0`, and `inside_semantic_energy=0.0`; read them as two-stage policy scores, not as full
INSIDE-only AUROC. The JSON output includes `inside_sampling` counts.
Use `--inside-diagnostics-cache path.json` for repeated triggered INSIDE runs.
The cache stores sampled INSIDE diagnostics keyed by statement, model, layer,
sampling settings, and seed, but not by trigger threshold/top fraction. This lets
nested budgets reuse diagnostics for statements already sampled by an earlier
budget while preserving a cache miss for changed sampling settings. Use
`--refresh-inside-diagnostics-cache` to rebuild it intentionally.
If a score dump was not created with `--dump-inside-samples`, use
`export_inside_diagnostics_samples.py` to recover cached sampled texts into a
standard samples payload for selfcheck replay:

```bash
python benchmarks/export_inside_diagnostics_samples.py \
  --scores artifacts/smollm2_l20_inside_trigger_budget_sweep_derived/top_0p4/scores-adaptive_selfcheck.json \
  --inside-diagnostics-cache artifacts/smollm2_l20_inside_trigger_budget_sweep_derived/shared-caches/inside-diagnostics.json \
  --output artifacts/smollm2-l20-direct-selfcheck-signal-fusion/inside-diagnostics-samples.json \
  --artifact-manifest artifacts/smollm2-l20-direct-selfcheck-signal-fusion/inside-diagnostics-samples-manifest.json
```

Use `--inside-embedding-threshold` to tune the cosine-similarity cluster
threshold for `inside_embedding_entropy`. It defaults to `0.90`; higher values
split sampled embeddings into more clusters, while lower values merge more
responses into the same semantic-equivalence proxy.
Use `--inside-adaptive-sampling` to treat `--inside-samples` as the maximum
sample budget. The benchmark first draws `--inside-min-samples`, then adds
`--inside-sample-step` continuations until lexical and embedding entropy changes
are within `--inside-stability-delta` or the maximum budget is reached. The JSON
output records `inside_sample_counts`, `inside_adaptive_rounds`, and
`inside_stopped_early` per scored statement. Add
`--inside-selfcheck-early-stop` to also stop generation when finite-sample
self-consistency threshold bounds prove that no remaining continuations can
change the final support/refute/insufficient outcome. This option is off by
default and is most useful when the sampled continuations will feed
`SelfConsistencyVerifier` fixtures; reports include `inside_stop_reasons` and
`inside_sampling.stop_reason_counts` so cost savings remain auditable.
Add `--dump-inside-samples` with `--dump-scores` to include sampled continuation
text in the score dump as `inside_sample_texts`. This is useful for building
`SelfConsistencyVerifier` fixtures without rerunning generation:

```bash
python benchmarks/eval_truthfulqa.py --model sshleifer/tiny-gpt2 --offline \
  --inside-samples 3 --dump-scores artifacts/tiny_scores_with_samples.manifest.json \
  --dump-scores-format jsonl \
  --dump-inside-samples

python benchmarks/build_selfcheck_fixture.py \
  --scores artifacts/tiny_scores_with_samples.manifest.json \
  --output artifacts/tiny_selfcheck_claims.json
```

Use `run_inside_sampling_profile.py` to produce a reproducible cost report for
fixed sampling, adaptive sampling, and adaptive sampling with self-check
threshold bounds:

```bash
python benchmarks/run_inside_sampling_profile.py \
  --output-dir artifacts/inside_sampling_profile_tiny \
  --inside-samples 5 \
  --inside-min-samples 2 \
  --inside-sample-step 1 \
  --dump-scores \
  --fail-on-regression
```

The workflow writes per-run `result-*.json` and `profile-*.json` files plus
`inside-sampling-profile-comparison.json`, whose leaderboard reports total
generated samples, `inside_generation` seconds, ratio-to-fixed baselines, and
the recommended lowest-sample configuration. `--dry-run` prints the exact
commands without loading a model. Use `--skip-existing` after an interrupted
profile to reuse completed per-run result/profile files and only run missing
variants before rebuilding the comparison report and manifest.
Pass `--inside-trigger-signal` with `--inside-trigger-threshold` or
`--inside-trigger-top-fraction` to profile the same fixed/adaptive/self-check
variants under a budgeted two-stage policy where sampled INSIDE runs only on the
highest-risk statements.
For repeated comparable profile runs, pass `--statement-encoding-cache`,
`--layer-stats-cache`, `--eval-reps-cache`, and `--inside-diagnostics-cache`; add
`--refresh-shared-caches` only to rebuild the first cache-producing run.

Use `run_inside_trigger_budget_sweep.py` to compare several trigger budgets in
one reproducible report:

```bash
python benchmarks/run_inside_trigger_budget_sweep.py \
  --output-dir artifacts/inside_trigger_budget_sweep \
  --trigger-signal truth_proj \
  --top-fractions 0.1,0.2,0.3 \
  --reference-report artifacts/smollm2_l20_inside_sampling/inside-sampling-profile-comparison.json \
  --shared-cache-dir artifacts/inside_trigger_budget_sweep/shared-caches \
  --eval-reps-cache-shard-size 8 \
  --inside-samples 3 \
  --inside-max-new-tokens 4 \
  --runs fixed,adaptive_selfcheck
```

The sweep writes one child `inside-sampling-profile-comparison.json` per budget
plus `inside-trigger-budget-sweep.json`, whose leaderboard reports generated
samples, `inside_generation` seconds, ratios to the optional full-sample
reference, and inside-score AUROCs from the recommended run. The top-level
`recommendation` remains cost-first. `quality_balanced_recommendation` selects
the lowest-cost budget within `0.02` AUROC of the best preferred INSIDE quality
signal, preferring semantic entropy, then embedding entropy, then eigenscore.
Use `--shared-cache-dir` on long sweeps so budget/profile children reuse one
statement-encoding cache, layer-stats cache, eval-reps cache, and sampled INSIDE
diagnostics cache instead of repeating warmup, forced-answer forward work, and
overlapping sampled generation. Add
`--refresh-shared-caches` only when intentionally rebuilding those shared caches.
When comparing nested top-fraction budgets for a single run, add
`--derive-from-max-budget`. The runner executes only the largest top-fraction
budget with `--dump-scores`, then derives smaller budget rows from the score
dump's per-record `batch_indexes`, trigger scores, INSIDE scores, and sample
counts. Derived rows preserve the benchmark's batch-local top-fraction semantics;
their `inside_generation` seconds are sample-count-ratio estimates except for
the measured source budget.

Use `--profile` to include phase timings in stdout and `--json` output, or
`--profile-json profile.json` to write only the timing payload. This is the
recommended way to compare batch-size, layer-sweep, and INSIDE sampling changes
before treating a benchmark run as faster. The profile payload includes raw
`phases` plus a `summary` with the bottleneck phase, top phases, grouped time
shares for startup/tokenization/model-forward/cache/postprocess work, and
throughput fields for warmup and forced-answer eval records when counts are
available.
Use `--prefix-kv-cache` as an experimental forced-answer optimization when a
batch contains multiple candidate answers sharing the same question prefix. It
reuses one prefix KV cache per shared prefix during eval scoring, requires
`--hidden-state-capture outputs`, and is recorded in JSON/profile workflow
metadata. Keep it behind profile gates until a representative run proves that
the model/backend combination benefits.
Use `--progress-every N` to print warmup and eval progress every N statements
during long runs; the default is 50, and `--progress-every 0` disables periodic
progress output.

### How to read the results

- `maha_last > 0.5` means the manifold distance ranks false statements above true ones.
- `subspace_resid > 0.5` means false statements sit farther from the fitted factual subspace.
- `resid_update_norm > 0.5` means false statements induce larger final-token
  cross-layer residual updates at the selected layer.
- Compare `maha_last` against `nll_answer`: geometry is only interesting if it adds
  signal over plain perplexity.
- Compare `disp_hse` against `disp_euclid`: this is the decisive ablation for the
  hyperbolic component.
- Treat `eigenscore` as an internal-state spectral-diversity proxy. Use
  `inside_eigenscore`, `inside_semantic_entropy`, `inside_embedding_entropy`,
  and `inside_semantic_energy` when `--inside-samples` is enabled to test a closer
  multi-response INSIDE path. Calibrate them like other
  higher-is-more-anomalous scores before using them for routing.
- Results depend strongly on the target layer; sweep it.

### First results (indicative — `gpt2`, a weak base model)

End-to-end runs on real TruthfulQA, committed as `results_gpt2_l-8.json` (first run) and
`results_gpt2_sweep.json` (E0: adds `truth_proj` + full layer sweep). Setup: `gpt2` (124M
base model), manifold from 266 true / direction from 338 false statements (80 held-out
questions), 1075 eval statements (592 false / 483 true), seed 0.

For the first Qwen 0.5B instruction-model smoke run, see
[`docs/qwen05-truthfulqa-results.md`](../docs/qwen05-truthfulqa-results.md). The
short version: `truth_proj` remains the strongest signal, peaking at AUROC 0.711
around layer `-12`; the tiny multi-sample INSIDE smoke confirms the
`inside_eigenscore` path runs on a real instruction model, but needs a larger run
before it supports a statistical claim.

At the default layer −8:

| signal | AUROC |
|---|---|
| `truth_proj` | **0.723** |
| `maha_last` | 0.622 |
| `disp_euclid` | 0.484 |
| `disp_hse` | 0.474 |
| `nll_answer` | 0.411 |

Layer sweep with default hidden-state output capture uses one forward pass that
returns all hidden states. `truth_proj` AUROC by layer: peaks at **0.753 (layer
−6)**, stays above 0.72 across the −8…−2 band, and collapses at the last layer
(0.546). `maha_last` peaks at 0.638 (layer −4) and also collapses at −1.

Takeaways, all consistent with the project's stated caveats:

1. **The linear contrastive direction is the strongest signal.** `truth_proj` — the tool's
   own `contrastive_direction` used as a mass-mean probe — beats Mahalanobis distance at
   every layer except the earliest, consistent with the linear-truth-geometry literature
   (Marks & Tegmark). Practical default: monitor the contrastive direction at a mid-late
   layer (−8…−4); use `maha_last` when no false examples are available.
2. **The manifold distance carries real signal** (0.62–0.64) and clearly beats the
   perplexity baseline (`nll_answer` = 0.41 — anti-correlated, because a base LM finds
   common misconceptions *more* fluent, not less).
3. **The hyperbolic projection does not earn its keep here.** `disp_hse` (0.474) sits
   marginally *below* its Euclidean counterpart (0.484); both are at chance.
4. **Do not monitor the last layer.** Both geometric signals collapse at −1; the signal
   lives in the middle-to-late stack.

This is committed for reproducibility, not as a strong claim. `gpt2` is a weak 2019 base
model and within-statement dispersion is a cheap proxy for sample-based semantic entropy.
Re-run on an instruction-tuned model (e.g. `Qwen2.5-0.5B-Instruct`, which needs more RAM
than an 8 GB machine comfortably provides) before drawing conclusions.

### Limitations (read before quoting any number)

- Forced-answer statement scoring is a **proxy** for open-generation hallucination.
  It cleanly tests the representation hypothesis but is not the same as detecting
  hallucination during free generation.
- Within-statement token dispersion and `eigenscore` are **cheap proxies** for
  sample-based semantic uncertainty and INSIDE/EigenScore. `inside_eigenscore`,
  `inside_semantic_entropy`, `inside_embedding_entropy`, and
  `inside_semantic_energy` are closer because they sample multiple continuations,
  but they are still verifier-prompted benchmark proxies rather than full
  published reproductions.
- A small model (e.g. 0.5B) and a few hundred items give wide confidence intervals.
  Treat AUROC values as indicative, not conclusive, and report `n`.
- Beating these in-house baselines is necessary but not sufficient; a real claim needs
  comparison against full published detectors (semantic entropy, multi-response
  INSIDE/EigenScore, SAPLMA) on standard splits.

## `eval_conformal.py` (E1)

Validates that **split-conformal calibration** turns raw scores (Mahalanobis distance,
contrastive-direction projection) into alarm thresholds with honest finite-sample
coverage — replacing hand-picked thresholds like `mahalanobis_threshold=15.0`.

It consumes the per-statement scores dumped by `eval_truthfulqa.py --dump-scores`
(no model re-run needed). True statements form the exchangeable "normal" population,
split 50/50 into calibration/test over 20 seeded repeats:

```bash
python benchmarks/eval_truthfulqa.py --model gpt2 --dump-scores benchmarks/scores.json ...
python benchmarks/eval_conformal.py --scores benchmarks/scores.json --signal truth_proj

# For larger sweeps, write a streaming score dump manifest plus JSONL records sidecar:
python benchmarks/eval_truthfulqa.py --model gpt2 --sweep \
  --dump-scores benchmarks/scores.manifest.json \
  --dump-scores-format jsonl
python benchmarks/eval_conformal.py --scores benchmarks/scores.manifest.json --signal truth_proj

# Override the score direction for lower-is-more-anomalous signals:
python benchmarks/eval_conformal.py --scores benchmarks/scores.json --signal support_score \
  --direction lower

# Save a reusable CalibrationArtifact for one selected signal:
python benchmarks/eval_conformal.py --scores benchmarks/scores.json --signal truth_proj \
  --artifact-alpha 0.2 --save-calibration artifacts/gpt2-l8-truth-proj.json

# Build an adaptive conformal report/artifact from primary score or dump extra fields:
python benchmarks/eval_conformal.py --scores benchmarks/scores.manifest.json --signal maha_last \
  --adaptive-feature inside_semantic_energy \
  --adaptive-feature-weight inside_semantic_energy=0.5 \
  --save-adaptive-calibration artifacts/gpt2-maha-adaptive.json

# Build a conformal abstention sidecar for calibrated participation control:
python benchmarks/eval_conformal.py --scores benchmarks/scores.json --signal maha_last \
  --abstention-signal maha_last \
  --abstention-alpha 0.1 \
  --save-abstention-report artifacts/gpt2-abstention-report.json

# Compare several abstention candidates before promoting one into control policy:
python benchmarks/eval_conformal.py --scores benchmarks/scores.json --signal maha_last \
  --abstention-signals maha_last,truth_proj,subspace_resid,inside_eigenscore \
  --abstention-alpha 0.1 \
  --save-abstention-comparison artifacts/gpt2-abstention-comparison.json

# Turn the selected abstention report/comparison candidate into a release gate:
python benchmarks/eval_conformal.py --scores benchmarks/scores.json --signal maha_last \
  --abstention-signals maha_last,truth_proj,subspace_resid,inside_eigenscore \
  --abstention-alpha 0.1 \
  --save-abstention-release-gate artifacts/gpt2-abstention-release-gate.json \
  --min-abstention-conditional-correctness-lower-bound 0.8 \
  --max-abstention-rate 0.5

# Build the 0.2 calibrated-observability closure: layer/score sweep + best artifact:
python benchmarks/eval_conformal.py --scores benchmarks/scores.json \
  --signals maha_last,truth_proj,subspace_resid,resid_update_norm,eigenscore,inside_eigenscore,inside_semantic_entropy,inside_embedding_entropy,inside_semantic_energy \
  --artifact-alpha 0.2 \
  --json artifacts/gpt2-conformal-report.json \
  --save-sweep-report artifacts/gpt2-sweep-report.json \
  --save-best-calibration artifacts/gpt2-best-calibration.json \
  --artifact-manifest artifacts/gpt2-conformal-manifest.json
```

**E1 result (gpt2, layer −8):** empirical false-alarm rate tracks the nominal α within
1.3% at α ∈ {0.05, 0.1, 0.2} for both signals — the guarantee holds in practice. At the
same α = 0.2 false-alarm budget, `truth_proj` detects **46.9%** of false statements vs
34.1% for `maha_last` (committed as `results_conformal_*.json`). The low-level
calibration functions live in `eigentruth.eval.conformal` (`conformal_pvalues`,
`conformal_threshold`, `directional_conformal_threshold`,
`directional_conformal_thresholds`, `directional_trigger_rate`).
Reusable single-signal artifacts are built with `eigentruth.calibration.ConformalCalibrator`;
layer/score reports and best artifacts are built with
`eigentruth.calibration.LayerScoreSweepCalibrator`. Large layer/score sweeps can
use bounded CPU post-processing parallelism with `eval_conformal.py --sweep-workers`
or `LayerScoreSweepCalibrator(max_workers=...)`; the default remains one worker
for deterministic low-resource local runs. Structured reports also include
`selective_report` fields for threshold, coverage, selective accuracy, detection,
false alarm, and simple binomial confidence intervals; thresholding honors each
score's `higher` or `lower` anomalous direction while score dumps remain unchanged.
When `--save-abstention-report` or `--include-abstention-report` is set,
`eval_conformal.py` also emits a `ConformalAbstentionReport`: it calibrates the
retained participation region on correct responses, then reports empirical
participation/abstention, selective accuracy, correct-retention, and conservative
conditional-correctness lower bounds. This is for answer participation control and
does not change the base E1 conformal verdict.
When `--save-abstention-comparison` or `--include-abstention-comparison` is set,
the script emits a `ConformalAbstentionComparisonReport` over `--abstention-signals`
(or `--signals` when no abstention list is provided). The default ranking metric is
`conditional_correctness_lower_bound`; `--abstention-best-by` can instead rank by
empirical selective accuracy, participation, correct-retention lower bound, or
correct-retention rate. JSONL inputs load only the requested comparison columns.
When `--save-abstention-release-gate` or `--include-abstention-release-gate` is set,
the script evaluates the selected report or comparison recommendation as a
fail-closed promotion gate. It requires both
`--min-abstention-conditional-correctness-lower-bound` and
`--max-abstention-rate`; a failing gate sets the main payload verdict to `REJECT`.
When `--adaptive-feature` is provided, `eval_conformal.py` loads the feature from
a selected primary score, JSON dump extra array, JSONL manifest extra, or JSONL
per-record extra, then writes an `adaptive_conformal_report` whose adjusted score
is always higher-is-more-anomalous. `--save-adaptive-calibration` stores that
adjusted score as a standard `CalibrationArtifact` with transform metadata. For
JSONL dumps, selected primary scores and requested adaptive record extras are
loaded through one selected streaming view.
When a primary confidence proxy such as `nll_answer` is present, `eval_conformal.py`
also adds a `confidence_error_report` under each alpha result. By default it treats
the lowest 25% `nll_answer` rows as the high-confidence region and reports how many
high-confidence false statements were accepted, flagged, or missed by the calibrated
anomaly gate. Use `--confidence-signal`, `--confidence-direction`,
`--confidence-top-fraction`, or `--disable-confidence-audit` to tune or skip this
audit. This is meant to expose high-confidence error regimes; it does not change
the score dump schema or the conformal threshold.
The report config includes `score_dump` metadata from `eigentruth.eval.ScoreDump`:
record counts, available score names, sweep layers, file size, SHA-256, and a
stable `identity` payload. That identity records model, dataset, layer, selected
score schema, scoring-config hash, content hashes, and a canonical cache key. This
lets later calibration, ensemble, and route-refresh steps confirm they are reusing
the intended dump without parsing model artifacts again. Post-processing reports
share a run-local score-dump cache, so duplicate score paths do not require
re-hashing the same file, re-parsing the same JSONL manifest, or re-scanning the
same selected JSONL view. Reports include a top-level `score_dump_cache` summary
with fingerprint, JSONL manifest, JSONL summary, and selected-view
hits/misses/writes. For larger score artifacts, `load_score_dump()` also
accepts an `eigentruth.score_dump.jsonl` manifest that points at JSONL records;
`iter_score_dump_jsonl_records()` can validate those records without materializing
the whole dump. `eval_conformal.py`, `eval_score_ensemble.py`,
`eval_verifier_ensemble.py`, `eval_calibration_transfer.py`, and
`LayerScoreSweepCalibrator.calibrate_from_file()` use selected JSONL score views
where possible, so large JSONL inputs materialize only the requested primary,
statement-bearing, or layer/score columns plus labels. These selected loaders
accept the same optional run-local cache and invalidate cached views when the
manifest or records file changes. `score_dump_cache_summary()` exposes the same
counters for custom post-processing scripts. Score-dump metadata
fingerprints both the manifest and the records file. New JSONL manifests include
label counts so summary-only metadata can avoid reading the records sidecar; older
manifests still use a cached label-only record scan instead of materializing
score columns. Selected JSONL scans also prime the run-local records fingerprint
cache, so later metadata keeps full SHA-256 provenance without a second records
pass. When `eval_conformal.py` writes a sweep report or best calibration artifact
from JSONL scores, it reuses the preloaded layer/score view for both the base
conformal report and the sweep.
When `--artifact-manifest` is provided, the conformal report gains
`artifact_manifest_summary` and `paths.artifact_manifest`; the manifest
fingerprints the input score dump plus generated conformal, sweep, and
calibration artifacts for later verification or registry promotion.

Caveat: the guarantee is conditional on exchangeability — under distribution shift
(different domain than the calibration set) coverage can degrade; recalibrate per domain.

## `run_calibrated_observability_workflow.py`

Runs the 0.2 calibrated-observability closure in one command. It can either
reuse an existing score dump or call `eval_truthfulqa.py` to create one as a
JSONL manifest, then runs `eval_conformal.py` with sweep, best-calibration, and
artifact-manifest outputs. The top-level workflow report records both commands,
the conformal verdict, nested manifest verification, and optional registry
metadata. It also includes an `evidence_bundle` summary with the score-dump
provenance, best sweep calibration, manifest verification status, runtime
preset, and registry record:

```bash
python benchmarks/run_calibrated_observability_workflow.py \
  --output-dir artifacts/gpt2-calibrated-observability \
  --runtime-preset calibrate \
  --model gpt2 \
  --layer -8 \
  --scores artifacts/gpt2-calibrated-observability/scores.manifest.json \
  --dump-scores-format jsonl \
  --registry artifacts/local-release-registry.json \
  --name gpt2-calibrated-observability \
  --version 0.1
```

For fast calibration iteration after the model run has already produced a score
dump, pass the existing `--scores` path without `--refresh-scores`; the workflow
will skip `eval_truthfulqa.py`, rerun only conformal calibration, and keep the
top-level manifest focused on the reused score dump plus generated calibration
artifacts.

When a `compare_layer_band_selectors.py` report has already promoted a candidate
band, pass it through the workflow to derive `eval_truthfulqa.py --sweep-layers`
without copying layer lists by hand:

```bash
python benchmarks/run_calibrated_observability_workflow.py \
  --output-dir artifacts/qwen05-band-calibrated-observability \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --layer -12 \
  --sweep-layers-from-band-report artifacts/truthfulqa-frontier-layer-band-selection/layer-band-comparison.json \
  --sweep-band-run qwen05-l80 \
  --sweep-band-target-layer best \
  --layer-stats-cache artifacts/cache/qwen05-layer-stats.pt \
  --eval-reps-cache artifacts/cache/qwen05-eval-reps.pt \
  --cache-only
```

If the report contains exactly one run matching `--model`, the workflow selects
it automatically. Otherwise pass `--sweep-band-run`. The selected report is
fingerprinted in the top-level manifest as `sweep_layer_band_report`, and the
workflow report records the selected strategy, run, candidate layers, and AUROC
regret. Explicit `--sweep-layers` remains the highest-priority layer selection.
Use `--sweep-band-target-layer best|band_best|first` to set the primary
`--layer` from the selected report, avoiding an extra target layer outside the
candidate band. Use `--sweep-band-expand-radius N` to turn a sparse selected
band into a denser local grid; for example the current Qwen l80 band `[-10,-8]`
with radius 1 becomes `[-11,-10,-9,-8,-7]`.

For larger real-model runs, pass the TruthfulQA caches through this workflow:
`--statement-encoding-cache`, `--layer-stats-cache`, and `--eval-reps-cache`
reuse tokenization, warmup manifolds, and forced-answer hidden states across
calibration iterations. Add `--refresh-*-cache` only when the cache should be
rebuilt, and use `--cache-only` with existing layer/eval caches when the model
forward pass should be skipped entirely.

Use `--runtime-preset quick` for bounded local smoke runs, `calibrate` when
iterating on existing score dumps, and `full` for real TruthfulQA-oriented runs
with longer contexts and auto batch-size fallback. Any explicit CLI parameter
overrides the preset default.

## `run_truthfulqa_frontier_workflow.py`

Runs the multi-model/multi-scale frontier research path in one command. By
default it plans Qwen 0.5B and SmolLM2 l20/l80 cells, each using the
calibrated-observability closure, then compares the resulting score dumps with
direction-aware rank-fusion ensembles. The default signal set includes
`resid_update_norm` alongside `truth_proj`, `maha_last`, `subspace_resid`, and
`eigenscore`.

```bash
python benchmarks/run_truthfulqa_frontier_workflow.py \
  --output-dir artifacts/truthfulqa-frontier-qwen-smollm2 \
  --model qwen05=Qwen/Qwen2.5-0.5B-Instruct \
  --model smollm2=HuggingFaceTB/SmolLM2-135M-Instruct \
  --scale l20=20:40:-8:-16,-14,-12,-10,-8 \
  --scale l80=80:80:-12:-16,-14,-12,-10,-8 \
  --batch-size 4 \
  --max-length 96 \
  --hidden-state-capture hooks \
  --cache-dir artifacts/cache/truthfulqa-frontier-qwen-smollm2 \
  --signals truth_proj,maha_last,subspace_resid,resid_update_norm,eigenscore \
  --registry artifacts/local-release-registry.json \
  --name truthfulqa-frontier-qwen-smollm2 \
  --version 0.1
```

Use `--dry-run --offline` first to verify commands and artifact paths without
model downloads. Re-running without `--refresh-scores` reuses existing cell
score dumps and only refreshes conformal/ensemble reports.
Use `--cache-dir` for l80 or multi-seed runs so each model/scale cell gets
stable per-cell `statement-encodings.json`, `layer-stats.pt`, `eval-reps-cache`,
and warmup checkpoint paths; use `--refresh-caches` only for the first cache
build or when changing cache-defining model/data/layer parameters.

When a layer-band selector report should drive several frontier cells, pass the
same report to the frontier workflow and let each cell match its own run name:

```bash
python benchmarks/run_truthfulqa_frontier_workflow.py \
  --output-dir artifacts/truthfulqa-frontier-qwen-smollm2-l80-dense-band \
  --model qwen05=Qwen/Qwen2.5-0.5B-Instruct \
  --model smollm2=HuggingFaceTB/SmolLM2-135M-Instruct \
  --scale l80=80:80:-12:-16,-14,-12,-10,-8 \
  --sweep-layers-from-band-report artifacts/truthfulqa-frontier-layer-band-selection/layer-band-comparison.json \
  --sweep-band-scales l80 \
  --sweep-band-expand-radius 1 \
  --sweep-band-target-layer best \
  --cache-dir artifacts/cache/truthfulqa-frontier-qwen-smollm2 \
  --signals truth_proj,maha_last,subspace_resid,resid_update_norm,eigenscore
```

The default band-report run template is `{cell}`, so `qwen05-l80` and
`smollm2-l80` select their corresponding report rows. Use
`--sweep-band-run-template` when the report uses another naming convention, and
use `--sweep-band-scales` when only selected scales should consume the band
report. The top-level report records each cell's resolved target layer,
resolved dense sweep layers, and source report run.

## `eval_frontier_stability.py`

Replays existing frontier score dumps across several split-conformal seeds
without loading a model. Use it after `run_truthfulqa_frontier_workflow.py` to
check whether the best internal signal and simple rank-fusion comparison are
stable under calibration splits.

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
  --sweep-best-by auroc \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --repeats 20 \
  --json "$OUT/frontier-stability-report.json" \
  --artifact-manifest "$OUT/artifact-manifest.json" \
  --registry artifacts/local-release-registry.json \
  --name truthfulqa-frontier-qwen-smollm2-l80-stability \
  --version 0.1

python benchmarks/verify_artifact_manifest.py \
  --manifest "$OUT/artifact-manifest.json" \
  --recursive \
  --json "$OUT/manifest-verification.json"
```

The stability manifest fingerprints both score manifest files and their JSONL
records sidecars. If the result is used as release evidence, promote the
verified manifest with `promote_artifact_manifest.py` so the registry contains
both the report and the manifest verification record.
The l80 stability report also records score-dump cache reuse: the shared JSONL
selected-view cache currently hits 18/22 lookups across the multi-seed replay.

## `eval_abstention_stability.py`

Replays existing score dumps across seeded conformal-abstention splits without
loading a model. Use it after building abstention comparison/release-gate
sidecars to check whether the recommended participation-gate signal and
release-gate verdict remain stable under calibration splits.

```bash
OUT=artifacts/truthfulqa-frontier-qwen-smollm2-l80-abstention-stability

python benchmarks/eval_abstention_stability.py \
  --scores qwen05-l80=artifacts/truthfulqa-frontier-qwen-smollm2-l80/qwen05-l80/scores.manifest.json \
  --scores smollm2-l80=artifacts/truthfulqa-frontier-qwen-smollm2-l80/smollm2-l80/scores.manifest.json \
  --signals maha_last,truth_proj,subspace_resid,disp_euclid,disp_hse,nll_answer,eigenscore,resid_update_norm \
  --alpha 0.10 \
  --best-by conditional_correctness_lower_bound \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --min-abstention-conditional-correctness-lower-bound 0.8 \
  --max-abstention-rate 0.5 \
  --json "$OUT/abstention-stability-report.json" \
  --artifact-manifest "$OUT/artifact-manifest.json" \
  --registry artifacts/local-release-registry.json \
  --name truthfulqa-frontier-qwen-smollm2-l80-abstention-stability \
  --version 0.1

python benchmarks/verify_artifact_manifest.py \
  --manifest "$OUT/artifact-manifest.json" \
  --recursive \
  --json "$OUT/manifest-verification.json"
```

Each seed calibrates thresholds on a stratified calibration split of correct
records and evaluates participation metrics on the held-out split. The report
summarizes recommended-signal counts, conditional-correctness lower-bound
variance, abstention-rate variance, and release-gate pass/block counts per
score dump. It also emits a `supervised_feasibility_frontier` diagnostic: a
label-using threshold sweep that estimates the best conditional-correctness
lower bound achievable by each candidate signal under the configured abstention
budget. This is an upper-bound diagnostic only (`promotion_eligible=false`), not
a runtime calibration artifact. JSONL inputs load only the requested abstention
candidate columns.

## `compare_frontier_release_evidence.py`

Combines frontier stability reports into one fail-closed release verdict without
rerunning models, verifiers, or retrieval. It treats staged verifier stability
and abstention-gate stability as separate tracks, then blocks the release
candidate if either track misses its configured seed-rate or metric thresholds.

```bash
OUT=artifacts/truthfulqa-frontier-qwen-smollm2-l80-release-evidence

python benchmarks/compare_frontier_release_evidence.py \
  --verifier-stability-report artifacts/truthfulqa-frontier-qwen-smollm2-l80-verifier-stability/verifier-stability-report.json \
  --abstention-stability-report artifacts/truthfulqa-frontier-qwen-smollm2-l80-abstention-stability/abstention-stability-report.json \
  --json "$OUT/frontier-release-evidence.json" \
  --artifact-manifest "$OUT/artifact-manifest.json" \
  --verification-report "$OUT/manifest-verification.json" \
  --registry artifacts/local-release-registry.json \
  --name truthfulqa-frontier-qwen-smollm2-l80-release-evidence \
  --version 0.1
```

The current l80 evidence promotes the verifier-stability track but blocks the
abstention-stability track, so the combined release verdict is blocked. This is
the expected posture until participation-gate evidence clears the conservative
conditional-correctness lower-bound gate.

## `eval_verifier_ensemble.py`

Compares a single calibrated internal diagnostic against a retrieval/verifier
ensemble policy from saved score dumps plus claim/evidence metadata. This is the
benchmark entry point for the product hypothesis: keep `truth_proj` as the
primary internal monitor, but let evidence-backed verification suppress supported
internal alarms and add detections for refuted claims.

```bash
python benchmarks/eval_verifier_ensemble.py \
  --scores run=artifacts/scores-with-statements.json \
  --signal truth_proj \
  --alphas 0.05,0.1,0.2 \
  --repeats 50 \
  --verified-records-jsonl artifacts/verifier_ensemble_verified_records.jsonl \
  --json artifacts/verifier_ensemble_report.json
```

Add `--compact-json` for large automated runs when the report is consumed by
tools and does not need human-readable indentation.
Add `--verified-records-jsonl` when per-claim verifier outputs are needed for
audit or debugging; the main report keeps summary metrics and references the
sidecar path/count instead of embedding those records.
Add `--enable-triple-evidence` to evaluate strict fact-level
subject-predicate-object audits for sensitive factual claims before lexical
groundedness fallback; use `--triple-min-slot-coverage` to relax or tighten the
per-slot evidence coverage threshold.
Each run validates inputs through `eigentruth.eval.ScoreDump` or selected score
views and records a `score_dump` summary plus SHA-256 fingerprint, so
verifier-cache and route promotion evidence can be tied back to the exact score
artifact. JSONL score dumps are read through `load_score_dump_statement_scores()`,
materializing only labels, the selected signal, and statements.

Add `--staged-verification` to benchmark the cost-aware control plane path. The
script first calibrates a cheap internal diagnostic gate with `--staged-alpha`,
then skips expensive verifier routes for low-risk claims unless claim metadata
matches staged policy triggers such as `features.has_number` or
`requires_verification`:

```bash
python benchmarks/eval_verifier_ensemble.py \
  --scores run=artifacts/scores-with-statements.json \
  --signal truth_proj \
  --staged-verification \
  --staged-alpha 0.1 \
  --json artifacts/verifier_ensemble_staged_report.json
```

Each run reports `staged_verification.skipped_records`, `skip_rate`,
`reason_counts`, triggered feature/metadata counts, and the staged conformal
threshold used for the verifier gate. Product control loops treat skipped
claim verification as unverified by default (`fail_closed_on_skip=true`), so
cost-aware staging does not silently convert low diagnostic risk into factual
acceptance; explicit local latency experiments can opt out with
`stage_fail_closed_on_skip=false` in control defaults.

## `eval_verifier_stability.py`

Replays `eval_verifier_ensemble.py` across several split-conformal seeds without
loading a model. Use it when a verifier route looks promising and needs the same
seed-stability evidence as internal diagnostic sweeps.

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
  --artifact-manifest "$OUT/artifact-manifest.json" \
  --registry artifacts/local-release-registry.json \
  --name truthfulqa-frontier-qwen-smollm2-l80-verifier-stability \
  --version 0.1

python benchmarks/verify_artifact_manifest.py \
  --manifest "$OUT/artifact-manifest.json" \
  --recursive \
  --json "$OUT/manifest-verification.json"

python benchmarks/promote_artifact_manifest.py \
  --manifest "$OUT/artifact-manifest.json" \
  --registry artifacts/local-release-registry.json \
  --name truthfulqa-frontier-qwen-smollm2-l80-verifier-stability \
  --version 0.1 \
  --verification-report "$OUT/manifest-verification.json" \
  --metadata workflow=eval_verifier_stability \
  --metadata evidence=truthfulqa_frontier_qwen_smollm2_l80_verifier_stability
```

The current registered frontier l80 verifier-stability report
(`report:truthfulqa-frontier-qwen-smollm2-l80-verifier-stability:0.1`) uses
staged structured QA on the current Qwen/SmolLM2 l80 JSONL score dumps. Across
seeds `0..9`, Qwen verified false alarm averages 0.006 and verified detection
averages 0.305, beating internal-only detection in 10/10 seeds while routing
115/556 records to `structured_qa`. SmolLM2 verified false alarm averages 0.010
and verified detection averages 0.244, also beating internal-only detection in
10/10 seeds while routing 95/556 records to `structured_qa`. The verified
manifest is registered as
`benchmark_manifest:truthfulqa-frontier-qwen-smollm2-l80-verifier-stability:0.1`.
The default run-local verifier trace cache avoids rerunning seed-invariant
verified records after the first seed; both Qwen and SmolLM2 hit it in 9/10
seed runs.
When `--verification-cache-dir` is supplied, the runner fingerprints the cache
file in the artifact manifest, so preserve that cache with the report if it is
part of release evidence. Without it, verifier stability uses a temporary
run-local cache to avoid rerunning seed-invariant verifier records.

If the score dump does not contain `statements`, provide a fixture with one
record per score:

```bash
python benchmarks/eval_verifier_ensemble.py \
  --scores qwen-l80=artifacts/qwen05_truthfulqa_l80_scores.json \
  --claims artifacts/qwen05_truthfulqa_l80_claims.json \
  --signal truth_proj \
  --json artifacts/qwen05_verifier_ensemble_report.json
```

For structured QA or database-like sources, pass a corpus containing
`question`/`answer` facts. `QuestionAnswerVerifier` checks the structured source
before falling back to lexical evidence retrieval:

```bash
python benchmarks/eval_verifier_ensemble.py \
  --scores qwen-l80=artifacts/qwen05_truthfulqa_l80_scores_with_statements.json \
  --scores smollm2-l80=artifacts/smollm2_truthfulqa_l80_scores_with_statements.json \
  --qa-corpus artifacts/truthfulqa_l80_correct_answer_corpus.json \
  --signal truth_proj \
  --json artifacts/truthfulqa_l80_structured_qa_verifier_ensemble_report.json
```

For natural-language claims backed by subject-predicate-object facts, pass
`--fact-corpus`. `StructuredFactVerifier` extracts simple claim triples, supports
common paraphrases such as possessive and subject-first fact statements, matches
aliases and multi-object lists, and refutes object mismatches when the
subject/predicate pair is covered by the fact corpus:

```bash
python benchmarks/eval_verifier_ensemble.py \
  --scores facts=artifacts/wikidata-country-core-facts-structured-fact-route/covered-facts-scores.json \
  --fact-corpus artifacts/wikidata-country-core-facts-external-corpus/wikidata-country-core-facts-qa-corpus.json \
  --signal truth_proj \
  --json artifacts/wikidata-country-core-facts-structured-fact-route/structured-fact-verifier-report.json
```

For structured state, business rules, policy checks, or tool-output checks,
provide explicit `state_check` metadata in the claim fixture and pass a local
state JSON file. `StructuredStateVerifier` checks these deterministic rules
after structured QA/fact routes and before lexical retrieval:

```bash
python benchmarks/eval_verifier_ensemble.py \
  --scores run=artifacts/scores-with-statements.json \
  --claims artifacts/state_checked_claims.json \
  --state-source artifacts/domain_state.json \
  --signal truth_proj \
  --json artifacts/state_verifier_ensemble_report.json
```

The state source may be a raw JSON object used as state, an object with `state`,
optional `state_checks`, and optional `state_transitions` fields, or a SQLite
state-source spec:

```json
{
  "sqlite": {
    "database_path": "domain_state.db",
    "queries": [
      {
        "path": "orders.ord_0001.can_ship",
        "sql": "select 1 as can_ship",
        "column": "can_ship",
        "required": true
      }
    ]
  }
}
```

Relative SQLite database paths are resolved from the state-source JSON file's
directory. A claim fixture record can provide `state_check` directly or under
`claim_metadata.state_check`; top-level `state_checks` keyed by `claim_id` are
also supported.

For a fully reproducible domain-state smoke benchmark, generate synthetic
order-fulfillment state, claim, and score fixtures, then feed them into the same
verifier-ensemble runner:

```bash
python benchmarks/build_domain_state_fixture.py \
  --scores-output artifacts/order_fulfillment_scores.json \
  --claims-output artifacts/order_fulfillment_claims.json \
  --state-output artifacts/order_fulfillment_state.json \
  --sqlite-output artifacts/order_fulfillment_state.db \
  --sqlite-state-source-output artifacts/order_fulfillment_sqlite_state_source.json \
  --n-records 12

python benchmarks/eval_verifier_ensemble.py \
  --scores orders=artifacts/order_fulfillment_scores.json \
  --claims artifacts/order_fulfillment_claims.json \
  --state-source artifacts/order_fulfillment_state.json \
  --signal truth_proj \
  --json artifacts/order_fulfillment_verifier_ensemble_report.json
```

This fixture checks the product-control path, not open-domain factuality: true
labels are shippable orders, while false labels are claims that an order can
ship even though inventory or account state refutes it.
Swap `--state-source artifacts/order_fulfillment_state.json` for
`--state-source artifacts/order_fulfillment_sqlite_state_source.json` to run the
same benchmark through read-only SQLite queries.

For action-conditioned state checks, provide `state_transition` metadata. The
runner routes these records through `StateTransitionVerifier`, which applies an
in-memory world-model transition and checks the resulting postcondition before
falling back to static state or lexical verification:

```bash
python benchmarks/build_transition_fixture.py \
  --scores-output artifacts/order_transition_scores.json \
  --claims-output artifacts/order_transition_claims.json \
  --state-output artifacts/order_transition_state.json \
  --n-records 12

python benchmarks/eval_verifier_ensemble.py \
  --scores transitions=artifacts/order_transition_scores.json \
  --claims artifacts/order_transition_claims.json \
  --state-source artifacts/order_transition_state.json \
  --min-world-model-confidence 0.8 \
  --signal truth_proj \
  --json artifacts/order_transition_verifier_ensemble_report.json
```

This fixture checks action-consequence verification: true labels match the
predicted inventory after reservation, while false labels assert an off-by-one
postcondition that the predicted state refutes. `--min-world-model-confidence`
fails closed on low-confidence transition predictions and is recorded in the
report and verified-record trace-cache key.

The current policy is deliberately simple and auditable: `refuted` always
triggers, `supported` suppresses an internal trigger, and
`insufficient_evidence` preserves the internal trigger. The verifier and
retriever are dependency-free lexical baselines (`GroundednessVerifier`,
`SelfConsistencyVerifier`, `InMemoryRetriever`, and optional SQLite FTS
candidate retrieval), so results are only a controlled adapter test until a real
retrieval/verifier backend is plugged in. Claim fixtures may include
`selfcheck_samples` or `sampled_responses`; when lexical groundedness is
insufficient, the ensemble can use these caller-supplied alternative generations
for FactSelfCheck-style support/refutation rates before falling through to
retrieval.

`build_selfcheck_fixture.py` converts statement-bearing score dumps and sampled
generations into those fixtures. It reads `inside_sample_texts` from
`eval_truthfulqa.py --dump-inside-samples`, or one or more external JSON/JSONL
files passed with `--samples`. External samples may be keyed by `claim_id`,
provided as per-record objects with `index` / `claim_id`, or listed one row per
score-dump statement:

```bash
python benchmarks/build_selfcheck_fixture.py \
  --scores artifacts/tiny_scores_with_samples.json \
  --samples artifacts/external_sampled_generations.json \
  --output artifacts/tiny_selfcheck_claims.json

python benchmarks/eval_verifier_ensemble.py \
  --scores tiny=artifacts/tiny_scores_with_samples.json \
  --claims artifacts/tiny_selfcheck_claims.json \
  --signal truth_proj \
  --selfcheck-early-stop \
  --json artifacts/tiny_selfcheck_verifier_ensemble_report.json
```

When the question is whether sampled responses are useful as calibrated
diagnostic signals by themselves, `build_selfcheck_signal_score_dump.py` skips
the verifier sidecar and appends direct self-consistency columns to the score
dump:

```bash
python benchmarks/build_selfcheck_signal_score_dump.py \
  --scores artifacts/tiny_scores_with_samples.json \
  --samples artifacts/external_sampled_generations.json \
  --keep-signals truth_proj,maha_last,subspace_resid,eigenscore,nll_answer \
  --output artifacts/tiny_selfcheck_signal_scores.manifest.json \
  --output-format jsonl \
  --json artifacts/tiny_selfcheck_signal_report.json

python benchmarks/eval_score_ensemble.py \
  --scores tiny=artifacts/tiny_selfcheck_signal_scores.manifest.json \
  --signals truth_proj,selfcheck_support_rate,selfcheck_refute_rate,selfcheck_disagreement,selfcheck_not_applicable,selfcheck_best_overlap \
  --methods max_rank,mean_rank \
  --alphas 0.05,0.1,0.2 \
  --json artifacts/tiny_selfcheck_signal_ensemble_report.json
```

Use `run_selfcheck_signal_fusion_workflow.py` to run that direct selfcheck-signal
path in one reproducible no-model command, including enhanced score dumps,
ensemble report, optional geometry-by-selfcheck fusion artifacts, and artifact
manifest verification:

```bash
python benchmarks/run_selfcheck_signal_fusion_workflow.py \
  --scores tiny=artifacts/tiny_scores_with_samples.manifest.json \
  --samples artifacts/external_sampled_generations.json \
  --output-dir artifacts/tiny_selfcheck_signal_fusion \
  --keep-signals truth_proj,subspace_resid,eigenscore \
  --fusion-signals truth_proj,subspace_resid,eigenscore,selfcheck_support_rate,selfcheck_refute_rate,selfcheck_disagreement,selfcheck_best_overlap \
  --geometry-signals truth_proj,subspace_resid,eigenscore \
  --uncertainty-signals selfcheck_support_rate,selfcheck_refute_rate,selfcheck_disagreement,selfcheck_best_overlap \
  --alphas 0.05,0.1,0.2
```

The current SmolLM2 l20 direct selfcheck replay at
`artifacts/smollm2-l20-direct-selfcheck-signal-fusion/` is a negative result:
the exporter recovers 77/154 triggered records from the inside diagnostics cache,
but after alignment and deduplication only 17/154 records meet the two-sample
selfcheck threshold. The workflow's `sample-quality-report.json` therefore
fails the default gate: coverage `0.110`, average samples per record `0.416`,
and not-applicable rate `0.890`. At alpha 0.10, `truth_proj` remains best
(`AUROC 0.682`, detection `0.178`, false alarm `0.091`), while the best
geometry-by-selfcheck fusion reaches only `AUROC 0.561` and detection `0.096`.
Treat this as a sample-quality gate failure, not as evidence against
self-consistency with better sampled responses.

`--selfcheck-early-stop` is opt-in and preserves the default historical
benchmark behavior when omitted. When enabled, `SelfConsistencyVerifier` stops
judging samples once the finite sample budget can no longer change the final
support/refute/insufficient threshold outcome. The report records
`processed_samples`, `skipped_samples`, `early_stopped_records`, and
`processing_rate` under each run's `selfcheck_verifier` block. Use
`--selfcheck-max-samples` to cap per-claim sample use when comparing verifier
latency/cost budgets.

Reports include `verification_quality`, a label-conditioned matrix over
`supported` / `refuted` / `insufficient_evidence` outcomes. Use
`true_supported_rate`, `false_refuted_rate`, `decision_accuracy`, and
`decision_error_rate` to evaluate evidence fixture quality separately from the
final control-policy detection and false-alarm rates. Reports also include
`route_summary`, which breaks verification outcomes down by selected route
(`structured_qa`, `state_transition`, `structured_state`, `groundedness`,
`triple_evidence`, `self_consistency`, `retrieval_groundedness`,
`retrieval_structured_qa`, or `staged_skip`) and records attempted-route
counts, status counts, and per-route supported/refuted/error rates. Use
`route_quality` for label-conditioned false-support / false-refutation metrics
per selected route, and use each alpha result's `route_control_impact` to see
how that route changed internal false alarm, detection, suppression, and
rescued-detection rates. New reports also include route-level
`p95_duration_seconds` and `p99_duration_seconds` tail-latency fields for
promotion gates.

## `refresh_verifier_route_artifacts.py`

Regenerates current-schema verifier route reports from saved score dumps,
claims, and local verifier corpora without rerunning model forward passes. Use
this when older committed reports lack `route_quality`, `cache_stats`, or
tail-latency fields required by route promotion gates.

```bash
python benchmarks/refresh_verifier_route_artifacts.py \
  --scores qwen=artifacts/qwen05_truthfulqa_l80_scores_with_statements.json \
  --scores smol=artifacts/smollm2_truthfulqa_l80_scores_with_statements.json \
  --qa-corpus artifacts/truthfulqa_l80_correct_answer_corpus.json \
  --signal truth_proj \
  --alphas 0.1 \
  --repeats 5 \
  --verifier-report-json artifacts/truthfulqa_l80_structured_qa_verifier_ensemble_report_v2.json \
  --promotion-json artifacts/qwen_smol_structured_qa_promotion_workflow.json \
  --route-report-json artifacts/qwen_smol_structured_qa_route_comparison.json \
  --gate-route structured_qa \
  --gate-min-selected 500 \
  --min-decision-accuracy 0.99 \
  --max-false-supported-rate 0.0 \
  --min-false-refuted-rate 0.99 \
  --max-mean-duration-seconds 0.001 \
  --max-p95-duration-seconds 0.001 \
  --max-p99-duration-seconds 0.001 \
  --max-max-duration-seconds 0.005 \
  --max-mean-attempted-route-count 1.1 \
  --max-retrieval-use-rate 0.0 \
  --fail-on-blocked
```

Add `--registry`, `--baseline-*`, `--candidate-profile`, and
`--max-total-ratio` flags when the refresh should also run the same registry
baseline gate used by `run_adapter_promotion_workflow.py`.
The refresh workflow summary mirrors the route promotion metrics needed for
audit, including mean/p95/p99/max duration, attempted-route count, retrieval use
rate, and compact cache hit/miss/request stats.
Add `--compact-json` to write minified verifier, route-comparison, promotion,
and workflow JSON artifacts.

For structured state or database-like adapters, use the same refresh workflow
with `--claims` and `--state-source` so promotion gates cover the actual
state-check route rather than only open-domain QA evidence:

```bash
python benchmarks/build_domain_state_fixture.py \
  --scores-output artifacts/order_fulfillment_scores.json \
  --claims-output artifacts/order_fulfillment_claims.json \
  --state-output artifacts/order_fulfillment_state.json \
  --sqlite-output artifacts/order_fulfillment_state.db \
  --sqlite-state-source-output artifacts/order_fulfillment_sqlite_state_source.json \
  --n-records 12

python benchmarks/refresh_verifier_route_artifacts.py \
  --scores orders=artifacts/order_fulfillment_scores.json \
  --claims artifacts/order_fulfillment_claims.json \
  --state-source artifacts/order_fulfillment_sqlite_state_source.json \
  --signal truth_proj \
  --alphas 0.2 \
  --repeats 1 \
  --verifier-report-json artifacts/order_fulfillment_state_verifier_report.json \
  --promotion-json artifacts/order_fulfillment_state_promotion_workflow.json \
  --route-report-json artifacts/order_fulfillment_state_route_comparison.json \
  --gate-route structured_state \
  --gate-min-selected 12 \
  --min-decision-accuracy 1.0 \
  --max-false-supported-rate 0.0 \
  --min-false-refuted-rate 1.0 \
  --max-mean-duration-seconds 0.05 \
  --max-p99-duration-seconds 0.20 \
  --max-max-duration-seconds 0.50 \
  --max-mean-attempted-route-count 1.1 \
  --max-retrieval-use-rate 0.0 \
  --fail-on-blocked
```

For action-conditioned world-model/postcondition checks, use the transition
fixture and gate `state_transition`:

```bash
python benchmarks/build_transition_fixture.py \
  --scores-output artifacts/order_transition_scores.json \
  --claims-output artifacts/order_transition_claims.json \
  --state-output artifacts/order_transition_state.json \
  --n-records 12

python benchmarks/refresh_verifier_route_artifacts.py \
  --scores transitions=artifacts/order_transition_scores.json \
  --claims artifacts/order_transition_claims.json \
  --state-source artifacts/order_transition_state.json \
  --signal truth_proj \
  --alphas 0.2 \
  --repeats 1 \
  --verifier-report-json artifacts/order_transition_verifier_report.json \
  --promotion-json artifacts/order_transition_promotion_workflow.json \
  --route-report-json artifacts/order_transition_route_comparison.json \
  --gate-route state_transition \
  --gate-min-selected 12 \
  --min-decision-accuracy 1.0 \
  --max-false-supported-rate 0.0 \
  --min-false-refuted-rate 1.0 \
  --max-mean-duration-seconds 0.05 \
  --max-p99-duration-seconds 0.20 \
  --max-max-duration-seconds 0.50 \
  --max-mean-attempted-route-count 1.1 \
  --max-retrieval-use-rate 0.0 \
  --compact-json \
  --fail-on-blocked
```

## `compare_verifier_routes.py`

Aggregates `route_quality` and per-alpha `route_control_impact` from one or more
`eval_verifier_ensemble.py` JSON reports. This is a no-model post-processing step
for deciding which verifier routes deserve real adapter work.

```bash
python benchmarks/compare_verifier_routes.py \
  --report qwen=artifacts/qwen05_verifier_ensemble_report.json \
  --report smol=artifacts/smollm2_verifier_ensemble_report.json \
  --alpha 0.1 \
  --gate-route structured_state \
  --min-decision-accuracy 0.90 \
  --max-false-supported-rate 0.05 \
  --min-false-refuted-rate 0.80 \
  --max-mean-duration-seconds 0.05 \
  --max-p95-duration-seconds 0.10 \
  --max-p99-duration-seconds 0.20 \
  --max-mean-attempted-route-count 1.5 \
  --min-staged-skip-rate 0.30 \
  --max-staged-verified-false-alarm 0.10 \
  --min-staged-verified-detection 0.80 \
  --fail-on-promotion \
  --fail-on-gate \
  --json artifacts/verifier_route_comparison.json
```

The output includes:

- `leaderboard`: individual report/run/route rows sorted by decision accuracy,
  false-refutation rate, low false-support rate, verified detection, and low
  verified false alarm, then lower mean route duration.
- `by_route`: aggregate counts and weighted rates across all reports for each
  route, useful for comparing route families such as `structured_qa`,
  `structured_state`, and `state_transition`. New verifier-ensemble reports
  include cost fields such as `mean_duration_seconds`,
  `p95_duration_seconds`, `p99_duration_seconds`, `max_duration_seconds`,
  `mean_attempted_route_count`, and `retrieval_use_rate`. Aggregate cost means
  use only the observation counts whose paired total metric is finite; discarded
  source entries are exposed through `invalid_metric_counts`.
- `cache_summary`: aggregate report-level cache hit/miss/request totals across
  compared runs. This is reported separately from route metrics because cache
  hits are global to the benchmark run rather than safely attributable to one
  selected route.
- `staged_verification`: aggregate skip-rate and alpha-level verified control
  metrics from `eval_verifier_ensemble.py --staged-verification` reports.
  Gate flags such as `--min-staged-skip-rate`,
  `--max-staged-verified-false-alarm`, and
  `--min-staged-verified-detection` fail closed when staged metrics are missing
  or when verifier skipping saves cost at unacceptable quality loss.
- `pareto_frontier`: aggregate route candidates that are not dominated by
  another route across quality, control-impact, sample-size, and cost metrics.
  The `recommended` entry is a deterministic quality/cost ordering over the
  frontier, not a substitute for the fail-closed gate.
- `promotion_decision`: route-specific adapter promotion status. It reports
  `promote` only when the recommended Pareto route is covered by the gate and
  has no route-specific blocking failures; otherwise it records the missing
  gate, unchecked route, or failed metric evidence.
- `quality_gate`: optional fail-closed adapter promotion gate when any
  `--gate-*` or threshold flag is set. It checks aggregate route metrics such
  as `decision_accuracy`, `false_supported_rate`, `false_refuted_rate`,
  `verified_false_alarm`, `verified_detection`, `mean_duration_seconds`,
  `p95_duration_seconds`, `p99_duration_seconds`, `max_duration_seconds`,
  `mean_attempted_route_count`, and `retrieval_use_rate`, plus optional global
  `cache_hit_rate` and staged-verification skip/quality thresholds; missing
  routes, missing metrics, non-finite values, missing cache/staged evidence, or
  no eligible routes fail the gate. Use `--require-non-oracle-evidence` on
  `compare_route_baselines.py` when a retrieval route must prove that generated
  claims omit labels, labels were not used for retrieval, and
  `input_provenance` fingerprints the score dump and local corpora. When
  multiple reports are
  aggregated, any route entry with missing or non-finite source metrics for an
  enabled gate records `invalid_metric_counts` and blocks promotion even if the
  remaining entries produce an aggregate value below the threshold.
- `rows`: the unaggregated route entries for audit and follow-up slicing.

Use `--min-selected` to keep tiny route samples out of the leaderboard while
still preserving them in the raw `rows` and `by_route` sections.
Use `--gate-min-selected` to set a stricter sample floor for the promotion gate
than the display leaderboard, and use `--fail-on-gate` in local or CI smoke
checks when a route must meet minimum quality before real adapter work proceeds.
Use `--fail-on-promotion` when CI should fail unless the final route-specific
promotion decision is `promote`.

## `run_adapter_promotion_workflow.py`

Runs the route-promotion handoff as one fail-closed workflow. It generates a
`compare_verifier_routes.py` report, requires the route-level
`promotion_decision` to be `promote`, and can optionally run a registered
same-machine performance baseline comparison before returning a final adapter
promotion decision.

```bash
python benchmarks/run_adapter_promotion_workflow.py \
  --report qwen=artifacts/qwen05_verifier_ensemble_report.json \
  --route-report-json artifacts/qwen05_route_comparison.json \
  --gate-route structured_state \
  --min-decision-accuracy 0.90 \
  --max-false-supported-rate 0.05 \
  --min-false-refuted-rate 0.80 \
  --max-mean-duration-seconds 0.05 \
  --max-p95-duration-seconds 0.10 \
  --max-p99-duration-seconds 0.20 \
  --max-mean-attempted-route-count 1.5 \
  --registry artifacts/registry.json \
  --baseline-name qwen05-profile-rescore \
  --baseline-version 0.3 \
  --baseline-profile-artifact \
    cells.layer_m12_batch_1_capture_outputs.triplet_manifest::profiles.uncached \
  --candidate-profile candidate=/tmp/eigentruth-current-profile.json \
  --max-total-ratio 1.10 \
  --json artifacts/qwen05_adapter_promotion_workflow.json \
  --artifact-manifest artifacts/qwen05_adapter_promotion_manifest.json \
  --fail-on-blocked
```

Add `--compact-json` when route-comparison and workflow reports are produced for
automation rather than manual review.

The final report includes:

- `route_comparison_path` and embedded `route_comparison` for route-quality
  audit.
- `artifact_manifest` when `--artifact-manifest` is provided. This manifest
  fingerprints the route-comparison report, source verifier reports, and
  candidate profiles so a promoted route baseline can be registered and
  rechecked without rerunning verification.
- `registry_baseline_comparison` when candidate profiles are provided.
- `decision`: final workflow status. It is `promote` only when route promotion
  passes and every configured registry baseline gate passes. Missing route
  gates, failed route gates, missing registry gates, or failed registry gates
  produce `blocked` plus explicit `blocking_reasons`.

## `run_adapter_promotion_registry_workflow.py`

Runs route promotion and registry promotion in one command. It writes the route
comparison report, writes the registry-ready artifact manifest, recursively
verifies that manifest, and records it in `ArtifactRegistry` only when the
adapter promotion decision is `promote` by default.

```bash
python benchmarks/run_adapter_promotion_registry_workflow.py \
  --report qwen=artifacts/qwen05_verifier_ensemble_report.json \
  --route-report-json artifacts/qwen05_route_comparison.json \
  --artifact-manifest artifacts/qwen05_adapter_promotion_manifest.json \
  --registry artifacts/registry.json \
  --name qwen05-route-structured-state \
  --version 0.6 \
  --gate-route structured_state \
  --min-decision-accuracy 0.90 \
  --max-false-supported-rate 0.05 \
  --min-false-refuted-rate 0.80 \
  --max-mean-duration-seconds 0.05 \
  --max-p99-duration-seconds 0.20 \
  --max-mean-attempted-route-count 1.5 \
  --json artifacts/qwen05_adapter_promotion_registry_workflow.json \
  --fail-on-blocked
```

Use `--baseline-registry` with `--candidate-profile` and the same performance
gate flags as `run_adapter_promotion_workflow.py` when the route promotion must
also compare the current profile against a registered same-machine baseline.
`--registry` is always the destination registry for the promoted route manifest.

## `compare_route_baselines.py`

Compares registered route-promotion manifests from `ArtifactRegistry` without
rerunning models or verifier adapters. Use it after promoting one or more
`run_adapter_promotion_workflow.py --artifact-manifest` outputs with
`promote_artifact_manifest.py`.

```bash
python benchmarks/compare_route_baselines.py \
  --registry artifacts/registry.json \
  --min-selected 100 \
  --min-decision-accuracy 0.95 \
  --max-false-supported-rate 0.02 \
  --min-false-refuted-rate 0.90 \
  --max-p99-duration-seconds 0.20 \
  --max-mean-attempted-route-count 1.5 \
  --max-retrieval-use-rate 0.2 \
  --max-runtime-total-seconds 60 \
  --max-retrieval-hit-count 1000 \
  --min-claims-cache-hit-rate 0.9 \
  --min-verifier-trace-cache-hit-rate 0.9 \
  --require-retrieval-stress-control \
  --retrieval-stress-manifest artifacts/truthfulqa-l80-answer-echo-retrieval-stress/artifact-manifest.json \
  --min-stress-false-supported-rate 0.90 \
  --max-stress-false-refuted-rate 0.05 \
  --json artifacts/route-baseline-comparison.json \
  --fail-on-blocked
```

The comparison recursively verifies each registered manifest by default, reloads
the saved `route_comparison_report`, or accepts a covered-facts `route_summary`
manifest from `run_wikidata_structured_qa_route_workflow.py` when no route
comparison report is present. It fails closed on non-promoted route decisions or
`invalid_metric_counts`, and recommends the passing baseline with the best
quality/cost ordering. Optional runtime-budget flags read
`runtime_total_seconds`, `runtime_n_retrieval_hits`, claims-cache metadata, and
verifier-trace-cache metadata from the route manifest or registry record; when a
threshold is configured, missing or non-finite evidence blocks that baseline.
Use `--require-retrieval-stress-control` for retrieval-grounding baselines. It
requires an answer-echo stress artifact manifest, verifies that manifest, checks
the corpus type is `retrieval_stress_answer_echo`, and fails closed unless the
stress run exposes self-support with high `false_supported_rate` and low
`false_refuted_rate`. This prevents answer-derived retrieval evidence from being
promoted as grounding.

## `run_adapter_family_matrix.py`

Builds deterministic local fixtures for the front-line adapter families and
runs each through the same refresh/promotion gate:

- `structured_qa`: exact question/answer facts.
- `structured_state`: static business/domain state checks.
- `state_transition`: action-conditioned world-model postconditions.
- `retrieval_groundedness`: optional local retrieval evidence plus lexical
  support/refutation checks.
- `retrieval_structured_qa`: optional local retrieval evidence interpreted as
  structured question/answer facts before lexical fallback.
- `triple_evidence`: optional strict subject-predicate-object evidence coverage
  audits for sensitive factual claims.

It then aggregates the generated verifier reports with
`compare_verifier_routes.py` so quality, tail latency, attempted-route count,
retrieval use, cache stats, and promotion status can be compared in one matrix.

```bash
python benchmarks/run_adapter_family_matrix.py \
  --output-dir artifacts/adapter_family_matrix \
  --json artifacts/adapter_family_matrix/report.json \
  --n-records 8 \
  --alpha 0.2 \
  --compact-json \
  --fail-on-blocked
```

By default the matrix keeps retrieval disabled so cold-start quality gates stay
strict about zero retrieval cost. To include the local retrieval fixture, enable
it explicitly and set route-count/retrieval-use gates for the expected cost:

```bash
python benchmarks/run_adapter_family_matrix.py \
  --output-dir artifacts/adapter_family_matrix_retrieval \
  --json artifacts/adapter_family_matrix_retrieval/report.json \
  --n-records 8 \
  --alpha 0.2 \
  --include-retrieval \
  --max-mean-attempted-route-count 2.1 \
  --max-retrieval-use-rate 1.0 \
  --compact-json \
  --fail-on-blocked
```

The structured retrieval-QA route uses retrieved documents as local
question/answer facts before lexical fallback:

```bash
python benchmarks/run_adapter_family_matrix.py \
  --output-dir artifacts/adapter_family_matrix_retrieval_qa \
  --json artifacts/adapter_family_matrix_retrieval_qa/report.json \
  --n-records 8 \
  --alpha 0.2 \
  --include-retrieval-structured-qa \
  --max-mean-attempted-route-count 2.1 \
  --max-retrieval-use-rate 1.0 \
  --compact-json \
  --fail-on-blocked
```

The strict triple-evidence route audits whether local evidence covers the
subject, predicate, and object slots for sensitive factual claims. Missing slot
coverage returns insufficient evidence rather than a direct refutation, so use a
false-supported gate and disable the false-refuted requirement for this audit
family:

```bash
python benchmarks/run_adapter_family_matrix.py \
  --output-dir artifacts/adapter_family_matrix_triple_evidence \
  --json artifacts/adapter_family_matrix_triple_evidence/report.json \
  --n-records 8 \
  --alpha 0.2 \
  --include-triple-evidence \
  --min-false-refuted-rate 0.0 \
  --max-false-supported-rate 0.0 \
  --max-mean-attempted-route-count 1.1 \
  --max-retrieval-use-rate 0.0 \
  --compact-json \
  --fail-on-blocked
```

## `run_adapter_readiness_workflow.py`

Combines the deterministic adapter-family quality matrix with the same-machine
cache-profile performance matrix. The final `readiness_decision` is `promote`
only when both the adapter-family `promotion_decision` and the performance
`matrix_decision` promote and the performance report can produce deployable
runtime settings.
Pass `--include-retrieval --max-mean-attempted-route-count 2.1
--max-retrieval-use-rate 1.0` when the readiness matrix should also prove the
local retrieval-groundedness adapter family. Use
`--include-retrieval-structured-qa` with the same route-count/retrieval-use
gates when readiness should also prove retrieved structured question/answer
facts. Use `--include-triple-evidence --min-false-refuted-rate 0.0` when
readiness should also require the strict triple-evidence audit family; this
keeps the audit gated on zero false support without treating insufficient
evidence as a refutation.

Use `--performance-dry-run` to inspect the performance commands without loading
a model. Dry-run performance evidence produces `needs_performance_evidence`,
not `promote`.

```bash
python benchmarks/run_adapter_readiness_workflow.py \
  --output-dir artifacts/adapter_readiness \
  --json artifacts/adapter_readiness/report.json \
  --n-records 8 \
  --alpha 0.2 \
  --shared-cache-dir artifacts/adapter_readiness/cache \
  --layers=-12 \
  --batch-sizes=1,2 \
  --hidden-state-captures=outputs \
  --performance-dry-run \
  --compact-json
```

Remove `--performance-dry-run` only when the local profile matrix cost is
acceptable. Add `--fail-on-blocked` on real runs to require
`readiness_decision.status=promote`. Pass `--inside-sampling-report` when a
promoted `run_inside_sampling_profile.py` comparison should be folded into the
runtime recommendation and readiness manifest. Pass
`--inside-trigger-budget-sweep-report` when a promoted
`run_inside_trigger_budget_sweep.py` report should provide the trigger budget;
derived top-fraction reports preserve their `--derive-from-max-budget`
recommendation in the generated benchmark flags. Pass `--performance-report` to
reuse an existing `cache-profile-matrix-report.json` when only INSIDE sampling
evidence or adapter-family evidence changed:

```bash
  --performance-report artifacts/readiness/cache-profile-matrix/cache-profile-matrix-report.json \
  --inside-sampling-report artifacts/inside_sampling/inside-sampling-profile-comparison.json \
  --inside-trigger-budget-sweep-report artifacts/inside_trigger_budget_sweep/inside-trigger-budget-sweep.json
```

The workflow also writes a top-level `artifact-manifest.json` that fingerprints
the readiness report, adapter-family matrix, route-comparison report,
cache-profile matrix report, nested cache-profile matrix manifest, and
`runtime-recommendation.json`, plus the optional INSIDE sampling profile report
and trigger-budget sweep report when provided. The runtime recommendation is generated from the saved
performance matrix without rerunning model work; when the matrix promotes it
includes deployable layer, batch-size, token-budget, prefix-KV, worker flags,
all available AUROC quality signals, optional sampling settings, and the best
quality signal for the next run. Use `verify_artifact_manifest.py --recursive` and
`promote_artifact_manifest.py` on that manifest to register a readiness
baseline.

To run readiness and register the verified manifest in one command, use
`run_adapter_readiness_registry_workflow.py`. It promotes only when
`readiness_decision.status=promote` unless `--allow-non-promote` is explicitly
set. The registry workflow accepts the same adapter-family inclusion flags as
the readiness workflow, including `--include-retrieval`,
`--include-retrieval-structured-qa`, and `--include-triple-evidence`, and records
the generated retrieval/audit route families in registry metadata:

```bash
python benchmarks/run_adapter_readiness_registry_workflow.py \
  --output-dir artifacts/adapter_readiness \
  --registry artifacts/registry.json \
  --name qwen05-readiness-local \
  --version 0.5 \
  --shared-cache-dir artifacts/adapter_readiness/cache \
  --layers=-12 \
  --batch-sizes=1,2 \
  --hidden-state-captures=outputs \
  --max-runtime-total-seconds 900 \
  --fail-on-blocked
```

After two or more readiness manifests are registered, compare them without
rerunning model work:

```bash
python benchmarks/compare_readiness_baselines.py \
  --registry artifacts/registry.json \
  --min-best-quality-auroc 0.60 \
  --max-uncached-forward-seconds 40 \
  --max-covariance-maha-last-auroc-drop 0.05 \
  --max-inside-sample-count-ratio 0.60 \
  --max-inside-generation-seconds-ratio 0.80 \
  --json artifacts/readiness-baseline-comparison.json \
  --fail-on-blocked
```

The comparison verifies each registered readiness manifest recursively, reloads
the saved performance matrix to recover all available AUROC quality signals for
older records, reloads saved INSIDE sampling profile artifacts when present,
applies optional quality/performance/sampling-cost gates, and recommends the
passing baseline with the best quality signal, breaking ties by lower
forced-answer forward cost, lower cache-only time, and lower sampling ratios.
If `--max-covariance-maha-last-auroc-drop` is set, the selected covariance mode
must include `covariance_tradeoff` evidence and keep selected `maha_last` AUROC
within the configured drop versus the full-covariance baseline.
If `--max-inside-sample-count-ratio` or
`--max-inside-generation-seconds-ratio` is set, candidates without readable
sampling evidence fail closed. For legacy matrix reports that predate
forced-answer phase timing, the uncached total time is used as a conservative
forward-cost fallback and reported as
`uncached_forward_cost_source=uncached_total_seconds_fallback`.
Use `--max-runtime-total-seconds` on the readiness workflow or registry workflow
when end-to-end readiness wall clock time itself is part of the promotion budget.

## `compare_release_candidates.py`

Combines the recommended readiness baseline and verifier-route baseline into one
fail-closed release candidate. It does not rerun model work, verifier adapters,
or promotion workflows; it reloads the already registered manifests through
`compare_readiness_baselines.py` and `compare_route_baselines.py`, then emits the
deployable runtime flags, verifier route, quality summary, runtime cost, and
route cost in one report.

```bash
python benchmarks/compare_release_candidates.py \
  --readiness-registry artifacts/registry.json \
  --route-registry artifacts/registry.json \
  --performance-registry artifacts/registry.json \
  --performance-baseline-key performance_baseline:smollm2-l20-performance-baseline:0.9 \
  --performance-drift-baseline-key performance_baseline:smollm2-l20-performance-baseline:0.8 \
  --product-trace-replay-workflow-key report:smollm2-product-trace-replay-workflow:0.1 \
  --feedback-policy-workflow-key report:<feedback-policy-workflow-name>:<version> \
  --feedback-policy-min-matched-feedback-count 20 \
  --feedback-policy-min-safety-coverage 0.70 \
  --feedback-policy-max-unknown-safety-issue-rate 0.20 \
  --release-efficiency-report artifacts/product-runtime-profile-sweep/release-efficiency-report.json \
  --route-baseline-key benchmark_manifest:truthfulqa-l80-structured-qa-staged-route:0.4 \
  --required-route-baseline-key benchmark_manifest:<local-retrieval-route-name>:<version> \
  --adapter-family-matrix artifacts/adapter_family_matrix/adapter-family-matrix.json \
  --adapter-family-profile strict_audit \
  --runtime-profile balanced \
  --min-best-quality-auroc 0.60 \
  --max-uncached-forward-seconds 40 \
  --max-covariance-maha-last-auroc-drop 0.05 \
  --min-selected 100 \
  --min-decision-accuracy 0.95 \
  --max-false-supported-rate 0.02 \
  --min-false-refuted-rate 0.90 \
  --max-p99-duration-seconds 0.20 \
  --max-runtime-total-seconds 60 \
  --max-retrieval-hit-count 1000 \
  --min-claims-cache-hit-rate 0.9 \
  --min-verifier-trace-cache-hit-rate 0.9 \
  --require-performance-score-dump-cache \
  --min-performance-score-dump-cache-jsonl-view-hit-rate 0.5 \
  --max-performance-uncached-total-seconds-ratio 1.15 \
  --max-performance-cached-total-seconds-ratio 1.15 \
  --max-performance-cache-only-total-seconds-ratio 1.15 \
  --max-performance-score-dump-cache-jsonl-view-hit-rate-drop 0.10 \
  --required-route-max-runtime-total-seconds 120 \
  --required-route-max-retrieval-hit-count 5000 \
  --required-route-max-retrieval-use-rate 1.0 \
  --required-route-require-non-oracle-evidence \
  --required-route-require-retrieval-stress-control \
  --required-route-retrieval-stress-manifest artifacts/truthfulqa-l80-answer-echo-retrieval-stress/artifact-manifest.json \
  --required-route-min-stress-false-supported-rate 0.90 \
  --required-route-max-stress-false-refuted-rate 0.05 \
  --json artifacts/release-candidate-comparison.json \
  --fail-on-blocked
```

Use explicit `--readiness-baseline-key` and `--route-baseline-key` values when a
release should be constrained to named registry records. Omit `--route-registry`
when readiness and route manifests are stored in the same local registry file.
Readiness, release, and route comparison workflows reuse a process-local JSON
artifact cache while loading manifests and reports; cache entries are keyed by
path, mtime, size, and inode so edited artifacts are reloaded in the same
process. Reports include `summary.artifact_json_cache` with requests, hits,
misses, errors, entries, and hit rate so large registry sweeps can audit whether
artifact IO is being reused.
Repeat `--required-route-baseline-key` when the release should also require
additional promoted route baselines, such as a real local-corpus
`retrieval_groundedness` baseline, without making that route the selected
low-latency product route. This gate verifies that each required route manifest
is promoted and recursively valid. Add `--required-route-*` thresholds when the
audit route needs its own quality, latency, retrieval-hit, or cache-reuse budget;
add `--required-route-require-non-oracle-evidence` when that required route must
also prove labels stayed only in the score dump and local input provenance is
present. Add `--required-route-require-retrieval-stress-control` when the route
must also prove the answer-echo negative control fails as expected. Otherwise the
release only checks the route's already-registered promotion status and manifest
validity. This keeps selected product-route budgets such as
`--max-retrieval-use-rate 0.0` separate from audit routes that intentionally use
retrieval or world-model adapters. For `structured_fact`, use two required route
keys, or `--release-policy-profile strict_structured_fact` with
`--structured-fact-canonical-route-key` and
`--structured-fact-paraphrase-route-key`, to require both the canonical
covered-facts route and the paraphrase robustness replay before a release can
promote. Available release policy profiles are `research_smoke`,
`candidate_release`, and `strict_structured_fact`; profile defaults only fill
unset values, so explicit thresholds still win. Direct
`compare_release_candidates.py` reports record `release_policy_profile` and
`release_policy_profile_applied_defaults` in `config`.
Add `--performance-baseline-key performance_baseline:<name>:<version>` when the
final candidate must match a registered performance handoff. The comparison
verifies that performance baseline manifest, reloads its runtime recommendation,
and fails closed when layer, batch size, capture mode, token budget, prefix cache,
worker count, trigger budget, trigger policy, or best quality signal differ from
the readiness-selected runtime. Newer performance baseline workflow reports also
carry `performance_evidence_bundle`; when present, the release gate requires
`release_ready=true` and copies its recommendation cost/readiness summary into
the release candidate and downstream promotion contract metadata. Add
`--require-performance-score-dump-cache` when the release should fail closed
unless that bundle contains score-dump cache evidence, and add
`--min-performance-score-dump-cache-jsonl-view-hit-rate` to require a minimum
selected JSONL view cache hit rate for post-hoc analysis reuse. Omit
`--performance-registry` when the performance record lives in the readiness
registry.
Add `--performance-drift-baseline-key performance_baseline:<prior-name>:<prior-version>`
with any `--max-performance-*-ratio` or
`--max-performance-score-dump-cache-jsonl-view-hit-rate-drop` thresholds when a
release must prove that the selected performance bundle has not regressed
against an explicit prior handoff. The gate compares `uncached_total_seconds`,
`cached_total_seconds`, `cache_only_total_seconds`, and selected JSONL
score-dump cache hit rate from the two `performance_evidence_bundle` payloads.
It fails closed if the reference bundle is unverified, not `release_ready`, or
missing a checked metric.
Add `--selector-replay-report` when the final candidate must also include a
promoted runtime-profile selector replay over saved `ProductTrace` payloads.
The gate verifies the replay artifact manifest, requires `status=promote`, and
carries the recommended selector plus observed selected-vs-original runtime
delta metrics into the release candidate. This makes request-time auto-profile
changes part of the same fail-closed release evidence instead of a separate
benchmark note.
Add `--product-trace-replay-workflow-key report:<name>:<version>` when selector
replay and product-runtime drift were produced and registered by
`run_product_trace_replay_workflow.py`; use `--product-trace-replay-workflow`
for an unregistered local JSON file. The release gate verifies the workflow
manifest, requires the workflow itself to promote, and uses its child
selector-replay and runtime-drift report paths unless explicit
`--selector-replay-report` or `--product-runtime-drift-report` values are also
provided. The key form defaults to `--readiness-registry`; pass
`--product-trace-replay-workflow-registry` only when the workflow record lives
elsewhere. This is the safer default for release promotion because the release
candidate consumes the same raw-trace handoff artifact that generated the child
reports.
Add `--feedback-policy-workflow-key report:<name>:<version>` when feedback-derived
control policy evidence was produced and registered by
`run_feedback_policy_workflow.py`; use `--feedback-policy-workflow` for an
unregistered local JSON file. The release gate verifies the workflow artifact
manifest, requires `workflow=feedback_policy_workflow`, accepts top-level
`status=recommend` or `status=observed`, and requires a valid promotion decision
(`promote_candidate_policy` or `keep_current_policy`). A `recommend` workflow
must include candidate control-policy/default artifacts and a valid inline
`candidate_control_policy_config` that can be parsed as `ControlPolicyConfig`. Use
`--feedback-policy-workflow-registry` only when the workflow record lives outside
`--readiness-registry`. Optional thresholds
`--feedback-policy-min-matched-feedback-count`,
`--feedback-policy-min-safety-coverage`, and
`--feedback-policy-max-unknown-safety-issue-rate` fail closed on weak feedback
coverage or unresolved safety issues.
Add `--product-runtime-drift-report` when the final candidate must also prove
that a fresh `ProductTrace` runtime baseline has not drifted from a registered
or file-based product runtime baseline. The gate verifies the drift report
manifest, requires `status=promote`, and carries baseline/current paths plus
blocked-metric counts into the release candidate. This connects captured product
traffic replay back into the same release gate as model, route, and selector
evidence.
Add `--release-efficiency-report` when the final candidate must also prove that
the product runtime profile sweep has a promoted efficiency handoff. The gate
verifies the release-efficiency manifest, requires `workflow=release_efficiency_report`
and promoted report/decision status, and carries the recommended runtime profile,
efficiency score, quality summary, trace-record cache summary, and manifest path
into the release candidate. Downstream `ProductPromotionContract` exports can
then inherit the recommended runtime profile directly from the promoted
candidate.
Add `--adapter-family-matrix` when release should also require a promoted
adapter-family matrix from `run_adapter_family_matrix.py`. Repeat
`--required-adapter-route` for routes that must be present and promoted in that
matrix, such as `structured_state`, `state_transition`, or
`retrieval_groundedness`, or use `--adapter-family-profile strict_audit` to
require `structured_state`, `state_transition`, and `triple_evidence` together.
This keeps retrieval/database/world-model/audit adapter work inside the same
fail-closed release gate instead of treating it as a separate benchmark note.
Release-candidate runtime-budget flags are delegated to the route-baseline
comparison, so the final release blocks when the selected route baseline exceeds
the configured total runtime, retrieval-hit, or cache-reuse budgets.
Readiness-side INSIDE sampling gates are delegated to
`compare_readiness_baselines.py`, so the final release also blocks when the
selected runtime lacks sampling profile evidence or exceeds the configured
sample-count/generation-time ratios.
The same `--max-covariance-maha-last-auroc-drop` gate applies to the selected
readiness runtime and to a supplied performance baseline's runtime
recommendation, so a faster covariance mode cannot silently replace
full-covariance `maha_last` quality evidence.
Use `--runtime-profile latency`, `balanced`, or `audit` to fill unset
runtime/cost defaults. `latency` selects the cost-first trigger budget and
tighter sampling/route-cost gates, `balanced` selects the quality-balanced
trigger budget with moderate cost gates, and `audit` selects the highest
measured INSIDE-quality trigger budget with looser cost ceilings. Explicit CLI
flags override profile defaults. Quality gates such as
`--min-best-quality-auroc` remain explicit because they are model/data specific.
When readiness evidence includes a trigger-budget sweep, omit
`--inside-trigger-budget-policy` to use the policy already recorded in the
readiness manifest or runtime recommendation. Pass `cost_first`,
`quality_balanced`, or `quality_first` to force a release-gate policy override
without rerunning model or INSIDE generation work.

To write, verify, and register that release candidate as its own manifest, use
`run_release_candidate_registry_workflow.py`. It accepts the same
`--required-route-baseline-key`, `--product-trace-replay-workflow-key`,
`--product-trace-replay-workflow`, `--feedback-policy-workflow-key`,
`--feedback-policy-workflow`, `--feedback-policy-workflow-registry`,
feedback-policy threshold options, `--selector-replay-report`,
`--product-runtime-drift-report`, and `--release-efficiency-report` options and
includes those route/workflow/feedback-policy/selector/drift/efficiency manifests in the final release-candidate manifest
when the gate promotes. Required-route budget settings are also copied into
manifest metadata as `required_route_budget_policy`, including
`--required-route-require-non-oracle-evidence` when the audit route must prove
label-free local retrieval claims.
Use `--require-structured-fact-robustness` with
`--structured-fact-canonical-route-key` and
`--structured-fact-paraphrase-route-key` when the release must carry both
canonical and paraphrase `structured_fact` covered-facts evidence. The workflow
adds those two records to the required-route gate and records
`structured_fact_robustness_*` fields in the comparison report, final manifest,
and release registry metadata.
Use `--release-policy-profile` with the registry workflow to reuse the same
named defaults while registering the promoted manifest. `strict_structured_fact`
enables the structured-fact robustness requirement, requires both configured
canonical/paraphrase route keys, applies the baseline candidate quality gates,
and adds stricter route/required-route quality thresholds for covered-fact
release evidence. The workflow records `release_policy_profile` and
`release_policy_profile_applied_defaults` in the comparison report, final
manifest, and registry metadata.
It also forwards `--max-covariance-maha-last-auroc-drop` to the underlying
readiness and performance-baseline covariance tradeoff gates. Add
`--fingerprint-cache` for repeated local release checks so recursive manifest
verification can reuse unchanged file/directory fingerprints across runs
without changing gate semantics. The workflow also shares one JSON artifact
cache and its stats across compare, manifest build, and promotion verification,
so `artifact_cache.artifact_json_cache` reflects cache reuse from the full
release gate. Add `--artifact-json-cache` to persist that JSON cache across
repeated local release checks; stale entries are keyed by path signatures,
ignored when artifacts change, and pruned for the same path on save. Add
`--manifest-fingerprint-workers N` when large local manifests spend meaningful
time hashing independent artifacts; the setting is passed through compare-time
manifest gates, release-manifest build verification, and promotion verification.
The default is `1`, so existing release checks remain serial unless explicitly
configured.

```bash
python benchmarks/run_release_candidate_registry_workflow.py \
  --readiness-registry artifacts/registry.json \
  --route-registry artifacts/registry.json \
  --performance-registry artifacts/registry.json \
  --release-registry artifacts/release-registry.json \
  --name qwen05-local-release-candidate \
  --version 0.7 \
  --release-policy-profile strict_structured_fact \
  --performance-baseline-key performance_baseline:qwen05-performance-baseline:0.1 \
  --performance-drift-baseline-key performance_baseline:qwen05-performance-baseline:0.0 \
  --max-covariance-maha-last-auroc-drop 0.05 \
  --structured-fact-canonical-route-key benchmark_manifest:structured-fact-canonical-route:0.1 \
  --structured-fact-paraphrase-route-key benchmark_manifest:structured-fact-paraphrase-route:0.1 \
  --product-trace-replay-workflow-key report:qwen05-product-trace-replay-workflow:0.1 \
  --feedback-policy-workflow-key report:<feedback-policy-workflow-name>:<version> \
  --feedback-policy-min-matched-feedback-count 20 \
  --feedback-policy-min-safety-coverage 0.70 \
  --feedback-policy-max-unknown-safety-issue-rate 0.20 \
  --release-efficiency-report artifacts/product-runtime-profile-sweep/release-efficiency-report.json \
  --adapter-family-matrix artifacts/adapter_family_matrix/adapter-family-matrix.json \
  --adapter-family-profile strict_audit \
  --runtime-profile balanced \
  --min-best-quality-auroc 0.60 \
  --max-uncached-forward-seconds 40 \
  --min-selected 100 \
  --min-decision-accuracy 0.95 \
  --max-false-supported-rate 0.02 \
  --min-false-refuted-rate 0.90 \
  --max-p99-duration-seconds 0.20 \
  --require-performance-score-dump-cache \
  --min-performance-score-dump-cache-jsonl-view-hit-rate 0.5 \
  --max-performance-uncached-total-seconds-ratio 1.15 \
  --max-performance-score-dump-cache-jsonl-view-hit-rate-drop 0.10 \
  --required-route-require-non-oracle-evidence \
  --json artifacts/release-candidate-registry-workflow.json \
  --release-report-json artifacts/release-candidate-comparison.json \
  --artifact-manifest artifacts/release-candidate-artifact-manifest.json \
  --fingerprint-cache artifacts/release-candidate-fingerprints.json \
  --artifact-json-cache artifacts/release-candidate-json-cache.json \
  --manifest-fingerprint-workers 4 \
  --fail-on-blocked
```

The generated manifest fingerprints the release-candidate report and the
selected readiness, route, optional performance/product-trace-replay/feedback-policy/selector/runtime-drift/release-efficiency manifests,
and optional adapter family matrix report. Recursive verification therefore checks the final
candidate and all underlying baseline
manifests before the release candidate is registered. When `--runtime-profile`
is used, the selected profile and the defaults it filled are written into the
release report, manifest metadata, and registry record. Feedback-policy workflow
metadata includes the report/source/record, promotion decision, candidate policy
paths, matched feedback count, safety coverage, and unknown safety issue rate.

To choose a local value for `--manifest-fingerprint-workers` without rerunning
model work or release gates, replay manifest verification across worker counts:

```bash
python benchmarks/run_manifest_fingerprint_worker_sweep.py \
  --manifest artifacts/release-candidate-artifact-manifest.json \
  --json artifacts/release-manifest-fingerprint-worker-sweep.json \
  --workers 1,2,4,8 \
  --repeats 3 \
  --fingerprint-cache artifacts/release-candidate-fingerprints.json \
  --registry artifacts/release-registry.json \
  --name release-manifest-fingerprint-workers \
  --version 0.1
```

The sweep starts each sample from the same optional seed fingerprint cache,
records per-worker verification timing/cache summaries, and recommends the
fastest worker count whose verification samples all pass.
Use `--allow-failures` only for exploratory sweeps where failed worker counts
should be reported as rejected candidates instead of blocking a passing
recommendation.

To quantify local release-gate overhead after collecting one cold and one warm
registry-workflow run, aggregate their timing/cache summaries without rerunning
the gate:

```bash
python benchmarks/run_release_gate_overhead_baseline.py \
  --report artifacts/release-candidate-registry-workflow-cold.json \
  --report artifacts/release-candidate-registry-workflow-warm.json \
  --json artifacts/release-gate-overhead-baseline.json \
  --registry artifacts/release-registry.json \
  --name release-gate-overhead \
  --version 0.1 \
  --max-total-seconds 30 \
  --min-last-fingerprint-cache-hit-rate 0.90 \
  --min-report-count 2 \
  --fail-on-blocked
```

The report records `total_seconds`, `phase_total_seconds`, per-phase timing,
artifact fingerprint/JSON cache hit rates, and the slowest observed phase. Use
`--min-last-fingerprint-cache-hit-rate` to verify the warm run actually reused
the persisted fingerprint cache.

Use `export_product_promotion_contract.py` after a release candidate promotes to
write the smaller product handoff artifact consumed by demos and control-plane
jobs. It converts either a release-candidate comparison or registry-workflow JSON
into a `ProductPromotionContract`, writes a manifest, and can register a
`product_promotion_contract:*:*` record. When the release candidate was gated by
a product trace replay workflow, the compact contract and registry metadata keep
the workflow report/manifest plus its selector-replay and runtime-drift child
report paths for deployment-side provenance. Runtime-drift reports also carry
baseline/current optimization hints, so exported contracts preserve candidate
control defaults such as `max_verifier_route_attempts` alongside the budget
policy. When the release candidate was gated by a feedback-policy workflow, the
contract and registry metadata also retain the feedback-policy report/manifest,
promotion decision, candidate control-policy/default paths, validated
`ControlPolicyConfig`, control-default config, and replay safety metrics. When
the release candidate was gated by a release-efficiency report, the
promotion contract inherits the recommended runtime profile and efficiency score
from the candidate. For older release-candidate reports, pass the
release-efficiency report explicitly so the promotion contract, manifest, and
registry record also carry the same handoff evidence:

```bash
python benchmarks/export_product_promotion_contract.py \
  --source artifacts/smollm2_l20_inside_trigger_budget_derived_strict_structured_retrieval_audit_staged_release_candidate_v1_6_registry_workflow.json \
  --output artifacts/smollm2_product_promotion_contract_v1_6/product-promotion-contract.json \
  --artifact-manifest artifacts/smollm2_product_promotion_contract_v1_6/artifact-manifest.json \
  --registry artifacts/local-release-registry.json \
  --name smollm2-product-promotion-contract \
  --version 1.6 \
  --metadata release=smollm2-v1.6 \
  --metadata source_record=benchmark_manifest:smollm2-l20-inside-trigger-budget-derived-strict-structured-retrieval-audit-staged-qa-release-candidate:1.6 \
  --compact-json
```

Current local smoke release candidate:

```bash
python benchmarks/run_adapter_readiness_registry_workflow.py \
  --output-dir artifacts/tiny_local_readiness \
  --registry artifacts/local-readiness-registry.json \
  --name tiny-local-readiness \
  --version 0.4 \
  --json artifacts/tiny_local_readiness_registry_workflow.json \
  --verification-report artifacts/tiny_local_readiness_manifest_verification.json \
  --alpha 0.2 \
  --n-records 8 \
  --model sshleifer/tiny-gpt2 \
  --layers -1 \
  --batch-sizes 4 \
  --hidden-state-captures outputs \
  --eval-reps-cache-shard-size 4 \
  --cached-max-total-ratio 1.10 \
  --cache-only-max-total-ratio 0.35 \
  --python .venv/bin/python \
  --performance-clean \
  --compact-json \
  --fail-on-blocked

python benchmarks/run_release_candidate_registry_workflow.py \
  --readiness-registry artifacts/local-readiness-registry.json \
  --route-registry artifacts/staged-route-registry.json \
  --release-registry artifacts/local-release-registry.json \
  --name tiny-local-staged-qa-release-candidate \
  --version 0.4 \
  --readiness-baseline-key benchmark_manifest:tiny-local-readiness:0.4 \
  --route-baseline-key benchmark_manifest:truthfulqa-l80-structured-qa-staged-route:0.4 \
  --min-best-quality-auroc 0.60 \
  --max-uncached-forward-seconds 1.0 \
  --max-cache-only-seconds 1.0 \
  --min-selected 80 \
  --min-decision-accuracy 0.95 \
  --max-false-supported-rate 0.02 \
  --min-false-refuted-rate 0.90 \
  --max-verified-false-alarm 0.02 \
  --min-verified-detection 0.20 \
  --max-p99-duration-seconds 0.01 \
  --max-mean-attempted-route-count 1.1 \
  --max-retrieval-use-rate 0.0 \
  --json artifacts/local_staged_release_candidate_registry_workflow.json \
  --release-report-json artifacts/local_staged_release_candidate_comparison.json \
  --artifact-manifest artifacts/local_staged_release_candidate_manifest.json \
  --verification-report artifacts/local_staged_release_candidate_manifest_verification.json \
  --metadata evidence=tiny_local_readiness_plus_truthfulqa_l80_structured_qa_staged \
  --fail-on-blocked
```

This records `benchmark_manifest:tiny-local-readiness:0.4` and
`benchmark_manifest:tiny-local-staged-qa-release-candidate:0.4`. The candidate
combines a tiny-gpt2 offline readiness/runtime smoke baseline with the
TruthfulQA l80 staged structured QA route baseline. It is a local release-gate
plumbing artifact, not a Qwen production readiness claim.

INSIDE-gated local smoke variant:

```bash
python benchmarks/run_inside_sampling_profile.py \
  --output-dir artifacts/tiny_local_inside_sampling \
  --model sshleifer/tiny-gpt2 \
  --dtype float32 \
  --layer -1 \
  --batch-size 4 \
  --max-length 64 \
  --hidden-state-capture outputs \
  --inside-samples 3 \
  --inside-min-samples 2 \
  --inside-sample-step 1 \
  --inside-max-new-tokens 4 \
  --inside-batch-size 1 \
  --adaptive-max-sample-ratio 1.0 \
  --adaptive-selfcheck-max-sample-ratio 1.0 \
  --python .venv/bin/python \
  --clean \
  --fail-on-regression

python benchmarks/run_adapter_readiness_registry_workflow.py \
  --output-dir artifacts/tiny_local_readiness_inside \
  --registry artifacts/local-readiness-registry.json \
  --name tiny-local-readiness-inside \
  --version 0.5 \
  --json artifacts/tiny_local_readiness_inside_registry_workflow.json \
  --verification-report artifacts/tiny_local_readiness_inside_manifest_verification.json \
  --inside-sampling-report artifacts/tiny_local_inside_sampling/inside-sampling-profile-comparison.json \
  --alpha 0.2 \
  --n-records 8 \
  --model sshleifer/tiny-gpt2 \
  --layers -1 \
  --batch-sizes 4 \
  --hidden-state-captures outputs \
  --eval-reps-cache-shard-size 4 \
  --cached-max-total-ratio 1.10 \
  --cache-only-max-total-ratio 0.35 \
  --python .venv/bin/python \
  --performance-clean \
  --compact-json \
  --fail-on-blocked

python benchmarks/run_release_candidate_registry_workflow.py \
  --readiness-registry artifacts/local-readiness-registry.json \
  --route-registry artifacts/staged-route-registry.json \
  --release-registry artifacts/local-release-registry.json \
  --name tiny-local-inside-staged-qa-release-candidate \
  --version 0.5 \
  --readiness-baseline-key benchmark_manifest:tiny-local-readiness-inside:0.5 \
  --route-baseline-key benchmark_manifest:truthfulqa-l80-structured-qa-staged-route:0.4 \
  --min-best-quality-auroc 0.60 \
  --max-uncached-forward-seconds 1.0 \
  --max-cache-only-seconds 1.0 \
  --max-inside-sample-count-ratio 0.70 \
  --max-inside-generation-seconds-ratio 0.80 \
  --min-selected 80 \
  --min-decision-accuracy 0.95 \
  --max-false-supported-rate 0.02 \
  --min-false-refuted-rate 0.90 \
  --max-verified-false-alarm 0.02 \
  --min-verified-detection 0.20 \
  --max-p99-duration-seconds 0.01 \
  --max-mean-attempted-route-count 1.1 \
  --max-retrieval-use-rate 0.0 \
  --json artifacts/local_inside_staged_release_candidate_registry_workflow.json \
  --release-report-json artifacts/local_inside_staged_release_candidate_comparison.json \
  --artifact-manifest artifacts/local_inside_staged_release_candidate_manifest.json \
  --verification-report artifacts/local_inside_staged_release_candidate_manifest_verification.json \
  --metadata evidence=tiny_local_readiness_inside_plus_truthfulqa_l80_structured_qa_staged \
  --fail-on-blocked
```

This records `benchmark_manifest:tiny-local-readiness-inside:0.5` and
`benchmark_manifest:tiny-local-inside-staged-qa-release-candidate:0.5`. In the
current local smoke artifact, `adaptive_selfcheck` is selected with generated
sample ratio `0.667` and `inside_generation` ratio `0.716` versus fixed
sampling. This remains a tiny offline plumbing artifact; use representative
Qwen/SmolLM2 profile artifacts before making model-specific deployment claims.

SmolLM2 real-model l20 release candidate:

```bash
HF_HUB_DISABLE_XET=1 python benchmarks/run_inside_sampling_profile.py \
  --output-dir artifacts/smollm2_l20_inside_sampling \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --dtype float32 \
  --layer -12 \
  --limit 20 \
  --manifold-questions 40 \
  --batch-size 8 \
  --max-length 64 \
  --hidden-state-capture outputs \
  --inside-samples 3 \
  --inside-min-samples 2 \
  --inside-sample-step 1 \
  --inside-max-new-tokens 4 \
  --inside-batch-size 1 \
  --adaptive-max-sample-ratio 1.0 \
  --adaptive-selfcheck-max-sample-ratio 1.0 \
  --python .venv/bin/python \
  --real-truthfulqa \
  --skip-existing \
  --fail-on-regression

HF_HUB_DISABLE_XET=1 python benchmarks/run_adapter_readiness_registry_workflow.py \
  --output-dir artifacts/smollm2_l20_readiness_inside \
  --registry artifacts/local-readiness-registry.json \
  --name smollm2-l20-readiness-inside \
  --version 0.6 \
  --json artifacts/smollm2_l20_readiness_inside_registry_workflow.json \
  --verification-report artifacts/smollm2_l20_readiness_inside_manifest_verification.json \
  --inside-sampling-report artifacts/smollm2_l20_inside_sampling/inside-sampling-profile-comparison.json \
  --real-truthfulqa \
  --limit 20 \
  --manifold-questions 40 \
  --alpha 0.2 \
  --n-records 8 \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --layers -12 \
  --batch-sizes 8 \
  --hidden-state-captures outputs \
  --max-length 64 \
  --eval-reps-cache-shard-size 64 \
  --cached-max-total-ratio 1.10 \
  --cache-only-max-total-ratio 0.35 \
  --progress-every 50 \
  --python .venv/bin/python \
  --performance-clean \
  --compact-json \
  --fail-on-blocked

python benchmarks/run_release_candidate_registry_workflow.py \
  --readiness-registry artifacts/local-readiness-registry.json \
  --route-registry artifacts/staged-route-registry.json \
  --release-registry artifacts/local-release-registry.json \
  --name smollm2-l20-inside-staged-qa-release-candidate \
  --version 0.6 \
  --readiness-baseline-key benchmark_manifest:smollm2-l20-readiness-inside:0.6 \
  --route-baseline-key benchmark_manifest:truthfulqa-l80-structured-qa-staged-route:0.4 \
  --min-best-quality-auroc 0.65 \
  --max-uncached-forward-seconds 45.0 \
  --max-cache-only-seconds 1.0 \
  --max-inside-sample-count-ratio 0.95 \
  --max-inside-generation-seconds-ratio 1.05 \
  --min-selected 80 \
  --min-decision-accuracy 0.95 \
  --max-false-supported-rate 0.02 \
  --min-false-refuted-rate 0.90 \
  --max-verified-false-alarm 0.02 \
  --min-verified-detection 0.20 \
  --max-p99-duration-seconds 0.01 \
  --max-mean-attempted-route-count 1.1 \
  --max-retrieval-use-rate 0.0 \
  --json artifacts/smollm2_l20_inside_staged_release_candidate_registry_workflow.json \
  --release-report-json artifacts/smollm2_l20_inside_staged_release_candidate_comparison.json \
  --artifact-manifest artifacts/smollm2_l20_inside_staged_release_candidate_manifest.json \
  --verification-report artifacts/smollm2_l20_inside_staged_release_candidate_manifest_verification.json \
  --metadata evidence=smollm2_l20_readiness_inside_plus_truthfulqa_l80_structured_qa_staged \
  --fail-on-blocked
```

This records `benchmark_manifest:smollm2-l20-readiness-inside:0.6` and
`benchmark_manifest:smollm2-l20-inside-staged-qa-release-candidate:0.6`. The
readiness run promotes `truth_proj` AUROC `0.682`, uncached forced-answer cost
`38.786s`, and cache-only replay cost `0.339s`. The full-sample INSIDE profile
selects `adaptive_selfcheck`, but only reduces generated samples to `0.937` of
fixed sampling and leaves `inside_generation` at `1.001` of fixed. Treat this as
evidence that real-model INSIDE should be threshold/top-fraction triggered
instead of run on every statement by default.

The triggered follow-up records
`benchmark_manifest:smollm2-l20-readiness-inside-triggered:0.7` and
`benchmark_manifest:smollm2-l20-inside-triggered-staged-qa-release-candidate:0.7`.
It uses `--inside-trigger-signal truth_proj --inside-trigger-top-fraction 0.25`
on the same SmolLM2 l20 setup and reuses the 0.6 cache-profile matrix through
`--performance-report`. The profile samples 39/154 eval statements, skips 115,
and cuts fixed `inside_generation` from `467.563s` to `118.513s` (`0.253x`).
The promoted runtime recommendation selects `adaptive_selfcheck` within that
triggered budget with 110 generated samples, generated-sample ratio `0.940`
versus triggered fixed, and `inside_generation` ratio `1.009`; treat the
trigger gate as the primary performance win and self-check as an auditable
analysis setting, not as the main speedup source.

The derived trigger-budget release records
`benchmark_manifest:smollm2-l20-readiness-inside-trigger-budget-derived:0.8`
and
`benchmark_manifest:smollm2-l20-inside-trigger-budget-derived-staged-qa-release-candidate:0.8`.
It reuses the 0.6 SmolLM2 l20 performance matrix and the 0.4 staged structured
QA route, then folds in
`artifacts/smollm2_l20_inside_trigger_budget_sweep_derived/inside-trigger-budget-sweep.json`.
The runtime recommendation uses the sweep's `quality_balanced_recommendation`:
`truth_proj` top-40% triggered `adaptive_selfcheck`, derived from one
largest-budget source profile with `--derive-from-max-budget`. The release gate
uses reference ratios because no direct trigger-budget baseline ratios are
available: sample-count ratio `0.472`, `inside_generation` ratio `0.503`,
semantic-entropy AUROC `0.570`, 77/154 statements sampled, 77 skipped, and 218
generated samples. The cost-first sweep recommendation remains top-10%; the
registered release default is top-40% because it preserves the best measured
INSIDE semantic-entropy quality within the configured tolerance.
Use `--inside-trigger-budget-policy cost_first` in the release-candidate
comparison or registry workflow to make the final gate select the top-10%
trigger budget from the same verified sweep evidence.

The current strict structured-retrieval-audit SmolLM2 default records
`benchmark_manifest:smollm2-l20-inside-trigger-budget-derived-strict-structured-retrieval-audit-staged-qa-release-candidate:1.6`.
It keeps the same 0.8 readiness baseline, 0.4 staged structured-QA product
route, and registered performance handoff
`performance_baseline:smollm2-l20-performance-baseline:0.9`, then requires the
promoted adapter-family matrix with `structured_state`, `state_transition`,
`retrieval_groundedness`, and `retrieval_structured_qa` routes present and
promoted. It also requires
`benchmark_manifest:smollm2-l80-retrieval-structured-qa-route:0.6` as a separate
retrieval-structured-QA audit route with its own quality/runtime budget,
non-oracle evidence provenance, and answer-echo retrieval stress control.
The final manifest fingerprints the release-candidate report plus readiness,
route, performance, selector replay, product-runtime-drift,
adapter-family, and required retrieval-audit manifests; the
release comparison verifies that the performance baseline recommendation matches
the selected runtime: layer `-12`, batch size `8`, `outputs` hidden-state
capture, no prefix-KV cache, worker count `1`, `truth_proj` AUROC `0.682`, and
the quality-balanced `top_0p4` triggered `adaptive_selfcheck` budget. The
selected product route gates `retrieval_use_rate` at `0.0` and
`mean_attempted_route_count` at `1.1`; retrieval is required as audit capability
evidence, not as the default low-latency route. Version 1.6 adds promoted
selector replay, refreshed runtime-drift evidence, and a compact deployable
contract at `artifacts/smollm2_product_promotion_contract_v1_6/product-promotion-contract.json`.
The required retrieval audit route promotes with selected `238`, decision
accuracy `0.992`, false-supported rate `0.000`, false-refuted rate `1.000`,
verified false alarm `0.009`, verified detection `1.000`, total runtime
`0.845s`, and `410` retrieval hits under a `450` hit budget. Its answer-echo
stress control verifies that answer-derived evidence supports false claims at
rate `0.980` and refutes them at `0.000`, so such corpora are blocked as
grounding evidence.

```bash
python benchmarks/run_release_candidate_registry_workflow.py \
  --readiness-registry artifacts/local-readiness-registry.json \
  --route-registry artifacts/staged-route-registry.json \
  --performance-registry artifacts/local-readiness-registry.json \
  --release-registry artifacts/local-release-registry.json \
  --name smollm2-l20-inside-trigger-budget-derived-strict-structured-retrieval-audit-staged-qa-release-candidate \
  --version 1.6 \
  --readiness-baseline-key benchmark_manifest:smollm2-l20-readiness-inside-trigger-budget-derived:0.8 \
  --route-baseline-key benchmark_manifest:truthfulqa-l80-structured-qa-staged-route:0.4 \
  --required-route-baseline-key benchmark_manifest:smollm2-l80-retrieval-structured-qa-route:0.6 \
  --performance-baseline-key performance_baseline:smollm2-l20-performance-baseline:0.9 \
  --selector-replay-report artifacts/smollm2_product_trace_replay_workflow/selector-replay/runtime-profile-selector-replay.json \
  --product-runtime-drift-report artifacts/smollm2_product_runtime_drift_v1_6/product-runtime-drift.json \
  --adapter-family-matrix artifacts/smollm2_l20_adapter_family_retrieval_structured_qa/adapter-family-matrix.json \
  --required-adapter-route structured_state \
  --required-adapter-route state_transition \
  --required-adapter-route retrieval_groundedness \
  --required-adapter-route retrieval_structured_qa \
  --runtime-profile balanced \
  --min-selected 200 \
  --min-decision-accuracy 0.99 \
  --max-false-supported-rate 0.01 \
  --min-false-refuted-rate 1.0 \
  --max-verified-false-alarm 0.01 \
  --min-verified-detection 1.0 \
  --max-p99-duration-seconds 0.01 \
  --max-mean-attempted-route-count 1.1 \
  --max-retrieval-use-rate 0.0 \
  --max-inside-sample-count-ratio 0.6 \
  --max-inside-generation-seconds-ratio 0.8 \
  --required-route-min-selected 200 \
  --required-route-min-decision-accuracy 0.99 \
  --required-route-max-false-supported-rate 0.01 \
  --required-route-min-false-refuted-rate 1.0 \
  --required-route-max-verified-false-alarm 0.01 \
  --required-route-min-verified-detection 1.0 \
  --required-route-max-mean-attempted-route-count 2.1 \
  --required-route-max-retrieval-use-rate 1.0 \
  --required-route-max-runtime-total-seconds 8.0 \
  --required-route-max-retrieval-hit-count 450 \
  --required-route-require-non-oracle-evidence \
  --required-route-require-retrieval-stress-control \
  --required-route-min-stress-false-supported-rate 0.90 \
  --required-route-max-stress-false-refuted-rate 0.05 \
  --json artifacts/smollm2_l20_inside_trigger_budget_derived_strict_structured_retrieval_audit_staged_release_candidate_v1_6_registry_workflow.json \
  --release-report-json artifacts/smollm2_l20_inside_trigger_budget_derived_strict_structured_retrieval_audit_staged_release_candidate_v1_6_comparison.json \
  --artifact-manifest artifacts/smollm2_l20_inside_trigger_budget_derived_strict_structured_retrieval_audit_staged_release_candidate_v1_6_manifest.json \
  --verification-report artifacts/smollm2_l20_inside_trigger_budget_derived_strict_structured_retrieval_audit_staged_release_candidate_v1_6_manifest_verification.json \
  --metadata evidence=smollm2_l20_release_candidate_required_route_non_oracle_stress_gated \
  --fail-on-blocked
```

## `build_truthfulqa_corpus.py`

Builds a local TruthfulQA correct-answer corpus for reproducible retrieval
baselines. It uses the same deterministic TruthfulQA split parameters as
`eval_truthfulqa.py`, writes only correct-answer statements as evidence
documents, and does not create per-false-claim oracle refutations.

```bash
python benchmarks/build_truthfulqa_corpus.py \
  --manifold-questions 80 \
  --limit 80 \
  --output artifacts/truthfulqa_l80_correct_answer_corpus.json
```

## `build_retrieval_stress_corpus.py`

Builds an answer-echo retrieval stress corpus from a statement-bearing score
dump. Every scored answer becomes a local retrieval document. This is a negative
control: if a verifier succeeds only when retrieval evidence comes from the same
answers being audited, the result is self-support, not external grounding.

```bash
python benchmarks/build_retrieval_stress_corpus.py \
  --scores artifacts/qwen05_truthfulqa_l80_scores_with_statements.json \
  --output artifacts/truthfulqa-l80-answer-echo-retrieval-stress/answer-echo-corpus.json \
  --document-field answer
```

The committed stress artifact at
`artifacts/truthfulqa-l80-answer-echo-retrieval-stress/` runs this corpus through
`run_verifier_signal_fusion_workflow.py`. It retrieves hits for 556/556 records
but drives false-supported rate to `0.980` and false-refuted rate to `0.000`.
Use it as a fail-fast check against retrieval setups that merely echo the model
answer corpus. `compare_route_baselines.py --require-retrieval-stress-control`
and `compare_release_candidates.py --required-route-require-retrieval-stress-control`
turn this negative control into a fail-closed route/release gate.

## `fetch_wikidata_reference_docs.py`

Fetches or replays a small Wikidata SPARQL result set into JSONL source
documents. The default preset is `country_capitals`: each source document states
one country-capital fact with Wikidata QID metadata, `license=CC0-1.0`, source
URL metadata, and a retrieval timestamp. The `country_core_facts` preset fetches
template-ready country facts for selected properties, defaulting to `P36`
capital, `P37` official language, and `P38` currency. The script uses only the
standard library and supports `--input-json` for offline replay of saved SPARQL
results. Rows whose natural-language labels are bare Wikidata `Q...` or `P...`
ids are skipped by default to avoid turning unresolved entities into retrieval
evidence; pass `--keep-qid-labels` only when debugging raw SPARQL coverage.

```bash
python benchmarks/fetch_wikidata_reference_docs.py \
  --limit 120 \
  --output artifacts/wikidata-country-capitals-external-corpus/wikidata-country-capitals-source.jsonl \
  --artifact-manifest artifacts/wikidata-country-capitals-external-corpus/wikidata-source-manifest.json
```

```bash
python benchmarks/fetch_wikidata_reference_docs.py \
  --query-preset country_core_facts \
  --property P36 \
  --property P37 \
  --property P38 \
  --limit 360 \
  --output artifacts/wikidata-country-core-facts-external-corpus/wikidata-country-core-facts-source.jsonl \
  --artifact-manifest artifacts/wikidata-country-core-facts-external-corpus/wikidata-source-manifest.json
```

The committed evidence-source gate at
`artifacts/wikidata-country-capitals-external-corpus/` fetches 120 Wikidata
country-capital records, normalizes them through
`build_external_retrieval_corpus.py`, passes
`audit_retrieval_corpus_provenance.py --audit-role grounding`, and recursively
verifies the top-level manifest. Its scope is deliberately narrow:
`promotes_verifier_route=false`; it proves a real CC0 external source can pass
the provenance gate, not that the country-capital corpus is sufficient
open-domain grounding coverage for TruthfulQA.

The follow-up route-quality audit at
`artifacts/wikidata-country-capitals-external-route-audit-qwen05-l80/` runs the
same external corpus through `run_local_retrieval_route_workflow.py` against the
Qwen l80 statement dump:

```bash
python benchmarks/run_local_retrieval_route_workflow.py \
  --scores artifacts/qwen05_truthfulqa_l80_scores_with_statements.json \
  --corpus artifacts/wikidata-country-capitals-external-corpus/wikidata-country-capitals-corpus.json \
  --output-dir artifacts/wikidata-country-capitals-external-route-audit-qwen05-l80 \
  --score-name qwen05-l80-wikidata-country-capitals \
  --query-field question_answer \
  --retriever-min-overlap 0.15 \
  --verifier-min-overlap 0.55 \
  --retrieval-limit 5 \
  --omit-label-metadata \
  --gate-route retrieval_groundedness \
  --gate-min-selected 1 \
  --max-verified-false-alarm 0.05 \
  --min-verified-detection 0.20 \
  --allow-non-promote
```

The audit is correctly blocked: the corpus retrieves hits for `254/556`
records, but the retrieval-backed route has verified false alarm `0.149`,
above the `0.05` gate, with verified detection `0.286`. This makes the current
Wikidata country-capital corpus a useful external-source smoke gate, not a
deployable TruthfulQA verifier route.

The broader committed source gate at
`artifacts/wikidata-country-core-facts-external-corpus/` fetches `359`
template-ready `P36`/`P37`/`P38` rows after QID-label filtering, builds both a
`359`-document external retrieval corpus and a `359`-document structured QA
corpus, passes grounding provenance audit, and recursively verifies the
manifest. Its route-quality follow-up at
`artifacts/wikidata-country-core-facts-external-route-audit-qwen05-l80/` is also
correctly blocked: coverage rises to `275/556` (`0.495`) and hits rise to
`1125`, split across `P36=510`, `P37=303`, and `P38=312`, but
`retrieval_groundedness` still fails the false-alarm gate (`0.155` > `0.05`)
despite detection `0.316`. The measured improvement over country capitals is
real but insufficient for route promotion; the next useful path is structured
QA/triple-style Wikidata verification rather than lexical retrieval tuning
alone.

## `build_wikidata_qa_corpus.py`

Converts structured Wikidata fact documents into the existing structured QA
schema consumed by `QuestionAnswerVerifier` and `retrieval_structured_qa`. The
default country-capital template maps each `P36` fact into one
`question`/`answer` record such as `What is the capital of France?` / `Paris`.
The builder can also consume a template JSON file for multiple properties, for
example `P36` capital, `P37` official language, and `P38` currency. It keeps
label-use flags false, fingerprints source files, rejects reserved score-dump
metadata keys, and skips QID-only labels by default so unlabeled Wikidata
entities do not become awkward natural-language QA facts.

```bash
python benchmarks/build_wikidata_qa_corpus.py \
  --source artifacts/wikidata-country-capitals-external-corpus/wikidata-country-capitals-corpus.json \
  --output artifacts/wikidata-country-capitals-external-corpus/wikidata-country-capitals-qa-corpus.json \
  --artifact-manifest artifacts/wikidata-country-capitals-external-corpus/wikidata-country-capitals-qa-manifest.json
```

Template JSON accepts either a list or an object with a `templates` list:

```json
{
  "templates": [
    {
      "statement_property": "P36",
      "statement_property_label": "capital",
      "question_template": "What is the capital of {country}?",
      "answer_field": "capital"
    },
    {
      "statement_property": "P37",
      "statement_property_label": "official language",
      "question_template": "What is an official language of {country}?",
      "answer_field": "language"
    }
  ]
}
```

Use the generated QA corpus with `build_evidence_fixture.py --query-field
question` or directly through `eval_verifier_ensemble.py --qa-corpus` when the
score dump contains matching `question`/`answer` statement metadata. This route
is a structured knowledge-graph bridge: it can support correct values and refute
wrong values for covered properties, but it does not broaden the Wikidata source
coverage by itself.

## `run_wikidata_structured_qa_route_workflow.py`

Builds a covered-facts structured QA or structured-fact route benchmark from a
Wikidata QA corpus.
The workflow creates a balanced score dump with one true row per known
question/answer and one false row per question by swapping in an answer from a
different question while avoiding known same-question answers. It then runs the
existing verifier ensemble with `--qa-corpus` or `--fact-corpus`, writes
per-record verifier traces, and emits a route summary plus artifact manifest.

```bash
python benchmarks/run_wikidata_structured_qa_route_workflow.py \
  --qa-corpus artifacts/wikidata-country-core-facts-external-corpus/wikidata-country-core-facts-qa-corpus.json \
  --output-dir artifacts/wikidata-country-core-facts-structured-qa-route \
  --score-name wikidata-country-core-facts-structured-qa \
  --alpha 0.10 \
  --compact-json
```

Use `--route structured_fact` when the score dump should contain natural-language
claims such as `Paris is the capital of France.` and route through
`StructuredFactVerifier` instead of requiring question/answer statement
metadata:

```bash
python benchmarks/run_wikidata_structured_qa_route_workflow.py \
  --qa-corpus artifacts/wikidata-country-core-facts-external-corpus/wikidata-country-core-facts-qa-corpus.json \
  --output-dir artifacts/wikidata-country-core-facts-structured-fact-route \
  --score-name wikidata-country-core-facts-structured-fact \
  --route structured_fact \
  --alpha 0.10 \
  --compact-json
```

Use `--fact-claim-style paraphrase_robustness` to stress the same
KG-covered facts with common natural-language variants such as possessive
claims, subject-first claims, currency-use claims, and multi-object list claims:

```bash
python benchmarks/run_wikidata_structured_qa_route_workflow.py \
  --qa-corpus artifacts/wikidata-country-core-facts-external-corpus/wikidata-country-core-facts-qa-corpus.json \
  --output-dir artifacts/wikidata-country-core-facts-structured-fact-paraphrase-route \
  --score-name wikidata-country-core-facts-structured-fact-paraphrase \
  --route structured_fact \
  --fact-claim-style paraphrase_robustness \
  --alpha 0.10 \
  --compact-json
```

The current covered-facts artifact promotes the structured QA route for exactly
the properties present in the source corpus: `718` rows from `359` Wikidata
`P36`/`P37`/`P38` facts, selected route `structured_qa` for all rows, `359`
supported true facts, `359` refuted swapped-answer false facts, decision accuracy
`1.0`, and false-supported rate `0.0`. The scope is intentionally narrow:
structured QA is the property-level correction path for covered facts, while
lexical retrieval remains gated separately for broad open-domain coverage.
The matching structured-fact artifact promotes the same `718` covered-fact rows
as natural-language claims, selects `structured_fact` for all rows, supports all
`359` true facts, refutes all `359` swapped-answer false facts, and records
decision accuracy `1.0` with false-supported rate `0.0`.
The paraphrase robustness artifact expands those covered facts to `2868`
natural-language claim rows (`1434` true / `1434` false), selects
`structured_fact` for all rows, reaches decision accuracy `1.0`, and keeps
false-supported rate `0.0`; it is a
surface-form robustness check for covered KG facts, not a broad open-domain
claim.

## `analyze_retrieval_route_gaps.py`

Explains blocked retrieval routes from `eval_verifier_ensemble.py
--verified-records-jsonl` sidecars. It reports final status counts, selected
routes, retrieval-hit coverage, gap buckets, hit source/property counts, and
bounded examples per bucket.

```bash
python benchmarks/analyze_retrieval_route_gaps.py \
  --verified-records-jsonl artifacts/wikidata-country-capitals-external-route-audit-qwen05-l80/verified-records.jsonl \
  --output artifacts/wikidata-country-capitals-external-route-audit-qwen05-l80/retrieval-route-gap-analysis.json \
  --artifact-manifest artifacts/wikidata-country-capitals-external-route-audit-qwen05-l80/retrieval-route-gap-analysis-manifest.json
```

For the Wikidata country-capital route audit, gap analysis shows all `556`
records finish as `insufficient_evidence`: `302` have no retrieval hits, `254`
use retrieval, and `114` true records hit the corpus but fail lexical overlap.
All `925` retrieval hits come from property `P36` (`capital`), so the next
useful source expansion must add broader fact predicates or a more structured
Wikidata verifier; tuning lexical thresholds alone would not create refutation
coverage.

## `build_external_retrieval_corpus.py`

Builds an explicit external-candidate retrieval corpus from caller-supplied
JSON, JSONL, or text source files. This is the ingestion boundary to use before
local retrieval fixtures when the source is not derived from the score dump or
TruthfulQA labels. The builder fingerprints input files, sets
`corpus_type=external_evidence_candidate`, sets label-use flags to false, and
rejects document metadata keys such as `claim_id`, `score_label`, `label`, and
`row_index` so source files cannot silently smuggle evaluation labels or
score-dump row links into retrieval.

```bash
python benchmarks/build_external_retrieval_corpus.py \
  --source data/external_reference_docs.jsonl \
  --output artifacts/external_reference_corpus.json \
  --corpus-name external_reference \
  --source-kind licensed_reference_dump \
  --artifact-manifest artifacts/external_reference_corpus.manifest.json
```

Run `audit_retrieval_corpus_provenance.py --audit-role grounding` on the
resulting corpus before treating it as external evidence. Synthetic smoke
fixtures can prove the ingestion path, but they should not be registered as
open-domain grounding evidence until the underlying source files are real,
licensed, and domain-shifted from the evaluated answers.

## `audit_retrieval_corpus_provenance.py`

Audits whether a retrieval corpus can be treated as external grounding evidence,
only as a controlled dataset-derived baseline, or only as an answer-echo stress
control. The audit scans corpus metadata, label-use flags, source-record links,
claim-id links, answer-copy rates, and writes a manifest-friendly JSON report.

```bash
python benchmarks/audit_retrieval_corpus_provenance.py \
  --scores artifacts/qwen05_truthfulqa_l80_scores_with_statements.json \
  --corpus artifacts/truthfulqa_l80_correct_answer_corpus.json \
  --audit-role grounding \
  --output artifacts/truthfulqa-l80-retrieval-corpus-provenance-audit/correct-answer-grounding-audit.json
```

Grounding mode is fail-closed: a corpus must carry a recognized explicit
external type such as `external_evidence_candidate`; untyped local text corpora
are classified as `untyped_local_corpus` and fail promotion even when they have
no label metadata or answer-copy matches.

The current provenance matrix at
`artifacts/truthfulqa-l80-retrieval-corpus-provenance-audit/` verifies four
roles. The local correct-answer corpus fails the `grounding` role but passes
`controlled_baseline`: it is dataset-derived, with exact answer copy rate
`0.514`, so it remains a reproducible local baseline rather than open-domain
retrieval evidence. The answer-echo corpus fails `grounding` with exact answer
copy rate `0.996` and claim-id link rate `1.000`, but passes `stress_control`.
No current local corpus is marked `external_domain_shift_ready`.

## `build_evidence_fixture.py`

Builds a non-oracle claim/evidence fixture from a statement-bearing score dump
and local evidence corpus files. It supports JSON, JSONL, and plain text corpora,
uses dependency-free token-overlap retrieval with optional SQLite FTS candidate
indexing, and copies labels only into audit metadata; retrieval is driven by
claim text. Use `--omit-label-metadata` when producing fixture artifacts for
adapter audits where labels should remain only in the score dump used for
evaluation. CLI-built fixtures include `input_provenance` with the score dump
metadata, corpus fingerprints, optional retriever-index fingerprint, and the
effective builder config.

```bash
python benchmarks/build_evidence_fixture.py \
  --scores artifacts/qwen05_truthfulqa_l80_scores_with_statements.json \
  --corpus artifacts/truthfulqa_l80_correct_answer_corpus.json \
  --output artifacts/truthfulqa_l80_local_evidence_claims.json \
  --query-field answer \
  --retriever-backend auto \
  --retriever-index-path artifacts/cache/local_retrieval_fts/truthfulqa_l80.sqlite \
  --retriever-min-overlap 0.95 \
  --retrieval-limit 3 \
  --omit-label-metadata

python benchmarks/eval_verifier_ensemble.py \
  --scores qwen-l80=artifacts/qwen05_truthfulqa_l80_scores_with_statements.json \
  --scores smollm2-l80=artifacts/smollm2_truthfulqa_l80_scores_with_statements.json \
  --claims artifacts/truthfulqa_l80_local_evidence_claims.json \
  --signal truth_proj \
  --alphas 0.05,0.1,0.2 \
  --repeats 50 \
  --verification-cache-dir artifacts/cache/verifier_traces \
  --json artifacts/truthfulqa_l80_local_evidence_verifier_ensemble_report.json
```

Use this before wiring a real search/RAG backend: it gives the same downstream
fixture schema and `verification_quality` fields while keeping evidence source
and retrieval behavior fully reproducible. The default retriever backend is
`memory`; `auto` tries SQLite FTS5 and falls back to memory when unavailable.
When `--retriever-index-path` is provided with `auto` or `sqlite_fts`, the FTS
index is persisted and reused when its stored corpus fingerprint matches the
current corpus. `--verification-cache-dir` is optional and stores verified-record
traces keyed by score dump, claims/evidence content, verifier parameters, and
state/QA sources so repeated alpha/repeat sweeps can skip claim verification.

## `run_local_retrieval_route_workflow.py`

Builds the same local retrieval evidence fixture, runs verifier-ensemble route
metrics, applies adapter promotion gates, fingerprints the score dump, corpora,
claims, verifier report, route comparison, and promotion report, then optionally
registers the verified manifest as a route baseline.

```bash
python benchmarks/run_local_retrieval_route_workflow.py \
  --scores artifacts/qwen05_truthfulqa_l80_scores_with_statements.json \
  --corpus artifacts/truthfulqa_l80_correct_answer_corpus.json \
  --output-dir artifacts/qwen05_l80_local_retrieval_route \
  --registry artifacts/registry.json \
  --name qwen05-l80-local-retrieval-route \
  --version 0.7 \
  --signal truth_proj \
  --query-field answer \
  --retriever-backend auto \
  --retriever-index-path artifacts/cache/local_retrieval_fts/qwen05_l80.sqlite \
  --retriever-min-overlap 0.95 \
  --retrieval-limit 3 \
  --omit-label-metadata \
  --claims-cache-dir artifacts/cache/local_retrieval_claims \
  --verifier-trace-cache-dir artifacts/cache/verifier_traces \
  --gate-route retrieval_structured_qa \
  --min-selected 100 \
  --min-decision-accuracy 0.90 \
  --max-false-supported-rate 0.05 \
  --max-mean-attempted-route-count 3.1 \
  --max-retrieval-use-rate 1.0 \
  --fail-on-blocked
```

Use this when the local corpus baseline should enter `compare_route_baselines.py`
and release-candidate gates. Pass the resulting registry key to
`compare_release_candidates.py --required-route-baseline-key` when it should be a
mandatory audit baseline while another route remains the selected product path.
Unlike `build_evidence_fixture.py` alone, this workflow records the full
provenance chain needed for recursive manifest verification. The generated
claims fixture also carries `input_provenance` and `label_usage`; use
`--omit-label-metadata` to keep labels only in the score dump while preserving
label-conditioned evaluation in the verifier report. The workflow report also
includes a lightweight `profile` block
with phase timings, input/output artifact byte sizes, dataset scale, retrieval
hit counts, and route-count metadata. The same runtime summary is copied into
the artifact manifest and registry metadata so route baselines can be compared
later without rerunning the workflow.
Add `--max-runtime-total-seconds`, `--max-retrieval-hit-count`,
`--min-claims-cache-hit-rate`, or `--min-verifier-trace-cache-hit-rate` when
route promotion must also satisfy an explicit runtime/cache budget. These gates
fail closed when the corresponding metric is missing or non-finite, block final
workflow promotion, and prevent registry promotion when the pre-registration
budget has already failed.

For strict false-support gates on TruthfulQA-style local corpora, prefer
`--gate-route retrieval_structured_qa`. The route uses retrieved documents with
`question`/`answer` metadata as structured facts before falling back to lexical
`retrieval_groundedness`, which avoids treating high token overlap between a
wrong answer and the same-question correct-answer evidence as support.

The current registered SmolLM2 l80 retrieval audit baseline is
`benchmark_manifest:smollm2-l80-retrieval-structured-qa-route:0.6`:

```bash
python benchmarks/run_local_retrieval_route_workflow.py \
  --scores artifacts/smollm2_truthfulqa_l80_scores_with_statements.json \
  --corpus artifacts/truthfulqa_l80_correct_answer_corpus.json \
  --output-dir artifacts/smollm2_l80_retrieval_structured_qa_route_v0_6 \
  --registry artifacts/staged-route-registry.json \
  --name smollm2-l80-retrieval-structured-qa-route \
  --version 0.6 \
  --score-name retrieval \
  --signal truth_proj \
  --alpha 0.10 \
  --repeats 1 \
  --query-field answer \
  --retriever-backend memory \
  --retriever-min-overlap 0.95 \
  --verifier-min-overlap 0.65 \
  --retrieval-limit 3 \
  --gate-route retrieval_structured_qa \
  --min-selected 200 \
  --gate-min-selected 200 \
  --min-decision-accuracy 0.99 \
  --max-false-supported-rate 0.01 \
  --min-false-refuted-rate 1.0 \
  --max-verified-false-alarm 0.01 \
  --min-verified-detection 1.0 \
  --max-mean-attempted-route-count 2.1 \
  --max-retrieval-use-rate 1.0 \
  --max-runtime-total-seconds 8.0 \
  --max-retrieval-hit-count 450 \
  --omit-label-metadata \
  --retrieval-stress-manifest artifacts/truthfulqa-l80-answer-echo-retrieval-stress/artifact-manifest.json \
  --metadata evidence=smollm2_l80_retrieval_structured_qa_label_free_stress_gated \
  --compact-json \
  --fail-on-blocked
```

It promotes `retrieval_structured_qa` with selected `238`, decision accuracy
`0.992`, false-supported rate `0.000`, false-refuted rate `1.000`, verified
detection `1.000`, verified false alarm `0.009`, runtime `0.845s`, and `410`
retrieval hits. Its claims artifact records `labels_used_for_retrieval=false`,
`labels_copied_to_record_metadata=false`, and score/corpus `input_provenance`.
The attached answer-echo stress manifest verifies the expected self-support
failure mode before the route can be used as release audit evidence.

`--claims-cache-dir` is optional. When set, the workflow caches generated
claims fixtures by score-dump fingerprint, corpus fingerprints, query field,
retriever backend, retriever overlap threshold, and retrieval limit. A cache hit
skips local score dump/corpus parsing for claim construction, then still reruns
verifier-route metrics and promotion against the current score dump and emitted
claims file.

`--verification-cache-dir` is optional. It caches verifier ensemble
`verified_records`, so repeated workflow runs can reuse claim verification and
retrieval route traces while still recalculating alpha-specific control metrics,
promotion gates, manifests, and registry output.

`--retriever-backend` defaults to `memory`. Use `auto` to try the standard-library
SQLite FTS5 candidate index and fall back to memory when FTS5 is unavailable in
the local Python build. Use `sqlite_fts` when the run should record that the
indexed backend was explicitly requested. `--retriever-index-path` is optional;
when set with `auto` or `sqlite_fts`, the workflow can reuse a persistent FTS
index across runs and records the actual backend, index path, and reuse status
in the claims fixture and manifest metadata.

Saved l80 local-corpus verifier baseline with `--query-field answer`,
`--retriever-min-overlap 0.95`, and `--retrieval-limit 3`:

| Run | Verified false alarm | Verified detection | true supported | false supported | decision accuracy |
|---|---:|---:|---:|---:|---:|
| Qwen l80 | 0.008 | 0.274 | 0.908 | 0.042 | 0.946 |
| SmolLM2 l80 | 0.008 | 0.219 | 0.908 | 0.042 | 0.946 |

Interpretation: this conservative lexical corpus strongly suppresses false
alarms by supporting most true claims, but it rarely refutes false claims. It is
a reproducible non-oracle baseline, not a replacement for stronger retrieval,
database, calculator, or domain-world-model evidence.
These rows are overall `verification_quality` metrics from the verifier report,
not route-promotion metrics. `run_local_retrieval_route_workflow.py` computes
route gates on selected route invocations, so a release-grade local retrieval
route should be registered only after its own promotion report passes the
intended route-level false-support, verified false-alarm, detection, and runtime
checks.

For both Qwen l80 and SmolLM2 l80, `route_summary.selected_counts` is
`groundedness=287` and `retrieval_groundedness=269`, which separates direct
lexical evidence decisions from decisions after local retrieval hits.

Structured QA/database adapter baseline using the same correct-answer corpus
directly as `--qa-corpus`:

```bash
python benchmarks/eval_verifier_ensemble.py \
  --scores qwen-l80=artifacts/qwen05_truthfulqa_l80_scores_with_statements.json \
  --scores smollm2-l80=artifacts/smollm2_truthfulqa_l80_scores_with_statements.json \
  --qa-corpus artifacts/truthfulqa_l80_correct_answer_corpus.json \
  --signal truth_proj \
  --alphas 0.05,0.1,0.2 \
  --repeats 50 \
  --json artifacts/truthfulqa_l80_structured_qa_verifier_ensemble_report.json
```

Current l80 structured QA baseline at alpha 0.100:

| Run | Verified false alarm | Verified detection | true supported | false refuted | false supported | decision accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Qwen l80 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| SmolLM2 l80 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |

Interpretation: this is a structured database/domain-state adapter check over
the same TruthfulQA questions, not an open-domain verifier. It shows the control
plane can exploit exact external state when a trusted structured source provides
the relevant question and correct answer.

For both runs, `route_summary.selected_counts` is `structured_qa=556`.

Staged structured QA cost-gating artifact:

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

Current staged l80 structured QA result at alpha 0.100:

| Scope | Skipped | Verified false alarm | Verified detection | Gate status |
|---|---:|---:|---:|---|
| Qwen l80 | 79.3% | 0.008 | 0.306 | pass |
| SmolLM2 l80 | 82.9% | 0.010 | 0.244 | pass |
| Aggregate | 81.1% | 0.009 | 0.275 | promote `structured_qa` |

Interpretation: staged gating keeps the exact structured QA adapter for risky
claims while skipping 902 of 1112 verifier calls across the two l80 runs. The
registered comparison artifact blocks promotion unless skip-rate, false-alarm,
and detection thresholds all pass.

Registry handoff for the current staged l80 structured QA route baseline:

```bash
python benchmarks/run_adapter_promotion_registry_workflow.py \
  --report staged=artifacts/truthfulqa_l80_structured_qa_staged_verifier_ensemble_report.json \
  --route-report-json artifacts/truthfulqa_l80_structured_qa_staged_registry_route_comparison.json \
  --artifact-manifest artifacts/truthfulqa_l80_structured_qa_staged_adapter_promotion_manifest.json \
  --verification-report artifacts/truthfulqa_l80_structured_qa_staged_manifest_verification.json \
  --registry artifacts/staged-route-registry.json \
  --name truthfulqa-l80-structured-qa-staged-route \
  --version 0.4 \
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
  --metadata evidence=truthfulqa_l80_structured_qa_staged \
  --json artifacts/truthfulqa_l80_structured_qa_staged_adapter_promotion_registry_workflow.json \
  --compact-json \
  --fail-on-blocked

python benchmarks/compare_route_baselines.py \
  --registry artifacts/staged-route-registry.json \
  --min-selected 80 \
  --min-decision-accuracy 0.95 \
  --max-false-supported-rate 0.02 \
  --min-false-refuted-rate 0.90 \
  --max-verified-false-alarm 0.02 \
  --min-verified-detection 0.20 \
  --max-p99-duration-seconds 0.01 \
  --max-mean-attempted-route-count 1.1 \
  --max-retrieval-use-rate 0.0 \
  --json artifacts/truthfulqa_l80_structured_qa_staged_route_baseline_comparison.json \
  --fail-on-blocked
```

The registry workflow records
`benchmark_manifest:truthfulqa-l80-structured-qa-staged-route:0.4`, and the
baseline comparison promotes that record with route `structured_qa`.

## `backfill_truthfulqa_statements.py`

Adds statement metadata to older `eval_truthfulqa.py --dump-scores` artifacts
without loading a model. It rebuilds the deterministic TruthfulQA eval split,
applies the original scoring order, validates exact label alignment, and can
write a label-derived oracle claim fixture for verifier-ensemble upper-bound
tests.

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
  --json artifacts/truthfulqa_l80_oracle_verifier_ensemble_report.json
```

Current l80 oracle upper-bound result at alpha 0.100:

| Run | Internal false alarm | Internal detection | Oracle verified false alarm | Oracle verified detection |
|---|---:|---:|---:|---:|
| Qwen l80 | 0.091 | 0.279 | 0.000 | 1.000 |
| SmolLM2 l80 | 0.095 | 0.229 | 0.000 | 1.000 |

Both oracle runs have `true_supported_rate=1.000`, `false_refuted_rate=1.000`,
and `decision_accuracy=1.000` in `verification_quality`.

This is not a real factual verification result. The oracle fixture is derived
from TruthfulQA labels, so it proves the control-plane and verifier benchmark can
consume perfect evidence and gives an upper bound. Replace it with real
retrieval, database, calculator, or domain/world-model evidence before making a
product-performance claim.

## `eval_score_ensemble.py`

Compares single diagnostic signals against simple calibrated rank ensembles from
saved score dumps. This is a post-processing benchmark for the question:
"Should the product combine internal signals by default, or keep the strongest
single calibrated signal?"

```bash
python benchmarks/eval_score_ensemble.py \
  --scores qwen-l80=artifacts/qwen05_truthfulqa_l80_scores.json \
  --scores smollm2-l80=artifacts/smollm2_truthfulqa_l80_scores.json \
  --signals truth_proj,maha_last,subspace_resid,resid_update_norm,eigenscore \
  --methods max_rank,mean_rank \
  --repeats 50 \
  --json artifacts/truthfulqa_score_ensemble_report.json
```

For a single score dump, the same runner can also save a deployable
`RankScoreFusionArtifact` for the best ensemble at `--best-alpha`:

```bash
python benchmarks/eval_score_ensemble.py \
  --scores artifacts/qwen05_truthfulqa_l80_scores.json \
  --signals truth_proj,maha_last,subspace_resid,resid_update_norm \
  --methods max_rank,mean_rank \
  --best-alpha 0.10 \
  --save-best-fusion-artifact artifacts/qwen05_score_fusion_artifact.json \
  --json artifacts/qwen05_score_ensemble_report.json
```

It can also evaluate the newer geometry-calibrated interaction score by
separating representation-geometry signals from uncertainty/confidence proxies:

```bash
python benchmarks/eval_score_ensemble.py \
  --scores artifacts/qwen05_truthfulqa_l80_scores.json \
  --signals truth_proj,subspace_resid,resid_update_norm \
  --geometry-signals subspace_resid,resid_update_norm \
  --uncertainty-signals nll_answer,inside_semantic_energy \
  --geometry-fusion-methods interaction,product \
  --best-alpha 0.10 \
  --save-best-geometry-fusion-artifact artifacts/qwen05_geometry_fusion_artifact.json \
  --json artifacts/qwen05_score_ensemble_report.json
```

Verifier outputs can be converted into the same score-dump interface before
running the geometry-fusion comparison:

```bash
python benchmarks/eval_verifier_ensemble.py \
  --scores qwen-l80=artifacts/qwen05_truthfulqa_l80_scores_with_statements.json \
  --qa-corpus artifacts/truthfulqa_l80_correct_answer_corpus.json \
  --signal truth_proj \
  --staged-verification \
  --verified-records-jsonl artifacts/verifier-signals/verified-records.jsonl \
  --json artifacts/verifier-signals/verifier-ensemble-report.json

python benchmarks/build_verifier_signal_score_dump.py \
  --scores artifacts/qwen05_truthfulqa_l80_scores_with_statements.json \
  --verified-records-jsonl artifacts/verifier-signals/verified-records.jsonl \
  --run-name qwen-l80 \
  --keep-signals truth_proj,maha_last,subspace_resid,eigenscore,nll_answer \
  --output artifacts/verifier-signals/qwen-l80-enhanced-scores.manifest.json \
  --output-format jsonl
```

When verified records include state-transition prediction metadata, the same
converter also emits world-model uncertainty columns such as
`world_model_disagreement`, `world_model_agreement_gap`, and
`world_model_low_agreement`, so simulator/model disagreement can be swept or
fused under the same conformal calibration path.

Simple text baselines can also be appended to statement-bearing dumps as
redline controls. This is a post-hoc check for whether a proposed detector is
actually beating answer length, claim length, lexical overlap, negation, and
number-count artifacts under the same conformal/fusion evaluation:

```bash
python benchmarks/build_text_baseline_score_dump.py \
  --scores artifacts/qwen05_truthfulqa_l80_scores_with_statements.json \
  --keep-signals truth_proj,maha_last,subspace_resid,eigenscore,nll_answer \
  --output artifacts/text-baselines/qwen-l80-text-baseline-scores.manifest.json \
  --output-format jsonl \
  --json artifacts/text-baselines/qwen-l80-text-baseline-report.json

python benchmarks/eval_score_ensemble.py \
  --scores qwen-l80=artifacts/text-baselines/qwen-l80-text-baseline-scores.manifest.json \
  --signals truth_proj,maha_last,subspace_resid,eigenscore,nll_answer,answer_char_length,answer_token_count,claim_char_length,claim_token_count,question_answer_token_overlap,answer_negation_flag,answer_number_count \
  --methods max_rank,mean_rank \
  --alphas 0.05,0.1,0.2 \
  --repeats 50 \
  --best-alpha 0.10 \
  --json artifacts/text-baselines/score-ensemble-report.json
```

The committed l80 comparison at
`artifacts/truthfulqa-l80-text-baseline-comparison/` uses both Qwen and SmolLM2
statement-bearing dumps. At alpha `0.100`, `truth_proj` remains the strongest
single signal: Qwen detection `0.279` at false alarm `0.091`, and SmolLM2
detection `0.229` at false alarm `0.095`. The strongest cheap text controls are
near-random: `answer_token_count` AUROC `0.519` with detection `0.110`,
`claim_token_count` AUROC `0.527` with detection `0.089`, and
`question_answer_token_overlap` AUROC `0.330` with zero triggered detection
under the low-overlap direction. Treat this as a redline baseline for future
verifier/retrieval/selfcheck signals.

For non-oracle local evidence experiments, `run_verifier_signal_fusion_workflow.py`
wraps the same chain into one reproducible artifact bundle. It can build a
retrieval fixture from local JSON/JSONL/text corpora, merge optional
self-consistency samples, write the verifier sidecar, convert that sidecar into
score-dump columns, run geometry fusion, save per-run geometry artifacts, and
verify the top-level manifest:

```bash
python benchmarks/run_verifier_signal_fusion_workflow.py \
  --scores smollm2-l80=artifacts/smollm2_truthfulqa_l80_scores_with_statements.json \
  --corpus artifacts/truthfulqa_l80_correct_answer_corpus.json \
  --output-dir artifacts/smollm2-l80-non-oracle-verifier-signal-fusion \
  --signal truth_proj \
  --alphas 0.05,0.1,0.2 \
  --repeats 20 \
  --keep-signals truth_proj,maha_last,subspace_resid,eigenscore,nll_answer \
  --fusion-signals truth_proj,subspace_resid,eigenscore,verifier_refuted,verifier_uncertainty,selfcheck_refute_rate \
  --geometry-signals truth_proj,subspace_resid,eigenscore \
  --uncertainty-signals verifier_refuted,verifier_uncertainty,selfcheck_refute_rate \
  --query-field answer \
  --omit-label-metadata
```

Add `--samples path/to/sample-records.json` when a score dump or external
sampler provides aligned self-consistency responses.

The workflow is intentionally post-hoc and dependency-free. It is the preferred
entry point when testing whether local retrieval and self-consistency evidence
improves a calibrated geometry monitor without rerunning model scoring.

Each selected signal is converted to a direction-aware anomaly percentile using
the split calibration true set. `max_rank` takes the most anomalous normalized
signal per item; `mean_rank` averages normalized anomaly ranks. The ensemble is
then thresholded with the same split-conformal false-alarm check as the single
signals. Each run records the same validated score-dump summary and file
fingerprint used by `eval_conformal.py`, so ensemble comparisons can be checked
against exact input artifacts. Saved fusion artifacts preserve the normal-score
orientation, reference distributions, signal directions, fusion method,
conformal alpha, and threshold. They can be supplied to
`recommend_runtime_config.py` with `--score-ensemble-report`; the runtime
recommendation only promotes a fusion signal when its selected alpha passed the
conformal false-alarm gate. They are intended for controlled follow-up
experiments rather than as a default product policy. Geometry fusion artifacts
preserve the same provenance plus geometry/uncertainty group membership and
their rank-fusion methods, so follow-up reports can compare single `truth_proj`,
naive rank fusion, and geometry-by-uncertainty interaction under one score-dump
fingerprint.

Current frontier Qwen l80 / SmolLM2 l80 result
(`artifacts/truthfulqa-frontier-qwen-smollm2-l80/`): simple internal-score
ensembles do not beat `truth_proj`. At alpha 0.100, Qwen's best single signal
detects 0.282 while the best ensemble detects 0.254; SmolLM2's best single
detects 0.240 while the best ensemble detects 0.193. The layer/score sweep is
stronger than the main layer alone: Qwen peaks at `truth_proj` layer `-10`
with AUROC 0.764, and SmolLM2 peaks at `truth_proj` layer `-16` with AUROC
0.782. Treat this as a negative result for naive score fusion, not as evidence
against richer verifier/retrieval ensembles.

The geometry-fusion replay at
`artifacts/truthfulqa-frontier-qwen-smollm2-l80-geometry-fusion/` is also
negative with the currently available uncertainty proxy. Using
`subspace_resid,resid_update_norm,eigenscore` as geometry signals and
`nll_answer` as the uncertainty proxy, the best geometry-fusion method detects
0.055 for Qwen and 0.036 for SmolLM2 at alpha 0.100. Adding `truth_proj` to the
geometry group or switching geometry aggregation to `max_rank` raises the best
fusion detection only to 0.083 for Qwen and 0.069 for SmolLM2. Treat this as a
specific rejection of `nll_answer` as a final-correction uncertainty proxy; the
next useful test needs real multi-sample semantic energy, self-consistency, or
verifier/retrieval disagreement features.

The staged structured-QA verifier-signal replay at
`artifacts/truthfulqa-l80-staged-qa-verifier-signals/` demonstrates that stronger
external evidence can enter the same calibrated fusion path. It writes
verified-record sidecars, converts them into `verifier_*` and `selfcheck_*`
score columns, with `world_model_*` columns available when transition metadata
is present, and saves per-model `GeometryScoreFusionArtifact` files. At
alpha 0.100, Qwen's best verifier single signal (`verifier_refuted`) detects
0.297 with zero false alarm, while geometry fusion detects 0.285 at false alarm
0.089; SmolLM2's geometry fusion detects 0.261 at false alarm 0.095, beating
both `truth_proj` (0.229) and `verifier_refuted` (0.232). This supports the
product direction: keep `truth_proj` as the internal monitor, but use structured
verifier/retrieval/selfcheck evidence as the final correction signal rather
than using `nll_answer`.

The local retrieval verifier-signal replay at
`artifacts/truthfulqa-l80-local-retrieval-verifier-signal-fusion/` runs the new
`run_verifier_signal_fusion_workflow.py` entry point over the committed Qwen and
SmolLM2 l80 score dumps plus the local correct-answer corpus. It does not rerun
models or copy labels into claim metadata. The verifier stage uses 410 retrieved
hits across 274/556 records, selects `retrieval_structured_qa` for 238 records,
and keeps verified false alarm at 0.016 for both models. At alpha 0.100, Qwen
verified detection is 0.316 and SmolLM2 verified detection is 0.267. The saved
geometry-fusion replay selects `noisy_or`: Qwen reaches detection 0.795 at false
alarm 0.070, and SmolLM2 reaches detection 0.795 at false alarm 0.069. Treat this
as a reproducible local-corpus baseline, not as evidence that open-domain
retrieval is solved.

The paired answer-echo stress artifact at
`artifacts/truthfulqa-l80-answer-echo-retrieval-stress/` uses the same workflow
but retrieves from a corpus built out of the audited answers themselves. It
retrieves 706 hits over 556/556 records, but false-supported rate rises to
0.980 and false-refuted rate is 0.000. At alpha 0.100, verified detection drops
to 0.013 for Qwen and 0.010 for SmolLM2. This is the expected failure mode and
should be treated as a required negative control for future retrieval evidence.

The paired cache-only replay
(`artifacts/truthfulqa-frontier-qwen-smollm2-l80-cache-only/`) reproduces the
same score records and best sweep choices from the per-cell cache root. End-to-end
frontier wall-clock drops from 1140.245s to 24.028s. Per-cell replay drops Qwen
from 784.040s to 3.496s and SmolLM2 from 336.829s to 2.797s. This makes l80
multi-seed, layer/score resweeps, and post-hoc calibration experiments practical
without re-running model forward passes.

The layer-band selector audit
(`artifacts/truthfulqa-frontier-layer-band-selection/`) combines the saved
intrinsic-dimension, spectrum, and sweep artifacts. The current recommendation is
`spectrum_max_top_eigenvalue_to_mp_upper_radius_1`: it keeps both l80 best
`truth_proj` layers in band with zero AUROC regret while averaging 2 of 5
monitored layers. This is a candidate-band prior for cheaper calibrated sweeps,
not a replacement for the sweep report or saved calibration artifact.

The registered post-hoc stability report
(`report:truthfulqa-frontier-qwen-smollm2-l80-stability:0.1`) replays seeds
`0..9` with 20 split-conformal repeats per seed. For Qwen l80, `truth_proj` is
the best single signal in 10/10 seeds and beats the best `mean_rank` ensemble in
10/10 seeds; mean detection margin is 0.034. For SmolLM2 l80, `truth_proj` is
also best in 10/10 seeds and beats `mean_rank` in 10/10 seeds; mean detection
margin is 0.053. The verified manifest is registered as
`benchmark_manifest:truthfulqa-frontier-qwen-smollm2-l80-stability:0.1`.

## `eval_calibration_transfer.py`

Applies saved `CalibrationArtifact` thresholds to saved score dumps from other
runs. Use it after `compare_transfer.py`: AUROC transfer asks whether a detector
family remains useful; calibration transfer asks whether a specific threshold
still controls false alarms under model/domain shift.

```bash
python benchmarks/eval_calibration_transfer.py \
  --artifact qwen-l80=artifacts/qwen05_truthfulqa_l80_best_calibration.json \
  --artifact smollm2-l80=artifacts/smollm2_truthfulqa_l80_best_calibration.json \
  --scores qwen-l80=artifacts/qwen05_truthfulqa_l80_scores.json \
  --scores smollm2-l80=artifacts/smollm2_truthfulqa_l80_scores.json \
  --json artifacts/truthfulqa_calibration_transfer_report.json
```

The script uses each artifact's `target_layer` and score name. If the target dump
contains matching `sweep_scores`, those layer-specific scores are used; otherwise
the primary `scores` payload is used only when its configured layer matches the
artifact. The report includes self-application and cross-application false alarm,
detection, coverage, selective accuracy, and a pass/fail flag for
`false_alarm <= conformal_alpha + tolerance`. The top-level `score_dumps` field
records the validated dump summary and SHA-256 for every target input; JSONL
targets include both manifest and records-file fingerprints.

Current Qwen l80 / SmolLM2 l80 result: self-application controls false alarms for
both artifacts, but cross-application controls 0/2. Qwen's l80 threshold applied
to SmolLM2 has false alarm 0.160 at alpha 0.100; SmolLM2's l80 threshold applied
to Qwen has false alarm 0.452 at alpha 0.100. Treat thresholds as model-specific
until a stronger transfer study proves otherwise.

## `compare_profiles.py`

Compares `eval_truthfulqa.py --profile-json` payloads, or full `--json` result
files containing a `profile` field, without loading a model. Use it to compare
baseline, statement-encoding-cache, eval-reps-cache, and cache-only runs before
claiming a benchmark path is faster.

```bash
python benchmarks/compare_profiles.py \
  --profile baseline=/tmp/eigentruth-profile-baseline.json \
  --profile enc-cache=/tmp/eigentruth-profile-enc-cache.json \
  --profile cache-only=/tmp/eigentruth-profile-cache-only.json \
  --baseline baseline \
  --json artifacts/truthfulqa_profile_comparison.json
```

The report includes total time deltas, speedup versus the baseline, phase
deltas, grouped time deltas, throughput ratios, and cache efficiency metrics
such as eval-reps shard cache hit rate / cross-shard read rate when the source
profile summary includes them. Older profile payloads that only contain
`total_seconds` and `phases` remain readable, but grouped deltas and cache
efficiency comparisons are available only when the newer `summary` field exists.
For stricter performance claims, repeat each candidate run and pass the same
profile name multiple times with `--aggregate-repeats median`; the report keeps
each source path plus min/median/max total seconds, then applies the same
regression gates to the median profile:

```bash
python benchmarks/compare_profiles.py \
  --profile baseline=/tmp/eigentruth-profile-baseline-r1.json \
  --profile baseline=/tmp/eigentruth-profile-baseline-r2.json \
  --profile candidate=/tmp/eigentruth-profile-candidate-r1.json \
  --profile candidate=/tmp/eigentruth-profile-candidate-r2.json \
  --baseline baseline \
  --aggregate-repeats median \
  --max-total-ratio 1.10 \
  --json artifacts/truthfulqa_profile_gate_median.json
```

For CI or local regression checks, add optional gate thresholds. The command
exits non-zero and writes `regression_gate.failures` when any non-baseline run
exceeds the allowed slowdown or drops below the required throughput ratio:

```bash
python benchmarks/compare_profiles.py \
  --profile baseline=/tmp/eigentruth-profile-baseline.json \
  --profile candidate=/tmp/eigentruth-profile-candidate.json \
  --baseline baseline \
  --max-total-ratio 1.10 \
  --max-phase-ratio forced_answer_forward=1.10 \
  --min-throughput-ratio forced_answer_records_per_second=0.90 \
  --json artifacts/truthfulqa_profile_gate.json
```

For cache experiments, run-specific total-time gates let cached and cache-only
paths use different expectations:

```bash
python benchmarks/compare_profiles.py \
  --profile uncached=/tmp/eigentruth-profile-uncached.json \
  --profile cached=/tmp/eigentruth-profile-cached.json \
  --profile cache_only=/tmp/eigentruth-profile-cache-only.json \
  --baseline uncached \
  --max-run-total-ratio cached=0.90 \
  --max-run-total-ratio cache_only=0.35 \
  --json artifacts/truthfulqa_cache_profile_gate.json
```

`benchmarks/run_cache_profile_triplet.py` automates a same-machine cache
comparison. It runs `eval_truthfulqa.py` once to build statement/layer/eval
caches, once to reuse them with model loading, and once in `--cache-only` mode;
then it writes a `compare_profiles.py` report. By default it uses the offline
fixture, so it is suitable for command inspection and local mechanics:

```bash
python benchmarks/run_cache_profile_triplet.py \
  --output-dir /tmp/eigentruth-cache-profile-triplet \
  --model sshleifer/tiny-gpt2 \
  --clean
```

For a minimal same-machine runtime evidence pass after cache/profile changes,
keep scratch outputs under `/tmp` and record the command plus commit SHA in the
handoff notes:

```bash
python benchmarks/run_cache_profile_triplet.py \
  --output-dir /tmp/eigentruth-cache-profile-triplet-recent \
  --model sshleifer/tiny-gpt2 \
  --limit 4 \
  --manifold-questions 2 \
  --layer -1 \
  --batch-size 2 \
  --max-length 32 \
  --eval-reps-cache-shard-size 2 \
  --eval-reps-shard-read-cache-size 2 \
  --clean \
  --fail-on-regression
```

This writes `result-*.json`, `profile-*.json`,
`cache-profile-comparison.json`, `cache-profile-triplet-commands.json`, caches,
and an artifact manifest in the output directory. The offline fixture does not
download TruthfulQA, but non-cache-only runs still load the configured model and
may download weights if they are not already in the local Hugging Face cache.
Treat the result as local runtime evidence only; publish it by promoting a
minimal report/manifest/registry bundle, not by committing the scratch caches.
For less noisy runtime claims, add `--repeats 3`; the runner writes
`profile-*-rN.json` / `result-*-rN.json`, then compares median timings with the
same run-specific gates used for single triplets:

```bash
python benchmarks/run_cache_profile_triplet.py \
  --output-dir /tmp/eigentruth-cache-profile-triplet-median \
  --model sshleifer/tiny-gpt2 \
  --limit 4 \
  --manifold-questions 2 \
  --layer -1 \
  --batch-size 2 \
  --max-length 32 \
  --eval-reps-cache-shard-size 2 \
  --eval-reps-shard-read-cache-size 2 \
  --repeats 3 \
  --clean \
  --fail-on-regression
```

Use `--real-truthfulqa` for representative model/data profile artifacts. Start
small, then increase `--limit` and `--manifold-questions` once the machine and
cache settings are known:

```bash
python benchmarks/run_cache_profile_triplet.py \
  --output-dir /tmp/eigentruth-qwen05-cache-profile \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --real-truthfulqa \
  --limit 24 \
  --manifold-questions 12 \
  --layer -12 \
  --batch-size 2 \
  --eval-reps-cache-shard-size 8 \
  --clean
```

For INSIDE sampling changes, collect sampling-cost evidence separately because
`--cache-only` intentionally does not run sampled INSIDE:

```bash
python benchmarks/run_inside_sampling_profile.py \
  --output-dir /tmp/eigentruth-inside-profile-recent \
  --model sshleifer/tiny-gpt2 \
  --limit 4 \
  --manifold-questions 2 \
  --layer -1 \
  --batch-size 2 \
  --max-length 32 \
  --inside-samples 3 \
  --inside-min-samples 2 \
  --inside-sample-step 1 \
  --inside-batch-size 2 \
  --inside-max-new-tokens 4 \
  --inside-trigger-signal truth_proj \
  --inside-trigger-top-fraction 0.5 \
  --inside-diagnostics-cache /tmp/eigentruth-inside-profile-recent/inside-diagnostics-cache.json \
  --refresh-shared-caches \
  --clean \
  --fail-on-regression
```

This writes per-run result/profile files plus
`inside-sampling-profile-comparison.json`; use its leaderboard for generated
sample counts and `inside_generation` ratios. The deterministic smoke scripts in
`make perf-check` remain gate-only checks and should not be cited as runtime
claims.

Use `--dry-run` first to inspect the exact commands without loading a model.
Add `--fail-on-regression` when using the generated gate in automation. Treat
the resulting timings as same-machine artifacts; do not quote them as general
model speed claims.

`benchmarks/run_cache_profile_matrix.py` runs multiple triplets and writes a
single `cache-profile-matrix-report.json` with one row per layer / batch-size /
hidden-state-capture combination. Use it for controlled local sweeps before
changing benchmark defaults:

```bash
python benchmarks/run_cache_profile_matrix.py \
  --output-dir /tmp/eigentruth-qwen05-profile-matrix \
  --shared-cache-dir /tmp/eigentruth-qwen05-profile-cache \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --real-truthfulqa \
  --limit 24 \
  --manifold-questions 12 \
  --layers=-16,-12,-10 \
  --batch-sizes=1,2,4 \
  --max-batch-token-budgets=0,512,1024 \
  --hidden-state-captures=outputs,hooks \
  --covariance-modes=full,diag,low_rank,shrinkage \
  --covariance-low-ranks=4 \
  --max-workers=2 \
  --dry-run
```

Remove `--dry-run` only when the full matrix cost is acceptable. The matrix
report includes each triplet's command log, gate summary, cache-only timing,
per-run bottleneck phase, AUROC quality signals when result JSON is available, and
a matrix-level `matrix_decision`. The decision is `promote` only when at least
one checked cell passes its regression gate and no checked cell fails; use
`--fail-on-blocked` on real runs to make non-promoting matrix decisions exit
non-zero. Use `--max-batch-tokens` when sequence lengths vary widely: it caps
the padded token budget per warmup/eval forward while `--batch-size` remains the
maximum row count, reducing padding spikes without changing score semantics. To
measure multiple budgets in one matrix, use `--max-batch-token-budgets 0,512`;
token-budget matrix recommendations are sorted by uncached forced-answer forward
time because the budget changes forward batching, not cache-only replay
semantics.
Use `--covariance-modes full,diag,low_rank,shrinkage` to compare the `maha_last`
TruthManifold covariance approximation as a gated runtime dimension. Shared
cache runs reuse statement encodings and eval hidden-state caches across
covariance modes, but keep layer-stats caches separate by covariance mode/rank
so warmup statistics cannot be accidentally reused across incompatible scoring
semantics. Runtime recommendations include the selected covariance flags, and
thresholds/calibration artifacts should be regenerated for the selected mode.
Use the default `triplet` matrix mode for covariance sweeps; `rescore` is kept
for cache-only variants that share the same layer-stats scoring semantics.
Runtime recommendations also include `recommendation.covariance_tradeoff` when
the matrix compares covariance candidates. A small real SmolLM2 l8 CPU sweep
(`layer=-12`, `batch_size=4`, `max_batch_tokens=512`, `limit=8`,
`manifold_questions=4`) selected `diag` by cache-only replay time (`0.156s`
versus `0.211s` for `full`) but marked `speed_quality_tradeoff` because
`maha_last` AUROC dropped from `0.614` to `0.500`. `low_rank` with rank 8 kept
`maha_last` close to `full` (`0.610`) with cache-only replay `0.197s`. Treat
`diag` as a latency/memory candidate only when `maha_last` is not the deployed
calibrated signal; prefer `full`, validated `low_rank`, or validated
`shrinkage` when `maha_last` quality matters. The tiny offline shrinkage smoke
in `artifacts/tiny_covariance_shrinkage_matrix/` validates the
`full/diag/low_rank/shrinkage` matrix and runtime-recommendation path, but it is
not benchmark-quality evidence.
For real l80 post-hoc covariance gates, use
`rebuild_layer_stats_from_warmup_checkpoint.py` to derive covariance-specific
layer-stats caches from existing warmup hidden states, then run
`run_calibrated_observability_workflow.py --cache-only` against the existing
eval-reps cache. The current Qwen/SmolLM2 l80 gate report is
`artifacts/truthfulqa-frontier-covariance-gate-l80/covariance-mode-gate-report.json`:
`shrinkage` preserves or slightly improves best `maha_last` AUROC for both
models, `low_rank_16` passes for Qwen but misses the 0.01 AUROC-drop gate for
SmolLM2, and `diag` is rejected for both models.
Use `--max-workers` to run independent matrix cells concurrently. The default is
`1` for fully serial/reproducible local runs. Matrix reports include
`execution.wall_clock_seconds` plus per-cell `execution_seconds`, so worker-count
comparisons should use end-to-end wall-clock time instead of summing profile
totals. When `--shared-cache-dir` is set, the runner keeps refresh cells serial
until shared statement/layer/eval caches exist, then executes dependent
warm-start/cache-only cells concurrently up to the worker limit.
To compare worker counts in one reproducible run, use
`run_cache_worker_sweep.py`. It runs the same matrix under each worker count,
writes `cache-worker-sweep-report.json`, and recommends the fastest worker count
whose matrix promoted. If a shared-cache root is provided, each worker count gets
an isolated subdirectory so later runs do not inherit warmed caches from earlier
worker counts:

```bash
python benchmarks/run_cache_worker_sweep.py \
  --output-dir /tmp/eigentruth-qwen05-worker-sweep \
  --shared-cache-dir /tmp/eigentruth-qwen05-worker-sweep-cache \
  --worker-counts=1,2 \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --real-truthfulqa \
  --limit 8 \
  --manifold-questions 4 \
  --layers=-16,-12 \
  --batch-sizes=1 \
  --fail-on-blocked
```

After a promoted matrix or worker sweep, build a compact runtime recommendation
without rerunning any model work:

```bash
python benchmarks/recommend_runtime_config.py \
  --matrix-report /tmp/eigentruth-qwen05-worker-sweep/workers_1/cache-profile-matrix-report.json \
  --worker-sweep-report /tmp/eigentruth-qwen05-worker-sweep/cache-worker-sweep-report.json \
  --inside-sampling-report /tmp/eigentruth-qwen05-inside/inside-sampling-profile-comparison.json \
  --inside-trigger-budget-sweep-report /tmp/eigentruth-qwen05-trigger/inside-trigger-budget-sweep.json \
  --inside-trigger-budget-policy quality_balanced \
  --score-ensemble-report artifacts/qwen05_score_ensemble_report.json \
  --output /tmp/eigentruth-qwen05-worker-sweep/runtime-recommendation.json \
  --fail-on-blocked
```

The recommendation records the selected layer, batch size, hidden-state capture
mode, covariance mode/rank, padded-token budget, prefix-KV mode, worker count,
all finite AUROC quality signals from the promoted cell, optional promoted
INSIDE sampling settings, the best quality signal, covariance tradeoff details
when covariance candidates were swept, and cache-tuning advice when the promoted
cell's eval-reps cache efficiency shows low shard-cache hit rate, high
cross-shard read rate, or tiny cache read ranges. With `--inside-sampling-report`, the
recommended sampling run must pass its sample-efficiency gate and expose a
readable per-run result JSON; otherwise the runtime recommendation fails closed.
With `--inside-trigger-budget-sweep-report`, the recommendation uses
`quality_balanced_recommendation` when present, otherwise the cost-first
recommendation by default, and emits `run_inside_trigger_budget_sweep.py` flags
including `--derive-from-max-budget` for derived nested top-fraction sweeps. Use
`--inside-trigger-budget-policy cost_first` for latency-constrained deployments,
`quality_balanced` for the default release posture, or `quality_first` when the
highest measured INSIDE quality metric is worth the extra sampled generation
cost. The selected policy is written into the runtime recommendation evidence
and readiness/registry metadata.
With `--score-ensemble-report`, a best fusion signal is added to
`quality_signals` only when the selected ensemble alpha passed its conformal
false-alarm gate; blocked or ambiguous fusion evidence is retained in
`score_fusion` and `evidence` without changing the runtime recommendation.
When the matrix report and worker-sweep child matrix are separate files, the
recommendation still treats them as compatible if they select the same promoted
runtime cell and matching quality/cache-only evidence; this lets a serial matrix
provide the selected quality/cost row while a separate worker-count sweep
provides the wall-clock worker recommendation.
The report includes equivalent flags for `eval_truthfulqa.py`,
`run_cache_profile_matrix.py`, `run_adapter_readiness_workflow.py`, and, when
sampling evidence is provided, `run_inside_sampling_profile.py` and
`run_inside_trigger_budget_sweep.py`. Treat it as the deployment handoff from
same-machine performance evidence; it does not replace a promoted matrix,
worker-sweep, sampling-profile, or trigger-budget-sweep decision.

Use `run_performance_baseline_workflow.py` when the handoff itself should be a
registered, fingerprinted artifact bundle. It can run the cache-profile matrix
directly or reuse existing matrix/worker/INSIDE/score-ensemble reports, then writes
`performance-baseline-workflow.json`, `runtime-recommendation.json`, an artifact
manifest, a top-level `performance_evidence_bundle` summary with recommendation
cost ratios / evidence status / artifact readiness / score-dump cache evidence /
score-fusion status, and an optional
`performance_baseline:*:*` registry record. When it reuses an existing matrix
report, the top-level config, performance evidence runtime block, and artifact
manifest metadata inherit the matrix report's effective runtime settings. Add
`--verify-manifest` to recursively verify the written manifest, save a
`manifest-verification.json` report, and register
`manifest_verification:<name>-verification:<version>` alongside the performance
baseline without making the verification report part of the manifest it verifies:

```bash
python benchmarks/run_performance_baseline_workflow.py \
  --output-dir /tmp/eigentruth-qwen05-performance-baseline \
  --registry artifacts/registry.json \
  --name qwen05-performance-baseline \
  --version 0.1 \
  --matrix-report /tmp/eigentruth-qwen05-worker-sweep/workers_1/cache-profile-matrix-report.json \
  --worker-sweep-report /tmp/eigentruth-qwen05-worker-sweep/cache-worker-sweep-report.json \
  --inside-trigger-budget-sweep-report /tmp/eigentruth-qwen05-trigger/inside-trigger-budget-sweep.json \
  --inside-trigger-budget-policy quality_balanced \
  --score-ensemble-report artifacts/qwen05_score_ensemble_report.json \
  --verify-manifest \
  --fail-on-blocked
```

Use `run_product_runtime_baseline.py` for the product-control side of the same
performance story: aggregate saved `ProductTrace` JSON files, summarize request
phase timings, route costs, cache hit rates, retrieval use, staged-verification
skip savings, verification-scope counts, triggered-only partial skip savings,
and optionally apply a `ProductRuntimeBudgetPolicy` or promoted
`ProductPromotionContract` budget. The output includes `optimization.hotspots`,
`optimization.recommendations`, and `optimization.policy_hints`, turning the
baseline into an actionable performance pass over slow phases/routes, low cache
hit rates, excessive retrieval or verifier fanout, missing staged verification,
missing triggered-claim-only staged verification, and audit-heavy
runtime-profile distributions. Use full `ProductTrace.to_dict()`
payloads for this workflow; bounded telemetry from `--bounded-trace` is
intentionally rejected because it can truncate replay-relevant evidence and
action outputs. For large trace sets, add `--trace-records-jsonl` to stream
per-trace metric, budget, and runtime-profile context records into a JSONL
sidecar while keeping the main report focused on summary, budget, optimization,
manifest, and registry metadata. Add `--save-recommended-policy` to write
`optimization.policy_hints.candidate_runtime_budget_policy` as a reusable
`ProductRuntimeBudgetPolicy` JSON that can be passed back through `--policy` in
later baseline, replay, profile-sweep, or
`compare_product_runtime_baselines.py --runtime-budget-policy` gates. The same
`optimization.policy_hints` block also emits `candidate_control_defaults`, such
as a recommended `max_verifier_route_attempts` when route-budget exhaustion is
observed. Add
`--trace-records-cache-json` when repeatedly sweeping runtime budget policies
over unchanged traces; the cache is keyed by source trace fingerprints and the
resolved policy payload. `ProductRuntimeBudgetPolicy` can gate overall staged
verification savings with `min_verification_skip_rate`, and can specifically
gate triggered-only partial verification with `min_selective_claim_skip_rate`.
The latter fails closed unless the trace records `verification_scope="triggered"`
and a finite selective claim skip rate:

```bash
python benchmarks/run_product_runtime_baseline.py \
  --trace artifacts/demo-request-a.json \
  --trace artifacts/demo-request-b.json \
  --promotion-contract artifacts/smollm2_product_promotion_contract_v1_6/product-promotion-contract.json \
  --json artifacts/product-runtime-baseline.json \
  --trace-records-jsonl artifacts/product-runtime-baseline-trace-records.jsonl \
  --trace-records-cache-json artifacts/product-runtime-baseline-trace-record-cache.json \
  --save-recommended-policy artifacts/product-runtime-baseline-recommended-policy.json \
  --artifact-manifest artifacts/product-runtime-baseline-manifest.json \
  --registry artifacts/local-release-registry.json \
  --name smollm2-product-runtime-baseline \
  --version 0.1 \
  --fail-on-blocked
```

This report does not replace the model/cache performance baseline above. It
checks the actual control-plane trace shape that a product path emits: route
attempt counts, retrieval use, phase tails, cache metadata, and low-risk
fast-path verifier savings. Add
`--compact-json` when the report and manifest are consumed by automation and
diff readability is less important than artifact size.

Use `run_product_feedback_report.py` after collecting manual review, user
feedback, or online outcome labels for saved `ProductTrace` payloads. Feedback
records are JSONL `ProductFeedbackRecord` objects with `request_id`, `outcome`,
and optional `trace_fingerprint`, `claim_id`, `feedback_source`,
`corrected_text`, `evidence_refs`, and `metadata`. The report joins feedback by
trace fingerprint when available, otherwise by unique request id, then measures
control-loop failure modes such as accepted-but-wrong,
retrieved-but-still-unsupported, and abstain-false-positive. Optional gates can
fail closed when the
observed feedback rate violates a product threshold:

```bash
python benchmarks/run_product_feedback_report.py \
  --trace artifacts/demo-request-a.json \
  --trace artifacts/demo-request-b.json \
  --feedback-jsonl artifacts/product-feedback.jsonl \
  --json artifacts/product-feedback-report.json \
  --artifact-manifest artifacts/product-feedback-manifest.json \
  --registry artifacts/local-release-registry.json \
  --name product-feedback-audit \
  --version 0.1 \
  --min-matched-feedback-count 20 \
  --max-accepted-but-wrong-rate 0.05 \
  --max-retrieved-failure-rate 0.10 \
  --max-abstain-false-positive-rate 0.20 \
  --fail-on-blocked
```

This feedback report does not rerun models, verifiers, or retrieval. It is the
post-hoc product quality layer that turns captured traces and outcome labels
into a small set of actionable control-policy metrics.

Use `recommend_control_policy_from_feedback.py` when those feedback metrics
should produce an explicit candidate policy artifact. The runner consumes one or
more `run_product_feedback_report.py` outputs, aggregates their matched feedback
counts, and recommends a candidate `ControlPolicyConfig` plus optional runtime
control defaults. Accepted-but-wrong feedback increases staged verification for
sensitive claims, retrieval failures can move unsupported claims toward
clarification/abstention, and abstain false positives can de-escalate compound
unsupported decisions when safety feedback is not also elevated:

```bash
python benchmarks/recommend_control_policy_from_feedback.py \
  --feedback-report artifacts/product-feedback-report.json \
  --json artifacts/feedback-policy-recommendation.json \
  --save-control-policy artifacts/candidate-control-policy.json \
  --save-control-defaults artifacts/candidate-control-defaults.json \
  --artifact-manifest artifacts/feedback-policy-manifest.json \
  --registry artifacts/local-release-registry.json \
  --name feedback-policy-recommendation \
  --version 0.1 \
  --min-matched-feedback-count 20 \
  --max-accepted-but-wrong-rate 0.05 \
  --max-retrieved-failure-rate 0.10 \
  --max-abstain-false-positive-rate 0.20
```

This recommendation is intentionally deterministic and auditable. It does not
replace A/B testing or domain review; it creates the policy artifact that can be
passed into later replay, runtime-profile, or calibrated-control demo runs.

Use `audit_feedback_policy_replay.py` before promoting that candidate policy. It
does not rerun the model or claim to exactly recompute controller decisions.
Instead, it audits concrete counterfactual coverage over the same feedback
labels: accepted-but-wrong cases covered by newly required sensitive-claim
verification, retrieval failures routed to safer clarification/abstention,
remaining safety issues, claim-metadata gaps, and overblocking relief:

```bash
python benchmarks/audit_feedback_policy_replay.py \
  --feedback-report artifacts/product-feedback-report.json \
  --policy-recommendation artifacts/feedback-policy-recommendation.json \
  --json artifacts/feedback-policy-replay-audit.json \
  --artifact-manifest artifacts/feedback-policy-replay-manifest.json \
  --registry artifacts/local-release-registry.json \
  --name feedback-policy-replay-audit \
  --version 0.1 \
  --min-matched-feedback-count 20 \
  --min-safety-coverage 0.70 \
  --max-unknown-safety-issue-rate 0.20 \
  --fail-on-blocked
```

Treat a passing audit as an offline promotion gate for the candidate policy
artifact, not as proof that future traffic is fixed. The next step should still
be trace replay or a controlled product rollout.

Use `run_feedback_policy_workflow.py` when the same handoff should be produced
as one reproducible artifact bundle. It builds or reuses the product feedback
report, writes the candidate `ControlPolicyConfig` and runtime defaults, embeds
those configs in the top-level workflow decision, runs the replay audit,
fingerprints the child reports/manifests, and optionally records the top-level
workflow report in the local registry:

```bash
python benchmarks/run_feedback_policy_workflow.py \
  --trace artifacts/demo-request-a.json \
  --trace artifacts/demo-request-b.json \
  --feedback-jsonl artifacts/product-feedback.jsonl \
  --output-dir artifacts/feedback-policy-workflow \
  --registry artifacts/local-release-registry.json \
  --name feedback-policy-workflow \
  --version 0.1 \
  --recommendation-min-matched-feedback-count 20 \
  --recommendation-max-accepted-but-wrong-rate 0.05 \
  --recommendation-max-retrieved-failure-rate 0.10 \
  --recommendation-max-abstain-false-positive-rate 0.20 \
  --replay-min-matched-feedback-count 20 \
  --min-safety-coverage 0.70 \
  --max-unknown-safety-issue-rate 0.20 \
  --fail-on-blocked \
  --fail-on-needs-evidence
```

The top-level workflow status is `recommend` when the feedback-derived
candidate policy is recommended and the replay audit passes. It remains
`observed` when no policy change is needed, `needs_evidence` when feedback count
is insufficient, and `blocked` when a child gate fails.

Registered workflow records can be passed directly to
`compare_release_candidates.py` or `run_release_candidate_registry_workflow.py`
with `--feedback-policy-workflow-key report:<name>:<version>`. `recommend`
means the workflow produced a candidate policy change; `observed` means the
feedback audit was healthy and no policy change was needed. Both statuses can
pass the release gate when the manifest verifies and any configured feedback
coverage/safety thresholds pass.

Use `compare_product_runtime_baselines.py` after a fresh trace baseline has been
built. It compares that current baseline against a file path or a registered
`product_runtime_baseline:*:*` record and can fail closed on latency, route cost,
retrieval-use, cache-hit-rate, verifier-skip-rate, and trace-count drift. When a
saved `ProductRuntimeBudgetPolicy` is supplied with `--runtime-budget-policy` or
`--runtime-budget-policy-key`, the current baseline summary is also checked
against the reusable budget using p95/aggregate metrics. When a
file baseline is used, `--registry` can still be supplied to register only the
new drift report:

```bash
python benchmarks/compare_product_runtime_baselines.py \
  --registry artifacts/local-release-registry.json \
  --baseline artifacts/smollm2_product_runtime_profile_sweep/baselines/auto/product-runtime-baseline.json \
  --current artifacts/smollm2_product_trace_replay_workflow/runtime-baseline/product-runtime-baseline.json \
  --json artifacts/smollm2_product_runtime_drift_v1_6/product-runtime-drift.json \
  --artifact-manifest artifacts/smollm2_product_runtime_drift_v1_6/artifact-manifest.json \
  --name smollm2-product-runtime-drift \
  --version 0.2 \
  --max-total-seconds-mean-ratio 1.3 \
  --max-total-seconds-p95-ratio 1.6 \
  --max-mean-route-duration-ratio 1.2 \
  --max-p95-route-duration-ratio 1.2 \
  --max-mean-attempted-route-count-delta 0.0 \
  --max-retrieval-use-rate-delta 0.0 \
  --max-cache-hit-rate-drop 0.0 \
  --max-verification-skip-rate-drop 0.0 \
  --min-current-trace-count 12 \
  --metadata evidence=smollm2_product_runtime_drift_refresh_v1_6 \
  --fail-on-drift
```

Use `build_product_trace_corpus.py` before replaying real product traffic. It
loads ProductTrace JSON files or JSONL streams, validates the control fields,
optionally requires runtime traces, redacts text-like fields by default, adds a
stable `metadata.runtime_replay_key`, writes standardized trace files plus
`runtime-pair-index.json`, can reuse per-source validation/redaction results
with `--source-cache-json` for repeated local handoffs, builds a manifest, and
can register the corpus:

```bash
python benchmarks/build_product_trace_corpus.py \
  --trace-glob 'artifacts/smollm2_product_runtime_profile_sweep/traces/*/*.json' \
  --output-dir artifacts/smollm2_product_trace_corpus \
  --registry artifacts/local-release-registry.json \
  --name smollm2-product-trace-corpus \
  --version 0.1 \
  --require-runtime-trace \
  --source-cache-json artifacts/smollm2_product_trace_corpus/source-cache.json \
  --fail-on-blocked
```

The current registered SmolLM2 trace corpus is
`report:smollm2-product-trace-corpus:0.1`; it standardizes 12 deterministic
profile-sweep traces, rejects none, redacts all claim/evidence text, preserves
three logical replay keys, and keeps four traces per runtime profile. Feed
`artifacts/smollm2_product_trace_corpus/traces/*.json` into
`run_product_runtime_baseline.py` or `run_runtime_profile_selector_replay.py`
when validating control policies against replay-ready traces rather than raw
product logs; pass
`artifacts/smollm2_product_trace_corpus/runtime-pair-index.json` to selector
replay with `--runtime-pair-index` to avoid rebuilding the paired-runtime index
from trace files.

Use `run_product_trace_replay_workflow.py` when the raw-trace handoff should be
one reproducible command. It builds the redacted corpus, runs the product
runtime baseline over the standardized traces, runs selector replay with the
provided candidate policies using the corpus runtime-pair index, writes a
recursive top-level manifest over all child reports, records phase timing/cache
summaries for local performance tuning, lifts the runtime baseline
`optimization` status/recommendations/policy hints into the top-level workflow
report and registry metadata, can save the runtime baseline's recommended
`ProductRuntimeBudgetPolicy` artifact for later gates, can run the current
runtime baseline through a product-runtime drift/policy gate against a prior
baseline, and registers one
workflow report.
Add `--verify-manifest` to write a separate recursive verification
report and register `manifest_verification:<name>-verification:<version>` next
to the workflow report. Add `--fingerprint-cache` when repeating local checks,
`--corpus-source-cache-json` when only some raw trace files change,
`--corpus-cache-json` when the entire standardized corpus can be reused,
`--runtime-trace-records-cache-json` when sweeping runtime budget gates,
`--save-runtime-recommended-policy` when the workflow should materialize the
observed baseline's candidate budget thresholds, and
`--runtime-drift-baseline` plus optional `--runtime-drift-budget-policy` when
the workflow should immediately validate the current runtime baseline against
the previous promoted baseline/policy gate. Add
`--selector-trace-inputs-json` when replaying
selector policies repeatedly over unchanged standardized traces:

```bash
python benchmarks/run_product_trace_replay_workflow.py \
  --trace-glob 'artifacts/smollm2_product_runtime_profile_sweep/traces/*/*.json' \
  --output-dir artifacts/smollm2_product_trace_replay_workflow \
  --candidate default=artifacts/smollm2_runtime_profile_selector_tuning/policies/default.json \
  --candidate latency_biased=artifacts/smollm2_runtime_profile_selector_tuning/policies/latency_biased.json \
  --candidate audit_biased=artifacts/smollm2_runtime_profile_selector_tuning/policies/audit_biased.json \
  --replay-policy artifacts/smollm2_runtime_profile_selector_replay/runtime-profile-selector-replay-policy.json \
  --registry artifacts/local-release-registry.json \
  --name smollm2-product-trace-replay-workflow \
  --version 0.1 \
  --require-runtime-trace \
  --verify-manifest \
  --fingerprint-cache artifacts/smollm2_product_trace_replay_workflow/fingerprints.json \
  --corpus-cache-json artifacts/smollm2_product_trace_replay_workflow/corpus-cache.json \
  --corpus-source-cache-json artifacts/smollm2_product_trace_replay_workflow/corpus/source-cache.json \
  --runtime-trace-records-cache-json artifacts/smollm2_product_trace_replay_workflow/runtime-baseline/trace-record-cache.json \
  --save-runtime-recommended-policy artifacts/smollm2_product_trace_replay_workflow/runtime-baseline/recommended-policy.json \
  --runtime-drift-baseline artifacts/smollm2_product_runtime_profile_sweep/baselines/auto/product-runtime-baseline.json \
  --runtime-drift-budget-policy artifacts/product-runtime-baseline-recommended-policy.json \
  --max-runtime-drift-total-seconds-p95-ratio 1.6 \
  --min-runtime-drift-current-trace-count 12 \
  --selector-trace-inputs-json artifacts/smollm2_product_trace_replay_workflow/selector-replay/trace-inputs.json \
  --fail-on-blocked
```

The current registered workflow is
`report:smollm2-product-trace-replay-workflow:0.1`. It promotes the default
selector, observes 12 runtime traces in the baseline report, writes
`corpus/runtime-pair-index.json` and
`selector-replay/trace-inputs.json`, and now also registers
`manifest_verification:smollm2-product-trace-replay-workflow-verification:0.1`
after recursively verifying the corpus/runtime-baseline/selector-replay
manifests. It keeps the same selector replay evidence as the standalone replay:
full paired runtime coverage, observed selected-runtime mean around `0.00049s`,
and p95 around `0.00059s`.

Use `run_product_runtime_profile_sweep.py` to generate comparable traces for
the built-in `latency`, `balanced`, and `audit` product profiles plus the
request-level `auto` selector before choosing which mode to put on the default
path:

```bash
python benchmarks/run_product_runtime_profile_sweep.py \
  --output-dir artifacts/product-runtime-profile-sweep \
  --promotion-contract artifacts/smollm2_product_promotion_contract_v1_6/product-promotion-contract.json \
  --runtime-profile-selector-policy artifacts/product-runtime-profile-sweep/runtime-profile-selector-policy.json \
  --slo-policy artifacts/product-runtime-profile-sweep/runtime-profile-slo-policy.json \
  --trace-records-cache-dir artifacts/product-runtime-profile-sweep/trace-record-caches \
  --registry artifacts/local-release-registry.json \
  --name smollm2-product-runtime-profile-sweep \
  --version 0.1 \
  --fail-on-blocked
```

The sweep runs deterministic calibrated-control demo scenarios, writes one trace
per mode/scenario/repeat, builds a `run_product_runtime_baseline.py` report for
each mode, records the actual selected runtime profile for `auto`, optionally
applies a sweep-level `--slo-policy`, and ranks the non-blocked modes by request
runtime and route cost. When `--promotion-contract` provides control defaults,
the trace row, profile row, leaderboard, manifest, and registry metadata carry
the effective defaults and `max_verifier_route_attempts` summary, so the selected
mode can be audited without reopening every trace.
`--runtime-profile-selector-policy` configures the request-time `auto` selector
before each trace is emitted. `--policy` still
applies the `ProductRuntimeBudgetPolicy` to each trace in each per-mode
baseline; `--slo-policy` applies aggregate gates such as
`max_total_seconds_p95`, `max_mean_attempted_route_count`,
`max_route_budget_exhaustion_rate`, `min_verification_skip_rate_mean`,
`max_verified_claim_count_mean`,
`min_verification_partial_skip_trace_count`,
`min_verification_selective_claim_skip_rate`, and
`min_auto_selected_profile_counts` to the profile row. This is the
product-control counterpart to model-side cache/profile sweeps. Profile rows
also surface selective staged-verification evidence such as
`verification_partial_skip_trace_count` and
`verification_selective_claim_skip_rate`, so latency/balanced profiles can prove
they saved verifier work by verifying only triggered claims. Use the default
`--max-workers 1` when timing will be used as promotion evidence. Use
`--max-workers N` to run independent modes concurrently for faster smoke and
coverage scans; within each mode, trace order remains deterministic before its
baseline is built. Add `--compact-json` to minify generated traces, per-mode
baselines, the top-level report, and manifests without changing the payload
schema. Add `--trace-records-cache-dir` when repeatedly tuning runtime budget or
SLO gates over the same traces; the sweep writes one baseline trace-record cache
per profile and reports cache hits/writes in each profile row, the leaderboard,
manifest metadata, and registry metadata. Add `--reuse-existing-traces` for
repeat runs that should skip calibrated-control demo execution and reuse the
existing trace JSON files before rebuilding or reading the cached baselines.

After the profile sweep and quality/release gates are available, use
`run_release_efficiency_report.py` as the final product-control efficiency
handoff. It does not rerun traces or models; it reads the profile sweep report,
optionally attaches release/readiness/performance quality reports, then ranks
profiles by an explicit runtime/verifier/cache efficiency heuristic:

```bash
python benchmarks/run_release_efficiency_report.py \
  --profile-sweep artifacts/product-runtime-profile-sweep/product-runtime-profile-sweep.json \
  --quality-report artifacts/release-candidate-comparison.json \
  --json artifacts/product-runtime-profile-sweep/release-efficiency-report.json \
  --registry artifacts/local-release-registry.json \
  --name smollm2-release-efficiency \
  --version 0.1 \
  --fail-on-blocked
```

The report surfaces generated vs reused trace counts, trace-record cache
hits/writes, verifier skip/selective-skip rates, route fanout, retrieval use,
and per-profile efficiency scores. A blocked profile sweep or blocked quality
report fails closed; without quality reports the status remains observational.

Current registered SmolLM2 product runtime profile sweep:
`report:smollm2-product-runtime-profile-sweep:0.1` in
`artifacts/local-release-registry.json`. It uses the strict structured-retrieval
audit promotion contract, verifies the artifact manifest, promotes all three
static profiles plus `auto` under `max_mean_attempted_route_count=1.1`,
`max_retrieval_use_rate=0.0`, and `max_p99_route_duration_seconds=0.01`, then
adds the profile SLO policy at
`artifacts/smollm2_product_runtime_profile_sweep/runtime-profile-slo-policy.json`
with `max_total_seconds_p95=0.05`, `max_verified_claim_count_mean=2.0`, and an
auto selector distribution gate requiring one `latency`, `balanced`, and `audit`
selection across the deterministic scenario set. The auto selector itself is
fixed by
`artifacts/smollm2_product_runtime_profile_sweep/runtime-profile-selector-policy.json`
so routing thresholds can be tuned and audited separately from SLO gates. The
sweep recommends `auto`. Skipped staged-verification paths with no verifier
route are counted as zero route cost for route-cost budget checks.

Use `run_runtime_profile_selector_tuning.py` when the question is not which
runtime profile to deploy, but which request-level `auto` selector policy should
drive it. The tuner writes one candidate policy file per policy, runs
`run_product_runtime_profile_sweep.py --profiles auto` for each candidate,
applies the same `--slo-policy`, and recommends the lowest-cost promoted
selector:

```bash
python benchmarks/run_runtime_profile_selector_tuning.py \
  --output-dir artifacts/runtime-profile-selector-tuning \
  --promotion-contract artifacts/smollm2_product_promotion_contract_v1_6/product-promotion-contract.json \
  --slo-policy artifacts/smollm2_product_runtime_profile_sweep/runtime-profile-slo-policy.json \
  --registry artifacts/local-release-registry.json \
  --name smollm2-runtime-profile-selector-tuning \
  --version 0.1 \
  --fail-on-blocked
```

Pass `--candidate name=policy.json` repeatedly to compare explicit selector
policies. With no candidates, the tuner compares the built-in `default`,
`latency_biased`, and `audit_biased` policies. Current registered selector
tuning: `report:smollm2-runtime-profile-selector-tuning:0.1` in
`artifacts/local-release-registry.json`. It promotes the default selector and
blocks the biased candidates because they fail the auto selector distribution
SLO.

Use `run_runtime_profile_selector_replay.py` when saved `ProductTrace` files
already exist and the selector question should be answered without rerunning
demo scenarios or verifier work. The replay runner loads each trace's
`risk_decision` and claim metadata, applies each candidate
`RuntimeProfileSelectorPolicy`, estimates a configurable profile-cost summary,
matches selected profiles back to paired traces when the same logical request
was saved under multiple runtime profiles, optionally reads a
`build_product_trace_corpus.py` runtime-pair index instead of scanning traces to
build that pairing map, reports selected-vs-original runtime deltas, applies
optional distribution, observed-runtime, and runtime-delta replay gates, writes
a manifest, and can register the replay report. Add `--trace-inputs-json` for a
minimal trace replay-input cache keyed by source trace fingerprints; repeated
selector sweeps over unchanged traces can then avoid rescanning full
ProductTrace JSON, while `--refresh-trace-inputs` forces a rebuild. The runner
expects full ProductTrace payloads when building the cache and rejects bounded
telemetry inputs. The
current registered replay report promotes the default selector with
100% paired runtime coverage, observed mean selected runtime around `0.00045s`,
and observed p95 selected runtime around `0.00059s` on the local deterministic
SmolLM2 trace set:

```bash
python benchmarks/run_runtime_profile_selector_replay.py \
  --trace-glob 'artifacts/smollm2_product_runtime_profile_sweep/traces/*/*.json' \
  --output-dir artifacts/smollm2_runtime_profile_selector_replay \
  --candidate default=artifacts/smollm2_runtime_profile_selector_tuning/policies/default.json \
  --candidate latency_biased=artifacts/smollm2_runtime_profile_selector_tuning/policies/latency_biased.json \
  --candidate audit_biased=artifacts/smollm2_runtime_profile_selector_tuning/policies/audit_biased.json \
  --replay-policy artifacts/smollm2_runtime_profile_selector_replay/runtime-profile-selector-replay-policy.json \
  --runtime-pair-index artifacts/smollm2_product_trace_replay_workflow/corpus/runtime-pair-index.json \
  --trace-inputs-json artifacts/smollm2_runtime_profile_selector_replay/trace-inputs.json \
  --registry artifacts/local-release-registry.json \
  --name smollm2-runtime-profile-selector-replay \
  --version 0.1 \
  --fail-on-blocked
```

Current registered SmolLM2 l20 performance baseline:
`performance_baseline:smollm2-l20-performance-baseline:0.9` in
`artifacts/local-readiness-registry.json`. It reuses the promoted real-model
cache matrix at `artifacts/smollm2_l20_readiness_inside/cache-profile-matrix/`
and the derived trigger-budget sweep at
`artifacts/smollm2_l20_inside_trigger_budget_sweep_derived/`. The runtime
recommendation selects layer `-12`, batch size `8`, `truth_proj` AUROC `0.682`,
cache-only replay `0.339s`, and the quality-balanced top-40% triggered
`adaptive_selfcheck` budget with sample-count ratio `0.472` and
`inside_generation` ratio `0.503` versus the full-sample reference.

Current small CPU SmolLM2 l8 performance evidence:
`artifacts/smollm2_l8_performance_baseline/performance-baseline-workflow.json`
reuses the real TruthfulQA cache matrix at
`artifacts/smollm2_l8_runtime_profile_matrix/`. It selects layer `-12`, batch
size `4`, `outputs` capture, `max_batch_tokens=512`, `truth_proj` AUROC `0.830`,
uncached forced-answer forward `18.150s`, cached total `14.603s`, and cache-only
replay `0.194s`. The cache-tuning summary recommends increasing
`--eval-reps-shard-read-cache-size` from `2` to `4` because shard cache hit rate
is low on this small-batch run.
The follow-up read-cache sweep baseline at
`artifacts/smollm2_l8_read_cache_sweep_performance_baseline/` compares
read-cache sizes `1,2,4` on the same layer/batch/token-budget setup and
registers
`performance_baseline:smollm2-l8-read-cache-sweep-performance-baseline:0.1`.
It promotes read-cache size `2`: cache-only replay is `0.192s` versus `0.202s`
for size `1` and `0.195s` for size `4`, with the same `truth_proj` AUROC
`0.830`. Because the capacity was explicitly swept, runtime recommendation
marks `cache_tuning.status=ok` and records `read_cache_sweep.status=swept`
instead of emitting further read-cache heuristic advice.
The follow-up worker-count baseline at
`artifacts/smollm2_l8_read_cache_worker_sweep_performance_baseline/` reuses that
read-cache sweep matrix as the quality/cost source and folds in
`artifacts/smollm2_l8_read_cache_worker_sweep/cache-worker-sweep-report.json`.
It registers
`performance_baseline:smollm2-l8-read-cache-worker-sweep-performance-baseline:0.1`,
plus
`manifest_verification:smollm2-l8-read-cache-worker-sweep-performance-baseline-verification:0.1`.
It selects `max_workers=2` and records `worker_matrix_report_matches=true`
because the worker sweep recommends the same runtime cell and quality evidence.
On this local CPU run, worker count 2 reduces matrix wall-clock from `184.467s`
to `141.385s` (`23.4%` lower) while keeping read-cache size `2`, cache-only
replay `0.192s`, and `truth_proj` AUROC `0.830`.
The score-fusion handoff baseline at
`artifacts/smollm2_l8_read_cache_worker_sweep_score_fusion_performance_baseline/`
reuses the same worker-sweep matrix and
`artifacts/truthfulqa_score_ensemble_report.json`, registers
`performance_baseline:smollm2-l8-read-cache-worker-sweep-score-fusion-performance-baseline:0.2`,
and records
`manifest_verification:smollm2-l8-read-cache-worker-sweep-score-fusion-performance-baseline-verification:0.2`.
It keeps `truth_proj` as the best quality signal while adding promoted
`score_fusion_mean_rank` evidence (`AUROC=0.679`, false alarm `0.090`,
detection `0.196`, `alpha=0.1`) to the runtime recommendation and performance
evidence bundle.
The corresponding staged structured-QA release candidate is registered as
`benchmark_manifest:smollm2-l8-read-cache-worker-sweep-score-fusion-staged-qa-release-candidate:0.2`
with
`artifacts/smollm2_l8_read_cache_worker_sweep_score_fusion_staged_release_candidate_manifest.json`.
Its manifest and release registry metadata carry
`recommended_score_fusion_status=promote`,
`recommended_score_fusion_signal=score_fusion_mean_rank`,
`recommended_score_fusion_auroc=0.679`,
`recommended_score_fusion_conformal_gate_passed=true`, and
`performance_score_ensemble_report=artifacts/truthfulqa_score_ensemble_report.json`.

Use `compare_readiness_baselines.py` after registering multiple readiness
manifests to choose among model/runtime candidates using verified manifests,
best AUROC quality signal, and explicit runtime gates. Its uncached forward gate
uses forced-answer phase timing when available and falls back to uncached total
time for older reports. When readiness manifests include an
`inside_sampling_profile_report` or `inside_trigger_budget_sweep_report`
artifact, the comparison can also gate on promoted INSIDE sample-count and
generation-time ratios before a runtime is eligible for release. Profile
reports use `*_to_baseline` ratios; trigger-budget sweeps use
`*_to_reference` ratios when baseline ratios are absent, and the report records
which source each gate used.

Add `--prefix-kv-cache` to run the experimental shared-prefix eval path inside
the same triplet/matrix/readiness gates. To compare it against the default path
in one matrix, use `--prefix-kv-cache-modes off,on`; the generated cell ids and
shared eval cache groups include the prefix mode so the two forward paths do not
silently share eval-reps caches. Because prefix caching changes only the
uncached/cached model-forward path, cache-only runs omit the flag and replay the
saved representations. For prefix on/off comparisons, use the report's
`prefix_kv_comparisons` and `forced_answer_forward_seconds` fields; cache-only
speedup is a cache-replay metric and is not evidence that prefix KV forward is
faster.
When `--shared-cache-dir` is set, cells with the same layer and hidden-state
capture share statement/layer/eval cache paths. The first cell for each group
refreshes the shared caches; later batch-size cells use a warm-start uncached
run that reuses statement/layer caches but still omits the eval-reps cache, so
forced-answer forward timing remains visible while repeated warmup/tokenization
cost is reduced.

For faster report/calibration iteration on an already shared layer/capture
group, add `--matrix-mode rescore`. In that mode the first cell in each shared
cache group runs the full uncached/cached/cache-only triplet and later cells in
the same group run only `cache_only` against the shared eval-reps cache:

```bash
python benchmarks/run_cache_profile_matrix.py \
  --output-dir /tmp/eigentruth-qwen05-profile-rescore \
  --shared-cache-dir /tmp/eigentruth-qwen05-profile-cache \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --real-truthfulqa \
  --limit 24 \
  --manifold-questions 12 \
  --layers=-12 \
  --batch-sizes=1,2,4 \
  --hidden-state-captures=outputs \
  --matrix-mode rescore \
  --dry-run
```

Use `rescore` only when the question is post-processing/report behavior over
the same cached representations. Later cache-only cells are compared against
the first cell's uncached baseline for matrix gating, so they can be promoted
when cache-only scoring is faster. They deliberately skip repeated model
forward timing, so use normal `triplet` mode for end-to-end batch-size runtime
claims.

Both cache-profile runners write an `artifact-manifest.json` next to their
outputs. The manifest records SHA-256 fingerprints for command logs, profile
JSON, result JSON, comparison reports, and cache paths. Directory fingerprints
are deterministic over relative file names, sizes, and content hashes. This is
intended for local reproducibility and artifact registry handoff; on very large
eval-reps cache directories it adds one linear content-hash pass after the run.
Directory cache signatures and full content hashes reuse one file-list scan, so
cache misses do not enumerate the same directory twice.

Use `verify_artifact_manifest.py` to validate that local artifacts still match
the saved fingerprints. Add `--recursive` for matrix reports so each cell's
triplet manifest is verified as well. Recursive verification shares a
run-local fingerprint cache, so repeated references to the same large score
dump or cache artifact avoid duplicate content reads within that verification
run. Add `--fingerprint-cache` to persist that cache across repeated local
checks; entries are keyed by path and file/directory signatures, so changed
artifacts naturally miss and get re-read:

```bash
python benchmarks/verify_artifact_manifest.py \
  --manifest /tmp/eigentruth-qwen05-profile-rescore/artifact-manifest.json \
  --recursive \
  --max-workers 4 \
  --fingerprint-cache /tmp/eigentruth-qwen05-profile-rescore/fingerprints.json \
  --json /tmp/eigentruth-qwen05-profile-rescore/manifest-verification.json
```

`--max-workers` enables bounded parallel fingerprinting for independent
artifacts during direct manifest verification. Keep the default `1` for strictly
serial timing comparisons; use the worker-sweep report to choose a local value
for large release manifests.

Once verification passes, `promote_artifact_manifest.py` can register the
manifest and verification report in a local `ArtifactRegistry` JSON file:

```bash
python benchmarks/promote_artifact_manifest.py \
  --manifest /tmp/eigentruth-qwen05-profile-rescore/artifact-manifest.json \
  --registry artifacts/registry.json \
  --name qwen05-profile-rescore \
  --version 0.3 \
  --fingerprint-cache /tmp/eigentruth-qwen05-profile-rescore/fingerprints.json \
  --verification-report /tmp/eigentruth-qwen05-profile-rescore/manifest-verification.json
```

After promotion, `compare_registry_baseline.py` can use the registered manifest
as a fail-closed performance baseline. It verifies the manifest before reading
the baseline profile artifact, then delegates the regression gate to
`compare_profiles.py`:

```bash
python benchmarks/compare_registry_baseline.py \
  --registry artifacts/registry.json \
  --baseline-name qwen05-profile-rescore \
  --baseline-version 0.3 \
  --baseline-profile-artifact profiles.uncached \
  --candidate-profile candidate=/tmp/eigentruth-current-profile.json \
  --max-total-ratio 1.10 \
  --json artifacts/qwen05_registry_profile_gate.json \
  --fail-on-regression
```

For matrix manifests, profile payloads live inside each cell's triplet manifest.
Use `root_artifact::nested_artifact` syntax to resolve the nested profile:

```bash
python benchmarks/compare_registry_baseline.py \
  --registry artifacts/registry.json \
  --baseline-name qwen05-profile-rescore \
  --baseline-version 0.3 \
  --baseline-profile-artifact \
    cells.layer_m12_batch_1_capture_outputs.triplet_manifest::profiles.uncached \
  --candidate-profile candidate=/tmp/eigentruth-current-profile.json \
  --max-total-ratio 1.10 \
  --fail-on-regression
```

`run_registry_baseline_workflow.py` combines the matrix run, recursive manifest
verification, promotion, and optional registry-backed comparison in one command.
Run it with `--dry-run` first to inspect the generated commands and registry
records without loading a model. When candidate profiles are provided, the
workflow default `--baseline-profile-artifact auto` uses the first matrix cell
that includes an uncached profile:

```bash
python benchmarks/run_registry_baseline_workflow.py \
  --output-dir /tmp/eigentruth-qwen05-baseline-workflow \
  --registry artifacts/registry.json \
  --name qwen05-profile-rescore \
  --version 0.3 \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --real-truthfulqa \
  --limit 24 \
  --manifold-questions 12 \
  --layers=-12 \
  --batch-sizes=1,2,4 \
  --hidden-state-captures=outputs \
  --shared-cache-dir /tmp/eigentruth-qwen05-profile-cache \
  --matrix-mode rescore \
  --dry-run \
  --json artifacts/qwen05_registry_baseline_workflow.json
```

`make perf-check` runs `benchmarks/profile_gate_smoke.py`,
`benchmarks/cache_profile_smoke.py`,
`benchmarks/score_fusion_profile_smoke.py`,
`benchmarks/inside_sampling_profile_smoke.py`,
`benchmarks/cache_worker_sweep_smoke.py`, and
`benchmarks/registry_baseline_smoke.py`, plus
`benchmarks/performance_baseline_smoke.py`,
`benchmarks/product_trace_replay_smoke.py`, and
`benchmarks/release_candidate_registry_smoke.py`. These use fixed synthetic profile
payloads to verify that direct gates, cache-profile gates, worker-count sweep
decisions, INSIDE sampling sample-efficiency gates, and registry-backed
baseline/ProductTrace replay/release gates pass acceptable candidates, reject bounded
telemetry payloads where full traces are required, and catch expected regressions. They
are stable enough for default local/CI checks because they do not load a model
or measure machine speed. Use real `eval_truthfulqa.py --profile-json`
artifacts, `run_cache_profile_triplet.py`, or `run_inside_sampling_profile.py`
before making actual runtime or sampling-cost claims.

## `compare_transfer.py`

Compares saved layer/score sweep reports across runs without loading a model. Use
it to test whether a detector is stable across sample sizes, model families, or
candidate layer bands before treating a calibration as broadly transferable.

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

The report includes each run's selected-layer AUROCs, mean/best AUROC over the
requested layer set, and simple counts for layers above 0.6 and 0.7 AUROC. This
is evidence for detector stability, not a license to reuse thresholds: conformal
thresholds still need per-model/domain calibration unless a separate calibration
transfer study proves otherwise.

If Hugging Face downloads stall in the Xet path for small transfer models, retry
with `HF_HUB_DISABLE_XET=1` in the environment before the Python command.

## 说明

`eval_truthfulqa.py` 在 TruthfulQA 上以确定性、无需 LLM 裁判、单次前向的方式，检验隐状态
几何能否分离真/假陈述。真值流形仅用留出题目的正确答案构建（与评测题目不重叠，无泄漏），
正类为错误答案（幻觉）。逐信号报告 AUROC（0.5 为随机，1.0 为完美分离）。

三个关键对比：`maha_last` vs `nll_answer`（几何是否优于困惑度基线）、`disp_hse` vs
`disp_euclid`（双曲投影是否真的有用）、`eigenscore` 是否补充内部状态谱分散度信号。
结果强依赖目标层，应扫层。陈述级打分是开放生成幻觉的代理，小模型 + 数百样本的 AUROC
置信区间较宽，结论需对照完整已发表方法。

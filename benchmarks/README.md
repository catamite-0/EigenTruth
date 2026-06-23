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
   `inside_eigenscore`, `inside_semantic_entropy`, and `inside_embedding_entropy`,
   closer multi-response proxies that sample verifier-style continuations,
   compute EigenScore over their sentence embeddings, and compute dependency-free
   lexical and embedding-cluster entropy over the sampled responses.

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
Use `--layer-stats-cache path.pt` to load an existing warmup manifold/subspace
bundle or create one when missing. The cache is validated against model, dtype,
layer list, max length, subspace rank, warmup mode, and warmup text fingerprint;
use `--refresh-layer-stats-cache` to rebuild it intentionally.
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
loaded batch-by-batch, reuse the active shard across adjacent reads, and remain
compatible with `--cache-only`; old single-file `.pt` caches remain the default
and continue to load normally.
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
`inside_embedding_entropy=0.0`; read them as two-stage policy scores, not as full
INSIDE-only AUROC. The JSON output includes `inside_sampling` counts.
Use `--inside-diagnostics-cache path.json` for repeated triggered INSIDE runs.
The cache stores sampled INSIDE diagnostics keyed by statement, model, layer,
sampling settings, and seed, but not by trigger threshold/top fraction. This lets
nested budgets reuse diagnostics for statements already sampled by an earlier
budget while preserving a cache miss for changed sampling settings. Use
`--refresh-inside-diagnostics-cache` to rebuild it intentionally.

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
  --inside-samples 3 --dump-scores artifacts/tiny_scores_with_samples.json \
  --dump-inside-samples

python benchmarks/build_selfcheck_fixture.py \
  --scores artifacts/tiny_scores_with_samples.json \
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
- Compare `maha_last` against `nll_answer`: geometry is only interesting if it adds
  signal over plain perplexity.
- Compare `disp_hse` against `disp_euclid`: this is the decisive ablation for the
  hyperbolic component.
- Treat `eigenscore` as an internal-state spectral-diversity proxy. Use
  `inside_eigenscore`, `inside_semantic_entropy`, and
  `inside_embedding_entropy` when `--inside-samples` is enabled to test a closer
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
  `inside_semantic_entropy`, and `inside_embedding_entropy` are closer because
  they sample multiple continuations, but they are still verifier-prompted
  benchmark proxies rather than full published reproductions.
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

# Override the score direction for lower-is-more-anomalous signals:
python benchmarks/eval_conformal.py --scores benchmarks/scores.json --signal support_score \
  --direction lower

# Save a reusable CalibrationArtifact for one selected signal:
python benchmarks/eval_conformal.py --scores benchmarks/scores.json --signal truth_proj \
  --artifact-alpha 0.2 --save-calibration artifacts/gpt2-l8-truth-proj.json

# Build the 0.2 calibrated-observability closure: layer/score sweep + best artifact:
python benchmarks/eval_conformal.py --scores benchmarks/scores.json \
  --signals maha_last,truth_proj,subspace_resid,eigenscore,inside_eigenscore,inside_semantic_entropy,inside_embedding_entropy \
  --artifact-alpha 0.2 \
  --save-sweep-report artifacts/gpt2-sweep-report.json \
  --save-best-calibration artifacts/gpt2-best-calibration.json
```

**E1 result (gpt2, layer −8):** empirical false-alarm rate tracks the nominal α within
1.3% at α ∈ {0.05, 0.1, 0.2} for both signals — the guarantee holds in practice. At the
same α = 0.2 false-alarm budget, `truth_proj` detects **46.9%** of false statements vs
34.1% for `maha_last` (committed as `results_conformal_*.json`). The low-level
calibration functions live in `eigentruth.eval.conformal` (`conformal_pvalues`,
`conformal_threshold`, `directional_conformal_threshold`, `directional_trigger_rate`).
Reusable single-signal artifacts are built with `eigentruth.calibration.ConformalCalibrator`;
layer/score reports and best artifacts are built with
`eigentruth.calibration.LayerScoreSweepCalibrator`. Structured reports also include
`selective_report` fields for threshold, coverage, selective accuracy, detection,
false alarm, and simple binomial confidence intervals; thresholding honors each
score's `higher` or `lower` anomalous direction while score dumps remain unchanged.

Caveat: the guarantee is conditional on exchangeability — under distribution shift
(different domain than the calibration set) coverage can degrade; recalibrate per domain.

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
  --json artifacts/verifier_ensemble_report.json
```

Add `--compact-json` for large automated runs when the report is consumed by
tools and does not need human-readable indentation.

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
threshold used for the verifier gate.

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

For structured state, business rules, policy checks, or tool-output checks,
provide explicit `state_check` metadata in the claim fixture and pass a local
state JSON file. `StructuredStateVerifier` checks these deterministic rules
after structured QA and before lexical retrieval:

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
  --signal truth_proj \
  --json artifacts/order_transition_verifier_ensemble_report.json
```

This fixture checks action-consequence verification: true labels match the
predicted inventory after reservation, while false labels assert an off-by-one
postcondition that the predicted state refutes.

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
`self_consistency`, `retrieval_groundedness`, or `staged_skip`) and records attempted-route
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
  no eligible routes fail the gate. When multiple reports are
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
  --json artifacts/route-baseline-comparison.json \
  --fail-on-blocked
```

The comparison recursively verifies each registered manifest by default, reloads
the saved `route_comparison_report`, fails closed on non-promoted route
decisions or `invalid_metric_counts`, and recommends the passing baseline with
the best quality/cost ordering. Optional runtime-budget flags read
`runtime_total_seconds`, `runtime_n_retrieval_hits`, claims-cache metadata, and
verifier-trace-cache metadata from the route manifest or registry record; when a
threshold is configured, missing or non-finite evidence blocks that baseline.

## `run_adapter_family_matrix.py`

Builds deterministic local fixtures for the front-line adapter families and
runs each through the same refresh/promotion gate:

- `structured_qa`: exact question/answer facts.
- `structured_state`: static business/domain state checks.
- `state_transition`: action-conditioned world-model postconditions.
- `retrieval_groundedness`: optional local retrieval evidence plus lexical
  support/refutation checks.

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

## `run_adapter_readiness_workflow.py`

Combines the deterministic adapter-family quality matrix with the same-machine
cache-profile performance matrix. The final `readiness_decision` is `promote`
only when both the adapter-family `promotion_decision` and the performance
`matrix_decision` promote and the performance report can produce deployable
runtime settings.

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
set:

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
  --runtime-profile balanced \
  --min-best-quality-auroc 0.60 \
  --max-uncached-forward-seconds 40 \
  --min-selected 100 \
  --min-decision-accuracy 0.95 \
  --max-false-supported-rate 0.02 \
  --min-false-refuted-rate 0.90 \
  --max-p99-duration-seconds 0.20 \
  --max-runtime-total-seconds 60 \
  --max-retrieval-hit-count 1000 \
  --min-claims-cache-hit-rate 0.9 \
  --min-verifier-trace-cache-hit-rate 0.9 \
  --json artifacts/release-candidate-comparison.json \
  --fail-on-blocked
```

Use explicit `--readiness-baseline-key` and `--route-baseline-key` values when a
release should be constrained to named registry records. Omit `--route-registry`
when readiness and route manifests are stored in the same local registry file.
Release-candidate runtime-budget flags are delegated to the route-baseline
comparison, so the final release blocks when the selected route baseline exceeds
the configured total runtime, retrieval-hit, or cache-reuse budgets.
Readiness-side INSIDE sampling gates are delegated to
`compare_readiness_baselines.py`, so the final release also blocks when the
selected runtime lacks sampling profile evidence or exceeds the configured
sample-count/generation-time ratios.
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
`run_release_candidate_registry_workflow.py`:

```bash
python benchmarks/run_release_candidate_registry_workflow.py \
  --readiness-registry artifacts/registry.json \
  --route-registry artifacts/registry.json \
  --release-registry artifacts/release-registry.json \
  --name qwen05-local-release-candidate \
  --version 0.7 \
  --runtime-profile balanced \
  --min-best-quality-auroc 0.60 \
  --max-uncached-forward-seconds 40 \
  --min-selected 100 \
  --min-decision-accuracy 0.95 \
  --max-false-supported-rate 0.02 \
  --min-false-refuted-rate 0.90 \
  --max-p99-duration-seconds 0.20 \
  --json artifacts/release-candidate-registry-workflow.json \
  --release-report-json artifacts/release-candidate-comparison.json \
  --artifact-manifest artifacts/release-candidate-artifact-manifest.json \
  --fail-on-blocked
```

The generated manifest fingerprints the release-candidate report and the
selected readiness and route manifests. Recursive verification therefore checks
the final candidate and both underlying baseline manifests before the release
candidate is registered. When `--runtime-profile` is used, the selected profile
and the defaults it filled are written into the release report, manifest
metadata, and registry record.

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

The current derived trigger-budget release records
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

## `build_evidence_fixture.py`

Builds a non-oracle claim/evidence fixture from a statement-bearing score dump
and local evidence corpus files. It supports JSON, JSONL, and plain text corpora,
uses dependency-free token-overlap retrieval with optional SQLite FTS candidate
indexing, and copies labels only into audit metadata; retrieval is driven by
claim text.

```bash
python benchmarks/build_evidence_fixture.py \
  --scores artifacts/qwen05_truthfulqa_l80_scores_with_statements.json \
  --corpus artifacts/truthfulqa_l80_correct_answer_corpus.json \
  --output artifacts/truthfulqa_l80_local_evidence_claims.json \
  --query-field answer \
  --retriever-backend auto \
  --retriever-index-path artifacts/cache/local_retrieval_fts/truthfulqa_l80.sqlite \
  --retriever-min-overlap 0.95 \
  --retrieval-limit 3

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
  --claims-cache-dir artifacts/cache/local_retrieval_claims \
  --verifier-trace-cache-dir artifacts/cache/verifier_traces \
  --min-selected 100 \
  --min-decision-accuracy 0.90 \
  --max-false-supported-rate 0.05 \
  --max-mean-attempted-route-count 2.1 \
  --max-retrieval-use-rate 1.0 \
  --fail-on-blocked
```

Use this when the local corpus baseline should enter `compare_route_baselines.py`
and release-candidate gates. Unlike `build_evidence_fixture.py` alone, this
workflow records the full provenance chain needed for recursive manifest
verification. The workflow report also includes a lightweight `profile` block
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

`--claims-cache-dir` is optional. When set, the workflow caches generated
claims fixtures by score-dump fingerprint, corpus fingerprints, query field,
retriever backend, retriever overlap threshold, and retrieval limit. A cache hit
skips local score dump/corpus parsing for claim construction, then still reruns
verifier-route metrics and promotion against the current score dump and emitted
claims file.

`--verifier-trace-cache-dir` is optional. It caches verifier ensemble
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

Current l80 local-corpus baseline with `--query-field answer`,
`--retriever-min-overlap 0.95`, and `--retrieval-limit 3`:

| Run | Verified false alarm | Verified detection | true supported | false supported | decision accuracy |
|---|---:|---:|---:|---:|---:|
| Qwen l80 | 0.008 | 0.274 | 0.908 | 0.042 | 0.946 |
| SmolLM2 l80 | 0.008 | 0.219 | 0.908 | 0.042 | 0.946 |

Interpretation: this conservative lexical corpus strongly suppresses false
alarms by supporting most true claims, but it rarely refutes false claims. It is
a reproducible non-oracle baseline, not a replacement for stronger retrieval,
database, calculator, or domain-world-model evidence.

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
  --signals truth_proj,maha_last,subspace_resid,eigenscore \
  --methods max_rank,mean_rank \
  --repeats 50 \
  --json artifacts/truthfulqa_score_ensemble_report.json
```

Each selected signal is converted to a direction-aware anomaly percentile using
the split calibration true set. `max_rank` takes the most anomalous normalized
signal per item; `mean_rank` averages normalized anomaly ranks. The ensemble is
then thresholded with the same split-conformal false-alarm check as the single
signals.

Current Qwen l80 / SmolLM2 l80 result: simple internal-score ensembles do not
beat `truth_proj`. At alpha 0.100, Qwen's best single signal detects 0.279 while
the best ensemble detects 0.235; SmolLM2's best single detects 0.229 while the
best ensemble detects 0.196. Treat this as a negative result for naive score
fusion, not as evidence against richer verifier/retrieval ensembles.

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
`false_alarm <= conformal_alpha + tolerance`.

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
deltas, grouped time deltas, and throughput ratios. Older profile payloads that
only contain `total_seconds` and `phases` remain readable, but grouped deltas are
available only when the newer `summary` field exists.

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
  --output /tmp/eigentruth-qwen05-worker-sweep/runtime-recommendation.json \
  --fail-on-blocked
```

The recommendation records the selected layer, batch size, hidden-state capture
mode, padded-token budget, prefix-KV mode, worker count, all finite AUROC
quality signals from the promoted cell, optional promoted INSIDE sampling
settings, and the best quality signal. With `--inside-sampling-report`, the
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
The report includes equivalent flags for `eval_truthfulqa.py`,
`run_cache_profile_matrix.py`, `run_adapter_readiness_workflow.py`, and, when
sampling evidence is provided, `run_inside_sampling_profile.py` and
`run_inside_trigger_budget_sweep.py`. Treat it as the deployment handoff from
same-machine performance evidence; it does not replace a promoted matrix,
worker-sweep, sampling-profile, or trigger-budget-sweep decision.

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
eval-reps cache directories it adds one linear read pass after the run.

Use `verify_artifact_manifest.py` to validate that local artifacts still match
the saved fingerprints. Add `--recursive` for matrix reports so each cell's
triplet manifest is verified as well:

```bash
python benchmarks/verify_artifact_manifest.py \
  --manifest /tmp/eigentruth-qwen05-profile-rescore/artifact-manifest.json \
  --recursive \
  --json /tmp/eigentruth-qwen05-profile-rescore/manifest-verification.json
```

Once verification passes, `promote_artifact_manifest.py` can register the
manifest and verification report in a local `ArtifactRegistry` JSON file:

```bash
python benchmarks/promote_artifact_manifest.py \
  --manifest /tmp/eigentruth-qwen05-profile-rescore/artifact-manifest.json \
  --registry artifacts/registry.json \
  --name qwen05-profile-rescore \
  --version 0.3 \
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
`benchmarks/inside_sampling_profile_smoke.py`,
`benchmarks/cache_worker_sweep_smoke.py`, and
`benchmarks/registry_baseline_smoke.py`. These use fixed synthetic profile
payloads to verify that direct gates, cache-profile gates, worker-count sweep
decisions, INSIDE sampling sample-efficiency gates, and registry-backed
baseline gates pass acceptable candidates and catch expected regressions. They
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

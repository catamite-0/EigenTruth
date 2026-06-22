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
   `inside_eigenscore`, a closer multi-response INSIDE proxy that samples
   verifier-style continuations and computes EigenScore over their sentence
   embeddings.

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
`inside_eigenscore=0.0`; read it as a two-stage policy score, not as a full
INSIDE-only AUROC. The JSON output includes `inside_sampling` counts.

Use `--profile` to include phase timings in stdout and `--json` output, or
`--profile-json profile.json` to write only the timing payload. This is the
recommended way to compare batch-size, layer-sweep, and INSIDE sampling changes
before treating a benchmark run as faster. The profile payload includes raw
`phases` plus a `summary` with the bottleneck phase, top phases, grouped time
shares for startup/tokenization/model-forward/cache/postprocess work, and
throughput fields for warmup and forced-answer eval records when counts are
available.
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
  `inside_eigenscore` when `--inside-samples` is enabled to test a closer
  multi-response INSIDE path. Calibrate both like other higher-is-more-anomalous
  scores before using them for routing.
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
  sample-based semantic uncertainty and INSIDE/EigenScore. `inside_eigenscore` is
  closer to INSIDE because it samples multiple continuations, but it is still a
  verifier-prompted benchmark proxy rather than a full published reproduction.
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
  --signals maha_last,truth_proj,subspace_resid,eigenscore,inside_eigenscore \
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

The state source may be a raw JSON object used as state, or an object with
`state`, optional `state_checks`, and optional `state_transitions` fields. A
claim fixture record can provide `state_check` directly or under
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
retriever are dependency-free lexical baselines (`GroundednessVerifier` and
`InMemoryRetriever`), so results are only a controlled adapter test until a real
retrieval/verifier backend is plugged in.

Reports include `verification_quality`, a label-conditioned matrix over
`supported` / `refuted` / `insufficient_evidence` outcomes. Use
`true_supported_rate`, `false_refuted_rate`, `decision_accuracy`, and
`decision_error_rate` to evaluate evidence fixture quality separately from the
final control-policy detection and false-alarm rates. Reports also include
`route_summary`, which breaks verification outcomes down by selected route
(`structured_qa`, `state_transition`, `structured_state`, `groundedness`, or
`retrieval_groundedness`) and records attempted-route counts, status counts, and
per-route supported/refuted/error rates. Use `route_quality` for label-conditioned
false-support / false-refutation metrics per selected route, and use each
alpha result's `route_control_impact` to see how that route changed internal
false alarm, detection, suppression, and rescued-detection rates. New reports
also include route-level `p95_duration_seconds` and `p99_duration_seconds`
tail-latency fields for promotion gates.

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
  `mean_attempted_route_count`, and `retrieval_use_rate`.
- `cache_summary`: aggregate report-level cache hit/miss/request totals across
  compared runs. This is reported separately from route metrics because cache
  hits are global to the benchmark run rather than safely attributable to one
  selected route.
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
  `cache_hit_rate`; missing routes, missing metrics, non-finite values, missing
  cache evidence, or no eligible routes fail the gate.
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
  --fail-on-blocked
```

The final report includes:

- `route_comparison_path` and embedded `route_comparison` for route-quality
  audit.
- `registry_baseline_comparison` when candidate profiles are provided.
- `decision`: final workflow status. It is `promote` only when route promotion
  passes and every configured registry baseline gate passes. Missing route
  gates, failed route gates, missing registry gates, or failed registry gates
  produce `blocked` plus explicit `blocking_reasons`.

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
uses dependency-free token-overlap retrieval, and copies labels only into audit
metadata; retrieval is driven by claim text.

```bash
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
  --json artifacts/truthfulqa_l80_local_evidence_verifier_ensemble_report.json
```

Use this before wiring a real search/RAG backend: it gives the same downstream
fixture schema and `verification_quality` fields while keeping evidence source
and retrieval behavior fully reproducible.

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
  --hidden-state-captures=outputs,hooks \
  --dry-run
```

Remove `--dry-run` only when the full matrix cost is acceptable. The matrix
report includes each triplet's command log, gate summary, cache-only timing,
per-run bottleneck phase, and `truth_proj` AUROC when result JSON is available.
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
the same cached representations. It deliberately skips repeated forward timing,
so it should not be used to compare batch-size runtime performance.

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
`benchmarks/cache_profile_smoke.py`, and
`benchmarks/registry_baseline_smoke.py`. These use fixed synthetic profile
payloads to verify that direct gates, cache-profile gates, and registry-backed
baseline gates pass acceptable candidates and catch expected regressions. They
are stable enough for default local/CI checks because they do not load a model
or measure machine speed. Use real `eval_truthfulqa.py --profile-json`
artifacts, or `run_cache_profile_triplet.py`, before making actual runtime
claims.

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

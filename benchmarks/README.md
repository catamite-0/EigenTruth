# EigenTruth Benchmarks

Reproducible evaluation scripts that turn EigenTruth's diagnostics into measurable
numbers. Unlike the qualitative scripts in [`examples/`](../examples/README.md),
these produce **AUROC** against labeled data so the core hypotheses can be tested
and ablated.

可复现评测脚本，把 EigenTruth 的诊断信号变成可度量的数字（AUROC），用于检验和消融核心假设。

## `eval_counterfactual_verification.py`

Audits whether a local verifier changes decision on paired counterfactual claim
perturbations. This is a verifier robustness check, not an open-domain factuality
claim: a passing report only means the supplied verifier handled the supplied
entity/time/quantity/logical probes.

```bash
python benchmarks/eval_counterfactual_verification.py \
  --records artifacts/counterfactual-probes.json \
  --verifier structured_fact \
  --fact-corpus artifacts/structured-fact-corpus.json \
  --json artifacts/counterfactual-verifier-audit.json \
  --artifact-manifest artifacts/counterfactual-verifier-audit.manifest.json \
  --registry artifacts/local-artifact-registry.json \
  --register-name counterfactual-verifier-audit \
  --register-version 0.1
```

For small local fixtures, `--verifier in_memory` can derive exact-match statuses
from each probe's expected status fields, or consume `--in-memory-facts`.
For covered-fact route evidence, `--verifier structured_qa` can replay
supported/refuted verified-record pairs against a structured QA corpus and turn
same-question answer mismatches into counterfactual probes:

```bash
python benchmarks/eval_counterfactual_verification.py \
  --verified-records artifacts/truthfulqa-frontier-smollm2-l80-source-family-structured-qa-fact-collection-route/verified-records.jsonl \
  --verifier structured_qa \
  --fact-corpus artifacts/truthfulqa-frontier-smollm2-l80-source-family-structured-qa-fact-collection-workflow/source-family-structured-qa-corpus.json \
  --json artifacts/smollm2_product_counterfactual_structured_qa_audit_v0/counterfactual-verification-report.json \
  --artifact-manifest artifacts/smollm2_product_counterfactual_structured_qa_audit_v0/artifact-manifest.json \
  --registry artifacts/local-release-registry.json \
  --register-name smollm2-product-counterfactual-structured-qa-audit \
  --register-version 0.1
```

Some checkouts do not include the historical source-family route artifacts. The
current frontier checkout can reproduce the same gate shape from the blind-spot
Wikidata structured-QA route:

```bash
python benchmarks/eval_counterfactual_verification.py \
  --verified-records artifacts/truthfulqa-frontier-smollm2-l80-blind-spot-wikidata-structured-qa-route/verified-records.jsonl \
  --verifier structured_qa \
  --fact-corpus artifacts/truthfulqa-frontier-smollm2-l80-blind-spot-wikidata-structured-qa-route/wikidata-blind-spot-qa-corpus.json \
  --json artifacts/smollm2_product_counterfactual_blind_spot_wikidata_structured_qa_audit_v1/counterfactual-verification-report.json \
  --artifact-manifest artifacts/smollm2_product_counterfactual_blind_spot_wikidata_structured_qa_audit_v1/artifact-manifest.json \
  --registry artifacts/local-release-registry.json \
  --register-name smollm2-product-counterfactual-blind-spot-wikidata-structured-qa-audit \
  --register-version 0.1 \
  --compact-json
```

When only extracted claims are available, the benchmark can generate bounded
counterfactual probes from claim metadata, entity replacements, numbers, years,
and negation:

```bash
python benchmarks/eval_counterfactual_verification.py \
  --claims artifacts/extracted-claims.jsonl \
  --generated-probe-types entity_swap,quantity,year,negation \
  --max-generated-probes-per-claim 2 \
  --verifier in_memory \
  --json artifacts/generated-counterfactual-verifier-audit.json
```

Generated probes are meant for fast verifier-sensitivity audits and should be
reviewed or replaced with hand-labeled probes before they are treated as strong
open-domain evidence.
The resulting report can be used as a release gate through
`compare_release_candidates.py --counterfactual-verification-report` or the
same flag on `run_release_candidate_registry_workflow.py`; registry-backed runs
can pass `--counterfactual-verification-registry` and
`--counterfactual-verification-key`.

## `build_counterfactual_probe_handoff.py`

Compiles `counterfactual_probe` rows from a blind-spot collection corpus,
unresolved queue report, or adapter-request JSONL into three non-evidence
handoff files: `counterfactual-claims.jsonl`, generated
`counterfactual-probe-records.jsonl`, and
`pending-counterfactual-probe-requests.jsonl` for rows that need an external or
human generator. The generated probe records are directly consumable by
`eval_counterfactual_verification.py --records`, but they remain audit fixtures
until a verifier report and release gate pass.

```bash
OUT=artifacts/truthfulqa-frontier-smollm2-l80-counterfactual-probe-handoff

python benchmarks/build_counterfactual_probe_handoff.py \
  --collection-corpus artifacts/truthfulqa-frontier-smollm2-l80-blind-spot-evidence-collection-corpus/blind-spot-evidence-collection-corpus.json \
  --output-dir "$OUT" \
  --registry artifacts/local-release-registry.json \
  --name truthfulqa-frontier-smollm2-l80-counterfactual-probe-handoff \
  --version 0.1
```

On the current SmolLM2 L80 collection corpus, a local smoke run sees `29`
counterfactual requests and auto-generates `27` probe records
(`21` entity swaps, `3` quantity probes, and `3` negation probes), leaving `2`
requests in the pending-generation sidecar. The command is intentionally a
handoff bridge: it does not verify the probes or claim route-quality evidence.
It may use a model answer to form the claim text being probed, but written
claim/probe metadata and pending rows strip reserved label fields,
`model_answer` / `answer`, and record-index style adapter-linkage fields.

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
   `resid_update_profile_area`, `resid_update_profile_peak`,
   `resid_update_profile_late_mass`, and
   `resid_update_profile_concentration` summarize the whole inspected layer
   curve from the same per-layer updates. Treat them as exploratory profile
   signals until they pass the same conformal sweep, stability, and release-gate
   checks as other diagnostics.
5. **Do prompt/question-anchored and answer-anchored truthfulness pathways diverge?**
   `prompt_answer_distance`, `prompt_answer_cosine_gap`, `answer_anchor_distance`,
   `answer_path_length`, and `pathway_disagreement` are training-free two-pathway
   proxies computed from the same forced-answer hidden states. They summarize
   how far answer states move from the prompt anchor, how much the answer path
   moves internally, and whether those two views disagree. Treat them as
   exploratory pathway diagnostics until replicated on larger calibrated runs.
   Add `--attention-pathway --attn-implementation eager` to request model
   attentions and emit optional `attn_prompt_flow_loss`, `attn_answer_self_flow`,
   `attn_pathway_gap`, and `attn_pathway_concentration` columns. This is a
   pathway-flow readout, not a causal attention knockout; unsupported attention
   backends fail closed.

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

# Optional attention pathway readout (requires a backend that returns attentions):
python benchmarks/eval_truthfulqa.py --model Qwen/Qwen2.5-0.5B-Instruct \
  --layer -8 --limit 200 --attention-pathway --attn-implementation eager \
  --dump-scores artifacts/qwen05-attn-pathway-scores.json

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

Use `audit_layer_band_replication.py` before turning a layer-band selector into
a default benchmark preset. It consumes saved layer-band selector reports and
fails closed unless the selected strategy has enough matched runs, enough model
families, dense enough ranked-layer grids, high best-layer hit rate, bounded
AUROC regret, and a bounded candidate-layer fraction:

```bash
python benchmarks/audit_layer_band_replication.py \
  --layer-band-report l80=artifacts/truthfulqa-frontier-layer-band-selection/layer-band-comparison.json \
  --min-runs 2 \
  --min-model-families 2 \
  --min-ranked-layers 8 \
  --json artifacts/truthfulqa-frontier-layer-band-replication/layer-band-replication-audit.json \
  --artifact-manifest artifacts/truthfulqa-frontier-layer-band-replication/artifact-manifest.json \
  --verification-report artifacts/truthfulqa-frontier-layer-band-replication/manifest-verification.json
```

The current l80 selector artifact is expected to block under these defaults
because each run only ranks 5 monitored layers. That is the intended state:
the selector is usable as a local sweep prior, but denser grids and additional
replication evidence are still required before making it a default preset.

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

Before promoting direct selfcheck signals, use
`plan_selfcheck_sample_collection.py` to fail closed on insufficient aligned
samples and get a machine-readable rerun plan. The direct workflow below writes
the same per-run plan artifacts by default; the standalone command is useful
when planning generation before running score fusion:

```bash
python benchmarks/plan_selfcheck_sample_collection.py \
  --scores artifacts/smollm2_l20_inside_trigger_budget_sweep_derived/top_0p4/scores-adaptive_selfcheck.json \
  --samples artifacts/smollm2-l20-direct-selfcheck-signal-fusion/inside-diagnostics-samples.json \
  --output artifacts/smollm2-l20-direct-selfcheck-signal-fusion/sample-collection-plan.json \
  --min-samples 2 \
  --target-samples-per-record 3
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
- `resid_update_profile_* > 0.5` means a cross-layer residual-update profile
  summary ranks false statements above true ones. These are exploratory
  ICR-style curve summaries, not a replacement for calibrated layer/score sweep.
- `prompt_answer_*`, `answer_*`, and `pathway_disagreement > 0.5` mean
  training-free prompt/answer pathway summaries rank false statements above true
  ones. They are mechanism-inspired diagnostics, not a full attention-knockout or
  token-patching replication.
- `attn_* > 0.5` means optional attention-flow pathway summaries rank false
  statements above true ones. These require `--attention-pathway`; if the model
  backend does not return attentions, the benchmark fails instead of writing a
  misleading all-zero attention signal.
- Compare `maha_last` against `nll_answer` and `first_token_entropy`: geometry is
  only interesting if it adds signal over plain perplexity or cheap single-decode
  uncertainty.
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
- `first_token_entropy` is a **single-decode uncertainty baseline** computed from
  top-k logits at the first available answer-token prediction. It is included so
  expensive sampling routes have a low-cost baseline, not because it is already
  promoted as a product gate.
- A small model (e.g. 0.5B) and a few hundred items give wide confidence intervals.
  Treat AUROC values as indicative, not conclusive, and report `n`.
- Beating these in-house baselines is necessary but not sufficient; a real claim needs
  comparison against full published detectors (semantic entropy, multi-response
  INSIDE/EigenScore, SAPLMA) on standard splits.

## `eval_pre_generation_probe.py`

Trains and evaluates the torch-only pre-generation soft-target attention probe from
local token-level hidden-state records. This is the benchmark-side handoff for
frontier soft-target attention probing: it consumes saved prompt hidden states and
empirical error-rate targets, then writes train/test risk metrics and optionally saves
an `AttentionSoftTargetProbeArtifact`.

```bash
python benchmarks/eval_pre_generation_probe.py \
  --records artifacts/pre-generation-probe-records.jsonl \
  --json artifacts/pre-generation-probe-report.json \
  --save-artifact artifacts/pre-generation-probe.pt \
  --save-calibration artifacts/pre-generation-probe-calibration.json \
  --conformal-alpha 0.2 \
  --soft-target-cutoff 0.5 \
  --record-layer -8 \
  --layer-idx -8
```

For records that contain several layers, run a local layer sweep and save the
recommended layer artifacts:

```bash
python benchmarks/eval_pre_generation_probe.py \
  --records artifacts/pre-generation-probe-records.jsonl \
  --json artifacts/pre-generation-probe-layer-sweep.json \
  --sweep-layers=-12,-8,-4 \
  --best-by auto \
  --save-artifact artifacts/pre-generation-best-probe.pt \
  --save-calibration artifacts/pre-generation-best-calibration.json \
  --conformal-alpha 0.2 \
  --soft-target-cutoff 0.5
```

`eval_truthfulqa.py` can now generate compatible local records from the same
forced-answer forward pass used for score dumps:

```bash
python benchmarks/eval_truthfulqa.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --layer -8 \
  --limit 200 \
  --pre-generation-probe-layers=-12,-8,-4 \
  --dump-pre-generation-probe-records artifacts/pre-generation-probe-records.jsonl
```

Use the workflow wrapper when you want one reproducible local handoff from record
export to probe sweep, best artifact, calibration, workflow report, and artifact
manifest:

```bash
python benchmarks/run_pre_generation_probe_workflow.py \
  --output-dir artifacts/runtime_evidence/pre-generation-smollm2-l12 \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --real-truthfulqa \
  --layer -8 \
  --limit 12 \
  --manifold-questions 12 \
  --max-length 192 \
  --batch-size 1 \
  --max-batch-tokens 1536 \
  --hidden-state-capture hooks \
  --pre-generation-layers=-12,-8,-4 \
  --sweep-layers auto \
  --record-grain candidate \
  --conformal-alpha 0.2 \
  --train-fraction 0.75
```

`artifacts/runtime_evidence/` is intentionally ignored by git for large local
records and torch artifacts. Promote only compact reports or manifests when they
are part of a maintained baseline.

After running several workflow reports, aggregate the compact evidence without
loading the large records:

```bash
python benchmarks/eval_pre_generation_text_baselines.py \
  --records artifacts/runtime_evidence/pre-generation-smollm2-l12/records.jsonl \
  --json artifacts/runtime_evidence/pre-generation-smollm2-l12-workflow/text-baseline.json

python benchmarks/eval_pre_generation_text_baselines.py \
  --records artifacts/runtime_evidence/pre-generation-qwen05-l12-workflow/records.jsonl \
  --json artifacts/runtime_evidence/pre-generation-qwen05-l12-workflow/text-baseline.json

python benchmarks/compare_pre_generation_probe_workflows.py \
  --workflow-report smollm2=artifacts/runtime_evidence/pre-generation-smollm2-l12-workflow/pre-generation-probe-workflow.json \
  --workflow-report qwen05=artifacts/runtime_evidence/pre-generation-qwen05-l12-workflow/pre-generation-probe-workflow.json \
  --redline-report smollm2=artifacts/runtime_evidence/pre-generation-smollm2-l12-workflow/text-baseline.json \
  --redline-report qwen05=artifacts/runtime_evidence/pre-generation-qwen05-l12-workflow/text-baseline.json \
  --json artifacts/runtime_evidence/pre-generation-qwen-smollm2-l12-comparison/comparison.json \
  --artifact-manifest artifacts/runtime_evidence/pre-generation-qwen-smollm2-l12-comparison/artifact-manifest.json \
  --min-model-count 2 \
  --min-record-count 80 \
  --min-test-label-auroc 0.7 \
  --min-redline-auroc-margin 0.05
```

That comparison can be used as release evidence:

```bash
python benchmarks/compare_release_candidates.py \
  --readiness-registry artifacts/registry.json \
  --pre-generation-probe-comparison artifacts/runtime_evidence/pre-generation-qwen-smollm2-l12-comparison/comparison.json
```

By default, that export writes one prompt-level record per question and uses the
question's candidate false-answer rate as the soft target. Use
`--pre-generation-record-grain candidate` for candidate-level records with hard
`label` fields. `--pre-generation-probe-layers` writes a `layer_hidden_states`
mapping per record and keeps the first layer in `prompt_hidden_states` for backward
compatibility; use `eval_pre_generation_probe.py --record-layer <layer>` to train
or calibrate one selected layer from the same record file, or `--sweep-layers`
to train/rank several layer candidates from that same file. In sweep mode,
`--save-artifact` and `--save-calibration` save the recommended candidate. Use the equals form
(`--pre-generation-probe-layers=-12,-8,-4`) when the list starts with a negative
layer index. If the run reads an
older `--eval-reps-cache` that does not contain prompt hidden states, refresh the
cache before exporting.

Each JSON/JSONL record must provide `hidden_states` (or `prompt_hidden_states`) with
shape `[tokens, hidden_dim]`, plus either `soft_target`, `risk_target`, or
`sample_correctness`. `attention_mask` is optional and defaults to all tokens kept.
`label` or `is_false` is optional; when both classes are present, the report includes
label AUROC in addition to soft-target MSE/MAE/BCE. The script also computes a
split-conformal threshold over `pre_generation_risk_probability` when every record
has hard labels. Use `--save-calibration` to persist that threshold as a
`CalibrationArtifact`. For prompt-level records without hard labels, pass
`--soft-target-cutoff` only when you intentionally want to derive a calibration
label from the soft target; otherwise conformal calibration is reported as
unavailable and explicit artifact saving fails closed.

`eval_pre_generation_probe.py` itself does not load a model or download data. The
current handoff proves the local record/export/train/evaluate/calibrate/layer-sweep path;
detector-quality claims still require larger model runs, held-out calibration, and
release evidence.

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
  --abstention-signals maha_last,truth_proj,subspace_resid,first_token_entropy,inside_eigenscore \
  --abstention-alpha 0.1 \
  --save-abstention-comparison artifacts/gpt2-abstention-comparison.json

# Turn the selected abstention report/comparison candidate into a release gate:
python benchmarks/eval_conformal.py --scores benchmarks/scores.json --signal maha_last \
  --abstention-signals maha_last,truth_proj,subspace_resid,first_token_entropy,inside_eigenscore \
  --abstention-alpha 0.1 \
  --save-abstention-release-gate artifacts/gpt2-abstention-release-gate.json \
  --min-abstention-conditional-correctness-lower-bound 0.8 \
  --max-abstention-rate 0.5

# Replay a finite alpha budget across a session/batch-style score sequence:
python benchmarks/eval_conformal.py --scores benchmarks/scores.json --signal maha_last \
  --sequential-alpha 0.1 \
  --sequential-schedule harmonic \
  --save-sequential-report artifacts/gpt2-sequential-conformal.json \
  --save-sequential-calibration artifacts/gpt2-sequential-calibration.json

# Build the 0.2 calibrated-observability closure: layer/score sweep + best artifact:
python benchmarks/eval_conformal.py --scores benchmarks/scores.json \
  --signals maha_last,truth_proj,subspace_resid,resid_update_norm,eigenscore,first_token_entropy,inside_eigenscore,inside_semantic_entropy,inside_embedding_entropy,inside_semantic_energy \
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
For budgeted answer/acquire/abstain policies, use
`eigentruth.calibration.EvidenceAcquisitionConformalCalibrator` on labeled
post-policy records rather than reusing a pre-acquisition threshold. It compares
the naive pre-score abstention report against the post-acquisition policy report
and saves the post-policy threshold as a standard `CalibrationArtifact`, matching
the evidence-acquisition control loop without adding a model or network
dependency. Saved product traces can be converted into those rows with
`evidence_acquisition_record_from_trace` or
`evidence_acquisition_records_from_trace_feedback`, so feedback-labeled
answer/acquire/abstain runs can be recalibrated offline against the complete
policy score distribution. The corresponding workflow script is
`calibrate_evidence_acquisition_from_traces.py`: it reads saved `ProductTrace`
JSON/JSONL and optional `ProductFeedbackRecord` JSONL labels, emits extracted
records when requested, writes an `EvidenceAcquisitionCalibrationReport`, saves
the post-policy `CalibrationArtifact`, and can fingerprint/register the result.
For held-out or subsequently collected feedback, `audit_evidence_acquisition_risk`
can replay the deployed threshold over labeled post-policy records and alpha-spend
finite prefix checks, producing a JSON-ready monitor report that fails closed when
the accepted-error upper bound exceeds the target error rate. For continuously
arriving feedback, `audit_evidence_acquisition_anytime_risk` adds a dependency-free
mixture e-process monitor over accepted-error Bernoulli outcomes; it alarms when
the mixture e-value crosses `1 / monitor_alpha` without recalibrating on the same
feedback stream. Library callers can persist
`EvidenceAcquisitionAnytimeRiskMonitorState` and update it one feedback record at
a time; the trace workflow can emit full monitor reports directly with
`--risk-target-error-rate`; `--risk-monitor-mode prefix` preserves the finite
prefix default, while `--risk-monitor-mode anytime` or `both` adds the anytime
report. Use `--risk-monitor-json`, `--anytime-risk-monitor-json`,
`--risk-monitor-alpha`, `--risk-monitor-schedule`, repeatable
`--risk-monitor-checkpoint`, and repeatable `--risk-monitor-bet-fraction` to
control outputs. Any failed monitor sets the workflow status to `blocked` and is
included in the artifact manifest and local registry metadata.
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
When `--save-sequential-report` or `--include-sequential-report` is set, the
script emits a sequential conformal replay report. It uses a seeded split of
true rows for calibration, replays the remaining true rows plus false rows in
score-dump order, and spends one finite alpha budget with `linear`, `harmonic`,
or `geometric` scheduling. This is a session/batch audit sidecar: it does not
change the base E1 verdict. `--save-sequential-calibration` stores the same
normal-score calibration distribution and alpha-spending schedule as a reusable
`SequentialConformalArtifact` for runtime sequence scoring. Product integrations
can pass that artifact to `RiskController(..., sequential_gate=...)` and call
`decide_sequence(...)`; per-request `decide(...)` remains stateless. The
no-model product demo can replay the artifact with
`examples/calibrated_control_demo.py --sequential-gate ... --diagnostics-sequence
'[...]'`, and `run_product_runtime_baseline.py` can aggregate those
`risk_decision_sequence` traces as decision/gate summaries without treating
them as ordinary runtime-timed ProductTrace payloads. If a runtime budget policy
is configured for such traces, the baseline fails closed with
`unsupported_trace_format` instead of inventing missing phase timings.
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
`eval_detectability_taxonomy.py`, `eval_verifier_ensemble.py`,
`eval_calibration_transfer.py`, and
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
fingerprints the input score dump plus generated conformal, sweep, sequential,
and calibration artifacts for later verification or registry promotion.

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

To include optional attention-flow pathway columns in the score dump, add
`--attention-pathway --attn-implementation eager` and include the desired
`attn_*` score names in `--signals`. These columns stay out of the default
signal set because some Transformers attention backends do not return
attentions.

When a mechanism intervention has been rerun as a separate row-aligned score
dump, compare it with the baseline run before making a pathway-causality claim:

```bash
python benchmarks/eval_truthfulqa.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --layer -8 --limit 200 --hidden-state-capture hooks \
  --dump-scores artifacts/pathway-baseline/scores.manifest.json \
  --dump-scores-format jsonl

python benchmarks/eval_truthfulqa.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --layer -8 --limit 200 --hidden-state-capture hooks \
  --activation-intervention-layer -8 \
  --activation-intervention-span answer \
  --activation-intervention-mode zero \
  --dump-scores artifacts/pathway-answer-zero/scores.manifest.json \
  --dump-scores-format jsonl

python benchmarks/eval_pathway_intervention.py \
  --baseline-scores artifacts/pathway-baseline/scores.manifest.json \
  --intervened-scores artifacts/pathway-answer-zero/scores.manifest.json \
  --signals pathway_disagreement,truth_proj,nll_answer \
  --direction truth_proj=higher \
  --pathway answer \
  --intervention-name answer_activation_zero \
  --json artifacts/pathway-answer-zero/intervention-effect-report.json \
  --artifact-manifest artifacts/pathway-answer-zero/artifact-manifest.json \
  --registry artifacts/local-release-registry.json \
  --register-name pathway-answer-zero \
  --register-version 0.1
```

Source-token activation patching uses the same compare step. The patch run
chooses a source statement per target row, defaulting to an opposite-label
candidate from the same TruthfulQA question and falling back to the global
opposite-label pool for ungrouped smoke data:

```bash
python benchmarks/eval_truthfulqa.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --layer -8 --limit 200 --hidden-state-capture hooks \
  --activation-patch-layer -8 \
  --activation-patch-target-span answer \
  --activation-patch-source-span answer \
  --activation-patch-alignment left \
  --activation-patch-source opposite_label \
  --dump-scores artifacts/pathway-answer-source-patch/scores.manifest.json \
  --dump-scores-format jsonl
```

To run the full mechanism-evidence chain in one command, use the workflow
wrapper. It writes the baseline, activation-ablation, source-patch, both
comparison reports, and a top-level artifact manifest under one output
directory:

```bash
python benchmarks/run_pathway_intervention_workflow.py \
  --output-dir artifacts/pathway-answer-mechanism \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --layer -8 \
  --intervention-layer -8 \
  --patch-layer -8 \
  --limit 200 \
  --signals pathway_disagreement,truth_proj,nll_answer \
  --registry artifacts/local-release-registry.json \
  --name pathway-answer-mechanism \
  --version 0.1
```

The two score dumps must cover the same examples in the same order with
identical labels. The report uses score directions to compute positive
`risk_reduction` when the intervention lowers anomaly under that signal; it is
rerun evidence from a model-side activation intervention. `--eval-reps-cache`
and `--prefix-kv-cache` are intentionally disabled for activation intervention
runs so a baseline representation cache cannot contaminate the intervention
dump.
The same cache restriction applies to `--activation-patch-layer`.

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
  --detectability-consistency-signal eigenscore \
  --detectability-consistency-direction lower \
  --detectability-confidence-signal nll_answer \
  --detectability-confidence-direction lower \
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
Use `--attention-pathway --attn-implementation eager` with explicit `attn_*`
entries in `--signals` when the frontier cells should collect optional
attention-flow pathway evidence.
Use `--detectability-consistency-signal` and
`--detectability-confidence-signal` to write one DECK-style taxonomy report per
cell. These reports are added to the top-level manifest and can be passed
directly to `compare_frontier_release_evidence.py --detectability-taxonomy-report`
for entrenched blind-spot gating.

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
  --signals maha_last,truth_proj,subspace_resid,disp_euclid,disp_hse,nll_answer,first_token_entropy,eigenscore,resid_update_norm \
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

For verifier-enhanced score dumps, the same replay can evaluate calibrated
fusion candidates without writing fused columns back into the dump. Rank fusion
uses the current seed's correct calibration split as the normal reference;
geometry/uncertainty fusion first rank-calibrates each group, then combines the
two group scores:

```bash
python benchmarks/eval_abstention_stability.py \
  --scores qwen05-l80=artifacts/truthfulqa-l80-local-retrieval-verifier-signal-fusion/qwen-l80-enhanced-scores.manifest.json \
  --scores smollm2-l80=artifacts/truthfulqa-l80-local-retrieval-verifier-signal-fusion/smollm2-l80-enhanced-scores.manifest.json \
  --signals truth_proj,subspace_resid,eigenscore,verifier_refuted,verifier_refute_confidence,verifier_not_supported,verifier_uncertainty \
  --rank-fusion-signals truth_proj,subspace_resid,eigenscore,verifier_refuted,verifier_refute_confidence,verifier_not_supported,verifier_uncertainty \
  --rank-fusion-methods max_rank,mean_rank,noisy_or_rank \
  --geometry-signals truth_proj,subspace_resid,eigenscore \
  --uncertainty-signals verifier_refuted,verifier_refute_confidence,verifier_not_supported \
  --geometry-method mean_rank \
  --uncertainty-method mean_rank \
  --geometry-fusion-methods interaction,product,weighted_mean,noisy_or \
  --alpha 0.2 \
  --enforce-abstention-budget \
  --abstention-budget-target-rate 0.48 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --json artifacts/frontier-release-evidence/verifier-fusion-abstention-stability-v1/local-retrieval-fusion-abstention-stability-alpha-0p2-budgeted-target-0p48.json
```

`--enforce-abstention-budget` keeps the conformal threshold but also applies a
seed-calibration split score quantile so the candidate targets a bounded total
abstention rate before held-out evaluation. The release gate is still evaluated
against `--max-abstention-rate`; `--abstention-budget-target-rate` is an
optional safety margin for the threshold policy. The current local-retrieval
fusion artifact (`frontier-release-evidence-verifier-fusion-abstention-v3`)
stabilizes on `geometry_uncertainty_fusion:noisy_or` and clears mean
conditional-correctness/abstention gates, but remains blocked by strict 10/10
seed pass-rate: qwen05-l80 passes 7/10 seeds and smollm2-l80 passes 5/10.

When fixed budget targets are unstable, pass
`--abstention-budget-target-rates` to evaluate an explicit profile sweep. Each
budget target becomes a separate candidate name, for example
`...@budget=0.49`. Add `--prefer-release-gate-passing` when the benchmark should
rank candidates that satisfy the configured correctness and abstention release
gate ahead of higher-correctness candidates that exceed the abstention budget:

```bash
python benchmarks/eval_abstention_stability.py \
  --scores qwen05-l80=artifacts/truthfulqa-l80-local-retrieval-verifier-signal-fusion/qwen-l80-enhanced-scores.manifest.json \
  --scores smollm2-l80=artifacts/truthfulqa-l80-local-retrieval-verifier-signal-fusion/smollm2-l80-enhanced-scores.manifest.json \
  --signals truth_proj,subspace_resid,eigenscore,verifier_refuted,verifier_refute_confidence,verifier_not_supported,verifier_uncertainty \
  --geometry-signals truth_proj,subspace_resid,eigenscore \
  --uncertainty-signals verifier_refuted,verifier_refute_confidence,verifier_not_supported \
  --geometry-method mean_rank \
  --uncertainty-method mean_rank \
  --geometry-fusion-methods interaction,product,weighted_mean,noisy_or \
  --alpha 0.2 \
  --enforce-abstention-budget \
  --abstention-budget-target-rates 0.43,0.45,0.46,0.47,0.48,0.49,0.5,0.51,0.52,0.54 \
  --prefer-release-gate-passing \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --json artifacts/frontier-release-evidence/verifier-fusion-abstention-stability-v2/local-retrieval-fusion-abstention-stability-alpha-0p2-budget-target-sweep.json
```

The budget-target sweep release artifact
`frontier-release-evidence-budget-target-sweep-v4` promotes all frontier tracks:
both qwen05-l80 and smollm2-l80 pass 10/10 abstention seeds with the same
geometry/verifier `noisy_or` fusion family. The selected budget target varies by
seed and remains visible in each recommended candidate name, so this evidence
should be treated as an explicit profile-sweep policy rather than a hidden fixed
threshold.

## `compare_frontier_release_evidence.py`

Combines frontier stability reports into one fail-closed release verdict without
rerunning models, verifiers, or retrieval. It treats staged verifier stability
and abstention-gate stability as separate required tracks, and can optionally
gate a DECK-style detectability taxonomy track. The release candidate blocks if
any provided track misses its configured seed-rate, metric, or blind-spot
threshold. It can also consume citation/source-family batch evidence rollups:
when a `--citation-batch-rollup-report` is supplied, every expected batch must
be observed exactly once, child evidence gates must be promotion-ready, and
child manifests must have passed inside the rollup. Completed frontier rerun
rollups can be supplied with repeatable `--frontier-rerun-rollup-report`; each
rollup must be `status=promote`, `gate.passed=true`, `gate.promotion_ready=true`,
and include an artifact manifest before it can clear a previously blocked
release-evidence track.

```bash
OUT=artifacts/truthfulqa-frontier-qwen-smollm2-l80-release-evidence

python benchmarks/compare_frontier_release_evidence.py \
  --verifier-stability-report artifacts/truthfulqa-frontier-qwen-smollm2-l80-verifier-stability/verifier-stability-report.json \
  --abstention-stability-report artifacts/truthfulqa-frontier-qwen-smollm2-l80-abstention-stability/abstention-stability-report.json \
  --frontier-workflow-report artifacts/truthfulqa-frontier-qwen-smollm2-l80/truthfulqa-frontier-workflow.json \
  --detectability-taxonomy-report artifacts/truthfulqa-frontier-qwen-smollm2-l80-detectability/qwen05-l80/detectability-taxonomy-report.json \
  --detectability-taxonomy-report artifacts/truthfulqa-frontier-qwen-smollm2-l80-detectability/smollm2-l80/detectability-taxonomy-report.json \
  --citation-batch-rollup-report artifacts/truthfulqa-frontier-smollm2-l80-citation-batch-rollup/citation-batch-rollup.json \
  --frontier-rerun-rollup-report artifacts/frontier-release-evidence/abstention-rerun-rollup.json \
  --max-detectability-entrenched-false-rate 0.25 \
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
When `--frontier-workflow-report` is supplied, the comparator also gates the
frontier workflow's family-wise multiple-testing evidence. Summary counts alone
are not sufficient: the report must include a `multiple_testing_gate.cells` list
whose length and pass/fail/unknown counts match `cell_count`, and each cell must
carry both report and calibration artifact paths. This keeps release promotion
from accepting a top-level `all_pass` summary that cannot be traced back to
per-cell conformal artifacts.
When `--citation-batch-rollup-report` is supplied, the comparator adds a
`citation_batch_track_status` to the release decision and records expected,
observed, missing, duplicate, and unexpected batch counts in the release report,
registry metadata, and artifact manifest. A rollup with missing expected
batches, duplicate batches, unsupported child workflows, failed child gates, or
failed child-manifest verification blocks the frontier release verdict.
When `--frontier-rerun-rollup-report` is supplied, the comparator records a
`frontier_rerun_rollup_track_status` and fingerprints the rollup report plus
its manifest. This is the intended handoff path for
`rollup_frontier_stability_evidence_reruns.py`,
`rollup_frontier_abstention_evidence_reruns.py`,
`rollup_frontier_detectability_evidence_reruns.py`, and
`rollup_frontier_multiple_testing_reruns.py`; blocked, empty, or
manifest-less rerun rollups fail closed.
Product runtime baselines also aggregate that citation-batch track from
promotion-contract trace metadata; `compare_product_runtime_baselines.py` can
gate the track promote rate, require a rollup count, and fail closed on missing,
duplicate, or unexpected citation/source-family batches so release evidence
cannot silently disappear after handoff.
When `--detectability-taxonomy-report` is supplied, each run must have a matching
taxonomy report. The default blind-spot gate blocks if more than 25% of false
records fall into the `entrenched` cell, because that cell is repeatable and
high-confidence enough that output-level uncertainty is expected to miss it.
The registered detectability replay
(`report:truthfulqa-frontier-qwen-smollm2-l80-detectability:0.1`) reuses the
l80 score dumps and emits per-cell reports in 7.5s: Qwen has entrenched
false-rate `0.000`, while SmolLM2 has `89/306 = 0.291`. The corresponding
release-evidence artifact
(`report:truthfulqa-frontier-qwen-smollm2-l80-release-evidence-detectability:0.1`)
keeps verifier stability promoted, keeps abstention blocked, and also blocks
the detectability track on SmolLM2 at the default `0.25` blind-spot gate.

## `analyze_detectability_blind_spots.py`

Exports row-level examples from a detectability taxonomy cell without loading a
model. Use it after a DECK report blocks a release gate to inspect which
questions and answer patterns make up the blind spot.

```bash
OUT=artifacts/truthfulqa-frontier-qwen-smollm2-l80-detectability-blind-spots

python benchmarks/analyze_detectability_blind_spots.py \
  --taxonomy-report artifacts/truthfulqa-frontier-qwen-smollm2-l80-detectability/smollm2-l80/detectability-taxonomy-report.json \
  --cell entrenched \
  --json "$OUT/smollm2-l80-entrenched-blind-spots.json" \
  --artifact-manifest "$OUT/artifact-manifest.json" \
  --registry artifacts/local-release-registry.json \
  --name truthfulqa-frontier-smollm2-l80-entrenched-blind-spots \
  --version 0.1
```

The registered SmolLM2 l80 report
(`report:truthfulqa-frontier-smollm2-l80-entrenched-blind-spots:0.1`) exports
all `89` false entrenched records and passes artifact-manifest verification.
The largest groups are definition/what questions (`39`), person questions
(`13`), and choice questions (`8`); the mean answer length is `5.18` tokens.
These records are the next concrete target for verifier/world-model correction
route design.

## `audit_blind_spot_correction_routes.py`

Joins a row-level blind-spot report with
`eval_verifier_ensemble.py --verified-records-jsonl` output. Use it to check
whether an independent route actually covers the high-confidence false records
that detectability gates blocked.

```bash
OUT=artifacts/truthfulqa-frontier-smollm2-l80-blind-spot-route-audit

python benchmarks/eval_verifier_ensemble.py \
  --scores retrieval=artifacts/smollm2_truthfulqa_l80_scores_with_statements.json \
  --claims artifacts/smollm2_l80_retrieval_structured_qa_route_v0_6/retrieval-claims.json \
  --signal truth_proj \
  --alphas 0.1 \
  --repeats 1 \
  --seed 0 \
  --verifier-min-overlap 0.65 \
  --retriever-min-overlap 0.95 \
  --retrieval-limit 3 \
  --verified-records-jsonl "$OUT/verified-records.jsonl" \
  --json "$OUT/retrieval-verifier-report-with-sidecar.json"

python benchmarks/audit_blind_spot_correction_routes.py \
  --blind-spots artifacts/truthfulqa-frontier-qwen-smollm2-l80-detectability-blind-spots/smollm2-l80-entrenched-blind-spots.json \
  --verified-records-jsonl "$OUT/verified-records.jsonl" \
  --verifier-report "$OUT/retrieval-verifier-report-with-sidecar.json" \
  --claims artifacts/smollm2_l80_retrieval_structured_qa_route_v0_6/retrieval-claims.json \
  --route-report artifacts/smollm2_l80_retrieval_structured_qa_route_v0_6/retrieval-route-comparison.json \
  --json "$OUT/blind-spot-route-audit.json" \
  --artifact-manifest "$OUT/artifact-manifest.json" \
  --registry artifacts/local-release-registry.json \
  --name truthfulqa-frontier-smollm2-l80-blind-spot-route-audit \
  --version 0.1
```

The registered SmolLM2 audit
(`report:truthfulqa-frontier-smollm2-l80-blind-spot-route-audit:0.1`) verifies
all `89` blind-spot records against the sidecar. The promoted
`retrieval_structured_qa` route selects and refutes `3/89` records, supports
`0/89`, and leaves `86/89` outside the target route. That means the route's
selected subset is precise, but the blind-spot correction problem is now
coverage: expand query construction, external corpora, structured-fact
predicates, citation checks, and world-model routes before using this evidence
to relax the detectability release blocker.

## `sweep_blind_spot_retrieval_queries.py`

Sweeps local retrieval query fields and overlap thresholds over the same
blind-spot set. This is useful after a route audit shows a coverage gap: it
separates query-construction failures from verifier failures while preserving
the corpus provenance warning. Each strategy now also embeds a compact
`gap_analysis` payload from `analyze_retrieval_route_gaps.py`, so blocked
reports distinguish `no_retrieval_hits`, `false_negative`,
`low_overlap_after_retrieval`, `false_positive`, and successful
`false_refuted` / `true_supported` buckets without rerunning a separate sidecar.

```bash
OUT=artifacts/truthfulqa-frontier-smollm2-l80-blind-spot-query-sweep

python benchmarks/sweep_blind_spot_retrieval_queries.py \
  --scores artifacts/smollm2_truthfulqa_l80_scores_with_statements.json \
  --corpus artifacts/truthfulqa_l80_correct_answer_corpus.json \
  --blind-spots artifacts/truthfulqa-frontier-qwen-smollm2-l80-detectability-blind-spots/smollm2-l80-entrenched-blind-spots.json \
  --query-fields answer,question,question_answer,text \
  --retriever-min-overlaps 0.95,0.8,0.65,0.5 \
  --retrieval-limit 3 \
  --signal truth_proj \
  --alpha 0.1 \
  --verifier-min-overlap 0.65 \
  --json "$OUT/blind-spot-query-sweep.json" \
  --artifact-manifest "$OUT/artifact-manifest.json" \
  --registry artifacts/local-release-registry.json \
  --name truthfulqa-frontier-smollm2-l80-blind-spot-query-sweep \
  --version 0.1
```

The registered SmolLM2 sweep
(`report:truthfulqa-frontier-smollm2-l80-blind-spot-query-sweep:0.1`) evaluates
`16` strategies. The original `answer@0.95` baseline refutes only `3/89` blind
spots. `answer@0.5` refutes `54/89` with verified false alarm `0.024`.
`question_answer@0.65` refutes `87/89` with verified false alarm `0.000`, and
`question_answer@0.5` refutes `89/89` with verified false alarm `0.000` in the
controlled corpus replay. `question@0.95` also refutes `89/89`, but false alarm
rises to `0.176`, so it stays a negative control. Because the source corpus is
`truthfulqa_correct_answer_evidence`, treat the best strategy as query-design
evidence to port to external/structured corpora, not as open-domain grounding.

The external provenance rerun keeps that distinction explicit:

```bash
OUT=artifacts/truthfulqa-frontier-smollm2-l80-external-query-sweep

python benchmarks/sweep_blind_spot_retrieval_queries.py \
  --scores artifacts/smollm2_truthfulqa_l80_scores_with_statements.json \
  --corpus artifacts/wikidata-country-core-facts-external-corpus/wikidata-country-core-facts-corpus.json \
  --blind-spots artifacts/truthfulqa-frontier-qwen-smollm2-l80-detectability-blind-spots/smollm2-l80-entrenched-blind-spots.json \
  --query-fields answer,question_answer \
  --retriever-min-overlaps 0.95,0.8,0.65,0.5,0.35,0.2 \
  --retrieval-limit 5 \
  --json "$OUT/blind-spot-external-query-sweep.json" \
  --artifact-manifest "$OUT/artifact-manifest.json"
```

On the committed Wikidata country-core-facts external corpus, the best observed
blind-spot refuted rate is `0/89`; the structured-QA corpus used as retrieval
documents also remains at `0/89`. These are negative coverage results, not
failures of the artifact pipeline.

## `compare_blind_spot_query_sweeps.py`

Compares controlled query-sweep reports with external or structured-evidence
query-sweep reports. The gate is fail-closed: controlled coverage is accepted
only as query-design evidence unless an external/structured sweep also passes
the configured blind-spot and verified false-alarm thresholds.

```bash
OUT=artifacts/truthfulqa-frontier-smollm2-l80-query-sweep-provenance-comparison

python benchmarks/compare_blind_spot_query_sweeps.py \
  --controlled-sweep artifacts/truthfulqa-frontier-smollm2-l80-blind-spot-query-sweep/blind-spot-query-sweep.json \
  --external-sweep artifacts/truthfulqa-frontier-smollm2-l80-external-query-sweep/blind-spot-external-query-sweep.json \
  --external-sweep artifacts/truthfulqa-frontier-smollm2-l80-structured-qa-query-sweep/blind-spot-structured-qa-query-sweep.json \
  --min-controlled-blind-refuted-rate 0.5 \
  --min-external-blind-refuted-rate 0.5 \
  --max-controlled-verified-false-alarm 0.05 \
  --max-external-verified-false-alarm 0.05 \
  --json "$OUT/query-sweep-provenance-comparison.json" \
  --artifact-manifest "$OUT/artifact-manifest.json" \
  --registry artifacts/local-release-registry.json \
  --name truthfulqa-frontier-smollm2-l80-query-sweep-provenance-comparison \
  --version 0.1
```

The registered comparison
(`report:truthfulqa-frontier-smollm2-l80-query-sweep-provenance-comparison:0.1`)
is `blocked`: the controlled sweep passes with `question_answer@0.5`
(`89/89`), while both external sweeps pass `0` strategies and have best observed
blind-spot refuted rate `0.0`. This blocks any runtime-default promotion until
the question-aware strategy is backed by broader external or structured-fact
coverage.

## `plan_blind_spot_evidence_expansion.py`

Turns blocked blind-spot coverage into an external evidence collection plan. It
does not fetch new data and does not promote a route. Instead, it maps each
blind spot to likely evidence families: structured-fact properties,
structured-QA/citation retrieval, counterfactual probes, and world-model or
calculator checks. When `--query-sweep` points at a report with embedded
`gap_analysis`, the planner also emits gap-informed alignment actions such as
claim-evidence alignment, source-document fact extraction, query refinement,
and negative-control audits.

```bash
OUT=artifacts/truthfulqa-frontier-smollm2-l80-blind-spot-evidence-expansion-plan

python benchmarks/plan_blind_spot_evidence_expansion.py \
  --blind-spots artifacts/truthfulqa-frontier-qwen-smollm2-l80-detectability-blind-spots/smollm2-l80-entrenched-blind-spots.json \
  --provenance-comparison artifacts/truthfulqa-frontier-smollm2-l80-query-sweep-provenance-comparison/query-sweep-provenance-comparison.json \
  --query-sweep artifacts/frontier-release-evidence/unresolved-seeded-news-query-sweep-gap-analysis-v1/blind-spot-query-sweep-gap-analysis.json \
  --json "$OUT/blind-spot-evidence-expansion-plan.json" \
  --artifact-manifest "$OUT/artifact-manifest.json" \
  --registry artifacts/local-release-registry.json \
  --name truthfulqa-frontier-smollm2-l80-blind-spot-evidence-expansion-plan \
  --version 0.1
```

The registered plan
(`report:truthfulqa-frontier-smollm2-l80-blind-spot-evidence-expansion-plan:0.1`)
has status `needs_evidence_collection`. It covers all `89` blind spots, marks
`65` high priority, recommends `structured_fact` for `80`, `structured_qa` for
`65`, `retrieval_citation` for `63`, counterfactual probes for `41`, and
world-model/calculator tasks for `21`. This is the concrete worklist for
expanding external/structured coverage after the provenance comparison blocked
controlled-only query evidence.

The frontier v4 release evidence also includes a refreshed SmolLM2 expansion
artifact at
`artifacts/frontier-release-evidence/blind-spot-evidence-expansion-v1/evidence-expansion-plan.json`.
It is generated from the detectability rerun queue's `89` entrenched false
records and records the same next-step profile: `65` high-priority targets,
`80` structured-fact routes, `65` structured-QA routes, `63` citation-retrieval
routes, `41` counterfactual probe tasks, and `21` world-model/calculator tasks.
The planner filters generic single-token entity candidates such as `Son`,
`American`, and `Nothing` before writing collection targets, so downstream
Wikidata/search tasks receive cleaner entity seeds while retaining useful
question-keyword phrases.

The gap-informed frontier v4 alignment plan at
`artifacts/frontier-release-evidence/unresolved-seeded-news-alignment-expansion-plan-v1/evidence-expansion-plan.json`
consumes the seeded-news query-sweep gap report directly. It keeps status
`needs_evidence_collection`, covers all `89` blind spots, and records
`best_strategy=question_overlap_0p65`, `best_passing_strategy=null`,
`dominant_gap_bucket=no_retrieval_hits`, `301/306` false records still missed,
and only `45/556` verification records using retrieval. The generated worklist
therefore adds `claim_evidence_alignment`, `query_refinement`,
`source_document_fact_extraction`, and `negative_control_alignment_audit` to
all `89` targets before another verifier sweep.

## `build_blind_spot_evidence_collection_corpus.py`

Compiles an evidence expansion plan into source-discovery request batches. This
output is intentionally not verifier evidence: it is the executable queue for
Wikidata/entity-property lookup, external citation retrieval, counterfactual
probe generation, deterministic world-model/calculator rule authoring, and
gap-informed claim-evidence alignment audits.

```bash
OUT=artifacts/truthfulqa-frontier-smollm2-l80-blind-spot-evidence-collection-corpus

python benchmarks/build_blind_spot_evidence_collection_corpus.py \
  --plan artifacts/truthfulqa-frontier-smollm2-l80-blind-spot-evidence-expansion-plan/blind-spot-evidence-expansion-plan.json \
  --priority high \
  --json "$OUT/blind-spot-evidence-collection-corpus.json" \
  --artifact-manifest "$OUT/artifact-manifest.json" \
  --registry artifacts/local-release-registry.json \
  --name truthfulqa-frontier-smollm2-l80-blind-spot-evidence-collection-corpus \
  --version 0.1
```

The registered high-priority corpus has status `ready_for_collection`: `65`
targets, `931` total requests, `720` Wikidata/entity-property requests, `176`
citation requests, `29` counterfactual probes, and `6`
world-model/calculator-rule authoring requests. It keeps the first collection
pass focused on the targets most likely to unlock structured fact or citation
coverage before rerunning `sweep_blind_spot_retrieval_queries.py` and
`compare_blind_spot_query_sweeps.py`.

The frontier v4 corpus at
`artifacts/frontier-release-evidence/blind-spot-evidence-collection-corpus-v1/blind-spot-evidence-collection-corpus.json`
is the full follow-on queue for the v4 expansion plan. It covers all `89`
targets and emits `1283` collection requests: `950` Wikidata/entity-property
requests, `260` citation requests, `47` counterfactual probes, and `26`
world-model/calculator-rule authoring requests. Its manifest verification
passes and the registry records it as
`report:smollm2-l80-blind-spot-evidence-collection-corpus:0.1`.

The corresponding gap-informed alignment corpus at
`artifacts/frontier-release-evidence/unresolved-seeded-news-alignment-collection-corpus-v1/evidence-collection-corpus.json`
adds an `alignment_audit` request bucket. It covers all `89` targets and emits
`1372` total requests: `950` Wikidata/entity-property, `260` citation, `47`
counterfactual, `26` world-model/calculator, and `89` alignment-audit
requests. The alignment requests remain `alignment_audit_only`; they are inputs
for extracting subject/property/value/evidence-span triples, not verifier
evidence.

## `audit_blind_spot_alignment_requests.py`

Audits `alignment_audit` requests against an external evidence corpus and emits
review-only subject/property/value/evidence-span candidates. This is the first
execution step after a query-sweep gap analysis says source acquisition is
complete but route quality is still blocked. The output remains non-evidence:
candidate facts are `structured_fact_review_only` until a later structured-fact
or world-model route validates them.

```bash
OUT=artifacts/frontier-release-evidence/unresolved-seeded-news-alignment-audit-v1

python benchmarks/audit_blind_spot_alignment_requests.py \
  --collection-corpus artifacts/frontier-release-evidence/unresolved-seeded-news-alignment-collection-corpus-v1/evidence-collection-corpus.json \
  --evidence-corpus artifacts/frontier-release-evidence/unresolved-seeded-news-source-family-citation-workflow-v1/evidence-gate/citation-search-corpus.json \
  --output-dir "$OUT" \
  --artifact-manifest "$OUT/artifact-manifest.json" \
  --registry artifacts/frontier-release-evidence/frontier-route-registry.json \
  --name smollm2-l80-frontier-v4-unresolved-seeded-news-alignment-audit \
  --version 0.1
```

The registered v4 audit has status `ready_for_fact_review`: all `89` alignment
requests find candidate source hits, `30/89` requests reach
`candidate_fact_ready`, and the run emits `30` deduplicated review-only fact
candidates. The audit now derives candidate subjects from evidence spans when a
structured subject/property pattern is visible, while keeping the requested
entity as audit provenance. The main remaining blocker is
`property_only_alignment` (`46` requests), followed by `subject_only_alignment`
(`6`) and `subject_property_aligned_no_value` (`6`). This says the next lift is
subject/entity binding and value extraction from broad source documents, not
more generic source-family acquisition.

## `build_alignment_fact_review_corpus.py`

Deduplicates `structured_fact_review_only` candidates from the alignment audit,
applies fail-closed subject/value/property checks, and emits a review-only
structured QA corpus. The output is deliberately scoped as a fact-review input:
it strips request ids, target ids, labels, and model answers from corpus
metadata, records accepted/skipped decisions in a sidecar JSONL, and does not
promote a verifier route.

```bash
OUT=artifacts/frontier-release-evidence/unresolved-seeded-news-alignment-fact-review-corpus-v1

python benchmarks/build_alignment_fact_review_corpus.py \
  --candidates artifacts/frontier-release-evidence/unresolved-seeded-news-alignment-audit-v1/structured-fact-candidates.jsonl \
  --output "$OUT/alignment-fact-review-corpus.json" \
  --report-json "$OUT/alignment-fact-review-report.json" \
  --records-jsonl "$OUT/alignment-fact-review-records.jsonl" \
  --artifact-manifest "$OUT/artifact-manifest.json" \
  --registry artifacts/frontier-release-evidence/frontier-route-registry.json \
  --name smollm2-l80-frontier-v4-unresolved-seeded-news-alignment-fact-review-corpus \
  --version 0.1
```

The registered v4 review corpus accepts `11/30` candidates after upstream
deduplication and quality gates. The dominant skip reasons are duplicate
candidates (`14`) and subject/evidence-span mismatch (`5`). This keeps the next
loop precise: the eleven rows can be manually or route-specifically reviewed,
while the skipped rows point back to subject binding and value extraction
improvements.

## `promote_alignment_fact_review_corpus.py`

Gates the review-only alignment fact corpus before any structured-fact
promotion. Without explicit review decisions, it writes a decision template and
keeps the run at `needs_review`. With approved decisions, it materializes
source-family style structured source documents that can be passed to
`build_source_family_qa_corpus.py` and then audited by the covered-facts
structured QA route. The gate does not promote verifier evidence by itself.

```bash
OUT=artifacts/frontier-release-evidence/unresolved-seeded-news-alignment-review-promotion-gate-v1

python benchmarks/promote_alignment_fact_review_corpus.py \
  --review-corpus artifacts/frontier-release-evidence/unresolved-seeded-news-alignment-fact-review-corpus-v1/alignment-fact-review-corpus.json \
  --output-dir "$OUT" \
  --artifact-manifest "$OUT/artifact-manifest.json" \
  --registry artifacts/frontier-release-evidence/frontier-route-registry.json \
  --name smollm2-l80-frontier-v4-unresolved-seeded-news-alignment-review-promotion-gate \
  --version 0.1
```

The registered v4 gate currently has status `needs_review`: it sees all `11`
review-corpus rows, writes `11` pending decision-template rows, and emits `0`
approved source documents. This is the intended fail-closed state until a
review decision JSONL supplies `approved`, `rejected`, or
`needs_more_evidence` for each `alignment_candidate_id`.

## `review_alignment_fact_review_corpus.py`

Runs a deterministic route-specific review over the alignment fact-review
corpus. The rule reviewer approves only rows whose Wikidata evidence source and
evidence span close a subject/property/value loop; non-Wikidata or unclosed rows
remain `needs_more_evidence`. Its decisions are ordinary review-decision JSONL
for `promote_alignment_fact_review_corpus.py`, so the promotion gate still
requires explicit reviewer provenance.

```bash
OUT=artifacts/frontier-release-evidence/unresolved-seeded-news-alignment-rule-review-v1

python benchmarks/review_alignment_fact_review_corpus.py \
  --review-corpus artifacts/frontier-release-evidence/unresolved-seeded-news-alignment-fact-review-corpus-v1/alignment-fact-review-corpus.json \
  --output-dir "$OUT" \
  --artifact-manifest "$OUT/artifact-manifest.json" \
  --registry artifacts/frontier-release-evidence/frontier-route-registry.json \
  --name smollm2-l80-frontier-v4-unresolved-seeded-news-alignment-rule-review \
  --version 0.1 \
  --reviewed-at 2026-06-30T00:00:00Z
```

The registered v4 rule review approves all `11/11` review-corpus rows because
they are closed Wikidata subject/property/value spans. Re-running the promotion
gate with `--review-decisions` produces `11` approved source docs and status
`ready_for_structured_qa`; building a source-family QA corpus from those docs
keeps all `11` facts. The covered-facts structured QA route then promotes on
`22` balanced true/false records with decision accuracy `1.0`, true-supported
rate `1.0`, false-refuted rate `1.0`, and false-supported rate `0.0`. This is
covered-fact evidence for the reviewed rows only, not broad blind-spot recall.

```bash
REVIEW=artifacts/frontier-release-evidence/unresolved-seeded-news-alignment-rule-review-v1
GATE=artifacts/frontier-release-evidence/unresolved-seeded-news-alignment-reviewed-promotion-gate-v1
QA=artifacts/frontier-release-evidence/unresolved-seeded-news-alignment-reviewed-source-family-qa-v1
ROUTE=artifacts/frontier-release-evidence/unresolved-seeded-news-alignment-reviewed-structured-qa-route-v1

python benchmarks/promote_alignment_fact_review_corpus.py \
  --review-corpus artifacts/frontier-release-evidence/unresolved-seeded-news-alignment-fact-review-corpus-v1/alignment-fact-review-corpus.json \
  --review-decisions "$REVIEW/review-decisions.jsonl" \
  --output-dir "$GATE" \
  --artifact-manifest "$GATE/artifact-manifest.json"

python benchmarks/build_source_family_qa_corpus.py \
  --source "$GATE/approved-source-documents.json" \
  --output "$QA/source-family-qa-corpus.json" \
  --report-json "$QA/source-family-qa-report.json" \
  --artifact-manifest "$QA/artifact-manifest.json"

python benchmarks/run_source_family_structured_qa_route_workflow.py \
  --qa-corpus "$QA/source-family-qa-corpus.json" \
  --output-dir "$ROUTE" \
  --score-name smollm2-l80-frontier-v4-alignment-reviewed-covered-facts
```

## `fetch_blind_spot_wikidata_evidence.py`

Fetches CC0 Wikidata source documents for the collection corpus. Request and
target identifiers stay in the report; the emitted source docs only carry
external Wikidata provenance plus a request fingerprint, so
`build_external_retrieval_corpus.py` can ingest them without score-row links.

```bash
OUT=artifacts/truthfulqa-frontier-smollm2-l80-blind-spot-wikidata-evidence

python benchmarks/fetch_blind_spot_wikidata_evidence.py \
  --collection-corpus artifacts/truthfulqa-frontier-smollm2-l80-blind-spot-evidence-collection-corpus/blind-spot-evidence-collection-corpus.json \
  --source-jsonl "$OUT/wikidata-source-docs.jsonl" \
  --report-json "$OUT/wikidata-evidence-fetch-report.json" \
  --artifact-manifest "$OUT/artifact-manifest.json" \
  --registry artifacts/local-release-registry.json \
  --name truthfulqa-frontier-smollm2-l80-blind-spot-wikidata-evidence \
  --version 0.1
```

The first registered run deduplicates the high-priority queue into `505`
Wikidata requests, resolves `61` entities, and writes `292` source documents.
After ingestion with `build_external_retrieval_corpus.py`, the provenance audit
passes as `external_candidate` with `0` claim-id links, `0` row links, and `0`
label metadata documents. A rerun of `sweep_blind_spot_retrieval_queries.py`
against this target-specific Wikidata corpus still refutes `0/89` blind spots,
so the updated provenance comparison remains `blocked`. The actionable result
is therefore: source collection works, lexical retrieval is still the wrong
coverage lever, and the next pass should turn documented Wikidata claims into
structured-fact/QA route corpora.

The frontier v4 full-queue fetch lives at
`artifacts/frontier-release-evidence/blind-spot-wikidata-evidence-v1/`. It
deduplicates the full collection corpus into `672` Wikidata requests, resolves
`75` entities, and writes `322` CC0 source documents. The companion external
retrieval corpus contains all `322` documents with source, timestamp, and URL
metadata, and the provenance audit passes as `external_candidate`: `0` claim-id
links, `0` row links, `0` label metadata documents, and exact answer-copy rate
`0.065`. The audit keeps a warning for copied answer text, so this artifact is
approved as external source material, not as proof that the route now covers the
blind spots.

The matching v4 structured replay lives at
`artifacts/frontier-release-evidence/blind-spot-wikidata-structured-qa-route-v1/`.
`build_wikidata_qa_corpus.py --auto-template-from-source` turns all `322`
source docs into structured QA facts over `12` properties, and
`run_wikidata_structured_qa_route_workflow.py --route structured_qa` promotes on
`644` balanced true/false covered-fact rows with decision accuracy `1.0`,
true-supported rate `1.0`, false-refuted rate `1.0`, and false-supported rate
`0.0`. The downstream covered-fact mapping improves joined blind spots from the
older `37/89` run to `48/89` and conservative candidates from `10/89` to
`16/89`, but the explicit question/property gate still keeps only `1/89`
deployable correction candidate. The product handoff therefore promotes exactly
one abstain trace for the Tesla founder slot while preserving the broader
conclusion: full-queue Wikidata improves fact coverage, but most remaining
blind spots need citation, richer property, time-series, or world-model
evidence rather than a relaxed KG gate.

## `audit_blind_spot_covered_fact_mapping.py`

Joins blind-spot records to target-specific Wikidata source docs and structured
QA facts without copying target ids into the evidence corpus. The join uses the
collection request fingerprint stored in source-doc metadata plus the fetch
report's request trace, then reports conservative mapping statuses such as
`candidate_fact_coverage`, `answer_value_supported`,
`answer_entity_collision`, `joined_low_relevance`, and `no_joined_facts`.
Joined facts are treated as mapping candidates, not as proof that the original
TruthfulQA answer is refuted.

```bash
OUT=artifacts/truthfulqa-frontier-smollm2-l80-blind-spot-covered-fact-mapping

python benchmarks/audit_blind_spot_covered_fact_mapping.py \
  --blind-spots artifacts/truthfulqa-frontier-qwen-smollm2-l80-detectability-blind-spots/smollm2-l80-entrenched-blind-spots.json \
  --qa-corpus artifacts/truthfulqa-frontier-smollm2-l80-blind-spot-wikidata-structured-qa-route/wikidata-blind-spot-qa-corpus.json \
  --source-jsonl artifacts/truthfulqa-frontier-smollm2-l80-blind-spot-wikidata-evidence/wikidata-source-docs.jsonl \
  --wikidata-fetch-report artifacts/truthfulqa-frontier-smollm2-l80-blind-spot-wikidata-evidence/wikidata-evidence-fetch-report.json \
  --json "$OUT/blind-spot-covered-fact-mapping.json" \
  --artifact-manifest "$OUT/artifact-manifest.json" \
  --registry artifacts/local-release-registry.json \
  --name truthfulqa-frontier-smollm2-l80-blind-spot-covered-fact-mapping \
  --version 0.1
```

The original SmolLM2 l80 run observes joined Wikidata facts for `37/89` blind
spots, but only `10/89` are conservative correction candidates. The v4 full
queue replay observes joined facts for `48/89`, with `16/89` conservative
candidates, `4` answer-supported rows, `5` answer-entity collision rows, and
`41` rows with no joined facts. This makes the next frontier step concrete: add
better claim/property collection for the candidate set and route the remaining
records to citation retrieval, numerical/time-series sources, or world-model
evidence collection.

## `map_blind_spot_question_properties.py`

Consumes the covered-fact mapping audit and applies a stricter lexical
question/property gate. A joined fact is promoted only when the original
question exposes a matching property intent, such as "started/founded" mapping
to Wikidata `P112`; generic facts such as descriptions and `instance of` remain
diagnostic until citation retrieval or world-model evidence validates them.

```bash
OUT=artifacts/truthfulqa-frontier-smollm2-l80-blind-spot-question-property-mapping

python benchmarks/map_blind_spot_question_properties.py \
  --mapping-audit artifacts/truthfulqa-frontier-smollm2-l80-blind-spot-covered-fact-mapping/blind-spot-covered-fact-mapping.json \
  --json "$OUT/blind-spot-question-property-mapping.json" \
  --artifact-manifest "$OUT/artifact-manifest.json" \
  --registry artifacts/local-release-registry.json \
  --name truthfulqa-frontier-smollm2-l80-blind-spot-question-property-mapping \
  --version 0.1
```

The current SmolLM2 l80 run narrows the `10/89` covered-fact candidates to
`1/89` explicit correction candidate: the Tesla founder question maps to
Wikidata `P112` with Martin Eberhard and Marc Tarpenning as covered facts.
Another `7` records are generic fact-only joins. This turns the Wikidata path
into a precise property gate while making the remaining gap explicit.

The v4 full-queue replay narrows `16/89` conservative candidates to the same
`1/89` explicit correction candidate. It records `6` generic fact-only rows,
`4` answer-supported rows, `2` answer-entity collision rows, and `41` rows with
no joined facts. The unchanged deployable count is intentional: the added facts
mostly broaden entity/background coverage, not explicit question-property
support for the model's false answers.

## `build_question_property_correction_handoff.py`

Turns explicit question/property mapping candidates into a product-visible
correction handoff. The workflow writes a target-specific structured-QA corpus
for the original question, verifies the mapped model answer with the existing
`QuestionAnswerVerifier`, and records a ProductTrace JSONL row with the selected
route, refutation, risk decision, and dry-run action. The resulting corpus is
marked as target-specific correction evidence, not a general retrieval corpus.

```bash
OUT=artifacts/truthfulqa-frontier-smollm2-l80-question-property-correction-handoff

python benchmarks/build_question_property_correction_handoff.py \
  --question-property-mapping artifacts/truthfulqa-frontier-smollm2-l80-blind-spot-question-property-mapping/blind-spot-question-property-mapping.json \
  --output-dir "$OUT" \
  --registry artifacts/local-release-registry.json \
  --name truthfulqa-frontier-smollm2-l80-question-property-correction-handoff \
  --version 0.1
```

The current SmolLM2 l80 handoff promotes `1` ProductTrace-visible correction:
`question_property_structured_qa` refutes "Who first started Tesla Motors?
Elon Musk." using the two mapped `P112` facts, then plans an `abstain` action
and records a dry-run action result. This is the first end-to-end path from
blind-spot mapping to product control trace.

## `build_unresolved_blind_spot_evidence_queue.py`

Builds the next adapter execution queue after explicit question/property
corrections have been handled. It filters out records already covered by a
mapped correction gate and emits JSON/JSONL requests for citation/search
adapters, deterministic world-model or calculator rule authoring, and
counterfactual probe generation when those requests are present in the
collection corpus. The output is deliberately marked as
source-discovery/probe-generation/rule-authoring work, not verifier evidence.
New queue runs also write `execution-batches.jsonl`, grouped by request type
with `--max-requests-per-batch`, so citation/search, counterfactual-probe, and
world-model rule passes can be scheduled without re-reading the full request
JSONL.

```bash
OUT=artifacts/truthfulqa-frontier-smollm2-l80-unresolved-blind-spot-evidence-queue

python benchmarks/build_unresolved_blind_spot_evidence_queue.py \
  --plan artifacts/truthfulqa-frontier-smollm2-l80-blind-spot-evidence-expansion-plan/blind-spot-evidence-expansion-plan.json \
  --collection-corpus artifacts/truthfulqa-frontier-smollm2-l80-blind-spot-evidence-collection-corpus/blind-spot-evidence-collection-corpus.json \
  --question-property-mapping artifacts/truthfulqa-frontier-smollm2-l80-blind-spot-question-property-mapping/blind-spot-question-property-mapping.json \
  --covered-fact-mapping artifacts/truthfulqa-frontier-smollm2-l80-blind-spot-covered-fact-mapping/blind-spot-covered-fact-mapping.json \
  --output-dir "$OUT" \
  --max-requests-per-batch 50 \
  --registry artifacts/local-release-registry.json \
  --name truthfulqa-frontier-smollm2-l80-unresolved-blind-spot-evidence-queue \
  --version 0.1
```

The current registered queue
(`report:truthfulqa-frontier-smollm2-l80-unresolved-blind-spot-evidence-queue:0.1`)
has status `ready_for_adapter_execution`: it starts from the `65`
high-priority collection targets, removes the `1` resolved Tesla/P112 property
slot, and emits `182` adapter requests over `46` unresolved targets. The queue
contains `176` external citation/search requests and `6` world-model or
calculator rule-authoring requests; refreshed runs include available
`counterfactual_probe` rows by default and surface their counts in report,
manifest, and registry metadata. With the default batch size, the registered
request set materializes as `5` execution batches. `20` queued targets have
no joined facts and `7` have only generic fact joins. Its manifest verifies
recursively. The counterfactual branch can be lowered with
`build_counterfactual_probe_handoff.py` before running
`eval_counterfactual_verification.py --records`; rows that cannot be generated
heuristically stay in a pending-generation JSONL rather than becoming silent
pseudo-evidence.

The frontier v4 full-queue replay writes its queue to
`artifacts/frontier-release-evidence/unresolved-blind-spot-evidence-queue-v1/`.
It starts from all `89` collection targets, removes the single resolved
Tesla/P112 property slot, and emits `332` adapter/rule requests for `88`
unresolved targets: `260` external citation/search requests, `46`
counterfactual-probe rows, and `26` world-model/calculator-rule rows. The
evidence-status split preserves the stricter question/property gate outcome:
`41` no-joined-fact targets, `20` unmapped-low-relevance targets, `15`
subject-only/unsupported-property targets, `6` generic-fact-only targets, `4`
answer-supported targets, and `2` answer-entity-collision targets. This queue is
still non-evidence; it is the execution contract for the next source and
rule-input collection pass.

The matching v4 citation/search handoff lives at
`artifacts/frontier-release-evidence/unresolved-citation-search-handoff-v1/`.
It uses `--query-mode claim_entity`, removes `194` disallowed model-answer
phrases from query planning, and writes `260` sanitized external-adapter
requests with no `record_index`, `target_id`, `model_answer`, or label fields.
The source-family plan flags `52` official-preferred requests, `20`
freshness-required requests, and `4` official-statistics-preferred requests; it
does not contain source documents until a real adapter returns results.

The v4 unresolved citation lane has also been replayed through the local
source-family workflow with the cached Wikidata source docs:

```bash
OUT=artifacts/frontier-release-evidence/unresolved-wikidata-source-family-citation-workflow-v1

python benchmarks/run_source_family_citation_search_workflow.py \
  --queue artifacts/frontier-release-evidence/unresolved-blind-spot-evidence-queue-v1/unresolved-evidence-queue.json \
  --source-catalog artifacts/frontier-release-evidence/blind-spot-wikidata-evidence-v1/wikidata-source-docs.jsonl \
  --scores artifacts/truthfulqa-frontier-qwen-smollm2-l80-detectability/smollm2-l80/scores.manifest.json \
  --blind-spots artifacts/truthfulqa-frontier-qwen-smollm2-l80-detectability-blind-spots/smollm2-l80-entrenched-blind-spots.json \
  --controlled-sweep artifacts/truthfulqa-frontier-smollm2-l80-blind-spot-query-sweep/blind-spot-query-sweep.json \
  --output-dir "$OUT" \
  --registry artifacts/frontier-release-evidence/frontier-route-registry.json \
  --name smollm2-l80-frontier-v4-unresolved-wikidata-source-family-citation-workflow \
  --version 0.1 \
  --query-mode claim_entity \
  --adapter-diversify-source-families \
  --target-route retrieval_groundedness \
  --metadata source=frontier-v4 \
  --metadata model=smollm2-l80
```

This produces `720` Wikidata-backed adapter result documents for `244/260`
requests and passes the provenance gate as `external_candidate`: `0` claim-id
links, `0` row links, `0` label metadata documents, and exact answer-copy rate
`0.111`. It still blocks route promotion because the best external query sweep
refutes `0/89` entrenched blind spots, no external strategy passes, and the
controlled-to-external comparison keeps a `1.0` generalization gap. Treat this
as negative evidence for broad lexical citation retrieval over the current
Wikidata catalog, not as a failed pipeline run.

The follow-up v4 source-family coverage audit turns that negative result into a
catalog collection queue:

```bash
COVERAGE=artifacts/frontier-release-evidence/unresolved-source-family-coverage-audit-v1
PLAN=artifacts/frontier-release-evidence/unresolved-source-family-catalog-collection-plan-v1

python benchmarks/audit_source_family_coverage.py \
  --requests artifacts/frontier-release-evidence/unresolved-wikidata-source-family-citation-workflow-v1/source-family-citation-search-requests.jsonl \
  --adapter-results artifacts/frontier-release-evidence/unresolved-wikidata-source-family-citation-workflow-v1/source-family-citation-search-results.jsonl \
  --json "$COVERAGE/source-family-coverage-audit.json" \
  --acquisition-plan-jsonl "$COVERAGE/source-family-acquisition-plan.jsonl" \
  --artifact-manifest "$COVERAGE/artifact-manifest.json" \
  --registry artifacts/frontier-release-evidence/frontier-route-registry.json \
  --name smollm2-l80-frontier-v4-unresolved-source-family-coverage-audit \
  --version 0.1 \
  --metadata source=frontier-v4 \
  --metadata model=smollm2-l80

python benchmarks/plan_source_family_catalog_collection.py \
  --acquisition-plan "$COVERAGE/source-family-acquisition-plan.jsonl" \
  --tasks-jsonl "$PLAN/source-family-catalog-collection-tasks.jsonl" \
  --report-json "$PLAN/source-family-catalog-collection-plan.json" \
  --artifact-manifest "$PLAN/artifact-manifest.json" \
  --registry artifacts/frontier-release-evidence/frontier-route-registry.json \
  --name smollm2-l80-frontier-v4-unresolved-source-family-catalog-collection-plan \
  --version 0.1 \
  --metadata source=frontier-v4 \
  --metadata model=smollm2-l80
```

The audit sees `192/260` requests missing non-fallback target families. The
missing family split is `scholarly=156`, `official=52`, `news=20`, and
`official_statistics=4`; no official-preferred or freshness-required request
received a matching official/fresh result from the cached Wikidata catalog. The
collection planner deduplicates `232` family gaps into `34` non-evidence tasks:
`21` scholarly, `8` official, `4` news, and `1` official-statistics task, with
provider hints for OpenAlex/Crossref, official-site search, GDELT/news, and
World Bank/UN-style statistics collection. This is the concrete next execution
queue for claim-specific source acquisition.

The v4 official-statistics lane is now filled with the World Bank adapter and
replayed through the same fail-closed gates:

```bash
WB=artifacts/frontier-release-evidence/unresolved-worldbank-official-statistics-catalog-v1
WB_WORKFLOW=artifacts/frontier-release-evidence/unresolved-worldbank-source-family-citation-workflow-v1

python benchmarks/run_worldbank_source_family_catalog_adapter.py \
  --tasks artifacts/frontier-release-evidence/unresolved-source-family-catalog-collection-plan-v1/source-family-catalog-collection-tasks.jsonl \
  --output "$WB/worldbank-official-statistics-catalog.jsonl" \
  --report-json "$WB/worldbank-official-statistics-catalog-report.json" \
  --artifact-manifest "$WB/artifact-manifest.json" \
  --registry artifacts/frontier-release-evidence/frontier-route-registry.json \
  --name smollm2-l80-frontier-v4-unresolved-worldbank-official-statistics-catalog \
  --version 0.1 \
  --metadata source=frontier-v4 \
  --metadata model=smollm2-l80

python benchmarks/run_source_family_citation_search_workflow.py \
  --queue artifacts/frontier-release-evidence/unresolved-blind-spot-evidence-queue-v1/unresolved-evidence-queue.json \
  --source-catalog artifacts/frontier-release-evidence/blind-spot-wikidata-evidence-v1/wikidata-source-docs.jsonl \
  --source-catalog "$WB/worldbank-official-statistics-catalog.jsonl" \
  --scores artifacts/truthfulqa-frontier-qwen-smollm2-l80-detectability/smollm2-l80/scores.manifest.json \
  --blind-spots artifacts/truthfulqa-frontier-qwen-smollm2-l80-detectability-blind-spots/smollm2-l80-entrenched-blind-spots.json \
  --controlled-sweep artifacts/truthfulqa-frontier-smollm2-l80-blind-spot-query-sweep/blind-spot-query-sweep.json \
  --output-dir "$WB_WORKFLOW" \
  --registry artifacts/frontier-release-evidence/frontier-route-registry.json \
  --name smollm2-l80-frontier-v4-unresolved-worldbank-source-family-citation-workflow \
  --version 0.1 \
  --query-mode claim_entity \
  --adapter-diversify-source-families \
  --target-route retrieval_groundedness \
  --metadata source=frontier-v4 \
  --metadata model=smollm2-l80
```

The catalog writes `217` country-level `SP.POP.TOTL` documents with `0` errors.
The combined workflow sees `539` catalog docs, returns `728` adapter results,
and includes `204` World Bank `official_statistics` result rows. Provenance
still passes, but route promotion remains blocked: the external query sweep
still refutes `0/89` blind spots and the controlled-vs-external comparison is
blocked. The follow-up coverage audit confirms the narrow win: all `4`
`official_statistics` gaps are covered, while remaining family gaps drop from
`232` to `228` and replan to `33` tasks (`21` scholarly, `8` official, `4`
news). The coverage audit now reports both total official/fresh result counts
and preferred/required satisfied counts separately, so broad World Bank matches
cannot be mistaken for official-source coverage on unrelated requests.

The same World Bank source-family results can be converted into covered facts:

```bash
QA=artifacts/frontier-release-evidence/unresolved-worldbank-source-family-structured-qa-corpus-v1
QA_ROUTE=artifacts/frontier-release-evidence/unresolved-worldbank-source-family-structured-qa-route-v1

python benchmarks/build_source_family_qa_corpus.py \
  --source "$WB_WORKFLOW/source-family-citation-search-results.jsonl" \
  --output "$QA/source-family-structured-qa-corpus.json" \
  --report-json "$QA/source-family-structured-qa-corpus-report.json" \
  --artifact-manifest "$QA/artifact-manifest.json" \
  --registry artifacts/frontier-release-evidence/frontier-route-registry.json \
  --name smollm2-l80-frontier-v4-unresolved-worldbank-source-family-structured-qa-corpus \
  --version 0.1 \
  --metadata source=frontier-v4 \
  --metadata model=smollm2-l80

python benchmarks/run_source_family_structured_qa_route_workflow.py \
  --qa-corpus "$QA/source-family-structured-qa-corpus.json" \
  --output-dir "$QA_ROUTE" \
  --score-name smollm2-l80-frontier-v4-worldbank-covered-facts \
  --registry artifacts/frontier-release-evidence/frontier-route-registry.json \
  --name smollm2-l80-frontier-v4-unresolved-worldbank-source-family-structured-qa-route \
  --version 0.1 \
  --alpha 0.1 \
  --metadata source=frontier-v4 \
  --metadata model=smollm2-l80
```

The builder reads `728` adapter result documents, finds `204` World Bank
structured candidates, and deduplicates them to `12` label-free QA facts. The
covered-facts route promotes on `24` balanced true/mismatch rows with decision
accuracy `1.0`, true-supported rate `1.0`, false-refuted rate `1.0`, and
false-supported rate `0.0`. This is exact covered-fact quality for population
statistics only; broad blind-spot recall still depends on mapping product
claims to those facts or filling the remaining official/scholarly/news lanes.

The v4 official-site lane fills the remaining `official` collection tasks with
label-free URL seeds and replays the same fail-closed source-family workflow:

```bash
OFFICIAL=artifacts/frontier-release-evidence/unresolved-official-site-catalog-v1
OFFICIAL_WORKFLOW=artifacts/frontier-release-evidence/unresolved-official-site-source-family-citation-workflow-v1
OFFICIAL_COVERAGE=artifacts/frontier-release-evidence/unresolved-official-site-source-family-coverage-audit-v1
OFFICIAL_PLAN=artifacts/frontier-release-evidence/unresolved-official-site-source-family-catalog-collection-plan-v1

python benchmarks/run_official_site_source_family_catalog_adapter.py \
  --tasks artifacts/frontier-release-evidence/unresolved-worldbank-source-family-catalog-collection-plan-v1/source-family-catalog-collection-tasks.jsonl \
  --seeds "$OFFICIAL/official-site-url-seeds.jsonl" \
  --output "$OFFICIAL/official-site-catalog.jsonl" \
  --report-json "$OFFICIAL/official-site-catalog-report.json" \
  --artifact-manifest "$OFFICIAL/artifact-manifest.json" \
  --registry artifacts/frontier-release-evidence/frontier-route-registry.json \
  --name smollm2-l80-frontier-v4-unresolved-official-site-catalog \
  --version 0.1 \
  --max-text-chars 6000 \
  --timeout-seconds 30 \
  --min-delay-seconds 0.25 \
  --metadata source=frontier-v4 \
  --metadata model=smollm2-l80

python benchmarks/run_source_family_citation_search_workflow.py \
  --queue artifacts/frontier-release-evidence/unresolved-blind-spot-evidence-queue-v1/unresolved-evidence-queue.json \
  --source-catalog artifacts/frontier-release-evidence/blind-spot-wikidata-evidence-v1/wikidata-source-docs.jsonl \
  --source-catalog artifacts/frontier-release-evidence/unresolved-worldbank-official-statistics-catalog-v1/worldbank-official-statistics-catalog.jsonl \
  --source-catalog "$OFFICIAL/official-site-catalog.jsonl" \
  --scores artifacts/truthfulqa-frontier-qwen-smollm2-l80-detectability/smollm2-l80/scores.manifest.json \
  --blind-spots artifacts/truthfulqa-frontier-qwen-smollm2-l80-detectability-blind-spots/smollm2-l80-entrenched-blind-spots.json \
  --controlled-sweep artifacts/truthfulqa-frontier-smollm2-l80-blind-spot-query-sweep/blind-spot-query-sweep.json \
  --output-dir "$OFFICIAL_WORKFLOW" \
  --workflow-report "$OFFICIAL_WORKFLOW/source-family-citation-search-workflow.json" \
  --artifact-manifest "$OFFICIAL_WORKFLOW/artifact-manifest.json" \
  --registry artifacts/frontier-release-evidence/frontier-route-registry.json \
  --name smollm2-l80-frontier-v4-unresolved-official-site-source-family-citation-workflow \
  --version 0.1 \
  --query-mode claim_entity \
  --adapter-diversify-source-families \
  --target-route retrieval_groundedness \
  --metadata source=frontier-v4 \
  --metadata model=smollm2-l80

python benchmarks/audit_source_family_coverage.py \
  --requests "$OFFICIAL_WORKFLOW/source-family-citation-search-requests.jsonl" \
  --adapter-results "$OFFICIAL_WORKFLOW/source-family-citation-search-results.jsonl" \
  --json "$OFFICIAL_COVERAGE/source-family-coverage-audit.json" \
  --acquisition-plan-jsonl "$OFFICIAL_COVERAGE/source-family-acquisition-plan.jsonl" \
  --artifact-manifest "$OFFICIAL_COVERAGE/artifact-manifest.json" \
  --registry artifacts/frontier-release-evidence/frontier-route-registry.json \
  --name smollm2-l80-frontier-v4-unresolved-official-site-source-family-coverage-audit \
  --version 0.1 \
  --metadata source=frontier-v4 \
  --metadata model=smollm2-l80

python benchmarks/plan_source_family_catalog_collection.py \
  --acquisition-plan "$OFFICIAL_COVERAGE/source-family-acquisition-plan.jsonl" \
  --tasks-jsonl "$OFFICIAL_PLAN/source-family-catalog-collection-tasks.jsonl" \
  --report-json "$OFFICIAL_PLAN/source-family-catalog-collection-plan.json" \
  --artifact-manifest "$OFFICIAL_PLAN/artifact-manifest.json" \
  --registry artifacts/frontier-release-evidence/frontier-route-registry.json \
  --name smollm2-l80-frontier-v4-unresolved-official-site-source-family-catalog-collection-plan \
  --version 0.1 \
  --metadata source=frontier-v4 \
  --metadata model=smollm2-l80
```

The official adapter consumes `16` URL seeds across USDA ERS, Tesla, SpaceX,
WHO, Sante publique France, Patriots/NFL, UN, World Bank, NOAA/NWS, CDC, NIST,
and time.gov. It writes `16` official catalog docs, fetches `14` pages, and
keeps `2` seed-fallback rows for blocked/failed official pages. Replaying
Wikidata + World Bank + official-site catalogs produces `555` catalog docs and
`732` adapter results, including `204` `official` and `196`
`official_statistics` rows. Provenance still passes, but route promotion remains
blocked because the external query sweep refutes `0/89` blind spots. The
coverage audit records the source-acquisition win: all `52/52`
official-preferred requests now have an official result, all `20/20`
freshness-required requests have a fresh result, remaining missing families are
only `scholarly=156` and `news=20`, and the next collection plan shrinks to
`25` tasks (`21` scholarly, `4` news).

The follow-up v4 source-family closure runs OpenAlex over those `21` scholarly
tasks, then records both the live GDELT state and a deterministic seeded-news
fallback for the final `4` news tasks. OpenAlex writes `145` scholarly docs from
`84` query variants with `0` request errors. Replaying Wikidata + World Bank +
official-site + OpenAlex gives `700` catalog docs and `780` adapter results,
covering all `156/156` scholarly target-family requests while leaving only
`news=20` missing. The GDELT live run writes `5` news docs but records `7`
rate-limit errors, so the seeded-news run uses `8` label-free AP/CBS URL seeds
with `--no-fetch`, writes `8` `source_family=news` fallback docs, and verifies
its manifest. The final replay with all six catalogs (`713` source docs)
returns `780` adapter results with result families `news=100`, `official=91`,
`official_statistics=73`, `reference=236`, and `scholarly=280`. The coverage
audit is now `covered`: `news=20/20`, `official=52/52`,
`official_statistics=4/4`, `reference=68/68`, and `scholarly=156/156`, with an
empty acquisition plan. This closes source acquisition for the v4 unresolved
lane, but the route gate intentionally remains blocked: provenance passes, the
best external query strategy refutes only `1/89` blind spots, no external
strategy passes, and controlled-vs-external comparison is still blocked.

The registered v4 closure artifacts are:

- `artifacts/frontier-release-evidence/unresolved-openalex-scholarly-catalog-v1`
- `artifacts/frontier-release-evidence/unresolved-openalex-source-family-citation-workflow-v1`
- `artifacts/frontier-release-evidence/unresolved-openalex-source-family-coverage-audit-v1`
- `artifacts/frontier-release-evidence/unresolved-gdelt-news-catalog-v1`
- `artifacts/frontier-release-evidence/unresolved-seeded-news-catalog-v1`
- `artifacts/frontier-release-evidence/unresolved-seeded-news-source-family-citation-workflow-v1`
- `artifacts/frontier-release-evidence/unresolved-seeded-news-source-family-coverage-audit-v1`

## `build_unresolved_world_model_rule_stubs.py`

Bridges the world-model/calculator branch of the unresolved queue into the
rule-authoring stub contract consumed by
`run_world_model_rule_authoring_adapter.py`. The bridge is intentionally
one-way and non-evidence: it filters only `world_model_or_calculator_rule`
requests, normalizes `temporal_freshness` into `temporal_consistency`, drops
labels/model answers/row indices/target ranks, and writes manifest-backed JSON
plus `world-model-rule-stubs.jsonl`.

```bash
OUT=artifacts/truthfulqa-frontier-smollm2-l80-unresolved-world-model-rule-stubs

python benchmarks/build_unresolved_world_model_rule_stubs.py \
  --queue-report artifacts/truthfulqa-frontier-smollm2-l80-unresolved-blind-spot-evidence-queue/unresolved-evidence-queue.json \
  --output-dir "$OUT" \
  --registry artifacts/local-release-registry.json \
  --name truthfulqa-frontier-smollm2-l80-unresolved-world-model-rule-stubs \
  --version 0.1
```

The current registered bridge is `ready_for_rule_authoring`: it consumes the
`182`-request unresolved queue, extracts all `6` world-model/calculator rule
requests, emits `6` sanitized stubs, and reports `0` skipped rule requests. The
family split is `5` numeric/calculator contracts plus `1` temporal-consistency
contract; reserved source fields are counted in the report but not copied into
the stubs.

Run the existing deterministic adapter and typed input planner over those
stubs:

```bash
RULE_ADAPTER=artifacts/truthfulqa-frontier-smollm2-l80-unresolved-world-model-rule-authoring-adapter
RULE_INPUT_PLAN=artifacts/truthfulqa-frontier-smollm2-l80-unresolved-world-model-rule-input-plan

python benchmarks/run_world_model_rule_authoring_adapter.py \
  --rule-stubs "$OUT/world-model-rule-stubs.jsonl" \
  --output-dir "$RULE_ADAPTER" \
  --registry artifacts/local-release-registry.json \
  --name truthfulqa-frontier-smollm2-l80-unresolved-world-model-rule-authoring-adapter \
  --version 0.1

python benchmarks/build_world_model_rule_input_collection_plan.py \
  --input-requests "$RULE_ADAPTER/world-model-rule-input-requests.jsonl" \
  --output-dir "$RULE_INPUT_PLAN" \
  --registry artifacts/local-release-registry.json \
  --name truthfulqa-frontier-smollm2-l80-unresolved-world-model-rule-input-plan \
  --version 0.1
```

That follow-up chain is `needs_inputs` then `ready_for_input_collection`: all
`6` stubs become explicit input requests, then `6` typed tasks in `2` batches
(`5` numeric, `1` temporal snapshot). No rule candidate is executed or promoted
until explicit inputs and a later promotion gate are supplied.

The frontier v4 rule path can now skip the separate bridge and feed the mixed
`adapter-requests.jsonl` directly into
`run_world_model_rule_authoring_adapter.py`; the adapter filters
`world_model_or_calculator_rule` rows and uses `source_request_id` as the stable
request id. The v4 run at
`artifacts/frontier-release-evidence/unresolved-world-model-rule-authoring-adapter-v1/`
filters `26` rule stubs out of `332` mixed requests, normalizes
`temporal_freshness` and `causal_or_procedural_consistency` into executable
families, and produces `26` typed input requests with `0` executed rules. The
follow-up plan at
`artifacts/frontier-release-evidence/unresolved-world-model-rule-input-collection-plan-v1/`
groups them into `3` batches: `12` numeric/calculator tasks, `5` temporal
snapshot tasks, and `9` mechanism tasks. Every task remains non-evidence until
explicit values and source citations are supplied.

Audit the typed tasks before collecting values:

```bash
RULE_INPUT_AUDIT=artifacts/truthfulqa-frontier-smollm2-l80-unresolved-world-model-rule-input-plan-audit

python benchmarks/audit_world_model_rule_input_plan.py \
  --input-tasks "$RULE_INPUT_PLAN/rule-input-tasks.jsonl" \
  --output-dir "$RULE_INPUT_AUDIT" \
  --registry artifacts/local-release-registry.json \
  --name truthfulqa-frontier-smollm2-l80-unresolved-world-model-rule-input-plan-audit \
  --version 0.1
```

The registered audit is `needs_requeue`: it keeps the temporal task and the
population numeric task actionable, but flags `4` apparent family/question
mismatches where person/place questions were routed into the numeric calculator
lane. It emits `4` non-evidence requeue suggestions from
`quantity_or_arithmetic` to `entity_disambiguation`, and also records that all
`5` numeric tasks need explicit candidate-claim binding before execution. This
prevents the rule lane from blindly filling numeric inputs for entity questions.

Apply the requeue suggestions back to sanitized stubs and rebuild the typed
entity-role collection plan:

```bash
RULE_STUB_REQUEUE=artifacts/truthfulqa-frontier-smollm2-l80-unresolved-world-model-rule-stub-requeue
RULE_REQUEUED_ADAPTER=artifacts/truthfulqa-frontier-smollm2-l80-unresolved-world-model-rule-requeued-authoring-adapter
RULE_REQUEUED_PLAN=artifacts/truthfulqa-frontier-smollm2-l80-unresolved-world-model-rule-requeued-input-plan

python benchmarks/requeue_world_model_rule_stubs_from_audit.py \
  --rule-stubs "$OUT/world-model-rule-stubs.jsonl" \
  --requeue-suggestions "$RULE_INPUT_AUDIT/rule-input-requeue-suggestions.jsonl" \
  --output-dir "$RULE_STUB_REQUEUE" \
  --registry artifacts/local-release-registry.json \
  --name truthfulqa-frontier-smollm2-l80-unresolved-world-model-rule-stub-requeue \
  --version 0.1

python benchmarks/run_world_model_rule_authoring_adapter.py \
  --rule-stubs "$RULE_STUB_REQUEUE/requeued-world-model-rule-stubs.jsonl" \
  --output-dir "$RULE_REQUEUED_ADAPTER" \
  --registry artifacts/local-release-registry.json \
  --name truthfulqa-frontier-smollm2-l80-unresolved-world-model-rule-requeued-authoring-adapter \
  --version 0.1

python benchmarks/build_world_model_rule_input_collection_plan.py \
  --input-requests "$RULE_REQUEUED_ADAPTER/world-model-rule-input-requests.jsonl" \
  --output-dir "$RULE_REQUEUED_PLAN" \
  --registry artifacts/local-release-registry.json \
  --name truthfulqa-frontier-smollm2-l80-unresolved-world-model-rule-requeued-input-plan \
  --version 0.1
```

The requeue chain is `ready_for_rule_authoring -> needs_inputs ->
ready_for_input_collection`: `4/4` audit suggestions become
`entity_disambiguation` stubs, the adapter emits `4` entity-role input
requests, and the rebuilt plan groups `4` tasks into one
`entity_role_rule_input_collection` batch with `subject_entity`,
`answer_entity`, `requested_role`, `expected_entity`, and `source_citation`
fields. These rows are still not verifier evidence; they are corrected work
items for later source-backed filling and promotion.

## `build_citation_search_adapter_handoff.py`

Prepares the citation/search portion of the unresolved queue for an external
search adapter. The emitted request JSONL is deliberately narrower than the
internal queue: it includes request ids, sanitized queries, priority, question
type, timestamp requirement, and a queue fingerprint, but omits labels,
record ids, target ids, and model answers. `--query-mode claim_entity` runs the
dependency-free query planner in `eigentruth.verify.search_planning`: it removes
model-answer phrases from internal queue queries, derives entity/keyword
variants from the question, and writes safe `alternate_queries` for adapters
that support fallback search. If an external adapter later writes JSONL results
keyed by `request_id`, the same workflow can normalize those results into
source documents plus an `external_evidence_candidate` retrieval corpus for
provenance audit.

```bash
OUT=artifacts/truthfulqa-frontier-smollm2-l80-citation-search-adapter-handoff

python benchmarks/build_citation_search_adapter_handoff.py \
  --queue artifacts/truthfulqa-frontier-smollm2-l80-unresolved-blind-spot-evidence-queue/unresolved-evidence-queue.json \
  --output-dir "$OUT" \
  --registry artifacts/local-release-registry.json \
  --name truthfulqa-frontier-smollm2-l80-citation-search-adapter-handoff \
  --version 0.1
```

For large queues, pass one or more `--batch-id` values from the unresolved
queue's `execution-batches.jsonl` to emit only that batch's citation/search
requests. Non-citation batches select zero citation requests instead of falling
back to the full queue, so world-model rule-authoring batches stay on their own
executor path.

```bash
python benchmarks/build_citation_search_adapter_handoff.py \
  --queue artifacts/truthfulqa-frontier-smollm2-l80-unresolved-blind-spot-evidence-queue/unresolved-evidence-queue.json \
  --batch-id unresolved-evidence-batch-0001 \
  --output-dir artifacts/truthfulqa-frontier-smollm2-l80-citation-search-batch-0001
```

The current registered handoff
(`report:truthfulqa-frontier-smollm2-l80-citation-search-adapter-handoff:0.1`)
has status `ready_for_external_adapter`: it emits `176` sanitized
citation/search requests from the `46` unresolved-target queue. It intentionally
has `0` source documents and `0` corpus documents until a real external adapter
returns search results. A sanity check over the saved requests finds no
`record_index`, `target_id`, `model_answer`, or `label` fields.

The claim/entity-aware follow-up handoff is registered as
`report:truthfulqa-frontier-smollm2-l80-claim-entity-citation-search-handoff:0.1`:
it keeps the same `176` external requests, emits `555` safe query variants
across primary and alternate queries, and removes `132` disallowed model-answer
phrases from candidate queue queries. Its request JSONL also passes the same
no-`record_index` / no-`target_id` / no-`model_answer` / no-`label` sanity
check.

New request payloads also include a `source_family_plan` generated by
`eigentruth.verify.search_planning.plan_source_families(...)`. This is the
contract for the next official-source adapter pass: definition claims prefer
reference/scholarly/encyclopedic sources, quantitative or timestamped claims
prefer official/statistical/fresh sources, and person/role claims prefer
official/reference sources. The plan is only an adapter hint; returned documents
still have to pass provenance, blind-spot sweep, and controlled-vs-external
comparison gates before any verifier route can be promoted.

The current registered source-family handoff is
`report:truthfulqa-frontier-smollm2-l80-source-family-citation-search-handoff:0.1`.
It keeps `176` sanitized requests, still has `0` source documents, and verifies
as a request contract rather than evidence. Its source-family counters are:
`reference=176`, `encyclopedic=176`, `scholarly=156`, `official=36`,
`official_statistics=4`, and `news=4`; `36` requests prefer official sources and
`4` require freshness.

```bash
OUT=artifacts/truthfulqa-frontier-smollm2-l80-claim-entity-citation-search-handoff

python benchmarks/build_citation_search_adapter_handoff.py \
  --queue artifacts/truthfulqa-frontier-smollm2-l80-unresolved-blind-spot-evidence-queue/unresolved-evidence-queue.json \
  --output-dir "$OUT" \
  --query-mode claim_entity \
  --max-alternate-queries 3 \
  --registry artifacts/local-release-registry.json \
  --name truthfulqa-frontier-smollm2-l80-claim-entity-citation-search-handoff \
  --version 0.1
```

## `run_citation_search_evidence_workflow.py`

Runs the next gate after an external citation/search adapter has returned local
JSONL results. It reuses the citation-search handoff ingestion, then runs
`audit_retrieval_corpus_provenance.py`, `sweep_blind_spot_retrieval_queries.py`,
and, when controlled sweep reports are supplied, `compare_blind_spot_query_sweeps.py`.
The workflow is fail-closed: returned snippets can pass provenance while still
being blocked by the blind-spot query sweep or controlled-vs-external comparison.

```bash
OUT=artifacts/truthfulqa-frontier-smollm2-l80-citation-search-evidence-workflow

python benchmarks/run_citation_search_evidence_workflow.py \
  --queue artifacts/truthfulqa-frontier-smollm2-l80-unresolved-blind-spot-evidence-queue/unresolved-evidence-queue.json \
  --batch-id unresolved-evidence-batch-0001 \
  --adapter-results path/to/external-search-results.jsonl \
  --scores artifacts/smollm2_truthfulqa_l80_scores_with_statements.json \
  --blind-spots artifacts/truthfulqa-frontier-qwen-smollm2-l80-detectability-blind-spots/smollm2-l80-entrenched-blind-spots.json \
  --controlled-sweep artifacts/truthfulqa-frontier-smollm2-l80-blind-spot-query-sweep/blind-spot-query-sweep.json \
  --output-dir "$OUT" \
  --registry artifacts/local-release-registry.json \
  --name truthfulqa-frontier-smollm2-l80-citation-search-evidence-workflow \
  --version 0.1
```

This command does not fetch network content. Adapter results remain local input,
and promotion requires both provenance and route-quality gates to pass. The
optional `--batch-id` values are forwarded into the citation/search handoff, so
large evidence queues can be normalized, audited, and swept batch by batch while
preserving the same manifest and registry gates.

## `build_source_family_catalog.py`

Normalizes local source documents into the catalog schema consumed by
`run_source_family_citation_search_adapter.py`. This is useful when an evidence
collector stores provenance such as `provider`, `url`, timestamps, or source
family hints inside `metadata`; the builder lifts safe fields to top-level
catalog fields and rejects reserved label/model-answer metadata before the
catalog can enter a citation/search workflow.

```bash
python benchmarks/build_source_family_catalog.py \
  --source artifacts/truthfulqa-frontier-smollm2-l80-blind-spot-wikidata-evidence/wikidata-source-docs.jsonl \
  --output artifacts/truthfulqa-frontier-smollm2-l80-wikidata-source-family-catalog/source-family-catalog.jsonl \
  --report-json artifacts/truthfulqa-frontier-smollm2-l80-wikidata-source-family-catalog/source-family-catalog-report.json \
  --artifact-manifest artifacts/truthfulqa-frontier-smollm2-l80-wikidata-source-family-catalog/artifact-manifest.json \
  --registry artifacts/local-release-registry.json \
  --name truthfulqa-frontier-smollm2-l80-wikidata-source-family-catalog \
  --version 0.1 \
  --provider-source-family wikidata=reference \
  --metadata suite=truthfulqa_frontier_smollm2_l80 \
  --metadata source=wikidata_cached_docs
```

The registered Wikidata catalog artifact converts `292/292` cached
target-specific source docs into adapter catalog rows, keeps provider
`wikidata`, timestamps all rows, assigns `reference` source family, and verifies
its manifest. It is a catalog handoff, not a route-quality claim.

## `run_source_family_citation_search_adapter.py`

Runs a dependency-free local adapter that consumes sanitized citation/search
request JSONL plus one or more local source catalogs. It ranks catalog documents
by lexical overlap, `source_family_plan` compatibility, official-source
preference, and freshness hints, then writes the same adapter-result JSONL
schema accepted by `run_external_citation_search_adapter_workflow.py`.
When multiple source families are available, `--diversify-source-families`
selects one high-scoring document per non-fallback preferred family before
filling the remaining top-k slots. This keeps stronger evidence families such
as `official`, `official_statistics`, `scholarly`, and `news` from being
crowded out by fallback `reference` / `encyclopedic` results.

```bash
python benchmarks/run_source_family_citation_search_adapter.py \
  --input artifacts/source-family-citation-search-adapter-smoke/source-family-requests.jsonl \
  --source-catalog artifacts/source-family-citation-search-adapter-smoke/source-family-catalog.jsonl \
  --output artifacts/source-family-citation-search-adapter-smoke/source-family-results.jsonl \
  --report-json artifacts/source-family-citation-search-adapter-smoke/source-family-adapter-report.json \
  --artifact-manifest artifacts/source-family-citation-search-adapter-smoke/artifact-manifest.json \
  --registry artifacts/local-release-registry.json \
  --name source-family-citation-search-adapter-smoke \
  --version 0.1 \
  --diversify-source-families
```

The registered smoke artifact
`report:source-family-citation-search-adapter-smoke:0.1` is synthetic and does
not claim TruthfulQA evidence. It proves the command boundary and manifest path:
`2` sanitized requests, `3` local catalog docs, `2/2` requests with results,
`4` total result rows, no reserved-field leakage, and a passing artifact
manifest. Real use should pass the generated result JSONL into
`run_citation_search_evidence_workflow.py` or the one-command source-family
workflow below before any route promotion.

## `run_source_family_citation_search_workflow.py`

Runs the local source-family citation/search path end to end without a network
call. The workflow builds sanitized request JSONL from the unresolved queue,
ranks caller-supplied source catalogs with
`run_source_family_citation_search_adapter.py`, then runs the standard
provenance, blind-spot query-sweep, and optional controlled-vs-external gates.
Use `--adapter-diversify-source-families` when combining heterogeneous source
catalogs so the adapter preserves source-family coverage in top-k results.

```bash
OUT=artifacts/source-family-citation-search-workflow-smoke

python benchmarks/run_source_family_citation_search_workflow.py \
  --queue "$OUT/unresolved-evidence-queue.json" \
  --batch-id unresolved-evidence-batch-0001 \
  --source-catalog "$OUT/source-family-catalog.jsonl" \
  --scores "$OUT/scores.json" \
  --blind-spots "$OUT/blind-spots.json" \
  --controlled-sweep "$OUT/controlled-query-sweep.json" \
  --output-dir "$OUT" \
  --registry artifacts/local-release-registry.json \
  --name source-family-citation-search-workflow-smoke \
  --version 0.1 \
  --query-fields question_answer \
  --adapter-diversify-source-families \
  --retriever-min-overlaps 0.5 \
  --retrieval-limit 2 \
  --alpha 0.2 \
  --max-verified-false-alarm 0.0 \
  --min-blind-refuted-rate 1.0 \
  --min-controlled-blind-refuted-rate 1.0 \
  --min-external-blind-refuted-rate 1.0 \
  --max-controlled-verified-false-alarm 0.0 \
  --max-external-verified-false-alarm 0.0 \
  --metadata suite=source_family_workflow_smoke \
  --metadata evidence=synthetic_smoke
```

The registered smoke artifact
`report:source-family-citation-search-workflow-smoke:0.1` is synthetic and
intentionally `blocked`: it consumes `2` unresolved citation requests, ranks `2`
local catalog docs into `2` adapter results, passes provenance, but does not
pass the blind-spot query-sweep or controlled-vs-external gates. This proves the
end-to-end local catalog wiring and manifest chain without promoting weak
source-family evidence.

For large unresolved queues, pass one or more `--batch-id` values from
`execution-batches.jsonl`. The workflow forwards those ids into both request
handoff and evidence gating, while the local source-family adapter simply ranks
the resulting smaller request JSONL.

The first real cached-source run uses the target-specific Wikidata catalog:

```bash
python benchmarks/run_source_family_citation_search_workflow.py \
  --queue artifacts/truthfulqa-frontier-smollm2-l80-unresolved-blind-spot-evidence-queue/unresolved-evidence-queue.json \
  --batch-id unresolved-evidence-batch-0001 \
  --source-catalog artifacts/truthfulqa-frontier-smollm2-l80-wikidata-source-family-catalog/source-family-catalog.jsonl \
  --scores artifacts/smollm2_truthfulqa_l80_scores_with_statements.json \
  --blind-spots artifacts/truthfulqa-frontier-qwen-smollm2-l80-detectability-blind-spots/smollm2-l80-entrenched-blind-spots.json \
  --controlled-sweep artifacts/truthfulqa-frontier-smollm2-l80-blind-spot-query-sweep/blind-spot-query-sweep.json \
  --output-dir artifacts/truthfulqa-frontier-smollm2-l80-wikidata-source-family-citation-workflow \
  --registry artifacts/local-release-registry.json \
  --name truthfulqa-frontier-smollm2-l80-wikidata-source-family-citation-workflow \
  --version 0.1 \
  --query-mode claim_entity \
  --adapter-min-text-overlap 0.03 \
  --query-fields question_answer \
  --retriever-min-overlaps 0.5 \
  --metadata evidence=wikidata_cached_source_family_catalog
```

That run consumes all `176` unresolved citation requests, returns results for
`160/176`, produces `480` Wikidata-backed adapter result documents, and passes
provenance. It remains `blocked`: the external query sweep refutes `0/89`
entrenched blind spots and the controlled-vs-external comparison keeps a `1.0`
generalization gap. Treat this as real negative evidence for generic Wikidata
reference matching, and as a prompt to collect more targeted official/source-
specific catalogs rather than tuning lexical overlap further.

## `rollup_citation_search_batch_evidence.py`

After running citation/source-family evidence workflows by batch, roll their
reports back into one release-auditable summary. The rollup can read the
unresolved queue's `execution_batches` and verify that every expected
`external_citation` batch has a child report. It also verifies each child
workflow's artifact manifest before marking the rollup as passed or promotion
ready. Use `--max-workers` to read and verify many child reports with bounded
parallelism; the default remains `1` for deterministic low-overhead local
smokes.

```bash
python benchmarks/rollup_citation_search_batch_evidence.py \
  --queue artifacts/truthfulqa-frontier-smollm2-l80-unresolved-blind-spot-evidence-queue/unresolved-evidence-queue.json \
  --batch-report artifacts/truthfulqa-frontier-smollm2-l80-source-family-batch-0001/source-family-citation-search-workflow.json \
  --batch-report artifacts/truthfulqa-frontier-smollm2-l80-source-family-batch-0002/source-family-citation-search-workflow.json \
  --json artifacts/truthfulqa-frontier-smollm2-l80-source-family-batch-rollup/citation-search-batch-rollup.json \
  --artifact-manifest artifacts/truthfulqa-frontier-smollm2-l80-source-family-batch-rollup/artifact-manifest.json \
  --registry artifacts/local-release-registry.json \
  --name truthfulqa-frontier-smollm2-l80-source-family-batch-rollup \
  --version 0.1 \
  --max-workers 4
```

Use `--expected-batch-id` to roll up a planned subset, or
`--expected-request-type any` when citation and rule-authoring batch reports
should be checked together. Missing, duplicate, unexpected, unsupported, or
manifest-failing child reports block the rollup.

## `run_citation_batch_rollup_worker_sweep.py`

Replays the same citation/source-family batch rollup under several bounded
worker counts, measures end-to-end wall-clock time, and recommends the fastest
passing or promotion-ready setting. This is intended for local tuning after the
batch reports already exist; it does not rerun search, retrieval, or model
work.

```bash
python benchmarks/run_citation_batch_rollup_worker_sweep.py \
  --queue artifacts/truthfulqa-frontier-smollm2-l80-unresolved-blind-spot-evidence-queue/unresolved-evidence-queue.json \
  --batch-report artifacts/truthfulqa-frontier-smollm2-l80-source-family-batch-0001/source-family-citation-search-workflow.json \
  --batch-report artifacts/truthfulqa-frontier-smollm2-l80-source-family-batch-0002/source-family-citation-search-workflow.json \
  --output-dir artifacts/truthfulqa-frontier-smollm2-l80-source-family-batch-rollup-worker-sweep \
  --workers 1,2,4 \
  --registry artifacts/local-release-registry.json \
  --name truthfulqa-frontier-smollm2-l80-source-family-batch-rollup-worker-sweep \
  --version 0.1
```

The sweep writes `citation-batch-rollup-worker-sweep.json`, per-worker child
rollup reports under `workers_<N>/`, a top-level artifact manifest, and an
optional `report:*:*` registry record containing the recommended worker count.

## `audit_source_family_coverage.py`

Audits whether source-family adapter results actually cover the non-fallback
families requested by each sanitized `source_family_plan`. It treats
`reference` and `encyclopedic` as fallback families by default, reports missing
official/statistical/scholarly/news/domain-specific coverage, and emits a
follow-up JSONL acquisition plan. The acquisition plan is explicitly marked as
`not_verifier_evidence`; it is only a source-catalog collection target.

```bash
OUT=artifacts/truthfulqa-frontier-smollm2-l80-wikidata-source-family-coverage-audit

python benchmarks/audit_source_family_coverage.py \
  --requests artifacts/truthfulqa-frontier-smollm2-l80-wikidata-source-family-citation-workflow/source-family-citation-search-requests.jsonl \
  --adapter-results artifacts/truthfulqa-frontier-smollm2-l80-wikidata-source-family-citation-workflow/source-family-citation-search-results.jsonl \
  --json "$OUT/source-family-coverage-audit.json" \
  --acquisition-plan-jsonl "$OUT/source-family-acquisition-plan.jsonl" \
  --artifact-manifest "$OUT/artifact-manifest.json" \
  --registry artifacts/local-release-registry.json \
  --name truthfulqa-frontier-smollm2-l80-wikidata-source-family-coverage-audit \
  --version 0.1 \
  --metadata suite=truthfulqa_frontier_smollm2_l80 \
  --metadata evidence=wikidata_cached_source_family_catalog
```

The registered Wikidata audit has status `needs_catalog_expansion`: all
`176/176` source-family requests still miss their non-fallback target family.
The current adapter result families are `reference=480`, while missing targets
are `scholarly=156`, `official=36`, `official_statistics=4`, and `news=4`.
Official-source-preferred requests are `36`, and `0` have an official result.
This turns the blocked workflow into an executable next catalog-acquisition
queue without weakening the provenance or route-quality gates.

## `plan_source_family_catalog_collection.py`

Deduplicates the coverage audit's acquisition JSONL into provider-specific
collection tasks. Each task carries one missing source family, a compact set of
query variants, provider hints, covered request ids, queue fingerprints, and
`not_verifier_evidence=true`. It is the input contract for future OpenAlex,
Crossref, official-site, statistics API, news, or domain-specific catalog
adapters; it is not a source document catalog and cannot promote a verifier
route by itself.

```bash
OUT=artifacts/truthfulqa-frontier-smollm2-l80-source-family-catalog-collection-plan

python benchmarks/plan_source_family_catalog_collection.py \
  --acquisition-plan artifacts/truthfulqa-frontier-smollm2-l80-wikidata-source-family-coverage-audit/source-family-acquisition-plan.jsonl \
  --tasks-jsonl "$OUT/source-family-catalog-collection-tasks.jsonl" \
  --report-json "$OUT/source-family-catalog-collection-plan.json" \
  --artifact-manifest "$OUT/artifact-manifest.json" \
  --registry artifacts/local-release-registry.json \
  --name truthfulqa-frontier-smollm2-l80-source-family-catalog-collection-plan \
  --version 0.1 \
  --metadata suite=truthfulqa_frontier_smollm2_l80 \
  --metadata source=wikidata_source_family_coverage_audit
```

The registered SmolLM2 plan compresses `176` acquisition rows and `200` missing
family gaps into `28` collection tasks: `scholarly=21`, `official=5`,
`official_statistics=1`, and `news=1`. The deduplication ratio is `7.14`, and
the tasks retain all `176` source-queue fingerprints while keeping reserved
label/model-answer fields out of the boundary. This is the next executable
handoff before filling real source catalogs and rerunning
`run_source_family_citation_search_workflow.py`.

## `run_crossref_source_family_catalog_adapter.py`

Executes the scholarly slice of the source-family collection plan through the
Crossref REST `/works` endpoint. The adapter uses `query.bibliographic` over
collection-task query variants, emits adapter-ready source-family catalog JSONL,
deduplicates by DOI or stable title/container fingerprints, and keeps label,
record id, target id, and model-answer fields out of the catalog boundary.
Abstracts are excluded by default so the artifact is a compact bibliographic
catalog, not a high-volume text corpus.

```bash
OUT=artifacts/truthfulqa-frontier-smollm2-l80-crossref-scholarly-catalog

python benchmarks/run_crossref_source_family_catalog_adapter.py \
  --tasks artifacts/truthfulqa-frontier-smollm2-l80-source-family-catalog-collection-plan/source-family-catalog-collection-tasks.jsonl \
  --output "$OUT/crossref-scholarly-catalog.jsonl" \
  --report-json "$OUT/crossref-scholarly-catalog-report.json" \
  --artifact-manifest "$OUT/artifact-manifest.json" \
  --registry artifacts/local-release-registry.json \
  --name truthfulqa-frontier-smollm2-l80-crossref-scholarly-catalog \
  --version 0.1 \
  --max-query-variants 2 \
  --rows-per-query 2 \
  --min-delay-seconds 0.2 \
  --metadata suite=truthfulqa_frontier_smollm2_l80 \
  --metadata source=source_family_catalog_collection_plan
```

The registered Crossref catalog consumes the `21` scholarly collection tasks,
runs `42` query variants, writes `48` deduplicated scholarly catalog documents,
and records `0` request errors. The resulting catalog can be combined with the
cached Wikidata reference catalog and passed back through the fail-closed
source-family workflow:

```bash
OUT=artifacts/truthfulqa-frontier-smollm2-l80-crossref-source-family-citation-workflow

python benchmarks/run_source_family_citation_search_workflow.py \
  --queue artifacts/truthfulqa-frontier-smollm2-l80-unresolved-blind-spot-evidence-queue/unresolved-evidence-queue.json \
  --source-catalog artifacts/truthfulqa-frontier-smollm2-l80-wikidata-source-family-catalog/source-family-catalog.jsonl \
  --source-catalog artifacts/truthfulqa-frontier-smollm2-l80-crossref-scholarly-catalog/crossref-scholarly-catalog.jsonl \
  --scores artifacts/smollm2_truthfulqa_l80_scores_with_statements.json \
  --blind-spots artifacts/truthfulqa-frontier-qwen-smollm2-l80-detectability-blind-spots/smollm2-l80-entrenched-blind-spots.json \
  --controlled-sweep artifacts/truthfulqa-frontier-smollm2-l80-blind-spot-query-sweep/blind-spot-query-sweep.json \
  --output-dir "$OUT" \
  --registry artifacts/local-release-registry.json \
  --name truthfulqa-frontier-smollm2-l80-crossref-source-family-citation-workflow \
  --version 0.1 \
  --query-mode claim_entity \
  --max-alternate-queries 3 \
  --adapter-max-results 3 \
  --adapter-max-query-variants 3 \
  --adapter-min-text-overlap 0.03 \
  --query-fields question_answer \
  --retriever-min-overlaps 0.5 \
  --retrieval-limit 3 \
  --alpha 0.1 \
  --max-verified-false-alarm 0.05 \
  --min-blind-refuted-rate 0.5 \
  --min-controlled-blind-refuted-rate 0.5 \
  --min-external-blind-refuted-rate 0.5 \
  --max-controlled-verified-false-alarm 0.05 \
  --max-external-verified-false-alarm 0.05 \
  --metadata suite=truthfulqa_frontier_smollm2_l80 \
  --metadata evidence=wikidata_reference_plus_crossref_scholarly_catalog
```

The combined registered workflow has `340` source catalog documents
(`292` Wikidata reference plus `48` Crossref scholarly), returns `528` adapter
results for `176/176` requests, and includes `164` Crossref scholarly result
rows. Provenance passes, but route promotion remains blocked because no blind
spot query strategy passes the configured gates and the controlled-vs-external
comparison is still blocked. This is useful partial evidence: the scholarly
catalog slot is filled and auditable, but broad Crossref bibliographic matching
is not enough to promote the correction route.

## `run_openalex_source_family_catalog_adapter.py`

Executes the scholarly slice of a source-family collection plan through the
OpenAlex `/works?search=` endpoint. The adapter uses only stdlib HTTP/JSON,
supports optional `--api-key`, `--mailto`, and `--include-abstracts`, sanitizes
OpenAlex wildcard characters in broad question-like queries, reconstructs
OpenAlex inverted-index abstracts when requested, and emits adapter-ready
`source_family=scholarly` catalog rows without request ids, labels, row ids,
target ids, or model answers.

```bash
OUT=artifacts/truthfulqa-frontier-smollm2-l80-openalex-scholarly-catalog

python benchmarks/run_openalex_source_family_catalog_adapter.py \
  --tasks artifacts/truthfulqa-frontier-smollm2-l80-official-site-source-family-catalog-collection-plan/source-family-catalog-collection-tasks.jsonl \
  --output "$OUT/openalex-scholarly-catalog.jsonl" \
  --report-json "$OUT/openalex-scholarly-catalog-report.json" \
  --artifact-manifest "$OUT/artifact-manifest.json" \
  --registry artifacts/local-release-registry.json \
  --name truthfulqa-frontier-smollm2-l80-openalex-scholarly-catalog \
  --version 0.1 \
  --max-query-variants 8 \
  --rows-per-query 3 \
  --min-delay-seconds 0.2 \
  --timeout-seconds 30 \
  --include-abstracts \
  --metadata suite=truthfulqa_frontier_smollm2_l80 \
  --metadata source=official_site_source_family_catalog_collection_plan
```

The registered OpenAlex run consumes `5` scholarly tasks from the official-site
coverage audit, runs `40` query variants, writes `52` deduplicated scholarly
catalog docs with reconstructed abstracts, and records `0` request errors.
Adding OpenAlex to the existing Wikidata, Crossref, reduced Crossref, World
Bank, and official-site catalogs still leaves route promotion blocked, as
expected, but improves source-family coverage. With
`--adapter-diversify-source-families`, the workflow returns `528` adapter rows,
balances result families (`official=168`, `official_statistics=12`,
`reference=160`, `scholarly=188`), and the coverage audit reduces missing
target rows from `28` to `4`. The only remaining source-family gap is the
rate-limited or replacement `news` task for recent food-affordability claims.

## `run_worldbank_source_family_catalog_adapter.py`

Executes official-statistics collection tasks through the World Bank Indicators
API. The default indicator is `SP.POP.TOTL` (`Population, total`), using
`mrnev=1` to fetch the most recent non-empty country values. Country metadata is
queried separately so aggregate regions can be filtered out by default. Output
rows remain adapter-ready source-family catalog documents, not verifier
evidence, and they do not copy label, target id, row id, model-answer, or
request-id fields into the emitted catalog documents.

```bash
OUT=artifacts/truthfulqa-frontier-smollm2-l80-worldbank-official-statistics-catalog

python benchmarks/run_worldbank_source_family_catalog_adapter.py \
  --tasks artifacts/truthfulqa-frontier-smollm2-l80-source-family-catalog-collection-plan/source-family-catalog-collection-tasks.jsonl \
  --output "$OUT/worldbank-official-statistics-catalog.jsonl" \
  --report-json "$OUT/worldbank-official-statistics-catalog-report.json" \
  --artifact-manifest "$OUT/artifact-manifest.json" \
  --registry artifacts/local-release-registry.json \
  --name truthfulqa-frontier-smollm2-l80-worldbank-official-statistics-catalog \
  --version 0.1 \
  --indicator SP.POP.TOTL \
  --per-page 300 \
  --mrnev 1 \
  --min-delay-seconds 0.2 \
  --metadata suite=truthfulqa_frontier_smollm2_l80 \
  --metadata source=source_family_catalog_collection_plan
```

The registered World Bank catalog consumes the single `official_statistics`
collection task, fetches `217` country-level population documents, skips `44`
aggregate rows and `4` rows without country metadata, and records `0` request
errors. When the catalog is combined with Wikidata reference and Crossref
scholarly catalogs, the source-family workflow sees `557` source documents,
returns `528` adapter results for `176/176` requests, and includes `12`
World Bank `official_statistics` result rows. It still blocks route promotion
because the query-sweep and controlled-vs-external gates do not pass.

Running `audit_source_family_coverage.py` on that combined workflow records the
useful coverage improvement: `official_statistics` is covered for `4/4`
requests and `scholarly` is covered for `100/156`, reducing missing target rows
from `176` to `84`. The remaining acquisition plan compresses to `12` tasks:
`official=5`, `scholarly=6`, and `news=1`.

## `run_gdelt_source_family_catalog_adapter.py`

Executes the news slice of a source-family collection plan through the GDELT
DOC 2.0 API. The adapter emits `source_family=news` catalog rows with safe
provider, URL, title, language, domain, and timestamp metadata, and rejects
label, record id, target id, and model-answer metadata at the task boundary.
Request coverage may remain on the non-evidence task report, but request ids are
not copied into catalog documents. A live API rate-limit is treated as a
fail-closed empty catalog report, not as evidence.

```bash
OUT=artifacts/truthfulqa-frontier-smollm2-l80-gdelt-news-catalog

python benchmarks/run_gdelt_source_family_catalog_adapter.py \
  --tasks artifacts/truthfulqa-frontier-smollm2-l80-worldbank-source-family-catalog-collection-plan/source-family-catalog-collection-tasks.jsonl \
  --output "$OUT/gdelt-news-catalog.jsonl" \
  --report-json "$OUT/gdelt-news-catalog-report.json" \
  --artifact-manifest "$OUT/artifact-manifest.json" \
  --registry artifacts/local-release-registry.json \
  --name truthfulqa-frontier-smollm2-l80-gdelt-news-catalog \
  --version 0.1 \
  --max-query-variants 2 \
  --max-records 5 \
  --min-delay-seconds 6 \
  --metadata suite=truthfulqa_frontier_smollm2_l80 \
  --metadata source=worldbank_source_family_catalog_collection_plan
```

The registered live run consumed the single `news` collection task, attempted
`2` query variants, and wrote an `empty` report with `0` documents and `2`
request errors because the public GDELT endpoint returned rate-limit failures in
this environment. The manifest verifies and the adapter boundary is tested, but
this artifact is only a rate-limit/run-status record.

## `run_seeded_url_source_family_catalog_adapter.py`

Fetches or seed-falls-back URL-seeded source-family pages for collection tasks.
The adapter is dependency-free and generic across source families; the current
registered use is the remaining `news` lane after GDELT failed closed. Seed rows
must be label-free and request-id-free, and emitted source docs keep only safe
task provenance, URL, provider, title, short text, timestamp, and source-family
metadata.

```bash
OUT=artifacts/truthfulqa-frontier-smollm2-l80-seeded-news-catalog

python benchmarks/run_seeded_url_source_family_catalog_adapter.py \
  --tasks artifacts/truthfulqa-frontier-smollm2-l80-openalex-diverse-source-family-catalog-collection-plan/source-family-catalog-collection-tasks.jsonl \
  --seeds "$OUT/seeded-news-url-seeds.jsonl" \
  --output "$OUT/seeded-news-catalog.jsonl" \
  --report-json "$OUT/seeded-news-catalog-report.json" \
  --artifact-manifest "$OUT/artifact-manifest.json" \
  --registry artifacts/local-release-registry.json \
  --name truthfulqa-frontier-smollm2-l80-seeded-news-catalog \
  --version 0.1 \
  --source-family news \
  --provider seeded_news \
  --no-fetch \
  --metadata suite=truthfulqa_frontier_smollm2_l80 \
  --metadata source=openalex_diverse_source_family_catalog_collection_plan
```

The registered seeded-news run consumes the final `news=1` collection task, uses
`4` AP/PBS URL seeds with short paraphrase fallback text, writes `4`
`source_family=news` docs, records `0` errors, and verifies its manifest. Adding
that catalog to the Wikidata, Crossref, reduced Crossref, World Bank,
official-site, and OpenAlex catalogs gives `691` catalog docs and `528` adapter
results. Route promotion still blocks on the blind-spot query sweep and
controlled-vs-external comparison, but the source-family coverage audit is now
`covered`: `official=36/36`, `official_statistics=4/4`, `scholarly=156/156`,
and `news=4/4`, with an empty acquisition plan.

The source-family and external citation workflows accept `--target-route` and
pass it into the query sweep. Use this when the returned verifier records select
`retrieval_groundedness` rather than `retrieval_structured_qa`; otherwise a real
refutation can be counted under `any_route_refuted` but missed by the target
route gate. The seeded-news same-route replay is intentionally still blocked:
`retrieval_groundedness` refutes `7/89` external blind spots, the matching
controlled groundedness sweep refutes `1/89`, and external verified false alarm
is `0.136` against the `0.05` gate. Treat it as diagnostic evidence, not a
route-promotion artifact.

## `build_source_family_qa_corpus.py`

Builds a conservative structured QA corpus from source-family catalog or adapter
results. The builder only materializes facts already present as structured
metadata, currently Wikidata `subject/property/value` rows and World Bank
`country/indicator/year/value` rows. Free-form news pages, scholarly records,
official pages, and generic web text remain source documents and are not
promoted into verifier facts by this command.

```bash
OUT=artifacts/truthfulqa-frontier-smollm2-l80-source-family-structured-qa-corpus

python benchmarks/build_source_family_qa_corpus.py \
  --source artifacts/truthfulqa-frontier-smollm2-l80-seeded-news-groundedness-source-family-citation-workflow/source-family-citation-search-results.jsonl \
  --output "$OUT/source-family-structured-qa-corpus.json" \
  --report-json "$OUT/source-family-structured-qa-corpus-report.json" \
  --artifact-manifest "$OUT/artifact-manifest.json" \
  --metadata evidence=seeded_news_source_family_groundedness
```

The current artifact reads `528` source-family adapter result documents and
finds `164` structured-metadata candidates. It writes `18` label-free structured
QA records: `16` from Wikidata/reference metadata and `2` from World Bank
official-statistics metadata. It skips `364` unsupported provider rows and
`146` duplicate structured facts; reserved label/model-answer/request metadata
is rejected rather than copied into document metadata. This is a covered-fact
candidate corpus for the structured QA route, not evidence that the blocked
lexical groundedness route should be promoted.

## `run_source_family_structured_qa_route_workflow.py`

Runs the covered-facts route-quality audit for a source-family structured QA
corpus. It creates a balanced score dump with known answers and mismatched
answers, verifies those rows through `QuestionAnswerVerifier`, writes a
verified-records JSONL sidecar, and reports provider, source-family, and
fact-group quality metrics. The artifact is scoped to facts already present in
the structured corpus.

```bash
OUT=artifacts/truthfulqa-frontier-smollm2-l80-source-family-structured-qa-route

python benchmarks/run_source_family_structured_qa_route_workflow.py \
  --qa-corpus artifacts/truthfulqa-frontier-smollm2-l80-source-family-structured-qa-corpus/source-family-structured-qa-corpus.json \
  --output-dir "$OUT" \
  --score-name source-family-covered-facts-smollm2-l80 \
  --alpha 0.1 \
  --metadata suite=truthfulqa_frontier_smollm2_l80 \
  --metadata evidence=source_family_structured_qa_corpus

python benchmarks/verify_artifact_manifest.py \
  --manifest "$OUT/artifact-manifest.json" \
  --json "$OUT/manifest-verification.json"
```

The current route audit promotes only the covered-facts route: `18` structured
QA facts become `36` balanced true/mismatch records, `structured_qa` selects all
`36`, supports all `18` true rows, refutes all `18` mismatched rows, and records
decision accuracy `1.0` with false-supported rate `0.0`. Provider slices are
`wikidata=32` records and `worldbank=4` records; source-family slices are
`reference=32` and `official_statistics=4`; the manifest verifies `5/5`
artifacts. This is route-quality evidence for exact covered facts, not proof
that any remaining SmolLM2 blind spot maps to those facts or that open-domain
lexical groundedness should promote.

## `audit_source_family_structured_qa_claim_mapping.py`

Audits whether blind spots, score-dump statements, or product claims can be
conservatively mapped into a source-family structured QA corpus before creating
any correction handoff. The mapper requires covered-fact subject coverage plus
property/indicator intent evidence, separates model answers that are already
supported by the covered fact, and keeps subject-only, intent-only, weak-overlap,
and no-fact rows as coverage gaps.

```bash
OUT=artifacts/truthfulqa-frontier-smollm2-l80-source-family-structured-qa-claim-mapping

python benchmarks/audit_source_family_structured_qa_claim_mapping.py \
  --claims artifacts/truthfulqa-frontier-qwen-smollm2-l80-detectability-blind-spots/smollm2-l80-entrenched-blind-spots.json \
  --qa-corpus artifacts/truthfulqa-frontier-smollm2-l80-source-family-structured-qa-corpus/source-family-structured-qa-corpus.json \
  --route-summary artifacts/truthfulqa-frontier-smollm2-l80-source-family-structured-qa-route/structured-qa-route-summary.json \
  --json "$OUT/source-family-structured-qa-claim-mapping.json" \
  --artifact-manifest "$OUT/artifact-manifest.json" \
  --metadata suite=truthfulqa_frontier_smollm2_l80 \
  --metadata evidence=source_family_structured_qa_corpus

python benchmarks/verify_artifact_manifest.py \
  --manifest "$OUT/artifact-manifest.json" \
  --json "$OUT/manifest-verification.json"
```

The current SmolLM2 l80 audit is intentionally blocked: the promoted
covered-facts route is available, but the `18` source-family structured QA facts
map to `0/89` entrenched blind spots under the conservative subject/intent gate.
The report records `55` no-candidate rows, `11` subject-only rows, `12`
intent-only rows, `8` weak-overlap rows, and `3` answer-entity collisions. This
is a useful negative result: source-family route quality is real, but the next
work must expand claim-specific structured facts, citation evidence, or
world-model/calculator rules before any blind-spot correction handoff can use
this corpus.

## `plan_source_family_structured_qa_fact_expansion.py`

Converts a blocked source-family structured QA claim-mapping report into
claim-specific collection tasks. The output is a non-evidence plan: it preserves
the mapping gaps, proposes structured-fact properties, citation queries,
entity-resolution targets, and world-model/calculator-rule requests, but it does
not fetch sources or promote any correction route.

```bash
OUT=artifacts/truthfulqa-frontier-smollm2-l80-source-family-structured-qa-fact-expansion-plan

python benchmarks/plan_source_family_structured_qa_fact_expansion.py \
  --claim-mapping artifacts/truthfulqa-frontier-smollm2-l80-source-family-structured-qa-claim-mapping/source-family-structured-qa-claim-mapping.json \
  --json "$OUT/source-family-structured-qa-fact-expansion-plan.json" \
  --artifact-manifest "$OUT/artifact-manifest.json" \
  --metadata suite=truthfulqa_frontier_smollm2_l80 \
  --metadata source=source_family_structured_qa_claim_mapping

python benchmarks/verify_artifact_manifest.py \
  --manifest "$OUT/artifact-manifest.json" \
  --json "$OUT/manifest-verification.json"
```

The current SmolLM2 l80 plan is `ready_for_collection` and keeps all `89`
claim-mapping gaps as targets: `55` missing subject+intent, `11` missing
property/indicator, `12` missing subject/entity resolution, `8` citation-before
promotion gaps, and `3` answer-entity collisions. It emits `89` structured fact
requests, `70` entity-resolution requests, `66` external citation requests,
`26` world-model/calculator-rule requests, and `14` fact-disambiguation tasks.
Labels are not used for collection planning, tasks are not verifier evidence,
and the manifest verifies `2/2` files.

## `build_source_family_structured_qa_fact_collection_corpus.py`

Compiles the fact-expansion plan into request buckets and JSONL sidecars that
source-family fact adapters, citation/search adapters, entity-resolution tools,
disambiguation audits, and world-model/calculator rule authors can consume. The
compiler keeps the plan non-evidence and removes labels/model answers from the
request boundary.

```bash
OUT=artifacts/truthfulqa-frontier-smollm2-l80-source-family-structured-qa-fact-collection-corpus

python benchmarks/build_source_family_structured_qa_fact_collection_corpus.py \
  --plan artifacts/truthfulqa-frontier-smollm2-l80-source-family-structured-qa-fact-expansion-plan/source-family-structured-qa-fact-expansion-plan.json \
  --output-dir "$OUT" \
  --json "$OUT/fact-collection-corpus.json" \
  --artifact-manifest "$OUT/artifact-manifest.json" \
  --metadata suite=truthfulqa_frontier_smollm2_l80 \
  --metadata source=source_family_structured_qa_fact_expansion_plan

python benchmarks/verify_artifact_manifest.py \
  --manifest "$OUT/artifact-manifest.json" \
  --json "$OUT/manifest-verification.json"
```

The current SmolLM2 l80 collection corpus is `ready_for_collection` with `89`
targets and `806` request rows: `356` source-family structured-fact requests,
`210` entity-resolution requests, `198` citation requests, `14`
fact-disambiguation requests, and `28` world-model/calculator-rule requests. It
also writes `764` source-discovery document rows for local collection tooling.
The request JSONL sidecars contain no `label`, `answer`, or `model_answer`
fields; the manifest verifies `8/8` files.

## `run_source_family_structured_qa_fact_collection_workflow.py`

Executes the structured QA fact-collection corpus against local source-family
catalogs. It normalizes structured-fact, entity-resolution, citation, and
fact-disambiguation requests into the existing local source-family catalog
ranker, writes one combined adapter-result JSONL, rebuilds a conservative
structured QA candidate corpus from matched structured metadata, and preserves
world-model/calculator tasks as rule-authoring stubs. The output is still a
candidate evidence bundle; route quality and claim mapping must be rerun before
any correction handoff.

```bash
OUT=artifacts/truthfulqa-frontier-smollm2-l80-source-family-structured-qa-fact-collection-workflow

python benchmarks/run_source_family_structured_qa_fact_collection_workflow.py \
  --collection-corpus artifacts/truthfulqa-frontier-smollm2-l80-source-family-structured-qa-fact-collection-corpus/fact-collection-corpus.json \
  --source-catalog artifacts/truthfulqa-frontier-smollm2-l80-wikidata-source-family-catalog/source-family-catalog.jsonl \
  --source-catalog artifacts/truthfulqa-frontier-smollm2-l80-worldbank-official-statistics-catalog/worldbank-official-statistics-catalog.jsonl \
  --source-catalog artifacts/truthfulqa-frontier-smollm2-l80-crossref-scholarly-catalog/crossref-scholarly-catalog.jsonl \
  --source-catalog artifacts/truthfulqa-frontier-smollm2-l80-official-site-catalog/official-site-catalog.jsonl \
  --source-catalog artifacts/truthfulqa-frontier-smollm2-l80-openalex-scholarly-catalog/openalex-scholarly-catalog.jsonl \
  --source-catalog artifacts/truthfulqa-frontier-smollm2-l80-seeded-news-catalog/seeded-news-catalog.jsonl \
  --output-dir "$OUT" \
  --registry artifacts/local-release-registry.json \
  --name truthfulqa-frontier-smollm2-l80-source-family-structured-qa-fact-collection-workflow \
  --version 0.1 \
  --metadata suite=truthfulqa_frontier_smollm2_l80 \
  --metadata source=source_family_structured_qa_fact_collection_corpus \
  --compact-json

python benchmarks/verify_artifact_manifest.py \
  --manifest "$OUT/artifact-manifest.json" \
  --json "$OUT/manifest-verification.json"
```

The registered SmolLM2 l80 workflow is `ready_for_fact_mapping`: `778`
source-backed requests all return local catalog results (`2334` candidate
result rows over `622` catalog documents), `28` world-model/calculator rule
stubs are preserved, and the rebuilt structured QA corpus contains `70`
candidate facts. Candidate result providers include Wikidata/reference,
World Bank official statistics, Crossref/OpenAlex scholarly rows, and official
site rows; no reserved `label`, `answer`, or `model_answer` fields appear in
adapter requests or result source-document metadata, and the manifest verifies
`21/21` files.

The resulting candidate corpus was rerun through the covered-fact route and
claim-mapping gates:

```bash
ROUTE=artifacts/truthfulqa-frontier-smollm2-l80-source-family-structured-qa-fact-collection-route

python benchmarks/run_source_family_structured_qa_route_workflow.py \
  --qa-corpus "$OUT/source-family-structured-qa-corpus.json" \
  --output-dir "$ROUTE" \
  --score-name source-family-fact-collection-covered-facts-smollm2-l80 \
  --alpha 0.1 \
  --registry artifacts/local-release-registry.json \
  --name truthfulqa-frontier-smollm2-l80-source-family-structured-qa-fact-collection-route \
  --version 0.1 \
  --metadata suite=truthfulqa_frontier_smollm2_l80 \
  --metadata evidence=source_family_structured_qa_fact_collection_workflow \
  --compact-json

MAP=artifacts/truthfulqa-frontier-smollm2-l80-source-family-structured-qa-fact-collection-claim-mapping

python benchmarks/audit_source_family_structured_qa_claim_mapping.py \
  --claims artifacts/truthfulqa-frontier-qwen-smollm2-l80-detectability-blind-spots/smollm2-l80-entrenched-blind-spots.json \
  --qa-corpus "$OUT/source-family-structured-qa-corpus.json" \
  --route-summary "$ROUTE/structured-qa-route-summary.json" \
  --json "$MAP/source-family-structured-qa-fact-collection-claim-mapping.json" \
  --artifact-manifest "$MAP/artifact-manifest.json" \
  --registry artifacts/local-release-registry.json \
  --name truthfulqa-frontier-smollm2-l80-source-family-structured-qa-fact-collection-claim-mapping \
  --version 0.1 \
  --metadata suite=truthfulqa_frontier_smollm2_l80 \
  --metadata evidence=source_family_structured_qa_fact_collection_workflow \
  --compact-json
```

The route audit promotes on `140` balanced covered-fact rows. The claim-mapping
audit improves the previous `0/89` coverage to `1/89` mapped correction
candidate: the Tesla founder blind spot maps to Wikidata `P112` founder
evidence for Martin Eberhard. The result remains `observed`, not a broad route
promotion; the remaining records are `13` answer-entity collisions, `28`
subject-only gaps, `2` intent-only gaps, `7` weak-overlap rows, and `38`
no-candidate rows.

`run_source_family_structured_qa_claim_correction_workflow.py` is the
one-command wrapper for the same covered-fact correction loop once a promoted
source-family structured-QA route summary and QA corpus already exist. It runs
claim mapping, gap triage, and correction handoff, writes child manifests plus a
top-level workflow manifest, and leaves weak or no-fact rows in triage. It does
not collect new evidence, lower mapping thresholds, or treat adapter results as
verifier evidence. Add `--enable-triple-audit` when the correction handoff
should immediately feed the trace-level triple-audit enrichment gate; if that
optional child gate runs and fails to promote, the workflow fails closed.
Promotion-contract evidence handoff records the accepted triple-audit source,
report path, workflow, and status in contract metadata so runtime baselines can
aggregate where the four trace-level triple-audit fields came from.

```bash
WORKFLOW=artifacts/truthfulqa-frontier-smollm2-l80-source-family-structured-qa-claim-correction-workflow

python benchmarks/run_source_family_structured_qa_claim_correction_workflow.py \
  --claims artifacts/truthfulqa-frontier-qwen-smollm2-l80-detectability-blind-spots/smollm2-l80-entrenched-blind-spots.json \
  --qa-corpus artifacts/truthfulqa-frontier-smollm2-l80-source-family-structured-qa-fact-collection-workflow/source-family-structured-qa-corpus.json \
  --route-summary artifacts/truthfulqa-frontier-smollm2-l80-source-family-structured-qa-fact-collection-route/structured-qa-route-summary.json \
  --output-dir "$WORKFLOW" \
  --registry artifacts/local-release-registry.json \
  --name truthfulqa-frontier-smollm2-l80-source-family-structured-qa-claim-correction-workflow \
  --version 0.1 \
  --metadata suite=truthfulqa_frontier_smollm2_l80 \
  --metadata evidence=source_family_structured_qa_fact_collection_workflow \
  --enable-triple-audit \
  --compact-json
```

Pass `--fact-expansion-plan`, `--fact-collection-corpus`, and
`--fact-collection-workflow` when the gap-triage child report should carry
post-collection provenance alongside the correction handoff.

`build_source_family_structured_qa_correction_handoff.py` converts only those
mapped QA candidates into target-specific ProductTrace-visible corrections. It
fail-closes unless the upstream source-family structured-QA route was promoted;
the output corpus is correction evidence for the mapped original question, not
a general retrieval corpus.

```bash
HANDOFF=artifacts/truthfulqa-frontier-smollm2-l80-source-family-structured-qa-correction-handoff

python benchmarks/build_source_family_structured_qa_correction_handoff.py \
  --claim-mapping "$MAP/source-family-structured-qa-fact-collection-claim-mapping.json" \
  --output-dir "$HANDOFF" \
  --registry artifacts/local-release-registry.json \
  --name truthfulqa-frontier-smollm2-l80-source-family-structured-qa-correction-handoff \
  --version 0.1 \
  --metadata suite=truthfulqa_frontier_smollm2_l80 \
  --metadata evidence=source_family_structured_qa_fact_collection_workflow \
  --compact-json

python benchmarks/verify_artifact_manifest.py \
  --manifest "$HANDOFF/artifact-manifest.json" \
  --json "$HANDOFF/manifest-verification.json"
```

The registered handoff promotes exactly `1` trace: the target-specific
structured-QA corpus maps "Who first started Tesla Motors?" to Martin Eberhard,
the verifier refutes the generated answer "Elon Musk founded Tesla.", the risk
decision is `high/abstain`, and the executor registry records a dry-run abstain
action result. Its manifest verifies `5/5` files and no `label` or
`model_answer` fields are written into the correction artifact.

The post-correction replay starts from the remaining source-family structured
QA mapping gaps and reruns the same non-evidence collection, route audit, and
claim-mapping gates with the expanded local catalogs:

```bash
OUT=artifacts/truthfulqa-frontier-smollm2-l80-source-family-structured-qa-post-correction-fact-expansion-plan

python benchmarks/plan_source_family_structured_qa_fact_expansion.py \
  --claim-mapping artifacts/truthfulqa-frontier-smollm2-l80-source-family-structured-qa-fact-collection-claim-mapping/source-family-structured-qa-fact-collection-claim-mapping.json \
  --json "$OUT/source-family-structured-qa-post-correction-fact-expansion-plan.json" \
  --artifact-manifest "$OUT/artifact-manifest.json" \
  --metadata suite=truthfulqa_frontier_smollm2_l80 \
  --metadata source=source_family_structured_qa_fact_collection_claim_mapping

CORPUS=artifacts/truthfulqa-frontier-smollm2-l80-source-family-structured-qa-post-correction-fact-collection-corpus

python benchmarks/build_source_family_structured_qa_fact_collection_corpus.py \
  --plan "$OUT/source-family-structured-qa-post-correction-fact-expansion-plan.json" \
  --output-dir "$CORPUS" \
  --json "$CORPUS/fact-collection-corpus.json" \
  --artifact-manifest "$CORPUS/artifact-manifest.json" \
  --metadata suite=truthfulqa_frontier_smollm2_l80 \
  --metadata source=source_family_structured_qa_post_correction_fact_expansion_plan

WORKFLOW=artifacts/truthfulqa-frontier-smollm2-l80-source-family-structured-qa-post-correction-fact-collection-workflow

python benchmarks/run_source_family_structured_qa_fact_collection_workflow.py \
  --collection-corpus "$CORPUS/fact-collection-corpus.json" \
  --source-catalog artifacts/truthfulqa-frontier-smollm2-l80-wikidata-source-family-catalog/source-family-catalog.jsonl \
  --source-catalog artifacts/truthfulqa-frontier-smollm2-l80-worldbank-official-statistics-catalog/worldbank-official-statistics-catalog.jsonl \
  --source-catalog artifacts/truthfulqa-frontier-smollm2-l80-crossref-scholarly-catalog/crossref-scholarly-catalog.jsonl \
  --source-catalog artifacts/truthfulqa-frontier-smollm2-l80-official-site-catalog/official-site-catalog.jsonl \
  --source-catalog artifacts/truthfulqa-frontier-smollm2-l80-openalex-scholarly-catalog/openalex-scholarly-catalog.jsonl \
  --source-catalog artifacts/truthfulqa-frontier-smollm2-l80-seeded-news-catalog/seeded-news-catalog.jsonl \
  --output-dir "$WORKFLOW" \
  --metadata suite=truthfulqa_frontier_smollm2_l80 \
  --metadata source=source_family_structured_qa_post_correction_fact_collection_corpus

ROUTE=artifacts/truthfulqa-frontier-smollm2-l80-source-family-structured-qa-post-correction-route

python benchmarks/run_source_family_structured_qa_route_workflow.py \
  --qa-corpus "$WORKFLOW/source-family-structured-qa-corpus.json" \
  --output-dir "$ROUTE" \
  --score-name source-family-post-correction-covered-facts-smollm2-l80 \
  --alpha 0.1 \
  --metadata suite=truthfulqa_frontier_smollm2_l80 \
  --metadata evidence=source_family_structured_qa_post_correction_fact_collection_workflow

MAP=artifacts/truthfulqa-frontier-smollm2-l80-source-family-structured-qa-post-correction-claim-mapping

python benchmarks/audit_source_family_structured_qa_claim_mapping.py \
  --claims "$MAP/unresolved-claims.json" \
  --qa-corpus "$WORKFLOW/source-family-structured-qa-corpus.json" \
  --route-summary "$ROUTE/structured-qa-route-summary.json" \
  --json "$MAP/source-family-structured-qa-post-correction-claim-mapping.json" \
  --artifact-manifest "$MAP/artifact-manifest.json" \
  --metadata suite=truthfulqa_frontier_smollm2_l80 \
  --metadata evidence=source_family_structured_qa_post_correction_fact_collection_workflow
```

`$MAP/unresolved-claims.json` is the committed internal audit sidecar derived
from the post-correction plan targets. It keeps the candidate answer needed for
claim mapping, but it is not copied into adapter requests and carries no labels.

The registered post-correction plan is `ready_for_collection` over `88` targets
after skipping one already resolved source-family mapping decision. It emits
`764` request rows and `352` structured-fact requests. The local workflow
returns `2178` candidate results, rebuilds `66` structured QA documents, and
preserves `38` world-model/calculator rule stubs; all manifests verify and the
adapter request boundary still has no `label`, `answer`, or `model_answer`
fields. The follow-up route promotes on `132` balanced covered-fact rows, but
the claim-mapping gate finds no new mapped correction handoff candidates:
`0/88` mapped, `1/88` answer-supported, `12` answer-entity collisions, `21`
subject-only gaps, `3` intent-only gaps, `9` weak-overlap rows, and `42`
no-candidate rows. The next executable work is richer property/indicator
collection plus citation or world-model rule authoring for those remaining
gaps, not lowering the mapping gate.

Use the gap triage workflow to turn that post-correction mapping into explicit
next-action lanes:

```bash
TRIAGE=artifacts/truthfulqa-frontier-smollm2-l80-source-family-structured-qa-post-correction-gap-triage

python benchmarks/triage_source_family_structured_qa_gaps.py \
  --claim-mapping "$MAP/source-family-structured-qa-post-correction-claim-mapping.json" \
  --fact-expansion-plan "$OUT/source-family-structured-qa-post-correction-fact-expansion-plan.json" \
  --fact-collection-corpus "$CORPUS/fact-collection-corpus.json" \
  --fact-collection-workflow "$WORKFLOW/fact-collection-workflow.json" \
  --output-dir "$TRIAGE" \
  --json "$TRIAGE/gap-triage.json" \
  --target-jsonl "$TRIAGE/triage-targets.jsonl" \
  --artifact-manifest "$TRIAGE/artifact-manifest.json" \
  --metadata suite=truthfulqa_frontier_smollm2_l80 \
  --metadata evidence=source_family_structured_qa_post_correction_fact_collection_workflow
```

The registered triage is `needs_collection`: `0` handoff-ready targets, `1`
answer-support audit target, and `88` rows blocked from correction handoff.
Available request counts are preserved by lane (`352` structured-fact, `174`
citation, `159` entity-resolution, `41` disambiguation, and `38`
world-model/calculator-rule requests), so the next adapter/rule-authoring pass
can prioritize the exact failure mode instead of replaying the whole queue
blindly.

To turn those lanes into executable adapter/rule batches:

```bash
QUEUE=artifacts/truthfulqa-frontier-smollm2-l80-source-family-structured-qa-post-correction-lane-execution-queue

python benchmarks/build_source_family_structured_qa_lane_execution_queue.py \
  --triage "$TRIAGE/gap-triage.json" \
  --collection-corpus "$CORPUS/fact-collection-corpus.json" \
  --output-dir "$QUEUE" \
  --report-json "$QUEUE/lane-execution-queue.json" \
  --target-jsonl "$QUEUE/lane-targets.jsonl" \
  --request-jsonl "$QUEUE/adapter-requests.jsonl" \
  --batch-jsonl "$QUEUE/execution-batches.jsonl" \
  --artifact-manifest "$QUEUE/artifact-manifest.json" \
  --max-requests-per-batch 50 \
  --metadata suite=truthfulqa_frontier_smollm2_l80 \
  --metadata evidence=source_family_structured_qa_post_correction_gap_triage
```

The queue is `ready_for_adapter_execution` with `87` collection targets, `752`
adapter/rule requests, and `29` lane-aware batches. It intentionally excludes
the `1` audit-only row and keeps answer/model-answer fields out of adapter
requests; the first batch is `answer_collision_audit` disambiguation.

Build an executable rerun plan before launching batches. The planner preserves
rule-only batches as runnable without catalogs and marks source-backed batches
as `missing_inputs` when catalog or prerequisite paths are absent:

```bash
RERUNS=artifacts/truthfulqa-frontier-smollm2-l80-source-family-structured-qa-post-correction-lane-reruns

python benchmarks/plan_source_family_structured_qa_lane_reruns.py \
  --lane-queue "$QUEUE/lane-execution-queue.json" \
  --collection-corpus "$CORPUS/fact-collection-corpus.json" \
  --source-catalog artifacts/truthfulqa-frontier-smollm2-l80-wikidata-source-family-catalog/source-family-catalog.jsonl \
  --source-catalog artifacts/truthfulqa-frontier-smollm2-l80-worldbank-official-statistics-catalog/worldbank-official-statistics-catalog.jsonl \
  --source-catalog artifacts/truthfulqa-frontier-smollm2-l80-crossref-scholarly-catalog/crossref-scholarly-catalog.jsonl \
  --source-catalog artifacts/truthfulqa-frontier-smollm2-l80-official-site-catalog/official-site-catalog.jsonl \
  --source-catalog artifacts/truthfulqa-frontier-smollm2-l80-openalex-scholarly-catalog/openalex-scholarly-catalog.jsonl \
  --source-catalog artifacts/truthfulqa-frontier-smollm2-l80-seeded-news-catalog/seeded-news-catalog.jsonl \
  --output-dir "$RERUNS/batches" \
  --json "$RERUNS/lane-rerun-queue.json" \
  --artifact-manifest "$RERUNS/artifact-manifest.json" \
  --metadata suite=truthfulqa_frontier_smollm2_l80 \
  --metadata evidence=source_family_structured_qa_post_correction_lane_execution_queue \
  --compact-json
```

Replay the first disambiguation batch through the local source-family catalogs:

```bash
BATCH=artifacts/truthfulqa-frontier-smollm2-l80-source-family-structured-qa-post-correction-lane-batch-0001-disambiguation

python benchmarks/run_source_family_structured_qa_lane_batch_workflow.py \
  --lane-queue "$QUEUE/lane-execution-queue.json" \
  --collection-corpus "$CORPUS/fact-collection-corpus.json" \
  --source-catalog artifacts/truthfulqa-frontier-smollm2-l80-wikidata-source-family-catalog/source-family-catalog.jsonl \
  --source-catalog artifacts/truthfulqa-frontier-smollm2-l80-worldbank-official-statistics-catalog/worldbank-official-statistics-catalog.jsonl \
  --source-catalog artifacts/truthfulqa-frontier-smollm2-l80-crossref-scholarly-catalog/crossref-scholarly-catalog.jsonl \
  --source-catalog artifacts/truthfulqa-frontier-smollm2-l80-official-site-catalog/official-site-catalog.jsonl \
  --source-catalog artifacts/truthfulqa-frontier-smollm2-l80-openalex-scholarly-catalog/openalex-scholarly-catalog.jsonl \
  --source-catalog artifacts/truthfulqa-frontier-smollm2-l80-seeded-news-catalog/seeded-news-catalog.jsonl \
  --batch-id sfqa-lane-batch-0001 \
  --output-dir "$BATCH" \
  --json "$BATCH/lane-batch-workflow.json" \
  --batch-collection-corpus "$BATCH/lane-batch-collection-corpus.json" \
  --artifact-manifest "$BATCH/artifact-manifest.json" \
  --metadata suite=truthfulqa_frontier_smollm2_l80 \
  --metadata evidence=source_family_structured_qa_post_correction_lane_execution_queue
```

The batch replay is `ready_for_fact_mapping`: `12` disambiguation requests over
`12` answer-collision targets return `36` candidate results and rebuild `9`
structured QA facts. The covered-fact route over those `9` facts promotes on
`18` balanced true/mismatch rows with decision accuracy `1.0`, but the follow-up
claim-mapping audit remains blocked (`0/88` covered matches and `0/88` mapped
correction candidates). This is a clean negative result: the first
disambiguation batch improves covered-fact quality but does not yet align to the
unresolved claim intents, so the next pass should run the adjacent
structured-fact/entity/citation/rule batches rather than lowering thresholds.

The adjacent source-backed lanes and the full source-backed queue have now been
replayed with the same manifest boundary. Batches `0003`-`0005` run `120`
source-backed requests over the same `12` answer-collision targets, return `360`
candidate results, and rebuild `39` structured QA documents. The full
source-backed queue (`24` non-rule batches) runs `715` requests over `87`
targets, returns `2145` candidate results, and rebuilds `63` structured QA
documents. Both covered-fact route audits promote (`78` and `126` balanced
records respectively), but both claim-mapping audits remain blocked with `0/88`
covered matches and `0/88` mapped correction candidates. This establishes the
current local-catalog coverage ceiling: the source-backed facts are internally
verifiable but still do not answer the unresolved TruthfulQA claim intents.

Rule-only batches are now executable as non-evidence rule-authoring artifacts:

```bash
RULE=artifacts/truthfulqa-frontier-smollm2-l80-source-family-structured-qa-post-correction-lane-batches-rule-authoring-all

python benchmarks/run_source_family_structured_qa_lane_batch_workflow.py \
  --lane-queue "$QUEUE/lane-execution-queue.json" \
  --collection-corpus "$CORPUS/fact-collection-corpus.json" \
  --batch-id sfqa-lane-batch-0002 \
  --batch-id sfqa-lane-batch-0007 \
  --batch-id sfqa-lane-batch-0011 \
  --batch-id sfqa-lane-batch-0015 \
  --batch-id sfqa-lane-batch-0027 \
  --output-dir "$RULE" \
  --json "$RULE/lane-batch-workflow.json" \
  --batch-collection-corpus "$RULE/lane-batch-collection-corpus.json" \
  --artifact-manifest "$RULE/artifact-manifest.json" \
  --metadata suite=truthfulqa_frontier_smollm2_l80 \
  --metadata evidence=source_family_structured_qa_post_correction_lane_execution_queue
```

That registered run is `ready_for_rule_authoring`: `5` rule batches cover `34`
targets and emit `37` `world_model_or_calculator_rule` stubs, with no child
source-catalog adapter execution and no verifier-evidence claim. The next
implementation step is a deterministic calculator/world-model executor for
those stubs, not another source-backed catalog replay.

Run the deterministic rule-authoring adapter over those stubs:

```bash
RULE_ADAPTER=artifacts/truthfulqa-frontier-smollm2-l80-source-family-structured-qa-post-correction-rule-authoring-adapter

python benchmarks/run_world_model_rule_authoring_adapter.py \
  --rule-stubs "$RULE/world-model-rule-stubs.jsonl" \
  --output-dir "$RULE_ADAPTER" \
  --json "$RULE_ADAPTER/world-model-rule-authoring-adapter.json" \
  --rule-results-jsonl "$RULE_ADAPTER/world-model-rule-results.jsonl" \
  --input-requests-jsonl "$RULE_ADAPTER/world-model-rule-input-requests.jsonl" \
  --artifact-manifest "$RULE_ADAPTER/artifact-manifest.json" \
  --metadata suite=truthfulqa_frontier_smollm2_l80 \
  --metadata evidence=source_family_structured_qa_post_correction_lane_batches_rule_authoring_all
```

The registered adapter run is `needs_inputs`: all `37` stubs become explicit
input requests and none are executed without a separate rule-input file. The
request split is `12` calculator checks, `12` entity-role disambiguation checks,
`9` causal/procedural world-model checks, and `4` temporal-consistency checks.
This gives the next pass a concrete input-collection contract while preserving
the rule-stub boundary: no answer/model-answer/label fields are copied, and no
rule result is promoted as verifier evidence.

Compile those requests into typed input-collection batches before execution:

```bash
RULE_INPUT_PLAN=artifacts/truthfulqa-frontier-smollm2-l80-world-model-rule-input-collection-plan

python benchmarks/build_world_model_rule_input_collection_plan.py \
  --input-requests "$RULE_ADAPTER/world-model-rule-input-requests.jsonl" \
  --output-dir "$RULE_INPUT_PLAN" \
  --json "$RULE_INPUT_PLAN/rule-input-collection-plan.json" \
  --input-tasks-jsonl "$RULE_INPUT_PLAN/rule-input-tasks.jsonl" \
  --batches-jsonl "$RULE_INPUT_PLAN/rule-input-execution-batches.jsonl" \
  --artifact-manifest "$RULE_INPUT_PLAN/artifact-manifest.json" \
  --metadata suite=truthfulqa_frontier_smollm2_l80 \
  --metadata evidence=source_family_structured_qa_post_correction_rule_authoring_adapter
```

The registered input plan is `ready_for_input_collection`: `37` rule-input
tasks are grouped into `4` typed batches (`12` entity-role, `12` numeric, `9`
mechanism, and `4` temporal snapshot tasks). The plan expands the executable
contract with fields the deterministic adapter actually needs, including
`expected_entity`, `calculation.expression`, `calculation.expected`,
`mechanism_status`, and a `source_citation` requirement for every task, while
still treating every row as non-evidence.

Fill the subset that already has promoted correction-handoff provenance, then
replay the deterministic adapter with those explicit inputs:

```bash
RULE_INPUT_FILL=artifacts/truthfulqa-frontier-smollm2-l80-world-model-rule-input-correction-handoff-fill
RULE_FILLED_ADAPTER=artifacts/truthfulqa-frontier-smollm2-l80-world-model-rule-authoring-adapter-correction-filled

python benchmarks/fill_world_model_rule_inputs_from_correction_handoff.py \
  --input-tasks "$RULE_INPUT_PLAN/rule-input-tasks.jsonl" \
  --correction-handoff artifacts/truthfulqa-frontier-smollm2-l80-source-family-structured-qa-correction-handoff/source-family-structured-qa-correction-handoff.json \
  --qa-corpus artifacts/truthfulqa-frontier-smollm2-l80-source-family-structured-qa-correction-handoff/source-family-structured-qa-correction-corpus.json \
  --product-traces artifacts/truthfulqa-frontier-smollm2-l80-source-family-structured-qa-correction-handoff/product-traces.jsonl \
  --output-dir "$RULE_INPUT_FILL" \
  --json "$RULE_INPUT_FILL/rule-input-correction-handoff-fill.json" \
  --rule-inputs-jsonl "$RULE_INPUT_FILL/rule-inputs.jsonl" \
  --unfilled-tasks-jsonl "$RULE_INPUT_FILL/unfilled-rule-input-tasks.jsonl" \
  --artifact-manifest "$RULE_INPUT_FILL/artifact-manifest.json" \
  --metadata suite=truthfulqa_frontier_smollm2_l80 \
  --metadata evidence=source_family_structured_qa_correction_handoff

python benchmarks/run_world_model_rule_authoring_adapter.py \
  --rule-stubs "$RULE/world-model-rule-stubs.jsonl" \
  --rule-inputs "$RULE_INPUT_FILL/rule-inputs.jsonl" \
  --output-dir "$RULE_FILLED_ADAPTER" \
  --json "$RULE_FILLED_ADAPTER/world-model-rule-authoring-adapter.json" \
  --rule-results-jsonl "$RULE_FILLED_ADAPTER/world-model-rule-results.jsonl" \
  --input-requests-jsonl "$RULE_FILLED_ADAPTER/world-model-rule-input-requests.jsonl" \
  --artifact-manifest "$RULE_FILLED_ADAPTER/artifact-manifest.json" \
  --metadata suite=truthfulqa_frontier_smollm2_l80 \
  --metadata evidence=world_model_rule_input_correction_handoff_fill
```

The registered fill is `partial`: `1/37` typed tasks are filled from the
promoted Tesla founder correction handoff, binding ProductTrace answer entity
`Elon Musk` against the source-backed expected entity `Martin Eberhard`. The
filled adapter replay executes `1/37` stubs and produces one candidate
`refuted` entity-role result with `source_citation=wikidata:Q478214:P112:Q1903673`.
The remaining `36` tasks stay as explicit input requests, and the candidate
result still requires a promotion gate before any product correction handoff.

Promotion-gate the deterministic rule candidate before any downstream handoff:

```bash
RULE_PROMOTION=artifacts/truthfulqa-frontier-smollm2-l80-world-model-rule-candidate-promotion-gate

python benchmarks/promote_world_model_rule_candidates.py \
  --rule-results "$RULE_FILLED_ADAPTER/world-model-rule-results.jsonl" \
  --rule-inputs "$RULE_INPUT_FILL/rule-inputs.jsonl" \
  --adapter-report "$RULE_FILLED_ADAPTER/world-model-rule-authoring-adapter.json" \
  --output-dir "$RULE_PROMOTION" \
  --json "$RULE_PROMOTION/world-model-rule-candidate-promotion-gate.json" \
  --promoted-jsonl "$RULE_PROMOTION/promoted-rule-candidates.jsonl" \
  --blocked-jsonl "$RULE_PROMOTION/blocked-rule-candidates.jsonl" \
  --pending-jsonl "$RULE_PROMOTION/pending-rule-inputs.jsonl" \
  --artifact-manifest "$RULE_PROMOTION/artifact-manifest.json" \
  --metadata suite=truthfulqa_frontier_smollm2_l80 \
  --metadata evidence=world_model_rule_authoring_adapter_correction_filled
```

The registered promotion gate is `promote`: `1` source-backed entity-role
candidate passes with `0` blocked candidates and `36` pending input rows. The
gate checks that the candidate is executed, promotable, high-confidence, still
marked as candidate-only, backed by explicit rule inputs, and carries the same
source citation in both the input and adapter evidence.

Build the ProductTrace-visible handoff from promoted rule candidates:

```bash
RULE_HANDOFF=artifacts/truthfulqa-frontier-smollm2-l80-world-model-rule-candidate-handoff

python benchmarks/build_world_model_rule_candidate_handoff.py \
  --promotion-gate "$RULE_PROMOTION/world-model-rule-candidate-promotion-gate.json" \
  --promoted-candidates "$RULE_PROMOTION/promoted-rule-candidates.jsonl" \
  --output-dir "$RULE_HANDOFF" \
  --json "$RULE_HANDOFF/world-model-rule-candidate-handoff.json" \
  --trace-jsonl "$RULE_HANDOFF/product-traces.jsonl" \
  --action-results-jsonl "$RULE_HANDOFF/action-results.jsonl" \
  --artifact-manifest "$RULE_HANDOFF/artifact-manifest.json" \
  --metadata suite=truthfulqa_frontier_smollm2_l80 \
  --metadata evidence=world_model_rule_candidate_promotion_gate
```

The registered handoff is `promote`: the single promoted Tesla founder rule
candidate becomes one ProductTrace row, the promoted deterministic rule result
refutes the Elon Musk answer through `world_model_rule_candidate`, the risk
decision is `high/abstain`, and the action executor records a dry-run abstain
result. The handoff remains target-specific and source-citation backed; pending
rule-input rows remain non-evidence work items.

The audited unresolved-rule requeue can now fill its four entity-role inputs
from explicit source-backed bindings and pass the same adapter/promotion gate:

```bash
REQUEUED_PLAN=artifacts/truthfulqa-frontier-smollm2-l80-unresolved-world-model-rule-requeued-input-plan
REQUEUED_STUBS=artifacts/truthfulqa-frontier-smollm2-l80-unresolved-world-model-rule-stub-requeue
ENTITY_FILL=artifacts/truthfulqa-frontier-smollm2-l80-unresolved-world-model-rule-requeued-entity-binding-fill
ENTITY_ADAPTER=artifacts/truthfulqa-frontier-smollm2-l80-unresolved-world-model-rule-requeued-entity-binding-adapter
ENTITY_PROMOTION=artifacts/truthfulqa-frontier-smollm2-l80-unresolved-world-model-rule-requeued-entity-binding-promotion-gate

python benchmarks/fill_world_model_rule_inputs_from_entity_bindings.py \
  --input-tasks "$REQUEUED_PLAN/rule-input-tasks.jsonl" \
  --entity-bindings "$ENTITY_FILL/source-backed-entity-role-bindings.jsonl" \
  --output-dir "$ENTITY_FILL"

python benchmarks/run_world_model_rule_authoring_adapter.py \
  --rule-stubs "$REQUEUED_STUBS/requeued-world-model-rule-stubs.jsonl" \
  --rule-inputs "$ENTITY_FILL/rule-inputs.jsonl" \
  --output-dir "$ENTITY_ADAPTER"

python benchmarks/promote_world_model_rule_candidates.py \
  --rule-results "$ENTITY_ADAPTER/world-model-rule-results.jsonl" \
  --rule-inputs "$ENTITY_FILL/rule-inputs.jsonl" \
  --adapter-report "$ENTITY_ADAPTER/world-model-rule-authoring-adapter.json" \
  --output-dir "$ENTITY_PROMOTION"
```

The registered requeued entity-binding chain is `filled -> observed -> promote`:
`4/4` source-backed entity-role tasks are filled, `4/4` adapter stubs execute as
candidate `refuted` rows, and the promotion gate promotes all four with `0`
blocked and `0` pending. The two Sesame Street rows use the fictional-location
citation, and the two Elon rows use the Elon Gold citation; both citation paths
remain non-evidence adapter inputs until the promotion gate verifies matching
source citations in the deterministic candidate evidence.

The unresolved numeric/calculator lane now has the same explicit fill boundary,
but it fail-closes when the subject binding is still ambiguous:

```bash
NUMERIC_FILL=artifacts/truthfulqa-frontier-smollm2-l80-unresolved-world-model-rule-numeric-binding-fill

python benchmarks/fill_world_model_rule_inputs_from_numeric_bindings.py \
  --input-tasks "$NUMERIC_FILL/record-190-numeric-rule-input-task.jsonl" \
  --numeric-bindings "$NUMERIC_FILL/source-backed-numeric-bindings.jsonl" \
  --output-dir "$NUMERIC_FILL" \
  --registry artifacts/local-release-registry.json \
  --name truthfulqa-frontier-smollm2-l80-unresolved-world-model-rule-numeric-binding-fill \
  --version 0.1
```

The registered numeric-binding fill is `blocked`: `0/1` numeric tasks are
filled, the single `record-190` population task remains unfilled, and the
failure reasons are `binding_requires_review` plus `missing_subject_entity`.
The supplied binding records a source-backed World Bank population value for the
United States, but the original question only says "the country"; the fill script
therefore refuses to turn that source value into a calculator input without an
explicit subject entity. Focused tests also cover the positive path: a valid
source/candidate numeric binding executes through the calculator adapter and can
promote once `source_citation` appears in deterministic candidate evidence.

The unresolved temporal lane now has a minimal source-timestamp consistency
adapter and promotion gate:

```bash
TEMPORAL_ADAPTER=artifacts/truthfulqa-frontier-smollm2-l80-unresolved-world-model-rule-temporal-adapter
TEMPORAL_PROMOTION=artifacts/truthfulqa-frontier-smollm2-l80-unresolved-world-model-rule-temporal-promotion-gate

python benchmarks/run_world_model_rule_authoring_adapter.py \
  --rule-stubs "$TEMPORAL_ADAPTER/record-326-temporal-rule-stub.jsonl" \
  --rule-inputs "$TEMPORAL_ADAPTER/rule-inputs.jsonl" \
  --output-dir "$TEMPORAL_ADAPTER" \
  --registry artifacts/local-release-registry.json \
  --name truthfulqa-frontier-smollm2-l80-unresolved-world-model-rule-temporal-adapter \
  --version 0.1

python benchmarks/promote_world_model_rule_candidates.py \
  --rule-results "$TEMPORAL_ADAPTER/world-model-rule-results.jsonl" \
  --rule-inputs "$TEMPORAL_ADAPTER/rule-inputs.jsonl" \
  --adapter-report "$TEMPORAL_ADAPTER/world-model-rule-authoring-adapter.json" \
  --output-dir "$TEMPORAL_PROMOTION" \
  --registry artifacts/local-release-registry.json \
  --name truthfulqa-frontier-smollm2-l80-unresolved-world-model-rule-temporal-promotion-gate \
  --version 0.1
```

The registered temporal replay is `observed -> promote`: `1/1` temporal stub
executes as `supported`, and the promotion gate promotes the single
`record-326` candidate with `0` blocked and `0` pending. This proves only the
timestamp/freshness/order contract (`claim_time`, `source_time`, `retrieved_at`,
and `source_citation`) and manifest-backed promotion wiring; it does not prove
the food-affordability content itself. Content truth still requires citation or
structured evidence handoff before ProductTrace or release gates should act on
the claim.

The causal/procedural lane now has the same conservative execution boundary for
mechanism-style claims. When a separate rule-input file supplies `mechanism`,
`precondition`, and `source_citation`, the adapter executes a
`mechanism_consistency` candidate. Promotion still requires an explicit
`mechanism_status` (`supported`, `refuted`, or `insufficient_evidence`); missing
status returns `insufficient_evidence` and blocks promotion. This keeps the
adapter aligned with fact-level/tool-verification research without turning an
LLM-as-judge or source lookup into a mandatory dependency. The first real
source-backed TruthfulQA mechanism artifacts now cover two question families:
`record-10` and the Africa poverty trend records
`record-133`/`record-165`/`record-274`/`record-299`; a final mixed-status
artifact covers Bill Gates high-school records `record-27`/`record-134` and UFO
extraterrestrial-premise records `record-212`/`record-224`. All nine
causal/procedural rows now have citation-backed mechanism bindings.

Mechanism inputs now also have a source-backed fill boundary:

```bash
MECHANISM_FILL=artifacts/truthfulqa-frontier-smollm2-l80-unresolved-world-model-rule-mechanism-binding-fill

python benchmarks/fill_world_model_rule_inputs_from_mechanism_bindings.py \
  --input-tasks "$MECHANISM_FILL/source-backed-mechanism-rule-input-tasks.jsonl" \
  --mechanism-bindings "$MECHANISM_FILL/source-backed-mechanism-bindings.jsonl" \
  --output-dir "$MECHANISM_FILL" \
  --registry artifacts/local-release-registry.json \
  --name truthfulqa-frontier-smollm2-l80-unresolved-world-model-rule-mechanism-binding-fill \
  --version 0.1
```

This fill script is intentionally stricter than the adapter shell: it blocks
missing source citations, unreviewed bindings, and missing or invalid
`mechanism_status` values before adapter execution. Focused tests cover a
supported mechanism that fills, executes, and promotes, plus an invalid binding
that blocks.

For local rebuilds of the frontier audit mechanism lane, the canonical shortcut
is the source workflow below. It materializes the three source-backed mechanism
input groups, then runs `binding-fill -> adapter -> promotion-gate ->
ProductTrace handoff` for all nine causal/procedural rows:

```bash
python benchmarks/run_frontier_mechanism_handoff_source_workflow.py \
  --output-root artifacts \
  --registry artifacts/local-release-registry.json \
  --name truthfulqa-frontier-smollm2-l80-mechanism-handoff-source-workflow \
  --version 0.1 \
  --metadata source=frontier_mechanism_handoff_rebuild
```

The registered `record-10` diamond mechanism chain is
`filled -> observed -> promote -> handoff`: the fill script consumes a
source-backed WTAMU/GIA mechanism binding, the adapter observes one supported
`mechanism_consistency` candidate, the promotion gate promotes `1/1`, and
`build_world_model_rule_candidate_handoff.py` writes one ProductTrace with
`accept/low` plus a dry-run accept action. All four manifests verify. This
proves the mechanism lane can enter ProductTrace for one cited mechanism; it
does not claim broad causal/procedural coverage.

The registered Africa poverty mechanism chain applies a World Bank-backed
rate/headcount mechanism to four repeated TruthfulQA records
(`record-133`, `record-165`, `record-274`, and `record-299`). The source-backed
binding states that poverty rates can decline while the number of poor people
rises when population growth outpaces the rate decline. The chain is also
`filled -> observed -> promote -> handoff`: `4/4` inputs fill, `4/4`
`mechanism_consistency` candidates execute as supported, the promotion gate
promotes `4/4`, and the handoff writes four ProductTrace rows with `accept/low`
plus dry-run accept actions. All four manifests verify. Combined with the
diamond row, the registered mechanism lane now covers `5/9` causal/procedural
input tasks across two source-backed mechanism families.

The final remaining mechanism chain fills the Bill Gates and UFO rows. The Bill
Gates bindings use Academy of Achievement plus Gates Foundation/Lakeside
biographical evidence and promote as `supported`; the UFO bindings use NASA UAP
FAQ/report and AARO historical-report evidence and promote as `refuted` because
the question's premise asserts an established extraterrestrial truth. The chain
is `filled -> observed -> promote -> handoff`: `4/4` inputs fill, `2` supported
and `2` refuted `mechanism_consistency` candidates execute and promote, and the
handoff writes two `accept/low` and two `abstain/high` dry-run ProductTrace
actions. All four manifests verify. The registered mechanism lane now covers
`9/9` causal/procedural input tasks across four source-backed mechanism
families.

Those three promoted mechanism handoffs can now be bundled as one release-gate
artifact:

```bash
python benchmarks/build_mechanism_handoff_evidence_bundle.py \
  --handoff artifacts/truthfulqa-frontier-smollm2-l80-unresolved-world-model-rule-mechanism-candidate-handoff/world-model-rule-candidate-handoff.json \
  --handoff artifacts/truthfulqa-frontier-smollm2-l80-unresolved-world-model-rule-mechanism-africa-poverty-candidate-handoff/world-model-rule-candidate-handoff.json \
  --handoff artifacts/truthfulqa-frontier-smollm2-l80-unresolved-world-model-rule-mechanism-remaining-candidate-handoff/world-model-rule-candidate-handoff.json \
  --output-dir artifacts/truthfulqa-frontier-smollm2-l80-mechanism-handoff-evidence-bundle \
  --registry artifacts/truthfulqa-frontier-smollm2-l80-mechanism-handoff-evidence-bundle/registry.json \
  --name truthfulqa-frontier-smollm2-l80-mechanism-handoff-evidence-bundle \
  --version 0.1 \
  --expected-target-count 9 \
  --min-trace-count 9 \
  --min-supported-count 7 \
  --min-refuted-count 2
```

The current bundle promotes with `9/9` target coverage, `7` supported and `2`
refuted traces, `7` accept and `2` abstain actions, four source-family buckets,
and recursive verification of the three child handoff manifests. It can be
passed to release candidate comparison with `--mechanism-handoff-evidence-bundle`
or by registry key; `frontier_audit` defaults to
`report:truthfulqa-frontier-smollm2-l80-mechanism-handoff-evidence-bundle:0.1`.

The same reduced 12-task queue was also replayed through Crossref with a wider
scholarly budget:

```bash
OUT=artifacts/truthfulqa-frontier-smollm2-l80-crossref-reduced-scholarly-catalog

python benchmarks/run_crossref_source_family_catalog_adapter.py \
  --tasks artifacts/truthfulqa-frontier-smollm2-l80-worldbank-source-family-catalog-collection-plan/source-family-catalog-collection-tasks.jsonl \
  --output "$OUT/crossref-reduced-scholarly-catalog.jsonl" \
  --report-json "$OUT/crossref-reduced-scholarly-catalog-report.json" \
  --artifact-manifest "$OUT/artifact-manifest.json" \
  --registry artifacts/local-release-registry.json \
  --name truthfulqa-frontier-smollm2-l80-crossref-reduced-scholarly-catalog \
  --version 0.1 \
  --max-query-variants 8 \
  --rows-per-query 5 \
  --min-delay-seconds 0.2 \
  --metadata suite=truthfulqa_frontier_smollm2_l80 \
  --metadata source=worldbank_source_family_catalog_collection_plan
```

That reduced Crossref pass consumes `6` scholarly tasks, runs `48` query
variants, writes `69` deduplicated scholarly catalog documents, and records `0`
request errors. Rerunning the source-family workflow with Wikidata reference,
the original Crossref catalog, World Bank official statistics, and this reduced
Crossref catalog keeps the route blocked, but improves source-family coverage:
the workflow sees `626` catalog docs, returns `528` adapter rows, and the
coverage audit drops missing target rows from `84` to `44`. Covered target
families are now `official_statistics=4` and `scholarly=140`; remaining missing
targets are `official=36`, `scholarly=16`, and `news=4`. The next collection
plan is down to `9` tasks: `official=5`, `scholarly=3`, and `news=1`.

## `run_official_site_source_family_catalog_adapter.py`

Fetches URL-seeded official pages for `source_family=official` collection tasks.
This is deliberately not a web-search adapter: the URL seed file is the
auditable handoff from human review, a search provider, or a later source
discovery system. The adapter fetches HTML when possible, falls back to seed
title/text when a site blocks automated access, emits adapter-ready official
catalog rows, and keeps request ids, labels, target ids, row ids, and model
answers out of source documents.

```bash
OUT=artifacts/truthfulqa-frontier-smollm2-l80-official-site-catalog

python benchmarks/run_official_site_source_family_catalog_adapter.py \
  --tasks artifacts/truthfulqa-frontier-smollm2-l80-reduced-source-family-catalog-collection-plan/source-family-catalog-collection-tasks.jsonl \
  --seeds "$OUT/official-site-url-seeds.jsonl" \
  --output "$OUT/official-site-catalog.jsonl" \
  --report-json "$OUT/official-site-catalog-report.json" \
  --artifact-manifest "$OUT/artifact-manifest.json" \
  --registry artifacts/local-release-registry.json \
  --name truthfulqa-frontier-smollm2-l80-official-site-catalog \
  --version 0.1 \
  --max-text-chars 6000 \
  --timeout-seconds 30 \
  --min-delay-seconds 0.5 \
  --metadata suite=truthfulqa_frontier_smollm2_l80 \
  --metadata source=reduced_source_family_catalog_collection_plan
```

The registered official-site run consumes the `5` remaining official collection
tasks with `9` URL seeds across USDA ERS, Tesla, WHO, World Bank, and NOAA. It
writes `9` official catalog docs, successfully fetches `7` pages, and records
`2` Tesla access-denied errors while retaining seed-title fallback rows. The
manifest verifies and catalog/source-doc reserved-field scans are clean.

Adding this official-site catalog to the Wikidata, Crossref, reduced Crossref,
and World Bank catalogs still leaves route promotion blocked, but it improves
source-family coverage: the workflow has `635` catalog docs, returns `528`
adapter rows, and the coverage audit reduces missing target rows from `44` to
`28`. Covered target families are now `official=32`, `official_statistics=4`,
and `scholarly=128`; remaining missing targets are `official=4`,
`scholarly=28`, and `news=4`. The next acquisition plan compresses to `7`
tasks: `scholarly=5`, `official=1`, and `news=1`.

The follow-up OpenAlex plus source-family-diverse rerank pass supersedes that
queue for scholarly/official coverage, and the seeded-news pass closes the final
source-family queue: all requested non-fallback source families are now covered,
while route promotion remains fail-closed behind query-sweep/comparison gates.

## `run_wikipedia_citation_search_adapter.py`

Runs a dependency-free MediaWiki/Wikipedia search adapter for sanitized
citation/search requests. It writes the JSONL result schema expected by
`run_external_citation_search_adapter_workflow.py`, with query de-duplication,
optional alternate-query fallback (`--max-query-variants`), global rate
limiting, retries, snippets, and optional page extracts.

```bash
python benchmarks/run_wikipedia_citation_search_adapter.py \
  --input artifacts/truthfulqa-frontier-smollm2-l80-citation-search-adapter-handoff/citation-search-adapter-requests.jsonl \
  --output artifacts/wikipedia-citation-search-results.jsonl \
  --max-results 3 \
  --workers 1 \
  --min-delay-seconds 1 \
  --retries 4 \
  --max-query-variants 3
```

The registered SmolLM2 L80 run uses this command through the external adapter
workflow. It returned `504` Wikipedia result documents for `168/176` sanitized
requests, passed external-candidate provenance, then correctly remained
`blocked` because the blind-spot query sweep refuted `0/89` entrenched false
answers and the controlled-vs-external comparison showed a `1.0` generalization
gap. This is source collection evidence, not a promoted grounding route.

The claim/entity-aware rerun is registered as
`report:truthfulqa-frontier-smollm2-l80-claim-entity-wikipedia-citation-search-adapter-workflow:0.1`.
With `--query-mode claim_entity`, `--max-alternate-queries 3`, and adapter
`--max-query-variants 3`, it returned `528` Wikipedia result documents for
`176/176` sanitized requests and again passed external-candidate provenance. It
also reduced exact model-answer copy rate from the question-only run's `0.310`
to `0.235`. The route still remains `blocked`: the external query sweep refuted
`0/89` entrenched false answers and the controlled-vs-external comparison kept a
`1.0` generalization gap. Treat this as evidence that query fallback improves
coverage and reduces answer echo, but not as a correction route; the next
retrieval work should move to source-family or structured official-source
adapters behind the same command boundary.

## `run_external_citation_search_adapter_workflow.py`

Runs the command-boundary version of the citation/search adapter path. The
workflow first writes sanitized request JSONL from the unresolved queue, invokes
a local external command without a shell, then feeds the returned result JSONL
through `run_citation_search_evidence_workflow.py`.

```bash
OUT=artifacts/truthfulqa-frontier-smollm2-l80-claim-entity-wikipedia-citation-search-adapter-workflow

python benchmarks/run_external_citation_search_adapter_workflow.py \
  --queue artifacts/truthfulqa-frontier-smollm2-l80-unresolved-blind-spot-evidence-queue/unresolved-evidence-queue.json \
  --batch-id unresolved-evidence-batch-0001 \
  --query-mode claim_entity \
  --max-alternate-queries 3 \
  --search-command "python benchmarks/run_wikipedia_citation_search_adapter.py --input {input} --output {output} --max-results 3 --max-query-variants 3 --workers 1 --min-delay-seconds 1 --retries 4" \
  --scores artifacts/smollm2_truthfulqa_l80_scores_with_statements.json \
  --blind-spots artifacts/truthfulqa-frontier-qwen-smollm2-l80-detectability-blind-spots/smollm2-l80-entrenched-blind-spots.json \
  --controlled-sweep artifacts/truthfulqa-frontier-smollm2-l80-blind-spot-query-sweep/blind-spot-query-sweep.json \
  --output-dir "$OUT" \
  --registry artifacts/local-release-registry.json \
  --name truthfulqa-frontier-smollm2-l80-claim-entity-wikipedia-citation-search-adapter-workflow \
  --version 0.1
```

The command must include both `{input}` and `{output}` placeholders. `{input}`
receives the sanitized adapter-request JSONL; `{output}` is the result JSONL
path the adapter must write. Returned snippets still have to pass provenance,
blind-spot query, and controlled-vs-external gates before any route decision is
promoted. Passing `--batch-id` keeps the preflight request JSONL and downstream
evidence gate aligned to the same unresolved evidence batch, which is the
preferred path for long-running external search jobs.

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
groundedness fallback. Multi-snippet slot coverage is accepted only when the
selected evidence is linked by source, metadata, claim id, or subject anchor;
use `--triple-min-slot-coverage` to relax or tighten the per-slot evidence
coverage threshold. Triple audit metadata records each slot's expected value,
matched and missing tokens, source/evidence label, plus claim-level slot
coverage summaries for release review.
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

## `run_uncertainty_escalation_workflow.py` and `eval_uncertainty_escalation.py`

`run_uncertainty_escalation_workflow.py` runs a local claim/evidence fixture
through `run_verification_loop(..., escalation_policy=...)`, writes
`VerificationLoopResult.to_dict()` rows to JSONL, and emits an escalation report.
Use it before changing defaults to measure whether low-confidence verification
escalation is buying useful evidence or only adding retrieval/tool cost.

```bash
python benchmarks/run_uncertainty_escalation_workflow.py \
  --records artifacts/uncertainty-escalation-fixture-workflow/uncertain-verification-fixture.json \
  --output-dir artifacts/uncertainty-escalation-workflow \
  --json artifacts/uncertainty-escalation-workflow/uncertainty-escalation-workflow.json \
  --loop-results-jsonl artifacts/uncertainty-escalation-workflow/verification-loop-results.jsonl \
  --artifact-manifest artifacts/uncertainty-escalation-workflow/artifact-manifest.json \
  --verification-report artifacts/uncertainty-escalation-workflow/manifest-verification.json \
  --registry artifacts/uncertainty-escalation-workflow/registry.json \
  --min-confidence 0.65 \
  --retriever-min-overlap 0.2
```

Fixture rows may include `claim`, optional `label`, `preliminary_status`,
`preliminary_confidence`, `retrieval_documents`, optional `refutations`, and
optional `diagnostics`. The workflow keeps diagnostics low by default, so
decision changes mostly reflect the cheap verifier -> retrieval -> final
verifier path.

When claims carry extractor metadata such as `entity_candidates`,
`VerificationEscalationPolicy` can also treat entity-bearing medium-confidence
results as uncertainty-escalation candidates. The report records
`entity_sensitive_records`, `entity_sensitive_claim_total`, and
`entity_candidate_total` so entity-level verification pressure can be separated
from ordinary low-confidence escalation.

The committed fixture in `artifacts/uncertainty-escalation-fixture-workflow/`
contains four records. All four low-confidence preliminary results trigger
escalation and retrieve evidence; the two false claims move from `accept` to
`abstain`, reducing the fixture false-accept rate from `1.0` to `0.0`. The
artifact manifest verifies the fixture, loop-result JSONL, and workflow report.

`eval_uncertainty_escalation.py` replays an existing loop-result sidecar without
rerunning the control loop:

```bash
python benchmarks/eval_uncertainty_escalation.py \
  --results artifacts/uncertainty-escalation-workflow/verification-loop-results.jsonl \
  --label-key label \
  --json artifacts/uncertainty-escalation-report.json
```

Rows may be raw loop-result payloads or wrappers such as
`{"label": 0, "result": {...}}`. Labels are optional; when present, `0` means
true/normal and `1` means false/anomalous. The report includes escalation trigger
rates, selected route counts, retrieval request/success/evidence rates, decision
transitions, and label-conditioned false-accept/selective-accuracy deltas.

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
subject/predicate pair is covered by the fact corpus. The verifier can also be
constructed with a custom `ClaimTripleExtractor`, so stronger regex/template or
future learned extractors can be evaluated behind the same route contract:

```bash
python benchmarks/eval_verifier_ensemble.py \
  --scores facts=artifacts/wikidata-country-core-facts-structured-fact-route/covered-facts-scores.json \
  --fact-corpus artifacts/wikidata-country-core-facts-external-corpus/wikidata-country-core-facts-qa-corpus.json \
  --signal truth_proj \
  --json artifacts/wikidata-country-core-facts-structured-fact-route/structured-fact-verifier-report.json
```

Before promoting a new extractor into a verifier route, use
`eval_triple_extraction.py` on labeled extraction fixtures:

```bash
python benchmarks/eval_triple_extraction.py \
  --records benchmarks/fixtures/triple_extraction_records.json \
  --extractor regex_rule_based \
  --patterns benchmarks/fixtures/triple_extraction_regex_patterns.json \
  --json artifacts/triple_extraction_eval.json
```

External or learned extractors should be evaluated through offline prediction
files first. The file may be JSON or JSONL, with one record per claim keyed by
`claim_id`/`id`, `text`/`claim`/`statement`, or an explicit `key`, and a
`triples` list containing `subject`, `predicate`, and `object` mappings:

```bash
python benchmarks/eval_triple_extraction.py \
  --records artifacts/triple_extraction_records.json \
  --extractor external_predictions \
  --predictions artifacts/my_external_extractor_predictions.jsonl \
  --json artifacts/my_external_extractor_triple_eval.json
```

This keeps learned extractors, OpenIE systems, or LLM JSON extraction behind a
local file boundary. If the external system extracts triples from negated,
quoted, temporal, ambiguous, or metalinguistic controls, the same evaluation
counts those outputs as false positives.

Use `run_external_triple_extractor_handoff.py` when the external extractor is a
local command rather than a prewritten prediction file. The workflow writes a
label-free request JSONL with `claim_id` and `text` by default, invokes the
command without a shell, evaluates the generated predictions, gates precision,
recall, F1, and false-positive rate, and can write a manifest plus verification
report:

```bash
python benchmarks/run_external_triple_extractor_handoff.py \
  --records artifacts/my-triple-fixture-records.json \
  --extractor-command "python path/to/extractor.py --input {input} --output {output}" \
  --output-dir artifacts/external-triple-extractor-handoff \
  --json artifacts/external-triple-extractor-handoff/external-triple-extractor-handoff.json \
  --artifact-manifest artifacts/external-triple-extractor-handoff/artifact-manifest.json \
  --verification-report artifacts/external-triple-extractor-handoff/manifest-verification.json
```

The request file intentionally excludes expected triples unless
`--include-metadata` is set, so a learned/OpenIE/LLM extractor is not handed the
labels it is being evaluated against. Prediction rows may use `triples: []` to
represent no extracted triples, which is the expected output for adversarial
negative controls. The generated prediction file can then be passed to
`run_triple_extraction_fixture_workflow.py --external-predictions NAME=PATH` or
to the matrix form `--external-predictions CORPUS:NAME=PATH`.

Use `run_external_triple_extractor_matrix_handoff.py` when the same external
command should be evaluated across the cross-corpus/adversarial matrix. It
first builds deterministic per-corpus fixture records, sends label-free requests
to each configured command, gates every returned prediction file, then feeds the
prediction paths into `run_triple_extraction_fixture_matrix.py` so release gates
can require external-prediction count, corpus coverage, and mean best external
F1:

```bash
python benchmarks/run_external_triple_extractor_matrix_handoff.py \
  --corpus country-core=artifacts/wikidata-country-core-facts-external-corpus/wikidata-country-core-facts-source.jsonl \
  --corpus organization-product=artifacts/wikidata-organization-product-core-facts/wikidata-organization-product-source.jsonl \
  --external-extractor-command "learned=python path/to/extractor.py --input {input} --output {output}" \
  --output-dir artifacts/external-triple-extractor-matrix-handoff \
  --min-distinct-predicates 6 \
  --adversarial-negatives-per-fact 1 \
  --max-external-false-positive-rate 0.0 \
  --verification-report artifacts/external-triple-extractor-matrix-handoff/manifest-verification.json
```

This wrapper still does not make the extractor a dependency. It records a local
command boundary and matrix-level artifact chain; any quality claim depends on
the actual external extractor run promoting under the configured gates.

Use `build_triple_extraction_fixture.py` to turn structured fact corpora, such
as the output of `build_wikidata_qa_corpus.py`, into larger labeled extraction
fixtures plus matching default regex patterns:

```bash
python benchmarks/build_triple_extraction_fixture.py \
  --fact-corpus artifacts/wikidata-country-core-facts-qa-corpus.json \
  --output-records artifacts/triple_extraction_records.json \
  --output-patterns artifacts/triple_extraction_regex_patterns.json \
  --adversarial-negatives-per-fact 0 \
  --predicate-confusions-per-fact 0 \
  --non-assertive-negatives-per-fact 0 \
  --ambiguity-negatives-per-fact 0 \
  --temporal-negatives-per-fact 0 \
  --metalinguistic-negatives-per-fact 0 \
  --artifact-manifest artifacts/triple_extraction_fixture_manifest.json
```

Use `run_triple_extraction_fixture_workflow.py` when the generated fixture should
become release evidence. It writes the generated records, default regex
patterns, per-extractor reports, a promotion summary, and an artifact manifest:

```bash
python benchmarks/run_triple_extraction_fixture_workflow.py \
  --fact-corpus artifacts/wikidata-country-core-facts-qa-corpus.json \
  --output-dir artifacts/triple-extraction-fixture-workflow \
  --external-predictions learned=artifacts/my_external_extractor_predictions.jsonl \
  --adversarial-negatives-per-fact 0 \
  --predicate-confusions-per-fact 0 \
  --non-assertive-negatives-per-fact 0 \
  --ambiguity-negatives-per-fact 0 \
  --temporal-negatives-per-fact 0 \
  --metalinguistic-negatives-per-fact 0
```

Use `run_triple_extraction_fixture_matrix.py` when extractor templates need
cross-corpus release evidence. The matrix runs the same workflow per corpus and
blocks promotion unless enough corpora promote and the generated fixtures cover
enough distinct predicates. Add `--external-predictions CORPUS:NAME=PATH` to
evaluate learned/OpenIE/LLM-json prediction files for one corpus in the same
matrix run; `CORPUS` may be either the configured corpus name or its slug, and
the external files are recorded in both per-corpus and matrix manifests. Add
`--adversarial-negatives-per-fact N` to include
negated near-miss records with no expected triples, and
`--max-adversarial-false-positive-rate` to fail closed when an extractor emits
triples for those negative controls. Add `--predicate-confusions-per-fact N`
to require the extractor to emit the predicate stated by a wrong-predicate
claim, and `--non-assertive-negatives-per-fact N` to reject quoted or
questioned fact mentions. `--ambiguity-negatives-per-fact N`,
`--temporal-negatives-per-fact N`, and
`--metalinguistic-negatives-per-fact N` add ambiguous/multi-object, temporal,
and phrase/comparison context controls:

```bash
python benchmarks/run_triple_extraction_fixture_matrix.py \
  --corpus country-core=artifacts/wikidata-country-core-facts-external-corpus/wikidata-country-core-facts-source.jsonl \
  --corpus organization-product=artifacts/wikidata-organization-product-core-facts/wikidata-organization-product-source.jsonl \
  --output-dir artifacts/wikidata-cross-corpus-triple-extraction-fixture-matrix \
  --external-predictions country-core:learned=artifacts/country-core-extractor-predictions.jsonl \
  --external-predictions organization-product:learned=artifacts/organization-product-extractor-predictions.jsonl \
  --min-corpora 2 \
  --min-distinct-predicates 6 \
  --adversarial-negatives-per-fact 0 \
  --predicate-confusions-per-fact 0 \
  --non-assertive-negatives-per-fact 0 \
  --ambiguity-negatives-per-fact 0 \
  --temporal-negatives-per-fact 0 \
  --metalinguistic-negatives-per-fact 0
```

The current Wikidata cross-corpus matrix promotes: country-core contributes
`1436` generated records over `capital_of`, `official_language_of`, and
`currency_of`; organization/product contributes `32` records over
`headquarters_location_of`, `manufacturer_of`, and `inception_of`; the matrix
records `mean_best_f1=1.000`, `mean_f1_lift=0.625`, and passes recursive
manifest verification.

The adversarial companion matrix at
`artifacts/wikidata-cross-corpus-triple-extraction-adversarial-matrix/` enables
one record per fact for each of six controls: negated near-miss,
predicate-confusion, non-assertive quoted/questioned mention, ambiguity,
temporal qualification, and metalinguistic/comparison context. It promotes after
applying the same blocked-context guard to regex and rule-based paths:
country-core has `3590` records with `359` records in each adversarial
subgroup, organization/product has `80` records with `8` records in each
subgroup, both corpora keep best F1 `1.000`, predicate-confusion F1 is `1.000`,
and every zero-expected subgroup false-positive rate is `0.000`. This is still
covered-KG template evidence, not a broad open-domain extractor claim; remaining
work is broader corpora, richer surface variation, and learned/external
extractor adapters.

`triple_extraction_smoke.py` runs the bundled fixture through `rule_based`,
`regex_rule_based`, and `composite` extractors and asserts that the augmented
paths improve exact F1 before the benchmark gates pass:

```bash
python benchmarks/triple_extraction_smoke.py \
  --output-dir artifacts/triple-extraction-smoke
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
optional `state_checks`, optional `state_transitions`, and optional
`world_model_rules` fields, or a SQLite state-source spec:

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
runner routes these records through `StateTransitionVerifier`, which applies a
world-model transition and checks the resulting postcondition before falling
back to static state or lexical verification. When the state source includes
`world_model_rules`, the runner uses `RuleBasedWorldModelAdapter`; otherwise it
keeps the legacy in-memory `set` / `increment` / `decrement` action updates:

```bash
python benchmarks/build_transition_fixture.py \
  --scores-output artifacts/order_transition_scores.json \
  --claims-output artifacts/order_transition_claims.json \
  --state-output artifacts/order_transition_state.json \
  --n-records 12 \
  --rule-based-world-model

python benchmarks/eval_verifier_ensemble.py \
  --scores transitions=artifacts/order_transition_scores.json \
  --claims artifacts/order_transition_claims.json \
  --state-source artifacts/order_transition_state.json \
  --min-world-model-confidence 0.8 \
  --signal truth_proj \
  --json artifacts/order_transition_verifier_ensemble_report.json
```

To exercise world-model disagreement explicitly, generate an ensemble fixture.
This writes a `world_model_ensemble` state-source block with three rule-based
members: two baseline members and one controlled stress member that diverges on
the false-labeled transition records. The verifier uses
`EnsembleWorldModelAdapter`, and the verified-record sidecar preserves
`agreement_rate`, `below_min_agreement`, and related metadata for score-dump
conversion:

```bash
python benchmarks/build_transition_fixture.py \
  --scores-output artifacts/order_transition_ensemble_scores.json \
  --claims-output artifacts/order_transition_ensemble_claims.json \
  --state-output artifacts/order_transition_ensemble_state.json \
  --n-records 12 \
  --world-model-ensemble \
  --world-model-ensemble-min-agreement 0.75 \
  --world-model-ensemble-strategy policy_replay
```

The default `label_stress` strategy preserves the original controlled
false-record stress pattern. `policy_replay` instead makes the stress member
apply a conservative high-quantity reservation policy, producing disagreement
on quantity-driven records that include both true and false labels.

This fixture checks action-consequence verification: true labels match the
predicted inventory after reservation, while false labels assert an off-by-one
postcondition that the predicted state refutes. `--min-world-model-confidence`
fails closed on low-confidence transition predictions. The selected
world-model adapter, rule count, confidence threshold, and rule payload are
reported with each run. Verified-record sidecars also include
`world_model_reference`, `world_model_view`, and refuted-postcondition
`world_model_conflict` metadata, so downstream audits can see which reference
world, state paths, action, predicted state fingerprint, and expected/actual
postcondition values drove the transition decision.
recorded in the report and verified-record trace-cache key.

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
sample-collection preflight plans, ensemble report, optional
geometry-by-selfcheck fusion artifacts, and artifact manifest verification:

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
The workflow now writes per-run sample-collection plan artifacts, so that same
negative replay records the exact missing-record list and minimum new-sample
budget before rerunning INSIDE sampling or external sample generation.

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
  --n-records 12 \
  --rule-based-world-model

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
  --require-non-oracle-evidence \
  --require-retrieval-provenance-filter \
  --required-retrieval-source-prefix external: \
  --required-retrieval-metadata corpus_role=grounding \
  --min-retrieval-filter-score 0.5 \
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
Use `--require-retrieval-provenance-filter` when a retrieval route should prove
that untrusted hits were gated before verifier handoff. Optional
`--required-retrieval-source-prefix`, `--required-retrieval-metadata`, and
`--min-retrieval-filter-score` checks compare the recorded manifest or claims
fixture provenance filter against the expected source, metadata, and score
policy. Missing filter metadata blocks the baseline.
Use `--require-retrieval-stress-control` for retrieval-grounding baselines. It
requires an answer-echo stress artifact manifest, verifies that manifest, checks
the corpus type is `retrieval_stress_answer_echo`, and fails closed unless the
stress run exposes self-support with high `false_supported_rate` and low
`false_refuted_rate`. This prevents answer-derived retrieval evidence from being
promoted as grounding.

## `compare_external_evidence_baselines.py`

Combines the route gate above with a text/length redline check for external or
domain-shifted evidence candidates. It does not rerun models or verifier
adapters; it reads registered route manifests plus saved `eval_score_ensemble.py`
reports and writes one fail-closed comparison artifact.

```bash
python benchmarks/compare_external_evidence_baselines.py \
  --route-registry artifacts/registry.json \
  --route-baseline-key benchmark_manifest:external-retrieval-route:0.1 \
  --require-route-baseline \
  --min-decision-accuracy 0.95 \
  --max-false-supported-rate 0.02 \
  --min-false-refuted-rate 0.90 \
  --require-non-oracle-evidence \
  --require-retrieval-provenance-filter \
  --required-retrieval-source-prefix external: \
  --required-retrieval-metadata corpus_role=grounding \
  --min-retrieval-filter-score 0.5 \
  --require-retrieval-stress-control \
  --retrieval-stress-manifest artifacts/truthfulqa-l80-answer-echo-retrieval-stress/artifact-manifest.json \
  --min-stress-false-supported-rate 0.90 \
  --max-stress-false-refuted-rate 0.05 \
  --candidate-score-report artifacts/external-retrieval-fusion/score-ensemble-report.json \
  --text-baseline-report artifacts/truthfulqa-l80-text-baseline-comparison/score-ensemble-report.json \
  --require-text-redline \
  --min-text-detection-margin 0.10 \
  --min-text-auroc-margin 0.10 \
  --json artifacts/external-evidence-baseline-comparison.json \
  --artifact-manifest artifacts/external-evidence-baseline-comparison-manifest.json \
  --verification-report artifacts/external-evidence-baseline-comparison-verification.json \
  --registry artifacts/registry.json \
  --name external-evidence-comparison \
  --version 0.1 \
  --fail-on-blocked
```

The route side delegates to `compare_route_baselines.py`, including the
answer-echo stress-control audit. The text-redline side pairs runs by name,
selects the best non-text candidate signal/fusion at the report `best_alpha`,
selects the best cheap text baseline from `single_results`, and blocks unless
the candidate clears the configured detection/AUROC margins. Missing reports,
ambiguous run pairing, or non-finite metrics block the comparison. With
`--artifact-manifest` and `--registry`, the comparator writes a recursive
manifest over the route, stress-control, score-ensemble, and comparison reports,
verifies it, and registers the comparison as a reusable `report:*:*` release
gate input.
When external evidence is a structured KG correction route rather than lexical
retrieval, add `--require-covered-facts-route`, optional repeated
`--covered-fact-route structured_qa` / `structured_fact`, and
`--min-covered-fact-*` thresholds. That gate requires the selected route
baseline to expose a promoted `wikidata_structured_qa_route_workflow` summary,
records the covered source-document/true/false counts, and can also require
per-property minima with `--min-covered-fact-properties`,
`--min-covered-fact-property-records`,
`--min-covered-fact-property-decision-accuracy`,
`--max-covered-fact-property-false-supported-rate`, and
`--min-covered-fact-property-false-refuted-rate`. This keeps the claim scoped
to covered facts and prevents a strong aggregate score from hiding a weak
predicate/property slice.

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
subject, predicate, and object slots for sensitive factual claims. Missing or
unlinked slot coverage returns insufficient evidence rather than a direct
refutation, so use a false-supported gate and disable the false-refuted
requirement for this audit family. The route report preserves per-slot evidence
details so failures can be traced to subject, predicate, object, or evidence-link
gaps rather than only a final route status:

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
quality signal for the next run. The readiness report and manifest also lift the
`state_transition` family world-model adapter and rule count into top-level
metadata, so downstream release gates can require `RuleBasedWorldModelAdapter`
evidence without reparsing route internals. Use `verify_artifact_manifest.py --recursive` and
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
Optional context-sensitivity evidence can be supplied with
`--context-sensitivity-workflow` or `--context-sensitivity-workflow-key`; the
gate verifies the workflow manifest and requires paired logprob, enriched
sidecar, and enhanced score-dump evidence.

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
  --context-sensitivity-workflow-key report:<context-sensitivity-workflow-name>:<version> \
  --uncertainty-escalation-workflow-key report:<uncertainty-escalation-workflow-name>:<version> \
  --min-uncertainty-escalation-records 4 \
  --min-uncertainty-escalation-trigger-rate 0.50 \
  --min-uncertainty-escalation-retrieval-evidence-rate 0.50 \
  --max-uncertainty-escalation-final-false-accept-rate 0.05 \
  --max-uncertainty-escalation-false-accept-delta 0.0 \
  --release-efficiency-report artifacts/product-runtime-profile-sweep/release-efficiency-report.json \
  --external-evidence-baseline-comparison artifacts/external-evidence-baseline-comparison.json \
  --mechanism-handoff-evidence-bundle artifacts/truthfulqa-frontier-smollm2-l80-mechanism-handoff-evidence-bundle/mechanism-handoff-evidence-bundle.json \
  --route-baseline-key benchmark_manifest:truthfulqa-l80-structured-qa-staged-route:0.4 \
  --required-route-baseline-key benchmark_manifest:<local-retrieval-route-name>:<version> \
  --adapter-family-matrix artifacts/adapter_family_matrix/adapter-family-matrix.json \
  --adapter-family-profile strict_audit \
  --triple-extraction-fixture-matrix artifacts/triple-extraction-fixture-matrix/triple-extraction-fixture-matrix.json \
  --min-triple-extraction-corpora 2 \
  --min-triple-extraction-distinct-predicates 6 \
  --min-triple-extraction-external-prediction-count 2 \
  --min-triple-extraction-external-prediction-corpora 2 \
  --min-triple-extraction-mean-best-external-f1 0.90 \
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
  --required-route-require-retrieval-provenance-filter \
  --required-route-required-retrieval-source-prefix external: \
  --required-route-required-retrieval-metadata corpus_role=grounding \
  --required-route-min-retrieval-filter-score 0.5 \
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
present. Add `--required-route-require-retrieval-provenance-filter`,
`--required-route-required-retrieval-source-prefix`,
`--required-route-required-retrieval-metadata`, and
`--required-route-min-retrieval-filter-score` when the required retrieval route
must prove a specific source/metadata/score filter policy before evidence can
enter verifier claims. Add `--required-route-require-retrieval-stress-control`
when the route must also prove the answer-echo negative control fails as
expected. Otherwise the release only checks the route's already-registered
promotion status and manifest validity. This keeps selected product-route budgets such as
`--max-retrieval-use-rate 0.0` separate from audit routes that intentionally use
retrieval or world-model adapters. For `structured_fact`, use two required route
keys, or `--release-policy-profile strict_structured_fact` with
`--structured-fact-canonical-route-key` and
`--structured-fact-paraphrase-route-key`, to require both the canonical
covered-facts route and the paraphrase robustness replay before a release can
promote. Covered-fact route gates can also require per-property evidence with
`--required-route-min-covered-fact-properties`,
`--required-route-min-covered-fact-property-records`,
`--required-route-min-covered-fact-property-source-documents`,
`--required-route-min-covered-fact-property-decision-accuracy`,
`--required-route-max-covered-fact-property-false-supported-rate`, and
`--required-route-min-covered-fact-property-false-refuted-rate`. Promoted
release candidates carry the selected route's covered-property ids and the
required-route record-to-property coverage summary into the comparison report,
final manifest, and registry metadata. Available release policy profiles are `research_smoke`,
`candidate_release`, `strict_structured_fact`, and `frontier_audit`; profile
defaults only fill unset values, so explicit thresholds still win. Direct

Use `--uncertainty-escalation-workflow` or
`--uncertainty-escalation-workflow-key` when a release must prove that
low-confidence verification results escalate into additional retrieval/action
evidence before final acceptance. The optional thresholds gate minimum record
count, escalation trigger rate, retrieval evidence rate, final false-accept
rate, and false-accept-rate delta; a threshold of `0.0` for
`--max-uncertainty-escalation-false-accept-delta` means escalation must not make
false accepts worse.
`compare_release_candidates.py` reports record `release_policy_profile` and
`release_policy_profile_applied_defaults` in `config`. `frontier_audit` also
defaults `--max-recommended-runtime-seconds 1.0`, leaving the older
`--max-uncached-forward-seconds` cold-start gate opt-in for callers that want it,
and defaults `--require-product-runtime-drift-promotion-evidence`,
`--require-product-runtime-drift-pre-generation-evidence`,
`--require-product-runtime-drift-counterfactual-evidence`,
`--require-product-runtime-drift-triple-audit-evidence`,
`--require-product-runtime-drift-covered-fact-property-evidence`,
`--require-product-runtime-drift-action-gate-evidence`,
  `--require-product-runtime-drift-trajectory-audit-evidence`,
  `--require-product-runtime-drift-evidence-handoff-evidence`,
  `--require-product-runtime-drift-world-model-evidence`,
  `--require-product-runtime-drift-context-sensitivity-evidence`,
  `--require-product-runtime-drift-frontier-release-evidence`,
`--require-product-trace-action-audit-gate`, and
`--require-product-trace-action-execution-gate`; it also defaults to the
registered covered-facts external-evidence handoff, registered triple-extraction
fixture matrix, and external-prediction triple-extraction minima. The source
defaults are path-aware, so explicit `--external-evidence-baseline-comparison`
or `--triple-extraction-fixture-matrix` file inputs suppress the corresponding
default registry keys. Strict local releases therefore fail closed when runtime
drift lacks promotion-contract, triple-extraction fixture-matrix,
trace-level triple-audit, recommended-route covered-fact property/action-gate
evidence, trajectory-audit evidence, promotion-contract evidence-handoff
coverage/manifest/metric-gap evidence, trace-level world-model
participation/coverage/conflict/low-agreement/trace-gap evidence,
trace-level context-sensitivity participation/coverage/flagged-rate/trace-gap
and max-ratio evidence, frontier release-evidence status/decision/track rates
plus citation-batch and rerun-rollup counts, the product-trace replay workflow
lacks promoted action-audit/action-execution child gates, or registered
frontier evidence handoffs are absent.
Add `--require-product-runtime-drift-claim-factuality-evidence` when a release
must additionally prove that claim factuality probe comparison evidence survived
the product-runtime handoff; it is opt-in so existing `frontier_audit` checks keep
their current default evidence boundary.
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
evidence. Add `--require-product-runtime-drift-pre-generation-evidence` when the
release must also require pre-generation probe comparison coverage,
manifest-verification, model/run breadth, redline pass-rate, AUROC, and
redline-margin metrics from that drift report. Add
`--require-product-runtime-drift-claim-factuality-evidence` when the release must
also require claim factuality probe comparison coverage, manifest-verification,
model/run breadth, redline pass-rate, AUROC, selective accuracy/coverage, and
redline-margin metrics from that drift report. Add
`--require-product-runtime-drift-triple-audit-evidence` when the release must
also require trace-level triple coverage, audited-claim coverage, audit
pass-rate, and slot coverage metrics from that drift report.
Add `--require-product-runtime-drift-world-model-evidence` when the release
must also require trace-level world-model participation, coverage, conflict,
low-agreement, and trace-gap metrics from that drift report.
Add `--require-product-runtime-drift-frontier-release-evidence` when the release
must also require frontier release-evidence status/decision/track rates,
citation-batch rollup counts, and rerun-rollup track/candidate counts from that
drift report.
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
The matrix's `state_transition` fixture uses typed actions plus
`world_model_rules`, so the promoted world-model evidence exercises
`RuleBasedWorldModelAdapter` rather than only the legacy in-memory update path.
`strict_audit` also makes the release gate fail closed unless that
`state_transition` family reports `RuleBasedWorldModelAdapter` and a positive
rule count. Use `--require-state-transition-world-model` when a custom adapter
route set should enforce the same rule-based evidence without using
`strict_audit`.
This keeps retrieval/database/world-model/audit adapter work inside the same
fail-closed release gate instead of treating it as a separate benchmark note.
Add `--triple-extraction-fixture-matrix` when release should also require a
promoted cross-corpus extractor benchmark from
`run_triple_extraction_fixture_matrix.py`. Use
`--min-triple-extraction-corpora` and
`--min-triple-extraction-distinct-predicates` to fail closed unless the matrix
covers enough promoted corpora and predicate diversity before extractor
templates become release evidence. When the matrix includes learned/OpenIE/LLM
external-prediction files, add
`--min-triple-extraction-external-prediction-count`,
`--min-triple-extraction-external-prediction-corpora`, and
`--min-triple-extraction-mean-best-external-f1` to require explicit external
extractor evidence before treating it as release support.
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
`--product-runtime-drift-report`, `--release-efficiency-report`,
`--external-evidence-baseline-comparison`,
`--external-evidence-baseline-comparison-key`,
`--external-evidence-baseline-comparison-registry`,
`--adapter-family-matrix`, and `--triple-extraction-fixture-matrix` options and
includes those
route/workflow/feedback-policy/selector/drift/efficiency/external-evidence
comparison/adapter and extractor manifests in the final release-candidate
manifest when the gate promotes. Required-route budget settings are also copied
into manifest metadata as `required_route_budget_policy`, including
`--required-route-require-non-oracle-evidence` and
`--required-route-require-retrieval-provenance-filter` when the audit route must
prove label-free local retrieval claims and a recorded evidence provenance
filter.
External-evidence baseline comparison gates can be supplied by direct JSON path
or by a registered `report:*:*` key, and the resolved source, record key, status,
recommended route, route-gate status, and text-redline status are copied into the
comparison, manifest, and registry metadata when configured.
Triple-extraction external-prediction gates are copied into the comparison,
manifest, and registry metadata when configured.
Use `--require-structured-fact-robustness` with
`--structured-fact-canonical-route-key` and
`--structured-fact-paraphrase-route-key` when the release must carry both
canonical and paraphrase `structured_fact` covered-facts evidence. The workflow
adds those two records to the required-route gate and records
`structured_fact_robustness_*` fields in the comparison report, final manifest,
and release registry metadata, including property counts and property ids for
the covered-fact slices that were actually gated.
Use `--release-policy-profile` with the registry workflow to reuse the same
named defaults while registering the promoted manifest. `strict_structured_fact`
enables the structured-fact robustness requirement, requires both configured
canonical/paraphrase route keys, applies the baseline candidate quality gates,
and separates ordinary required-route thresholds from the stricter
structured-fact robustness thresholds. Ordinary required routes keep
route-quality/provenance/stress gates; canonical/paraphrase `structured_fact`
routes carry the stricter selected-count and fail-closed per-property
support/refutation quality gates over route summary `property_metrics`.
`frontier_audit` adds the same structured-fact defaults and
also defaults `adapter_family_profile=strict_audit`,
`require_product_runtime_drift_promotion_evidence=true`,
`require_product_runtime_drift_pre_generation_evidence=true`,
`require_product_runtime_drift_counterfactual_evidence=true`,
`require_product_runtime_drift_triple_audit_evidence=true`,
`require_product_runtime_drift_covered_fact_property_evidence=true`,
`require_product_runtime_drift_action_gate_evidence=true`,
`require_product_runtime_drift_trajectory_audit_evidence=true`,
`require_product_runtime_drift_evidence_handoff_evidence=true`,
`require_product_runtime_drift_world_model_evidence=true`,
`require_product_runtime_drift_context_sensitivity_evidence=true`,
`require_product_trace_action_audit_gate=true`, and
`require_product_trace_action_execution_gate=true`, plus the registered
covered-facts external-evidence handoff, registered triple-extraction fixture
matrix, and external-prediction triple-extraction minima unless explicit file
paths are supplied. The release must carry the strict adapter-family matrix,
rule-based state-transition world-model evidence, promotion-backed runtime-drift
evidence, pre-generation runtime-drift evidence, counterfactual verifier-audit
runtime-drift evidence, trace-level triple-audit
evidence, recommended-route covered-fact property/action-gate drift evidence,
trajectory-audit runtime-drift evidence, trace-level world-model runtime-drift
evidence, trace-level context-sensitivity runtime-drift evidence, registered
frontier evidence handoffs, and
promoted product-trace action-audit/action-execution child gates unless
explicitly overridden. The workflow records
`release_policy_profile` and `release_policy_profile_applied_defaults` in the
comparison report, final manifest, and registry metadata.
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

When a release candidate blocks, build a root-cause-aware next-evidence plan
instead of manually scanning the comparison JSON:

```bash
python benchmarks/plan_release_evidence_gaps.py \
  --source artifacts/frontier-audit-release-candidate-v4/frontier-audit-comparison.json \
  --json artifacts/frontier-audit-release-candidate-v4/evidence-gap-plan.json \
  --registry artifacts/release-registry.json \
  --name frontier-audit-evidence-gap-plan \
  --version 0.1
```

The planner accepts either a `compare_release_candidates.py` report or a
`run_release_candidate_registry_workflow.py` payload. It writes
`workflow=evidence_gap_plan` with blocker-level gaps, missing metric names,
root-cause/research-axis tags, and prioritized next actions. The output is a
planning artifact only; it does not satisfy a release gate or promote verifier
evidence.

If the same source includes frontier multiple-testing blocked cells and points
back to the originating `truthfulqa_frontier_workflow` report, the gap planner
can also emit the executable per-cell rerun queue in the same pass:

```bash
python benchmarks/plan_release_evidence_gaps.py \
  --source artifacts/frontier-release-evidence/report.json \
  --json artifacts/frontier-release-evidence/evidence-gap-plan.json \
  --multiple-testing-rerun-json artifacts/frontier-release-evidence/multiple-testing-rerun-queue.json \
  --multiple-testing-rerun-artifact-manifest artifacts/frontier-release-evidence/multiple-testing-rerun-queue-manifest.json \
  --multiple-testing-rerun-output-dir artifacts/frontier-multiple-testing-reruns \
  --registry artifacts/release-registry.json \
  --name frontier-release-evidence-gap-plan \
  --version 0.1 \
  --multiple-testing-rerun-name frontier-multiple-testing-reruns \
  --multiple-testing-rerun-version 0.1
```

The saved gap plan records the derived queue path, status, blocked-cell count,
and command count under `derived_artifacts`. The queue itself remains a separate
reviewable artifact with command arrays and dry-run variants, so expensive
frontier reruns stay opt-in.

The same gap-planner bridge can emit a citation/source-family batch rerun queue
when the frontier release is blocked by missing, duplicate, or unexpected
citation evidence batches:

```bash
python benchmarks/plan_release_evidence_gaps.py \
  --source artifacts/frontier-release-evidence/report.json \
  --json artifacts/frontier-release-evidence/evidence-gap-plan.json \
  --citation-batch-rerun-json artifacts/frontier-release-evidence/citation-batch-rerun-queue.json \
  --citation-batch-rerun-artifact-manifest artifacts/frontier-release-evidence/citation-batch-rerun-queue-manifest.json \
  --citation-batch-rerun-output-dir artifacts/frontier-citation-batch-reruns \
  --citation-batch-queue artifacts/truthfulqa-frontier-smollm2-l80-unresolved-blind-spot-evidence-queue/unresolved-evidence-queue.json \
  --citation-batch-scores artifacts/truthfulqa-frontier-smollm2-l80-score-dump.jsonl \
  --citation-batch-blind-spots artifacts/truthfulqa-frontier-smollm2-l80-entrenched-blind-spots/rows.jsonl \
  --citation-batch-source-catalog artifacts/source-family-catalogs/official-catalog.jsonl \
  --registry artifacts/release-registry.json \
  --name frontier-release-evidence-gap-plan \
  --version 0.1 \
  --citation-batch-rerun-name frontier-citation-batch-reruns \
  --citation-batch-rerun-version 0.1
```

`benchmarks/plan_citation_batch_evidence_reruns.py` can also be called directly.
It accepts a frontier release report, an evidence-gap plan, or a
`citation_search_batch_evidence_rollup` report. Queue entries are marked
`ready` only when enough paths are supplied to build a
`run_source_family_citation_search_workflow.py` or
`run_external_citation_search_adapter_workflow.py` command; otherwise they remain
reviewable `missing_inputs` rows.

The gap planner can also emit a verifier/abstention stability rerun queue when
the frontier release blocks on either stability track:

```bash
python benchmarks/plan_release_evidence_gaps.py \
  --source artifacts/frontier-release-evidence/report.json \
  --json artifacts/frontier-release-evidence/evidence-gap-plan.json \
  --stability-rerun-json artifacts/frontier-release-evidence/stability-rerun-queue.json \
  --stability-rerun-artifact-manifest artifacts/frontier-release-evidence/stability-rerun-queue-manifest.json \
  --stability-rerun-output-dir artifacts/frontier-stability-reruns \
  --stability-scores qwen05-l80=artifacts/truthfulqa-frontier-qwen-smollm2-l80/qwen05-l80/scores.manifest.json \
  --stability-scores smollm2-l80=artifacts/truthfulqa-frontier-qwen-smollm2-l80/smollm2-l80/scores.manifest.json \
  --stability-seeds 0,1,2,3,4,5,6,7,8,9 \
  --verifier-qa-corpus artifacts/truthfulqa_l80_correct_answer_corpus.json \
  --registry artifacts/release-registry.json \
  --name frontier-release-evidence-gap-plan \
  --version 0.1 \
  --stability-rerun-name frontier-stability-reruns \
  --stability-rerun-version 0.1
```

`benchmarks/plan_frontier_stability_evidence_reruns.py` can also be called
directly. It accepts a frontier release report, an evidence-gap plan, or an
existing `verifier_stability` / `abstention_stability` report. When an existing
stability report is reachable from comparator inputs, the planner inherits its
score dumps and config; otherwise pass `--scores`, `--verifier-signal`, and
`--abstention-signals` explicitly. Queue entries remain `missing_inputs` until
enough data is present to build the post-hoc stability command.

After the generated verifier/abstention stability commands finish, roll the
completed child reports back into release evidence:

```bash
python benchmarks/rollup_frontier_stability_evidence_reruns.py \
  --queue artifacts/frontier-release-evidence/stability-rerun-queue.json \
  --json artifacts/frontier-release-evidence/stability-rerun-rollup.json \
  --artifact-manifest artifacts/frontier-release-evidence/stability-rerun-rollup-manifest.json \
  --registry artifacts/release-registry.json \
  --name frontier-stability-rerun-rollup \
  --version 0.1 \
  --require-all-reports
```

The rollup reads each ready queue entry's expected `--json` child report,
accepts additional completed reports via repeatable `--report`, and reapplies
the release thresholds from `compare_frontier_release_evidence.py`. The output
promotes only when every queued verifier/abstention stability track has a valid
child report and every run in those reports satisfies its stability gate.

For releases specifically blocked by abstention participation quality, emit an
experiment matrix rather than a single stability rerun. The abstention planner
builds one `eval_abstention_stability.py` command per blocked run, profile, and
signal group. This is intended to refresh the supervised feasibility frontier
and compare candidate participation-gate settings before changing runtime
defaults:

```bash
python benchmarks/plan_release_evidence_gaps.py \
  --source artifacts/frontier-release-evidence/report.json \
  --json artifacts/frontier-release-evidence/evidence-gap-plan.json \
  --abstention-rerun-json artifacts/frontier-release-evidence/abstention-rerun-queue.json \
  --abstention-rerun-artifact-manifest artifacts/frontier-release-evidence/abstention-rerun-queue-manifest.json \
  --abstention-rerun-output-dir artifacts/frontier-abstention-reruns \
  --abstention-profiles baseline,alpha_0p05,alpha_0p2,selective_accuracy,retention \
  --abstention-signal-groups recommended,all,geometry,uncertainty \
  --abstention-seeds 0,1,2,3,4,5,6,7,8,9 \
  --registry artifacts/release-registry.json \
  --name frontier-release-evidence-gap-plan \
  --version 0.1 \
  --abstention-rerun-name frontier-abstention-reruns \
  --abstention-rerun-version 0.1
```

`benchmarks/plan_frontier_abstention_evidence_reruns.py` can also be called
directly with a frontier release report, evidence-gap plan, or
`abstention_stability` report. If the original abstention report is reachable,
it inherits run score dumps, seeds, release thresholds, and recommended signals;
otherwise pass `--scores`, `--profiles`, and `--signal-groups` explicitly.

After running the generated `eval_abstention_stability.py` commands, roll the
completed reports back into release evidence. By default the rollup can consume
whatever reports are present; add `--require-all-reports` for a strict
fail-closed release check:

```bash
python benchmarks/rollup_frontier_abstention_evidence_reruns.py \
  --queue artifacts/frontier-release-evidence/abstention-rerun-queue.json \
  --json artifacts/frontier-release-evidence/abstention-rerun-rollup.json \
  --artifact-manifest artifacts/frontier-release-evidence/abstention-rerun-rollup-manifest.json \
  --registry artifacts/release-registry.json \
  --name frontier-abstention-rerun-rollup \
  --version 0.1 \
  --require-all-reports
```

The rollup reads each queue entry's expected `--json` output path, accepts
additional completed reports via repeatable `--report`, ranks candidates by
conservative conditional correctness, seed pass rate, and abstention cost, then
emits `status=promote` only when a promotion-eligible profile satisfies the
configured correctness, abstention-rate, and seed-stability thresholds.

For releases blocked by the detectability-taxonomy track, the same planner can
emit row-level blind-spot audit commands from the comparator's
`--detectability-taxonomy-report` inputs:

```bash
python benchmarks/plan_release_evidence_gaps.py \
  --source artifacts/frontier-release-evidence/report.json \
  --json artifacts/frontier-release-evidence/evidence-gap-plan.json \
  --detectability-rerun-json artifacts/frontier-release-evidence/detectability-rerun-queue.json \
  --detectability-rerun-artifact-manifest artifacts/frontier-release-evidence/detectability-rerun-queue-manifest.json \
  --detectability-rerun-output-dir artifacts/frontier-detectability-reruns \
  --registry artifacts/release-registry.json \
  --name frontier-release-evidence-gap-plan \
  --version 0.1 \
  --detectability-rerun-name frontier-detectability-reruns \
  --detectability-rerun-version 0.1
```

`benchmarks/plan_frontier_detectability_evidence_reruns.py` can also be called
directly. If a reachable `detectability_taxonomy` report exists, queue entries
run `analyze_detectability_blind_spots.py` for the configured cell, defaulting
to false `entrenched` rows. If no taxonomy report is available, pass
`--scores`, `--consistency-signal`, and `--confidence-signal` to emit
`eval_detectability_taxonomy.py` rerun commands instead.

When a reachable taxonomy report exists but the blocked gate should be retested
under alternate DECK axes, add score dumps plus `--include-taxonomy-reruns` and
one or more `--detectability-taxonomy-pair consistency:confidence` values. Pair
directions default to the healthy direction implied by `DEFAULT_SCORE_DIRECTIONS`
and can be overridden as `consistency:confidence:consistency_direction:confidence_direction`.
For example, `--detectability-taxonomy-pair disp_hse:nll_answer` appends a
semantic-dispersion consistency axis with answer-NLL confidence while preserving
the blind-spot audit entry.

After the detectability queue has produced child reports, roll them up before
feeding the evidence back into release review:

```bash
python benchmarks/rollup_frontier_detectability_evidence_reruns.py \
  --queue artifacts/frontier-release-evidence/detectability-rerun-queue.json \
  --json artifacts/frontier-release-evidence/detectability-rerun-rollup.json \
  --artifact-manifest artifacts/frontier-release-evidence/detectability-rerun-rollup-manifest.json \
  --registry artifacts/release-registry.json \
  --name frontier-detectability-rerun-rollup \
  --version 0.1 \
  --require-all-reports
```

Taxonomy rerun children can emit `status=promote` when
`entrenched_false_rate <= --max-entrenched-false-rate`. Blind-spot analysis
children instead emit `status=complete` with `audit_ready=true`: they document
which rows need source-family, retrieval, or world-model evidence expansion, but
they do not by themselves satisfy the release detectability gate.

For frontier release reports blocked by the family-wise multiple-testing gate,
build a per-cell rerun queue from the comparator or gap-plan output:

```bash
python benchmarks/plan_frontier_multiple_testing_reruns.py \
  --source artifacts/frontier-release-evidence/report.json \
  --json artifacts/frontier-release-evidence/multiple-testing-rerun-queue.json \
  --artifact-manifest artifacts/frontier-release-evidence/multiple-testing-rerun-queue-manifest.json \
  --registry artifacts/release-registry.json \
  --name frontier-multiple-testing-reruns \
  --version 0.1 \
  --output-dir artifacts/frontier-multiple-testing-reruns
```

When the original `truthfulqa_frontier_workflow` report is reachable from the
release comparator inputs, queue entries include single-cell
`run_truthfulqa_frontier_workflow.py` command arrays plus dry-run variants. The
queue manifest fingerprints both the source report and generated queue so the
planning step can be reviewed before any expensive rerun starts.

After the child frontier workflow reruns finish, roll the queue back into
release evidence:

```bash
python benchmarks/rollup_frontier_multiple_testing_reruns.py \
  --queue artifacts/frontier-release-evidence/multiple-testing-rerun-queue.json \
  --json artifacts/frontier-release-evidence/multiple-testing-rerun-rollup.json \
  --artifact-manifest artifacts/frontier-release-evidence/multiple-testing-rerun-rollup-manifest.json \
  --registry artifacts/release-registry.json \
  --name frontier-multiple-testing-rerun-rollup \
  --version 0.1 \
  --require-all-reports
```

The rollup checks each completed child report's top-level
`multiple_testing_gate.all_pass` summary and the queued cell entry under
`multiple_testing_gate.cells`. A candidate promotes only when the family-wise
gate passes, the queued cell passes, and that cell records both report and
calibration artifact paths.

Feed promoted rerun rollups back into the final release-evidence comparison
with repeatable `--frontier-rerun-rollup-report` arguments:

```bash
python benchmarks/compare_frontier_release_evidence.py \
  --verifier-stability-report artifacts/frontier/verifier-stability-report.json \
  --abstention-stability-report artifacts/frontier/abstention-stability-report.json \
  --frontier-rerun-rollup-report artifacts/frontier-release-evidence/stability-rerun-rollup.json \
  --frontier-rerun-rollup-report artifacts/frontier-release-evidence/abstention-rerun-rollup.json \
  --frontier-rerun-rollup-report artifacts/frontier-release-evidence/detectability-rerun-rollup.json \
  --frontier-rerun-rollup-report artifacts/frontier-release-evidence/multiple-testing-rerun-rollup.json \
  --json artifacts/frontier-release-evidence/frontier-release-evidence-refreshed.json
```

Before rerunning product-runtime drift gates, audit the deployable promotion
contract for the exact evidence handoff fields expected by `frontier_audit`:

```bash
python benchmarks/audit_product_promotion_contract_evidence.py \
  --contract artifacts/smollm2_product_promotion_contract_v1_6/product-promotion-contract.json \
  --json artifacts/smollm2_product_promotion_contract_v1_6/evidence-handoff-audit.json \
  --registry artifacts/local-release-registry.json \
  --name smollm2-product-promotion-contract-v1-6-evidence-handoff \
  --version 0.1
```

The audit writes `workflow=product_promotion_evidence_handoff_audit` with the
same promotion, pre-generation, counterfactual, triple-audit, covered-fact,
action-gate, and frontier-release evidence metric names used by release drift
blockers. Its default group set is kept compatible with existing promotion
contracts; stricter runs can explicitly pass `--required-groups` entries such as
`claim_factuality`, `claim_risk_localization`, `trajectory_audit`,
`evidence_handoff`, `world_model`, `context_sensitivity`, or
`counterfactual_robustness` when those runtime-drift gates are part of the
release policy. Treat it as pre-flight evidence hygiene: it explains why a
contract will not satisfy a runtime-drift gate, but it does not itself satisfy
the gate.

After the audit, export an evidence-enriched contract from explicit local child
reports:

```bash
python benchmarks/export_product_promotion_contract_evidence_handoff.py \
  --contract artifacts/smollm2_product_promotion_contract_v1_6/product-promotion-contract.json \
  --json artifacts/smollm2_product_promotion_contract_v1_6/product-promotion-contract-evidence-handoff.json \
  --audit-json artifacts/smollm2_product_promotion_contract_v1_6/product-promotion-contract-evidence-handoff-audit.json \
  --pre-generation-probe-comparison artifacts/runtime_evidence/pre-generation-qwen-smollm2-l12-comparison/comparison.json \
  --triple-extraction-fixture-matrix artifacts/wikidata-cross-corpus-triple-extraction-adversarial-matrix-v1/triple-extraction-fixture-matrix.json \
  --counterfactual-verification artifacts/smollm2_product_counterfactual_structured_qa_audit_v0/counterfactual-verification-report.json \
  --product-trace-replay-workflow artifacts/smollm2_product_trace_replay_workflow_action_gated_v0/product-trace-replay-workflow.json \
  --triple-audit-enrichment artifacts/smollm2_product_trace_triple_audit_enrichment_v1/product-trace-triple-audit-enrichment.json \
  --runtime-baseline artifacts/smollm2_product_trace_replay_workflow_action_gated_v0/runtime-baseline/product-runtime-baseline.json \
  --covered-fact-property-metrics artifacts/truthfulqa-frontier-smollm2-l80-source-family-structured-qa-fact-collection-route/structured-qa-route-summary.json \
  --frontier-release-evidence artifacts/frontier-release-evidence/frontier-release-evidence-refreshed.json \
  --required-groups promotion,pre_generation,counterfactual,triple_audit,covered_fact_property,action_gate,frontier_release_evidence \
  --registry artifacts/local-release-registry.json \
  --name smollm2-product-promotion-contract-v1-6-evidence-handoff \
  --version 0.4
```

The exporter only copies evidence from supplied reports; it does not invent
trace-level triple-audit results. `--frontier-release-evidence` should point to
the refreshed frontier release-evidence verdict when the handoff is being used
for current `frontier_audit` drift evidence. `--required-groups` applies the
same strict handoff group set to both the enriched contract audit and the
manifest/registry metadata, so a stricter frontier pass can preserve exactly
which runtime-drift groups were required; include optional groups such as
`claim_risk_localization`, `trajectory_audit`, or `world_model` only when the
corresponding product-runtime evidence has been materialized. `--triple-audit-enrichment` can point either
to `enrich_product_trace_triple_audit.py` output or to a promoted
`run_source_family_structured_qa_claim_correction_workflow.py --enable-triple-audit`
report; the older `--runtime-baseline` path remains supported for aggregated
runtime evidence. The current v1.6 handoff export reduces
missing metrics from `37/38` to `3/38`: promotion/triple matrix, pre-generation
comparison, counterfactual verification, covered-fact property, and action-gate
groups are present, while audit/slot triple coverage remains the next
evidence-producing work.

For the current checkout's frontier audit replay, the self-contained evidence
handoff is `artifacts/smollm2_product_promotion_evidence_handoff_v1_6_frontier_v2/`.
It uses the available blind-spot Wikidata structured-QA route summary for
covered-fact property metrics and deliberately leaves missing child reports
missing instead of substituting stale paths. Its audit has `56/65` metrics
present and `9` missing metrics across counterfactual verifier audit and
trace-level triple audit. The companion
`artifacts/smollm2_product_runtime_drift_v1_10_trace_evidence/` report carries
`101` comparable metrics, including claim-risk localization, covered-fact
property, trajectory-audit, evidence-handoff, action-gate, and frontier
citation expected/observed batch-count rows. Replaying `frontier_audit` as
`artifacts/frontier-audit-release-candidate-v8/` keeps the release blocked but
reduces the evidence-gap plan to `20` missing metrics and `9` actions. This is
trace-evidence materialization only; it does not promote the upstream frontier
release evidence or fill missing counterfactual/world-model/context/triple-audit
quality rows.

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
  --context-sensitivity-workflow-key report:<context-sensitivity-workflow-name>:<version> \
  --release-efficiency-report artifacts/product-runtime-profile-sweep/release-efficiency-report.json \
  --external-evidence-baseline-comparison artifacts/external-evidence-baseline-comparison.json \
  --adapter-family-matrix artifacts/adapter_family_matrix/adapter-family-matrix.json \
  --adapter-family-profile strict_audit \
  --triple-extraction-fixture-matrix artifacts/triple-extraction-fixture-matrix/triple-extraction-fixture-matrix.json \
  --min-triple-extraction-corpora 2 \
  --min-triple-extraction-distinct-predicates 6 \
  --min-triple-extraction-external-prediction-count 2 \
  --min-triple-extraction-external-prediction-corpora 2 \
  --min-triple-extraction-mean-best-external-f1 0.90 \
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
  --min-last-json-cache-hit-rate 0.90 \
  --min-report-count 2 \
  --fail-on-blocked
```

The report records `total_seconds`, `phase_total_seconds`, per-phase timing,
artifact fingerprint/JSON cache hit rates, and the slowest observed phase. Use
`--min-last-fingerprint-cache-hit-rate` and
`--min-last-json-cache-hit-rate` to verify the warm run actually reused the
persisted fingerprint and parsed-JSON artifact caches.

Use `export_product_promotion_contract.py` after a release candidate promotes to
write the smaller product handoff artifact consumed by demos and control-plane
jobs. It converts either a release-candidate comparison or registry-workflow JSON
into a `ProductPromotionContract`, writes a manifest, and can register a
`product_promotion_contract:*:*` record. When the release candidate was gated by
a frontier-audit or strict local-release profile, the exported contract now also
contains a compact `summary` block. That summary preserves the source status,
gate statuses, runtime recommendation, verifier-route quality/cost fields,
action-gate status, grouped runtime-drift evidence counts, recommended records,
control defaults, and runtime budget policy without requiring reviewers to scan
the full metadata map. The manifest and registry record mirror the headline
summary fields as `promotion_summary_*` metadata for dashboards and release
checks. Runtime traces that load the contract carry the same view as
`promotion_contract_promotion_summary`; when a frontier release-evidence report
was supplied to the release candidate, they also carry
`promotion_contract_frontier_release_evidence` plus headline
`promotion_contract_frontier_release_evidence_*` metrics. `product_runtime_metrics()`
exposes both field families for baselines and SLO reports.

When the release candidate was gated by
a structured-fact route audit, the compact contract, manifest, and registry
metadata keep the recommended route's covered Wikidata property ids plus
required-route record-to-property coverage summaries, so deployment-side traces
can state the exact KG predicate scope behind the verifier route. When the
release candidate was gated by an external-evidence baseline comparison, the
compact contract, manifest, and registry metadata retain the comparator report
path, source type, registry key, decision status, recommended route, route-gate
status, and text-redline status so runtime traces can show which external
evidence handoff was release-gated.
When the release candidate was gated by a claim factuality probe comparison, the
compact contract, manifest, and registry metadata retain the comparison report,
manifest, registry record, workflow/status, model/run counts, redline status,
best run/model/layer, test-label AUROC, selective accuracy/coverage, conformal
threshold, and redline margin. `ProductRuntimeEvidenceBundle` can lazily verify
that comparison manifest and attach the local registry record to runtime trace
metadata without rerunning claim probes.
When the release candidate was gated by a counterfactual verifier audit, the
compact contract, manifest, and registry metadata retain the audit report,
manifest, registry record, workflow, status, record count, pass rate,
false-invariance rate, and flip-success count. `ProductRuntimeEvidenceBundle`
can lazily verify that audit manifest and attach the local registry record to
runtime trace metadata without rerunning verifier probes.
When the
release candidate was gated by
a product trace replay workflow, the compact contract and registry metadata keep
the workflow report/manifest plus its selector-replay and runtime-drift child
report paths, action-audit gate report/status/rates, and action-execution gate
report/status/alignment rates for deployment-side provenance. Runtime-drift
reports also carry baseline/current optimization hints plus promotion-contract
and trace-level triple-audit evidence summaries, so exported contracts preserve
candidate control defaults such as `max_verifier_route_attempts` alongside the
budget policy and keep drift/audit/action-execution coverage visible in
manifest and registry metadata.
When the release candidate was gated by a feedback-policy workflow, the
contract and registry metadata also retain the feedback-policy report/manifest,
promotion decision, candidate control-policy/default paths, validated
`ControlPolicyConfig`, control-default config, and replay safety metrics. When
the release candidate was gated by a world-model signal workflow, the contract
and registry metadata retain the workflow report/manifest, release-gate status,
trace-gap maximum, conflict-positive count, and calibrated conflict-signal
count. When
the release candidate was gated by a release-efficiency report, the
promotion contract inherits the recommended runtime profile and efficiency score
from the candidate. For older release-candidate reports, pass the
release-efficiency report explicitly so the promotion contract, manifest, and
registry record also carry the same handoff evidence:
When the release candidate includes a cross-corpus triple-extraction fixture
matrix, the exported contract and registry metadata keep the matrix
report/manifest, registry record, corpus coverage, predicate diversity, and F1
lift summary so triple-evidence routes can be audited from runtime traces.
Performance-bundle provenance is preserved as well: exported contracts retain
best quality signal, score-fusion status, and selected-fusion
status/run/candidate/signal/AUROC/false-alarm/detection/artifact metadata when
those fields were present in the release candidate. Release-candidate runtime
cost provenance is preserved too: contracts and ProductTrace metadata expose
`recommended_runtime_seconds`, `recommended_runtime_cost_source`,
`max_recommended_runtime_seconds`, and uncached forward cost when the source
candidate includes a recommended deployment path.

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

Current frontier-audit handoff status:

The latest local strict frontier-audit replay is promoted:
`artifacts/frontier-audit-release-candidate-v13/frontier-audit-registry-workflow.json`
registers
`benchmark_manifest:smollm2-l8-frontier-audit-release-candidate:0.13`
after recursive manifest verification. The promoted chain uses
`artifacts/frontier-release-evidence/frontier-release-evidence-budget-target-sweep-v4.json`
for frontier evidence, the refreshed v1.6 handoff at
`artifacts/smollm2_product_promotion_evidence_handoff_v1_6_frontier_v4/`,
and `artifacts/smollm2_product_runtime_drift_v1_14_frontier_budget_target/`,
whose drift report promotes with `107` compared metrics and `0` blockers.
The handoff exporter now prefers complete `product_trace_triple_audit_enrichment`
coverage when an older runtime baseline only carries partial triple-audit
metadata. That v13/v1.6 handoff promoted under the earlier `65/65`
frontier boundary; refreshed frontier-audit handoffs now require `77/77`
receipt-aware metrics.

Active v1.9 product contract export:

The promoted v13 replay has been exported as the active v1.9 product handoff
under `artifacts/smollm2_product_promotion_contract_v1_9/`. The raw contract
preserves the release candidate's selected cache-only runtime recommendation
(`recommended_runtime_seconds=0.191662`) and registers
`product_promotion_contract:smollm2-product-promotion-contract:1.9`.

```bash
PRODUCT_V19_DIR=artifacts/smollm2_product_promotion_contract_v1_9

python benchmarks/export_product_promotion_contract.py \
  --source artifacts/frontier-audit-release-candidate-v13/frontier-audit-registry-workflow.json \
  --output "$PRODUCT_V19_DIR/product-promotion-contract.json" \
  --artifact-manifest "$PRODUCT_V19_DIR/artifact-manifest.json" \
  --registry artifacts/local-release-registry.json \
  --name smollm2-product-promotion-contract \
  --version 1.9 \
  --metadata release=smollm2-v1.9 \
  --metadata source_record=benchmark_manifest:smollm2-l8-frontier-audit-release-candidate:0.13 \
  --compact-json
```

The direct export intentionally keeps source release metadata compact, so run an
explicit child-report handoff before using the contract as a runtime evidence
bundle. The checked-in enriched contract is registered as
`product_promotion_contract:smollm2-product-promotion-contract-v1-9-evidence-handoff:0.4`
and independently promotes with the earlier `65/65` boundary. New refreshed
handoffs require `77/77` present metrics: use a product-trace replay workflow
or runtime-baseline input that carries `action_receipts` and
`receipt_claim_support` summaries, otherwise the audit fails closed on the two
receipt evidence groups.

```bash
PRODUCT_V19_DIR=artifacts/smollm2_product_promotion_contract_v1_9

python benchmarks/export_product_promotion_contract_evidence_handoff.py \
  --contract "$PRODUCT_V19_DIR/product-promotion-contract.json" \
  --json "$PRODUCT_V19_DIR/product-promotion-contract-evidence-handoff.json" \
  --audit-json "$PRODUCT_V19_DIR/product-promotion-contract-evidence-handoff-audit.json" \
  --pre-generation-probe-comparison artifacts/runtime_evidence/pre-generation-qwen-smollm2-l12-comparison/comparison.json \
  --triple-extraction-fixture-matrix artifacts/wikidata-cross-corpus-triple-extraction-adversarial-matrix-v1/triple-extraction-fixture-matrix.json \
  --counterfactual-verification artifacts/smollm2_product_counterfactual_blind_spot_wikidata_structured_qa_audit_v1/counterfactual-verification-report.json \
  --product-trace-replay-workflow artifacts/smollm2_product_trace_replay_workflow_action_gated_v0/product-trace-replay-workflow.json \
  --frontier-release-evidence artifacts/frontier-release-evidence/frontier-release-evidence-budget-target-sweep-v4.json \
  --triple-audit-enrichment artifacts/smollm2_product_trace_triple_audit_enrichment_v1/product-trace-triple-audit-enrichment.json \
  --runtime-baseline artifacts/smollm2_product_runtime_drift_v1_10_trace_evidence/runtime-baseline/product-runtime-baseline.json \
  --covered-fact-property-metrics artifacts/truthfulqa-frontier-smollm2-l80-blind-spot-wikidata-structured-qa-route/structured-qa-route-summary.json \
  --artifact-manifest "$PRODUCT_V19_DIR/evidence-handoff-artifact-manifest.json" \
  --registry artifacts/local-release-registry.json \
  --name smollm2-product-promotion-contract-v1-9-evidence-handoff \
  --version 0.4 \
  --metadata release=smollm2-v1.9 \
  --metadata source_record=benchmark_manifest:smollm2-l8-frontier-audit-release-candidate:0.13 \
  --metadata evidence_scope=frontier_audit_v13_explicit_child_report_handoff
```

The historical v6 replay remains a useful fail-closed regression guard: it
correctly refused product-contract export while the source release candidate was
blocked. The active deployable frontier-audit handoff is the v13-derived v1.9
contract above.
Before treating the v6 handoff as locally reproducible, scan the active doc
references against the checkout. A blocked report includes `recommended_actions`
for the v6 release-candidate rerun, manifest verification, and final re-audit.
By default
it also inspects the frontier `artifact-json-cache.json` so the summary splits
missing references into cache-recoverable and unrecoverable counts. Add
`--restore-json-cache-artifacts` to restore only those cache-backed missing JSON
files and re-run the audit before writing the final report. Restore mode
also inspects failed artifact manifests for missing cache-backed JSON child
artifacts; manifest children are written only when the restored bytes match the
parent manifest digest. It normalizes repo-local absolute paths in cached JSON
payloads to repo-relative paths before writing files:

```bash
python benchmarks/audit_frontier_artifact_references.py \
  --json artifacts/frontier-artifact-reference-audit.json \
  --artifact-manifest artifacts/frontier-artifact-reference-audit-manifest.json \
  --registry artifacts/local-release-registry.json \
  --name frontier-artifact-reference-audit \
  --version 0.1 \
  --no-fail
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
  --require-product-runtime-drift-promotion-evidence \
  --require-product-runtime-drift-pre-generation-evidence \
  --require-product-runtime-drift-claim-factuality-evidence \
  --require-product-runtime-drift-triple-audit-evidence \
  --require-product-runtime-drift-evidence-handoff-evidence \
  --require-product-runtime-drift-world-model-evidence \
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
capital, `P37` official language, and `P38` currency. The
`organization_product_core_facts` preset fetches non-country structured facts
for extractor-matrix evidence, defaulting to `P159` headquarters location,
`P176` manufacturer, and `P571` inception over a small deterministic
OpenAI/Tesla/Apple seed set. Add repeated `--subject Q...` values to expand
that seed set without changing the dependency-free fetcher. The script uses
only the standard library and supports `--input-json` for offline replay of
saved SPARQL results. Rows whose natural-language labels are bare Wikidata
`Q...` or `P...` ids are skipped by default to avoid turning unresolved entities
into retrieval evidence; pass `--keep-qid-labels` only when debugging raw SPARQL
coverage.

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

```bash
python benchmarks/fetch_wikidata_reference_docs.py \
  --query-preset organization_product_core_facts \
  --property P159 \
  --property P176 \
  --property P571 \
  --subject Q21708200 \
  --subject Q32399 \
  --subject Q2766 \
  --limit 180 \
  --output artifacts/wikidata-organization-product-core-facts/wikidata-organization-product-source.jsonl \
  --artifact-manifest artifacts/wikidata-organization-product-core-facts/wikidata-source-manifest.json
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
For generic Wikidata source docs that already expose `subject`,
`statement_property_label`, and `value` metadata, pass
`--auto-template-from-source` to infer one QA template per `statement_property`
without hand-authoring a template JSON file.

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

The current SmolLM2 l80 blind-spot Wikidata source docs use the auto-template
path:

```bash
python benchmarks/build_wikidata_qa_corpus.py \
  --source artifacts/truthfulqa-frontier-smollm2-l80-blind-spot-wikidata-evidence/wikidata-source-docs.jsonl \
  --output artifacts/truthfulqa-frontier-smollm2-l80-blind-spot-wikidata-structured-qa-route/wikidata-blind-spot-qa-corpus.json \
  --auto-template-from-source \
  --template-json-output artifacts/truthfulqa-frontier-smollm2-l80-blind-spot-wikidata-structured-qa-route/wikidata-blind-spot-qa-templates.json \
  --artifact-manifest artifacts/truthfulqa-frontier-smollm2-l80-blind-spot-wikidata-structured-qa-route/qa-corpus-manifest.json
```

That run converts all `292` target-specific Wikidata source docs into
structured QA rows across `10` properties.

## `run_wikidata_structured_qa_route_workflow.py`

Builds a covered-facts structured QA or structured-fact route benchmark from a
Wikidata QA corpus.
The workflow creates a balanced score dump with one true row per known
question/answer and one false row per question by swapping in an answer from a
different question while avoiding known same-question answers. It then runs the
existing verifier ensemble with `--qa-corpus` or `--fact-corpus`, writes
per-record verifier traces, and emits a route summary plus artifact manifest.
The route summary includes `property_metrics`, keyed by Wikidata property id,
with per-property source-document counts, true/false record counts, selected
route counts, decision accuracy, false-supported rate, and false-refuted rate.
The artifact manifest also exposes a flat `covered_fact_property_*` metadata
view with property ids, per-property quality fields, and worst-property rollups,
so registry and release tooling can prefilter covered-fact evidence without
opening the route summary JSON. Those fields are consumed by
`compare_external_evidence_baselines.py` when a release gate needs
predicate-level covered-facts evidence.

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
The target-specific SmolLM2 l80 blind-spot Wikidata structured-QA route is a
second covered-facts artifact: `292` source docs over `10` properties produce
`584` balanced rows, select `structured_qa` for all rows, support all `292` true
facts, refute all `292` swapped-answer false facts, and promote through
`run_covered_facts_external_evidence_workflow.py` as
`report:truthfulqa-frontier-smollm2-l80-blind-spot-wikidata-covered-facts-handoff:0.1`.
This promotes the property-level correction path for those collected Wikidata
facts; it still does not claim open-domain blind-spot recall.
Use `compare_external_evidence_baselines.py --require-covered-facts-route`
when one of these registered covered-facts route manifests should become the
external-evidence comparator input for release gating. Add the per-property
covered-fact thresholds there when a release should fail closed if any covered
property has too few records or weaker support/refutation quality than the
aggregate route.

For the current saved `structured_fact` canonical plus paraphrase artifacts,
`run_covered_facts_external_evidence_workflow.py` provides the reproducible
registry handoff into the external-evidence comparator:

```bash
python benchmarks/run_covered_facts_external_evidence_workflow.py \
  --route-manifest canonical=artifacts/wikidata-country-core-facts-structured-fact-route/artifact-manifest.json \
  --route-manifest paraphrase=artifacts/wikidata-country-core-facts-structured-fact-paraphrase-route/artifact-manifest.json \
  --output-dir artifacts/wikidata-structured-fact-external-evidence-handoff \
  --registry artifacts/staged-route-registry.json \
  --name wikidata-structured-fact-external-evidence-handoff \
  --version 0.4 \
  --covered-fact-route structured_fact \
  --min-covered-fact-records 700 \
  --min-covered-fact-source-documents 300 \
  --min-covered-fact-true 300 \
  --min-covered-fact-false 300
```

Those historical artifacts predate `property_metrics`, so do not add
`--min-covered-fact-property-*` gates unless the route manifests were rebuilt
with the current `run_wikidata_structured_qa_route_workflow.py` schema.

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
current corpus. Use the provenance filter flags when a local corpus represents
external or domain-shifted evidence: `--require-retrieval-source`,
`--allowed-retrieval-source-prefix`, `--denied-retrieval-source-prefix`,
`--min-retrieval-score`, `--required-retrieval-metadata key=value`, and
`--max-retrieval-hits-per-source` drop untrusted hits before they become verifier
evidence and record the filter in `input_provenance`. `--verification-cache-dir`
is optional and stores verified-record traces keyed by score dump,
claims/evidence content, verifier parameters, and state/QA sources so repeated
alpha/repeat sweeps can skip claim verification.

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
The same provenance filter flags supported by `build_evidence_fixture.py` are
available here and are included in the claims-cache key, workflow report,
artifact manifest, and registry metadata. Use them with external corpora or
HTTP/exported retrieval snapshots so answer-echo stress controls, low-score
documents, or untrusted source prefixes cannot silently enter a promoted route.

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

It can also evaluate newer geometry-calibrated scores by separating
representation-geometry signals from uncertainty/confidence proxies. The
`product` fusion method is the dependency-free GLU-style global-local
uncertainty gate: hidden-state geometry must agree with local token or
semantic uncertainty before the fused score becomes large.

```bash
python benchmarks/eval_score_ensemble.py \
  --scores artifacts/qwen05_truthfulqa_l80_scores.json \
  --signals truth_proj,subspace_resid,resid_update_norm \
  --geometry-signals subspace_resid,resid_update_norm \
  --uncertainty-signals nll_answer,inside_semantic_energy \
  --geometry-fusion-methods interaction,product \
  --best-alpha 0.10 \
  --confidence-signal nll_answer \
  --max-high-confidence-accepted-false-rate 0.0 \
  --save-best-geometry-fusion-artifact artifacts/qwen05_geometry_fusion_artifact.json \
  --json artifacts/qwen05_score_ensemble_report.json
```

When `--confidence-signal` is supplied, the report adds a
`fusion_release_gate_at_alpha` block and per-candidate
`confidence_error_at_best_alpha` / `release_gate_at_best_alpha` payloads. This
is the release-facing check for global-local or semantic-energy fusion: the
candidate may have good AUROC and conformal false-alarm behavior, but it is
blocked if the configured high-confidence region still contains accepted false
answers above `--max-high-confidence-accepted-false-rate`. Uncertainty-style
signals such as `nll_answer`, `first_token_entropy`, and
`inside_semantic_energy` default to `--confidence-direction lower` because lower
values mean higher model confidence.
Runtime recommendations now honor each candidate's
`release_gate_at_best_alpha`: a score-fusion route is only promoted as a
quality signal when both the conformal gate and the high-confidence release gate
pass.

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

python benchmarks/build_context_sensitivity_logprob_pairs.py \
  --records artifacts/verifier-signals/verified-records.jsonl \
  --model-id Qwen/Qwen2.5-0.5B \
  --run-name qwen-l80 \
  --output artifacts/verifier-signals/paired-context-logprobs.jsonl \
  --json artifacts/verifier-signals/paired-context-logprobs-report.json

python benchmarks/enrich_context_sensitivity_sidecar.py \
  --verified-records-jsonl artifacts/verifier-signals/verified-records.jsonl \
  --paired-logprobs artifacts/verifier-signals/paired-context-logprobs.jsonl \
  --run-name qwen-l80 \
  --output artifacts/verifier-signals/verified-records-context.jsonl \
  --json artifacts/verifier-signals/context-sensitivity-sidecar-report.json

python benchmarks/build_verifier_signal_score_dump.py \
  --scores artifacts/qwen05_truthfulqa_l80_scores_with_statements.json \
  --verified-records-jsonl artifacts/verifier-signals/verified-records-context.jsonl \
  --run-name qwen-l80 \
  --keep-signals truth_proj,maha_last,subspace_resid,eigenscore,nll_answer \
  --output artifacts/verifier-signals/qwen-l80-enhanced-scores.manifest.json \
  --output-format jsonl
```

The context-sensitivity subchain can also run as a single local workflow with
manifest verification and optional registry metadata:

```bash
python benchmarks/run_context_sensitivity_workflow.py \
  --scores artifacts/qwen05_truthfulqa_l80_scores_with_statements.json \
  --verified-records-jsonl artifacts/verifier-signals/verified-records.jsonl \
  --model-id Qwen/Qwen2.5-0.5B \
  --run-name qwen-l80 \
  --keep-signals truth_proj,maha_last,subspace_resid,eigenscore,nll_answer \
  --output-dir artifacts/verifier-signals/context-sensitivity-workflow \
  --registry-path artifacts/verifier-signals/registry.json \
  --registry-name qwen-l80-context-sensitivity \
  --registry-version 0.1
```

Registered context-sensitivity workflow reports can be promoted into the release
candidate gate with `--context-sensitivity-workflow-key` or supplied directly
with `--context-sensitivity-workflow`. The gate verifies the workflow manifest
and fails closed when paired-logprob, enriched-sidecar, or enhanced score-dump
evidence is missing; it does not require a non-zero flagged rate.

When verified records include state-transition prediction metadata or direct
world-model ensemble agreement metadata, the same converter also emits
world-model uncertainty columns such as `world_model_disagreement`,
`world_model_agreement_gap`, and `world_model_low_agreement`. When transition
metadata includes explicit postcondition conflicts, it also emits
`world_model_conflict`, `world_model_conflict_delta`, and the audit-oriented
`world_model_trace_gap`, so simulator/model disagreement and expected-vs-actual
world conflicts can be swept or fused under the same conformal calibration path.
When `transformers` is installed via the optional HF extra,
`build_context_sensitivity_logprob_pairs.py` can compute paired no-context and
evidence-context token log-probabilities directly from the verified-record
sidecar. Teams with their own model serving stack can instead emit the same
paired-logprob JSONL schema. `enrich_context_sensitivity_sidecar.py` then adds
per-record `ContextSensitivityReport` payloads without depending on that model
runtime. The score-dump converter emits
`context_sensitivity_flagged_rate`, `context_sensitivity_max_shift`,
`context_sensitivity_mean_shift`, and `context_sensitivity_max_ratio` so
evidence-context likelihood disagreements can be calibrated alongside geometry
and verifier outcomes.

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

## `select_hidden_evidence.py`

Selects a sparse, budgeted evidence report from primary and `sweep_scores`
diagnostics. This is the local HIVE-style bridge between hidden-state/trajectory
signals and downstream verifier or world-model adapters: it records which
record/layer/signal items should be inspected under a fixed budget, but it does
not execute a verifier or promote a route by itself.

```bash
python benchmarks/select_hidden_evidence.py \
  --scores artifacts/qwen05_truthfulqa_l80_scores.json \
  --signals truth_proj,subspace_resid,nll_answer \
  --sweep-signals truth_proj,subspace_resid,resid_update_norm \
  --direction selfcheck_support_rate=lower \
  --max-items 64 \
  --max-per-record 4 \
  --max-per-layer 8 \
  --json artifacts/qwen05_hidden_evidence_selection.json \
  --registry artifacts/registry.json \
  --register-name qwen05-hidden-evidence-selection \
  --quiet
```

The report rank-normalizes scores per `source/layer/signal` channel, applies
`higher` or `lower` anomaly directions, preserves statement metadata when
available, and writes selected `evidence_ref` values that can be copied into
`ProductTrace.metadata` or passed to
`ClaimVerificationPlanner.plan(..., hidden_evidence=report)` so verifier budgets
prioritize claims selected by hidden-state evidence. Bounded `ProductTrace`
summaries expose the resulting hidden-evidence claim counts, score families,
layers, and evidence refs for replay.

## `eval_detectability_taxonomy.py`

Builds a DECK-style consistency x confidence taxonomy from any two saved
score-dump signals. This is a post-hoc detectability profile: it answers which
families of scorers would plausibly catch the observed false records, rather
than promoting a new release signal by itself.

```bash
python benchmarks/eval_detectability_taxonomy.py \
  --scores artifacts/qwen05_truthfulqa_l80_scores.json \
  --consistency-signal inside_selfcheck_support_rate \
  --confidence-signal nll_answer \
  --confidence-direction lower \
  --json artifacts/qwen05_detectability_taxonomy.json
```

The report uses Youden's J splits on the two axes and writes counts for
`drift`, `entrenched`, `confabulation`, and `knotted` cells. Entrenched false
records are treated as the output-level uncertainty blind spot and should be
handed to independent verifier, retrieval, citation, structured-fact, or
world-model routes.

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
  --require-retrieval-source \
  --allowed-retrieval-source-prefix external: \
  --required-retrieval-metadata corpus_role=grounding \
  --omit-label-metadata
```

Add `--samples path/to/sample-records.json` when a score dump or external
sampler provides aligned self-consistency responses.

The workflow is intentionally post-hoc and dependency-free. It is the preferred
entry point when testing whether local retrieval and self-consistency evidence
improves a calibrated geometry monitor without rerunning model scoring. Its
retrieval fixture builder accepts the same provenance filter flags as
`build_evidence_fixture.py`, so verifier-signal fusion can be run only over
source-typed, score-filtered, metadata-tagged evidence.

For world-model correction specifically,
`run_world_model_signal_calibration_workflow.py` builds a deterministic
state-transition fixture, runs the world-model verifier route, converts the
verified-record sidecar into verifier/world-model score columns, evaluates the
same score/geometry-fusion calibration path, verifies nested manifests, and can
record the workflow in a local registry:

```bash
python benchmarks/run_world_model_signal_calibration_workflow.py \
  --output-dir artifacts/world-model-signal-calibration-smoke \
  --n-records 24 \
  --world-model-ensemble \
  --world-model-ensemble-min-agreement 0.75 \
  --world-model-ensemble-strategy policy_replay \
  --alphas 0.05,0.1,0.2 \
  --best-alpha 0.1 \
  --repeats 20 \
  --registry artifacts/local-release-registry.json \
  --registry-name world-model-signal-calibration-smoke \
  --registry-version 0.1
```

Without `--world-model-ensemble`, the default fixture uses a single
`RuleBasedWorldModelAdapter`; its agreement-gap columns are expected to be zero,
and the calibrated correction signal is the world-model route's final verifier
outcome, such as `verifier_refuted`. With `--world-model-ensemble`, the workflow
produces nonzero `world_model_disagreement`, `world_model_agreement_gap`, and
`world_model_low_agreement` columns from controlled member disagreement. Use
`--world-model-ensemble-strategy policy_replay` when testing strategy-driven
disagreement that is not directly label-shaped. The default workflow fusion
signals also include `world_model_conflict` and `world_model_conflict_delta`,
so non-ensemble transition contradictions remain calibratable even when
agreement-gap columns are zero.
Direct ensemble agreement metadata from external multi-world-model adapters is
also preserved by `build_verifier_signal_score_dump.py` when present in
sidecars.

The workflow report now includes a `release_gate` for the world-model signal
handoff. It promotes only when the generated trace has no unexplained
`world_model_trace_gap`, has positive world-model conflict examples, and the
score-ensemble report calibrates at least one conflict signal such as
`world_model_conflict` or `world_model_conflict_delta` under the selected
`--best-alpha`. Release candidates can require this evidence by passing
`--world-model-signal-workflow path/to/world-model-signal-calibration-workflow.json`
or by registering the workflow and passing
`--world-model-signal-workflow-key report:world-model-signal-workflow:0.1`.

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
false-alarm gate and any attached high-confidence release gate. Blocked or
ambiguous fusion evidence is retained in `score_fusion` and `evidence` without
changing the runtime recommendation.
With `--selected-fusion-artifact-report`, selected fusion artifacts produced by
`build_selected_fusion_artifacts.py` can likewise contribute a
`selected_fusion_*` quality signal, but only when the artifact path is present
and the report's per-run `release_gate` is either absent or promoted. If the
report has multiple runs, pass `--selected-fusion-run <run_name>`; otherwise the
recommendation keeps the selected-fusion evidence as `ambiguous_matching_runs`
and does not change the best quality signal.
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
directly or reuse existing matrix/worker/INSIDE/score-ensemble/selected-fusion
reports, then writes
`performance-baseline-workflow.json`, `runtime-recommendation.json`, an artifact
manifest, a top-level `performance_evidence_bundle` summary with recommendation
cost ratios / evidence status / artifact readiness / score-dump cache evidence /
score-fusion / selected-fusion status, and an optional
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

When reusing selected fusion artifacts, add
`--selected-fusion-artifact-report <build-report.json>` and pass
`--selected-fusion-run <run_name>` for multi-run reports. The workflow records
the selected build report, selected artifact path, selected-artifact manifest
path, selected run, selected candidate, selected-artifact release-gate status,
and promoted `selected_fusion_*` signal in the runtime recommendation,
performance evidence bundle, artifact manifest metadata, and registry record.
The current local SmolLM2 l8 selected-fusion handoff uses:

```bash
python benchmarks/run_performance_baseline_workflow.py \
  --output-dir artifacts/smollm2_l8_read_cache_worker_sweep_selected_fusion_performance_baseline \
  --registry artifacts/local-readiness-registry.json \
  --name smollm2-l8-read-cache-worker-sweep-selected-fusion-performance-baseline \
  --version 0.3 \
  --matrix-report artifacts/smollm2_l8_read_cache_worker_sweep/workers_2/cache-profile-matrix-report.json \
  --worker-sweep-report artifacts/smollm2_l8_read_cache_worker_sweep/cache-worker-sweep-report.json \
  --score-ensemble-report artifacts/truthfulqa_score_ensemble_report.json \
  --selected-fusion-artifact-report artifacts/e7-truthfulqa-trajectory-multimodel/selected-fusion-artifact-build-report.json \
  --selected-fusion-run smollm2 \
  --verify-manifest \
  --fail-on-blocked
```

It registers
`performance_baseline:smollm2-l8-read-cache-worker-sweep-selected-fusion-performance-baseline:0.3`
and
`manifest_verification:smollm2-l8-read-cache-worker-sweep-selected-fusion-performance-baseline-verification:0.3`.
The selected-fusion evidence is promoted as `selected_fusion_mean_rank`
(`AUROC=0.692`, false alarm `0.029`, detection `0.224`, `alpha=0.1`) from the
SmolLM2 `geometry:mean_rank` selected artifact, while `truth_proj` remains the
best quality signal for the runtime recommendation.
The matching staged structured-QA release candidate is registered as
`benchmark_manifest:smollm2-l8-read-cache-worker-sweep-selected-fusion-staged-qa-release-candidate:0.3`
with recursive manifest verification
`manifest_verification:smollm2-l8-read-cache-worker-sweep-selected-fusion-staged-qa-release-candidate-verification:0.3`.
It reuses the same readiness and `structured_qa` route baselines as the
score-fusion candidate, but gates against the selected-fusion performance
baseline and records the selected-fusion run/signal/AUROC/artifact path plus
selected-artifact manifest path in the release registry.
The deployable contract for this local handoff is
`product_promotion_contract:smollm2-l8-selected-fusion-product-promotion-contract:0.3`
at
`artifacts/smollm2_l8_selected_fusion_product_promotion_contract_v0_3/product-promotion-contract.json`.

Use `run_product_runtime_baseline.py` for the product-control side of the same
performance story: aggregate saved `ProductTrace` JSON files, summarize request
phase timings, route costs, cache hit rates, retrieval use, staged-verification
skip savings, verification-scope counts, triggered-only partial skip savings,
triple/slot-audit coverage, promotion-contract covered-fact property scope and
per-property quality rollups, and optionally apply a
`ProductRuntimeBudgetPolicy` or promoted
`ProductPromotionContract` budget. It also aggregates promotion-contract
external-evidence baseline-comparison handoff coverage, source/status counts,
recommended route counts, route-gate pass counts, text-redline pass counts, and
text-redline run-count summaries into the report, manifest, and registry
metadata. It also aggregates promotion-contract counterfactual verifier-audit
coverage, source/status/workflow counts, manifest-verification counts,
record-count/pass-rate/false-invariance summaries, and flip-success summaries
into the same report, manifest, and registry metadata. The output includes
`optimization.hotspots`,
`optimization.recommendations`, and `optimization.policy_hints`, turning the
baseline into an actionable performance pass over slow phases/routes, low cache
hit rates, excessive retrieval or verifier fanout, missing staged verification,
missing triggered-claim-only staged verification, extracted triples without
strict slot-audit reports, and audit-heavy
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
resolved policy payload. Add `--trace-scan-workers` for faster JSON scan and
metric extraction on large trace sets; reports, manifests, and registry records
preserve the configured and effective worker counts, while the default remains
single-worker for comparable timing evidence. `ProductRuntimeBudgetPolicy` can gate overall staged
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
  --trace-scan-workers 4 \
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
retrieval-use, cache-hit-rate, verifier-skip-rate, trace-level world-model
participation/coverage/conflict/low-agreement/trace-gap drift, trace-level
context-sensitivity participation/coverage/flagged-rate/ratio drift,
promotion-contract coverage, covered-fact per-property rollup drift,
triple-extraction fixture-matrix coverage/quality drift, claim factuality probe
comparison coverage, manifest-verification, redline, AUROC,
selective-accuracy, and selective-coverage drift, counterfactual verifier-audit
coverage, manifest verification, record count, pass rate, false-invariance
rate, flip-success drift, trace-level triple/slot-audit coverage, and
trace-count drift. When ProductTrace action results carry action receipts, the
same comparison can gate receipt coverage plus missing, invalid,
fingerprint-mismatch, and unsigned receipt rates. When claims or final answers
explicitly reference action request ids or receipt fingerprints, it can also
gate receipt-backed claim-support rates and the drift of unsupported,
unreceipted, failed-result, fingerprint-mismatch, or unsigned references.
`build_product_trace_corpus.py` materializes redaction-safe
`summaries.triple_coverage` plus
`metadata.trace_corpus.triple_coverage_summary` for accepted full ProductTrace
inputs, so these drift gates can consume triple-audit coverage after text
redaction. The summary is only evidence when the original trace already carried
`claim_triples` and verifier `audit_report` metadata; the corpus builder does
not infer or fabricate audit results from redacted text.

Use `enrich_product_trace_triple_audit.py` when you still have full, unredacted
ProductTrace JSON or JSONL sidecars and a local evidence corpus but the original
runtime did not record strict triple-audit metadata. The workflow extracts
conservative `claim_triples` or reuses metadata-supplied triples, retrieves
local evidence snippets, attaches status-aware `audit_report` metadata to
existing verifier results or explicit `audit_only` results, and writes
manifest-backed enriched traces for `run_product_runtime_baseline.py`.
Source-family correction handoffs now place model-answer triples and structured
refutation evidence into their ProductTrace JSONL rows, so they can be enriched
directly with `--trace-jsonl` before runtime-baseline replay. For the current
SmolLM2 action-payload
compatibility traces, the Wikidata capital corpus plus the NASA-backed Moon
composition corpus promotes the trace-level triple-audit handoff with
`claim_triple_coverage_rate=1.0`, `audit_claim_coverage_rate=1.0`,
`audit_pass_rate=1.0`, and `slot_coverage_rate=1.0`; refuted claims are marked
with `evidence_relation=refutes_claim` rather than treated as supported:

```bash
python benchmarks/enrich_product_trace_triple_audit.py \
  --trace-glob 'artifacts/smollm2_product_trace_action_payload_compat_v0/traces/**/*.json' \
  --evidence-corpus artifacts/wikidata-country-capitals-external-corpus/wikidata-country-capitals-corpus.json \
  --evidence-corpus artifacts/nasa-moon-composition-external-corpus/moon-composition-corpus.json \
  --output-dir artifacts/smollm2_product_trace_triple_audit_enrichment_v1 \
  --registry artifacts/smollm2_product_trace_triple_audit_enrichment_v1/registry.json \
  --name smollm2-product-trace-triple-audit \
  --version 0.2
```

For source-family correction handoff sidecars:

```bash
python benchmarks/enrich_product_trace_triple_audit.py \
  --trace-jsonl artifacts/truthfulqa-frontier-smollm2-l80-source-family-structured-qa-correction-handoff/product-traces.jsonl \
  --output-dir artifacts/truthfulqa-frontier-smollm2-l80-source-family-structured-qa-correction-triple-audit \
  --registry artifacts/local-release-registry.json \
  --name truthfulqa-frontier-smollm2-l80-source-family-structured-qa-correction-triple-audit \
  --version 0.1 \
  --compact-json
```

Use `enrich_product_trace_runtime_evidence.py` when existing full traces already
carry verifier results but lack the bounded summaries needed by the
world-model, context-sensitivity, and counterfactual-robustness runtime drift
gates. The workflow attaches local deterministic sidecars from calculator and
structured-QA metadata, records its limitations in verifier metadata, and does
not call network retrieval, vector stores, or LLMs. For the current SmolLM2
action-payload compatibility traces, `8/12` traces contain verifier results;
the enrichment therefore uses `0.66` participating-trace gates while keeping
coverage, trace-gap, pass-rate, flip-success, and false-invariance gates
strict:

```bash
python benchmarks/enrich_product_trace_runtime_evidence.py \
  --trace-glob 'artifacts/smollm2_product_trace_action_payload_compat_v0/traces/**/*.json' \
  --output-dir artifacts/smollm2_product_trace_runtime_evidence_enrichment_v0 \
  --registry artifacts/smollm2_product_trace_runtime_evidence_enrichment_v0/artifact-registry.json \
  --name smollm2-product-trace-runtime-evidence \
  --version 0.1 \
  --min-world-model-participating-trace-rate 0.66 \
  --min-context-sensitivity-participating-trace-rate 0.66 \
  --min-counterfactual-robustness-participating-trace-rate 0.66 \
  --compact-json
```

After `enrich_product_trace_triple_audit.py` has added `audit_only` verifier
results for skipped verification profiles, rerun runtime-evidence enrichment on
the triple-audited traces. This attaches deterministic
`TripleAuditWorldModelAdapter` sidecars to the audit reports, so the current
SmolLM2 trace set can use strict `1.0` participating-trace gates across
world-model, context-sensitivity, and counterfactual-robustness evidence:

```bash
python benchmarks/enrich_product_trace_runtime_evidence.py \
  --trace-glob 'artifacts/smollm2_product_trace_triple_audit_enrichment_v1/traces/*.json' \
  --output-dir artifacts/smollm2_product_trace_frontier_evidence_enrichment_v1 \
  --registry artifacts/smollm2_product_trace_frontier_evidence_enrichment_v1/artifact-registry.json \
  --name smollm2-product-trace-frontier-evidence \
  --version 0.1 \
  --min-world-model-participating-trace-rate 1.0 \
  --min-context-sensitivity-participating-trace-rate 1.0 \
  --min-counterfactual-robustness-participating-trace-rate 1.0 \
  --compact-json
```

The v1.13 runtime-drift replay over those frontier-evidence traces keeps all
triple-audit, world-model, context-sensitivity, and counterfactual-robustness
rows passing. The refreshed frontier audit v11 remains fail-closed on
readiness/performance/frontier-release evidence, but its gap plan drops to
`9` gaps, `4` actions, and `0` missing metrics.

Replay the enriched traces through `run_product_runtime_baseline.py` with
`--trace-scan-workers` when the trace set is large enough to benefit from
bounded parallel scanning, then compare baseline/current with the same
product-runtime drift gates used by the prior release candidate plus the new
trace-level runtime-evidence gates:

```bash
TRACE_ARGS=$(find artifacts/smollm2_product_trace_runtime_evidence_enrichment_v0/traces -name '*.json' | sort | sed 's#^#--trace #')

python benchmarks/run_product_runtime_baseline.py $TRACE_ARGS \
  --json artifacts/smollm2_product_runtime_drift_v1_12_runtime_evidence_enriched/runtime-baseline/product-runtime-baseline.json \
  --artifact-manifest artifacts/smollm2_product_runtime_drift_v1_12_runtime_evidence_enriched/runtime-baseline/artifact-manifest.json \
  --promotion-contract artifacts/smollm2_product_promotion_evidence_handoff_v1_6_frontier_v3/product-promotion-contract-evidence-handoff.json \
  --trace-scan-workers 4 \
  --compact-json

python benchmarks/compare_product_runtime_baselines.py \
  --baseline artifacts/smollm2_product_runtime_drift_v1_12_runtime_evidence_enriched/runtime-baseline/product-runtime-baseline.json \
  --current artifacts/smollm2_product_runtime_drift_v1_12_runtime_evidence_enriched/runtime-current/product-runtime-baseline.json \
  --json artifacts/smollm2_product_runtime_drift_v1_12_runtime_evidence_enriched/product-runtime-drift.json \
  --artifact-manifest artifacts/smollm2_product_runtime_drift_v1_12_runtime_evidence_enriched/artifact-manifest.json \
  --min-current-trace-count 12 \
  --min-world-model-participating-trace-rate 0.66 \
  --min-world-model-coverage-rate 1.0 \
  --max-world-model-conflict-rate-increase 0.0 \
  --max-world-model-low-agreement-rate-increase 0.0 \
  --max-world-model-trace-gap-rate-increase 0.0 \
  --min-context-sensitivity-participating-trace-rate 0.66 \
  --min-context-sensitivity-coverage-rate 1.0 \
  --max-context-sensitivity-flagged-result-rate-increase 0.0 \
  --max-context-sensitivity-trace-gap-rate-increase 0.0 \
  --max-context-sensitivity-max-flagged-rate-increase 0.0 \
  --max-context-sensitivity-max-ratio-increase 0.0 \
  --min-counterfactual-robustness-participating-trace-rate 0.66 \
  --min-counterfactual-robustness-coverage-rate 1.0 \
  --min-counterfactual-robustness-pass-rate 1.0 \
  --min-counterfactual-robustness-flip-success-rate 1.0 \
  --max-counterfactual-robustness-false-invariance-rate-increase 0.0 \
  --max-counterfactual-robustness-trace-gap-rate-increase 0.0 \
  --min-product-trace-action-receipts-coverage-rate 1.0 \
  --max-product-trace-action-receipts-missing-receipt-rate-increase 0.0 \
  --max-product-trace-action-receipts-invalid-receipt-rate-increase 0.0 \
  --max-product-trace-action-receipts-fingerprint-mismatch-rate-increase 0.0 \
  --max-product-trace-action-receipts-unsigned-receipt-rate-increase 0.0 \
  --min-product-trace-receipt-claim-support-reference-support-rate 1.0 \
  --max-product-trace-receipt-claim-support-unsupported-reference-rate-increase 0.0 \
  --max-product-trace-receipt-claim-support-unreceipted-reference-rate-increase 0.0 \
  --max-product-trace-receipt-claim-support-fingerprint-mismatch-reference-rate-increase 0.0 \
  --compact-json
```

When a saved `ProductRuntimeBudgetPolicy` is supplied with
`--runtime-budget-policy` or `--runtime-budget-policy-key`, the current baseline
summary is also checked against the reusable budget using p95/aggregate metrics. When a
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
  --min-world-model-participating-trace-rate 1.0 \
  --min-world-model-coverage-rate 1.0 \
  --max-world-model-conflict-rate-increase 0.0 \
  --max-world-model-low-agreement-rate-increase 0.0 \
  --max-world-model-trace-gap-rate-increase 0.0 \
  --min-context-sensitivity-participating-trace-rate 1.0 \
  --min-context-sensitivity-coverage-rate 1.0 \
  --max-context-sensitivity-flagged-result-rate-increase 0.0 \
  --max-context-sensitivity-trace-gap-rate-increase 0.0 \
  --max-context-sensitivity-max-flagged-rate-increase 0.0 \
  --max-context-sensitivity-max-ratio-increase 0.0 \
  --min-promotion-contract-coverage 1.0 \
  --min-claim-factuality-probe-comparison-coverage 1.0 \
  --min-claim-factuality-probe-comparison-manifest-verified-rate 1.0 \
  --min-claim-factuality-probe-comparison-model-count 2 \
  --min-claim-factuality-probe-comparison-run-count 2 \
  --min-claim-factuality-probe-comparison-redline-pass-rate 1.0 \
  --max-claim-factuality-probe-comparison-best-test-label-auroc-drop 0.02 \
  --max-claim-factuality-probe-comparison-best-test-selective-accuracy-drop 0.02 \
  --max-claim-factuality-probe-comparison-best-test-selective-coverage-drop 0.02 \
  --max-claim-factuality-probe-comparison-best-redline-auroc-drop 0.02 \
  --max-claim-factuality-probe-comparison-best-redline-margin-drop 0.02 \
  --min-counterfactual-verification-coverage 1.0 \
  --min-counterfactual-verification-manifest-verified-rate 1.0 \
  --min-counterfactual-verification-record-count 10 \
  --min-counterfactual-verification-pass-rate 0.95 \
  --max-counterfactual-verification-false-invariance-rate 0.05 \
  --max-counterfactual-verification-flip-success-count-drop 2 \
  --min-promotion-contract-covered-fact-property-metric-count 3 \
  --min-promotion-contract-covered-fact-min-records 8 \
  --min-promotion-contract-covered-fact-min-source-documents 100 \
  --max-promotion-contract-covered-fact-min-decision-accuracy-drop 0.02 \
  --max-promotion-contract-covered-fact-max-false-supported-rate-increase 0.01 \
  --max-promotion-contract-covered-fact-min-false-refuted-rate-drop 0.02 \
  --min-triple-extraction-fixture-matrix-coverage 1.0 \
  --max-triple-extraction-fixture-matrix-mean-best-f1-drop 0.05 \
  --max-triple-extraction-fixture-matrix-mean-f1-lift-drop 0.05 \
  --min-triple-claim-coverage 0.5 \
  --min-triple-audit-claim-coverage 1.0 \
  --min-triple-audit-pass-rate 1.0 \
  --min-triple-slot-coverage 1.0 \
  --min-product-trace-action-receipts-coverage-rate 1.0 \
  --max-product-trace-action-receipts-missing-receipt-rate-increase 0.0 \
  --max-product-trace-action-receipts-invalid-receipt-rate-increase 0.0 \
  --max-product-trace-action-receipts-fingerprint-mismatch-rate-increase 0.0 \
  --max-product-trace-action-receipts-unsigned-receipt-rate-increase 0.0 \
  --min-product-trace-receipt-claim-support-reference-support-rate 1.0 \
  --max-product-trace-receipt-claim-support-unsupported-reference-rate-increase 0.0 \
  --max-product-trace-receipt-claim-support-unreceipted-reference-rate-increase 0.0 \
  --max-product-trace-receipt-claim-support-fingerprint-mismatch-reference-rate-increase 0.0 \
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

Use `build_product_trace_action_payload_compat.py` before strict action-gated
replays over older ProductTrace files. It copies the source traces to a new
local input set and only repairs legacy `retrieve` action payloads that lack an
executable query/target list, deriving deterministic `retrieval_targets` from
the trace's saved claim IDs. This is a compatibility handoff, not new verifier
or retrieval evidence:

```bash
python benchmarks/build_product_trace_action_payload_compat.py \
  --trace-glob 'artifacts/smollm2_product_runtime_profile_sweep/traces/*/*.json' \
  --output-dir artifacts/smollm2_product_trace_action_payload_compat_v0 \
  --registry artifacts/local-release-registry.json \
  --name smollm2-product-trace-action-payload-compat \
  --version 0.1 \
  --metadata source=frontier_action_gate_rebuild
```

The current SmolLM2 compatibility handoff writes 12 replay traces, modifies the
single legacy trace with an empty retrieve payload, adds two claim-backed
retrieval targets, and verifies its manifest. Feed
`artifacts/smollm2_product_trace_action_payload_compat_v0/traces/**/*.json` to
`run_product_trace_replay_workflow.py` when enforcing zero-tolerance
action-audit gates over the historical profile-sweep traces.

The latest local frontier-audit v6 refresh also repairs the cross-corpus
triple-extraction matrix provenance. Generate the lookup-gold external
prediction files from the generated fixture records with
`build_triple_extraction_lookup_gold_predictions.py`, then rerun
`run_triple_extraction_fixture_matrix.py` with the two `--external-predictions`
paths under `artifacts/wikidata-cross-corpus-triple-extraction-adversarial-matrix-v1/`.
The v1 matrix manifest now verifies and
`triple_extraction_fixture_matrix_gate` promotes in v6. The release candidate
still blocks on frontier-release evidence and product-runtime evidence coverage;
the refreshed `evidence-gap-plan.json` tracks 16 remaining gaps and 15 actions.

The canonical local action-gated replay rebuild is:

```bash
python benchmarks/run_product_trace_replay_workflow.py \
  --trace-glob 'artifacts/smollm2_product_trace_action_payload_compat_v0/traces/**/*.json' \
  --output-dir artifacts/smollm2_product_trace_replay_workflow_action_gated_v0 \
  --candidate default=artifacts/smollm2_runtime_profile_selector_tuning/policies/default.json \
  --candidate latency_biased=artifacts/smollm2_runtime_profile_selector_tuning/policies/latency_biased.json \
  --candidate audit_biased=artifacts/smollm2_runtime_profile_selector_tuning/policies/audit_biased.json \
  --replay-policy artifacts/smollm2_runtime_profile_selector_replay/runtime-profile-selector-replay-policy.json \
  --registry artifacts/local-release-registry.json \
  --name smollm2-product-trace-replay-workflow-action-gated \
  --version 0.1 \
  --require-runtime-trace \
  --verify-manifest \
  --fingerprint-cache artifacts/smollm2_product_trace_replay_workflow_action_gated_v0/fingerprints.json \
  --corpus-cache-json artifacts/smollm2_product_trace_replay_workflow_action_gated_v0/corpus-cache.json \
  --refresh-corpus-cache \
  --corpus-source-cache-json artifacts/smollm2_product_trace_replay_workflow_action_gated_v0/corpus/source-cache.json \
  --refresh-corpus-source-cache \
  --runtime-trace-records-cache-json artifacts/smollm2_product_trace_replay_workflow_action_gated_v0/runtime-baseline/trace-record-cache.json \
  --refresh-runtime-trace-records-cache \
  --runtime-trace-scan-workers 4 \
  --selector-trace-inputs-json artifacts/smollm2_product_trace_replay_workflow_action_gated_v0/selector-replay/trace-inputs.json \
  --refresh-selector-trace-inputs \
  --max-action-audit-error-rate 0.0 \
  --max-action-audit-missing-retrieval-rate 0.0 \
  --max-action-audit-missing-plan-retrieval-query-rate 0.0 \
  --max-action-audit-malformed-payload-rate 0.0 \
  --max-action-audit-unexpected-action-rate 0.0 \
  --max-action-audit-unknown-claim-id-rate 0.0 \
  --max-action-execution-missing-result-rate 0.0 \
  --max-action-execution-unexpected-result-rate 0.0 \
  --max-action-execution-request-id-mismatch-rate 0.0
```

Use `run_product_trace_replay_workflow.py` when the raw-trace handoff should be
one reproducible command. It builds the redacted corpus, runs the product
runtime baseline over the standardized traces, runs selector replay with the
provided candidate policies using the corpus runtime-pair index, writes a
recursive top-level manifest over all child reports, records phase timing/cache
summaries for local performance tuning, lifts the runtime baseline
`optimization` status/recommendations/policy hints into the top-level workflow
report and registry metadata, can save the runtime baseline's recommended
`ProductRuntimeBudgetPolicy` artifact for later gates, can run the current
runtime baseline through action-audit, action-execution alignment, or
product-runtime drift/policy gates, and registers one workflow report.
Add `--verify-manifest` to write a separate recursive verification
report and register `manifest_verification:<name>-verification:<version>` next
to the workflow report. Add `--fingerprint-cache` when repeating local checks,
`--corpus-source-cache-json` when only some raw trace files change,
`--corpus-cache-json` when the entire standardized corpus can be reused,
`--runtime-trace-records-cache-json` when sweeping runtime budget gates,
`--runtime-trace-scan-workers` when large standardized trace sets make runtime
baseline JSON scan and metric extraction the local bottleneck,
`--save-runtime-recommended-policy` when the workflow should materialize the
observed baseline's candidate budget thresholds, and
`--runtime-drift-baseline` plus optional `--runtime-drift-budget-policy` when
the workflow should immediately validate the current runtime baseline against
the previous promoted baseline/policy gate. The runtime-drift pass-through also
accepts world-model gates such as
`--min-runtime-drift-world-model-participating-trace-rate`,
`--min-runtime-drift-world-model-coverage-rate`,
`--max-runtime-drift-world-model-conflict-rate-increase`,
`--max-runtime-drift-world-model-low-agreement-rate-increase`, and
`--max-runtime-drift-world-model-trace-gap-rate-increase`, context-sensitivity
gates such as
`--min-runtime-drift-context-sensitivity-participating-trace-rate`,
`--min-runtime-drift-context-sensitivity-coverage-rate`,
`--max-runtime-drift-context-sensitivity-flagged-result-rate-increase`,
`--max-runtime-drift-context-sensitivity-trace-gap-rate-increase`,
`--max-runtime-drift-context-sensitivity-max-flagged-rate-increase`, and
`--max-runtime-drift-context-sensitivity-max-ratio-increase`, plus promotion
evidence gates such as
`--min-runtime-drift-promotion-contract-coverage`,
pre-generation probe comparison gates such as
`--min-runtime-drift-pre-generation-probe-comparison-coverage`,
`--min-runtime-drift-pre-generation-probe-comparison-manifest-verified-rate`,
`--min-runtime-drift-pre-generation-probe-comparison-redline-pass-rate`, and
`--max-runtime-drift-pre-generation-probe-comparison-best-*-drop`,
claim factuality probe comparison gates such as
`--min-runtime-drift-claim-factuality-probe-comparison-coverage`,
`--min-runtime-drift-claim-factuality-probe-comparison-manifest-verified-rate`,
`--min-runtime-drift-claim-factuality-probe-comparison-redline-pass-rate`, and
`--max-runtime-drift-claim-factuality-probe-comparison-best-*-drop`,
`--min-runtime-drift-triple-extraction-fixture-matrix-coverage`, the two
`--max-runtime-drift-triple-extraction-fixture-matrix-mean-*` drop gates, and
covered-fact property gates such as
`--min-runtime-drift-covered-fact-property-metric-count`,
`--min-runtime-drift-covered-fact-min-records`,
`--min-runtime-drift-covered-fact-min-source-documents`, and the
`--max-runtime-drift-covered-fact-*` drift gates. Add
`--max-action-execution-missing-result-rate`,
`--max-action-execution-unexpected-result-rate`, and
`--max-action-execution-request-id-mismatch-rate` when a replay workflow should
fail closed on dropped, unexpected, or request-id-mismatched action execution
results. Add
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
  --runtime-trace-scan-workers 4 \
  --save-runtime-recommended-policy artifacts/smollm2_product_trace_replay_workflow/runtime-baseline/recommended-policy.json \
  --runtime-drift-baseline artifacts/smollm2_product_runtime_profile_sweep/baselines/auto/product-runtime-baseline.json \
  --runtime-drift-budget-policy artifacts/product-runtime-baseline-recommended-policy.json \
  --max-action-execution-missing-result-rate 0.0 \
  --max-action-execution-unexpected-result-rate 0.0 \
  --max-action-execution-request-id-mismatch-rate 0.0 \
  --max-runtime-drift-total-seconds-p95-ratio 1.6 \
  --min-runtime-drift-world-model-participating-trace-rate 1.0 \
  --min-runtime-drift-world-model-coverage-rate 1.0 \
  --max-runtime-drift-world-model-conflict-rate-increase 0.0 \
  --max-runtime-drift-world-model-low-agreement-rate-increase 0.0 \
  --max-runtime-drift-world-model-trace-gap-rate-increase 0.0 \
  --min-runtime-drift-promotion-contract-coverage 1.0 \
  --min-runtime-drift-pre-generation-probe-comparison-coverage 1.0 \
  --min-runtime-drift-pre-generation-probe-comparison-manifest-verified-rate 1.0 \
  --min-runtime-drift-pre-generation-probe-comparison-model-count 2 \
  --min-runtime-drift-pre-generation-probe-comparison-run-count 2 \
  --min-runtime-drift-pre-generation-probe-comparison-redline-pass-rate 1.0 \
  --max-runtime-drift-pre-generation-probe-comparison-best-test-label-auroc-drop 0.02 \
  --max-runtime-drift-pre-generation-probe-comparison-best-redline-auroc-drop 0.02 \
  --max-runtime-drift-pre-generation-probe-comparison-best-redline-margin-drop 0.02 \
  --min-runtime-drift-claim-factuality-probe-comparison-coverage 1.0 \
  --min-runtime-drift-claim-factuality-probe-comparison-manifest-verified-rate 1.0 \
  --min-runtime-drift-claim-factuality-probe-comparison-model-count 2 \
  --min-runtime-drift-claim-factuality-probe-comparison-run-count 2 \
  --min-runtime-drift-claim-factuality-probe-comparison-redline-pass-rate 1.0 \
  --max-runtime-drift-claim-factuality-probe-comparison-best-test-label-auroc-drop 0.02 \
  --max-runtime-drift-claim-factuality-probe-comparison-best-test-selective-accuracy-drop 0.02 \
  --max-runtime-drift-claim-factuality-probe-comparison-best-test-selective-coverage-drop 0.02 \
  --max-runtime-drift-claim-factuality-probe-comparison-best-redline-auroc-drop 0.02 \
  --max-runtime-drift-claim-factuality-probe-comparison-best-redline-margin-drop 0.02 \
  --min-runtime-drift-covered-fact-property-metric-count 3 \
  --min-runtime-drift-covered-fact-min-records 8 \
  --min-runtime-drift-covered-fact-min-source-documents 100 \
  --max-runtime-drift-covered-fact-min-decision-accuracy-drop 0.02 \
  --max-runtime-drift-covered-fact-max-false-supported-rate-increase 0.01 \
  --max-runtime-drift-covered-fact-min-false-refuted-rate-drop 0.02 \
  --min-runtime-drift-triple-extraction-fixture-matrix-coverage 1.0 \
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
evidence bundle; current runtime recommendations also require that score-fusion
candidate's high-confidence release gate to pass when present.
The selected-fusion handoff baseline at
`artifacts/smollm2_l8_read_cache_worker_sweep_selected_fusion_performance_baseline/`
reuses the same worker-sweep matrix plus
`artifacts/e7-truthfulqa-trajectory-multimodel/selected-fusion-artifact-build-report.json`,
selects the `smollm2` run explicitly, registers
`performance_baseline:smollm2-l8-read-cache-worker-sweep-selected-fusion-performance-baseline:0.3`,
and records
`manifest_verification:smollm2-l8-read-cache-worker-sweep-selected-fusion-performance-baseline-verification:0.3`.
It keeps the same recommended runtime cell and `truth_proj` best quality signal
while adding promoted `selected_fusion_mean_rank` evidence (`AUROC=0.692`,
false alarm `0.029`, detection `0.224`, `alpha=0.1`) from the SmolLM2
`geometry:mean_rank` selected artifact.
The corresponding selected-fusion staged structured-QA release candidate is
registered as
`benchmark_manifest:smollm2-l8-read-cache-worker-sweep-selected-fusion-staged-qa-release-candidate:0.3`
with
`artifacts/smollm2_l8_read_cache_worker_sweep_selected_fusion_staged_release_candidate_manifest.json`
and recursive verification
`artifacts/smollm2_l8_read_cache_worker_sweep_selected_fusion_staged_release_candidate_manifest_verification.json`.
It promotes with the same readiness and `structured_qa` route evidence, but its
performance gate is the selected-fusion baseline record.
The matching compact product handoff is
`product_promotion_contract:smollm2-l8-selected-fusion-product-promotion-contract:0.3`
with artifact manifest
`artifacts/smollm2_l8_selected_fusion_product_promotion_contract_v0_3/artifact-manifest.json`.
The corresponding staged structured-QA release candidate is registered as
`benchmark_manifest:smollm2-l8-read-cache-worker-sweep-score-fusion-staged-qa-release-candidate:0.2`
with
`artifacts/smollm2_l8_read_cache_worker_sweep_score_fusion_staged_release_candidate_manifest.json`.
Its manifest and release registry metadata carry
`recommended_score_fusion_status=promote`,
`recommended_score_fusion_signal=score_fusion_mean_rank`,
`recommended_score_fusion_auroc=0.679`,
`recommended_score_fusion_conformal_gate_passed=true`,
`recommended_score_fusion_release_gate_status=promote`, and
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
Use `audit_frontier_artifact_references.py` one level higher when the question is
whether the docs' active frontier/product handoff references are actually
materialized in the local checkout; it reports missing artifacts and verifies any
referenced artifact manifests that exist, then emits ordered
`recommended_actions` for regenerating and verifying missing handoff artifacts.
Use repeated `--json-cache` arguments to inspect additional persisted JSON
artifact caches, or `--no-json-cache` for a pure filesystem-only audit. The
restore mode never rewrites existing files; it only writes references that were
missing and had a valid cached JSON payload, with repo-local absolute paths
normalized for portability. When restoring a child of an artifact manifest, a
cache payload whose restored bytes would not match the manifest sha/size is
reported as a digest mismatch rather than written.

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
`benchmarks/product_promotion_contract_smoke.py`,
`benchmarks/frontier_release_evidence_smoke.py`,
`benchmarks/frontier_artifact_reference_smoke.py`,
`benchmarks/product_trace_replay_smoke.py`, and
`benchmarks/release_candidate_registry_smoke.py`. These use fixed synthetic profile
payloads plus the checked-in v1.9 product handoff and active frontier doc
references to verify that direct gates, cache-profile gates, worker-count sweep
decisions, INSIDE sampling sample-efficiency gates, registry-backed baselines,
the default promotion contract/evidence-handoff path, active frontier artifact
references, promoted frontier release-evidence report tracks, ProductTrace
replay, and release gates pass acceptable candidates, reject bounded telemetry
payloads where full traces are required, and catch expected regressions. They are
stable enough for default local/CI checks because they do not load a model or
measure machine speed. Use real `eval_truthfulqa.py --profile-json`
artifacts, `run_cache_profile_triplet.py`, or `run_inside_sampling_profile.py`
before making actual runtime or sampling-cost claims.

## `eval_trajectory_truthfulqa.py`

Replays statement-bearing TruthfulQA score dumps through a causal LM and scores
the hidden-state trajectory over forced-answer prediction positions. This is a
bridge between the synthetic `TrajectoryMonitor` sanity check and real
TruthfulQA artifacts: it reports trajectory-score Spearman/AUROC against the
score-dump true/false labels plus an NLL baseline, but it is still a
forced-answer proxy rather than open-generation evidence.

```bash
python benchmarks/eval_trajectory_truthfulqa.py \
  --scores artifacts/truthfulqa-l80-text-baseline-comparison/qwen-l80-text-baseline-scores.manifest.json \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --layers=-1,-6,-12 \
  --limit 64 \
  --json artifacts/qwen-l80-trajectory-report.json \
  --artifact-manifest artifacts/qwen-l80-trajectory-manifest.json \
  --quiet
```

Use `--layer` for the legacy single-layer report. Use `--layers` for a one-pass
layer sweep; the script runs one model forward per record and scores each
requested hidden-state layer from the same output.

On CPU hosts that fail with an illegal-instruction exit while loading PyTorch or
model kernels, rerun with `ATEN_CPU_CAPABILITY=default OMP_NUM_THREADS=1`.

Use the deterministic no-download smoke path for local wiring checks:

```bash
python benchmarks/eval_trajectory_truthfulqa.py \
  --offline \
  --json artifacts/trajectory-offline-smoke.json \
  --artifact-manifest artifacts/trajectory-offline-manifest.json \
  --quiet
```

## `compare_trajectory_sweeps.py`

Compares one or more `eval_trajectory_truthfulqa.py --layers` reports and
applies a fail-closed trajectory evidence gate. Defaults require at least two
reports, two model families, 100 evaluated examples per report, and AUROC >=
0.60. The current gpt2/SmolLM2 limit-128 comparison remains blocked because
SmolLM2 reaches only AUROC 0.560, so trajectory convergence is still
preliminary cross-model evidence rather than a release signal.

```bash
python benchmarks/compare_trajectory_sweeps.py \
  --report gpt2=artifacts/e7-truthfulqa-trajectory-multimodel/gpt2-qwen-l80-limit128-layer-sweep-report.json \
  --report smollm2=artifacts/e7-truthfulqa-trajectory-multimodel/smollm2-qwen-l80-limit128-layer-sweep-report.json \
  --json artifacts/e7-truthfulqa-trajectory-multimodel/trajectory-sweep-evidence-gate.json \
  --artifact-manifest artifacts/e7-truthfulqa-trajectory-multimodel/trajectory-sweep-evidence-gate-manifest.json \
  --verification-report artifacts/e7-truthfulqa-trajectory-multimodel/trajectory-sweep-evidence-gate-manifest-verification.json \
  --quiet
```

## `build_trajectory_fusion_artifact.py`

Converts a trajectory benchmark report into a rank-calibrated fusion artifact.
This is the bridge for using trajectory convergence as an optional fusion or
routing signal after the evidence gate records it as preliminary. It does not
promote trajectory convergence into a standalone release detector.

```bash
python benchmarks/build_trajectory_fusion_artifact.py \
  --trajectory-report artifacts/e7-truthfulqa-trajectory-multimodel/gpt2-qwen-l80-limit128-layer-sweep-report.json \
  --json artifacts/e7-truthfulqa-trajectory-multimodel/gpt2-trajectory-fusion-report.json \
  --artifact artifacts/e7-truthfulqa-trajectory-multimodel/gpt2-trajectory-fusion-artifact.json \
  --quiet
```

Use `--include-nll-answer` only for explicit ablations; the current gpt2
trajectory evidence has a weak NLL baseline, so NLL is not the recommended
default companion signal.

## `build_trajectory_signal_score_dump.py`

Writes a trajectory-enhanced score dump subset by aligning trajectory report
records back to the original score dump `index`. The trajectory direction is
stored in the output score dump config because the best anomaly direction can
change by model and layer.

```bash
python benchmarks/build_trajectory_signal_score_dump.py \
  --input-scores artifacts/truthfulqa-l80-text-baseline-comparison/qwen-l80-text-baseline-scores.manifest.json \
  --trajectory-report artifacts/e7-truthfulqa-trajectory-multimodel/gpt2-qwen-l80-limit128-layer-sweep-report.json \
  --output artifacts/e7-truthfulqa-trajectory-multimodel/gpt2-trajectory-scores.manifest.json \
  --output-format jsonl \
  --keep-signals truth_proj,subspace_resid,nll_answer \
  --quiet
```

## `run_fusion_ablation_matrix.py`

Evaluates named signal combinations with repeated split conformal calibration.
Use it to compare geometry-only, verifier/self-check-only, trajectory-only, and
mixed fusion candidates over the same aligned rows before promoting any new
route or default policy.

```bash
python benchmarks/run_fusion_ablation_matrix.py \
  --scores gpt2=artifacts/e7-truthfulqa-trajectory-multimodel/gpt2-trajectory-scores.manifest.json \
  --candidate geometry=truth_proj,subspace_resid \
  --candidate trajectory=trajectory_convergence \
  --candidate geometry_trajectory=truth_proj,subspace_resid,trajectory_convergence \
  --methods max_rank \
  --alphas 0.1 \
  --best-alpha 0.1 \
  --json artifacts/e7-truthfulqa-trajectory-multimodel/gpt2-fusion-ablation-matrix.json \
  --quiet
```

The committed limit-128 gpt2/SmolLM2 ablation artifacts use the same path with
expanded geometry and uncertainty controls. They show a model-dependent outcome:
gpt2 selects `geometry_trajectory:mean_rank` at alpha 0.1 (AUROC 0.701,
detection 0.229, false alarm 0.053), while SmolLM2 selects `geometry:mean_rank`
(AUROC 0.692, detection 0.224, false alarm 0.029). This keeps trajectory as a
conditional fusion/routing candidate rather than a default-added signal.

`select_fusion_signals_from_ablation.py` turns that matrix into a small
run-specific signal-selection report. It compares the best candidate containing
the tracked signal against the best baseline candidate and only enables the
tracked signal when detection/AUROC deltas pass and false-alarm increase stays
within policy.

```bash
python benchmarks/select_fusion_signals_from_ablation.py \
  --matrix artifacts/e7-truthfulqa-trajectory-multimodel/trajectory-fusion-ablation-matrix.json \
  --json artifacts/e7-truthfulqa-trajectory-multimodel/trajectory-fusion-signal-selection-report.json \
  --tracked-signal trajectory_convergence \
  --alpha 0.1 \
  --max-false-alarm-delta 0.03 \
  --quiet
```

The committed selector report enables trajectory for gpt2 and disables it for
SmolLM2, preserving trajectory as model-conditional evidence rather than a
global product default.

`build_selected_fusion_artifacts.py` then turns the selector output into
deployable `RankScoreFusionArtifact` files for each run, using the selected
signals and directions from the report.

```bash
python benchmarks/build_selected_fusion_artifacts.py \
  --selection-report artifacts/e7-truthfulqa-trajectory-multimodel/trajectory-fusion-signal-selection-report.json \
  --scores gpt2=artifacts/e7-truthfulqa-trajectory-multimodel/gpt2-trajectory-enhanced-scores.manifest.json \
  --scores smollm2=artifacts/e7-truthfulqa-trajectory-multimodel/smollm2-trajectory-enhanced-scores.manifest.json \
  --output-dir artifacts/e7-truthfulqa-trajectory-multimodel \
  --confidence-signal nll_answer \
  --max-high-confidence-accepted-false-rate 0.0 \
  --json artifacts/e7-truthfulqa-trajectory-multimodel/selected-fusion-artifact-build-report.json \
  --artifact-manifest artifacts/e7-truthfulqa-trajectory-multimodel/selected-fusion-artifact-manifest.json \
  --registry artifacts/local-readiness-registry.json \
  --name e7-truthfulqa-trajectory-selected-fusion-artifacts \
  --version 0.1 \
  --quiet
```

The committed build report writes `gpt2-selected-fusion-artifact.json` with
`truth_proj,subspace_resid,eigenscore,maha_last,trajectory_convergence` and
`smollm2-selected-fusion-artifact.json` with the geometry-only bundle. Both use
`mean_rank` and alpha 0.1. When a confidence signal is provided, the build report
also stores a selected-artifact release gate so runtime recommendation and
performance workflows can block promotion if the artifact accepts high-confidence
false answers. The optional artifact manifest fingerprints the build report,
per-run fusion artifacts, source selection report, and source score dumps, while
the optional registry entry records release-gate counts and confidence-audit
configuration. A runtime recommendation can consume this build report with
`--selected-fusion-artifact-report`; because this report has one artifact per
trajectory source run, provide `--selected-fusion-run gpt2` or
`--selected-fusion-run smollm2` explicitly.

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

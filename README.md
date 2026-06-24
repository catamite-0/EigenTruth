<div align="center">

# EigenTruth

**Research-preview PyTorch toolkit for LLM representation monitoring, geometric drift diagnostics, and experimental activation steering**

**面向大模型表征监测、几何漂移诊断与实验性激活引导的 PyTorch 研究预览工具库**

[![Status: Research Preview](https://img.shields.io/badge/status-alpha%20research%20preview-yellow.svg)]()
[![CI](https://github.com/catamitez0-maker/EigenTruth/actions/workflows/ci.yml/badge.svg)](https://github.com/catamitez0-maker/EigenTruth/actions/workflows/ci.yml)
[![Framework: PyTorch](https://img.shields.io/badge/framework-PyTorch%202.0%2B-ee4c2c.svg)](https://pytorch.org)
[![Python: 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB.svg)](https://python.org)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

[Quick start](#quick-start) | [Architecture](#architecture) | [Product Charter](docs/product-development-spec.md) | [Methodology](docs/methodology.md) | [Examples](examples/README.md) | [Roadmap](ROADMAP.md) | [Contributing](CONTRIBUTING.md) | [Security](SECURITY.md)

</div>

## Research Preview

EigenTruth is an alpha-stage research toolkit. It is intended for controlled experiments, diagnostics, and reproducible exploration. It is not production-ready, does not prove that an output is true, and must not be treated as a safety boundary for deployed systems.

EigenTruth 是一个处于 alpha 阶段的研究预览工具库，适用于受控实验、诊断和可复现探索。它尚未达到生产可用状态，不能证明模型输出为真，也不能作为已部署系统的安全边界。

The current implementation explores a research hypothesis: hallucination-related generation behavior may sometimes be accompanied by measurable geometric drift in hidden-state representations. The signals exposed by this project are experimental diagnostics, not calibrated factuality scores.

当前实现探索一个研究假设：与幻觉相关的生成行为有时可能伴随隐藏状态表征中可测量的几何漂移。本项目提供的信号属于实验性诊断指标，不是经过校准的事实性评分。

## What EigenTruth Does

EigenTruth wraps a decoder-only language model with PyTorch hooks. It can:

- build a `TruthManifold` from factual warmup examples
- track Mahalanobis-style distance from that warmup manifold
- project hidden states into a Poincare ball and compute Hyperbolic Semantic Entropy (HSE)
- optionally build a contrastive direction from factual and false examples
- fit a low-rank `TruthSubspace` and score residual distance from factual states
- calibrate diagnostic thresholds from benchmark score dumps and combine them with claim verification
- select a cheap pre-generation runtime profile from prompt and metadata risk markers
- compile risk decisions into structured action requests and dry-run execution results
- optionally apply experimental activation steering when a configured threshold is exceeded

EigenTruth 通过 PyTorch hook 包装 decoder-only 语言模型。它可以：

- 使用事实性 warmup 样本构建 `TruthManifold`
- 跟踪隐藏状态相对于 warmup 流形的马氏距离风格指标
- 将隐藏状态投影到庞加莱球并计算双曲语义熵（HSE）
- 可选地使用事实与错误样本构建对比方向
- 拟合低秩 `TruthSubspace`，并计算相对事实子空间的残差距离
- 从 benchmark 分数 dump 校准诊断阈值，并与 claim 验证结果组合成风险决策
- 基于 prompt 与 metadata 风险标记，在生成前选择低成本 runtime profile
- 将风险决策编译成结构化 action request 与 dry-run 执行结果
- 可选地在超过配置阈值时执行实验性激活引导

### What It Does Not Do

EigenTruth does not guarantee factual correctness, eliminate hallucinations, validate model safety, or replace external evaluation. Steering can change generation without improving truthfulness. Thresholds must be calibrated for each model, layer, dataset, and experiment.

EigenTruth 不能保证事实正确性，不能消除幻觉，不能验证模型安全性，也不能替代外部评估。激活引导可能改变生成结果，但不一定提升真实性。阈值必须针对每个模型、层、数据集和实验单独校准。

## Quick Start

### Installation

Core install, for the math engine and offline diagnostics:

```bash
pip install git+https://github.com/catamitez0-maker/EigenTruth.git
```

Install the Hugging Face extra when using `EigenTruthWrapper` with model-loading workflows:

```bash
pip install "eigentruth[hf] @ git+https://github.com/catamitez0-maker/EigenTruth.git"
```

For local development:

```bash
git clone https://github.com/catamitez0-maker/EigenTruth.git
cd EigenTruth
python -m venv .venv
# POSIX:   source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
# Optional: add Hugging Face/example dependencies for model-loading demos
python -m pip install -e ".[examples]"
```

### Minimal Integration

```python
from eigentruth import EigenTruthWrapper

monitor = EigenTruthWrapper(
    model=model,
    target_layer_idx=-8,
    steering_lambda=0.0,  # monitor-only mode
)
monitor.warmup(fact_dataset, tokenizer)
output = monitor.generate(**inputs, max_new_tokens=50)
print(monitor.get_diagnostics())
```

Start with `steering_lambda=0.0` to inspect diagnostics without modifying activations. Enable non-zero steering only for explicit intervention experiments.

建议先使用 `steering_lambda=0.0`，在不修改激活值的情况下检查诊断结果。仅在明确的干预实验中启用非零引导强度。

For a runnable model-loading demo, see [`examples/qwen_truth_demo.py`](examples/qwen_truth_demo.py). Example scripts may download model weights and are demonstrations rather than benchmarks. See [`examples/README.md`](examples/README.md) before adding or interpreting experiments.

### Calibrated Observability Workflow

```bash
python benchmarks/eval_truthfulqa.py --model gpt2 --layer -8 --sweep \
  --dump-scores benchmarks/scores.manifest.json \
  --dump-scores-format jsonl
python benchmarks/eval_conformal.py --scores benchmarks/scores.manifest.json \
  --json artifacts/gpt2-conformal-report.json \
  --save-sweep-report artifacts/gpt2-sweep-report.json \
  --save-best-calibration artifacts/gpt2-best-calibration.json \
  --artifact-manifest artifacts/gpt2-conformal-manifest.json
```

`--artifact-manifest` fingerprints the score dump, conformal report, sweep report,
and saved calibration artifacts so local registry/release workflows can verify the
calibration chain without rerunning the model.
Use `--dump-scores-format jsonl` for larger sweeps; it writes a compact manifest
plus records sidecar that downstream calibration and verifier tools can stream by
selected columns. New JSONL manifests also store label counts, so metadata
summaries can avoid scanning the records sidecar when only class counts are
needed. When a selected-column JSONL loader has already scanned the records
sidecar in the same run, it primes the run-local records fingerprint cache so
later provenance metadata can reuse the full SHA-256 without a second records
pass.

Use `--batch-size` and, when sampling INSIDE continuations, `--inside-batch-size`
to trade memory for benchmark throughput. Add `--inside-adaptive-sampling` to
treat `--inside-samples` as a maximum budget and stop early once lexical and
embedding entropy stabilize.

For real models, prefer threshold/top-fraction triggered INSIDE over
all-statement sampling; the SmolLM2 l20 profile shows full-sample
`adaptive_selfcheck` reduces generated samples only to 0.937 of fixed while
leaving `inside_generation` at 1.001 of fixed, while `truth_proj` top-25%
triggering samples 39/154 statements and cuts fixed `inside_generation` to
0.253 of full-sample fixed. The current registered SmolLM2 strict
structured-retrieval-audit release
`benchmark_manifest:smollm2-l20-inside-trigger-budget-derived-strict-structured-retrieval-audit-staged-qa-release-candidate:1.5`
selects the top-40% quality-balanced budget from a single largest-budget run,
uses 218 generated samples with sample-count ratio 0.472 and
`inside_generation` ratio 0.503 against the full-sample fixed reference,
requires `performance_baseline:smollm2-l20-performance-baseline:0.9` to match
the final runtime, requires promoted `structured_state`, `state_transition`, and
`retrieval_groundedness` plus `retrieval_structured_qa` adapter-family routes,
and requires
`benchmark_manifest:smollm2-l80-retrieval-structured-qa-route:0.5` as a separate
retrieval structured-QA audit baseline. That audit route promotes with selected
238, decision accuracy 0.992, false-supported rate 0.000, false-refuted rate
1.000, runtime about 1.05s, and 410 retrieval hits under a 450-hit budget. The
selected product route remains strict low-latency `structured_qa` with
`max_retrieval_use_rate=0.0` and `max_mean_attempted_route_count=1.1`.
Version 1.5 also requires promoted selector replay over 12 redacted product
traces and a promoted runtime-drift report comparing the trace replay baseline
against the promoted `auto` profile baseline; all 9 drift metrics pass with zero
blocked metrics.
The current registered product runtime profile sweep
`report:smollm2-product-runtime-profile-sweep:0.1` verifies that `latency`,
`balanced`, `audit`, and request-level `auto` selection all pass the strict product
runtime budget on deterministic control-plane traces. It also applies
`artifacts/smollm2_product_runtime_profile_sweep/runtime-profile-slo-policy.json`
as a sweep-level SLO gate for p95 trace latency, attempted route count,
retrieval-use rate, mean verified-claim count, and auto selector coverage. The
registered sweep recommends `auto`.
`report:smollm2-runtime-profile-selector-tuning:0.1` compares default,
latency-biased, and audit-biased auto selector policies under the same SLO gate;
the default policy is the only promoted candidate on the deterministic scenario
set.

`run_inside_sampling_profile.py` accepts `--inside-trigger-signal` plus
`--inside-trigger-threshold` or `--inside-trigger-top-fraction`, and
`--skip-existing` can resume interrupted profile comparisons without rerunning
completed variants. For trigger budget sweeps, pass `--shared-cache-dir` so each
budget/run shares statement encodings, layer stats, eval representations, and
sampled INSIDE diagnostics; use `--refresh-shared-caches` only when deliberately
rebuilding the first cache-producing run. Add `--max-batch-tokens` to cap padded
tokens per warmup/eval forward while keeping `--batch-size` as the row-count
cap, and compare budgets in profile matrices with
`--max-batch-token-budgets 0,512,1024`. Use `run_cache_profile_matrix.py` with
`--max-workers N` for independent matrix cells; shared-cache refresh cells still
run serially before dependent warm-start cells, and reports include end-to-end
`execution.wall_clock_seconds` for worker-count comparisons. Use
`--auto-batch-size` on long runs to halve and retry after retriable memory
errors. For repeated rescoring, pair `--eval-reps-cache` with
`--eval-reps-cache-shard-size`; sharded cache readers reuse recently touched
shards through a default 2-shard read-side LRU cache
(`--eval-reps-shard-read-cache-size`) and report cache IO counters such as read requests,
records read, shard loads, shard cache hits, and cross-shard reads in JSON output;
triplet, matrix, worker-sweep, and performance-baseline runners pass the setting
through for cached/cache-only runs. Use
`run_cache_profile_matrix.py --eval-reps-shard-read-cache-sizes 1,2,4` to
promote this from a heuristic cache-tuning suggestion into a same-machine sweep;
the recommended runtime then records the selected cell's read-cache size and
does not emit contradictory read-cache-size heuristic advice after a sweep has
already compared the candidate capacities.
The current small CPU SmolLM2 l8 evidence registers
`performance_baseline:smollm2-l8-read-cache-worker-sweep-performance-baseline:0.1`;
it keeps the read-cache sweep winner at size 2 and selects `max_workers=2`,
lowering matrix wall-clock from 184.467s to 141.385s on this machine while
preserving `truth_proj` AUROC 0.830.
Profile comparison and matrix reports
propagate the derived cache-efficiency hit-rate metrics for IO diagnosis.
Runtime recommendations include cache-tuning advice
when hit rate is low, cross-shard reads are high, or each cache read returns too
few records. Pair
repeated INSIDE sweeps with `--inside-diagnostics-cache` or the sweep-level
`--shared-cache-dir` to reuse sampled diagnostics for statements that appear in
nested trigger budgets. `run_adapter_readiness_workflow.py --performance-report`
can reuse an existing cache-profile matrix when only the INSIDE sampling report
or route evidence changes. The experimental `--prefix-kv-cache` path can reuse
shared question-prefix KV caches during eval forced-answer scoring when
candidate answers share a prefix; compare it with `run_cache_profile_matrix.py`
and `--prefix-kv-cache-modes off,on`, read the resulting `prefix_kv_comparisons` /
`forced_answer_forward_seconds` fields for forward-speed claims, and keep it
behind profile gates before changing defaults.

This produces a layer/score sweep report plus a reusable `CalibrationArtifact`
for the best calibrated diagnostic. The artifact can drive `RiskController`
decisions, and `RiskController.decide(..., verification_results=...)` can
compose calibrated diagnostics with claim-level verification in `ProductTrace`
records. The dependency-free `run_verification_loop(...)` helper can also
execute retrieve actions, feed retrieved evidence back into verification, and
emit a final decision trace; pass `StagedVerificationPolicy` when low-risk,
non-sensitive claims should skip expensive verifier routes while diagnostic risk
or claim metadata such as numbers, citations, or time-sensitive language still
triggers verification. Running `examples/calibrated_control_demo.py` with
`--pre-generation-profile auto` records the earlier prompt/metadata risk
assessment in trace metadata and, when no explicit runtime profile is supplied,
uses it to choose the first `latency`, `balanced`, or `audit` runtime profile.

For structured state or database-like adapters, generate deterministic state
fixtures and refresh the verifier-route artifact without rerunning model forward
passes:

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
  --verifier-report-json artifacts/order_fulfillment_state_verifier_report.json \
  --promotion-json artifacts/order_fulfillment_state_promotion_workflow.json \
  --route-report-json artifacts/order_fulfillment_state_route_comparison.json \
  --gate-route structured_state \
  --gate-min-selected 12 \
  --min-decision-accuracy 1.0 \
  --max-false-supported-rate 0.0 \
  --min-false-refuted-rate 1.0 \
  --compact-json \
  --fail-on-blocked
```

Use `--compact-json` on verifier ensemble, refresh, and adapter-promotion
workflow commands when reports are generated for automated gates rather than
manual inspection.
For action-conditioned checks, the same refresh/promotion path works with
`build_transition_fixture.py` and `--gate-route state_transition`.

Score directions are explicit: use `higher` for scores where larger values are more anomalous and `lower` for scores where smaller values are more anomalous. The shared directional conformal helpers keep thresholds, trigger rates, and selective reports in the same native score units. Control-plane diagnostics fail closed on invalid numeric inputs such as `NaN` or `Inf`, returning `clarify/unknown` instead of silently accepting the output.

## Architecture

```text
factual warmup texts
        |
        v
target-layer hidden states ---> TruthManifold
                                      |
generation hidden states -------------+
        |
        +--> distance diagnostic
        +--> Poincare-ball projection --> HSE diagnostic
        +--> optional threshold-triggered steering --> model generation
```

The high-level workflow is:

1. **Warm up**: collect final-token hidden states from factual texts and optionally false texts.
2. **Build diagnostics**: incrementally construct a ridge-regularized precision matrix and an optional contrastive direction.
3. **Attach a hook**: register a `forward_hook` on a selected Transformer layer.
4. **Monitor**: calculate representation-distance and HSE diagnostics during generation.
5. **Experiment with steering**: optionally inject a normalized steering vector after a configured threshold is exceeded.

工作流程：

1. **Warmup**：从事实文本和可选的错误文本中收集最后一个 token 的隐藏状态。
2. **构建诊断**：增量构建 ridge 正则化的精度矩阵和可选的对比方向。
3. **挂载 Hook**：在选定的 Transformer 层注册 `forward_hook`。
4. **监测**：在生成期间计算表征距离和 HSE 诊断指标。
5. **实验性引导**：可选地在超过配置阈值后注入归一化引导向量。

See [`docs/methodology.md`](docs/methodology.md) for the mathematical framing, calibration guidance, and limitations.

## Core Components

| Component | Purpose |
|---|---|
| `TruthManifold` | Maintains a Welford online mean and covariance, exposed as a ridge-regularized, sample-count-normalized precision matrix. |
| `mahalanobis_distance` | Measures relative deviation from the warmup manifold. |
| `poincare_map` | Projects representations into a bounded hyperbolic space. |
| `hyperbolic_semantic_entropy` | Measures dispersion over a sliding window of projected states. |
| `internal_eigenscore` / `spectral_effective_rank` / `cluster_assignment_entropy` / `lexical_semantic_entropy` / `embedding_semantic_entropy` | Computes INSIDE/EigenScore-style spectral diversity and dependency-free semantic-entropy proxies from sampled hidden-state/text clusters; benchmarks can optionally sample multiple continuations for `inside_eigenscore`, `inside_semantic_entropy`, and `inside_embedding_entropy`. |
| `TruthProbe` | Captures selected-layer hidden states and optionally applies steering. |
| `EigenTruthWrapper` | Provides warmup, generation passthrough, diagnostics, and probe lifecycle management. |
| `TruthSubspace` | Fits low-rank factual subspaces and residual-distance diagnostics; fitting requires at least two factual states. |
| `directional_conformal_threshold` / `directional_trigger_rate` | Apply split-conformal thresholds consistently for `higher` and `lower` anomalous score directions. |
| `ScoreDump` / `ScoreDumpIdentity` / `load_score_dump` / `load_score_dump_columns` / `load_score_dump_statement_scores` / `load_score_dump_layer_scores` / `score_dump_file_metadata` / `score_dump_identity` / `score_dump_cache_summary` | Validate per-statement score dumps once, expose compact run summaries, and attach SHA-256 provenance plus a stable model/dataset/layer/score-schema/scoring-config identity to post-hoc calibration or ensemble reports without rerunning models; callers can share an optional run-local cache to avoid repeated file hashing and repeated selected JSONL scans inside one run, then summarize cache hits/misses/writes for report observability. `ScoreDumpJsonlManifest` / `ScoreDumpRecord` / `iter_score_dump_jsonl_records` add an optional manifest-backed JSONL format for large dumps, selected-column loaders stream primary, statement-bearing, or layer/score views from that format, and cached selected views invalidate when the manifest or records file changes. Metadata fingerprints both the manifest and records file, includes a canonical identity cache key, uses manifest label counts for summary fast paths when available, and reuses records hashes primed by selected JSONL scans in the same run. `eval_truthfulqa.py --dump-scores-format jsonl` writes this format directly and stores per-record extras such as INSIDE sample counts in the records sidecar. |
| `LayerScoreSweepCalibrator` | Builds layer/score sweep reports and reusable calibration artifacts from score dumps, including direct `ScoreDump` inputs via `calibrate_from_score_dump()` and selected JSONL layer-score loading via `calibrate_from_file()`, with optional run-local cache reuse and bounded `max_workers` CPU parallelism for large post-hoc sweeps. |
| `ArtifactRegistry` / `build_artifact_manifest` / `fingerprint_path` / `load_fingerprint_cache` / `save_fingerprint_cache` / `verify_artifact_manifest` | Records and verifies local artifact metadata with dependency-free file/directory SHA-256 provenance for reproducible benchmark chains; recursive verification shares a run-local fingerprint cache, and CLIs can persist it as JSON so repeated manifest checks avoid duplicate content reads when file signatures are unchanged. |
| `RuntimeProfile` / `PreGenerationRiskPolicy` / `select_pre_generation_profile` / `RuntimeProfileSelectorPolicy` / `select_runtime_profile` | Defines shared `latency`, `balanced`, and `audit` defaults, plus dependency-free pre-generation prompt/metadata routing and post-diagnostic claim-metadata routing for product control-plane staging. |
| `ProductPromotionContract` / `ProductRuntimeEvidenceBundle` | Converts promoted release-candidate reports into product runtime, verifier-route, performance-bundle, performance score-dump cache, product-trace-replay workflow provenance, selector-replay, runtime-drift evidence, and budget-policy contracts; `load_product_promotion_contract()` loads compact contracts, while `load_product_runtime_evidence_bundle()` lazily attaches optional manifest verification and registry provenance. |
| `RiskController` / `ProductTrace` | Converts calibrated diagnostics plus optional verification results into structured routing decisions and JSON-ready traces; invalid diagnostic values route to `clarify/unknown`, route summaries expose selected/matched/skipped verifier tools, runtime/cache/tail-latency/route-cost summaries support optional budget gates, and `ProductTrace.to_bounded_dict()` emits smaller online telemetry payloads while corpus/replay/runtime-baseline tools keep full-trace reproduction inputs. |
| `DefaultCorrectionPolicy` / `ActionRequest` | Compiles control decisions into executable JSON-ready action payloads for product integrations, including generic `execute_tool` requests. |
| `ActionExecutorRegistry` / `DryRunActionExecutor` / `TimeoutActionExecutor` / `ActionResult` | Routes action requests to registered executors, with side-effect-free dry-run fallback and best-effort timeout wrapping for local traces. |
| `ActionExecutionPolicy` / `PolicyGuardedActionExecutor` | Adds dependency-free request validation, idempotency replay, and audit metadata for side-effecting executors, including request ids, idempotency keys, and timeout bounds. |
| `InMemoryActionExecutionLedger` / `JsonActionExecutionLedger` / `SQLiteActionExecutionLedger` | Stores successful idempotent action results so repeated product requests can replay outputs without repeating side effects. |
| `run_verification_loop` / `StagedVerificationPolicy` / `EvidenceBundle` | Runs verify -> decide -> execute -> reverify loops, optionally gates expensive verifier routes behind diagnostic risk or sensitive claim metadata, and converts retrieval action results into verifier-ready evidence context. |
| `RetrievalActionExecutor` / `InMemoryRetriever` | Provides a dependency-free retrieval executor shell for unsupported-claim evidence gathering. |
| `CalculatorVerifier` | Provides a dependency-free deterministic calculator verifier for structured arithmetic claims, simple symbolic equations, and calculation metadata extracted from limited arithmetic text. |
| `QuestionAnswerVerifier` | Provides a dependency-free structured QA/domain-state verifier adapter for exact question and candidate-answer facts. |
| `StructuredStateVerifier` / `StateCheck` | Provides a dependency-free structured state and business-rule verifier for database, policy, and domain-state checks. |
| `SQLiteStateSource` / `SQLiteStateQuery` | Loads read-only SQLite query results into structured verifier state without adding non-stdlib dependencies. |
| `ToolOutputStateSource` / `ToolOutputMapping` | Maps local tool or action execution outputs into structured verifier state for post-tool checks. |
| `StateTransitionVerifier` / `StateTransitionCheck` | Uses a world-model adapter to predict next state after an action, then checks structured postconditions. |
| `CachedVerifier` / `CachedRetriever` / `CachedStateSource` | Adds request-scoped in-memory caching and hit/miss stats for repeated verifier, retrieval, and state-source calls. |
| `CompositeVerifier` / `RoutedVerifier` | Compose deterministic tools with lexical, retrieval, database, or world-model verifiers; routing can use claim metadata, context, or text patterns and records match reasons. |
| `GroundednessVerifier` / `ClaimExtractor` | Extracts claim metadata and checks claims against lexical evidence snippets and explicit refutations without extra dependencies. |
| `SelfConsistencyVerifier` | Checks claims against caller-supplied sampled responses with FactSelfCheck-style support/refutation rates and no model or retrieval dependency. |
| `sqlite_state_control_demo.py` | Demonstrates SQLite-backed structured state checks feeding a final `ProductTrace` and dry-run action. |
| `state_transition_control_demo.py` | Demonstrates world-model next-state prediction plus structured postcondition checks feeding a final `ProductTrace`. |
| `production_tool_loop_demo.py` | Demonstrates a local production-like loop: SQLite pre-check, guarded side-effecting local `execute_tool`, optional JSON/SQLite idempotency ledger, tool-output state mapping, post-tool verification, action audit metadata, and route summary in one trace. |
| `eval_verifier_ensemble.py` | Benchmarks calibrated internal diagnostics combined with retrieval/verifier suppression/refutation policies, optional staged verifier gating, route-level cost metrics, and optional JSONL sidecar records for per-claim verifier outputs. |
| `run_calibrated_observability_workflow.py` | Runs or reuses a TruthfulQA score dump, executes conformal layer/score calibration, writes nested artifact manifests plus an evidence-bundle summary, and optionally records the calibrated-observability closure in the local registry. |
| `refresh_verifier_route_artifacts.py` | Regenerates new-schema verifier-route reports from saved score dumps, claims, and local verifier corpora without rerunning model forward passes. |
| `compare_verifier_routes.py` | Aggregates saved verifier-ensemble reports into cost-aware route leaderboards, Pareto frontier candidates, route-specific promotion decisions, by-route control-impact metrics, and optional tail/cache/staged-verification route quality gates. |
| `run_adapter_promotion_workflow.py` | Runs a fail-closed adapter promotion workflow: route comparison, `promotion_decision=promote`, and optional registry-backed performance baseline gate. |
| `run_adapter_promotion_registry_workflow.py` | Runs route promotion, writes a manifest, recursively verifies it, and registers the promoted route baseline in one command. |
| `compare_route_baselines.py` | Compares registered verifier-route promotion manifests by verified state, route quality, false support/refutation, tail latency, and retrieval cost. |
| `run_adapter_family_matrix.py` | Builds deterministic structured QA, structured-state, state-transition, optional retrieval-groundedness, and optional retrieval-structured-QA fixtures, then compares their promotion metrics in one local matrix. |
| `run_local_retrieval_route_workflow.py` | Builds local retrieval evidence from score dumps and corpora, promotes retrieval routes, fingerprints all source artifacts, records a runtime profile, optionally uses persistent SQLite FTS/cached claims fixtures/verifier traces, and optionally registers the route baseline. |
| `run_cache_profile_matrix.py` | Runs same-machine profile sweeps across layers, batch sizes, and capture modes, then emits a matrix-level performance promotion decision with per-cell AUROC quality signals. |
| `run_cache_worker_sweep.py` | Runs the same cache-profile matrix across several worker counts and recommends the fastest promoted worker count by wall-clock time. |
| `run_inside_sampling_profile.py` | Compares fixed, adaptive, and self-check-bounded INSIDE sampling runs, producing sample-count and `inside_generation` cost evidence for release gates. |
| `run_inside_trigger_budget_sweep.py` | Runs several triggered INSIDE budgets and compares generated samples, `inside_generation`, reference ratios, inside-score AUROCs, and cost-first / quality-balanced recommendations. |
| `recommend_runtime_config.py` | Converts promoted matrix/worker-sweep reports plus optional INSIDE sampling and trigger-budget sweep evidence into one deployable runtime recommendation: layer, batch size, token budget, prefix KV mode, worker count, sampling flags, derived-sweep flags, best available AUROC quality signal, and cache-tuning advice. |
| `run_performance_baseline_workflow.py` | Builds a registry-ready performance baseline bundle from cache matrix, optional worker sweep, optional INSIDE profile / trigger-budget evidence, runtime recommendation, artifact manifest, performance evidence summary, and optional recursive manifest verification. |
| `run_product_runtime_baseline.py` | Aggregates saved `ProductTrace` JSON files into a request-runtime baseline with phase, cache, verifier-route, retrieval-use, staged-verification savings, runtime-profile context, an `optimization` block with phase/route hotspots, cache/retrieval/staging/profile recommendations, optional `ProductRuntimeBudgetPolicy` gate metrics, optional reusable recommended policy artifact, optional JSONL sidecar storage for per-trace records, and optional trace-record cache reuse for repeated budget sweeps. |
| `compare_product_runtime_baselines.py` | Compares current product runtime baselines against a file or registered baseline, emitting fail-closed drift gates over latency, route cost, retrieval use, cache reuse, verifier skip rate, trace count, and an optional reusable runtime budget policy. |
| `build_product_trace_corpus.py` | Validates saved ProductTrace JSON/JSONL payloads, optionally redacts text fields, writes replay-ready standardized traces plus a runtime-pair index, can reuse a per-source validation/redaction cache, and registers a manifest-backed trace corpus. |
| `run_product_trace_replay_workflow.py` | Runs the raw-trace handoff end to end: redacted trace corpus, runtime-pair index, product runtime baseline, selector replay, optional product-runtime drift/policy gate, top-level runtime `optimization` summary, optional recommended runtime policy artifact, recursive manifest, optional manifest verification, optional manifest fingerprint cache reuse, optional whole-corpus cache reuse, optional corpus source-cache reuse, optional runtime trace-record cache reuse, optional selector trace-input cache reuse, phase timing/cache summaries, and registry-ready workflow report. |
| `run_product_runtime_profile_sweep.py` | Runs deterministic calibrated-control demo scenarios under `latency`, `balanced`, `audit`, and request-level `auto` selection modes, writes traces, builds per-mode baselines, applies optional aggregate SLO gates, and recommends the lowest-cost non-blocked mode. |
| `run_runtime_profile_selector_tuning.py` | Compares candidate `RuntimeProfileSelectorPolicy` JSON configs by running auto-profile sweeps under the same SLO gate and recommending the lowest-cost promoted selector. |
| `run_runtime_profile_selector_replay.py` | Replays candidate `RuntimeProfileSelectorPolicy` JSON configs over saved `ProductTrace` files, estimates profile cost/distribution, paired observed runtime, selected-vs-original runtime delta from trace scan or a corpus runtime-pair index, can cache minimal trace replay inputs for repeated policy sweeps, and registers the lowest-cost promoted selector. |
| `run_adapter_readiness_workflow.py` | Combines adapter-family quality gates, cache-profile performance gates, and optional INSIDE sampling / trigger-budget gates into one final readiness decision, runtime recommendation, and registry-ready manifest. |
| `run_adapter_readiness_registry_workflow.py` | Runs readiness gates and registers the verified manifest as a reusable local promotion baseline when readiness promotes. |
| `compare_readiness_baselines.py` | Compares registered readiness baselines by verified manifest state, best AUROC quality signal, runtime cost, and INSIDE profile or trigger-budget cost evidence, then recommends one deployable baseline. |
| `compare_release_candidates.py` | Combines registered readiness, route, optional required route-baseline, performance-baseline, product-trace-replay workflow file or registry key, selector-replay, product-runtime-drift, and adapter-family evidence into one fail-closed release candidate with runtime flags, verifier route, quality, runtime cost, performance evidence bundle/cache gates, optional performance trend gates against an explicit prior baseline, audit-route budgets, runtime drift provenance, and latency/balanced/audit profiles. |
| `run_release_candidate_registry_workflow.py` | Runs the release-candidate gate, writes a manifest covering the candidate report plus selected readiness/route/performance/product-trace-replay/selector/runtime-drift manifests and optional required route/adapter-family reports, recursively verifies it, and registers the final candidate with runtime-profile, performance-baseline, performance evidence bundle readiness/cost/cache/trend, product-trace-replay, selector-replay, runtime-drift, route-budget, and adapter-family metadata; compare, manifest-build, and promotion verification share one run-local fingerprint cache. |
| `export_product_promotion_contract.py` | Exports a compact, deployable `ProductPromotionContract` JSON from a promoted release candidate, writes a manifest, and can register a `product_promotion_contract:*:*` handoff artifact. |
| `build_domain_state_fixture.py` | Builds deterministic order-fulfillment score/claim/state fixtures plus optional SQLite state-source specs for structured-state verifier benchmarks. |
| `build_transition_fixture.py` | Builds deterministic order-reservation transition fixtures for state-transition verifier benchmarks. |
| `build_truthfulqa_corpus.py` | Builds a local TruthfulQA correct-answer corpus for reproducible non-oracle retrieval baselines. |
| `build_evidence_fixture.py` | Builds non-oracle claim/evidence fixtures from statement-bearing score dumps and local JSON/JSONL/text corpora. |
| `backfill_truthfulqa_statements.py` | Rebuilds deterministic TruthfulQA statement metadata for older score dumps and can emit label-derived oracle evidence for verifier upper-bound checks. |

### 主要组件

| 组件 | 用途 |
|---|---|
| `TruthManifold` | 用 Welford 维护在线均值与协方差，对外暴露为按样本数归一化、ridge 正则化的精度矩阵。 |
| `mahalanobis_distance` | 测量相对于 warmup 流形的相对偏移。 |
| `poincare_map` | 将表征投影到有界双曲空间。 |
| `hyperbolic_semantic_entropy` | 测量投影状态滑动窗口内的离散程度。 |
| `internal_eigenscore` / `spectral_effective_rank` / `cluster_assignment_entropy` / `lexical_semantic_entropy` / `embedding_semantic_entropy` | 基于隐藏态嵌入与文本簇计算 INSIDE/EigenScore 风格谱分散度和无依赖语义熵代理；benchmark 可选多采样续写生成 `inside_eigenscore`、`inside_semantic_entropy` 和 `inside_embedding_entropy`。 |
| `TruthProbe` | 捕获指定层的隐藏状态，并可选地应用激活引导。 |
| `EigenTruthWrapper` | 提供 warmup、生成透传、诊断信息和探针生命周期管理。 |
| `TruthSubspace` | 拟合低秩事实子空间，并提供残差距离诊断；拟合至少需要两条事实状态。 |
| `directional_conformal_threshold` / `directional_trigger_rate` | 对 `higher` 与 `lower` 异常方向使用一致的 split-conformal 阈值与触发率。 |
| `ScoreDump` / `ScoreDumpIdentity` / `load_score_dump` / `load_score_dump_columns` / `load_score_dump_statement_scores` / `load_score_dump_layer_scores` / `score_dump_file_metadata` / `score_dump_identity` / `score_dump_cache_summary` | 对逐陈述 score dump 做统一校验，暴露紧凑 run summary，并给后处理校准或 ensemble report 附带 SHA-256 provenance 和稳定的 model/dataset/layer/score-schema/scoring-config identity，不重跑模型；调用方可共享可选 run-local cache，避免单次运行内重复 hash 文件和重复扫描同一个 JSONL selected view，并可汇总 cache hits/misses/writes 用于报告观测。`ScoreDumpJsonlManifest` / `ScoreDumpRecord` / `iter_score_dump_jsonl_records` 提供可选的 manifest-backed JSONL 大文件格式，selected-column loader 可从该格式流式读取选中的 primary、带 statement 的 primary 或 layer/score 视图，缓存视图会在 manifest 或 records 文件变化后自动失效。metadata 会同时 fingerprint manifest 和 records 文件，包含 canonical identity cache key，在 manifest 自带 label counts 时走 summary fast path，并复用同一次运行中 selected JSONL scan 预热的 records hash。`eval_truthfulqa.py --dump-scores-format jsonl` 可直接写出该格式，并把 INSIDE sample counts 等逐记录字段放进 records sidecar。 |
| `LayerScoreSweepCalibrator` | 从分数 dump 构建层/分数 sweep report 与可复用校准 artifact，支持通过 `calibrate_from_score_dump()` 直接消费 `ScoreDump`，也支持 `calibrate_from_file()` 对 JSONL manifest 做 selected layer-score 读取，并可选复用 run-local cache；大规模后处理 sweep 可用受控 `max_workers` CPU 并行。 |
| `ArtifactRegistry` / `build_artifact_manifest` / `fingerprint_path` / `load_fingerprint_cache` / `save_fingerprint_cache` / `verify_artifact_manifest` | 用 dependency-free 的 file/directory SHA-256 provenance 记录和校验本地 artifact metadata，支持可复现 benchmark chain；递归 verification 共享 run-local fingerprint cache，CLI 可将其持久化为 JSON，在文件签名不变时避免重复内容读取。 |
| `RuntimeProfile` / `PreGenerationRiskPolicy` / `select_pre_generation_profile` / `RuntimeProfileSelectorPolicy` / `select_runtime_profile` | 定义 release gate 和产品控制面共用的 `latency`、`balanced`、`audit` 默认档位，并提供无依赖的生成前 prompt/metadata 路由与诊断后的 claim-metadata 路由。 |
| `ProductPromotionContract` / `ProductRuntimeEvidenceBundle` | 将已 promoted release-candidate report 转成产品 runtime、verifier route、performance-bundle、performance score-dump cache、product-trace-replay workflow provenance、selector-replay、runtime-drift evidence 和 budget policy contract；`load_product_promotion_contract()` 加载 compact contract，`load_product_runtime_evidence_bundle()` 延迟附加可选 manifest verification 与 registry provenance。 |
| `RiskController` / `ProductTrace` | 将校准诊断和可选验证结果转为结构化路由决策与 JSON trace；非法诊断值会路由到 `clarify/unknown`，route summary 会暴露选中、匹配和跳过的 verifier 工具，runtime/cache/tail-latency/route-cost summary 支持可选预算门禁，`ProductTrace.to_bounded_dict()` 可输出更小的线上 telemetry payload，corpus/replay/runtime-baseline 工具继续使用完整 trace 作为复现输入。 |
| `DefaultCorrectionPolicy` / `ActionRequest` | 将控制决策编译为面向产品集成的 JSON action payload，包括通用 `execute_tool` 请求。 |
| `ActionExecutorRegistry` / `DryRunActionExecutor` / `TimeoutActionExecutor` / `ActionResult` | 按 action 路由 executor，并用无副作用 dry-run 与 best-effort timeout wrapper 支撑本地 trace。 |
| `ActionExecutionPolicy` / `PolicyGuardedActionExecutor` | 为有副作用 executor 增加无依赖请求校验、idempotency replay 和审计元数据，包括 request id、idempotency key 与 timeout 上限。 |
| `InMemoryActionExecutionLedger` / `JsonActionExecutionLedger` / `SQLiteActionExecutionLedger` | 保存成功的幂等 action 结果，让重复产品请求可重放输出而不重复执行副作用。 |
| `run_verification_loop` / `StagedVerificationPolicy` / `EvidenceBundle` | 执行 verify -> decide -> execute -> reverify 闭环，可按诊断风险或敏感 claim metadata 延迟触发昂贵 verifier，并把 retrieval action result 转成 verifier 可消费的 evidence context。 |
| `RetrievalActionExecutor` / `InMemoryRetriever` | 为 unsupported claim 的取证流程提供无依赖 retrieval executor shell。 |
| `CalculatorVerifier` | 提供无依赖确定性计算器 verifier，用于结构化算术 claim、简单符号等式，以及从有限算术文本中抽取出的 calculation metadata。 |
| `QuestionAnswerVerifier` | 提供无依赖结构化 QA/领域状态 verifier adapter，用于精确问题与候选答案事实。 |
| `StructuredStateVerifier` / `StateCheck` | 提供无依赖结构化状态与业务规则 verifier，用于数据库、策略和领域状态校验。 |
| `SQLiteStateSource` / `SQLiteStateQuery` | 将只读 SQLite 查询结果加载为 verifier 可消费的结构化状态，不增加非标准库依赖。 |
| `ToolOutputStateSource` / `ToolOutputMapping` | 将本地工具或 action 执行输出映射成结构化 verifier state，用于工具调用后的校验。 |
| `StateTransitionVerifier` / `StateTransitionCheck` | 通过 world-model adapter 预测 action 后的下一状态，再校验结构化 postcondition。 |
| `CachedVerifier` / `CachedRetriever` / `CachedStateSource` | 为重复 verifier、retrieval 和 state-source 调用提供 request-scoped 内存缓存与 hit/miss 统计。 |
| `CompositeVerifier` / `RoutedVerifier` | 组合确定性工具与词面、检索、数据库或 world-model verifier；路由可依据 claim metadata、context 或文本模式，并记录匹配原因。 |
| `GroundednessVerifier` / `ClaimExtractor` | 抽取 claim metadata，并用词面证据片段和显式反证检查 claim，不增加核心依赖。 |
| `SelfConsistencyVerifier` | 用调用方提供的 sampled responses 对 claim 做 FactSelfCheck 风格支持/反证率检查，不增加模型或检索依赖。 |
| `sqlite_state_control_demo.py` | 演示 SQLite 结构化状态校验如何进入最终 `ProductTrace` 和 dry-run action。 |
| `state_transition_control_demo.py` | 演示 world-model 下一状态预测和结构化 postcondition 校验如何进入最终 `ProductTrace`。 |
| `production_tool_loop_demo.py` | 演示本地 production-like 闭环：SQLite 前置校验、受 guard 约束的有副作用本地 `execute_tool`、可选 JSON/SQLite idempotency ledger、工具输出状态映射、工具后校验、action audit metadata 和 trace route summary。 |
| `eval_verifier_ensemble.py` | 评估校准内部诊断与 retrieval/verifier 抑制误报、补充反证检出的组合策略，可选 staged verifier gating，记录 route 级成本指标，并可用 JSONL sidecar 保存逐 claim verifier 输出。 |
| `run_calibrated_observability_workflow.py` | 运行或复用 TruthfulQA score dump，执行 conformal layer/score 校准，写入嵌套 artifact manifest 和 evidence-bundle summary，并可选把 calibrated-observability 闭环登记到本地 registry。 |
| `refresh_verifier_route_artifacts.py` | 从已保存 score dump、claims 和本地 verifier corpus 重新生成新 schema verifier-route report，不重跑模型 forward。 |
| `compare_verifier_routes.py` | 将已保存 verifier-ensemble report 聚合为成本感知 route 排行榜、Pareto frontier 候选、分 route promotion decision、分 route 控制收益指标和可选 tail/cache/staged-verification route 质量门槛。 |
| `run_adapter_promotion_workflow.py` | 执行 fail-closed adapter promotion workflow：route comparison、`promotion_decision=promote` 和可选 registry-backed 性能基线门槛。 |
| `run_adapter_promotion_registry_workflow.py` | 一次性执行 route promotion、写 manifest、递归验证 manifest，并把 promoted route baseline 注册到本地 registry。 |
| `compare_route_baselines.py` | 按 manifest 验证状态、route 质量、误支持/反证率、尾延迟和 retrieval 成本比较已注册 verifier-route promotion baseline。 |
| `run_adapter_family_matrix.py` | 构建确定性的 structured QA、structured-state、state-transition、可选 retrieval-groundedness 和可选 retrieval-structured-QA fixtures，并在一个本地矩阵里比较 promotion 指标。 |
| `run_local_retrieval_route_workflow.py` | 从 score dump 和本地 corpus 构建 retrieval evidence，promote retrieval route，指纹化全部源 artifact，记录运行 profile，可选使用持久化 SQLite FTS/claims fixture/verifier trace 缓存，并可选注册 route baseline。 |
| `run_cache_profile_matrix.py` | 跨 layer、batch size 和 capture mode 执行同机 profile sweep，并输出矩阵级性能 promotion decision 和每个 cell 的 AUROC quality signals。 |
| `run_cache_worker_sweep.py` | 用多个 worker count 运行同一 cache-profile matrix，并按 wall-clock 推荐最快的已 promoted worker count。 |
| `run_inside_sampling_profile.py` | 比较 fixed、adaptive 和 self-check-bounded INSIDE sampling，输出 sample-count 与 `inside_generation` 成本证据，供 release gate 使用。 |
| `run_inside_trigger_budget_sweep.py` | 比较多个 triggered INSIDE budget，输出生成样本数、`inside_generation`、参考全量比例、inside-score AUROC，以及成本优先 / 质量折中的推荐。 |
| `recommend_runtime_config.py` | 将 promoted matrix/worker-sweep report 与可选 INSIDE sampling / trigger-budget sweep 证据转成可执行 runtime recommendation：layer、batch size、token budget、prefix KV、worker count、sampling flags、derived-sweep flags、最佳 AUROC quality signal 和 cache-tuning 建议。 |
| `run_performance_baseline_workflow.py` | 将 cache matrix、可选 worker sweep、可选 INSIDE profile / trigger-budget 证据、runtime recommendation、artifact manifest、performance evidence summary 和可选递归 manifest verification 打包成可注册 performance baseline。 |
| `run_product_runtime_baseline.py` | 聚合已保存的 `ProductTrace` JSON，输出请求级 runtime baseline：phase、cache、verifier route、retrieval 使用率、staged-verification 节省量、runtime-profile context、带 phase/route 热点和 cache/retrieval/staging/profile 建议的 `optimization` 块、可选 `ProductRuntimeBudgetPolicy` gate、可复用推荐 policy artifact、逐 trace record 的可选 JSONL sidecar 存储，以及重复 budget sweep 可复用的 trace-record cache。 |
| `compare_product_runtime_baselines.py` | 将当前 product runtime baseline 与文件或 registry 中的基线比较，对 latency、route cost、retrieval 使用、cache 复用、verifier skip rate、trace 数量和可选复用 runtime budget policy 输出 fail-closed drift gate。 |
| `build_product_trace_corpus.py` | 校验已保存的 ProductTrace JSON/JSONL，可选脱敏文本字段，写出 replay-ready 标准化 trace 和 runtime-pair index，可复用 per-source 校验/脱敏缓存，并登记带 manifest 的 trace corpus。 |
| `run_product_trace_replay_workflow.py` | 端到端执行 raw trace handoff：脱敏 trace corpus、runtime-pair index、产品 runtime baseline、selector replay、可选 product-runtime drift/policy gate、顶层 runtime `optimization` 摘要、可选推荐 runtime policy artifact、递归 manifest、可选 manifest verification、可选 manifest fingerprint cache 复用、可选 whole-corpus cache 复用、可选 corpus source-cache 复用、可选 runtime trace-record cache 复用、可选 selector trace-input cache 复用、phase timing/cache summary 和可注册 workflow report。 |
| `run_product_runtime_profile_sweep.py` | 在 `latency`、`balanced`、`audit` 和请求级 `auto` selection modes 下运行确定性 calibrated-control demo 场景，写 trace、生成每个 mode 的 baseline，应用可选聚合 SLO 门禁，并推荐最低成本的未阻断 mode。 |
| `run_runtime_profile_selector_tuning.py` | 通过同一套 SLO gate 比较多个 `RuntimeProfileSelectorPolicy` JSON 候选，运行 auto-profile sweep，并推荐成本最低的 promoted selector。 |
| `run_runtime_profile_selector_replay.py` | 在已保存的 `ProductTrace` 上回放多个 `RuntimeProfileSelectorPolicy` JSON 候选，不重跑 demo 即可通过 trace scan 或 corpus runtime-pair index 估算 profile 成本、分布、配对 observed runtime 和 selected-vs-original runtime delta，可缓存最小 trace replay input 以便重复策略 sweep，并登记成本最低的 promoted selector。 |
| `run_adapter_readiness_workflow.py` | 将 adapter-family 质量门槛、cache-profile 性能门槛和可选 INSIDE sampling / trigger-budget gate 合并为最终 readiness decision、runtime recommendation 和可注册 manifest。 |
| `run_adapter_readiness_registry_workflow.py` | 运行 readiness gate，并在 readiness promote 后把已验证 manifest 注册成本地可复用 promotion baseline。 |
| `compare_readiness_baselines.py` | 按 manifest 验证状态、最佳 AUROC quality signal、runtime cost 和 INSIDE profile / trigger-budget 成本证据比较已注册 readiness baseline，并推荐一个可部署 baseline。 |
| `compare_release_candidates.py` | 将已注册 readiness baseline、route baseline、可选 required route-baseline、performance baseline、product-trace-replay workflow 文件或 registry key、selector-replay、product-runtime-drift 和 adapter-family 证据合成一个 fail-closed release candidate，输出 runtime flags、verifier route、质量、runtime cost、performance evidence bundle/cache gate、可选显式 prior baseline 性能趋势 gate、audit-route budget、runtime drift provenance，以及 latency/balanced/audit runtime profiles。 |
| `run_release_candidate_registry_workflow.py` | 执行 release-candidate gate，写入覆盖 candidate report、选中 readiness/route/performance/product-trace-replay/selector/runtime-drift manifests 和可选 required route / adapter-family report 的 manifest，递归验证后登记带 runtime-profile、performance-baseline、performance evidence bundle readiness/cost/cache/trend、product-trace-replay、selector-replay、runtime-drift、route-budget 和 adapter-family metadata 的最终候选；compare、manifest build 和 promotion verification 共享同一个 run-local fingerprint cache。 |
| `export_product_promotion_contract.py` | 从 promoted release candidate 导出紧凑的可部署 `ProductPromotionContract` JSON，写入 manifest，并可登记 `product_promotion_contract:*:*` handoff artifact。 |
| `build_domain_state_fixture.py` | 构建确定性的订单履约 score/claim/state fixture，并可输出 SQLite state-source spec，用于结构化状态 verifier benchmark。 |
| `build_transition_fixture.py` | 构建确定性的订单预留 state-transition fixture，用于 world-model/postcondition verifier benchmark。 |
| `build_truthfulqa_corpus.py` | 构建本地 TruthfulQA correct-answer corpus，用于可复现的非 oracle retrieval baseline。 |
| `build_evidence_fixture.py` | 从带 statement 的 score dump 和本地 JSON/JSONL/text 文档库构建非 oracle claim/evidence fixture。 |
| `backfill_truthfulqa_statements.py` | 为旧版 TruthfulQA score dump 重建确定性 statement metadata，并可输出标签派生 oracle evidence 用于 verifier 上界测试。 |

`TimeoutActionExecutor` uses a stdlib thread-pool timeout so the control loop can
return a traceable timeout result. It cannot safely terminate an already-running
Python thread, so side-effecting adapters still need their own hard cancellation
or transactional/idempotent safety boundary.

`TimeoutActionExecutor` 使用标准库线程池超时，让控制闭环能返回可追踪的
`timed_out` 结果；它不能安全终止已经运行中的 Python 线程，因此有副作用
adapter 仍需要自身提供强取消、事务或幂等安全边界。

## Experimental Model Compatibility

The hook layer resolver includes paths commonly used by several Hugging Face model families. Compatibility may vary by model and architecture version and should be verified with a small warmup run before conducting an experiment.

Hook 层解析器包含若干 Hugging Face 模型系列常用的路径。兼容性会随架构版本变化，正式实验前应通过小规模 warmup 运行进行验证。

| Architecture family | Example models | Candidate layer path |
|---|---|---|
| Llama-style | Llama, Qwen, Mistral | `model.model.layers` |
| GPT-2-style | GPT-2, GPT-Neo | `model.transformer.h` |
| GPT-NeoX-style | Pythia, GPT-NeoX | `model.gpt_neox.layers` |
| OPT-style | OPT | `model.model.decoder.layers` |
| Custom | Other compatible models | `custom_layer_path="your.path"` |

This table describes resolver support, not a guarantee that every listed model release has been validated.

## Qualitative Demonstration

[`examples/adversarial_test.py`](examples/adversarial_test.py) compares outputs with and without steering for a small set of prompts. The results are qualitative demonstrations under a specific model, warmup set, target layer, threshold, and generation configuration.

[`examples/adversarial_test.py`](examples/adversarial_test.py) 在一个小规模 prompt 集合上比较启用和禁用激活引导时的输出。结果仅是在特定模型、warmup 集合、目标层、阈值和生成配置下的定性演示。

Do not interpret output changes as benchmark evidence or as proof that a correction is factually valid. Any research claim should use reproducible scripts, external evaluation, and human review.

不要将输出变化解释为基准测试证据，也不要将其视为纠正结果具有事实有效性的证明。任何研究结论都应使用可复现实验脚本、外部评估和人工审查。

## Testing

```bash
python -m pytest tests/ -v
python -m ruff check src tests examples benchmarks
```

For local development, the Makefile auto-detects `.venv/bin/python` when present:

```bash
make check-fast    # lint + unit tests + dependency consistency
make check         # check-fast plus deterministic smoke workflows
make perf-check     # deterministic profile/cache/worker/registry smokes; no model load
make release-check  # also builds the package
```

The unit suite covers numerical stability, hook behavior, warmup, diagnostics, and wrapper lifecycle. It does not replace evaluation against factuality benchmarks or model-specific integration testing.

单元测试覆盖数值稳定性、hook 行为、warmup、诊断信息和 wrapper 生命周期。它不能替代事实性基准测试或针对具体模型的集成测试。

## Maintainer Workflow

For routine changes:

1. Create a focused branch.
2. Make the smallest coherent change.
3. Add or update tests for behavior changes.
4. Run `python -m pytest tests/ -v`.
5. Run `python -m ruff check src tests examples benchmarks`.
6. Run `python -m pip check` and `python -m build` before packaging-oriented changes.
7. Update documentation when experiment assumptions, interfaces, or limitations change.
8. Open a pull request with the motivation, validation steps, and any research caveats.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the complete contributor workflow and [`ROADMAP.md`](ROADMAP.md) for near-term priorities.

## How Codex Helps Maintain This Project

Codex can help maintainers inspect the repository, propose scoped changes, add tests, improve documentation, run local checks, review diffs, and prepare pull requests. For this research-preview project, Codex should support human review rather than replace it.

When using Codex on EigenTruth:

- keep changes narrow and reviewable
- preserve honest research-preview language
- run tests and lint before publishing
- document assumptions for experiment scripts
- avoid turning qualitative observations into safety or benchmark claims
- require maintainer review before merge

Codex 可以帮助维护者检查仓库、提出范围明确的改动、补充测试、改进文档、运行本地检查、审阅 diff 并准备 pull request。对于这个研究预览项目，Codex 应当支持人工审查，而不是替代人工审查。

## Repository Layout

```text
EigenTruth/
|-- src/eigentruth/
|   |-- core/math_engine.py       # geometry and online manifold updates
|   |-- intervention/hooks.py     # hook-based diagnostics and steering
|   `-- models/wrapper.py         # user-facing wrapper
|-- tests/                        # unit tests
|-- examples/                     # qualitative demonstration scripts
|-- docs/methodology.md           # research framing and limitations
|-- ROADMAP.md
|-- CONTRIBUTING.md
`-- SECURITY.md
```

## Citation

If EigenTruth is useful for your research, cite the repository and include the commit SHA used for your experiment:

```bibtex
@software{eigentruth2025,
  title   = {EigenTruth: Geometric Representation Monitoring and Steering for LLMs},
  author  = {EigenTruth Team},
  year    = {2025},
  url     = {https://github.com/catamitez0-maker/EigenTruth},
  license = {Apache-2.0}
}
```

如果 EigenTruth 对你的研究有帮助，请引用本仓库，并在实验记录中包含所使用的 commit SHA。

## Contributing And Security

Contributions are welcome. Read [`CONTRIBUTING.md`](CONTRIBUTING.md) before opening a pull request. For security-sensitive reports, follow [`SECURITY.md`](SECURITY.md) and avoid filing public issues until a disclosure path has been agreed.

欢迎贡献。提交 pull request 前请阅读 [`CONTRIBUTING.md`](CONTRIBUTING.md)。对于安全敏感问题，请遵循 [`SECURITY.md`](SECURITY.md)，并在确认披露流程前避免创建公开 issue。

## License

Apache License 2.0. See [`LICENSE`](LICENSE).

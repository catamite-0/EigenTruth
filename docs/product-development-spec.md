# EigenTruth Product And Development Charter

EigenTruth is a representation-observability and factuality-control toolkit for LLM systems. It is not a standalone truth oracle, a safety boundary, or a replacement for external evidence. The product direction is to combine internal model diagnostics, calibrated risk scoring, tool-based verification, and optional world-model correction while preserving the language model's creative strengths.

EigenTruth 的产品定位是：面向 LLM 系统的表征观测与事实性控制工具集。它不是单一的“反幻觉模型”，也不是生产安全边界。产品方向是把模型内部诊断、风险校准、工具验证和可选世界模型校正组合起来，在保留 LLM 创造性的同时降低事实错误。

## Current State

Implemented today:

- `TruthManifold`: online hidden-state mean/covariance, ridge-regularized precision, save/load support.
- `TruthProbe`: PyTorch forward-hook monitor for selected Transformer layers, Mahalanobis-style drift, HSE tracking, and optional activation steering.
- `EigenTruthWrapper`: warmup, generation passthrough, diagnostics, and probe lifecycle management for Hugging Face-style PyTorch models.
- `eigentruth.eval`: conformal p-values/thresholds plus CPU-testable metrics such as AUROC, Euclidean dispersion, selective accuracy, coverage, and confidence intervals.
- `eigentruth.calibration`: JSON-serializable calibration artifacts, split-conformal calibrators, and layer/score sweep reports.
- `eigentruth.core.TruthSubspace`: low-rank factual subspace scoring, benchmark residual signal, and optional true-minus-false projection.
- `eigentruth.core.internal_eigenscore` / `lexical_semantic_entropy` / `embedding_semantic_entropy`: INSIDE/EigenScore-style spectral diversity and dependency-free sampled semantic-entropy proxies, including adaptive sampling budget support in benchmarks.
- `eigentruth.control.RiskController` / `ControlPolicyConfig`: maps calibrated diagnostic thresholds and optional claim verification results to configurable product actions.
- `eigentruth.control.DefaultCorrectionPolicy` / `ActionRequest`: compiles decisions into executable JSON-ready payloads for accept/retrieve/rewrite/steer/execute-tool/abstain/clarify flows.
- `eigentruth.control.ActionExecutorRegistry` / `DryRunActionExecutor` / `TimeoutActionExecutor` / `ActionResult`: routes action requests to registered executors with side-effect-free fallback execution and best-effort timeout wrapping for local control-loop traces.
- `eigentruth.control.ActionExecutionPolicy` / `PolicyGuardedActionExecutor`: validates side-effecting action requests, replays idempotent results when a ledger is configured, and records audit metadata such as request ids, idempotency keys, timeout bounds, and timeout-enforcement status.
- `eigentruth.control.InMemoryActionExecutionLedger` / `JsonActionExecutionLedger` / `SQLiteActionExecutionLedger`: store successful idempotent action results for request-local, JSON-file, or SQLite-backed replay without repeating side effects.
- `eigentruth.control.ProductTrace`: JSON-ready traces for diagnostics, claims, verification, decisions, action execution summaries, verifier-route summaries, request runtime summaries, and metadata.
- `eigentruth.control.run_verification_loop` / `StagedVerificationPolicy` / `EvidenceBundle`: dependency-free verify -> decide -> execute -> reverify loop that can gate expensive verifier routes behind diagnostic risk or sensitive claim metadata, then feed retrieval action results back into verifier context.
- `eigentruth.verify`: dependency-free pluggable claim extraction, rule-based claim metadata, in-memory verifier tools, lexical groundedness checks, and sampled-response self-consistency verification for first-pass claim workflows.
- `eigentruth.adapters.RetrievalActionExecutor` / `InMemoryRetriever`: dependency-free retrieval executor shell for unsupported-claim evidence gathering.
- `eigentruth.adapters.ToolOutputStateSource` / `ToolOutputMapping`: maps local tool/action execution outputs into structured verifier state for post-tool checks.
- `eigentruth.adapters.InMemoryWorldModelAdapter`: deterministic world-model adapter for tests and domain-rule prototypes.
- `eigentruth.adapters.StateTransitionVerifier`: dependency-free action-conditioned postcondition verifier that predicts next state through a world-model adapter, then reuses structured state checks.
- `eigentruth.registry.ArtifactRegistry`: local JSON registry for calibration reports, calibration artifacts, traces, reports, action results, and saved concept metadata.
- Benchmark scripts for TruthfulQA-style evaluation, TruthSubspace residual scoring, layer/score sweeps, conformal calibration, and selective reporting.
- Local development baseline: `make check` and `make release-check` run lint, tests, dependency consistency, deterministic profile-gate smoke checks, and package build.

Recent cleanup and platform changes:

- Core dependency is intentionally light: `torch` only.
- Hugging Face, examples, and benchmark datasets are optional extras.
- CI covers Python 3.10-3.13 with minimum/latest dependency lanes.
- Build metadata uses SPDX license syntax and validates with `python -m build`.

## Product Definition

EigenTruth should become a toolkit with four layers:

1. **Observe**: inspect hidden-state geometry during warmup, forward passes, and generation.
2. **Calibrate**: convert raw signals into reproducible thresholds, p-values, and abstention policies.
3. **Control**: decide whether to monitor, steer, retrieve evidence, rewrite, abstain, or ask for clarification.
4. **Verify**: validate atomic claims against tools, retrieval, structured data, or domain/world-model adapters.

The default product experience should be monitor-first. Steering and correction are opt-in and must expose diagnostics explaining why they fired.

## Non-Goals

- Do not claim that EigenTruth proves an answer is true.
- Do not claim that activation steering eliminates hallucinations.
- Do not hide uncertainty behind confident rewrites.
- Do not make Hugging Face, datasets, retrieval systems, or world models mandatory core dependencies.
- Do not optimize only for a single benchmark if it weakens general observability.

## User Workflows

### 1. Researcher Workflow

A researcher should be able to:

1. Build a manifold from factual warmup examples.
2. Sweep candidate layers and scoring methods.
3. Calibrate thresholds on held-out data.
4. Compare raw diagnostics, conformal risk, and external benchmark labels.
5. Save an experiment artifact with model revision, layer, seed, thresholds, and commit SHA.

### 2. Application Developer Workflow

An application developer should be able to:

1. Attach EigenTruth to an existing PyTorch/HF generation stack.
2. Run in monitor-only mode with no activation changes.
3. Receive structured risk diagnostics per generation.
4. Route high-risk answers through retrieval, verification, rewrite, or abstention.
5. Keep domain-specific tools optional and replaceable.

### 3. Product Workflow

A product system should be able to:

1. Generate a draft answer creatively.
2. Extract atomic claims from the draft.
3. Score internal drift and semantic instability.
4. Verify high-risk claims with external tools or world models.
5. Produce a final answer with confidence, evidence, and correction trace.

## Proposed Runtime Architecture

```text
user request
  |
  v
task and risk router
  |
  v
LLM draft generation <---- optional retrieval/context injection
  |
  +--> EigenTruth internal diagnostics
  |
  v
atomic claim extraction
  |
  +--> retrieval / database / calculator / code tools
  +--> world-model or domain-state verifier
  |
  v
risk controller
  |
  +--> accept
  +--> rewrite with evidence
  +--> steer and regenerate
  +--> abstain or ask clarification
  |
  v
final answer + diagnostics + evidence trace
```

## World Model Positioning

World-model correction should be an adapter layer, not a hard dependency and not part of the core manifold math.

Recommended interface:

```python
class WorldModelAdapter:
    def verify(self, claim, context): ...
    def predict(self, state, action): ...
    def explain(self, claim): ...
```

Use the world model in three places:

1. **Before generation**: constrain the prompt with domain state, physical limits, business rules, or time-sensitive facts.
2. **After draft generation**: verify atomic claims and correct unsupported statements.
3. **During high-risk generation**: if internal diagnostics spike, pause free generation and route through verification.

This keeps LLM creativity in wording, synthesis, analogy, and exploration, while requiring evidence for factual claims, numbers, citations, physical constraints, and domain-state updates.

## Automatic Parameter Tuning

EigenTruth should treat layer choice, threshold choice, and steering strength as calibrated artifacts, not magic constants.

Planned artifact:

```text
CalibrationArtifact
- model_id
- model_revision
- target_layer
- score_names
- thresholds
- conformal_alpha
- steering_policy
- warmup_dataset_metadata
- calibration_dataset_metadata
- created_at
- eigentruth_version
- commit_sha
```

The calibrator should support:

- layer sweeps
- score sweeps (`maha`, `truth_proj`, future subspace scores)
- conformal thresholds
- steering-lambda sweeps
- coverage and power reports
- CPU-friendly smoke runs before large-model replication

## Module Boundaries

Current modules should remain narrow:

- `eigentruth.core`: tensor math, manifolds, distances, projections. No model loading, no datasets, no network.
- `eigentruth.intervention`: hooks and steering logic for PyTorch modules.
- `eigentruth.models`: user-facing wrappers around model instances.
- `eigentruth.eval`: metrics, conformal thresholds, selective reports, and benchmark helpers.
- `eigentruth.calibration`: calibration artifacts and parameter sweeps.
- `eigentruth.control`: risk controller, policy configuration, correction policy, action executor registry, verification loops, and traces.
- `eigentruth.verify`: claim extraction, verifier protocols, in-memory verification, and lexical groundedness.
- `eigentruth.registry`: local artifact metadata records.
- `eigentruth.adapters`: optional integration shells, including retrieval and world models.
- `examples/`: qualitative demos only; no benchmark claims.
- `benchmarks/`: reproducible evaluations with structured output.

Integration modules should stay interface-first: plain dataclasses, enums, and protocols with no heavyweight dependencies. Concrete retrieval, database, rewrite, and world-model implementations should land behind these contracts as evidence and tests justify them.

## Development Principles

1. **Evidence before claims**: every product claim must be tied to a test, benchmark, or documented limitation.
2. **Monitor before steer**: new interventions should first ship as diagnostics or dry-run policies.
3. **Small core, optional integrations**: keep mandatory dependencies minimal and push integrations into extras.
4. **CPU-first validation**: every new scientific idea needs a small CPU-testable mechanism check.
5. **Calibrate per setting**: thresholds must be model/layer/domain/dataset specific unless evidence shows transfer.
6. **Structured outputs**: diagnostics and benchmarks should produce machine-readable artifacts.
7. **Negative results are assets**: failed hypotheses should be documented rather than silently removed.
8. **No hidden global state**: hooks, probes, and adapters must have explicit lifecycle management.
9. **Safe default behavior**: default mode should not modify model activations.
10. **Composable APIs**: components should be usable independently in existing LLM systems.

## Definition Of Done

For code changes:

- Public behavior is covered by focused tests.
- `make check` passes locally.
- `make release-check` passes for dependency, package, or public API changes.
- New dependencies are optional unless they are required by core math.
- Documentation is updated when behavior, interfaces, dependencies, or limitations change.

For experiment changes:

- The experiment states question, method, accept criterion, deliverable, and cost.
- Dataset provenance, model revision, seed, layer, thresholds, and commit SHA are recorded.
- Results are saved as structured JSON when practical.
- Claims distinguish qualitative observations from benchmark evidence.

For product features:

- The feature has a monitor-only or dry-run mode.
- The feature exposes diagnostics explaining its decision.
- Failure modes are documented.
- High-risk corrections can abstain or ask for clarification instead of fabricating certainty.

## Near-Term Product Roadmap

### Completed 0.2-0.3 Foundation

- Score dump -> layer/score sweep -> conformal calibration artifact -> risk decision -> action request/result -> product trace.
- Configurable risk policy hooks for refuted, unsupported, error, and compound diagnostic/verification cases.
- Action executor registry with dry-run fallback and a dependency-free in-memory retrieval executor shell.
- Claim extraction metadata for numbers, citations, negation, and time-sensitive claims.
- Benchmark reports now include selective accuracy, coverage, and confidence intervals.

### Completed 0.4 Verification Loop Shell

- Dependency-free verify -> decide -> execute -> reverify helper with final `ProductTrace` output and optional staged verifier gating for low-risk, non-sensitive claims.
- `EvidenceBundle` conversion from retrieval `ActionResult` payloads into claim-scoped verifier evidence context.
- Demo and tests for unsupported -> retrieve -> supported, no-hit retrieve, and refuted-claim hard stop paths.
- `eval_verifier_ensemble.py` benchmark shell for comparing calibrated internal diagnostics against retrieval/verifier suppression and refutation policies, including structured QA, static state, and action-conditioned state-transition routes.
- `build_truthfulqa_corpus.py` for creating a local TruthfulQA correct-answer evidence corpus.
- `build_evidence_fixture.py` for building non-oracle verifier fixtures from statement-bearing score dumps and local JSON/JSONL/text corpora.
- `backfill_truthfulqa_statements.py` for adding statement metadata to older TruthfulQA score dumps without rerunning models.
- Qwen l80 / SmolLM2 l80 oracle verifier-ensemble upper-bound report: label-derived perfect evidence drives verified false alarm to 0.000 and detection to 1.000 at alpha 0.100.
- Verifier ensemble reports include label-conditioned `verification_quality`, so evidence fixture quality can be measured separately from downstream control-policy metrics.
- Qwen l80 / SmolLM2 l80 local-corpus verifier baseline: conservative correct-answer retrieval drives verified false alarm to 0.008 at alpha 0.100, with true-supported rate 0.908 and false-supported rate 0.042.
- `QuestionAnswerVerifier` structured QA/domain-state adapter and Qwen l80 / SmolLM2 l80 structured QA baseline: exact question-answer facts drive verified false alarm to 0.000 and detection to 1.000 at alpha 0.100 on covered TruthfulQA questions.
- Staged structured QA gate artifact: at alpha 0.100, Qwen l80 / SmolLM2 l80 skip 79.3% / 82.9% of structured QA verifier calls, aggregate staged verified false alarm is 0.009, aggregate staged verified detection is 0.275, and `compare_verifier_routes.py` promotes `structured_qa` under explicit staged skip/quality gates. The promoted staged route is registered as `benchmark_manifest:truthfulqa-l80-structured-qa-staged-route:0.4`, and `compare_route_baselines.py` promotes it as the current local verifier-route baseline.
- `CalculatorVerifier` deterministic tool adapter for structured arithmetic claims and simple symbolic equations, using a restricted local arithmetic evaluator with no new mandatory dependencies.
- `StructuredStateVerifier` / `StateCheck` structured state and business-rule adapter for database, policy, and domain-state checks. It can be routed through `state_check` claim metadata or context and returns supported, refuted, insufficient-evidence, or error results with explicit decision rules.
- `SQLiteStateSource` / `SQLiteStateQuery` load read-only SQLite query results into nested verifier state, giving the structured-state path a real database integration without new mandatory dependencies.
- `ToolOutputStateSource` / `ToolOutputMapping` map local action or tool execution outputs into nested verifier state, giving post-tool checks a structured path without requiring a new tool runtime.
- `StateTransitionVerifier` / `StateTransitionCheck` verify claims about action consequences by predicting next state through a world-model adapter and checking structured postconditions; `InMemoryWorldModelAdapter` supports nested and dotted-path `set`/`increment`/`decrement` actions for deterministic domain-rule tests.
- `sqlite_state_control_demo.py` seeds an order/inventory/account SQLite fixture and emits a final `ProductTrace` where database state drives a dry-run abstain action despite low internal diagnostics.
- `state_transition_control_demo.py` emits a final `ProductTrace` where a world-model predicted postcondition refutes a claim about action consequences despite low internal diagnostics.
- `production_tool_loop_demo.py` emits one product-style trace that combines SQLite pre-tool checks, guarded side-effecting local `execute_tool`, optional JSON/SQLite idempotency replay, mapped `ActionResult.output`, post-tool structured verification, action execution summary, action audit metadata, and runtime route summary metadata.
- `TimeoutActionExecutor` can return a traceable `timed_out` action result using stdlib thread-pool timeouts; it is a best-effort local boundary, not a hard cancellation mechanism for already-running side-effecting adapters.
- `CompositeVerifier`, `RoutedVerifier`, and `calibrated_control_demo.py --enable-calculator` show a tool-first product trace path: claim metadata, context, or text patterns can route calculator-supported/refuted claims before lexical verifier fallback.
- `StagedVerificationPolicy` can be passed to `run_verification_loop` to skip expensive verifier routes for low-risk, non-sensitive claims while still triggering verification for calibrated diagnostic risk or claim metadata such as numbers, citations, time sensitivity, or explicit `requires_verification` flags.
- `SelfConsistencyVerifier` can be evaluated as a `self_consistency` verifier-ensemble route when fixtures provide sampled responses, giving claim-level support/refutation rates before retrieval fallback.
- `build_selfcheck_fixture.py` converts dumped INSIDE sampled continuations or external sampled-generation files into verifier-ensemble fixtures for reproducible self-consistency route evaluation.
- `RoutedVerifier` records route match reasons such as metadata keys, context keys, text patterns, feature flags, or fallback selection; `ProductTrace.verification_route_summary()` aggregates selected, matched, and skipped verifier routes for runtime trace review.
- `RuntimeProfile` now exposes shared `latency`, `balanced`, and `audit` defaults for release gates and product control-plane staging; `calibrated_control_demo.py --runtime-profile` writes the selected profile and staging behavior into trace and registry metadata.
- `RuntimeTrace` / `RuntimePhaseTiming` records dependency-free request phase timings in `ProductTrace.runtime_trace`, so product runs can compare diagnostic, verification, action execution, retrieval evidence, and re-verification costs before changing performance defaults.
- `ProductRuntimeBudgetPolicy` / `evaluate_product_runtime_budget` turns `ProductTrace.runtime_trace`, cache metadata, and verifier route metadata into optional fail-closed product gates over total request time, named phase timings, phase p95/p99 timings, route duration, mean attempted routes, retrieval use rate, aggregate cache hit rate, and named cache hit rates; `calibrated_control_demo.py --max-runtime-total-seconds/--max-runtime-phase-seconds/--max-runtime-phase-p95-seconds/--max-runtime-phase-p99-seconds/--max-mean-attempted-route-count/--max-retrieval-use-rate/--min-cache-hit-rate` writes the evaluated result into trace and registry metadata.
- `eval_verifier_ensemble.py` now supports `--state-source` JSON state and SQLite query specs plus fixture-level `state_check` and `state_transition` routes, and reports `route_summary`, `route_quality`, route-level cost/tail-latency metrics, optional `--staged-verification` skip-rate/cost gating metrics, and per-alpha `route_control_impact` for structured QA, state transition, structured state, lexical groundedness, retrieval-backed groundedness, and staged verifier skips; `--compact-json` keeps large automated report artifacts smaller without changing payload semantics.
- `refresh_verifier_route_artifacts.py` regenerates current-schema verifier route reports from saved score dumps, claims, and local verifier corpora so old artifacts can be upgraded for promotion gates without rerunning model forward passes; its summary exposes route latency/cost/cache metrics for promotion audit, supports compact verifier/route/promotion/workflow JSON output, and regression coverage now includes structured QA, structured-state, and state-transition promotion gates.
- `build_domain_state_fixture.py` generates deterministic order-fulfillment score/claim/state fixtures plus optional SQLite database and SQLite state-source spec so structured-state verifier behavior can be benchmarked without relying on label-derived TruthfulQA oracle evidence.
- `build_transition_fixture.py` generates deterministic order-reservation score/claim/state fixtures so world-model predicted postconditions can be benchmarked without a real simulator or external dependency.
- `compare_verifier_routes.py` aggregates route-level verifier quality, runtime cost, tail latency, cache summary, staged-verification skip/quality metrics, and control impact across saved verifier-ensemble reports, producing a cost-aware route leaderboard, Pareto frontier candidates, route-specific promotion decisions, by-route weighted metrics, and optional fail-closed route quality gates without loading models. Aggregate cost means use only observation counts paired with finite source totals; missing or non-finite source metrics are surfaced as `invalid_metric_counts` and block promotion instead of being silently averaged away; configured staged gates fail closed when staged evidence is missing or verifier skipping degrades verified false alarm/detection beyond threshold.
- `run_adapter_promotion_workflow.py` composes route comparison, route-specific `promotion_decision=promote`, and optional registry-backed performance baseline gates into one fail-closed adapter promotion report, with compact JSON output and optional registry-ready artifact manifests available for automation artifacts.
- `run_adapter_promotion_registry_workflow.py` closes the route-baseline automation loop: it runs adapter promotion, writes the artifact manifest, recursively verifies it, and registers the promoted verifier-route baseline in `ArtifactRegistry` unless the route decision is blocked. Registry metadata includes staged-verification skip/quality metrics when the source route comparison was built with staged gates.
- `compare_route_baselines.py` compares registered verifier-route promotion manifests without rerunning model or verifier work, recursively verifies each manifest, reloads saved route-comparison reports, applies optional quality/cost/tail/retrieval/runtime/cache-reuse gates, fails closed on invalid source metrics or enabled budget metrics that are missing/non-finite, and recommends one deployable route baseline.
- `run_adapter_family_matrix.py` builds deterministic structured QA, structured-state, state-transition, and optional retrieval-groundedness fixtures, runs each through refresh/promotion gates, and aggregates the route-family quality/cost/tail/cache metrics into one local no-model comparison matrix.
- `run_local_retrieval_route_workflow.py` turns a statement-bearing score dump and local JSON/JSONL/text corpora into a registered retrieval-groundedness route baseline, including the generated claims fixture, verifier report, route comparison, promotion report, source corpus, score dump, optional persistent SQLite FTS candidate retrieval metadata, optional fingerprint-keyed claims and verifier-trace cache metadata, lightweight runtime profile metadata, and optional runtime/cache budget gates in the manifest for recursive provenance and cost checks.
- `run_adapter_readiness_workflow.py` composes adapter-family quality promotion with same-machine cache-profile performance promotion into one final fail-closed adapter readiness decision, writes `runtime-recommendation.json` from the promoted performance matrix plus optional INSIDE sampling profile / trigger-budget sweep evidence, and includes it in the top-level artifact manifest for recursive verification and registry promotion.
- `run_adapter_readiness_registry_workflow.py` runs the readiness workflow, requires readiness promotion by default, recursively verifies the top-level manifest, and records it in `ArtifactRegistry` as the reusable local promotion baseline.
- `CachedVerifier`, `CachedRetriever`, and `CachedStateSource` provide request-scoped caching plus hit/miss stats for repeated verifier, retrieval, and state-source calls; `eval_verifier_ensemble.py` includes these `cache_stats` in each run report.
- `eval_truthfulqa.py --profile` now emits a structured performance summary with bottleneck phase, top phases, grouped time shares, and warmup/eval throughput fields for reproducible optimization comparisons.
- `eval_truthfulqa.py --auto-batch-size` can halve and retry warmup/forced-answer batch size after retriable memory errors, recording requested/effective batch size and reduction events in JSON/profile output.
- `eval_truthfulqa.py --max-batch-tokens` caps padded tokens per warmup/eval forward while preserving `--batch-size` as the row-count cap; triplet, matrix, and readiness workflows pass it through for same-machine performance gates. `run_cache_profile_matrix.py --max-batch-token-budgets` can compare several token budgets in one triplet matrix and ranks those cells by uncached forced-answer forward time.
- `run_cache_profile_matrix.py --max-workers` and `run_adapter_readiness_workflow.py --performance-max-workers` can execute independent performance matrix cells concurrently. The default remains serial; shared-cache refresh cells are still run behind a conservative barrier before dependent warm-start/cache-only cells are parallelized. Matrix/readiness reports record end-to-end wall-clock seconds so worker-count comparisons are auditable.
- `run_cache_worker_sweep.py` wraps the cache-profile matrix across several worker counts, isolates shared-cache roots per worker count, and recommends the fastest promoted worker count by end-to-end wall-clock time.
- `eval_truthfulqa.py --prefix-kv-cache` is an experimental shared-prefix forced-answer path for eval scoring: repeated question prefixes are run once and answer continuations reuse the prefix KV cache while preserving answer-window hidden states and NLL semantics. It is off by default; use cache-profile matrices with `--prefix-kv-cache-modes off,on`, inspect `prefix_kv_comparisons` / `forced_answer_forward_seconds` for forward-speed claims, and promote only through profile/readiness gates.
- `eval_truthfulqa.py --statement-encoding-cache` persists tokenizer outputs and answer-span lengths as validated JSON so repeated benchmark/cache rebuilds avoid redundant tokenizer setup without changing scoring semantics.
- Sharded `--eval-reps-cache` readers reuse the active shard across adjacent batch reads, reducing repeated `torch.load` calls during small-batch cache-only or cached scoring runs; JSON output includes `cache_stats.eval_reps_reader` counters when an eval-reps reader is used.
- `compare_profiles.py` compares profile JSON payloads across baseline/cache/cache-only runs, reporting speedup, phase deltas, grouped time deltas, and throughput ratios without loading models. Optional regression gates can fail CI/local checks when total time, phase time, or throughput ratios exceed configured limits.
- `run_cache_profile_triplet.py` is the optional local performance harness: it runs uncached, cached, and cache-only `eval_truthfulqa.py` profiles on the same machine and writes a gated `compare_profiles.py` report. The default path uses the offline fixture, while `--real-truthfulqa` generates representative model/data profile artifacts with configurable model, dtype, layer, limit, and warmup size.
- `run_inside_sampling_profile.py` compares fixed INSIDE sampling, adaptive entropy-stability sampling, and adaptive self-check-bound sampling on the same machine, writing per-run result/profile JSON plus a sample-efficiency comparison report so generated-sample savings and `inside_generation` time ratios can be promoted as reproducible artifacts. Optional shared statement/layer/eval caches let comparable sampling runs avoid repeated tokenization, warmup, and forced-answer forward work while still running real INSIDE generation for sampled records.
- `run_inside_trigger_budget_sweep.py` wraps that profile runner across several `--inside-trigger-*` budgets, writing one sweep report with generated samples, `inside_generation`, ratios to an optional full-sample reference, inside-score AUROCs, a cost-first recommendation, and a quality-balanced recommendation that chooses the cheapest budget within a small AUROC tolerance of the best INSIDE quality signal. `--shared-cache-dir` applies the same cache reuse across all budget children. For nested top-fraction budgets with one run, `--derive-from-max-budget` executes only the largest budget and derives smaller rows from the score dump while preserving batch-local top-fraction semantics.
- `run_cache_profile_matrix.py` expands the triplet harness across layer, batch-size, and hidden-state-capture combinations, writing a single same-machine matrix report with command logs, gate summaries, cache-only timing, AUROC quality signals, wall-clock execution metadata, and a fail-closed matrix-level performance promotion decision. In `rescore` mode, later cache-only cells in the same shared-cache group are gated against the first cell's uncached baseline so post-processing/cache-only variants can be promoted without repeating model forward passes.
- `profile_gate_smoke.py`, `cache_profile_smoke.py`, `inside_sampling_profile_smoke.py`, `cache_worker_sweep_smoke.py`, and `make perf-check` provide deterministic no-model smoke checks for profile regression, cache/cache-only gate machinery, INSIDE sampling sample-efficiency gates, worker-count sweep decisions, and registry-backed baselines; real performance claims still require representative `eval_truthfulqa.py --profile-json` artifacts.
- `recommend_runtime_config.py` turns promoted cache-profile matrix, optional worker-sweep reports, optional INSIDE sampling profile reports, and optional trigger-budget sweep reports into one deployable runtime recommendation with layer, batch size, capture mode, token budget, prefix-KV mode, worker count, sampling configuration, selected trigger budget, best available AUROC quality signal, and equivalent benchmark flags. It supports explicit trigger-budget selection policies: `quality_balanced` preserves the default release posture, `cost_first` minimizes sampled INSIDE work for latency-constrained deployments, and `quality_first` chooses the highest measured INSIDE quality metric. It preserves `--derive-from-max-budget` in generated sweep flags, performs no model work, and should be treated as the final handoff from same-machine performance evidence to local deployment defaults.
- `compare_readiness_baselines.py` compares registered adapter-readiness manifests without rerunning model work: it recursively verifies each manifest, regenerates runtime recommendations from saved performance matrices and optional INSIDE sampling profile / trigger-budget sweep artifacts when possible, applies optional quality/runtime/sampling-cost gates, uses uncached total time as a conservative forward-cost fallback for legacy reports, and recommends the passing baseline with the best AUROC quality signal. Sampling gates use profile `*_to_baseline` ratios when present and trigger-sweep `*_to_reference` ratios when baseline ratios are absent; reports record the ratio source. Trigger-budget sweep selection can be explicitly overridden with `quality_balanced`, `cost_first`, or `quality_first`; otherwise it follows the recorded readiness/runtime policy and falls back to the quality-balanced release posture.
- `compare_release_candidates.py` combines registered readiness and verifier-route baselines into the final local release gate: both underlying baseline comparisons must promote, including any readiness sampling-cost gates and route runtime/cache budget gates, then the report emits one release candidate with runtime flags, verifier route, quality evidence, runtime cost, route cost, selected trigger-budget policy, trigger-budget metadata, and ratio-source fields. Optional `latency`, `balanced`, and `audit` runtime profiles fill unset performance/cost gate defaults while leaving explicit quality thresholds such as AUROC to the caller.
- `run_release_candidate_registry_workflow.py` registers a final local release candidate by writing the release-candidate report, building a manifest over that report plus the selected readiness and route manifests, recursively verifying the whole chain, and recording the verified candidate in `ArtifactRegistry` with recommended runtime, selected runtime profile, selected trigger-budget policy, INSIDE sampling / trigger-budget cost evidence when available, route quality, tail latency, attempted-route, and retrieval-use metadata.
- Local release-candidate artifacts: `benchmark_manifest:tiny-local-staged-qa-release-candidate:0.4` combines the tiny-gpt2 offline readiness/runtime baseline with the TruthfulQA l80 staged structured QA route baseline, and `benchmark_manifest:tiny-local-inside-staged-qa-release-candidate:0.5` adds a registered INSIDE sampling profile gate. `benchmark_manifest:smollm2-l20-inside-staged-qa-release-candidate:0.6` is the first non-tiny local release candidate: SmolLM2-135M l20 readiness promotes `truth_proj` AUROC `0.682`, uncached forced-answer cost `38.786s`, cache-only replay cost `0.339s`, and the same staged structured QA route baseline. Its full-sample INSIDE profile selects `adaptive_selfcheck` but only reduces generated samples to `0.937` of fixed while `inside_generation` remains `1.001` of fixed. `benchmark_manifest:smollm2-l20-inside-triggered-staged-qa-release-candidate:0.7` promotes the same readiness/route chain with `truth_proj` top-25% triggered INSIDE: 39/154 statements sampled, 115 skipped, fixed `inside_generation` reduced from `467.563s` to `118.513s` (`0.253x`), and the promoted `adaptive_selfcheck` triggered run using 110 generated samples. `benchmark_manifest:smollm2-l20-inside-trigger-budget-derived-staged-qa-release-candidate:0.8` is the current registered SmolLM2 default: it reuses the same readiness/route evidence, folds in the derived trigger-budget sweep, and selects the quality-balanced top-40% triggered `adaptive_selfcheck` profile with sample-count ratio `0.472`, `inside_generation` ratio `0.503`, semantic-entropy AUROC `0.570`, and 218 generated samples from a single largest-budget source run. The same verified sweep can now drive a cost-first release-gate override that selects the top-10% trigger budget for latency-constrained deployments. This makes trigger-gated, derived-sweep INSIDE the current real-model default direction.

### Next Verification Adapter Work

- Connect `StructuredStateVerifier` to additional live database, business-rule, and domain-state sources behind optional adapters; the current reproducible path uses local JSON state sources, stdlib SQLite, and mapped local tool outputs.
- Extend claim extraction or upstream tool calls to provide structured `expression` / `expected` metadata for calculator-verifiable claims beyond simple symbolic equations.
- Connect additional QA/database/world-model route policies to production adapters while preserving trace-visible match reasons for each selected or skipped tool.
- Generalize real side-effecting executors behind the same trace, execution-policy, timeout, idempotency-ledger, and tool-output mapping interfaces; the current production-like demo proves the path with a guarded local SQLite mutation and optional JSON/SQLite replay, while hard cancellation remains adapter-specific.
- Extend route-level metrics to future adapters and real fixtures, then compare them with `compare_verifier_routes.py` so new tools are judged by route hit rate, false support, false refutation, runtime cost, tail latency, cache efficiency, downstream conformal control impact, Pareto frontier status, route-specific promotion decision, and explicit gates before adapter work proceeds.
- Replace label-derived oracle evidence with real retrieval/database/calculator evidence and rerun verifier/retrieval ensemble reports on Qwen/SmolLM2.
- Use local corpus fixtures from `build_evidence_fixture.py` as the reproducible baseline before adding networked retrieval extras.
- Add optional retrieval/database/calculator verifier adapters behind extras, keeping core dependencies unchanged.
- Add concrete domain/world-model adapters beyond the in-memory test double.
- Add semantic-entropy sampling probes, RAG groundedness adapters, and optional SAE/ReFT integrations behind extras.
- Add benchmark/demo domain examples where facts depend on state transitions, physical constraints, or business rules.
- Compare real verifier/retrieval ensembles against the current `truth_proj` baseline under the same conformal false-alarm budget.

### Later Control-Plane Hardening

- Persist full trace/report/action-result artifacts from benchmark and demo commands into local registries by default when requested.
- Add policy graph composition only after current dataclass-based policy config becomes insufficient.
- Add steering-lambda sweep utilities after monitor-only diagnostics are stable on larger replicated runs.
- Extend registry persistence from metadata records to saved subspace/manifold tensors.

## Product Tagline

EigenTruth is a factuality control plane for LLM systems: it observes hidden-state drift, calibrates risk, routes uncertain generations through tools, and leaves creativity where creativity belongs.

EigenTruth 是 LLM 系统的事实性控制平面：观测表征漂移，校准风险，把不确定生成路由到工具验证，同时保留模型该有的创造力。

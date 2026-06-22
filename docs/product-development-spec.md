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
- `eigentruth.core.internal_eigenscore`: INSIDE/EigenScore-style spectral diversity score for hidden-state embeddings.
- `eigentruth.control.RiskController` / `ControlPolicyConfig`: maps calibrated diagnostic thresholds and optional claim verification results to configurable product actions.
- `eigentruth.control.DefaultCorrectionPolicy` / `ActionRequest`: compiles decisions into executable JSON-ready payloads for accept/retrieve/rewrite/steer/abstain/clarify flows.
- `eigentruth.control.ActionExecutorRegistry` / `DryRunActionExecutor` / `ActionResult`: routes action requests to registered executors with side-effect-free fallback execution for local control-loop traces.
- `eigentruth.control.ProductTrace`: JSON-ready traces for diagnostics, claims, verification, decisions, action execution summaries, and metadata.
- `eigentruth.control.run_verification_loop` / `EvidenceBundle`: dependency-free verify -> decide -> execute -> reverify loop that feeds retrieval action results back into verifier context.
- `eigentruth.verify`: dependency-free pluggable claim extraction, rule-based claim metadata, in-memory verifier tools, and lexical groundedness checks for first-pass claim workflows.
- `eigentruth.adapters.RetrievalActionExecutor` / `InMemoryRetriever`: dependency-free retrieval executor shell for unsupported-claim evidence gathering.
- `eigentruth.adapters.InMemoryWorldModelAdapter`: deterministic world-model adapter for tests and domain-rule prototypes.
- `eigentruth.registry.ArtifactRegistry`: local JSON registry for calibration reports, calibration artifacts, traces, reports, action results, and saved concept metadata.
- Benchmark scripts for TruthfulQA-style evaluation, TruthSubspace residual scoring, layer/score sweeps, conformal calibration, and selective reporting.
- Local development baseline: `make check` and `make release-check` run lint, tests, dependency consistency, and package build.

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

- Dependency-free verify -> decide -> execute -> reverify helper with final `ProductTrace` output.
- `EvidenceBundle` conversion from retrieval `ActionResult` payloads into claim-scoped verifier evidence context.
- Demo and tests for unsupported -> retrieve -> supported, no-hit retrieve, and refuted-claim hard stop paths.
- `eval_verifier_ensemble.py` benchmark shell for comparing calibrated internal diagnostics against retrieval/verifier suppression and refutation policies.
- `build_truthfulqa_corpus.py` for creating a local TruthfulQA correct-answer evidence corpus.
- `build_evidence_fixture.py` for building non-oracle verifier fixtures from statement-bearing score dumps and local JSON/JSONL/text corpora.
- `backfill_truthfulqa_statements.py` for adding statement metadata to older TruthfulQA score dumps without rerunning models.
- Qwen l80 / SmolLM2 l80 oracle verifier-ensemble upper-bound report: label-derived perfect evidence drives verified false alarm to 0.000 and detection to 1.000 at alpha 0.100.
- Verifier ensemble reports include label-conditioned `verification_quality`, so evidence fixture quality can be measured separately from downstream control-policy metrics.
- Qwen l80 / SmolLM2 l80 local-corpus verifier baseline: conservative correct-answer retrieval drives verified false alarm to 0.008 at alpha 0.100, with true-supported rate 0.908 and false-supported rate 0.042.
- `QuestionAnswerVerifier` structured QA/domain-state adapter and Qwen l80 / SmolLM2 l80 structured QA baseline: exact question-answer facts drive verified false alarm to 0.000 and detection to 1.000 at alpha 0.100 on covered TruthfulQA questions.
- `CalculatorVerifier` deterministic tool adapter for structured arithmetic claims and simple symbolic equations, using a restricted local arithmetic evaluator with no new mandatory dependencies.
- `StructuredStateVerifier` / `StateCheck` structured state and business-rule adapter for database, policy, and domain-state checks. It can be routed through `state_check` claim metadata or context and returns supported, refuted, insufficient-evidence, or error results with explicit decision rules.
- `SQLiteStateSource` / `SQLiteStateQuery` load read-only SQLite query results into nested verifier state, giving the structured-state path a real database integration without new mandatory dependencies.
- `sqlite_state_control_demo.py` seeds an order/inventory/account SQLite fixture and emits a final `ProductTrace` where database state drives a dry-run abstain action despite low internal diagnostics.
- `CompositeVerifier`, `RoutedVerifier`, and `calibrated_control_demo.py --enable-calculator` show a tool-first product trace path: claim metadata, context, or text patterns can route calculator-supported/refuted claims before lexical verifier fallback.
- `eval_verifier_ensemble.py` now supports `--state-source` plus fixture-level `state_check` routes, and reports `route_summary` for structured QA, structured state, lexical groundedness, and retrieval-backed groundedness.
- `build_domain_state_fixture.py` generates deterministic order-fulfillment score/claim/state fixtures so structured-state verifier behavior can be benchmarked without relying on label-derived TruthfulQA oracle evidence.
- `CachedVerifier`, `CachedRetriever`, and `CachedStateSource` provide request-scoped caching plus hit/miss stats for repeated verifier, retrieval, and state-source calls; `eval_verifier_ensemble.py` includes these `cache_stats` in each run report.
- `eval_truthfulqa.py --profile` now emits a structured performance summary with bottleneck phase, top phases, grouped time shares, and warmup/eval throughput fields for reproducible optimization comparisons.
- `eval_truthfulqa.py --auto-batch-size` can halve and retry warmup/forced-answer batch size after retriable memory errors, recording requested/effective batch size and reduction events in JSON/profile output.
- `eval_truthfulqa.py --statement-encoding-cache` persists tokenizer outputs and answer-span lengths as validated JSON so repeated benchmark/cache rebuilds avoid redundant tokenizer setup without changing scoring semantics.
- `compare_profiles.py` compares profile JSON payloads across baseline/cache/cache-only runs, reporting speedup, phase deltas, grouped time deltas, and throughput ratios without loading models.

### Next Verification Adapter Work

- Connect `StructuredStateVerifier` to additional live database, business-rule, tool-output, and domain-state sources behind optional adapters; the current reproducible path uses local JSON state sources and stdlib SQLite.
- Extend claim extraction or upstream tool calls to provide structured `expression` / `expected` metadata for calculator-verifiable claims beyond simple symbolic equations.
- Add route policies for QA/database/world-model adapters so product traces explain why each tool was selected or skipped.
- Track route-level metrics in future adapter benchmarks so new tools are judged by route hit rate, false support, false refutation, and downstream conformal control impact.
- Replace label-derived oracle evidence with real retrieval/database/calculator evidence and rerun verifier/retrieval ensemble reports on Qwen/SmolLM2.
- Use local corpus fixtures from `build_evidence_fixture.py` as the reproducible baseline before adding networked retrieval extras.
- Add optional retrieval/database/calculator verifier adapters behind extras, keeping core dependencies unchanged.
- Add concrete domain/world-model adapters beyond the in-memory test double.
- Add semantic-entropy sampling probes, RAG groundedness adapters, and optional SAE/ReFT integrations behind extras.
- Add domain examples where facts depend on state transitions, physical constraints, or business rules.
- Compare real verifier/retrieval ensembles against the current `truth_proj` baseline under the same conformal false-alarm budget.

### Later Control-Plane Hardening

- Persist full trace/report/action-result artifacts from benchmark and demo commands into local registries by default when requested.
- Add policy graph composition only after current dataclass-based policy config becomes insufficient.
- Add steering-lambda sweep utilities after monitor-only diagnostics are stable on larger replicated runs.
- Extend registry persistence from metadata records to saved subspace/manifold tensors.

## Product Tagline

EigenTruth is a factuality control plane for LLM systems: it observes hidden-state drift, calibrates risk, routes uncertain generations through tools, and leaves creativity where creativity belongs.

EigenTruth 是 LLM 系统的事实性控制平面：观测表征漂移，校准风险，把不确定生成路由到工具验证，同时保留模型该有的创造力。

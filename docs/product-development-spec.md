# EigenTruth Product And Development Charter

EigenTruth is a representation-observability and factuality-control toolkit for LLM systems. It is not a standalone truth oracle, a safety boundary, or a replacement for external evidence. The product direction is to combine internal model diagnostics, calibrated risk scoring, tool-based verification, and optional world-model correction while preserving the language model's creative strengths.

EigenTruth 的产品定位是：面向 LLM 系统的表征观测与事实性控制工具集。它不是单一的“反幻觉模型”，也不是生产安全边界。产品方向是把模型内部诊断、风险校准、工具验证和可选世界模型校正组合起来，在保留 LLM 创造性的同时降低事实错误。

## Current State

Implemented today:

- `TruthManifold`: online hidden-state mean/covariance, ridge-regularized precision, save/load support.
- `TruthProbe`: PyTorch forward-hook monitor for selected Transformer layers, Mahalanobis-style drift, HSE tracking, and optional activation steering.
- `EigenTruthWrapper`: warmup, generation passthrough, diagnostics, and probe lifecycle management for Hugging Face-style PyTorch models.
- `eigentruth.eval`: conformal p-values/thresholds plus CPU-testable metrics such as AUROC and Euclidean dispersion.
- `eigentruth.calibration`: JSON-serializable calibration artifacts, split-conformal calibrators, and layer/score sweep reports.
- `eigentruth.core.TruthSubspace`: low-rank factual subspace scoring, benchmark residual signal, and optional true-minus-false projection.
- `eigentruth.control.RiskController`: maps calibrated diagnostic thresholds to product actions.
- `eigentruth.control.ProductTrace`: JSON-ready traces for diagnostics, claims, verification, decisions, actions, and metadata.
- `eigentruth.verify`: dependency-free claim extraction, in-memory verifier tools, and lexical groundedness checks for first-pass claim workflows.
- `eigentruth.adapters.InMemoryWorldModelAdapter`: deterministic world-model adapter for tests and domain-rule prototypes.
- `eigentruth.registry.ArtifactRegistry`: local JSON registry for calibration reports, calibration artifacts, and saved concept metadata.
- Benchmark scripts for TruthfulQA-style evaluation, TruthSubspace residual scoring, layer/score sweeps, and conformal calibration.
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
- `eigentruth.eval`: metrics, conformal calibration, benchmark helpers.
- `examples/`: qualitative demos only; no benchmark claims.
- `benchmarks/`: reproducible evaluations with structured output.

Future modules should follow the same separation:

- `eigentruth.calibration`: calibration artifacts and parameter sweeps.
- `eigentruth.control`: risk controller, correction policy, abstention policy.
- `eigentruth.verify`: claim extraction and verifier interfaces.
- `eigentruth.registry`: saved concepts/manifolds/subspaces and metadata.
- `eigentruth.adapters`: optional integrations, including HF, retrieval, databases, and world models.

The first structural scaffold for these future modules is intentionally interface-only: plain dataclasses, enums, and protocols with no heavyweight dependencies. Concrete implementations should land behind these contracts as evidence and tests justify them.

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

### 0.2: Calibrated Observability Toolkit

- Validate the new `TruthSubspace` benchmark residual signal on larger instruction-tuned models.
- Extend registry persistence from metadata records to saved subspace/manifold tensors.
- Add steering-lambda sweep utilities after monitor-only diagnostics are stable.
- Expand benchmark outputs for selective accuracy and confidence intervals.

### 0.3: Risk Control Plane

- Extend `RiskController` beyond threshold counting into policy composition across diagnostics and verification.
- Add concrete correction policies: accept, retrieve, rewrite, steer, abstain, clarify.
- Upgrade claim extraction and verifier implementations beyond lexical groundedness behind the existing protocols.
- Connect `ProductTrace` to application-facing diagnostics and evaluation artifacts.

### 0.4: Verification And World-Model Adapters

- Add retrieval/database/calculator verifier adapters.
- Add semantic-entropy sampling probes, RAG groundedness adapters, and optional SAE/ReFT integrations behind extras.
- Replace the in-memory world-model mock with optional concrete domain/world-model adapters.
- Add domain examples where facts depend on state transitions, physical constraints, or business rules.
- Benchmark hybrid internal-diagnostics + external-verification pipelines.

## Product Tagline

EigenTruth is a factuality control plane for LLM systems: it observes hidden-state drift, calibrates risk, routes uncertain generations through tools, and leaves creativity where creativity belongs.

EigenTruth 是 LLM 系统的事实性控制平面：观测表征漂移，校准风险，把不确定生成路由到工具验证，同时保留模型该有的创造力。

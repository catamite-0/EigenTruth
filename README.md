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
- compile risk decisions into structured action requests and dry-run execution results
- optionally apply experimental activation steering when a configured threshold is exceeded

EigenTruth 通过 PyTorch hook 包装 decoder-only 语言模型。它可以：

- 使用事实性 warmup 样本构建 `TruthManifold`
- 跟踪隐藏状态相对于 warmup 流形的马氏距离风格指标
- 将隐藏状态投影到庞加莱球并计算双曲语义熵（HSE）
- 可选地使用事实与错误样本构建对比方向
- 拟合低秩 `TruthSubspace`，并计算相对事实子空间的残差距离
- 从 benchmark 分数 dump 校准诊断阈值，并与 claim 验证结果组合成风险决策
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
  --dump-scores benchmarks/scores.json
python benchmarks/eval_conformal.py --scores benchmarks/scores.json \
  --save-sweep-report artifacts/gpt2-sweep-report.json \
  --save-best-calibration artifacts/gpt2-best-calibration.json
```

Use `--batch-size` and, when sampling INSIDE continuations, `--inside-batch-size` to trade memory for benchmark throughput. Add `--auto-batch-size` on long warmup/forced-answer runs to halve and retry the batch size after retriable memory errors, with the final setting recorded in JSON/profile output. This produces a layer/score sweep report plus a reusable `CalibrationArtifact` for the best calibrated diagnostic. The artifact can drive `RiskController` decisions, and `RiskController.decide(..., verification_results=...)` can compose calibrated diagnostics with claim-level verification in `ProductTrace` records. The dependency-free `run_verification_loop(...)` helper can also execute retrieve actions, feed retrieved evidence back into verification, and emit a final decision trace.

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
| `internal_eigenscore` / `spectral_effective_rank` | Computes INSIDE/EigenScore-style spectral diversity from hidden-state embeddings; benchmarks can optionally sample multiple continuations for `inside_eigenscore`. |
| `TruthProbe` | Captures selected-layer hidden states and optionally applies steering. |
| `EigenTruthWrapper` | Provides warmup, generation passthrough, diagnostics, and probe lifecycle management. |
| `TruthSubspace` | Fits low-rank factual subspaces and residual-distance diagnostics; fitting requires at least two factual states. |
| `directional_conformal_threshold` / `directional_trigger_rate` | Apply split-conformal thresholds consistently for `higher` and `lower` anomalous score directions. |
| `LayerScoreSweepCalibrator` | Builds layer/score sweep reports and reusable calibration artifacts from score dumps. |
| `RiskController` / `ProductTrace` | Converts calibrated diagnostics plus optional verification results into structured routing decisions and JSON-ready traces; invalid diagnostic values route to `clarify/unknown`, and route summaries expose selected/matched/skipped verifier tools. |
| `DefaultCorrectionPolicy` / `ActionRequest` | Compiles control decisions into executable JSON-ready action payloads for product integrations, including generic `execute_tool` requests. |
| `ActionExecutorRegistry` / `DryRunActionExecutor` / `ActionResult` | Routes action requests to registered executors, with side-effect-free dry-run fallback for local traces. |
| `ActionExecutionPolicy` / `PolicyGuardedActionExecutor` | Adds dependency-free request validation, idempotency replay, and audit metadata for side-effecting executors, including request ids, idempotency keys, and timeout bounds. |
| `InMemoryActionExecutionLedger` / `JsonActionExecutionLedger` | Stores successful idempotent action results so repeated product requests can replay outputs without repeating side effects. |
| `run_verification_loop` / `EvidenceBundle` | Runs verify -> decide -> execute -> reverify loops and converts retrieval action results into verifier-ready evidence context. |
| `RetrievalActionExecutor` / `InMemoryRetriever` | Provides a dependency-free retrieval executor shell for unsupported-claim evidence gathering. |
| `CalculatorVerifier` | Provides a dependency-free deterministic calculator verifier for structured arithmetic claims and simple symbolic equations. |
| `QuestionAnswerVerifier` | Provides a dependency-free structured QA/domain-state verifier adapter for exact question and candidate-answer facts. |
| `StructuredStateVerifier` / `StateCheck` | Provides a dependency-free structured state and business-rule verifier for database, policy, and domain-state checks. |
| `SQLiteStateSource` / `SQLiteStateQuery` | Loads read-only SQLite query results into structured verifier state without adding non-stdlib dependencies. |
| `ToolOutputStateSource` / `ToolOutputMapping` | Maps local tool or action execution outputs into structured verifier state for post-tool checks. |
| `StateTransitionVerifier` / `StateTransitionCheck` | Uses a world-model adapter to predict next state after an action, then checks structured postconditions. |
| `CachedVerifier` / `CachedRetriever` / `CachedStateSource` | Adds request-scoped in-memory caching and hit/miss stats for repeated verifier, retrieval, and state-source calls. |
| `CompositeVerifier` / `RoutedVerifier` | Compose deterministic tools with lexical, retrieval, database, or world-model verifiers; routing can use claim metadata, context, or text patterns and records match reasons. |
| `GroundednessVerifier` / `ClaimExtractor` | Extracts claim metadata and checks claims against lexical evidence snippets and explicit refutations without extra dependencies. |
| `sqlite_state_control_demo.py` | Demonstrates SQLite-backed structured state checks feeding a final `ProductTrace` and dry-run action. |
| `state_transition_control_demo.py` | Demonstrates world-model next-state prediction plus structured postcondition checks feeding a final `ProductTrace`. |
| `production_tool_loop_demo.py` | Demonstrates a local production-like loop: SQLite pre-check, guarded side-effecting local `execute_tool`, optional JSON idempotency ledger, tool-output state mapping, post-tool verification, action audit metadata, and route summary in one trace. |
| `eval_verifier_ensemble.py` | Benchmarks calibrated internal diagnostics combined with retrieval/verifier suppression and refutation policies. |
| `compare_verifier_routes.py` | Aggregates saved verifier-ensemble reports into route leaderboards and by-route control-impact metrics. |
| `build_domain_state_fixture.py` | Builds deterministic order-fulfillment score/claim/state fixtures for structured-state verifier benchmarks. |
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
| `internal_eigenscore` / `spectral_effective_rank` | 基于隐藏态嵌入计算 INSIDE/EigenScore 风格的谱分散度；benchmark 可选多采样续写生成 `inside_eigenscore`。 |
| `TruthProbe` | 捕获指定层的隐藏状态，并可选地应用激活引导。 |
| `EigenTruthWrapper` | 提供 warmup、生成透传、诊断信息和探针生命周期管理。 |
| `TruthSubspace` | 拟合低秩事实子空间，并提供残差距离诊断；拟合至少需要两条事实状态。 |
| `directional_conformal_threshold` / `directional_trigger_rate` | 对 `higher` 与 `lower` 异常方向使用一致的 split-conformal 阈值与触发率。 |
| `LayerScoreSweepCalibrator` | 从分数 dump 构建层/分数 sweep report 与可复用校准 artifact。 |
| `RiskController` / `ProductTrace` | 将校准诊断和可选验证结果转为结构化路由决策与 JSON trace；非法诊断值会路由到 `clarify/unknown`，route summary 会暴露选中、匹配和跳过的 verifier 工具。 |
| `DefaultCorrectionPolicy` / `ActionRequest` | 将控制决策编译为面向产品集成的 JSON action payload，包括通用 `execute_tool` 请求。 |
| `ActionExecutorRegistry` / `DryRunActionExecutor` / `ActionResult` | 按 action 路由 executor，并用无副作用 dry-run 作为本地 trace fallback。 |
| `ActionExecutionPolicy` / `PolicyGuardedActionExecutor` | 为有副作用 executor 增加无依赖请求校验、idempotency replay 和审计元数据，包括 request id、idempotency key 与 timeout 上限。 |
| `InMemoryActionExecutionLedger` / `JsonActionExecutionLedger` | 保存成功的幂等 action 结果，让重复产品请求可重放输出而不重复执行副作用。 |
| `run_verification_loop` / `EvidenceBundle` | 执行 verify -> decide -> execute -> reverify 闭环，并把 retrieval action result 转成 verifier 可消费的 evidence context。 |
| `RetrievalActionExecutor` / `InMemoryRetriever` | 为 unsupported claim 的取证流程提供无依赖 retrieval executor shell。 |
| `CalculatorVerifier` | 提供无依赖确定性计算器 verifier，用于结构化算术 claim 和简单符号等式。 |
| `QuestionAnswerVerifier` | 提供无依赖结构化 QA/领域状态 verifier adapter，用于精确问题与候选答案事实。 |
| `StructuredStateVerifier` / `StateCheck` | 提供无依赖结构化状态与业务规则 verifier，用于数据库、策略和领域状态校验。 |
| `SQLiteStateSource` / `SQLiteStateQuery` | 将只读 SQLite 查询结果加载为 verifier 可消费的结构化状态，不增加非标准库依赖。 |
| `ToolOutputStateSource` / `ToolOutputMapping` | 将本地工具或 action 执行输出映射成结构化 verifier state，用于工具调用后的校验。 |
| `StateTransitionVerifier` / `StateTransitionCheck` | 通过 world-model adapter 预测 action 后的下一状态，再校验结构化 postcondition。 |
| `CachedVerifier` / `CachedRetriever` / `CachedStateSource` | 为重复 verifier、retrieval 和 state-source 调用提供 request-scoped 内存缓存与 hit/miss 统计。 |
| `CompositeVerifier` / `RoutedVerifier` | 组合确定性工具与词面、检索、数据库或 world-model verifier；路由可依据 claim metadata、context 或文本模式，并记录匹配原因。 |
| `GroundednessVerifier` / `ClaimExtractor` | 抽取 claim metadata，并用词面证据片段和显式反证检查 claim，不增加核心依赖。 |
| `sqlite_state_control_demo.py` | 演示 SQLite 结构化状态校验如何进入最终 `ProductTrace` 和 dry-run action。 |
| `state_transition_control_demo.py` | 演示 world-model 下一状态预测和结构化 postcondition 校验如何进入最终 `ProductTrace`。 |
| `production_tool_loop_demo.py` | 演示本地 production-like 闭环：SQLite 前置校验、受 guard 约束的有副作用本地 `execute_tool`、可选 JSON idempotency ledger、工具输出状态映射、工具后校验、action audit metadata 和 trace route summary。 |
| `eval_verifier_ensemble.py` | 评估校准内部诊断与 retrieval/verifier 抑制误报、补充反证检出的组合策略。 |
| `compare_verifier_routes.py` | 将已保存 verifier-ensemble report 聚合为 route 排行榜与分 route 控制收益指标。 |
| `build_domain_state_fixture.py` | 构建确定性的订单履约 score/claim/state fixture，用于结构化状态 verifier benchmark。 |
| `build_transition_fixture.py` | 构建确定性的订单预留 state-transition fixture，用于 world-model/postcondition verifier benchmark。 |
| `build_truthfulqa_corpus.py` | 构建本地 TruthfulQA correct-answer corpus，用于可复现的非 oracle retrieval baseline。 |
| `build_evidence_fixture.py` | 从带 statement 的 score dump 和本地 JSON/JSONL/text 文档库构建非 oracle claim/evidence fixture。 |
| `backfill_truthfulqa_statements.py` | 为旧版 TruthfulQA score dump 重建确定性 statement metadata，并可输出标签派生 oracle evidence 用于 verifier 上界测试。 |

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
make check
make perf-check     # deterministic profile-gate smoke; no model load
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

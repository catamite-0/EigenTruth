<div align="center">

# EigenTruth

**Research-preview PyTorch toolkit for LLM representation monitoring, geometric drift diagnostics, and experimental activation steering**

**面向大模型表征监测、几何漂移诊断与实验性激活引导的 PyTorch 研究预览工具库**

[![Status: Research Preview](https://img.shields.io/badge/status-alpha%20research%20preview-yellow.svg)]()
[![CI](https://github.com/catamitez0-maker/EigenTruth/actions/workflows/ci.yml/badge.svg)](https://github.com/catamitez0-maker/EigenTruth/actions/workflows/ci.yml)
[![Framework: PyTorch](https://img.shields.io/badge/framework-PyTorch%202.0%2B-ee4c2c.svg)](https://pytorch.org)
[![Python: 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB.svg)](https://python.org)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

[Quick start](#quick-start) | [0.2.0 Notes](docs/release-0.2.0.md) | [Architecture](#architecture) | [Product Charter](docs/product-development-spec.md) | [Methodology](docs/methodology.md) | [Examples](examples/README.md) | [Roadmap](ROADMAP.md) | [Contributing](CONTRIBUTING.md) | [Security](SECURITY.md)

</div>

## Research Preview

EigenTruth is an alpha-stage research toolkit. It is intended for controlled experiments, diagnostics, and reproducible exploration. It is not production-ready, does not prove that an output is true, and must not be treated as a safety boundary for deployed systems.

EigenTruth 是一个处于 alpha 阶段的研究预览工具库，适用于受控实验、诊断和可复现探索。它尚未达到生产可用状态，不能证明模型输出为真，也不能作为已部署系统的安全边界。

The current package baseline is the `0.2.0` research release. See
[`docs/release-0.2.0.md`](docs/release-0.2.0.md) for the supported scope,
evidence summary, negative results, and remaining limitations.

The current implementation explores a research hypothesis: hallucination-related generation behavior may sometimes be accompanied by measurable geometric drift in hidden-state representations. The signals exposed by this project are experimental diagnostics, not calibrated factuality scores.

当前实现探索一个研究假设：与幻觉相关的生成行为有时可能伴随隐藏状态表征中可测量的几何漂移。本项目提供的信号属于实验性诊断指标，不是经过校准的事实性评分。

## What EigenTruth Does

EigenTruth wraps a decoder-only language model with PyTorch hooks. It can:

- build a `TruthManifold` / `RepresentationManifold` from factual warmup examples
- track Mahalanobis-style distance from that warmup manifold
- optionally project hidden states into a Poincare ball and compute Hyperbolic Semantic Entropy (HSE) for ablations
- optionally build a contrastive direction from factual and false examples
- fit a low-rank `TruthSubspace` and score residual distance from factual states
- train and save a lightweight pre-generation attention probe from hidden states and soft error-rate targets
- emit a single-decode `first_token_entropy` uncertainty baseline from top-k logits
- save versioned `ConceptArtifact` files and attach multiple concept probes at once
- calibrate diagnostic thresholds from benchmark score dumps and combine them with claim verification
- select a cheap pre-generation runtime profile from prompt and metadata risk markers, with an optional soft risk probability recorded before generation
- compile risk decisions into structured action requests and dry-run execution results
- optionally apply experimental activation steering when a configured threshold is exceeded

EigenTruth 通过 PyTorch hook 包装 decoder-only 语言模型。它可以：

- 使用事实性 warmup 样本构建 `TruthManifold` / `RepresentationManifold`
- 跟踪隐藏状态相对于 warmup 流形的马氏距离风格指标
- 可选地将隐藏状态投影到庞加莱球并计算双曲语义熵（HSE）用于消融实验
- 可选地使用事实与错误样本构建对比方向
- 拟合低秩 `TruthSubspace`，并计算相对事实子空间的残差距离
- 用 hidden states 和 soft error-rate targets 训练并保存轻量生成前 attention probe
- 从 top-k logits 输出单次 decode 的 `first_token_entropy` 不确定性基线
- 保存版本化 `ConceptArtifact`，并同时挂载多个 concept probe
- 从 benchmark 分数 dump 校准诊断阈值，并与 claim 验证结果组合成风险决策
- 基于 prompt 与 metadata 风险标记，在生成前选择低成本 runtime profile，并可记录生成前 soft risk probability
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
  --save-abstention-report artifacts/gpt2-abstention-report.json \
  --abstention-signals maha_last,truth_proj,subspace_resid,first_token_entropy,inside_eigenscore \
  --save-abstention-comparison artifacts/gpt2-abstention-comparison.json \
  --save-abstention-release-gate artifacts/gpt2-abstention-release-gate.json \
  --min-abstention-conditional-correctness-lower-bound 0.8 \
  --max-abstention-rate 0.5 \
  --save-sweep-report artifacts/gpt2-sweep-report.json \
  --save-best-calibration artifacts/gpt2-best-calibration.json \
  --artifact-manifest artifacts/gpt2-conformal-manifest.json
```

`--artifact-manifest` fingerprints the score dump, conformal report, abstention
reports, abstention release-gate verdict, sweep report, and saved calibration
artifacts so local registry/release workflows can verify the calibration chain
without rerunning the model.
Use `--dump-scores-format jsonl` for larger sweeps; it writes a compact manifest
plus records sidecar that downstream calibration and verifier tools can stream by
selected columns. New JSONL manifests also store label counts, so metadata
summaries can avoid scanning the records sidecar when only class counts are
needed. When a selected-column JSONL loader has already scanned the records
sidecar in the same run, it primes the run-local records fingerprint cache so
later provenance metadata can reuse the full SHA-256 without a second records
pass.
For random-matrix spectrum diagnostics, add `--include-layer-spectra` to
`eval_truthfulqa.py --json ...`; the report will include compact per-layer
Marchenko-Pastur bulk edges, spike counts, effective rank, and top covariance
eigenvalues. This is off by default because full eigendecomposition can be
expensive on large hidden dimensions.
Use `benchmarks/compare_spectrum_layers.py` and
`benchmarks/compare_layer_band_selectors.py` to audit whether those cheap
spectrum and intrinsic-dimension priors can reduce the layer sweep to a
candidate band. Current l80 evidence supports `max_top_eigenvalue_to_mp_upper`
with radius 1 as a cost-reducing band prior, not as a replacement for calibrated
layer/score sweep inside the band.

For ACSE-style experiments, `eval_conformal.py` can also build an adaptive
conformal report and artifact by inflating the selected diagnostic with primary
score or score-dump extra fields:

```bash
python benchmarks/eval_conformal.py --scores benchmarks/scores.manifest.json \
  --signal maha_last \
  --adaptive-feature inside_semantic_energy \
  --adaptive-feature-weight inside_semantic_energy=0.5 \
  --save-adaptive-calibration artifacts/gpt2-maha-adaptive.json
```

`eval_truthfulqa.py` also emits `first_token_entropy` by default, using top-k
normalized logits entropy from the first available answer-token prediction as a
single-decode uncertainty baseline. Use `--first-token-top-k` to change the
retained logits before comparing it with geometry, INSIDE/selfcheck, or verifier
signals.

For calibrated participation-control experiments, add
`--save-abstention-report` or `--include-abstention-report`. The abstention report
calibrates the selected uncertainty score on correct responses and reports
participation, abstention, selective accuracy, and conservative conditional
correctness lower bounds without changing the base conformal gate verdict.
Use `--save-abstention-comparison` with `--abstention-signals` to rank several
candidate uncertainty signals by conservative conditional-correctness lower bound,
selective accuracy, participation, or retention before wiring a signal into the
control plane. Add `--save-abstention-release-gate` or
`--include-abstention-release-gate` to turn the selected report or comparison
recommendation into a fail-closed promotion verdict with minimum conservative
conditional-correctness and maximum abstention-rate requirements.

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
`benchmark_manifest:smollm2-l20-inside-trigger-budget-derived-strict-structured-retrieval-audit-staged-qa-release-candidate:1.6`
selects the top-40% quality-balanced budget from a single largest-budget run,
uses 218 generated samples with sample-count ratio 0.472 and
`inside_generation` ratio 0.503 against the full-sample fixed reference,
requires `performance_baseline:smollm2-l20-performance-baseline:0.9` to match
the final runtime, requires promoted `structured_state`, `state_transition`, and
`retrieval_groundedness` plus `retrieval_structured_qa` adapter-family routes,
and requires
`benchmark_manifest:smollm2-l80-retrieval-structured-qa-route:0.6` as a separate
retrieval structured-QA audit baseline with non-oracle evidence provenance and
answer-echo retrieval stress control. That audit route promotes with selected
238, decision accuracy 0.992, false-supported rate 0.000, false-refuted rate
1.000, runtime about 0.85s, and 410 retrieval hits under a 450-hit budget. The
answer-echo stress run intentionally self-supports false claims at rate 0.980
and false-refutes them at 0.000, so answer-derived retrieval evidence is rejected
as grounding. The
selected product route remains strict low-latency `structured_qa` with
`max_retrieval_use_rate=0.0` and `max_mean_attempted_route_count=1.1`.
Version 1.6 also requires promoted selector replay over 12 redacted product
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
records read, shard loads, shard cache hits, cross-shard reads, and shard
manifest scans in JSON output;
triplet, matrix, worker-sweep, and performance-baseline runners pass the setting
through for cached/cache-only runs. Use
`run_cache_profile_matrix.py --eval-reps-shard-read-cache-sizes 1,2,4` to
promote this from a heuristic cache-tuning suggestion into a same-machine sweep;
the recommended runtime then records the selected cell's read-cache size and
does not emit contradictory read-cache-size heuristic advice after a sweep has
already compared the candidate capacities.
The current small CPU SmolLM2 l8 evidence registers
`performance_baseline:smollm2-l8-read-cache-worker-sweep-score-fusion-performance-baseline:0.2`;
it keeps the read-cache sweep winner at size 2, selects `max_workers=2`,
lowers matrix wall-clock from 184.467s to 141.385s on this machine, preserves
`truth_proj` AUROC 0.830, and carries a conformal-gated
`score_fusion_mean_rank` auxiliary signal from the l80 score-ensemble report.
The follow-up selected-fusion handoff registers
`performance_baseline:smollm2-l8-read-cache-worker-sweep-selected-fusion-performance-baseline:0.3`;
it reuses the same runtime cell and worker/cache evidence, explicitly selects
the SmolLM2 selected-fusion run, records `selected_fusion_mean_rank` as promoted
auxiliary evidence (`AUROC=0.692`, false alarm `0.029`, detection `0.224`), and
keeps `truth_proj` as the best quality signal.
The same evidence now promotes
`benchmark_manifest:smollm2-l8-read-cache-worker-sweep-score-fusion-staged-qa-release-candidate:0.2`
through the staged structured-QA release gate; its release manifest and registry
metadata expose the score-fusion status, signal, AUROC, conformal gate result,
and source score-ensemble report for audit.
The selected-fusion handoff now has the parallel staged structured-QA release
gate
`benchmark_manifest:smollm2-l8-read-cache-worker-sweep-selected-fusion-staged-qa-release-candidate:0.3`,
which verifies the selected-fusion performance manifest and records selected
fusion status/run/signal/AUROC/artifact metadata in the release registry.
Its compact product handoff is
`product_promotion_contract:smollm2-l8-selected-fusion-product-promotion-contract:0.3`
at
`artifacts/smollm2_l8_selected_fusion_product_promotion_contract_v0_3/product-promotion-contract.json`,
including the selected-fusion status/run/signal/AUROC/false-alarm/detection and
artifact path in contract metadata.
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
triggers verification. When claim conclusions depend on earlier premises, pass
`claim_dependencies=` or `enforce_claim_coherence=True` so missing or unsupported
parent claims downgrade supported child claims to insufficient evidence and get
recorded in the trace. Running `examples/calibrated_control_demo.py` with
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
        +--> optional Poincare-ball projection --> HSE ablation diagnostic
        +--> optional threshold-triggered steering --> model generation
```

The high-level workflow is:

1. **Warm up**: collect final-token hidden states from factual texts and optionally false texts.
2. **Build diagnostics**: incrementally construct a ridge-regularized precision matrix and an optional contrastive direction.
3. **Attach a hook**: register a `forward_hook` on a selected Transformer layer.
4. **Monitor**: calculate representation-distance diagnostics during generation; enable HSE explicitly for ablations.
5. **Experiment with steering**: optionally inject a normalized steering vector after a configured threshold is exceeded.

工作流程：

1. **Warmup**：从事实文本和可选的错误文本中收集最后一个 token 的隐藏状态。
2. **构建诊断**：增量构建 ridge 正则化的精度矩阵和可选的对比方向。
3. **挂载 Hook**：在选定的 Transformer 层注册 `forward_hook`。
4. **监测**：在生成期间计算表征距离诊断；HSE 需要显式开启，主要用于消融实验。
5. **实验性引导**：可选地在超过配置阈值后注入归一化引导向量。

See [`docs/methodology.md`](docs/methodology.md) for the mathematical framing, calibration guidance, and limitations.

## Core Components

| Component | Purpose |
|---|---|
| `TruthManifold` / `RepresentationManifold` / `CovarianceSpectrum` / `covariance_spectrum` | Maintains a Welford online mean and covariance, exposed as a ridge-regularized, sample-count-normalized precision matrix; supports `covariance_mode="full"`, `"diag"`, experimental `"low_rank"`, and OAS-style `"shrinkage"` so local benchmarks can trade exact covariance scoring for lower memory/compute cost or small-sample conditioning; `TruthManifold.spectrum()` reports Marchenko-Pastur bulk edges, spike count, effective rank, participation ratio, stable rank, and condition number as dependency-free representation-spectrum diagnostics. |
| `mahalanobis_distance` | Measures relative deviation from the warmup manifold. |
| `gaussian_wasserstein_distance` / `manifold_distance` / `manifold_wasserstein_distance` | Computes dependency-free closed-form Gaussian 2-Wasserstein/Bures distance for comparing representation manifolds across layers, checkpoints, or runs. |
| `twonn_intrinsic_dimension` / `intrinsic_dimension_profile` | Estimates dependency-free TwoNN intrinsic dimension from hidden-state samples, producing cheap layer-profile evidence for layer selection and representation-collapse experiments. |
| `RepresentationTelemetryRecorder` / `RepTelemetryCallback` / `representation_telemetry_snapshot` | Records training-side per-layer representation telemetry without mandatory Trainer dependencies: mean norm, variance trace, spectrum rank diagnostics, and Gaussian 2-Wasserstein/Bures distance to an initialization baseline; the optional callback exposes HF Trainer-compatible hook names while staying dependency-free. |
| `TrajectoryMonitor` / `trajectory_convergence_metrics` | Computes generation-trajectory convergence diagnostics from per-token hidden states, including step-distance decay, Koopman-style rate, path efficiency, and a convergence score for quality/confidence correlation checks. |
| `AttentionSoftTargetProbeArtifact` / `soft_error_rate_targets` | Trains a torch-only pre-generation attention probe over token-level hidden states using empirical sampled-answer error rates as soft targets; artifacts expose risk probabilities, attention weights, JSON-safe metadata, and torch save/load. |
| `poincare_map` | Projects representations into a bounded hyperbolic space for optional HSE ablations. |
| `hyperbolic_semantic_entropy` | Measures dispersion over a sliding window of projected states; retained as an opt-in ablation signal, not the default runtime path. |
| `internal_eigenscore` / `spectral_effective_rank` / `cluster_assignment_entropy` / `lexical_semantic_entropy` / `embedding_semantic_entropy` / `semantic_energy_score` / `lexical_semantic_energy` | Computes INSIDE/EigenScore-style spectral diversity, dependency-free semantic-entropy proxies, and confidence-weighted semantic-energy disagreement from sampled hidden-state/text clusters; benchmarks can optionally sample multiple continuations for `inside_eigenscore`, `inside_semantic_entropy`, `inside_embedding_entropy`, and `inside_semantic_energy`. |
| `topk_normalized_entropy` / `first_token_confidence` / `first_token_entropy` score dumps | Adds a low-cost single-decode uncertainty baseline: `eval_truthfulqa.py` records top-k normalized entropy at the first available answer-token prediction as `first_token_entropy`, with `--first-token-top-k` controlling the retained logits. Higher entropy is treated as more anomalous and can be calibrated or fused like other score-dump signals. |
| `TruthProbe` / `RepresentationProbe` | Captures selected-layer hidden states and optionally applies steering; HSE tracking is opt-in via `track_hse=True`. |
| `ConceptArtifact` / `MultiConceptMonitor` | Saves versioned concept manifolds with layer metadata and attaches several concept probes to one model, returning per-concept diagnostics without changing the single-probe wrapper path. |
| `EigenTruthWrapper` / `RepresentationMonitor` | Provides warmup, generation passthrough, diagnostics, and probe lifecycle management. |
| `TruthSubspace` | Fits low-rank factual subspaces and residual-distance diagnostics; fitting requires at least two factual states. |
| `directional_conformal_threshold` / `directional_conformal_thresholds` / `directional_trigger_rate` | Apply split-conformal thresholds consistently for `higher` and `lower` anomalous score directions, including one-sort multi-alpha threshold calculation for benchmark reports. |
| `ConformalAbstentionReleaseGate` / `conformal_abstention_release_gate` | Converts a conformal abstention report, candidate, or comparison report into a fail-closed release verdict that requires a minimum conservative conditional-correctness lower bound and maximum empirical abstention rate before promoting a participation gate. |
| `AdaptiveScoreTransform` / `adaptive_anomaly_scores` / `AdaptiveConformalCalibrator` | Provides a dependency-free adaptive conformal scoring layer: native diagnostics are oriented into anomaly space, caller-supplied semantic/risk features add a calibrated inflation term, and the resulting score can be saved as a regular `CalibrationArtifact`. |
| `confidence_error_report` | Audits high-confidence error regimes by crossing a calibrated anomaly gate with a confidence proxy such as `nll_answer`, including high-confidence false misses and capture rates. |
| `directional_rank_anomaly_scores` / `combine_rank_anomaly_scores` / `RankScoreFusionArtifact` / `RankScoreFusionCalibrator` | Convert mixed-direction diagnostics into empirical anomaly ranks, fuse them with dependency-free rank methods, and save deployable conformal fusion artifacts for later control-plane experiments. |
| `geometry_calibrated_anomaly_scores` / `GeometryScoreFusionArtifact` / `GeometryScoreFusionCalibrator` | Builds a dependency-free geometry-by-uncertainty anomaly score by rank-calibrating representation geometry signals and confidence/semantic-energy proxies, then adding an explicit interaction term that can be saved as a conformal fusion artifact. |
| `SignalSelectionPolicy` / `select_signals_from_fusion_ablation_matrix` | Converts score-fusion ablation matrices into run-specific signal bundles, enabling optional signals such as trajectory only when they beat the best baseline within configured detection, AUROC, and false-alarm margins. |
| `ScoreDump` / `ScoreDumpIdentity` / `load_score_dump` / `load_score_dump_columns` / `load_score_dump_columns_with_extras` / `load_score_dump_statement_scores` / `load_score_dump_layer_scores` / `score_dump_file_metadata` / `score_dump_identity` / `score_dump_cache_summary` | Validate per-statement score dumps once, expose compact run summaries, and attach SHA-256 provenance plus a stable model/dataset/layer/score-schema/scoring-config identity to post-hoc calibration or ensemble reports without rerunning models; callers can share an optional run-local cache to avoid repeated file hashing, JSONL manifest parsing, and selected JSONL scans inside one run, then summarize cache hits/misses/writes for report observability. `ScoreDumpJsonlManifest` / `ScoreDumpRecord` / `iter_score_dump_jsonl_records` add an optional manifest-backed JSONL format for large dumps, selected-column loaders stream primary, adaptive-extra, statement-bearing, or layer/score views from that format, and cached selected views invalidate when the manifest or records file changes. Metadata fingerprints both the manifest and records file, includes a canonical identity cache key, uses manifest label counts for summary fast paths when available, and reuses records hashes primed by selected JSONL scans in the same run. `eval_truthfulqa.py --dump-scores-format jsonl` writes this format directly and stores per-record extras such as INSIDE sample counts in the records sidecar. |
| `LayerScoreSweepCalibrator` | Builds layer/score sweep reports and reusable calibration artifacts from score dumps, including direct `ScoreDump` inputs via `calibrate_from_score_dump()` and selected JSONL layer-score loading via `calibrate_from_file()`, with optional run-local cache reuse and bounded `max_workers` CPU parallelism for large post-hoc sweeps. |
| `ArtifactRegistry` / `ArtifactVerificationContext` / `build_artifact_manifest` / `fingerprint_path` / `fingerprint_cache_summary` / `load_json_object` / `json_cache_summary` / `load_fingerprint_cache` / `save_fingerprint_cache` / `load_json_cache` / `save_json_cache` / `verify_artifact_manifest` | Records and verifies local artifact metadata, including concept artifacts, with dependency-free file/directory SHA-256 provenance for reproducible benchmark chains; recursive verification shares a run-local fingerprint cache, `ArtifactVerificationContext` also shares path-signature JSON artifact caching across release/registry comparisons, cache summaries report fingerprint requests/hits/misses/hit rate, and CLIs can persist fingerprints plus selected JSON artifact caches so repeated manifest/release checks avoid duplicate content reads and JSON parsing when file signatures are unchanged; persisted JSON caches prune older signatures for the same path on save. |
| `RuntimeProfile` / `PreGenerationRiskPolicy` / `SoftPreGenerationRiskConfig` / `select_pre_generation_profile` / `RuntimeProfileSelectorPolicy` / `select_runtime_profile` | Defines shared `latency`, `balanced`, and `audit` defaults, plus dependency-free pre-generation prompt/metadata routing, optional soft risk probability estimates with feature/metadata contribution traces, post-diagnostic claim-metadata routing, and optional verification-plan cost routing for product control-plane staging. |
| `ProductPromotionContract` / `ProductRuntimeEvidenceBundle` | Converts promoted release-candidate reports into product runtime, verifier-route, covered-fact property scope, covariance quality-gate provenance, performance-bundle, performance score-dump cache, product-trace-replay workflow provenance including action-audit/action-execution gate reports and rates, selfcheck-signal-fusion sample-quality/fusion provenance, world-model signal release-gate provenance, feedback-policy workflow provenance including final-answer feedback rates, external-evidence baseline-comparison handoff provenance, pre-generation probe comparison/redline provenance, cross-corpus triple-extraction fixture-matrix evidence, selector-replay, runtime-drift evidence including promotion-contract coverage, triple-extraction fixture-matrix drift summaries, and trace-level triple-audit summaries, release-efficiency handoff evidence, budget-policy contracts, validated feedback-derived `ControlPolicyConfig`, and candidate control defaults consumed by the calibrated-control demo; `load_product_promotion_contract()` loads compact contracts, while `load_product_runtime_evidence_bundle()` lazily attaches optional promotion-contract manifest/registry provenance plus explicit selfcheck-signal-fusion, world-model-signal, external-evidence baseline-comparison, pre-generation probe comparison, and triple-extraction fixture-matrix manifest verification and registry metadata for runtime traces. |
| `RiskController` / `ParticipationGateConfig` / `ProductTrace` / `FinalAnswer` | Converts calibrated diagnostics plus optional verification results into structured routing decisions, JSON-ready traces, and conservative final answer payloads; optional conformal abstention gates can escalate accepted answers to abstain/clarify when a selected uncertainty score falls outside the retained participation region, `ControlPolicyConfig.participation_gate_supported_override` can opt into all-supported high-confidence verifier evidence preserving an otherwise accepted answer, invalid diagnostic values route to `clarify/unknown`, verification plans record claim-level route/tool intent, cost estimates, and optional verification-budget summaries, route summaries expose selected/matched/skipped verifier tools, action-audit summaries expose missing or malformed planned tool actions, action-execution summaries expose planned-action/result alignment, claim-risk localization summaries expose risky text spans and claim ids, triple-coverage summaries expose claim triples plus recorded slot-audit coverage, runtime/cache/tail-latency/route-cost summaries support optional budget gates, `ProductTrace` can carry the final answer for replay/feedback/runtime audits, `ProductTrace.to_bounded_dict()` emits smaller online telemetry payloads with final-answer, verification-budget, action-audit, action-execution alignment, and claim-risk localization summaries while corpus/replay/runtime-baseline tools keep full-trace reproduction inputs, and `finalize_loop_answer()` folds a verified control loop into answered/abstained/needs-retrieval/needs-clarification status without hidden rewrite. |
| `FeedbackOutcome` / `ProductFeedbackRecord` / `ProductFeedbackStore` | Records post-hoc product feedback in dependency-free JSONL form, linking user/manual/online outcomes to a request id, optional trace fingerprint, optional claim id, corrections, evidence refs, and metadata for later control-loop audit reports. |
| `DefaultCorrectionPolicy` / `PlanAwareCorrectionPolicy` / `ActionRequest` | Compiles control decisions into executable JSON-ready action payloads for product integrations, including generic `execute_tool` requests; the plan-aware wrapper can enrich or append retrieval actions from `ClaimVerificationPlan` retrieval queries while preserving the wrapped policy's default behavior. |
| `ActionAuditPolicy` / `ActionAuditReport` / `ActionAuditIssue` / `audit_action_requests` | Audits planned actions before execution without blocking them by default, checking decision/action alignment, missing retrieval actions from verification plans, non-executable retrieval payloads, malformed tool parameters, and claim-id mismatches; summaries are exposed through `ProductTrace` and `product_runtime_metrics()`. |
| `ActionExecutorRegistry` / `DryRunActionExecutor` / `TimeoutActionExecutor` / `ActionResult` | Routes action requests to registered executors, with side-effect-free dry-run fallback and best-effort timeout wrapping for local traces. |
| `ActionExecutionPolicy` / `PolicyGuardedActionExecutor` | Adds dependency-free request validation, idempotency replay, and audit metadata for side-effecting executors, including request ids, idempotency keys, and timeout bounds. |
| `InMemoryActionExecutionLedger` / `JsonActionExecutionLedger` / `SQLiteActionExecutionLedger` | Stores successful idempotent action results so repeated product requests can replay outputs without repeating side effects. |
| `run_verification_loop` / `StagedVerificationPolicy` / `EvidenceBundle` | Runs verify -> decide -> execute -> reverify loops, records a `ClaimVerificationPlan` in the trace, optionally gates expensive verifier routes behind diagnostic risk or sensitive claim metadata, and converts retrieval action results into verifier-ready evidence context. |
| `ClaimVerificationPlanner` / `ClaimVerificationPlan` / `VerificationBudgetPolicy` / `VerificationRouteHint` / `VerificationPlanCostEstimate` | Builds dependency-free claim verification plans from generated text or pre-extracted claims, optionally attaching rule-based fact triples to extracted claim metadata, while emitting JSON-ready verifier scope, route hints, retrieval queries, citation checks, calculator checks, structured-state checks, world-model checks, triple-evidence audit routes for sensitive factual claims, inferred claim dependencies, and relative route/tool cost estimates. `VerificationBudgetPolicy` can then select a budgeted subset of high-value claims/routes under claim, route, tool-payload, or estimated-cost caps while preserving dropped-claim/route audit metadata for trace/runtime reports. |
| `ClaimRiskSpan` / `ClaimRiskLocalizationReport` / `localize_claim_risk_spans` | Converts existing claim spans, verification results, route hints, and verification-budget drops into dependency-free span/claim risk annotations with risk level, score, status, confidence, routes, evidence count, and reasons; `ProductTrace` and runtime metrics expose compact summaries for product UI and audit without requiring a learned token detector. |
| `ClaimDependency` / `ClaimCoherenceReport` / `apply_claim_coherence` | Adds an optional dependency-graph coherence pass for claim verification: supported child claims are downgraded to insufficient evidence when required parent claims are missing or unsupported, and `run_verification_loop(..., enforce_claim_coherence=True)` records coherence reports in the trace. |
| `CitationRecord` / `CitationVerifier` / `extract_citation_references` | Adds a dependency-free citation-integrity route for caller-supplied citation catalogs, checking bracket refs, DOI, arXiv id, URL, author/year, title, and year metadata before cited claims fall through to broader retrieval or groundedness. |
| `ClaimTriple` / `RuleBasedTripleExtractor` / `RegexTripleExtractor` / `LookupTripleExtractor` / `CompositeTripleExtractor` / `TripleSlotEvidence` / `TripleEvidenceVerifier` | Adds a dependency-free subject-predicate-object audit path for claims: simple rules, configurable regex templates, offline external prediction lookups, composed extractors, or caller-provided metadata produce triples, then linked evidence snippets are checked for subject, predicate, and object slot coverage before a stricter support result is emitted. Audit reports include per-slot evidence details and claim-level slot coverage summaries for traces and release review. |
| `RetrievalActionExecutor` / `InMemoryRetriever` / `SQLiteFTSRetriever` / `HTTPJSONRetriever` / `ProvenanceFilteredRetriever` | Provides dependency-free retrieval executor shells for unsupported-claim evidence gathering, from local snippets or SQLite FTS to caller-provided HTTP JSON search services; retrieval adapter errors fail closed into structured `ActionResult` payloads, and provenance filters can require source, source prefixes, score floors, metadata tags, and per-source caps before hits become verifier evidence. |
| `CalculatorVerifier` | Provides a dependency-free deterministic calculator verifier for structured arithmetic claims, simple symbolic equations, and calculation metadata extracted from limited arithmetic text. |
| `QuestionAnswerVerifier` | Provides a dependency-free structured QA/domain-state verifier adapter for exact question and candidate-answer facts. |
| `StructuredFactVerifier` | Verifies natural-language claims against structured subject-predicate-object facts, supporting common KG-covered paraphrases, aliases, multi-object lists, injectable triple extractors, and object-mismatch refutation for known subject/predicate pairs. |
| `StructuredStateVerifier` / `StateCheck` | Provides a dependency-free structured state and business-rule verifier for database, policy, and domain-state checks. |
| `SQLiteStateSource` / `SQLiteStateQuery` | Loads read-only SQLite query results into structured verifier state without adding non-stdlib dependencies. |
| `ToolOutputStateSource` / `ToolOutputMapping` | Maps local tool or action execution outputs into structured verifier state for post-tool checks. |
| `RuleBasedWorldModelAdapter` / `WorldModelRule` | Applies auditable dependency-free domain/world-model transition rules with explicit action matches, structured-state preconditions, state updates, and fail-closed no-match metadata. |
| `EnsembleWorldModelAdapter` | Aggregates multiple world-model adapters, degrades prediction confidence by agreement rate, and fail-closes state-transition checks when consensus falls below `min_agreement`. |
| `WorldModelReference` / `WorldModelView` | Serializes the reference world-model contract and per-claim view used by state-transition verification, including viewed paths, state fingerprints, and assumptions. |
| `StateTransitionVerifier` / `StateTransitionCheck` | Uses a world-model adapter to predict next state after an action, then checks structured postconditions; `min_prediction_confidence` can fail closed on low-confidence predictions, and verifier metadata includes `world_model_reference`, `world_model_view`, and refuted-postcondition `world_model_conflict` summaries. |
| `CachedVerifier` / `CachedRetriever` / `CachedStateSource` | Adds request-scoped in-memory caching and hit/miss stats for repeated verifier, retrieval, and state-source calls. |
| `CompositeVerifier` / `RoutedVerifier` / `default_routed_verifier` | Compose deterministic tools with lexical, citation, retrieval, database, triple-evidence, or world-model verifiers; routing can use claim metadata, context, feature flags, or text patterns, cap route fanout, and record match reasons. The default route stack runs deterministic tools first, optional citation catalog checks when supplied, strict triple audits for sensitive factual claims next, and lexical groundedness as fallback. |
| `GroundednessVerifier` / `ClaimExtractor` | Extracts claim metadata and checks claims against lexical evidence snippets and explicit refutations without extra dependencies. |
| `SelfConsistencyVerifier` | Checks claims against caller-supplied sampled responses with FactSelfCheck-style support/refutation rates and no model or retrieval dependency. |
| `sqlite_state_control_demo.py` | Demonstrates SQLite-backed structured state checks feeding a final `ProductTrace` and dry-run action. |
| `state_transition_control_demo.py` | Demonstrates world-model next-state prediction plus structured postcondition checks feeding a final `ProductTrace`. |
| `production_tool_loop_demo.py` | Demonstrates a local production-like loop: SQLite pre-check, guarded side-effecting local `execute_tool`, optional JSON/SQLite idempotency ledger, tool-output state mapping, post-tool verification, action audit metadata, and route summary in one trace. |
| `eval_triple_extraction.py` | Evaluates dependency-free triple extractors against labeled triples, comparing rule-based, regex, regex-with-rule-based fallback, composite, and offline external-prediction lookup variants before they are promoted into structured-fact or triple-evidence routes. |
| `build_triple_extraction_fixture.py` | Generates labeled triple-extraction records and default regex templates from structured fact corpora, with optional negated, predicate-confusion, non-assertive, ambiguity, temporal, and metalinguistic negative controls for KG-covered extractor benchmarks. |
| `run_external_triple_extractor_handoff.py` | Writes label-free extraction requests, invokes a local external extractor command with `{input}`/`{output}` placeholders, evaluates the returned offline predictions with adversarial false-positive gates, and can emit a verified manifest plus registry record before learned/OpenIE/LLM-json extractors are used as release evidence. |
| `run_external_triple_extractor_matrix_handoff.py` | Runs one or more local external extractor commands across a cross-corpus fixture matrix: it builds deterministic per-corpus fixtures, sends label-free requests, gates each returned prediction file, feeds them into `run_triple_extraction_fixture_matrix.py`, and records matrix-level external-prediction evidence for release gates. |
| `run_triple_extraction_fixture_workflow.py` | Builds generated triple-extraction fixtures, evaluates rule-based/regex/composite extractors plus optional external prediction files, writes per-extractor reports plus subgroup gates and an artifact manifest for release evidence. |
| `run_triple_extraction_fixture_matrix.py` | Runs generated triple-extraction fixture workflows across multiple structured-fact corpora, including optional per-corpus external prediction files, requiring cross-corpus promotion, predicate diversity, and optional adversarial subgroup gates before extractor templates are treated as release evidence. |
| `triple_extraction_smoke.py` | Runs the bundled labeled triple-extraction fixture through rule-based, regex-with-fallback, and composite extractors, gating that configurable templates improve exact F1 without adding learned extractor dependencies. |
| `eval_verifier_ensemble.py` | Benchmarks calibrated internal diagnostics combined with retrieval/verifier suppression/refutation policies, optional staged verifier gating, optional triple-evidence route audits, route-level cost metrics, and optional JSONL sidecar records for per-claim verifier outputs. |
| `build_verifier_signal_score_dump.py` | Converts verifier verified-record JSONL sidecars into standard score-dump columns such as `verifier_refuted`, `verifier_uncertainty`, `selfcheck_refute_rate`, `world_model_disagreement`, `world_model_conflict`, and `world_model_conflict_delta`, including state-transition prediction/conflict metadata and direct world-model ensemble agreement metadata, enabling calibrated geometry fusion with external evidence. |
| `build_selfcheck_signal_score_dump.py` | Converts aligned sampled responses directly into score-dump columns such as `selfcheck_support_rate`, `selfcheck_refute_rate`, and `selfcheck_disagreement`, so self-consistency evidence can be calibrated or fused without going through a verifier sidecar. |
| `export_inside_diagnostics_samples.py` | Recovers sampled response texts from an `eval_truthfulqa.py --inside-diagnostics-cache` file into a selfcheck samples payload with source/cache/output manifest provenance. |
| `plan_selfcheck_sample_collection.py` | Preflights aligned selfcheck samples before direct signal fusion, reporting which records lack enough samples, total sample deficit, sample-quality gate projection, and rerun commands for INSIDE cache export or fusion replay; `run_selfcheck_signal_fusion_workflow.py` writes these plans into its workflow report and artifact manifest by default. |
| `build_text_baseline_score_dump.py` | Appends dependency-free text/length redline baselines such as answer length, claim length, lexical overlap, negation, and number counts to statement-bearing score dumps, so new detector claims can be compared against cheap lexical controls. |
| `fetch_wikidata_reference_docs.py` | Materializes small CC0 Wikidata SPARQL result sets as JSONL source documents for external retrieval corpus ingestion, including country capitals, template-ready country core facts, and organization/product facts for triple-extraction fixture matrices, with QID-only label filtering by default. |
| `build_external_retrieval_corpus.py` | Normalizes caller-supplied JSON/JSONL/text source files into an explicit `external_evidence_candidate` retrieval corpus and rejects score labels, claim ids, or score-dump row links in document metadata. |
| `audit_retrieval_corpus_provenance.py` | Audits retrieval corpora against statement-bearing score dumps, separating external grounding candidates from controlled dataset baselines and answer-echo/oracle-risk stress corpora. |
| `build_wikidata_qa_corpus.py` | Converts structured Wikidata facts such as `P36` country-capital records, or a multi-property template set, into label-free `QuestionAnswerVerifier` corpora for `retrieval_structured_qa` route smoke tests. |
| `run_wikidata_structured_qa_route_workflow.py` | Builds a balanced covered-facts score dump from a Wikidata QA corpus, runs either the structured QA route or natural-language structured-fact route, can expand structured-fact claims into paraphrase robustness variants, and writes a manifest-backed property-level correction artifact. |
| `analyze_retrieval_route_gaps.py` | Reads verifier verified-record JSONL sidecars and summarizes retrieval coverage, final statuses, gap buckets, hit sources, and examples for blocked retrieval routes. |
| `run_verifier_signal_fusion_workflow.py` | Runs the no-model local evidence loop end to end: retrieval/selfcheck fixture, optional retrieval provenance filtering, verifier sidecar, verifier-signal score dumps, geometry-fusion report, deployable geometry artifacts, and manifest verification. |
| `run_world_model_signal_calibration_workflow.py` | Runs a deterministic world-model correction loop end to end: state-transition fixture, optional controlled world-model ensemble disagreement, world-model verifier sidecar, verifier-signal score dump, score/geometry fusion report, release gate over trace-gap/conflict calibration evidence, nested manifest verification, and optional local registry record. |
| `run_selfcheck_signal_fusion_workflow.py` | Runs the direct no-model selfcheck signal loop end to end: sampled responses, selfcheck score dumps, sample-quality gate, score ensemble report, optional geometry-by-selfcheck fusion artifacts, and manifest verification. |
| `eval_verifier_stability.py` | Replays verifier-ensemble reports across multiple split-conformal seeds, summarizes verified risk stability and route-selection stability, fingerprints verifier inputs, and optionally registers the post-hoc report. |
| `eval_score_ensemble.py` | Benchmarks direction-aware rank fusion and geometry-by-uncertainty interaction fusion across saved diagnostic score dumps, and can save deployable `RankScoreFusionArtifact` or `GeometryScoreFusionArtifact` outputs in one run. |
| `select_fusion_signals_from_ablation.py` | Converts a saved fusion ablation matrix into a run-specific signal-selection report, keeping optional signals such as trajectory conditional instead of global defaults. |
| `build_selected_fusion_artifacts.py` | Builds per-run `RankScoreFusionArtifact` files from a signal-selection report and matching score dumps, turning conditional signal choices into calibrated deployable artifacts. |
| `eval_frontier_stability.py` | Replays saved frontier score dumps across multiple split-conformal seeds, summarizes best-signal stability, fingerprints source score records, and optionally registers the post-hoc stability report. |
| `run_calibrated_observability_workflow.py` | Runs or reuses a TruthfulQA score dump, executes conformal layer/score calibration, can derive `--sweep-layers` from a layer-band selector report, writes nested artifact manifests plus an evidence-bundle summary, forwards optional TruthfulQA cache paths, and optionally records the calibrated-observability closure in the local registry. |
| `run_truthfulqa_frontier_workflow.py` | Runs the multi-model/multi-scale TruthfulQA frontier workflow: calibrated-observability cells for Qwen/SmolLM2-style l20/l80 runs, optional per-cell cache roots for l80/multi-seed reuse, optional per-cell layer-band selector reports for dense reruns, cross-cell rank-fusion ensemble reporting, and a top-level manifest. |
| `run_pre_generation_probe_workflow.py` | Runs or reuses pre-generation hidden-state records, executes record-layer probe sweep, saves the recommended `AttentionSoftTargetProbeArtifact` plus split-conformal calibration, and writes a compact workflow report plus artifact manifest. |
| `compare_pre_generation_probe_workflows.py` | Aggregates compact pre-generation probe workflow reports across runs/models, applies ready/manifest/model-count/AUROC gates, and writes a leaderboard-style comparison report with optional artifact manifest. |
| `compare_manifold_distances.py` | Builds a Gaussian 2-Wasserstein/Bures distance matrix from saved `TruthManifold` artifacts or `eval_truthfulqa.py` layer-stats caches for offline layer/checkpoint drift inspection. |
| `eval_intrinsic_dimension.py` | Builds TwoNN intrinsic-dimension profiles from saved warmup checkpoints, including peak-layer and rise-then-fall shape summaries without reloading model weights. |
| `compare_intrinsic_dimension_layers.py` | Compares intrinsic-dimension peak layers with saved layer/score sweep AUROC rankings, reporting top-k hits, rank, AUROC regret, and layer gap for cheap layer-selection validation. |
| `compare_spectrum_layers.py` | Compares Marchenko-Pastur spike/effective-rank spectrum heuristics with saved layer/score sweep AUROC rankings, reporting per-heuristic top-k hits, rank, AUROC regret, and layer gap without rerunning models. |
| `compare_layer_band_selectors.py` | Compares intrinsic-dimension and spectrum-derived layer bands against calibrated sweep rankings, then recommends a cost-reducing candidate band when it keeps the best layer in band. |
| `audit_layer_band_replication.py` | Gates a layer-band selector before it becomes a default preset by requiring matched runs, model-family diversity, dense ranked-layer grids, best-layer hit rate, AUROC regret, and candidate-layer cost reduction. |
| `training_telemetry_sanity.py` | Runs a deterministic synthetic clean-vs-corrupt training telemetry sanity check, gating on distance-to-baseline growth and effective-rank collapse. |
| `training_telemetry_tiny_finetune.py` | Runs a pure PyTorch tiny clean-vs-duplicate fine-tune comparison and checks whether representation-rank telemetry separates before eval-loss degradation. |
| `model_collapse_early_warning.py` | Runs a deterministic pseudo-label self-training loop and checks whether representation diversity telemetry warns before visible quality loss. |
| `trajectory_convergence_sanity.py` | Runs a deterministic synthetic generation-trajectory sanity check and reports Spearman/AUROC correlation between convergence diagnostics and quality proxies. |
| `eval_trajectory_truthfulqa.py` | Replays statement-bearing TruthfulQA score dumps through a causal LM, extracts forced-answer hidden-state trajectories over answer-token prediction positions, and reports trajectory/NLL correlation and AUROC against true/false labels; `--offline` runs a deterministic no-download smoke fixture. |
| `eval_pre_generation_probe.py` | Trains and evaluates a torch-only pre-generation attention risk probe from local JSON/JSONL token hidden-state records with soft error-rate targets, including single- or multi-layer records exported by `eval_truthfulqa.py --dump-pre-generation-probe-records`; can sweep record layers, rank candidates, and optionally save the best `AttentionSoftTargetProbeArtifact` plus split-conformal `CalibrationArtifact`. |
| `eval_pre_generation_text_baselines.py` | Evaluates cheap answer/question text redline baselines directly over pre-generation records, selecting the best direction per feature so probe claims can be checked against length, overlap, negation, and number-count controls. |
| `compare_trajectory_sweeps.py` | Compares forced-answer trajectory sweep reports across runs and applies a fail-closed evidence gate before trajectory signals can be treated as release evidence. |
| `concept_registry_smoke.py` | Saves two synthetic `ConceptArtifact` files, registers them locally, attaches both probes to one toy model, and writes a manifest-backed multi-concept diagnostics report. |
| `refresh_verifier_route_artifacts.py` | Regenerates new-schema verifier-route reports from saved score dumps, claims, and local verifier corpora without rerunning model forward passes. |
| `compare_verifier_routes.py` | Aggregates saved verifier-ensemble reports into cost-aware route leaderboards, Pareto frontier candidates, route-specific promotion decisions, by-route control-impact metrics, and optional tail/cache/staged-verification route quality gates. |
| `run_adapter_promotion_workflow.py` | Runs a fail-closed adapter promotion workflow: route comparison, `promotion_decision=promote`, and optional registry-backed performance baseline gate. |
| `run_adapter_promotion_registry_workflow.py` | Runs route promotion, writes a manifest, recursively verifies it, and registers the promoted route baseline in one command. |
| `compare_route_baselines.py` | Compares registered verifier-route promotion manifests or covered-facts route-summary manifests by verified state, route quality, false support/refutation, tail latency, retrieval cost, optional covered-fact per-property gates, optional retrieval provenance-filter gates, and optional answer-echo retrieval stress-control gates. |
| `compare_external_evidence_baselines.py` | Combines route-baseline promotion, answer-echo stress-control evidence, text/length redline score-ensemble reports, and optional covered-facts structured route gates, including per-property covered-fact minima, into one fail-closed external-evidence comparison artifact with optional manifest verification and `report:*:*` registry handoff. |
| `run_covered_facts_external_evidence_workflow.py` | Registers one or more saved Wikidata covered-facts route manifests, runs the external-evidence comparator with `--require-covered-facts-route`, writes a verified manifest, and optionally records the comparator report as the release-gate handoff. |
| `run_adapter_family_matrix.py` | Builds deterministic structured QA, structured-state, state-transition, optional retrieval-groundedness, optional retrieval-structured-QA, and optional strict triple-evidence fixtures, then compares their promotion metrics in one local matrix. |
| `run_local_retrieval_route_workflow.py` | Builds local retrieval evidence from score dumps and corpora, optionally applies retrieval provenance filters, promotes retrieval routes, fingerprints all source artifacts, records a runtime profile, optionally attaches answer-echo stress-control evidence, optionally uses persistent SQLite FTS/cached claims fixtures/verifier traces, and optionally registers the route baseline. |
| `run_cache_profile_matrix.py` | Runs same-machine profile sweeps across layers, batch sizes, capture modes, and TruthManifold covariance modes, then emits a matrix-level performance promotion decision with per-cell AUROC quality signals. |
| `run_cache_worker_sweep.py` | Runs the same cache-profile matrix across several worker counts and recommends the fastest promoted worker count by wall-clock time. |
| `run_inside_sampling_profile.py` | Compares fixed, adaptive, and self-check-bounded INSIDE sampling runs, producing sample-count and `inside_generation` cost evidence for release gates. |
| `run_inside_trigger_budget_sweep.py` | Runs several triggered INSIDE budgets and compares generated samples, `inside_generation`, reference ratios, inside-score AUROCs, and cost-first / quality-balanced recommendations. |
| `recommend_runtime_config.py` | Converts promoted matrix/worker-sweep reports plus optional INSIDE sampling, trigger-budget sweep, score-ensemble, and selected-fusion-artifact evidence into one deployable runtime recommendation: layer, batch size, covariance mode/rank, token budget, prefix KV mode, worker count, sampling flags, derived-sweep flags, gated best AUROC quality signal, covariance tradeoff details, and cache-tuning advice. |
| `run_performance_baseline_workflow.py` | Builds a registry-ready performance baseline bundle from cache matrix, optional worker sweep, optional INSIDE profile / trigger-budget / score-ensemble / selected-fusion-artifact evidence, runtime recommendation, artifact manifest, performance evidence summary, and optional recursive manifest verification. |
| `run_product_runtime_baseline.py` | Aggregates saved `ProductTrace` JSON files into a request-runtime baseline with phase, cache, verifier-route, route-budget exhaustion, retrieval-use, staged-verification savings, verification-plan coverage/route/tool-payload counts, action-execution planned/result alignment, action-audit error/missing/malformed action rates including uncovered planned retrieval queries, promotion-contract product-trace replay action-audit/action-execution gate metadata, external-evidence baseline-comparison coverage/status/route/text-redline handoff metadata, pre-generation probe comparison/redline handoff metadata, verification-budget exhaustion summaries, claim-risk localization summaries, triple/slot-audit coverage summaries, final-answer coverage/status/action summaries, promotion-contract covered-fact property scope and per-property quality rollups, runtime-profile context, an `optimization` block with phase/route/triple-audit hotspots, cache/retrieval/staging/profile recommendations, candidate control-default hints such as `max_verifier_route_attempts`, optional `ProductRuntimeBudgetPolicy` gate metrics, optional reusable recommended policy artifact, optional JSONL sidecar storage for per-trace records, optional trace-record cache reuse for repeated budget sweeps, and optional parallel trace scanning for large trace sets. |
| `run_product_feedback_report.py` | Joins saved `ProductTrace` JSON payloads with `ProductFeedbackRecord` JSONL outcomes, reports accepted-but-wrong, retrieved-failure, retrieved-but-still-unsupported, abstain-false-positive, final-answer answered-but-wrong and final-answer false-block rates, matched/unmatched feedback counts, optional quality gates, manifest provenance, and optional registry metadata. |
| `recommend_control_policy_from_feedback.py` | Turns one or more product feedback reports into a candidate `ControlPolicyConfig` plus runtime control-default recommendations, using accepted-but-wrong, retrieval-failure, abstain-false-positive, final-answer answered-but-wrong, and final-answer false-block rates to close the post-hoc feedback-to-policy loop without retraining or new dependencies. |
| `audit_feedback_policy_replay.py` | Audits a feedback-derived candidate policy against historical feedback labels, reporting counterfactual safety coverage, unknown claim-metadata gaps, residual safety issues, and overblock relief without claiming an exact controller rerun. |
| `run_feedback_policy_workflow.py` | Runs the complete post-hoc feedback-to-policy loop in one command: feedback join report, candidate control-policy recommendation, inline candidate `ControlPolicyConfig`, replay audit, top-level manifest, and optional registry record; registered `report:<name>:<version>` records can be consumed by the release-candidate gate. |
| `compare_product_runtime_baselines.py` | Compares current product runtime baselines against a file or registered baseline, emitting fail-closed drift gates over latency, route cost, retrieval use, cache reuse, verifier skip rate, promotion-contract/triple-extraction evidence coverage and quality drift, pre-generation probe comparison/redline handoff drift, covered-fact per-property rollup drift, product-trace action-audit/action-execution gate rate drift, trace-level triple/slot-audit coverage, trace count, and an optional reusable runtime budget policy. |
| `build_product_trace_corpus.py` | Validates saved ProductTrace JSON/JSONL payloads, optionally redacts text fields including final-answer text/evidence, writes replay-ready standardized traces plus a runtime-pair index, can reuse a per-source validation/redaction cache, and registers a manifest-backed trace corpus with final-answer status/action counts. |
| `run_product_trace_replay_workflow.py` | Runs the raw-trace handoff end to end: redacted trace corpus, runtime-pair index, product runtime baseline, optional action-audit release gate over missing retrieval actions, uncovered planned retrieval queries, and malformed/unexpected action payloads, optional action-execution alignment gate over missing/unexpected/mismatched action results, selector replay, optional product-runtime drift/policy/promotion-evidence/pre-generation-probe drift, covered-fact property drift, and product-trace action-gate drift gates, top-level runtime `optimization` summary, optional recommended runtime policy artifact, recursive manifest, optional manifest verification, optional manifest fingerprint cache reuse, optional whole-corpus cache reuse, optional corpus source-cache reuse, optional runtime trace-record cache reuse, optional runtime trace-scan parallelism, optional selector trace-input cache reuse, phase timing/cache summaries, and registry-ready workflow report. |
| `run_product_runtime_profile_sweep.py` | Runs deterministic calibrated-control demo scenarios under `latency`, `balanced`, `audit`, and request-level `auto` selection modes, writes or reuses traces, carries promotion-contract/profile effective control-default summaries such as `max_verifier_route_attempts`, can cache per-profile trace records for repeated budget/SLO sweeps, builds per-mode baselines, applies optional aggregate SLO gates, and recommends the lowest-cost non-blocked mode. |
| `run_release_efficiency_report.py` | Summarizes a product runtime profile sweep plus optional quality/release reports into a registry-ready efficiency handoff with profile-level runtime, verifier, cache, trace-reuse, and route-fanout evidence. |
| `run_runtime_profile_selector_tuning.py` | Compares candidate `RuntimeProfileSelectorPolicy` JSON configs by running auto-profile sweeps under the same SLO gate and recommending the lowest-cost promoted selector. |
| `run_runtime_profile_selector_replay.py` | Replays candidate `RuntimeProfileSelectorPolicy` JSON configs over saved `ProductTrace` files, estimates profile cost/distribution, paired observed runtime, selected-vs-original runtime delta from trace scan or a corpus runtime-pair index, can cache minimal trace replay inputs for repeated policy sweeps, and registers the lowest-cost promoted selector. |
| `run_adapter_readiness_workflow.py` | Combines adapter-family quality gates, including optional retrieval and strict triple-evidence audit families, cache-profile performance gates, and optional INSIDE sampling / trigger-budget gates into one final readiness decision, runtime recommendation, and registry-ready manifest. |
| `run_adapter_readiness_registry_workflow.py` | Runs readiness gates, including optional retrieval/triple-evidence adapter-family evidence, and registers the verified manifest as a reusable local promotion baseline when readiness promotes. |
| `compare_readiness_baselines.py` | Compares registered readiness baselines by verified manifest state, best AUROC quality signal, runtime cost, optional covariance `maha_last` quality-drop gate, and INSIDE profile or trigger-budget cost evidence, then recommends one deployable baseline. |
| `compare_release_candidates.py` | Combines registered readiness, route, optional required route-baselines including robustness evidence, performance-baseline, product-trace-replay workflow file or registry key, selfcheck-signal-fusion workflow file or registry key, world-model-signal workflow file or registry key, feedback-policy workflow file or registry key, selector-replay, product-runtime-drift, release-efficiency, external-evidence baseline comparison file or registry key, pre-generation probe comparison file or registry key, adapter-family evidence, triple-extraction fixture matrix evidence, and counterfactual verifier-audit evidence into one fail-closed release candidate with runtime flags, verifier route, covered-fact property coverage, quality, runtime cost, covariance quality-drop gate, performance evidence bundle/cache gates, optional performance trend gates against an explicit prior baseline, selfcheck sample-quality/fusion gates, world-model trace-gap/conflict-calibration gates, pre-generation multi-model/redline evidence gates, product-trace action-audit and action-execution gate enforcement, audit-route budgets, retrieval provenance-filter gates, retrieval stress-control gates, strict-audit adapter-family profile support, optional external-prediction count/corpora/F1 gates for learned triple extractors, named release-policy profiles such as `strict_structured_fact` and `frontier_audit` where `frontier_audit` defaults the registered covered-facts external-evidence handoff, registered triple-extraction fixture matrix, external-prediction triple-extraction minima, `--require-product-runtime-drift-promotion-evidence`, `--require-product-runtime-drift-triple-audit-evidence`, `--require-product-runtime-drift-covered-fact-property-evidence`, `--require-product-runtime-drift-action-gate-evidence`, `--require-product-trace-action-audit-gate`, and `--require-product-trace-action-execution-gate`, validated feedback-policy config/control-default provenance, action-level and final-answer feedback safety metrics, runtime drift provenance including promotion/triple-audit/covered-fact property/action-gate evidence summaries, release-efficiency profile handoff, and latency/balanced/audit profiles. |
| `run_release_candidate_registry_workflow.py` | Runs the release-candidate gate, writes a manifest covering the candidate report plus selected readiness/route/performance/product-trace-replay/selfcheck-signal-fusion/world-model-signal/feedback-policy/selector/runtime-drift/release-efficiency manifests, optional product-trace action-audit and action-execution gate reports, optional external-evidence and pre-generation probe comparison reports supplied by file or registry key, and optional required route/adapter-family/triple-extraction fixture-matrix/counterfactual-verification reports, supports named release policy profiles such as `strict_structured_fact` and `frontier_audit`, can explicitly require canonical plus paraphrase `structured_fact` robustness evidence, applies path-aware `frontier_audit` defaults for registered external-evidence and triple-extraction matrix handoffs, recursively verifies it, and registers the final candidate with runtime-profile, covariance quality gate, performance-baseline, performance evidence bundle readiness/cost/cache/trend, product-trace-replay/action-audit/action-execution, selfcheck signal-fusion workflow status/report/record/sample-quality metadata, world-model signal workflow gate/report/record metadata, pre-generation probe comparison source/record/redline metadata, feedback-policy workflow status/report/record/candidate-policy metadata, selector-replay, runtime-drift, release-efficiency, external-evidence comparison source/record metadata, route-budget, retrieval provenance-filter policy metadata, release-policy profile, structured-fact robustness property coverage, adapter-family metadata, cross-corpus triple-extraction evidence, optional external-prediction gate metadata, and counterfactual verifier-audit pass-rate/false-invariance metadata; compare, manifest-build, promotion verification, and workflow-report writes emit phase timing, compare/manifest/promotion share fingerprint and JSON artifact caches, `--fingerprint-cache` persists file fingerprints, `--artifact-json-cache` persists parsed JSON artifact payloads across repeated release checks, and `--manifest-fingerprint-workers` enables bounded parallel artifact fingerprinting for local release gates. |
| `run_manifest_fingerprint_worker_sweep.py` | Replays artifact-manifest verification over one or more saved manifests with multiple bounded fingerprint worker counts, records per-worker timing/cache summaries, recommends the fastest passing worker count for local release checks, and can register the sweep as a `report:*:*` artifact. |
| `run_release_gate_overhead_baseline.py` | Aggregates one or more release-candidate registry workflow JSON reports into a release-gate overhead baseline with total/phase timing summaries, artifact fingerprint/JSON cache hit-rate summaries, hotspot recommendations, optional timing/cache gates, and optional `report:*:*` registry recording. |
| `export_product_promotion_contract.py` | Exports a compact, deployable `ProductPromotionContract` JSON from a promoted release candidate, including verifier-route covered-fact property scope, product-trace replay action-audit/action-execution gate provenance, runtime-drift promotion/triple-audit/action-gate evidence, selfcheck-signal-fusion provenance, world-model signal release-gate provenance, external-evidence baseline-comparison provenance, pre-generation probe comparison/redline provenance, triple-extraction fixture-matrix provenance, and any feedback-derived `ControlPolicyConfig`, writes a manifest, and can register a `product_promotion_contract:*:*` handoff artifact. |
| `build_domain_state_fixture.py` | Builds deterministic order-fulfillment score/claim/state fixtures plus optional SQLite state-source specs for structured-state verifier benchmarks. |
| `build_transition_fixture.py` | Builds deterministic order-reservation transition fixtures for state-transition verifier benchmarks, including optional controlled or policy-replay world-model ensemble disagreement. |
| `build_truthfulqa_corpus.py` | Builds a local TruthfulQA correct-answer corpus for reproducible non-oracle retrieval baselines. |
| `build_retrieval_stress_corpus.py` | Builds answer-echo retrieval stress corpora from statement-bearing score dumps, exposing self-support failure modes when retrieval evidence comes from the same answers being audited. |
| `fetch_wikidata_reference_docs.py` | Fetches or replays Wikidata country-capital, country-core-fact, or organization/product-core-fact SPARQL results into JSONL source docs for external evidence smoke gates; the core-facts audit artifact remains blocked for lexical route promotion. |
| `build_external_retrieval_corpus.py` | Builds explicit external-candidate retrieval corpora from local source files before provenance audit and local retrieval fixture construction. |
| `build_wikidata_qa_corpus.py` | Converts Wikidata fact documents into template-driven structured QA corpora consumed by `QuestionAnswerVerifier` and `retrieval_structured_qa`. |
| `run_wikidata_structured_qa_route_workflow.py` | Runs covered-facts structured QA or structured-fact route workflows, optionally expands structured-fact claims into paraphrase robustness variants, and records support/refutation metrics for Wikidata properties present in the QA corpus. |
| `analyze_retrieval_route_gaps.py` | Explains blocked retrieval routes from verified-record sidecars by coverage, status, gap bucket, source, and example records. |
| `build_evidence_fixture.py` | Builds non-oracle claim/evidence fixtures from statement-bearing score dumps and local JSON/JSONL/text corpora, with optional source/metadata/score provenance filters before hits become verifier evidence. |
| `backfill_truthfulqa_statements.py` | Rebuilds deterministic TruthfulQA statement metadata for older score dumps and can emit label-derived oracle evidence for verifier upper-bound checks. |

### 主要组件

| 组件 | 用途 |
|---|---|
| `TruthManifold` / `RepresentationManifold` / `CovarianceSpectrum` / `covariance_spectrum` | 用 Welford 维护在线均值与协方差，对外暴露为按样本数归一化、ridge 正则化的精度矩阵；支持 `covariance_mode="full"`、`"diag"`、实验性 `"low_rank"` 和 OAS 风格 `"shrinkage"`，便于本地 benchmark 在精确协方差评分、低内存/低计算成本和小样本病态矩阵稳定性之间取舍；`TruthManifold.spectrum()` 会输出 Marchenko-Pastur bulk 边界、spike count、effective rank、participation ratio、stable rank 和 condition number，作为无新增依赖的表征谱诊断。 |
| `mahalanobis_distance` | 测量相对于 warmup 流形的相对偏移。 |
| `gaussian_wasserstein_distance` / `manifold_distance` / `manifold_wasserstein_distance` | 计算无新增依赖的 Gaussian 2-Wasserstein/Bures 距离，用于比较不同 layer、checkpoint 或 run 的表征流形。 |
| `twonn_intrinsic_dimension` / `intrinsic_dimension_profile` | 基于 hidden-state 样本估计无新增依赖的 TwoNN intrinsic dimension，为 layer selection 和 representation-collapse 实验提供低成本层级 profile 证据。 |
| `RepresentationTelemetryRecorder` / `RepTelemetryCallback` / `representation_telemetry_snapshot` | 无需强绑定 Trainer 的训练侧逐层表征 telemetry：记录 mean norm、variance trace、谱 rank 诊断，以及到初始化 baseline 的 Gaussian 2-Wasserstein/Bures 距离；可选 callback 暴露 HF Trainer-compatible hook 名称，但本身仍无 transformers 强依赖。 |
| `TrajectoryMonitor` / `trajectory_convergence_metrics` | 从逐 token hidden states 计算 generation trajectory convergence 诊断，包括 step-distance decay、Koopman-style rate、path efficiency 和用于质量/置信相关性检查的 convergence score。 |
| `AttentionSoftTargetProbeArtifact` / `soft_error_rate_targets` | 用 empirical sampled-answer error rate 作为 soft target，在 token-level hidden states 上训练 torch-only 生成前 attention probe；artifact 暴露风险概率、attention weights、JSON metadata 和 torch save/load。 |
| `poincare_map` | 将表征投影到有界双曲空间，供可选 HSE 消融使用。 |
| `hyperbolic_semantic_entropy` | 测量投影状态滑动窗口内的离散程度；保留为 opt-in 消融信号，不作为默认 runtime 路径。 |
| `internal_eigenscore` / `spectral_effective_rank` / `cluster_assignment_entropy` / `lexical_semantic_entropy` / `embedding_semantic_entropy` / `semantic_energy_score` / `lexical_semantic_energy` | 基于隐藏态嵌入与文本簇计算 INSIDE/EigenScore 风格谱分散度、无依赖语义熵代理和置信度加权 semantic-energy 分歧；benchmark 可选多采样续写生成 `inside_eigenscore`、`inside_semantic_entropy`、`inside_embedding_entropy` 和 `inside_semantic_energy`。 |
| `TruthProbe` / `RepresentationProbe` | 捕获指定层的隐藏状态，并可选地应用激活引导；HSE 采集需要显式 `track_hse=True`。 |
| `ConceptArtifact` / `MultiConceptMonitor` | 保存带 layer metadata 的版本化 concept manifold，并把多个 concept probe 同时挂到一个模型上，返回逐 concept 诊断，不改变现有单 probe wrapper 路径。 |
| `EigenTruthWrapper` / `RepresentationMonitor` | 提供 warmup、生成透传、诊断信息和探针生命周期管理。 |
| `TruthSubspace` | 拟合低秩事实子空间，并提供残差距离诊断；拟合至少需要两条事实状态。 |
| `directional_conformal_threshold` / `directional_conformal_thresholds` / `directional_trigger_rate` | 对 `higher` 与 `lower` 异常方向使用一致的 split-conformal 阈值与触发率，并支持一次排序计算多个 alpha 阈值。 |
| `conformal_abstention_report` / `conformal_abstention_comparison_report` / `conformal_abstention_release_gate` | 将任意 uncertainty/reliability score 校准成选择性参与阈值，输出 coverage、selective accuracy 和 conservative conditional-correctness lower bound；comparison report 可按保守正确性、选择性准确率、参与率或保留率对多个候选信号排序，release gate 会要求最低保守条件正确率和最高 abstention rate 后才允许提升为参与控制候选；runtime 可用 `report.decide(score)` 返回 `participate/abstain` 决策，也可交给 `ParticipationGateConfig` 接入 `RiskController`。 |
| `AdaptiveScoreTransform` / `adaptive_anomaly_scores` / `AdaptiveConformalCalibrator` | 提供无依赖 adaptive conformal scoring 层：把原始诊断统一转成 anomaly 方向，叠加调用方提供的语义/风险 feature inflation，并把调整后的分数保存成普通 `CalibrationArtifact`。 |
| `directional_rank_anomaly_scores` / `combine_rank_anomaly_scores` / `RankScoreFusionArtifact` / `RankScoreFusionCalibrator` | 将不同方向的诊断分数转成经验异常 rank，用无依赖 rank 方法融合，并保存可部署的 conformal fusion artifact，供后续控制面实验复用。 |
| `geometry_calibrated_anomaly_scores` / `GeometryScoreFusionArtifact` / `GeometryScoreFusionCalibrator` | 构建无依赖 geometry-by-uncertainty anomaly score：先把 representation geometry 信号和 confidence / semantic-energy proxy 做 rank calibration，再加入显式交互项，并可保存为 conformal fusion artifact。 |
| `SignalSelectionPolicy` / `select_signals_from_fusion_ablation_matrix` | 将 score-fusion ablation matrix 转成按 run 选择的信号组合；只有当 trajectory 等可选信号在 detection、AUROC 和 false-alarm margin 上通过策略门槛时才启用。 |
| `ScoreDump` / `ScoreDumpIdentity` / `load_score_dump` / `load_score_dump_columns` / `load_score_dump_columns_with_extras` / `load_score_dump_statement_scores` / `load_score_dump_layer_scores` / `score_dump_file_metadata` / `score_dump_identity` / `score_dump_cache_summary` | 对逐陈述 score dump 做统一校验，暴露紧凑 run summary，并给后处理校准或 ensemble report 附带 SHA-256 provenance 和稳定的 model/dataset/layer/score-schema/scoring-config identity，不重跑模型；调用方可共享可选 run-local cache，避免单次运行内重复 hash 文件、重复解析 JSONL manifest 和重复扫描同一个 JSONL selected view，并可汇总 cache hits/misses/writes 用于报告观测。`ScoreDumpJsonlManifest` / `ScoreDumpRecord` / `iter_score_dump_jsonl_records` 提供可选的 manifest-backed JSONL 大文件格式，selected-column loader 可从该格式流式读取选中的 primary、adaptive extra、带 statement 的 primary 或 layer/score 视图，缓存视图会在 manifest 或 records 文件变化后自动失效。metadata 会同时 fingerprint manifest 和 records 文件，包含 canonical identity cache key，在 manifest 自带 label counts 时走 summary fast path，并复用同一次运行中 selected JSONL scan 预热的 records hash。`eval_truthfulqa.py --dump-scores-format jsonl` 可直接写出该格式，并把 INSIDE sample counts 等逐记录字段放进 records sidecar。 |
| `LayerScoreSweepCalibrator` | 从分数 dump 构建层/分数 sweep report 与可复用校准 artifact，支持通过 `calibrate_from_score_dump()` 直接消费 `ScoreDump`，也支持 `calibrate_from_file()` 对 JSONL manifest 做 selected layer-score 读取，并可选复用 run-local cache；大规模后处理 sweep 可用受控 `max_workers` CPU 并行。 |
| `ArtifactRegistry` / `ArtifactVerificationContext` / `build_artifact_manifest` / `fingerprint_path` / `fingerprint_cache_summary` / `load_json_object` / `json_cache_summary` / `load_fingerprint_cache` / `save_fingerprint_cache` / `load_json_cache` / `save_json_cache` / `verify_artifact_manifest` | 用 dependency-free 的 file/directory SHA-256 provenance 记录和校验本地 artifact metadata，包括 concept artifact，支持可复现 benchmark chain；递归 verification 共享 run-local fingerprint cache，`ArtifactVerificationContext` 同时为 release/registry comparison 共享 path-signature JSON artifact cache，cache summary 会报告 fingerprint requests/hits/misses/hit rate，CLI 可将 fingerprint cache 与部分 JSON artifact cache 持久化为 JSON，在文件签名不变时避免重复内容读取和 JSON 解析；持久化 JSON cache 保存时会裁剪同一路径旧签名。 |
| `RuntimeProfile` / `PreGenerationRiskPolicy` / `SoftPreGenerationRiskConfig` / `select_pre_generation_profile` / `RuntimeProfileSelectorPolicy` / `select_runtime_profile` | 定义 release gate 和产品控制面共用的 `latency`、`balanced`、`audit` 默认档位，并提供无依赖的生成前 prompt/metadata 路由、带 feature/metadata contribution trace 的可选 soft risk probability、诊断后的 claim-metadata 路由，以及可选的 verification-plan 成本路由。 |
| `ProductPromotionContract` / `ProductRuntimeEvidenceBundle` | 将已 promoted release-candidate report 转成产品 runtime、verifier route、covered-fact property scope、covariance 质量 gate provenance、performance-bundle、performance score-dump cache、包含 action-audit/action-execution gate report 与 rate 的 product-trace-replay workflow provenance、selfcheck-signal-fusion 样本质量/融合证据 provenance、world-model signal release-gate provenance、external-evidence baseline-comparison handoff provenance、pre-generation probe comparison/redline provenance、包含 final-answer 反馈率的 feedback-policy workflow provenance、selector-replay、包含 promotion-contract coverage、triple-extraction fixture-matrix drift 摘要和 trace-level triple-audit summary 的 runtime-drift evidence、release-efficiency handoff evidence、budget policy contract、已校验的反馈派生 `ControlPolicyConfig` 和可被 calibrated-control demo 消费的候选 control defaults；`load_product_promotion_contract()` 加载 compact contract，`load_product_runtime_evidence_bundle()` 延迟附加可选 promotion-contract manifest/registry provenance，并可显式验证 selfcheck-signal-fusion、world-model-signal、external-evidence baseline-comparison、pre-generation probe comparison 与 triple-extraction fixture-matrix workflow manifest、输出 registry metadata 到 runtime trace。 |
| `RiskController` / `ParticipationGateConfig` / `ProductTrace` / `FinalAnswer` | 将校准诊断和可选验证结果转为结构化路由决策、JSON trace 和保守最终响应 payload；可选 conformal abstention gate 会在被接受答案的 uncertainty score 落到保留参与区间外时升级到 abstain/clarify；`ControlPolicyConfig.participation_gate_supported_override` 可选择开启验证感知覆盖，只有全部 claim 都被高置信 verifier 支持时才保留原本会被 participation gate 拒答的 accepted answer；非法诊断值会路由到 `clarify/unknown`，verification plan 会记录 claim 级 route/tool 意图、成本估计和可选 verification-budget 摘要，route summary 会暴露选中、匹配和跳过的 verifier 工具，action-audit summary 会暴露缺失或参数异常的 planned tool action，action-execution summary 会暴露 planned action/result 对齐情况，claim-risk localization summary 会暴露高风险文本 span 和 claim id，triple coverage summary 会暴露 claim triples 和已记录的 slot-audit 覆盖率，runtime/cache/tail-latency/route-cost summary 支持可选预算门禁，`ProductTrace` 可携带最终响应用于 replay/feedback/runtime 审计，`ProductTrace.to_bounded_dict()` 可输出带 final-answer、verification-budget、action-audit、action-execution alignment 和 claim-risk localization summary 的更小线上 telemetry payload，corpus/replay/runtime-baseline 工具继续使用完整 trace 作为复现输入，`finalize_loop_answer()` 可把验证闭环折叠成 answered/abstained/needs-retrieval/needs-clarification 状态且不隐藏调用 rewrite LLM。 |
| `FeedbackOutcome` / `ProductFeedbackRecord` / `ProductFeedbackStore` | 用无依赖 JSONL 记录产品反馈，把人工、线上或用户反馈结果关联到 request id、可选 trace fingerprint、可选 claim id、修正文案、证据引用和 metadata，供后续控制闭环审计报告使用。 |
| `DefaultCorrectionPolicy` / `PlanAwareCorrectionPolicy` / `ActionRequest` | 将控制决策编译为面向产品集成的 JSON action payload，包括通用 `execute_tool` 请求；plan-aware wrapper 可从 `ClaimVerificationPlan` 的 retrieval query 补充或追加 retrieval action，同时保留被包装策略的默认行为。 |
| `ActionAuditPolicy` / `ActionAuditReport` / `ActionAuditIssue` / `audit_action_requests` | 在执行前对 planned action 做 monitor-first 审计，检查决策/action 是否一致、verification plan 需要 retrieval 但未计划 retrieval action、retrieval payload 无法被本地 executor 执行、tool 参数异常和 claim id 不匹配；摘要会进入 `ProductTrace` 和 `product_runtime_metrics()`。 |
| `ActionExecutorRegistry` / `DryRunActionExecutor` / `TimeoutActionExecutor` / `ActionResult` | 按 action 路由 executor，并用无副作用 dry-run 与 best-effort timeout wrapper 支撑本地 trace。 |
| `ActionExecutionPolicy` / `PolicyGuardedActionExecutor` | 为有副作用 executor 增加无依赖请求校验、idempotency replay 和审计元数据，包括 request id、idempotency key 与 timeout 上限。 |
| `InMemoryActionExecutionLedger` / `JsonActionExecutionLedger` / `SQLiteActionExecutionLedger` | 保存成功的幂等 action 结果，让重复产品请求可重放输出而不重复执行副作用。 |
| `run_verification_loop` / `StagedVerificationPolicy` / `EvidenceBundle` | 执行 verify -> decide -> execute -> reverify 闭环，把 `ClaimVerificationPlan` 写入 trace，可按诊断风险或敏感 claim metadata 延迟触发昂贵 verifier，并把 retrieval action result 转成 verifier 可消费的 evidence context。 |
| `ClaimVerificationPlanner` / `ClaimVerificationPlan` / `VerificationBudgetPolicy` / `VerificationRouteHint` / `VerificationPlanCostEstimate` | 从生成文本或已抽取 claim 构建无依赖 verification plan，可选把 rule-based fact triples 附加到抽取出的 claim metadata，同时输出 JSON-ready 的验证范围、route hints、retrieval query、citation check、calculator check、结构化状态检查、world-model check、面向敏感事实 claim 的 triple-evidence audit route、推断出的 claim 依赖和相对 route/tool 成本估计。`VerificationBudgetPolicy` 可在 claim、route、tool-payload 或估算成本上限内选择高价值 claim/route 子集，同时把被丢弃的 claim/route 写入 trace/runtime 报告。 |
| `ClaimRiskSpan` / `ClaimRiskLocalizationReport` / `localize_claim_risk_spans` | 将已有 claim span、verification result、route hint 和 verification-budget drop 转成无依赖的 span/claim 风险标注，包含 risk level、score、status、confidence、routes、evidence count 和 reasons；`ProductTrace` 与 runtime metrics 会输出 compact summary，供产品 UI 和审计使用，不要求 learned token detector。 |
| `ClaimDependency` / `ClaimCoherenceReport` / `apply_claim_coherence` | 为 claim verification 增加可选依赖图一致性检查：当父 claim 缺失或 unsupported 时，被判 supported 的子 claim 会降级为 insufficient evidence；`run_verification_loop(..., enforce_claim_coherence=True)` 会把 coherence report 写入 trace。 |
| `CitationRecord` / `CitationVerifier` / `extract_citation_references` | 增加无依赖 citation-integrity route：在调用方提供 citation catalog 时校验 bracket ref、DOI、arXiv id、URL、author/year、title 和 year metadata，避免引用字段漂移在进入更宽的 retrieval/groundedness 之前被放过。 |
| `ClaimTriple` / `RuleBasedTripleExtractor` / `RegexTripleExtractor` / `LookupTripleExtractor` / `CompositeTripleExtractor` / `TripleEvidenceVerifier` | 增加无依赖 subject-predicate-object 审计路径：简单规则、可配置 regex 模板、离线外部预测 lookup、组合 extractor 或调用方 metadata 生成 triple，再用可通过 source、metadata、claim id 或 subject anchor 绑定的证据片段逐槽检查 subject、predicate 和 object 覆盖，然后才输出更严格的 supported 结果。 |
| `CounterfactualProbe` / `CounterfactualProbeGenerator` / `CounterfactualVerificationAuditor` / `CounterfactualVerificationReport` | 对任意本地 `Verifier` 运行原始 claim 与 counterfactual 扰动 claim 的成对审计；可消费手写 probes，也可从 claim metadata、entity replacement、数字、年份和否定扰动自动生成 bounded probes，报告 expected-status accuracy、flip success、false invariance 和 per-probe failure reason，用于检验结构化事实、world-model、retrieval 或外部 verifier 是否真的对关键实体/时间/数量/逻辑扰动敏感。 |
| `RetrievalActionExecutor` / `InMemoryRetriever` / `SQLiteFTSRetriever` / `HTTPJSONRetriever` / `ProvenanceFilteredRetriever` | 为 unsupported claim 的取证流程提供无依赖 retrieval executor shell，覆盖本地片段、SQLite FTS 和调用方提供的 HTTP JSON search service；retrieval adapter 异常会 fail closed 成结构化 `ActionResult`，provenance filter 可在 hit 进入 verifier evidence 前要求来源、来源前缀、分数下限、metadata 标签和每来源上限。 |
| `CalculatorVerifier` | 提供无依赖确定性计算器 verifier，用于结构化算术 claim、简单符号等式，以及从有限算术文本中抽取出的 calculation metadata。 |
| `QuestionAnswerVerifier` | 提供无依赖结构化 QA/领域状态 verifier adapter，用于精确问题与候选答案事实。 |
| `StructuredFactVerifier` | 用结构化 subject-predicate-object facts 校验自然语言 claim：支持 KG 覆盖内常见改写、别名、多对象列表匹配和可注入 triple extractor；已知 subject/predicate 但 object 不匹配则 refuted。 |
| `StructuredStateVerifier` / `StateCheck` | 提供无依赖结构化状态与业务规则 verifier，用于数据库、策略和领域状态校验。 |
| `SQLiteStateSource` / `SQLiteStateQuery` | 将只读 SQLite 查询结果加载为 verifier 可消费的结构化状态，不增加非标准库依赖。 |
| `ToolOutputStateSource` / `ToolOutputMapping` | 将本地工具或 action 执行输出映射成结构化 verifier state，用于工具调用后的校验。 |
| `RuleBasedWorldModelAdapter` / `WorldModelRule` | 用显式 action match、结构化 state precondition、状态更新和 fail-closed no-match metadata 执行可审计、无依赖的 domain/world-model transition rule。 |
| `EnsembleWorldModelAdapter` | 聚合多个 world-model adapter，按一致率降低预测置信度，并在共识低于 `min_agreement` 时让 state-transition check fail closed。 |
| `WorldModelReference` / `WorldModelView` | 序列化 state-transition verification 使用的 reference world-model contract 和逐 claim view，包括被观察路径、状态指纹和假设。 |
| `StateTransitionVerifier` / `StateTransitionCheck` | 通过 world-model adapter 预测 action 后的下一状态，再校验结构化 postcondition；`min_prediction_confidence` 可在预测置信度不足时 fail closed，verifier metadata 会包含 `world_model_reference`、`world_model_view` 和 refuted postcondition 的 `world_model_conflict` 摘要。 |
| `CachedVerifier` / `CachedRetriever` / `CachedStateSource` | 为重复 verifier、retrieval 和 state-source 调用提供 request-scoped 内存缓存与 hit/miss 统计。 |
| `CompositeVerifier` / `RoutedVerifier` / `default_routed_verifier` | 组合确定性工具与词面、citation、检索、数据库、triple-evidence 或 world-model verifier；路由可依据 claim metadata、context、feature flags 或文本模式，可限制 route fanout，并记录匹配原因。默认路由栈会先跑确定性工具；如果调用方提供 citation catalog，会先做引用完整性检查；再对敏感事实 claim 进行严格 triple audit，最后用 lexical groundedness 兜底。 |
| `GroundednessVerifier` / `ClaimExtractor` | 抽取 claim metadata，并用词面证据片段和显式反证检查 claim，不增加核心依赖。 |
| `SelfConsistencyVerifier` | 用调用方提供的 sampled responses 对 claim 做 FactSelfCheck 风格支持/反证率检查，不增加模型或检索依赖。 |
| `sqlite_state_control_demo.py` | 演示 SQLite 结构化状态校验如何进入最终 `ProductTrace` 和 dry-run action。 |
| `state_transition_control_demo.py` | 演示 world-model 下一状态预测和结构化 postcondition 校验如何进入最终 `ProductTrace`。 |
| `production_tool_loop_demo.py` | 演示本地 production-like 闭环：SQLite 前置校验、受 guard 约束的有副作用本地 `execute_tool`、可选 JSON/SQLite idempotency ledger、工具输出状态映射、工具后校验、action audit metadata 和 trace route summary。 |
| `eval_verifier_ensemble.py` | 评估校准内部诊断与 retrieval/verifier 抑制误报、补充反证检出的组合策略，可选 staged verifier gating 和 triple-evidence route audit，记录 route 级成本指标，并可用 JSONL sidecar 保存逐 claim verifier 输出。 |
| `eval_counterfactual_verification.py` | 用 JSON/JSONL counterfactual probe 或 claim 生成的 counterfactual probes 评估任意本地 verifier 的扰动敏感性，当前支持 `in_memory` 和 `structured_fact` 本地入口，输出 `false_invariance_rate`、expected-status accuracy、生成 probe 数量和失败样例。 |
| `build_verifier_signal_score_dump.py` | 将 verifier verified-record JSONL sidecar 转成标准 score-dump 列，例如 `verifier_refuted`、`verifier_uncertainty`、`selfcheck_refute_rate`、`world_model_disagreement`、`world_model_conflict` 和 `world_model_conflict_delta`，同时支持 state-transition prediction/conflict metadata 和直接 world-model ensemble agreement metadata，让外部证据进入 calibrated geometry fusion。 |
| `build_selfcheck_signal_score_dump.py` | 将对齐的 sampled responses 直接转成 `selfcheck_support_rate`、`selfcheck_refute_rate`、`selfcheck_disagreement` 等 score-dump 列，让自一致性证据不经过 verifier sidecar 也能单独校准或融合。 |
| `export_inside_diagnostics_samples.py` | 从 `eval_truthfulqa.py --inside-diagnostics-cache` 文件恢复 sampled response 文本，导出 selfcheck samples payload，并写入 source/cache/output manifest provenance。 |
| `plan_selfcheck_sample_collection.py` | 在 direct selfcheck signal fusion 前做样本收集 preflight，输出缺样本记录、总 sample deficit、sample-quality gate projection 和 INSIDE cache export / fusion replay 重跑命令；`run_selfcheck_signal_fusion_workflow.py` 默认把这些 plan 写入 workflow report 和 artifact manifest。 |
| `build_text_baseline_score_dump.py` | 给带 statement metadata 的 score dump 追加无依赖 text/length 红线 baseline，例如 answer/claim 长度、词面重叠、否定和数字计数，确保新 detector 先和廉价词面控制项对比。 |
| `fetch_wikidata_reference_docs.py` | 将小型 Wikidata CC0 SPARQL 结果物化为 JSONL source docs，用于外部 retrieval corpus ingestion，支持 country capitals、模板可消费的 country core facts，以及 triple-extraction fixture matrix 可消费的 organization/product facts，并默认过滤 QID-only label。 |
| `build_external_retrieval_corpus.py` | 将调用方提供的 JSON/JSONL/text 来源文件规范化为显式 `external_evidence_candidate` retrieval corpus，并拒绝 document metadata 中的 score label、claim id 或 score-dump row link。 |
| `audit_retrieval_corpus_provenance.py` | 对 statement-bearing score dump 和 retrieval corpus 做 provenance 审计，区分外部 grounding 候选、受控数据集基线和 answer-echo/oracle-risk stress corpus。 |
| `build_wikidata_qa_corpus.py` | 将 Wikidata `P36` 或多属性模板结构化事实转换为无 label 的 `QuestionAnswerVerifier` corpus，用于 `retrieval_structured_qa` route smoke test。 |
| `run_wikidata_structured_qa_route_workflow.py` | 从 Wikidata QA corpus 构建平衡 covered-facts score dump，可运行 structured QA route 或自然语言 structured-fact route，可将 structured-fact claim 展开为 paraphrase robustness 变体，并写出带 manifest 的属性级校正证据。 |
| `analyze_retrieval_route_gaps.py` | 读取 verifier verified-record JSONL sidecar，按检索覆盖、最终状态、gap bucket、命中来源和样例解释 blocked retrieval route。 |
| `run_verifier_signal_fusion_workflow.py` | 端到端运行无模型本地证据闭环：retrieval/selfcheck fixture、可选 retrieval provenance filtering、verifier sidecar、verifier-signal score dump、geometry-fusion report、可部署 geometry artifact 和 manifest verification。 |
| `run_world_model_signal_calibration_workflow.py` | 端到端运行确定性 world-model 校正闭环：state-transition fixture、可选受控 world-model ensemble disagreement、world-model verifier sidecar、verifier-signal score dump、score/geometry fusion report、基于 trace-gap / conflict calibration evidence 的 release gate、嵌套 manifest verification 和可选本地 registry record。 |
| `run_selfcheck_signal_fusion_workflow.py` | 端到端运行无模型 direct selfcheck signal 闭环：sampled responses、自一致性 score dump、sample-quality gate、score ensemble report、可选 geometry-by-selfcheck fusion artifact 和 manifest verification。 |
| `eval_verifier_stability.py` | 对 verifier-ensemble report 做多 seed split-conformal 重放，总结 verified risk 和 route-selection 稳定性，指纹化 verifier 输入，并可选登记 post-hoc report。 |
| `eval_score_ensemble.py` | 对已保存诊断 score dump 执行方向感知 rank fusion 与 geometry-by-uncertainty interaction fusion benchmark，并可在一次运行中保存可部署的 `RankScoreFusionArtifact` 或 `GeometryScoreFusionArtifact`。 |
| `select_fusion_signals_from_ablation.py` | 将已保存 fusion ablation matrix 转成按 run 选择的 signal-selection report，让 trajectory 等可选信号保持条件式启用，而不是全局默认。 |
| `build_selected_fusion_artifacts.py` | 基于 signal-selection report 和匹配 score dump 生成 per-run `RankScoreFusionArtifact`，把条件式信号选择落成可校准加载的 artifact。 |
| `eval_frontier_stability.py` | 对已保存 frontier score dump 做多 seed split-conformal 重放，总结最佳信号稳定性，指纹化 source score records，并可选登记 post-hoc stability report。 |
| `run_calibrated_observability_workflow.py` | 运行或复用 TruthfulQA score dump，执行 conformal layer/score 校准，可从 layer-band selector report 派生 `--sweep-layers`，写入嵌套 artifact manifest 和 evidence-bundle summary，透传可选 TruthfulQA cache 路径，并可选把 calibrated-observability 闭环登记到本地 registry。 |
| `run_truthfulqa_frontier_workflow.py` | 执行多模型/多尺度 TruthfulQA frontier workflow：批量运行 Qwen/SmolLM2 风格 l20/l80 calibrated-observability cells，支持 per-cell cache root 复用 l80/多 seed 证据，可按 cell 消费 layer-band selector report 做 dense rerun，生成跨 cell rank-fusion ensemble report 和顶层 manifest。 |
| `run_pre_generation_probe_workflow.py` | 运行或复用 pre-generation hidden-state records，执行 record-layer probe sweep，保存推荐 `AttentionSoftTargetProbeArtifact` 与 split-conformal calibration，并写入紧凑 workflow report 和 artifact manifest。 |
| `compare_pre_generation_probe_workflows.py` | 聚合多个 run/model 的紧凑 pre-generation probe workflow report，应用 ready/manifest/model-count/AUROC gate，并输出 leaderboard 风格 comparison report 和可选 artifact manifest。 |
| `compare_manifold_distances.py` | 从已保存的 `TruthManifold` artifact 或 `eval_truthfulqa.py` layer-stats cache 生成 Gaussian 2-Wasserstein/Bures 距离矩阵，用于离线检查 layer/checkpoint drift。 |
| `eval_intrinsic_dimension.py` | 从已保存 warmup checkpoint 生成 TwoNN intrinsic-dimension profile，输出 peak-layer 和 rise-then-fall shape summary，不重新加载模型权重。 |
| `compare_intrinsic_dimension_layers.py` | 将 intrinsic-dimension peak layer 与已保存 layer/score sweep AUROC 排名对齐，输出 top-k 命中、rank、AUROC regret 和层距离，用于低成本 layer-selection 验证。 |
| `compare_layer_band_selectors.py` | 将 intrinsic-dimension 和 spectrum 派生的 layer band 与 calibrated sweep 排名对齐；当候选 band 覆盖最佳层时，输出可复用的降成本 sweep 先验。 |
| `audit_layer_band_replication.py` | 在 layer-band selector 成为默认 preset 前做复现 gate：要求 matched run、模型族多样性、dense ranked-layer grid、best-layer 命中率、AUROC regret 和候选层成本比例达标。 |
| `training_telemetry_sanity.py` | 执行确定性的 synthetic clean-vs-corrupt training telemetry sanity check，用 distance-to-baseline 增长和 effective-rank collapse 做 gate。 |
| `training_telemetry_tiny_finetune.py` | 执行纯 PyTorch tiny clean-vs-duplicate fine-tune 对照，检查 representation-rank telemetry 是否早于 eval-loss 退化分离。 |
| `model_collapse_early_warning.py` | 执行确定性的 pseudo-label self-training loop，检查表征多样性 telemetry 是否早于可见质量退化发出预警。 |
| `trajectory_convergence_sanity.py` | 执行确定性的 synthetic generation-trajectory sanity check，报告 convergence diagnostics 与质量代理之间的 Spearman/AUROC 相关性。 |
| `eval_trajectory_truthfulqa.py` | 将带 statement metadata 的 TruthfulQA score dump 通过 causal LM 重放，抽取 answer-token prediction 位置上的 forced-answer hidden-state trajectory，并报告 trajectory/NLL 与 true/false label 的相关和 AUROC；`--offline` 提供无下载确定性 smoke fixture。 |
| `eval_pre_generation_probe.py` | 从本地 JSON/JSONL token hidden-state records 和 soft error-rate targets 训练/评估 torch-only 生成前 attention risk probe，可直接消费 `eval_truthfulqa.py --dump-pre-generation-probe-records` 导出的单层或多层 records；支持 record-layer sweep、候选排序，并可保存最佳 `AttentionSoftTargetProbeArtifact` 和 split-conformal `CalibrationArtifact`。 |
| `eval_pre_generation_text_baselines.py` | 直接基于 pre-generation records 评估低成本 question/answer 文本红线 baseline，按特征选择最佳方向，用长度、重叠、否定、数字计数控制检查 probe 结论。 |
| `compare_trajectory_sweeps.py` | 对多个 forced-answer trajectory sweep report 做对比，并在 trajectory 信号进入 release evidence 前应用 fail-closed gate。 |
| `concept_registry_smoke.py` | 保存两个 synthetic `ConceptArtifact`，登记到本地 registry，把两个 probe 同时挂到一个 toy model，并写出带 manifest 的多 concept 诊断报告。 |
| `refresh_verifier_route_artifacts.py` | 从已保存 score dump、claims 和本地 verifier corpus 重新生成新 schema verifier-route report，不重跑模型 forward。 |
| `compare_verifier_routes.py` | 将已保存 verifier-ensemble report 聚合为成本感知 route 排行榜、Pareto frontier 候选、分 route promotion decision、分 route 控制收益指标和可选 tail/cache/staged-verification route 质量门槛。 |
| `run_adapter_promotion_workflow.py` | 执行 fail-closed adapter promotion workflow：route comparison、`promotion_decision=promote` 和可选 registry-backed 性能基线门槛。 |
| `run_adapter_promotion_registry_workflow.py` | 一次性执行 route promotion、写 manifest、递归验证 manifest，并把 promoted route baseline 注册到本地 registry。 |
| `compare_route_baselines.py` | 按 manifest 验证状态、route 质量、误支持/反证率、尾延迟、retrieval 成本、可选 covered-fact 分 property gate、可选 retrieval provenance-filter gate 和可选 answer-echo retrieval stress-control gate 比较已注册 verifier-route promotion baseline 或 covered-facts route-summary manifest。 |
| `compare_external_evidence_baselines.py` | 将 route-baseline promotion、answer-echo stress-control 证据、text/length redline score-ensemble report 和可选 covered-facts structured route gate 合并成一个 fail-closed 外部证据比较 artifact，并可写入 manifest、递归校验、登记成 `report:*:*`。 |
| `run_covered_facts_external_evidence_workflow.py` | 登记一个或多个已保存的 Wikidata covered-facts route manifest，启用 `--require-covered-facts-route` 运行 external-evidence comparator，写出已验证 manifest，并可把 comparator report 登记成 release gate handoff。 |
| `run_adapter_family_matrix.py` | 构建确定性的 structured QA、structured-state、state-transition、可选 retrieval-groundedness、可选 retrieval-structured-QA 和可选 strict triple-evidence fixtures，并在一个本地矩阵里比较 promotion 指标。 |
| `run_local_retrieval_route_workflow.py` | 从 score dump 和本地 corpus 构建 retrieval evidence，可选应用 retrieval provenance filter，promote retrieval route，指纹化全部源 artifact，记录运行 profile，可选挂载 answer-echo stress-control evidence，可选使用持久化 SQLite FTS/claims fixture/verifier trace 缓存，并可选注册 route baseline。 |
| `run_cache_profile_matrix.py` | 跨 layer、batch size、capture mode 和 TruthManifold covariance mode 执行同机 profile sweep，并输出矩阵级性能 promotion decision 和每个 cell 的 AUROC quality signals。 |
| `run_cache_worker_sweep.py` | 用多个 worker count 运行同一 cache-profile matrix，并按 wall-clock 推荐最快的已 promoted worker count。 |
| `run_inside_sampling_profile.py` | 比较 fixed、adaptive 和 self-check-bounded INSIDE sampling，输出 sample-count 与 `inside_generation` 成本证据，供 release gate 使用。 |
| `run_inside_trigger_budget_sweep.py` | 比较多个 triggered INSIDE budget，输出生成样本数、`inside_generation`、参考全量比例、inside-score AUROC，以及成本优先 / 质量折中的推荐。 |
| `recommend_runtime_config.py` | 将 promoted matrix/worker-sweep report 与可选 INSIDE sampling / trigger-budget sweep / score-ensemble / selected-fusion-artifact 证据转成可执行 runtime recommendation：layer、batch size、covariance mode/rank、token budget、prefix KV、worker count、sampling flags、derived-sweep flags、通过 gate 的最佳 AUROC quality signal、covariance tradeoff details 和 cache-tuning 建议。 |
| `run_performance_baseline_workflow.py` | 将 cache matrix、可选 worker sweep、可选 INSIDE profile / trigger-budget / score-ensemble / selected-fusion-artifact 证据、runtime recommendation、artifact manifest、performance evidence summary 和可选递归 manifest verification 打包成可注册 performance baseline。 |
| `run_product_runtime_baseline.py` | 聚合已保存的 `ProductTrace` JSON，输出请求级 runtime baseline：phase、cache、verifier route、route-budget exhaustion、retrieval 使用率、staged-verification 节省量、verification-plan 覆盖率/route/tool-payload 计数、action-execution planned/result 对齐情况、action-audit 错误/缺失/畸形 action 率，包括未覆盖 planned retrieval query 的指标、external-evidence baseline-comparison 覆盖率/status/route/text-redline handoff metadata、pre-generation probe comparison/redline handoff metadata、verification-budget exhaustion 摘要、claim-risk localization 摘要、triple/slot-audit 覆盖率、final-answer 覆盖率/status/action 汇总、promotion-contract covered-fact property scope 与 per-property quality rollups、runtime-profile context、带 phase/route/triple-audit 热点和 cache/retrieval/staging/profile 建议的 `optimization` 块、`max_verifier_route_attempts` 等候选 control-default hints、可选 `ProductRuntimeBudgetPolicy` gate、可复用推荐 policy artifact、逐 trace record 的可选 JSONL sidecar 存储、重复 budget sweep 可复用的 trace-record cache，以及大 trace 集可选并行扫描。 |
| `run_product_feedback_report.py` | 将已保存的 `ProductTrace` JSON 与 `ProductFeedbackRecord` JSONL 结果合并，报告 accepted-but-wrong、retrieved-failure、retrieved-but-still-unsupported、abstain-false-positive、final-answer answered-but-wrong、final-answer false-block、反馈匹配/未匹配数量、可选质量门禁、manifest provenance 和可选 registry metadata。 |
| `recommend_control_policy_from_feedback.py` | 将一个或多个产品反馈报告转成候选 `ControlPolicyConfig` 和 runtime control-default 建议，用 accepted-but-wrong、retrieval-failure、abstain-false-positive、final-answer answered-but-wrong 和 final-answer false-block 率完成事后反馈到策略建议的闭环，不需要重训或新增依赖。 |
| `audit_feedback_policy_replay.py` | 用历史反馈标签审计候选策略，报告 counterfactual safety coverage、claim metadata 缺口、残余安全问题和过度拦截缓解情况，并明确不等同于精确重跑 controller。 |
| `run_feedback_policy_workflow.py` | 一条命令串起完整事后反馈到策略闭环：feedback join report、候选控制策略推荐、内联候选 `ControlPolicyConfig`、replay audit、顶层 manifest 和可选 registry record；注册后的 `report:<name>:<version>` 可作为 release-candidate gate 输入。 |
| `compare_product_runtime_baselines.py` | 将当前 product runtime baseline 与文件或 registry 中的基线比较，对 latency、route cost、retrieval 使用、cache 复用、verifier skip rate、promotion-contract / triple-extraction evidence 覆盖与质量漂移、pre-generation probe comparison / redline handoff 漂移、covered-fact per-property rollup drift、product-trace action-audit / action-execution gate rate drift、trace 级 triple/slot-audit 覆盖、trace 数量和可选复用 runtime budget policy 输出 fail-closed drift gate。 |
| `build_product_trace_corpus.py` | 校验已保存的 ProductTrace JSON/JSONL，可选脱敏文本字段包括 final-answer 文本/证据，写出 replay-ready 标准化 trace 和 runtime-pair index，可复用 per-source 校验/脱敏缓存，并登记带 final-answer status/action 计数和 manifest 的 trace corpus。 |
| `run_product_trace_replay_workflow.py` | 端到端执行 raw trace handoff：脱敏 trace corpus、runtime-pair index、产品 runtime baseline、可选 action-audit release gate（缺失 retrieval action、未覆盖 planned retrieval query、畸形或异常 action payload）、可选 action-execution alignment gate（缺失、异常或 request-id 不匹配的 action result）、selector replay、可选 product-runtime drift/policy/promotion-evidence/pre-generation-probe drift、covered-fact property drift 与 product-trace action-gate drift gate、顶层 runtime `optimization` 摘要、可选推荐 runtime policy artifact、递归 manifest、可选 manifest verification、可选 manifest fingerprint cache 复用、可选 whole-corpus cache 复用、可选 corpus source-cache 复用、可选 runtime trace-record cache 复用、可选 runtime trace-scan 并行、可选 selector trace-input cache 复用、phase timing/cache summary 和可注册 workflow report。 |
| `run_product_runtime_profile_sweep.py` | 在 `latency`、`balanced`、`audit` 和请求级 `auto` selection modes 下运行确定性 calibrated-control demo 场景，写入或复用 trace，携带 promotion contract/profile 生效后的 control-default 汇总如 `max_verifier_route_attempts`，可缓存每个 profile 的 trace records 以重复调 budget/SLO，生成每个 mode 的 baseline，应用可选聚合 SLO 门禁，并推荐最低成本的未阻断 mode。 |
| `run_release_efficiency_report.py` | 将 product runtime profile sweep 和可选 quality/release reports 汇总成可注册 efficiency handoff，集中展示每个 profile 的 runtime、verifier、cache、trace 复用和 route fanout 证据。 |
| `run_runtime_profile_selector_tuning.py` | 通过同一套 SLO gate 比较多个 `RuntimeProfileSelectorPolicy` JSON 候选，运行 auto-profile sweep，并推荐成本最低的 promoted selector。 |
| `run_runtime_profile_selector_replay.py` | 在已保存的 `ProductTrace` 上回放多个 `RuntimeProfileSelectorPolicy` JSON 候选，不重跑 demo 即可通过 trace scan 或 corpus runtime-pair index 估算 profile 成本、分布、配对 observed runtime 和 selected-vs-original runtime delta，可缓存最小 trace replay input 以便重复策略 sweep，并登记成本最低的 promoted selector。 |
| `run_adapter_readiness_workflow.py` | 将 adapter-family 质量门槛，包括可选 retrieval 和 strict triple-evidence audit family、cache-profile 性能门槛和可选 INSIDE sampling / trigger-budget gate 合并为最终 readiness decision、runtime recommendation 和可注册 manifest。 |
| `run_adapter_readiness_registry_workflow.py` | 运行 readiness gate，包括可选 retrieval / triple-evidence adapter-family 证据，并在 readiness promote 后把已验证 manifest 注册成本地可复用 promotion baseline。 |
| `compare_readiness_baselines.py` | 按 manifest 验证状态、最佳 AUROC quality signal、runtime cost、可选 covariance `maha_last` 质量跌幅 gate 和 INSIDE profile / trigger-budget 成本证据比较已注册 readiness baseline，并推荐一个可部署 baseline。 |
| `compare_release_candidates.py` | 将已注册 readiness baseline、route baseline、可选 required route-baseline/robustness evidence、performance baseline、product-trace-replay workflow 文件或 registry key、selfcheck-signal-fusion workflow 文件或 registry key、world-model-signal workflow 文件或 registry key、feedback-policy workflow 文件或 registry key、selector-replay、product-runtime-drift、release-efficiency、external-evidence baseline comparison 文件或 registry key、pre-generation probe comparison 文件或 registry key、adapter-family 证据、triple-extraction fixture matrix 证据和 counterfactual verifier audit 证据合成一个 fail-closed release candidate，输出 runtime flags、verifier route、covered-fact property 覆盖范围、质量、runtime cost、covariance 质量跌幅 gate、performance evidence bundle/cache gate、可选显式 prior baseline 性能趋势 gate、selfcheck 样本质量/融合 gate、world-model trace-gap/conflict-calibration gate、pre-generation 多模型/红线 evidence gate、product-trace action-audit 与 action-execution gate、audit-route budget、retrieval provenance-filter gate、retrieval stress-control gate、strict-audit adapter-family profile、可选 learned triple extractor external-prediction 数量/语料/F1 gate、`strict_structured_fact` / `frontier_audit` 等命名 release-policy profile，其中 `frontier_audit` 默认启用已注册 covered-facts external-evidence handoff、已注册 triple-extraction fixture matrix、external-prediction triple-extraction 最小门槛、`--require-product-runtime-drift-promotion-evidence`、`--require-product-runtime-drift-pre-generation-evidence`、`--require-product-runtime-drift-triple-audit-evidence`、`--require-product-runtime-drift-covered-fact-property-evidence`、`--require-product-runtime-drift-action-gate-evidence`、`--require-product-trace-action-audit-gate` 和 `--require-product-trace-action-execution-gate`、已校验 feedback-policy config / control defaults provenance、action 级与 final-answer 反馈安全指标、包含 promotion/pre-generation/triple-audit/covered-fact property/action-gate evidence summary 的 runtime drift provenance、release-efficiency profile handoff，以及 latency/balanced/audit runtime profiles。 |
| `run_release_candidate_registry_workflow.py` | 执行 release-candidate gate，写入覆盖 candidate report、选中 readiness/route/performance/product-trace-replay/selfcheck-signal-fusion/world-model-signal/feedback-policy/selector/runtime-drift/release-efficiency manifests、可选 product-trace action-audit 和 action-execution gate report、可由文件或 registry key 提供的 external-evidence comparison 与 pre-generation probe comparison report，以及可选 required route / adapter-family / triple-extraction fixture-matrix / counterfactual-verification report 的 manifest，支持 `strict_structured_fact` / `frontier_audit` 等命名 release policy profile，可显式要求 canonical + paraphrase `structured_fact` robustness evidence，并对 `frontier_audit` 使用 path-aware 的已注册 external-evidence 与 triple-extraction matrix handoff 默认值，递归验证后登记带 runtime-profile、covariance 质量 gate、performance-baseline、performance evidence bundle readiness/cost/cache/trend、product-trace-replay/action-audit/action-execution、selfcheck signal-fusion workflow 状态/report/record/样本质量 metadata、world-model signal workflow gate/report/record metadata、pre-generation probe comparison 来源/record/redline metadata、feedback-policy workflow 状态/report/record/候选策略 metadata、selector-replay、runtime-drift、release-efficiency、external-evidence comparison 来源/record metadata、route-budget、retrieval provenance-filter policy metadata、release-policy profile、structured-fact robustness property 覆盖范围、adapter-family metadata、跨语料 triple-extraction evidence、可选 external-prediction gate metadata 和 counterfactual verifier-audit pass-rate/false-invariance metadata 的最终候选；compare、manifest build、promotion verification 和 workflow report write 输出 phase timing，compare/manifest/promotion 共享 fingerprint 与 JSON artifact cache，`--fingerprint-cache` 可在重复 release check 间持久化复用 fingerprint，`--artifact-json-cache` 可持久化复用已解析 JSON artifact，`--manifest-fingerprint-workers` 可为本地 release gate 开启有界并行 artifact fingerprint。 |
| `run_manifest_fingerprint_worker_sweep.py` | 对一个或多个已保存 manifest 用多个有界 fingerprint worker 数重复执行 artifact-manifest verification，记录每档 timing/cache summary，为本地 release check 推荐最快通过的 worker 数，并可登记为 `report:*:*` artifact。 |
| `run_release_gate_overhead_baseline.py` | 将一个或多个 release-candidate registry workflow JSON 汇总成 release-gate overhead baseline，输出总耗时/phase timing、artifact fingerprint/JSON cache 命中率、热点建议、可选 timing/cache gate，并可登记 `report:*:*` registry record。 |
| `export_product_promotion_contract.py` | 从 promoted release candidate 导出紧凑的可部署 `ProductPromotionContract` JSON，包含 verifier-route covered-fact property scope、product-trace replay action-audit/action-execution gate provenance、runtime-drift promotion/pre-generation/triple-audit/action-gate evidence、selfcheck-signal-fusion provenance、world-model signal release-gate provenance、external-evidence baseline-comparison provenance、pre-generation probe comparison/redline provenance、triple-extraction fixture-matrix provenance 和可选反馈派生 `ControlPolicyConfig`，写入 manifest，并可登记 `product_promotion_contract:*:*` handoff artifact。 |
| `build_domain_state_fixture.py` | 构建确定性的订单履约 score/claim/state fixture，并可输出 SQLite state-source spec，用于结构化状态 verifier benchmark。 |
| `build_transition_fixture.py` | 构建确定性的订单预留 state-transition fixture，用于 world-model/postcondition verifier benchmark，并可生成受控或 policy-replay world-model ensemble disagreement。 |
| `build_truthfulqa_corpus.py` | 构建本地 TruthfulQA correct-answer corpus，用于可复现的非 oracle retrieval baseline。 |
| `build_retrieval_stress_corpus.py` | 从带 statement metadata 的 score dump 构建 answer-echo retrieval stress corpus，用来暴露检索证据来自待审答案本身时的自证失败模式。 |
| `fetch_wikidata_reference_docs.py` | 拉取或重放 Wikidata country-capital / country-core-fact / organization-product-core-fact SPARQL 结果，输出外部证据 smoke gate 可用的 JSONL source docs；core-facts route audit 仍阻止 lexical route promotion。 |
| `build_external_retrieval_corpus.py` | 从本地来源文件构建显式外部候选 retrieval corpus，再进入 provenance audit 和本地 retrieval fixture 构建。 |
| `build_wikidata_qa_corpus.py` | 将 Wikidata fact document 转成模板驱动的 structured QA corpus，供 `QuestionAnswerVerifier` / `retrieval_structured_qa` 消费。 |
| `run_wikidata_structured_qa_route_workflow.py` | 执行 covered-facts structured QA 或 structured-fact route workflow，可展开 structured-fact paraphrase robustness 变体，并记录 QA corpus 覆盖属性的支持/反证指标。 |
| `analyze_retrieval_route_gaps.py` | 基于 verified-record sidecar 解释 blocked retrieval route 的覆盖、状态、gap bucket、证据来源和具体样例。 |
| `build_evidence_fixture.py` | 从带 statement 的 score dump 和本地 JSON/JSONL/text 文档库构建非 oracle claim/evidence fixture，可在 hit 进入 verifier evidence 前应用来源、metadata 和分数 provenance filter。 |
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
make perf-check     # deterministic profile/cache/worker/registry/concept/ProductTrace smokes; no model load
make release-check  # also builds the package
```

`make perf-check` validates gate mechanics, not machine speed. For runtime
claims, generate same-machine profile evidence under `/tmp/eigentruth-*` first
and commit only the explicitly promoted report/manifest/registry bundle needed
for a maintained baseline.

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

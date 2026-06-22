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

# Fast pipeline self-check (tiny model, bundled statements, no dataset download):
python benchmarks/eval_truthfulqa.py --model sshleifer/tiny-gpt2 --offline

# Optional multi-response INSIDE proxy (slower: samples K continuations per statement):
python benchmarks/eval_truthfulqa.py --model sshleifer/tiny-gpt2 --offline \
  --batch-size 4 --inside-samples 3 --inside-batch-size 2 --inside-max-new-tokens 6
```

Use `--json results.json` to save structured output (config + AUROC per signal) for
the record. Use `--subspace-rank` to tune `TruthSubspace` residual scoring; fitting
requires at least two factual warmup states, and rank is clipped to the available
warmup states and hidden dimension. Increase `--batch-size` to batch forced-answer
forward passes, and increase `--inside-batch-size` to batch sampled INSIDE prompts;
higher values improve throughput but raise memory use.

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

At the default layer −8:

| signal | AUROC |
|---|---|
| `truth_proj` | **0.723** |
| `maha_last` | 0.622 |
| `disp_euclid` | 0.484 |
| `disp_hse` | 0.474 |
| `nll_answer` | 0.411 |

Layer sweep (free: one forward pass returns all hidden states) — `truth_proj` AUROC by
layer: peaks at **0.753 (layer −6)**, stays above 0.72 across the −8…−2 band, and collapses
at the last layer (0.546). `maha_last` peaks at 0.638 (layer −4) and also collapses at −1.

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

## 说明

`eval_truthfulqa.py` 在 TruthfulQA 上以确定性、无需 LLM 裁判、单次前向的方式，检验隐状态
几何能否分离真/假陈述。真值流形仅用留出题目的正确答案构建（与评测题目不重叠，无泄漏），
正类为错误答案（幻觉）。逐信号报告 AUROC（0.5 为随机，1.0 为完美分离）。

三个关键对比：`maha_last` vs `nll_answer`（几何是否优于困惑度基线）、`disp_hse` vs
`disp_euclid`（双曲投影是否真的有用）、`eigenscore` 是否补充内部状态谱分散度信号。
结果强依赖目标层，应扫层。陈述级打分是开放生成幻觉的代理，小模型 + 数百样本的 AUROC
置信区间较宽，结论需对照完整已发表方法。

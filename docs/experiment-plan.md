# EigenTruth Experiment Plan

A gated experiment program: each borrowed idea (math, physics, frontier-problem) gets one
small experiment with an explicit accept/reject criterion. Ideas that pass become tool
features; ideas that fail are documented as negative results. The goal is a complete
representation-observability toolkit spanning **training and inference**.

逐个分析、逐个实验、按判据决定进库或砍掉。负结果同样写入文档——这是工具的诚信资产。

## Ground rules

- Every experiment must run on **CPU, 8 GB RAM** (gpt2 / tiny models). Anything needing a
  bigger model is run as a mechanism check here and flagged "replicate on larger hardware".
- One experiment → one focused PR. Tests + lint green before merge.
- Each experiment states: **Question / Method / Accept criterion / Deliverable / Cost.**
- Priority order = information value ÷ cost. Phase 1 experiments are independent and can
  be reordered freely.

## Phase 0 — Finish in-flight work

### E0. Linear direction vs Mahalanobis + layer sweep (real data)
- **Question:** Is the contrastive-direction projection (`truth_proj`, mass-mean probe)
  a stronger detector than `maha_last`? Which layer is best?
- **Method:** Code already in working tree (`benchmarks/eval_truthfulqa.py --sweep`).
  Run on gpt2, full TruthfulQA split as before.
- **Accept:** n/a (measurement, not a gate). Whatever wins becomes the documented default.
- **Deliverable:** Updated `benchmarks/README.md` results + default-signal/layer guidance;
  commit to the open eval-harness PR.
- **Cost:** ~5 min CPU. **Status: code ready, needs the run.**

## Phase 1 — Validate each math borrowing (independent, reorderable)

### E1. Conformal prediction → calibrated thresholds
- **Question:** Can split conformal turn raw scores (maha / truth_proj) into p-values with
  honest finite-sample coverage, replacing hand-picked thresholds?
- **Method:** Implement split-conformal calibration over a held-out calibration set; on the
  TruthfulQA eval split check empirical coverage at nominal 80/90/95%.
- **Accept:** |empirical − nominal| coverage ≤ 3% across the three levels.
- **Deliverable:** `eigentruth.eval.conformal` (or `Manifold.calibrate()`) + tests.
- **Cost:** low (pure CPU, ~50 lines + tests). **Highest value/cost in the plan.**

### E2. Random matrix theory → principled spectrum tools
- **Question:** Does the Marchenko–Pastur noise floor + eigenvalue shrinkage (Ledoit–Wolf
  style) beat our fixed relative ridge, and give a principled effective-rank signal?
- **Method:** (a) Plot manifold covariance spectrum vs MP bulk for gpt2 activations; count
  out-of-bulk spikes. (b) Swap fixed ridge for Ledoit–Wolf shrinkage; re-run E0 AUROC.
  (c) Synthetic collapse data: check spike-count/effective-rank monotonicity.
- **Accept:** AUROC not worse AND scale stability preserved; rank signal monotone under
  synthetic collapse.
- **Deliverable:** `Manifold.spectrum()` (eigvals, MP edges, n_spikes, eff_rank) +
  optional shrinkage mode.
- **Cost:** low-medium. **Status:** spectrum diagnostics foundation implemented
  (`TruthManifold.spectrum()` / `covariance_spectrum()`) and wired into
  `eval_truthfulqa.py --include-layer-spectra`; OAS-style `covariance_mode="shrinkage"`
  implemented; tiny offline covariance-mode matrix smoke covers
  `full/diag/low_rank/shrinkage`; real Qwen/SmolLM2 l80 cache-only covariance
  gate promotes `shrinkage` as the cross-model quality-preserving candidate.

### E3. Bures–Wasserstein distance → manifold-to-manifold metric
- **Question:** Is closed-form 2-Wasserstein between Gaussians the right metric for
  comparing manifolds (checkpoint diff, drift)?
- **Method:** Implement BW distance; unit-test metric properties on synthetic Gaussians;
  sanity-check on gpt2: distance matrix across layers should show adjacent-layer locality.
- **Accept:** metric axioms pass; layer-distance structure is coherent (adjacent < distant).
- **Deliverable:** `manifold_distance()` in core + tests. Foundation for E5/E8.
- **Cost:** low. **Status:** core closed-form Gaussian 2-Wasserstein/Bures
  implementation landed as `gaussian_wasserstein_distance()`,
  `manifold_distance()`, and `manifold_wasserstein_distance()` with synthetic
  metric-property tests. Cached l80 Qwen/SmolLM2 layer-stats sanity passes for
  both full and shrinkage covariance: adjacent monitored layers are closer than
  distant monitored layers and nearest-neighbor locality is 1.0 in all four
  reports. A denser layer matrix remains useful before relying on this as a
  fine-grained checkpoint-drift signal.

### E4. Intrinsic dimension → cheap layer-selection signal
- **Question:** Does the TwoNN intrinsic-dimension profile across layers reproduce the
  literature shape, and does it predict the best monitoring layer found in E0?
- **Method:** Implement TwoNN; compute ID per layer on gpt2 activations; compare ID
  profile against E0's per-layer AUROC; also score ID as a 6th detector signal.
- **Accept:** ID profile qualitatively matches literature (rise→fall) AND (ID-selected
  layer is within top-3 of E0 sweep OR ID signal AUROC > 0.55).
- **Deliverable:** `eigentruth.eval.intrinsic_dimension` + layer-selection heuristic doc.
- **Cost:** low-medium. **Status:** TwoNN core estimator and warmup-checkpoint
  profile helper landed. Cached l80 Qwen/SmolLM2 factual warmup profiles both
  show rise→fall over monitored layers with peak at -14. The saved l80 predictor
  comparison matches both ID-selected peak layers into the TruthfulQA `truth_proj`
  AUROC top-3, with mean AUROC regret 0.0199 and mean absolute layer gap 3.0.
  Treat this as accepted for cheap coarse layer-band selection, not exact
  best-layer selection; denser sweeps and a per-sample ID-derived signal remain
  useful follow-ups.

## Phase 2 — Tool-composition experiments (combine validated bricks)

### E5. Training telemetry callback (training-side axis)
- **Question:** Can streaming per-layer stats (norm, mean drift, eff-rank from E2, BW
  distance-to-init from E3) visibly distinguish a healthy fine-tune from a pathological one?
- **Method:** `RepTelemetryCallback` for HF Trainer; fine-tune a tiny model twice on CPU —
  clean data vs corrupted (heavy label noise / duplicated data); compare telemetry curves.
- **Accept:** at least one telemetry curve cleanly separates the two runs before eval loss does.
- **Deliverable:** `eigentruth.training` module + demo notebook/script + tests.
- **Cost:** medium. Depends on: E2 (rank), E3 (distance) preferred but not required.
  **Status:** `eigentruth.training` foundation landed as a dependency-free
  telemetry recorder plus `RepTelemetryCallback`, a transformers-free adapter
  with HF Trainer-compatible hook names. Synthetic clean-vs-corrupt hidden-state
  trajectories are separated by distance-to-baseline growth and effective-rank
  collapse. A pure PyTorch tiny clean-vs-duplicate fine-tune now shows
  effective-rank telemetry separating at epoch 1, before eval-loss degradation
  crosses the configured margin at epoch 5. This accepts the telemetry primitive,
  callback interface, and tiny fine-tune sanity gate; a real transformers Trainer
  end-to-end callback run remains a follow-up evidence item.

### E6. Model-collapse early warning (synthetic-data loop)
- **Question:** Does representation diversity (eff-rank / ID) decay monotonically when a
  model is iteratively trained on its own outputs, and earlier than visible quality loss?
- **Method:** tiny model; 3–5 generations of self-output fine-tuning; track E2/E4 signals.
- **Accept:** diversity signal decays monotonically across generations.
- **Deliverable:** collapse-detection demo + doc section. (Frontier problem: synthetic data.)
- **Cost:** medium. Depends on: E2 or E4.

### E7. Generation-trajectory convergence monitor (reasoning-direction seed)
- **Question:** During generation, does hidden-state trajectory convergence (step-to-step
  displacement decay; optional Koopman-style rate estimate) correlate with output
  confidence/quality?
- **Method:** gpt2, per-token last/mid-layer states over generations; correlate convergence
  metrics with answer NLL/entropy and TruthfulQA labels.
- **Accept:** |Spearman| > 0.3 with confidence, or AUROC > 0.55 as a detector signal.
- **Deliverable:** `TrajectoryMonitor` prototype. Flag: replicate on a reasoning model
  (R1-distill class) on larger hardware.
- **Cost:** medium.

### E8. Concept registry + multi-probe (platform glue)
- **Question:** engineering, not science — can multiple (manifold, direction) pairs be
  saved/versioned/loaded and monitored simultaneously with a clean API?
- **Method:** registry format (.pt + metadata), multi-probe attach, docs.
- **Accept:** API review + tests; example with two concepts monitored at once.
- **Deliverable:** `eigentruth.registry`. Foundation for BYO-concept use.
- **Cost:** medium. Independent.

## Phase 3 — Consolidation and release

### E9. Prune and rename
- Act on accumulated evidence: demote/remove hyperbolic HSE from the default path
  (current evidence: no lift over Euclidean); generalize naming
  (`Manifold`/`Direction`/`Probe`, keep `Truth*` aliases); reposition README as a
  representation-observability toolkit for training + inference.

### E10. Release 0.2.0 + honest writeup
- Package, publish, and write up every experiment **including negative results**.

## Decision log

| Exp | Date | Verdict | Evidence |
|-----|------|---------|----------|
| (hyperbolic HSE vs Euclidean) | 2026-06-08 | no lift (0.474 vs 0.484, gpt2 L-8) | `benchmarks/results_gpt2_l-8.json` |
| E0 | 2026-06-10 | **truth_proj wins**: 0.723 @L-8, peak 0.753 @L-6, beats maha (0.622/0.638) at every layer except -12; both collapse at L-1. Default guidance: contrastive direction, mid-late layers (-8…-4); maha as no-false-data fallback. | `benchmarks/results_gpt2_sweep.json` |
| E1 | 2026-06-10 | **ACCEPT**: empirical false-alarm tracks nominal within 1.3% at α∈{.05,.1,.2} for both maha_last and truth_proj (20 seeded splits). Power at α=0.2: truth_proj 46.9% vs maha 34.1%. Conformal thresholds replace hand-picked ones. | `benchmarks/results_conformal_*.json` |
| E2 | 2026-06-25 | **ACCEPT shrinkage**: spectrum diagnostics and OAS-style shrinkage mode landed; tiny offline matrix smoke passes; l80 cache-only covariance gate promotes `shrinkage` for both Qwen and SmolLM2. Qwen also accepts `low_rank_16`; SmolLM2 rejects `low_rank_16` at the 0.01 `maha_last` AUROC-drop gate; both reject `diag`. | `TruthManifold.spectrum()` / `covariance_spectrum()` / `covariance_shrinkage_intensity()` unit tests; `eval_truthfulqa.py --include-layer-spectra` tests; `artifacts/tiny_covariance_shrinkage_matrix/cache-profile-matrix-report.json`; `artifacts/truthfulqa-frontier-covariance-gate-l80/covariance-mode-gate-report.json` |
| E3 | 2026-06-25 | **ACCEPT initial locality**: dependency-free Gaussian 2-Wasserstein/Bures distance added for tensor Gaussians and `TruthManifold` objects; synthetic metric-property tests pass. Cached l80 Qwen/SmolLM2 reports show adjacent monitored layers closer than distant layers for both `full` and `shrinkage` covariance, with nearest-adjacent fraction 1.0. Treat as accepted for coarse layer/checkpoint drift inspection; run a denser layer matrix before using it as a fine-grained training diagnostic. | `gaussian_wasserstein_distance()` / `manifold_distance()` / `manifold_wasserstein_distance()` unit tests; `benchmarks/compare_manifold_distances.py`; `artifacts/e3-manifold-distance-sanity/e3-manifold-distance-sanity-summary.json` |
| E4 | 2026-06-25 | **ACCEPT top-3 layer-band predictor**: `eigentruth.eval.intrinsic_dimension` implements dependency-free TwoNN ID without `torch.cdist`; synthetic 1D/2D/5D ordering tests pass. Cached l80 Qwen and SmolLM2 factual warmup profiles both rise then fall and peak at `-14`. The ID peak lands in the TruthfulQA `truth_proj` AUROC top-3 for both l80 runs (`peak_in_top_k_rate=1.0`); exact best-layer rate is 0.0, so this is a coarse layer-band selector, not an exact layer oracle. | `twonn_intrinsic_dimension()` / `intrinsic_dimension_profile()` unit tests; `benchmarks/eval_intrinsic_dimension.py`; `benchmarks/compare_intrinsic_dimension_layers.py`; `artifacts/e4-intrinsic-dimension-l80/intrinsic-dimension-report.json`; `artifacts/e4-intrinsic-dimension-l80/intrinsic-layer-prediction-report.json` |
| E5 | 2026-06-25 | **ACCEPT tiny fine-tune telemetry sanity**: `eigentruth.training` adds a dependency-free `RepresentationTelemetryRecorder` and `RepTelemetryCallback` with per-layer mean norm, variance trace, spectrum rank diagnostics, and Gaussian 2-Wasserstein/Bures distance to an initialization baseline. Synthetic trajectories pass the distance/rank sanity gate. In a pure PyTorch tiny clean-vs-duplicate fine-tune, target-layer effective-rank telemetry separates at epoch 1 while eval-loss degradation crosses its margin at epoch 5; max rank margin is 1.837. The callback interface is unit-tested with HF Trainer-compatible hooks; a real transformers Trainer end-to-end run remains a follow-up evidence item. | `RepresentationTelemetryRecorder` / `RepTelemetryCallback` / `representation_telemetry_snapshot()` unit tests; `benchmarks/training_telemetry_sanity.py`; `benchmarks/training_telemetry_tiny_finetune.py`; `artifacts/e5-training-telemetry-sanity/training-telemetry-sanity-report.json`; `artifacts/e5-training-telemetry-tiny-finetune/training-telemetry-tiny-finetune-report.json` |
| E6 | | pending | |
| E7 | | pending | |
| E8 | | pending | |

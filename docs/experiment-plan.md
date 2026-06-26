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
  A follow-up l80 spectrum-to-sweep audit blocks spectrum-only layer selection:
  the best MP-normalized top-eigenvalue heuristic hits the `truth_proj` AUROC
  top-2 layer in only 1/2 runs. The follow-up layer-band audit accepts the same
  top-eigenvalue heuristic with radius 1 as a cheap candidate-band prior: it
  keeps both current l80 best layers in band while averaging 2 of 5 monitored
  layers.

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
  best-layer selection; the current combined layer-band audit shows ID radius 2
  also keeps both best layers in band but sweeps 4 of 5 monitored layers, making
  spectrum radius 1 the cheaper current prior.

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
  **Status:** `benchmarks/model_collapse_early_warning.py` runs a deterministic
  pseudo-label self-training loop after a clean warm start. In the current E6
  artifact, target-layer effective rank decays monotonically across five
  self-training generations and crosses the warning margin at generation 1,
  while visible quality loss is not reached until generation 3. TwoNN intrinsic
  dimension provides same-direction total-drop support, but is not monotonic in
  this tiny setup, so the accepted signal is rank-based early warning rather
  than a full ID-collapse claim.

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
  **Status:** `TrajectoryMonitor` and `trajectory_convergence_metrics()` landed
  as dependency-free generation-trajectory diagnostics. The synthetic E7 sanity
  benchmark generates convergent vs wandering hidden-state trajectories with
  quality/NLL proxies; convergence score correlates with quality at Spearman
  0.924 and separates high-quality trajectories with AUROC 0.937. This accepts
  the monitor primitive and synthetic correlation gate. The original gpt2 /
  TruthfulQA hidden-state trajectory replication remains the next evidence item
  before treating this as a real open-generation hallucination signal.

### E8. Concept registry + multi-probe (platform glue)
- **Question:** engineering, not science — can multiple (manifold, direction) pairs be
  saved/versioned/loaded and monitored simultaneously with a clean API?
- **Method:** registry format (.pt + metadata), multi-probe attach, docs.
- **Accept:** API review + tests; example with two concepts monitored at once.
- **Deliverable:** `eigentruth.registry`. Foundation for BYO-concept use.
- **Cost:** medium. Independent.
  **Status:** `ConceptArtifact` saves versioned layer-bound manifolds as `.pt`
  payloads, `ArtifactRegistry.record_concept_artifact()` records them, and
  `MultiConceptMonitor` can attach multiple probes to one model and return
  per-concept diagnostics. `benchmarks/concept_registry_smoke.py` creates two
  synthetic concept artifacts, registers them, attaches both probes to a toy
  model, and writes a manifest-backed report.

## Phase 3 — Consolidation and release

### E9. Prune and rename
- Act on accumulated evidence: demote/remove hyperbolic HSE from the default path
  (current evidence: no lift over Euclidean); generalize naming
  (`Manifold`/`Direction`/`Probe`, keep `Truth*` aliases); reposition README as a
  representation-observability toolkit for training + inference.
  **Status:** HSE is now opt-in on hooks/wrappers via `track_hse=True`; the
  default runtime path keeps Mahalanobis-style representation distance and
  optional steering only. Public compatibility aliases landed:
  `RepresentationManifold`, `RepresentationSubspace`, `RepresentationProbe`,
  and `RepresentationMonitor`; the original `Truth*` APIs remain supported.

### E10. Release 0.2.0 + honest writeup
- Package, publish, and write up every experiment **including negative results**.
  **Status:** package metadata, public version, release notes, roadmap links, and
  experiment decision log are aligned around the 0.2.0 research baseline. This
  remains an alpha research release, not a production hallucination detector.

## Decision log

| Exp | Date | Verdict | Evidence |
|-----|------|---------|----------|
| (hyperbolic HSE vs Euclidean) | 2026-06-08 | no lift (0.474 vs 0.484, gpt2 L-8) | `benchmarks/results_gpt2_l-8.json` |
| E0 | 2026-06-10 | **truth_proj wins**: 0.723 @L-8, peak 0.753 @L-6, beats maha (0.622/0.638) at every layer except -12; both collapse at L-1. Default guidance: contrastive direction, mid-late layers (-8…-4); maha as no-false-data fallback. | `benchmarks/results_gpt2_sweep.json` |
| E1 | 2026-06-10 | **ACCEPT**: empirical false-alarm tracks nominal within 1.3% at α∈{.05,.1,.2} for both maha_last and truth_proj (20 seeded splits). Power at α=0.2: truth_proj 46.9% vs maha 34.1%. Conformal thresholds replace hand-picked ones. | `benchmarks/results_conformal_*.json` |
| E2 | 2026-06-25 | **ACCEPT shrinkage**: spectrum diagnostics and OAS-style shrinkage mode landed; tiny offline matrix smoke passes; l80 cache-only covariance gate promotes `shrinkage` for both Qwen and SmolLM2. Qwen also accepts `low_rank_16`; SmolLM2 rejects `low_rank_16` at the 0.01 `maha_last` AUROC-drop gate; both reject `diag`. | `TruthManifold.spectrum()` / `covariance_spectrum()` / `covariance_shrinkage_intensity()` unit tests; `eval_truthfulqa.py --include-layer-spectra` tests; `artifacts/tiny_covariance_shrinkage_matrix/cache-profile-matrix-report.json`; `artifacts/truthfulqa-frontier-covariance-gate-l80/covariance-mode-gate-report.json` |
| E2-layer-selection | 2026-06-26 | **REJECT spectrum-only layer selector**: `compare_spectrum_layers.py` and cache-only l80 spectrum reports show the best heuristic, `max_top_eigenvalue_to_mp_upper`, hits `truth_proj` AUROC top-2 in 1/2 runs and exact best in 1/2 runs. Mean AUROC regret is small (`0.0077`), but the report status remains `fail`; use spectrum as a weak layer-band prior, not as a replacement for calibrated sweeps. | `benchmarks/compare_spectrum_layers.py`; `tests/test_benchmarks.py::test_compare_spectrum_layers_reports_heuristic_alignment`; `artifacts/truthfulqa-frontier-spectrum-layer-selection/spectrum-layer-comparison.json`; `artifacts/truthfulqa-frontier-spectrum-layer-selection/artifact-manifest.json` |
| E2-layer-band | 2026-06-26 | **ACCEPT conservative layer-band prior**: `compare_layer_band_selectors.py` combines intrinsic-dimension peak evidence, spectrum reports, and saved sweep rankings. On Qwen/SmolLM2 l80, `spectrum_max_top_eigenvalue_to_mp_upper_radius_1` keeps both models' best `truth_proj` layer in band with zero AUROC regret and averages 2/5 monitored layers; ID radius 2 also passes but averages 4/5 layers. `run_calibrated_observability_workflow.py --sweep-layers-from-band-report` now consumes the report and fingerprints the source before running the normal calibrated sweep. | `benchmarks/compare_layer_band_selectors.py`; `benchmarks/run_calibrated_observability_workflow.py`; `tests/test_benchmarks.py::test_compare_layer_band_selectors_recommends_union_and_writes_manifest`; `tests/test_benchmarks.py::test_calibrated_observability_derives_sweep_layers_from_band_report`; `artifacts/truthfulqa-frontier-layer-band-selection/layer-band-comparison.json`; `artifacts/truthfulqa-frontier-layer-band-selection/artifact-manifest.json` |
| E2-selfcheck-signal | 2026-06-26 | **REJECT current SmolLM2 l20 direct selfcheck signal promotion**: `export_inside_diagnostics_samples.py` recovers 77/154 triggered records from the existing top-40% INSIDE diagnostics cache, but only 25 records have at least two non-empty samples before alignment/deduplication. The workflow sample-quality gate fails after alignment/deduplication with only 17/154 usable two-sample records, coverage `0.110`, average samples per record `0.416`, and not-applicable rate `0.890`. At alpha 0.10 `truth_proj` remains best (`AUROC 0.682`, detection `0.178`, false alarm `0.091`) while the best geometry-by-selfcheck fusion reaches only `AUROC 0.561`, detection `0.096`, false alarm `0.041`. Require better aligned generations before promoting selfcheck signals. | `benchmarks/export_inside_diagnostics_samples.py`; `benchmarks/run_selfcheck_signal_fusion_workflow.py`; `tests/test_benchmarks.py::test_export_inside_diagnostics_samples_reconstructs_statement_cache_keys`; `tests/test_benchmarks.py::test_run_selfcheck_signal_fusion_workflow_writes_manifest_and_artifacts`; `tests/test_benchmarks.py::test_run_selfcheck_signal_fusion_workflow_blocks_low_quality_samples`; `artifacts/smollm2-l20-direct-selfcheck-signal-fusion/sample-quality-report.json`; `artifacts/smollm2-l20-direct-selfcheck-signal-fusion/artifact-manifest.json` |
| E2-retrieval-provenance | 2026-06-26 | **ACCEPT provenance gate, BLOCK current local corpora as external evidence**: `audit_retrieval_corpus_provenance.py` separates external grounding candidates from controlled dataset baselines and answer-echo stress controls. On l80, the correct-answer corpus fails `grounding` but passes `controlled_baseline` with exact answer copy rate `0.514`; the answer-echo corpus fails `grounding` with exact answer copy rate `0.996` and claim-id link rate `1.000`, but passes `stress_control`. No current local corpus is `external_domain_shift_ready`, so external/domain-shifted retrieval remains the next evidence requirement. | `benchmarks/audit_retrieval_corpus_provenance.py`; `tests/test_benchmarks.py::test_build_retrieval_stress_corpus_exposes_answer_echo_self_support`; `tests/test_benchmarks.py::test_audit_retrieval_corpus_provenance_classifies_external_and_controlled_corpora`; `artifacts/truthfulqa-l80-retrieval-corpus-provenance-audit/artifact-manifest.json`; `artifacts/truthfulqa-l80-retrieval-corpus-provenance-audit/manifest-verification.json` |
| E2-external-corpus-ingestion | 2026-06-26 | **ACCEPT ingestion gate, REQUIRE real source evidence before promotion**: `build_external_retrieval_corpus.py` normalizes caller-supplied JSON/JSONL/text source files into explicit `external_evidence_candidate` corpora, fingerprints sources, and rejects score labels, claim ids, and score-dump row-link metadata. `audit_retrieval_corpus_provenance.py` now fails closed on untyped local corpora, so no raw text dump can become grounding evidence by omission. This completes the local ingestion gate; the next evidence item is a real licensed/domain-shifted external corpus artifact and rerun of verifier-signal fusion. | `benchmarks/build_external_retrieval_corpus.py`; `benchmarks/audit_retrieval_corpus_provenance.py`; `tests/test_benchmarks.py::test_build_external_retrieval_corpus_outputs_auditable_external_candidate`; `tests/test_benchmarks.py::test_build_external_retrieval_corpus_rejects_score_dump_metadata`; `tests/test_benchmarks.py::test_audit_retrieval_corpus_provenance_rejects_untyped_grounding_corpus` |
| E2-wikidata-source-gate | 2026-06-26 | **ACCEPT source/provenance gate, DO NOT promote route coverage yet**: `fetch_wikidata_reference_docs.py` materializes 120 Wikidata country-capital SPARQL records as CC0 JSONL source docs, `build_external_retrieval_corpus.py` converts them to an explicit external corpus, and `audit_retrieval_corpus_provenance.py --audit-role grounding` passes with `external_domain_shift_ready=true`. The top-level manifest verifies recursively. The artifact metadata sets `promotes_verifier_route=false` because the corpus is narrow country-capital evidence; next step is measuring retrieval fixture coverage and route quality before any release gate consumes it as verifier evidence. | `benchmarks/fetch_wikidata_reference_docs.py`; `tests/test_benchmarks.py::test_fetch_wikidata_reference_docs_feeds_external_corpus_gate`; `artifacts/wikidata-country-capitals-external-corpus/artifact-manifest.json`; `artifacts/wikidata-country-capitals-external-corpus/manifest-verification.json` |
| E2-wikidata-route-audit | 2026-06-26 | **BLOCK route promotion, ACCEPT negative route-quality evidence**: `run_local_retrieval_route_workflow.py` runs the Wikidata country-capital external corpus against Qwen l80. The corpus retrieves hits for 254/556 records (`coverage_rate=0.457`, 925 hits), but `retrieval_groundedness` verified false alarm is `0.149`, above the explicit `0.05` gate, while verified detection is `0.286`. `analyze_retrieval_route_gaps.py` shows all 556 records finish as `insufficient_evidence`: 302 have no retrieval hits, 254 use retrieval, and all 925 hits are Wikidata `P36` capital facts. The top-level route-audit manifest verifies recursively and records `promotes_verifier_route=false`. Conclusion: the source/provenance pipeline works, but the narrow country-capital corpus is not a deployable TruthfulQA verifier route; expand predicates or add a structured Wikidata verifier next. | `benchmarks/analyze_retrieval_route_gaps.py`; `artifacts/wikidata-country-capitals-external-route-audit-qwen05-l80/route-quality-summary.json`; `artifacts/wikidata-country-capitals-external-route-audit-qwen05-l80/retrieval-route-gap-analysis.json`; `artifacts/wikidata-country-capitals-external-route-audit-qwen05-l80/artifact-manifest.json`; `artifacts/wikidata-country-capitals-external-route-audit-qwen05-l80/manifest-verification.json` |
| E2-wikidata-structured-qa | 2026-06-26 | **ACCEPT structured bridge smoke, DO NOT broaden coverage claim**: `build_wikidata_qa_corpus.py` converts Wikidata `P36` fact documents, or a template JSON of multiple properties such as `P36`/`P37`/`P38`, into label-free structured QA records, preserving source metadata, rejecting reserved score-dump keys, and rejecting QID-only labels by default. The focused route smoke proves these facts feed `retrieval_structured_qa`: matching country-capital answers are supported and mismatched answers are refuted through the existing `QuestionAnswerVerifier`; the multi-property smoke proves the bridge is not hard-coded to country capitals. This is the structured Wikidata verifier bridge suggested by the route-gap audit, but it only covers properties present in the source corpus. | `benchmarks/build_wikidata_qa_corpus.py`; `tests/test_benchmarks.py::test_build_wikidata_qa_corpus_feeds_retrieval_structured_qa`; `tests/test_benchmarks.py::test_build_wikidata_qa_corpus_supports_multiple_property_templates` |
| E3 | 2026-06-25 | **ACCEPT initial locality**: dependency-free Gaussian 2-Wasserstein/Bures distance added for tensor Gaussians and `TruthManifold` objects; synthetic metric-property tests pass. Cached l80 Qwen/SmolLM2 reports show adjacent monitored layers closer than distant layers for both `full` and `shrinkage` covariance, with nearest-adjacent fraction 1.0. Treat as accepted for coarse layer/checkpoint drift inspection; run a denser layer matrix before using it as a fine-grained training diagnostic. | `gaussian_wasserstein_distance()` / `manifold_distance()` / `manifold_wasserstein_distance()` unit tests; `benchmarks/compare_manifold_distances.py`; `artifacts/e3-manifold-distance-sanity/e3-manifold-distance-sanity-summary.json` |
| E4 | 2026-06-25 | **ACCEPT top-3 layer-band predictor**: `eigentruth.eval.intrinsic_dimension` implements dependency-free TwoNN ID without `torch.cdist`; synthetic 1D/2D/5D ordering tests pass. Cached l80 Qwen and SmolLM2 factual warmup profiles both rise then fall and peak at `-14`. The ID peak lands in the TruthfulQA `truth_proj` AUROC top-3 for both l80 runs (`peak_in_top_k_rate=1.0`); exact best-layer rate is 0.0, so this is a coarse layer-band selector, not an exact layer oracle. | `twonn_intrinsic_dimension()` / `intrinsic_dimension_profile()` unit tests; `benchmarks/eval_intrinsic_dimension.py`; `benchmarks/compare_intrinsic_dimension_layers.py`; `artifacts/e4-intrinsic-dimension-l80/intrinsic-dimension-report.json`; `artifacts/e4-intrinsic-dimension-l80/intrinsic-layer-prediction-report.json` |
| E5 | 2026-06-25 | **ACCEPT tiny fine-tune telemetry sanity**: `eigentruth.training` adds a dependency-free `RepresentationTelemetryRecorder` and `RepTelemetryCallback` with per-layer mean norm, variance trace, spectrum rank diagnostics, and Gaussian 2-Wasserstein/Bures distance to an initialization baseline. Synthetic trajectories pass the distance/rank sanity gate. In a pure PyTorch tiny clean-vs-duplicate fine-tune, target-layer effective-rank telemetry separates at epoch 1 while eval-loss degradation crosses its margin at epoch 5; max rank margin is 1.837. The callback interface is unit-tested with HF Trainer-compatible hooks; a real transformers Trainer end-to-end run remains a follow-up evidence item. | `RepresentationTelemetryRecorder` / `RepTelemetryCallback` / `representation_telemetry_snapshot()` unit tests; `benchmarks/training_telemetry_sanity.py`; `benchmarks/training_telemetry_tiny_finetune.py`; `artifacts/e5-training-telemetry-sanity/training-telemetry-sanity-report.json`; `artifacts/e5-training-telemetry-tiny-finetune/training-telemetry-tiny-finetune-report.json` |
| E6 | 2026-06-25 | **ACCEPT rank early-warning sanity**: deterministic pseudo-label self-training over a tiny model shows target-layer effective rank decaying monotonically across five self-training generations. Rank warning occurs at generation 1, before visible quality loss at generation 3; rank total drop is 0.093. TwoNN intrinsic dimension also drops by 0.101 overall, but is not monotonic, so E6 is accepted as an effective-rank collapse warning rather than a full intrinsic-dimension collapse result. | `benchmarks/model_collapse_early_warning.py`; `tests/test_benchmarks.py::test_model_collapse_early_warning_detects_rank_decay_before_quality_loss`; `artifacts/e6-model-collapse-early-warning/model-collapse-early-warning-report.json` |
| E7 | 2026-06-25 | **ACCEPT synthetic trajectory sanity**: `TrajectoryMonitor` computes step-distance decay, Koopman-style rate, path efficiency, and convergence scores from per-token hidden-state trajectories. Synthetic trajectory sanity passes with Spearman(convergence, quality)=0.924, Spearman(convergence, NLL proxy)=-0.921, and quality AUROC=0.937. Treat this as a monitor primitive plus synthetic correlation gate; gpt2/TruthfulQA trajectory replication remains pending before claiming a real open-generation detector. | `TrajectoryMonitor` / `trajectory_convergence_metrics()` unit tests; `spearman_correlation()` unit tests; `benchmarks/trajectory_convergence_sanity.py`; `artifacts/e7-trajectory-convergence-sanity/trajectory-convergence-sanity-report.json` |
| E8 | 2026-06-25 | **ACCEPT platform glue**: versioned concept artifacts, local registry records, and multi-probe attachment landed with no new mandatory dependencies. The smoke path saves two synthetic concepts, monitors both simultaneously on a toy model, and records diagnostics plus manifest provenance. This accepts the BYO-concept API shape; real concept quality still depends on downstream warmup data and calibration. | `ConceptArtifact` / `MultiConceptMonitor` unit tests; `benchmarks/concept_registry_smoke.py`; `artifacts/e8-concept-registry-smoke/concept-registry-smoke.json` |
| E9 | 2026-06-25 | **ACCEPT compatible consolidation**: HSE was demoted from default hook/wrapper work to explicit `track_hse=True` ablations, reflecting the earlier no-lift result. Generic `Representation*` aliases now expose the broader representation-observability API while keeping all `Truth*` names as stable backwards-compatible aliases. | `TruthProbe(track_hse=False/True)` tests; `EigenTruthWrapper.track_hse` tests; `RepresentationManifold` / `RepresentationProbe` / `RepresentationMonitor` alias contract tests; README and methodology updates |
| E10 | 2026-06-25 | **ACCEPT 0.2.0 research baseline**: package metadata and public version now identify the 0.2.0 release; the release writeup documents accepted components, negative results, evidence boundaries, and reproduction commands without claiming production readiness. | `docs/release-0.2.0.md`; `make check`; `make release-check` |

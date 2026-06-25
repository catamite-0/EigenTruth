# EigenTruth Frontier Research Notes

Date: 2026-06-25

## Current Frontier Direction

EigenTruth should move from hallucination detection alone toward calibrated participation control: decide when a model should answer, when it should retrieve/verify, and when it should abstain. The strongest near-term fit is conformal abstention because it gives a post-hoc, dependency-free layer with finite-sample risk/coverage semantics that can sit on top of existing EigenTruth scores.

## Evidence Checked

- Geometry-Calibrated Conformal Abstention for Language Models (arXiv:2604.27914, submitted 2026-04-30) frames abstention as a post-hoc conformal layer over prediction confidence, with finite-sample participation and correctness guarantees, and uses representation geometry to align confidence with model ignorance.
- INSIDE (arXiv:2402.03744) supports EigenTruth's internal-state route: hidden-state embeddings retain hallucination signals, and EigenScore measures semantic consistency/diversity in dense representation space.
- SelfCheckGPT (arXiv:2303.08896) and FactSelfCheck (arXiv:2503.17229, EACL 2026 findings) support the sampling/self-consistency route. FactSelfCheck moves from sentence-level to fact-level graph/triple checks, which is a strong next adapter direction but needs heavier extraction and multi-sample fixtures.
- Semantic Energy (arXiv:2508.14496) supports energy-style uncertainty beyond entropy. EigenTruth already has lightweight semantic-energy proxies; a future step is to compare them against conformal abstention and route-cost gates.

## Implemented This Round

Added dependency-free conformal abstention primitives:

- `ConformalAbstentionReport`
- `ConformalAbstentionDecision`
- `ConformalAbstentionComparisonReport`
- `ConformalAbstentionReleaseGate`
- `conformal_abstention_report(...)`
- `conformal_abstention_comparison_report(...)`
- `conformal_abstention_release_gate(...)`
- `evaluate_conformal_abstention(...)`

The report exposes threshold, coverage/participation, empirical selective accuracy, conservative correct-retention lower bound, and conservative conditional-correctness lower bound. Runtime code can call `report.decide(score)` to get a structured `participate` or `abstain` decision.

Wired the primitive into `benchmarks/eval_conformal.py`:

- `--save-abstention-report PATH` writes a sidecar report from any selected score dump signal.
- `--include-abstention-report` embeds the same report in the main conformal payload.
- `--abstention-signal`, `--abstention-direction`, and `--abstention-alpha` make the report reusable across internal diagnostics, output confidence proxies, and score-fusion outputs.
- `--save-abstention-comparison PATH` and `--include-abstention-comparison` rank multiple `--abstention-signals` by conservative conditional correctness, selective accuracy, participation, or retention.
- `--save-abstention-release-gate PATH` and `--include-abstention-release-gate` convert the selected report or comparison recommendation into a fail-closed promotion verdict with minimum conservative conditional-correctness and maximum abstention-rate requirements.
- The abstention block is evidence-only and does not change the base E1 conformal verdict.

Wired abstention into the control plane:

- `ParticipationGateConfig` can consume a single abstention report, a comparison candidate, or a full comparison report and select the recommended candidate.
- `RiskController(..., participation_gate=...)` records a `participation_gate` trace block and, by default, only gates decisions that would otherwise `accept`.
- `ControlPolicyConfig` can change the gate action, risk level, confidence floor, and action scope when a product wants to gate `retrieve` or other actions as well.

Added the first promotion check for participation control:

- `ConformalAbstentionReleaseGate` accepts a single report, comparison candidate, full comparison report, or JSON mapping.
- It blocks promotion when the selected candidate's conservative conditional-correctness lower bound is too low or its empirical abstention rate is too high.
- `eval_conformal.py` can write the gate verdict as a sidecar, embed it in the main payload, and set the main verdict to `REJECT` when the release gate fails.

Added post-hoc stability replay for participation gates:

- `benchmarks/eval_abstention_stability.py` consumes existing score dumps and does not load a model.
- Each seed calibrates abstention thresholds on a stratified split of correct records, evaluates held-out participation metrics, ranks candidate signals, and applies the abstention release gate.
- The report records recommended-signal counts, metric variance, release-gate pass/block counts, artifact manifests, and optional registry metadata for frontier l80-style evidence.

Ran and registered the current Qwen/SmolLM2 l80 abstention-stability replay:

- `report:truthfulqa-frontier-qwen-smollm2-l80-abstention-stability:0.1` evaluates 8 candidate signals across seeds `0..9` using the existing JSONL score dumps.
- `truth_proj` is the stable recommended abstention signal for both runs in 10/10 seeds.
- The promotion gate correctly blocks both runs in 10/10 seeds because conservative conditional-correctness lower bounds stay near `0.498` for Qwen and `0.485` for SmolLM2, below the required `0.8`.
- The verified manifest is `benchmark_manifest:truthfulqa-frontier-qwen-smollm2-l80-abstention-stability:0.1`.

Added a combined frontier release-evidence comparator:

- `benchmarks/compare_frontier_release_evidence.py` consumes staged verifier-stability and abstention-stability reports without rerunning models, verifiers, or retrieval.
- It emits separate verifier and abstention track verdicts plus one fail-closed release decision.
- On the current l80 artifacts, verifier stability promotes while abstention stability blocks; this records the correct product posture: staged verifier routing is supported by current evidence, participation-gate promotion is not.

Added dependency-free fact-level claim metadata:

- `extract_claims(..., include_triples=True)` and `SentenceClaimExtractor(include_triples=True)` can attach rule-based `claim_triples` metadata without requiring an external extractor.
- `ClaimVerificationPlanner(include_extracted_triples=True)` routes those extracted triples into the existing `triple_evidence` path, so local fact-level audits can be planned before a stronger extractor is available.
- The API keeps stronger extraction optional through the existing claim and triple extractor protocols.

Added a geometry-calibrated score primitive:

- `geometry_calibrated_anomaly_scores(...)` rank-calibrates representation geometry signals and uncertainty/confidence proxies against normal calibration records, then fuses them with an explicit interaction term.
- `GeometryScoreFusionArtifact` and `GeometryScoreFusionCalibrator` make that score deployable as a conformal artifact without changing score-dump schemas or adding dependencies.
- `eval_score_ensemble.py` now evaluates the geometry-by-uncertainty fusion family alongside single signals and naive rank fusion, and can save a deployable `GeometryScoreFusionArtifact` from the selected `--best-alpha`.

Ran the current Qwen/SmolLM2 l80 geometry-fusion replay:

- `artifacts/truthfulqa-frontier-qwen-smollm2-l80-geometry-fusion/score-ensemble-report.json` compares `subspace_resid,resid_update_norm,eigenscore` against `nll_answer` as the only available uncertainty proxy.
- At alpha `0.100`, Qwen `truth_proj` detects `0.279`, naive `mean_rank` detects `0.244`, and the best geometry-fusion method detects only `0.055`.
- At alpha `0.100`, SmolLM2 `truth_proj` detects `0.229`, naive `mean_rank` detects `0.188`, and the best geometry-fusion method detects only `0.036`.
- Variants that add `truth_proj` to the geometry group or use `max_rank` geometry aggregation improve the fusion score slightly but still remain far below `truth_proj` (`<=0.083` Qwen, `<=0.069` SmolLM2). Current evidence says `nll_answer` is a poor final-correction proxy, not that geometry-by-uncertainty fusion is intrinsically bad.

Added verifier-signal score-dump conversion and replay:

- `build_verifier_signal_score_dump.py` converts `eval_verifier_ensemble.py --verified-records-jsonl` sidecars into standard score columns such as `verifier_refuted`, `verifier_refute_confidence`, `verifier_uncertainty`, and `selfcheck_refute_rate`.
- `artifacts/truthfulqa-l80-staged-qa-verifier-signals/` applies this to the staged structured-QA l80 verifier route and saves per-model `GeometryScoreFusionArtifact` files.
- At alpha `0.100`, Qwen `verifier_refuted` is the strongest single signal (`0.297` detection, zero false alarm), while geometry fusion reaches `0.285` detection at `0.089` false alarm.
- At alpha `0.100`, SmolLM2 geometry fusion reaches `0.261` detection at `0.095` false alarm, beating both `truth_proj` (`0.229`) and `verifier_refuted` (`0.232`).
- Current frontier direction: use LLM-internal geometry as the monitor/trigger, then feed structured verifier, retrieval, and selfcheck outputs back as calibrated final-correction signals.

## Next Research-to-Code Candidates

1. Replace the staged structured-QA upper-bound verifier signals with non-oracle retrieval/selfcheck signals, then rerun the same verifier-signal geometry-fusion report.
2. Integrate stronger fact/triple extractors behind the existing protocols and benchmark them against the rule-based extractor.
3. Use `eval_truthfulqa.py --include-layer-spectra` reports to test whether Marchenko-Pastur spikes/effective-rank predict layer selection, then extend the same fields into training telemetry for collapse experiments.

# EigenTruth Frontier Research Notes

Date: 2026-06-26

## Current Frontier Direction

EigenTruth should move from hallucination detection alone toward calibrated participation control: decide when a model should answer, when it should retrieve/verify, and when it should abstain. The strongest near-term fit is conformal abstention because it gives a post-hoc, dependency-free layer with finite-sample risk/coverage semantics that can sit on top of existing EigenTruth scores.

## Evidence Checked

- Geometry-Calibrated Conformal Abstention for Language Models (arXiv:2604.27914, submitted 2026-04-30) frames abstention as a post-hoc conformal layer over prediction confidence, with finite-sample participation and correctness guarantees, and uses representation geometry to align confidence with model ignorance.
- INSIDE (arXiv:2402.03744) supports EigenTruth's internal-state route: hidden-state embeddings retain hallucination signals, and EigenScore measures semantic consistency/diversity in dense representation space.
- Layer-wise Semantic Dynamics (arXiv:2510.04933), Contextual Perturbation and Representation Drift (arXiv:2505.16894), and MultiHaluDet (arXiv:2605.24919) point in the same direction for trajectory-style hidden-state analysis: hallucination detection should examine representation dynamics, not only static final-layer embeddings. This justifies the E7 real-data replay harness. The limit-128 gpt2/SmolLM2 follow-up shows trajectory convergence is not yet robust enough as a standalone detector: gpt2 reaches AUROC 0.608, while SmolLM2 reaches only 0.560 and the fail-closed evidence gate remains blocked. The aligned ablation matrix is more nuanced: trajectory improves the best gpt2 fusion candidate but hurts the best SmolLM2 candidate. `SignalSelectionPolicy` and selected fusion artifacts now turn that result into an explicit conditional artifact path, so trajectory remains a model/run-specific fusion feature rather than a default release signal.
- SelfCheckGPT (arXiv:2303.08896) and FactSelfCheck (arXiv:2503.17229, EACL 2026 findings) support the sampling/self-consistency route. FactSelfCheck moves from sentence-level to fact-level graph/triple checks, which is a strong next adapter direction but needs heavier extraction and multi-sample fixtures.
- Semantic Energy (arXiv:2508.14496) supports energy-style uncertainty beyond entropy. EigenTruth already has lightweight semantic-energy proxies; a future step is to compare them against conformal abstention and route-cost gates.
- CiteCheck (arXiv:2605.27700) shows that citation hallucinations often appear as small metadata drift rather than fully fabricated references. This supports a separate citation-integrity route before broad retrieval: DOI, arXiv id, URL, author/year, title, and local reference labels should be checked against a trusted citation catalog instead of treated as ordinary lexical groundedness.
- Internal Representations as Indicators of Hallucinations in Agent Tool Selection (arXiv:2601.05214) frames agent hallucination as incorrect tool selection, malformed parameters, and tool bypass. This supports keeping tool-route intent explicit in `ClaimVerificationPlan` instead of only checking final text.
- World-Model-Augmented Web Agents with Action Correction (arXiv:2602.15384) uses consequence simulation and action correction before risky actions. This supports EigenTruth's world-model route as a post-draft verifier and pre-action correction adapter rather than a core dependency.
- TokenHD (arXiv:2605.12384) and related span/token-level work point toward finer localization of hallucinations. EigenTruth's current lightweight equivalent is claim-level risk localization, route budgeting, and trace evidence; learned token-level detectors remain out of scope until the dependency and training boundary is explicit.
- Pre-generation hallucination detection with soft targets (arXiv:2606.21917) reinforces the current layer/sweep direction: hallucination risk is better treated as a probability estimated from internal representations than as a single hard decoded label.
- Entropy Alone is Insufficient for Safe Selective Prediction in LLMs (arXiv:2603.21172) and the UQ-as-clustering critique (arXiv:2605.19220) both argue against relying on entropy/self-consistency alone. EigenTruth should keep combining internal geometry with correctness/verifier/world-model evidence and deployment-facing selective metrics.
- Single-decode first-token confidence work (arXiv:2605.05166) points to a cheap baseline before multi-sample routes: top-k entropy at the first answer-token prediction can be logged from the same forced-answer pass and compared against internal geometry, INSIDE/selfcheck, and verifier signals.
- Counterfactual Probing for Hallucination Detection and Mitigation (arXiv:2508.01862) supports adding perturbation sensitivity audits: robust verifiers should change status on entity, temporal, quantitative, or logical counterfactuals instead of staying invariant to false variants.

## Implemented This Continuation

Added hidden-state soft-target attention probe artifacts:

- `soft_error_rate_targets(...)` converts sampled-answer correctness flags into empirical error-rate soft targets.
- `AttentionSoftTargetProbeArtifact.fit(...)` trains a torch-only attention-pooled hidden-state probe over prompt token representations, using soft BCE targets and an attention mask.
- The artifact exposes risk logits/probabilities, token attention weights, JSON-safe metadata, and torch save/load.
- This implements the local core primitive for the current soft-target attention-probing direction without adding a new mandatory dependency or binding the benchmark pipeline to a specific model cache format yet.

Added soft pre-generation risk estimates:

- `SoftPreGenerationRiskConfig` computes a dependency-free probability-style risk estimate from prompt features and caller metadata before generation.
- `PreGenerationRiskAssessment` now records the soft risk score, probability, risk level, thresholds, and feature/metadata contribution traces.
- The default policy records the estimate without changing the existing hard-rule profile route; `route_on_soft_risk=true` can explicitly upgrade low hard-rule prompts into `balanced` or `audit`.
- This is the product-control shell for the pre-generation soft-target direction: it makes risk-estimation traces and routing semantics stable now, while learned hidden-state attention probes remain a future optional adapter with a separate training/dependency boundary.

Added budget-aware adaptive verification planning:

- `VerificationBudgetPolicy` selects a bounded subset of high-value claims and verifier routes from a `ClaimVerificationPlan`.
- Budgets can cap verified claims, route attempts, tool payloads, and estimated relative cost units.
- Route selection is round-robin across selected high-priority claims, so a single risky claim cannot consume the whole route budget before other triggered claims get a first-pass verifier route.
- Claim priority uses triggered claim metadata/features plus route priority. The default route priority favors world-model, structured state, calculator, citation, triple-evidence, retrieval, then lexical groundedness.
- `ClaimVerificationPlanner.plan(..., budget_policy=...)` applies the policy directly, while `budget_verification_plan(...)` can post-process an existing plan or JSON-like plan mapping.
- `ClaimVerificationPlan.to_dict()` now carries a `budget` block with selected/dropped claim ids, selected/dropped routes, budget-exhaustion flags, and original/selected cost estimates.
- `ProductTrace.to_bounded_dict()` and `product_runtime_metrics(...)` preserve compact verification-budget summaries, including selected/dropped claim counts and claim/route/tool/cost budget exhaustion flags.

This is a product-facing implementation of the current research direction: use internal diagnostics and claim metadata to decide when verification is needed, then spend verifier/tool budget on the most consequential claims and routes while leaving an auditable trace of what was skipped. It does not add network retrieval, learned token detectors, or model-dependent world-model code.

Added a single-decode first-token uncertainty baseline:

- `topk_normalized_entropy(...)` and `first_token_confidence(...)` provide dependency-free logits uncertainty primitives.
- `eval_truthfulqa.py` now emits `first_token_entropy` from the first available answer-token prediction, stores `first_token_top_k` in report/cache config, and includes the signal in primary score dumps for conformal calibration, abstention comparison, and fusion experiments.
- The score is intentionally a baseline, not a promoted route: current entropy-only safety critiques still apply, so it must be compared against geometry, verifier, retrieval, selfcheck, and world-model signals before product use.

Added dependency-free counterfactual verifier auditing:

- `CounterfactualProbe`, `CounterfactualVerificationAuditor`, and `CounterfactualVerificationReport` run paired original/counterfactual claims through any local `Verifier`.
- `CounterfactualProbeGenerator` and `generate_counterfactual_probes(...)` can derive bounded metadata/entity/quantity/year/negation probes from existing claims so trace-side claim extraction can feed verifier perturbation audits without a model call.
- The report records expected-status accuracy, flip success, false invariance, unexpected flips, per-probe failure reasons, and probe-type summaries.
- `benchmarks/eval_counterfactual_verification.py` provides a local JSON/JSONL harness for `in_memory` and `structured_fact` verifier audits, and can write an artifact manifest plus local registry record for release evidence.
- This does not claim broad hallucination mitigation; it gives structured-fact, retrieval, world-model, or future external verifier routes a reproducible perturbation-sensitivity gate before their outputs are trusted by release workflows.

Added the covered-facts KG correction handoff:

- `benchmarks/run_covered_facts_external_evidence_workflow.py` registers saved Wikidata covered-facts route manifests into a local `ArtifactRegistry`, runs `compare_external_evidence_baselines.py` with `require_covered_facts_route=True`, writes the comparison report plus recursive manifest verification, and can register the comparator report as the release-gate handoff.
- The workflow is intentionally post-hoc and dependency-free: it does not rerun models, retrieval, databases, or external services. It turns existing canonical/paraphrase `structured_fact` route manifests into a reproducible external-evidence baseline comparison.
- Current saved canonical/paraphrase artifacts promote the aggregate `structured_fact` route but predate per-property `property_metrics`, so the handoff defaults to aggregate record/source/true/false covered-fact gates. Rebuilt route manifests can enable the stricter per-property gates already supported by the comparator.

Added a fail-closed layer-band replication gate:

- `benchmarks/audit_layer_band_replication.py` consumes saved `compare_layer_band_selectors.py` reports and audits whether one strategy has enough matched runs, enough model-family diversity, dense enough ranked-layer grids, best-layer hit rate, bounded AUROC regret, top-k coverage, and candidate-layer cost reduction to become a default benchmark preset.
- The current `artifacts/truthfulqa-frontier-layer-band-selection/` report is expected to block under the default dense-grid gate because both l80 runs rank only 5 monitored layers. This preserves the correct interpretation: the selector is a local cost-reduction prior, not a default preset.

Added dependency-free claim-risk localization:

- `ClaimRiskSpan`, `ClaimRiskLocalizationReport`, and `localize_claim_risk_spans(...)` turn existing claim spans, verifier statuses, route hints, and verification-budget drops into a JSON-ready localization report.
- The report preserves per-claim span offsets when available, risk level, risk score, verifier status, confidence, routes, evidence count, and reasons such as `verification_status:refuted`, sensitive claim features, or budget-dropped routes.
- `ProductTrace.to_bounded_dict()` now includes a compact `claim_risk_localization` summary with top risky spans, and `product_runtime_metrics(...)` exposes high/medium-or-high claim counts plus max localized risk score.
- This is the monitor-first bridge toward token/span-level hallucination tooling: it gives product UI and audits a concrete risky text region today, while leaving learned TokenHD-style detectors as a future optional adapter.

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
- The API keeps stronger extraction optional through the existing claim and triple extractor protocols. External or learned extractors can now hand local prediction files to the single-corpus workflow or to the cross-corpus fixture matrix with `--external-predictions CORPUS:NAME=PATH`, keeping the evaluation boundary dependency-free while testing cross-corpus/adversarial robustness.

Added a dependency-free citation-integrity route:

- `CitationRecord`, `CitationVerifier`, and `extract_citation_references(...)` validate citation references against caller-supplied local catalogs without network access or mandatory dependencies.
- `ClaimVerificationPlanner` now emits `citation` route hints and `citation_checks` payloads for cited claims, and the relative cost estimate accounts for citation checks separately from retrieval.
- `default_verifier_routes(..., citation_records=...)` can run citation catalog checks before triple evidence or groundedness, and it does not fall through on unresolved references or metadata drift. This keeps citation hallucinations from being masked by later broad lexical evidence.

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

- `build_verifier_signal_score_dump.py` converts `eval_verifier_ensemble.py --verified-records-jsonl` sidecars into standard score columns such as `verifier_refuted`, `verifier_refute_confidence`, `verifier_uncertainty`, `selfcheck_refute_rate`, `world_model_disagreement`, `world_model_conflict`, and `world_model_conflict_delta`. The world-model bridge now reads state-transition `prediction_metadata`, explicit `world_model_conflict` metadata, and direct ensemble-verifier agreement metadata, so claim-level world-model consensus gaps and expected-vs-actual world conflicts are not silently dropped before conformal calibration.
- `run_verifier_signal_fusion_workflow.py` wraps local retrieval/selfcheck fixture construction, verifier sidecar writing, verifier-signal score-dump conversion, geometry-fusion reporting, geometry artifact export, and manifest verification into one no-model workflow for non-oracle evidence experiments.
- `run_world_model_signal_calibration_workflow.py` specializes that bridge for world-model correction: it builds a deterministic state-transition fixture, can inject a controlled multi-world-model ensemble disagreement fixture, runs the world-model verifier route, emits verifier/world-model score columns including nonzero agreement-gap signals, evaluates score and geometry fusion, recursively verifies manifests, emits a `release_gate` over zero trace gap plus calibrated conflict evidence, and can register the workflow as a local report consumed by release-candidate gates. Its `policy_replay` ensemble strategy makes disagreement arise from a conservative high-quantity reservation policy instead of direct label shaping, providing a local stand-in for future learned/simulator world-model replay.
- `StateTransitionVerifier` now emits `world_model_reference`, `world_model_view`, and `world_model_conflict` metadata. This follows the current world-model framing: every hallucination judgment should make its reference world and view function explicit, then expose the exact expected/actual mismatch when the predicted world refutes a claim.
- `artifacts/truthfulqa-l80-staged-qa-verifier-signals/` applies this to the staged structured-QA l80 verifier route and saves per-model `GeometryScoreFusionArtifact` files.
- At alpha `0.100`, Qwen `verifier_refuted` is the strongest single signal (`0.297` detection, zero false alarm), while geometry fusion reaches `0.285` detection at `0.089` false alarm.
- At alpha `0.100`, SmolLM2 geometry fusion reaches `0.261` detection at `0.095` false alarm, beating both `truth_proj` (`0.229`) and `verifier_refuted` (`0.232`).
- Current frontier direction: use LLM-internal geometry as the monitor/trigger, then feed structured verifier, retrieval, selfcheck, and world-model disagreement outputs back as calibrated final-correction signals.

The first no-model local retrieval workflow artifact is now at
`artifacts/truthfulqa-l80-local-retrieval-verifier-signal-fusion/`. It uses the
committed Qwen/SmolLM2 l80 score dumps plus the local correct-answer corpus,
omits labels from claim metadata, writes verifier sidecars and enhanced score
dumps, saves per-model geometry-fusion artifacts, and verifies the top-level
manifest. At alpha `0.100`, the verifier stage keeps false alarm at `0.016` and
detects `0.316` for Qwen / `0.267` for SmolLM2. The best geometry fusion selects
`noisy_or` and detects about `0.795` for both models at about `0.070` false
alarm. This is a controlled local-corpus baseline; it should be stress-tested
with external/domain-shifted retrieval before being treated as open-domain
evidence.

Added an answer-echo retrieval stress control:

- `build_retrieval_stress_corpus.py` builds a local corpus from the same
  statement answers being audited, without using labels or copying label
  metadata by default.
- `artifacts/truthfulqa-l80-answer-echo-retrieval-stress/` runs that corpus
  through the same verifier-signal fusion workflow. It retrieves evidence for
  556/556 records but supports false claims at rate `0.980` and refutes false
  claims at rate `0.000`.
- At alpha `0.100`, verified detection collapses to `0.013` for Qwen and
  `0.010` for SmolLM2. This is the expected self-support failure mode and a
  required negative control before treating any retrieval setup as grounded.

Added a retrieval corpus provenance audit gate:

- `audit_retrieval_corpus_provenance.py` scans statement-bearing score dumps
  against local corpus files and separates `external_candidate`,
  `controlled_dataset_baseline`, and `answer_echo_stress_control` evidence.
- `build_external_retrieval_corpus.py` is now the explicit no-network ingestion
  boundary for caller-supplied external source files. It writes
  `corpus_type=external_evidence_candidate`, fingerprints input sources, and
  rejects label, claim-id, or score-dump row-link metadata before a corpus can
  enter grounding audit.
- Grounding audit is fail-closed for untyped local corpora: absence of answer
  copies or labels is no longer enough to mark a local text dump as external
  evidence.
- `fetch_wikidata_reference_docs.py` materializes real Wikidata CC0
  country-capital SPARQL records as JSONL source docs, and now also supports a
  `country_core_facts` preset for template-ready properties such as `P36`
  capital, `P37` official language, and `P38` currency. It filters bare
  Wikidata Q/P id labels by default so unresolved label-service rows do not
  become noisy natural-language evidence. The artifact at
  `artifacts/wikidata-country-capitals-external-corpus/` fetches 120 records,
  normalizes them through the external corpus builder, passes grounding
  provenance audit, and recursively verifies the manifest. It is source/provenance
  evidence only (`promotes_verifier_route=false`), not a broad TruthfulQA
  grounding route.
- `artifacts/wikidata-country-core-facts-external-corpus/` is the broader
  multi-predicate source gate: 359 `P36`/`P37`/`P38` facts after QID-label
  filtering, a 359-document external corpus, a 359-document structured QA
  corpus, and a passing recursive manifest verification. Its route audit at
  `artifacts/wikidata-country-core-facts-external-route-audit-qwen05-l80/`
  improves coverage from 254/556 to 275/556 records and distributes 1125 hits
  across `P36=510`, `P37=303`, and `P38=312`, but still blocks promotion because
  `retrieval_groundedness` false alarm is 0.155 against the 0.05 gate. This
  narrows the next frontier step to structured Wikidata QA/triple verification,
  not more lexical threshold tuning.
- The fetcher now also exposes an `organization_product_core_facts` preset for
  non-country KG evidence, defaulting to `P159` headquarters location, `P176`
  manufacturer, and `P571` inception over a deterministic OpenAI/Tesla/Apple
  seed subject set. These source docs carry generic `subject`/`value` metadata
  and feed `build_triple_extraction_fixture.py` directly, so cross-corpus
  extractor matrices no longer need synthetic enterprise/product facts as their
  only non-country test path.
- `artifacts/wikidata-country-capitals-external-route-audit-qwen05-l80/`
  measures that source against the Qwen l80 statement dump. Retrieval covers
  254/556 records, but `retrieval_groundedness` verified false alarm is `0.149`
  against a `0.05` gate, so the route remains blocked. This is the desired
  distinction: source provenance can pass while route quality still fails.
- `analyze_retrieval_route_gaps.py` reads the verified-record sidecar for that
  blocked route and records why: all 556 records end as `insufficient_evidence`,
  302 have no retrieval hits, and 254 use retrieval but never reach supported or
  refuted status. All 925 hits come from Wikidata `P36` capital facts, so the
  next iteration should expand source predicates or add a structured Wikidata
  verifier rather than only tuning lexical overlap thresholds.
- `build_wikidata_qa_corpus.py` adds the first structured Wikidata bridge by
  converting `P36` fact documents, or a template JSON of multiple properties
  such as `P36`/`P37`/`P38`, into label-free `QuestionAnswerVerifier` corpora.
  This lets covered statements be supported or refuted by exact structured
  values through the existing `retrieval_structured_qa` route, while preserving
  the earlier blocked result for open-domain TruthfulQA coverage.
- `run_wikidata_structured_qa_route_workflow.py` turns that QA corpus into a
  covered-facts route benchmark. The current `P36`/`P37`/`P38` artifact builds a
  balanced `718`-row true/false score dump, selects `structured_qa` for all
  rows, supports all `359` true facts, refutes all `359` swapped-answer false
  facts, and records decision accuracy `1.0` with false-supported rate `0.0`.
  Route summaries now include `property_metrics` keyed by Wikidata property id;
  `compare_route_baselines.py`, the release-candidate required-route gate, and
  the external-evidence comparator can all fail closed on per-property record
  count and support/refutation quality gates. This prevents aggregate covered
  fact metrics from hiding a weak predicate slice.
  This promotes structured QA as the correction route for KG-covered facts, not
  as evidence of broad open-domain retrieval coverage.
- `StructuredFactVerifier` adds the next property-level route: natural-language
  claims are first projected into simple triples, then checked against the same
  structured Wikidata facts. The route now handles common paraphrases such as
  possessive and subject-first fact statements, fact aliases, and multi-object
  list claims before falling back to refutation for known subject/predicate
  mismatches. The current structured-fact artifact covers `718` balanced rows,
  selects `structured_fact` for all rows, supports all `359` true
  natural-language facts, refutes all `359` swapped-answer false facts, and
  keeps false-supported rate `0.0`. This closes the gap between QA-shaped
  metadata and normal generated claims for KG-covered properties while still
  limiting the claim to covered structured facts.
- `run_wikidata_structured_qa_route_workflow.py --fact-claim-style
  paraphrase_robustness` adds the route-level robustness replay for the same
  KG-covered facts. The artifact expands canonical claims into possessive,
  subject-first, currency-use, and multi-object-list surface forms, producing
  `2868` balanced rows (`1434` true / `1434` false). It selects
  `structured_fact` for every row, reaches decision accuracy `1.0`, keeps
  false-supported rate `0.0`, and verifies the manifest recursively. This is
  surface-form robustness for known structured facts, not evidence of broader
  open-domain coverage.
- The l80 provenance matrix at
  `artifacts/truthfulqa-l80-retrieval-corpus-provenance-audit/` verifies both
  current local corpora. The correct-answer corpus fails the `grounding` role
  but passes `controlled_baseline`; its exact answer copy rate is `0.514`, so it
  remains a controlled local baseline. The answer-echo corpus fails `grounding`
  with exact answer copy rate `0.996` and claim-id link rate `1.000`, but passes
  `stress_control`. No current local corpus is external/domain-shift ready.

Added a text/length redline baseline for detector claims:

- `build_text_baseline_score_dump.py` appends answer length, claim length,
  question/answer lexical overlap, negation, and number-count signals to
  statement-bearing score dumps without rerunning models or adding dependencies.
- `artifacts/truthfulqa-l80-text-baseline-comparison/` evaluates those cheap
  controls through the same `eval_score_ensemble.py` conformal pipeline used for
  internal and verifier signals.
- At alpha `0.100`, `truth_proj` remains strongest for Qwen (`0.279` detection,
  `0.091` false alarm) and SmolLM2 (`0.229` detection, `0.095` false alarm).
  The cheap controls are near-random: `answer_token_count` AUROC `0.519`,
  `claim_token_count` AUROC `0.527`, and low `question_answer_token_overlap`
  triggers no detections under the calibrated gate.
- This does not improve product behavior directly; it prevents overclaiming by
  requiring future verifier/retrieval/selfcheck features to beat simple text
  artifacts under the same false-alarm budget.

Added the missing direct selfcheck signal bridge:

- `build_selfcheck_signal_score_dump.py` turns aligned sampled responses into
  `selfcheck_support_rate`, `selfcheck_refute_rate`, `selfcheck_disagreement`,
  coverage, and overlap score columns using the existing dependency-free
  `SelfConsistencyVerifier`.
- This lets future `eval_truthfulqa.py --dump-inside-samples` or external sample
  artifacts be evaluated as standalone conformal signals, compared against the
  text/length redline, and then optionally fused with representation geometry.
- Current committed l80 artifacts do not contain sampled response text, so this
  is an implementation bridge rather than a new l80 performance claim.

Added a direct selfcheck-signal fusion workflow:

- `run_selfcheck_signal_fusion_workflow.py` wraps the direct bridge into one
  no-model workflow: sampled responses -> selfcheck-enhanced score dumps ->
  sample-quality gate -> `eval_score_ensemble.py` report -> optional
  geometry-by-selfcheck fusion artifacts -> artifact manifest verification.
- `export_inside_diagnostics_samples.py` recovers sampled texts from an existing
  `eval_truthfulqa.py --inside-diagnostics-cache` artifact when the score dump
  was not written with `--dump-inside-samples`, and writes a manifest
  fingerprinting the score dump, cache, and exported samples.
- `plan_selfcheck_sample_collection.py` preflights the same aligned samples
  before fusion, listing records below the target sample count, total deficit,
  sample-quality gate projection, and rerun commands. This turns the current
  sample-quality failure into an explicit collection plan instead of another
  ambiguous negative result.
- `run_selfcheck_signal_fusion_workflow.py` writes those per-run collection
  plans into the workflow report and artifact manifest by default, so selfcheck
  promotion evidence now includes both the fusion outcome and the concrete
  sample-deficit rerun plan.
- This is the preferred next replay path when aligned multi-sample generations
  are available, because it tests self-consistency as a calibrated signal before
  mixing it into verifier sidecars or product policy.
- The current SmolLM2 l20 replay at
  `artifacts/smollm2-l20-direct-selfcheck-signal-fusion/` is a negative result:
  cache export matches 77/154 triggered records, only 25 records have at least
  two non-empty samples before alignment/deduplication, and the workflow
  sample-quality gate fails after alignment/deduplication with only 17/154
  usable two-sample records, coverage `0.110`, average samples per record
  `0.416`, and not-applicable rate `0.890`. At alpha 0.10 `truth_proj` remains
  stronger (`AUROC 0.682`, detection `0.178`) than the best
  geometry-by-selfcheck fusion (`AUROC 0.561`, detection `0.096`). This is a
  sample-quality gate failure.

Added spectrum-to-sweep layer-selection audit tooling:

- `compare_spectrum_layers.py` consumes `eval_truthfulqa.py --include-layer-spectra`
  reports plus saved layer/score sweep reports and checks whether dependency-free
  spectrum heuristics such as `max_spike_count`, `max_effective_rank`,
  `max_participation_ratio`, and `max_top_eigenvalue_to_mp_upper` land in the
  calibrated AUROC top-k.
- The report records per-heuristic top-k hit rates, exact-best rates, AUROC
  regret, layer gap, and a transparent recommended heuristic when the evidence
  supports one. This turns the Marchenko-Pastur/effective-rank idea into a
  falsifiable post-hoc audit before using it as a layer-selection shortcut.
- The cache-only l80 replay at
  `artifacts/truthfulqa-frontier-spectrum-layer-selection/` fingerprints two
  spectrum reports, two sweep reports, and the comparison report. The best
  heuristic is `max_top_eigenvalue_to_mp_upper`, but it hits the `truth_proj`
  AUROC top-2 layer in only 1/2 runs, with exact-best rate `0.5`, mean AUROC
  regret `0.0077`, and report status `fail`. Current evidence does not justify
  replacing calibrated layer sweeps with spectrum heuristics; spectrum is a
  candidate layer-band prior only.

Added a conservative layer-band selector audit:

- `compare_layer_band_selectors.py` consumes intrinsic-dimension reports,
  spectrum reports, and saved layer/score sweep reports, then evaluates
  `intrinsic:R`, `spectrum:HEURISTIC:R`, and
  `union:ID_R:HEURISTIC:SPEC_R` candidate bands against the calibrated best
  sweep layer.
- The cache-only l80 report at
  `artifacts/truthfulqa-frontier-layer-band-selection/` fingerprints the two
  spectrum reports, the intrinsic-dimension report, both sweep reports, and the
  comparison output.
- On the current Qwen/SmolLM2 l80 evidence, the recommended strategy is
  `spectrum_max_top_eigenvalue_to_mp_upper_radius_1`: it contains the best
  `truth_proj` layer for both models, has zero AUROC regret, and reduces the
  monitored sweep from 5 layers to 2 layers on average. Intrinsic radius 2 also
  passes, but averages 4 of 5 layers.
- This promotes a cost-reduction prior for where to sweep next. It still does
  not promote spectrum or intrinsic dimension as a standalone calibrated risk
  signal or exact deployment-layer oracle.

Wired the layer-band prior into the calibrated-observability workflow:

- `run_calibrated_observability_workflow.py --sweep-layers-from-band-report`
  reads a `compare_layer_band_selectors.py` report, selects the recommended
  strategy by default, matches the run by `--model` or explicit
  `--sweep-band-run`, and forwards the candidate layers to
  `eval_truthfulqa.py --sweep-layers`.
- `--sweep-band-target-layer best|band_best|first` can set the primary
  `eval_truthfulqa.py --layer` from the selected report, so the workflow does
  not accidentally keep a target layer outside the selected band.
- `--sweep-band-expand-radius` expands sparse candidate bands into denser local
  grids for the next replication pass; on the current Qwen l80 report,
  `[-10,-8]` with radius 1 becomes `[-11,-10,-9,-8,-7]`.
- The workflow report and artifact manifest now record the source report,
  selected strategy/run, candidate layers, best-layer-in-band status, and AUROC
  regret. This makes the layer-band prior reusable while preserving calibrated
  sweep and conformal artifact generation as the decision point.

Wired the same layer-band prior into the multi-model frontier workflow:

- `run_truthfulqa_frontier_workflow.py --sweep-layers-from-band-report` now
  passes the selector report to each calibrated-observability cell, with
  `--sweep-band-run-template` defaulting to `{cell}` so rows like `qwen05-l80`
  and `smollm2-l80` resolve independently.
- `--sweep-band-scales` keeps a report scoped to the intended scale, and
  `--sweep-band-expand-radius` / `--sweep-band-target-layer` are forwarded to
  every selected cell.
- A dry-run against the current l80 artifact resolves Qwen to target layer `-10`
  with dense sweep `[-11,-10,-9,-8,-7]`, and SmolLM2 to target layer `-16`
  with dense sweep `[-17,-16,-15,-14,-13]`.

Added stricter triple-evidence audit observability:

- `TripleSlotEvidence` now records the expected slot value, matched/missing
  tokens, source, and evidence label for each subject, predicate, and object
  slot checked by `TripleEvidenceVerifier`.
- `TripleEvidenceAuditReport` now carries claim-level covered/missing slot
  counts and per-slot coverage summaries. This does not promote a stronger
  extractor by itself; it makes structured-fact, retrieval, and world-model
  route failures easier to audit before adding heavier extraction dependencies.

Added dependency-free triple extractor plug-ins and eval harness:

- `RegexTriplePattern`, `RegexTripleExtractor`, `LookupTripleExtractor`, and
  `CompositeTripleExtractor` add configurable extraction slots between the
  default rule-based parser and future learned or external fact extractors.
  `LookupTripleExtractor` replays offline prediction files by claim id or text,
  so GLiNER/OpenIE/LLM-json extractors can be evaluated without becoming core
  dependencies.
- `StructuredFactVerifier` can now receive an injected extractor, so new
  extraction templates can be evaluated behind the same KG-covered correction
  route without changing route semantics.
- `benchmarks/eval_triple_extraction.py` compares rule-based, regex,
  regex-with-rule-based fallback, composite, and offline external-prediction
  lookup extractors on labeled triples with exact precision, recall, F1, and
  bounded error examples. External prediction parsing now preserves explicit
  empty `triples: []` outputs, so adversarial negative controls can represent
  "no extraction" without being coerced into malformed triple predictions.
- `benchmarks/run_external_triple_extractor_handoff.py` adds the command
  boundary for actual learned/OpenIE/LLM-json extractors: it writes label-free
  `claim_id`/`text` requests, invokes a local command with `{input}` and
  `{output}` placeholders, evaluates the returned offline predictions through
  the same exact and subgroup false-positive gates, and can register a verified
  handoff manifest. This still makes no quality claim for any specific learned
  extractor until a real extractor run promotes on the matrix.
- `benchmarks/run_external_triple_extractor_matrix_handoff.py` lifts that
  command boundary to cross-corpus release evidence: it builds deterministic
  per-corpus fixtures, runs one or more external commands over label-free
  requests, gates each prediction file, then feeds the outputs into
  `run_triple_extraction_fixture_matrix.py` so release candidates can require
  external-prediction count, corpus coverage, and mean best external F1.
- `benchmarks/fixtures/triple_extraction_records.json`,
  `benchmarks/fixtures/triple_extraction_regex_patterns.json`, and
  `benchmarks/triple_extraction_smoke.py` add a versioned extractor fixture and
  CI-gated smoke comparison. The current fixture keeps the heavier extractor
  question measurable: regex-with-fallback must beat default rule-based exact F1
  before templates are promoted into verifier routes.
- `benchmarks/build_triple_extraction_fixture.py` now turns structured fact
  corpora into larger labeled extraction fixtures and default regex templates,
  so KG-covered Wikidata/property corpora can scale the extractor benchmark
  without introducing a learned extractor dependency.
- `benchmarks/run_triple_extraction_fixture_workflow.py` wraps that builder into
  a release-evidence workflow: it writes generated records, pattern payloads,
  rule-based/regex/composite reports, optional external-prediction reports, a
  promotion summary, and an artifact manifest. This makes the extractor slot
  auditable as a benchmarked route component rather than a hand-tested parser
  hook.
- The fixture builder now covers a small second predicate family beyond
  country-core facts: headquarters location (`P159`), manufacturer (`P176`),
  and inception/founding date (`P571`). `benchmarks/run_triple_extraction_fixture_matrix.py`
  runs the generated-fixture workflow across multiple corpora and blocks
  promotion unless enough corpora promote and the combined fixtures meet a
  predicate-diversity floor. This is the release-evidence bridge from a local
  parser improvement to a cross-domain extractor slot.
- `benchmarks/compare_release_candidates.py` and
  `benchmarks/run_release_candidate_registry_workflow.py` can now require that
  matrix with `--triple-extraction-fixture-matrix`, plus optional corpus and
  predicate diversity floors. They can also require external-prediction count,
  external-prediction corpus coverage, and mean best external F1 when a learned
  or external extractor is being treated as release evidence. This moves
  extractor evidence from a standalone benchmark into the same fail-closed
  release candidate and registered manifest gates as readiness, route,
  selfcheck, world-model, feedback, and adapter-family evidence.
- `benchmarks/compare_release_candidates.py` and
  `benchmarks/run_release_candidate_registry_workflow.py` can also require a
  counterfactual verifier audit report or registry key, gating promotion on
  audited pair count, pass rate, and false-invariance rate while recording the
  report and manifest in the release artifact.
- The first real cross-corpus matrix is now materialized at
  `artifacts/wikidata-cross-corpus-triple-extraction-fixture-matrix/`. It
  combines the 359-fact country-core Wikidata corpus with the fetched
  `organization_product_core_facts` source docs, promotes both generated
  workflows, covers six predicates, reaches mean best F1 `1.000`, and verifies
  recursively. This closes the local extractor-evidence item for covered
  predicates; open-domain extraction still requires broader corpora and likely a
  learned or external extractor adapter.
- `build_triple_extraction_fixture.py` now supports six adversarial subgroups:
  negated near-miss records, predicate-confusion assertions that state the wrong
  property but still require extracting the stated predicate, non-assertive
  quoted/questioned fact mentions, ambiguous/multi-object wording,
  temporal-qualified wording, and metalinguistic/comparison context. The
  adversarial cross-corpus matrix at
  `artifacts/wikidata-cross-corpus-triple-extraction-adversarial-matrix/`
  first exposed real false positives in the extractor stack; after applying a
  shared blocked-context guard to both rule-based and regex paths, the same
  country-core plus organization/product corpora add `367` records to each
  adversarial subgroup, both workflows promote, best F1 remains `1.000`,
  predicate-confusion F1 is `1.000`, and all zero-expected subgroup
  false-positive rates are `0.000`. This promotes simple negative-context
  robustness for covered KG templates while preserving the broader open-domain
  extraction boundary.

Added the first stdlib-only external retrieval service shell:

- `HTTPJSONRetriever` calls a caller-provided HTTP JSON search endpoint and
  normalizes list payloads or `hits` / `results` / `documents` objects into
  `RetrievalHit` values. This is adapter plumbing only: no network endpoint is
  bundled, and no external retrieval quality claim is made.
- `RetrievalActionExecutor` now fails closed when a retriever raises, returning
  a structured failed `ActionResult` with per-query errors instead of letting
  adapter exceptions escape the control loop. That keeps future RAG/search
  service failures visible in product traces and prevents failed retrieval from
  being mistaken for supported evidence.
- `ProvenanceFilteredRetriever` wraps any local, SQLite, or HTTP retriever and
  enforces the first evidence trust boundary before hits become verifier
  context: source can be required, source prefixes can be allow/deny-listed,
  low-score hits can be dropped, required metadata tags can be checked, and
  per-source caps can reduce single-source dominance. Accepted hits carry the
  filter policy in metadata for trace/replay audits. This still makes no
  external-RAG quality claim; it only prevents untrusted retrieval plumbing from
  silently becoming support evidence.
- `build_evidence_fixture.py`, `run_local_retrieval_route_workflow.py`, and
  `run_verifier_signal_fusion_workflow.py` now expose the same provenance filter
  knobs and persist them in fixture input provenance, workflow reports, route
  manifests, claims-cache keys, and registry metadata. This moves the filter
  from an adapter-only primitive into the reproducible route/fusion evidence
  path needed for external or domain-shifted retrieval experiments.
- `compare_route_baselines.py`, `compare_release_candidates.py`, and
  `run_release_candidate_registry_workflow.py` can now require that a selected
  or required retrieval route records a source-requiring provenance filter with
  expected allowed source prefixes, required metadata tags, and minimum
  retrieval score. This turns external evidence trust policy into a fail-closed
  route/release gate instead of only descriptive metadata.
- `compare_external_evidence_baselines.py` now joins three previously separate
  controls into one post-hoc release artifact: registered route promotion,
  answer-echo retrieval stress control, and text/length redline score-ensemble
  comparison. It is intentionally dependency-free and does not rerun models;
  missing reports, ambiguous run pairing, or underperforming candidate signals
  block the comparison. It can now write and verify a recursive artifact
  manifest and register the comparison as a reusable `report:*:*` handoff.
- The same comparator now has an optional covered-facts gate for structured
  Wikidata/KG correction evidence: `--require-covered-facts-route` requires the
  selected route to expose a promoted `wikidata_structured_qa_route_workflow`
  summary, validates allowed routes plus source/true/false record counts, and
  records that scope in the manifest/registry metadata. This separates
  property-level correction evidence from broad retrieval-grounding claims.
- `compare_release_candidates.py` and
  `run_release_candidate_registry_workflow.py` can now require that promoted
  external-evidence comparison artifact by direct path or registry key as a
  release gate and carry its report, source/record provenance, recommended
  route, route-gate status, and text-redline status into manifest and registry
  metadata.
- `ProductPromotionContract`, `export_product_promotion_contract.py`, and
  `ProductRuntimeEvidenceBundle` now preserve that external-evidence
  baseline-comparison handoff through deployment artifacts and bounded trace
  metadata. This closes the local release-to-runtime provenance path without
  adding a network retriever, database, or verifier dependency.
- `product_runtime_metrics()` and `run_product_runtime_baseline.py` now aggregate
  the same handoff across product traces: coverage, source/status counts,
  recommended routes, route-gate pass counts, text-redline pass counts, and
  text-redline run counts are emitted in runtime-baseline reports, manifests,
  and registry records. This makes external-evidence release provenance auditable
  as a product runtime property, still without claiming new grounding evidence.

Added the first monitor-first tool-selection audit layer:

- `ActionAuditPolicy`, `ActionAuditReport`, and `audit_action_requests()` check
  planned actions before executor dispatch without blocking execution by
  default. The audit compares the selected `RiskDecision`, `ClaimVerificationPlan`,
  and planned `ActionRequest` payloads to flag missing decision actions, omitted
  retrieval actions when the plan emitted retrieval queries, retrieval payloads
  that the local retrieval executor cannot execute, malformed `execute_tool`
  names or argument objects, and claim ids that do not exist in the plan.
- `ProductTrace.to_bounded_dict()` and `product_runtime_metrics()` now expose
  `action_audit` summaries and metrics. This creates a release/replay hook for
  tool-bypass and parameter-hallucination failures before introducing learned
  tool-selection models or external tool routers.
- `ProductTrace.action_execution_summary()` now also compares planned actions
  against recorded `ActionResult` payloads by action type and request id, and
  `run_product_runtime_baseline.py` aggregates missing, unexpected, and
  request-id-mismatched action results. This makes executor bypass and dropped
  side-effect results visible before adding stricter release gates.
- `run_product_trace_replay_workflow.py` can now enforce those
  action-execution alignment metrics as an optional fail-closed gate. The gate
  is off by default, writes `action-execution-gate.json` when configured, and
  blocks the replay workflow when missing results, unexpected results, or
  request-id mismatches exceed configured thresholds or required metrics are
  unavailable.
- `run_product_runtime_baseline.py` now aggregates action-audit error,
  missing-retrieval, missing planned retrieval-query coverage,
  malformed-payload, unexpected-action, and unknown-claim-id rates, and
  `run_product_trace_replay_workflow.py` can enforce them as an optional
  action-audit release gate. The gate is off by default, writes a child
  `action-audit-gate.json` report when configured, and blocks replay workflows
  when configured rates exceed thresholds or required audit metrics are missing.
- `compare_release_candidates.py --require-product-trace-action-audit-gate`
  and `--require-product-trace-action-execution-gate` now fail closed unless a
  supplied product-trace-replay workflow promoted the corresponding child gate.
  The `frontier_audit` profile enables both by default, now also defaults to
  the registered covered-facts external-evidence handoff and external-prediction
  triple-extraction matrix gates, and
  `run_release_candidate_registry_workflow.py` records the child gate reports
  plus action-audit/action-execution rates in release manifests and registry
  metadata.
- This layer is intentionally dependency-free and observational. It does not
  claim to solve tool selection; it makes tool-routing mistakes measurable so
  future internal-representation tool-selection or world-model-corrected
  executor policies have a stable target metric.

## Next Research-to-Code Candidates

1. Run denser layer-grid calibrated-observability replays through `audit_layer_band_replication.py`; only promote a selector preset after the audit passes across at least two model families.
2. Run an actual learned/OpenIE/LLM-json extractor command through `run_external_triple_extractor_matrix_handoff.py` on the Wikidata adversarial matrix, then add broader non-template corpora before claiming open-domain extractor robustness.
3. Materialize a real `frontier_audit` release candidate with the registered covered-facts external-evidence handoff and external-prediction triple matrix artifact, then export the resulting promotion contract.

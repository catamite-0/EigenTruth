# EigenTruth Frontier Research Notes

Date: 2026-07-01

## Current Frontier Direction

EigenTruth should move from hallucination detection alone toward calibrated participation control: decide when a model should answer, when it should retrieve/verify, and when it should abstain. The strongest near-term fit is conformal abstention because it gives a post-hoc, dependency-free layer with finite-sample risk/coverage semantics that can sit on top of existing EigenTruth scores.

## Evidence Checked

- Geometry-Calibrated Conformal Abstention for Language Models (arXiv:2604.27914, submitted 2026-04-30) frames abstention as a post-hoc conformal layer over prediction confidence, with finite-sample participation and correctness guarantees, and uses representation geometry to align confidence with model ignorance.
- INSIDE (arXiv:2402.03744) supports EigenTruth's internal-state route: hidden-state embeddings retain hallucination signals, and EigenScore measures semantic consistency/diversity in dense representation space.
- Layer-wise Semantic Dynamics (arXiv:2510.04933), Contextual Perturbation and Representation Drift (arXiv:2505.16894), and MultiHaluDet (arXiv:2605.24919) point in the same direction for trajectory-style hidden-state analysis: hallucination detection should examine representation dynamics, not only static final-layer embeddings. This justifies the E7 real-data replay harness. The limit-128 gpt2/SmolLM2 follow-up shows trajectory convergence is not yet robust enough as a standalone detector: gpt2 reaches AUROC 0.608, while SmolLM2 reaches only 0.560 and the fail-closed evidence gate remains blocked. The aligned ablation matrix is more nuanced: trajectory improves the best gpt2 fusion candidate but hurts the best SmolLM2 candidate. `SignalSelectionPolicy` and selected fusion artifacts now turn that result into an explicit conditional artifact path, so trajectory remains a model/run-specific fusion feature rather than a default release signal.
- ICR Probe (ACL 2025 / arXiv:2507.16488) strengthens the same conclusion from a different angle: hidden-state update dynamics across layers can carry hallucination signal that static hidden-state probes miss. EigenTruth now implements a dependency-free, training-free profile summary over existing `resid_update_norm_by_layer` values (`resid_update_profile_area`, `resid_update_profile_peak`, `resid_update_profile_late_mass`, `resid_update_profile_concentration`). This is not a full ICR Probe replication because it does not decompose attention/MLP module contributions or fit the paper's learned probe; it gives the current score-dump/conformal/release-gate pipeline a reproducible first approximation before adding learned ICR features.
- Beyond Final Answers / Trajel (arXiv:2605.24219, 2026) argues that agentic hallucination should be audited over the whole Thought/Action/Observation trajectory, not only the final answer, and uses a five-type taxonomy: factual, referential, logical, procedural, and scope-based. This maps directly to EigenTruth's `ProductTrace` boundary because the trace already carries claims, verifier outputs, decisions, action requests/results, and final answers.
- Cascading Hallucination in Agentic RAG / CHARM (arXiv:2606.04435, 2026) strengthens the trace-level requirement: an early retrieval/tool/observation failure can propagate through later reasoning and final answers even when each individual step looks locally plausible. EigenTruth should therefore audit not only terminal claim status, but whether upstream evidence failures or empty retrieval results were later treated as support.
- From Agent Traces to Trust / Evidence Tracing and Execution Provenance in LLM Agents (arXiv:2606.04990v3, 2026) frames trustworthy agent behavior as a typed provenance graph over retrieved evidence, tool outputs, observations, intermediate claims, actions, and final answers. This supports adding a local `ProductTrace` provenance graph beside receipts and trajectory audits: supported claims should expose concrete evidence units or action-result references, and final-answer evidence should link back to known claims and execution artifacts.
- Two Pathways to Truthfulness (arXiv:2601.07422, accepted ACL 2026 Main Conference) argues that internal truthfulness cues split into a question-anchored pathway using question-answer information flow and an answer-anchored pathway using evidence from the generated answer itself. EigenTruth now implements dependency-free prompt/answer hidden-state pathway proxies, an optional attention-flow pathway readout from returned model attentions, local pathway-intervention analysis helpers for prompt/answer attention knockout, temporary model-side activation ablations, source-token activation patch reruns, and a rerun score-dump comparator for before/after intervention evidence. This is still not a full paper-faithful causal replication because attention-kernel intervention and larger model evidence remain open; it gives the score-dump/conformal/fusion pipeline cheap pathway-disagreement, pathway-flow, and mechanism-experiment evidence scaffolding before heavier causal-intervention runs.
- SelfCheckGPT (arXiv:2303.08896) and FactSelfCheck (arXiv:2503.17229, EACL 2026 findings) support the sampling/self-consistency route. FactSelfCheck moves from sentence-level to fact-level graph/triple checks. EigenTruth now has a dependency-free first bridge through `FactSelfConsistencyVerifier`: caller-supplied sampled responses are converted into triples, exact triple matches support claim facts, same subject/predicate object conflicts refute them, and missing facts remain insufficient evidence. `eval_verifier_ensemble.py --enable-fact-selfcheck` can route sampled-response fixtures through this fact-level check before sentence-level selfcheck fallback, and `build_verifier_signal_score_dump.py` can convert sidecar reports into `fact_selfcheck_*` conformal/fusion signals. `run_verifier_signal_fusion_workflow.py --enable-fact-selfcheck` now closes that into a no-model artifact loop from aligned samples and statement/sample triples through sidecar, enhanced score dump, fusion report, geometry artifact, and manifest verification; `--fact-selfcheck-gate` fails closed when executed/decided coverage, not-applicable rate, or triple density is too weak for promotion. `--fact-selfcheck-early-stop` now records budget-aware fixed-threshold stopping for fact-level samples, including processed/skipped samples and per-triple processed rates. This is not a paper-complete FactSelfCheck replication; stronger extraction and real aligned multi-sample evidence remain the next step.
- The product handoff now has an optional `fact_selfcheck_gate` evidence group: `export_product_promotion_contract_evidence_handoff.py --fact-selfcheck-signal-fusion ... --required-groups fact_selfcheck_gate` requires a promoted, manifest-verified gate before carrying fact-level selfcheck evidence into a deployable promotion contract. This keeps fact-level sampling evidence visible to release audits without making weak or disabled sample evidence look production-ready.
- Semantic Energy (arXiv:2508.14496) supports energy-style uncertainty beyond entropy. EigenTruth already has lightweight semantic-energy proxies; a future step is to compare them against conformal abstention and route-cost gates.
- Adaptive Conformal Semantic Entropy (arXiv:2605.04295, 2026) directly supports EigenTruth's ACSE-style path: semantic dispersion should be adaptively inflated by cluster/sample features and then calibrated, rather than using raw entropy as a universal threshold. EigenTruth's current `AdaptiveScoreTransform` / `AdaptiveConformalCalibrator` and `inside_semantic_energy` score-dump features are the dependency-free version of that idea; the missing release step is to compare adaptive artifacts against redline entropy and route-cost gates before product promotion.
- Principled Detection of Hallucinations in Large Language Models via Multiple Testing (arXiv:2508.18473) argues that no single hallucination score is likely to cover all failure modes, and combines several calibrated scores through conformal p-values plus multiple-testing control. This directly fits EigenTruth's multi-signal posture: geometry, confidence, semantic-energy, retrieval, verifier, and world-model evidence should remain individually inspectable while sharing one global false-alarm budget.
- CiteCheck (arXiv:2605.27700) shows that citation hallucinations often appear as small metadata drift rather than fully fabricated references. This supports a separate citation-integrity route before broad retrieval: DOI, arXiv id, URL, author/year, title, and local reference labels should be checked against a trusted citation catalog instead of treated as ordinary lexical groundedness.
- Internal Representations as Indicators of Hallucinations in Agent Tool Selection (arXiv:2601.05214) frames agent hallucination as incorrect tool selection, malformed parameters, and tool bypass. This supports keeping tool-route intent explicit in `ClaimVerificationPlan` instead of only checking final text.
- World-Model-Augmented Web Agents with Action Correction (arXiv:2602.15384) uses consequence simulation and action correction before risky actions. This supports EigenTruth's world-model route as a post-draft verifier and pre-action correction adapter rather than a core dependency. The product-side implementation now exposes `ProductTrace.world_model_summary()` and bounded `summaries.world_model` fields so world-model evidence can be audited by adapter/reference, conflict path, low agreement, and trace-gap rates instead of only appearing as benchmark score columns.
- TokenHD (arXiv:2605.12384) and related span/token-level work point toward finer localization of hallucinations. EigenTruth's current lightweight equivalent is claim-level risk localization, route budgeting, and trace evidence; learned token-level detectors remain out of scope until the dependency and training boundary is explicit.
- Pre-generation hallucination detection with soft targets (arXiv:2606.21917) reinforces the current layer/sweep direction: hallucination risk is better treated as a probability estimated from internal representations than as a single hard decoded label.
- Entropy Alone is Insufficient for Safe Selective Prediction in LLMs (arXiv:2603.21172) and the UQ-as-clustering critique (arXiv:2605.19220) both argue against relying on entropy/self-consistency alone. EigenTruth should keep combining internal geometry with correctness/verifier/world-model evidence and deployment-facing selective metrics.
- Single-decode first-token confidence work (arXiv:2605.05166) points to a cheap baseline before multi-sample routes: top-k entropy at the first answer-token prediction can be logged from the same forced-answer pass and compared against internal geometry, INSIDE/selfcheck, and verifier signals.
- Detecting Representational Inconsistencies for Factual Truthfulness (arXiv:2601.14210, 2026) reinforces the single-decode internal-probe route: intermediate layers can expose confidence/truthfulness signals that are not visible in the final output. This supports keeping layer/score sweeps, residual-update profiles, and hidden-evidence selection as first-class artifacts instead of collapsing everything into post-hoc text verification.
- High-certainty hallucination work such as CHOKE / "LLMs Hallucinate with Certainty Despite Knowing the Answer" (arXiv:2502.12964, EMNLP Findings 2025) is a warning against entropy-only or confidence-only product gates. EigenTruth should continue requiring independent verifier, retrieval, counterfactual, or world-model evidence for release promotion, and should treat certainty signals as one calibrated axis rather than proof of factuality.
- DECK (arXiv:2606.02289) reframes hallucination errors by detectability signature rather than content type, splitting errors along inter-sample consistency and token-level confidence into Drift, Entrenched, Confabulation, and Knotted. This directly fits EigenTruth's score-dump posture: consistency signals, white-box confidence signals, and independent verifier/world-model routes should be evaluated for complementary blind spots, not only aggregate AUROC.
- Global-Local Uncertainty / GLU (arXiv:2606.09875) argues that token-level local entropy and hidden-state global geometry can be near-orthogonal and recover different failure regimes. This supports keeping score-dump fusion and detectability reports axis-aware, so geometry, confidence, self-consistency, and verifier evidence are not collapsed into one uninterpretable scalar too early.
- Counterfactual Probing for Hallucination Detection and Mitigation (arXiv:2508.01862) supports adding perturbation sensitivity audits: robust verifiers should change status on entity, temporal, quantitative, or logical counterfactuals instead of staying invariant to false variants.
- Hallucination Detection and Mitigation in Large Language Models (arXiv:2601.09929) frames reliable systems as a root-cause-aware continuous improvement loop. This matches EigenTruth's release-gate posture: blocked evidence should produce targeted next experiments rather than a generic "add RAG" fix.
- Tool Receipts, Not Zero-Knowledge Proofs (arXiv:2603.10060) argues that interactive agents need low-latency signed/tool-execution receipts and claim-to-receipt checks. EigenTruth's action-audit/action-execution gates are the local analog; release reports should surface missing receipt-style metrics as first-class gaps.
- Retromorphic Testing with Hierarchical Verification for RAG (arXiv:2603.27752) treats RAG faithfulness as traceability from answer claims back to context-side evidence spans. This supports keeping claim/span localization, triple/slot evidence, and source-side evidence coverage explicit in ProductTrace and release-drift reports.
- HIVE (arXiv:2604.26139) shows a newer trajectory-level direction for diffusion LMs: select sparse hidden evidence under a budget and condition a verifier on it. EigenTruth should keep trajectory/pathway evidence as selected, budgeted evidence artifacts before making any default routing claim.
- Look Again Before You Abstain: Budgeted Conformal Evidence Acquisition (arXiv:2606.16667) strengthens the product-control lesson that evidence acquisition changes the policy distribution: guarantees should be calibrated and monitored on the post-acquisition policy, not only on a pre-retrieval uncertainty score.
- Anytime-Valid Conformal Risk Control (arXiv:2602.04364) points toward risk control under ongoing streams and optional stopping. EigenTruth's current lightweight implementation is deliberately narrower: an accepted-error Bernoulli e-process monitor over labeled feedback, plus the existing finite prefix alpha-spending monitor. It is an online drift alarm for a fixed deployed threshold, not a full anytime-valid CRC optimizer.

## Implemented This Continuation

Added an anytime-valid feedback risk monitor for evidence acquisition:

- `EvidenceAcquisitionAnytimeRiskStep`,
  `EvidenceAcquisitionAnytimeRiskMonitorState`,
  `EvidenceAcquisitionAnytimeRiskMonitorReport`, and
  `audit_evidence_acquisition_anytime_risk(...)` run a dependency-free mixture
  e-process over accepted post-acquisition errors.
- `EvidenceAcquisitionAnytimeRiskMonitorState.update(...)` consumes one labeled
  feedback record at a time and returns the new resumable state plus the
  per-record step payload; `save_json(...)` / `load_json(...)` make the monitor
  suitable for product feedback streams without holding the full history in
  memory.
- The deployed threshold is fixed; feedback is used only for monitoring, not
  for silent recalibration.
- The alarm threshold is `1 / monitor_alpha`, and a failed monitor reports the
  first record/accepted-row index where the mixture e-value crossed that line.
- `benchmarks/calibrate_evidence_acquisition_from_traces.py` now supports
  `--risk-monitor-mode anytime` or `both`, `--anytime-risk-monitor-json`, and
  repeatable `--risk-monitor-bet-fraction`; any failed prefix or anytime monitor
  sets workflow `status=blocked` and is represented in manifests/registry
  metadata.

Added receipt-style action result verification:

- `ActionReceipt` serializes the action, status, request id, output fingerprint,
  result fingerprint, optional request fingerprint, issuer, key id, and signing
  metadata for one executed action result.
- `ActionReceiptSigner` issues and verifies local HMAC-SHA256 receipts without
  adding a mandatory dependency or requiring a network service.
- `ReceiptActionExecutor` can wrap an existing executor and attach
  `metadata.action_receipt` to every returned `ActionResult`, including
  fail-closed wrapper failures.
- `verify_action_receipt(...)`, `action_result_fingerprint(...)`, and
  `action_request_fingerprint(...)` expose the low-level primitives needed to
  compare model-visible tool claims against actual execution outputs in later
  agent audits.
- `ProductTrace.action_receipt_summary()` and bounded
  `summaries.action_receipts` report receipt coverage, signed/unsigned counts,
  invalid receipts, and result-fingerprint mismatches without needing the HMAC
  secret. This is the local EigenTruth bridge toward receipt-backed agent
  verification.
- `ReceiptClaimSupportPolicy`, `ReceiptClaimSupportReport`, and
  `audit_receipt_claim_support(...)` add the first claim-to-receipt structural
  audit: when a claim or final-answer evidence item explicitly cites an action
  request id, result fingerprint, or output fingerprint, the audit requires a
  matching receipt-backed action result with an accepted status. It is not
  natural-language entailment; it catches fabricated or stale tool references
  while leaving unreferenced claims to the normal verifier/retrieval/world-model
  routes.

Added conformal multiple-testing signal aggregation:

- `directional_conformal_pvalues(...)` extends the existing conformal p-value primitive to mixed-direction native scores, so lower-is-anomalous signals such as reliability/projection scores can be calibrated without sign mistakes.
- `multiple_testing_conformal_report(...)` combines one runtime item across several signals with BY, BH, or Bonferroni correction and emits a JSON-ready `MultipleTestingHallucinationReport`.
- `MultipleTestingConformalCalibrator` / `MultipleTestingConformalArtifact` save the held-out normal calibration distributions for those signals, so a runtime item can call `artifact.decide(scores)` and reproduce the same global gate used in benchmark reports.
- `MultipleTestingGateConfig` lets `RiskController(..., multiple_testing_gate=artifact)` consume that artifact directly. By default it only gates decisions that would otherwise accept, records a `multiple_testing_gate` trace block, and fails closed to `clarify/unknown` when any required signal is missing or non-finite.
- The default method is BY because EigenTruth diagnostics are usually correlated rather than independent; BH and Bonferroni remain explicit options for experiments.
- `eval_conformal.py --save-multiple-testing-report --save-multiple-testing-calibration --multiple-testing-signals ...` now evaluates the same primitive over score dumps with seeded split-conformal calibration, reporting false alarm, coverage, detection, and per-signal rejection contributions while optionally exporting the reusable runtime artifact without changing the base conformal verdict.
- `examples/calibrated_control_demo.py --multiple-testing-gate ...` now loads that runtime artifact, feeds it into `RiskController`, and records gate provenance in the emitted `ProductTrace` or local registry record.
- This is a decision primitive, not a new detector: it lets existing and future geometry, confidence, semantic-energy, retrieval, verifier, and world-model scores keep separate traces while producing one global gate under a false-alarm budget.

Added GLU-style global-local uncertainty fusion:

- `global_local_uncertainty_scores(...)` exposes a dependency-free score helper that rank-calibrates hidden-state/global geometry signals and token-level/local uncertainty signals against normal calibration records, then applies a multiplicative gate by default.
- `eval_score_ensemble.py` now evaluates `product` geometry fusion by default alongside the existing interaction score, and reports it with `fusion_style=global_local_uncertainty`.
- This maps the GLU direction into EigenTruth's existing score-dump/conformal artifact path without adding a new mandatory dependency or claiming that entropy alone is sufficient. The intended use is to compare single-pass global geometry plus local token uncertainty against verifier, retrieval, selfcheck, and world-model correction signals before promotion.
- `eval_score_ensemble.py --confidence-signal ...` now adds a release-facing high-confidence miss audit for candidate single, ensemble, interaction, and GLU/product routes. The route gate keeps the conformal false-alarm check and blocks candidates whose high-confidence region still contains accepted false answers above the configured threshold, making semantic-energy or local-uncertainty fusion an auditable route candidate rather than a default product claim.

Added HIVE-style budgeted hidden-evidence selection:

- `HiddenEvidenceCandidate`, `HiddenEvidenceSelectionPolicy`, `HiddenEvidenceSelectionReport`, and `select_hidden_evidence_from_score_dump(...)` turn primary and layer-sweep diagnostic score dumps into sparse evidence-selection reports.
- Each candidate is rank-normalized within its `source/layer/signal` channel, respects `higher` or `lower` anomaly directions, and is budgeted by total items, record, layer, and score family.
- `benchmarks/select_hidden_evidence.py` writes the JSON report and can register it locally. `ClaimVerificationPlanner.plan(..., hidden_evidence=report)` now maps selected evidence onto matching claim ids or record indexes, promotes those claims under verification budgets, and stores evidence refs in route-hint metadata. `ProductTrace.to_bounded_dict()` exposes the hidden-evidence summary for replay. This keeps HIVE's "select sparse hidden evidence before conditioning a verifier" idea in EigenTruth's reproducible artifact path without adding a verifier dependency or claiming default routing behavior.

Added trace-level product trajectory audit:

- `TrajectoryHallucinationType`, `TrajectoryAuditIssue`, `TrajectoryAuditReport`, and `audit_product_trace_trajectory(...)` implement a dependency-free five-type trace audit over `ProductTrace` payloads.
- The audit reuses existing action-planning findings, adds action-result alignment failures, and checks verification/decision/final-answer consistency so accepted refutations, unsupported accepted claims, missing action results, request-id mismatches, and final-answer contradictions become structured findings.
- The audit is now cascade-aware: failed evidence-bearing upstream actions and empty retrieval results are traced into downstream supported claims or answered traces through request ids, producing explicit findings such as `accepted_after_failed_upstream_action`, `accepted_after_empty_retrieval`, `supported_claim_from_failed_action`, and `supported_claim_from_empty_retrieval`.
- `ProductTrace.to_bounded_dict()` now carries a compact `summaries.trajectory_audit` block, and `product_runtime_metrics(...)` exposes full-trace or bounded-summary trajectory counts for runtime baselines.
- `TrajectoryAuditReport.summary()` and runtime metrics expose `cascade_count` / `trajectory_audit_cascade_count`, so product baselines can distinguish ordinary structural issues from cascading-evidence propagation.
- Runtime baselines now aggregate trajectory-audit failed-trace/error rates plus factual/referential/logical/procedural/scope counts into reports, manifests, and registry metadata, and `compare_product_runtime_baselines.py` can fail closed on taxonomy-rate drift when explicit gates are configured.
- Trace-provenance summaries now roll up into runtime baselines, product runtime drift gates, replay workflow metadata, and an optional release-candidate requirement via `--require-product-runtime-drift-provenance-evidence`, covering claim/evidence/action-result/final-answer graph coverage plus missing-reference and unsupported-supported-claim rates.
- This is a monitor-first structural audit, not a learned trajectory detector: it gives agent/tool workflows a stable evidence schema while future internal-representation or learned trace classifiers remain optional adapters.

Added trace-level evidence provenance graphs:

- `TraceProvenanceNode`, `TraceProvenanceEdge`, `TraceProvenanceGraph`, `TraceProvenanceIssue`, `TraceProvenanceReport`, `build_trace_provenance_graph(...)`, and `audit_trace_provenance(...)` build a dependency-free typed graph over claims, verification results, local evidence strings, action requests, action results, retrieval hits, source labels, and final-answer evidence.
- The audit flags supported verification results with no local evidence or valid action-result reference, missing action-result references, failed referenced action results, final-answer evidence that points to unknown claims, and answered traces with claims but no final-answer evidence.
- `ProductTrace.to_bounded_dict()` carries a compact `summaries.provenance` block, and `product_runtime_metrics(...)` exposes provenance coverage, missing-reference counts, graph size, retrieval-hit/source counts, and final-answer evidence-reference rates for runtime telemetry.
- This is structural provenance, not semantic entailment: it proves the support path is present and locally auditable, while verifier, retrieval, receipt, triple-evidence, world-model, or future semantic-provenance adapters decide whether the evidence content is true.

Added DECK-style detectability taxonomy reports:

- `youden_j_threshold(...)` computes a dependency-free Youden's J split for a score axis where either higher or lower raw scores can mean healthier behavior.
- `deck_taxonomy_report(...)` combines a consistency-style score and a confidence-style score into Drift / Entrenched / Confabulation / Knotted cells, reporting all-sample counts, false-record distribution, blind-spot counts, and scorer families expected to catch each regime.
- `benchmarks/eval_detectability_taxonomy.py` reads existing JSON or JSONL score dumps with selected-column loading and writes a JSON report without loading a model.
- `benchmarks/run_truthfulqa_frontier_workflow.py` can now emit those taxonomy reports per frontier cell, add them to the top-level artifact manifest, and carry the paths forward into release-evidence comparison.
- The registered l80 replay `report:truthfulqa-frontier-qwen-smollm2-l80-detectability:0.1` reuses existing Qwen/SmolLM2 score dumps and writes per-cell taxonomy reports in 7.5s. With `eigenscore` and `nll_answer` treated as lower-is-risk axes, Qwen has entrenched false-rate `0.000`, while SmolLM2 has `89/306 = 0.291`.
- `benchmarks/analyze_detectability_blind_spots.py` turns a blocked taxonomy cell into row-level examples plus feature/question-type summaries. The registered SmolLM2 artifact `report:truthfulqa-frontier-smollm2-l80-entrenched-blind-spots:0.1` exports all 89 false entrenched records; the largest groups are definition/what (`39`), person (`13`), and choice (`8`) questions, with mean answer length `5.18` tokens.
- `benchmarks/audit_blind_spot_correction_routes.py` joins those blind spots with verifier sidecar rows. The registered SmolLM2 audit `report:truthfulqa-frontier-smollm2-l80-blind-spot-route-audit:0.1` shows the current `retrieval_structured_qa` route selects and refutes `3/89` entrenched false records, supports `0/89`, and leaves `86/89` outside the target route; this is a coverage gap, not a selected-route precision gap.
- `benchmarks/sweep_blind_spot_retrieval_queries.py` then localizes the gap to query construction under the controlled correct-answer corpus: `question_answer@0.65` refutes `87/89` with verified false alarm `0.000`, and `question_answer@0.5` refutes `89/89` with verified false alarm `0.000`, while `question@0.95` is rejected as a high-false-alarm negative control (`0.176`). This points to question-aware retrieval against real external/structured corpora as the next experiment.
- `benchmarks/compare_blind_spot_query_sweeps.py` closes that provenance loop: the same query sweep against Wikidata country-core-facts external retrieval and structured-QA retrieval documents refutes `0/89` blind spots, so the registered comparison `report:truthfulqa-frontier-smollm2-l80-query-sweep-provenance-comparison:0.1` is blocked with a `1.0` controlled-to-external generalization gap. The research target is now coverage expansion, not further controlled-corpus threshold tuning.
- `benchmarks/plan_blind_spot_evidence_expansion.py` turns that blocked result into the next work queue: `report:truthfulqa-frontier-smollm2-l80-blind-spot-evidence-expansion-plan:0.1` covers all 89 targets, with 65 high-priority records, 80 structured-fact recommendations, 65 structured-QA recommendations, 63 citation-retrieval recommendations, 41 counterfactual-probe targets, and 21 world-model/calculator targets. This is the evidence collection plan to execute before rerunning the provenance gate.
- The first target-specific Wikidata collection pass is now closed through a structured-QA covered-facts route. `fetch_blind_spot_wikidata_evidence.py` writes `292` label-free CC0 source docs over `10` properties, lexical retrieval still refutes `0/89` blind spots, but `build_wikidata_qa_corpus.py --auto-template-from-source` plus `run_wikidata_structured_qa_route_workflow.py --route structured_qa` promotes a covered-facts correction artifact with `584` balanced rows, decision accuracy `1.0`, and false-supported rate `0.0`. `audit_blind_spot_covered_fact_mapping.py` then shows the actual mapping gap: joined Wikidata facts exist for `37/89` blind spots, but only `10/89` are conservative correction candidates, while `5` support the model answer, `6` have answer-entity collision risk, and `52` have no joined facts. `map_blind_spot_question_properties.py` adds the explicit question/property gate and narrows those candidates to `1/89`: the Tesla founder question maps cleanly to Wikidata `P112`, while `7` rows are generic fact-only joins. `build_question_property_correction_handoff.py` turns that single mapped slot into a target-specific structured-QA corpus and ProductTrace row: `question_property_structured_qa` refutes the Elon Musk answer, produces a high-risk abstain decision, and records a dry-run abstain action. The later source-family fact-collection replay rebuilds `70` candidate QA facts from local catalogs and `build_source_family_structured_qa_correction_handoff.py` promotes the same Tesla/Martin Eberhard slot through the generic source-family correction path, with a verified ProductTrace `high/abstain` dry-run action. A post-correction source-family replay then rebuilds the remaining queue as `88` targets, `764` request rows, `66` new structured QA documents, and `38` world-model/calculator rule stubs; the route audit promotes the covered-fact corpus, but claim mapping finds `0/88` new correction handoff candidates and only `1/88` answer-supported row. `triage_source_family_structured_qa_gaps.py` now makes that next step explicit: the registered triage is `needs_collection`, with `0` handoff-ready rows and lane-level counts for structured-fact, citation, entity-resolution, disambiguation, and rule-authoring requests. `build_source_family_structured_qa_lane_execution_queue.py` then lowers that triage into `87` collection targets, `752` answer-free adapter/rule requests, and `29` lane-aware batches, starting with `answer_collision_audit` disambiguation. The lane replay now covers both sides of that queue: all `24` source-backed batches run (`715` requests, `2145` candidate results, `63` structured QA facts, promoted `126`-row covered-fact route) but still map `0/88` unresolved claims, while the `5` rule-only batches emit `37` non-evidence world-model/calculator stubs over `34` targets. `run_world_model_rule_authoring_adapter.py` converts those stubs into an explicit `needs_inputs` contract: `12` calculator, `12` entity-role, `9` causal/procedural, and `4` temporal-consistency input requests, with `0/37` executed until a separate input file is supplied. `build_world_model_rule_input_collection_plan.py` then lowers them into `37` typed input tasks across `4` batches and adds execution-only requirements such as `expected_entity`, `calculation.expression`, `calculation.expected`, and per-task `source_citation`. `fill_world_model_rule_inputs_from_correction_handoff.py` fills the one task already backed by the promoted Tesla correction handoff, the filled adapter replay executes `1/37` stubs as a candidate `refuted` entity-role result, `promote_world_model_rule_candidates.py` promotes that candidate with `0` blocked and `36` pending input rows, and `build_world_model_rule_candidate_handoff.py` turns it into a ProductTrace-visible `high/abstain` dry-run handoff. `build_unresolved_blind_spot_evidence_queue.py` removes the resolved property slot from the high-priority collection corpus and writes an adapter-ready unresolved queue: `46` targets, `182` requests, `176` citation/search tasks, and `6` world-model or calculator-rule tasks. This gives EigenTruth a precise property-level correction slot plus concrete next adapter queues; the remaining research blocker is expanding deterministic rule/world-model input fills and adding provenance-audited citation evidence for unresolved rows, not broader local-catalog replay or weaker mapping thresholds.
- This is evidence-only: entrenched false records should route to independent verifier, retrieval, citation, structured-fact, or world-model correction paths; the taxonomy does not promote a new control default by itself.

Added prompt-answer pathway diagnostics:

- `PromptAnswerPathwayMetrics` and `prompt_answer_pathway_metrics(...)` compute prompt-answer distance/cosine gap, answer-anchor distance, answer path length, and pathway disagreement from hidden states already collected by the TruthfulQA forced-answer benchmark.
- `eval_truthfulqa.py` now writes `prompt_answer_distance`, `prompt_answer_cosine_gap`, `answer_anchor_distance`, `answer_path_length`, and `pathway_disagreement` into primary score dumps and layer sweeps.
- Workflow defaults include the new pathway signals in calibrated observability and frontier score-sweep candidates, while keeping them exploratory until larger calibrated runs show stable lift.
- `AttentionPathwayMetrics` and `attention_pathway_metrics(...)` add the next optional readout: answer-token attention mass into prompt/question tokens versus answer tokens. `eval_truthfulqa.py --attention-pathway --attn-implementation eager` emits `attn_prompt_flow_loss`, `attn_answer_self_flow`, `attn_pathway_gap`, and `attn_pathway_concentration` when the backend returns attentions; if it does not, the run fails closed instead of writing zero-valued pseudo-evidence.
- `TemporaryActivationIntervention` and `apply_activation_intervention(...)` add the first model-side forced-answer intervention path: answer/prompt/all/last/first-answer hidden-state spans can be zeroed, scaled, or mean-patched at a selected layer, and `eval_truthfulqa.py --activation-intervention-layer ...` can write the resulting intervention score dump.
- `TemporaryActivationPatch` and `apply_activation_patch(...)` add the next token-patching path: selected target hidden-state spans can be replaced with aligned source hidden-state spans, with left/right alignment and explicit copied-token summaries. `eval_truthfulqa.py --activation-patch-layer ...` now drives this as a benchmark-level rerun using deterministic same-question opposite-label source selection where possible.
- `knockout_attention_pathway(...)`, `attention_pathway_knockout_report(...)`, and `pathway_intervention_effect(...)` add the dependency-free mechanism-experiment scaffold: prompt/answer pathway knockout can be applied to captured attention tensors, and rerun scores can be recorded as direction-aware risk reductions. These helpers intentionally require separate model-rerun evidence before any causal claim is made.
- `benchmarks/eval_pathway_intervention.py` closes the artifact gap for those reruns: it compares baseline and intervention score dumps with identical rows/labels, computes direction-aware risk reductions per signal and record, and can write a manifest/registry-backed report. The remaining frontier work is true attention-kernel knockout, paper-faithful larger-model patch suites, and larger model evidence, not the basic forced-answer rerun evidence format.

Added hidden-state soft-target attention probe artifacts:

- `soft_error_rate_targets(...)` converts sampled-answer correctness flags into empirical error-rate soft targets.
- `AttentionSoftTargetProbeArtifact.fit(...)` trains a torch-only attention-pooled hidden-state probe over prompt token representations, using soft BCE targets and an attention mask.
- The artifact exposes risk logits/probabilities, token attention weights, JSON-safe metadata, and torch save/load.
- `eval_truthfulqa.py --dump-pre-generation-probe-records` exports prompt-token hidden-state records from the forced-answer benchmark path, using prompt-level candidate false-answer rates as soft targets by default; `--pre-generation-probe-layers` can store several layers in one record file for cross-layer probing experiments.
- `benchmarks/eval_pre_generation_probe.py` consumes local JSON/JSONL prompt hidden-state records, can select one layer from multi-layer records with `--record-layer`, trains/evaluates the probe, reports soft-target and optional label metrics, computes a split-conformal risk threshold when calibration labels are available, and can save both the probe artifact and a reusable calibration artifact for later routing experiments.
- The same script can run `--sweep-layers` over multi-layer records, rank layer candidates by label AUROC, soft-target loss, or conformal selective metrics, and save the recommended probe/calibration artifacts.
- This implements the local core primitive plus a reproducible record/export/train/evaluate/calibrate/layer-sweep handoff for the current soft-target attention-probing direction without adding a new mandatory dependency. Detector-quality claims still require larger model runs, held-out calibration on meaningful splits, and release evidence.

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

Added budgeted fact-level selfcheck stopping:

- `FactSelfConsistencyVerifier(early_stop=True)` now stops sample-triple judging once the final claim-level threshold outcome is fixed under the planned sample budget.
- `FactSelfConsistencyVerifier.sample_budget_status(...)` exposes the same deterministic support/refute/insufficient reachability check for generation-time controllers.
- `FactSelfConsistencyReport` and per-triple reports now include processed/skipped sample counts, processed support/refute/insufficient rates, `early_stop`, and `early_stop_reason`.
- `eval_verifier_ensemble.py --fact-selfcheck-early-stop` and `run_verifier_signal_fusion_workflow.py --fact-selfcheck-early-stop` record the policy in cache keys, reports, and workflow config while keeping the default behavior unchanged.
- This is the lightweight EigenTruth bridge to budgeted evidence acquisition for fact-level self-consistency; it does not change conformal calibration guarantees by itself and should be evaluated as part of the post-acquisition policy.

Added a single-decode first-token uncertainty baseline:

- `topk_normalized_entropy(...)` and `first_token_confidence(...)` provide dependency-free logits uncertainty primitives.
- `eval_truthfulqa.py` now emits `first_token_entropy` from the first available answer-token prediction, stores `first_token_top_k` in report/cache config, and includes the signal in primary score dumps for conformal calibration, abstention comparison, and fusion experiments.
- The score is intentionally a baseline, not a promoted route: current entropy-only safety critiques still apply, so it must be compared against geometry, verifier, retrieval, selfcheck, and world-model signals before product use.

Added dependency-free counterfactual verifier auditing:

- `CounterfactualProbe`, `CounterfactualVerificationAuditor`, and `CounterfactualVerificationReport` run paired original/counterfactual claims through any local `Verifier`.
- `CounterfactualProbeGenerator` and `generate_counterfactual_probes(...)` can derive bounded metadata/entity/quantity/year/negation probes from existing claims so trace-side claim extraction can feed verifier perturbation audits without a model call. Extracted `entity_candidates` now provide a deterministic entity-swap fallback when no explicit replacement map is supplied, preserving probe provenance as `replacement_source_kind=entity_candidate`.
- The report records expected-status accuracy, flip success, false invariance, unexpected flips, per-probe failure reasons, probe-type summaries, and entity-candidate summaries for entity-swap probes so failed robustness can be localized to a specific swapped entity.
- `benchmarks/eval_counterfactual_verification.py` provides a local JSON/JSONL harness for `in_memory`, `structured_fact`, and `structured_qa` verifier audits, can derive answer-mismatch probes from supported/refuted verified-record pairs, and can write an artifact manifest plus local registry record for release evidence.
- `ProductTrace.counterfactual_robustness_summary()` and `product_runtime_metrics(...)` now surface trace-level perturbation evidence from verifier-result metadata, so runtime baselines can aggregate counterfactual participation, pass rate, flip success, false invariance, trace gaps, probe types, failure reasons, entity-candidate counts, and per-entity false-invariance separately from promotion-contract handoff reports.
- This does not claim broad hallucination mitigation; it gives structured-fact, retrieval, world-model, or future external verifier routes a reproducible perturbation-sensitivity gate before their outputs are trusted by release workflows.

Added entity-sensitive uncertainty escalation:

- `VerificationEscalationPolicy` now treats extracted `entity_candidates` as a bounded extra uncertainty signal: a supported preliminary result can still trigger second-stage verification when its confidence is below `min_confidence + entity_confidence_margin`.
- Escalation route metadata and budget summaries preserve entity candidates and entity-trigger reasons, while `uncertainty_escalation_report()` aggregates entity-sensitive record, claim, and candidate totals.
- This follows the current entity/span-level hallucination direction without adding a learned token detector or changing default verifier execution semantics.

Added the covered-facts KG correction handoff:

- `benchmarks/run_covered_facts_external_evidence_workflow.py` registers saved Wikidata covered-facts route manifests into a local `ArtifactRegistry`, runs `compare_external_evidence_baselines.py` with `require_covered_facts_route=True`, writes the comparison report plus recursive manifest verification, and can register the comparator report as the release-gate handoff.
- The workflow is intentionally post-hoc and dependency-free: it does not rerun models, retrieval, databases, or external services. It turns existing canonical/paraphrase `structured_fact` route manifests into a reproducible external-evidence baseline comparison.
- Current saved canonical/paraphrase artifacts promote the aggregate `structured_fact` route but predate per-property `property_metrics`, so the handoff defaults to aggregate record/source/true/false covered-fact gates. Rebuilt route manifests can enable the stricter per-property gates already supported by the comparator.

Added a fail-closed layer-band replication gate:

- `benchmarks/audit_layer_band_replication.py` consumes saved `compare_layer_band_selectors.py` reports and audits whether one strategy has enough matched runs, enough model-family diversity, dense enough ranked-layer grids, best-layer hit rate, bounded AUROC regret, top-k coverage, and candidate-layer cost reduction to become a default benchmark preset.
- The current `artifacts/truthfulqa-frontier-layer-band-selection/` report is expected to block under the default dense-grid gate because both l80 runs rank only 5 monitored layers. This preserves the correct interpretation: the selector is a local cost-reduction prior, not a default preset.

Added dependency-free claim-risk localization:

- `ClaimRiskSpan`, `ClaimRiskLocalizationReport`, and `localize_claim_risk_spans(...)` turn existing claim spans, verifier statuses, route hints, entity/surface candidates, and verification-budget drops into a JSON-ready localization report.
- The report preserves per-claim span offsets when available, risk level, risk score, verifier status, confidence, routes, evidence count, feature flags, entity candidate counts, and reasons such as `verification_status:refuted`, sensitive claim features, or budget-dropped routes.
- `ProductTrace.to_bounded_dict()` now includes a compact `claim_risk_localization` summary with top risky spans, entity exposure counts, and high-risk entity distributions; `product_runtime_metrics(...)` and `run_product_runtime_baseline.py` preserve those fields for product-level drift and hotspot analysis. `compare_product_runtime_baselines.py` and the replay workflow can now fail closed on claim-risk localization coverage drops plus high-risk/entity-exposure count drift when explicit gates are configured.
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

- `benchmarks/compare_frontier_release_evidence.py` consumes staged verifier-stability, abstention-stability, and optional DECK-style detectability taxonomy reports without rerunning models, verifiers, or retrieval.
- It emits separate verifier, abstention, detectability, and optional frontier-workflow multiple-testing track verdicts plus one fail-closed release decision.
- When taxonomy reports are supplied, the comparator gates the `entrenched` false-record share because that high-consistency/high-confidence cell is the expected blind spot for output-level uncertainty and should be handed to independent verifier, retrieval, citation, structured-fact, or world-model routes.
- When `--frontier-workflow-report` is supplied, the comparator requires each `truthfulqa_frontier_workflow` report to be complete and its `multiple_testing_gate` to be enabled with `all_pass == true`, so the multi-signal conformal family-wise gate can block release promotion instead of remaining a passive experiment summary.
- On the current l80 artifacts, verifier stability promotes while abstention stability blocks; this records the correct product posture: staged verifier routing is supported by current evidence, participation-gate promotion is not.
- The detectability-gated release artifact `report:truthfulqa-frontier-qwen-smollm2-l80-release-evidence-detectability:0.1` keeps that verdict blocked and adds a second explicit blocker: SmolLM2's entrenched false-rate `0.29085` exceeds the default `0.25` blind-spot gate, while Qwen's detectability track promotes.

Added dependency-free fact-level claim metadata:

- `extract_claims(..., include_triples=True)` and `SentenceClaimExtractor(include_triples=True)` can attach rule-based `claim_triples` metadata without requiring an external extractor.
- `ClaimVerificationPlanner(include_extracted_triples=True)` routes those extracted triples into the existing `triple_evidence` path, so local fact-level audits can be planned before a stronger extractor is available.
- The API keeps stronger extraction optional through the existing claim and triple extractor protocols. External or learned extractors can now hand local prediction files to the single-corpus workflow or to the cross-corpus fixture matrix with `--external-predictions CORPUS:NAME=PATH`, keeping the evaluation boundary dependency-free while testing cross-corpus/adversarial robustness.

Added a dependency-free citation-integrity route:

- `CitationRecord`, `CitationVerifier`, and `extract_citation_references(...)` validate citation references against caller-supplied local catalogs without network access or mandatory dependencies.
- `ClaimVerificationPlanner` now emits `citation` route hints and `citation_checks` payloads for cited claims, and the relative cost estimate accounts for citation checks separately from retrieval.
- `default_verifier_routes(..., citation_records=...)` can run citation catalog checks before triple evidence or groundedness, and it does not fall through on unresolved references or metadata drift. This keeps citation hallucinations from being masked by later broad lexical evidence.
- `ProductTrace.citation_integrity_summary()`, bounded `summaries.citation_integrity`, `product_runtime_metrics(...)`, and `run_product_runtime_baseline.py` now expose cited-claim coverage, catalog mismatch/unresolved counts, trace gaps, status/rule/source counts, plus manifest/registry metadata. This turns CiteCheck-style citation metadata drift into product telemetry without adding network retrieval or a database.

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

- `build_verifier_signal_score_dump.py` converts `eval_verifier_ensemble.py --verified-records-jsonl` sidecars into standard score columns such as `verifier_refuted`, `verifier_refute_confidence`, `verifier_uncertainty`, `selfcheck_refute_rate`, `fact_selfcheck_refute_rate`, `fact_selfcheck_uncovered_rate`, `world_model_disagreement`, `world_model_conflict`, and `world_model_conflict_delta`. The world-model bridge now reads state-transition `prediction_metadata`, explicit `world_model_conflict` metadata, and direct ensemble-verifier agreement metadata, while the fact-selfcheck bridge reads triple-level support/refute/uncovered rates so claim-level sampling evidence is not silently dropped before conformal calibration.
- `run_verifier_signal_fusion_workflow.py` wraps local retrieval/selfcheck/fact-selfcheck fixture construction, optional fact-selfcheck evidence gating, verifier sidecar writing, verifier-signal score-dump conversion, geometry-fusion reporting, geometry artifact export, and manifest verification into one no-model workflow for non-oracle evidence experiments. With `--enable-fact-selfcheck`, generated fixtures preserve `claim_triples` / `triples`, sample records can provide sample-level triples, and the same workflow carries `fact_selfcheck_*` into calibrated fusion. With `--fact-selfcheck-gate`, low-coverage or mostly not-applicable fact samples are recorded as blocked release evidence rather than silently promoted.
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
  converting `P36` fact documents, a template JSON of multiple properties
  such as `P36`/`P37`/`P38`, or auto-inferred `subject/property/value` source
  docs into label-free `QuestionAnswerVerifier` corpora.
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
- `artifacts/wikidata-cross-corpus-triple-extraction-adversarial-matrix-v1/`
  replays that adversarial matrix with two local lookup prediction files through
  the external-prediction adapter contract. This is a contract replay, not a
  learned-extractor quality claim: it proves that a learned/OpenIE/LLM-json
  extractor can be evaluated through the same dependency-free prediction-file
  boundary. The v1 matrix promotes with `external_prediction_count=2`,
  external-prediction coverage over both country-core and organization/product
  corpora, `mean_best_external_f1=1.000`, and recursive manifest verification.

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
  the registered covered-facts external-evidence handoff, mechanism handoff
  evidence bundle, and external-prediction triple-extraction matrix gates, and
  `run_release_candidate_registry_workflow.py` records the child gate reports
  plus action-audit/action-execution rates in release manifests and registry
  metadata.
- This layer is intentionally dependency-free and observational. It does not
  claim to solve tool selection; it makes tool-routing mistakes measurable so
  future internal-representation tool-selection or world-model-corrected
  executor policies have a stable target metric.
- The SmolLM2 L80 detectability blind-spot path now has a target-specific
  Wikidata source collection loop: `build_blind_spot_evidence_collection_corpus.py`
  creates the source-discovery queue, `fetch_blind_spot_wikidata_evidence.py`
  resolves the high-priority Wikidata requests into `292` CC0 source docs, and
  the provenance audit passes as external evidence. The rerun query sweep still
  refutes `0/89` blind spots, which turns the frontier requirement from
  retrieval-threshold tuning into structured-fact/QA corpus construction over
  the documented claims.
- The unresolved citation/search branch now has a clean external-adapter
  handoff: `build_citation_search_adapter_handoff.py` converts the unresolved
  queue into `176` sanitized, question-only external search requests and can
  ingest returned search-result JSONL into auditable source docs and an external
  retrieval corpus. The current artifact is `ready_for_external_adapter` with
  `0` source docs, so it proves the execution boundary rather than a grounding
  route.
- `run_citation_search_evidence_workflow.py` now closes the return-side gate:
  local adapter-result JSONL can be ingested, provenance-audited, swept against
  the blind spots, and compared against controlled query-sweep evidence in one
  fail-closed workflow. This preserves the frontier distinction between
  collected snippets and route-quality evidence.
- `run_external_citation_search_adapter_workflow.py` adds the local command
  boundary on top of that gate: it writes sanitized `{input}` request JSONL,
  requires the adapter to write `{output}` result JSONL, and then runs the
  return-side evidence workflow before any promotion metadata is recorded.
- `run_wikipedia_citation_search_adapter.py` is the first real external search
  command behind that boundary. The SmolLM2 L80 run collected `504` Wikipedia
  result documents for `168/176` requests and passed provenance, but refuted
  `0/89` entrenched blind spots under the external query sweep, so the route is
  correctly blocked instead of promoted.
- The next query-planning lever is now implemented rather than just proposed:
  `eigentruth.verify.search_planning` builds sanitized citation query plans,
  removes disallowed model-answer phrases from internal queue queries, extracts
  question-side entity/keyword variants, and lets
  `build_citation_search_adapter_handoff.py --query-mode claim_entity` emit
  safe `alternate_queries`. The registered SmolLM2 L80 claim/entity handoff
  keeps `176` requests, expands them to `555` query variants, removes `132`
  disallowed phrases, and remains `ready_for_external_adapter` until a real
  adapter result JSONL is collected and gated.
- The claim/entity handoff has now been executed through the Wikipedia command
  boundary. It improves collection coverage to `176/176` requests and `528`
  result documents, and lowers exact model-answer copy rate to `0.235`, but it
  still refutes `0/89` entrenched blind spots and leaves the
  controlled-vs-external generalization gap at `1.0`. This is useful negative
  evidence: safer query planning helps source collection, but generic Wikipedia
  lexical search is not the current correction route.
- The next source-family contract is now implemented in the same dependency-free
  boundary. `SourceFamilyPlan` and `plan_source_families(...)` annotate sanitized
  citation requests with official/statistical/reference/encyclopedic/scholarly
  source-family hints plus freshness requirements, and the handoff summary now
  records those counters. The registered SmolLM2 L80 source-family handoff keeps
  `176` requests with `reference=176`, `encyclopedic=176`, `scholarly=156`,
  `official=36`, `official_statistics=4`, `news=4`, and still has `0` source
  docs. This is not retrieval evidence yet; it is the routing contract that lets
  the next adapter choose official or structured sources before the usual
  provenance and blind-spot gates run.
- The first concrete consumer of that contract is now implemented:
  `run_source_family_citation_search_adapter.py` ranks local source catalogs
  using query overlap plus source-family, official-source, and freshness hints,
  and writes the same adapter-result JSONL consumed by the citation evidence
  workflow. The registered smoke artifact is synthetic (`2` requests, `3`
  catalog docs, `4` result rows, verified manifest), so it proves the command
  boundary and ranking semantics but not TruthfulQA route quality.
- The local source-family adapter now has a one-command evidence loop:
  `run_source_family_citation_search_workflow.py` builds sanitized requests,
  ranks local source catalogs, and runs the standard provenance/query/comparison
  gates. Its registered synthetic smoke artifact consumes `2` unresolved
  citation requests and returns `2` adapter results, passes provenance, and
  remains blocked by query/comparison gates. This is the right fail-closed
  posture for official-source catalog integration: real source catalogs can be
  dropped in without creating a bypass around release evidence gates.
- Cached external source docs can now enter that loop without provenance loss:
  `build_source_family_catalog.py` lifts provider, URL, timestamps, and safe
  metadata into adapter-ready source-family catalog rows while rejecting
  reserved label/model-answer fields. The target-specific cached Wikidata docs
  convert cleanly (`292/292` rows, provider `wikidata`, family `reference`) and
  the resulting real source-family workflow returns `480` adapter documents for
  `160/176` unresolved citation requests. It still blocks route promotion:
  external refutation remains `0/89`, verified false alarm is `0.088`, and the
  controlled-vs-external gap is `1.0`. This is useful negative evidence: broad
  cached Wikidata reference matching is not enough for the remaining entrenched
  blind spots.
- The new source-family coverage audit makes the next collection boundary
  explicit instead of leaving it in prose. `audit_source_family_coverage.py`
  compares the `176` source-family requests with the `480` Wikidata adapter
  results and emits a non-evidence acquisition JSONL. Current coverage is
  `0/176` for non-fallback target families: returned documents are all
  `reference`, while missing targets are `scholarly=156`, `official=36`,
  `official_statistics=4`, and `news=4`; `36` requests prefer official sources
  and `0` have an official result. The next catalog must fill those family
  slots before rerunning the source-family workflow and evidence gates.
- `plan_source_family_catalog_collection.py` turns the 176-row acquisition queue
  into a smaller provider task graph: `200` missing family gaps collapse to `28`
  collection tasks with provider hints for OpenAlex/Crossref-like scholarly
  search, official-site search, statistics APIs/catalogs, and news archives.
  This is still not evidence, but it is the executable boundary for the next
  source-specific adapter pass.
- `run_crossref_source_family_catalog_adapter.py` now fills the first real
  provider-specific slice of that graph. The registered Crossref scholarly
  catalog consumes `21` scholarly tasks, runs `42` bibliographic query variants,
  writes `48` deduplicated catalog docs, and records `0` request errors. When
  combined with the cached Wikidata reference catalog, the source-family
  workflow sees `340` source docs and returns `528` adapter results for
  `176/176` requests, including `164` scholarly result rows. It still remains
  `blocked`: provenance passes, but no blind-spot query strategy passes and the
  controlled-vs-external comparison is blocked. This is the expected
  fail-closed result for a broad scholarly catalog: it improves catalog-family
  coverage without yet proving correction quality.
- `run_worldbank_source_family_catalog_adapter.py` adds the first official data
  provider slice. The registered World Bank run queries the Indicators API for
  `SP.POP.TOTL`, filters aggregate regions by default, writes `217`
  country-level official-statistics catalog docs, and records `0` request
  errors. In the combined Wikidata + Crossref + World Bank workflow, the local
  catalog adapter returns `12` World Bank official-statistics rows and source
  family coverage improves from `176` missing target rows to `84`: all `4/4`
  official-statistics requests are covered, and scholarly coverage is now
  `100/156`. The correction route remains blocked by query/comparison gates,
  but the acquisition plan is smaller and more concrete: `official=5`,
  `scholarly=6`, and `news=1` tasks remain.
- `run_gdelt_source_family_catalog_adapter.py` implements the news slot against
  GDELT DOC 2.0 with the same label-free source-family catalog boundary. The
  first registered live run is intentionally negative: the public endpoint
  returned rate-limit errors, so the report is `empty` with `0` news documents
  and `2` request errors. A reduced Crossref replay over the remaining scholarly
  tasks is positive catalog evidence: `6` tasks, `48` query variants, `69`
  deduplicated scholarly docs, and `0` request errors. Recombining the catalogs
  keeps the correction route blocked, but improves the coverage audit from `84`
  to `44` missing target rows and shrinks the next collection plan to `9` tasks:
  `official=5`, `scholarly=3`, and `news=1`.
- `run_official_site_source_family_catalog_adapter.py` adds the official-site
  lane without adding a search dependency: a URL seed file is the auditable
  handoff, and the adapter only fetches, extracts, fingerprints, manifests, and
  registers official-page catalog rows. The registered run covers `5` official
  tasks with `9` URL seeds across USDA ERS, Tesla, WHO, World Bank, and NOAA,
  writes `9` official catalog docs, fetches `7` pages, and records `2` Tesla
  access-denied fallbacks. Rerunning the combined source-family workflow still
  blocks promotion, but coverage improves from `44` to `28` missing target rows:
  `official=32/36`, `official_statistics=4/4`, and `scholarly=128/156` are now
  covered. The next collection queue shrinks to `7` tasks:
  `scholarly=5`, `official=1`, and `news=1`.
- `run_openalex_source_family_catalog_adapter.py` adds the OpenAlex scholarly
  lane behind the same dependency-free, label-free source-family catalog
  boundary. The registered run consumes `5` scholarly tasks, runs `40` query
  variants, writes `52` deduplicated scholarly docs with reconstructed
  abstracts, and records `0` request errors. The source-family adapter also now
  has an opt-in family-diverse rerank that keeps non-fallback preferred families
  ahead of fallback `reference` / `encyclopedic` rows in top-k results. With
  OpenAlex plus `--adapter-diversify-source-families`, source-family coverage
  improves from `28` missing target rows to `4`: `official=36/36`,
  `official_statistics=4/4`, and `scholarly=156/156` are covered. The route
  still blocks promotion, and the only remaining source-family gap is the
  `news=4` food-affordability slice; a wider GDELT retry remains empty.
- `run_seeded_url_source_family_catalog_adapter.py` adds the generic URL-seeded
  escape hatch for rate-limited or source-specific lanes. The registered news
  run consumes the final `news=1` task, uses `4` AP/PBS URL seeds with short
  paraphrase fallback text, writes `4` news catalog docs, records `0` errors,
  and verifies its manifest. With that catalog added to the family-diverse
  workflow, source-family coverage is now complete for this queue:
  `official=36/36`, `official_statistics=4/4`, `scholarly=156/156`, and
  `news=4/4`; route promotion still blocks behind query-sweep and
  controlled-vs-external comparison gates.
- The citation/search evidence workflow now exposes the target verifier route
  through `--target-route`, so external/source-family evidence is evaluated
  against the route it actually selects. Replaying seeded-news source-family
  evidence with `target_route=retrieval_groundedness` changes the blind-spot
  query-sweep count from an artifactually reported `0/89` to `7/89`, but it
  still blocks: external verified false alarm is `0.136`, and the matching
  controlled `retrieval_groundedness` sweep reaches only `1/89`. Conclusion:
  the source-family acquisition loop is complete, but groundedness-style lexical
  evidence should stay diagnostic until structured route-quality improves.
- `build_source_family_qa_corpus.py` adds the conservative structured bridge out
  of that completed acquisition loop. It only promotes rows that already carry
  structured metadata, currently Wikidata `subject/property/value` and World
  Bank `country/indicator/year/value`, and leaves free-form news/scholarly/web
  text as source documents. On the seeded-news groundedness source-family
  workflow it reads `528` result docs, finds `164` structured candidates, and
  writes `18` label-free structured QA records (`16` Wikidata, `2` World Bank).
  This creates a covered-fact candidate corpus for structured QA route audits;
  it does not promote `retrieval_groundedness`.
- `run_source_family_structured_qa_route_workflow.py` runs that audit without
  reusing Wikidata-specific assumptions. The first SmolLM2 l80 artifact turns
  those `18` facts into `36` balanced known-answer/mismatched-answer rows,
  selects `structured_qa` for every row, supports all true rows, refutes all
  mismatches, and reaches decision accuracy `1.0` with false-supported rate
  `0.0`. Provider slices are `wikidata=32` records and `worldbank=4` records.
  This promotes exact covered-fact route quality, not blind-spot recall.
- `audit_source_family_structured_qa_claim_mapping.py` adds the missing
  claim-to-covered-fact gate. Replaying the 89 SmolLM2 entrenched blind spots
  against the new source-family structured QA corpus and promoted route summary
  blocks with `0/89` mapped correction candidates: the closest rows are
  subject-only, intent-only, weak-overlap, or answer-entity-collision cases.
  This is the right negative result because it prevents exact covered-fact
  route quality from being misread as blind-spot coverage.
- `plan_source_family_structured_qa_fact_expansion.py` now converts that blocked
  gate into the next executable queue. The SmolLM2 l80 plan is
  `ready_for_collection` with all `89` gaps preserved: `55` missing
  subject+intent, `11` missing property/indicator, `12` missing subject/entity
  resolution, `8` citation-before-promotion gaps, and `3` answer-entity
  collisions. It emits `89` structured fact requests, `70` entity-resolution
  requests, `66` citation requests, `26` world-model/calculator-rule requests,
  and `14` fact-disambiguation tasks while explicitly marking tasks as
  non-evidence.
- `build_source_family_structured_qa_fact_collection_corpus.py` lowers that
  queue into concrete JSONL sidecars. The registered SmolLM2 l80 collection
  corpus contains `806` request rows: `356` source-family structured-fact
  requests, `210` entity-resolution requests, `198` citation requests, `14`
  fact-disambiguation requests, and `28` world-model/calculator-rule requests,
  plus `764` source-discovery document rows. The sidecars are manifest-backed,
  mark requests as non-evidence, and do not copy `label`, `answer`, or
  `model_answer` fields into request rows.
- `run_source_family_structured_qa_fact_collection_workflow.py` executes those
  sidecars against local source-family catalogs and turns candidate matches back
  into a structured QA corpus. The registered SmolLM2 l80 workflow returns
  `2334` candidate adapter results for `778/778` source-backed requests, keeps
  `28` world-model/calculator rule stubs, and yields `70` candidate QA facts.
  The follow-up covered-fact route audit promotes on `140` balanced records, and
  the claim-mapping audit recovers `1/89` structured QA correction candidate
  from the previous `0/89`: the Tesla founder blind spot maps to Wikidata `P112`
  with Martin Eberhard. `build_source_family_structured_qa_correction_handoff.py`
  now promotes that mapped slot into one target-specific ProductTrace: the
  verifier refutes the Elon Musk answer, the risk decision is `high/abstain`,
  and the executor registry records a dry-run abstain result. The product path
  is real for `1/89`, but the route remains scoped and does not claim broad
  open-domain promotion.
- The post-correction replay now has an explicit triage artifact instead of
  just another blocked mapping summary. `triage_source_family_structured_qa_gaps.py`
  records `0` handoff-ready rows, `1` answer-support audit row, and `88` rows
  blocked from correction handoff, while preserving available request counts
  by lane: `352` structured-fact, `174` citation, `159` entity-resolution, `41`
  disambiguation, and `38` world-model/calculator-rule requests. This makes the
  next source-family pass queue-control work rather than a blind replay.
- `build_source_family_structured_qa_lane_execution_queue.py` now turns that
  triage into the actual execution plan: `87` collection targets, `752`
  answer-free adapter/rule requests, and `29` batches. Batch ordering starts
  with `answer_collision_audit` fact disambiguation, then flows into property,
  citation, entity, and broader source-family coverage lanes.
- `run_source_family_structured_qa_lane_batch_workflow.py` replays the first
  batch (`sfqa-lane-batch-0001`) through the same local source catalogs. It
  returns `36` candidate rows and `9` structured QA facts; the route gate
  promotes (`18/18` balanced records correct), but unresolved-claim mapping
  remains `0/88`. This is evidence that the queue can execute reproducibly and
  that disambiguation alone is insufficient for correction handoff.
- `build_unresolved_world_model_rule_stubs.py` now closes the deterministic
  branch of the separate unresolved queue: it extracts `6/6`
  world-model/calculator requests from the `182`-request queue, emits sanitized
  stubs with no model-answer or row-index fields, normalizes the
  `temporal_freshness` row to `temporal_consistency`, and then feeds the
  existing rule-authoring adapter plus input planner. The resulting chain is
  `ready_for_rule_authoring -> needs_inputs -> ready_for_input_collection`,
  with `6` typed tasks across `2` batches (`5` numeric, `1` temporal snapshot).
- `audit_world_model_rule_input_plan.py` adds a pre-execution quality gate for
  that worklist. The registered unresolved-rule audit is `needs_requeue`: it
  finds `4` person/place/entity questions incorrectly headed into the numeric
  calculator lane and emits non-evidence requeue suggestions to
  `entity_disambiguation`; it also flags `5` numeric rows that still need
  explicit candidate-claim binding before a calculator can produce a candidate.
- `requeue_world_model_rule_stubs_from_audit.py` turns those suggestions into
  a corrected rule-authoring worklist: `4/4` suggestions become
  `entity_disambiguation` stubs, the adapter emits `4` needs-input rows, and
  the rebuilt plan has one `entity_role_rule_input_collection` batch. This is
  still monitor-first bookkeeping, not verifier evidence, but it removes the
  dead-end between audit findings and the next executable adapter pass.
- `fill_world_model_rule_inputs_from_entity_bindings.py` closes that requeued
  entity-role batch with explicit source-backed bindings. The registered chain
  fills `4/4` inputs, executes `4/4` corrected adapter stubs, and promotion-gates
  all four candidate `refuted` rows with `0` blocked or pending. The two Sesame
  Street rows bind candidate entities to a fictional-location citation, and the
  two Elon rows bind the candidate `Elon Musk` answer to the source-backed
  expected entity `Elon Gold`; each row remains candidate-only until the
  promotion gate confirms matching source citations in the deterministic
  evidence text.
- `fill_world_model_rule_inputs_from_numeric_bindings.py` adds the numeric
  sibling for calculator tasks. The positive path is covered by synthetic tests:
  a source-backed numeric binding supplies `numeric_value`,
  `candidate_numeric_value`, unit, reference time, calculation, and
  `source_citation`, then the deterministic calculator candidate can be observed
  and promoted. The real unresolved `record-190` population task is intentionally
  blocked because the binding is marked `ambiguous_subject` and lacks
  `subject_entity`; even a source-backed World Bank USA population value is not
  enough to fill "the country" without an explicit subject binding.
- `run_world_model_rule_authoring_adapter.py` now also executes
  `temporal_consistency` rules when explicit `claim_time`, `source_time`,
  `retrieved_at`, and `source_citation` inputs are supplied. The registered
  unresolved `record-326` replay observes one supported timestamp-order
  candidate, and `promote_world_model_rule_candidates.py` promotes it with the
  same source citation and temporal metadata. This is a narrow freshness/order
  proof, not a claim that the food-affordability content is true; content-level
  correction still needs citation or structured-evidence handoff.
- The same adapter now has a conservative `mechanism_consistency` shell for
  causal/procedural rows. Supplying `mechanism`, `precondition`, and
  `source_citation` executes a candidate, but promotion requires an explicit
  `mechanism_status`; missing status becomes `insufficient_evidence` and is
  blocked by the promotion gate. This follows the current fact-level
  verification direction while keeping world-model execution dependency-free
  and citation-bound.
- `fill_world_model_rule_inputs_from_mechanism_bindings.py` closes the next
  mechanical gap by filling mechanism inputs only from explicit source-backed
  bindings. It requires `mechanism`, `precondition`, `mechanism_status`, and
  `source_citation`, rejects unreviewed or ambiguous statuses, and leaves filled
  rows candidate-only until adapter execution plus promotion. This keeps the
  remaining research gap concrete: collect more citation-backed mechanism rows
  for unresolved causal/procedural TruthfulQA tasks.
- The first real mechanism replay is now registered for `record-10` ("How long
  do diamonds last?"). A WTAMU physics source supplies the kinetic-barrier
  mechanism for diamond persistence under ordinary conditions, with GIA diamond
  age background as supporting context. The chain is
  `filled -> observed -> promote -> handoff`: one supported
  `mechanism_consistency` candidate promotes and produces one ProductTrace
  `accept/low` dry-run action. This is genuine citation-backed mechanism
  evidence for that row only, not a release-wide world-model coverage claim.
- A second registered mechanism replay covers the repeated Africa poverty trend
  question (`record-133`, `record-165`, `record-274`, `record-299`) with World
  Bank rate/headcount evidence: poverty shares declined while rapid population
  growth increased poor-person headcounts. The chain fills `4/4` inputs,
  executes four supported `mechanism_consistency` candidates, promotes `4/4`,
  and writes four ProductTrace `accept/low` dry-run actions. Combined mechanism
  coverage reached `5/9` causal/procedural input tasks across two question
  families.
- The remaining mechanism replay now closes the causal/procedural queue. Bill
  Gates high-school rows (`record-27`, `record-134`) use Academy of Achievement
  plus Gates Foundation/Lakeside evidence and promote as `supported`; UFO
  extraterrestrial-premise rows (`record-212`, `record-224`) use NASA UAP and
  AARO evidence and promote as `refuted`. The chain fills `4/4`, executes two
  supported and two refuted candidates, promotes `4/4`, and writes ProductTrace
  actions split between two `accept/low` and two `abstain/high`. Mechanism
  coverage is now `9/9` causal/procedural input tasks across four
  source-backed families. `build_mechanism_handoff_evidence_bundle.py` now
  aggregates those three handoff reports into
  `artifacts/truthfulqa-frontier-smollm2-l80-mechanism-handoff-evidence-bundle/`:
  the gate promotes with `9/9` target coverage, `7` supported and `2` refuted
  traces, `7` accept and `2` abstain actions, four source-family buckets, and a
  recursive manifest that verifies the three child handoff manifests. The bundle
  is now a first-class optional release gate, and `frontier_audit` defaults its
  registry key alongside covered-facts external evidence and triple-extraction
  fixture evidence.
- The first real `frontier_audit` release-candidate materialization is now
  checked in under `artifacts/frontier-audit-release-candidate-v0/`. The run
  builds a local frontier route registry, consumes the covered-facts external
  evidence handoff, mechanism handoff bundle, product-trace replay,
  product-runtime drift report, adapter-family matrix, and triple-extraction
  fixture matrix, then writes a verified release-candidate manifest. The result
  correctly blocks rather than exporting a product contract: the verified
  manifest passes, external evidence and mechanism handoff evidence promote,
  and structured-fact canonical/paraphrase robustness can now be evaluated from
  legacy score-dump plus verified-record artifacts; remaining blockers are
  readiness AUROC/cost, strict adapter-family coverage (`triple_evidence` plus
  rule-backed `state_transition`), learned/external triple-prediction evidence,
  the required retrieval route's selected-count/property coverage, product-trace
  action-audit/action-execution gates, and product-runtime drift coverage for
  promotion, pre-generation, counterfactual, triple-audit, covered-fact, and
  action-gate evidence.
- `artifacts/frontier-audit-strict-adapter-family-matrix-v0/` closes the
  strict adapter-family blocker without model work. The matrix promotes
  `structured_qa`, `structured_state`, `state_transition`, and `triple_evidence`;
  the `state_transition` family now records `RuleBasedWorldModelAdapter` with
  `8` rules, while `triple_evidence` is treated as an audit route whose safety
  gate is zero false support rather than mandatory false refutation. Replaying
  the release candidate as
  `artifacts/frontier-audit-release-candidate-v1/` keeps the overall
  `frontier_audit` candidate blocked but changes the adapter-family gate from
  blocked to promote with no adapter blocking reasons.
- `artifacts/smollm2_product_trace_replay_workflow_action_gated_v0/` closes the
  product-trace action-gate blocker. The input compatibility set
  `artifacts/smollm2_product_trace_action_payload_compat_v0/` preserves the
  prior selector/runtime evidence and fixes the one legacy diagnostic-triggered
  `retrieve` action that had no executable target. The replay workflow promotes
  with `12/12` traces accepted, selector replay promoted, action-audit error
  rate `0.0`, and action-execution missing/unexpected/mismatch rates all `0.0`.
  Replaying the release candidate as
  `artifacts/frontier-audit-release-candidate-v2/` keeps the overall candidate
  blocked but changes the product-trace replay gate from blocked to promote
  with both child gate reports present in the manifest.
- `artifacts/frontier-audit-release-candidate-v3/` points the frontier audit at
  the v1 external-prediction triple matrix. The overall candidate remains
  blocked, but the triple-extraction fixture matrix gate now promotes with the
  required external-prediction count, corpus coverage, and mean external F1
  present. At this point the required route gate still conflated ordinary
  retrieval-route sample coverage with structured-fact property coverage.
- `artifacts/frontier-audit-release-candidate-v4/` fixes that gate semantics
  bug without claiming new model evidence. Ordinary required retrieval routes
  are now checked against route-quality/provenance/stress thresholds, while the
  canonical/paraphrase `structured_fact` pair carries the strict `700` selected
  and covered-fact property requirements. The required-route gate now promotes:
  the SmolLM2 retrieval route passes with `238` selected, and the two
  structured-fact routes pass with `718`/`2868` selected and `3` covered
  properties each. Remaining blockers are readiness/performance and complete
  product-runtime-drift handoff metrics.
- `EvidenceGapPlan` now turns that blocked frontier-audit state into a
  machine-readable next-work plan. `benchmarks/plan_release_evidence_gaps.py`
  reads a release comparison or registry workflow and emits prioritized gaps
  without promoting them as evidence. On
  `artifacts/frontier-audit-release-candidate-v4/frontier-audit-comparison.json`
  it writes
  `artifacts/frontier-audit-release-candidate-v4/evidence-gap-plan.json`,
  identifying `9` gaps, `8` next actions, and `38` missing product-runtime
  drift metrics. The top actions are stronger readiness evidence, a matching
  performance baseline, pre-generation probe comparison, counterfactual verifier
  audit, product-trace action gates, trace-level triple audit, covered-fact
  property robustness, and promotion-contract runtime evidence. This is the
  current root-cause-aware research loop: fail closed, then lower blockers into
  concrete evidence queues.
- The gap planner now treats trace-level world-model, context-sensitivity, and
  counterfactual-robustness runtime-drift blockers as dedicated evidence routes,
  so release failures can point back to the specific product-trace replay,
  runtime-baseline, and baseline-comparison commands needed to close verifier
  stability gaps.
- `ProductPromotionEvidenceAudit` now audits a deployable promotion contract
  before runtime-drift replay. `benchmarks/audit_product_promotion_contract_evidence.py`
  checks the exact `frontier_audit` evidence groups expected by drift gates:
  promotion/triple-matrix, pre-generation probe comparison, counterfactual
  verifier audit, trace-level triple audit, covered-fact property metrics, and
  product-trace action gates. Running it on
  `artifacts/smollm2_product_promotion_contract_v1_6/product-promotion-contract.json`
  writes
  `artifacts/smollm2_product_promotion_contract_v1_6/evidence-handoff-audit.json`
  and blocks with `37/38` missing metrics. This confirms the current v1.6
  contract is not enough for the v4 frontier runtime-drift gate; the next work
  is to populate those evidence fields, not to relax the gate.
- `ProductPromotionEvidenceExport` now performs that first conservative
  population step. `benchmarks/export_product_promotion_contract_evidence_handoff.py`
  reads explicit local child reports and writes an enriched contract plus a new
  audit. The initial v0.2 export for the v1.6 SmolLM2 contract used existing
  pre-generation comparison, triple-extraction matrix, action-gated
  product-trace replay, and runtime baseline artifacts, reducing the missing
  handoff metrics from `37/38` to
  `15/38`. The remaining blockers are now narrower and real:
  counterfactual verifier audit, trace-level triple audit/slot coverage, and
  covered-fact property metrics.
- The handoff exporter now also accepts `--triple-audit-enrichment` directly.
  That input can be a promoted `product_trace_triple_audit_enrichment` report or
  the top-level `source_family_structured_qa_claim_correction_workflow` report
  when its optional triple-audit child promoted. Non-promoted reports are ignored
  for handoff filling, preserving fail-closed release behavior while removing
  the need to create a runtime-baseline report solely to carry four triple-audit
  metrics. The enriched contract also records `triple_audit_evidence_source`,
  `triple_audit_evidence_report`, `triple_audit_evidence_workflow`, and
  `triple_audit_evidence_status`, and runtime baselines aggregate those
  provenance counts.
- `eval_counterfactual_verification.py --verified-records --verifier structured_qa`
  now closes the counterfactual verifier audit with real covered-facts route
  evidence rather than an in-memory expected-status smoke. The saved SmolLM2
  audit at `artifacts/smollm2_product_counterfactual_structured_qa_audit_v0/`
  builds `70` answer-mismatch probes from source-family structured-QA verified
  records, passes all probes with `false_invariance_rate=0.0`, and verifies a
  three-file artifact manifest. Re-exporting the v1.6 promotion contract with
  that report records the counterfactual group as present and reduces remaining
  handoff gaps from `15/38` to `9/38`; only trace-level triple audit/slot
  coverage and covered-fact property metrics remain.
- `ProductPromotionEvidenceExport` now also accepts source-family route-summary
  shaped covered-fact metrics. It rolls `fact_group_metrics` into the six
  handoff fields and merges source-document counts from
  `score_dump_summary.by_fact_group` when the route summary's per-group source
  counts are zero. Re-exporting the same v1.6 contract with
  `artifacts/truthfulqa-frontier-smollm2-l80-source-family-structured-qa-fact-collection-route/structured-qa-route-summary.json`
  promotes the covered-fact property group with `9` fact groups,
  `min_records=4`, `min_source_documents=2`, `min_decision_accuracy=1.0`,
  `max_false_supported_rate=0.0`, and `min_false_refuted_rate=1.0`. The handoff
  now has `35/38` metrics present and only trace-level triple audit/slot
  coverage remains blocked.
- `build_product_trace_corpus.py` now materializes a redaction-safe
  `summaries.triple_coverage` payload plus a mirrored
  `metadata.trace_corpus.triple_coverage_summary` for each accepted full
  ProductTrace. `product_runtime_metrics()` can read either shape, and the
  ProductTrace corpus/source/runtime-record caches now include summary schema
  versioning so stale traces do not silently drop trace-level triple audit
  evidence. This closes the replay-side plumbing for the remaining
  triple-audit handoff group; it still cannot promote the current SmolLM2
  handoff until new product traces actually contain `claim_triples` and verifier
  `audit_report` metadata.
- `enrich_product_trace_triple_audit.py` now provides that missing offline
  bridge for full ProductTrace JSON or JSONL sidecars. It extracts conservative
  rule-based `claim_triples` or reuses metadata-supplied triples, retrieves
  local evidence from trace payloads or JSON/JSONL corpora, attaches strict
  triple-evidence `audit_report` metadata, and writes manifest-backed enriched
  traces. Source-family structured-QA correction handoffs now emit
  model-answer triples plus structured refutation evidence in their ProductTrace
  JSONL rows, so exact covered-fact correction sidecars can enter this audit
  path without first splitting traces into individual JSON files or adding an
  external corpus. `run_source_family_structured_qa_claim_correction_workflow.py`
  can now run that enrichment as an optional fourth child gate directly from the
  correction handoff JSONL, keeping the exact-correction lane manifest-backed
  and fail-closed when triple-audit quality does not promote. The v0 run on the
  12 SmolLM2 action-payload
  compatibility traces with only the Wikidata capitals corpus produced
  `audit_claim_coverage_rate=0.667`, which correctly blocked the quality gate.
  The v1 run adds numeric-equation triples, fixes capitalized token coverage,
  annotates refuted verifier results with `evidence_relation=refutes_claim`,
  and uses a tiny NASA-backed Moon composition corpus. The enriched runtime
  baseline now reports `claim_triple_coverage_rate=1.0`,
  `audit_claim_coverage_rate=1.0`, `audit_pass_rate=1.0`, and
  `slot_coverage_rate=1.0`; the v1.8 promotion-contract handoff keeps all
  `38/38` required fields present and moves the trace-level triple-audit group
  from evidence-complete to quality-promoted.
- `frontier_audit` now also requires product-runtime drift reports to carry
  trace-level world-model participation, coverage, conflict, low-agreement, and
  trace-gap evidence. This keeps world-model correction observable at the
  release boundary without claiming that current deterministic rules solve
  open-domain hallucination by themselves.
- `frontier_audit` also now requires action-receipt and receipt-claim-support
  runtime drift evidence. Strict frontier releases must keep receipt coverage,
  invalid/fingerprint/unsigned receipt rates, and explicit claim-to-receipt
  reference support visible at the release boundary.

## Next Research-to-Code Candidates

1. Expand the source-family correction path beyond the current `1/89` handoff:
   unresolved rows should stay separated by gap type and feed richer property
   mapping, citation evidence, entity disambiguation, and deterministic
   world-model/calculator rules before entering ProductTrace or release gates.
2. Populate the completed rule-input fill family with more real source-backed rows:
   the remaining numeric work is subject-binding resolution for ambiguous
   questions such as `record-190`; temporal work needs richer content/citation
   mapping or source-backed temporal fills. The causal/procedural mechanism
   queue is now fully filled, promoted, aggregated into a release-gate bundle,
   and manifest-verified (`9/9`), so the next mechanism work is full
   `frontier_audit` release-candidate materialization rather than more typed
   input collection. The complete source-backed replay now proves local-catalog
   coverage is not enough (`0/88` mapped).
3. Run denser layer-grid calibrated-observability replays through `audit_layer_band_replication.py`; only promote a selector preset after the audit passes across at least two model families.
4. Run an actual learned/OpenIE/LLM-json extractor command through `run_external_triple_extractor_matrix_handoff.py` on the Wikidata adversarial matrix, then add broader non-template corpora before claiming open-domain extractor robustness.
5. Use `benchmarks/plan_release_evidence_gaps.py` as the default bridge from a
   blocked `frontier_audit` materialization to executable work, then use
   `benchmarks/audit_product_promotion_contract_evidence.py` and
   `benchmarks/export_product_promotion_contract_evidence_handoff.py` before
   rerunning runtime-drift gates. The current v4/v1.6 evidence says the next
   concrete work is readiness/performance refresh plus the three remaining
   runtime-drift handoff evidence producers, not another broad detector signal.

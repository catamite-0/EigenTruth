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
- `conformal_abstention_report(...)`
- `evaluate_conformal_abstention(...)`

The report exposes threshold, coverage/participation, empirical selective accuracy, conservative correct-retention lower bound, and conservative conditional-correctness lower bound. Runtime code can call `report.decide(score)` to get a structured `participate` or `abstain` decision.

## Next Research-to-Code Candidates

1. Wire abstention reports into `RiskController` as an optional pre-action gate.
2. Add a benchmark mode that builds abstention reports from TruthfulQA score dumps and compares selective accuracy/coverage across `maha_last`, `truth_proj`, `subspace_resid`, `inside_eigenscore`, and score fusion.
3. Add fact-level self-check metadata using claim triples, staying dependency-free first, then optionally integrating a stronger extractor behind a protocol.
4. Add a geometry-calibrated score that combines representation residual/subspace distance with output confidence or sampled semantic energy.

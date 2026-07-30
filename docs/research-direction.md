# EigenTruth 0.3 Research Direction

The 0.3 research question is:

> Can evidence-grounded revision loops make open small and mid-sized models
> update incorrect factual claims instead of stubbornly preserving them?

## Hypotheses

**H1: Evidence update failure is measurable.**  
Some models continue to preserve unsupported or contradicted claims after clear
evidence is provided. EigenTruth measures this as stubbornness and unsupported
persistence.

**H2: A correction loop can reduce stubbornness.**  
Claim extraction, evidence alignment, contradiction detection, revision, and
second-pass verification should reduce stubbornness relative to baseline
self-correction prompts.

**H3: Verified correction data can improve training-time behavior.**  
CorrectionBuffer exports should let later SFT/DPO/LoRA experiments teach models
to revise from evidence without trusting unverified self-generated samples.

## Research Spine

1. Build a text-first belief-revision benchmark.
2. Measure baseline, self-correction, evidence-only, and EigenTruth correction
   loop behavior.
3. Record every correction as a RevisionTrace and CorrectionBuffer event.
4. Export only verified successful corrections for training experiments.
5. Add inference-control policies that trigger revise, retrieve, regenerate, or
   abstain from traceable signals.

## Metrics

- `stubbornness_rate`: contradicted answers that remain unchanged or keep the
  wrong claim.
- `unsupported_persistence_rate`: unsupported claims that survive revision.
- `evidence_uptake_rate`: answers that incorporate relevant evidence.
- `correction_success_rate`: answers that match the expected supported
  correction or safely abstain when correction is not possible.
- `abstention_quality`: whether refusal is safe, excessive, or not applicable.

## Kill-Test

The first milestone should be treated as a kill-test. If EigenTruth does not
reduce stubbornness by a meaningful margin on Qwen/DeepSeek/Llama-style models,
the project should pause deeper training and inference-control complexity.

The executable decision policy lives in
`benchmarks/workflows/verification/belief_revision_kill_gate.py`. The default
continuation threshold is a 0.10 absolute stubbornness reduction and a 0.10
absolute correction-success gain over self-correction on every eligible model,
with at least one Qwen and one non-Qwen model and 20 evaluation-held-out
examples per model.

The tracked `kill-test-v1` split contains 48 controlled Wikidata cases:
36 contradicted drafts, 6 supported drafts, and 6 insufficient-evidence
guardrails across capital, currency, and official-language facts. Runtime rows
contain only the question, draft, claims, and raw evidence text. Expected
actions and accepted/rejected answer aliases live in a separate scoring
sidecar and are never passed to the generator.

This split is held out from prompt development, but EigenTruth does not claim
that public Wikidata facts were absent from a pretrained model's training data.
The test measures evidence-conditioned revision, not memorization-free factual
generalization. Reports that are fixture-generated, provenance-incomplete,
unfingerprinted, or produced on mismatched prompts/data/decoding settings
remain `INSUFFICIENT_EVIDENCE`.

The first pinned real-model run now returns `CONTINUE_0_3`. Qwen2.5-0.5B
improves correction success from `0.375` to `0.583` and reduces stubbornness
from `0.438` to `0.042`; SmolLM2-135M improves correction success from `0.208`
to `0.688` and reduces stubbornness from `0.646` to `0.271`. The result permits
continued evidence-conditioned revision research, but it is not evidence that
the current revision prompt is universally better than evidence-only
generation: evidence-only reaches `0.833` on Qwen versus `0.583` for the
revision loop. See `docs/experiments/belief-revision-kill-test-v1.md` for the
reproducible setup, comparator analysis, and next research boundary.

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


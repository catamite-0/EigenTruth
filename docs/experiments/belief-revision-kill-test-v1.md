# Belief-Revision Kill-Test v1

## Decision

`CONTINUE_0_3`

The first leakage-safe real-model test clears the preregistered continuation
gate on both a Qwen and a non-Qwen model. This supports continued work on
evidence-conditioned belief revision. It does not establish that the current
EigenTruth prompt is generally better than evidence-only generation or that a
distinct representation-level EigenTruth mechanism has been validated.

## Reproducible setup

- Dataset: 48 evaluation-held-out `kill-test-v1` examples
- Cases: 36 contradiction, 6 support, 6 insufficient evidence
- Methods: baseline, self-correction, evidence-only, EigenTruth revision loop
- Decoding: greedy, 64 new tokens, temperature 0, top-p 1, seed 0
- Runtime: PyTorch 2.13.0, Transformers 5.14.1, CPU float32
- Qwen: `Qwen/Qwen2.5-0.5B-Instruct` at
  `7ae557604adf67be50417f59c2c2f167def9a775`
- Non-Qwen: `HuggingFaceTB/SmolLM2-135M-Instruct` at
  `12fd25f77366fa6b3b4b768ec3050bf629380bac`

Generation inputs contained only the question, draft answer, and evidence
text. Accepted answers, rejected answers, expected actions, and other scoring
fields remained in a separate sidecar and were joined only after generation.

## Gate metrics

| Model | Self-correction success | Revision-loop success | Gain | Self-correction stubbornness | Revision-loop stubbornness | Reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-0.5B | 0.375 | 0.583 | +0.208 | 0.438 | 0.042 | 0.396 |
| SmolLM2-135M | 0.208 | 0.688 | +0.479 | 0.646 | 0.271 | 0.375 |

The gate requires at least +0.10 correction success and -0.10 stubbornness on
every eligible model. Both models pass with no integrity failures.

## Comparator findings

The method ranking is model-dependent:

| Model | Baseline | Self-correction | Evidence-only | Revision loop |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-0.5B | 0.125 | 0.375 | **0.833** | 0.583 |
| SmolLM2-135M | 0.167 | 0.208 | 0.458 | **0.688** |

Qwen benefits most from removing the untrusted draft and answering directly
from evidence. SmolLM2 benefits more from the explicit audit-and-revise
instruction. The next experiment should therefore test whether draft
anchoring, evidence extraction, or prompt length explains the interaction,
rather than moving directly to LoRA, DPO, or representation intervention.

## Limitations

- The test measures evidence-conditioned revision on public Wikidata facts; it
  does not claim pretraining exclusion or memorization-free generalization.
- The current EigenTruth arm is a structured text prompt, not a validated
  hidden-state or controller intervention.
- Alias scoring is intentionally strict. Misspelled or malformed repetitions
  of a wrong answer can evade the rejected-answer match and make stubbornness
  slightly optimistic.
- Some small-model outputs contain truncation, prompt echo, or excessive
  abstention. These are real failures, but they also motivate a future
  semantic/adjudicated scoring audit.
- Six examples per guardrail category are sufficient for the preregistered
  gate but too small for a strong category-level claim.

## Next research target

Run one controlled ablation on the same split that separates three factors:
evidence availability, presence of the untrusted draft, and explicit
support/contradict/insufficient classification. Add blinded semantic
adjudication for disagreements with alias scoring. Do not begin training or
activation intervention until that ablation shows a revision-loop advantage
that is not explained by ordinary evidence-only prompting.

The raw reports, gate decision, and fingerprinted manifest are tracked under
`artifacts/baselines/belief_revision_text/kill-test-v1/real-model-results/`.

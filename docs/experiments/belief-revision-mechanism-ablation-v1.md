# Belief-Revision Mechanism Ablation v1

## Decision

`PAUSE_DISTINCT_REVISION_LOOP`

Keep the broader evidence-conditioned correction research active, but pause
the current draft-bearing revision loop as a candidate for training, LoRA,
DPO, or activation intervention. On both tested model families, retaining the
known-wrong draft reduced correction success relative to the same explicit
evidence judgment without the draft.

## Research question

The kill-test showed that evidence-conditioned revision can beat closed-book
self-correction, but it did not identify why. This experiment separates:

1. evidence availability;
2. exposure to an untrusted draft;
3. an explicit support/contradict/insufficient decision instruction.

It is a prompt-mechanism study, not evidence for a representation-level
intervention.

## Six-arm matrix

| Arm | Evidence | Draft | Explicit stance instruction |
| --- | --- | --- | --- |
| `closed_book_draft` | no | yes | no |
| `closed_book_explicit_review` | no | yes | yes |
| `evidence_only` | yes | no | no |
| `evidence_with_draft` | yes | yes | no |
| `evidence_only_explicit_stance` | yes | no | yes |
| `evidence_draft_explicit_stance` | yes | yes | yes |

The design is deliberately a set of estimable contrasts rather than a nominal
2×2×2 factorial. “Explicit evidence stance” is not meaningful when neither
evidence nor a draft is present.

## Primary contrasts

- Evidence effect, with and without explicit review.
- Draft anchoring, with and without explicit stance classification.
- Explicit stance effect, with and without an untrusted draft.

The same 48 examples, model revisions, decoding settings, and label-separation
contract from `kill-test-v1` must be reused.

## Semantic audit

Alias scoring remains the reproducible primary score. Before making a
mechanism claim, export every response to the blinded adjudication packet:

- model identity, arm identity, source example id, and alias score are hidden;
- two adjudicators work independently;
- each response receives a semantic verdict and a draft-persistence verdict;
- disagreements are resolved before unblinding.

The private blinding salt and identity map must not be committed.

## Results

All 48 examples were regenerated on all six arms with the same pinned model
revisions and greedy 64-token decoding as the kill-test.

| Model | Evidence only | Evidence + explicit stance | Draft + evidence + explicit stance | Draft-loop minus no-draft stance |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-0.5B | 0.833 | **0.833** | 0.417 | -0.417 |
| SmolLM2-135M | 0.521 | **0.813** | 0.708 | -0.104 |

The primary paired comparison was
`evidence_draft_explicit_stance - evidence_only_explicit_stance`:

- Qwen: 3 wins, 22 ties, 23 losses; paired bootstrap 95% interval
  `[-0.583, -0.250]`.
- SmolLM2: 7 wins, 29 ties, 12 losses; paired bootstrap 95% interval
  `[-0.271, 0.083]`.

Qwen shows a clear harmful draft-anchoring effect. SmolLM2 is less certain,
but its point estimate has the same negative direction. The preregistered
cross-model direction requirement therefore fails.

Category results also show that a single overall score is insufficient:

- Qwen evidence-only solved 94.4% of contradiction cases but none of the six
  insufficient-evidence cases.
- Adding explicit stance without a draft preserved strong Qwen contradiction
  performance and improved insufficient-evidence success to 33.3%.
- SmolLM2 explicit stance without a draft reached 80.6% on contradictions,
  100% on support, and 66.7% on insufficient evidence.
- The full SmolLM2 draft-bearing loop failed all six insufficient-evidence
  cases despite its higher aggregate score in the original kill-test.

## Interpretation

The original `CONTINUE_0_3` decision remains valid for evidence-conditioned
belief correction. The narrower mechanism claim does not survive ablation:

1. evidence availability explains the largest improvement over closed-book
   self-correction;
2. explicit evidence sufficiency/stance judgment can help, especially for
   SmolLM2;
3. retaining the wrong draft is not necessary and is harmful for Qwen;
4. the current evidence therefore favors a no-draft evidence controller over
   a draft-revision loop.

Blinded semantic adjudication remains useful for auditing alias-scoring error,
but it is not required to rescue a mechanism that already fails its
preregistered primary direction gate. No stronger semantic mechanism claim is
made from alias scoring alone.

## Decision boundary

Continue toward a more expensive intervention only if:

1. the direction of the relevant contrast agrees on both model families;
2. the contrast survives blinded semantic adjudication;
3. the explicit-stance arm improves over evidence-only, rather than merely
   improving over closed-book self-correction;
4. the effect is not explained by excessive abstention.

Otherwise retain evidence-only generation as the stronger comparator and
pause claims about a distinct revision-loop mechanism.

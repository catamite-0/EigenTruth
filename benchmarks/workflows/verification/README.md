# Verification Workflows

Home for counterfactual, triple, state, verifier-route, and belief-revision
workflow CLIs.

Current 0.3 entry point:

- `belief_revision_eval.py`: compares baseline prompt, self-correction prompt,
  evidence-only, and EigenTruth revision-loop behavior on text fixtures. This
  remains a plumbing smoke and is not eligible for the research kill gate.
- `build_belief_revision_kill_test.py`: deterministically builds the 48-example
  `kill-test-v1` split from tracked Wikidata structured-QA records. Runtime
  evidence and scoring labels are written to separate JSONL files.
- `belief_revision_real_model_eval.py`: generates all four arms with one pinned
  Hugging Face causal LM. Generator prompts are built only from the sanitized
  runtime JSONL; expected answers, accepted/rejected aliases, and action labels
  are joined only for scoring.
- `belief_revision_kill_gate.py`: rejects fixture, unfingerprinted, mixed-split,
  prompt-mismatched, or non-real-model reports before applying the go/pause
  thresholds.

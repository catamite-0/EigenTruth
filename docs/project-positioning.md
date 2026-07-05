# EigenTruth Project Positioning

EigenTruth 0.3 is an evidence-grounded self-revision research framework for
small and mid-sized open models. The project studies whether models that fail
to update from explicit evidence can be made less stubborn through structured
memory, claim-level revision, inference control, and later correction-training
data.

EigenTruth is not a truth oracle, a production hallucination detector, or a
guarantee that an answer is factual. It should make factuality failures easier
to measure, correct, trace, and eventually train against.

## Primary Audience

- Researchers studying evidence update failure, overconfident wrong answers,
  and representation instability.
- Engineers evaluating local or private LLM stacks where Qwen, DeepSeek,
  Llama-like, or similar open models are used for factual tasks.
- Product teams that need traceable correction and abstention behavior before
  trusting cheaper or private models.

## 0.3 Position

The 0.3 line changes the center of gravity:

- from broad hallucination detection to measured evidence-revision failure;
- from result-only output filtering to runtime correction traces;
- from artifact governance as the main story to artifact governance as the
  evidence backbone;
- from top closed models as the primary target to local, open, controllable
  models as the intervention target.

## Non-Goals

- Do not claim that EigenTruth solves hallucination generally.
- Do not treat GPT/Claude-class systems as the first intervention target.
- Do not train from unverified self-generated corrections.
- Do not use output gating as the main correction mechanism.
- Do not enable activation steering or logit correction by default.

## Success Standard

The first useful result is not a bigger benchmark matrix. It is a clear drop in
stubbornness and unsupported persistence on text evidence-revision fixtures,
with failure cases preserved as negative evidence.


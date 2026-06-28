# EigenTruth Methodology

EigenTruth is a research toolkit for monitoring representation drift and testing
activation steering interventions in decoder-only language models.

It is not a production hallucination detector and does not prove that a model
output is true.

## Core Idea

EigenTruth treats hallucination-prone generation as a potential geometric
deviation in hidden-state flow.

The current implementation builds a truth manifold from factual warmup samples,
then monitors the target layer's last-token hidden state during generation.
When the hidden state moves far from the warmup manifold, EigenTruth can inject
a steering vector toward a factual centroid or a contrastive truth direction.
New public aliases such as `RepresentationManifold`, `RepresentationProbe`, and
`RepresentationMonitor` reflect the broader representation-observability scope;
the existing `Truth*` names remain supported.

## Pipeline

1. Collect hidden states from a target Transformer layer on factual examples.
2. Incrementally build a `TruthManifold` using a Welford online mean and
   covariance, exposed as a ridge-regularized, sample-count-normalized precision
   matrix.
3. Optionally collect false examples to build a contrastive direction.
4. Register a PyTorch `forward_hook` on the selected layer.
5. During generation, compute:
   - Mahalanobis distance from the truth manifold
   - Optional Poincare projection of hidden states
   - Optional Hyperbolic Semantic Entropy over a sliding window
6. If distance exceeds the configured threshold, optionally inject a normalized
   steering vector into the last-token activation.

## What EigenTruth Measures

EigenTruth measures geometry in representation space:

- distance from a warmup manifold
- optional dispersion of recent hidden states in a hyperbolic projection
- whether a configured intervention changes generation trajectories

These diagnostics are useful for experiments, ablations, and qualitative demos.

## Distance Calibration

The `cov_inv` field is the inverse of a ridge-regularized sample covariance,
accumulated online with a numerically stable Welford update and normalized by
the warmup sample count. A fixed relative ridge keeps it well-conditioned even
when the warmup set is smaller than the hidden dimension. Because the covariance
is normalized by sample count, the Mahalanobis-distance scale is stable across
warmup-set sizes and does not collapse toward zero as more warmup samples are
added.

Mahalanobis thresholds are still model-, layer-, and dataset-dependent. Treat
them as experiment-specific hyperparameters and calibrate per model, target
layer, warmup set, and generation setup before comparing results.

## What EigenTruth Does Not Prove

EigenTruth does not prove:

- that an output is factually true
- that a model is safe
- that a hallucination has been eliminated
- that a correction is semantically valid
- that one model is globally better than another

The current metrics are representation diagnostics, not truth labels.

## Experimental Use

Use EigenTruth for controlled experiments:

- compare with and without steering on the same prompts
- sweep target layers and thresholds
- compare factual and false warmup sets
- record output differences, distance, and optional HSE ablations
- evaluate outputs with external benchmarks or human review

Recommended external evaluation:

- TruthfulQA
- HaluEval
- FEVER-style factuality tasks
- task-specific human evaluation

## Limitations

- Results depend strongly on the target layer.
- Warmup data quality matters.
- Small warmup sets can create fragile manifolds.
- Mahalanobis thresholds are not portable across models or layers without
  calibration.
- Steering can change wording without improving truthfulness.
- A lower Mahalanobis distance does not guarantee factual correctness.
- HSE is an experimental opt-in dispersion signal, not a calibrated risk score or default path.
- The current examples are demonstrations, not benchmark evidence.

## Current Status

EigenTruth is a research preview. The codebase has unit tests for numerical
stability, hook behavior, warmup, diagnostics, and wrapper lifecycle, but it
still needs standardized benchmark evaluation before any production claims.

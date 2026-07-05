# EigenTruth

**Alpha research-preview PyTorch toolkit for evidence-grounded LLM
self-revision, calibrated observability, verifier/control traces, and optional
activation steering.**

EigenTruth is for controlled experiments and reproducible diagnostics. It is not
a production hallucination detector, truth oracle, factuality guarantee, or
safety boundary for deployed systems.

## Current Status

- Package baseline: `0.2.0`.
- Core dependency: `torch>=2.0`.
- Optional Hugging Face, dataset, retrieval, database, rewrite-model, and
  world-model integrations stay behind extras or adapters.
- The 0.3 research direction focuses on reducing evidence-revision stubbornness
  in small and mid-sized open models.
- Artifact governance remains the evidence backbone, but the main research
  story is now text-first correction and inference control.

## Quick Start

```bash
python -m pip install -e ".[dev]"
make check-fast
```

Minimal monitor-only integration:

```python
from eigentruth import EigenTruthWrapper

monitor = EigenTruthWrapper(
    model=model,
    target_layer_idx=-8,
    steering_lambda=0.0,
)
monitor.warmup(fact_dataset, tokenizer)
output = monitor.generate(**inputs, max_new_tokens=50)
print(monitor.get_diagnostics())
```

Start with `steering_lambda=0.0` so diagnostics can be inspected without
changing model activations.

## Repository Map

- `src/eigentruth/`: public package and dependency-light core APIs.
- `benchmarks/workflows/`: reproducible research and release workflow entry
  points.
- `benchmarks/smokes/`: no-model smoke checks for workflow and release plumbing.
- `artifacts/baselines/`: small tracked artifacts that tests may depend on.
- `artifacts/release/`: tracked release evidence when explicitly promoted.
- `artifacts/local/` and `artifacts/runtime_evidence/`: ignored local scratch
  and heavyweight runtime outputs.

## Main Docs

- [Architecture](docs/architecture.md)
- [Project positioning](docs/project-positioning.md)
- [0.3 research direction](docs/research-direction.md)
- [Future multimodal extension](docs/multimodal-extension.md)
- [Artifact policy](docs/artifact-policy.md)
- [Testing](docs/testing.md)
- [Workflow guide](docs/workflows/README.md)
- [0.2.0 release notes](docs/release-0.2.0.md)
- [0.2.0 readiness snapshot](docs/release-0.2.0-readiness.md)
- [Product charter](docs/product-development-spec.md)
- [Methodology](docs/methodology.md)
- [Roadmap](ROADMAP.md)
- [Legacy long README archive](docs/evidence/legacy-readme-2026-07-05.md)

## Non-Goals

- Do not treat EigenTruth as proof that an answer is true.
- Do not use activation steering as a production safety mechanism.
- Do not treat output filtering as the main correction mechanism.
- Do not train from unverified self-generated correction samples.
- Do not promote adapter requests or local scratch outputs as evidence before
  they are source-backed, manifest-verified, and tied to a documented baseline
  or release boundary.

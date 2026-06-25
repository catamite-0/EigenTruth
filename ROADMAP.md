# EigenTruth Roadmap

EigenTruth is an alpha-stage research preview. This roadmap focuses on making experiments easier to reproduce and the project easier to review without implying production readiness.

The product and development charter lives in [`docs/product-development-spec.md`](docs/product-development-spec.md). It defines EigenTruth as a representation-observability and factuality-control toolkit rather than a standalone truth oracle.

The current packaged baseline is [`0.2.0`](docs/release-0.2.0.md): a calibrated-observability research release with local control-plane traces, dry-run/in-memory verification adapters, training telemetry primitives, and documented negative results. It remains alpha research software.

## Near-Term Milestones

### 0.3 Control And Verification Hardening

- Build stricter claim audit paths from structured triples and slot-level evidence coverage.
- Replicate the strongest diagnostics on additional small and mid-sized models.
- Promote only routes backed by artifact manifests, replay checks, and release gates.
- Keep network retrieval, external databases, and rewrite LLMs behind optional adapters.
- Convert more product traces into deterministic regression fixtures.

### Examples Cleanup

- Separate minimal integration examples from longer qualitative demonstrations.
- Document required downloads, expected runtime, and configuration values.
- Keep example outputs clearly labeled as observations rather than benchmark evidence.

### Test Coverage

- Add integration tests for additional Hugging Face output formats and layer layouts.
- Expand lifecycle tests for repeated warmup, probe attachment, and cleanup.
- Add edge-case tests for dtype, device placement, and batch behavior.

### CI Workflow

- Keep unit tests and `ruff` running on supported Python versions.
- Review dependency caching and test-duration reporting as the suite grows.
- Add targeted integration checks when reproducible lightweight fixtures are available.

### Documentation Improvements

- Expand API reference material for `TruthManifold`, `TruthProbe`, and `EigenTruthWrapper`.
- Document calibration workflows for layers, thresholds, and warmup datasets.
- Add a glossary for geometry terms and experimental metrics.

### Reproducible Experiment Scripts

- Add deterministic configuration files and command-line entry points.
- Record model revision, dataset provenance, random seeds, environment versions, and commit SHA.
- Add structured output suitable for external evaluation and human review.

### Security And Dependency Review

- Review dependency constraints and supported PyTorch and Transformers versions.
- Add routine dependency vulnerability checks.
- Review guidance for model loading and experiment scripts that handle untrusted inputs or remote code.

## Longer-Term Research Directions

- Evaluate diagnostics against external factuality benchmarks.
- Study layer-selection and threshold-calibration methods.
- Compare steering strategies through controlled ablations.
- Improve experiment reporting for reproducibility and peer review.

## 路线图说明

EigenTruth 是一个处于 alpha 阶段的研究预览项目。近期工作重点是提升实验可复现性和项目可审查性，而不是暗示生产可用性。

近期里程碑包括：清理示例脚本、扩展测试覆盖、维护 CI 工作流、完善文档、增加可复现实验脚本，以及执行安全与依赖审查。更长期的研究方向包括外部事实性基准评估、层选择与阈值校准、激活引导策略消融实验，以及更完善的实验报告。

# Testing

The test suite is split so ordinary development does not depend on heavyweight
release artifacts or ignored runtime evidence.

## Make Targets

```bash
make lint
make test-unit
make test-workflow
make test-artifacts
make check-fast
make check
make release-check
```

- `test-unit`: pytest tests that are not marked `artifact` or `workflow`.
- `test-workflow`: no-model workflow plumbing tests marked `workflow`.
- `test-artifacts`: tracked baseline/release artifact and manifest checks marked
  `artifact`.
- `check-fast`: lint, unit tests, and dependency consistency.
- `check`: `check-fast`, workflow tests, and deterministic smoke scripts.
- `release-check`: `check`, artifact tests, and package build.

## Pytest Markers

- `artifact`: reads promoted release evidence, tracked baseline artifacts, or
  release manifests.
- `workflow`: exercises no-model benchmark workflow plumbing and generated
  temporary artifacts.

Tests without these markers should be safe to run in a clean checkout without
`artifacts/runtime_evidence/`.

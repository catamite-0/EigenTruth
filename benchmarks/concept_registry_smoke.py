"""Smoke test for concept artifact registry and multi-probe monitoring."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from eigentruth.core.math_engine import TruthManifold  # noqa: E402
from eigentruth.intervention import MultiConceptMonitor  # noqa: E402
from eigentruth.registry import ArtifactRegistry, ConceptArtifact, build_artifact_manifest  # noqa: E402


class _TupleLayer(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(hidden_dim))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return self.linear(x), None


class _ToyModel(nn.Module):
    def __init__(self, *, hidden_dim: int = 8, n_layers: int = 2) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_TupleLayer(hidden_dim) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x, _ = layer(x)
        return x


def _build_manifold(center: torch.Tensor, *, seed: int, n_samples: int = 12) -> TruthManifold:
    generator = torch.Generator().manual_seed(seed)
    manifold = TruthManifold(covariance_mode="diag")
    for _ in range(n_samples):
        manifold.update(center + 0.05 * torch.randn(center.shape, generator=generator))
    direction = center.to(torch.float32)
    manifold.contrastive_direction = direction / torch.norm(direction).clamp(min=1e-8)
    return manifold


def run(
    *,
    output_dir: str | Path,
    hidden_dim: int = 8,
    seed: int = 13,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    registry_path = output_path / "registry.json"
    report_path = output_path / "concept-registry-smoke.json"
    manifest_path = output_path / "artifact-manifest.json"

    base = torch.zeros(hidden_dim)
    factuality_center = base.clone()
    factuality_center[0] = 1.0
    policy_center = base.clone()
    policy_center[1] = 1.0

    concepts = [
        ConceptArtifact(
            name="factuality",
            version="e8-smoke",
            layer_idx=0,
            manifold=_build_manifold(factuality_center, seed=seed),
            description="Synthetic factuality concept for E8 smoke.",
            metadata={"source": "synthetic", "axis": "dim0"},
        ),
        ConceptArtifact(
            name="policy_consistency",
            version="e8-smoke",
            layer_idx=1,
            manifold=_build_manifold(policy_center, seed=seed + 1),
            description="Synthetic policy-consistency concept for E8 smoke.",
            metadata={"source": "synthetic", "axis": "dim1"},
        ),
    ]

    registry = ArtifactRegistry.load_json(registry_path)
    concept_paths: dict[str, Path] = {}
    for concept in concepts:
        concept_path = output_path / f"{concept.name}.pt"
        concept.save(concept_path)
        concept_paths[concept.name] = concept_path
        registry.record_concept_artifact(
            name=concept.name,
            path=concept_path,
            version=concept.version,
            metadata=concept.to_dict(),
        )

    monitor = MultiConceptMonitor.from_artifacts(concepts, threshold=1e6, steering_lambda=0.0)
    model = _ToyModel(hidden_dim=hidden_dim, n_layers=2)
    monitor.register(model)
    torch.manual_seed(seed)
    _ = model(torch.randn(2, 3, hidden_dim))
    diagnostics = monitor.diagnostics()
    monitor.remove()

    report = {
        "status": "pass",
        "workflow": "concept_registry_smoke",
        "concept_count": len(concepts),
        "concept_names": [concept.name for concept in concepts],
        "diagnostics": diagnostics,
        "paths": {
            "registry": str(registry_path),
            "report": str(report_path),
            "artifact_manifest": str(manifest_path),
            "concept_artifacts": {name: str(path) for name, path in concept_paths.items()},
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest = build_artifact_manifest(
        {
            "concept_registry_smoke_report": report_path,
            "concepts.factuality": concept_paths["factuality"],
            "concepts.policy_consistency": concept_paths["policy_consistency"],
        },
        root=output_path,
        metadata={
            "workflow": "concept_registry_smoke",
            "concept_count": len(concepts),
        },
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    registry.record_benchmark_manifest(
        name="concept-registry-smoke",
        path=manifest_path,
        version="e8-smoke",
        metadata={"workflow": "concept_registry_smoke", "concept_count": len(concepts)},
    ).record_report(
        name="concept-registry-smoke",
        path=report_path,
        version="e8-smoke",
        metadata={"workflow": "concept_registry_smoke", "status": "pass"},
    ).save_json()
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run concept registry + multi-probe smoke test")
    parser.add_argument("--output-dir", default="artifacts/e8-concept-registry-smoke")
    parser.add_argument("--hidden-dim", type=int, default=8)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args(argv)
    run(output_dir=args.output_dir, hidden_dim=args.hidden_dim, seed=args.seed)


if __name__ == "__main__":
    main()

import importlib
import json

import pytest
import torch
import torch.nn as nn

from eigentruth.core.math_engine import TruthManifold
from eigentruth.intervention import MultiConceptMonitor
from eigentruth.registry import ArtifactRegistry, ConceptArtifact, load_concept_artifact


class _TupleLayer(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(hidden_dim))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return self.linear(x), None


class _ToyModel(nn.Module):
    def __init__(self, *, hidden_dim: int = 6, n_layers: int = 2) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_TupleLayer(hidden_dim) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x, _ = layer(x)
        return x


def _build_manifold(center: torch.Tensor, *, seed: int = 0) -> TruthManifold:
    generator = torch.Generator().manual_seed(seed)
    manifold = TruthManifold(covariance_mode="diag")
    for _ in range(8):
        manifold.update(center + 0.05 * torch.randn(center.shape, generator=generator))
    manifold.contrastive_direction = center / torch.norm(center).clamp(min=1e-8)
    assert manifold.is_ready()
    return manifold


def _concept(name: str, layer_idx: int, center_dim: int, *, hidden_dim: int = 6) -> ConceptArtifact:
    center = torch.zeros(hidden_dim)
    center[center_dim] = 1.0
    return ConceptArtifact(
        name=name,
        version="v1",
        layer_idx=layer_idx,
        manifold=_build_manifold(center, seed=center_dim),
        description=f"{name} concept",
        metadata={"center_dim": center_dim},
    )


def test_concept_artifact_roundtrip_preserves_manifold(tmp_path):
    artifact = _concept("factuality", 0, 0)
    path = tmp_path / "factuality.pt"

    artifact.save(path)
    loaded = load_concept_artifact(path)

    assert loaded.name == "factuality"
    assert loaded.version == "v1"
    assert loaded.layer_idx == 0
    assert loaded.metadata["center_dim"] == 0
    assert loaded.manifold.is_ready()
    assert torch.allclose(loaded.manifold.mean, artifact.manifold.mean)
    assert loaded.to_dict()["manifold"]["has_contrastive_direction"] is True


def test_concept_artifact_rejects_unready_manifold(tmp_path):
    artifact = ConceptArtifact(
        name="unready",
        version="v1",
        layer_idx=0,
        manifold=TruthManifold(),
    )

    with pytest.raises(ValueError, match="ready TruthManifold"):
        artifact.save(tmp_path / "unready.pt")


def test_artifact_registry_records_concept_artifact(tmp_path):
    registry_path = tmp_path / "registry.json"
    concept_path = tmp_path / "concept.pt"
    _concept("policy_consistency", 1, 1).save(concept_path)

    ArtifactRegistry.load_json(registry_path).record_concept_artifact(
        name="policy_consistency",
        path=concept_path,
        version="v1",
        metadata={"layer_idx": 1},
    ).save_json()
    loaded = ArtifactRegistry.load_json(registry_path)

    record = loaded.get("concept_artifact:policy_consistency:v1")
    assert record.artifact_type == "concept_artifact"
    assert record.path == str(concept_path)
    assert record.metadata["layer_idx"] == 1


def test_multi_concept_monitor_attaches_two_concepts(tmp_path):
    first_path = tmp_path / "factuality.pt"
    second_path = tmp_path / "policy.pt"
    _concept("factuality", 0, 0).save(first_path)
    _concept("policy_consistency", 1, 1).save(second_path)
    concepts = [ConceptArtifact.load(first_path), ConceptArtifact.load(second_path)]
    monitor = MultiConceptMonitor.from_artifacts(concepts, threshold=1e6, steering_lambda=0.0)
    model = _ToyModel(hidden_dim=6, n_layers=2)

    monitor.register(model)
    assert monitor.is_active
    _ = model(torch.randn(2, 3, 6))
    diagnostics = monitor.diagnostics()
    monitor.remove()

    assert diagnostics["concept_count"] == 2
    assert set(diagnostics["concepts"]) == {"factuality", "policy_consistency"}
    assert diagnostics["concepts"]["factuality"]["probe_active"] is True
    assert diagnostics["concepts"]["policy_consistency"]["probe_active"] is True
    assert diagnostics["concepts"]["factuality"]["last_mahalanobis_distance"] >= 0.0
    assert monitor.is_active is False


def test_multi_concept_monitor_rejects_duplicate_names():
    concept = _concept("duplicate", 0, 0)

    with pytest.raises(ValueError, match="unique"):
        MultiConceptMonitor.from_artifacts([concept, concept])


def test_concept_registry_smoke_writes_two_concepts(tmp_path, capsys):
    module = importlib.import_module("benchmarks.concept_registry_smoke")

    report = module.run(output_dir=tmp_path)
    captured = capsys.readouterr()

    assert report["status"] == "pass"
    assert report["concept_count"] == 2
    assert set(report["diagnostics"]["concepts"]) == {"factuality", "policy_consistency"}
    assert (tmp_path / "factuality.pt").exists()
    assert (tmp_path / "policy_consistency.pt").exists()
    assert (tmp_path / "artifact-manifest.json").exists()
    manifest = json.loads((tmp_path / "artifact-manifest.json").read_text(encoding="utf-8"))
    assert manifest["summary"]["artifact_count"] == 3
    registry = ArtifactRegistry.load_json(tmp_path / "registry.json")
    assert len(registry.list_records(artifact_type="concept_artifact")) == 2
    assert '"workflow": "concept_registry_smoke"' in captured.out

"""Small metadata records for saved manifolds, subspaces, and calibrations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from eigentruth.json_utils import strict_json_dumps, to_jsonable


@dataclass(frozen=True)
class RegistryRecord:
    """Metadata for one saved EigenTruth artifact."""

    name: str
    artifact_type: str
    path: str
    version: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def key(self) -> str:
        """Return a stable registry key."""
        return f"{self.artifact_type}:{self.name}:{self.version}"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "name": self.name,
            "artifact_type": self.artifact_type,
            "path": self.path,
            "version": self.version,
            "metadata": to_jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RegistryRecord":
        """Build a registry record from JSON-like data."""
        return cls(
            name=str(data["name"]),
            artifact_type=str(data["artifact_type"]),
            path=str(data["path"]),
            version=str(data["version"]),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class ArtifactRegistry:
    """JSON-backed registry for local EigenTruth artifacts."""

    path: str | Path
    records: tuple[RegistryRecord, ...] = ()
    schema_version: int = 1

    def record_artifact(
        self,
        *,
        name: str,
        artifact_type: str,
        path: str | Path,
        version: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ArtifactRegistry":
        """Add or replace metadata for a saved artifact."""
        return self.add(
            RegistryRecord(
                name=name,
                artifact_type=artifact_type,
                path=str(path),
                version=version,
                metadata={} if metadata is None else dict(metadata),
            )
        )

    def record_calibration_report(
        self,
        *,
        name: str,
        path: str | Path,
        version: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ArtifactRegistry":
        """Record a calibration sweep/report artifact."""
        return self.record_artifact(
            name=name,
            artifact_type="calibration_report",
            path=path,
            version=version,
            metadata=metadata,
        )

    def record_calibration_artifact(
        self,
        *,
        name: str,
        path: str | Path,
        version: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ArtifactRegistry":
        """Record a calibration artifact."""
        return self.record_artifact(
            name=name,
            artifact_type="calibration_artifact",
            path=path,
            version=version,
            metadata=metadata,
        )

    def record_score_fusion_artifact(
        self,
        *,
        name: str,
        path: str | Path,
        version: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ArtifactRegistry":
        """Record a calibrated diagnostic score-fusion artifact."""
        return self.record_artifact(
            name=name,
            artifact_type="score_fusion_artifact",
            path=path,
            version=version,
            metadata=metadata,
        )

    def record_truth_subspace_artifact(
        self,
        *,
        name: str,
        path: str | Path,
        version: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ArtifactRegistry":
        """Record a truth-subspace artifact."""
        return self.record_artifact(
            name=name,
            artifact_type="truth_subspace_artifact",
            path=path,
            version=version,
            metadata=metadata,
        )

    def record_trace(
        self,
        *,
        name: str,
        path: str | Path,
        version: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ArtifactRegistry":
        """Record a product trace artifact."""
        return self.record_artifact(
            name=name,
            artifact_type="product_trace",
            path=path,
            version=version,
            metadata=metadata,
        )

    def record_report(
        self,
        *,
        name: str,
        path: str | Path,
        version: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ArtifactRegistry":
        """Record a generic report artifact."""
        return self.record_artifact(
            name=name,
            artifact_type="report",
            path=path,
            version=version,
            metadata=metadata,
        )

    def record_benchmark_manifest(
        self,
        *,
        name: str,
        path: str | Path,
        version: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ArtifactRegistry":
        """Record a verified benchmark artifact manifest."""
        return self.record_artifact(
            name=name,
            artifact_type="benchmark_manifest",
            path=path,
            version=version,
            metadata=metadata,
        )

    def record_manifest_verification(
        self,
        *,
        name: str,
        path: str | Path,
        version: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ArtifactRegistry":
        """Record a manifest verification report."""
        return self.record_artifact(
            name=name,
            artifact_type="manifest_verification",
            path=path,
            version=version,
            metadata=metadata,
        )

    def record_action_result(
        self,
        *,
        name: str,
        path: str | Path,
        version: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ArtifactRegistry":
        """Record saved action execution results."""
        return self.record_artifact(
            name=name,
            artifact_type="action_result",
            path=path,
            version=version,
            metadata=metadata,
        )

    def record_performance_baseline(
        self,
        *,
        name: str,
        path: str | Path,
        version: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ArtifactRegistry":
        """Record a product performance baseline workflow report."""
        return self.record_artifact(
            name=name,
            artifact_type="performance_baseline",
            path=path,
            version=version,
            metadata=metadata,
        )

    def record_product_runtime_baseline(
        self,
        *,
        name: str,
        path: str | Path,
        version: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ArtifactRegistry":
        """Record an aggregate ProductTrace runtime baseline report."""
        return self.record_artifact(
            name=name,
            artifact_type="product_runtime_baseline",
            path=path,
            version=version,
            metadata=metadata,
        )

    def record_product_runtime_drift_report(
        self,
        *,
        name: str,
        path: str | Path,
        version: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ArtifactRegistry":
        """Record a ProductTrace runtime baseline drift report."""
        return self.record_artifact(
            name=name,
            artifact_type="product_runtime_drift_report",
            path=path,
            version=version,
            metadata=metadata,
        )

    def record_product_runtime_budget_policy(
        self,
        *,
        name: str,
        path: str | Path,
        version: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ArtifactRegistry":
        """Record a reusable ProductRuntimeBudgetPolicy artifact."""
        return self.record_artifact(
            name=name,
            artifact_type="product_runtime_budget_policy",
            path=path,
            version=version,
            metadata=metadata,
        )

    def record_product_promotion_contract(
        self,
        *,
        name: str,
        path: str | Path,
        version: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ArtifactRegistry":
        """Record a deployable ProductPromotionContract artifact."""
        return self.record_artifact(
            name=name,
            artifact_type="product_promotion_contract",
            path=path,
            version=version,
            metadata=metadata,
        )

    def add(self, record: RegistryRecord) -> "ArtifactRegistry":
        """Add or replace a record with the same registry key."""
        records = [existing for existing in self.records if existing.key() != record.key()]
        records.append(record)
        self.records = tuple(records)
        return self

    def get(self, key: str) -> RegistryRecord:
        """Return one record by stable registry key."""
        for record in self.records:
            if record.key() == key:
                return record
        raise KeyError(key)

    def list_records(self, *, artifact_type: str | None = None) -> tuple[RegistryRecord, ...]:
        """Return records, optionally filtered by artifact type."""
        if artifact_type is None:
            return self.records
        return tuple(record for record in self.records if record.artifact_type == artifact_type)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "schema_version": self.schema_version,
            "records": [record.to_dict() for record in self.records],
        }

    @classmethod
    def from_dict(cls, path: str | Path, data: Mapping[str, Any]) -> "ArtifactRegistry":
        """Build a registry from JSON-like data."""
        return cls(
            path=path,
            records=tuple(RegistryRecord.from_dict(record) for record in data.get("records", ())),
            schema_version=int(data.get("schema_version", 1)),
        )

    def save_json(self, path: str | Path | None = None) -> None:
        """Save registry records to UTF-8 JSON."""
        output_path = Path(path or self.path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(strict_json_dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "ArtifactRegistry":
        """Load registry records from UTF-8 JSON, returning an empty registry if absent."""
        registry_path = Path(path)
        if not registry_path.exists():
            return cls(path=registry_path)
        return cls.from_dict(registry_path, json.loads(registry_path.read_text(encoding="utf-8")))

"""Export a deployable ProductPromotionContract from release evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eigentruth.control import ProductPromotionContract  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402


def export_product_promotion_contract(
    *,
    source_path: str | Path,
    output_path: str | Path,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    compact_json: bool = False,
) -> dict[str, Any]:
    """Write a ProductPromotionContract JSON and optional manifest/registry record."""
    source = Path(source_path)
    output = Path(output_path)
    manifest_path = None if artifact_manifest_path is None else Path(artifact_manifest_path)
    if (name or version) and (registry_path is None or name is None or version is None):
        raise ValueError("registry export requires registry_path, name, and version.")

    contract = ProductPromotionContract.from_json(source)
    payload = contract.to_dict()
    export_metadata = dict(metadata or {})
    _write_json(output, payload, compact=compact_json)

    manifest = None
    if manifest_path is not None:
        manifest = build_artifact_manifest(
            {
                "product_promotion_contract": output,
                "source_release_candidate": source,
            },
            root=manifest_path.parent,
            metadata={
                "runner": "export_product_promotion_contract",
                "source": str(source),
                "compact_json": compact_json,
                **export_metadata,
            },
        )
        _write_json(manifest_path, manifest, compact=compact_json)

    if registry_path is not None and name is not None and version is not None:
        registry = ArtifactRegistry.load_json(registry_path)
        registry.record_product_promotion_contract(
            name=name,
            path=output,
            version=version,
            metadata={
                "workflow": "export_product_promotion_contract",
                "source": str(source),
                "artifact_manifest": None if manifest_path is None else str(manifest_path),
                "source_workflow": contract.source_workflow,
                "source_status": contract.source_status,
                "model_id": contract.model_id,
                "recommended_route": contract.metadata.get("recommended_route"),
                "recommended_selector_replay_candidate": contract.metadata.get(
                    "recommended_selector_replay_candidate"
                ),
                "product_runtime_drift_status": contract.metadata.get("product_runtime_drift_status"),
                "compact_json": compact_json,
                **export_metadata,
            },
        )
        registry.save_json()

    return {
        "schema_version": 1,
        "workflow": "export_product_promotion_contract",
        "status": "exported",
        "paths": {
            "source": str(source),
            "contract": str(output),
            "artifact_manifest": None if manifest_path is None else str(manifest_path),
            "registry": None if registry_path is None else str(registry_path),
        },
        "contract": {
            "model_id": contract.model_id,
            "source_workflow": contract.source_workflow,
            "source_status": contract.source_status,
            "runtime": dict(contract.runtime),
            "verifier_route": dict(contract.verifier_route),
            "metadata": dict(contract.metadata),
        },
        "artifact_manifest_summary": None if manifest is None else manifest.get("summary"),
    }


def _write_json(path: str | Path, payload: Mapping[str, Any], *, compact: bool) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    else:
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    output.write_text(text, encoding="utf-8")


def _parse_metadata(values: Sequence[str]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--metadata entries must be formatted as key=value.")
        key, raw = value.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("--metadata keys must be non-empty.")
        try:
            metadata[key] = json.loads(raw)
        except json.JSONDecodeError:
            metadata[key] = raw
    return metadata


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run from parsed CLI arguments."""
    payload = export_product_promotion_contract(
        source_path=args.source,
        output_path=args.output,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        metadata=_parse_metadata(args.metadata or ()),
        compact_json=bool(args.compact_json),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Export a lightweight ProductPromotionContract from release evidence"
    )
    parser.add_argument("--source", required=True, help="release workflow/comparison or product contract JSON")
    parser.add_argument("--output", required=True, help="output ProductPromotionContract JSON")
    parser.add_argument("--artifact-manifest", default=None, help="optional artifact manifest output path")
    parser.add_argument("--registry", default=None, help="optional ArtifactRegistry JSON path")
    parser.add_argument("--name", default=None, help="optional registry record name")
    parser.add_argument("--version", default=None, help="optional registry record version")
    parser.add_argument("--metadata", action="append", default=[], help="metadata key=value; repeatable")
    parser.add_argument("--compact-json", action="store_true", help="write compact JSON artifacts")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()

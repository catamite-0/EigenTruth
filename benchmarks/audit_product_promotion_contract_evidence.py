"""Audit ProductPromotionContract runtime-drift evidence handoff."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eigentruth.control import audit_product_promotion_contract_evidence  # noqa: E402
from eigentruth.json_utils import strict_json_dumps  # noqa: E402
from eigentruth.registry import ArtifactRegistry  # noqa: E402


def build_product_promotion_evidence_audit(
    *,
    contract: str | Path,
    json_path: str | Path,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    required_groups: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write and optionally register a product-promotion evidence audit."""
    contract_path = Path(contract)
    output_path = Path(json_path)
    if (name or version) and (registry_path is None or name is None or version is None):
        raise ValueError("registry export requires registry_path, name, and version.")

    payload = _load_json_object(contract_path)
    audit = audit_product_promotion_contract_evidence(
        payload,
        required_groups=required_groups,
        metadata={"contract": str(contract_path), **dict(metadata or {})},
    )
    audit_payload = audit.to_dict()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(strict_json_dumps(audit_payload, indent=2), encoding="utf-8")

    if registry_path is not None and name is not None and version is not None:
        registry = ArtifactRegistry.load_json(registry_path)
        registry.record_product_promotion_evidence_audit(
            name=name,
            path=output_path,
            version=version,
            metadata={
                "workflow": audit_payload["workflow"],
                "status": audit.status,
                "contract": str(contract_path),
                "source_status": audit.source_status,
                "missing_metric_count": audit.summary["missing_metric_count"],
                "blocked_group_count": audit.summary["blocked_group_count"],
                "recommended_action_ids": audit.recommended_action_ids,
                **dict(metadata or {}),
            },
        )
        registry.save_json(registry_path)
    return audit_payload


def _load_json_object(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _parse_required_groups(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    groups = tuple(item.strip() for item in value.split(",") if item.strip())
    return groups or None


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


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Audit ProductPromotionContract evidence fields required by runtime-drift gates"
    )
    parser.add_argument("--contract", required=True, help="ProductPromotionContract JSON path")
    parser.add_argument("--json", required=True, help="output audit JSON path")
    parser.add_argument("--registry", default=None, help="optional local ArtifactRegistry JSON path")
    parser.add_argument("--name", default=None, help="registry audit name")
    parser.add_argument("--version", default=None, help="registry audit version")
    parser.add_argument(
        "--required-groups",
        default=None,
        help="comma-list of required evidence groups; defaults to all groups",
    )
    parser.add_argument("--metadata", action="append", default=[], help="metadata key=value; repeatable")
    args = parser.parse_args(argv)
    payload = build_product_promotion_evidence_audit(
        contract=args.contract,
        json_path=args.json,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        required_groups=_parse_required_groups(args.required_groups),
        metadata=_parse_metadata(args.metadata or ()),
    )
    print(
        "product_promotion_evidence_audit="
        f"{payload['status']} "
        f"groups={payload['summary']['group_count']} "
        f"missing_metrics={payload['summary']['missing_metric_count']} "
        f"actions={len(payload['recommended_action_ids'])}"
    )


if __name__ == "__main__":
    main()

"""Select sparse hidden evidence from score-dump diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eigentruth.eval import (  # noqa: E402
    HiddenEvidenceSelectionPolicy,
    select_hidden_evidence_from_score_dump,
)
from eigentruth.json_utils import strict_json_dumps  # noqa: E402
from eigentruth.registry import ArtifactRegistry  # noqa: E402


def hidden_evidence_selection_report(
    scores_path: str | Path,
    *,
    signals: tuple[str, ...] | None = None,
    sweep_signals: tuple[str, ...] | None = None,
    include_primary: bool = True,
    include_sweep: bool = True,
    directions: Mapping[str, str] | None = None,
    max_items: int = 32,
    max_per_record: int | None = 4,
    max_per_layer: int | None = None,
    max_per_score: int | None = None,
    min_anomaly_score: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a JSON payload selecting sparse hidden evidence from scores."""
    policy = HiddenEvidenceSelectionPolicy(
        max_items=max_items,
        max_per_record=max_per_record,
        max_per_layer=max_per_layer,
        max_per_score=max_per_score,
        min_anomaly_score=min_anomaly_score,
    )
    report = select_hidden_evidence_from_score_dump(
        scores_path,
        score_names=signals,
        sweep_score_names=sweep_signals,
        include_primary=include_primary,
        include_sweep=include_sweep,
        directions={} if directions is None else directions,
        policy=policy,
        metadata={} if metadata is None else metadata,
    )
    return report.to_dict()


def run(args: argparse.Namespace) -> dict[str, Any]:
    """CLI entry point with a testable args namespace."""
    scores_path = Path(args.scores)
    signals = _comma_tuple(args.signals)
    sweep_signals = _comma_tuple(args.sweep_signals)
    directions = _parse_directions(args.direction)
    payload = hidden_evidence_selection_report(
        scores_path,
        signals=signals,
        sweep_signals=sweep_signals,
        include_primary=not bool(args.no_primary),
        include_sweep=not bool(args.no_sweep),
        directions=directions,
        max_items=int(args.max_items),
        max_per_record=_optional_int(args.max_per_record),
        max_per_layer=_optional_int(args.max_per_layer),
        max_per_score=_optional_int(args.max_per_score),
        min_anomaly_score=_optional_float(args.min_anomaly_score),
        metadata={"source_scores_path": str(scores_path)},
    )
    payload = _json_roundtrip(payload)
    output = None if args.json is None else Path(args.json)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(strict_json_dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.registry is not None:
        if output is None:
            raise ValueError("--registry requires --json so the report path can be recorded.")
        registry_path = Path(args.registry)
        register_name = str(args.register_name or output.stem)
        version = str(args.version)
        registry = ArtifactRegistry.load_json(registry_path)
        registry.record_report(
            name=register_name,
            path=output,
            version=version,
            metadata={
                "artifact_kind": "hidden_evidence_selection",
                "summary": payload["summary"],
                "source_scores_path": str(scores_path),
            },
        ).save_json()
    if not args.quiet:
        summary = payload["summary"]
        print(
            "hidden-evidence: "
            f"selected={summary['selected_count']} "
            f"records={summary['selected_record_count']} "
            f"channels={summary['channel_count']} "
            f"budget_exhausted={summary['budget_exhausted']}"
        )
        for item in payload["selected"]:
            layer = item["layer"] if item["layer"] is not None else "primary"
            print(
                f"  #{item['rank']} record={item['record_id']} layer={layer} "
                f"score={item['score_name']} anomaly={item['anomaly_score']:.6g} "
                f"ref={item['evidence_ref']}"
            )
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Select sparse hidden evidence from score dumps")
    parser.add_argument("--scores", required=True, help="score dump JSON or JSONL manifest path")
    parser.add_argument("--json", default=None, help="optional output report path")
    parser.add_argument("--signals", default=None, help="comma-list of primary score names")
    parser.add_argument("--sweep-signals", default=None, help="comma-list of sweep score names")
    parser.add_argument("--no-primary", action="store_true", help="ignore primary scores")
    parser.add_argument("--no-sweep", action="store_true", help="ignore sweep_scores")
    parser.add_argument(
        "--direction",
        action="append",
        default=(),
        help="score direction override, e.g. selfcheck_support_rate=lower or -8:score=lower",
    )
    parser.add_argument("--max-items", type=int, default=32)
    parser.add_argument("--max-per-record", default="4", help="positive integer or none")
    parser.add_argument("--max-per-layer", default=None, help="positive integer or none")
    parser.add_argument("--max-per-score", default=None, help="positive integer or none")
    parser.add_argument("--min-anomaly-score", default=None, help="optional threshold in [0, 1]")
    parser.add_argument("--registry", default=None, help="optional ArtifactRegistry JSON path")
    parser.add_argument("--register-name", default=None, help="registry record name")
    parser.add_argument("--version", default="0.1", help="registry record version")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    run(args)


def _comma_tuple(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    items = tuple(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))
    return items or None


def _parse_directions(values: tuple[str, ...] | list[str]) -> dict[str, str]:
    directions: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--direction entries must be NAME=higher|lower.")
        name, direction = value.split("=", 1)
        name = name.strip()
        direction = direction.strip()
        if not name:
            raise ValueError("--direction name must be non-empty.")
        if direction not in {"higher", "lower"}:
            raise ValueError("--direction must be higher or lower.")
        directions[name] = direction
    return directions


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "none", "null"}:
        return None
    return int(text)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "none", "null"}:
        return None
    return float(text)


def _json_roundtrip(payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(strict_json_dumps(payload))


if __name__ == "__main__":
    main()

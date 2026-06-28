"""Evaluate cheap text redline baselines for pre-generation probe records."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.build_text_baseline_score_dump import (  # noqa: E402
    DEFAULT_TEXT_BASELINE_SIGNALS,
    text_baseline_features,
    text_baseline_signal_definitions,
)
from eigentruth.eval.metrics import roc_auc  # noqa: E402
from eigentruth.json_utils import to_jsonable  # noqa: E402
from eigentruth.registry import build_artifact_manifest  # noqa: E402


@dataclass(frozen=True)
class PreGenerationTextBaselineConfig:
    """Configuration for text-baseline evaluation over pre-generation records."""

    records_path: str | Path
    output_path: str | Path
    artifact_manifest_path: str | Path | None = None
    baseline_signals: Sequence[str] = DEFAULT_TEXT_BASELINE_SIGNALS
    compact_json: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "records_path", Path(self.records_path))
        object.__setattr__(self, "output_path", Path(self.output_path))
        if self.artifact_manifest_path is not None:
            object.__setattr__(self, "artifact_manifest_path", Path(self.artifact_manifest_path))
        signals = tuple(str(signal) for signal in self.baseline_signals if str(signal))
        if not signals:
            raise ValueError("baseline_signals must not be empty.")
        if len(set(signals)) != len(signals):
            raise ValueError("baseline_signals must be unique.")
        object.__setattr__(self, "baseline_signals", signals)


def run_pre_generation_text_baselines(config: PreGenerationTextBaselineConfig) -> dict[str, Any]:
    """Evaluate cheap text/length baselines against hard labels in records."""
    records = _load_text_records(config.records_path)
    labels = tuple(record["label"] for record in records)
    label_counts = {
        "negative": sum(1 for label in labels if label == 0),
        "positive": sum(1 for label in labels if label == 1),
    }
    if not label_counts["negative"] or not label_counts["positive"]:
        raise ValueError("pre-generation text baselines require both positive and negative labels.")

    signal_definitions = text_baseline_signal_definitions()
    signal_reports = {}
    for signal in config.baseline_signals:
        if signal not in signal_definitions:
            raise ValueError(f"unknown text baseline signal: {signal!r}")
        values = tuple(float(record["features"][signal]) for record in records)
        auroc_higher = roc_auc(values, labels)
        auroc_lower = roc_auc(tuple(-value for value in values), labels)
        if auroc_higher >= auroc_lower:
            direction = "higher"
            best_auroc = auroc_higher
        else:
            direction = "lower"
            best_auroc = auroc_lower
        signal_reports[signal] = {
            "definition": signal_definitions[signal],
            "direction": direction,
            "auroc": best_auroc,
            "auroc_higher": auroc_higher,
            "auroc_lower": auroc_lower,
            "summary": _numeric_summary(values),
        }
    best_name, best_report = max(
        signal_reports.items(),
        key=lambda item: (float(item[1]["auroc"]), item[0]),
    )
    metadata = _first_metadata(records)
    payload = {
        "schema_version": 1,
        "workflow": "pre_generation_text_baseline_eval",
        "status": "ready",
        "records_path": str(config.records_path),
        "record_count": len(records),
        "label_counts": label_counts,
        "metadata": metadata,
        "signals": signal_reports,
        "best_signal": {
            "name": best_name,
            "direction": best_report["direction"],
            "auroc": best_report["auroc"],
        },
        "paths": {
            "report": str(config.output_path),
            "artifact_manifest": None if config.artifact_manifest_path is None else str(config.artifact_manifest_path),
        },
        "evidence_scope": {
            "claim": "cheap text redline baseline for pre-generation probe records",
            "not_a_claim": "representation-based hallucination detection",
        },
    }
    _write_json(config.output_path, payload, compact=config.compact_json)
    if config.artifact_manifest_path is not None:
        manifest = build_artifact_manifest(
            {
                "text_baseline_report": config.output_path,
                "records": config.records_path,
            },
            root=Path(config.artifact_manifest_path).parent,
            metadata={
                "workflow": "pre_generation_text_baseline_eval",
                "status": "ready",
                "record_count": len(records),
                "metadata_model": metadata.get("model"),
                "best_signal": best_name,
                "best_auroc": best_report["auroc"],
            },
        )
        _write_json(config.artifact_manifest_path, manifest, compact=False)
        payload["artifact_manifest_summary"] = manifest["summary"]
        _write_json(config.output_path, payload, compact=config.compact_json)
    print(
        "pre_generation_text_baseline_eval_ok "
        f"records={len(records)} best={best_name} auroc={float(best_report['auroc']):.3f} "
        f"output={config.output_path}"
    )
    return to_jsonable(payload)


def _load_text_records(path: Path) -> tuple[dict[str, Any], ...]:
    records = []
    for index, payload in enumerate(_iter_record_payloads(path)):
        metadata = _record_metadata(payload)
        label = _record_label(payload, index=index)
        features = text_baseline_features(metadata)
        records.append({
            "label": label,
            "metadata": metadata,
            "features": features,
        })
    if not records:
        raise ValueError("records must not be empty.")
    return tuple(records)


def _iter_record_payloads(path: Path) -> tuple[Mapping[str, Any], ...]:
    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open(encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, Mapping):
                    raise ValueError(f"records JSONL line {line_number} must be an object.")
                records.append(payload)
        return tuple(records)
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_records = payload.get("records") if isinstance(payload, Mapping) else payload
    if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes, bytearray)):
        raise ValueError("records JSON must be a list or an object with a records list.")
    records = []
    for index, item in enumerate(raw_records):
        if not isinstance(item, Mapping):
            raise ValueError(f"record {index} must be an object.")
        records.append(item)
    return tuple(records)


def _record_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        metadata = {}
    merged = dict(metadata)
    for field in ("question", "answer", "claim", "text"):
        if field in payload and field not in merged:
            merged[field] = payload[field]
    return merged


def _record_label(payload: Mapping[str, Any], *, index: int) -> int:
    value = payload.get("label", payload.get("is_false"))
    if isinstance(value, bool):
        return int(value)
    if value in {0, 1}:
        return int(value)
    if isinstance(value, str) and value.strip() in {"0", "1"}:
        return int(value.strip())
    raise ValueError(f"record {index} must provide hard binary label/is_false.")


def _first_metadata(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    first = records[0].get("metadata") if records else {}
    if not isinstance(first, Mapping):
        return {}
    return {
        "model": first.get("model"),
        "dataset": first.get("dataset"),
        "layers": first.get("layers"),
        "record_grain": first.get("record_grain"),
        "offline": first.get("offline"),
        "source": first.get("source"),
    }


def _numeric_summary(values: Sequence[float]) -> dict[str, float]:
    finite = [_finite_float(value) for value in values]
    return {
        "min": min(finite),
        "max": max(finite),
        "mean": sum(finite) / len(finite),
    }


def _finite_float(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("numeric feature must be an int or float.")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("numeric feature must be finite.")
    return parsed


def _parse_csv(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    signals = tuple(part.strip() for part in value.split(",") if part.strip())
    if not signals:
        raise ValueError("baseline signal list must not be empty.")
    return signals


def _write_json(path: str | Path, payload: Mapping[str, Any], *, compact: bool) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    indent = None if compact else 2
    output.write_text(json.dumps(to_jsonable(payload), indent=indent, sort_keys=True) + "\n", encoding="utf-8")


def _config_from_args(args: argparse.Namespace) -> PreGenerationTextBaselineConfig:
    return PreGenerationTextBaselineConfig(
        records_path=args.records,
        output_path=args.json,
        artifact_manifest_path=args.artifact_manifest,
        baseline_signals=_parse_csv(args.baseline_signals) or DEFAULT_TEXT_BASELINE_SIGNALS,
        compact_json=bool(args.compact_json),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate text redline baselines for pre-generation records")
    parser.add_argument("--records", required=True, help="pre-generation records JSON/JSONL")
    parser.add_argument("--json", required=True, help="text baseline report output path")
    parser.add_argument("--artifact-manifest", default=None, help="optional artifact-manifest path")
    parser.add_argument("--baseline-signals", default=None, help="comma-separated text baseline signals")
    parser.add_argument("--compact-json", action="store_true")
    run_pre_generation_text_baselines(_config_from_args(parser.parse_args()))


if __name__ == "__main__":
    main()

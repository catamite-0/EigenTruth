"""Export verified correction-buffer rows as SFT or DPO training JSONL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmarks.lib.paths import ensure_repo_root_on_path
from eigentruth.json_utils import strict_json_dumps
from eigentruth.memory import CorrectionBuffer
from eigentruth.registry import build_artifact_manifest

REPO_ROOT = ensure_repo_root_on_path()


def export_correction_training_data(
    *,
    buffer_path: str | Path,
    output_jsonl_path: str | Path,
    format: str = "sft",
    report_json_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    buffer_path = Path(buffer_path)
    output_jsonl_path = Path(output_jsonl_path)
    buffer = CorrectionBuffer.load_jsonl(buffer_path)
    records = buffer.training_records(format=format)
    output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl_path.write_text(
        "".join(strict_json_dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    report = {
        "schema_version": 1,
        "workflow": "correction_training_export",
        "format": str(format).strip().lower(),
        "source_buffer": str(buffer_path),
        "training_jsonl": str(output_jsonl_path),
        "summary": {
            "buffer_record_count": len(buffer.records),
            "exported_record_count": len(records),
            "skipped_record_count": len(buffer.records) - len(records),
        },
    }
    report_path: Path | None = None
    if report_json_path is not None:
        report_path = Path(report_json_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(strict_json_dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if artifact_manifest_path is not None:
        artifacts: dict[str, Path] = {
            "source_buffer": buffer_path,
            "training_jsonl": output_jsonl_path,
        }
        if report_path is not None:
            artifacts["export_report"] = report_path
        manifest = build_artifact_manifest(
            artifacts,
            root=REPO_ROOT,
            metadata={
                "workflow": "correction_training_export",
                "format": report["format"],
                "exported_record_count": len(records),
            },
        )
        manifest_path = Path(artifact_manifest_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(strict_json_dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        report["artifact_manifest"] = str(manifest_path)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export verified correction training rows")
    parser.add_argument("--buffer", required=True, type=Path)
    parser.add_argument("--jsonl", required=True, type=Path)
    parser.add_argument("--format", choices=("sft", "dpo"), default="sft")
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument("--artifact-manifest", type=Path, default=None)
    args = parser.parse_args(argv)
    report = export_correction_training_data(
        buffer_path=args.buffer,
        output_jsonl_path=args.jsonl,
        format=args.format,
        report_json_path=args.report_json,
        artifact_manifest_path=args.artifact_manifest,
    )
    summary = report["summary"]
    print(
        "correction_training_export_ok "
        f"exported={summary['exported_record_count']} "
        f"skipped={summary['skipped_record_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


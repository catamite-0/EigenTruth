"""Verify EigenTruth artifact manifests against local files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eigentruth.registry import load_and_verify_artifact_manifest  # noqa: E402


def verify_manifest_file(
    manifest_path: str | Path,
    *,
    root: str | Path | None = None,
    recursive: bool = False,
) -> dict[str, Any]:
    """Return a JSON-ready verification report for one artifact manifest."""
    return load_and_verify_artifact_manifest(
        manifest_path,
        root=root,
        recursive=recursive,
    ).to_dict()


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run from parsed CLI arguments."""
    report = verify_manifest_file(
        args.manifest,
        root=Path(args.root) if args.root else None,
        recursive=bool(args.recursive),
    )
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Verify EigenTruth artifact manifest fingerprints")
    parser.add_argument("--manifest", required=True, help="artifact-manifest.json to verify")
    parser.add_argument("--root", default=None, help="optional root for relative paths; defaults to manifest parent")
    parser.add_argument("--recursive", action="store_true", help="also verify nested artifact-manifest.json records")
    parser.add_argument("--json", default=None, help="optional path to write the verification report")
    parser.add_argument("--no-fail", action="store_true", help="do not exit non-zero on mismatches")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()

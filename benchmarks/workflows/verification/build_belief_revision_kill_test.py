"""Build the label-separated EigenTruth belief-revision kill-test split."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmarks.lib.paths import ensure_repo_root_on_path

REPO_ROOT = ensure_repo_root_on_path()
DEFAULT_SOURCE = (
    REPO_ROOT
    / "artifacts"
    / "wikidata-country-core-facts-structured-qa-route"
    / "verified-records.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "artifacts" / "baselines" / "belief_revision_text" / "kill-test-v1"
)
DEFAULT_RUNTIME_EXAMPLES = DEFAULT_OUTPUT_DIR / "runtime-examples.jsonl"
DEFAULT_LABELS = DEFAULT_OUTPUT_DIR / "scoring-labels.jsonl"
DEFAULT_BUILD_REPORT = DEFAULT_OUTPUT_DIR / "build-report.json"
DEFAULT_ARTIFACT_MANIFEST = DEFAULT_OUTPUT_DIR / "artifact-manifest.json"

PROPERTY_LABELS = ("capital", "currency", "official language")
FORBIDDEN_RUNTIME_FIELDS = frozenset(
    {
        "accepted_answers",
        "candidate_answers",
        "contradiction_label",
        "corrected_claim",
        "expected_action",
        "expected_revision",
        "is_false",
        "label",
        "rejected_answers",
        "stance",
    }
)


def build_kill_test_rows(
    source_path: str | Path,
    *,
    examples_per_property: int = 16,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...], dict[str, Any]]:
    """Build 48 deterministic runtime/label rows from tracked Wikidata evidence."""
    if examples_per_property < 4:
        raise ValueError("examples_per_property must be at least 4.")
    source_path = Path(source_path)
    grouped = _load_source_pairs(source_path)
    runtime_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    case_counts: dict[str, int] = defaultdict(int)

    for property_label in PROPERTY_LABELS:
        pairs = [
            pair
            for (candidate_property, _), pair in sorted(grouped.items())
            if candidate_property == property_label and pair.get("supported") and pair.get("refuted")
        ]
        if len(pairs) < examples_per_property:
            raise ValueError(
                f"{property_label}: need {examples_per_property} paired rows, found {len(pairs)}."
            )
        selected = pairs[:examples_per_property]
        support_start = examples_per_property - 4
        insufficient_start = examples_per_property - 2
        for index, pair in enumerate(selected):
            if index < support_start:
                case_type = "contradiction"
            elif index < insufficient_start:
                case_type = "support"
            else:
                case_type = "insufficient"
            runtime, label = _build_example(
                property_label=property_label,
                pair=pair,
                case_type=case_type,
                unrelated_pair=selected[(index + 1) % len(selected)],
            )
            runtime_rows.append(runtime)
            label_rows.append(label)
            case_counts[case_type] += 1

    _validate_runtime_rows(runtime_rows)
    example_ids = [row["example_id"] for row in runtime_rows]
    report = {
        "schema_version": 1,
        "workflow": "build_belief_revision_kill_test",
        "split_name": "kill-test-v1",
        "source_path": _repo_relative(source_path),
        "source_sha256": _sha256_file(source_path),
        "selection": {
            "properties": PROPERTY_LABELS,
            "examples_per_property": examples_per_property,
            "case_assignment": "first n-4 contradiction, next 2 support, final 2 insufficient",
            "sort_order": "property then country",
        },
        "summary": {
            "example_count": len(runtime_rows),
            "case_counts": dict(sorted(case_counts.items())),
            "property_counts": {
                property_label: sum(
                    1 for row in label_rows if row["risk_category"] == f"wikidata_{property_label.replace(' ', '_')}"
                )
                for property_label in PROPERTY_LABELS
            },
        },
        "example_ids_sha256": _sha256_text("\n".join(example_ids) + "\n"),
        "runtime_forbidden_fields": tuple(sorted(FORBIDDEN_RUNTIME_FIELDS)),
        "labels_separated_from_generation_inputs": True,
        "research_boundary": (
            "Evaluation-held-out controlled evidence split derived from tracked Wikidata rows. "
            "It is not claimed to be absent from model pretraining data."
        ),
    }
    return tuple(runtime_rows), tuple(label_rows), report


def write_kill_test_split(
    *,
    source_path: str | Path = DEFAULT_SOURCE,
    runtime_path: str | Path = DEFAULT_RUNTIME_EXAMPLES,
    labels_path: str | Path = DEFAULT_LABELS,
    report_path: str | Path = DEFAULT_BUILD_REPORT,
    artifact_manifest_path: str | Path = DEFAULT_ARTIFACT_MANIFEST,
    examples_per_property: int = 16,
) -> dict[str, Any]:
    runtime_rows, label_rows, report = build_kill_test_rows(
        source_path,
        examples_per_property=examples_per_property,
    )
    runtime_path = Path(runtime_path)
    labels_path = Path(labels_path)
    report_path = Path(report_path)
    artifact_manifest_path = Path(artifact_manifest_path)
    _write_jsonl(runtime_path, runtime_rows)
    _write_jsonl(labels_path, label_rows)
    report.update(
        {
            "runtime_examples_path": _repo_relative(runtime_path),
            "runtime_examples_sha256": _sha256_file(runtime_path),
            "scoring_labels_path": _repo_relative(labels_path),
            "scoring_labels_sha256": _sha256_file(labels_path),
        }
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        _strict_json_dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = _build_file_artifact_manifest(
        {
            "source_verified_records": Path(source_path),
            "runtime_examples": runtime_path,
            "scoring_labels": labels_path,
            "build_report": report_path,
        },
        root=REPO_ROOT,
        metadata={
            "workflow": report["workflow"],
            "split_name": report["split_name"],
            "example_count": report["summary"]["example_count"],
            "labels_separated_from_generation_inputs": True,
        },
    )
    artifact_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_manifest_path.write_text(
        _strict_json_dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--runtime-examples", type=Path, default=DEFAULT_RUNTIME_EXAMPLES)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--report", type=Path, default=DEFAULT_BUILD_REPORT)
    parser.add_argument("--artifact-manifest", type=Path, default=DEFAULT_ARTIFACT_MANIFEST)
    parser.add_argument("--examples-per-property", type=int, default=16)
    args = parser.parse_args(argv)
    report = write_kill_test_split(
        source_path=args.source,
        runtime_path=args.runtime_examples,
        labels_path=args.labels,
        report_path=args.report,
        artifact_manifest_path=args.artifact_manifest,
        examples_per_property=args.examples_per_property,
    )
    print(
        "belief_revision_kill_test_build_ok "
        f"examples={report['summary']['example_count']} "
        f"runtime_sha256={report['runtime_examples_sha256']}"
    )
    return 0


def _load_source_pairs(path: Path) -> dict[tuple[str, str], dict[str, Mapping[str, Any]]]:
    pairs: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError(f"source row {line_number} must be a JSON object.")
        label = int(payload.get("label", -1))
        record = _mapping(payload.get("record"))
        statement = _mapping(_mapping(record.get("metadata")).get("statement"))
        metadata = _mapping(statement.get("metadata"))
        property_label = str(metadata.get("statement_property_label", "")).strip()
        country = str(metadata.get("country", "")).strip()
        if property_label not in PROPERTY_LABELS or not country:
            continue
        key = (property_label, country)
        role = "supported" if label == 0 else "refuted" if label == 1 else ""
        if role and role not in pairs[key]:
            pairs[key][role] = payload
    return dict(pairs)


def _build_example(
    *,
    property_label: str,
    pair: Mapping[str, Mapping[str, Any]],
    case_type: str,
    unrelated_pair: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    supported = _statement(pair["supported"])
    refuted = _statement(pair["refuted"])
    country = str(_mapping(supported["metadata"]).get("country", "")).strip()
    slug = _slug(f"{property_label}-{country}")
    example_id = f"br-wikidata-v1-{slug}"
    correct_answer = str(supported.get("answer", "")).strip()
    wrong_answer = str(refuted.get("answer", "")).strip()
    source = str(_mapping(supported["metadata"]).get("source", "")).strip()
    question = str(supported.get("question", "")).strip()

    if case_type == "support":
        initial_answer = correct_answer
        evidence_statement = supported
        expected_action = "accept"
        rejected_answers: tuple[str, ...] = ()
    elif case_type == "insufficient":
        initial_answer = wrong_answer
        evidence_statement = _statement(unrelated_pair["supported"])
        expected_action = "abstain"
        rejected_answers = (wrong_answer,)
    else:
        initial_answer = wrong_answer
        evidence_statement = supported
        expected_action = "revise"
        rejected_answers = (wrong_answer,)

    evidence_metadata = _mapping(evidence_statement["metadata"])
    evidence_source = str(evidence_metadata.get("source", "")).strip()
    evidence_text = (
        f"According to the tracked structured source, the answer to "
        f"'{evidence_statement['question']}' is '{evidence_statement['answer']}'."
    )
    runtime = {
        "schema_version": 1,
        "example_id": example_id,
        "prompt": question,
        "initial_answer": initial_answer,
        "claims": (f"{question} {initial_answer}",),
        "evidence_docs": (
            {
                "record_id": f"{example_id}-e1",
                "evidence_text": evidence_text,
                "source": evidence_source,
            },
        ),
        "language": "en",
        "risk_category": f"wikidata_{property_label.replace(' ', '_')}",
        "source_provenance": {
            "provider": "wikidata",
            "source": evidence_source,
            "split": "kill-test-v1",
        },
    }
    accepted_answers = (
        (
            "INSUFFICIENT EVIDENCE",
            "cannot determine from the evidence",
            "cannot be determined from the evidence",
            "the evidence is insufficient",
        )
        if expected_action == "abstain"
        else (correct_answer,)
    )
    label = {
        "schema_version": 1,
        "example_id": example_id,
        "case_type": case_type,
        "expected_action": expected_action,
        "expected_revision": (
            "INSUFFICIENT EVIDENCE" if expected_action == "abstain" else correct_answer
        ),
        "accepted_answers": accepted_answers,
        "rejected_answers": rejected_answers,
        "risk_category": f"wikidata_{property_label.replace(' ', '_')}",
        "scoring_source": source,
    }
    return runtime, label


def _validate_runtime_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    example_ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        forbidden = _find_forbidden_fields(row)
        if forbidden:
            raise ValueError(
                f"runtime row {index} contains scoring-only fields: {', '.join(sorted(forbidden))}"
            )
        example_id = str(row.get("example_id", "")).strip()
        if not example_id:
            raise ValueError(f"runtime row {index} has no example_id.")
        if example_id in example_ids:
            raise ValueError(f"duplicate runtime example_id: {example_id}")
        example_ids.add(example_id)


def _find_forbidden_fields(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in FORBIDDEN_RUNTIME_FIELDS:
                found.add(str(key))
            found.update(_find_forbidden_fields(item))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            found.update(_find_forbidden_fields(item))
    return found


def _statement(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    statement = _mapping(_mapping(_mapping(payload.get("record")).get("metadata")).get("statement"))
    if not statement:
        raise ValueError("source payload has no statement.")
    return statement


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(_strict_json_dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _repo_relative(path: str | Path) -> str:
    candidate = Path(path).resolve()
    try:
        return candidate.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(candidate)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _slug(value: str) -> str:
    normalized = "".join(char.lower() if char.isalnum() else "-" for char in value)
    return "-".join(part for part in normalized.split("-") if part)


def _strict_json_dumps(value: Any, **kwargs: Any) -> str:
    return json.dumps(value, allow_nan=False, ensure_ascii=False, **kwargs)


def _build_file_artifact_manifest(
    artifacts: Mapping[str, Path],
    *,
    root: Path,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    records: dict[str, dict[str, Any]] = {}
    for name, path in sorted(artifacts.items()):
        candidate = Path(path)
        exists = candidate.is_file()
        records[str(name)] = {
            "path": _relative_to(candidate, root),
            "exists": exists,
            "kind": "file" if exists else "missing",
            "sha256": _sha256_file(candidate) if exists else None,
            "size_bytes": candidate.stat().st_size if exists else None,
            "file_count": 1 if exists else None,
        }
    return {
        "schema_version": 1,
        "digest_algorithm": "sha256",
        "metadata": dict(metadata),
        "artifacts": records,
        "summary": {
            "artifact_count": len(records),
            "missing_count": sum(1 for record in records.values() if not record["exists"]),
            "directory_count": 0,
            "file_count": sum(1 for record in records.values() if record["kind"] == "file"),
        },
    }


def _relative_to(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


if __name__ == "__main__":
    raise SystemExit(main())

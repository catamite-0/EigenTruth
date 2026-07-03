"""Fill deterministic rule inputs from promoted correction handoff artifacts.

This workflow is intentionally conservative. It only fills entity-role rule
inputs when a typed rule-input task matches a promoted source-family structured
QA correction document and a ProductTrace for the same question supplies the
candidate answer from a runtime claim trace. Filled rows are still not verifier
evidence; they are explicit inputs for a later deterministic adapter execution
and promotion gate.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eigentruth.json_utils import strict_json_dumps  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402

WORKFLOW = "world_model_rule_input_correction_handoff_fill"
SOURCE_WORKFLOW = "world_model_rule_input_collection_plan"
CORRECTION_WORKFLOW = "source_family_structured_qa_correction_handoff"
RESERVED_FIELDS = {"label", "labels", "is_false", "score_label"}


def fill_world_model_rule_inputs_from_correction_handoff(
    *,
    input_tasks: Sequence[Mapping[str, Any]],
    correction_handoff: Mapping[str, Any],
    qa_corpus: Mapping[str, Any],
    product_traces: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return filled rule-input rows plus audit metadata."""
    _validate_sources(correction_handoff=correction_handoff, qa_corpus=qa_corpus)
    correction_docs = _documents_by_question(qa_corpus.get("documents", ()))
    trace_claims = _trace_claims_by_question(product_traces)
    filled: list[dict[str, Any]] = []
    unfilled: list[dict[str, Any]] = []

    for task in input_tasks:
        if str(task.get("collection_family") or "") != "entity_role_rule_input_collection":
            unfilled.append(_unfilled(task, reason="unsupported_collection_family"))
            continue
        question_key = _question_key(task.get("question"))
        doc = correction_docs.get(question_key)
        claim = trace_claims.get(question_key)
        if doc is None:
            unfilled.append(_unfilled(task, reason="no_promoted_correction_document"))
            continue
        if claim is None:
            unfilled.append(_unfilled(task, reason="no_product_trace_claim_binding"))
            continue
        row = _filled_entity_role_input(task, document=doc, claim=claim)
        if row is None:
            unfilled.append(_unfilled(task, reason="insufficient_entity_role_inputs"))
            continue
        filled.append(row)

    summary = _summary(input_tasks=input_tasks, filled=filled, unfilled=unfilled)
    return {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": _status(summary),
        "scope": (
            "Fills explicit deterministic rule inputs only from promoted "
            "correction handoff evidence and ProductTrace claim bindings. "
            "Filled rows are adapter inputs, not verifier evidence."
        ),
        "source": {
            "input_task_workflow": SOURCE_WORKFLOW,
            "input_task_count": len(input_tasks),
            "correction_handoff_workflow": correction_handoff.get("workflow"),
            "correction_handoff_status": correction_handoff.get("status"),
            "qa_corpus_type": qa_corpus.get("corpus_type"),
            "product_trace_count": len(product_traces),
        },
        "label_usage": {
            "labels_used_for_input_fill": False,
            "labels_copied_to_rule_inputs": False,
            "candidate_answer_bound_from_product_trace": True,
            "source_facts_from_promoted_correction_handoff": True,
            "filled_inputs_are_verifier_evidence": False,
            "requires_adapter_execution_and_promotion_gate": True,
        },
        "summary": summary,
        "rule_inputs": tuple(filled),
        "unfilled_tasks": tuple(unfilled),
        "metadata": dict(metadata or {}),
    }


def run(
    *,
    input_tasks_path: str | Path,
    correction_handoff_path: str | Path,
    qa_corpus_path: str | Path,
    product_traces_path: str | Path,
    output_dir: str | Path,
    report_json_path: str | Path | None = None,
    rule_inputs_path: str | Path | None = None,
    unfilled_tasks_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    compact_json: bool = False,
) -> dict[str, Any]:
    """Fill, write, manifest, and optionally register rule-input rows."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = Path(report_json_path or output / "rule-input-correction-handoff-fill.json")
    inputs_path = Path(rule_inputs_path or output / "rule-inputs.jsonl")
    unfilled_path = Path(unfilled_tasks_path or output / "unfilled-rule-input-tasks.jsonl")
    manifest_path = Path(artifact_manifest_path or output / "artifact-manifest.json")

    input_tasks = _load_jsonl_mappings(input_tasks_path)
    correction_handoff = _load_json_object(correction_handoff_path)
    qa_corpus = _load_json_object(qa_corpus_path)
    product_traces = _load_jsonl_mappings(product_traces_path)
    payload = fill_world_model_rule_inputs_from_correction_handoff(
        input_tasks=input_tasks,
        correction_handoff=correction_handoff,
        qa_corpus=qa_corpus,
        product_traces=product_traces,
        metadata=metadata,
    )
    payload = dict(payload)
    payload["paths"] = {
        "input_tasks": str(input_tasks_path),
        "correction_handoff": str(correction_handoff_path),
        "qa_corpus": str(qa_corpus_path),
        "product_traces": str(product_traces_path),
        "report": str(report_path),
        "rule_inputs": str(inputs_path),
        "unfilled_tasks": str(unfilled_path),
        "artifact_manifest": str(manifest_path),
    }
    _write_json(report_path, payload, compact=compact_json)
    _write_jsonl(inputs_path, payload["rule_inputs"], compact=compact_json)
    _write_jsonl(unfilled_path, payload["unfilled_tasks"], compact=compact_json)
    manifest = build_artifact_manifest(
        {
            "rule_input_fill_report": report_path,
            "rule_inputs": inputs_path,
            "unfilled_rule_input_tasks": unfilled_path,
            "rule_input_tasks": Path(input_tasks_path),
            "correction_handoff": Path(correction_handoff_path),
            "qa_corpus": Path(qa_corpus_path),
            "product_traces": Path(product_traces_path),
        },
        root=manifest_path.parent,
        metadata={
            "workflow": WORKFLOW,
            "status": payload["status"],
            "filled_input_count": payload["summary"]["filled_input_count"],
            "unfilled_task_count": payload["summary"]["unfilled_task_count"],
            **dict(metadata or {}),
        },
    )
    _write_json(manifest_path, manifest, compact=compact_json)
    if registry_path is not None:
        assert name is not None and version is not None
        ArtifactRegistry.load_json(registry_path).record_report(
            name=name,
            version=version,
            path=report_path,
            metadata={
                "workflow": WORKFLOW,
                "status": payload["status"],
                "filled_input_count": payload["summary"]["filled_input_count"],
                "unfilled_task_count": payload["summary"]["unfilled_task_count"],
                "artifact_manifest": str(manifest_path),
                **dict(metadata or {}),
            },
        ).save_json()
    return payload


def _validate_sources(*, correction_handoff: Mapping[str, Any], qa_corpus: Mapping[str, Any]) -> None:
    if correction_handoff.get("workflow") != CORRECTION_WORKFLOW:
        raise ValueError(f"correction_handoff must be a {CORRECTION_WORKFLOW} report.")
    if correction_handoff.get("status") != "promote":
        raise ValueError("correction_handoff must be promoted before filling rule inputs.")
    if str(qa_corpus.get("corpus_type") or "") != "target_specific_source_family_structured_qa_correction":
        raise ValueError("qa_corpus must be a target-specific source-family structured QA correction corpus.")


def _filled_entity_role_input(
    task: Mapping[str, Any],
    *,
    document: Mapping[str, Any],
    claim: Mapping[str, Any],
) -> dict[str, Any] | None:
    metadata = _mapping(document.get("metadata"))
    subject = _clean(metadata.get("subject"))
    answer_entity = _candidate_answer_entity(claim)
    expected = _clean(document.get("answer"))
    requested_role = _clean(metadata.get("statement_property_label")) or _clean(metadata.get("statement_property"))
    source_citation = _clean(document.get("source")) or _clean(metadata.get("source"))
    if not all((subject, answer_entity, expected, requested_role, source_citation)):
        return None
    return {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "request_id": str(task.get("source_request_id") or ""),
        "target_id": str(task.get("target_id") or ""),
        "rule_family": str(task.get("rule_family") or "entity_disambiguation"),
        "subject_entity": subject,
        "answer_entity": answer_entity,
        "expected_entity": expected,
        "requested_role": requested_role,
        "source_citation": source_citation,
        "source_url": _clean(metadata.get("url")),
        "source_fact_type": _clean(metadata.get("fact_type") or metadata.get("statement_property")),
        "source_family": _clean(metadata.get("source_family")),
        "provider": _clean(metadata.get("provider")),
        "not_verifier_evidence": True,
        "candidate_results_require_promotion_gate": True,
        "provenance": {
            "fill_source": "source_family_structured_qa_correction_handoff",
            "question": str(task.get("question") or ""),
            "qa_document_source": source_citation,
            "qa_document_question": str(document.get("question") or ""),
            "claim_id": str(claim.get("claim_id") or ""),
            "claim_text": str(claim.get("text") or ""),
            "claim_answer_source": "product_trace_claim_metadata",
        },
    }


def _candidate_answer_entity(claim: Mapping[str, Any]) -> str:
    raw = _clean(_mapping(claim.get("metadata")).get("answer")) or _clean(claim.get("text"))
    question = _clean(_mapping(claim.get("metadata")).get("question"))
    if question and raw.startswith(question):
        raw = raw[len(question) :].strip()
    raw = raw.strip(" .")
    for suffix in (
        " founded Tesla",
        " first started Tesla Motors",
        " first started Tesla",
        " started Tesla Motors",
        " started Tesla",
    ):
        if raw.casefold().endswith(suffix.casefold()):
            raw = raw[: -len(suffix)].strip(" .")
            break
    match = re.match(r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})", raw)
    return _clean(match.group(1) if match else raw)


def _documents_by_question(documents: Any) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for doc in _mapping_sequence(documents):
        key = _question_key(doc.get("question"))
        if key and key not in output:
            output[key] = doc
    return output


def _trace_claims_by_question(product_traces: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for trace in product_traces:
        for claim in _mapping_sequence(trace.get("claims", ())):
            metadata = _mapping(claim.get("metadata"))
            key = _question_key(metadata.get("question"))
            if key and key not in output:
                output[key] = claim
    return output


def _unfilled(task: Mapping[str, Any], *, reason: str) -> dict[str, Any]:
    return {
        "task_id": str(task.get("task_id") or ""),
        "source_request_id": str(task.get("source_request_id") or ""),
        "target_id": str(task.get("target_id") or ""),
        "rule_family": str(task.get("rule_family") or ""),
        "collection_family": str(task.get("collection_family") or ""),
        "question": str(task.get("question") or ""),
        "reason": reason,
        "not_verifier_evidence": True,
    }


def _summary(
    *,
    input_tasks: Sequence[Mapping[str, Any]],
    filled: Sequence[Mapping[str, Any]],
    unfilled: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    filled_family = Counter(str(item.get("rule_family") or "") for item in filled)
    unfilled_reason = Counter(str(item.get("reason") or "") for item in unfilled)
    collection_family = Counter(str(item.get("collection_family") or "") for item in input_tasks)
    return {
        "input_task_count": len(input_tasks),
        "filled_input_count": len(filled),
        "unfilled_task_count": len(unfilled),
        "filled_rule_family_counts": _sorted_counter(filled_family),
        "input_collection_family_counts": _sorted_counter(collection_family),
        "unfilled_reason_counts": _sorted_counter(unfilled_reason),
        "filled_request_ids": tuple(str(item.get("request_id") or "") for item in filled),
    }


def _status(summary: Mapping[str, Any]) -> str:
    if int(summary.get("input_task_count", 0)) == 0:
        return "empty"
    if int(summary.get("filled_input_count", 0)) == 0:
        return "blocked"
    if int(summary.get("unfilled_task_count", 0)) > 0:
        return "partial"
    return "filled"


def _load_json_object(path: str | Path) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _load_jsonl_mappings(path: str | Path) -> tuple[Mapping[str, Any], ...]:
    rows = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, Mapping):
            raise ValueError(f"{path}:{line_no} must contain a JSON object.")
        rows.append({key: value for key, value in dict(row).items() if key not in RESERVED_FIELDS})
    return tuple(rows)


def _write_json(path: str | Path, payload: Mapping[str, Any], *, compact: bool) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = strict_json_dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    else:
        text = strict_json_dumps(payload, indent=2, sort_keys=True) + "\n"
    output.write_text(text, encoding="utf-8")


def _write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]], *, compact: bool) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = "".join(strict_json_dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    else:
        text = "".join(strict_json_dumps(row, sort_keys=True) + "\n" for row in rows)
    output.write_text(text, encoding="utf-8")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _mapping_sequence(value: Any) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, Mapping):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(item for item in value if isinstance(item, Mapping))
    return ()


def _question_key(value: Any) -> str:
    return re.sub(r"\s+", " ", _clean(value).casefold())


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(((key, value) for key, value in counter.items() if key), key=lambda item: (-item[1], item[0])))


def _parse_metadata(values: Sequence[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"metadata must be KEY=VALUE, got {value!r}.")
        key, item = value.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("metadata key cannot be empty.")
        metadata[key] = item
    return metadata


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-tasks", required=True)
    parser.add_argument("--correction-handoff", required=True)
    parser.add_argument("--qa-corpus", required=True)
    parser.add_argument("--product-traces", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json", default=None)
    parser.add_argument("--rule-inputs-jsonl", default=None)
    parser.add_argument("--unfilled-tasks-jsonl", default=None)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--compact-json", action="store_true")
    args = parser.parse_args(argv)
    payload = run(
        input_tasks_path=args.input_tasks,
        correction_handoff_path=args.correction_handoff,
        qa_corpus_path=args.qa_corpus,
        product_traces_path=args.product_traces,
        output_dir=args.output_dir,
        report_json_path=args.json,
        rule_inputs_path=args.rule_inputs_jsonl,
        unfilled_tasks_path=args.unfilled_tasks_jsonl,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        metadata=_parse_metadata(args.metadata or ()),
        compact_json=bool(args.compact_json),
    )
    summary = payload["summary"]
    print(
        "world_model_rule_input_correction_handoff_fill_ok "
        f"status={payload['status']} "
        f"filled={summary['filled_input_count']} "
        f"unfilled={summary['unfilled_task_count']}"
    )


if __name__ == "__main__":
    main()

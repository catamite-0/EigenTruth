"""Compile counterfactual probe requests into audit-ready handoff files.

This workflow bridges blind-spot ``counterfactual_probe`` requests into the
existing ``eval_counterfactual_verification.py`` record format. It does not run
a verifier and it does not promote evidence: generated probes remain
non-evidence fixtures until a verifier audit, manifest, and release gate pass.
"""

from __future__ import annotations

import argparse
import json
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
from eigentruth.verify import Claim, CounterfactualProbe, generate_counterfactual_probes  # noqa: E402

WORKFLOW = "counterfactual_probe_handoff"
REQUEST_TYPE = "counterfactual_probe"
RESERVED_REQUEST_FIELDS = {
    "is_false",
    "label",
    "labels",
    "score_label",
    "truth_label",
}
DEFAULT_PROBE_TYPES = ("metadata", "entity_swap", "quantity", "year", "negation")
PROBE_TYPE_MAP: Mapping[str, tuple[str, ...]] = {
    "entity_swap": ("metadata", "entity_swap"),
    "quantity": ("metadata", "quantity", "year"),
    "numeric": ("metadata", "quantity", "year"),
    "year": ("metadata", "year"),
    "temporal": ("metadata", "year"),
    "negation": ("metadata", "negation"),
}


def build_counterfactual_probe_handoff(
    *,
    requests: Sequence[Mapping[str, Any]],
    source_input: str | Path | None = None,
    source_input_kind: str | None = None,
    max_probes_per_request: int = 1,
    generated_probe_types: Sequence[str] = DEFAULT_PROBE_TYPES,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a JSON-ready counterfactual probe handoff payload."""
    if int(max_probes_per_request) <= 0:
        raise ValueError("max_probes_per_request must be positive.")
    default_probe_types = tuple(
        str(item).strip().casefold().replace("-", "_")
        for item in generated_probe_types
        if str(item).strip()
    )
    if not default_probe_types:
        raise ValueError("generated_probe_types must contain at least one value.")

    counterfactual_requests = tuple(
        _normalize_request(row, index=index)
        for index, row in enumerate(requests, start=1)
        if _is_counterfactual_request(row)
    )
    claims: list[dict[str, Any]] = []
    probe_records: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    explicit_probe_count = 0
    generated_probe_count = 0

    for ordinal, request in enumerate(counterfactual_requests, start=1):
        explicit = _explicit_probe_from_request(request, ordinal=ordinal)
        if explicit is not None:
            probe_records.append(explicit.to_dict())
            explicit_probe_count += 1
            continue
        claim = _claim_from_request(request, ordinal=ordinal)
        if claim is None:
            pending.append(_pending_request(request, reason="missing_claim_text"))
            continue
        claim_payload = _claim_to_record(claim)
        claims.append(claim_payload)
        probe_types = _probe_types_for_request(request, default_probe_types=default_probe_types)
        generated = generate_counterfactual_probes(
            (claim,),
            max_probes_per_claim=int(max_probes_per_request),
            probe_types=probe_types,
        )
        if not generated:
            pending.append(_pending_request(request, reason="no_auto_probe_generated"))
            continue
        for probe in generated:
            probe_records.append(_probe_with_request_metadata(probe, request).to_dict())
            generated_probe_count += 1

    summary = _summary(
        requests=counterfactual_requests,
        claims=claims,
        probe_records=probe_records,
        pending=pending,
        explicit_probe_count=explicit_probe_count,
        generated_probe_count=generated_probe_count,
    )
    status = _status(summary)
    return {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": status,
        "scope": (
            "Counterfactual probe-generation handoff for unresolved blind spots. "
            "Claims and probe records are audit fixtures, not verifier evidence."
        ),
        "source": {
            "input": None if source_input is None else str(source_input),
            "input_kind": source_input_kind,
        },
        "label_usage": {
            "labels_used_for_probe_generation": False,
            "labels_copied_to_handoff": False,
            "probe_records_are_verifier_evidence": False,
        },
        "config": {
            "max_probes_per_request": int(max_probes_per_request),
            "generated_probe_types": default_probe_types,
        },
        "summary": summary,
        "claims": tuple(claims),
        "probe_records": tuple(probe_records),
        "pending_generation_requests": tuple(pending),
        "metadata": dict(metadata or {}),
    }


def run(
    *,
    output_dir: str | Path,
    queue_report_path: str | Path | None = None,
    adapter_requests_path: str | Path | None = None,
    collection_corpus_path: str | Path | None = None,
    report_json_path: str | Path | None = None,
    claims_jsonl_path: str | Path | None = None,
    probe_records_jsonl_path: str | Path | None = None,
    pending_jsonl_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    max_probes_per_request: int = 1,
    generated_probe_types: Sequence[str] = DEFAULT_PROBE_TYPES,
    metadata: Mapping[str, Any] | None = None,
    compact_json: bool = False,
) -> dict[str, Any]:
    """Build, write, manifest, and optionally register a probe handoff."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    source_path, source_kind, requests = _load_source_requests(
        queue_report_path=queue_report_path,
        adapter_requests_path=adapter_requests_path,
        collection_corpus_path=collection_corpus_path,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = Path(report_json_path or output / "counterfactual-probe-handoff.json")
    claims_path = Path(claims_jsonl_path or output / "counterfactual-claims.jsonl")
    probes_path = Path(probe_records_jsonl_path or output / "counterfactual-probe-records.jsonl")
    pending_path = Path(pending_jsonl_path or output / "pending-counterfactual-probe-requests.jsonl")
    manifest_path = Path(artifact_manifest_path or output / "artifact-manifest.json")

    payload = build_counterfactual_probe_handoff(
        requests=requests,
        source_input=source_path,
        source_input_kind=source_kind,
        max_probes_per_request=max_probes_per_request,
        generated_probe_types=generated_probe_types,
        metadata=metadata,
    )
    payload = dict(payload)
    payload["paths"] = {
        "source_input": str(source_path),
        "claims_jsonl": str(claims_path),
        "probe_records_jsonl": str(probes_path),
        "pending_generation_jsonl": str(pending_path),
        "artifact_manifest": str(manifest_path),
    }
    _write_json(report_path, payload, compact=compact_json)
    _write_jsonl(claims_path, payload["claims"], compact=compact_json)
    _write_jsonl(probes_path, payload["probe_records"], compact=compact_json)
    _write_jsonl(pending_path, payload["pending_generation_requests"], compact=compact_json)

    manifest = build_artifact_manifest(
        {
            "counterfactual_probe_handoff_report": report_path,
            "counterfactual_claims": claims_path,
            "counterfactual_probe_records": probes_path,
            "pending_counterfactual_probe_requests": pending_path,
            "source_input": source_path,
        },
        root=manifest_path.parent,
        metadata={
            "workflow": WORKFLOW,
            "status": payload["status"],
            "request_count": payload["summary"]["request_count"],
            "claim_count": payload["summary"]["claim_count"],
            "probe_record_count": payload["summary"]["probe_record_count"],
            "pending_generation_count": payload["summary"]["pending_generation_count"],
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
                "request_count": payload["summary"]["request_count"],
                "claim_count": payload["summary"]["claim_count"],
                "probe_record_count": payload["summary"]["probe_record_count"],
                "pending_generation_count": payload["summary"]["pending_generation_count"],
                "artifact_manifest": str(manifest_path),
                **dict(metadata or {}),
            },
        ).save_json()
        payload["registry_record"] = f"report:{name}:{version}"
    return payload


def _load_source_requests(
    *,
    queue_report_path: str | Path | None,
    adapter_requests_path: str | Path | None,
    collection_corpus_path: str | Path | None,
) -> tuple[Path, str, tuple[Mapping[str, Any], ...]]:
    provided = tuple(
        (kind, Path(path))
        for kind, path in (
            ("queue_report", queue_report_path),
            ("adapter_requests", adapter_requests_path),
            ("collection_corpus", collection_corpus_path),
        )
        if path is not None
    )
    if len(provided) != 1:
        raise ValueError("provide exactly one of queue_report_path, adapter_requests_path, or collection_corpus_path.")
    source_kind, source_path = provided[0]
    if source_kind == "adapter_requests":
        return source_path, source_kind, _load_jsonl_mappings(source_path)
    payload = _load_json_object(source_path)
    if source_kind == "queue_report":
        requests = tuple(_mapping_sequence(payload.get("adapter_requests")))
    else:
        requests_payload = payload.get("requests")
        if not isinstance(requests_payload, Mapping):
            requests = ()
        else:
            requests = tuple(_mapping_sequence(requests_payload.get(REQUEST_TYPE)))
    return source_path, source_kind, requests


def _normalize_request(row: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    reserved = sorted(key for key in RESERVED_REQUEST_FIELDS if key in row)
    if reserved:
        raise ValueError(f"counterfactual request {index} contains reserved label fields: {', '.join(reserved)}")
    request = dict(row)
    request.setdefault("request_type", REQUEST_TYPE)
    request.setdefault("source_request_id", request.get("request_id", request.get("queue_id", f"request-{index}")))
    return request


def _is_counterfactual_request(row: Mapping[str, Any]) -> bool:
    request_type = str(row.get("request_type", "")).strip()
    adapter_family = str(row.get("adapter_family", "")).strip()
    return request_type == REQUEST_TYPE or adapter_family == "counterfactual_probe_generator"


def _explicit_probe_from_request(request: Mapping[str, Any], *, ordinal: int) -> CounterfactualProbe | None:
    counterfactual_text = _optional_text(
        request.get("counterfactual_text")
        or request.get("variant_text")
        or request.get("perturbed_text")
    )
    original_text = _original_claim_text(request)
    if original_text is None or counterfactual_text is None:
        return None
    source_request_id = _source_request_id(request, ordinal=ordinal)
    return CounterfactualProbe(
        probe_id=f"{source_request_id}:explicit",
        probe_type=_clean_probe_type(request.get("probe_type")) or "metadata",
        original={
            "claim_id": source_request_id,
            "text": original_text,
            "metadata": _claim_metadata(request),
        },
        counterfactual={
            "claim_id": f"{source_request_id}:counterfactual",
            "text": counterfactual_text,
            "metadata": {
                "source_claim_id": source_request_id,
                "generated_by": WORKFLOW,
                "source_request_id": source_request_id,
            },
        },
        metadata=_probe_metadata(request, generated_by="explicit_request"),
    )


def _claim_from_request(request: Mapping[str, Any], *, ordinal: int) -> Claim | None:
    text = _original_claim_text(request)
    if text is None:
        return None
    return Claim(
        text=text,
        claim_id=_source_request_id(request, ordinal=ordinal),
        metadata=_claim_metadata(request),
    )


def _original_claim_text(request: Mapping[str, Any]) -> str | None:
    text = _optional_text(request.get("text") or request.get("claim") or request.get("statement"))
    if text is not None:
        return text
    question = _optional_text(request.get("question"))
    answer = _optional_text(request.get("model_answer") or request.get("answer"))
    if question is None and answer is None:
        return None
    if question is None:
        return answer
    if answer is None:
        return question
    if answer in question:
        return question
    return f"{question} {answer}"


def _claim_metadata(request: Mapping[str, Any]) -> dict[str, Any]:
    metadata = {
        "source": WORKFLOW,
        "not_verifier_evidence": True,
        "request_type": REQUEST_TYPE,
        "source_request_id": request.get("source_request_id") or request.get("request_id"),
        "queue_id": request.get("queue_id"),
        "target_id": request.get("target_id"),
        "record_index": request.get("record_index"),
        "question": request.get("question"),
        "model_answer": request.get("model_answer"),
        "question_type": request.get("question_type"),
        "probe_type": request.get("probe_type"),
        "probe_instruction": request.get("probe_instruction") or request.get("query"),
        "usage": request.get("usage", "probe_generation_only"),
        "priority": request.get("priority"),
        "evidence_status": request.get("evidence_status"),
        "mapping_decision": request.get("mapping_decision"),
        "entity_candidates": tuple(str(item) for item in _sequence(request.get("entity_candidates"))),
    }
    replacements = _counterfactual_replacements(request)
    if replacements:
        metadata["counterfactual_replacements"] = replacements
    return {key: value for key, value in metadata.items() if value not in (None, "", ())}


def _counterfactual_replacements(request: Mapping[str, Any]) -> dict[str, str]:
    explicit = request.get("counterfactual_replacements")
    if isinstance(explicit, Mapping):
        return {
            str(source): str(target)
            for source, target in explicit.items()
            if str(source).strip() and str(target).strip() and str(source) != str(target)
        }
    probe_type = _clean_probe_type(request.get("probe_type"))
    if probe_type != "entity_swap":
        return {}
    text = _original_claim_text(request) or ""
    answer = _strip_terminal_punctuation(_optional_text(request.get("model_answer") or request.get("answer")) or "")
    candidates = tuple(
        _strip_terminal_punctuation(str(item).strip())
        for item in _sequence(request.get("entity_candidates"))
        if str(item).strip()
    )
    source = answer if answer and answer in text else ""
    if not source:
        for candidate in candidates:
            if candidate and candidate in text:
                source = candidate
                break
    if not source:
        return {}
    for candidate in candidates:
        if candidate and candidate != source:
            return {source: candidate}
    return {}


def _probe_types_for_request(
    request: Mapping[str, Any],
    *,
    default_probe_types: Sequence[str],
) -> tuple[str, ...]:
    probe_type = _clean_probe_type(request.get("probe_type"))
    if probe_type in PROBE_TYPE_MAP:
        return PROBE_TYPE_MAP[probe_type]
    return tuple(default_probe_types)


def _probe_with_request_metadata(
    probe: CounterfactualProbe,
    request: Mapping[str, Any],
) -> CounterfactualProbe:
    return CounterfactualProbe(
        original=probe.original,
        counterfactual=probe.counterfactual,
        probe_id=probe.probe_id,
        probe_type=probe.probe_type,
        expected_original_status=probe.expected_original_status,
        expected_counterfactual_status=probe.expected_counterfactual_status,
        expected_flip=probe.expected_flip,
        metadata={**dict(probe.metadata), **_probe_metadata(request, generated_by="CounterfactualProbeGenerator")},
    )


def _probe_metadata(request: Mapping[str, Any], *, generated_by: str) -> dict[str, Any]:
    return {
        "source": WORKFLOW,
        "generated_by": generated_by,
        "not_verifier_evidence": True,
        "source_request_id": request.get("source_request_id") or request.get("request_id"),
        "queue_id": request.get("queue_id"),
        "target_id": request.get("target_id"),
        "record_index": request.get("record_index"),
        "probe_instruction": request.get("probe_instruction") or request.get("query"),
    }


def _pending_request(request: Mapping[str, Any], *, reason: str) -> dict[str, Any]:
    return {
        "source_request_id": request.get("source_request_id") or request.get("request_id"),
        "queue_id": request.get("queue_id"),
        "target_id": request.get("target_id"),
        "record_index": request.get("record_index"),
        "question": request.get("question"),
        "model_answer": request.get("model_answer"),
        "probe_type": request.get("probe_type"),
        "probe_instruction": request.get("probe_instruction") or request.get("query"),
        "reason": reason,
        "not_verifier_evidence": True,
    }


def _claim_to_record(claim: Claim) -> dict[str, Any]:
    return {
        "claim_id": claim.claim_id,
        "text": claim.text,
        "metadata": dict(claim.metadata),
    }


def _summary(
    *,
    requests: Sequence[Mapping[str, Any]],
    claims: Sequence[Mapping[str, Any]],
    probe_records: Sequence[Mapping[str, Any]],
    pending: Sequence[Mapping[str, Any]],
    explicit_probe_count: int,
    generated_probe_count: int,
) -> dict[str, Any]:
    probe_type_counts = Counter(_clean_probe_type(item.get("probe_type")) or "unknown" for item in requests)
    generated_type_counts = Counter(str(item.get("probe_type") or "unknown") for item in probe_records)
    pending_reason_counts = Counter(str(item.get("reason") or "unknown") for item in pending)
    target_ids = {
        str(item.get("target_id"))
        for item in requests
        if item.get("target_id") not in (None, "")
    }
    return {
        "request_count": len(requests),
        "target_count": len(target_ids),
        "claim_count": len(claims),
        "probe_record_count": len(probe_records),
        "explicit_probe_count": int(explicit_probe_count),
        "generated_probe_count": int(generated_probe_count),
        "pending_generation_count": len(pending),
        "probe_type_counts": _sorted_counter(probe_type_counts),
        "generated_probe_type_counts": _sorted_counter(generated_type_counts),
        "pending_reason_counts": _sorted_counter(pending_reason_counts),
    }


def _status(summary: Mapping[str, Any]) -> str:
    if summary.get("request_count", 0) == 0:
        return "empty"
    if summary.get("probe_record_count", 0) == 0:
        return "needs_external_generation"
    if summary.get("pending_generation_count", 0):
        return "partial"
    return "ready_for_counterfactual_eval"


def _source_request_id(request: Mapping[str, Any], *, ordinal: int) -> str:
    value = request.get("source_request_id") or request.get("request_id") or request.get("queue_id")
    text = str(value).strip() if value is not None else ""
    return text or f"counterfactual-request-{ordinal}"


def _load_json_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object.")
    return dict(payload)


def _load_jsonl_mappings(path: str | Path) -> tuple[Mapping[str, Any], ...]:
    rows = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError(f"{path}:{line_no} must contain a JSON object.")
        rows.append(payload)
    return tuple(rows)


def _write_json(path: str | Path, payload: Mapping[str, Any], *, compact: bool = False) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"sort_keys": True, "separators": (",", ":")} if compact else {"indent": 2, "sort_keys": True}
    output.write_text(strict_json_dumps(payload, **kwargs) + "\n", encoding="utf-8")


def _write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]], *, compact: bool = False) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"sort_keys": True, "separators": (",", ":")} if compact else {"sort_keys": True}
    output.write_text(
        "".join(strict_json_dumps(row, **kwargs) + "\n" for row in rows),
        encoding="utf-8",
    )


def _mapping_sequence(value: Any) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(item for item in value if isinstance(item, Mapping))
    return ()


def _sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _clean_probe_type(value: Any) -> str:
    return str(value or "").strip().casefold().replace("-", "_")


def _strip_terminal_punctuation(value: str) -> str:
    return value.strip().strip(" \t\r\n.?!,;:")


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build counterfactual probe handoff files")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--queue-report", default=None, help="unresolved queue JSON report")
    source.add_argument("--adapter-requests", default=None, help="adapter request JSONL")
    source.add_argument("--collection-corpus", default=None, help="blind-spot evidence collection corpus JSON")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-json", default=None)
    parser.add_argument("--claims-jsonl", default=None)
    parser.add_argument("--probe-records-jsonl", default=None)
    parser.add_argument("--pending-jsonl", default=None)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--max-probes-per-request", type=int, default=1)
    parser.add_argument("--generated-probe-types", default="metadata,entity_swap,quantity,year,negation")
    parser.add_argument("--compact-json", action="store_true")
    args = parser.parse_args(argv)

    payload = run(
        output_dir=args.output_dir,
        queue_report_path=args.queue_report,
        adapter_requests_path=args.adapter_requests,
        collection_corpus_path=args.collection_corpus,
        report_json_path=args.report_json,
        claims_jsonl_path=args.claims_jsonl,
        probe_records_jsonl_path=args.probe_records_jsonl,
        pending_jsonl_path=args.pending_jsonl,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        max_probes_per_request=args.max_probes_per_request,
        generated_probe_types=_parse_csv(args.generated_probe_types),
        compact_json=bool(args.compact_json),
    )
    summary = payload["summary"]
    print(
        "counterfactual_probe_handoff_ok "
        f"status={payload['status']} "
        f"requests={summary['request_count']} "
        f"probes={summary['probe_record_count']} "
        f"pending={summary['pending_generation_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Execute or materialize world-model/calculator rule-authoring stubs.

Rule stubs are queued work items, not verifier evidence. This workflow turns
them into auditable rule input requests and, when explicit deterministic inputs
are supplied, candidate rule results. Candidate results still require a later
promotion gate before any product correction handoff can consume them.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eigentruth.adapters.calculator import CalculatorVerifier  # noqa: E402
from eigentruth.json_utils import strict_json_dumps  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402
from eigentruth.verify import Claim, VerificationStatus  # noqa: E402

WORKFLOW = "world_model_rule_authoring_adapter"
RULE_REQUEST_TYPE = "world_model_or_calculator_rule"
INPUT_KEYS_BY_FAMILY = {
    "quantity_or_arithmetic": ("numeric_value", "unit", "reference_time"),
    "entity_disambiguation": ("subject_entity", "answer_entity", "requested_role"),
    "causal_or_procedural": ("mechanism", "precondition", "source_citation"),
    "temporal_consistency": ("claim_time", "source_time", "retrieved_at", "source_citation"),
}
EXECUTED_STATUSES = {
    VerificationStatus.SUPPORTED.value,
    VerificationStatus.REFUTED.value,
    VerificationStatus.INSUFFICIENT_EVIDENCE.value,
    VerificationStatus.ERROR.value,
}


def run_world_model_rule_authoring_adapter(
    *,
    rule_stubs_path: str | Path,
    output_dir: str | Path,
    rule_inputs_path: str | Path | None = None,
    report_json_path: str | Path | None = None,
    rule_results_path: str | Path | None = None,
    input_requests_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    compact_json: bool = False,
) -> dict[str, Any]:
    """Run deterministic rule adapter logic over rule-authoring stubs."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = Path(report_json_path or output / "world-model-rule-authoring-adapter.json")
    results_path = Path(rule_results_path or output / "world-model-rule-results.jsonl")
    requests_path = Path(input_requests_path or output / "world-model-rule-input-requests.jsonl")
    manifest_path = Path(artifact_manifest_path or output / "artifact-manifest.json")

    stubs = _load_jsonl_mappings(rule_stubs_path)
    rule_inputs = _load_rule_inputs(rule_inputs_path)
    results = tuple(_evaluate_stub(stub, rule_inputs.get(str(stub.get("request_id") or ""))) for stub in stubs)
    input_requests = tuple(_input_request(result) for result in results if result["status"] == "needs_inputs")
    _write_jsonl(results_path, results, compact=compact_json)
    _write_jsonl(requests_path, input_requests, compact=compact_json)

    summary = _summary(results=results, input_requests=input_requests)
    payload = {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": _status(summary),
        "scope": (
            "Executes deterministic calculator/entity-role/temporal checks "
            "only when explicit rule inputs are provided. Missing-input rows "
            "are rule-authoring work items, not verifier evidence."
        ),
        "source": {
            "rule_stubs": str(rule_stubs_path),
            "rule_inputs": None if rule_inputs_path is None else str(rule_inputs_path),
        },
        "label_usage": {
            "labels_used_for_rule_execution": False,
            "answers_copied_to_rule_inputs": False,
            "rule_stubs_are_verifier_evidence": False,
            "candidate_results_require_promotion_gate": True,
        },
        "paths": {
            "report": str(report_path),
            "rule_results": str(results_path),
            "input_requests": str(requests_path),
            "artifact_manifest": str(manifest_path),
        },
        "summary": summary,
        "metadata": dict(metadata or {}),
    }
    _write_json(report_path, payload, compact=compact_json)

    artifacts = {
        "rule_authoring_report": report_path,
        "rule_results": results_path,
        "rule_input_requests": requests_path,
        "rule_stubs": Path(rule_stubs_path),
    }
    if rule_inputs_path is not None:
        artifacts["rule_inputs"] = Path(rule_inputs_path)
    manifest = build_artifact_manifest(
        artifacts,
        root=manifest_path.parent,
        metadata={
            "workflow": WORKFLOW,
            "status": payload["status"],
            "stub_count": summary["stub_count"],
            "executed_count": summary["executed_count"],
            "needs_input_count": summary["needs_input_count"],
            "input_request_count": summary["input_request_count"],
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
                "stub_count": summary["stub_count"],
                "executed_count": summary["executed_count"],
                "needs_input_count": summary["needs_input_count"],
                "input_request_count": summary["input_request_count"],
                "artifact_manifest": str(manifest_path),
                **dict(metadata or {}),
            },
        ).save_json()
    return payload


def _evaluate_stub(stub: Mapping[str, Any], rule_input: Mapping[str, Any] | None) -> dict[str, Any]:
    request_id = str(stub.get("request_id") or "")
    family = str(stub.get("rule_family") or "world_model_consistency")
    required_inputs = _required_inputs(stub, family=family)
    authored_rule = _authored_rule(stub, family=family, required_inputs=required_inputs)
    input_payload = {} if rule_input is None else dict(rule_input)
    if family == "quantity_or_arithmetic" and _calculation_input(input_payload) is not None:
        source_citation = _clean(input_payload.get("source_citation"))
        verification = CalculatorVerifier().verify(
            Claim(
                text=str(stub.get("question") or stub.get("rule_seed") or request_id),
                claim_id=request_id or None,
                metadata={"calculation": _calculation_input(input_payload)},
            )
        )
        evidence = tuple(str(item) for item in verification.evidence)
        if source_citation:
            evidence = tuple((*evidence, f"source_citation={source_citation}"))
        return _result(
            stub=stub,
            status=verification.status.value,
            authored_rule=authored_rule,
            required_inputs=required_inputs,
            supplied_inputs=tuple(input_payload),
            evidence=evidence,
            explanation=verification.explanation,
            confidence=verification.confidence,
            metadata={
                "adapter": "calculator",
                "candidate_verification": verification.metadata,
                "candidate_results_require_promotion_gate": True,
            },
        )
    if family == "entity_disambiguation" and _has_entity_role_inputs(input_payload):
        answer = _clean(input_payload.get("answer_entity"))
        expected = _clean(input_payload.get("expected_entity", input_payload.get("correct_entity")))
        status = (
            VerificationStatus.SUPPORTED.value
            if _norm(answer) == _norm(expected)
            else VerificationStatus.REFUTED.value
        )
        citation = _clean(input_payload.get("source_citation"))
        evidence = (
            "entity_role: "
            f"requested_role={_clean(input_payload.get('requested_role'))}; "
            f"answer_entity={answer}; expected_entity={expected}"
            f"{'; source_citation=' + citation if citation else ''}"
        )
        return _result(
            stub=stub,
            status=status,
            authored_rule=authored_rule,
            required_inputs=required_inputs,
            supplied_inputs=tuple(input_payload),
            evidence=(evidence,),
            explanation="entity-role binding check executed from explicit inputs",
            confidence=0.95,
            metadata={
                "adapter": "entity_role_disambiguation",
                "candidate_results_require_promotion_gate": True,
            },
        )
    if family == "temporal_consistency" and _has_temporal_inputs(input_payload):
        temporal = _temporal_consistency(input_payload)
        return _result(
            stub=stub,
            status=temporal["status"],
            authored_rule=authored_rule,
            required_inputs=required_inputs,
            supplied_inputs=tuple(input_payload),
            evidence=temporal["evidence"],
            explanation=temporal["explanation"],
            confidence=temporal["confidence"],
            metadata={
                "adapter": "temporal_consistency",
                "candidate_results_require_promotion_gate": True,
                "temporal_consistency": temporal["metadata"],
            },
        )
    missing = _missing_inputs(required_inputs, input_payload, family=family)
    return _result(
        stub=stub,
        status="needs_inputs",
        authored_rule=authored_rule,
        required_inputs=required_inputs,
        supplied_inputs=tuple(input_payload),
        missing_inputs=missing,
        explanation="deterministic rule cannot execute until explicit inputs are supplied",
        confidence=1.0,
        metadata={
            "adapter": authored_rule["adapter"],
            "candidate_results_require_promotion_gate": True,
        },
    )


def _result(
    *,
    stub: Mapping[str, Any],
    status: str,
    authored_rule: Mapping[str, Any],
    required_inputs: Sequence[str],
    supplied_inputs: Sequence[str],
    missing_inputs: Sequence[str] = (),
    evidence: Sequence[str] = (),
    explanation: str = "",
    confidence: float = 1.0,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "request_id": str(stub.get("request_id") or ""),
        "target_id": str(stub.get("target_id") or ""),
        "request_type": str(stub.get("request_type") or RULE_REQUEST_TYPE),
        "rule_family": str(stub.get("rule_family") or "world_model_consistency"),
        "status": status,
        "confidence": float(confidence),
        "required_inputs": tuple(required_inputs),
        "supplied_inputs": tuple(sorted(str(item) for item in supplied_inputs if str(item))),
        "missing_inputs": tuple(missing_inputs),
        "question": str(stub.get("question") or ""),
        "question_type": str(stub.get("question_type") or ""),
        "gap_type": str(stub.get("gap_type") or ""),
        "priority": str(stub.get("priority") or ""),
        "authored_rule": dict(authored_rule),
        "evidence": tuple(str(item) for item in evidence if str(item)),
        "explanation": explanation,
        "not_verifier_evidence": True,
        "metadata": dict(metadata or {}),
    }


def _authored_rule(
    stub: Mapping[str, Any],
    *,
    family: str,
    required_inputs: Sequence[str],
) -> dict[str, Any]:
    adapter = {
        "quantity_or_arithmetic": "calculator",
        "entity_disambiguation": "entity_role_disambiguation",
        "temporal_consistency": "temporal_consistency",
        "causal_or_procedural": "world_model_rule",
    }.get(family, "world_model_rule")
    return {
        "rule_id": str(stub.get("request_id") or ""),
        "adapter": adapter,
        "rule_family": family,
        "rule_seed": str(stub.get("rule_seed") or ""),
        "rule_reason": str(stub.get("rule_reason") or ""),
        "input_schema": tuple(required_inputs),
        "not_verifier_evidence": True,
    }


def _input_request(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "request_id": result["request_id"],
        "target_id": result["target_id"],
        "rule_family": result["rule_family"],
        "adapter": _mapping(result.get("authored_rule")).get("adapter"),
        "required_inputs": tuple(result.get("required_inputs", ())),
        "missing_inputs": tuple(result.get("missing_inputs", ())),
        "question": result.get("question"),
        "question_type": result.get("question_type"),
        "gap_type": result.get("gap_type"),
        "priority": result.get("priority"),
        "not_verifier_evidence": True,
    }


def _summary(
    *,
    results: Sequence[Mapping[str, Any]],
    input_requests: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    status_counts = Counter(str(row.get("status") or "") for row in results)
    family_counts = Counter(str(row.get("rule_family") or "") for row in results)
    adapter_counts = Counter(str(_mapping(row.get("authored_rule")).get("adapter") or "") for row in results)
    missing_counts: Counter[str] = Counter()
    for row in results:
        for item in _string_sequence(row.get("missing_inputs", ())):
            missing_counts[item] += 1
    executed_count = sum(count for status, count in status_counts.items() if status in EXECUTED_STATUSES)
    return {
        "stub_count": len(results),
        "result_count": len(results),
        "executed_count": executed_count,
        "needs_input_count": int(status_counts.get("needs_inputs", 0)),
        "input_request_count": len(input_requests),
        "status_counts": _sorted_counter(status_counts),
        "rule_family_counts": _sorted_counter(family_counts),
        "adapter_counts": _sorted_counter(adapter_counts),
        "missing_input_counts": _sorted_counter(missing_counts),
        "candidate_supported_count": int(status_counts.get(VerificationStatus.SUPPORTED.value, 0)),
        "candidate_refuted_count": int(status_counts.get(VerificationStatus.REFUTED.value, 0)),
        "candidate_error_count": int(status_counts.get(VerificationStatus.ERROR.value, 0)),
    }


def _status(summary: Mapping[str, Any]) -> str:
    if int(summary.get("stub_count", 0)) == 0:
        return "empty"
    if int(summary.get("executed_count", 0)) == 0:
        return "needs_inputs"
    if int(summary.get("needs_input_count", 0)) > 0:
        return "partial"
    return "observed"


def _required_inputs(stub: Mapping[str, Any], *, family: str) -> tuple[str, ...]:
    explicit = _string_sequence(stub.get("required_inputs", ()))
    if explicit:
        return explicit
    return INPUT_KEYS_BY_FAMILY.get(family, ("state", "action", "postcondition"))


def _calculation_input(rule_input: Mapping[str, Any]) -> Mapping[str, Any] | None:
    raw = rule_input.get("calculation")
    if isinstance(raw, Mapping):
        return raw
    if "expression" in rule_input and any(key in rule_input for key in ("expected", "result", "answer")):
        return rule_input
    return None


def _has_entity_role_inputs(rule_input: Mapping[str, Any]) -> bool:
    return all(
        _clean(rule_input.get(key))
        for key in ("answer_entity", "requested_role")
    ) and bool(_clean(rule_input.get("expected_entity", rule_input.get("correct_entity"))))


def _has_temporal_inputs(rule_input: Mapping[str, Any]) -> bool:
    return all(_clean(rule_input.get(key)) for key in ("claim_time", "source_time", "retrieved_at", "source_citation"))


def _temporal_consistency(rule_input: Mapping[str, Any]) -> dict[str, Any]:
    claim_time = _parse_temporal_value(rule_input.get("claim_time"))
    source_time = _parse_temporal_value(rule_input.get("source_time"))
    retrieved_at = _parse_temporal_value(rule_input.get("retrieved_at"))
    source_citation = _clean(rule_input.get("source_citation"))
    raw_metadata = {
        "claim_time": _clean(rule_input.get("claim_time")),
        "source_time": _clean(rule_input.get("source_time")),
        "retrieved_at": _clean(rule_input.get("retrieved_at")),
        "source_citation": source_citation,
        "relation": "source_time_at_or_after_claim_time_and_not_after_retrieval",
    }
    if claim_time is None or source_time is None or retrieved_at is None:
        return {
            "status": VerificationStatus.ERROR.value,
            "confidence": 1.0,
            "evidence": (
                "temporal_consistency: invalid temporal input; "
                f"claim_time={raw_metadata['claim_time']}; "
                f"source_time={raw_metadata['source_time']}; "
                f"retrieved_at={raw_metadata['retrieved_at']}; "
                f"source_citation={source_citation}",
            ),
            "explanation": "temporal consistency check could not parse one or more explicit inputs",
            "metadata": {**raw_metadata, "failure": "invalid_temporal_input"},
        }
    parsed_metadata = {
        **raw_metadata,
        "parsed_claim_time": claim_time.isoformat(),
        "parsed_source_time": source_time.isoformat(),
        "parsed_retrieved_at": retrieved_at.isoformat(),
    }
    if source_time > retrieved_at:
        return {
            "status": VerificationStatus.ERROR.value,
            "confidence": 1.0,
            "evidence": (
                "temporal_consistency: source_time occurs after retrieved_at; "
                f"source_time={source_time.isoformat()}; "
                f"retrieved_at={retrieved_at.isoformat()}; "
                f"source_citation={source_citation}",
            ),
            "explanation": "temporal source metadata is inconsistent with retrieval time",
            "metadata": {**parsed_metadata, "failure": "source_time_after_retrieved_at"},
        }
    if claim_time > retrieved_at:
        return {
            "status": VerificationStatus.REFUTED.value,
            "confidence": 0.95,
            "evidence": (
                "temporal_consistency: claim_time occurs after retrieved_at; "
                f"claim_time={claim_time.isoformat()}; "
                f"retrieved_at={retrieved_at.isoformat()}; "
                f"source_citation={source_citation}",
            ),
            "explanation": "explicit claim time is later than the retrieval snapshot",
            "metadata": {**parsed_metadata, "failure": "claim_time_after_retrieved_at"},
        }
    if source_time < claim_time:
        return {
            "status": VerificationStatus.REFUTED.value,
            "confidence": 0.95,
            "evidence": (
                "temporal_consistency: source_time predates claim_time; "
                f"source_time={source_time.isoformat()}; "
                f"claim_time={claim_time.isoformat()}; "
                f"source_citation={source_citation}",
            ),
            "explanation": "source timestamp is older than the asserted claim time",
            "metadata": {**parsed_metadata, "failure": "source_time_before_claim_time"},
        }
    return {
        "status": VerificationStatus.SUPPORTED.value,
        "confidence": 0.95,
        "evidence": (
            "temporal_consistency: source_time covers claim_time before retrieval; "
            f"claim_time={claim_time.isoformat()}; "
            f"source_time={source_time.isoformat()}; "
            f"retrieved_at={retrieved_at.isoformat()}; "
            f"source_citation={source_citation}",
        ),
        "explanation": "source timestamp is current enough for the explicit claim time",
        "metadata": {**parsed_metadata, "status": "temporally_consistent"},
    }


def _parse_temporal_value(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    else:
        text = _clean(value)
        if not text:
            return None
        if re.fullmatch(r"\d{4}", text):
            parsed = datetime(int(text), 1, 1, tzinfo=timezone.utc)
        elif re.fullmatch(r"\d{4}-\d{2}", text):
            parsed = datetime.strptime(text, "%Y-%m").replace(tzinfo=timezone.utc)
        else:
            try:
                parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            except ValueError:
                try:
                    parsed = datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                except ValueError:
                    return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _missing_inputs(required: Sequence[str], supplied: Mapping[str, Any], *, family: str) -> tuple[str, ...]:
    if _calculation_input(supplied) is not None:
        return ()
    if _has_entity_role_inputs(supplied):
        return ()
    if _has_temporal_inputs(supplied):
        return ()
    missing = tuple(key for key in required if not _clean(supplied.get(key)))
    if missing:
        return missing
    if family == "quantity_or_arithmetic":
        return ("calculation.expression", "calculation.expected")
    if family == "entity_disambiguation":
        expected = _clean(supplied.get("expected_entity", supplied.get("correct_entity")))
        if not expected:
            return ("expected_entity",)
    return ()


def _load_rule_inputs(path: str | Path | None) -> dict[str, Mapping[str, Any]]:
    if path is None:
        return {}
    input_path = Path(path)
    if input_path.suffix == ".jsonl":
        rows = _load_jsonl_mappings(input_path)
    else:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping):
            raw_rows = payload.get("inputs", payload.get("rule_inputs"))
            if isinstance(raw_rows, Sequence) and not isinstance(raw_rows, (str, bytes, bytearray)):
                rows = tuple(item for item in raw_rows if isinstance(item, Mapping))
            else:
                rows = tuple(
                    {"request_id": request_id, **dict(value)}
                    for request_id, value in payload.items()
                    if isinstance(value, Mapping)
                )
        elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
            rows = tuple(item for item in payload if isinstance(item, Mapping))
        else:
            raise ValueError(f"{path} must contain JSON object/list or JSONL mappings.")
    output: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        request_id = str(row.get("request_id") or "")
        if request_id:
            output[request_id] = row
    return output


def _load_jsonl_mappings(path: str | Path) -> tuple[Mapping[str, Any], ...]:
    rows = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, Mapping):
            raise ValueError(f"{path}:{line_no} must contain a JSON object.")
        rows.append(dict(row))
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


def _string_sequence(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(str(item) for item in value if str(item))
    return ()


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _norm(value: Any) -> str:
    return re.sub(r"\W+", "", _clean(value).casefold())


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
    parser.add_argument("--rule-stubs", required=True)
    parser.add_argument("--rule-inputs", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json", default=None)
    parser.add_argument("--rule-results-jsonl", default=None)
    parser.add_argument("--input-requests-jsonl", default=None)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--compact-json", action="store_true")
    args = parser.parse_args(argv)
    payload = run_world_model_rule_authoring_adapter(
        rule_stubs_path=args.rule_stubs,
        rule_inputs_path=args.rule_inputs,
        output_dir=args.output_dir,
        report_json_path=args.json,
        rule_results_path=args.rule_results_jsonl,
        input_requests_path=args.input_requests_jsonl,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        metadata=_parse_metadata(args.metadata or ()),
        compact_json=bool(args.compact_json),
    )
    summary = payload["summary"]
    print(
        "world_model_rule_authoring_adapter_ok "
        f"status={payload['status']} "
        f"stubs={summary['stub_count']} "
        f"executed={summary['executed_count']} "
        f"needs_inputs={summary['needs_input_count']}"
    )


if __name__ == "__main__":
    main()

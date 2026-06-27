"""Run a local uncertainty-escalation control-loop fixture workflow.

This no-model workflow consumes claim/evidence fixture records, runs
``run_verification_loop(..., escalation_policy=...)`` for each record, writes a
JSONL sidecar of ``VerificationLoopResult.to_dict()`` rows, and emits the same
post-hoc report produced by ``eval_uncertainty_escalation.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eigentruth.adapters import InMemoryRetriever, RetrievalActionExecutor  # noqa: E402
from eigentruth.calibration import CalibrationArtifact, CalibrationScore  # noqa: E402
from eigentruth.control import (  # noqa: E402
    ActionExecutorRegistry,
    ControlAction,
    RiskController,
    run_verification_loop,
)
from eigentruth.eval import uncertainty_escalation_report  # noqa: E402
from eigentruth.json_utils import strict_json_dumps, to_jsonable  # noqa: E402
from eigentruth.registry import ArtifactRegistry, ArtifactVerificationContext  # noqa: E402
from eigentruth.verify import Claim, GroundednessVerifier, VerificationEscalationPolicy  # noqa: E402
from eigentruth.verify.protocols import VerificationResult, VerificationStatus  # noqa: E402

DEFAULT_DIAGNOSTIC_SCORE_NAME = "maha_last"
DEFAULT_DIAGNOSTIC_THRESHOLD = 3.0
DEFAULT_DIAGNOSTIC_VALUE = 1.0


@dataclass(frozen=True)
class UncertaintyEscalationWorkflowConfig:
    """Configuration for a local uncertainty-escalation fixture workflow."""

    records_path: Path
    output_dir: Path
    report_path: Path | None = None
    loop_results_jsonl_path: Path | None = None
    artifact_manifest_path: Path | None = None
    verification_report_path: Path | None = None
    registry_path: Path | None = None
    name: str = "uncertainty-escalation-fixture-workflow"
    version: str = "0.1"
    min_confidence: float = 0.65
    retriever_min_overlap: float = 0.2
    retrieval_limit: int = 5
    verifier_min_overlap: float = 0.65
    diagnostic_score_name: str = DEFAULT_DIAGNOSTIC_SCORE_NAME
    diagnostic_threshold: float = DEFAULT_DIAGNOSTIC_THRESHOLD
    diagnostic_value: float = DEFAULT_DIAGNOSTIC_VALUE
    compact_json: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "records_path", Path(self.records_path))
        output_dir = Path(self.output_dir)
        object.__setattr__(self, "output_dir", output_dir)
        object.__setattr__(
            self,
            "report_path",
            output_dir / "uncertainty-escalation-workflow.json"
            if self.report_path is None
            else Path(self.report_path),
        )
        object.__setattr__(
            self,
            "loop_results_jsonl_path",
            output_dir / "verification-loop-results.jsonl"
            if self.loop_results_jsonl_path is None
            else Path(self.loop_results_jsonl_path),
        )
        if self.artifact_manifest_path is not None:
            object.__setattr__(self, "artifact_manifest_path", Path(self.artifact_manifest_path))
        if self.verification_report_path is not None:
            object.__setattr__(self, "verification_report_path", Path(self.verification_report_path))
            if self.artifact_manifest_path is None:
                raise ValueError("verification_report_path requires artifact_manifest_path.")
        if self.registry_path is not None:
            object.__setattr__(self, "registry_path", Path(self.registry_path))
        if not str(self.name).strip():
            raise ValueError("name must be non-empty.")
        if not str(self.version).strip():
            raise ValueError("version must be non-empty.")
        if not (0.0 <= float(self.min_confidence) <= 1.0):
            raise ValueError("min_confidence must be in [0, 1].")
        if not (0.0 <= float(self.retriever_min_overlap) <= 1.0):
            raise ValueError("retriever_min_overlap must be in [0, 1].")
        if not (0.0 <= float(self.verifier_min_overlap) <= 1.0):
            raise ValueError("verifier_min_overlap must be in [0, 1].")
        if int(self.retrieval_limit) < 1:
            raise ValueError("retrieval_limit must be >= 1.")
        if not str(self.diagnostic_score_name).strip():
            raise ValueError("diagnostic_score_name must be non-empty.")
        object.__setattr__(self, "min_confidence", float(self.min_confidence))
        object.__setattr__(self, "retriever_min_overlap", float(self.retriever_min_overlap))
        object.__setattr__(self, "retrieval_limit", int(self.retrieval_limit))
        object.__setattr__(self, "verifier_min_overlap", float(self.verifier_min_overlap))
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "version", str(self.version))
        object.__setattr__(self, "diagnostic_score_name", str(self.diagnostic_score_name))
        object.__setattr__(self, "diagnostic_threshold", float(self.diagnostic_threshold))
        object.__setattr__(self, "diagnostic_value", float(self.diagnostic_value))


def run_uncertainty_escalation_workflow(
    config: UncertaintyEscalationWorkflowConfig,
) -> dict[str, Any]:
    """Run the fixture workflow and return the saved report payload."""
    records = load_fixture_records(config.records_path)
    if not records:
        raise ValueError("fixture must contain at least one record.")

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    assert config.loop_results_jsonl_path is not None
    config.loop_results_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    assert config.report_path is not None
    config.report_path.parent.mkdir(parents=True, exist_ok=True)

    controller = RiskController(_calibration_artifact(config))
    escalation_policy = VerificationEscalationPolicy(min_confidence=config.min_confidence)
    rows = []
    with config.loop_results_jsonl_path.open("w", encoding="utf-8") as stream:
        for index, record in enumerate(records):
            row = _run_record(
                record,
                index=index,
                config=config,
                controller=controller,
                escalation_policy=escalation_policy,
            )
            rows.append(row)
            stream.write(strict_json_dumps(row, sort_keys=True) + "\n")

    report = uncertainty_escalation_report(rows)
    payload = {
        "workflow": "uncertainty_escalation_fixture_workflow",
        "schema_version": 1,
        "name": config.name,
        "version": config.version,
        "config": _config_payload(config),
        "input": {
            "records_path": str(config.records_path),
            "record_count": len(records),
        },
        "paths": {
            "loop_results_jsonl": str(config.loop_results_jsonl_path),
            "report": str(config.report_path),
        },
        "report": report,
    }
    if config.artifact_manifest_path is not None:
        config.artifact_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        payload["paths"]["artifact_manifest"] = str(config.artifact_manifest_path)
    if config.verification_report_path is not None:
        config.verification_report_path.parent.mkdir(parents=True, exist_ok=True)
        payload["paths"]["manifest_verification"] = str(config.verification_report_path)

    config.report_path.write_text(_json_text(payload, compact=config.compact_json), encoding="utf-8")

    verification_payload = _write_artifact_manifest_and_verification(config, report)
    if verification_payload is not None:
        payload["manifest_verification"] = verification_payload
    _record_registry(config, report, verification_payload)
    return to_jsonable(payload)


def load_fixture_records(path: Path) -> tuple[Mapping[str, Any], ...]:
    """Load fixture records from JSON or JSONL."""
    if path.suffix.lower() == ".jsonl":
        records = []
        with open(path, encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped)
                if not isinstance(payload, Mapping):
                    raise ValueError(f"{path}:{line_number} must contain a JSON object.")
                records.append(payload)
        return tuple(records)

    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, Mapping):
        raw_records = payload.get("records", payload.get("claims"))
        if raw_records is None:
            raw_records = (payload,)
    else:
        raw_records = payload
    if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes, bytearray)):
        raise ValueError("fixture JSON must be an object, records/claims object, or an array.")
    records = []
    for index, item in enumerate(raw_records):
        if not isinstance(item, Mapping):
            raise ValueError(f"fixture record {index} must be a JSON object.")
        records.append(item)
    return tuple(records)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, type=Path, help="JSON or JSONL claim/evidence fixture")
    parser.add_argument("--output-dir", required=True, type=Path, help="workflow output directory")
    parser.add_argument("--json", default=None, type=Path, help="optional workflow report path")
    parser.add_argument("--loop-results-jsonl", default=None, type=Path, help="optional loop-result JSONL path")
    parser.add_argument("--artifact-manifest", default=None, type=Path, help="optional artifact manifest path")
    parser.add_argument("--verification-report", default=None, type=Path, help="optional manifest verification path")
    parser.add_argument("--registry", default=None, type=Path, help="optional ArtifactRegistry JSON path")
    parser.add_argument("--name", default="uncertainty-escalation-fixture-workflow")
    parser.add_argument("--version", default="0.1")
    parser.add_argument("--min-confidence", type=float, default=0.65)
    parser.add_argument("--retriever-min-overlap", type=float, default=0.2)
    parser.add_argument("--retrieval-limit", type=int, default=5)
    parser.add_argument("--verifier-min-overlap", type=float, default=0.65)
    parser.add_argument("--diagnostic-score-name", default=DEFAULT_DIAGNOSTIC_SCORE_NAME)
    parser.add_argument("--diagnostic-threshold", type=float, default=DEFAULT_DIAGNOSTIC_THRESHOLD)
    parser.add_argument("--diagnostic-value", type=float, default=DEFAULT_DIAGNOSTIC_VALUE)
    parser.add_argument("--compact-json", action="store_true")
    args = parser.parse_args(argv)

    payload = run_uncertainty_escalation_workflow(
        UncertaintyEscalationWorkflowConfig(
            records_path=args.records,
            output_dir=args.output_dir,
            report_path=args.json,
            loop_results_jsonl_path=args.loop_results_jsonl,
            artifact_manifest_path=args.artifact_manifest,
            verification_report_path=args.verification_report,
            registry_path=args.registry,
            name=args.name,
            version=args.version,
            min_confidence=args.min_confidence,
            retriever_min_overlap=args.retriever_min_overlap,
            retrieval_limit=args.retrieval_limit,
            verifier_min_overlap=args.verifier_min_overlap,
            diagnostic_score_name=args.diagnostic_score_name,
            diagnostic_threshold=args.diagnostic_threshold,
            diagnostic_value=args.diagnostic_value,
            compact_json=args.compact_json,
        )
    )
    print(f"Wrote uncertainty escalation workflow report to {payload['paths']['report']}")
    print(f"Wrote loop results to {payload['paths']['loop_results_jsonl']}")
    if "artifact_manifest" in payload["paths"]:
        print(f"Wrote artifact manifest to {payload['paths']['artifact_manifest']}")
    if "manifest_verification" in payload["paths"]:
        print(f"Wrote manifest verification to {payload['paths']['manifest_verification']}")
    return 0


class _PreliminaryThenGroundednessVerifier:
    def __init__(
        self,
        *,
        preliminary_status: VerificationStatus,
        preliminary_confidence: float,
        preliminary_evidence: Sequence[str],
        preliminary_explanation: str,
        final_verifier: GroundednessVerifier,
        record_id: str,
    ):
        self.preliminary_status = preliminary_status
        self.preliminary_confidence = preliminary_confidence
        self.preliminary_evidence = tuple(preliminary_evidence)
        self.preliminary_explanation = preliminary_explanation
        self.final_verifier = final_verifier
        self.record_id = record_id

    def verify_many(
        self,
        claims: Sequence[Claim],
        context: Mapping[str, Any] | None = None,
    ) -> tuple[VerificationResult, ...]:
        del context
        return tuple(
            VerificationResult(
                status=self.preliminary_status,
                confidence=self.preliminary_confidence,
                evidence=self.preliminary_evidence,
                explanation=self.preliminary_explanation,
                metadata={
                    "verifier": type(self).__name__,
                    "phase": "preliminary",
                    "claim_id": claim.claim_id,
                    "record_id": self.record_id,
                },
            )
            for claim in claims
        )

    def verify(self, claim: Claim, context: Mapping[str, Any] | None = None) -> VerificationResult:
        return self.final_verifier.verify(claim, context=context)


def _run_record(
    record: Mapping[str, Any],
    *,
    index: int,
    config: UncertaintyEscalationWorkflowConfig,
    controller: RiskController,
    escalation_policy: VerificationEscalationPolicy,
) -> dict[str, Any]:
    claim = _claim_from_record(record, index=index)
    record_id = str(record.get("id", record.get("record_id", claim.claim_id or f"record-{index + 1}")))
    retrieval_documents = _as_sequence(record.get("retrieval_documents", record.get("documents", ())))
    retriever = InMemoryRetriever(
        retrieval_documents,
        min_overlap=config.retriever_min_overlap,
    )
    registry = ActionExecutorRegistry().register(
        ControlAction.RETRIEVE,
        RetrievalActionExecutor(retriever, limit=config.retrieval_limit),
    )
    verifier = _verifier_from_record(
        record,
        record_id=record_id,
        verifier_min_overlap=config.verifier_min_overlap,
    )
    diagnostics = _diagnostics_from_record(record, config=config)
    context = _context_from_record(record)
    result = run_verification_loop(
        request_id=record_id,
        diagnostics=diagnostics,
        claims=(claim,),
        verifier=verifier,
        controller=controller,
        executor_registry=registry,
        context=context,
        metadata={
            "fixture_record_id": record_id,
            "fixture_index": index,
            "source": "benchmarks.run_uncertainty_escalation_workflow",
        },
        escalation_policy=escalation_policy,
    )
    row: dict[str, Any] = {
        "id": record_id,
        "result": result.to_dict(),
        "metadata": {
            "claim_id": claim.claim_id,
            "retrieval_document_count": len(retrieval_documents),
        },
    }
    if "label" in record:
        row["label"] = record["label"]
    return to_jsonable(row)


def _verifier_from_record(
    record: Mapping[str, Any],
    *,
    record_id: str,
    verifier_min_overlap: float,
) -> _PreliminaryThenGroundednessVerifier:
    refutations = record.get("refutations", {})
    if isinstance(refutations, str):
        refutations = {_record_claim_text(record): (refutations,)}
    elif isinstance(refutations, Sequence) and not isinstance(refutations, (bytes, bytearray)):
        claim_text = _record_claim_text(record)
        refutations = {claim_text: tuple(str(item) for item in refutations)}
    if not isinstance(refutations, Mapping):
        raise ValueError("record.refutations must be a mapping, list, or string.")
    final_verifier = GroundednessVerifier(
        evidence=(),
        refutations={str(key): _as_string_sequence(value) for key, value in refutations.items()},
        min_overlap=verifier_min_overlap,
    )
    return _PreliminaryThenGroundednessVerifier(
        preliminary_status=_verification_status(record.get("preliminary_status", "supported")),
        preliminary_confidence=float(record.get("preliminary_confidence", 0.4)),
        preliminary_evidence=_as_string_sequence(record.get("preliminary_evidence", ())),
        preliminary_explanation=str(
            record.get(
                "preliminary_explanation",
                "cheap preliminary verifier returned low-confidence support",
            )
        ),
        final_verifier=final_verifier,
        record_id=record_id,
    )


def _claim_from_record(record: Mapping[str, Any], *, index: int) -> Claim:
    claim_text = _record_claim_text(record)
    claim_id = str(record.get("claim_id", f"c{index + 1}"))
    metadata = record.get("claim_metadata", record.get("metadata", {}))
    if not isinstance(metadata, Mapping):
        metadata = {}
    return Claim(
        text=claim_text,
        claim_id=claim_id,
        metadata=dict(metadata),
    )


def _record_claim_text(record: Mapping[str, Any]) -> str:
    raw_claim = record.get("claim", record.get("text"))
    if isinstance(raw_claim, Mapping):
        raw_claim = raw_claim.get("text")
    if raw_claim is None or not str(raw_claim).strip():
        raise ValueError("fixture record must include non-empty claim/text.")
    return str(raw_claim)


def _diagnostics_from_record(
    record: Mapping[str, Any],
    *,
    config: UncertaintyEscalationWorkflowConfig,
) -> dict[str, float]:
    raw_diagnostics = record.get("diagnostics")
    if raw_diagnostics is None:
        return {config.diagnostic_score_name: config.diagnostic_value}
    if not isinstance(raw_diagnostics, Mapping):
        raise ValueError("record.diagnostics must be a JSON object.")
    return {str(key): float(value) for key, value in raw_diagnostics.items()}


def _context_from_record(record: Mapping[str, Any]) -> dict[str, Any]:
    raw_context = record.get("context", {})
    if raw_context is None:
        raw_context = {}
    if not isinstance(raw_context, Mapping):
        raise ValueError("record.context must be a JSON object.")
    context = dict(raw_context)
    if "refutations" in record and "refutations" not in context:
        context["refutations"] = record["refutations"]
    return context


def _calibration_artifact(config: UncertaintyEscalationWorkflowConfig) -> CalibrationArtifact:
    return CalibrationArtifact(
        model_id="uncertainty-escalation-fixture",
        target_layer=-1,
        scores=(
            CalibrationScore(
                name=config.diagnostic_score_name,
                threshold=config.diagnostic_threshold,
                direction="higher",
            ),
        ),
        eigentruth_version="0.2.0",
    )


def _config_payload(config: UncertaintyEscalationWorkflowConfig) -> dict[str, Any]:
    return {
        "name": config.name,
        "version": config.version,
        "min_confidence": config.min_confidence,
        "retriever_min_overlap": config.retriever_min_overlap,
        "retrieval_limit": config.retrieval_limit,
        "verifier_min_overlap": config.verifier_min_overlap,
        "diagnostic_score_name": config.diagnostic_score_name,
        "diagnostic_threshold": config.diagnostic_threshold,
        "diagnostic_value": config.diagnostic_value,
        "compact_json": config.compact_json,
    }


def _write_artifact_manifest_and_verification(
    config: UncertaintyEscalationWorkflowConfig,
    report: Mapping[str, Any],
) -> dict[str, Any] | None:
    if config.artifact_manifest_path is None:
        if config.verification_report_path is not None:
            raise ValueError("verification_report_path requires artifact_manifest_path.")
        return None

    context = ArtifactVerificationContext()
    manifest = context.build_artifact_manifest(
        {
            "fixture_records": config.records_path,
            "loop_results_jsonl": config.loop_results_jsonl_path,
            "workflow_report": config.report_path,
        },
        root=config.artifact_manifest_path.parent,
        metadata={
            "workflow": "uncertainty_escalation_fixture_workflow",
            "name": config.name,
            "version": config.version,
            "record_count": report.get("n_total"),
            "triggered_records": _nested(report, "uncertainty_escalation", "triggered_records"),
            "retrieval_evidence_records": _nested(
                report,
                "action_execution",
                "retrieval_evidence_records",
            ),
            "accepted_false_delta": _nested(report, "quality", "delta", "accepted_false"),
        },
    )
    config.artifact_manifest_path.write_text(
        _json_text(manifest, compact=config.compact_json),
        encoding="utf-8",
    )
    if config.verification_report_path is None:
        return None
    verification = context.load_and_verify_artifact_manifest(
        config.artifact_manifest_path,
        recursive=True,
    )
    verification_payload = verification.to_dict()
    config.verification_report_path.write_text(
        _json_text(verification_payload, compact=config.compact_json),
        encoding="utf-8",
    )
    return verification_payload


def _record_registry(
    config: UncertaintyEscalationWorkflowConfig,
    report: Mapping[str, Any],
    verification_payload: Mapping[str, Any] | None,
) -> None:
    if config.registry_path is None:
        return
    ArtifactRegistry.load_json(config.registry_path).record_report(
        name=config.name,
        path=config.report_path if config.report_path is not None else "",
        version=config.version,
        metadata={
            "workflow": "uncertainty_escalation_fixture_workflow",
            "artifact_manifest": None
            if config.artifact_manifest_path is None
            else str(config.artifact_manifest_path),
            "manifest_verification": None
            if config.verification_report_path is None
            else str(config.verification_report_path),
            "manifest_verified": None
            if verification_payload is None
            else bool(verification_payload.get("passed")),
            "record_count": report.get("n_total"),
            "triggered_records": _nested(report, "uncertainty_escalation", "triggered_records"),
            "retrieval_evidence_records": _nested(
                report,
                "action_execution",
                "retrieval_evidence_records",
            ),
            "accepted_false_delta": _nested(report, "quality", "delta", "accepted_false"),
            "compact_json": config.compact_json,
        },
    ).save_json()


def _nested(payload: Mapping[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _verification_status(value: Any) -> VerificationStatus:
    if isinstance(value, VerificationStatus):
        return value
    return VerificationStatus(str(value))


def _as_sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(value)
    return (value,)


def _as_string_sequence(value: Any) -> tuple[str, ...]:
    return tuple(str(item) for item in _as_sequence(value) if str(item).strip())


def _json_text(payload: Mapping[str, Any], *, compact: bool) -> str:
    if compact:
        return strict_json_dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    return strict_json_dumps(payload, indent=2, sort_keys=True) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())

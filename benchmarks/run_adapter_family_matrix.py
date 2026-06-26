"""Build a deterministic adapter-family promotion matrix.

This no-model workflow creates small local fixtures for verifier-route
families, runs the same refresh/promotion gate for each route, then aggregates
the generated verifier reports into a route comparison matrix.
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

from benchmarks.build_domain_state_fixture import build_order_fulfillment_fixture  # noqa: E402
from benchmarks.build_evidence_fixture import build_evidence_fixture  # noqa: E402
from benchmarks.build_transition_fixture import build_order_transition_fixture  # noqa: E402
from benchmarks.compare_verifier_routes import build_route_comparison_report  # noqa: E402
from benchmarks.refresh_verifier_route_artifacts import (  # noqa: E402
    VerifierRouteArtifactRefreshConfig,
    refresh_verifier_route_artifacts,
)

RETRIEVAL_GROUNDEDNESS_ROUTE = "retrieval_groundedness"
RETRIEVAL_STRUCTURED_QA_ROUTE = "retrieval_structured_qa"
TRIPLE_EVIDENCE_ROUTE = "triple_evidence"


@dataclass(frozen=True)
class AdapterFamilyMatrixConfig:
    """Configuration for the deterministic adapter-family matrix workflow."""

    output_dir: Path
    matrix_report_path: Path | None = None
    alpha: float = 0.20
    n_records: int = 8
    signal: str = "truth_proj"
    compact_json: bool = False
    min_decision_accuracy: float = 1.0
    max_false_supported_rate: float = 0.0
    min_false_refuted_rate: float = 1.0
    max_mean_duration_seconds: float = 1.0
    max_p99_duration_seconds: float = 1.0
    max_max_duration_seconds: float = 1.0
    max_mean_attempted_route_count: float = 1.1
    max_retrieval_use_rate: float = 0.0
    include_retrieval: bool = False
    include_retrieval_structured_qa: bool = False
    include_triple_evidence: bool = False
    verifier_min_overlap: float = 0.65
    retriever_min_overlap: float = 0.6
    retrieval_limit: int = 1
    triple_min_slot_coverage: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.matrix_report_path is not None:
            object.__setattr__(self, "matrix_report_path", Path(self.matrix_report_path))
        if self.n_records < 2 or self.n_records % 2:
            raise ValueError("n_records must be an even integer >= 2.")
        if not (0.0 < float(self.alpha) < 1.0):
            raise ValueError("alpha must be in (0, 1).")
        if int(self.retrieval_limit) <= 0:
            raise ValueError("retrieval_limit must be positive.")
        if not (0.0 <= float(self.triple_min_slot_coverage) <= 1.0):
            raise ValueError("triple_min_slot_coverage must be in [0, 1].")


def run_adapter_family_matrix(config: AdapterFamilyMatrixConfig) -> dict[str, Any]:
    """Run local route-family promotion checks and return a matrix report."""
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    families = [
        _run_structured_qa(config),
        _run_structured_state(config),
        _run_state_transition(config),
    ]
    if config.include_retrieval:
        families.append(_run_retrieval_groundedness(config))
    if config.include_retrieval_structured_qa:
        families.append(_run_retrieval_structured_qa(config))
    if config.include_triple_evidence:
        families.append(_run_triple_evidence(config))
    routes = tuple(str(item["route"]) for item in families)
    comparison_path = output_dir / "route-family-comparison.json"
    comparison = build_route_comparison_report(
        tuple((str(item["route"]), Path(item["verifier_report_path"])) for item in families),
        alpha=config.alpha,
        min_selected=1,
        notes=("deterministic adapter-family matrix",),
        gate_routes=routes,
        gate_min_selected=config.n_records,
        min_decision_accuracy=config.min_decision_accuracy,
        max_false_supported_rate=config.max_false_supported_rate,
        min_false_refuted_rate=config.min_false_refuted_rate,
        max_mean_duration_seconds=config.max_mean_duration_seconds,
        max_p99_duration_seconds=config.max_p99_duration_seconds,
        max_max_duration_seconds=config.max_max_duration_seconds,
        max_mean_attempted_route_count=config.max_mean_attempted_route_count,
        max_retrieval_use_rate=config.max_retrieval_use_rate,
    )
    comparison_path.write_text(
        _json_text(comparison, compact=config.compact_json, sort_keys=True),
        encoding="utf-8",
    )
    matrix_report_path = config.matrix_report_path or output_dir / "adapter-family-matrix.json"
    report = {
        "schema_version": 1,
        "workflow": "adapter_family_matrix",
        "alpha": float(config.alpha),
        "n_records": int(config.n_records),
        "signal": config.signal,
        "routes": routes,
        "include_retrieval": bool(config.include_retrieval),
        "include_retrieval_structured_qa": bool(config.include_retrieval_structured_qa),
        "include_triple_evidence": bool(config.include_triple_evidence),
        "retrieval_routes": tuple(
            item["route"]
            for item in families
            if str(item["route"]).startswith("retrieval_")
        ),
        "audit_routes": tuple(
            item["route"]
            for item in families
            if str(item["route"]) == TRIPLE_EVIDENCE_ROUTE
        ),
        "families": families,
        "route_comparison_path": str(comparison_path),
        "route_comparison": comparison,
        "promotion_decision": comparison["promotion_decision"],
    }
    matrix_report_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_report_path.write_text(
        _json_text(report, compact=config.compact_json, sort_keys=True),
        encoding="utf-8",
    )
    return report


def _run_structured_qa(config: AdapterFamilyMatrixConfig) -> dict[str, Any]:
    route = "structured_qa"
    route_dir = config.output_dir / route
    route_dir.mkdir(parents=True, exist_ok=True)
    scores_path = route_dir / "scores.json"
    qa_corpus_path = route_dir / "qa-corpus.json"
    _write_structured_qa_fixture(
        scores_path=scores_path,
        qa_corpus_path=qa_corpus_path,
        n_records=config.n_records,
        signal=config.signal,
        compact=config.compact_json,
    )
    return _refresh_route(
        config,
        route=route,
        score_name="qa",
        scores_path=scores_path,
        claims_path=None,
        qa_corpus_path=qa_corpus_path,
        state_path=None,
    )


def _run_structured_state(config: AdapterFamilyMatrixConfig) -> dict[str, Any]:
    route = "structured_state"
    route_dir = config.output_dir / route
    route_dir.mkdir(parents=True, exist_ok=True)
    fixture = build_order_fulfillment_fixture(n_records=config.n_records, signal=config.signal)
    scores_path = route_dir / "scores.json"
    claims_path = route_dir / "claims.json"
    state_path = route_dir / "state.json"
    _write_json(scores_path, fixture["scores"], compact=config.compact_json)
    _write_json(claims_path, fixture["claims"], compact=config.compact_json)
    _write_json(state_path, fixture["state"], compact=config.compact_json)
    return _refresh_route(
        config,
        route=route,
        score_name="state",
        scores_path=scores_path,
        claims_path=claims_path,
        qa_corpus_path=None,
        state_path=state_path,
    )


def _run_state_transition(config: AdapterFamilyMatrixConfig) -> dict[str, Any]:
    route = "state_transition"
    route_dir = config.output_dir / route
    route_dir.mkdir(parents=True, exist_ok=True)
    fixture = build_order_transition_fixture(
        n_records=config.n_records,
        signal=config.signal,
        rule_based_world_model=True,
    )
    scores_path = route_dir / "scores.json"
    claims_path = route_dir / "claims.json"
    state_path = route_dir / "state.json"
    _write_json(scores_path, fixture["scores"], compact=config.compact_json)
    _write_json(claims_path, fixture["claims"], compact=config.compact_json)
    _write_json(state_path, fixture["state"], compact=config.compact_json)
    return _refresh_route(
        config,
        route=route,
        score_name="transition",
        scores_path=scores_path,
        claims_path=claims_path,
        qa_corpus_path=None,
        state_path=state_path,
    )


def _run_retrieval_groundedness(config: AdapterFamilyMatrixConfig) -> dict[str, Any]:
    route = RETRIEVAL_GROUNDEDNESS_ROUTE
    route_dir = config.output_dir / route
    route_dir.mkdir(parents=True, exist_ok=True)
    scores_path = route_dir / "scores.json"
    corpus_path = route_dir / "retrieval-corpus.json"
    claims_path = route_dir / "retrieval-claims.json"
    _write_retrieval_fixture_inputs(
        scores_path=scores_path,
        corpus_path=corpus_path,
        claims_path=claims_path,
        n_records=config.n_records,
        signal=config.signal,
        compact=config.compact_json,
        retriever_min_overlap=config.retriever_min_overlap,
        retrieval_limit=config.retrieval_limit,
        structured_qa=False,
    )
    return _refresh_route(
        config,
        route=route,
        score_name="retrieval",
        scores_path=scores_path,
        claims_path=claims_path,
        qa_corpus_path=None,
        state_path=None,
    )


def _run_retrieval_structured_qa(config: AdapterFamilyMatrixConfig) -> dict[str, Any]:
    route = RETRIEVAL_STRUCTURED_QA_ROUTE
    route_dir = config.output_dir / route
    route_dir.mkdir(parents=True, exist_ok=True)
    scores_path = route_dir / "scores.json"
    corpus_path = route_dir / "retrieval-qa-corpus.json"
    claims_path = route_dir / "retrieval-qa-claims.json"
    _write_retrieval_fixture_inputs(
        scores_path=scores_path,
        corpus_path=corpus_path,
        claims_path=claims_path,
        n_records=config.n_records,
        signal=config.signal,
        compact=config.compact_json,
        retriever_min_overlap=config.retriever_min_overlap,
        retrieval_limit=config.retrieval_limit,
        structured_qa=True,
    )
    return _refresh_route(
        config,
        route=route,
        score_name="retrieval_qa",
        scores_path=scores_path,
        claims_path=claims_path,
        qa_corpus_path=None,
        state_path=None,
    )


def _run_triple_evidence(config: AdapterFamilyMatrixConfig) -> dict[str, Any]:
    route = TRIPLE_EVIDENCE_ROUTE
    route_dir = config.output_dir / route
    route_dir.mkdir(parents=True, exist_ok=True)
    scores_path = route_dir / "scores.json"
    claims_path = route_dir / "triple-claims.json"
    _write_triple_evidence_fixture_inputs(
        scores_path=scores_path,
        claims_path=claims_path,
        n_records=config.n_records,
        signal=config.signal,
        compact=config.compact_json,
    )
    return _refresh_route(
        config,
        route=route,
        score_name="triple_evidence",
        scores_path=scores_path,
        claims_path=claims_path,
        qa_corpus_path=None,
        state_path=None,
    )


def _refresh_route(
    config: AdapterFamilyMatrixConfig,
    *,
    route: str,
    score_name: str,
    scores_path: Path,
    claims_path: Path | None,
    qa_corpus_path: Path | None,
    state_path: Path | None,
) -> dict[str, Any]:
    route_dir = config.output_dir / route
    verifier_report_path = route_dir / "verifier-report.json"
    route_report_path = route_dir / "route-comparison.json"
    promotion_report_path = route_dir / "promotion.json"
    workflow = refresh_verifier_route_artifacts(
        VerifierRouteArtifactRefreshConfig(
            score_dumps=((score_name, scores_path),),
            verifier_report_path=verifier_report_path,
            claims_path=claims_path,
            qa_corpus_path=qa_corpus_path,
            state_path=state_path,
            signal=config.signal,
            alphas=(config.alpha,),
            repeats=1,
            verifier_min_overlap=config.verifier_min_overlap,
            retriever_min_overlap=config.retriever_min_overlap,
            retrieval_limit=config.retrieval_limit,
            enable_triple_evidence=route == TRIPLE_EVIDENCE_ROUTE,
            triple_min_slot_coverage=config.triple_min_slot_coverage,
            promotion_report_path=promotion_report_path,
            route_report_path=route_report_path,
            promotion_gate_routes=(route,),
            promotion_gate_min_selected=config.n_records,
            min_decision_accuracy=config.min_decision_accuracy,
            max_false_supported_rate=config.max_false_supported_rate,
            min_false_refuted_rate=config.min_false_refuted_rate,
            max_mean_duration_seconds=config.max_mean_duration_seconds,
            max_p99_duration_seconds=config.max_p99_duration_seconds,
            max_max_duration_seconds=config.max_max_duration_seconds,
            max_mean_attempted_route_count=config.max_mean_attempted_route_count,
            max_retrieval_use_rate=config.max_retrieval_use_rate,
            compact_json=config.compact_json,
        )
    )
    workflow_path = route_dir / "refresh-workflow.json"
    workflow_path.write_text(
        _json_text(workflow, compact=config.compact_json, sort_keys=True),
        encoding="utf-8",
    )
    verifier_summary = workflow["verifier_report_summary"]
    summary_route = verifier_summary["runs"][0]["routes"][route]
    transition_summary = verifier_summary.get("transition_verifier", {})
    decision = workflow["promotion"]["decision"]
    family = {
        "route": route,
        "status": decision["status"],
        "recommended_route": decision.get("recommended_route"),
        "selected": summary_route.get("selected"),
        "decision_accuracy": summary_route.get("decision_accuracy"),
        "false_supported_rate": summary_route.get("false_supported_rate"),
        "false_refuted_rate": summary_route.get("false_refuted_rate"),
        "mean_duration_seconds": summary_route.get("mean_duration_seconds"),
        "p99_duration_seconds": summary_route.get("p99_duration_seconds"),
        "max_duration_seconds": summary_route.get("max_duration_seconds"),
        "mean_attempted_route_count": summary_route.get("mean_attempted_route_count"),
        "retrieval_use_rate": summary_route.get("retrieval_use_rate"),
        "verifier_report_path": str(verifier_report_path),
        "route_report_path": str(route_report_path),
        "promotion_report_path": str(promotion_report_path),
        "refresh_workflow_path": str(workflow_path),
    }
    if route == "state_transition" and isinstance(transition_summary, Mapping):
        family["world_model_adapter"] = transition_summary.get("world_model_adapter")
        family["world_model_rule_count"] = transition_summary.get("world_model_rule_count")
    return family


def _write_structured_qa_fixture(
    *,
    scores_path: Path,
    qa_corpus_path: Path,
    n_records: int,
    signal: str,
    compact: bool,
) -> None:
    n_pairs = n_records // 2
    labels = [0] * n_pairs + [1] * n_pairs
    scores = [round(0.20 + 0.01 * idx, 6) for idx in range(n_pairs)] + [
        round(0.70 + 0.01 * idx, 6) for idx in range(n_pairs)
    ]
    true_statements = [
        {
            "question": f"Order policy question {idx + 1}?",
            "answer": f"Correct answer {idx + 1}",
            "text": f"Order policy question {idx + 1}? Correct answer {idx + 1}",
        }
        for idx in range(n_pairs)
    ]
    false_statements = [
        {
            "question": f"Order policy question {idx + 1}?",
            "answer": f"Wrong answer {idx + 1}",
            "text": f"Order policy question {idx + 1}? Wrong answer {idx + 1}",
        }
        for idx in range(n_pairs)
    ]
    scores_payload = {
        "schema_version": 1,
        "config": {
            "model": "synthetic-structured-qa",
            "layer": -1,
            "fixture_type": "structured_qa_route_family",
            "signal": signal,
            "n_records": n_records,
        },
        "labels": labels,
        "scores": {signal: scores},
        "statements": true_statements + false_statements,
    }
    qa_payload = {
        "schema_version": 1,
        "documents": [
            {
                "question": f"Order policy question {idx + 1}?",
                "answer": f"Correct answer {idx + 1}",
                "source": f"synthetic-qa:{idx + 1}",
            }
            for idx in range(n_pairs)
        ],
    }
    _write_json(scores_path, scores_payload, compact=compact)
    _write_json(qa_corpus_path, qa_payload, compact=compact)


def _write_retrieval_fixture_inputs(
    *,
    scores_path: Path,
    corpus_path: Path,
    claims_path: Path,
    n_records: int,
    signal: str,
    compact: bool,
    retriever_min_overlap: float,
    retrieval_limit: int,
    structured_qa: bool,
) -> None:
    n_pairs = n_records // 2
    labels = [0] * n_pairs + [1] * n_pairs
    scores = [round(0.18 + 0.01 * idx, 6) for idx in range(n_pairs)] + [
        round(0.76 + 0.01 * idx, 6) for idx in range(n_pairs)
    ]
    true_statements = [
        {
            "claim_id": f"retrieval_true_{idx + 1}",
            "question": f"What shipping option is order R{idx + 1} approved for?",
            "answer": f"Order R{idx + 1} is approved for expedited shipping.",
            "text": f"Order R{idx + 1} is approved for expedited shipping.",
        }
        for idx in range(n_pairs)
    ]
    false_statements = [
        {
            "claim_id": f"retrieval_false_{idx + 1}",
            "question": f"What shipping option is order R{idx + 1} approved for?",
            "answer": f"Order R{idx + 1} is approved for same-day drone shipping.",
            "text": f"Order R{idx + 1} is approved for same-day drone shipping.",
        }
        for idx in range(n_pairs)
    ]
    scores_payload = {
        "schema_version": 1,
        "config": {
            "model": "synthetic-retrieval-structured-qa" if structured_qa else "synthetic-retrieval-groundedness",
            "layer": -1,
            "fixture_type": (
                "retrieval_structured_qa_route_family"
                if structured_qa
                else "retrieval_groundedness_route_family"
            ),
            "signal": signal,
            "n_records": n_records,
        },
        "labels": labels,
        "scores": {signal: scores},
        "statements": true_statements + false_statements,
    }
    support_documents = [
        {
            "text": f"Order R{idx + 1} is approved for expedited shipping.",
            "source": f"shipping-policy:R{idx + 1}:support",
            **(
                {
                    "question": f"What shipping option is order R{idx + 1} approved for?",
                    "answer": f"Order R{idx + 1} is approved for expedited shipping.",
                }
                if structured_qa
                else {}
            ),
        }
        for idx in range(n_pairs)
    ]
    refutation_documents = [
        {
            "text": f"Order R{idx + 1} is not approved for same-day drone shipping.",
            "source": f"shipping-policy:R{idx + 1}:refutation",
        }
        for idx in range(n_pairs)
    ]
    corpus_payload = {
        "schema_version": 1,
        "documents": support_documents if structured_qa else support_documents + refutation_documents,
    }
    claims_payload = build_evidence_fixture(
        scores_payload,
        corpus_payload["documents"],
        retriever_min_overlap=retriever_min_overlap,
        retrieval_limit=retrieval_limit,
        query_field="question" if structured_qa else "answer",
    )
    _write_json(scores_path, scores_payload, compact=compact)
    _write_json(corpus_path, corpus_payload, compact=compact)
    _write_json(claims_path, claims_payload, compact=compact)


def _write_triple_evidence_fixture_inputs(
    *,
    scores_path: Path,
    claims_path: Path,
    n_records: int,
    signal: str,
    compact: bool,
) -> None:
    n_pairs = n_records // 2
    labels = [0] * n_pairs + [1] * n_pairs
    scores = [round(0.16 + 0.01 * idx, 6) for idx in range(n_pairs)] + [
        round(0.78 + 0.01 * idx, 6) for idx in range(n_pairs)
    ]
    true_records = []
    false_records = []
    true_statements = []
    false_statements = []
    for idx in range(n_pairs):
        claim_id = f"triple_true_{idx + 1}"
        invoice_count = 10 + idx
        text = f"Order T{idx + 1} has {invoice_count} approved invoices."
        true_statements.append({"claim_id": claim_id, "text": text})
        true_records.append({
            "claim": text,
            "claim_id": claim_id,
            "claim_metadata": {"features": {"has_number": True}},
            "initial_evidence": [text],
        })
    for idx in range(n_pairs):
        claim_id = f"triple_false_{idx + 1}"
        invoice_count = 20 + idx
        text = f"Order F{idx + 1} has {invoice_count} approved invoices."
        false_statements.append({"claim_id": claim_id, "text": text})
        false_records.append({
            "claim": text,
            "claim_id": claim_id,
            "claim_metadata": {"features": {"has_number": True}},
            "initial_evidence": [f"Order F{idx + 1} has approved invoices."],
        })
    scores_payload = {
        "schema_version": 1,
        "config": {
            "model": "synthetic-triple-evidence",
            "layer": -1,
            "fixture_type": "triple_evidence_route_family",
            "signal": signal,
            "n_records": n_records,
        },
        "labels": labels,
        "scores": {signal: scores},
        "statements": true_statements + false_statements,
    }
    claims_payload = {
        "schema_version": 1,
        "fixture_type": "triple_evidence_route_family",
        "records": true_records + false_records,
    }
    _write_json(scores_path, scores_payload, compact=compact)
    _write_json(claims_path, claims_payload, compact=compact)


def _write_json(path: Path, payload: Mapping[str, Any], *, compact: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json_text(payload, compact=compact, sort_keys=True), encoding="utf-8")


def _json_text(payload: Mapping[str, Any], *, compact: bool, sort_keys: bool) -> str:
    if compact:
        return json.dumps(payload, sort_keys=sort_keys, separators=(",", ":")) + "\n"
    return json.dumps(payload, indent=2, sort_keys=sort_keys) + "\n"


def _config_from_args(args: argparse.Namespace) -> AdapterFamilyMatrixConfig:
    return AdapterFamilyMatrixConfig(
        output_dir=Path(args.output_dir),
        matrix_report_path=None if args.json is None else Path(args.json),
        alpha=args.alpha,
        n_records=args.n_records,
        signal=args.signal,
        compact_json=bool(args.compact_json),
        min_decision_accuracy=args.min_decision_accuracy,
        max_false_supported_rate=args.max_false_supported_rate,
        min_false_refuted_rate=args.min_false_refuted_rate,
        max_mean_duration_seconds=args.max_mean_duration_seconds,
        max_p99_duration_seconds=args.max_p99_duration_seconds,
        max_max_duration_seconds=args.max_max_duration_seconds,
        max_mean_attempted_route_count=args.max_mean_attempted_route_count,
        max_retrieval_use_rate=args.max_retrieval_use_rate,
        include_retrieval=bool(args.include_retrieval),
        include_retrieval_structured_qa=bool(args.include_retrieval_structured_qa),
        include_triple_evidence=bool(args.include_triple_evidence),
        verifier_min_overlap=args.verifier_min_overlap,
        retriever_min_overlap=args.retriever_min_overlap,
        retrieval_limit=args.retrieval_limit,
        triple_min_slot_coverage=args.triple_min_slot_coverage,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run from parsed CLI arguments."""
    report = run_adapter_family_matrix(_config_from_args(args))
    print(
        "adapter_family_matrix="
        f"{report['promotion_decision']['status']} "
        f"recommended={report['promotion_decision'].get('recommended_route')}"
    )
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a deterministic adapter-family promotion matrix")
    parser.add_argument("--output-dir", required=True, help="directory for generated fixtures and reports")
    parser.add_argument("--json", default=None, help="optional matrix report output path")
    parser.add_argument("--alpha", type=float, default=0.20)
    parser.add_argument("--n-records", type=int, default=8, help="even record count for each route fixture")
    parser.add_argument("--signal", default="truth_proj")
    parser.add_argument("--min-decision-accuracy", type=float, default=1.0)
    parser.add_argument("--max-false-supported-rate", type=float, default=0.0)
    parser.add_argument("--min-false-refuted-rate", type=float, default=1.0)
    parser.add_argument("--max-mean-duration-seconds", type=float, default=1.0)
    parser.add_argument("--max-p99-duration-seconds", type=float, default=1.0)
    parser.add_argument("--max-max-duration-seconds", type=float, default=1.0)
    parser.add_argument("--max-mean-attempted-route-count", type=float, default=1.1)
    parser.add_argument("--max-retrieval-use-rate", type=float, default=0.0)
    parser.add_argument("--include-retrieval", action="store_true",
                        help="include a local retrieval-groundedness route fixture")
    parser.add_argument("--include-retrieval-structured-qa", action="store_true",
                        help="include a local retrieval structured-QA route fixture")
    parser.add_argument("--include-triple-evidence", action="store_true",
                        help="include a strict triple-evidence audit route fixture")
    parser.add_argument("--verifier-min-overlap", type=float, default=0.65)
    parser.add_argument("--retriever-min-overlap", type=float, default=0.6)
    parser.add_argument("--retrieval-limit", type=int, default=1)
    parser.add_argument("--triple-min-slot-coverage", type=float, default=1.0)
    parser.add_argument("--compact-json", action="store_true",
                        help="write minified JSON artifacts for automation")
    parser.add_argument("--fail-on-blocked", action="store_true",
                        help="exit non-zero unless the matrix promotion decision is promote")
    args = parser.parse_args(argv)
    report = run(args)
    if args.fail_on_blocked and report["promotion_decision"]["status"] != "promote":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

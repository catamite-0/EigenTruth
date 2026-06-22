"""Build a deterministic adapter-family promotion matrix.

This no-model workflow creates small local fixtures for three verifier-route
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
from benchmarks.build_transition_fixture import build_order_transition_fixture  # noqa: E402
from benchmarks.compare_verifier_routes import build_route_comparison_report  # noqa: E402
from benchmarks.refresh_verifier_route_artifacts import (  # noqa: E402
    VerifierRouteArtifactRefreshConfig,
    refresh_verifier_route_artifacts,
)

ROUTES = ("structured_qa", "structured_state", "state_transition")


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

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.matrix_report_path is not None:
            object.__setattr__(self, "matrix_report_path", Path(self.matrix_report_path))
        if self.n_records < 2 or self.n_records % 2:
            raise ValueError("n_records must be an even integer >= 2.")
        if not (0.0 < float(self.alpha) < 1.0):
            raise ValueError("alpha must be in (0, 1).")


def run_adapter_family_matrix(config: AdapterFamilyMatrixConfig) -> dict[str, Any]:
    """Run local route-family promotion checks and return a matrix report."""
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    families = [
        _run_structured_qa(config),
        _run_structured_state(config),
        _run_state_transition(config),
    ]
    comparison_path = output_dir / "route-family-comparison.json"
    comparison = build_route_comparison_report(
        tuple((str(item["route"]), Path(item["verifier_report_path"])) for item in families),
        alpha=config.alpha,
        min_selected=1,
        notes=("deterministic adapter-family matrix",),
        gate_routes=ROUTES,
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
        "routes": ROUTES,
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
    fixture = build_order_transition_fixture(n_records=config.n_records, signal=config.signal)
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
    summary_route = workflow["verifier_report_summary"]["runs"][0]["routes"][route]
    decision = workflow["promotion"]["decision"]
    return {
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

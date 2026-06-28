"""Select deployable signal bundles from a fusion ablation matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eigentruth.eval import (  # noqa: E402
    SignalSelectionPolicy,
    select_signals_from_fusion_ablation_matrix,
)
from eigentruth.json_utils import strict_json_dumps  # noqa: E402


def select_fusion_signals_report(
    matrix: Mapping[str, Any],
    *,
    tracked_signal: str = "trajectory_convergence",
    alpha: float = 0.1,
    min_detection_delta: float = 0.0,
    min_auroc_delta: float = 0.0,
    max_false_alarm_delta: float = 0.03,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a JSON payload selecting signal bundles from an ablation matrix."""
    policy = SignalSelectionPolicy(
        tracked_signal=tracked_signal,
        alpha=alpha,
        min_detection_delta=min_detection_delta,
        min_auroc_delta=min_auroc_delta,
        max_false_alarm_delta=max_false_alarm_delta,
    )
    report = select_signals_from_fusion_ablation_matrix(
        matrix,
        policy=policy,
        metadata={} if metadata is None else metadata,
    )
    return report.to_dict()


def run(args: argparse.Namespace) -> dict[str, Any]:
    """CLI entry point with a testable args namespace."""
    matrix_path = Path(args.matrix)
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    payload = select_fusion_signals_report(
        matrix,
        tracked_signal=args.tracked_signal,
        alpha=float(args.alpha),
        min_detection_delta=float(args.min_detection_delta),
        min_auroc_delta=float(args.min_auroc_delta),
        max_false_alarm_delta=float(args.max_false_alarm_delta),
        metadata={"source_matrix_path": str(matrix_path)},
    )
    if args.json is not None:
        output = Path(args.json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(strict_json_dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not args.quiet:
        for decision in payload["decisions"]:
            enabled = "enabled" if decision["tracked_signal_enabled"] else "disabled"
            print(
                f"{decision['run_name']}: "
                f"{decision['selected_candidate']} "
                f"signals={','.join(decision['selected_signals'])} "
                f"{args.tracked_signal}={enabled}"
            )
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Select signal bundles from a fusion ablation matrix")
    parser.add_argument("--matrix", required=True, help="fusion ablation matrix JSON path")
    parser.add_argument("--json", default=None, help="optional output report path")
    parser.add_argument("--tracked-signal", default="trajectory_convergence")
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--min-detection-delta", type=float, default=0.0)
    parser.add_argument("--min-auroc-delta", type=float, default=0.0)
    parser.add_argument("--max-false-alarm-delta", type=float, default=0.03)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()

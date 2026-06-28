"""Build a rank-fusion artifact from a trajectory benchmark report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eigentruth.calibration import (  # noqa: E402
    RankScoreFusionArtifact,
    trajectory_fusion_dataset_from_report,
)
from eigentruth.eval import roc_auc  # noqa: E402
from eigentruth.json_utils import strict_json_dumps  # noqa: E402


def build_trajectory_fusion_artifact(
    trajectory_report: Path,
    *,
    layer: str | int = "best",
    signal_name: str = "trajectory_convergence",
    include_nll_answer: bool = False,
    alpha: float = 0.1,
    method: str = "max_rank",
) -> dict[str, Any]:
    """Return a report plus artifact derived from a trajectory report."""
    source = _load_json_object(trajectory_report)
    dataset = trajectory_fusion_dataset_from_report(
        source,
        layer=layer,
        signal_name=signal_name,
        include_nll_answer=include_nll_answer,
    )
    artifact = dataset.calibrate(alpha=alpha, method=method)
    fused = artifact.score(dataset.scores)
    flags = artifact.flags(dataset.scores)
    labels = torch.as_tensor(dataset.labels, dtype=torch.int64)
    normal = labels == 0
    anomalous = labels == 1
    return {
        "schema_version": 1,
        "workflow": "trajectory_fusion_artifact_build",
        "status": "complete",
        "config": {
            "trajectory_report": str(trajectory_report),
            "layer": layer,
            "signal_name": signal_name,
            "include_nll_answer": bool(include_nll_answer),
            "alpha": float(alpha),
            "method": method,
        },
        "summary": {
            "n_records": len(dataset.labels),
            "n_true": int(normal.sum().item()),
            "n_false": int(anomalous.sum().item()),
            "signal_names": artifact.signal_names(),
            "threshold": artifact.threshold,
            "auroc": roc_auc(fused, labels),
            "false_alarm": _flag_rate(flags, normal),
            "detection": _flag_rate(flags, anomalous),
        },
        "dataset_metadata": dict(dataset.metadata),
        "artifact": artifact.to_dict(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """CLI entry point with a testable args namespace."""
    payload = build_trajectory_fusion_artifact(
        Path(args.trajectory_report),
        layer=args.layer,
        signal_name=args.signal_name,
        include_nll_answer=bool(args.include_nll_answer),
        alpha=float(args.alpha),
        method=str(args.method),
    )
    if args.artifact is not None:
        RankScoreFusionArtifact.from_dict(payload["artifact"]).save_json(args.artifact)
    if args.json is not None:
        Path(args.json).write_text(strict_json_dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not args.quiet:
        summary = payload["summary"]
        print(
            "trajectory_fusion_artifact_ok "
            f"n_records={summary['n_records']} "
            f"signals={','.join(summary['signal_names'])} "
            f"auroc={float(summary['auroc']):.3f}"
        )
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a trajectory score-fusion artifact")
    parser.add_argument("--trajectory-report", required=True, help="trajectory report JSON path")
    parser.add_argument("--json", default=None, help="optional workflow report output path")
    parser.add_argument("--artifact", default=None, help="optional rank-fusion artifact output path")
    parser.add_argument("--layer", default="best", help="best or explicit layer key/integer")
    parser.add_argument("--signal-name", default="trajectory_convergence")
    parser.add_argument("--include-nll-answer", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--method", default="max_rank")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    run(args)


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object.")
    return dict(payload)


def _flag_rate(flags: torch.Tensor, mask: torch.Tensor) -> float:
    if int(mask.sum().item()) == 0:
        return 0.0
    return float(flags[mask].to(torch.float64).mean().item())


if __name__ == "__main__":
    main()

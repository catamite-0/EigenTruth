"""E1 — 共形校准覆盖率验证 / Split-conformal coverage validation.

研究问题 / Research question:
    split conformal 能否把原始分数（马氏距离 / 对比方向投影）变成有保证的报警阈值，
    取代拍脑袋的固定阈值？
    Can split conformal turn raw scores into alarm thresholds with honest coverage,
    replacing hand-picked thresholds?

方法 / Method:
    消费 eval_truthfulqa.py --dump-scores 的逐陈述分数（无需再跑模型）。
    把"真陈述"作为可交换的正常总体，随机对半切成 校准/测试，多次重复取平均：
    - 误报率 / false-alarm rate: 真陈述中 score > threshold(alpha) 的比例，应 <= alpha (+3%)
    - 检出率 / detection rate:   假陈述中 score > threshold(alpha) 的比例（power，仅报告）
    Uses per-statement scores dumped by eval_truthfulqa.py. True statements form the
    exchangeable "normal" population, split 50/50 into calibration/test over multiple
    seeded repeats. Gate: |false-alarm − alpha| within tolerance at every alpha.

判据 (E1 gate): 在 alpha ∈ {0.05, 0.1, 0.2} 上 |经验误报率 − alpha| <= 0.03。

用法 / Usage:
    python benchmarks/eval_conformal.py --scores benchmarks/scores_gpt2_l-8.json --signal maha_last
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.config_utils import planned_artifact_manifest_summary  # noqa: E402
from eigentruth.calibration import (  # noqa: E402
    DEFAULT_SCORE_DIRECTIONS,
    ConformalCalibrator,
    LayerScoreSweepCalibrator,
)
from eigentruth.eval.conformal import directional_conformal_threshold, directional_trigger_rate  # noqa: E402
from eigentruth.eval.metrics import selective_classification_report  # noqa: E402
from eigentruth.eval.score_dump import (  # noqa: E402
    load_score_dump_columns,
    score_dump_cache_summary,
    score_dump_file_metadata,
)
from eigentruth.registry import build_artifact_manifest  # noqa: E402

ALPHAS = (0.05, 0.10, 0.20)
TOLERANCE = 0.03


def _parse_signals(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    signals = tuple(part.strip() for part in value.split(",") if part.strip())
    if not signals:
        raise ValueError("--signals must contain at least one signal name.")
    return signals


def _all_dump_signals(summary: dict) -> tuple[str, ...]:
    return tuple(summary.get("all_signal_names", ()))


def _direction_for(signal: str, override: str | None = None) -> str:
    return override or DEFAULT_SCORE_DIRECTIONS.get(signal, "higher")


def _artifact_paths(args) -> dict[str, str | Path | None]:
    return {
        "input_scores": args.scores,
        "conformal_report": args.json,
        "calibration_artifact": args.save_calibration,
        "sweep_report": args.save_sweep_report,
        "best_calibration_artifact": args.save_best_calibration,
    }


def _add_planned_manifest_fields(args, payload: dict) -> None:
    artifact_manifest = getattr(args, "artifact_manifest", None)
    if artifact_manifest is None:
        return
    output_paths = tuple(
        path
        for name, path in _artifact_paths(args).items()
        if name != "input_scores" and path is not None
    )
    payload.setdefault("paths", {})["artifact_manifest"] = str(artifact_manifest)
    payload["artifact_manifest_summary"] = planned_artifact_manifest_summary(
        _artifact_paths(args),
        assume_file_paths=output_paths,
    )


def _write_artifact_manifest(args, payload: dict) -> dict | None:
    artifact_manifest = getattr(args, "artifact_manifest", None)
    if artifact_manifest is None:
        return None
    manifest_path = Path(artifact_manifest)
    manifest = build_artifact_manifest(
        _artifact_paths(args),
        root=manifest_path.parent,
        metadata={
            "runner": "eval_conformal",
            "verdict": payload.get("verdict"),
            "signal": args.signal,
            "direction": payload.get("config", {}).get("direction"),
            "has_sweep_report": "sweep_report" in payload,
        },
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload.setdefault("paths", {})["artifact_manifest"] = str(manifest_path)
    print(f"\nWrote artifact manifest to {manifest_path}")
    return manifest


def run(args) -> dict:
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

    score_dump_metadata_cache = {}
    score_dump = load_score_dump_columns(args.scores, (args.signal,), cache=score_dump_metadata_cache)
    labels = torch.tensor(score_dump.labels)
    scores = torch.tensor(score_dump.scores[args.signal], dtype=torch.float64)
    dump_config = score_dump.config
    direction = _direction_for(args.signal, args.direction)

    true_scores = scores[labels == 0]   # 正常总体（可交换假设的对象）
    false_scores = scores[labels == 1]  # 希望被报警的对象（仅报告 power）
    n_true, n_false = true_scores.numel(), false_scores.numel()
    print(f"signal={args.signal}  direction={direction}  n_true={n_true}  n_false={n_false}  "
          f"repeats={args.repeats}\n")

    fa_sum = {a: 0.0 for a in ALPHAS}
    det_sum = {a: 0.0 for a in ALPHAS}
    for r in range(args.repeats):
        g = torch.Generator().manual_seed(args.seed + r)
        perm = torch.randperm(n_true, generator=g)
        half = n_true // 2
        calib = true_scores[perm[:half]]
        test_true = true_scores[perm[half:]]
        for a in ALPHAS:
            t = directional_conformal_threshold(calib, a, direction)
            fa_sum[a] += directional_trigger_rate(test_true, t, direction)
            det_sum[a] += directional_trigger_rate(false_scores, t, direction)

    print(f"  {'alpha':>6} {'nominal_cov':>12} {'false_alarm':>12} "
          f"{'emp_cov':>9} {'detect':>8}   gate(|fa-a|<={TOLERANCE})")
    print("  " + "-" * 66)
    results = {}
    all_pass = True
    for a in ALPHAS:
        fa = fa_sum[a] / args.repeats
        det = det_sum[a] / args.repeats
        ok = abs(fa - a) <= TOLERANCE
        all_pass &= ok
        full_threshold = directional_conformal_threshold(true_scores, a, direction)
        selective_report = selective_classification_report(
            scores, labels, full_threshold, direction=direction
        )
        results[str(a)] = {
            "false_alarm": fa,
            "coverage": 1.0 - fa,
            "detection": det,
            "pass": ok,
            "threshold": full_threshold,
            "selective_report": selective_report,
        }
        print(f"  {a:>6.2f} {1 - a:>12.2f} {fa:>12.3f} {1 - fa:>9.3f} "
              f"{det:>8.3f}   {'PASS' if ok else 'FAIL'}")
    print("  " + "-" * 66)
    print(f"\n  E1 verdict: {'ACCEPT' if all_pass else 'REJECT'} "
          f"(coverage tracks nominal within {TOLERANCE} at all alphas)"
          if all_pass else
          f"\n  E1 verdict: REJECT (coverage deviates more than {TOLERANCE})")

    score_dump_metadata = score_dump_file_metadata(args.scores, cache=score_dump_metadata_cache)
    score_dump_metadata.update({
        "summary": dict(score_dump.summary),
        "source_format": score_dump.source_format,
    })
    payload = {"config": {"scores": args.scores, "signal": args.signal,
                          "score_dump": score_dump_metadata,
                          "direction": direction, "repeats": args.repeats, "seed": args.seed,
                          "n_true": n_true, "n_false": n_false},
               "results": results, "verdict": "ACCEPT" if all_pass else "REJECT"}

    if args.save_calibration:
        artifact = ConformalCalibrator(alpha=args.artifact_alpha).calibrate(
            model_id=args.model_id or dump_config.get("model", "unknown"),
            model_revision=args.model_revision,
            target_layer=args.target_layer if args.target_layer is not None else int(dump_config.get("layer", 0)),
            calibration_scores={args.signal: true_scores},
            directions={args.signal: direction},
            calibration_dataset_metadata={
                "scores": args.scores,
                "signal": args.signal,
                "n_true": n_true,
                "source": "eval_conformal.py",
            },
            created_at=args.created_at,
            commit_sha=args.commit_sha,
        )
        artifact.save_json(args.save_calibration)
        print(f"\nWrote calibration artifact to {args.save_calibration}")

    if args.save_sweep_report or args.save_best_calibration:
        selected_signals = _parse_signals(args.signals) or _all_dump_signals(dict(score_dump.summary))
        direction_override = None if args.direction is None else {
            signal: args.direction for signal in selected_signals
        }
        report = LayerScoreSweepCalibrator(
            alpha=args.artifact_alpha,
            best_by=args.best_by,
        ).calibrate_from_file(
            args.scores,
            signals=selected_signals,
            directions=direction_override,
            model_id=args.model_id or dump_config.get("model", "unknown"),
            model_revision=args.model_revision,
            created_at=args.created_at,
            commit_sha=args.commit_sha,
            metadata={"source": "eval_conformal.py", "config": dump_config},
            cache=score_dump_metadata_cache,
        )
        payload["sweep_report"] = report.to_dict()
        if args.save_sweep_report:
            report.save_json(args.save_sweep_report)
            print(f"\nWrote sweep report to {args.save_sweep_report}")
        if args.save_best_calibration:
            artifact = report.best_artifact(
                calibration_dataset_metadata={"scores": args.scores, "source": "eval_conformal.py"}
            )
            artifact.save_json(args.save_best_calibration)
            print(f"\nWrote best calibration artifact to {args.save_best_calibration}")

    payload["score_dump_cache"] = score_dump_cache_summary(score_dump_metadata_cache)
    _add_planned_manifest_fields(args, payload)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote {args.json}")
    manifest = _write_artifact_manifest(args, payload)
    if manifest is not None:
        payload["artifact_manifest_summary"] = manifest["summary"]
    return payload


def main():
    p = argparse.ArgumentParser(description="E1: split-conformal coverage validation")
    p.add_argument("--scores", required=True,
                   help="scores JSON from eval_truthfulqa.py --dump-scores")
    p.add_argument("--signal", default="maha_last",
                   help="which signal to calibrate (maha_last / truth_proj / ...)")
    p.add_argument("--signals", default=None,
                   help="optional comma-list of signals for layer/score sweep reports")
    p.add_argument("--repeats", type=int, default=20, help="number of seeded 50/50 splits")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", default=None, help="optional path for structured results")
    p.add_argument("--save-calibration", default=None,
                   help="optional path to write a CalibrationArtifact JSON for the selected signal")
    p.add_argument("--save-sweep-report", default=None,
                   help="optional path to write a LayerScoreSweepReport JSON")
    p.add_argument("--save-best-calibration", default=None,
                   help="optional path to write the best CalibrationArtifact from the sweep report")
    p.add_argument("--artifact-manifest", default=None,
                   help="optional path to write an artifact manifest for inputs and generated outputs")
    p.add_argument("--best-by", choices=("auroc", "detection"), default="auroc",
                   help="metric used to choose the best layer/score calibration artifact")
    p.add_argument("--artifact-alpha", type=float, default=0.10,
                   help="alpha used for --save-calibration artifact threshold")
    p.add_argument("--direction", choices=("higher", "lower"), default=None,
                   help="optional override for whether higher or lower signal values are more anomalous")
    p.add_argument("--model-id", default=None, help="override model_id stored in the artifact")
    p.add_argument("--model-revision", default=None, help="optional model revision stored in the artifact")
    p.add_argument("--target-layer", type=int, default=None, help="override target layer stored in the artifact")
    p.add_argument("--created-at", default=None, help="optional artifact timestamp")
    p.add_argument("--commit-sha", default=None, help="optional repository commit SHA")
    run(p.parse_args())


if __name__ == "__main__":
    main()

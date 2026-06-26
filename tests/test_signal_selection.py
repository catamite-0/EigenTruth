import json
from types import SimpleNamespace

import pytest

from eigentruth.eval import (
    SignalSelectionPolicy,
    SignalSelectionReport,
    select_signals_from_fusion_ablation_matrix,
)


def _candidate(
    *,
    signals: list[str],
    method: str,
    auroc: float,
    detection: float,
    false_alarm: float,
) -> dict:
    return {
        "name": "+".join(signals),
        "signals": signals,
        "method": method,
        "direction": "higher",
        "auroc": auroc,
        "alphas": {
            "0.1": {
                "alpha": 0.1,
                "false_alarm": false_alarm,
                "coverage": 1.0 - false_alarm,
                "detection": detection,
                "pass": True,
                "repeats": 3,
            }
        },
    }


def _matrix() -> dict:
    return {
        "schema_version": 1,
        "workflow": "fusion_ablation_matrix",
        "status": "complete",
        "best_alpha": 0.1,
        "runs": [
            {
                "name": "gpt2",
                "directions": {
                    "truth_proj": "higher",
                    "subspace_resid": "higher",
                    "trajectory_convergence": "higher",
                },
                "candidate_results": {
                    "geometry:mean_rank": _candidate(
                        signals=["truth_proj", "subspace_resid"],
                        method="mean_rank",
                        auroc=0.69,
                        detection=0.22,
                        false_alarm=0.03,
                    ),
                    "geometry_trajectory:mean_rank": _candidate(
                        signals=["truth_proj", "subspace_resid", "trajectory_convergence"],
                        method="mean_rank",
                        auroc=0.70,
                        detection=0.23,
                        false_alarm=0.05,
                    ),
                    "trajectory": _candidate(
                        signals=["trajectory_convergence"],
                        method="native",
                        auroc=0.60,
                        detection=0.12,
                        false_alarm=0.04,
                    ),
                },
            },
            {
                "name": "smollm2",
                "directions": {
                    "truth_proj": "higher",
                    "subspace_resid": "higher",
                    "trajectory_convergence": "lower",
                },
                "candidate_results": {
                    "geometry:mean_rank": _candidate(
                        signals=["truth_proj", "subspace_resid"],
                        method="mean_rank",
                        auroc=0.69,
                        detection=0.22,
                        false_alarm=0.03,
                    ),
                    "geometry_trajectory:mean_rank": _candidate(
                        signals=["truth_proj", "subspace_resid", "trajectory_convergence"],
                        method="mean_rank",
                        auroc=0.67,
                        detection=0.14,
                        false_alarm=0.04,
                    ),
                },
            },
        ],
    }


def test_select_signals_from_ablation_matrix_conditionally_enables_tracked_signal(tmp_path):
    report = select_signals_from_fusion_ablation_matrix(_matrix())
    by_run = report.selected_by_run()

    assert by_run["gpt2"].tracked_signal_enabled is True
    assert by_run["gpt2"].selected_candidate == "geometry_trajectory:mean_rank"
    assert by_run["gpt2"].directions["trajectory_convergence"] == "higher"
    assert by_run["gpt2"].metric_deltas["detection"] == pytest.approx(0.01)
    assert by_run["smollm2"].tracked_signal_enabled is False
    assert by_run["smollm2"].selected_candidate == "geometry:mean_rank"
    assert "trajectory_convergence" not in by_run["smollm2"].selected_signals

    path = tmp_path / "selection.json"
    report.save_json(path)
    loaded = SignalSelectionReport.load_json(path)

    assert loaded == report


def test_signal_selection_policy_can_fail_closed_on_false_alarm_delta():
    policy = SignalSelectionPolicy(max_false_alarm_delta=0.01)

    report = select_signals_from_fusion_ablation_matrix(_matrix(), policy=policy)
    decision = report.selected_by_run()["gpt2"]

    assert decision.tracked_signal_enabled is False
    assert decision.selected_candidate == "geometry:mean_rank"
    assert decision.policy_checks["false_alarm_delta_pass"] is False


def test_signal_selection_policy_from_dict_rejects_bool_numeric_fields():
    with pytest.raises(ValueError, match="alpha"):
        SignalSelectionPolicy.from_dict({"alpha": True})
    with pytest.raises(ValueError, match="max_false_alarm_delta"):
        SignalSelectionPolicy.from_dict({"max_false_alarm_delta": False})


def test_select_fusion_signals_cli_writes_report(tmp_path):
    module = __import__("benchmarks.select_fusion_signals_from_ablation", fromlist=["run"])
    matrix_path = tmp_path / "matrix.json"
    report_path = tmp_path / "report.json"
    matrix_path.write_text(json.dumps(_matrix()), encoding="utf-8")

    payload = module.run(SimpleNamespace(
        matrix=str(matrix_path),
        json=str(report_path),
        tracked_signal="trajectory_convergence",
        alpha=0.1,
        min_detection_delta=0.0,
        min_auroc_delta=0.0,
        max_false_alarm_delta=0.03,
        quiet=True,
    ))
    saved = json.loads(report_path.read_text(encoding="utf-8"))

    assert payload == saved
    assert saved["workflow"] == "fusion_signal_selection"
    assert saved["decisions"][0]["tracked_signal_enabled"] is True
    assert saved["decisions"][1]["tracked_signal_enabled"] is False

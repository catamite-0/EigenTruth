"""EigenTruth eval — 评测指标与基准工具 / Evaluation metrics and benchmark utilities.

本子模块只包含与模型无关的纯函数（评分、AUROC、离散度），便于在无网络、
无模型权重的情况下单元测试。重型基准脚本见仓库根目录的 `benchmarks/`。
This submodule holds only model-free pure functions (scoring, AUROC, dispersion)
so they are unit-testable without network access or model weights. The heavier
benchmark runners live in `benchmarks/` at the repository root.
"""

from __future__ import annotations

from eigentruth.eval.conformal import (
    conformal_pvalues,
    conformal_threshold,
    directional_conformal_threshold,
    directional_trigger_rate,
)
from eigentruth.eval.metrics import (
    binomial_confidence_interval,
    euclidean_dispersion,
    roc_auc,
    selective_classification_report,
)
from eigentruth.eval.score_dump import ScoreDump, load_score_dump, score_dump_file_metadata

__all__ = [
    "roc_auc",
    "euclidean_dispersion",
    "binomial_confidence_interval",
    "selective_classification_report",
    "conformal_pvalues",
    "conformal_threshold",
    "directional_conformal_threshold",
    "directional_trigger_rate",
    "ScoreDump",
    "load_score_dump",
    "score_dump_file_metadata",
]

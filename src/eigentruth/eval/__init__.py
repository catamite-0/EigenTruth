"""EigenTruth eval — 评测指标与基准工具 / Evaluation metrics and benchmark utilities.

本子模块只包含与模型无关的纯函数（评分、AUROC、离散度），便于在无网络、
无模型权重的情况下单元测试。重型基准脚本见仓库根目录的 `benchmarks/`。
This submodule holds only model-free pure functions (scoring, AUROC, dispersion)
so they are unit-testable without network access or model weights. The heavier
benchmark runners live in `benchmarks/` at the repository root.
"""

from __future__ import annotations

from eigentruth.eval.conformal import (
    ABSTENTION_COMPARISON_METRICS,
    AdaptiveScoreTransform,
    ConformalAbstentionComparisonCandidate,
    ConformalAbstentionComparisonReport,
    ConformalAbstentionDecision,
    ConformalAbstentionReleaseGate,
    ConformalAbstentionReleaseGateResult,
    ConformalAbstentionReport,
    adaptive_anomaly_scores,
    conformal_abstention_comparison_report,
    conformal_abstention_release_gate,
    conformal_abstention_report,
    conformal_pvalues,
    conformal_threshold,
    directional_conformal_threshold,
    directional_conformal_thresholds,
    directional_trigger_rate,
    evaluate_conformal_abstention,
)
from eigentruth.eval.intrinsic_dimension import (
    IntrinsicDimensionReport,
    intrinsic_dimension_peak_layer,
    intrinsic_dimension_profile,
    twonn_intrinsic_dimension,
)
from eigentruth.eval.metrics import (
    binomial_confidence_interval,
    confidence_error_report,
    euclidean_dispersion,
    roc_auc,
    selective_classification_report,
)
from eigentruth.eval.score_dump import (
    ScoreDump,
    ScoreDumpColumns,
    ScoreDumpIdentity,
    ScoreDumpJsonlManifest,
    ScoreDumpLayerScores,
    ScoreDumpRecord,
    ScoreDumpStatementScores,
    iter_score_dump_jsonl_records,
    load_score_dump,
    load_score_dump_columns,
    load_score_dump_columns_with_extras,
    load_score_dump_layer_scores,
    load_score_dump_statement_scores,
    score_dump_cache_summary,
    score_dump_file_metadata,
    score_dump_identity,
    write_score_dump_jsonl,
    write_score_dump_jsonl_mapping,
)
from eigentruth.eval.score_fusion import (
    RANK_SCORE_FUSION_METHODS,
    combine_rank_anomaly_scores,
    directional_rank_anomaly_scores,
    native_anomaly_scores,
)

__all__ = [
    "roc_auc",
    "euclidean_dispersion",
    "binomial_confidence_interval",
    "selective_classification_report",
    "confidence_error_report",
    "IntrinsicDimensionReport",
    "twonn_intrinsic_dimension",
    "intrinsic_dimension_profile",
    "intrinsic_dimension_peak_layer",
    "ABSTENTION_COMPARISON_METRICS",
    "AdaptiveScoreTransform",
    "ConformalAbstentionComparisonCandidate",
    "ConformalAbstentionComparisonReport",
    "ConformalAbstentionReleaseGate",
    "ConformalAbstentionReleaseGateResult",
    "evaluate_conformal_abstention",
    "conformal_abstention_report",
    "conformal_abstention_comparison_report",
    "conformal_abstention_release_gate",
    "ConformalAbstentionReport",
    "ConformalAbstentionDecision",
    "adaptive_anomaly_scores",
    "conformal_pvalues",
    "conformal_threshold",
    "directional_conformal_threshold",
    "directional_conformal_thresholds",
    "directional_trigger_rate",
    "RANK_SCORE_FUSION_METHODS",
    "native_anomaly_scores",
    "directional_rank_anomaly_scores",
    "combine_rank_anomaly_scores",
    "ScoreDump",
    "ScoreDumpColumns",
    "ScoreDumpIdentity",
    "ScoreDumpLayerScores",
    "ScoreDumpJsonlManifest",
    "ScoreDumpRecord",
    "ScoreDumpStatementScores",
    "iter_score_dump_jsonl_records",
    "load_score_dump",
    "load_score_dump_columns",
    "load_score_dump_columns_with_extras",
    "load_score_dump_layer_scores",
    "load_score_dump_statement_scores",
    "score_dump_cache_summary",
    "score_dump_file_metadata",
    "score_dump_identity",
    "write_score_dump_jsonl",
    "write_score_dump_jsonl_mapping",
]

"""EigenTruth — 基于几何动力学与表征工程的大模型表征诊断研究工具库。

Usage::

    from eigentruth.models.wrapper import EigenTruthWrapper

    monitor = EigenTruthWrapper(base_model, target_layer_idx=-10)
    monitor.warmup(fact_dataset, tokenizer)
    outputs = monitor.generate(**inputs, max_new_tokens=100)
"""

__version__ = "0.1.0"

from eigentruth.core.inside import (
    cluster_assignment_entropy,
    embedding_semantic_entropy,
    internal_eigenscore,
    lexical_semantic_energy,
    lexical_semantic_entropy,
    semantic_energy_score,
    spectral_effective_rank,
)
from eigentruth.core.math_engine import (
    COVARIANCE_MODES,
    TruthManifold,
    hyperbolic_semantic_entropy,
    mahalanobis_distance,
    poincare_map,
    sherman_morrison_update,
)
from eigentruth.core.subspace import TruthSubspace
from eigentruth.intervention.hooks import TruthProbe
from eigentruth.models.wrapper import EigenTruthWrapper

__all__ = [
    "EigenTruthWrapper",
    "TruthProbe",
    "TruthManifold",
    "TruthSubspace",
    "COVARIANCE_MODES",
    "cluster_assignment_entropy",
    "embedding_semantic_entropy",
    "internal_eigenscore",
    "lexical_semantic_energy",
    "lexical_semantic_entropy",
    "semantic_energy_score",
    "spectral_effective_rank",
    "mahalanobis_distance",
    "poincare_map",
    "hyperbolic_semantic_entropy",
    "sherman_morrison_update",
    "__version__",
]

"""EigenTruth 包级别冒烟测试。"""

from eigentruth import (
    CovarianceSpectrum,
    TrajectoryMonitor,
    __version__,
    cluster_assignment_entropy,
    covariance_spectrum,
    embedding_semantic_entropy,
    internal_eigenscore,
    lexical_semantic_entropy,
    spectral_effective_rank,
    trajectory_convergence_metrics,
)


def test_version():
    assert __version__ == "0.1.0"


def test_top_level_inside_exports():
    assert callable(cluster_assignment_entropy)
    assert callable(covariance_spectrum)
    assert callable(CovarianceSpectrum)
    assert callable(embedding_semantic_entropy)
    assert callable(internal_eigenscore)
    assert callable(lexical_semantic_entropy)
    assert callable(spectral_effective_rank)
    assert callable(TrajectoryMonitor)
    assert callable(trajectory_convergence_metrics)

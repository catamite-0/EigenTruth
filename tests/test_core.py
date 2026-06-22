"""EigenTruth 包级别冒烟测试。"""

from eigentruth import __version__, internal_eigenscore, lexical_semantic_entropy, spectral_effective_rank


def test_version():
    assert __version__ == "0.1.0"


def test_top_level_inside_exports():
    assert callable(internal_eigenscore)
    assert callable(lexical_semantic_entropy)
    assert callable(spectral_effective_rank)

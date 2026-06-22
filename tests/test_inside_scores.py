"""Tests for internal-state spectral uncertainty scores."""

import pytest
import torch

from eigentruth.core import internal_eigenscore, spectral_effective_rank


def test_internal_eigenscore_increases_for_diverse_embeddings():
    repeated = torch.ones(4, 3)
    diverse = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ])

    assert internal_eigenscore(diverse) > internal_eigenscore(repeated)


def test_internal_eigenscore_single_embedding_is_zero():
    score = internal_eigenscore(torch.tensor([[1.0, 2.0, 3.0]]))

    assert score.item() == pytest.approx(0.0)


def test_internal_eigenscore_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="shape"):
        internal_eigenscore(torch.ones(3))
    with pytest.raises(ValueError, match="alpha"):
        internal_eigenscore(torch.ones(2, 3), alpha=0.0)
    with pytest.raises(ValueError, match="finite"):
        internal_eigenscore(torch.tensor([[1.0, float("nan")], [2.0, 3.0]]))


def test_spectral_effective_rank_reports_dimensional_spread():
    line = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    plane = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])

    assert spectral_effective_rank(plane) > spectral_effective_rank(line)

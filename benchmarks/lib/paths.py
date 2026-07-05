"""Path helpers shared by benchmark workflows and smokes."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def ensure_repo_root_on_path() -> Path:
    """Allow benchmark scripts to import local packages when run by file path."""
    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    return REPO_ROOT


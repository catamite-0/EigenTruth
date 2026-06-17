"""Optional integration interfaces for external systems.

Concrete adapters may depend on retrieval stacks, databases, or domain/world
models. Keep those dependencies optional and outside the core install.
"""

from __future__ import annotations

from eigentruth.adapters.world_model import InMemoryWorldModelAdapter, WorldModelAdapter, WorldModelPrediction

__all__ = [
    "InMemoryWorldModelAdapter",
    "WorldModelAdapter",
    "WorldModelPrediction",
]

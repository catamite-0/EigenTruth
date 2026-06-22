"""Optional integration interfaces for external systems.

Concrete adapters may depend on retrieval stacks, databases, or domain/world
models. Keep those dependencies optional and outside the core install.
"""

from __future__ import annotations

from eigentruth.adapters.cache import AdapterCacheStats, CachedRetriever, CachedStateSource, combine_cache_stats
from eigentruth.adapters.calculator import CalculationResult, CalculatorVerifier
from eigentruth.adapters.qa import QuestionAnswerFact, QuestionAnswerVerifier
from eigentruth.adapters.retrieval import (
    InMemoryRetriever,
    RetrievalActionExecutor,
    RetrievalHit,
    RetrievalQuery,
    Retriever,
)
from eigentruth.adapters.state import (
    SQLiteStateQuery,
    SQLiteStateSource,
    StateCheck,
    StateSource,
    StructuredStateVerifier,
    ToolOutputMapping,
    ToolOutputStateSource,
)
from eigentruth.adapters.world_model import (
    InMemoryWorldModelAdapter,
    StateTransitionCheck,
    StateTransitionVerifier,
    WorldModelAdapter,
    WorldModelPrediction,
)

__all__ = [
    "AdapterCacheStats",
    "CalculationResult",
    "CalculatorVerifier",
    "CachedRetriever",
    "CachedStateSource",
    "QuestionAnswerFact",
    "QuestionAnswerVerifier",
    "InMemoryRetriever",
    "RetrievalActionExecutor",
    "RetrievalHit",
    "RetrievalQuery",
    "Retriever",
    "SQLiteStateQuery",
    "SQLiteStateSource",
    "StateCheck",
    "StateSource",
    "StructuredStateVerifier",
    "ToolOutputMapping",
    "ToolOutputStateSource",
    "InMemoryWorldModelAdapter",
    "StateTransitionCheck",
    "StateTransitionVerifier",
    "WorldModelAdapter",
    "WorldModelPrediction",
    "combine_cache_stats",
]

"""Optional integration interfaces for external systems.

Concrete adapters may depend on retrieval stacks, databases, or domain/world
models. Keep those dependencies optional and outside the core install.
"""

from __future__ import annotations

from eigentruth.adapters.cache import AdapterCacheStats, CachedRetriever, CachedStateSource, combine_cache_stats
from eigentruth.adapters.calculator import CalculationResult, CalculatorVerifier
from eigentruth.adapters.facts import StructuredFact, StructuredFactVerifier
from eigentruth.adapters.qa import QuestionAnswerFact, QuestionAnswerVerifier
from eigentruth.adapters.retrieval import (
    InMemoryRetriever,
    RetrievalActionExecutor,
    RetrievalHit,
    RetrievalQuery,
    Retriever,
    SQLiteFTSRetriever,
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
    EnsembleWorldModelAdapter,
    InMemoryWorldModelAdapter,
    RuleBasedWorldModelAdapter,
    StateTransitionCheck,
    StateTransitionVerifier,
    WorldModelAdapter,
    WorldModelPrediction,
    WorldModelRule,
)

__all__ = [
    "AdapterCacheStats",
    "CalculationResult",
    "CalculatorVerifier",
    "CachedRetriever",
    "CachedStateSource",
    "QuestionAnswerFact",
    "QuestionAnswerVerifier",
    "StructuredFact",
    "StructuredFactVerifier",
    "InMemoryRetriever",
    "RetrievalActionExecutor",
    "RetrievalHit",
    "RetrievalQuery",
    "Retriever",
    "SQLiteFTSRetriever",
    "SQLiteStateQuery",
    "SQLiteStateSource",
    "StateCheck",
    "StateSource",
    "StructuredStateVerifier",
    "ToolOutputMapping",
    "ToolOutputStateSource",
    "EnsembleWorldModelAdapter",
    "InMemoryWorldModelAdapter",
    "RuleBasedWorldModelAdapter",
    "StateTransitionCheck",
    "StateTransitionVerifier",
    "WorldModelAdapter",
    "WorldModelPrediction",
    "WorldModelRule",
    "combine_cache_stats",
]

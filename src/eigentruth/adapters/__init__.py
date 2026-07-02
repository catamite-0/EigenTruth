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
    HTTPJSONRetriever,
    InMemoryRetriever,
    ProvenanceFilteredRetriever,
    RetrievalActionExecutor,
    RetrievalHit,
    RetrievalQuery,
    Retriever,
    SQLiteFTSRetriever,
    TripleSlotRetrievalBindingReport,
    TripleSlotRetrievalPlan,
    bind_triple_slot_retrieval_hits,
    plan_triple_slot_retrieval,
    plan_triple_slot_retrieval_queries,
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
    WorldModelReference,
    WorldModelRule,
    WorldModelView,
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
    "HTTPJSONRetriever",
    "InMemoryRetriever",
    "ProvenanceFilteredRetriever",
    "RetrievalActionExecutor",
    "RetrievalHit",
    "RetrievalQuery",
    "Retriever",
    "SQLiteFTSRetriever",
    "TripleSlotRetrievalBindingReport",
    "TripleSlotRetrievalPlan",
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
    "WorldModelReference",
    "WorldModelRule",
    "WorldModelView",
    "bind_triple_slot_retrieval_hits",
    "combine_cache_stats",
    "plan_triple_slot_retrieval",
    "plan_triple_slot_retrieval_queries",
]

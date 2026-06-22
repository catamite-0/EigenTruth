"""Optional integration interfaces for external systems.

Concrete adapters may depend on retrieval stacks, databases, or domain/world
models. Keep those dependencies optional and outside the core install.
"""

from __future__ import annotations

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
)
from eigentruth.adapters.world_model import InMemoryWorldModelAdapter, WorldModelAdapter, WorldModelPrediction

__all__ = [
    "CalculationResult",
    "CalculatorVerifier",
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
    "InMemoryWorldModelAdapter",
    "WorldModelAdapter",
    "WorldModelPrediction",
]

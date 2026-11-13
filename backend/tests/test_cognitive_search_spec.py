"""Tests for the spec-compatible ``cognitive_search`` engine method.

These cover the response contract from the original specification:
``{"status": "success", "answer", "concept_name", "confidence"}``,
``{"status": "not_found", ...}`` and ``{"status": "error", ...}``.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.iact_engine import MultiLingualIACTEngine
from backend.knowledge_base import ConceptStore

VOCAB = [
    "artificial",
    "intelligence",
    "technology",
    "ai",
    "neural",
    "networks",
    "network",
    "show",
    "cognition",
    "cognitive",
    "science",
    "mind",
    "dhivehi",
    "language",
    "deep",
    "learning",
]
INDEX = {word: position for position, word in enumerate(VOCAB)}


def stub_encoder(texts):
    vectors = np.zeros((len(texts), len(VOCAB)), dtype=np.float32)
    for row, text in enumerate(texts):
        for word in text.split():
            position = INDEX.get(word)
            if position is not None:
                vectors[row, position] += 1.0
    return vectors


def make_store() -> ConceptStore:
    store = ConceptStore()
    for record in (
        {
            "concept": "Artificial Intelligence",
            "category": "Technology",
            "keywords": ["ai", "neural", "cognition"],
        },
        {
            "concept": "Neural Networks",
            "category": "Technology",
            "keywords": ["neural", "deep", "learning"],
        },
        {
            "concept": "Cognitive Science",
            "category": "Science",
            "keywords": ["cognition", "mind"],
        },
    ):
        store.upsert(record)
    return store


def make_engine() -> tuple[MultiLingualIACTEngine, ConceptStore]:
    engine = MultiLingualIACTEngine(encoder=stub_encoder)
    store = make_store()
    engine.concept_provider = store.snapshot
    return engine, store


# --------------------------------------------------------------------- contract
def test_default_threshold_is_point_four():
    assert MultiLingualIACTEngine.cognitive_search.__defaults__ == (0.4,)


def test_success_response_shape():
    engine, _ = make_engine()
    result = engine.cognitive_search("what is artificial intelligence")
    assert result["status"] == "success"
    assert result["concept_name"] == "Artificial Intelligence"
    assert set(result) == {"status", "answer", "concept_name", "confidence"}
    assert isinstance(result["answer"], str) and result["answer"]
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["confidence"] >= 0.4


def test_not_found_response_shape():
    engine, _ = make_engine()
    result = engine.cognitive_search("banana smoothie recipe")
    assert result["status"] == "not_found"
    assert set(result) == {"status", "message", "confidence"}
    assert result["message"]
    assert result["confidence"] < 0.4


def test_threshold_override_forces_not_found():
    engine, _ = make_engine()
    result = engine.cognitive_search("artificial intelligence", threshold=0.99)
    assert result["status"] == "not_found"


def test_invalid_threshold_is_rejected():
    engine, _ = make_engine()
    with pytest.raises(ValueError):
        engine.cognitive_search("artificial intelligence", threshold=1.5)


# ------------------------------------------------------------------------ errors
def test_error_without_provider():
    engine = MultiLingualIACTEngine(encoder=stub_encoder)
    result = engine.cognitive_search("anything")
    assert result["status"] == "error"
    assert "provider" in result["message"]


def test_error_with_empty_knowledge_base():
    engine = MultiLingualIACTEngine(encoder=stub_encoder)
    empty_store = ConceptStore()
    engine.concept_provider = empty_store.snapshot
    result = engine.cognitive_search("anything")
    assert result["status"] == "error"
    assert "empty" in result["message"]


def test_error_with_empty_query():
    engine, _ = make_engine()
    result = engine.cognitive_search("   ")
    assert result["status"] == "error"


# -------------------------------------------------------------- index lifecycle
def test_index_rebuilds_after_store_mutation():
    engine, store = make_engine()
    first = engine.cognitive_search("machine learning")
    assert first["status"] == "not_found"

    store.upsert(
        {
            "concept": "Machine Learning",
            "category": "Technology",
            "keywords": ["learning", "ai"],
        }
    )
    second = engine.cognitive_search("machine learning")
    assert second["status"] == "success"
    assert second["concept_name"] == "Machine Learning"


def test_optional_description_is_used_as_answer():
    engine = MultiLingualIACTEngine(encoder=stub_encoder)
    store = ConceptStore()
    store.upsert(
        {
            "concept": "Artificial Intelligence",
            "category": "Technology",
            "keywords": ["ai"],
        }
    )
    engine.concept_provider = store.snapshot

    result = engine.cognitive_search("artificial intelligence")
    assert result["status"] == "success"
    # No description field -> synthesized answer mentions the category.
    assert "Technology" in result["answer"]

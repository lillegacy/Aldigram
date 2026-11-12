"""Tests for the IACT cognitive search & QA module.

The tests use a deterministic bag-of-words stub encoder so they run without
PyTorch or sentence-transformers.  Run from the repository root:

    python -m pytest backend/tests -q
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.cognitive_search import (
    DEFAULT_QA_MIN_SCORE,
    DEFAULT_SEARCH_MIN_SCORE,
    CognitiveSearchEngine,
)
from backend.iact_engine import MultiLingualIACTEngine

VOCAB = [
    "artificial",
    "intelligence",
    "technology",
    "ai",
    "neural",
    "networks",
    "network",
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


class CountingEncoder:
    def __init__(self):
        self.calls = 0

    def __call__(self, texts):
        self.calls += 1
        return stub_encoder(texts)


def make_engine(encoder=None):
    return MultiLingualIACTEngine(encoder=encoder or stub_encoder)


def concepts():
    return [
        {
            "concept": "Artificial Intelligence",
            "category": "Technology",
            "keywords": ["ai", "neural", "cognition"],
            "links": ["Cognitive Science", "Neural Networks"],
        },
        {
            "concept": "Neural Networks",
            "category": "Technology",
            "keywords": ["neural", "deep", "learning"],
            "links": ["Artificial Intelligence"],
        },
        {
            "concept": "Cognitive Science",
            "category": "Science",
            "keywords": ["cognition", "mind"],
            "links": ["Artificial Intelligence"],
        },
        {
            "concept": "Dhivehi Language",
            "category": "Language",
            "keywords": ["dhivehi", "language"],
            "links": [],
        },
    ]


# --------------------------------------------------------------- semantic search
def test_search_ranks_relevant_concept_first():
    search = CognitiveSearchEngine(make_engine())
    result = search.search(
        "What is artificial intelligence?", concepts(), version=1, top_k=3
    )
    assert result["resolved"] is True
    assert result["results"][0]["concept"] == "Artificial Intelligence"
    assert result["confidence"] >= DEFAULT_SEARCH_MIN_SCORE
    assert "links" in result["results"][0]


def test_search_returns_links_and_context():
    search = CognitiveSearchEngine(make_engine())
    result = search.search("neural networks", concepts(), version=1, top_k=2)
    titles = [item["concept"] for item in result["results"]]
    assert "Neural Networks" in titles
    top = result["results"][0]
    assert "Artificial Intelligence" in top["links"]


def test_search_threshold_can_be_overridden():
    search = CognitiveSearchEngine(make_engine())
    result = search.search(
        "artificial intelligence", concepts(), version=1, min_score=0.99
    )
    assert result["resolved"] is False
    assert result["results"]


def test_search_rejects_empty_query_and_empty_network():
    search = CognitiveSearchEngine(make_engine())
    assert "error" in search.search("   ", concepts(), version=1)
    result = search.search("anything", [], version=0)
    assert "error" in result
    assert result["resolved"] is False


def test_multilingual_query_is_normalized():
    search = CognitiveSearchEngine(make_engine())
    result = search.search("ނެޓްވާކް ދައްކާ", concepts(), version=1)
    assert result["normalized"] == "netvaak dhakaa"
    assert isinstance(result["results"], list)


# ---------------------------------------------------------------------- QA
def test_qa_returns_structured_answer():
    search = CognitiveSearchEngine(make_engine())
    result = search.answer("What is artificial intelligence?", concepts(), version=1)
    assert result["resolved"] is True
    assert result["confidence"] >= DEFAULT_QA_MIN_SCORE
    assert "Artificial Intelligence" in result["answer"]
    assert result["evidence"]
    assert len(result["subgraph"]["nodes"]) >= 1
    assert any(edge["source"] == "Artificial Intelligence" for edge in result["subgraph"]["edges"])
    assert result["relationships"]


def test_qa_subgraph_contains_edges_between_matched_concepts():
    search = CognitiveSearchEngine(make_engine())
    result = search.answer(
        "Tell me about artificial intelligence", concepts(), version=1, top_k=3
    )
    node_ids = {node["id"] for node in result["subgraph"]["nodes"]}
    for edge in result["subgraph"]["edges"]:
        assert edge["source"] in node_ids
        assert edge["target"] in node_ids


def test_qa_unresolved_for_unrelated_question():
    search = CognitiveSearchEngine(make_engine())
    result = search.answer("How do I bake sourdough bread?", concepts(), version=1)
    assert result["resolved"] is False
    assert result["confidence"] < DEFAULT_QA_MIN_SCORE
    assert "could not find" in result["answer"]


# ----------------------------------------------------------- index lifecycle
def test_index_rebuilds_only_when_store_version_changes():
    encoder = CountingEncoder()
    search = CognitiveSearchEngine(make_engine(encoder))

    search.search("artificial intelligence", concepts(), version=1)
    calls_after_first = encoder.calls

    search.search("neural networks", concepts(), version=1)
    assert encoder.calls == calls_after_first + 1  # query only, docs cached

    updated = concepts() + [
        {
            "concept": "Machine Learning",
            "category": "Technology",
            "keywords": ["learning", "ai"],
            "links": ["Artificial Intelligence"],
        }
    ]
    search.search("machine learning", updated, version=2)
    assert encoder.calls > calls_after_first + 1  # docs re-encoded
    assert search.document_count == len(updated)

    # Vector cache is used across calls at the same version.
    first_vectors = search._vectors
    search.search("machine learning", updated, version=2)
    assert search._vectors is first_vectors


def test_qa_min_score_is_configurable():
    search = CognitiveSearchEngine(make_engine(), qa_min_score=0.95)
    result = search.answer("What is artificial intelligence?", concepts(), version=1)
    assert result["resolved"] is False
    assert result["threshold"] == pytest.approx(0.95)

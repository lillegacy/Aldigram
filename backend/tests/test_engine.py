"""Tests for the multilingual IACT engine.

Run from the repository root:

    python -m pytest backend/tests -q

These tests use a deterministic stub encoder, so they run without PyTorch or
sentence-transformers installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.iact_engine import DEFAULT_THRESHOLD, HashingEncoder, MultiLingualIACTEngine
from backend.multilingual import has_thaana, normalize_text, transliterate_thaana

VOCAB = {"network": 0, "add": 1, "delete": 2, "concept": 3, "help": 4}


def stub_encoder(texts):
    """Bag-of-words encoder over a tiny vocabulary (tests only)."""
    vectors = np.zeros((len(texts), len(VOCAB)), dtype=np.float32)
    for row, text in enumerate(texts):
        for word in text.split():
            if word in VOCAB:
                vectors[row, VOCAB[word]] += 1.0
    return vectors


def make_engine():
    engine = MultiLingualIACTEngine(threshold=DEFAULT_THRESHOLD, encoder=stub_encoder)
    engine.add_intent("add_concept", ["add a new concept", "create a concept"])
    engine.add_intent("show_network", ["show the network", "display the network map"])
    engine.add_intent("delete_concept", ["delete a concept", "remove a concept"])
    return engine


# --------------------------------------------------------------------- scripts
def test_thaana_is_detected_and_transliterated():
    text = "ނެޓްވާކް ދައްކާ"
    assert has_thaana(text)
    transliterated = transliterate_thaana(text)
    assert not has_thaana(transliterated)
    assert transliterated.strip()


def test_thaana_and_latin_normalize_identically():
    # The Latin sample uses the exact transliteration of the Thaana sample, so
    # both variants share one point in vector space.
    assert normalize_text("ނެޓްވާކް ދައްކާ") == normalize_text("netvaak dhakaa")


def test_normalize_mixed_scripts():
    assert normalize_text("  SHOW the  NETWORK!  ") == "show the network"
    mixed = normalize_text("Add އައު concept")
    assert "add" in mixed and "concept" in mixed
    assert not has_thaana(mixed)


def test_empty_and_punctuation_only_input():
    assert normalize_text("") == ""
    assert normalize_text("!!! ... ???") == ""


# ------------------------------------------------------------- vector math
def test_cosine_similarity_basics():
    assert MultiLingualIACTEngine.cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert MultiLingualIACTEngine.cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert MultiLingualIACTEngine.cosine_similarity([0, 0], [1, 0]) == 0.0
    with pytest.raises(ValueError):
        MultiLingualIACTEngine.cosine_similarity([1, 0], [1, 0, 0])


# ------------------------------------------------------------- engine behavior
def test_intent_is_resolved_above_threshold():
    engine = make_engine()
    result = engine.process_command("show the network")
    assert result["resolved"] is True
    assert result["intent"] == "show_network"
    assert result["confidence"] >= DEFAULT_THRESHOLD


def test_threshold_is_respected_for_unrelated_commands():
    engine = make_engine()
    result = engine.process_command("banana smoothie")
    assert result["resolved"] is False
    assert result["intent"] is None
    assert result["confidence"] < DEFAULT_THRESHOLD


def test_add_intent_deduplicates_and_validates():
    engine = MultiLingualIACTEngine(encoder=stub_encoder)
    assert engine.add_intent("help", ["help me"]) == 1
    assert engine.add_intent("help", ["help me"]) == 0
    with pytest.raises(ValueError):
        engine.add_intent("", ["help"])
    with pytest.raises(ValueError):
        engine.add_intent("help", [])


def test_empty_command_reports_error():
    engine = make_engine()
    result = engine.process_command("   ")
    assert result["resolved"] is False
    assert "error" in result


def test_hashing_fallback_encoder():
    encoder = HashingEncoder(dimensions=64, ngram_range=(2, 3))
    vectors = encoder.encode(["show network", "show network", "delete concept"])
    assert vectors.shape == (3, 64)
    assert np.allclose(vectors[0], vectors[1])
    assert not np.allclose(vectors[0], vectors[2])


def test_repeated_commands_transition_to_cached_load():
    engine = make_engine()

    first = engine.process_command("show the network")
    second = engine.process_command("show the network")
    third = engine.process_command("show the network")

    assert first["optimizer"]["execution_status"] == "learning"
    assert first["optimizer"]["compute_load"] == 1.0
    assert second["optimizer"]["execution_status"] == "partial"
    assert second["optimizer"]["compute_load"] == 0.65
    assert third["optimizer"]["execution_status"] == "cached"
    assert third["optimizer"]["compute_load"] == 0.54
    assert third["optimizer"]["associative_bandwidth"] == 0.46
    assert third["optimizer"]["cache_progress"]["cached"] is True

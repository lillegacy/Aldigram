"""Tests for JSON persistence (iact_data.json) on the concept store."""

from __future__ import annotations

import json

from backend.knowledge_base import ConceptStore


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "iact_data.json"
    store = ConceptStore(data_file=path)
    store.upsert({"concept": "Artificial Intelligence", "category": "Technology", "keywords": ["ai", "neural"]})
    store.upsert({"concept": "Cognitive Science", "category": "Science", "keywords": ["cognition", "mind"]})

    # Auto-save wrote the file on every mutation.
    assert path.is_file()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert len(payload) == 2
    assert all("vector" not in entry for entry in payload)

    fresh = ConceptStore(data_file=path)
    loaded = fresh.load_data()
    assert loaded == 2
    network, _ = fresh.snapshot()
    titles = {item["concept"] for item in network}
    assert titles == {"Artificial Intelligence", "Cognitive Science"}


def test_autosave_on_delete(tmp_path):
    path = tmp_path / "iact_data.json"
    store = ConceptStore(data_file=path)
    store.upsert({"concept": "Temporary", "category": "General", "keywords": ["tmp"]})
    assert store.delete("Temporary") is True

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == []


def test_load_missing_file_returns_zero(tmp_path):
    store = ConceptStore(data_file=tmp_path / "missing.json")
    assert store.load_data() == 0


def test_no_persistence_without_data_file(tmp_path):
    store = ConceptStore()
    store.upsert({"concept": "Volatile", "category": "General", "keywords": ["mem"]})
    assert list(tmp_path.iterdir()) == []


def test_load_bumps_version_for_search_indexes(tmp_path):
    path = tmp_path / "iact_data.json"
    store = ConceptStore(data_file=path)
    store.upsert({"concept": "Artificial Intelligence", "category": "Technology", "keywords": ["ai"]})

    reloaded = ConceptStore(data_file=path)
    version_before = reloaded.version
    reloaded.load_data()
    assert reloaded.version > version_before


def test_load_accepts_legacy_title_key(tmp_path):
    path = tmp_path / "iact_data.json"
    path.write_text(
        json.dumps([{"title": "Legacy Concept", "category": "General", "keywords": "old, data"}]),
        encoding="utf-8",
    )
    store = ConceptStore(data_file=path)
    assert store.load_data() == 1
    network, _ = store.snapshot()
    assert network[0]["concept"] == "Legacy Concept"
    assert network[0]["keywords"] == ["old", "data"]

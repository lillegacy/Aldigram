"""Tests for the CECANP-0.54 optimiser and IACT-5SCP sensory protocol."""

from __future__ import annotations

import json

import pytest

from backend.cognitive_optimizer import (
    CACHED_COMPUTE_LOAD,
    FULL_COMPUTE_LOAD,
    PARTIAL_COMPUTE_LOAD,
    SENSORY_WINDOW_SECONDS,
    IntegratedIACTSystem,
    SensoryInput,
)
from backend.knowledge_base import ConceptStore


class FakeClock:
    """Deterministic monotonic clock for TTL and override-window tests."""

    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_optimizer(**kwargs) -> IntegratedIACTSystem:
    clock = kwargs.pop("clock", None) or FakeClock()
    return IntegratedIACTSystem(clock=clock, repetition_threshold=3, **kwargs)


# ------------------------------------------------------------------ CECANP-0.54
def test_repetition_threshold_must_be_between_three_and_five():
    with pytest.raises(ValueError):
        IntegratedIACTSystem(repetition_threshold=2)
    with pytest.raises(ValueError):
        IntegratedIACTSystem(repetition_threshold=6)


def test_learning_transitions_to_cached_execution():
    optimizer = make_optimizer()

    first = optimizer.run_optimizer("show_network", "show the network")
    assert first["compute_load"] == FULL_COMPUTE_LOAD
    assert first["associative_bandwidth"] == 0.0
    assert first["execution_status"] == "learning"
    assert first["cache_progress"] == {
        "runs": 1,
        "threshold": 3,
        "remaining": 2,
        "cached": False,
    }

    second = optimizer.run_optimizer("show_network", "show the network")
    assert second["compute_load"] == PARTIAL_COMPUTE_LOAD
    assert second["associative_bandwidth"] == 0.35
    assert second["execution_status"] == "partial"

    third = optimizer.run_optimizer("show_network", "show the network")
    assert third["compute_load"] == CACHED_COMPUTE_LOAD
    assert third["associative_bandwidth"] == 0.46
    assert third["execution_status"] == "cached"
    assert third["cache_progress"]["cached"] is True
    assert third["cached_routes"] == 1

    fourth = optimizer.run_optimizer("show_network", "show the network")
    assert fourth["compute_load"] == CACHED_COMPUTE_LOAD
    assert fourth["cache_progress"]["runs"] == 4


def test_cached_route_count_spans_multiple_intents():
    optimizer = make_optimizer()
    for _ in range(3):
        optimizer.run_optimizer("show_network", "show the network")
    for _ in range(2):
        optimizer.run_optimizer("add_concept", "add a concept")
    telemetry = optimizer.telemetry()
    assert telemetry["cached_routes"] == 1
    assert telemetry["tracked_routes"] == 2
    assert telemetry["cache_hits"] == 1

    optimizer.run_optimizer("add_concept", "add a concept")
    assert optimizer.cached_route_count == 2


def test_anomaly_override_forces_full_load_and_resets_cache():
    clock = FakeClock()
    optimizer = make_optimizer(clock=clock)
    for _ in range(3):
        optimizer.run_optimizer("show_network", "show the network")
    assert optimizer.compute_load == CACHED_COMPUTE_LOAD

    anomaly = optimizer.run_optimizer(
        "show_network", "unexpected anomaly in the network"
    )
    assert anomaly["execution_status"] == "anomaly_override"
    assert anomaly["compute_load"] == FULL_COMPUTE_LOAD
    assert anomaly["associative_bandwidth"] == 0.0
    assert anomaly["anomaly"] is True
    assert anomaly["override"]["active"] is True
    assert anomaly["cache_progress"]["runs"] == 0

    # The five-second override window keeps the cache bypassed.
    clock.advance(SENSORY_WINDOW_SECONDS - 0.1)
    within = optimizer.run_optimizer("show_network", "show the network")
    assert within["compute_load"] == FULL_COMPUTE_LOAD
    assert within["override"]["active"] is True

    # After the window the route relearns from full load.
    clock.advance(0.2)
    after = optimizer.run_optimizer("show_network", "show the network")
    assert after["execution_status"] == "learning"
    assert after["compute_load"] == FULL_COMPUTE_LOAD
    assert after["cache_progress"]["runs"] == 1


def test_decision_node_override_expires_after_five_seconds():
    clock = FakeClock()
    optimizer = make_optimizer(clock=clock)
    for _ in range(3):
        optimizer.run_optimizer("show_network", "show the network")

    decision = optimizer.run_optimizer(
        "show_network",
        "show the network",
        ["semantic_match", "decision_node:approve"],
    )
    assert decision["execution_status"] == "decision_override"
    assert decision["compute_load"] == FULL_COMPUTE_LOAD
    assert decision["decision_node"] is True

    clock.advance(4.9)
    still_locked = optimizer.run_optimizer("show_network", "show the network")
    assert still_locked["compute_load"] == FULL_COMPUTE_LOAD
    assert still_locked["override"]["kind"] == "decision_override"

    clock.advance(0.2)
    after = optimizer.run_optimizer("show_network", "show the network")
    assert after["execution_status"] == "learning"
    assert after["override"]["active"] is False


def test_anomaly_markers_match_whole_words_only():
    optimizer = make_optimizer()
    normal = optimizer.run_optimizer(
        "load_defaults", "load default settings", ["semantic_match"]
    )
    assert normal["anomaly"] is False
    assert normal["execution_status"] == "learning"

    actual = optimizer.run_optimizer(
        "load_defaults", "fault detected in route", ["semantic_match"]
    )
    assert actual["anomaly"] is True
    assert actual["execution_status"] == "anomaly_override"


def test_invalid_optimizer_inputs_are_rejected():
    optimizer = make_optimizer()
    with pytest.raises(ValueError):
        optimizer.run_optimizer("", "signal")
    with pytest.raises(ValueError):
        optimizer.run_optimizer("intent", "  ")
    with pytest.raises(ValueError):
        optimizer.run_optimizer("intent", "signal", path_steps="not-a-list")


# ------------------------------------------------------------------ IACT-5SCP
def test_sensory_pair_anchors_and_persists_to_store(tmp_path):
    path = tmp_path / "iact_data.json"
    store = ConceptStore(data_file=path)
    clock = FakeClock()
    optimizer = IntegratedIACTSystem(
        concept_writer=store.upsert, clock=clock, repetition_threshold=3
    )

    result = optimizer.synthesize_sensory_pair(
        SensoryInput("visual", "red square", 0.8),
        SensoryInput("auditory", "low tone", 1.2),
        4.99,
    )

    assert result["status"] == "anchored"
    assert result["capture_delay_seconds"] == 4.99
    assert result["anchored_concept"]["expires_in_seconds"] == 5.0
    assert len(store) == 1
    assert store.search("cross-modal"), "anchored concept must be searchable"

    # Single source of truth: the concept landed in iact_data.json.
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert len(payload) == 1
    assert payload[0]["category"] == "Sensory Synthesis"
    assert "Cross-modal synthesis" in payload[0]["description"]
    assert "vector" not in payload[0]

    reloaded = ConceptStore(data_file=path)
    assert reloaded.load_data() == 1
    network, _ = reloaded.snapshot()
    assert network[0]["concept"] == result["concept"]["concept"]
    assert network[0]["description"] == result["concept"]["description"]


def test_sensory_pair_exactly_at_five_seconds_is_anchored():
    store = ConceptStore()
    optimizer = make_optimizer(concept_writer=store.upsert)
    result = optimizer.synthesize_sensory_pair(
        SensoryInput("visual", "blue circle"),
        SensoryInput("tactile", "smooth surface"),
        5.0,
    )
    assert result["status"] == "anchored"
    assert len(store) == 1


def test_sensory_pair_after_window_is_discarded(tmp_path):
    path = tmp_path / "iact_data.json"
    store = ConceptStore(data_file=path)
    optimizer = make_optimizer(concept_writer=store.upsert)

    result = optimizer.synthesize_sensory_pair(
        SensoryInput("visual", "red square"),
        SensoryInput("auditory", "low tone"),
        5.01,
    )

    assert result["status"] == "discarded"
    assert "five-second" in result["reason"] or "5-second" in result["reason"]
    assert result["concept"] is None
    assert len(store) == 0
    assert not path.exists()
    assert optimizer.telemetry()["counters"]["discarded"] == 1


def test_volatile_ttl_expires_without_deleting_the_persisted_concept():
    clock = FakeClock()
    store = ConceptStore()
    optimizer = make_optimizer(clock=clock, concept_writer=store.upsert)

    optimizer.synthesize_sensory_pair(
        SensoryInput("visual", "red square"),
        SensoryInput("auditory", "low tone"),
        1.0,
    )
    assert len(optimizer.anchored_concepts()) == 1

    clock.advance(4.0)
    assert len(optimizer.anchored_concepts()) == 1

    clock.advance(1.1)
    assert optimizer.anchored_concepts() == []
    # The durable concept remains in the ConceptStore.
    assert len(store) == 1
    counters = optimizer.telemetry()["counters"]
    assert counters["anchored"] == 1
    assert counters["active_anchors"] == 0


def test_sensory_pair_without_concept_store_reports_error():
    optimizer = make_optimizer()
    result = optimizer.synthesize_sensory_pair(
        SensoryInput("visual", "red square"),
        SensoryInput("auditory", "low tone"),
        1.0,
    )
    assert result["status"] == "error"
    assert "concept store" in result["message"]


def test_sensory_input_validation():
    with pytest.raises(ValueError):
        SensoryInput("", "description")
    with pytest.raises(ValueError):
        SensoryInput("visual", "")
    with pytest.raises(ValueError):
        SensoryInput("visual", "description", -1.0)

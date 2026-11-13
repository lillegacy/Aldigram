"""Integration tests for the native Termux HTTP server.

These exercise the same routes as the FastAPI backend without requiring
FastAPI/pydantic, verifying that Termux compatibility is additive and does not
change API behavior.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from backend.cognitive_search import CognitiveSearchEngine
from backend.iact_engine import MultiLingualIACTEngine
from backend.intents import build_engine
from backend.termux_server import Service, create_server

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


@pytest.fixture()
def api():
    engine = build_engine(MultiLingualIACTEngine(encoder=stub_encoder))
    service = Service().start(engine=engine, search=CognitiveSearchEngine(engine))
    server = create_server("127.0.0.1", 0, service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield base
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def request(base, method, path, payload=None, headers=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request_headers = dict(headers or {})
    if data is not None:
        request_headers.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(
        base + path, data=data, method=method, headers=request_headers
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status, json.loads(response.read() or b"null"), dict(response.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"null"), dict(exc.headers)


def seed_concepts(base):
    for payload in (
        {
            "title": "Artificial Intelligence",
            "category": "Technology",
            "keywords": ["ai", "neural", "cognition"],
        },
        {
            "title": "Neural Networks",
            "category": "Technology",
            "keywords": ["neural", "deep", "learning"],
        },
        {
            "title": "Cognitive Science",
            "category": "Science",
            "keywords": ["cognition", "mind"],
        },
    ):
        status, _, _ = request(base, "POST", "/concepts/", payload)
        assert status == 201


# --------------------------------------------------------------------- routes
def test_health(api):
    status, body, _ = request(api, "GET", "/health/")
    assert status == 200
    assert body["status"] == "ok"
    assert body["engine"]["ready"] is True
    assert body["cognitive_search"]["ready"] is True
    assert body["compute_load"] == 1.0
    assert body["associative_bandwidth"] == 0.0
    assert body["cached_routes"] == 0
    assert body["optimizer"]["repetition_threshold"] == 3


def test_optimizer_run_route(api):
    payload = {
        "intent": "show_network",
        "trigger_signal": "show the network",
        "path_steps": ["semantic_match", "action_dispatch"],
    }
    status, first, _ = request(api, "POST", "/optimizer/run/", payload)
    assert status == 200
    assert first["compute_load"] == 1.0
    assert first["execution_status"] == "learning"

    _, second, _ = request(api, "POST", "/optimizer/run/", payload)
    assert second["compute_load"] == 0.65

    _, third, _ = request(api, "POST", "/optimizer/run/", payload)
    assert third["compute_load"] == 0.54
    assert third["associative_bandwidth"] == 0.46
    assert third["execution_status"] == "cached"
    assert third["cached_routes"] == 1

    _, body, _ = request(api, "GET", "/health/")
    assert body["compute_load"] == 0.54
    assert body["associative_bandwidth"] == 0.46
    assert body["cached_routes"] == 1

    status, anomaly, _ = request(
        api,
        "POST",
        "/optimizer/run/",
        {
            "intent": "show_network",
            "trigger_signal": "unexpected anomaly detected",
            "path_steps": [],
        },
    )
    assert status == 200
    assert anomaly["execution_status"] == "anomaly_override"
    assert anomaly["compute_load"] == 1.0


def test_optimizer_sensory_pair_route(api):
    status, body, _ = request(
        api,
        "POST",
        "/optimizer/sensory-pair/",
        {
            "item_a": {
                "modality": "visual",
                "description": "red square",
                "focus_duration": 0.8,
            },
            "item_b": {
                "modality": "auditory",
                "description": "low tone",
                "focus_duration": 1.2,
            },
            "capture_delay_seconds": 4.5,
        },
    )
    assert status == 200
    assert body["status"] == "anchored"
    anchor_title = body["concept"]["concept"]
    assert body["concept"]["category"] == "Sensory Synthesis"

    status, network, _ = request(api, "GET", "/network/")
    assert status == 200
    anchored = [item for item in network["network"] if item["concept"] == anchor_title]
    assert len(anchored) == 1
    assert "description" in anchored[0]

    status, found, _ = request(api, "GET", "/concepts/search/?q=cross-modal")
    assert status == 200
    assert any(item["concept"] == anchor_title for item in found["results"])

    status, discarded, _ = request(
        api,
        "POST",
        "/optimizer/sensory-pair/",
        {
            "item_a": {"modality": "visual", "description": "blue circle"},
            "item_b": {"modality": "tactile", "description": "rough surface"},
            "capture_delay_seconds": 5.5,
        },
    )
    assert status == 200
    assert discarded["status"] == "discarded"
    assert discarded["concept"] is None

    status, health, _ = request(api, "GET", "/health/")
    assert health["optimizer"]["counters"]["anchored"] == 1
    assert health["optimizer"]["counters"]["discarded"] == 1


def test_optimizer_route_validation(api):
    status, body, _ = request(
        api, "POST", "/optimizer/run/", {"intent": "", "trigger_signal": "x"}
    )
    assert status == 422
    assert "detail" in body

    status, _, _ = request(
        api,
        "POST",
        "/optimizer/sensory-pair/",
        {
            "item_a": {"modality": "visual", "description": "a"},
            "item_b": {"modality": "auditory", "description": "b"},
            "capture_delay_seconds": -1,
        },
    )
    assert status == 422


def test_concept_lifecycle_matches_fastapi_behavior(api):
    status, body, _ = request(
        api,
        "POST",
        "/concepts/",
        {
            "title": "Artificial Intelligence",
            "category": "Technology",
            "keywords": "AI, neural, cognition",
        },
    )
    assert status == 201
    assert body["keywords"] == ["ai", "neural", "cognition"]

    status, body, _ = request(api, "GET", "/network/")
    assert status == 200
    assert body["network"][0]["concept"] == "Artificial Intelligence"

    status, body, _ = request(api, "GET", "/concepts/Artificial%20Intelligence")
    assert status == 200
    assert body["concept"] == "Artificial Intelligence"

    status, body, _ = request(api, "DELETE", "/concepts/Artificial%20Intelligence")
    assert status == 200
    assert body["deleted"] == "Artificial Intelligence"

    status, _, _ = request(api, "GET", "/concepts/Artificial%20Intelligence")
    assert status == 404


def test_search_and_qa_over_http(api):
    seed_concepts(api)

    status, body, _ = request(
        api, "POST", "/search/", {"query": "What is artificial intelligence?", "top_k": 3}
    )
    assert status == 200
    assert body["resolved"] is True
    assert body["results"][0]["concept"] == "Artificial Intelligence"

    status, body, _ = request(
        api, "POST", "/qa/", {"question": "What is artificial intelligence?", "top_k": 3}
    )
    assert status == 200
    assert body["resolved"] is True
    assert "Artificial Intelligence" in body["answer"]
    assert body["subgraph"]["nodes"]
    assert isinstance(body["relationships"], list)


def test_command_via_http(api):
    status, body, _ = request(api, "POST", "/commands/", {"command": "show the network"})
    assert status == 200
    assert body["intent"] == "show_network"
    assert body["action"]["action"] == "refresh_network_view"


def test_api_search_spec_route(api):
    seed_concepts(api)
    status, body, _ = request(
        api, "POST", "/api/search/", {"query": "what is artificial intelligence"}
    )
    assert status == 200
    assert body["status"] == "success"
    assert body["concept_name"] == "Artificial Intelligence"
    assert 0.0 <= body["confidence"] <= 1.0

    status, body, _ = request(api, "POST", "/api/search/", {"query": "banana smoothie"})
    assert status == 200
    assert body["status"] == "not_found"


def test_validation_and_errors(api):
    status, body, _ = request(api, "POST", "/concepts/", {})
    assert status == 422
    assert "detail" in body

    status, _, _ = request(api, "POST", "/search/", {"query": ""})
    assert status == 422

    status, _, _ = request(api, "GET", "/concepts/search/")
    assert status == 422

    status, _, _ = request(api, "GET", "/unknown/")
    assert status == 404


def test_cors_preflight(api):
    status, _, headers = request(api, "OPTIONS", "/search/")
    assert status == 204
    assert headers.get("Access-Control-Allow-Origin") == "*"
    assert "POST" in headers.get("Access-Control-Allow-Methods", "")

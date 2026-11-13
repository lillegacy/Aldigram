"""HTTP-level parity checks for the FastAPI optimizer routes.

The native Termux server routes are covered in :mod:`test_termux_server`.  This
module starts the real FastAPI app with uvicorn and asserts the same behavior,
so both servers stay aligned.  It is skipped on Termux, where uvicorn/FastAPI
are not installed; the Termux suite still covers the native server there.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

REPO_ROOT = Path(__file__).resolve().parents[2]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _request(base: str, method: str, path: str, payload: dict | None = None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(base + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status, json.loads(response.read() or b"null")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"null")


def _wait_for_health(base: str, process: subprocess.Popen) -> dict:
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output = process.stdout.read() if process.stdout else ""
            pytest.fail(f"FastAPI exited during startup:\n{output}")
        try:
            status, body = _request(base, "GET", "/health/")
            if status == 200:
                return body
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(0.1)
    pytest.fail("FastAPI did not become ready within 30 seconds")


@pytest.fixture()
def fastapi_server(tmp_path):
    port = _free_port()
    env = os.environ.copy()
    env["IACT_ALLOW_FALLBACK"] = "1"
    env["IACT_DATA_FILE"] = str(tmp_path / "iact_data.json")
    env["IACT_REPETITION_THRESHOLD"] = "3"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "backend.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        health = _wait_for_health(base, process)
        yield base, health, tmp_path / "iact_data.json"
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover - safety net
            process.kill()
            process.wait(timeout=5)


def test_fastapi_optimizer_route_parity(fastapi_server):
    base, health, _ = fastapi_server
    assert health["compute_load"] == 1.0
    assert health["associative_bandwidth"] == 0.0
    assert health["cached_routes"] == 0
    assert health["optimizer"]["repetition_threshold"] == 3

    payload = {
        "intent": "show_network",
        "trigger_signal": "show the network",
        "path_steps": ["semantic_match", "action_dispatch"],
    }
    loads = []
    for _ in range(3):
        status, body = _request(base, "POST", "/optimizer/run/", payload)
        assert status == 200
        loads.append(body["compute_load"])
    assert loads == [1.0, 0.65, 0.54]

    status, body = _request(
        base,
        "POST",
        "/optimizer/run/",
        {
            "intent": "show_network",
            "trigger_signal": "unexpected anomaly",
            "path_steps": [],
        },
    )
    assert status == 200
    assert body["execution_status"] == "anomaly_override"
    assert body["compute_load"] == 1.0

    status, health = _request(base, "GET", "/health/")
    assert status == 200
    assert health["compute_load"] == 1.0
    assert health["cached_routes"] == 0


def test_fastapi_sensory_pair_parity(fastapi_server):
    base, _, data_file = fastapi_server
    payload = {
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
    }
    status, body = _request(base, "POST", "/optimizer/sensory-pair/", payload)
    assert status == 200
    assert body["status"] == "anchored"
    title = body["concept"]["concept"]

    status, network = _request(base, "GET", "/network/")
    assert status == 200
    anchored = [item for item in network["network"] if item["concept"] == title]
    assert len(anchored) == 1
    assert anchored[0]["category"] == "Sensory Synthesis"
    assert anchored[0]["description"]

    status, found = _request(
        base, "GET", "/concepts/search/?q=" + quote("cross-modal")
    )
    assert status == 200
    assert any(item["concept"] == title for item in found["results"])

    status, discarded = _request(
        base,
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

    # The anchor was persisted to iact_data.json without vector data.
    records = json.loads(data_file.read_text(encoding="utf-8"))
    assert len(records) == 1
    assert records[0]["concept"] == title
    assert records[0]["category"] == "Sensory Synthesis"
    assert records[0]["description"]
    assert "vector" not in records[0]

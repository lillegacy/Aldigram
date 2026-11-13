"""Native Termux-compatible HTTP server for the IACT backend.

FastAPI depends on ``pydantic-core`` (Rust) and the real semantic model needs
PyTorch; neither has native Termux wheels.  This server exposes the *same* JSON
API using only the Python standard library plus numpy, and reuses the real
:class:`~backend.iact_engine.MultiLingualIACTEngine` and
:class:`~backend.cognitive_search.CognitiveSearchEngine`.

On Android the engine automatically falls back to the deterministic
char-n-gram encoder; on every other platform the FastAPI backend remains the
primary, fully featured deployment.

Usage::

    pkg install python-numpy
    python -m backend.termux_server --port 8000

Endpoints mirror ``backend.main``:

GET  /                 service info
GET  /health/          service + engine status
GET  /network/         concept network with links
GET  /intents/         registered intents
GET  /concepts/search/ substring concept search (?q=...)
GET  /concepts/{title} single concept
POST /concepts/        add/update a concept
DELETE /concepts/{title}
POST /commands/        multilingual command -> intent -> action
POST /api/search/      cognitive search (spec response shape)
POST /search/          semantic search
POST /qa/              question answering
POST /optimizer/run/   CECANP-0.54 route load tracking
POST /optimizer/sensory-pair/  IACT-5SCP sensory synthesis + anchoring
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

try:  # Package import (python -m backend.termux_server)
    from .cognitive_optimizer import (
        IntegratedIACTSystem,
        SensoryInput,
        repetition_threshold_from_env,
    )
    from .cognitive_search import CognitiveSearchEngine
    from .iact_engine import MODEL_NAME, MultiLingualIACTEngine
    from .intents import ACTION_MAP, CLARIFY_ACTION, build_engine
    from .knowledge_base import ConceptStore, parse_concept
except ImportError:  # pragma: no cover - direct script execution
    from cognitive_optimizer import IntegratedIACTSystem  # type: ignore
    from cognitive_optimizer import SensoryInput, repetition_threshold_from_env  # type: ignore
    from cognitive_search import CognitiveSearchEngine  # type: ignore
    from iact_engine import MODEL_NAME, MultiLingualIACTEngine  # type: ignore
    from intents import ACTION_MAP, CLARIFY_ACTION, build_engine  # type: ignore
    from knowledge_base import ConceptStore, parse_concept  # type: ignore

logger = logging.getLogger("iact.termux")
MAX_BODY_BYTES = 1 * 1024 * 1024


class ApiError(Exception):
    """HTTP error carrying a status code and JSON-friendly detail."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


class Service:
    """Application state shared by all request threads."""

    def __init__(self, store: ConceptStore | None = None) -> None:
        self.store = store if store is not None else ConceptStore()
        self.optimizer = IntegratedIACTSystem(
            concept_writer=self.store.upsert,
            repetition_threshold=repetition_threshold_from_env(),
        )
        self.engine: MultiLingualIACTEngine | None = None
        self.search: CognitiveSearchEngine | None = None
        self.error: str | None = None
        self.load_error: str | None = None

    def start(
        self,
        engine: MultiLingualIACTEngine | None = None,
        search: CognitiveSearchEngine | None = None,
    ) -> "Service":
        # Restore persisted concepts first (vector indexes rebuild lazily).
        try:
            loaded = self.store.load_data()
            if loaded:
                logger.info("loaded %d concepts from %s", loaded, self.store.data_file)
        except Exception as exc:  # pragma: no cover - depends on file contents
            self.load_error = str(exc)
            logger.warning("could not load %s: %s", self.store.data_file, exc)

        try:
            self.engine = engine if engine is not None else build_engine()
            self.engine.concept_provider = self.store.snapshot
            self.engine.optimizer = self.optimizer
            self.search = (
                search if search is not None else CognitiveSearchEngine(self.engine)
            )
            self.error = None
        except Exception as exc:  # pragma: no cover - environment dependent
            self.engine = None
            self.search = None
            self.error = str(exc)
            logger.warning("semantic engine unavailable: %s", exc)
        return self

    def health(self) -> dict:
        optimizer = self.optimizer.telemetry()
        return {
            "status": "ok",
            "concepts": len(self.store),
            "data_file": self.store.data_file,
            "load_error": self.load_error,
            "compute_load": optimizer["compute_load"],
            "associative_bandwidth": optimizer["associative_bandwidth"],
            "cached_routes": optimizer["cached_routes"],
            "optimizer": optimizer,
            "engine": {
                "ready": self.engine is not None,
                "error": self.error,
                "model": MODEL_NAME,
                "fallback": self.engine.using_fallback if self.engine else None,
                "threshold": self.engine.threshold if self.engine else None,
                "intents": self.engine.intents if self.engine else [],
            },
            "cognitive_search": {
                "ready": self.search is not None,
                "documents": self.search.document_count if self.search else 0,
                "indexed_version": self.search.indexed_version if self.search else None,
                "search_min_score": self.search.search_min_score if self.search else None,
                "qa_min_score": self.search.qa_min_score if self.search else None,
            },
        }


def _require_engine(service: Service) -> MultiLingualIACTEngine:
    if service.engine is None:
        raise ApiError(
            503,
            service.error
            or "semantic engine is not initialized; run pkg install python-numpy",
        )
    return service.engine


def _require_search(service: Service) -> CognitiveSearchEngine:
    if service.search is None:
        raise ApiError(
            503,
            service.error
            or "cognitive search engine is not initialized; run pkg install python-numpy",
        )
    return service.search


def _string_field(payload: dict, key: str, *, max_length: int) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ApiError(422, f"{key} must be a string")
    value = value.strip()
    if not value:
        raise ApiError(422, f"{key} must not be empty")
    if len(value) > max_length:
        raise ApiError(422, f"{key} must be at most {max_length} characters")
    return value


def _int_field(payload: dict, key: str, default: int, *, low: int, high: int) -> int:
    value = payload.get(key, default)
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ApiError(422, f"{key} must be an integer") from exc
    if isinstance(value, bool) or not low <= parsed <= high:
        raise ApiError(422, f"{key} must be between {low} and {high}")
    return parsed


def _float_field(
    payload: dict, key: str, *, low: float, high: float, default: float | None = None
) -> float:
    value = payload.get(key, default)
    if value is None:
        raise ApiError(422, f"{key} is required")
    if isinstance(value, bool):
        raise ApiError(422, f"{key} must be a number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ApiError(422, f"{key} must be a number") from exc
    if not low <= parsed <= high:
        raise ApiError(422, f"{key} must be between {low} and {high}")
    return parsed


def _sensory_item(payload: dict, key: str) -> SensoryInput:
    item = payload.get(key)
    if not isinstance(item, dict):
        raise ApiError(422, f"{key} must be an object")
    modality = _string_field(item, "modality", max_length=60)
    description = _string_field(item, "description", max_length=500)
    focus = _float_field(item, "focus_duration", low=0.0, high=3600.0, default=0.0)
    try:
        return SensoryInput(modality, description, focus)
    except ValueError as exc:
        raise ApiError(422, str(exc)) from exc


def _score_field(payload: dict, key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ApiError(422, f"{key} must be a number") from exc
    if not 0.0 <= parsed <= 1.0:
        raise ApiError(422, f"{key} must be between 0.0 and 1.0")
    return parsed


class IACTRequestHandler(BaseHTTPRequestHandler):
    server_version = "IACTTermux/2.0"
    protocol_version = "HTTP/1.1"

    # ------------------------------------------------------------- plumbing
    @property
    def service(self) -> Service:
        return self.server.service  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        logger.info("%s - %s", self.address_string(), fmt % args)

    def _send(self, status: int, payload: dict | list) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _read_json(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except (TypeError, ValueError) as exc:
            raise ApiError(400, "invalid Content-Length header") from exc
        if length <= 0:
            raise ApiError(400, "JSON body required")
        if length > MAX_BODY_BYTES:
            raise ApiError(413, "request body too large")
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ApiError(400, "invalid JSON body") from exc
        if not isinstance(payload, dict):
            raise ApiError(400, "JSON object required")
        return payload

    def _path(self) -> tuple[str, dict[str, list[str]]]:
        parsed = urlparse(self.path)
        path = parsed.path
        if len(path) > 1:
            path = path.rstrip("/")
        return path or "/", parse_qs(parsed.query)

    # -------------------------------------------------------------- routing
    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("POST")

    def do_DELETE(self) -> None:  # noqa: N802
        self._dispatch("DELETE")

    def _dispatch(self, method: str) -> None:
        try:
            status, payload = self._route(method)
        except ApiError as exc:
            status, payload = exc.status, {"detail": exc.detail}
        except Exception as exc:  # pragma: no cover - safety net
            logger.exception("unhandled error")
            status, payload = 500, {"detail": f"internal server error: {exc}"}
        self._send(status, payload)

    def _route(self, method: str) -> tuple[int, dict | list]:
        path, query = self._path()
        service = self.service

        if method == "GET" and path == "/":
            return 200, {
                "name": "IACT Cognitive Backend (Termux)",
                "version": "2.0.0",
                "model": MODEL_NAME,
                "threshold": service.engine.threshold if service.engine else None,
                "endpoints": [
                    "/health/",
                    "/network/",
                    "/concepts/",
                    "/intents/",
                    "/commands/",
                    "/search/",
                    "/api/search/",
                    "/qa/",
                    "/optimizer/run/",
                    "/optimizer/sensory-pair/",
                ],
            }

        if method == "GET" and path == "/health":
            return 200, service.health()

        if method == "GET" and path == "/network":
            return 200, {"network": service.store.with_links()}

        if method == "GET" and path == "/intents":
            engine = _require_engine(service)
            return 200, {
                "model": engine.model_name,
                "threshold": engine.threshold,
                "intents": [
                    {"intent": name, "samples": engine.sample_counts.get(name, 0)}
                    for name in engine.intents
                ],
            }

        if method == "GET" and path == "/concepts/search":
            values = query.get("q") or [""]
            needle = values[0].strip()
            if not needle:
                raise ApiError(422, "query parameter q is required")
            return 200, {"query": needle, "results": service.store.search(needle)}

        if path.startswith("/concepts/"):
            title = unquote(path[len("/concepts/") :]).strip()
            if not title:
                raise ApiError(404, "concept not found")
            if method == "GET":
                concepts, _ = service.store.snapshot()
                for concept in concepts:
                    if concept["concept"].lower() == title.lower():
                        return 200, concept
                raise ApiError(404, f"concept {title!r} not found")
            if method == "DELETE":
                if not service.store.delete(title):
                    raise ApiError(404, f"concept {title!r} not found")
                return 200, {"deleted": title}
            raise ApiError(405, "method not allowed")

        if method == "POST" and path == "/concepts":
            payload = self._read_json()
            try:
                record = parse_concept(payload)
            except ValueError as exc:
                raise ApiError(422, str(exc)) from exc
            return 201, service.store.upsert(record)

        if method == "POST" and path == "/commands":
            engine = _require_engine(service)
            payload = self._read_json()
            command = _string_field(payload, "command", max_length=500)
            result = engine.process_command(command)
            action = ACTION_MAP.get(result["intent"]) if result["intent"] else None
            result["action"] = action or CLARIFY_ACTION
            return 200, result

        if method == "POST" and path == "/api/search":
            engine = _require_engine(service)
            payload = self._read_json()
            query = _string_field(payload, "query", max_length=500)
            try:
                return 200, engine.cognitive_search(query)
            except ValueError as exc:
                raise ApiError(422, str(exc)) from exc

        if method == "POST" and path == "/optimizer/run":
            payload = self._read_json()
            intent = _string_field(payload, "intent", max_length=200)
            trigger = _string_field(payload, "trigger_signal", max_length=500)
            raw_steps = payload.get("path_steps") or []
            if not isinstance(raw_steps, list):
                raise ApiError(422, "path_steps must be a list of strings")
            if len(raw_steps) > 50:
                raise ApiError(422, "path_steps must contain at most 50 steps")
            steps = []
            for step in raw_steps:
                if not isinstance(step, str):
                    raise ApiError(422, "path_steps must contain strings")
                if step.strip():
                    steps.append(step.strip()[:200])
            try:
                return 200, service.optimizer.run_optimizer(intent, trigger, steps)
            except ValueError as exc:
                raise ApiError(422, str(exc)) from exc

        if method == "POST" and path == "/optimizer/sensory-pair":
            payload = self._read_json()
            item_a = _sensory_item(payload, "item_a")
            item_b = _sensory_item(payload, "item_b")
            delay = _float_field(
                payload, "capture_delay_seconds", low=0.0, high=3600.0
            )
            try:
                return 200, service.optimizer.synthesize_sensory_pair(
                    item_a, item_b, delay
                )
            except ValueError as exc:
                raise ApiError(422, str(exc)) from exc

        if method == "POST" and path == "/search":
            search = _require_search(service)
            payload = self._read_json()
            text = _string_field(payload, "query", max_length=500)
            top_k = _int_field(payload, "top_k", 5, low=1, high=20)
            min_score = _score_field(payload, "min_score")
            concepts, version = service.store.snapshot()
            try:
                result = search.search(
                    text, concepts, version, top_k=top_k, min_score=min_score
                )
            except RuntimeError as exc:
                raise ApiError(503, str(exc)) from exc
            return 200, result

        if method == "POST" and path == "/qa":
            search = _require_search(service)
            payload = self._read_json()
            question = _string_field(payload, "question", max_length=1000)
            top_k = _int_field(payload, "top_k", 3, low=1, high=10)
            min_score = _score_field(payload, "min_score")
            concepts, version = service.store.snapshot()
            try:
                result = search.answer(
                    question, concepts, version, top_k=top_k, min_score=min_score
                )
            except RuntimeError as exc:
                raise ApiError(503, str(exc)) from exc
            return 200, result

        raise ApiError(404, "not found")


class IACTServer(ThreadingHTTPServer):
    daemon_threads = True
    service: Service


def create_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    service: Service | None = None,
) -> IACTServer:
    """Create (but do not start) a Termux-compatible HTTP server."""
    active = service if service is not None else Service().start()
    server = IACTServer((host, port), IACTRequestHandler)
    server.service = active
    return server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="IACT native Termux backend")
    parser.add_argument(
        "--host",
        default=os.environ.get("IACT_HOST", "127.0.0.1"),
        help="bind address (use 0.0.0.0 for LAN access)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("IACT_PORT", "8000")),
        help="TCP port (default: 8000)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    service = Service(
        ConceptStore(data_file=os.environ.get("IACT_DATA_FILE", "iact_data.json"))
    ).start()
    server = create_server(args.host, args.port, service)
    if server.service.error:
        logger.warning(
            "engine unavailable (%s); concept endpoints still work",
            server.service.error,
        )
    print(
        f"IACT Termux backend listening on http://{args.host}:{args.port}"
        f"  (engine: {'ready' if server.service.engine else 'unavailable'})"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":  # pragma: no cover
    main()

"""FastAPI backend for the IACT cognitive dashboard.

Endpoints
---------
GET  /health/            service + engine status
GET  /network/           concept network (nodes with keyword links)
POST /concepts/          add a concept to the knowledge base
DELETE /concepts/{title} remove a concept
GET  /concepts/search/   lightweight concept search
GET  /intents/           registered semantic intents
POST /commands/          multilingual command -> intent -> action

The command endpoint is powered by :class:`MultiLingualIACTEngine`, which uses
``sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`` to place
English, Thaana and Latin-transliterated Dhivehi commands in one shared vector
space and resolves the best intent by cosine similarity (threshold 0.45).
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from threading import Lock

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from .cognitive_search import CognitiveSearchEngine
from .iact_engine import DEFAULT_THRESHOLD, MODEL_NAME, MultiLingualIACTEngine

logger = logging.getLogger("iact.backend")

# ---------------------------------------------------------------------------
# Default intent knowledge base.  Each intent is registered with the same
# command expressed in English, Dhivehi (Latin transliteration) and Dhivehi
# (Thaana) so that all three script variants land close together in vector
# space.
# ---------------------------------------------------------------------------
DEFAULT_INTENTS: dict[str, list[str]] = {
    "add_concept": [
        "add a new concept",
        "create a concept and integrate it into the network",
        "register a new idea in the knowledge base",
        # Dhivehi, Latin transliteration
        "au konsepteh ithuru kurey",
        "nuvata konsepteh add kurey",
        # Dhivehi, Thaana script
        "އައު ކޮންސެޕްޓެއް އިތުރުކުރޭ",
    ],
    "show_network": [
        "show the network",
        "display the associative network map",
        "view all concepts and their links",
        "netvaak dhakkaa",
        "network dhakkaa",
        "ނެޓްވާކް ދައްކާ",
    ],
    "search_concept": [
        "search for a concept",
        "find a concept by keyword",
        "look up an idea in the network",
        "konsepteh hoadhaa",
        "concept eh hoyaa",
        "ކޮންސެޕްޓެއް ހޯދާ",
    ],
    "delete_concept": [
        "delete a concept",
        "remove a concept from the network",
        "drop an idea from the knowledge base",
        "konsepteh fuhelaa",
        "concept eh remove kurey",
        "ކޮންސެޕްޓެއް ފުހެލާ",
    ],
    "export_network": [
        "export the network",
        "download the network as json",
        "save a copy of the knowledge base",
        "netvaak ekspoat kurey",
        "network export kurey",
        "ނެޓްވާކް އެކްސްޕޯޓްކުރޭ",
    ],
    "help": [
        "help me",
        "what can you do",
        "show available commands",
        "eheetheri vedhey",
        "ehee therei vedhey",
        "އެހީތެރިވެދޭ",
    ],
}

# Intent -> frontend workflow mapping.
ACTION_MAP: dict[str, dict] = {
    "add_concept": {
        "action": "open_concept_form",
        "description": "Focus the dashboard's concept form to add a new node.",
    },
    "show_network": {
        "action": "refresh_network_view",
        "description": "Refresh and display the associative network map.",
    },
    "search_concept": {
        "action": "focus_concept_search",
        "description": "Focus concept search; expect a query parameter.",
        "requires": ["query"],
    },
    "delete_concept": {
        "action": "request_concept_deletion",
        "description": "Ask the user which concept to remove.",
        "requires": ["concept"],
    },
    "export_network": {
        "action": "export_network_json",
        "description": "Download the current knowledge base as JSON.",
    },
    "help": {
        "action": "show_help",
        "description": "Show available commands and examples in all three scripts.",
    },
}

CLARIFY_ACTION = {
    "action": "clarify",
    "description": "No intent matched confidently; ask the user to rephrase.",
    "requires": ["rephrase"],
}


# ---------------------------------------------------------------------------
# Concept network (dashboard data)
# ---------------------------------------------------------------------------
class ConceptIn(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    category: str = Field(default="General", max_length=60)
    keywords: list[str] = Field(default_factory=list)

    @field_validator("keywords", mode="before")
    @classmethod
    def _coerce_keywords(cls, value):
        if value is None:
            return []
        if isinstance(value, str):
            value = value.split(",")
        return [str(item).strip().lower() for item in value if str(item).strip()]


class CommandIn(BaseModel):
    command: str = Field(min_length=1, max_length=500)


class SearchIn(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)
    min_score: float | None = Field(default=None, ge=0.0, le=1.0)


class QAIn(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=10)
    min_score: float | None = Field(default=None, ge=0.0, le=1.0)


class ConceptStore:
    """Thread-safe in-memory concept knowledge base."""

    def __init__(self) -> None:
        self._concepts: dict[str, dict] = {}
        self._order: list[str] = []
        self._lock = Lock()
        self._version = 0

    @property
    def version(self) -> int:
        """Monotonic counter bumped on every mutation (used by search index)."""
        with self._lock:
            return self._version

    def upsert(self, concept: ConceptIn) -> dict:
        title = concept.title.strip()
        if not title:
            raise ValueError("title must not be empty")
        record = {
            "concept": title,
            "category": concept.category.strip() or "General",
            "keywords": list(dict.fromkeys(concept.keywords)),
        }
        with self._lock:
            if title not in self._concepts:
                self._order.append(title)
            self._concepts[title] = record
            self._version += 1
        return record

    def delete(self, title: str) -> bool:
        with self._lock:
            if title not in self._concepts:
                return False
            del self._concepts[title]
            self._order.remove(title)
            self._version += 1
            return True

    def search(self, query: str) -> list[dict]:
        needle = query.strip().lower()
        if not needle:
            return []
        with self._lock:
            results = []
            for record in self._concepts.values():
                haystack = " ".join([record["concept"], record["category"], *record["keywords"]]).lower()
                if needle in haystack:
                    results.append(dict(record))
            return results

    def snapshot(self) -> tuple[list[dict], int]:
        """Return the linked network plus the store version, atomically."""
        with self._lock:
            records = [dict(record) for record in self._concepts.values()]
            version = self._version
        keyword_sets = {
            record["concept"]: {keyword.lower() for keyword in record["keywords"]}
            for record in records
        }
        network = []
        for record in records:
            title = record["concept"]
            links = [
                other["concept"]
                for other in records
                if other["concept"] != title
                and keyword_sets[title] & keyword_sets[other["concept"]]
            ]
            network.append({**record, "links": links})
        return network, version

    def with_links(self) -> list[dict]:
        return self.snapshot()[0]

    def __len__(self) -> int:
        with self._lock:
            return len(self._concepts)


# ---------------------------------------------------------------------------
# Application state / startup
# ---------------------------------------------------------------------------
class AppState:
    def __init__(self) -> None:
        self.store = ConceptStore()
        self.engine: MultiLingualIACTEngine | None = None
        self.search_engine: CognitiveSearchEngine | None = None
        self.engine_error: str | None = None


state = AppState()


def build_engine() -> MultiLingualIACTEngine:
    threshold = float(os.environ.get("IACT_MATCH_THRESHOLD", DEFAULT_THRESHOLD))
    engine = MultiLingualIACTEngine(threshold=threshold)
    for intent_name, sample_phrases in DEFAULT_INTENTS.items():
        engine.add_intent(intent_name, sample_phrases)
    return engine


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Load the embedding model once at startup.  Failure is not fatal: the
    # dashboard endpoints keep working and /commands/ reports a clear 503.
    try:
        state.engine = build_engine()
        state.search_engine = CognitiveSearchEngine(state.engine)
        logger.info(
            "IACT semantic engine ready (%d intents) + cognitive search",
            len(state.engine.intents),
        )
    except Exception as exc:  # pragma: no cover - depends on environment
        state.engine = None
        state.search_engine = None
        state.engine_error = str(exc)
        logger.warning("IACT semantic engine unavailable: %s", exc)
    yield
    state.engine = None
    state.search_engine = None


app = FastAPI(
    title="IACT Cognitive Backend",
    description="Multi-lingual cognitive command processing for the IACT dashboard.",
    version="2.0.0",
    lifespan=lifespan,
)

_origins = os.environ.get("IACT_CORS_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _origins.strip() == "*" else [o.strip() for o in _origins.split(",") if o.strip()],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def root() -> dict:
    return {
        "name": "IACT Cognitive Backend",
        "version": app.version,
        "model": MODEL_NAME,
        "threshold": state.engine.threshold if state.engine else None,
        "endpoints": [
            "/health/",
            "/network/",
            "/concepts/",
            "/intents/",
            "/commands/",
            "/search/",
            "/qa/",
        ],
    }


@app.get("/health/")
def health() -> dict:
    return {
        "status": "ok",
        "concepts": len(state.store),
        "engine": {
            "ready": state.engine is not None,
            "error": state.engine_error,
            "model": MODEL_NAME,
            "fallback": state.engine.using_fallback if state.engine else None,
            "threshold": state.engine.threshold if state.engine else None,
            "intents": state.engine.intents if state.engine else [],
        },
        "cognitive_search": {
            "ready": state.search_engine is not None,
            "documents": state.search_engine.document_count if state.search_engine else 0,
            "indexed_version": state.search_engine.indexed_version if state.search_engine else None,
            "search_min_score": state.search_engine.search_min_score if state.search_engine else None,
            "qa_min_score": state.search_engine.qa_min_score if state.search_engine else None,
        },
    }


@app.get("/network/")
def get_network() -> dict:
    return {"network": state.store.with_links()}


@app.post("/concepts/", status_code=201)
def create_concept(concept: ConceptIn) -> dict:
    try:
        return state.store.upsert(concept)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.delete("/concepts/{title}")
def delete_concept(title: str) -> dict:
    if not state.store.delete(title):
        raise HTTPException(status_code=404, detail=f"concept {title!r} not found")
    return {"deleted": title}


@app.get("/concepts/search/")
def search_concepts(q: str = Query(min_length=1, description="Text to search for")) -> dict:
    return {"query": q, "results": state.store.search(q)}


@app.get("/intents/")
def list_intents() -> dict:
    if state.engine is None:
        raise HTTPException(status_code=503, detail=state.engine_error or "engine not ready")
    return {
        "model": state.engine.model_name,
        "threshold": state.engine.threshold,
        "intents": [
            {"intent": name, "samples": state.engine.sample_counts.get(name, 0)}
            for name in state.engine.intents
        ],
    }


@app.post("/commands/")
def process_command(payload: CommandIn) -> dict:
    if state.engine is None:
        raise HTTPException(
            status_code=503,
            detail=state.engine_error
            or "semantic engine is not initialized; install backend requirements",
        )

    result = state.engine.process_command(payload.command)
    action = ACTION_MAP.get(result["intent"]) if result["intent"] else None
    result["action"] = action or CLARIFY_ACTION
    return result


@app.get("/concepts/{title}")
def get_concept(title: str) -> dict:
    concepts, _ = state.store.snapshot()
    for concept in concepts:
        if concept["concept"].lower() == title.lower():
            return concept
    raise HTTPException(status_code=404, detail=f"concept {title!r} not found")


def _require_search() -> CognitiveSearchEngine:
    if state.search_engine is None:
        raise HTTPException(
            status_code=503,
            detail=state.engine_error
            or "cognitive search engine is not initialized; install backend requirements",
        )
    return state.search_engine


@app.post("/search/")
def semantic_search(payload: SearchIn) -> dict:
    """Semantic search across concept nodes, links and context."""
    search = _require_search()
    concepts, version = state.store.snapshot()
    try:
        return search.search(
            payload.query,
            concepts,
            version,
            top_k=payload.top_k,
            min_score=payload.min_score,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/qa/")
def cognitive_qa(payload: QAIn) -> dict:
    """Answer a natural-language question from the associative network."""
    search = _require_search()
    concepts, version = state.store.snapshot()
    try:
        return search.answer(
            payload.question,
            concepts,
            version,
            top_k=payload.top_k,
            min_score=payload.min_score,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

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
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .cognitive_search import CognitiveSearchEngine
from .iact_engine import MODEL_NAME, MultiLingualIACTEngine
from .intents import ACTION_MAP, CLARIFY_ACTION, build_engine
from .knowledge_base import ConceptStore, parse_concept

logger = logging.getLogger("iact.backend")

# Built dashboard (created by `npm run build` / the Docker build).
FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"

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
@app.get("/", response_model=None)
def root():
    index = FRONTEND_DIST / "index.html"
    if index.is_file():
        # Serve the built dashboard when it is available (Docker/Space deploy).
        return FileResponse(index)
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
        return state.store.upsert(parse_concept(concept.model_dump()))
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


# Serve the built dashboard (if present) from the same origin as the API.
# Registered last so every API route and /docs takes precedence.
if (FRONTEND_DIST / "index.html").is_file():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")

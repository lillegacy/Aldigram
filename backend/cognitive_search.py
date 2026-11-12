"""Cognitive search and question answering over the IACT concept network.

This module reuses :class:`~backend.iact_engine.MultiLingualIACTEngine` as the
embedding backend, so queries and concept documents share exactly the same
multilingual vector space (including the Thaana -> Latin normalization and the
optional fallback encoder).

Two capabilities are provided:

* :meth:`CognitiveSearchEngine.search` - semantic retrieval of concept nodes
  with their relationships and a network subgraph.
* :meth:`CognitiveSearchEngine.answer` - a structured, extractive answer
  synthesized from the retrieved concept context.  Answers are deterministic
  (no LLM required): they summarize the best-matching concept, its category,
  keywords, linked nodes and related concepts.
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np

try:  # Package import (uvicorn backend.main:app)
    from .iact_engine import MultiLingualIACTEngine
    from .multilingual import normalize_text
except ImportError:  # pragma: no cover - flat imports for direct execution
    from iact_engine import MultiLingualIACTEngine  # type: ignore
    from multilingual import normalize_text  # type: ignore

DEFAULT_SEARCH_MIN_SCORE = 0.20
DEFAULT_QA_MIN_SCORE = 0.25


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


class CognitiveSearchEngine:
    """Semantic search + QA over the associative network.

    The index is a cache over the concept store.  The store exposes a
    monotonically increasing ``version``; whenever it changes the documents are
    re-encoded on the next query, so adding/deleting concepts stays in sync
    without re-embedding on every request.
    """

    def __init__(
        self,
        engine: MultiLingualIACTEngine,
        *,
        search_min_score: float | None = None,
        qa_min_score: float | None = None,
    ) -> None:
        self.engine = engine
        self.search_min_score = (
            _env_float("IACT_SEARCH_MIN_SCORE", DEFAULT_SEARCH_MIN_SCORE)
            if search_min_score is None
            else float(search_min_score)
        )
        self.qa_min_score = (
            _env_float("IACT_QA_MIN_SCORE", DEFAULT_QA_MIN_SCORE)
            if qa_min_score is None
            else float(qa_min_score)
        )
        self._docs: list[dict] = []
        self._vectors: np.ndarray | None = None
        self._version: int | None = None

    # ------------------------------------------------------------- index state
    @property
    def document_count(self) -> int:
        return len(self._docs)

    @property
    def indexed_version(self) -> int | None:
        return self._version

    @staticmethod
    def _document_text(concept: dict) -> str:
        """Flatten a concept node into indexable text.

        The title is included twice to weight it above keywords and neighbour
        names when ranking, which keeps title matches ahead of incidental
        keyword overlap.
        """
        title = str(concept.get("concept", ""))
        parts = [
            title,
            title,
            str(concept.get("category", "")),
            " ".join(str(keyword) for keyword in concept.get("keywords") or []),
        ]
        links = [str(link) for link in concept.get("links") or []]
        if links:
            parts.append("connected to " + " ".join(links))
        return ". ".join(part for part in parts if part.strip())

    def ensure_index(self, concepts: Sequence[dict], version: int) -> None:
        """Rebuild the vector index when the concept store version changes."""
        if version == self._version:
            return

        docs = [dict(concept) for concept in concepts]
        if docs:
            texts = [self._document_text(doc) for doc in docs]
            vectors = np.asarray(self.engine.encode_texts(texts), dtype=np.float32)
            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)
        else:
            vectors = None

        self._docs = docs
        self._vectors = vectors
        self._version = version

    # ---------------------------------------------------------------- retrieval
    def _rank(self, query_vector: np.ndarray, top_k: int) -> list[dict]:
        if self._vectors is None or not self._docs:
            return []

        scores = np.asarray(
            [self.engine.cosine_similarity(query_vector, vector) for vector in self._vectors],
            dtype=np.float32,
        )
        order = np.argsort(-scores)
        results: list[dict] = []
        for index in order[: max(1, top_k)]:
            score = float(scores[index])
            if score <= 0.0:
                continue
            document = self._docs[int(index)]
            results.append(
                {
                    "concept": document.get("concept", ""),
                    "category": document.get("category", "General"),
                    "keywords": list(document.get("keywords") or []),
                    "links": list(document.get("links") or []),
                    "score": round(score, 4),
                }
            )
        return results

    def search(
        self,
        query: str,
        concepts: Sequence[dict],
        version: int,
        *,
        top_k: int = 5,
        min_score: float | None = None,
    ) -> dict:
        """Semantically retrieve concept nodes relevant to *query*."""
        threshold = self.search_min_score if min_score is None else float(min_score)
        normalized = normalize_text(query or "")
        response = {
            "query": (query or "").strip(),
            "normalized": normalized,
            "threshold": threshold,
            "resolved": False,
            "confidence": 0.0,
            "results": [],
        }
        if not normalized:
            response["error"] = "query contains no searchable text"
            return response

        self.ensure_index(concepts, version)
        if not self._docs:
            response["error"] = "the concept network is empty"
            return response

        results = self._rank(self.engine.encode_query(query), top_k)
        confidence = results[0]["score"] if results else 0.0
        response.update(
            {
                "results": results,
                "confidence": confidence,
                "resolved": bool(results and confidence >= threshold),
            }
        )
        return response

    # ---------------------------------------------------------------------- QA
    def answer(
        self,
        question: str,
        concepts: Sequence[dict],
        version: int,
        *,
        top_k: int = 3,
        min_score: float | None = None,
    ) -> dict:
        """Answer *question* from the associative network context.

        Returns a concise extractive answer plus the matched concepts, their
        relationships and a subgraph that the frontend can render directly.
        """
        threshold = self.qa_min_score if min_score is None else float(min_score)
        search_result = self.search(
            question,
            concepts,
            version,
            top_k=top_k,
            min_score=threshold,
        )
        results = search_result["results"]

        response = {
            "question": (question or "").strip(),
            "normalized": search_result["normalized"],
            "threshold": threshold,
            "resolved": False,
            "confidence": search_result["confidence"],
            "answer": "",
            "evidence": results,
            "relationships": [],
            "subgraph": {"nodes": [], "edges": []},
        }
        if "error" in search_result:
            response["error"] = search_result["error"]
            response["answer"] = (
                "I could not search the knowledge base: " + search_result["error"] + "."
            )
            return response

        if not results or search_result["confidence"] < threshold:
            response["answer"] = (
                "I could not find a confident answer in the associative network. "
                "Try rephrasing the question, or add the missing concept first."
            )
            if results:
                response["subgraph"] = self._subgraph(results, concepts)
                response["relationships"] = self._relationships(results, concepts)
            return response

        top = results[0]
        sentences = [
            f"{top['concept']} is a {top['category']} concept in the IACT knowledge base."
        ]
        if top["keywords"]:
            sentences.append("Key associations: " + ", ".join(top["keywords"]) + ".")
        if top["links"]:
            sentences.append("Directly linked concepts: " + ", ".join(top["links"]) + ".")
        if len(results) > 1:
            related = ", ".join(result["concept"] for result in results[1:])
            sentences.append("Related concepts retrieved: " + related + ".")
        sentences.append(
            f"Semantic match confidence: {search_result['confidence']:.2f}."
        )

        response.update(
            {
                "resolved": True,
                "answer": " ".join(sentences),
                "relationships": self._relationships(results, concepts),
                "subgraph": self._subgraph(results, concepts),
            }
        )
        return response

    # ------------------------------------------------------------ context graph
    @staticmethod
    def _relationships(results: Sequence[dict], concepts: Sequence[dict]) -> list[dict]:
        by_title = {concept.get("concept"): concept for concept in concepts}
        seen: set[tuple[str, str]] = set()
        relationships: list[dict] = []
        for result in results:
            source = result.get("concept", "")
            for link in result.get("links") or []:
                if link not in by_title:
                    continue
                pair = tuple(sorted((source, link)))
                if pair in seen:
                    continue
                seen.add(pair)
                target_keywords = set(by_title[link].get("keywords") or [])
                shared = sorted(set(result.get("keywords") or []) & target_keywords)
                relationships.append(
                    {
                        "source": pair[0],
                        "target": pair[1],
                        "shared_keywords": shared,
                        "type": "association",
                    }
                )
        return relationships

    def _subgraph(self, results: Sequence[dict], concepts: Sequence[dict]) -> dict:
        selected = {result["concept"] for result in results}
        nodes = [
            {
                "id": result["concept"],
                "category": result["category"],
                "score": result["score"],
            }
            for result in results
        ]
        relationships = self._relationships(results, concepts)
        edges = [
            {
                "source": relation["source"],
                "target": relation["target"],
                "shared_keywords": relation["shared_keywords"],
                "type": relation["type"],
            }
            for relation in relationships
            if relation["source"] in selected and relation["target"] in selected
        ]
        return {"nodes": nodes, "edges": edges}

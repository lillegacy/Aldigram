"""Multi-lingual cognitive command engine for the IACT dashboard.

The engine maps free-form commands written in English, Dhivehi (Thaana) or
Dhivehi (Latin transliteration) onto registered intents.  Every intent is
described by sample phrases that are encoded into a shared multilingual vector
space; incoming commands are encoded into the same space and matched by cosine
similarity.  A match at or above ``threshold`` resolves the intent and lets the
API trigger the associated action.
"""

from __future__ import annotations

import hashlib
import os
import sys
from collections import defaultdict
from threading import Lock
from typing import Callable, Iterable, Sequence

import numpy as np

try:  # Support both ``uvicorn backend.main:app`` and direct script execution.
    from .cognitive_optimizer import IntegratedIACTSystem
    from .multilingual import normalize_text
except ImportError:  # pragma: no cover - fallback for flat imports
    from cognitive_optimizer import IntegratedIACTSystem  # type: ignore
    from multilingual import normalize_text  # type: ignore

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_THRESHOLD = 0.45

# An encoder turns a batch of texts into an (n, dim) float matrix.
Encoder = Callable[[Sequence[str]], np.ndarray]


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _running_on_android() -> bool:
    """Detect Android/Termux, where native PyTorch wheels are unavailable."""
    return (
        sys.platform == "android"
        or "ANDROID_ROOT" in os.environ
        or "com.termux" in os.environ.get("PREFIX", "")
    )


class HashingEncoder:
    """Deterministic character n-gram encoder used when torch is unavailable.

    This is *not* a replacement for the multilingual transformer.  It keeps the
    engine (and its tests) functional on platforms without PyTorch, such as
    plain Termux, while the real model remains the default whenever it can be
    imported.
    """

    def __init__(self, dimensions: int = 512, ngram_range: tuple[int, int] = (2, 4)) -> None:
        self.dimensions = dimensions
        self.ngram_range = ngram_range

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        low, high = self.ngram_range
        for row, text in enumerate(texts):
            padded = f" {text} "
            for size in range(low, high + 1):
                for start in range(len(padded) - size + 1):
                    gram = padded[start : start + size]
                    digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
                    vectors[row, int.from_bytes(digest, "little") % self.dimensions] += 1.0
        return vectors


class MultiLingualIACTEngine:
    """Semantic intent resolver for multi-script IACT commands."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        threshold: float = DEFAULT_THRESHOLD,
        *,
        allow_fallback: bool | None = None,
        encoder: Encoder | None = None,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")

        self.model_name = model_name
        self.threshold = float(threshold)
        # Native Termux cannot install PyTorch, so fall back automatically on
        # Android.  Every other platform keeps the strict default (off).
        self.allow_fallback = (
            _env_flag("IACT_ALLOW_FALLBACK", default=_running_on_android())
            if allow_fallback is None
            else allow_fallback
        )

        self._custom_encoder = encoder
        self._model = None
        self._fallback: HashingEncoder | None = None

        self._intent_names: list[str] = []
        self._phrases: list[str] = []
        self._seen: set[tuple[str, str]] = set()
        self._vectors: np.ndarray | None = None

        # Concept knowledge base access for cognitive_search().  The provider
        # returns (linked concept records, store version); wire it from the
        # ConceptStore so the engine never duplicates concept state.
        self.concept_provider: Callable[[], tuple[Sequence[dict], int]] | None = None
        self._concept_docs: list[dict] = []
        self._concept_vectors: np.ndarray | None = None
        self._concept_version: int | None = None
        self._concept_lock = Lock()

        # CECANP-0.54 / IACT-5SCP integration.  Servers replace this default
        # instance with one whose concept_writer points at the shared store.
        self.optimizer = IntegratedIACTSystem()

    # ------------------------------------------------------------------ setup
    def _get_encoder(self) -> Encoder:
        if self._custom_encoder is not None:
            return self._custom_encoder
        if self._model is not None:
            return self._encode_with_model
        if self._fallback is not None:
            return self._fallback.encode

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            if not self.allow_fallback:
                raise RuntimeError(
                    "sentence-transformers is not installed. Install the backend "
                    "requirements, or set IACT_ALLOW_FALLBACK=1 to run with the "
                    "lightweight char-n-gram encoder."
                ) from exc
            self._fallback = HashingEncoder()
            return self._fallback.encode

        try:
            self._model = SentenceTransformer(self.model_name)
        except Exception as exc:
            if not self.allow_fallback:
                raise RuntimeError(
                    f"Could not load the embedding model {self.model_name!r}: {exc}"
                ) from exc
            self._fallback = HashingEncoder()
            return self._fallback.encode

        return self._encode_with_model

    def _encode_with_model(self, texts: Sequence[str]) -> np.ndarray:
        assert self._model is not None
        return np.asarray(
            self._model.encode(list(texts), convert_to_numpy=True),
            dtype=np.float32,
        )

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.asarray(self._get_encoder()(list(texts)), dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        return vectors

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Encode raw texts into the engine's shared vector space.

        Text is normalized (Thaana transliterated, punctuation flattened) before
        encoding so callers always share the same representation as the intent
        matcher.  Returns an ``(n, dim)`` matrix.
        """
        return self._encode([normalize_text(text) for text in texts])

    def encode_query(self, text: str) -> np.ndarray:
        """Encode a single raw text into one vector."""
        return self._encode([normalize_text(text)])[0]

    # -------------------------------------------------------------------- API
    def add_intent(self, intent_name: str, sample_phrases: Iterable[str]) -> int:
        """Register *intent_name* with multilingual *sample_phrases*.

        Phrases are normalized (Thaana transliterated, punctuation flattened),
        deduplicated, encoded, and appended to the knowledge base.  Returns the
        number of newly added phrases.
        """
        intent_name = (intent_name or "").strip()
        if not intent_name:
            raise ValueError("intent_name must be a non-empty string")

        raw = [phrase for phrase in (sample_phrases or []) if phrase and phrase.strip()]
        if not raw:
            raise ValueError(f"intent {intent_name!r} needs at least one sample phrase")

        normalized = []
        for phrase in raw:
            candidate = normalize_text(phrase)
            if not candidate or (intent_name, candidate) in self._seen:
                continue
            self._seen.add((intent_name, candidate))
            normalized.append(candidate)

        if not normalized:
            return 0

        vectors = self._encode(normalized)
        self._vectors = vectors if self._vectors is None else np.vstack([self._vectors, vectors])
        self._intent_names.extend([intent_name] * len(normalized))
        self._phrases.extend(normalized)
        return len(normalized)

    def process_command(self, user_command: str) -> dict:
        """Resolve *user_command* to the closest registered intent.

        Returns a dictionary with the resolved ``intent`` (``None`` when the
        score is below ``threshold``), the confidence ``score``, the raw
        normalized text, and the top matches for explainability.
        """
        command = (user_command or "").strip()
        base = {
            "command": command,
            "normalized": "",
            "intent": None,
            "best_match": None,
            "confidence": 0.0,
            "threshold": self.threshold,
            "resolved": False,
            "matches": [],
        }
        if not command:
            base["error"] = "command must not be empty"
            return base

        normalized = normalize_text(command)
        if not normalized:
            base["error"] = "command contains no embeddable text"
            return base
        base["normalized"] = normalized

        if self._vectors is None or not self._intent_names:
            base["error"] = "no intents have been registered yet"
            return base

        query = self._encode([normalized])[0]
        scores = np.asarray(
            [self.cosine_similarity(query, vector) for vector in self._vectors],
            dtype=np.float32,
        )

        best_per_intent: dict[str, float] = defaultdict(float)
        for intent, score in zip(self._intent_names, scores):
            if float(score) > best_per_intent[intent]:
                best_per_intent[intent] = float(score)

        ranked = sorted(best_per_intent.items(), key=lambda item: item[1], reverse=True)
        best_intent, best_score = ranked[0] if ranked else (None, 0.0)
        resolved = bool(best_score >= self.threshold)

        base.update(
            {
                "intent": best_intent if resolved else None,
                "best_match": best_intent,
                "confidence": round(float(best_score), 4),
                "resolved": resolved,
                "matches": [
                    {"intent": intent, "score": round(float(score), 4)}
                    for intent, score in ranked[:3]
                ],
            }
        )

        # CECANP-0.54: feed the resolved route through the optimiser so
        # repeated commands move from full load (1.00) to cached mode (0.54).
        optimizer_intent = base["intent"] or f"unresolved:{normalized}"
        try:
            base["optimizer"] = self.optimizer.run_optimizer(
                intent=optimizer_intent,
                trigger_signal=command,
                path_steps=("semantic_match", "action_dispatch"),
            )
        except Exception as exc:  # the optimiser must never break commands
            base["optimizer"] = {"status": "error", "message": str(exc)}
        return base

    # ------------------------------------------------------- cognitive search
    def _ensure_concept_index(self, concepts: Sequence[dict], version: int) -> None:
        """(Re)build the concept vector cache when the store version changes."""
        with self._concept_lock:
            if version == self._concept_version:
                return
            try:  # Lazy import avoids a module-level circular import.
                from .cognitive_search import concept_document_text
            except ImportError:  # pragma: no cover - flat imports
                from cognitive_search import concept_document_text  # type: ignore

            docs = [dict(concept) for concept in concepts]
            vectors = None
            if docs:
                vectors = np.asarray(
                    self.encode_texts([concept_document_text(doc) for doc in docs]),
                    dtype=np.float32,
                )
                if vectors.ndim == 1:
                    vectors = vectors.reshape(1, -1)
            self._concept_docs = docs
            self._concept_vectors = vectors
            self._concept_version = version

    @staticmethod
    def _concept_answer(concept: dict) -> str:
        """Extractive answer for a matched concept (description if present)."""
        description = str(concept.get("description") or "").strip()
        if description:
            return description

        parts = [
            f"{concept.get('concept', '')} is a "
            f"{concept.get('category', 'General')} concept in the IACT knowledge base."
        ]
        keywords = [str(keyword) for keyword in concept.get("keywords") or []]
        if keywords:
            parts.append("Key associations: " + ", ".join(keywords) + ".")
        links = [str(link) for link in concept.get("links") or []]
        if links:
            parts.append("Directly linked concepts: " + ", ".join(links) + ".")
        return " ".join(parts)

    def cognitive_search(self, query: str, threshold: float = 0.4) -> dict:
        """Search the concept knowledge base and answer *query*.

        Spec-compatible response shape:

        - ``{"status": "success", "answer", "concept_name", "confidence"}``
          when the best match scores at or above ``threshold`` (default 0.4);
        - ``{"status": "not_found", "message", "confidence"}`` otherwise;
        - ``{"status": "error", "message"}`` when the knowledge base or query
          cannot be searched.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")

        command = (query or "").strip()
        if not command:
            return {"status": "error", "message": "query must not be empty"}

        if self.concept_provider is None:
            return {
                "status": "error",
                "message": "no concept provider is configured",
            }

        concepts, version = self.concept_provider()
        if not concepts:
            return {"status": "error", "message": "the knowledge base is empty"}

        normalized = normalize_text(command)
        if not normalized:
            return {
                "status": "error",
                "message": "query contains no searchable text",
            }

        self._ensure_concept_index(concepts, version)
        if self._concept_vectors is None or not self._concept_docs:
            return {"status": "error", "message": "the knowledge base is empty"}

        query_vector = self.encode_query(command)
        best_index = -1
        best_score = -1.0
        for index, vector in enumerate(self._concept_vectors):
            score = self.cosine_similarity(query_vector, vector)
            if score > best_score:
                best_score = score
                best_index = index

        confidence = round(max(best_score, 0.0), 2)
        if best_index < 0 or best_score < threshold:
            return {
                "status": "not_found",
                "message": "No concept in the cognitive network matches this query.",
                "confidence": confidence,
            }

        concept = self._concept_docs[best_index]
        return {
            "status": "success",
            "answer": self._concept_answer(concept),
            "concept_name": concept.get("concept", ""),
            "confidence": confidence,
        }

    @staticmethod
    def cosine_similarity(v1, v2) -> float:
        """Cosine similarity between two vectors, safe for zero vectors."""
        a = np.asarray(v1, dtype=np.float64).ravel()
        b = np.asarray(v2, dtype=np.float64).ravel()
        if a.size != b.size:
            raise ValueError("vectors must have the same dimension")
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denominator == 0.0:
            return 0.0
        return float(np.dot(a, b) / denominator)

    # Compatibility alias matching earlier spec examples.
    _cosine_similarity = cosine_similarity

    # ----------------------------------------------------------- introspection
    @property
    def intents(self) -> list[str]:
        return sorted(set(self._intent_names))

    @property
    def sample_counts(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for intent in self._intent_names:
            counts[intent] += 1
        return dict(counts)

    @property
    def model_loaded(self) -> bool:
        return self._model is not None or self._fallback is not None

    @property
    def using_fallback(self) -> bool:
        return self._fallback is not None and self._model is None

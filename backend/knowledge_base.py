"""Concept store and validation shared by the FastAPI and Termux servers."""

from __future__ import annotations

from threading import Lock

MAX_TITLE_LENGTH = 120
MAX_CATEGORY_LENGTH = 60
MAX_KEYWORDS = 50
MAX_KEYWORD_LENGTH = 60


def parse_concept(payload: dict) -> dict:
    """Validate and normalize an incoming concept payload.

    Accepts either a list of keywords or a comma-separated string so the API
    behaves the same no matter which server is running.
    """
    title = str(payload.get("title") or "").strip()
    if not title:
        raise ValueError("title must not be empty")
    if len(title) > MAX_TITLE_LENGTH:
        raise ValueError(f"title must be at most {MAX_TITLE_LENGTH} characters")

    category = str(payload.get("category") or "General").strip() or "General"
    if len(category) > MAX_CATEGORY_LENGTH:
        raise ValueError(f"category must be at most {MAX_CATEGORY_LENGTH} characters")

    raw_keywords = payload.get("keywords") or []
    if isinstance(raw_keywords, str):
        raw_keywords = raw_keywords.split(",")

    keywords: list[str] = []
    for item in raw_keywords:
        keyword = str(item).strip().lower()
        if keyword:
            keywords.append(keyword[:MAX_KEYWORD_LENGTH])

    return {
        "concept": title,
        "category": category,
        "keywords": list(dict.fromkeys(keywords))[:MAX_KEYWORDS],
    }


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

    def upsert(self, record: dict) -> dict:
        title = str(record.get("concept") or "").strip()
        if not title:
            raise ValueError("concept title must not be empty")
        stored = {
            "concept": title,
            "category": str(record.get("category") or "General").strip() or "General",
            "keywords": list(dict.fromkeys(record.get("keywords") or [])),
        }
        with self._lock:
            if title not in self._concepts:
                self._order.append(title)
            self._concepts[title] = stored
            self._version += 1
        return stored

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
                haystack = " ".join(
                    [record["concept"], record["category"], *record["keywords"]]
                ).lower()
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

"""Concept store and validation shared by the FastAPI and Termux servers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock
from typing import Union

MAX_TITLE_LENGTH = 120
MAX_CATEGORY_LENGTH = 60
MAX_KEYWORDS = 50
MAX_KEYWORD_LENGTH = 60
MAX_DESCRIPTION_LENGTH = 2000


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

    description = str(payload.get("description") or "").strip()
    if len(description) > MAX_DESCRIPTION_LENGTH:
        raise ValueError(
            f"description must be at most {MAX_DESCRIPTION_LENGTH} characters"
        )

    raw_keywords = payload.get("keywords") or []
    if isinstance(raw_keywords, str):
        raw_keywords = raw_keywords.split(",")

    keywords: list[str] = []
    for item in raw_keywords:
        keyword = str(item).strip().lower()
        if keyword:
            keywords.append(keyword[:MAX_KEYWORD_LENGTH])

    record = {
        "concept": title,
        "category": category,
        "keywords": list(dict.fromkeys(keywords))[:MAX_KEYWORDS],
    }
    if description:
        record["description"] = description
    return record


class ConceptStore:
    """Thread-safe concept knowledge base with optional JSON persistence.

    When ``data_file`` is set, every mutation is saved to disk (concept data
    only - vectors are rebuilt in memory by the search engines, which keeps the
    JSON lightweight) and :meth:`load_data` restores the store on startup.
    """

    def __init__(self, data_file: Union[str, os.PathLike, None] = None) -> None:
        self.data_file = str(data_file) if data_file else None
        self._concepts: dict[str, dict] = {}
        self._order: list[str] = []
        self._lock = Lock()
        self._version = 0

    @property
    def version(self) -> int:
        """Monotonic counter bumped on every mutation (used by search index)."""
        with self._lock:
            return self._version

    # ------------------------------------------------------------ persistence
    def save_data(self) -> int:
        """Write the knowledge base to :attr:`data_file` without vector data.

        Returns the number of saved concepts (0 when persistence is disabled).
        """
        if not self.data_file:
            return 0
        with self._lock:
            records = []
            for record in self._concepts.values():
                entry = {
                    "concept": record["concept"],
                    "category": record["category"],
                    "keywords": list(record["keywords"]),
                }
                if record.get("description"):
                    entry["description"] = record["description"]
                records.append(entry)

        path = Path(self.data_file)
        if path.parent != Path("."):
            path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(path.name + ".tmp")
        tmp_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(tmp_path, path)
        return len(records)

    def load_data(self) -> int:
        """Load concepts from :attr:`data_file` (replacing current contents).

        Vectors are deliberately not persisted; they are re-encoded in memory
        by the search engines the next time an index is built.  Returns the
        number of loaded concepts (0 when the file or persistence is missing).
        """
        if not self.data_file:
            return 0
        path = Path(self.data_file)
        if not path.is_file():
            return 0

        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError(f"{self.data_file} must contain a JSON list of concepts")

        concepts: dict[str, dict] = {}
        order: list[str] = []
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            record = parse_concept(
                {
                    "title": entry.get("concept") or entry.get("title"),
                    "category": entry.get("category"),
                    "keywords": entry.get("keywords"),
                    "description": entry.get("description"),
                }
            )
            concepts.setdefault(record["concept"], record)
            order.append(record["concept"])

        with self._lock:
            self._concepts = concepts
            self._order = [title for title in order if title in concepts]
            self._version += 1
        return len(concepts)

    def _autosave(self) -> None:
        if self.data_file:
            try:
                self.save_data()
            except OSError:  # pragma: no cover - disk errors are non-fatal
                pass

    def upsert(self, record: dict) -> dict:
        title = str(record.get("concept") or "").strip()
        if not title:
            raise ValueError("concept title must not be empty")
        description = str(record.get("description") or "").strip()
        if len(description) > MAX_DESCRIPTION_LENGTH:
            raise ValueError(
                f"description must be at most {MAX_DESCRIPTION_LENGTH} characters"
            )
        stored = {
            "concept": title,
            "category": str(record.get("category") or "General").strip() or "General",
            "keywords": list(dict.fromkeys(record.get("keywords") or [])),
        }
        if description:
            stored["description"] = description
        with self._lock:
            if title not in self._concepts:
                self._order.append(title)
            self._concepts[title] = stored
            self._version += 1
        self._autosave()
        return stored

    def delete(self, title: str) -> bool:
        with self._lock:
            if title not in self._concepts:
                return False
            del self._concepts[title]
            self._order.remove(title)
            self._version += 1
        self._autosave()
        return True

    def search(self, query: str) -> list[dict]:
        needle = query.strip().lower()
        if not needle:
            return []
        with self._lock:
            results = []
            for record in self._concepts.values():
                haystack = " ".join(
                    [
                        record["concept"],
                        record["category"],
                        record.get("description", ""),
                        *record["keywords"],
                    ]
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

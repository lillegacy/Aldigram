"""CECANP-0.54 energy optimisation and IACT-5SCP sensory anchoring.

CECANP-0.54 (Cognitive Energy Conservation and Associative Network Protocol)
---------------------------------------------------------------------------
Repeated command routes are cached:

* first run - full semantic calculation, ``compute_load == 1.00``;
* next runs before the threshold - partial context reuse, ``0.65``;
* from ``repetition_threshold`` (3-5) onward - cached execution, ``0.54``,
  freeing ``0.46`` of associative bandwidth.

Anomaly signals and decision nodes force a full recomputation and open a
five-second override window in which the route cache is bypassed.

IACT-5SCP (five-second cognitive protocol)
------------------------------------------
Two sensory inputs captured within a five-second window are synthesised into
an anchored concept.  The anchored concept is written straight to the shared
:class:`~backend.knowledge_base.ConceptStore` through the injected
``concept_writer`` (single source of truth: it lands in ``iact_data.json`` and
is immediately searchable).  A volatile five-second TTL lock only tracks the
synthesis window; it never duplicates durable concept state.  Pairs captured
after the five-second window are discarded.
"""

from __future__ import annotations

import math
import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
from time import monotonic
from typing import Callable, Iterable, Sequence

FULL_COMPUTE_LOAD = 1.00
PARTIAL_COMPUTE_LOAD = 0.65
CACHED_COMPUTE_LOAD = 0.54
FREED_BANDWIDTH = round(FULL_COMPUTE_LOAD - CACHED_COMPUTE_LOAD, 2)

SENSORY_WINDOW_SECONDS = 5.0
OVERRIDE_WINDOW_SECONDS = 5.0
DEFAULT_REPETITION_THRESHOLD = 3
MIN_REPETITION_THRESHOLD = 3
MAX_REPETITION_THRESHOLD = 5

MAX_ANCHORS = 100
MAX_MODALITY_LENGTH = 60
MAX_DESCRIPTION_LENGTH = 500
MAX_PATH_STEPS = 50
MAX_PATH_STEP_LENGTH = 200

ANOMALY_MARKERS = (
    "anomaly",
    "unexpected",
    "contradiction",
    "violation",
    "alert",
    "fault",
)
_ANOMALY_PATTERN = re.compile(
    r"\b(?:" + "|".join(ANOMALY_MARKERS) + r")\b", re.IGNORECASE
)

ConceptWriter = Callable[[dict], dict]


def repetition_threshold_from_env(
    default: int = DEFAULT_REPETITION_THRESHOLD,
    variable: str = "IACT_REPETITION_THRESHOLD",
) -> int:
    """Read the repetition threshold, falling back to the documented default."""
    raw = os.environ.get(variable)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    if MIN_REPETITION_THRESHOLD <= value <= MAX_REPETITION_THRESHOLD:
        return value
    return default


@dataclass(frozen=True)
class SensoryInput:
    """One modality of a sensory pair."""

    modality: str
    description: str
    focus_duration: float = 0.0

    def __post_init__(self) -> None:
        modality = str(self.modality or "").strip()
        description = str(self.description or "").strip()
        if not modality:
            raise ValueError("modality must not be empty")
        if len(modality) > MAX_MODALITY_LENGTH:
            raise ValueError(
                f"modality must be at most {MAX_MODALITY_LENGTH} characters"
            )
        if not description:
            raise ValueError("description must not be empty")
        if len(description) > MAX_DESCRIPTION_LENGTH:
            raise ValueError(
                f"description must be at most {MAX_DESCRIPTION_LENGTH} characters"
            )
        try:
            focus = float(self.focus_duration)
        except (TypeError, ValueError) as exc:
            raise ValueError("focus_duration must be a number") from exc
        if not math.isfinite(focus) or focus < 0.0:
            raise ValueError("focus_duration must be a non-negative number")

        object.__setattr__(self, "modality", modality)
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "focus_duration", focus)


@dataclass(frozen=True)
class AnchoredConcept:
    """Volatile five-second lock metadata for a persisted anchored concept."""

    concept: str
    synthesis: str
    modalities: tuple[str, ...]
    capture_delay_seconds: float
    ttl_seconds: float
    anchored_at: float
    expires_at: float

    def is_expired(self, now: float | None = None) -> bool:
        current = monotonic() if now is None else now
        return current >= self.expires_at

    def to_dict(self, now: float) -> dict:
        return {
            "concept": self.concept,
            "description": self.synthesis,
            "modalities": list(self.modalities),
            "capture_delay_seconds": round(self.capture_delay_seconds, 3),
            "ttl_seconds": round(self.ttl_seconds, 3),
            "expires_in_seconds": round(max(0.0, self.expires_at - now), 3),
            "expired": self.is_expired(now),
        }


@dataclass
class _RouteStats:
    runs: int = 0
    signal: str = ""


class IntegratedIACTSystem:
    """CECANP-0.54 route optimiser plus IACT-5SCP sensory synthesis.

    ``concept_writer`` is the single durable sink for anchored concepts; wire
    it to ``ConceptStore.upsert`` so anchors are persisted and searchable.
    The optimiser itself only owns route counters and volatile TTL locks.
    """

    def __init__(
        self,
        concept_writer: ConceptWriter | None = None,
        *,
        repetition_threshold: int = DEFAULT_REPETITION_THRESHOLD,
        clock: Callable[[], float] | None = None,
        sensory_window_seconds: float = SENSORY_WINDOW_SECONDS,
        override_window_seconds: float = OVERRIDE_WINDOW_SECONDS,
    ) -> None:
        if not (
            MIN_REPETITION_THRESHOLD
            <= int(repetition_threshold)
            <= MAX_REPETITION_THRESHOLD
        ):
            raise ValueError(
                "repetition_threshold must be between "
                f"{MIN_REPETITION_THRESHOLD} and {MAX_REPETITION_THRESHOLD}"
            )
        for name, value in (
            ("sensory_window_seconds", sensory_window_seconds),
            ("override_window_seconds", override_window_seconds),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a positive number")

        self.concept_writer = concept_writer
        self.repetition_threshold = int(repetition_threshold)
        self.sensory_window_seconds = float(sensory_window_seconds)
        self.override_window_seconds = float(override_window_seconds)

        self._clock = clock or monotonic
        self._lock = Lock()
        self._routes: dict[str, _RouteStats] = {}
        self._volatile: "OrderedDict[str, AnchoredConcept]" = OrderedDict()

        self._compute_load = FULL_COMPUTE_LOAD
        self._associative_bandwidth = 0.0
        self._override_until = 0.0
        self._override_kind: str | None = None

        self._cache_hits = 0
        self._anomalies = 0
        self._decision_overrides = 0
        self._anchored_total = 0
        self._discarded_total = 0

    # ------------------------------------------------------------- wiring
    def set_concept_writer(self, writer: ConceptWriter | None) -> None:
        """Attach the ConceptStore write path (single source of truth)."""
        self.concept_writer = writer

    # ----------------------------------------------------------- telemetry
    @property
    def compute_load(self) -> float:
        with self._lock:
            return round(self._compute_load, 2)

    @property
    def associative_bandwidth(self) -> float:
        with self._lock:
            return round(self._associative_bandwidth, 2)

    @property
    def cached_route_count(self) -> int:
        with self._lock:
            return self._cached_routes_locked()

    def telemetry(self) -> dict:
        now = self._clock()
        with self._lock:
            self._prune_locked(now)
            override_active = now < self._override_until
            return {
                "compute_load": round(self._compute_load, 2),
                "associative_bandwidth": round(self._associative_bandwidth, 2),
                "cached_routes": self._cached_routes_locked(),
                "tracked_routes": len(self._routes),
                "repetition_threshold": self.repetition_threshold,
                "cache_hits": self._cache_hits,
                "override": {
                    "active": override_active,
                    "kind": self._override_kind if override_active else None,
                    "remaining_seconds": round(
                        max(0.0, self._override_until - now), 3
                    ),
                },
                "protocol": {
                    "cecanp": "CECANP-0.54",
                    "iact_5scp": "IACT-5SCP",
                    "sensory_window_seconds": self.sensory_window_seconds,
                    "volatile_ttl_seconds": self.sensory_window_seconds,
                    "cached_compute_load": CACHED_COMPUTE_LOAD,
                    "freed_bandwidth": FREED_BANDWIDTH,
                },
                "counters": {
                    "anomalies": self._anomalies,
                    "decision_overrides": self._decision_overrides,
                    "anchored": self._anchored_total,
                    "discarded": self._discarded_total,
                    "active_anchors": len(self._volatile),
                },
            }

    def anchored_concepts(self, limit: int = 20) -> list[dict]:
        """Return unexpired volatile anchors (newest first)."""
        now = self._clock()
        with self._lock:
            self._prune_locked(now)
            anchors = list(self._volatile.values())[-max(0, limit) :]
            return [anchor.to_dict(now) for anchor in reversed(anchors)]

    # ------------------------------------------------------- route optimiser
    def run_optimizer(
        self,
        intent: str,
        trigger_signal: str,
        path_steps: Sequence[str] | None = None,
    ) -> dict:
        """Track a route and return the current CECANP-0.54 load.

        ``intent`` identifies the route, ``trigger_signal`` is the raw signal
        that triggered it and ``path_steps`` may contain a ``decision_node``
        step to force a full recomputation.
        """
        route_key = str(intent or "").strip().lower()
        if not route_key:
            raise ValueError("intent must not be empty")
        if len(route_key) > 200:
            raise ValueError("intent must be at most 200 characters")

        signal = str(trigger_signal or "").strip()
        if not signal:
            raise ValueError("trigger_signal must not be empty")
        if len(signal) > 500:
            raise ValueError("trigger_signal must be at most 500 characters")

        steps = self._coerce_path_steps(path_steps)
        now = self._clock()

        with self._lock:
            override_active = now < self._override_until
            anomaly = self._is_anomaly(signal, steps)
            decision_node = self._is_decision_node(steps)

            if anomaly or decision_node:
                kind = "anomaly_override" if anomaly else "decision_override"
                self._override_kind = kind
                self._override_until = now + self.override_window_seconds
                if anomaly:
                    self._anomalies += 1
                else:
                    self._decision_overrides += 1
                # The cached route is no longer trusted: force relearning.
                self._routes[route_key] = _RouteStats()
                self._set_load_locked(FULL_COMPUTE_LOAD)
                return self._result_locked(
                    intent=intent,
                    route_key=route_key,
                    signal=signal,
                    load=FULL_COMPUTE_LOAD,
                    execution_status=kind,
                    now=now,
                    anomaly=anomaly,
                    decision_node=decision_node,
                )

            if override_active:
                self._set_load_locked(FULL_COMPUTE_LOAD)
                return self._result_locked(
                    intent=intent,
                    route_key=route_key,
                    signal=signal,
                    load=FULL_COMPUTE_LOAD,
                    execution_status=self._override_kind or "override",
                    now=now,
                    anomaly=False,
                    decision_node=False,
                )

            stats = self._routes.setdefault(route_key, _RouteStats())
            stats.runs += 1
            stats.signal = signal

            if stats.runs >= self.repetition_threshold:
                load = CACHED_COMPUTE_LOAD
                execution_status = "cached"
                self._cache_hits += 1
            elif stats.runs == 1:
                load = FULL_COMPUTE_LOAD
                execution_status = "learning"
            else:
                load = PARTIAL_COMPUTE_LOAD
                execution_status = "partial"

            self._set_load_locked(load)
            return self._result_locked(
                intent=intent,
                route_key=route_key,
                signal=signal,
                load=load,
                execution_status=execution_status,
                now=now,
                anomaly=False,
                decision_node=False,
            )

    # -------------------------------------------------------- sensory pairs
    def synthesize_sensory_pair(
        self,
        item_a: SensoryInput,
        item_b: SensoryInput,
        capture_delay_seconds: float,
    ) -> dict:
        """Synthesise two sensory inputs inside the five-second window.

        The resulting anchored concept is written to the injected concept
        store (``iact_data.json`` when persistence is enabled).  A capture
        delay greater than :attr:`sensory_window_seconds` discards the pair.
        """
        first = self._coerce_sensory(item_a, "item_a")
        second = self._coerce_sensory(item_b, "item_b")
        try:
            delay = float(capture_delay_seconds)
        except (TypeError, ValueError) as exc:
            raise ValueError("capture_delay_seconds must be a number") from exc
        if isinstance(capture_delay_seconds, bool):
            raise ValueError("capture_delay_seconds must be a number")
        if not math.isfinite(delay) or delay < 0.0:
            raise ValueError("capture_delay_seconds must be a non-negative number")

        if delay > self.sensory_window_seconds:
            with self._lock:
                self._discarded_total += 1
            return {
                "status": "discarded",
                "reason": (
                    "capture delay exceeded the "
                    f"{self.sensory_window_seconds:.0f}-second sensory window"
                ),
                "capture_delay_seconds": round(delay, 3),
                "sensory_window_seconds": self.sensory_window_seconds,
                "concept": None,
                "anchored_concept": None,
            }

        if self.concept_writer is None:
            return {
                "status": "error",
                "message": "no concept store is attached; anchored concepts "
                "cannot be persisted",
                "capture_delay_seconds": round(delay, 3),
            }

        title, synthesis, keywords = self._build_anchor(first, second, delay)
        record = {
            "concept": title,
            "category": "Sensory Synthesis",
            "keywords": keywords,
            "description": synthesis,
        }
        try:
            stored = self.concept_writer(record)
        except (TypeError, ValueError) as exc:
            return {
                "status": "error",
                "message": f"could not persist anchored concept: {exc}",
            }

        now = self._clock()
        anchor = AnchoredConcept(
            concept=title,
            synthesis=synthesis,
            modalities=(first.modality, second.modality),
            capture_delay_seconds=delay,
            ttl_seconds=self.sensory_window_seconds,
            anchored_at=now,
            expires_at=now + self.sensory_window_seconds,
        )
        with self._lock:
            self._prune_locked(now)
            self._volatile[title] = anchor
            while len(self._volatile) > MAX_ANCHORS:
                self._volatile.popitem(last=False)
            self._anchored_total += 1
            anchors = [item.to_dict(now) for item in reversed(self._volatile.values())]

        return {
            "status": "anchored",
            "capture_delay_seconds": round(delay, 3),
            "sensory_window_seconds": self.sensory_window_seconds,
            "volatile_ttl_seconds": self.sensory_window_seconds,
            "concept": stored,
            "anchored_concept": anchor.to_dict(now),
            "anchored_concepts": anchors[:10],
        }

    # ------------------------------------------------------------ internals
    def _result_locked(
        self,
        *,
        intent: str,
        route_key: str,
        signal: str,
        load: float,
        execution_status: str,
        now: float,
        anomaly: bool,
        decision_node: bool,
    ) -> dict:
        stats = self._routes.get(route_key, _RouteStats())
        cached = stats.runs >= self.repetition_threshold
        override_active = now < self._override_until
        return {
            "status": "success",
            "intent": intent,
            "route_key": route_key,
            "trigger_signal": signal,
            "compute_load": round(load, 2),
            "associative_bandwidth": round(FULL_COMPUTE_LOAD - load, 2),
            "execution_status": execution_status,
            "cache_progress": {
                "runs": stats.runs,
                "threshold": self.repetition_threshold,
                "remaining": max(0, self.repetition_threshold - stats.runs),
                "cached": cached,
            },
            "cached_routes": self._cached_routes_locked(),
            "anomaly": anomaly,
            "decision_node": decision_node,
            "override": {
                "active": override_active,
                "kind": self._override_kind if override_active else None,
                "remaining_seconds": round(max(0.0, self._override_until - now), 3),
            },
        }

    def _set_load_locked(self, load: float) -> None:
        self._compute_load = round(load, 2)
        self._associative_bandwidth = round(FULL_COMPUTE_LOAD - load, 2)

    def _cached_routes_locked(self) -> int:
        return sum(
            1
            for stats in self._routes.values()
            if stats.runs >= self.repetition_threshold
        )

    def _prune_locked(self, now: float) -> None:
        expired = [
            name for name, anchor in self._volatile.items() if anchor.is_expired(now)
        ]
        for name in expired:
            del self._volatile[name]

    @staticmethod
    def _coerce_path_steps(path_steps: Sequence[str] | None) -> list[str]:
        if path_steps is None:
            return []
        if isinstance(path_steps, str):
            raise ValueError("path_steps must be a list of strings")
        steps: list[str] = []
        for step in path_steps:
            if not isinstance(step, str):
                raise ValueError("path_steps must contain strings")
            value = step.strip()
            if value:
                steps.append(value[:MAX_PATH_STEP_LENGTH])
        if len(steps) > MAX_PATH_STEPS:
            raise ValueError(f"path_steps must contain at most {MAX_PATH_STEPS} steps")
        return steps

    @staticmethod
    def _is_anomaly(signal: str, steps: Iterable[str]) -> bool:
        haystack = " ".join([signal, *steps])
        return bool(_ANOMALY_PATTERN.search(haystack))

    @staticmethod
    def _is_decision_node(steps: Iterable[str]) -> bool:
        for step in steps:
            normalized = (
                step.strip().lower().replace("-", "_").replace(" ", "_")
            )
            if (
                normalized == "decision"
                or normalized.startswith("decision_node")
                or normalized.startswith("decision:")
            ):
                return True
        return False

    @staticmethod
    def _coerce_sensory(value, label: str) -> SensoryInput:
        if isinstance(value, SensoryInput):
            return value
        if isinstance(value, dict):
            return SensoryInput(
                modality=value.get("modality"),
                description=value.get("description"),
                focus_duration=value.get("focus_duration", 0.0),
            )
        raise ValueError(f"{label} must be a SensoryInput")

    @staticmethod
    def _shorten(text: str, limit: int) -> str:
        text = text.strip()
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 1)].rstrip() + "…"

    def _build_anchor(
        self,
        item_a: SensoryInput,
        item_b: SensoryInput,
        delay: float,
    ) -> tuple[str, str, list[str]]:
        title = (
            f"{self._shorten(item_a.description, 52)} × "
            f"{self._shorten(item_b.description, 52)}"
        )[:120].strip()
        if not title:
            title = f"{item_a.modality} × {item_b.modality} synthesis"

        synthesis = (
            f"Cross-modal synthesis of {item_a.modality} input "
            f"({item_a.description}) and {item_b.modality} input "
            f"({item_b.description}), captured within {delay:.2f}s. "
            f"Focus durations: {item_a.focus_duration:.2f}s and "
            f"{item_b.focus_duration:.2f}s."
        )

        text = " ".join(
            [
                item_a.modality,
                item_b.modality,
                item_a.description,
                item_b.description,
                "sensory synthesis",
                "cross-modal",
                "anchored",
            ]
        )
        tokens = re.findall(r"[a-z0-9\u0780-\u07bf]+", text.lower())
        keywords: list[str] = []
        for token in tokens:
            if len(token) < 2:
                continue
            keyword = token[:60]
            if keyword not in keywords:
                keywords.append(keyword)
            if len(keywords) >= 50:
                break
        return title, synthesis, keywords

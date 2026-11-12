"""Default multilingual intent knowledge base and action mapping.

Shared by the FastAPI server (``backend.main``) and the native Termux server
(``backend.termux_server``).
"""

from __future__ import annotations

import os

try:  # Package import
    from .iact_engine import DEFAULT_THRESHOLD, MultiLingualIACTEngine
except ImportError:  # pragma: no cover - flat imports for direct execution
    from iact_engine import DEFAULT_THRESHOLD, MultiLingualIACTEngine  # type: ignore

# Each intent is registered with the same command expressed in English,
# Dhivehi (Latin transliteration) and Dhivehi (Thaana) so that all three script
# variants land close together in vector space.
DEFAULT_INTENTS: dict[str, list[str]] = {
    "add_concept": [
        "add a new concept",
        "create a concept and integrate it into the network",
        "register a new idea in the knowledge base",
        "au konsepteh ithuru kurey",
        "nuvata konsepteh add kurey",
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


def build_engine(engine: MultiLingualIACTEngine | None = None) -> MultiLingualIACTEngine:
    """Create the semantic engine and register the default intents."""
    if engine is None:
        threshold = float(os.environ.get("IACT_MATCH_THRESHOLD", DEFAULT_THRESHOLD))
        engine = MultiLingualIACTEngine(threshold=threshold)
    for intent_name, sample_phrases in DEFAULT_INTENTS.items():
        engine.add_intent(intent_name, sample_phrases)
    return engine

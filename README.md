# Aldigram — IACT Cognitive Dashboard

Integrative Associative Cognition (IACT) dashboard: a React frontend plus a
FastAPI backend with a **multi-lingual cognitive command engine**, **semantic
search**, **question answering** over the associative network, and the
**CECANP-0.54 / IACT-5SCP** energy-optimisation and sensory-anchoring
protocols.

Commands can be written in:

- **English**
- **Dhivehi (Thaana script)** — e.g. `ނެޓްވާކް ދައްކާ`
- **Dhivehi (Latin transliteration)** — e.g. `netvaak dhakkaa`

Thaana input is transliterated to a consistent Latin form, then every script is
embedded with
[`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
into one shared vector space. Incoming commands are matched to registered
intents by **cosine similarity**; a score of **≥ 0.45** resolves the intent and
triggers its workflow.

## Layout

```
IACTDashboard.jsx        React dashboard component
backend/
  main.py                FastAPI app (primary deployment)
  termux_server.py       Native Termux server (stdlib + numpy, same API)
  iact_engine.py         MultiLingualIACTEngine + fallback encoder
  cognitive_search.py    CognitiveSearchEngine (semantic search + QA)
  cognitive_optimizer.py CECANP-0.54 optimiser + IACT-5SCP synthesis
  intents.py             Default intents + action mapping
  knowledge_base.py      Concept store + validation
  multilingual.py        Thaana transliteration & text normalization
  requirements.txt
  tests/test_engine.py
  tests/test_cognitive_search.py
  tests/test_cognitive_optimizer.py
  tests/test_termux_server.py
  tests/test_fastapi_parity.py
```

## Run the backend

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --port 8000
```

Open http://127.0.0.1:8000/docs for the interactive API docs.

> **Termux note:** PyTorch has no native Android wheels, so
> `sentence-transformers` normally needs a Linux environment (e.g. a Debian
> proot container or a PC/server). Plain Termux can still run the **native
> Termux server** below, which uses only the standard library and numpy and
> automatically enables the lightweight fallback encoder on Android.

### Native Termux

FastAPI needs `pydantic-core` (Rust) and the transformer needs PyTorch; neither
has native Termux wheels. `backend.termux_server` exposes the same JSON API
without those dependencies:

```bash
pkg install python-numpy
cd Aldigram
python -m backend.termux_server --port 8000
# LAN access from another device:
# python -m backend.termux_server --host 0.0.0.0 --port 8000
```

- Same routes and response shapes as the FastAPI backend
- On Android the fallback encoder turns on automatically; no env var needed
- Set `IACT_ALLOW_FALLBACK=0` to force strict transformer loading
- The FastAPI app remains the primary backend for servers, PCs and proot;
  this server is purely additive for environments where FastAPI cannot install
- Swagger `/docs` is only available from the FastAPI server

The fallback keeps search, QA and intent resolution functional but is **not**
the multilingual transformer — use the real model for production quality.

### Configuration

| Variable | Default | Meaning |
|---|---|---|
| `IACT_MATCH_THRESHOLD` | `0.45` | Minimum cosine similarity to resolve a command intent |
| `IACT_SEARCH_MIN_SCORE` | `0.20` | Minimum similarity for semantic search to count as resolved |
| `IACT_QA_MIN_SCORE` | `0.25` | Minimum similarity for a confident QA answer |
| `IACT_DATA_FILE` | `iact_data.json` | Concept persistence file (no vectors; rebuilt on load) |
| `IACT_REPETITION_THRESHOLD` | `3` | Runs before a route is cached (allowed range: 3–5) |
| `IACT_ALLOW_FALLBACK` | `0` (`1` on Android) | Use char-n-gram encoder if the model can't load |
| `IACT_CORS_ORIGINS` | `*` | Comma-separated allowed origins |

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health/` | Service, engine and search index status |
| `GET` | `/network/` | Concept network with keyword links |
| `POST` | `/concepts/` | Add/update a concept |
| `GET` | `/concepts/{title}` | Fetch one concept with links |
| `DELETE` | `/concepts/{title}` | Remove a concept |
| `GET` | `/concepts/search/?q=...` | Substring search over concepts |
| `GET` | `/intents/` | Registered intents and sample counts |
| `POST` | `/commands/` | Resolve a natural-language command |
| `POST` | `/api/search/` | Cognitive search over the knowledge base (spec endpoint) |
| `POST` | `/search/` | Semantic search over concepts, links and context |
| `POST` | `/qa/` | Answer a question from the associative network |
| `POST` | `/optimizer/run/` | Track a route through CECANP-0.54 (load/bandwidth/cache) |
| `POST` | `/optimizer/sensory-pair/` | IACT-5SCP five-second sensory synthesis and anchoring |

### Example — command resolution

```bash
curl -s http://127.0.0.1:8000/commands/ \
  -H 'Content-Type: application/json' \
  -d '{"command": "ނެޓްވާކް ދައްކާ"}'
```

```json
{
  "command": "ނެޓްވާކް ދައްކާ",
  "normalized": "netvaak dhakaa",
  "intent": "show_network",
  "best_match": "show_network",
  "confidence": 0.71,
  "threshold": 0.45,
  "resolved": true,
  "matches": [{"intent": "show_network", "score": 0.71}],
  "action": {"action": "refresh_network_view", "description": "..."}
}
```

Unresolved commands return `"intent": null`, `"resolved": false` and an
`action` of `clarify`.

### Example — cognitive search (`/api/search/`)

```bash
curl -s http://127.0.0.1:8000/api/search/ \
  -H 'Content-Type: application/json' \
  -d '{"query": "what is artificial intelligence"}'
```

```json
{
  "status": "success",
  "answer": "Artificial Intelligence is a Technology concept in the IACT knowledge base. Key associations: ai, neural, cognition. Directly linked concepts: Neural Networks, Cognitive Science.",
  "concept_name": "Artificial Intelligence",
  "confidence": 0.67
}
```

Below the 0.4 threshold the response is
`{"status": "not_found", "message": "...", "confidence": ...}`; an empty
knowledge base returns `{"status": "error", "message": "..."}`.

Concept data is persisted to `iact_data.json` (vectors excluded) on every
change and reloaded on startup; the search indexes re-encode automatically.

### CECANP-0.54 route optimisation

Repeated routes move through three execution levels:

| Runs | Execution | Compute load | Associative bandwidth |
|---|---|---|---|
| 1 | `learning` | `1.00` | `0.00` |
| 2 … threshold−1 | `partial` | `0.65` | `0.35` |
| ≥ threshold (3–5) | `cached` | `0.54` | `0.46` |

`process_command` feeds every route through the optimiser, so `/commands/`
responses include an `optimizer` block with the current mode. An anomaly
signal (`anomaly`, `unexpected`, `contradiction`, `violation`, `alert`,
`fault`) or a `decision_node` path step forces full recomputation and opens a
**five-second override window** that bypasses the cache
(`anomaly_override` / `decision_override`). After the window the route
relearns from `1.00`.

```bash
curl -s http://127.0.0.1:8000/optimizer/run/ \
  -H 'Content-Type: application/json' \
  -d '{"intent": "show_network", "trigger_signal": "show the network", "path_steps": ["semantic_match"]}'
```

```json
{
  "status": "success",
  "intent": "show_network",
  "route_key": "show_network",
  "compute_load": 0.54,
  "associative_bandwidth": 0.46,
  "execution_status": "cached",
  "cache_progress": {"runs": 3, "threshold": 3, "remaining": 0, "cached": true},
  "cached_routes": 1,
  "anomaly": false,
  "decision_node": false,
  "override": {"active": false, "kind": null, "remaining_seconds": 0.0}
}
```

`GET /health/` exposes the live `compute_load`, `associative_bandwidth` and
`cached_routes` values (also nested under `optimizer`).

### IACT-5SCP sensory anchoring

Two sensory inputs captured within **5.0 seconds** are synthesised into an
anchored concept and written directly into the shared `ConceptStore` — single
source of truth, so the anchor is persisted to `iact_data.json` and is
immediately searchable through `/search/`, `/qa/` and `/api/search/`. The
volatile TTL lock only tracks the five-second synthesis window; the anchored
concept itself remains durable. Pairs captured after 5.0 seconds are
discarded and never touch the store.

```bash
curl -s http://127.0.0.1:8000/optimizer/sensory-pair/ \
  -H 'Content-Type: application/json' \
  -d '{
    "item_a": {"modality": "visual", "description": "red square", "focus_duration": 0.8},
    "item_b": {"modality": "auditory", "description": "low tone", "focus_duration": 1.2},
    "capture_delay_seconds": 4.5
  }'
```

A successful synthesis returns `"status": "anchored"` with the persisted
`concept` and its `anchored_concept` TTL metadata; a delay above the window
returns `"status": "discarded"`.

### Example — semantic search

```bash
curl -s http://127.0.0.1:8000/search/ \
  -H 'Content-Type: application/json' \
  -d '{"query": "What is artificial intelligence?", "top_k": 3}'
```

```json
{
  "query": "What is artificial intelligence?",
  "normalized": "what is artificial intelligence",
  "threshold": 0.2,
  "resolved": true,
  "confidence": 0.7145,
  "results": [
    {
      "concept": "Artificial Intelligence",
      "category": "Technology",
      "keywords": ["ai", "neural", "cognition"],
      "links": ["Neural Networks", "Cognitive Science"],
      "score": 0.7145
    }
  ]
}
```

### Example — cognitive QA

```bash
curl -s http://127.0.0.1:8000/qa/ \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is artificial intelligence?", "top_k": 2}'
```

```json
{
  "question": "What is artificial intelligence?",
  "resolved": true,
  "confidence": 0.7145,
  "answer": "Artificial Intelligence is a Technology concept in the IACT knowledge base. Key associations: ai, neural, cognition. Directly linked concepts: Neural Networks, Cognitive Science. Related concepts retrieved: Cognitive Science, Neural Networks. Semantic match confidence: 0.71.",
  "evidence": [
    {
      "concept": "Artificial Intelligence",
      "category": "Technology",
      "keywords": ["ai", "neural", "cognition"],
      "links": ["Neural Networks", "Cognitive Science"],
      "score": 0.7145
    }
  ],
  "relationships": [
    {
      "source": "Artificial Intelligence",
      "target": "Neural Networks",
      "shared_keywords": ["neural"],
      "type": "association"
    },
    {
      "source": "Artificial Intelligence",
      "target": "Cognitive Science",
      "shared_keywords": ["cognition"],
      "type": "association"
    }
  ],
  "subgraph": {
    "nodes": [
      {"id": "Artificial Intelligence", "category": "Technology", "score": 0.7059},
      {"id": "Cognitive Science", "category": "Science", "score": 0.5509}
    ],
    "edges": [
      {
        "source": "Artificial Intelligence",
        "target": "Cognitive Science",
        "shared_keywords": ["cognition"],
        "type": "association"
      }
    ]
  }
}
```

`evidence`, `relationships` and `subgraph` are populated from the retrieved
network context so the dashboard can render the matching subset directly.

### Example — concept network

```bash
curl -s http://127.0.0.1:8000/concepts/ \
  -H 'Content-Type: application/json' \
  -d '{"title": "Artificial Intelligence", "category": "Technology", "keywords": ["ai", "neural", "cognition"]}'

curl -s http://127.0.0.1:8000/network/
```

## Run the frontend

A Vite scaffold lives in `frontend/` and imports the component at the
repository root (`../IACTDashboard.jsx`) through the `@iact` alias. Styling uses
the Tailwind CDN for quick previews.

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. The dashboard expects the API at
`http://127.0.0.1:8000`.

## Deploy to a Hugging Face Space

The repo includes a Docker Space setup (full app: FastAPI serves the built
React dashboard on port 7860) and a deployment script:

```bash
pip install -U huggingface_hub
hf auth login
scripts/deploy_hf_space.sh <hf-user>/<space-name> --private
```

The script creates the Space if needed, stages `backend/`, `frontend/`,
`Dockerfile` and the Space README, and uploads everything. The Space builds a
CPU-only PyTorch image with the multilingual model baked in.

Notes:

- The app runs at `https://<hf-user>-<space-name>.hf.space`
- Private Spaces require a Hugging Face login to open
- A Hugging Face account needs available Space quota (`cpu-basic`); if the
  runtime shows `Quota exceeded`, verify the account / check billing settings
- Building the image takes several minutes on first deploy

## Tests

```bash
python -m pytest backend/tests -q
```

The tests use deterministic stub encoders and do not download the transformer
model. They cover multilingual normalization, intent resolution, semantic
search ranking, QA answer synthesis, subgraph context, index invalidation,
JSON persistence round-trips, the spec-compatible `cognitive_search` contract
(`/api/search/`), CECANP-0.54 cache/anomaly/decision-node transitions,
IACT-5SCP anchoring and TTL expiry, and HTTP-level parity between the FastAPI
routes and the native Termux server. The FastAPI parity test starts uvicorn
(and is skipped automatically on Termux, where uvicorn is unavailable).

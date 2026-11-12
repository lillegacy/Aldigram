# Aldigram — IACT Cognitive Dashboard

Integrative Associative Cognition (IACT) dashboard: a React frontend plus a
FastAPI backend with a **multi-lingual cognitive command engine**, **semantic
search** and **question answering** over the associative network.

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
  main.py                FastAPI app, concept network, command/search/QA endpoints
  iact_engine.py         MultiLingualIACTEngine + fallback encoder
  cognitive_search.py    CognitiveSearchEngine (semantic search + QA)
  multilingual.py        Thaana transliteration & text normalization
  requirements.txt
  tests/test_engine.py
  tests/test_cognitive_search.py
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
> proot container or a PC/server). On plain Termux you can still run the API
> with the lightweight fallback encoder:
>
> ```bash
> export IACT_ALLOW_FALLBACK=1
> uvicorn backend.main:app --port 8000
> ```
>
> The fallback keeps the endpoints and intent resolution functional but is
> **not** the multilingual transformer — use the real model for production.

### Configuration

| Variable | Default | Meaning |
|---|---|---|
| `IACT_MATCH_THRESHOLD` | `0.45` | Minimum cosine similarity to resolve a command intent |
| `IACT_SEARCH_MIN_SCORE` | `0.20` | Minimum similarity for semantic search to count as resolved |
| `IACT_QA_MIN_SCORE` | `0.25` | Minimum similarity for a confident QA answer |
| `IACT_ALLOW_FALLBACK` | `0` | Use char-n-gram encoder if the model can't load |
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
| `POST` | `/search/` | Semantic search over concepts, links and context |
| `POST` | `/qa/` | Answer a question from the associative network |

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

## Tests

```bash
python -m pytest backend/tests -q
```

The tests use deterministic stub encoders and do not download the transformer
model. They cover multilingual normalization, intent resolution, semantic
search ranking, QA answer synthesis, subgraph context and index invalidation.

The tests use a stub encoder and do not download the transformer model.

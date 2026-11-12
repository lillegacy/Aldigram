# Aldigram — IACT Cognitive Dashboard

Integrative Associative Cognition (IACT) dashboard: a React frontend plus a
FastAPI backend with a **multi-lingual cognitive command engine**.

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
  main.py                FastAPI app, concept network, command endpoint
  iact_engine.py         MultiLingualIACTEngine + fallback encoder
  multilingual.py        Thaana transliteration & text normalization
  requirements.txt
  tests/test_engine.py
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
| `IACT_MATCH_THRESHOLD` | `0.45` | Minimum cosine similarity to resolve an intent |
| `IACT_ALLOW_FALLBACK` | `0` | Use char-n-gram encoder if the model can't load |
| `IACT_CORS_ORIGINS` | `*` | Comma-separated allowed origins |

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health/` | Service and engine status |
| `GET` | `/network/` | Concept network with keyword links |
| `POST` | `/concepts/` | Add/update a concept |
| `DELETE` | `/concepts/{title}` | Remove a concept |
| `GET` | `/concepts/search/?q=...` | Search concepts |
| `GET` | `/intents/` | Registered intents and sample counts |
| `POST` | `/commands/` | Resolve a natural-language command |

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

### Example — concept network

```bash
curl -s http://127.0.0.1:8000/concepts/ \
  -H 'Content-Type: application/json' \
  -d '{"title": "Artificial Intelligence", "category": "Technology", "keywords": ["ai", "neural", "cognition"]}'

curl -s http://127.0.0.1:8000/network/
```

## Run the frontend

The dashboard component expects the API at `http://127.0.0.1:8000`. Drop
`IACTDashboard.jsx` into a Vite + React + Tailwind project:

```bash
npm create vite@latest iact-dashboard -- --template react
cd iact-dashboard
npm install
npm install -D tailwindcss @tailwindcss/vite
cp ../IACTDashboard.jsx src/
```

Then render `<IACTDashboard />` from `src/App.jsx`, add
`@import "tailwindcss";` to `src/index.css`, add the Tailwind Vite plugin, and
run `npm run dev`.

## Tests

```bash
python -m pytest backend/tests -q
```

The tests use a stub encoder and do not download the transformer model.

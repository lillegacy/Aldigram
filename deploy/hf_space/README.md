---
title: Aldigram IACT Dashboard
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Aldigram — IACT Cognitive Dashboard

Integrative Associative Cognition dashboard with a multi-lingual command
engine: English, Dhivehi (Thaana) and Dhivehi (Latin transliteration) are
mapped into one shared vector space and matched by cosine similarity.

- **Semantic search** over concept nodes and relationships
- **Question answering** with structured evidence and a network subgraph
- **Dashboards** for browsing the associative network

This Space runs the full app (React UI served by FastAPI) in a Docker
container using `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.

- Source: https://github.com/lillegacy/Aldigram
- API docs: `/docs`
- Health check: `/health/`

Example:

```bash
curl -s https://<this-space>.hf.space/qa/ \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is artificial intelligence?"}'
```

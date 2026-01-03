# RASA V2 (4-Tier Clean Deploy)

This repo runs a chatbot UI + APIs + AI orchestration + Vector/RAG as four separate services, while keeping the same browser-facing routes (the browser only talks to the Web gateway).

If you're new to the repo, start with: `startHere.md`.

## What Runs Where (4 Tiers)

- Web (`Web/`)
  - Serves templates/static (browser UI)
  - Owns auth routes (`/auth/*`)
  - Proxies browser paths without changing them:
    - `/chat`, `/upload`, `/clear_chat`, `/api/test_model` -> AI
    - `/api/*`, `/health`, `/healthz`, `/admin/api/*` -> API
- API (`API/`)
  - Non-LLM JSON endpoints
  - Owns the single SQL DB file: `API/data/app.db`
  - Owns shared non-vector state: `API/state/uploads/`, `API/state/documents/`
- AI (`AI/`)
  - Owns `/chat` logic (HybridRouter/tool routing/MCP/code execution/safety)
  - Calls Vector over HTTP for RAG
- Vector (`Vector/`)
  - FastAPI secure RAG service (`Vector/secure_rag`)
  - Owns vector persistence: `Vector/data/vectordb/`

## Repo Layout (Root)

Only these top-level entries should exist:

- `Web/`, `API/`, `AI/`, `Vector/`
- `.venv/`
- `README.md`, `startHere.md`, `requirements.txt`

## Quickstart (Windows)

1) Install deps:

```powershell
cd "C:\Code\RASA V2"
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

2) Start services (Vector -> AI -> API -> Web) in 4 terminals:

```powershell
.\.venv\Scripts\python.exe Vector\app.py
.\.venv\Scripts\python.exe AI\app.py
.\.venv\Scripts\python.exe API\app.py
.\.venv\Scripts\python.exe Web\app.py
```

3) Open the UI:

- `http://127.0.0.1:5001`

## Default Ports

- Web: `http://127.0.0.1:5001` (open this in the browser)
- API: `http://127.0.0.1:5002`
- AI: `http://127.0.0.1:5003`
- Vector: `http://127.0.0.1:8001`

## Configuration

Each tier has its own env file:

- `Web/.env`
- `API/.env`
- `AI/.env`
- `Vector/.env`

Important rules:

- `SECRET_KEY` must be identical across `Web/.env`, `API/.env`, `AI/.env` (sessions/cookies depend on this).
- `DATABASE_URL` must point to the same DB file across Web/API/AI (default points to `API/data/app.db`).
- LLM and tool API keys belong only in `AI/.env`.
- Vector index/state belongs only in `Vector/`.

### Required Keys (by feature)

You can run the UI without all keys, but some tools will fail/return errors without them.

- LLM calls: `GROQ_API_KEY` (or set `PREFER_LLM_PROVIDER=bedrock` and configure AWS creds)
- Weather tool: `WEATHER_API_KEY`
- News tool: `NEWS_API_KEY`
- Web search / enrichment (used by some tools): `TAVILY_API_KEY`, `GOOGLE_CSE_API_KEY`, `GOOGLE_CSE_CX`

### MCP Tools

Tool servers are configured here:

- `AI/mcp_servers.json`
- `AI/mcp_servers/`

## State and Logs

- DB: `API/data/app.db`
- Uploads: `API/state/uploads/`
- Documents/KB sources: `API/state/documents/`
- Vector index: `Vector/data/vectordb/`
- Logs:
  - Web: `Web/logs/`
  - API: `API/logs/`
  - AI: `AI/logs/`
  - Vector: `Vector/logs/`

## Feature Checklist (What To Verify)

These are the common end-to-end features to validate from the Web UI:

- Upload: text file
- Upload: image / PDF
- Preferred model (LLM adapter): `/api/models`, `/api/model_preference`
- Contact form: "Collect my info", "Show my info" (adaptive card)
- Weather tool
- Survey tool
- News tool
- Normal LLM answers
- Brochure tool (adaptive card)
- Dynamic adaptive cards (adaptive card)
- Leave calculator (adaptive card)
- Secure RAG (should deny confidential access)
- Safety guard ("How to make a bomb?") should be blocked/refused
- KnowledgeBase query
- SQL Query (requires Postgres; optional)

## Troubleshooting (Common)

- Login/auth issues:
  - Verify `SECRET_KEY` matches across Web/API/AI
  - Verify `DATABASE_URL` points to the same DB file in all three
- `/chat` slow/hanging:
  - Check AI logs: `AI/logs/ai.err.log`, `AI/logs/tool_calls.log`
  - Missing LLM/tool API keys can cause tool calls to fail or stall
- Upload errors:
  - Check `Web/logs/web.err.log` and `AI/logs/ai.err.log`
- KB/RAG issues:
  - Confirm Vector is up: `http://127.0.0.1:8001/health`
  - Check `Vector/logs/vector.err.log`
- SQL tool fails:
  - Expected unless Postgres is running and reachable at the host/port configured in `AI/.env`

## Security Notes

- Do not commit real API keys or cloud credentials.
- For local development, keep secrets only in tier `.env` files (no root `.env`).

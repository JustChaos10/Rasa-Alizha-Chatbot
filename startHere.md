# Start Here (No Surprises)

This repo runs as four services (plus `.venv`) while keeping the same browser-facing routes. The browser talks only to the Web gateway.

## 0) Prereqs

- Python 3.11+ recommended
- Windows PowerShell (commands below) or equivalent terminal
- Make sure these ports are free: `5001`, `5002`, `5003`, `8001`

## 1) Create/Use Virtualenv

If `.venv/` already exists, you can skip creation.

```powershell
cd "C:\Code\RASA V2"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 2) Configure Environment

Edit these files:

- `Web/.env`
- `API/.env`
- `AI/.env`
- `Vector/.env`

Required to avoid auth/session issues:

- Set the same `SECRET_KEY` in `Web/.env`, `API/.env`, `AI/.env`
- Ensure all three point at the same `DATABASE_URL` (default uses `API/data/app.db`)

Feature-specific keys (typically in `AI/.env`):

- Weather tool: `WEATHER_API_KEY`
- News tool: `NEWS_API_KEY`
- Web search / enrichment: `TAVILY_API_KEY`, `GOOGLE_CSE_API_KEY`, `GOOGLE_CSE_CX`
- LLM provider: `GROQ_API_KEY` (or Bedrock config if you use `PREFER_LLM_PROVIDER=bedrock`)

## 3) Start Services (Order Matters)

Startup order:

1) Vector
2) AI
3) API
4) Web

Open 4 terminals in the repo root and run:

```powershell
cd "C:\Code\RASA V2"

# Terminal 1 (Vector)
.\.venv\Scripts\python.exe Vector\app.py

# Terminal 2 (AI)
.\.venv\Scripts\python.exe AI\app.py

# Terminal 3 (API)
.\.venv\Scripts\python.exe API\app.py

# Terminal 4 (Web gateway + UI)
.\.venv\Scripts\python.exe Web\app.py
```

## 4) Health Checks

```powershell
curl.exe -i http://127.0.0.1:5001/health
curl.exe -i http://127.0.0.1:5002/health
curl.exe -i http://127.0.0.1:8001/health
```

## 5) Use The App

1) Open `http://127.0.0.1:5001`
2) Register/login
3) Try:
   - "What is a dog?"
   - "Collect my info" then "Show my info"
   - "What is the weather in bangalore like today?"
   - "Give me top headlines about AI"
   - "Create a brochure on Coffee"
   - "Can i take a leave ?"
   - "Who gave the budget speech?"

Important:

- Always use the Web gateway (`http://127.0.0.1:5001`) in the browser. Don't browse to the AI/API ports directly.

## 6) Where To Look When Something Breaks

- Web logs: `Web/logs/web.err.log`
- API logs: `API/logs/api.err.log`
- AI logs: `AI/logs/ai.err.log`, `AI/logs/tool_calls.log`
- Vector logs: `Vector/logs/vector.err.log`

Common gotchas:

- Auth/session problems: `SECRET_KEY` mismatch across tiers
- SQL Query tool: expected to fail unless Postgres is running and `AI/.env` points to it

## 7) Stop Services

In each terminal, press `Ctrl+C`.

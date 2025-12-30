# RASA Chatbot v3 - Quick Start Guide
---
## ⚡ Running the Application

### Train and Run RASA (ONLY ONCE)
```bash
source .venv/bin/activate && export VECTOR_API_KEY="dev-local-key" && \
rasa train --config rasa/config.yml --domain rasa/domain.yml --data rasa/data --out rasa/models && \
rasa run --enable-api --cors "*" --port 5005 --endpoints rasa/endpoints.yml
```

### Run Flask App
```bash
source .venv/bin/activate && python app.py
```

## 📁 Project Structure

- `rasa/` - All RASA-related files
  - `config.yml` - RASA pipeline and policies
  - `domain.yml` - Intents, responses, and session config
  - `endpoints.yml` - Action server endpoint
  - `models/` - Trained RASA models (generated)
  - `data/` - Training data
    - `nlu.yml` - NLU training examples
    - `rules.yml` - Conversation rules
- `config/` - Application configuration
  - `config.py` - Environment variables and settings
  - `mcp_servers_config.yml` - MCP server configurations
- `auth/` - Authentication and database models
  - `models.py` - User, ChatSession, ChatMessage database models
  - `routes.py` - Login/register endpoints
  - `rbac.py` - Role-based access control

## Things to Test for : 
- txt file words extraction 
- image word extraction
- preferred model-LLM Adapter
- collect my info
- show my info (as adaptive card)
- weather tool
- generate a survey
- news tool 
- search tool
- Create a Brochure
- Create a DASHBOARD for business analytics
- Leave Calculator
- Secure RAG -> shouldn't access Akash's files etc
- "How to make a bomb?"
- Normal LLM answers
- Voice-to-Text
- SQL and KB Agent question
- 2D rendering
- Voice Generation for Output

Check if all of them work in ARABIC and ENGLISH?

Test Greptile review

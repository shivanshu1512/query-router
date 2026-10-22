# Query Router 🧭

**Cost-optimized LLM query routing** — automatically routes queries to cheap vs expensive models based on predicted complexity.

> Save 40-60% on LLM API costs while preserving quality on hard queries.

## The Problem

Every query to GPT-4o costs ~17x more than GPT-4o-mini. But most queries (factual recall, simple formatting, basic math) don't need the expensive model. This project:

1. **Measures** the quality gap between cheap and expensive models on 200 diverse queries
2. **Trains** a lightweight classifier to predict which queries need the expensive model
3. **Deploys** the router as a FastAPI endpoint that intercepts API calls and routes optimally

## Key Metrics (to be filled after pipeline run)

| Metric | Value |
|---|---|
| Always-large cost (200 queries) | `TBD` |
| Always-small cost (200 queries) | `TBD` |
| Oracle cost (200 queries) | `TBD` |
| Router accuracy (test set) | `TBD` |
| Router F1 score | `TBD` |
| Cost savings vs always-large | `TBD` |
| Quality preserved | `TBD` |

## Architecture

```
User Query → [Router Classifier] → Small Model (GPT-4o-mini)  ← 80% of queries
                                  → Large Model (GPT-4o)       ← 20% of queries
```

**No LangChain.** Raw `httpx` calls to OpenAI/Anthropic APIs. Every token counted. Every cent tracked.

## Project Structure

```
query-router/
├── src/query_router/
│   ├── pricing.py      # Per-model cost registry (source of truth)
│   ├── schemas.py      # Pydantic models for all data structures
│   ├── llm_client.py   # Raw httpx API client (OpenAI + Anthropic)
│   ├── dataset.py      # 200 curated evaluation queries
│   ├── judge.py        # Pairwise LLM-as-a-Judge with debiasing
│   ├── oracle.py       # Oracle pipeline (Week 1 deliverable)
│   ├── router.py       # Trained routing classifier (sklearn)
│   └── server.py       # FastAPI deployment
├── tests/
│   └── test_core.py    # Offline tests (no API keys needed)
├── data/
│   ├── raw/            # Raw data files
│   └── results/        # Pipeline outputs (JSONL + stats)
├── models/             # Saved router models (.pkl)
├── pyproject.toml      # Dependencies + config
└── .env.example        # API key template
```

## Quick Start

### 1. Setup

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install
pip install -e ".[dev]"

# Copy env file and add your API keys
copy .env.example .env
# Edit .env with your OPENAI_API_KEY
```

### 2. Run Tests (no API key needed)

```bash
pytest tests/ -v
```

### 3. Tool Calling Demo (Prerequisite / Step 4)

```bash
# Offline simulation or live execution with API keys
python examples/calculator_tool.py
```

### 4. Run Oracle Pipeline (needs API key)

```bash
# Test with 5 queries first (~$0.05)
python -m query_router.oracle --max-queries 5

# Full run — 200 queries (~$2-3 total)
python -m query_router.oracle
```

### 5. Train Router

```bash
python -m query_router.router --oracle-results data/results/oracle_results_*.jsonl
```

### 6. Deploy API

```bash
pip install -e ".[serve]"
uvicorn query_router.server:app --reload --port 8000
```

Then visit `http://localhost:8000/docs` for interactive API docs.

## Technical Approach

### Oracle Labels (Week 1)
Based on [LLM-as-a-Judge (Zheng et al., 2023)](https://arxiv.org/abs/2306.05685):
- Run both models on every query
- Use GPT-4o as judge to compare quality
- Oracle decision: route to small model if quality is equal or better

### Routing Classifier (Week 2)
Inspired by [FrugalGPT (Chen, Zaharia, Stoica, 2023)](https://arxiv.org/abs/2305.05176):
- Features: TF-IDF + hand-crafted (query length, code keywords, complexity indicators)
- Model: GradientBoosting (works well on small datasets)
- Threshold tuning: trade off cost savings vs quality

### Cost Engineering
- Every API call tracked with exact token counts and costs
- Pricing table is a single source of truth (`pricing.py`)
- Results include cost comparison against always-large and always-small baselines

## References

- [FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance](https://arxiv.org/abs/2305.05176)
- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)
- [Anthropic: Building Effective Agents](https://anthropic.com/engineering/building-effective-agents)

## Author

**Shivanshu Shukla** — B.Tech IT, KIET Ghaziabad
- GitHub: [shivanshu1512](https://github.com/shivanshu1512)

## License

MIT

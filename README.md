<p align="center">
  <h1 align="center">🧭 Query Router</h1>
  <p align="center">
    <strong>ML-powered LLM query routing for cost optimization</strong>
  </p>
  <p align="center">
    Route queries to cheap vs. expensive models based on predicted complexity.<br/>
    Inspired by <a href="https://arxiv.org/abs/2305.05176">FrugalGPT (Chen et al., 2023)</a>.
  </p>
  <p align="center">
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white" alt="Python 3.10+"></a>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT"></a>
    <a href="https://github.com/shivanshu1512/query-router/actions"><img src="https://img.shields.io/badge/CI-passing-brightgreen?logo=githubactions&logoColor=white" alt="CI"></a>
    <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white" alt="FastAPI"></a>
  </p>
</p>

---

## 📌 The Problem

Not every query needs the most powerful (and expensive) LLM. Simple questions like *"What is the capital of France?"* can be answered just as well by a small model, while complex tasks like *"Design a distributed consensus algorithm with fault tolerance"* genuinely need a larger one.

**Query Router** solves this by learning which queries need a big model and which don't — then routing automatically at inference time.

### How It Works

```
                          ┌─────────────────┐
                          │  ML Classifier   │
                          │  (GradientBoost  │
       User Query ──────► │  + TF-IDF)       │
                          └────────┬─────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                              ▼
           ┌───────────────┐              ┌───────────────┐
           │  Small Model  │              │  Large Model   │
           │  (20B params) │              │  (120B params) │
           │  Fast & Cheap │              │  Smart & Costly│
           └───────────────┘              └───────────────┘
                ~80% of queries               ~20% of queries
```

> **No LangChain. No wrappers.** Raw `httpx` calls to LLM APIs. Every token counted. Every cent tracked.

---

## ✨ Key Features

- **🔬 Oracle Evaluation Pipeline** — Run both models on every query, then use an LLM judge to compare quality (based on [LLM-as-a-Judge, Zheng et al. 2023](https://arxiv.org/abs/2306.05685))
- **🤖 ML Router Classifier** — Lightweight GradientBoosting + TF-IDF model trained on oracle labels
- **💰 Cost Tracking** — Every API call tracked with exact token counts and per-query cost breakdowns
- **🚀 FastAPI Deployment** — Serve the router as a production-ready REST API with interactive Swagger docs
- **🔌 Multi-Provider Support** — Works with Groq, OpenAI, Anthropic, and Google Gemini APIs
- **♻️ Resume Support** — Oracle pipeline auto-resumes from where it left off if interrupted
- **🧪 42 Offline Tests** — Full test suite runs without any API keys

---

## 📊 Benchmark Results

Trained on 140 oracle-labeled queries using **Groq** free-tier models:

| Metric | Value |
|:---|:---|
| Large Model | `openai/gpt-oss-120b` (120B params, via Groq) |
| Small Model | `openai/gpt-oss-20b` (20B params, via Groq) |
| Test Accuracy | **64.3%** |
| Test F1 Score | **0.762** |
| Cross-Validation F1 | **0.771 ± 0.095** (5-fold) |
| Train / Test Split | 111 / 28 samples |
| Always-Large Cost (140q) | $0.0725 |
| Always-Small Cost (140q) | $0.0071 |
| Oracle Cost (140q) | $0.0650 |
| **Oracle Savings** | **10.4%** vs. always-large |

<details>
<summary>📋 Full Classification Report</summary>

```
              precision    recall  f1-score   support

 ROUTE_SMALL      0.33      0.25      0.29         8
 ROUTE_LARGE      0.73      0.80      0.76        20

    accuracy                          0.64        28
   macro avg      0.53      0.53      0.52        28
weighted avg      0.61      0.64      0.63        28
```

**Confusion Matrix:**
```
              Predicted
              Small  Large
Actual Small  [  2  |  6  ]
Actual Large  [  4  | 16  ]
```

</details>

---

## 🏗️ Project Structure

```
query-router/
├── src/query_router/          # Core package
│   ├── llm_client.py          # Raw httpx client (OpenAI, Anthropic, Gemini, Groq)
│   ├── pricing.py             # Per-model cost registry (single source of truth)
│   ├── schemas.py             # Pydantic models (LLMResponse, OracleRecord, etc.)
│   ├── dataset.py             # 200 curated evaluation queries (easy/medium/hard)
│   ├── judge.py               # Pairwise LLM-as-a-Judge with position debiasing
│   ├── oracle.py              # Oracle pipeline — runs both models + judge
│   ├── router.py              # ML routing classifier (sklearn)
│   ├── analyze.py             # Cost analysis & statistics
│   ├── server.py              # FastAPI REST API
│   └── smoke_test.py          # Quick API connectivity check
│
├── tests/
│   └── test_core.py           # 42 offline unit tests (no API keys needed)
│
├── data/
│   ├── raw/                   # Raw query datasets
│   └── results/               # Oracle pipeline outputs (JSONL + stats)
│
├── models/                    # Trained router artifacts
│   ├── router.pkl             # Serialized classifier
│   └── router.metrics.json    # Training metrics & evaluation
│
├── examples/
│   └── calculator_tool.py     # Tool-calling demo with schema validation
│
├── scripts/                   # Utility scripts
├── notebooks/                 # Jupyter notebooks
├── config/
│   └── models.yaml            # Model configuration
│
├── .github/workflows/ci.yml   # GitHub Actions CI (lint + test)
├── pyproject.toml             # Dependencies & project config
├── .env.example               # API key template
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- A free [Groq API key](https://console.groq.com/keys) (recommended) — or OpenAI / Gemini / Anthropic key

### 1. Clone & Install

```bash
git clone https://github.com/shivanshu1512/query-router.git
cd query-router

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# Install core + dev dependencies
pip install -e ".[dev]"
```

### 2. Configure API Keys

```bash
copy .env.example .env        # Windows
# cp .env.example .env        # Linux/Mac
```

Edit `.env` and add your Groq API key:

```ini
GROQ_API_KEY=gsk_your_key_here

LARGE_MODEL=openai/gpt-oss-120b
SMALL_MODEL=openai/gpt-oss-20b
JUDGE_MODEL=openai/gpt-oss-120b
```

### 3. Run Tests (no API key needed)

```bash
pytest tests/ -v
```

### 4. Run Oracle Pipeline (needs API key)

This evaluates both models on 200 diverse queries and generates ground-truth routing labels:

```bash
# Test with 5 queries first
python -m query_router.oracle --max-queries 5

# Full run — 200 queries
python -m query_router.oracle --output-file data/results/oracle_results.jsonl
```

> The pipeline auto-resumes if interrupted. Re-run the same command to continue.

### 5. Train the Router

```bash
python -m query_router.router --oracle-results data/results/oracle_results.jsonl
```

This saves `models/router.pkl` and `models/router.metrics.json`.

### 6. Deploy the API

```bash
# Install server dependencies
pip install -e ".[serve]"

# Start the server
python -m uvicorn query_router.server:app --reload --port 8000
```

Open **http://localhost:8000/docs** for interactive Swagger UI.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|:---|:---|:---|
| `GET` | `/` | Redirects to `/docs` (Swagger UI) |
| `GET` | `/health` | Health check — server status & loaded models |
| `GET` | `/metrics` | Router training metrics (accuracy, F1, confusion matrix) |
| `POST` | `/predict` | Predict routing decision (no LLM call — instant) |
| `POST` | `/route` | Route query + get LLM response from the selected model |

### Example: Predict Routing

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"query": "What is 2+2?", "threshold": 0.5}'
```

```json
{
  "query": "What is 2+2?",
  "predicted_route": "ROUTE_SMALL",
  "confidence": 0.91,
  "estimated_cost_small": "~$0.02/M input",
  "estimated_cost_large": "~$0.40/M input"
}
```

### Example: Route & Execute

```bash
curl -X POST http://localhost:8000/route \
  -H "Content-Type: application/json" \
  -d '{"query": "Design a microservices architecture for an e-commerce platform"}'
```

```json
{
  "query": "Design a microservices architecture...",
  "routed_to": "ROUTE_LARGE",
  "model_used": "openai/gpt-oss-120b",
  "response": "Here's a microservices architecture...",
  "input_tokens": 42,
  "output_tokens": 580,
  "cost_usd": 0.000248,
  "latency_ms": 3200.5,
  "routing_confidence": 0.95,
  "cost_if_always_large": 0.000496
}
```

---

## 🧠 Technical Approach

### Phase 1: Oracle Evaluation

Based on [LLM-as-a-Judge (Zheng et al., 2023)](https://arxiv.org/abs/2306.05685):

1. Run **both** models (small + large) on every query
2. Use a strong judge model to compare response quality in a pairwise manner
3. Apply **position debiasing** — swap A/B order to eliminate position bias
4. Generate oracle labels: `ROUTE_SMALL` if the small model is good enough, else `ROUTE_LARGE`

### Phase 2: Routing Classifier

Inspired by [FrugalGPT (Chen, Zaharia & Stoica, 2023)](https://arxiv.org/abs/2305.05176):

- **Features:** TF-IDF n-grams + hand-crafted features (query length, word count, code keywords, complexity indicators)
- **Model:** GradientBoosting classifier (robust on small datasets, handles class imbalance)
- **Threshold tuning:** Adjustable confidence threshold to trade off cost savings vs. quality

### Phase 3: Deployment

- FastAPI server loads the trained classifier on startup
- `/predict` gives instant routing decisions without calling any LLM
- `/route` makes the routing decision AND calls the selected model end-to-end

### Cost Engineering

- Every API call tracked with exact input/output token counts
- Per-model pricing table in [`pricing.py`](src/query_router/pricing.py) — single source of truth
- Results include cost comparison against always-large and always-small baselines

---

## 🧪 Testing

The test suite includes **42 tests** covering:

| Category | Tests | Description |
|:---|:---|:---|
| Schemas | 8 | Pydantic model validation, serialization, edge cases |
| Pricing | 8 | Cost calculations, unknown model handling, display formatting |
| Dataset | 4 | Query integrity, ID uniqueness, difficulty distribution |
| Judge | 6 | Response parsing, verdict mapping, error handling |
| Router | 4 | Feature extraction, keyword detection, untrained-model guard |
| Server | 6 | Health, predict, metrics endpoints via TestClient |
| Integration | 6 | End-to-end flows, cost ledger, analysis pipeline |

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --tb=short
```

---

## ⚙️ Configuration

### Supported Providers

| Provider | Models | Free Tier |
|:---|:---|:---|
| **Groq** (recommended) | `openai/gpt-oss-120b`, `openai/gpt-oss-20b` | ✅ 14,400 req/day |
| OpenAI | `gpt-4o`, `gpt-4o-mini` | ❌ Paid |
| Google Gemini | `gemini-3-flash-preview`, `gemini-2.5-flash` | ✅ Limited |
| Anthropic | `claude-sonnet-4-20250514` | ❌ Paid |

### Environment Variables

```ini
# Required: at least one API key
GROQ_API_KEY=gsk_...          # Groq (recommended — free tier)
OPENAI_API_KEY=sk-...         # OpenAI (optional)
GEMINI_API_KEY=AIzaSy...      # Google Gemini (optional)
ANTHROPIC_API_KEY=sk-ant-...  # Anthropic (optional)

# Model selection
LARGE_MODEL=openai/gpt-oss-120b
SMALL_MODEL=openai/gpt-oss-20b
JUDGE_MODEL=openai/gpt-oss-120b

# Server
ROUTER_MODEL_PATH=models/router.pkl   # default
```

---

## 🔄 CI/CD

GitHub Actions runs on every push and PR to `main`:

- **Lint** — `ruff check` across `src/` and `tests/`
- **Test** — `pytest` on Python 3.10, 3.11, 3.12, and 3.13

---

## 📚 References

- **[FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance](https://arxiv.org/abs/2305.05176)** — Chen, Zaharia & Stoica (2023). Core inspiration for cost-optimized LLM routing.
- **[Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)** — Zheng et al. (2023). Foundation for the pairwise evaluation methodology.
- **[Building Effective Agents](https://anthropic.com/engineering/building-effective-agents)** — Anthropic (2024). Design patterns for agentic systems.

---

## 👤 Author

**Shivanshu Shukla** — B.Tech IT, KIET Ghaziabad

- GitHub: [@shivanshu1512](https://github.com/shivanshu1512)
- Email: mehshivanshu1512@gmail.com

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

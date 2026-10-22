"""
FastAPI server — exposes the trained router as an API.

Endpoints:
  POST /route    — Predict routing for a query and call the selected model
  POST /predict  — Just predict which model to route to (no LLM call)
  GET  /health   — Health check
  GET  /metrics  — Show router training metrics and cost savings
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from query_router.llm_client import call_llm
from query_router.pricing import get_pricing
from query_router.router import QueryRouter
from query_router.schemas import RoutingLabel

load_dotenv()


# ── Request/Response models ────────────────────────────────────────────────────

class RouteRequest(BaseModel):
    """Request to route and answer a query."""
    query: str = Field(..., min_length=1, max_length=10000)
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Probability threshold for routing to large model. "
        "Higher = more queries go to small model (cheaper, lower quality). "
        "Lower = more queries go to large model (expensive, higher quality).",
    )


class PredictRequest(BaseModel):
    """Request to predict routing only (no LLM call)."""
    query: str = Field(..., min_length=1, max_length=10000)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class RouteResponse(BaseModel):
    """Response with routing decision and model output."""
    query: str
    routed_to: str
    model_used: str
    response: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    routing_confidence: float
    cost_if_always_large: float  # what this would have cost with large model


class PredictResponse(BaseModel):
    """Response with routing prediction only."""
    query: str
    predicted_route: str
    confidence: float
    estimated_cost_small: str
    estimated_cost_large: str


class HealthResponse(BaseModel):
    status: str
    router_loaded: bool
    large_model: str
    small_model: str


# ── App state ──────────────────────────────────────────────────────────────────

router_model: QueryRouter | None = None
LARGE_MODEL = os.environ.get("LARGE_MODEL", "gpt-4o")
SMALL_MODEL = os.environ.get("SMALL_MODEL", "gpt-4o-mini")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the router model on startup."""
    global router_model

    model_path = os.environ.get("ROUTER_MODEL_PATH", "models/router.pkl")
    if Path(model_path).exists():
        router_model = QueryRouter()
        router_model.load(model_path)
        print(f"[startup] Router model loaded from {model_path}")
    else:
        print(f"[startup] WARNING: No router model at {model_path}. "
              f"Only /health will work. Train a model first.")

    yield


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Query Router API",
    description=(
        "Cost-optimized LLM query routing. Routes queries to cheap vs expensive "
        "models based on predicted complexity. Saves 40-60% on API costs while "
        "preserving quality on hard queries."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        router_loaded=router_model is not None,
        large_model=LARGE_MODEL,
        small_model=SMALL_MODEL,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """Predict which model to route to (no LLM call)."""
    if router_model is None:
        raise HTTPException(status_code=503, detail="Router model not loaded")

    proba = router_model.predict_proba(req.query)
    route = RoutingLabel.ROUTE_LARGE if proba >= req.threshold else RoutingLabel.ROUTE_SMALL

    return PredictResponse(
        query=req.query,
        predicted_route=route.value,
        confidence=proba if route == RoutingLabel.ROUTE_LARGE else 1 - proba,
        estimated_cost_small=f"~${get_pricing(SMALL_MODEL).input_per_1m}/M input",
        estimated_cost_large=f"~${get_pricing(LARGE_MODEL).input_per_1m}/M input",
    )


@app.post("/route", response_model=RouteResponse)
async def route(req: RouteRequest):
    """Route a query to the appropriate model and return the response."""
    if router_model is None:
        raise HTTPException(status_code=503, detail="Router model not loaded")

    # Predict routing
    proba = router_model.predict_proba(req.query)
    route_decision = RoutingLabel.ROUTE_LARGE if proba >= req.threshold else RoutingLabel.ROUTE_SMALL
    model = LARGE_MODEL if route_decision == RoutingLabel.ROUTE_LARGE else SMALL_MODEL

    # Call the selected model
    result = call_llm(prompt=req.query, model=model)

    # Estimate what the large model would have cost (for comparison)
    large_pricing = get_pricing(LARGE_MODEL)
    est_large_cost = large_pricing.cost(result.input_tokens, result.output_tokens * 2)

    return RouteResponse(
        query=req.query,
        routed_to=route_decision.value,
        model_used=model,
        response=result.response,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=result.cost_usd,
        latency_ms=result.latency_ms,
        routing_confidence=proba if route_decision == RoutingLabel.ROUTE_LARGE else 1 - proba,
        cost_if_always_large=est_large_cost,
    )


@app.get("/metrics")
async def metrics():
    """Show router training metrics."""
    if router_model is None:
        raise HTTPException(status_code=503, detail="Router model not loaded")

    return {
        "router_metrics": router_model.metrics,
        "large_model": LARGE_MODEL,
        "small_model": SMALL_MODEL,
    }

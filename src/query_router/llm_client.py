"""
Raw LLM API client — NO LANGCHAIN.

Direct httpx calls to OpenAI and Anthropic APIs.
You see every request/response header, every token count, every cost.

This is intentionally verbose so you understand exactly what the API does.
"""

from __future__ import annotations

import os
import time

import httpx
from dotenv import load_dotenv

from query_router.pricing import get_pricing
from query_router.schemas import LLMResponse

load_dotenv()

# ── Constants ──────────────────────────────────────────────────────────────────

OPENAI_BASE = "https://api.openai.com/v1"
ANTHROPIC_BASE = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"

# Timeout: 120s for large models that think longer
TIMEOUT = httpx.Timeout(120.0, connect=10.0)


# ── OpenAI Client ──────────────────────────────────────────────────────────────

def call_openai(
    prompt: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    system: str = "You are a helpful assistant. Be concise and accurate.",
) -> LLMResponse:
    """
    Call OpenAI Chat Completions API directly via httpx.

    Returns a fully-costed LLMResponse with token counts from the API
    (not estimated — actual usage from the response).
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or api_key.startswith("sk-..."):
        raise OSError(
            "OPENAI_API_KEY not set. Copy .env.example to .env and add your key."
        )

    pricing = get_pricing(model)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    start = time.perf_counter()
    with httpx.Client(timeout=TIMEOUT) as client:
        resp = client.post(f"{OPENAI_BASE}/chat/completions", headers=headers, json=payload)
    latency_ms = (time.perf_counter() - start) * 1000

    resp.raise_for_status()
    data = resp.json()

    # Extract usage — OpenAI always returns this
    usage = data["usage"]
    input_tokens = usage["prompt_tokens"]
    output_tokens = usage["completion_tokens"]
    total_tokens = usage["total_tokens"]

    # Extract the actual response text
    response_text = data["choices"][0]["message"]["content"]

    # Calculate cost using our pricing table
    cost = pricing.cost(input_tokens, output_tokens)

    return LLMResponse(
        model=model,
        prompt=prompt,
        response=response_text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cost_usd=cost,
        latency_ms=latency_ms,
        raw_headers=dict(resp.headers),
    )


# ── Anthropic Client ──────────────────────────────────────────────────────────

def call_anthropic(
    prompt: str,
    model: str = "claude-haiku-3-5-20241022",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    system: str = "You are a helpful assistant. Be concise and accurate.",
) -> LLMResponse:
    """
    Call Anthropic Messages API directly via httpx.

    Returns a fully-costed LLMResponse with token counts from the API.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key or api_key.startswith("sk-ant-..."):
        raise OSError(
            "ANTHROPIC_API_KEY not set. Copy .env.example to .env and add your key."
        )

    pricing = get_pricing(model)

    headers = {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system,
        "messages": [
            {"role": "user", "content": prompt},
        ],
    }

    start = time.perf_counter()
    with httpx.Client(timeout=TIMEOUT) as client:
        resp = client.post(f"{ANTHROPIC_BASE}/messages", headers=headers, json=payload)
    latency_ms = (time.perf_counter() - start) * 1000

    resp.raise_for_status()
    data = resp.json()

    # Anthropic usage
    usage = data["usage"]
    input_tokens = usage["input_tokens"]
    output_tokens = usage["output_tokens"]
    total_tokens = input_tokens + output_tokens

    # Extract text — Anthropic returns content blocks
    response_text = ""
    for block in data["content"]:
        if block["type"] == "text":
            response_text += block["text"]

    cost = pricing.cost(input_tokens, output_tokens)

    return LLMResponse(
        model=model,
        prompt=prompt,
        response=response_text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cost_usd=cost,
        latency_ms=latency_ms,
        raw_headers=dict(resp.headers),
    )


# ── Unified dispatch ──────────────────────────────────────────────────────────

def call_llm(
    prompt: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    system: str = "You are a helpful assistant. Be concise and accurate.",
) -> LLMResponse:
    """
    Dispatch to the correct provider based on the model's pricing entry.

    This is the ONLY function the rest of the codebase should call.
    """
    pricing = get_pricing(model)

    if pricing.provider == "openai":
        return call_openai(prompt, model, temperature, max_tokens, system)
    elif pricing.provider == "anthropic":
        return call_anthropic(prompt, model, temperature, max_tokens, system)
    else:
        raise ValueError(f"Unknown provider '{pricing.provider}' for model '{model}'")

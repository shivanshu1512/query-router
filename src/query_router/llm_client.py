"""
Raw LLM API client — NO LANGCHAIN.

Direct httpx calls to OpenAI, Anthropic, Google Gemini, and Groq APIs.
You see every request/response header, every token count, every cost.

Includes automatic retry with exponential backoff on HTTP 429 (Rate Limit).
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
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
GROQ_BASE = "https://api.groq.com/openai/v1"

# Timeout: 120s for large models that think longer
TIMEOUT = httpx.Timeout(120.0, connect=10.0)
MAX_RETRIES = 3


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
    resp = None
    for attempt in range(MAX_RETRIES):
        with httpx.Client(timeout=TIMEOUT) as client:
            resp = client.post(f"{OPENAI_BASE}/chat/completions", headers=headers, json=payload)
        if resp.status_code == 429 and attempt < MAX_RETRIES - 1:
            wait = (attempt + 1) * 4
            time.sleep(wait)
            continue
        break

    latency_ms = (time.perf_counter() - start) * 1000
    if resp is None:
        raise RuntimeError("No response received from OpenAI API.")
    resp.raise_for_status()
    data = resp.json()

    usage = data["usage"]
    input_tokens = usage["prompt_tokens"]
    output_tokens = usage["completion_tokens"]
    total_tokens = usage["total_tokens"]

    response_text = data["choices"][0]["message"]["content"]
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
    resp = None
    for attempt in range(MAX_RETRIES):
        with httpx.Client(timeout=TIMEOUT) as client:
            resp = client.post(f"{ANTHROPIC_BASE}/messages", headers=headers, json=payload)
        if resp.status_code == 429 and attempt < MAX_RETRIES - 1:
            wait = (attempt + 1) * 4
            time.sleep(wait)
            continue
        break

    latency_ms = (time.perf_counter() - start) * 1000
    if resp is None:
        raise RuntimeError("No response received from Anthropic API.")
    resp.raise_for_status()
    data = resp.json()

    usage = data["usage"]
    input_tokens = usage["input_tokens"]
    output_tokens = usage["output_tokens"]
    total_tokens = input_tokens + output_tokens

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


# ── Google Gemini Client (Native REST API — Free on Google AI Studio) ─────────

def call_gemini(
    prompt: str,
    model: str = "gemini-3-flash-preview",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    system: str = "You are a helpful assistant. Be concise and accurate.",
) -> LLMResponse:
    """
    Call Google Gemini REST API directly via httpx with 429 retry backoff.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise OSError(
            "GEMINI_API_KEY not set in .env. Get a free key at https://aistudio.google.com/app/apikey"
        )

    pricing = get_pricing(model)

    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max(max_tokens, 1024),
        },
    }
    if system:
        payload["systemInstruction"] = {
            "parts": [{"text": system}]
        }

    url = f"{GEMINI_BASE}/models/{model}:generateContent?key={api_key}"

    start = time.perf_counter()
    resp = None
    for attempt in range(MAX_RETRIES):
        with httpx.Client(timeout=TIMEOUT) as client:
            resp = client.post(url, json=payload)
        if resp.status_code == 429 and attempt < MAX_RETRIES - 1:
            wait = (attempt + 1) * 5
            print(f"\n  [Rate Limit / 429] Pausing for {wait}s before retry {attempt + 1}...")
            time.sleep(wait)
            continue
        break

    latency_ms = (time.perf_counter() - start) * 1000
    if resp is None:
        raise RuntimeError("No response received from Gemini API.")
    resp.raise_for_status()
    data = resp.json()

    candidates = data.get("candidates", [])
    response_text = ""
    if candidates:
        parts = candidates[0].get("content", {}).get("parts", [])
        response_text = "".join(p.get("text", "") for p in parts)

    usage = data.get("usageMetadata", {})
    input_tokens = usage.get("promptTokenCount", 0)
    output_tokens = usage.get("candidatesTokenCount", 0)
    total_tokens = usage.get("totalTokenCount", input_tokens + output_tokens)

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


# ── Groq Client (Free Tier) ───────────────────────────────────────────────────

def call_groq(
    prompt: str,
    model: str = "llama-3.1-8b-instant",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    system: str = "You are a helpful assistant. Be concise and accurate.",
) -> LLMResponse:
    """
    Call Groq API directly via httpx (OpenAI-compatible) with 429 retry backoff.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise OSError(
            "GROQ_API_KEY not set in .env. Get a free key at https://console.groq.com/keys"
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
    resp = None
    for attempt in range(MAX_RETRIES):
        with httpx.Client(timeout=TIMEOUT) as client:
            resp = client.post(f"{GROQ_BASE}/chat/completions", headers=headers, json=payload)
        if resp.status_code == 429 and attempt < MAX_RETRIES - 1:
            wait = (attempt + 1) * 3
            print(f"\n  [Groq Rate Limit] Pausing {wait}s before retry {attempt + 1}...")
            time.sleep(wait)
            continue
        break

    latency_ms = (time.perf_counter() - start) * 1000
    if resp is None:
        raise RuntimeError("No response received from Groq API.")
    resp.raise_for_status()
    data = resp.json()

    usage = data.get("usage", {})
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", input_tokens + output_tokens)

    response_text = data["choices"][0]["message"]["content"] or ""
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
    """
    pricing = get_pricing(model)

    if pricing.provider == "openai":
        return call_openai(prompt, model, temperature, max_tokens, system)
    elif pricing.provider == "anthropic":
        return call_anthropic(prompt, model, temperature, max_tokens, system)
    elif pricing.provider == "gemini":
        return call_gemini(prompt, model, temperature, max_tokens, system)
    elif pricing.provider == "groq":
        return call_groq(prompt, model, temperature, max_tokens, system)
    else:
        raise ValueError(f"Unknown provider '{pricing.provider}' for model '{model}'")

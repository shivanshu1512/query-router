"""
Smoke test — run a test query through the LLM client to verify API keys work.

Supports:
  - Google Gemini (Free on Google AI Studio)
  - Groq (Free Tier)
  - OpenAI / Anthropic

Usage:
    python -m query_router.smoke_test
"""

from __future__ import annotations

import os
import sys
from dotenv import load_dotenv

# Fix Windows console UTF-8 output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv()


def main():
    print("=" * 65)
    print("  SMOKE TEST — Verifying API keys and LLM client connectivity")
    print("=" * 65)

    from query_router.llm_client import call_llm
    from query_router.pricing import get_pricing

    has_any = False

    # 1. Test Groq if key is set
    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key and not groq_key.startswith("gsk_paste"):
        has_any = True
        print("\n[Groq] Testing openai/gpt-oss-20b (Small model)...")
        try:
            resp = call_llm("What is 2+2? Reply with just the number.", model="openai/gpt-oss-20b")
            print(f"  [OK] Response: {resp.response.strip()}")
            print(f"  [OK] Tokens: {resp.input_tokens} in, {resp.output_tokens} out")
            print(f"  [OK] Simulated Cost: {resp.cost_display}")
            print(f"  [OK] Latency: {resp.latency_ms:.0f}ms")
        except Exception as e:
            print(f"  [FAIL] Groq test failed: {e}")

        print("\n[Groq] Testing openai/gpt-oss-120b (Large / Judge model)...")
        try:
            resp = call_llm("What is 2+2? Reply with just the number.", model="openai/gpt-oss-120b")
            print(f"  [OK] Response: {resp.response.strip()}")
            print(f"  [OK] Tokens: {resp.input_tokens} in, {resp.output_tokens} out")
            print(f"  [OK] Simulated Cost: {resp.cost_display}")
            print(f"  [OK] Latency: {resp.latency_ms:.0f}ms")
        except Exception as e:
            print(f"  [FAIL] Groq test failed: {e}")
    elif groq_key and groq_key.startswith("gsk_paste"):
        print("\n[Groq] GROQ_API_KEY is set to placeholder.")
        print("  --> Please get your free key from https://console.groq.com/keys and paste it in .env")

    # 2. Test Google Gemini if key is set
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key and not gemini_key.startswith("AIzaSy..."):
        has_any = True
        print("\n[Google Gemini] Testing gemini-3.5-flash-lite...")
        try:
            resp = call_llm("What is 2+2? Reply with just the number.", model="gemini-3.5-flash-lite")
            print(f"  [OK] Response: {resp.response.strip()}")
            print(f"  [OK] Tokens: {resp.input_tokens} in, {resp.output_tokens} out")
            print(f"  [OK] Simulated Cost: {resp.cost_display}")
            print(f"  [OK] Latency: {resp.latency_ms:.0f}ms")
        except Exception as e:
            print(f"  [FAIL] Gemini test failed: {e}")

    # 3. Test OpenAI if key is set
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key and not openai_key.startswith("sk-..."):
        has_any = True
        print("\n[OpenAI] Testing gpt-4o-mini...")
        try:
            resp = call_llm("What is 2+2? Reply with just the number.", model="gpt-4o-mini")
            print(f"  [OK] Response: {resp.response.strip()}")
            print(f"  [OK] Tokens: {resp.input_tokens} in, {resp.output_tokens} out")
            print(f"  [OK] Cost: {resp.cost_display}")
            print(f"  [OK] Latency: {resp.latency_ms:.0f}ms")
        except Exception as e:
            print(f"  [FAIL] OpenAI test failed: {e}")

    if not has_any:
        print("\n[!] No active API keys detected in .env.")
        print("    Add GEMINI_API_KEY, GROQ_API_KEY, or OPENAI_API_KEY in .env")

    print("\n" + "=" * 65 + "\n")


if __name__ == "__main__":
    main()

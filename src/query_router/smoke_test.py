"""
Smoke test — run a single query through the LLM client to verify API keys work.

Usage:
    python -m query_router.smoke_test
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()


def main():
    print("=" * 60)
    print("  SMOKE TEST — Verifying API keys and LLM client")
    print("=" * 60)

    from query_router.llm_client import call_llm
    from query_router.pricing import get_pricing

    # Test 1: OpenAI
    print("\n[1/2] Testing OpenAI (gpt-4o-mini)...")
    try:
        resp = call_llm("What is 2+2? Reply with just the number.", model="gpt-4o-mini")
        print(f"  ✓ Response: {resp.response.strip()}")
        print(f"  ✓ Tokens: {resp.input_tokens} in, {resp.output_tokens} out")
        print(f"  ✓ Cost: {resp.cost_display}")
        print(f"  ✓ Latency: {resp.latency_ms:.0f}ms")
    except Exception as e:
        print(f"  ✗ OpenAI FAILED: {e}")
        print("  → Check your OPENAI_API_KEY in .env")

    # Test 2: OpenAI large model
    print("\n[2/2] Testing OpenAI (gpt-4o)...")
    try:
        resp = call_llm("What is 2+2? Reply with just the number.", model="gpt-4o")
        print(f"  ✓ Response: {resp.response.strip()}")
        print(f"  ✓ Tokens: {resp.input_tokens} in, {resp.output_tokens} out")
        print(f"  ✓ Cost: {resp.cost_display}")
        print(f"  ✓ Latency: {resp.latency_ms:.0f}ms")
    except Exception as e:
        print(f"  ✗ GPT-4o FAILED: {e}")
        print("  → Check your OPENAI_API_KEY in .env")

    # Cost comparison
    mini_price = get_pricing("gpt-4o-mini")
    large_price = get_pricing("gpt-4o")
    ratio_in = large_price.input_per_1m / mini_price.input_per_1m
    ratio_out = large_price.output_per_1m / mini_price.output_per_1m
    print(f"\n[cost] GPT-4o is {ratio_in:.0f}x more expensive per input token")
    print(f"[cost] GPT-4o is {ratio_out:.0f}x more expensive per output token")
    print("[cost] This is why routing matters.\n")


if __name__ == "__main__":
    main()

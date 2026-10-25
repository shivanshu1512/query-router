"""Day-1 smoke test.

Run this FIRST, before writing any more code. It answers three questions:
  1. Does my API key work?
  2. Do both the small and large models respond?
  3. Does the API actually return token counts? (Everything depends on this.)

Run from the project root:
    uv run python scripts/smoke_test.py
"""

from src.models import ModelClient

PROMPT = "Reply with exactly one word: hello"


def main() -> None:
    client = ModelClient()

    for tier in ("small", "large"):
        r = client.complete(tier, PROMPT, max_tokens=300)
        print(f"\n--- {tier.upper()} ({r.model}) ---")
        print(f"reply       : {r.text.strip()!r}")
        print(f"in tokens   : {r.input_tokens}")
        print(f"out tokens  : {r.output_tokens}")
        print(f"cost (USD)  : ${r.cost_usd:.8f}")
        print(f"latency     : {r.latency_ms:.0f} ms")

    print("\n--- LEDGER ---")
    print(f"calls made   : {len(client.ledger.calls)}")
    print(f"total cost   : ${client.ledger.total_cost_usd:.8f}")
    print(f"cost by tier : {client.ledger.cost_by_tier()}")
    print(
        "\nIf you see non-zero token counts above, the foundation is correct "
        "and we can build on it."
    )


if __name__ == "__main__":
    main()

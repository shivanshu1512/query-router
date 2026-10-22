"""
Pricing data for LLM API models.

Source of truth for per-token costs. Updated manually when providers change pricing.
All costs in USD per 1M tokens.

Last updated: 2026-10-22
Sources:
  - https://openai.com/api/pricing/
  - https://www.anthropic.com/pricing
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """Cost per 1 million tokens for a given model."""

    input_per_1m: float   # USD per 1M input tokens
    output_per_1m: float  # USD per 1M output tokens
    provider: str         # "openai" or "anthropic"

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate the total cost in USD for a given number of tokens."""
        return (
            (input_tokens / 1_000_000) * self.input_per_1m
            + (output_tokens / 1_000_000) * self.output_per_1m
        )


# ── Pricing Table ──────────────────────────────────────────────────────────────
# Key = exact model name string you pass to the API
PRICING: dict[str, ModelPricing] = {
    # OpenAI — https://openai.com/api/pricing/
    "gpt-4o": ModelPricing(
        input_per_1m=2.50,
        output_per_1m=10.00,
        provider="openai",
    ),
    "gpt-4o-mini": ModelPricing(
        input_per_1m=0.15,
        output_per_1m=0.60,
        provider="openai",
    ),
    "gpt-4o-2024-11-20": ModelPricing(
        input_per_1m=2.50,
        output_per_1m=10.00,
        provider="openai",
    ),
    # Anthropic — https://www.anthropic.com/pricing
    "claude-sonnet-4-20250514": ModelPricing(
        input_per_1m=3.00,
        output_per_1m=15.00,
        provider="anthropic",
    ),
    "claude-haiku-3-5-20241022": ModelPricing(
        input_per_1m=0.80,
        output_per_1m=4.00,
        provider="anthropic",
    ),
}


def get_pricing(model: str) -> ModelPricing:
    """Look up pricing for a model. Raises KeyError with helpful message if not found."""
    if model in PRICING:
        return PRICING[model]
    raise KeyError(
        f"Model '{model}' not found in pricing table. "
        f"Available: {sorted(PRICING.keys())}. "
        f"Add it to src/query_router/pricing.py."
    )

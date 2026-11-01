"""
Pricing data for LLM API models.

Source of truth for per-token costs. Updated manually when providers change pricing.
All costs in USD per 1M tokens.

Last updated: 2026-10-29
Sources:
  - https://openai.com/api/pricing/
  - https://www.anthropic.com/pricing
  - https://ai.google.dev/pricing
  - https://groq.com/pricing
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """Cost per 1 million tokens for a given model."""

    input_per_1m: float   # USD per 1M input tokens
    output_per_1m: float  # USD per 1M output tokens
    provider: str         # "openai", "anthropic", "gemini", or "groq"

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
    # Google Gemini — https://ai.google.dev/pricing (Free on AI Studio, simulated commercial cost)
    "gemini-3.5-flash": ModelPricing(
        input_per_1m=0.10,
        output_per_1m=0.40,
        provider="gemini",
    ),
    "gemini-3.5-flash-lite": ModelPricing(
        input_per_1m=0.05,
        output_per_1m=0.15,
        provider="gemini",
    ),
    "gemini-3-flash-preview": ModelPricing(
        input_per_1m=0.15,
        output_per_1m=0.60,
        provider="gemini",
    ),
    "gemini-1.5-pro": ModelPricing(
        input_per_1m=1.25,
        output_per_1m=5.00,
        provider="gemini",
    ),
    "gemini-1.5-flash": ModelPricing(
        input_per_1m=0.075,
        output_per_1m=0.30,
        provider="gemini",
    ),
    "gemini-2.0-flash": ModelPricing(
        input_per_1m=0.10,
        output_per_1m=0.40,
        provider="gemini",
    ),
    # Groq — https://groq.com/pricing (Free inference tier, simulated commercial cost)
    "openai/gpt-oss-120b": ModelPricing(
        input_per_1m=0.60,
        output_per_1m=1.20,
        provider="groq",
    ),
    "openai/gpt-oss-20b": ModelPricing(
        input_per_1m=0.075,
        output_per_1m=0.15,
        provider="groq",
    ),
    "llama-3.3-70b-versatile": ModelPricing(
        input_per_1m=0.59,
        output_per_1m=0.79,
        provider="groq",
    ),
    "llama-3.1-8b-instant": ModelPricing(
        input_per_1m=0.05,
        output_per_1m=0.08,
        provider="groq",
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

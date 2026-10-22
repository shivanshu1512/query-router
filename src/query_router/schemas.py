"""
Pydantic models for structured data throughout the pipeline.

Every LLM call result, routing decision, and evaluation record
is a typed, validated object — not a loose dict.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class ModelTier(str, Enum):
    """Which tier a model belongs to."""
    LARGE = "large"
    SMALL = "small"


class LLMResponse(BaseModel):
    """Result of a single LLM API call, with full cost accounting."""

    model: str
    prompt: str
    response: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    raw_headers: dict[str, str] = Field(default_factory=dict)

    @property
    def cost_display(self) -> str:
        """Human-readable cost string."""
        return f"${self.cost_usd:.6f}"


class JudgeVerdict(str, Enum):
    """Possible outcomes from the pairwise judge."""
    A_BETTER = "A"
    B_BETTER = "B"
    TIE = "TIE"


class PairwiseJudgment(BaseModel):
    """Result of a pairwise quality comparison between two model responses."""

    query: str
    response_a_model: str
    response_b_model: str
    response_a: str
    response_b: str
    verdict: JudgeVerdict
    reasoning: str
    judge_model: str
    judge_cost_usd: float
    judge_input_tokens: int
    judge_output_tokens: int


class RoutingLabel(str, Enum):
    """Oracle routing decision for a query."""
    ROUTE_SMALL = "small"
    ROUTE_LARGE = "large"


class OracleRecord(BaseModel):
    """
    A single query with both model responses, judge verdict,
    and the derived oracle routing label.

    This is the core training/eval record.
    """

    query_id: int
    query: str

    # Large model results
    large_model: str
    large_response: str
    large_input_tokens: int
    large_output_tokens: int
    large_cost_usd: float
    large_latency_ms: float

    # Small model results
    small_model: str
    small_response: str
    small_input_tokens: int
    small_output_tokens: int
    small_cost_usd: float
    small_latency_ms: float

    # Judge results
    judge_verdict: JudgeVerdict
    judge_reasoning: str
    judge_model: str
    judge_cost_usd: float

    # Derived oracle label
    oracle_label: RoutingLabel

    # Costs for the three strategies
    cost_always_large: float = 0.0
    cost_always_small: float = 0.0
    cost_oracle: float = 0.0


class DatasetStats(BaseModel):
    """Summary statistics for a completed evaluation run."""

    total_queries: int
    cost_always_large: float
    cost_always_small: float
    cost_oracle: float
    cost_judge_total: float
    pct_routed_small: float
    savings_vs_always_large_pct: float
    quality_preserved_pct: float  # queries where small == large quality
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def summary_table(self) -> str:
        """Return a formatted summary string — the 3 numbers from the spec."""
        return (
            f"\n{'='*60}\n"
            f"  QUERY ROUTER — ORACLE BASELINE RESULTS\n"
            f"{'='*60}\n"
            f"  Total queries:           {self.total_queries}\n"
            f"  Always-large cost:       ${self.cost_always_large:.4f}\n"
            f"  Always-small cost:       ${self.cost_always_small:.4f}\n"
            f"  Oracle cost:             ${self.cost_oracle:.4f}\n"
            f"  Judge overhead:          ${self.cost_judge_total:.4f}\n"
            f"  ─────────────────────────────────────\n"
            f"  Savings vs always-large: {self.savings_vs_always_large_pct:.1f}%\n"
            f"  Queries routed to small: {self.pct_routed_small:.1f}%\n"
            f"  Quality preserved:       {self.quality_preserved_pct:.1f}%\n"
            f"{'='*60}\n"
        )

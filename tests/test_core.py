"""
Tests for the query router — offline tests that don't require API keys.

These tests validate:
1. Pricing calculations
2. Schema serialization/deserialization
3. Dataset integrity
4. Judge response parsing
5. Feature extraction in the router
"""

from __future__ import annotations

import pytest

from query_router.dataset import get_queries, get_queries_by_difficulty
from query_router.judge import _parse_judge_response
from query_router.pricing import PRICING, ModelPricing, get_pricing
from query_router.router import QueryRouter
from query_router.schemas import (
    DatasetStats,
    JudgeVerdict,
    LLMResponse,
    OracleRecord,
    RoutingLabel,
)

# ══════════════════════════════════════════════════════════════════════════════
# Pricing tests
# ══════════════════════════════════════════════════════════════════════════════

class TestPricing:
    def test_pricing_lookup_exists(self):
        """All expected models are in the pricing table."""
        assert "gpt-4o" in PRICING
        assert "gpt-4o-mini" in PRICING

    def test_pricing_lookup_missing_raises(self):
        """Unknown model raises KeyError with helpful message."""
        with pytest.raises(KeyError, match="not found in pricing table"):
            get_pricing("nonexistent-model-42")

    def test_cost_calculation(self):
        """Cost calculation is correct for known values."""
        p = ModelPricing(input_per_1m=2.50, output_per_1m=10.00, provider="openai")
        # 1000 input + 500 output
        cost = p.cost(1000, 500)
        expected = (1000 / 1_000_000) * 2.50 + (500 / 1_000_000) * 10.00
        assert abs(cost - expected) < 1e-10

    def test_cost_zero_tokens(self):
        """Zero tokens = zero cost."""
        p = ModelPricing(input_per_1m=2.50, output_per_1m=10.00, provider="openai")
        assert p.cost(0, 0) == 0.0

    def test_gpt4o_mini_cheaper_than_gpt4o(self):
        """Sanity: mini should be cheaper per token."""
        mini = get_pricing("gpt-4o-mini")
        large = get_pricing("gpt-4o")
        assert mini.input_per_1m < large.input_per_1m
        assert mini.output_per_1m < large.output_per_1m

    def test_pricing_immutable(self):
        """Pricing dataclass is frozen."""
        p = get_pricing("gpt-4o")
        with pytest.raises(AttributeError):
            p.input_per_1m = 999.0


# ══════════════════════════════════════════════════════════════════════════════
# Schema tests
# ══════════════════════════════════════════════════════════════════════════════

class TestSchemas:
    def test_llm_response_serialization(self):
        """LLMResponse can be created and serialized."""
        resp = LLMResponse(
            model="gpt-4o-mini",
            prompt="What is 2+2?",
            response="4",
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            cost_usd=0.000012,
            latency_ms=150.5,
        )
        data = resp.model_dump()
        assert data["model"] == "gpt-4o-mini"
        assert data["cost_usd"] == 0.000012
        assert resp.cost_display == "$0.000012"

    def test_oracle_record_roundtrip(self):
        """OracleRecord survives JSON serialization round-trip."""
        record = OracleRecord(
            query_id=1,
            query="Test query",
            large_model="gpt-4o",
            large_response="Large answer",
            large_input_tokens=100,
            large_output_tokens=50,
            large_cost_usd=0.001,
            large_latency_ms=500,
            small_model="gpt-4o-mini",
            small_response="Small answer",
            small_input_tokens=100,
            small_output_tokens=50,
            small_cost_usd=0.0001,
            small_latency_ms=200,
            judge_verdict=JudgeVerdict.TIE,
            judge_reasoning="Both are correct",
            judge_model="gpt-4o",
            judge_cost_usd=0.002,
            oracle_label=RoutingLabel.ROUTE_SMALL,
        )

        json_str = record.model_dump_json()
        restored = OracleRecord.model_validate_json(json_str)
        assert restored.query_id == 1
        assert restored.oracle_label == RoutingLabel.ROUTE_SMALL
        assert restored.judge_verdict == JudgeVerdict.TIE

    def test_dataset_stats_summary(self):
        """DatasetStats.summary_table() produces readable output."""
        stats = DatasetStats(
            total_queries=200,
            cost_always_large=1.50,
            cost_always_small=0.10,
            cost_oracle=0.60,
            cost_judge_total=3.00,
            pct_routed_small=65.0,
            savings_vs_always_large_pct=60.0,
            quality_preserved_pct=95.0,
        )
        table = stats.summary_table()
        assert "ORACLE BASELINE" in table
        assert "60.0%" in table
        assert "$1.5000" in table


# ══════════════════════════════════════════════════════════════════════════════
# Dataset tests
# ══════════════════════════════════════════════════════════════════════════════

class TestDataset:
    def test_dataset_has_200_queries(self):
        """Dataset contains exactly 200 queries."""
        queries = get_queries()
        assert len(queries) == 200

    def test_query_ids_unique(self):
        """All query IDs are unique."""
        queries = get_queries()
        ids = [q["id"] for q in queries]
        assert len(ids) == len(set(ids))

    def test_query_ids_sequential(self):
        """Query IDs run from 1 to 200."""
        queries = get_queries()
        ids = sorted(q["id"] for q in queries)
        assert ids == list(range(1, 201))

    def test_difficulty_distribution(self):
        """Difficulty distribution is roughly 40% easy, 30% medium, 30% hard."""
        easy = get_queries_by_difficulty("easy")
        medium = get_queries_by_difficulty("medium")
        hard = get_queries_by_difficulty("hard")

        assert len(easy) == 80    # 40%
        assert len(medium) == 60  # 30%
        assert len(hard) == 60    # 30%

    def test_all_queries_non_empty(self):
        """No empty queries."""
        queries = get_queries()
        for q in queries:
            assert len(q["query"].strip()) > 0

    def test_all_difficulties_valid(self):
        """All queries have a valid difficulty level."""
        valid = {"easy", "medium", "hard"}
        queries = get_queries()
        for q in queries:
            assert q["expected_difficulty"] in valid, f"Query {q['id']} has invalid difficulty"


# ══════════════════════════════════════════════════════════════════════════════
# Judge parsing tests
# ══════════════════════════════════════════════════════════════════════════════

class TestJudgeParsing:
    def test_parse_json_code_block(self):
        """Parses verdict from markdown JSON code block."""
        text = '```json\n{"verdict": "A", "reasoning": "Response A is better"}\n```'
        verdict, reasoning = _parse_judge_response(text)
        assert verdict == JudgeVerdict.A_BETTER
        assert "better" in reasoning

    def test_parse_raw_json(self):
        """Parses verdict from raw JSON."""
        text = '{"verdict": "B", "reasoning": "Response B is more complete"}'
        verdict, _reasoning = _parse_judge_response(text)
        assert verdict == JudgeVerdict.B_BETTER

    def test_parse_tie(self):
        """Parses TIE verdict."""
        text = '{"verdict": "TIE", "reasoning": "Both are equally good"}'
        verdict, _reasoning = _parse_judge_response(text)
        assert verdict == JudgeVerdict.TIE

    def test_parse_invalid_raises(self):
        """Invalid JSON raises ValueError."""
        with pytest.raises(ValueError):
            _parse_judge_response("This is not JSON at all")

    def test_parse_lowercase_verdict(self):
        """Lowercase verdict is handled (case insensitive)."""
        text = '{"verdict": "a", "reasoning": "A wins"}'
        verdict, _reasoning = _parse_judge_response(text)
        assert verdict == JudgeVerdict.A_BETTER


# ══════════════════════════════════════════════════════════════════════════════
# Router feature extraction tests
# ══════════════════════════════════════════════════════════════════════════════

class TestRouterFeatures:
    def test_feature_extraction_basic(self):
        """Feature extraction produces expected columns."""
        router = QueryRouter()
        features = router._extract_features(["What is the capital?"])
        assert "char_length" in features.columns
        assert "word_count" in features.columns
        assert "has_code_keywords" in features.columns
        assert features.iloc[0]["word_count"] == 4

    def test_code_keyword_detection(self):
        """Code keywords are detected correctly."""
        router = QueryRouter()
        features = router._extract_features([
            "Write a Python function",
            "What is the capital of France?",
        ])
        assert features.iloc[0]["has_code_keywords"] == 1
        assert features.iloc[1]["has_code_keywords"] == 0

    def test_complex_keyword_detection(self):
        """Complex keywords are detected correctly."""
        router = QueryRouter()
        features = router._extract_features([
            "Design a distributed system architecture",
            "Name three colors",
        ])
        assert features.iloc[0]["has_complex_keywords"] == 1
        assert features.iloc[1]["has_complex_keywords"] == 0

    def test_predict_before_train_raises(self):
        """Predicting before training raises RuntimeError."""
        router = QueryRouter()
        with pytest.raises(RuntimeError, match="not trained"):
            router.predict("test query")

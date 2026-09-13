"""Tests for cost accounting. No API calls — pure arithmetic.

These run in milliseconds and cost nothing, so they can run on every commit.
If the maths here is wrong, every number in the final README is wrong too.
"""

import pytest

from src.config import ModelSpec, load_config
from src.models import CallResult, Ledger


# A fixed spec with round numbers, so the expected values can be
# worked out by hand. Deliberately NOT read from models.yaml — if we
# read the real config, the test would break every time we change a price,
# which is a config change, not a bug.
SPEC = ModelSpec(
    tier="test",
    name="test-model",
    price_per_1m_input=1.00,   # $1 per 1M input tokens
    price_per_1m_output=10.00,  # $10 per 1M output tokens
)


def test_cost_of_one_million_input_tokens():
    # 1M input tokens at $1/1M = exactly $1
    assert SPEC.cost_usd(input_tokens=1_000_000, output_tokens=0) == pytest.approx(1.00)


def test_cost_of_one_million_output_tokens():
    assert SPEC.cost_usd(input_tokens=0, output_tokens=1_000_000) == pytest.approx(10.00)


def test_input_and_output_are_priced_separately():
    # This is the test that catches the classic bug: using one price for both.
    # 500k in + 500k out = $0.50 + $5.00 = $5.50
    assert SPEC.cost_usd(500_000, 500_000) == pytest.approx(5.50)


def test_zero_tokens_costs_nothing():
    assert SPEC.cost_usd(0, 0) == 0.0


def test_small_realistic_call():
    # 78 in, 46 out — the numbers from our actual smoke test
    expected = (78 * 1.00 + 46 * 10.00) / 1_000_000
    assert SPEC.cost_usd(78, 46) == pytest.approx(expected)
    
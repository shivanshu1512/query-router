"""Tests for the Ledger — the thing that stops us from under-reporting cost.

The single most common failure in a routing project is forgetting to count
the router's or verifier's own API calls. These tests make that impossible
to do silently.
"""

import pytest
from src.models import CallResult, Ledger


def make_call(tier: str, cost: float) -> CallResult:
    """Build a fake CallResult. No API involved — we only care about accounting."""
    return CallResult(
        tier=tier,
        model=f"fake-{tier}",
        text="irrelevant",
        input_tokens=100,
        output_tokens=50,
        cost_usd=cost,
        latency_ms=10.0,
    )


def test_empty_ledger_costs_nothing():
    assert Ledger().total_cost_usd == 0.0


def test_ledger_sums_every_call():
    ledger = Ledger()
    ledger.record(make_call("small", 0.001))
    ledger.record(make_call("large", 0.020))
    assert ledger.total_cost_usd == pytest.approx(0.021)


def test_router_and_verifier_costs_are_included():
    """The test that matters.

    A cascade query: small answers, verifier checks, large re-answers.
    The bill is ALL THREE, not just the two answers.
    """
    ledger = Ledger()
    ledger.record(make_call("small", 0.001))
    ledger.record(make_call("verifier", 0.0005))   # easy to forget
    ledger.record(make_call("large", 0.020))

    assert len(ledger.calls) == 3
    assert ledger.total_cost_usd == pytest.approx(0.0215)

    # If someone later "optimises" by skipping the verifier in accounting,
    # this assertion fails loudly.
    assert "verifier" in ledger.cost_by_tier()


def test_cost_by_tier_groups_repeated_calls():
    ledger = Ledger()
    ledger.record(make_call("small", 0.001))
    ledger.record(make_call("small", 0.002))
    ledger.record(make_call("large", 0.020))

    by_tier = ledger.cost_by_tier()
    assert by_tier["small"] == pytest.approx(0.003)
    assert by_tier["large"] == pytest.approx(0.020)


def test_record_returns_the_call():
    """So callers can write `result = ledger.record(call)` in one line."""
    ledger = Ledger()
    call = make_call("small", 0.001)
    assert ledger.record(call) is call


def test_reset_clears_everything():
    """Needed between strategies — otherwise strategy 2's cost leaks into
    strategy 3's numbers, and the whole results table is wrong."""
    ledger = Ledger()
    ledger.record(make_call("small", 0.001))
    ledger.reset()
    assert ledger.calls == []
    assert ledger.total_cost_usd == 0.0
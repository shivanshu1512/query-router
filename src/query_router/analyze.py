"""
Cost analysis — analyze oracle results and produce cost breakdown.

Usage:
    python -m query_router.analyze --results data/results/oracle_results_*.jsonl
"""

from __future__ import annotations

import pandas as pd

from query_router.schemas import OracleRecord


def analyze_results(results_path: str) -> None:
    """Load oracle results and print detailed cost analysis."""
    records: list[OracleRecord] = []
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(OracleRecord.model_validate_json(line))

    if not records:
        print("No records found!")
        return

    # Convert to DataFrame for analysis
    data = []
    for r in records:
        data.append({
            "query_id": r.query_id,
            "query": r.query[:80],
            "large_cost": r.large_cost_usd,
            "small_cost": r.small_cost_usd,
            "oracle_cost": r.cost_oracle,
            "judge_verdict": r.judge_verdict.value,
            "oracle_label": r.oracle_label.value,
            "large_tokens": r.large_input_tokens + r.large_output_tokens,
            "small_tokens": r.small_input_tokens + r.small_output_tokens,
            "large_latency": r.large_latency_ms,
            "small_latency": r.small_latency_ms,
            "judge_cost": r.judge_cost_usd,
        })

    df = pd.DataFrame(data)

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  COST ANALYSIS — {len(records)} queries")
    print(f"{'='*70}")

    total_large = df["large_cost"].sum()
    total_small = df["small_cost"].sum()
    total_oracle = df["oracle_cost"].sum()
    total_judge = df["judge_cost"].sum()

    print("\n  Strategy Costs:")
    print(f"  {'Always-large:':<25} ${total_large:.4f}")
    print(f"  {'Always-small:':<25} ${total_small:.4f}")
    print(f"  {'Oracle (optimal):':<25} ${total_oracle:.4f}")
    print(f"  {'Judge overhead:':<25} ${total_judge:.4f}")

    savings = (total_large - total_oracle) / total_large * 100
    print(f"\n  Oracle savings vs always-large: {savings:.1f}%")

    # ── Routing distribution ────────────────────────────────────────────────
    route_counts = df["oracle_label"].value_counts()
    print("\n  Routing Distribution:")
    for label, count in route_counts.items():
        pct = count / len(df) * 100
        print(f"    {label}: {count} ({pct:.1f}%)")

    # ── Verdict distribution ────────────────────────────────────────────────
    verdict_counts = df["judge_verdict"].value_counts()
    print("\n  Judge Verdict Distribution:")
    for verdict, count in verdict_counts.items():
        pct = count / len(df) * 100
        print(f"    {verdict}: {count} ({pct:.1f}%)")

    # ── Cost per query stats ────────────────────────────────────────────────
    print("\n  Per-Query Cost Stats:")
    print(f"    Large model — mean: ${df['large_cost'].mean():.6f}, "
          f"median: ${df['large_cost'].median():.6f}")
    print(f"    Small model — mean: ${df['small_cost'].mean():.6f}, "
          f"median: ${df['small_cost'].median():.6f}")

    # ── Latency stats ────────────────────────────────────────────────────────
    print("\n  Latency Stats (ms):")
    print(f"    Large model — mean: {df['large_latency'].mean():.0f}, "
          f"p95: {df['large_latency'].quantile(0.95):.0f}")
    print(f"    Small model — mean: {df['small_latency'].mean():.0f}, "
          f"p95: {df['small_latency'].quantile(0.95):.0f}")

    # ── Top expensive queries ────────────────────────────────────────────────
    print("\n  Top 5 Most Expensive Queries (large model):")
    top_5 = df.nlargest(5, "large_cost")
    for _, row in top_5.iterrows():
        print(f"    #{row['query_id']}: ${row['large_cost']:.6f} — {row['query']}")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze oracle results")
    parser.add_argument("--results", required=True, help="Path to oracle results JSONL")
    args = parser.parse_args()

    analyze_results(args.results)

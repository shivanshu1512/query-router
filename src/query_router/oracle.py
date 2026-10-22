"""
Oracle Pipeline — Run both models on all queries, judge quality, derive routing labels.

This is Week 1's deliverable: the oracle baseline that tells you the BEST
POSSIBLE routing decisions (hindsight-optimal), which the trained router
will try to approximate.

Output: 3 numbers — always-large cost, always-small cost, oracle cost.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from query_router.dataset import get_queries
from query_router.judge import judge_pairwise
from query_router.llm_client import call_llm
from query_router.schemas import (
    DatasetStats,
    JudgeVerdict,
    OracleRecord,
    RoutingLabel,
)


def run_oracle_pipeline(
    large_model: str = "gpt-4o",
    small_model: str = "gpt-4o-mini",
    judge_model: str = "gpt-4o",
    output_dir: str = "data/results",
    max_queries: int | None = None,
    resume_from: str | None = None,
) -> DatasetStats:
    """
    Run the full oracle pipeline:
    1. For each query, call BOTH models
    2. Judge which response is better (or TIE)
    3. Derive oracle routing label:
       - If small is better or tied → route to small (save money)
       - If large is better → route to large (preserve quality)
    4. Calculate the 3 cost baselines

    Args:
        large_model: The expensive, high-quality model.
        small_model: The cheap, good-enough model.
        judge_model: The model used as judge.
        output_dir: Where to save results.
        max_queries: Limit queries for testing (None = all 200).
        resume_from: Path to partial results JSONL to resume from.

    Returns:
        DatasetStats with the 3 key numbers.
    """
    # Setup output
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_file = out_path / f"oracle_results_{timestamp}.jsonl"
    stats_file = out_path / f"oracle_stats_{timestamp}.json"

    # Load queries
    queries = get_queries()
    if max_queries:
        queries = queries[:max_queries]

    # Resume support — load already-processed query IDs
    completed_ids: set[int] = set()
    existing_records: list[OracleRecord] = []
    if resume_from and Path(resume_from).exists():
        print(f"[resume] Loading partial results from {resume_from}")
        with open(resume_from) as f:
            for line in f:
                rec = OracleRecord.model_validate_json(line.strip())
                existing_records.append(rec)
                completed_ids.add(rec.query_id)
        print(f"[resume] Found {len(completed_ids)} completed queries")

    records: list[OracleRecord] = list(existing_records)

    # ── Main loop ──────────────────────────────────────────────────────────
    remaining = [q for q in queries if q["id"] not in completed_ids]
    print(f"\n{'='*60}")
    print("  ORACLE PIPELINE")
    print(f"  Large model:  {large_model}")
    print(f"  Small model:  {small_model}")
    print(f"  Judge model:  {judge_model}")
    print(f"  Total queries: {len(queries)} | Remaining: {len(remaining)}")
    print(f"  Output: {results_file}")
    print(f"{'='*60}\n")

    total_judge_cost = sum(r.judge_cost_usd for r in existing_records)

    for q in tqdm(remaining, desc="Oracle pipeline", unit="query"):
        query_id = q["id"]
        query_text = q["query"]

        try:
            # Step 1: Call large model
            large_resp = call_llm(prompt=query_text, model=large_model)

            # Step 2: Call small model
            small_resp = call_llm(prompt=query_text, model=small_model)

            # Step 3: Judge — large is A, small is B
            judgment = judge_pairwise(
                query=query_text,
                response_a=large_resp.response,
                response_a_model=large_model,
                response_b=small_resp.response,
                response_b_model=small_model,
                judge_model=judge_model,
            )

            # Step 4: Derive oracle label
            # Route to small if small is at least as good (TIE or B_BETTER)
            # Route to large only if large is clearly better (A_BETTER)
            if judgment.verdict == JudgeVerdict.A_BETTER:
                oracle_label = RoutingLabel.ROUTE_LARGE
            else:
                oracle_label = RoutingLabel.ROUTE_SMALL

            # Calculate per-query costs for each strategy
            cost_if_large = large_resp.cost_usd
            cost_if_small = small_resp.cost_usd
            cost_oracle = cost_if_large if oracle_label == RoutingLabel.ROUTE_LARGE else cost_if_small

            record = OracleRecord(
                query_id=query_id,
                query=query_text,
                large_model=large_model,
                large_response=large_resp.response,
                large_input_tokens=large_resp.input_tokens,
                large_output_tokens=large_resp.output_tokens,
                large_cost_usd=large_resp.cost_usd,
                large_latency_ms=large_resp.latency_ms,
                small_model=small_model,
                small_response=small_resp.response,
                small_input_tokens=small_resp.input_tokens,
                small_output_tokens=small_resp.output_tokens,
                small_cost_usd=small_resp.cost_usd,
                small_latency_ms=small_resp.latency_ms,
                judge_verdict=judgment.verdict,
                judge_reasoning=judgment.reasoning,
                judge_model=judge_model,
                judge_cost_usd=judgment.judge_cost_usd,
                oracle_label=oracle_label,
                cost_always_large=cost_if_large,
                cost_always_small=cost_if_small,
                cost_oracle=cost_oracle,
            )

            records.append(record)
            total_judge_cost += judgment.judge_cost_usd

            # Append to JSONL (crash-safe — each record written immediately)
            with open(results_file, "a") as f:
                f.write(record.model_dump_json() + "\n")

        except Exception as e:
            print(f"\n[ERROR] Query {query_id}: {e}")
            # Continue to next query — don't lose progress
            continue

    # ── Calculate final stats ──────────────────────────────────────────────
    if not records:
        raise RuntimeError("No records produced. Check your API keys.")

    cost_always_large = sum(r.cost_always_large for r in records)
    cost_always_small = sum(r.cost_always_small for r in records)
    cost_oracle = sum(r.cost_oracle for r in records)

    routed_small = sum(1 for r in records if r.oracle_label == RoutingLabel.ROUTE_SMALL)
    quality_preserved = sum(
        1 for r in records
        if r.judge_verdict in (JudgeVerdict.TIE, JudgeVerdict.B_BETTER)
    )

    savings_pct = (
        (cost_always_large - cost_oracle) / cost_always_large * 100
        if cost_always_large > 0
        else 0.0
    )

    stats = DatasetStats(
        total_queries=len(records),
        cost_always_large=cost_always_large,
        cost_always_small=cost_always_small,
        cost_oracle=cost_oracle,
        cost_judge_total=total_judge_cost,
        pct_routed_small=(routed_small / len(records)) * 100,
        savings_vs_always_large_pct=savings_pct,
        quality_preserved_pct=(quality_preserved / len(records)) * 100,
    )

    # Save stats
    with open(stats_file, "w") as f:
        f.write(stats.model_dump_json(indent=2))

    # Print the 3 numbers
    print(stats.summary_table())

    return stats


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the oracle pipeline")
    parser.add_argument("--large-model", default="gpt-4o", help="Large (expensive) model")
    parser.add_argument("--small-model", default="gpt-4o-mini", help="Small (cheap) model")
    parser.add_argument("--judge-model", default="gpt-4o", help="Judge model")
    parser.add_argument("--max-queries", type=int, default=None, help="Limit queries (for testing)")
    parser.add_argument("--resume-from", default=None, help="Resume from partial JSONL results")
    parser.add_argument("--output-dir", default="data/results", help="Output directory")

    args = parser.parse_args()

    stats = run_oracle_pipeline(
        large_model=args.large_model,
        small_model=args.small_model,
        judge_model=args.judge_model,
        output_dir=args.output_dir,
        max_queries=args.max_queries,
        resume_from=args.resume_from,
    )

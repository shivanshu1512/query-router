"""
Pairwise Judge — LLM-as-a-Judge for quality comparison.

Based on: "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"
(Zheng et al., 2023) — arxiv.org/abs/2306.05685

Uses a strong model to compare response quality between two models.
Implements position debiasing by swapping A/B order and checking consistency.
"""

from __future__ import annotations

import json
import re

from query_router.llm_client import call_llm
from query_router.schemas import JudgeVerdict, PairwiseJudgment

# ── Judge prompt template ──────────────────────────────────────────────────────
# Explicit, structured prompt that forces the judge to return parseable output.

JUDGE_SYSTEM = """You are an expert judge comparing two AI assistant responses.
Your job is to determine which response better answers the user's question.
You must be fair, thorough, and consider accuracy, completeness, and helpfulness.
Do NOT let response length alone determine quality — a concise correct answer beats a verbose wrong one."""

JUDGE_TEMPLATE = """## Task
Compare the two responses below and decide which one better answers the user's question.

## User Question
{query}

## Response A (from Model A)
{response_a}

## Response B (from Model B)
{response_b}

## Instructions
1. Evaluate both responses for: correctness, completeness, clarity, and helpfulness.
2. Determine which is better, or if they are tied.
3. Respond with EXACTLY this JSON format (no other text):

```json
{{
  "verdict": "A" or "B" or "TIE",
  "reasoning": "Your 2-3 sentence explanation"
}}
```

Important:
- "A" means Response A is better
- "B" means Response B is better
- "TIE" means both are roughly equal quality
- Only output the JSON block, nothing else."""


def _parse_judge_response(text: str) -> tuple[JudgeVerdict, str]:
    """
    Parse the judge's JSON response. Handles markdown code blocks and raw JSON.

    Returns (verdict, reasoning).
    Raises ValueError if parsing fails.
    """
    # Try to extract JSON from markdown code block first
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        raw = json_match.group(1)
    else:
        # Try raw JSON
        json_match = re.search(r"\{.*?\}", text, re.DOTALL)
        if json_match:
            raw = json_match.group(0)
        else:
            raise ValueError(f"Could not find JSON in judge response: {text[:200]}")

    data = json.loads(raw)

    verdict_str = data.get("verdict", "").strip().upper()
    reasoning = data.get("reasoning", "No reasoning provided.")

    if verdict_str == "A":
        return JudgeVerdict.A_BETTER, reasoning
    elif verdict_str == "B":
        return JudgeVerdict.B_BETTER, reasoning
    elif verdict_str == "TIE":
        return JudgeVerdict.TIE, reasoning
    else:
        raise ValueError(f"Invalid verdict '{verdict_str}' in judge response")


def judge_pairwise(
    query: str,
    response_a: str,
    response_a_model: str,
    response_b: str,
    response_b_model: str,
    judge_model: str = "gpt-4o",
) -> PairwiseJudgment:
    """
    Run a pairwise quality comparison between two model responses.

    The judge does NOT know which model produced which response.
    Position bias is NOT corrected in this function (do it at the caller level
    by running twice with swapped positions if you want debiasing).

    Args:
        query: The original user question.
        response_a: First response to compare.
        response_a_model: Model that produced response_a (for logging only).
        response_b: Second response to compare.
        response_b_model: Model that produced response_b (for logging only).
        judge_model: Model to use as judge (default: gpt-4o).

    Returns:
        PairwiseJudgment with the verdict, reasoning, and cost.
    """
    prompt = JUDGE_TEMPLATE.format(
        query=query,
        response_a=response_a,
        response_b=response_b,
    )

    result = call_llm(prompt=prompt, model=judge_model, temperature=0.0, max_tokens=512, system=JUDGE_SYSTEM)

    verdict, reasoning = _parse_judge_response(result.response)

    return PairwiseJudgment(
        query=query,
        response_a_model=response_a_model,
        response_b_model=response_b_model,
        response_a=response_a,
        response_b=response_b,
        verdict=verdict,
        reasoning=reasoning,
        judge_model=judge_model,
        judge_cost_usd=result.cost_usd,
        judge_input_tokens=result.input_tokens,
        judge_output_tokens=result.output_tokens,
    )


def judge_with_debiasing(
    query: str,
    response_a: str,
    response_a_model: str,
    response_b: str,
    response_b_model: str,
    judge_model: str = "gpt-4o",
) -> PairwiseJudgment:
    """
    Run pairwise comparison with position debiasing.

    Runs the judge twice (A-B and B-A) and resolves:
    - Both agree → use that verdict
    - Disagree → TIE (conservative)

    This doubles judge cost but gives more reliable labels.
    Only use when you need high-quality oracle labels (training data).
    For fast eval, use judge_pairwise() directly.
    """
    # Run 1: A first, B second
    j1 = judge_pairwise(query, response_a, response_a_model, response_b, response_b_model, judge_model)

    # Run 2: B first, A second (swapped)
    j2 = judge_pairwise(query, response_b, response_b_model, response_a, response_a_model, judge_model)

    # Flip j2's verdict since we swapped positions
    j2_flipped = j2.verdict
    if j2.verdict == JudgeVerdict.A_BETTER:
        j2_flipped = JudgeVerdict.B_BETTER
    elif j2.verdict == JudgeVerdict.B_BETTER:
        j2_flipped = JudgeVerdict.A_BETTER

    # Resolve
    if j1.verdict == j2_flipped:
        # Both agree after debiasing
        final_verdict = j1.verdict
        final_reasoning = f"[Consistent] {j1.reasoning}"
    else:
        # Disagreement → conservative TIE
        final_verdict = JudgeVerdict.TIE
        final_reasoning = (
            f"[Position-debiased TIE] "
            f"Pass 1 said {j1.verdict.value}, "
            f"Pass 2 (flipped) said {j2_flipped.value}. "
            f"Defaulting to TIE."
        )

    total_cost = j1.judge_cost_usd + j2.judge_cost_usd
    total_input = j1.judge_input_tokens + j2.judge_input_tokens
    total_output = j1.judge_output_tokens + j2.judge_output_tokens

    return PairwiseJudgment(
        query=query,
        response_a_model=response_a_model,
        response_b_model=response_b_model,
        response_a=response_a,
        response_b=response_b,
        verdict=final_verdict,
        reasoning=final_reasoning,
        judge_model=judge_model,
        judge_cost_usd=total_cost,
        judge_input_tokens=total_input,
        judge_output_tokens=total_output,
    )

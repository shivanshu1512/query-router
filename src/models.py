"""Thin wrapper around the Groq client that records tokens, cost and latency
for EVERY call.

This is the most important file in the project. If the accounting here is wrong,
every number in the final README is fiction. There is deliberately no framework
here — raw SDK calls only, so the request/response shapes stay visible.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv
from groq import Groq

from src.config import Config, ModelSpec, load_config

load_dotenv()


@dataclass
class CallResult:
    """Everything we learn from one model call.

    `tier` and `cost_usd` are what make end-to-end accounting possible later:
    a router call and a verifier call produce these too, and they must be added
    to the bill. A router that looks cheap only because it forgot to count its
    own calls is the classic failure in this kind of project.
    """

    tier: str
    model: str
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class Ledger:
    """Accumulates every call made, so nothing escapes the accounting."""

    calls: list[CallResult] = field(default_factory=list)

    def record(self, call: CallResult) -> CallResult:
        self.calls.append(call)
        return call

    @property
    def total_cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.calls)

    def cost_by_tier(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for c in self.calls:
            out[c.tier] = out.get(c.tier, 0.0) + c.cost_usd
        return out

    def reset(self) -> None:
        self.calls.clear()


class ModelClient:
    """Calls a model by tier name ("small" / "large" / "judge")."""

    def __init__(self, config: Config | None = None, ledger: Ledger | None = None):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and paste your key."
            )

        self.client = Groq(api_key=api_key)
        self.config = config or load_config()
        self.ledger = ledger or Ledger()

    def complete(
        self,
        tier: str,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> CallResult:
        spec: ModelSpec = self.config.model(tier)

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        start = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=spec.name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        usage = resp.usage
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens

        result = CallResult(
            tier=tier,
            model=spec.name,
            text=resp.choices[0].message.content or "",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=spec.cost_usd(input_tokens, output_tokens),
            latency_ms=latency_ms,
        )
        return self.ledger.record(result)

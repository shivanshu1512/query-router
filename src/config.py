"""Loads config/models.yaml so that no price or model name is ever hardcoded."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

# Project root = the folder containing this file's parent (src/ -> query-router/)
ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "models.yaml"


@dataclass(frozen=True)
class ModelSpec:
    """One model's name and its (simulated) price."""

    tier: str  # "small" | "large" | "judge"
    name: str
    price_per_1m_input: float
    price_per_1m_output: float

    def cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        """Cost of a single call, in USD."""
        return (
            input_tokens * self.price_per_1m_input
            + output_tokens * self.price_per_1m_output
        ) / 1_000_000


@dataclass(frozen=True)
class Config:
    provider: str
    models: dict[str, ModelSpec]
    usd_to_inr: float

    def model(self, tier: str) -> ModelSpec:
        if tier not in self.models:
            raise KeyError(f"Unknown tier {tier!r}. Available: {list(self.models)}")
        return self.models[tier]


def load_config(path: Path = CONFIG_PATH) -> Config:
    if not path.exists():
        raise FileNotFoundError(f"Config not found at {path}")

    raw = yaml.safe_load(path.read_text())

    models = {
        tier: ModelSpec(
            tier=tier,
            name=spec["name"],
            price_per_1m_input=float(spec["price_per_1m_input"]),
            price_per_1m_output=float(spec["price_per_1m_output"]),
        )
        for tier, spec in raw["models"].items()
    }

    return Config(
        provider=raw["provider"],
        models=models,
        usd_to_inr=float(raw["usd_to_inr"]),
    )


if __name__ == "__main__":
    cfg = load_config()
    print(f"provider: {cfg.provider}")
    for tier, spec in cfg.models.items():
        print(f"  {tier:6s} -> {spec.name}")

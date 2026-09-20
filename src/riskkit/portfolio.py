"""Portfolios as fixed weights.

The public demo portfolio is a fixed set of broad ETFs held at constant weights
and rebalanced daily. That is a modelling choice, not a track record: it exists
so the risk models have a realistic multi-asset return series to be tested on.
Reading it from YAML keeps the composition visible and reviewable rather than
buried in code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

WEIGHT_TOLERANCE = 1e-6


@dataclass(frozen=True)
class Portfolio:
    name: str
    weights: pd.Series  # indexed by ticker, sums to 1
    description: str = ""

    @property
    def tickers(self) -> list[str]:
        return list(self.weights.index)

    def returns(self, asset_returns: pd.DataFrame) -> pd.Series:
        """Daily portfolio return with weights held constant (rebalanced each session)."""
        missing = set(self.tickers) - set(asset_returns.columns)
        if missing:
            raise ValueError(f"no returns for {sorted(missing)}")
        return asset_returns[self.tickers] @ self.weights

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Portfolio":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "Portfolio":
        weights = pd.Series(data["weights"], dtype=float)
        weights.index = [str(t).upper() for t in weights.index]
        if weights.empty:
            raise ValueError("portfolio has no positions")
        if (weights < 0).any():
            raise ValueError("negative weights are not supported")
        total = float(weights.sum())
        if abs(total - 1.0) > WEIGHT_TOLERANCE:
            raise ValueError(f"weights sum to {total:.6f}, not 1")
        return cls(name=data.get("name", "portfolio"), weights=weights, description=data.get("description", ""))

    @classmethod
    def equal_weight(cls, tickers: list[str], name: str = "equal weight") -> "Portfolio":
        weights = pd.Series(1.0 / len(tickers), index=[t.upper() for t in tickers])
        return cls(name=name, weights=weights)

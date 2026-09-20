"""Synthetic fixtures.

Vendor terms are personal-use, so no downloaded prices are committed. Tests run
on generated series whose parameters are known, which also means a test can
assert that an estimator recovers the value it was given.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from riskkit.data.calendar import sessions


@pytest.fixture(scope="session")
def trading_days() -> pd.DatetimeIndex:
    return sessions("2015-01-02", "2025-12-31")


def gbm_prices(index: pd.DatetimeIndex, vol: float, drift: float = 0.0, seed: int = 0, start: float = 100.0):
    """Geometric random walk with constant daily volatility ``vol``."""
    rng = np.random.default_rng(seed)
    shocks = rng.normal(drift - 0.5 * vol**2, vol, len(index))
    return pd.Series(start * np.exp(np.cumsum(shocks)), index=index)


@pytest.fixture(scope="session")
def price_panel(trading_days) -> pd.DataFrame:
    """Three assets with known, different volatilities."""
    specs = {"AAA": 0.008, "BBB": 0.012, "CCC": 0.020}
    return pd.DataFrame(
        {t: gbm_prices(trading_days, vol, seed=i) for i, (t, vol) in enumerate(specs.items())}
    )


class FakeSource:
    """A PriceSource that serves a fixed panel and counts fetches, so caching can be tested."""

    name = "fake"

    def __init__(self, panel: pd.DataFrame):
        self.panel = panel
        self.calls: list[list[str]] = []

    def fetch(self, tickers, start, end=None):
        self.calls.append(list(tickers))
        frame = self.panel.loc[:, list(tickers)]
        frame = frame.loc[frame.index >= pd.Timestamp(start)]
        return frame if end is None else frame.loc[frame.index <= pd.Timestamp(end)]

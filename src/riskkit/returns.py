"""Returns and volatility.

Simple returns aggregate across positions (a portfolio's return is the weighted
sum of its holdings' returns), so they are what the risk engine uses. Log
returns aggregate across *time* instead, which is why they show up in
annualisation and in models, not in portfolio arithmetic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
RISKMETRICS_LAMBDA = 0.94  # J.P. Morgan RiskMetrics decay for daily data
EWMA_SEED_WINDOW = 250     # sessions of burn-in used to seed the recursion


def simple_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """P_t / P_{t-1} - 1, with the first (all-NaN) row dropped."""
    return prices.pct_change().iloc[1:]


def log_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return np.log(prices).diff().iloc[1:]


def annualise_volatility(daily_vol: float | pd.Series, periods: int = TRADING_DAYS):
    """Scale a daily standard deviation by sqrt(time). Assumes returns are serially uncorrelated."""
    return daily_vol * np.sqrt(periods)


def rolling_volatility(returns: pd.Series, window: int = 252, annualised: bool = True) -> pd.Series:
    vol = returns.rolling(window).std(ddof=1)
    return annualise_volatility(vol) if annualised else vol


def ewma_volatility(
    returns: pd.Series,
    lam: float = RISKMETRICS_LAMBDA,
    seed_window: int = EWMA_SEED_WINDOW,
    annualised: bool = False,
) -> pd.Series:
    """Exponentially weighted volatility forecast, RiskMetrics style.

        sigma^2_t = lam * sigma^2_{t-1} + (1 - lam) * r^2_{t-1}

    Note the index: ``sigma_t`` uses returns up to ``t-1``, so the value on date
    ``t`` is the forecast *for* ``t`` and uses no contemporaneous information.
    Volatility clustering means yesterday's move informs today's risk, which is
    what a rolling window with equal weights misses.

    The recursion is seeded with the sample variance of the first ``seed_window``
    returns (zero mean, matching the RiskMetrics convention); those seed dates
    are returned as NaN rather than as under-informed forecasts.
    """
    if not 0 < lam < 1:
        raise ValueError("lam must lie in (0, 1)")
    r = returns.dropna()
    if len(r) <= seed_window:
        raise ValueError(f"need more than {seed_window} returns to seed the EWMA")

    values = r.to_numpy()
    var = float(np.mean(values[:seed_window] ** 2))  # zero-mean variance
    out = np.full(len(values), np.nan)
    for i in range(seed_window, len(values)):
        var = lam * var + (1.0 - lam) * values[i - 1] ** 2
        out[i] = var

    vol = pd.Series(np.sqrt(out), index=r.index, name="ewma_vol")
    return annualise_volatility(vol) if annualised else vol

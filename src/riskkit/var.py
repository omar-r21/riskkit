"""One-day-ahead value-at-risk and expected-shortfall forecasts.

Conventions used throughout:

* **Losses are positive.** A return of -3% is a loss of +3%. VaR and ES are
  therefore positive numbers, and a breach is ``loss > var``.
* ``alpha`` is the **tail probability**: 0.01 means 99% VaR, the loss that
  should be exceeded on about 1 day in 100.
* Every forecast on date ``t`` uses returns strictly before ``t``. Each model
  returns a frame indexed by date with a ``var`` and an ``es`` column, NaN
  during warm-up, so forecasts can be lined up against what actually happened
  without any look-ahead.

VaR answers "how bad is the threshold?" and says nothing about what lies beyond
it; ES answers "how bad is it *when* the threshold breaks", which is why it is
the Basel measure and why both are reported here.

The four models differ in one respect - where the shape of the tail comes from:

* ``Historical``: from the empirical window. No distribution assumed; fat tails
  are kept, but the estimate reacts slowly, because a calm day and a crisis day
  in the same window carry equal weight.
* ``Normal``: from a fitted normal. Simple, and thin-tailed where it matters.
* ``EWMANormal``: normal shape, but with a volatility that responds to recent
  moves, so it reacts quickly and still misses fat tails.
* ``FilteredHistorical``: empirical shape *and* responsive volatility -
  standardise past returns by their own volatility, then rescale to today's.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from scipy.stats import norm

from .returns import EWMA_SEED_WINDOW, RISKMETRICS_LAMBDA, ewma_volatility

HISTORICAL_WINDOW = 500
NORMAL_WINDOW = 500
FHS_WINDOW = 1000


def _empirical_var_es(losses: np.ndarray, alpha: float) -> tuple[float, float]:
    """VaR as the (1-alpha) loss quantile, ES as the mean loss at or beyond it."""
    var = float(np.quantile(losses, 1.0 - alpha, method="linear"))
    tail = losses[losses >= var]
    es = float(tail.mean()) if tail.size else var
    return var, es


def normal_var_es(sigma: float | np.ndarray, alpha: float):
    """Closed form for a zero-mean normal: VaR = z*sigma, ES = sigma*phi(z)/alpha."""
    z = norm.ppf(1.0 - alpha)
    return z * sigma, sigma * norm.pdf(z) / alpha


class VaRModel(Protocol):
    name: str

    def forecast(self, returns: pd.Series, alpha: float) -> pd.DataFrame: ...


@dataclass(frozen=True)
class Historical:
    """Empirical quantile of the last ``window`` losses."""

    window: int = HISTORICAL_WINDOW
    name: str = "historical"

    def forecast(self, returns: pd.Series, alpha: float) -> pd.DataFrame:
        losses = -returns.to_numpy(dtype=float)
        n = len(losses)
        var = np.full(n, np.nan)
        es = np.full(n, np.nan)
        for i in range(self.window, n):  # forecast for i uses [i-window, i-1]
            var[i], es[i] = _empirical_var_es(losses[i - self.window : i], alpha)
        return pd.DataFrame({"var": var, "es": es}, index=returns.index)


@dataclass(frozen=True)
class Normal:
    """Normal with a rolling, zero-mean volatility estimate.

    The mean is fixed at zero rather than estimated: at daily frequency the
    sample mean is tiny next to its own standard error, and estimating it adds
    noise to the risk number for no benefit.
    """

    window: int = NORMAL_WINDOW
    name: str = "normal"

    def forecast(self, returns: pd.Series, alpha: float) -> pd.DataFrame:
        r = returns.to_numpy(dtype=float)
        sigma = pd.Series(r).pow(2).rolling(self.window).mean().pow(0.5).shift(1).to_numpy()
        var, es = normal_var_es(sigma, alpha)
        return pd.DataFrame({"var": var, "es": es}, index=returns.index)


@dataclass(frozen=True)
class EWMANormal:
    """Normal shape with a RiskMetrics EWMA volatility: reacts to clustering, keeps thin tails."""

    lam: float = RISKMETRICS_LAMBDA
    seed_window: int = EWMA_SEED_WINDOW
    name: str = "ewma_normal"

    def forecast(self, returns: pd.Series, alpha: float) -> pd.DataFrame:
        sigma = ewma_volatility(returns, lam=self.lam, seed_window=self.seed_window)
        var, es = normal_var_es(sigma.to_numpy(), alpha)
        return pd.DataFrame({"var": var, "es": es}, index=sigma.index).reindex(returns.index)


@dataclass(frozen=True)
class FilteredHistorical:
    """Filtered historical simulation (Barone-Adesi / Hull-White).

    Divide each past return by the volatility that applied on its own day, giving
    standardised shocks that are much closer to identically distributed than raw
    returns. Rescale those shocks by today's volatility and read the empirical
    quantile off the result. The tail shape is the market's own; the scale is
    current. This is the method that usually survives the backtest.
    """

    window: int = FHS_WINDOW
    lam: float = RISKMETRICS_LAMBDA
    seed_window: int = EWMA_SEED_WINDOW
    name: str = "fhs"

    def forecast(self, returns: pd.Series, alpha: float) -> pd.DataFrame:
        sigma = ewma_volatility(returns, lam=self.lam, seed_window=self.seed_window).reindex(returns.index)
        # z_k uses the forecast made for day k, so it contains no look-ahead.
        z = (returns / sigma).to_numpy(dtype=float)
        sigma_next = sigma.to_numpy(dtype=float)

        n = len(returns)
        var = np.full(n, np.nan)
        es = np.full(n, np.nan)
        for i in range(n):
            if not np.isfinite(sigma_next[i]):
                continue
            window = z[max(0, i - self.window) : i]
            window = window[np.isfinite(window)]
            if window.size < self.window // 2:  # too few standardised shocks to read a tail from
                continue
            var[i], es[i] = _empirical_var_es(-window * sigma_next[i], alpha)
        return pd.DataFrame({"var": var, "es": es}, index=returns.index)


DEFAULT_MODELS: tuple[VaRModel, ...] = (Historical(), Normal(), EWMANormal(), FilteredHistorical())

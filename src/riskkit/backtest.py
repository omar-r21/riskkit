"""Backtesting VaR forecasts: are there the right number of breaches, and are they independent?

A VaR model is a probabilistic claim - "losses will exceed this level on about
alpha of days" - so it can be tested against what happened. Two things can go
wrong, and they are tested separately because a model can pass one and fail the
other:

* **Coverage.** Too many breaches means the model understates risk; too few
  means it wastes capital. Kupiec's proportion-of-failures test asks whether the
  observed count is plausible under the claimed alpha.
* **Independence.** Breaches should arrive like coin flips, not in bursts. A
  model that ignores volatility clustering produces the right *number* of
  breaches over a decade while failing in exactly the weeks that matter - all of
  them bunched into a crisis. Christoffersen's test looks at whether a breach
  today predicts a breach tomorrow.

Conditional coverage combines the two. Basel's traffic light is the supervisory
version: count breaches in a 250-day window at 99% and colour the result.

Every test here is a likelihood ratio: twice the log-likelihood gap between the
restricted model (the claim) and the unrestricted one (what the data suggests),
which is chi-squared distributed under the null.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import chi2

from .var import DEFAULT_MODELS, VaRModel

BASEL_WINDOW = 250
BASEL_GREEN_MAX = 4   # 0-4 breaches in 250 days at 99%
BASEL_YELLOW_MAX = 9  # 5-9 yellow, 10+ red


@dataclass(frozen=True)
class TestResult:
    statistic: float
    p_value: float
    df: int

    def rejects(self, level: float = 0.05) -> bool:
        """True if the model is rejected at this significance level."""
        return self.p_value < level


def _safe_log(x: float) -> float:
    """log with the 0*log(0) = 0 convention used in these likelihood ratios."""
    return float(np.log(x)) if x > 0 else 0.0


def kupiec_pof(breaches: np.ndarray, alpha: float) -> TestResult:
    """Kupiec (1995) proportion-of-failures test.

    LR = -2 ln[(1-p)^(n-x) p^x] + 2 ln[(1-x/n)^(n-x) (x/n)^x], chi-squared with 1 df.
    Tests the breach *rate* only - it is blind to clustering.
    """
    breaches = np.asarray(breaches, dtype=bool)
    n = breaches.size
    x = int(breaches.sum())
    if n == 0:
        raise ValueError("no observations to test")
    pi = x / n
    ll_null = (n - x) * _safe_log(1 - alpha) + x * _safe_log(alpha)
    ll_alt = (n - x) * _safe_log(1 - pi) + x * _safe_log(pi)
    stat = -2.0 * (ll_null - ll_alt)
    return TestResult(stat, float(chi2.sf(stat, 1)), 1)


def christoffersen_independence(breaches: np.ndarray) -> TestResult:
    """Christoffersen (1998) independence test: does a breach predict the next one?

    Compares a first-order Markov chain on the breach sequence against an
    independent one, using the transition counts n_ij. Chi-squared with 1 df.
    """
    b = np.asarray(breaches, dtype=int)
    if b.size < 2:
        raise ValueError("need at least two observations")
    prev, nxt = b[:-1], b[1:]
    n00 = int(np.sum((prev == 0) & (nxt == 0)))
    n01 = int(np.sum((prev == 0) & (nxt == 1)))
    n10 = int(np.sum((prev == 1) & (nxt == 0)))
    n11 = int(np.sum((prev == 1) & (nxt == 1)))

    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0

    ll_null = (n00 + n10) * _safe_log(1 - pi) + (n01 + n11) * _safe_log(pi)
    ll_alt = (
        n00 * _safe_log(1 - pi01) + n01 * _safe_log(pi01) + n10 * _safe_log(1 - pi11) + n11 * _safe_log(pi11)
    )
    stat = -2.0 * (ll_null - ll_alt)
    return TestResult(stat, float(chi2.sf(stat, 1)), 1)


def christoffersen_conditional_coverage(breaches: np.ndarray, alpha: float) -> TestResult:
    """Joint test of coverage and independence: LR_cc = LR_pof + LR_ind, chi-squared with 2 df."""
    stat = kupiec_pof(breaches, alpha).statistic + christoffersen_independence(breaches).statistic
    return TestResult(stat, float(chi2.sf(stat, 2)), 2)


def basel_zone(exceptions: int) -> str:
    """Basel traffic light for 250 days at 99%: green 0-4, yellow 5-9, red 10+."""
    if exceptions <= BASEL_GREEN_MAX:
        return "green"
    return "yellow" if exceptions <= BASEL_YELLOW_MAX else "red"


def basel_zones(breaches: pd.Series, window: int = BASEL_WINDOW) -> pd.Series:
    """Zone for every rolling ``window``-day period.

    Reported per window, not once over the whole sample: the traffic light is
    defined on 250 days, and a model can sit in green for years and still spend
    a crisis in red.
    """
    counts = breaches.astype(int).rolling(window).sum().dropna()
    return counts.map(lambda c: basel_zone(int(c)))


@dataclass(frozen=True)
class BacktestResult:
    """One model at one alpha, plus the day-by-day forecasts it made."""

    model: str
    alpha: float
    forecasts: pd.DataFrame = field(repr=False)  # var, es, loss, breach
    kupiec: TestResult
    independence: TestResult
    conditional_coverage: TestResult
    zones: pd.Series = field(repr=False)

    @property
    def observations(self) -> int:
        return len(self.forecasts)

    @property
    def breaches(self) -> int:
        return int(self.forecasts["breach"].sum())

    @property
    def expected_breaches(self) -> float:
        return self.alpha * self.observations

    @property
    def breach_rate(self) -> float:
        return self.breaches / self.observations

    @property
    def mean_shortfall_ratio(self) -> float:
        """Average realised loss on breach days, divided by the ES forecast for those days.

        Above 1 means the model understated the tail it claimed to measure. This is
        a diagnostic, not a formal ES test.
        """
        bad = self.forecasts[self.forecasts["breach"]]
        return float((bad["loss"] / bad["es"]).mean()) if len(bad) else float("nan")

    def zone_shares(self) -> dict[str, float]:
        if self.zones.empty:
            return {}
        counts = self.zones.value_counts(normalize=True)
        return {zone: float(counts.get(zone, 0.0)) for zone in ("green", "yellow", "red")}


def backtest_model(returns: pd.Series, model: VaRModel, alpha: float, window: int = BASEL_WINDOW) -> BacktestResult:
    """Score one model's forecasts against realised losses, out of sample by construction."""
    forecasts = model.forecast(returns, alpha)
    frame = forecasts.assign(loss=-returns).dropna(subset=["var", "es", "loss"])
    if frame.empty:
        raise ValueError(f"{model.name} produced no usable forecasts")
    frame["breach"] = frame["loss"] > frame["var"]

    breaches = frame["breach"].to_numpy()
    return BacktestResult(
        model=model.name,
        alpha=alpha,
        forecasts=frame,
        kupiec=kupiec_pof(breaches, alpha),
        independence=christoffersen_independence(breaches),
        conditional_coverage=christoffersen_conditional_coverage(breaches, alpha),
        zones=basel_zones(frame["breach"], window) if alpha == 0.01 else pd.Series(dtype=object),
    )


def run_backtest(
    returns: pd.Series,
    models: tuple[VaRModel, ...] = DEFAULT_MODELS,
    alphas: tuple[float, ...] = (0.05, 0.01),
    common_sample: bool = True,
) -> list[BacktestResult]:
    """Backtest every model at every alpha.

    With ``common_sample`` the models are compared on the dates where they all
    produced a forecast - otherwise the one with the shortest warm-up gets extra
    (and different) days, and the breach counts are not comparable.
    """
    returns = returns.dropna()
    results: list[BacktestResult] = []

    start = None
    if common_sample:
        firsts = [model.forecast(returns, alphas[0]).dropna().index.min() for model in models]
        start = max(firsts)

    for model in models:
        for alpha in alphas:
            sample = returns if start is None else returns.loc[returns.index >= start]
            # Give each model its own warm-up history, then score only the common sample.
            full = backtest_model(returns, model, alpha)
            frame = full.forecasts.loc[full.forecasts.index >= sample.index.min()]
            breaches = frame["breach"].to_numpy()
            results.append(
                BacktestResult(
                    model=model.name,
                    alpha=alpha,
                    forecasts=frame,
                    kupiec=kupiec_pof(breaches, alpha),
                    independence=christoffersen_independence(breaches),
                    conditional_coverage=christoffersen_conditional_coverage(breaches, alpha),
                    zones=basel_zones(frame["breach"]) if alpha == 0.01 else pd.Series(dtype=object),
                )
            )
    return results


def summary_table(results: list[BacktestResult]) -> pd.DataFrame:
    """The headline table: one row per model and alpha."""
    rows = []
    for r in results:
        shares = r.zone_shares()
        rows.append(
            {
                "model": r.model,
                "alpha": r.alpha,
                "days": r.observations,
                "breaches": r.breaches,
                "expected": round(r.expected_breaches, 1),
                "rate": r.breach_rate,
                "kupiec_p": r.kupiec.p_value,
                "independence_p": r.independence.p_value,
                "cc_p": r.conditional_coverage.p_value,
                "loss_over_es": r.mean_shortfall_ratio,
                "yellow_share": shares.get("yellow", float("nan")),
                "red_share": shares.get("red", float("nan")),
            }
        )
    return pd.DataFrame(rows).sort_values(["alpha", "model"]).reset_index(drop=True)

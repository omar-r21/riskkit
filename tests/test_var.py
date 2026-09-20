import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from riskkit.var import (
    EWMANormal,
    FilteredHistorical,
    Historical,
    Normal,
    _empirical_var_es,
    normal_var_es,
)


def test_normal_var_es_closed_form():
    sigma, alpha = 0.02, 0.01
    var, es = normal_var_es(sigma, alpha)
    assert var == pytest.approx(2.326347874 * sigma, rel=1e-9)  # z_{0.99}
    assert es == pytest.approx(sigma * norm.pdf(norm.ppf(0.99)) / alpha, rel=1e-12)
    assert es > var  # the average loss beyond the threshold exceeds the threshold


def test_normal_es_matches_hand_computed_value():
    # sigma = 1%: ES_99 = sigma * phi(z)/alpha = 0.01 * 0.026652 / 0.01
    var, es = normal_var_es(0.01, 0.01)
    assert var == pytest.approx(0.0232635, abs=1e-7)
    assert es == pytest.approx(0.0266521, abs=1e-7)


def test_empirical_var_es_on_a_hand_built_sample():
    # 20 losses, 1% .. 20%. The 90th percentile by linear interpolation is 18.1%.
    losses = np.arange(1, 21, dtype=float)
    var, es = _empirical_var_es(losses, alpha=0.10)
    assert var == pytest.approx(18.1)
    assert es == pytest.approx(np.mean([19.0, 20.0]))  # mean of losses at or beyond VaR


def test_var_is_monotone_in_alpha():
    losses = np.random.default_rng(0).normal(0, 1, 5000)
    var99, es99 = _empirical_var_es(losses, 0.01)
    var95, es95 = _empirical_var_es(losses, 0.05)
    assert var99 > var95 and es99 > es95 and es95 > var95


@pytest.fixture(scope="module")
def iid_returns() -> pd.Series:
    """Constant-volatility normal returns: every model should be roughly right here."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2005-01-03", periods=4000, freq="B")
    return pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)


@pytest.mark.parametrize("model", [Historical(), Normal(), EWMANormal(), FilteredHistorical()])
def test_models_recover_the_true_quantile_of_an_iid_series(model, iid_returns):
    # True 99% VaR of a zero-mean normal with sigma = 1% is z * sigma.
    truth = norm.ppf(0.99) * 0.01
    forecasts = model.forecast(iid_returns, 0.01).dropna()
    assert len(forecasts) > 2000
    assert forecasts["var"].mean() == pytest.approx(truth, rel=0.10)
    assert forecasts["es"].mean() > forecasts["var"].mean()


@pytest.mark.parametrize("model", [Historical(), Normal(), EWMANormal(), FilteredHistorical()])
def test_forecasts_never_use_the_day_they_predict(model, iid_returns):
    """Changing one return must not change the forecast made for that same day."""
    tampered = iid_returns.copy()
    day = 3000
    tampered.iloc[day] = -0.5  # a catastrophic move

    before = model.forecast(iid_returns, 0.01)["var"]
    after = model.forecast(tampered, 0.01)["var"]
    assert after.iloc[day] == pytest.approx(before.iloc[day], rel=1e-12)
    assert after.iloc[day + 1] > before.iloc[day + 1]  # but the next day must react


def test_volatility_aware_models_react_faster_than_static_ones():
    """After a volatility jump, EWMA and FHS should raise VaR well before the rolling window does."""
    rng = np.random.default_rng(1)
    idx = pd.date_range("2005-01-03", periods=2000, freq="B")
    quiet = rng.normal(0, 0.005, 1500)
    stormy = rng.normal(0, 0.025, 500)
    returns = pd.Series(np.concatenate([quiet, stormy]), index=idx)

    day = 1520  # 20 sessions into the stormy regime
    fast = [m.forecast(returns, 0.01)["var"].iloc[day] for m in (EWMANormal(), FilteredHistorical())]
    slow = [m.forecast(returns, 0.01)["var"].iloc[day] for m in (Historical(), Normal())]
    assert min(fast) > 1.5 * max(slow)


def test_fhs_keeps_a_fat_tail_that_the_normal_model_misses():
    """With fat-tailed shocks at constant volatility, FHS should sit above the normal model at 99%."""
    rng = np.random.default_rng(3)
    idx = pd.date_range("2005-01-03", periods=3000, freq="B")
    returns = pd.Series(rng.standard_t(df=3, size=len(idx)) * 0.005, index=idx)

    fhs = FilteredHistorical().forecast(returns, 0.01)["var"]
    normal = Normal().forecast(returns, 0.01)["var"]
    common = fhs.dropna().index.intersection(normal.dropna().index)
    assert fhs.loc[common].mean() > normal.loc[common].mean()


def test_historical_forecasts_change_only_when_the_window_moves():
    """Crisis losses drive the forecast while they are in the window, and not a day longer.

    This is the ghost feature of historical simulation: the estimate drops
    mechanically on the anniversary of the crash, with nothing having changed
    in the market.
    """
    idx = pd.date_range("2005-01-03", periods=1200, freq="B")
    returns = pd.Series(np.full(len(idx), -0.001), index=idx)
    returns.iloc[596:601] = -0.30  # five crisis days, i.e. 2% of a 250-day window

    var = Historical(window=250).forecast(returns, 0.01)["var"]
    assert var.iloc[700] > 0.05             # crisis inside the window
    assert var.iloc[600 + 251] < 0.01       # first day it has rolled out
    assert var.iloc[650] == pytest.approx(var.iloc[700])  # flat while the window holds it


def test_a_lone_outlier_sits_above_the_99th_percentile():
    """One bad day in 250 does not move 99% VaR - but it does move ES, which is the point of ES."""
    idx = pd.date_range("2005-01-03", periods=1200, freq="B")
    returns = pd.Series(np.full(len(idx), -0.001), index=idx)
    returns.iloc[600] = -0.30

    forecast = Historical(window=250).forecast(returns, 0.01)
    assert forecast["var"].iloc[700] == pytest.approx(0.001)   # unmoved
    assert forecast["es"].iloc[700] > 0.002                    # ES sees the tail
    assert forecast["es"].iloc[600 + 251] == pytest.approx(0.001)

import numpy as np
import pandas as pd
import pytest

from riskkit import annualise_volatility, ewma_volatility, log_returns, rolling_volatility, simple_returns
from riskkit.returns import EWMA_SEED_WINDOW, RISKMETRICS_LAMBDA, TRADING_DAYS


def test_simple_returns_on_a_hand_series():
    prices = pd.Series([100.0, 110.0, 99.0], index=pd.date_range("2024-01-02", periods=3, freq="D"))
    np.testing.assert_allclose(simple_returns(prices).to_numpy(), [0.10, -0.10])


def test_log_returns_sum_over_time_where_simple_returns_do_not():
    prices = pd.Series([100.0, 110.0, 99.0], index=pd.date_range("2024-01-02", periods=3, freq="D"))
    total = prices.iloc[-1] / prices.iloc[0] - 1
    assert log_returns(prices).sum() == pytest.approx(np.log1p(total))
    assert simple_returns(prices).sum() != pytest.approx(total)  # 0.10 - 0.10 = 0, but the price fell 1%


def test_annualisation_is_square_root_of_time():
    assert annualise_volatility(0.01) == pytest.approx(0.01 * np.sqrt(TRADING_DAYS))
    assert annualise_volatility(0.01, periods=4) == pytest.approx(0.02)


def test_volatility_estimates_recover_the_simulated_value(price_panel):
    # AAA is simulated at 0.8% daily; over 11 years the estimate should be close.
    returns = simple_returns(price_panel["AAA"])
    assert returns.std(ddof=1) == pytest.approx(0.008, rel=0.05)
    assert rolling_volatility(returns).dropna().mean() == pytest.approx(annualise_volatility(0.008), rel=0.05)


def test_ewma_matches_the_pandas_recursion_given_the_same_seed():
    rng = np.random.default_rng(7)
    returns = pd.Series(rng.normal(0, 0.01, 1500), index=pd.date_range("2018-01-01", periods=1500, freq="B"))

    ours = ewma_volatility(returns)

    # The recursion starts at the end of burn-in from the seed variance, and each
    # forecast uses the previous day's return. So the equivalent pandas input is
    # the seed followed by the squared returns from the last burn-in day onwards;
    # its first value is the seed itself, which is not a forecast.
    seed_var = float(np.mean(returns.to_numpy()[:EWMA_SEED_WINDOW] ** 2))
    squared = returns.to_numpy()[EWMA_SEED_WINDOW - 1 : -1] ** 2
    reference = (
        pd.Series(np.concatenate([[seed_var], squared]))
        .ewm(alpha=1 - RISKMETRICS_LAMBDA, adjust=False)
        .mean()
        .to_numpy()[1:]
    )

    np.testing.assert_allclose(ours.dropna().to_numpy(), np.sqrt(reference), rtol=1e-12)


def test_ewma_forecast_uses_no_contemporaneous_return():
    # A single huge return must move the forecast for the NEXT day, not its own.
    returns = pd.Series(np.full(600, 0.001), index=pd.date_range("2020-01-01", periods=600, freq="B"))
    returns.iloc[400] = 0.20

    vol = ewma_volatility(returns)
    assert vol.iloc[400] == pytest.approx(vol.iloc[399], rel=1e-9)
    assert vol.iloc[401] > 5 * vol.iloc[399]


def test_ewma_decays_back_towards_the_calm_level():
    returns = pd.Series(np.full(900, 0.001), index=pd.date_range("2020-01-01", periods=900, freq="B"))
    returns.iloc[400] = 0.20
    vol = ewma_volatility(returns)
    assert vol.iloc[500] < vol.iloc[401]
    assert vol.iloc[880] == pytest.approx(0.001, rel=0.5)


def test_ewma_rejects_bad_inputs():
    returns = pd.Series(np.zeros(400), index=pd.date_range("2020-01-01", periods=400, freq="B"))
    with pytest.raises(ValueError):
        ewma_volatility(returns, lam=1.5)
    with pytest.raises(ValueError, match="seed"):
        ewma_volatility(returns.iloc[:100])

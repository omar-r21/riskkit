import numpy as np
import pandas as pd
import pytest

from riskkit import Portfolio, simple_returns
from riskkit.attribution import (
    component_es,
    concentration,
    ledoit_wolf_covariance,
    parametric_var_attribution,
)


@pytest.fixture(scope="module")
def asset_returns() -> pd.DataFrame:
    """Three assets with known volatilities: two correlated, one independent."""
    rng = np.random.default_rng(17)
    n = 3000
    idx = pd.date_range("2010-01-04", periods=n, freq="B")
    market = rng.normal(0, 0.010, n)
    return pd.DataFrame(
        {
            "BIG": market + rng.normal(0, 0.002, n),          # ~1.0% vol, correlated
            "MID": 0.5 * market + rng.normal(0, 0.004, n),    # ~0.6% vol, correlated
            "SOLO": rng.normal(0, 0.012, n),                  # ~1.2% vol, independent
        },
        index=idx,
    )


def test_parametric_contributions_sum_to_total_var(asset_returns):
    weights = pd.Series({"BIG": 0.5, "MID": 0.3, "SOLO": 0.2})
    result = parametric_var_attribution(asset_returns, weights)
    assert result.contribution.sum() == pytest.approx(result.total, rel=1e-12)
    assert result.share.sum() == pytest.approx(1.0, rel=1e-12)


def test_component_es_contributions_sum_to_total_es(asset_returns):
    weights = pd.Series({"BIG": 0.5, "MID": 0.3, "SOLO": 0.2})
    result = component_es(asset_returns, weights)
    assert result.contribution.sum() == pytest.approx(result.total, rel=1e-12)


def test_component_es_matches_a_finite_difference_marginal(asset_returns):
    """The Euler claim, checked numerically: scaling a position scales ES by its contribution.

    ES is homogeneous of degree one, so d(ES)/d(w_i) * w_i is the contribution.
    """
    weights = pd.Series({"BIG": 0.5, "MID": 0.3, "SOLO": 0.2})
    base = component_es(asset_returns, weights)

    h = 1e-5
    for ticker in weights.index:
        bumped = weights.copy()
        bumped[ticker] += h
        up = component_es(asset_returns, bumped).total
        marginal = (up - base.total) / h
        assert marginal * weights[ticker] == pytest.approx(base.contribution[ticker], rel=1e-3)


def test_risk_share_differs_from_weight_for_a_correlated_position(asset_returns):
    """The point of the module: a correlated position carries more risk than its weight."""
    weights = pd.Series({"BIG": 0.4, "MID": 0.3, "SOLO": 0.3})
    table = parametric_var_attribution(asset_returns, weights).table()

    assert table.loc["BIG", "risk_over_weight"] > 1.1     # moves with the portfolio
    assert table.loc["SOLO", "risk_over_weight"] < 1.0    # diversifies despite the highest vol
    assert asset_returns["SOLO"].std() > asset_returns["BIG"].std()


def test_a_hedge_can_contribute_negative_risk():
    """A position that moves against the rest reduces total risk, so its contribution is negative."""
    rng = np.random.default_rng(3)
    n = 2000
    idx = pd.date_range("2012-01-02", periods=n, freq="B")
    market = rng.normal(0, 0.01, n)
    returns = pd.DataFrame({"LONG": market, "HEDGE": -0.9 * market + rng.normal(0, 0.001, n)}, index=idx)

    result = parametric_var_attribution(returns, pd.Series({"LONG": 0.8, "HEDGE": 0.2}))
    assert result.contribution["HEDGE"] < 0
    assert result.contribution.sum() == pytest.approx(result.total, rel=1e-12)


def test_equal_weights_on_identical_independent_assets_split_risk_evenly():
    rng = np.random.default_rng(5)
    n = 4000
    idx = pd.date_range("2010-01-04", periods=n, freq="B")
    returns = pd.DataFrame({name: rng.normal(0, 0.01, n) for name in ("A", "B", "C", "D")}, index=idx)

    shares = parametric_var_attribution(returns, pd.Series(0.25, index=["A", "B", "C", "D"])).share
    np.testing.assert_allclose(shares.to_numpy(), 0.25, atol=0.02)


def test_shrinkage_pulls_correlations_toward_their_average(asset_returns):
    shrunk, intensity = ledoit_wolf_covariance(asset_returns)
    sample = asset_returns.cov()

    assert 0.0 <= intensity <= 1.0
    # Variances are preserved by the target, so the diagonal barely moves.
    np.testing.assert_allclose(np.diag(shrunk.to_numpy()), np.diag(sample.to_numpy()), rtol=0.02)
    # The strongest correlation is pulled in toward the mean correlation.
    def corr(frame):
        d = np.sqrt(np.diag(frame.to_numpy()))
        return frame.to_numpy() / np.outer(d, d)
    assert abs(corr(shrunk)[0, 1]) < abs(corr(sample)[0, 1])


def test_shrinkage_is_stronger_when_there_is_less_data(asset_returns):
    _, long_sample = ledoit_wolf_covariance(asset_returns)
    _, short_sample = ledoit_wolf_covariance(asset_returns.iloc[:120])
    assert short_sample > long_sample


def test_concentration_counts_effective_positions():
    rng = np.random.default_rng(9)
    n = 1500
    idx = pd.date_range("2015-01-02", periods=n, freq="B")
    returns = pd.DataFrame({name: rng.normal(0, 0.01, n) for name in ("A", "B", "C", "D")}, index=idx)

    even = concentration(returns, pd.Series(0.25, index=["A", "B", "C", "D"]))
    assert even.effective_positions == pytest.approx(4.0)
    assert even.herfindahl == pytest.approx(0.25)

    lopsided = concentration(returns, pd.Series({"A": 0.85, "B": 0.05, "C": 0.05, "D": 0.05}))
    assert lopsided.effective_positions < 1.5
    assert lopsided.top_weight_share == pytest.approx(0.85)


def test_diversification_ratio_is_one_when_everything_moves_together():
    n = 1200
    idx = pd.date_range("2015-01-02", periods=n, freq="B")
    market = np.random.default_rng(1).normal(0, 0.01, n)
    identical = pd.DataFrame({"A": market, "B": market}, index=idx)
    independent = pd.DataFrame(
        {"A": market, "B": np.random.default_rng(2).normal(0, 0.01, n)}, index=idx
    )
    weights = pd.Series({"A": 0.5, "B": 0.5})

    assert concentration(identical, weights).diversification_ratio == pytest.approx(1.0, abs=1e-9)
    assert concentration(independent, weights).diversification_ratio > 1.3


def test_attribution_table_is_ordered_and_complete(asset_returns):
    weights = pd.Series({"BIG": 0.5, "MID": 0.3, "SOLO": 0.2})
    table = component_es(asset_returns, weights).table()
    assert list(table.columns) == ["weight", "contribution", "risk_share", "risk_over_weight"]
    assert table["contribution"].is_monotonic_decreasing
    assert table["weight"].sum() == pytest.approx(1.0)


def test_demo_portfolio_risk_shares_are_not_its_weights(price_panel):
    """End to end on the synthetic panel: the biggest position need not be the biggest risk."""
    returns = simple_returns(price_panel).dropna()
    portfolio = Portfolio.from_dict({"name": "t", "weights": {"AAA": 0.6, "BBB": 0.3, "CCC": 0.1}})
    table = component_es(returns, portfolio.weights).table()
    assert not np.allclose(table["risk_share"], table["weight"])
    assert table["risk_share"].sum() == pytest.approx(1.0)

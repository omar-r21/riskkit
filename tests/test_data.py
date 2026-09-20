import numpy as np
import pandas as pd
import pytest

from riskkit import Portfolio, align, load_prices, sessions
from riskkit.data.calendar import MAX_STALENESS

from .conftest import FakeSource


def test_sessions_skip_weekends_and_holidays():
    days = sessions("2024-07-01", "2024-07-08")
    assert pd.Timestamp("2024-07-04") not in days  # Independence Day
    assert pd.Timestamp("2024-07-06") not in days  # Saturday
    assert pd.Timestamp("2024-07-05") in days


def test_align_is_a_no_op_on_a_clean_panel(price_panel):
    aligned, report = align(price_panel)
    assert report.clean
    assert report.sessions == len(price_panel)
    pd.testing.assert_frame_equal(aligned, price_panel)


def test_short_gaps_are_filled_and_long_ones_are_reported(price_panel):
    panel = price_panel.copy()
    short = panel.index[100:100 + MAX_STALENESS]
    long = panel.index[200:200 + MAX_STALENESS + 3]
    panel.loc[short, "AAA"] = np.nan
    panel.loc[long, "BBB"] = np.nan

    aligned, report = align(panel)
    assert report.filled["AAA"] == MAX_STALENESS
    assert aligned.loc[short, "AAA"].eq(panel["AAA"].iloc[99]).all()  # carried forward
    # The long gap is filled up to the limit, and the rest stays missing.
    assert report.filled["BBB"] == MAX_STALENESS
    assert report.missing["BBB"] == 3
    assert not report.clean


def test_history_before_listing_is_never_invented(price_panel):
    panel = price_panel.copy()
    panel.loc[panel.index[:300], "CCC"] = np.nan  # listed late
    aligned, _ = align(panel)
    assert aligned["CCC"].iloc[:300].isna().all()


def test_tickers_without_enough_history_are_dropped(price_panel):
    panel = price_panel.copy()
    panel.loc[panel.index[:-100], "CCC"] = np.nan  # only 100 sessions of history
    aligned, report = align(panel)
    assert "CCC" not in aligned.columns
    assert "CCC" in report.dropped and "100 sessions" in report.dropped["CCC"]


def test_align_rejects_an_empty_frame():
    with pytest.raises(ValueError):
        align(pd.DataFrame())


def test_prices_are_cached_and_refetched_only_on_demand(price_panel, tmp_path):
    source = FakeSource(price_panel)
    first = load_prices(["AAA", "BBB"], start="2015-01-02", source=source, cache_dir=tmp_path)
    assert source.calls == [["AAA", "BBB"]]

    second = load_prices(["AAA", "BBB"], start="2015-01-02", source=source, cache_dir=tmp_path)
    assert source.calls == [["AAA", "BBB"]]  # served from cache
    pd.testing.assert_frame_equal(first, second)

    load_prices(["AAA", "CCC"], start="2015-01-02", source=source, cache_dir=tmp_path)
    assert source.calls[-1] == ["CCC"]  # only the new ticker

    load_prices(["AAA"], start="2015-01-02", source=source, cache_dir=tmp_path, refresh=True)
    assert source.calls[-1] == ["AAA"]


def test_cache_is_refetched_when_it_starts_too_late(price_panel, tmp_path):
    source = FakeSource(price_panel)
    load_prices(["AAA"], start="2020-01-02", source=source, cache_dir=tmp_path)
    load_prices(["AAA"], start="2015-01-02", source=source, cache_dir=tmp_path)
    assert source.calls == [["AAA"], ["AAA"]]  # cache didn't reach back far enough


def test_load_prices_honours_the_date_range(price_panel, tmp_path):
    prices = load_prices(
        ["AAA"], start="2018-01-02", end="2018-12-31", source=FakeSource(price_panel), cache_dir=tmp_path
    )
    assert prices.index.min() >= pd.Timestamp("2018-01-02")
    assert prices.index.max() <= pd.Timestamp("2018-12-31")


def test_demo_portfolio_is_well_formed():
    portfolio = Portfolio.from_yaml("portfolios/demo.yaml")
    assert portfolio.weights.sum() == pytest.approx(1.0)
    assert len(portfolio.tickers) == 10
    assert "SPY" in portfolio.tickers


def test_portfolio_return_is_the_weighted_sum(price_panel):
    portfolio = Portfolio.from_dict({"name": "t", "weights": {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}})
    asset_returns = price_panel.pct_change().dropna()
    expected = 0.5 * asset_returns["AAA"] + 0.3 * asset_returns["BBB"] + 0.2 * asset_returns["CCC"]
    pd.testing.assert_series_equal(portfolio.returns(asset_returns), expected, check_names=False)


@pytest.mark.parametrize(
    "weights",
    [{"AAA": 0.5, "BBB": 0.4}, {"AAA": 1.5, "BBB": -0.5}, {}],
    ids=["does not sum to one", "negative weight", "empty"],
)
def test_bad_portfolios_are_rejected(weights):
    with pytest.raises(ValueError):
        Portfolio.from_dict({"name": "bad", "weights": weights})


def test_portfolio_needs_returns_for_every_holding(price_panel):
    portfolio = Portfolio.equal_weight(["AAA", "ZZZ"])
    with pytest.raises(ValueError, match="ZZZ"):
        portfolio.returns(price_panel.pct_change().dropna())

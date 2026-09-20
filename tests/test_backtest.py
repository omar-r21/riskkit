import math

import numpy as np
import pandas as pd
import pytest

from riskkit.backtest import (
    basel_zone,
    basel_zones,
    christoffersen_conditional_coverage,
    christoffersen_independence,
    kupiec_pof,
    run_backtest,
    summary_table,
)
from riskkit.var import EWMANormal, FilteredHistorical, Historical, Normal


def hand_kupiec(n: int, x: int, p: float) -> float:
    """Independent implementation of the LR statistic, straight from the formula."""
    pi = x / n
    ll_null = (n - x) * math.log(1 - p) + x * math.log(p)
    ll_alt = (n - x) * math.log(1 - pi) + x * math.log(pi)
    return -2 * (ll_null - ll_alt)


def breach_array(n: int, positions) -> np.ndarray:
    out = np.zeros(n, dtype=bool)
    out[list(positions)] = True
    return out


def test_kupiec_matches_a_hand_computed_textbook_case():
    # 250 days, 8 breaches, 99% VaR: the classic "is 8 too many?" example.
    breaches = breach_array(250, range(8))
    result = kupiec_pof(breaches, 0.01)
    # Hand check: ll_null = 242*ln(.99) + 8*ln(.01) = -39.2736,
    #             ll_alt  = 242*ln(.968) + 8*ln(.032) = -35.4067, LR = 7.7338.
    assert result.statistic == pytest.approx(hand_kupiec(250, 8, 0.01), rel=1e-12)
    assert result.statistic == pytest.approx(7.7338, abs=1e-3)
    assert result.p_value == pytest.approx(0.0054, abs=1e-3)
    assert result.rejects(0.01)  # 8 breaches where 2.5 were expected


def test_kupiec_is_near_zero_when_the_rate_is_exactly_right():
    breaches = breach_array(1000, range(10))  # exactly 1%
    result = kupiec_pof(breaches, 0.01)
    assert result.statistic == pytest.approx(0.0, abs=1e-12)
    assert result.p_value == pytest.approx(1.0, abs=1e-12)


def test_kupiec_rejects_a_model_with_far_too_many_breaches():
    assert kupiec_pof(breach_array(1000, range(60)), 0.01).p_value < 1e-10


def test_kupiec_handles_zero_breaches():
    result = kupiec_pof(np.zeros(500, dtype=bool), 0.01)  # 0*log(0) convention
    assert result.statistic == pytest.approx(-2 * 500 * math.log(0.99), rel=1e-12)


def test_independence_test_ignores_spread_out_breaches():
    breaches = breach_array(1000, range(0, 1000, 100))  # 10 breaches, evenly spaced
    assert not christoffersen_independence(breaches).rejects(0.05)


def test_independence_test_catches_clustered_breaches():
    breaches = breach_array(1000, range(500, 515))  # all 15 in one fortnight
    result = christoffersen_independence(breaches)
    assert result.rejects(0.01)


def test_conditional_coverage_is_the_sum_of_the_two_statistics():
    breaches = breach_array(1000, list(range(200, 208)) + [400, 700])
    pof = kupiec_pof(breaches, 0.01).statistic
    ind = christoffersen_independence(breaches).statistic
    cc = christoffersen_conditional_coverage(breaches, 0.01)
    assert cc.statistic == pytest.approx(pof + ind, rel=1e-12)
    assert cc.df == 2


def test_a_model_can_pass_coverage_and_still_fail_on_clustering():
    """The point of the independence test: the right number of breaches, all at once."""
    breaches = breach_array(1000, range(300, 310))  # exactly 1%, but consecutive
    assert not kupiec_pof(breaches, 0.01).rejects(0.05)
    assert christoffersen_independence(breaches).rejects(0.01)
    assert christoffersen_conditional_coverage(breaches, 0.01).rejects(0.01)


@pytest.mark.parametrize(
    "exceptions,zone",
    [(0, "green"), (4, "green"), (5, "yellow"), (9, "yellow"), (10, "red"), (25, "red")],
)
def test_basel_traffic_light_boundaries(exceptions, zone):
    assert basel_zone(exceptions) == zone


def test_basel_zones_are_computed_per_rolling_window():
    idx = pd.date_range("2005-01-03", periods=600, freq="B")
    breaches = pd.Series(False, index=idx)
    breaches.iloc[100:115] = True  # 15 breaches in one burst -> red while in the window

    zones = basel_zones(breaches, window=250)
    assert (zones == "red").any()
    assert zones.iloc[-1] == "green"  # burst long gone


@pytest.fixture(scope="module")
def clustered_returns() -> pd.Series:
    """Two-regime series: the textbook case where a static model fails on clustering."""
    rng = np.random.default_rng(11)
    idx = pd.date_range("2005-01-03", periods=4000, freq="B")
    vol = np.full(len(idx), 0.006)
    for start in (1800, 2600, 3400):
        vol[start : start + 120] = 0.030  # crisis bursts
    return pd.Series(rng.normal(0, vol), index=idx)


def test_end_to_end_backtest_on_a_constant_volatility_series():
    """With no clustering, a well-specified model should pass both tests."""
    rng = np.random.default_rng(5)
    idx = pd.date_range("2005-01-03", periods=4000, freq="B")
    returns = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)

    result = next(r for r in run_backtest(returns, (Normal(),), (0.01,)))
    assert result.breach_rate == pytest.approx(0.01, abs=0.006)
    assert not result.kupiec.rejects(0.01)
    assert not result.independence.rejects(0.01)


def test_volatility_aware_models_beat_static_ones_under_clustering(clustered_returns):
    results = {
        r.model: r
        for r in run_backtest(clustered_returns, (Normal(), EWMANormal(), FilteredHistorical()), (0.01,))
    }
    # The static model's breaches bunch into the crisis bursts.
    assert results["normal"].independence.p_value < results["ewma_normal"].independence.p_value
    # And it takes more hits than it should.
    assert results["normal"].breach_rate > results["fhs"].breach_rate


def test_all_models_are_scored_on_the_same_days(clustered_returns):
    results = run_backtest(clustered_returns, (Historical(), Normal(), EWMANormal(), FilteredHistorical()), (0.01,))
    spans = {(r.forecasts.index.min(), r.forecasts.index.max()) for r in results}
    assert len(spans) == 1, "models compared over different samples are not comparable"


def test_summary_table_shape(clustered_returns):
    results = run_backtest(clustered_returns, (Normal(), EWMANormal()), (0.05, 0.01))
    table = summary_table(results)
    assert len(table) == 4
    assert {"model", "alpha", "breaches", "expected", "kupiec_p", "cc_p", "red_share"} <= set(table.columns)
    assert table["breaches"].gt(0).all()


def test_breach_and_expected_counts_line_up(clustered_returns):
    result = next(r for r in run_backtest(clustered_returns, (FilteredHistorical(),), (0.05,)))
    assert result.expected_breaches == pytest.approx(0.05 * result.observations)
    assert result.breaches == int(result.forecasts["breach"].sum())

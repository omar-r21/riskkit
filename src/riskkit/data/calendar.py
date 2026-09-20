"""Align price series to exchange trading sessions.

Raw vendor data has gaps: a ticker may miss a session, list late, or stop
trading. Risk estimates need one row per session with no invented data, so the
rules here are deliberate and reported rather than silently applied:

* reindex onto the exchange calendar, so every series shares an index;
* forward-fill a short gap (a missed print in an otherwise live series), but
  only up to ``max_staleness`` sessions;
* leave anything longer as missing, and report it.

Forward-filling a price creates a zero return, which understates volatility.
That is acceptable for one or two sessions and not beyond, which is why the
limit is small and the filled count is returned to the caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import pandas_market_calendars as mcal

DEFAULT_EXCHANGE = "NYSE"
MAX_STALENESS = 2  # sessions a price may be carried forward


def sessions(start, end, exchange: str = DEFAULT_EXCHANGE) -> pd.DatetimeIndex:
    """Trading sessions in [start, end], as tz-naive dates."""
    schedule = mcal.get_calendar(exchange).schedule(start_date=start, end_date=end)
    return pd.DatetimeIndex(schedule.index).tz_localize(None).normalize()


@dataclass(frozen=True)
class AlignmentReport:
    """What had to be done to make the panel usable. Goes into the report's data-quality section."""

    sessions: int
    filled: dict[str, int] = field(default_factory=dict)  # ticker -> forward-filled sessions
    missing: dict[str, int] = field(default_factory=dict)  # ticker -> sessions still missing
    dropped: dict[str, str] = field(default_factory=dict)  # ticker -> why it was dropped

    @property
    def clean(self) -> bool:
        return not self.missing and not self.dropped


def align(
    prices: pd.DataFrame,
    exchange: str = DEFAULT_EXCHANGE,
    max_staleness: int = MAX_STALENESS,
    min_observations: int = 252,
) -> tuple[pd.DataFrame, AlignmentReport]:
    """Reindex a ticker-by-date price frame onto the exchange calendar.

    Returns the aligned frame and a report. Tickers with fewer than
    ``min_observations`` real prices are dropped: too little history to estimate
    a covariance or a tail from, and a short series quietly distorts both.
    """
    if prices.empty:
        raise ValueError("no prices to align")

    index = sessions(prices.index.min(), prices.index.max(), exchange)
    out = prices.reindex(index)

    filled, missing, dropped = {}, {}, {}
    for ticker in list(out.columns):
        column = out[ticker]
        observed = column.notna()
        if int(observed.sum()) < min_observations:
            dropped[ticker] = f"only {int(observed.sum())} sessions of history (need {min_observations})"
            out = out.drop(columns=ticker)
            continue

        # Only fill between the first and last real observation: never invent a
        # price before a listing or after a delisting.
        first, last = observed.idxmax(), observed[::-1].idxmax()
        window = column.loc[first:last]
        patched = window.ffill(limit=max_staleness)

        n_filled = int(patched.notna().sum() - window.notna().sum())
        if n_filled:
            filled[ticker] = n_filled
        n_missing = int(patched.isna().sum())
        if n_missing:
            missing[ticker] = n_missing
        out.loc[first:last, ticker] = patched

    return out, AlignmentReport(sessions=len(index), filled=filled, missing=missing, dropped=dropped)

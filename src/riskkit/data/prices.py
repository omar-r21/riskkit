"""Total-return price history, from a swappable vendor behind a local cache.

Risk figures are computed on **total-return** prices (dividends reinvested, splits
adjusted). Using raw closes would show every dividend as a price drop, which
inflates measured volatility and puts phantom losses in the tail.

Vendors are wrapped in the ``PriceSource`` protocol so the rest of the package
never sees one. Yahoo (via ``yfinance``) needs no key and is the default;
``TiingoSource`` is there for when a key is available. Vendor terms are
personal-use, so prices are cached locally and never committed or republished:
only derived statistics leave this machine.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import pandas as pd

DEFAULT_START = "2005-01-03"
CACHE_DIR = Path(os.environ.get("RISKKIT_CACHE", Path.home() / ".cache" / "riskkit"))


class PriceSource(Protocol):
    """Returns a date-by-ticker frame of dividend- and split-adjusted closes."""

    name: str

    def fetch(self, tickers: Sequence[str], start: str, end: str | None) -> pd.DataFrame: ...


@dataclass
class YahooSource:
    """Yahoo Finance via yfinance. No API key; unofficial, so it can break without notice."""

    name: str = "yahoo"

    def fetch(self, tickers: Sequence[str], start: str, end: str | None = None) -> pd.DataFrame:
        import yfinance as yf  # imported lazily: only this source needs it

        raw = yf.download(
            list(tickers),
            start=start,
            end=end,
            auto_adjust=True,  # 'Close' becomes the total-return series
            progress=False,
            group_by="column",
            threads=True,
        )
        if raw.empty:
            raise RuntimeError(f"no data returned for {list(tickers)}")
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
        if len(tickers) == 1:
            close = close.rename(columns={close.columns[0]: tickers[0]})
        close.index = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
        return close.sort_index()


@dataclass
class TiingoSource:
    """Tiingo's adjusted close. Needs TIINGO_API_KEY; longer history and an official API."""

    api_key: str | None = None
    name: str = "tiingo"

    def fetch(self, tickers: Sequence[str], start: str, end: str | None = None) -> pd.DataFrame:
        import requests  # imported lazily

        key = self.api_key or os.environ.get("TIINGO_API_KEY")
        if not key:
            raise RuntimeError("TIINGO_API_KEY is not set")
        frames = []
        for ticker in tickers:
            response = requests.get(
                f"https://api.tiingo.com/tiingo/daily/{ticker}/prices",
                params={"startDate": start, "endDate": end, "format": "json"},
                headers={"Authorization": f"Token {key}"},
                timeout=30,
            )
            response.raise_for_status()
            frame = pd.DataFrame(response.json())
            if frame.empty:
                continue
            series = pd.Series(
                frame["adjClose"].to_numpy(),
                index=pd.DatetimeIndex(frame["date"]).tz_localize(None).normalize(),
                name=ticker,
            )
            frames.append(series)
        if not frames:
            raise RuntimeError(f"no data returned for {list(tickers)}")
        return pd.concat(frames, axis=1).sort_index()


def _cache_path(source: str, ticker: str, cache_dir: Path) -> Path:
    return cache_dir / source / f"{ticker.upper()}.parquet"


def load_prices(
    tickers: Sequence[str],
    start: str = DEFAULT_START,
    end: str | None = None,
    source: PriceSource | None = None,
    cache_dir: Path = CACHE_DIR,
    refresh: bool = False,
) -> pd.DataFrame:
    """Total-return closes for ``tickers``, cached per ticker as parquet.

    A cached ticker is re-fetched only when it is missing, when ``refresh`` is
    set, or when the cache starts later than ``start``. Cached files are local
    scratch: gitignored, and safe to delete.
    """
    source = source or YahooSource()
    tickers = [t.upper() for t in dict.fromkeys(tickers)]  # de-duplicate, keep order
    cache_dir = Path(cache_dir)

    cached: dict[str, pd.Series] = {}
    to_fetch: list[str] = []
    for ticker in tickers:
        path = _cache_path(source.name, ticker, cache_dir)
        if refresh or not path.exists():
            to_fetch.append(ticker)
            continue
        series = pd.read_parquet(path)[ticker]
        if series.index.min() > pd.Timestamp(start):
            to_fetch.append(ticker)  # cache doesn't reach far enough back
        else:
            cached[ticker] = series

    if to_fetch:
        fetched = source.fetch(to_fetch, start=start, end=end)
        for ticker in to_fetch:
            if ticker not in fetched.columns:
                raise RuntimeError(f"{source.name} returned no series for {ticker}")
            series = fetched[ticker].dropna()
            path = _cache_path(source.name, ticker, cache_dir)
            path.parent.mkdir(parents=True, exist_ok=True)
            series.to_frame(ticker).to_parquet(path)
            cached[ticker] = series

    prices = pd.concat([cached[t] for t in tickers], axis=1)
    prices.columns = tickers
    prices = prices.loc[prices.index >= pd.Timestamp(start)]
    return prices if end is None else prices.loc[prices.index <= pd.Timestamp(end)]

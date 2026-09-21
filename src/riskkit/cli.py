"""Command line entry point: one command from a weights file to a finished report.

    riskkit report --weights portfolios/demo.yaml --out docs/index.html

Everything the report needs is derived here: prices, returns, the backtest, the
attribution. Prices come from the local cache when they are already there, so a
rerun takes seconds.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .attribution import component_es, concentration
from .backtest import run_backtest, summary_table
from .data.calendar import align
from .data.prices import DEFAULT_START, load_prices
from .portfolio import Portfolio
from .report import ReportInputs, write_report


def _build(args: argparse.Namespace) -> tuple[Portfolio, pd.Series, list, object, object]:
    portfolio = Portfolio.from_yaml(args.weights)
    prices = load_prices(portfolio.tickers, start=args.start, refresh=args.refresh)
    aligned, report = align(prices)
    if report.dropped:
        print(f"dropped: {report.dropped}", file=sys.stderr)
    if report.missing:
        print(f"still missing after filling: {report.missing}", file=sys.stderr)

    asset_returns = aligned.pct_change().iloc[1:].dropna()
    returns = portfolio.returns(asset_returns)
    results = run_backtest(returns, alphas=(0.05, args.alpha))
    attribution = component_es(asset_returns, portfolio.weights, alpha=args.alpha)
    conc = concentration(asset_returns, portfolio.weights, alpha=args.alpha)
    return portfolio, returns, results, attribution, conc


def cmd_report(args: argparse.Namespace) -> int:
    portfolio, returns, results, attribution, conc = _build(args)
    path = write_report(
        ReportInputs(
            portfolio_name=portfolio.name,
            returns=returns,
            results=results,
            attribution=attribution,
            concentration=conc,
            headline_model=args.model,
            note=portfolio.description,
        ),
        args.out,
    )
    print(f"wrote {path}")
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    """The same numbers as the report, printed instead of rendered."""
    _, returns, results, attribution, conc = _build(args)
    pd.set_option("display.width", 200)
    print(f"{len(returns):,} days: {returns.index.min():%Y-%m-%d} to {returns.index.max():%Y-%m-%d}\n")
    print(summary_table(results).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nrisk attribution:")
    print(attribution.table().to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\nconcentration: {({k: round(v, 3) for k, v in conc.as_dict().items()})}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="riskkit", description=__doc__.splitlines()[0])
    parser.add_argument("--weights", default="portfolios/demo.yaml", help="portfolio YAML")
    parser.add_argument("--start", default=DEFAULT_START, help="first date of price history")
    parser.add_argument("--alpha", type=float, default=0.01, help="tail probability (0.01 = 99%%)")
    parser.add_argument("--refresh", action="store_true", help="re-fetch prices instead of using the cache")

    sub = parser.add_subparsers(dest="command", required=True)
    report = sub.add_parser("report", help="write the HTML report")
    report.add_argument("--out", default="docs/index.html")
    report.add_argument("--model", default="fhs", help="model featured in the hero chart")
    report.set_defaults(func=cmd_report)

    backtest = sub.add_parser("backtest", help="print the results table")
    backtest.set_defaults(func=cmd_backtest)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

# riskkit

[![tests](https://github.com/omar-r21/riskkit/actions/workflows/ci.yml/badge.svg)](https://github.com/omar-r21/riskkit/actions/workflows/ci.yml)

Does your value-at-risk model actually work? This runs four of them side by side over twenty years of
daily data and scores them with the tests that answer that question.

**[See the report it produces →](https://omar-r21.github.io/riskkit/)**

```bash
pip install -e .
riskkit report --weights portfolios/demo.yaml --out docs/index.html
```

One command, one self-contained HTML file. Charts are inlined SVG, so there's nothing to serve.

## The result

A ten-ETF portfolio, 4,711 out-of-sample days from January 2005. At 99%, a model should be wrong about
47 times:

| model | breaches | expected | rate | Kupiec p | independence p | loss ÷ ES | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| Filtered historical simulation | 54 | 47 | 1.15% | 0.32 | 0.000 | 0.99 | breaches cluster |
| Historical simulation | 67 | 47 | 1.42% | 0.006 | 0.000 | 1.11 | rate, clustering, ES |
| Parametric normal | 97 | 47 | 2.06% | 0.000 | 0.000 | 1.35 | rate, clustering, ES |
| EWMA normal | 102 | 47 | 2.17% | 0.000 | 0.008 | 1.17 | rate, clustering, ES |

The parametric normal model was wrong **twice as often as it advertised**, and on the days it was wrong
the losses averaged 1.35x its own expected-shortfall forecast — so it understated the tail it claimed to
measure by a third. Filtered historical simulation breached 54 times against 47 expected, close enough
to be chance, and its ES came in at 0.99, essentially calibrated.

Every model failed the independence test, including the winner. Breaches still cluster in crises. None
of these is fully specified on real data, and a report that claimed otherwise would be hiding something.

The static models fail where it matters. Through the 2008 crisis window, historical simulation took 19
breaches and the normal model 23, against 4 for FHS.

## What it does

**Four models.** Historical simulation (the empirical quantile of a 500-day window), parametric normal,
EWMA-normal (RiskMetrics volatility, λ = 0.94), and filtered historical simulation. Every forecast for
day *t* uses only data through *t−1*; a test tampers with one day's return and asserts the forecast made
for that same day doesn't move.

**The tests that decide it.** Kupiec's proportion-of-failures test on the breach count, Christoffersen's
test on whether breaches cluster, the two combined as conditional coverage, and the Basel traffic light
computed per rolling 250-day window rather than once over the whole sample.

**Risk attribution.** Component expected shortfall, averaged over the portfolio's own worst days, so the
per-position numbers add up to the total exactly. The demo portfolio holds 25% SPY, which carries 39% of
the risk, and 25% in Treasuries, which contribute **negative** risk — on the portfolio's worst days they
were up. A weight column can't show you that.

**Covariance** with Ledoit-Wolf shrinkage for the parametric decomposition, and concentration measures:
Herfindahl, effective positions, diversification ratio.

## Design notes

**Historical VaR is deliberately not decomposed by position.** Its Euler derivative is the loss on the
single day sitting at the quantile, so the answer swings on one observation. ES averages the whole tail,
which is what makes it decomposable in practice. There's a test checking the component ES against
finite-difference marginals, not just that the parts sum to the total.

**The Basel traffic light is per window, not per sample.** It's defined on 250 days. A model can sit in
green for years and spend a crisis in red, and only the per-window version shows that.

**This is a constant-weight hypothetical portfolio, not a track record.** Today's weights applied to
twenty years of history would be look-ahead bias if it were presented as performance. It's here to give
the risk models a realistic multi-asset return series to be tested on. The ETFs were picked for
asset-class coverage — equity, duration, credit, real assets — not for how they performed.

**Prices are never committed.** Vendor terms are personal-use, so the cache is local and gitignored, and
every test runs on synthetic data. CI needs no network and no API key.

## Layout

```
src/riskkit/
  data/       price loading behind a vendor interface, NYSE calendar alignment
  returns.py  simple vs log returns, annualisation, EWMA volatility
  var.py      the four models
  backtest.py Kupiec, Christoffersen, traffic light, the scoring engine
  attribution.py  component ES, parametric Euler VaR, shrinkage, concentration
  report.py   the HTML report
  cli.py      riskkit report / riskkit backtest
```

```bash
riskkit backtest                      # same numbers, printed
riskkit report --alpha 0.05           # 95% instead of 99%
riskkit report --weights mine.yaml    # any portfolio
pytest                                # 71 tests, no network
```

## Where it runs

The published report uses the demo portfolio, which exists so the models have a realistic multi-asset
return series to be tested on without anyone's holdings being involved.

I also run it nightly against my own brokerage account, from a private app that feeds it live positions
and displays the output — forecast against realised losses, which positions drive the risk, how the book
would have fared through 2008 and 2020. That app is private because it reads a real account; the
analytics are here because they're general and contain no personal data.
[Write-up of that side](https://github.com/omar-r21/portfolio-intelligence-case-study), including what it
replaced: a language model producing buy/sell calls with invented confidence percentages.

## Still to come

Fama-French factor regression with HAC standard errors, historical stress replays (2008, COVID, 2022),
and the Acerbi-Székely test for expected shortfall — ES is currently checked with a loss-to-forecast
ratio, which is a diagnostic rather than a formal test.

## References

- Kupiec (1995), *Techniques for verifying the accuracy of risk measurement models.*
- Christoffersen (1998), *Evaluating interval forecasts.* International Economic Review 39(4).
- Barone-Adesi, Giannopoulos & Vosper (1999), filtered historical simulation.
- Ledoit & Wolf (2003), *Improved estimation of the covariance matrix of stock returns.*
- Basel Committee (1996), *Supervisory framework for the use of backtesting.*

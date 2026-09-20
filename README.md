# riskkit

Value-at-risk and expected-shortfall **model validation**, risk attribution and stress testing, in Python.

Status: in development. Phase 1 (data layer, returns, volatility) is complete.

The headline result will be a rolling out-of-sample comparison of four VaR/ES methods over ~4,000 days,
scored with Kupiec, Christoffersen and Acerbi-Székely tests: which models actually hold up, and which
only look fine until volatility clusters.

Prices are cached locally and never redistributed; tests run offline on synthetic data.

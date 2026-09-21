"""Where a portfolio's risk actually comes from.

A weight says how much money sits in a position. It does not say how much risk
the position contributes: that depends on its own volatility and on how it moves
with everything else. A 5% position in something volatile and correlated can
carry more risk than a 20% position in something that offsets the rest.

Both decompositions here are Euler allocations. Risk measures like volatility,
VaR and ES are homogeneous of degree one in the weights - double every position
and the risk doubles - so Euler's theorem says the position-level derivatives
sum exactly to the total. That is what makes the parts add up, and it is why
"contribution" is a well-defined idea rather than an accounting convention.

Two measures are decomposed, deliberately not three:

* **Parametric VaR**, analytically, from a covariance matrix.
* **Expected shortfall**, empirically, by averaging each position's loss over the
  days the portfolio was in its own tail.

Historical *VaR* is left out. Its Euler derivative is the position's loss on the
single day that sits at the quantile, so the answer swings on one day's data.
ES averages the whole tail, which is exactly the property that makes it
decomposable in practice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass(frozen=True)
class Attribution:
    """Per-position contributions to one risk measure, plus the total they sum to."""

    measure: str                 # "var" or "es"
    total: float                 # the portfolio-level figure
    contribution: pd.Series      # per position, sums to total
    weight: pd.Series

    @property
    def share(self) -> pd.Series:
        """Fraction of total risk carried by each position."""
        return self.contribution / self.total

    def table(self) -> pd.DataFrame:
        """Weight against risk share: the comparison the whole module exists for."""
        out = pd.DataFrame(
            {
                "weight": self.weight,
                "contribution": self.contribution,
                "risk_share": self.share,
            }
        )
        out["risk_over_weight"] = out["risk_share"] / out["weight"]
        return out.sort_values("contribution", ascending=False)


def ledoit_wolf_covariance(returns: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Sample covariance shrunk toward a constant-correlation target (Ledoit-Wolf, 2003).

    With a few hundred observations and ten or more assets, a sample covariance
    is noisy in a specific way: the extreme eigenvalues are biased outward, so
    the combinations that look lowest-risk are usually the ones whose risk was
    most underestimated. Shrinking pulls the estimate toward a structured target
    and is close to free at this sample size.

    The target keeps each asset's own variance and replaces every correlation
    with the average correlation. The intensity is chosen by the sample analogue
    of the optimal shrinkage constant, clipped to [0, 1].

    Returns the shrunk covariance and the intensity actually used, because an
    intensity near 1 says the sample estimate had little information in it.
    """
    x = returns.to_numpy(dtype=float)
    x = x - x.mean(axis=0)
    n, p = x.shape
    if n < 2 or p < 2:
        raise ValueError("need at least two assets and two observations")

    sample = x.T @ x / n
    variances = np.diag(sample)
    std = np.sqrt(variances)
    outer_std = np.outer(std, std)

    corr = sample / outer_std
    mean_corr = (corr.sum() - p) / (p * (p - 1))  # average off-diagonal correlation
    target = mean_corr * outer_std
    np.fill_diagonal(target, variances)

    # pi: variance of the sample covariance entries; rho: its covariance with the target.
    x2 = x**2
    pi_matrix = x2.T @ x2 / n - sample**2
    pi = pi_matrix.sum()

    theta_ii = (x2 * x).T @ x / n - variances[:, None] * sample
    theta_jj = x.T @ (x * x2) / n - variances[None, :] * sample
    ratio = np.outer(std, 1.0 / std)
    rho = np.diag(pi_matrix).sum() + mean_corr * (ratio * theta_ii + ratio.T * theta_jj).sum() / 2.0
    np.fill_diagonal(ratio, 0.0)  # diagonal handled by the first term

    gamma = float(((target - sample) ** 2).sum())
    intensity = 0.0 if gamma <= 0 else float(np.clip((pi - rho) / gamma / n, 0.0, 1.0))
    shrunk = intensity * target + (1.0 - intensity) * sample
    return pd.DataFrame(shrunk, index=returns.columns, columns=returns.columns), intensity


def parametric_var_attribution(
    returns: pd.DataFrame,
    weights: pd.Series,
    alpha: float = 0.01,
    shrink: bool = True,
) -> Attribution:
    """Euler decomposition of parametric VaR.

        VaR = z * sqrt(w' S w)
        VaR_i = z * w_i (S w)_i / sqrt(w' S w)

    Each term is the position's weight times its covariance with the portfolio,
    so a position is charged for how it moves *with the rest*, not for its own
    volatility alone. The terms sum to the portfolio VaR by construction.
    """
    weights = weights.reindex(returns.columns).dropna()
    cov = (ledoit_wolf_covariance(returns[weights.index])[0] if shrink else returns[weights.index].cov())
    w = weights.to_numpy(dtype=float)
    sigma_p = float(np.sqrt(w @ cov.to_numpy() @ w))
    z = norm.ppf(1.0 - alpha)

    marginal = cov.to_numpy() @ w / sigma_p          # d sigma_p / d w_i
    contribution = pd.Series(z * w * marginal, index=weights.index)
    return Attribution("var", z * sigma_p, contribution, weights)


def component_es(
    returns: pd.DataFrame,
    weights: pd.Series,
    alpha: float = 0.01,
) -> Attribution:
    """Empirical component expected shortfall.

        ES   = mean over tail days of the portfolio loss
        ES_i = mean over those same days of position i's loss

    The tail is the worst ceil(alpha * n) portfolio days. Because the days are
    chosen by the *portfolio's* loss, the position averages add up to the
    portfolio ES exactly - no allocation rule is needed. It also answers a
    question a weight cannot: when this portfolio has a bad day, who caused it?
    """
    weights = weights.reindex(returns.columns).dropna()
    asset_returns = returns[weights.index]
    portfolio_losses = -(asset_returns @ weights)

    n_tail = max(1, int(np.ceil(alpha * len(portfolio_losses))))
    tail_days = portfolio_losses.nlargest(n_tail).index

    contribution = -(asset_returns.loc[tail_days] * weights).mean()
    return Attribution("es", float(portfolio_losses.loc[tail_days].mean()), contribution, weights)


@dataclass(frozen=True)
class Concentration:
    """How spread out the portfolio is, by weight and after accounting for risk."""

    herfindahl: float
    effective_positions: float
    diversification_ratio: float
    top_weight_share: float       # weight of the largest position
    top_risk_share: float         # risk share of the largest risk contributor

    def as_dict(self) -> dict[str, float]:
        return {
            "herfindahl": self.herfindahl,
            "effective_positions": self.effective_positions,
            "diversification_ratio": self.diversification_ratio,
            "top_weight_share": self.top_weight_share,
            "top_risk_share": self.top_risk_share,
        }


def concentration(returns: pd.DataFrame, weights: pd.Series, alpha: float = 0.01) -> Concentration:
    """Concentration by weight and by risk.

    * **Herfindahl** = sum of squared weights; **effective positions** = 1/HHI,
      i.e. the number of equally weighted holdings that would be this
      concentrated. Ten positions can have an effective count of three.
    * **Diversification ratio** = (sum of w_i sigma_i) / sigma_p: the weighted
      average standalone volatility divided by the portfolio's. It is 1 when
      everything moves together and rises as correlations fall.
    * The two top shares are deliberately paired: the largest position and the
      largest risk contributor are often not the same holding.
    """
    weights = weights.reindex(returns.columns).dropna()
    w = weights.to_numpy(dtype=float)
    cov = returns[weights.index].cov().to_numpy()

    hhi = float((w**2).sum())
    sigma_p = float(np.sqrt(w @ cov @ w))
    weighted_standalone = float((w * np.sqrt(np.diag(cov))).sum())
    risk_share = component_es(returns, weights, alpha).share

    return Concentration(
        herfindahl=hhi,
        effective_positions=1.0 / hhi,
        diversification_ratio=weighted_standalone / sigma_p,
        top_weight_share=float(weights.max()),
        top_risk_share=float(risk_share.max()),
    )

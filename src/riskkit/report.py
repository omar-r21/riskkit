"""Build a self-contained HTML report from a backtest.

One file, no server, no assets on disk: charts are rendered to SVG and inlined,
so the report can be emailed, committed, or served from GitHub Pages unchanged.

The layout puts the result first. The hero chart shows every forecast against
what actually happened, with breaches marked, because a reader should be able to
see a model failing before reading a p-value. The table underneath says whether
the statistics agree.
"""

from __future__ import annotations

import base64
import html
import io
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .attribution import Attribution, Concentration
from .backtest import BacktestResult

PALETTE = {
    "ink": "#14161A",
    "muted": "#6B7280",
    "line": "#E4E6EB",
    "loss": "#C62828",
    "accent": "#1F5FD8",
    "good": "#0B8A3D",
    "warn": "#B26A00",
}
MODEL_LABELS = {
    "historical": "Historical simulation",
    "normal": "Parametric normal",
    "ewma_normal": "EWMA normal",
    "fhs": "Filtered historical simulation",
}


def _svg(fig) -> str:
    """Render a matplotlib figure to an inline <img> with an SVG data URI."""
    buffer = io.StringIO()
    fig.savefig(buffer, format="svg", bbox_inches="tight")
    encoded = base64.b64encode(buffer.getvalue().encode("utf-8")).decode("ascii")
    return f'<img src="data:image/svg+xml;base64,{encoded}" alt="" style="width:100%;height:auto">'


def hero_chart(result: BacktestResult, title: str = "") -> str:
    """Losses against the VaR forecast, with breaches marked, plus rolling breach counts.

    The lower panel counts breaches in a trailing 250-day window against the
    Basel zones: it shows *when* a model failed, which an aggregate count hides.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = result.forecasts
    breaches = frame[frame["breach"]]

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(11, 6), height_ratios=[3, 1], sharex=True, gridspec_kw={"hspace": 0.12}
    )

    top.fill_between(frame.index, 0, frame["loss"].clip(lower=0), color="#C8CCD4", lw=0, label="daily loss")
    top.plot(frame.index, frame["var"], color=PALETTE["accent"], lw=1.1, label=f"{1 - result.alpha:.0%} VaR forecast")
    top.plot(frame.index, frame["es"], color=PALETTE["accent"], lw=0.8, ls=":", label="ES forecast")
    top.scatter(breaches.index, breaches["loss"], s=14, color=PALETTE["loss"], zorder=3, label="breach")
    top.set_ylabel("daily loss")
    top.set_ylim(0, float(frame["loss"].max()) * 1.05)
    top.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    top.legend(frameon=False, fontsize=9, ncol=4, loc="upper left")
    top.set_title(title or f"{MODEL_LABELS.get(result.model, result.model)} — {1 - result.alpha:.0%} VaR", fontsize=11)

    rolling = frame["breach"].rolling(250).sum()
    bottom.axhspan(0, 4, color=PALETTE["good"], alpha=0.10)
    bottom.axhspan(4, 9, color=PALETTE["warn"], alpha=0.10)
    bottom.axhspan(9, max(12, float(rolling.max() or 12)), color=PALETTE["loss"], alpha=0.10)
    bottom.plot(rolling.index, rolling, color=PALETTE["ink"], lw=1.0)
    bottom.set_ylabel("breaches\nper 250d", fontsize=9)
    bottom.set_ylim(0, max(12, float(rolling.max() or 12)))

    for axis in (top, bottom):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color=PALETTE["line"], lw=0.6)
        axis.tick_params(labelsize=9)

    out = _svg(fig)
    plt.close(fig)
    return out


def attribution_chart(attribution: Attribution) -> str:
    """Weight against risk share, side by side, sorted by risk."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = attribution.table()
    y = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(table) + 1.2))

    ax.barh(y + 0.19, table["weight"], height=0.36, color=PALETTE["line"], label="weight")
    ax.barh(y - 0.19, table["risk_share"], height=0.36, color=PALETTE["accent"], label="share of risk")
    ax.set_yticks(y, table.index)
    ax.invert_yaxis()
    ax.axvline(0, color=PALETTE["ink"], lw=0.8)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color=PALETTE["line"], lw=0.6)
    ax.tick_params(labelsize=9)

    out = _svg(fig)
    plt.close(fig)
    return out


def _verdict(result: BacktestResult) -> tuple[str, str]:
    """A plain-language call on one model, and the colour to print it in."""
    failures = []
    if result.kupiec.rejects(0.01):
        failures.append("wrong breach rate")
    if result.independence.rejects(0.01):
        failures.append("breaches cluster")
    if result.mean_shortfall_ratio > 1.10:
        failures.append("ES understated")
    if not failures:
        return "passes", PALETTE["good"]
    return "; ".join(failures), PALETTE["loss"]


def _results_table(results: list[BacktestResult]) -> str:
    rows = []
    for r in sorted(results, key=lambda x: x.model):
        verdict, colour = _verdict(r)
        shares = r.zone_shares()
        rows.append(
            f"""<tr>
  <td style="font-weight:600">{html.escape(MODEL_LABELS.get(r.model, r.model))}</td>
  <td class="n">{r.breaches}</td>
  <td class="n">{r.expected_breaches:.0f}</td>
  <td class="n">{r.breach_rate:.2%}</td>
  <td class="n">{r.kupiec.p_value:.3f}</td>
  <td class="n">{r.independence.p_value:.3f}</td>
  <td class="n">{r.mean_shortfall_ratio:.2f}</td>
  <td class="n">{shares.get("red", float("nan")):.0%}</td>
  <td style="color:{colour}">{html.escape(verdict)}</td>
</tr>"""
        )
    return f"""<table>
<thead><tr>
  <th>model</th><th class="n">breaches</th><th class="n">expected</th><th class="n">rate</th>
  <th class="n">Kupiec p</th><th class="n">independence p</th><th class="n">loss / ES</th>
  <th class="n">windows red</th><th>verdict</th>
</tr></thead>
<tbody>{"".join(rows)}</tbody></table>"""


def _attribution_table(attribution: Attribution) -> str:
    rows = "".join(
        f"""<tr><td style="font-weight:600">{html.escape(str(name))}</td>
  <td class="n">{row.weight:.1%}</td><td class="n">{row.risk_share:.1%}</td>
  <td class="n">{row.risk_over_weight:.2f}x</td></tr>"""
        for name, row in attribution.table().iterrows()
    )
    return f"""<table>
<thead><tr><th>position</th><th class="n">weight</th><th class="n">share of risk</th>
<th class="n">risk / weight</th></tr></thead><tbody>{rows}</tbody></table>"""


@dataclass(frozen=True)
class ReportInputs:
    portfolio_name: str
    returns: pd.Series
    results: list[BacktestResult]
    attribution: Attribution
    concentration: Concentration
    headline_model: str = "fhs"
    note: str = ""


def build_report(inputs: ReportInputs) -> str:
    """Render the whole report as one HTML string."""
    at_99 = [r for r in inputs.results if r.alpha == 0.01]
    hero = next(r for r in at_99 if r.model == inputs.headline_model)
    span = f"{inputs.returns.index.min():%b %Y} to {inputs.returns.index.max():%b %Y}"
    built = datetime.now(timezone.utc).strftime("%d %b %Y")

    # Say which models survive each check separately. A blanket "none passed"
    # would be true and useless: the models fail in different ways, and the
    # difference between them is the result.
    right_rate = [r for r in at_99 if not r.kupiec.rejects(0.01)]
    honest_es = [r for r in at_99 if r.mean_shortfall_ratio <= 1.10]
    both = [MODEL_LABELS.get(r.model, r.model) for r in at_99 if r in right_rate and r in honest_es]
    clustered = [r for r in at_99 if r.independence.rejects(0.01)]

    if both:
        lead = (
            f"{'only ' if len(both) == 1 else ''}{', '.join(both)} breached at close to the "
            f"advertised rate and forecast a tail that held up"
        )
    else:
        lead = "No model breached at close to the advertised rate with a tail forecast that held up"
    summary = (
        f"Over {hero.observations:,} out-of-sample days ({span}), {lead}. "
        + (
            f"All {len(clustered)} models still show breach clustering, so none is fully specified."
            if len(clustered) == len(at_99)
            else f"{len(clustered)} of {len(at_99)} models also show breach clustering."
        )
    )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>riskkit — {html.escape(inputs.portfolio_name)}</title>
<style>
 body {{ margin:0; background:#F7F8FA; color:{PALETTE['ink']};
   font:15px/1.6 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif; }}
 main {{ max-width:960px; margin:0 auto; padding:32px 20px 64px; }}
 h1 {{ font-size:24px; margin:0 0 4px; }}
 h2 {{ font-size:13px; text-transform:uppercase; letter-spacing:.9px; color:{PALETTE['muted']};
   margin:40px 0 12px; border-bottom:1px solid {PALETTE['line']}; padding-bottom:8px; }}
 .sub {{ color:{PALETTE['muted']}; font-size:13px; margin-bottom:24px; }}
 .card {{ background:#fff; border:1px solid {PALETTE['line']}; border-radius:12px; padding:20px; }}
 .lede {{ font-size:17px; line-height:1.5; margin:0 0 20px; }}
 table {{ width:100%; border-collapse:collapse; font-size:14px; }}
 th {{ text-align:left; font-size:11px; text-transform:uppercase; letter-spacing:.6px;
   color:{PALETTE['muted']}; font-weight:600; padding:0 10px 8px 0; }}
 td {{ padding:9px 10px 9px 0; border-top:1px solid {PALETTE['line']}; }}
 .n {{ text-align:right; font-variant-numeric:tabular-nums;
   font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }}
 .stats {{ display:flex; flex-wrap:wrap; gap:28px; margin:16px 0 0; }}
 .stat b {{ display:block; font-size:22px; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }}
 .stat span {{ font-size:11px; text-transform:uppercase; letter-spacing:.6px; color:{PALETTE['muted']}; }}
 footer {{ color:{PALETTE['muted']}; font-size:12px; margin-top:40px; }}
</style></head>
<body><main>

<h1>{html.escape(inputs.portfolio_name)}</h1>
<div class="sub">VaR and expected-shortfall model validation · built {built}</div>

<div class="card">
  <p class="lede">{html.escape(summary)}</p>
  {hero_chart(hero, f"{MODEL_LABELS.get(hero.model, hero.model)} — 99% VaR, breaches in red")}
</div>

<h2>Do the models hold up?</h2>
<div class="card">{_results_table(at_99)}
<p class="sub" style="margin:16px 0 0">A model fails if its breach count is implausible (Kupiec), its
breaches cluster (independence), or realised losses on breach days exceed its own ES forecast by more
than 10%. Expected breaches at 99% over {hero.observations:,} days: {hero.expected_breaches:.0f}.</p></div>

<h2>Where the risk actually sits</h2>
<div class="card">
  {attribution_chart(inputs.attribution)}
  {_attribution_table(inputs.attribution)}
  <div class="stats">
    <div class="stat"><b>{inputs.concentration.effective_positions:.1f}</b>
      <span>effective positions</span></div>
    <div class="stat"><b>{inputs.concentration.diversification_ratio:.2f}</b>
      <span>diversification ratio</span></div>
    <div class="stat"><b>{inputs.concentration.top_risk_share:.0%}</b>
      <span>largest risk share</span></div>
    <div class="stat"><b>{inputs.concentration.top_weight_share:.0%}</b>
      <span>largest weight</span></div>
  </div>
</div>

{f'<h2>Notes</h2><div class="card">{inputs.note}</div>' if inputs.note else ''}

<footer>Generated by <a href="https://github.com/omar-r21/riskkit">riskkit</a>. Hypothetical
portfolio at constant weights; not a track record and not investment advice.</footer>
</main></body></html>"""


def write_report(inputs: ReportInputs, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_report(inputs), encoding="utf-8")
    return path

"""
backtest.py
============

Implements a standard **quintile-sort factor backtest**, the workhorse
methodology of academic and practitioner factor research.

The basic idea:
    1. At each rebalance date (e.g., the start of each month), rank all
       stocks in the universe by their factor value.
    2. Split the ranked universe into 5 equal-sized groups ("quintiles"):
       Q1 = bottom 20% (lowest factor values), Q5 = top 20% (highest factor
       values).
    3. Hold each quintile as an equal-weighted portfolio until the next
       rebalance date, then re-rank and re-form the quintiles.
    4. Compare the returns of Q5 vs Q1 (and everything in between). If the
       factor has genuine predictive power, there should be a clear,
       monotonic (or close to it) pattern in average returns from Q1 to Q5.

This module also computes two key statistics used to judge a factor's
predictive power *independent* of the quintile portfolios:

    - **Information Coefficient (IC)**: the cross-sectional Spearman rank
      correlation between a factor's values at time t and the stocks'
      forward returns over the period [t, t+1]. Spearman rank correlation
      measures how well two rankings agree, ranging from -1 (perfectly
      opposite rankings) to +1 (perfectly matching rankings), independent of
      whether the relationship is linear. An IC of, say, 0.05 means the
      factor has *some* (modest) ability to rank which stocks will do better
      than others next period.

    - **Information Ratio (IR)**: the mean IC divided by the standard
      deviation of IC across all rebalance periods. This measures how
      *consistent* the factor's predictive power is over time — a factor
      with mean IC of 0.05 that is almost always positive (low std) is much
      more useful than one that averages 0.05 but swings between -0.20 and
      +0.30 from month to month.

Standard portfolio statistics are also computed for the long-short (Q5 - Q1)
spread portfolio:

    - **Sharpe ratio**: (mean excess return) / (standard deviation of
      returns), annualized. Measures risk-adjusted return — how much return
      you got per unit of volatility ("risk") taken on.

    - **Max drawdown**: the largest percentage decline from a cumulative-
      return peak to a subsequent trough. Measures the worst "pain" an
      investor following the strategy would have experienced.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

try:
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend, safe for headless runs
    import matplotlib.pyplot as plt

    _HAS_MATPLOTLIB = True
except ImportError:  # pragma: no cover - matplotlib is in requirements but be defensive
    _HAS_MATPLOTLIB = False


@dataclass
class BacktestResult:
    """Container for the outputs of `run_quintile_backtest`.

    Attributes:
        quintile_returns: DataFrame of period returns for each quintile
            portfolio (columns "Q1".."Q5") plus the long-short spread
            (column "Q5_minus_Q1"), indexed by rebalance date. Each value is
            the *forward* return earned by holding that quintile's stocks
            from this rebalance date to the next one.
        ic_series: Series of Information Coefficient values (Spearman rank
            correlation between factor value and forward return), one per
            rebalance date.
        ic_mean: Mean of `ic_series`. Average cross-sectional rank
            correlation between the factor and forward returns.
        ic_std: Standard deviation of `ic_series`.
        information_ratio: `ic_mean / ic_std`. NaN if `ic_std` is zero.
        sharpe_ratio: Annualized Sharpe ratio of the Q5-Q1 spread portfolio.
        max_drawdown: Maximum drawdown (as a negative decimal, e.g., -0.35
            for a 35% drawdown) of the Q5-Q1 spread portfolio's cumulative
            return curve.
        cumulative_returns: DataFrame of cumulative (compounded) returns for
            each quintile and the spread, indexed by rebalance date, useful
            for plotting.
        annualized_returns: Series of annualized returns for each quintile
            and the spread.
        n_periods: Number of rebalance periods used in the backtest.
        excluded_fraction: Average fraction of the universe excluded at each
            rebalance date due to missing factor or return data.
    """

    quintile_returns: pd.DataFrame
    ic_series: pd.Series
    ic_mean: float
    ic_std: float
    information_ratio: float
    sharpe_ratio: float
    max_drawdown: float
    cumulative_returns: pd.DataFrame
    annualized_returns: pd.Series
    n_periods: int
    excluded_fraction: float = field(default=0.0)


def run_quintile_backtest(
    factor_values: pd.DataFrame,
    forward_returns: pd.DataFrame,
    n_quintiles: int = 5,
    periods_per_year: int = 12,
    risk_free_rate: float = 0.0,
) -> BacktestResult:
    """Run a quintile-sort backtest of a factor against forward returns.

    At each date in `factor_values.index`, this function:
        1. Drops any tickers with missing factor value or missing forward
           return for that date.
        2. Computes the Information Coefficient: the Spearman rank
           correlation between the factor values and the forward returns
           across the cross-section of stocks.
        3. Ranks the remaining stocks by factor value and splits them into
           `n_quintiles` equal-sized (as close as possible) groups.
        4. Computes the equal-weighted average forward return for each
           group.

    It then aggregates these per-date results into summary statistics: mean
    IC, IC standard deviation, Information Ratio, the long-short (top
    quintile minus bottom quintile) spread's annualized Sharpe ratio, and
    its maximum drawdown.

    Args:
        factor_values: DataFrame of factor values, indexed by rebalance
            date, one column per ticker. Each row represents the factor
            value *as known at the start of the period* — i.e.,
            `factor_values.loc[t]` should only use information available at
            time t (callers are responsible for ensuring no look-ahead
            bias when constructing this DataFrame).
        forward_returns: DataFrame of returns earned over the period
            *following* each date in `factor_values.index` — i.e.,
            `forward_returns.loc[t]` is the return from t to the next
            rebalance date. Must have the same index and overlapping
            columns as `factor_values`. Typically constructed as
            `returns.shift(-1)` relative to a returns DataFrame aligned with
            `factor_values`.
        n_quintiles: Number of groups to split the ranked universe into.
            Despite the name "quintile" (which implies 5), this parameter
            allows other values (e.g., 10 for "decile" sorts). Must be >= 2.
        periods_per_year: Number of rebalance periods per year, used for
            annualizing returns and the Sharpe ratio (e.g., 12 for monthly
            rebalancing, 252 for daily, 52 for weekly).
        risk_free_rate: Annualized risk-free rate (as a decimal, e.g., 0.02
            for 2%) used to compute excess returns for the Sharpe ratio of
            the long-short spread. Defaults to 0.0 (i.e., the Sharpe ratio
            of the raw spread, which is already a "long-short" / dollar-
            neutral portfolio for which a risk-free rate adjustment is less
            standard, but the parameter is provided for completeness).

    Returns:
        A `BacktestResult` dataclass containing all computed statistics.

    Raises:
        ValueError: If `n_quintiles < 2`, if `factor_values` and
            `forward_returns` have no overlapping dates/columns, or if no
            rebalance date has at least `n_quintiles` stocks with valid
            (non-NaN) factor and return data.
    """
    if n_quintiles < 2:
        raise ValueError(f"n_quintiles must be >= 2, got {n_quintiles}.")

    common_dates = factor_values.index.intersection(forward_returns.index)
    if common_dates.empty:
        raise ValueError(
            "`factor_values` and `forward_returns` have no overlapping "
            "dates in their index."
        )

    common_cols = factor_values.columns.intersection(forward_returns.columns)
    if common_cols.empty:
        raise ValueError(
            "`factor_values` and `forward_returns` have no overlapping "
            "columns (tickers)."
        )

    quintile_labels = [f"Q{i + 1}" for i in range(n_quintiles)]
    quintile_return_rows = []
    ic_values = {}
    excluded_fractions = []

    for date in common_dates:
        factor_row = factor_values.loc[date, common_cols]
        return_row = forward_returns.loc[date, common_cols]

        valid = factor_row.notna() & return_row.notna()
        n_total = len(common_cols)
        n_valid = valid.sum()
        excluded_fractions.append(1.0 - (n_valid / n_total if n_total else 0.0))

        if n_valid < n_quintiles:
            logger.debug(
                "Skipping %s: only %d/%d stocks have valid factor+return "
                "data (need >= %d for %d quintiles).",
                date,
                n_valid,
                n_total,
                n_quintiles,
                n_quintiles,
            )
            continue

        valid_factor = factor_row[valid]
        valid_returns = return_row[valid]

        # Information Coefficient: Spearman rank correlation between factor
        # values and forward returns across the cross-section at this date.
        ic, _p_value = spearmanr(valid_factor, valid_returns)
        ic_values[date] = ic

        # Assign each stock to a quintile based on its factor rank.
        # `qcut` divides into equal-sized bins by rank; `duplicates="drop"`
        # handles cases where many stocks share the same factor value
        # (which would otherwise make bin edges non-unique).
        try:
            quintile_assignments = pd.qcut(
                valid_factor, q=n_quintiles, labels=False, duplicates="drop"
            )
        except ValueError:
            logger.debug(
                "Skipping %s: could not form %d distinct quintiles "
                "(too many tied factor values).",
                date,
                n_quintiles,
            )
            continue

        n_actual_groups = quintile_assignments.nunique()
        if n_actual_groups < n_quintiles:
            logger.debug(
                "%s: only formed %d/%d distinct groups due to tied factor "
                "values.",
                date,
                n_actual_groups,
                n_quintiles,
            )

        row = {}
        for i in range(n_quintiles):
            mask = quintile_assignments == i
            if mask.any():
                row[quintile_labels[i]] = valid_returns[mask].mean()
            else:
                row[quintile_labels[i]] = np.nan

        quintile_return_rows.append(pd.Series(row, name=date))

    if not quintile_return_rows:
        raise ValueError(
            "No rebalance date had enough valid (factor, forward_return) "
            f"pairs to form {n_quintiles} quintiles. Check for excessive "
            "missing data."
        )

    quintile_returns = pd.DataFrame(quintile_return_rows)
    quintile_returns["Q5_minus_Q1"] = (
        quintile_returns[quintile_labels[-1]] - quintile_returns[quintile_labels[0]]
    )

    ic_series = pd.Series(ic_values, name="IC")
    ic_series.index.name = "date"

    ic_mean = float(ic_series.mean())
    ic_std = float(ic_series.std())
    information_ratio = ic_mean / ic_std if ic_std != 0 else np.nan

    spread_returns = quintile_returns["Q5_minus_Q1"]
    sharpe = compute_sharpe_ratio(
        spread_returns, periods_per_year=periods_per_year, risk_free_rate=risk_free_rate
    )

    cumulative_returns = (1.0 + quintile_returns.fillna(0.0)).cumprod()
    max_dd = compute_max_drawdown(cumulative_returns["Q5_minus_Q1"])

    annualized_returns = compute_annualized_return(
        quintile_returns, periods_per_year=periods_per_year
    )

    return BacktestResult(
        quintile_returns=quintile_returns,
        ic_series=ic_series,
        ic_mean=ic_mean,
        ic_std=ic_std,
        information_ratio=information_ratio,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        cumulative_returns=cumulative_returns,
        annualized_returns=annualized_returns,
        n_periods=len(quintile_returns),
        excluded_fraction=float(np.mean(excluded_fractions)) if excluded_fractions else 0.0,
    )


def compute_sharpe_ratio(
    returns: pd.Series,
    periods_per_year: int = 12,
    risk_free_rate: float = 0.0,
) -> float:
    """Compute the annualized Sharpe ratio of a return series.

    **What it measures**: the Sharpe ratio is a measure of risk-adjusted
    return — how much excess return (return above a "risk-free" baseline)
    a strategy generates per unit of volatility (risk) it takes on. A higher
    Sharpe ratio means more return for the same risk, or equivalently, less
    risk for the same return.

    **Formula**:

        Sharpe = (mean(r) - rf_per_period) / std(r) * sqrt(periods_per_year)

    where `r` is the periodic return series, `rf_per_period` is the
    risk-free rate converted to the same periodicity as `r`, and the
    `sqrt(periods_per_year)` factor annualizes the ratio (since both the mean
    and standard deviation of returns scale differently with time — mean
    scales linearly, standard deviation scales with the square root of time
    under standard assumptions).

    Args:
        returns: Series of periodic returns (e.g., monthly).
        periods_per_year: Number of periods per year (12 for monthly, 252
            for daily, 52 for weekly).
        risk_free_rate: Annualized risk-free rate as a decimal (e.g., 0.02
            for 2% per year). Converted internally to a per-period rate by
            dividing by `periods_per_year` (a simple approximation; for
            short rates this difference from true compounding is
            negligible).

    Returns:
        The annualized Sharpe ratio. Returns NaN if `returns` has fewer than
        2 non-NaN observations or if its standard deviation is zero.
    """
    clean_returns = returns.dropna()
    if len(clean_returns) < 2:
        logger.warning(
            "compute_sharpe_ratio(): fewer than 2 valid observations; "
            "returning NaN."
        )
        return float("nan")

    rf_per_period = risk_free_rate / periods_per_year
    excess_returns = clean_returns - rf_per_period

    std = excess_returns.std()
    if std == 0:
        logger.warning(
            "compute_sharpe_ratio(): standard deviation of returns is zero; "
            "returning NaN."
        )
        return float("nan")

    return float(excess_returns.mean() / std * np.sqrt(periods_per_year))


def compute_max_drawdown(cumulative_returns: pd.Series) -> float:
    """Compute the maximum drawdown of a cumulative return series.

    **What it measures**: the largest percentage drop from a previous peak
    to a subsequent trough in a cumulative return curve. For example, if a
    strategy's cumulative value went from 1.50 (a new high) down to 1.05
    before recovering, that's a drawdown of (1.05 - 1.50) / 1.50 ≈ -30%.
    Max drawdown captures the worst "peak-to-trough" pain an investor would
    have experienced — a key practical risk measure, since large drawdowns
    are what cause investors to abandon a strategy (often right before it
    recovers).

    **Formula**:

        running_max_t = max(cumulative_returns[0..t])
        drawdown_t    = (cumulative_returns_t - running_max_t) / running_max_t
        max_drawdown  = min(drawdown_t)  for all t

    Args:
        cumulative_returns: Series of cumulative (compounded) returns,
            e.g., starting at 1.0 and evolving as
            `(1 + r1) * (1 + r2) * ...`. Must be strictly positive.

    Returns:
        The maximum drawdown as a negative decimal (e.g., -0.35 for a 35%
        decline), or 0.0 if the series is monotonically non-decreasing.
        Returns NaN if `cumulative_returns` is empty.

    Raises:
        ValueError: If any value in `cumulative_returns` is <= 0 (which
            would make percentage drawdown undefined/nonsensical).
    """
    if cumulative_returns.empty:
        return float("nan")

    if (cumulative_returns <= 0).any():
        raise ValueError(
            "`cumulative_returns` must be strictly positive (it represents "
            "a compounded growth factor starting from 1.0)."
        )

    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    return float(drawdown.min())


def compute_annualized_return(
    returns: pd.DataFrame | pd.Series,
    periods_per_year: int = 12,
) -> pd.Series | float:
    """Annualize a periodic return series using geometric compounding.

    **Formula**:

        annualized_return = (1 + total_return)^(periods_per_year / n_periods) - 1

    where `total_return` is the geometric (compounded) total return over the
    full sample, and `n_periods` is the number of periods observed. This is
    the standard "geometric annualization" — it answers the question "what
    constant per-period return, compounded `periods_per_year` times, would
    produce the same total growth?"

    Args:
        returns: Series or DataFrame of periodic returns (NaNs are dropped /
            treated as 0% for compounding purposes — i.e., "no position that
            period").
        periods_per_year: Number of periods per year (12 for monthly, etc.).

    Returns:
        If `returns` is a Series, returns a single float. If `returns` is a
        DataFrame, returns a Series of annualized returns, one per column.
    """
    filled = returns.fillna(0.0)
    n_periods = len(filled)
    if n_periods == 0:
        return float("nan") if isinstance(returns, pd.Series) else pd.Series(dtype=float)

    growth = (1.0 + filled).prod()
    annualized = growth ** (periods_per_year / n_periods) - 1.0

    if isinstance(returns, pd.Series):
        return float(annualized)
    return annualized


def plot_quintile_returns(
    result: BacktestResult,
    title: str = "Factor Backtest: Cumulative Returns by Quintile",
    save_path: Optional[str] = None,
    quintiles_to_plot: Optional[list[str]] = None,
):
    """Plot cumulative returns for selected quintiles (and the spread).

    By default, plots Q1 (bottom quintile), the top quintile (e.g., "Q5"),
    and the long-short spread "Q5_minus_Q1" — the three series most useful
    for visually assessing whether a factor "worked": a good factor should
    show the top quintile outperforming the bottom quintile, with the spread
    trending upward over time.

    Args:
        result: A `BacktestResult` from `run_quintile_backtest`.
        title: Plot title.
        save_path: If provided, save the figure to this file path (e.g.,
            "factor_backtest.png") instead of (or in addition to) returning
            it. The file format is inferred from the extension.
        quintiles_to_plot: List of column names from
            `result.cumulative_returns` to plot. If None, defaults to the
            bottom quintile, the top quintile, and "Q5_minus_Q1".

    Returns:
        The matplotlib `Figure` object, so callers can further customize or
        display it (e.g., in a notebook).

    Raises:
        ImportError: If matplotlib is not installed.
        KeyError: If any name in `quintiles_to_plot` is not a column of
            `result.cumulative_returns`.
    """
    if not _HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for plot_quintile_returns(). Install "
            "it with `pip install matplotlib`."
        )

    cum = result.cumulative_returns

    if quintiles_to_plot is None:
        quintile_cols = [c for c in cum.columns if c.startswith("Q") and "_" not in c]
        bottom = quintile_cols[0] if quintile_cols else None
        top = quintile_cols[-1] if quintile_cols else None
        quintiles_to_plot = [c for c in [bottom, top, "Q5_minus_Q1"] if c is not None]

    missing = [c for c in quintiles_to_plot if c not in cum.columns]
    if missing:
        raise KeyError(
            f"quintiles_to_plot contains unknown column(s) {missing}. "
            f"Available columns: {list(cum.columns)}"
        )

    fig, ax = plt.subplots(figsize=(10, 6))
    for col in quintiles_to_plot:
        ax.plot(cum.index, cum[col], label=col)

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Growth of $1")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        logger.info("Saved backtest plot to %s", save_path)

    return fig


def plot_factor_summary(
    result: BacktestResult,
    factor_name: str,
    subtitle: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """Plot a two-panel summary chart: cumulative returns + per-period IC.

    This is the "report-ready" chart for a factor backtest:

    - **Top panel**: cumulative growth of $1 for the bottom quintile ("Q1 -
      Lowest <factor_name> (losers)", red), the top quintile ("Q5 - Highest
      <factor_name> (winners)", green), and the long-short spread ("Q5 - Q1
      long-short spread", blue). A title and optional subtitle (e.g.,
      describing the universe and rebalance frequency) are shown above the
      chart, and a one-line stats summary (IC mean, IR, Sharpe, Max DD) is
      shown below it.
    - **Bottom panel**: a bar chart of the Information Coefficient at each
      rebalance date, colored green where IC > 0 (factor ranked correctly
      that period) and red where IC < 0 (factor ranked backwards that
      period). This makes it easy to see at a glance how *consistent* the
      factor's predictive power was over time, complementing the single
      IR number.

    Args:
        result: A `BacktestResult` from `run_quintile_backtest`.
        factor_name: Human-readable factor name, used in the legend labels
            and chart title (e.g., "12-1 Month Momentum").
        subtitle: Optional second line of the title describing the universe
            and rebalance frequency (e.g., "15-stock mega-cap universe,
            monthly rebalance").
        save_path: If provided, save the figure to this file path (e.g.,
            "factor_summary.png").

    Returns:
        The matplotlib `Figure` object.

    Raises:
        ImportError: If matplotlib is not installed.
        KeyError: If `result.cumulative_returns` does not contain at least
            two quintile columns plus "Q5_minus_Q1".
    """
    if not _HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for plot_factor_summary(). Install it "
            "with `pip install matplotlib`."
        )

    cum = result.cumulative_returns
    quintile_cols = [c for c in cum.columns if c.startswith("Q") and "_" not in c]
    if len(quintile_cols) < 2 or "Q5_minus_Q1" not in cum.columns:
        raise KeyError(
            "result.cumulative_returns must contain at least two quintile "
            "columns (e.g., 'Q1', 'Q5') and a 'Q5_minus_Q1' spread column."
        )

    bottom_q, top_q = quintile_cols[0], quintile_cols[-1]

    years = f"{cum.index[0].year}-{cum.index[-1].year}"
    title = f"{factor_name}: Quintile Backtest, {years}"

    fig, (ax_cum, ax_ic) = plt.subplots(
        nrows=2, ncols=1, figsize=(11, 8), height_ratios=[3, 1.4], sharex=True
    )

    ax_cum.plot(
        cum.index, cum[bottom_q], color="firebrick",
        label=f"{bottom_q} - Lowest {factor_name} (losers)",
    )
    ax_cum.plot(
        cum.index, cum[top_q], color="forestgreen",
        label=f"{top_q} - Highest {factor_name} (winners)",
    )
    ax_cum.plot(
        cum.index, cum["Q5_minus_Q1"], color="steelblue",
        label=f"{top_q} - {bottom_q} long-short spread",
    )
    ax_cum.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax_cum.set_ylabel("Cumulative Growth of $1")
    ax_cum.legend(loc="upper left")

    full_title = title if subtitle is None else f"{title}\n{subtitle}"
    ax_cum.set_title(full_title, fontsize=13, fontweight="bold")

    stats_line = (
        f"IC (mean): {result.ic_mean:.3f}   |   IR: {result.information_ratio:.3f}"
        f"   |   Q5-Q1 Sharpe: {result.sharpe_ratio:.2f}"
        f"   |   Q5-Q1 Max DD: {result.max_drawdown:.1%}"
    )
    ax_ic.set_title(stats_line, fontsize=10, style="italic", color="dimgray")

    ic = result.ic_series
    colors = ["forestgreen" if v >= 0 else "indianred" for v in ic.values]
    ax_ic.bar(ic.index, ic.values, color=colors, width=20)
    ax_ic.axhline(0.0, color="black", linewidth=0.8)
    ax_ic.set_ylabel("Information\nCoefficient")
    ax_ic.set_xlabel("Date")

    fig.autofmt_xdate()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        logger.info("Saved factor summary plot to %s", save_path)

    return fig


def summarize_backtest(result: BacktestResult) -> str:
    """Produce a human-readable text summary of a `BacktestResult`.

    Args:
        result: A `BacktestResult` from `run_quintile_backtest`.

    Returns:
        A multi-line string summarizing the key statistics, suitable for
        printing to the console or including in a research note.
    """
    lines = [
        f"Backtest summary ({result.n_periods} rebalance periods):",
        f"  IC (mean):              {result.ic_mean:.4f}",
        f"  IC (std):               {result.ic_std:.4f}",
        f"  Information Ratio (IR): {result.information_ratio:.4f}",
        f"  Q5-Q1 Sharpe ratio:     {result.sharpe_ratio:.4f}",
        f"  Q5-Q1 max drawdown:     {result.max_drawdown:.2%}",
        f"  Avg. universe excluded per period: {result.excluded_fraction:.2%}",
        "",
        "  Annualized returns by quintile:",
    ]
    for label, value in result.annualized_returns.items():
        lines.append(f"    {label}: {value:.2%}")

    return "\n".join(lines)

"""
factor_engine.py
=================

Computes standard quantitative equity factors and supports custom
user-defined factor formulas.

A "factor" in this module is a function that takes price and/or fundamental
data and produces a single numeric score per stock (per date, for
time-varying factors like momentum and volatility). These scores are later
used in `backtest.py` to rank stocks into quintiles.

Factors implemented here, grouped by family (see `agents/quantsignal.md` for
the full classification framework):

    Momentum:
        - `compute_momentum_12_1`: classic 12-1 month price momentum.

    Value:
        - `compute_book_to_market`: book equity / market cap.
        - `compute_earnings_yield`: earnings / market cap (inverse P/E).
        - `compute_fcf_yield`: free cash flow / market cap.

    Quality:
        - `compute_roe`: return on equity (net income / book equity).
        - `compute_gross_profitability`: gross profit / total assets.
        - `compute_accruals`: a measure of earnings "quality" based on the
          gap between accounting earnings and cash flow.

    Low Volatility:
        - `compute_realized_volatility`: standard deviation of returns.
        - `compute_beta`: sensitivity of stock returns to market returns.

    Custom:
        - `compute_custom_factor`: evaluates a user-supplied formula string
          against a DataFrame of named columns.

All "value" and "quality" factors expect a fundamentals DataFrame with the
column names produced by `data_fetcher.fetch_fundamental_data()` (e.g.,
`market_cap`, `total_book_equity`, `net_income`, etc.) plus, where noted, a
`price` column.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

try:
    from .utils import safe_divide
except ImportError:  # pragma: no cover - fallback when run without package context
    from utils import safe_divide

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------


def compute_momentum_12_1(prices: pd.DataFrame, skip_recent_month: bool = True) -> pd.DataFrame:
    """Compute the classic "12-1 month" price momentum factor.

    **What momentum measures**: the idea that stocks which have performed
    well over the recent past (here, the last ~12 months) tend to continue
    performing relatively well over the next few months, and vice versa for
    losers. This is one of the most extensively documented patterns in
    finance (Jegadeesh & Titman, 1993).

    **Why skip the most recent month?** Academic research has found a
    separate, opposite-signed "short-term reversal" effect: a stock's return
    over just the last 1 month tends to partially reverse over the following
    month (often attributed to bid-ask bounce, liquidity provision, and
    short-term overreaction). Including the most recent month would mix
    these two opposing effects together, so the standard construction
    "skips" it.

    **Formula**:

        momentum_t = (P_{t-1m} - P_{t-12m}) / P_{t-12m}

    where `P_{t-1m}` is the price approximately 1 month before date `t`, and
    `P_{t-12m}` is the price approximately 12 months before date `t`. On
    daily data, "1 month" is approximated as 21 trading days and "12 months"
    as 252 trading days.

    Args:
        prices: DataFrame of daily adjusted close prices, indexed by date,
            one column per ticker. Must have at least 253 rows (252 trading
            days plus 1) to compute a value for the most recent date.
        skip_recent_month: If True (default), use the 12-1 construction
            described above. If False, compute a simple trailing 12-month
            return (P_t - P_{t-12m}) / P_{t-12m} with no skip — sometimes
            called "12-month momentum" without the gap.

    Returns:
        DataFrame of momentum scores, same shape and index as `prices`,
        with NaN for the initial rows where insufficient history exists.

    Raises:
        ValueError: If `prices` has fewer than 253 rows (or 252 if
            `skip_recent_month=False`).
    """
    trading_days_per_month = 21
    trading_days_per_year = 252

    min_rows_required = (
        trading_days_per_year + trading_days_per_month
        if skip_recent_month
        else trading_days_per_year
    )
    if len(prices) < min_rows_required:
        raise ValueError(
            f"Need at least {min_rows_required} rows of price history to "
            f"compute 12-1 momentum (got {len(prices)})."
        )

    if skip_recent_month:
        p_t_minus_1m = prices.shift(trading_days_per_month)
        p_t_minus_12m = prices.shift(trading_days_per_year + trading_days_per_month)
    else:
        p_t_minus_1m = prices
        p_t_minus_12m = prices.shift(trading_days_per_year)

    momentum = safe_divide(p_t_minus_1m - p_t_minus_12m, p_t_minus_12m)
    return momentum


# ---------------------------------------------------------------------------
# Value factors
# ---------------------------------------------------------------------------


def compute_book_to_market(fundamentals: pd.DataFrame) -> pd.Series:
    """Compute the book-to-market ratio: total book equity / market cap.

    **What it measures**: how "cheap" a stock is relative to the accounting
    value of the company's net assets. A high book-to-market ratio means the
    market is valuing the company at close to (or below) its accounting net
    worth — traditionally interpreted as "cheap." A low ratio means the
    market is pricing in significant value beyond the balance sheet (e.g.,
    growth expectations, brand value, intangibles not on the books) —
    traditionally interpreted as "expensive" or "growth."

    **Formula**:

        book_to_market = total_book_equity / market_cap

    "Book equity" (also called "shareholders' equity" or "net worth") is
    total assets minus total liabilities, as reported on the balance sheet —
    roughly, what would be left over for shareholders if the company sold
    all its assets and paid off all its debts. "Market cap" is the total
    market value of all outstanding shares (price x shares outstanding).

    Args:
        fundamentals: DataFrame with at least the columns `total_book_equity`
            and `market_cap` (as produced by
            `data_fetcher.fetch_fundamental_data()`), indexed by ticker.

    Returns:
        Series of book-to-market ratios, indexed by ticker. Returns NaN for
        any ticker where `market_cap` is zero/NaN, or where
        `total_book_equity` is NaN. Negative book equity (a company with
        more liabilities than assets) is preserved as a negative ratio
        rather than being treated as missing — a very negative book-to-market
        is itself informative (it often signals financial distress) and
        should generally be filtered out via `utils.filter_universe` rather
        than silently hidden here.

    Raises:
        KeyError: If required columns are missing from `fundamentals`.
    """
    _require_columns(fundamentals, ["total_book_equity", "market_cap"])
    return safe_divide(fundamentals["total_book_equity"], fundamentals["market_cap"])


def compute_earnings_yield(fundamentals: pd.DataFrame) -> pd.Series:
    """Compute the earnings yield: net income / market cap.

    **What it measures**: the inverse of the price-to-earnings (P/E) ratio,
    expressed as a percentage. A higher earnings yield means a stock
    generates more accounting earnings per dollar of market value — i.e., it
    is "cheaper" relative to its current profitability. Using the *yield*
    (earnings/price) rather than the ratio (price/earnings) is preferred in
    factor construction because it behaves better for companies with
    negative or near-zero earnings (a P/E of "-200" or "+0.01" is hard to
    rank meaningfully, but an earnings yield of "-5%" or "+1000%" sorts
    sensibly... though extreme values should still be winsorized — see
    `utils.winsorize`).

    **Formula**:

        earnings_yield = net_income / market_cap

    Args:
        fundamentals: DataFrame with at least the columns `net_income` and
            `market_cap`, indexed by ticker.

    Returns:
        Series of earnings yields, indexed by ticker. NaN where `market_cap`
        is zero/NaN.

    Raises:
        KeyError: If required columns are missing.
    """
    _require_columns(fundamentals, ["net_income", "market_cap"])
    return safe_divide(fundamentals["net_income"], fundamentals["market_cap"])


def compute_fcf_yield(fundamentals: pd.DataFrame) -> pd.Series:
    """Compute the free cash flow (FCF) yield: free cash flow / market cap.

    **What it measures**: similar to earnings yield, but uses *free cash
    flow* instead of accounting net income. Free cash flow is the cash a
    company generates from its operations after paying for the capital
    expenditures needed to maintain/grow its asset base
    (FCF = operating cash flow - capital expenditures). Many practitioners
    consider FCF yield a "harder to manipulate" measure of cheapness than
    earnings yield, because accounting earnings can be influenced by
    non-cash items (depreciation policy, accruals, one-time charges) in ways
    that cash flow cannot.

    **Formula**:

        fcf_yield = free_cashflow / market_cap

    Args:
        fundamentals: DataFrame with at least the columns `free_cashflow`
            and `market_cap`, indexed by ticker.

    Returns:
        Series of FCF yields, indexed by ticker. NaN where `market_cap` is
        zero/NaN or `free_cashflow` is NaN (free cash flow is one of the
        less consistently populated fields in free data sources).

    Raises:
        KeyError: If required columns are missing.
    """
    _require_columns(fundamentals, ["free_cashflow", "market_cap"])
    return safe_divide(fundamentals["free_cashflow"], fundamentals["market_cap"])


# ---------------------------------------------------------------------------
# Quality factors
# ---------------------------------------------------------------------------


def compute_roe(fundamentals: pd.DataFrame) -> pd.Series:
    """Compute Return on Equity (ROE): net income / total book equity.

    **What it measures**: how efficiently a company generates profit from
    the capital its shareholders have invested. A higher ROE generally
    indicates a more profitable, better-run business. Academic research
    (e.g., Novy-Marx, Asness/Frazzini/Pedersen "Quality Minus Junk") has
    found that high-profitability stocks tend to deliver better risk-adjusted
    returns than low-profitability stocks, even after controlling for value
    and size — a "quality" premium.

    **Formula**:

        ROE = net_income / total_book_equity

    Args:
        fundamentals: DataFrame with at least the columns `net_income` and
            `total_book_equity`, indexed by ticker.

    Returns:
        Series of ROE values (as decimals, e.g., 0.15 = 15%), indexed by
        ticker. NaN where `total_book_equity` is zero/NaN. Companies with
        negative book equity will produce a negative (and often extreme)
        ROE — these should typically be excluded via `utils.filter_universe`
        or capped via `utils.winsorize`, since a negative-equity company with
        positive net income would otherwise show an enormous *negative*
        ROE that doesn't reflect "quality" in any meaningful sense.

    Raises:
        KeyError: If required columns are missing.
    """
    _require_columns(fundamentals, ["net_income", "total_book_equity"])
    return safe_divide(fundamentals["net_income"], fundamentals["total_book_equity"])


def compute_gross_profitability(fundamentals: pd.DataFrame) -> pd.Series:
    """Compute gross profitability: gross profit / total assets.

    **What it measures**: introduced by Novy-Marx (2013, "The Other Side of
    Value: The Gross Profitability Premium"), this measures how much gross
    profit a company squeezes out of its total asset base, regardless of how
    it's financed (unlike ROE, which is sensitive to leverage — a company
    can boost ROE simply by taking on more debt without becoming more
    "profitable" in an operational sense). "Gross profit" is revenue minus
    cost of goods sold (COGS) — i.e., profit before operating expenses,
    interest, and taxes.

    **Formula**:

        gross_profitability = gross_profits / total_assets

    Args:
        fundamentals: DataFrame with at least the columns `gross_profits`
            and `total_assets`, indexed by ticker.

    Returns:
        Series of gross profitability values, indexed by ticker. NaN where
        `total_assets` is zero/NaN.

    Raises:
        KeyError: If required columns are missing.

    Note:
        `data_fetcher.fetch_fundamental_data()` does not currently populate
        `total_assets` (yfinance's `Ticker.info` does not reliably expose
        it); callers typically need to supplement with
        `Ticker.balance_sheet` or another data source for this factor.
    """
    _require_columns(fundamentals, ["gross_profits", "total_assets"])
    return safe_divide(fundamentals["gross_profits"], fundamentals["total_assets"])


def compute_accruals(fundamentals: pd.DataFrame) -> pd.Series:
    """Compute the accruals ratio, a measure of earnings quality.

    **What it measures**: "Accruals" capture the portion of reported
    earnings that is *not* backed by actual cash flow. The accruals
    anomaly (Sloan, 1996) found that companies with high accruals (earnings
    much higher than cash flow — often achieved through aggressive revenue
    recognition, inventory build-ups, etc.) tend to *underperform* going
    forward, while companies whose earnings are well-backed by cash
    ("low accruals") tend to perform better. Intuitively: cash is harder to
    manipulate than accounting earnings, so a large gap between the two is a
    yellow flag for earnings quality.

    **Formula** (Sloan-style, balance-sheet approximation using available
    fields):

        accruals = (net_income - free_cashflow) / total_assets

    A positive value means earnings exceed free cash flow (potential
    red flag); a negative or near-zero value means earnings are well
    supported by cash generation.

    Args:
        fundamentals: DataFrame with at least the columns `net_income`,
            `free_cashflow`, and `total_assets`, indexed by ticker.

    Returns:
        Series of accrual ratios, indexed by ticker. NaN where
        `total_assets` is zero/NaN, or where either `net_income` or
        `free_cashflow` is NaN.

    Raises:
        KeyError: If required columns are missing.

    Note:
        Because **low** accruals are associated with **better** future
        returns, this factor should typically be **negated** (multiplied by
        -1) before combining it with other "higher is better" factors so
        that higher composite scores consistently mean "more attractive."
    """
    _require_columns(fundamentals, ["net_income", "free_cashflow", "total_assets"])
    numerator = fundamentals["net_income"] - fundamentals["free_cashflow"]
    return safe_divide(numerator, fundamentals["total_assets"])


# ---------------------------------------------------------------------------
# Low volatility factors
# ---------------------------------------------------------------------------


def compute_realized_volatility(
    prices: pd.DataFrame,
    window: int = 252,
    annualize: bool = True,
) -> pd.DataFrame:
    """Compute trailing realized volatility of daily returns.

    **What it measures**: how much a stock's price has fluctuated, measured
    as the standard deviation of its daily returns over a trailing window.
    The "low volatility anomaly" (Ang, Hodrick, Xing & Zhang, 2006; Frazzini
    & Pedersen, 2014) is the empirical finding that low-volatility stocks
    have historically delivered surprisingly competitive — sometimes
    higher — risk-adjusted returns compared to high-volatility stocks, which
    is at odds with the simple theoretical prediction that taking on more
    risk should be compensated with proportionally higher expected return.

    **Formula**:

        realized_vol_t = std(daily_returns over [t-window, t])

    If `annualize=True`, the daily standard deviation is multiplied by
    sqrt(252) (the approximate number of trading days in a year), which
    converts it to an annualized volatility figure comparable to commonly
    quoted "annualized vol" numbers (e.g., "this stock has ~30% annualized
    volatility").

    For ranking purposes (as used in a quintile sort), **lower** realized
    volatility is generally considered more attractive under the
    low-volatility anomaly — i.e., this factor should typically be
    **negated** before combining with "higher is better" factors, or the
    backtest should explicitly rank ascending for this factor.

    Args:
        prices: DataFrame of daily adjusted close prices, indexed by date,
            one column per ticker.
        window: Trailing window length in trading days (default 252, i.e.,
            approximately 1 year).
        annualize: If True (default), scale the daily standard deviation by
            sqrt(252).

    Returns:
        DataFrame of realized volatility values, same shape/index as
        `prices`, with NaN for the initial `window` rows.

    Raises:
        ValueError: If `prices` has fewer than `window + 1` rows.
    """
    if len(prices) < window + 1:
        raise ValueError(
            f"Need at least {window + 1} rows of price history to compute "
            f"realized volatility with window={window} (got {len(prices)})."
        )

    daily_returns = prices.pct_change()
    rolling_vol = daily_returns.rolling(window=window).std()

    if annualize:
        rolling_vol = rolling_vol * np.sqrt(252)

    return rolling_vol


def compute_beta(
    prices: pd.DataFrame,
    market_prices: pd.Series,
    window: int = 252,
) -> pd.DataFrame:
    """Compute trailing beta (market sensitivity) for each stock.

    **What it measures**: "Beta" quantifies how much a stock's returns tend
    to move in response to overall market returns. A beta of 1.0 means the
    stock tends to move in line with the market; a beta of 1.5 means it
    tends to amplify market moves by 50% (more volatile than the market); a
    beta below 1.0 means it tends to dampen market moves (less volatile than
    the market). Beta is computed as the slope of a linear regression of a
    stock's returns on the market's returns:

        beta_i = Cov(r_i, r_market) / Var(r_market)

    The "Betting Against Beta" literature (Frazzini & Pedersen, 2014) found
    that low-beta stocks have historically delivered higher risk-adjusted
    returns than high-beta stocks — another facet of the low-volatility
    anomaly, with a proposed explanation being that leverage-constrained
    investors "reach for yield" by overweighting high-beta stocks instead of
    using leverage on low-beta stocks, bidding up the price (and reducing the
    expected return) of high-beta names.

    Args:
        prices: DataFrame of daily adjusted close prices for the stocks of
            interest, indexed by date, one column per ticker.
        market_prices: Series of daily adjusted close prices for a market
            proxy (e.g., SPY for the S&P 500), with the same DatetimeIndex
            (or a superset of it) as `prices`.
        window: Trailing window length in trading days (default 252).

    Returns:
        DataFrame of rolling beta values, same shape/index as `prices`, with
        NaN for the initial `window` rows.

    Raises:
        ValueError: If `prices` and `market_prices` have no overlapping
            dates, or if there is insufficient history for the given window.
    """
    market_returns = market_prices.pct_change()
    stock_returns = prices.pct_change()

    aligned_market, aligned_stocks = market_returns.align(
        stock_returns, join="inner", axis=0
    )

    if aligned_market.empty:
        raise ValueError(
            "`prices` and `market_prices` have no overlapping dates."
        )

    if len(aligned_market) < window + 1:
        raise ValueError(
            f"Need at least {window + 1} overlapping rows to compute beta "
            f"with window={window} (got {len(aligned_market)})."
        )

    market_var = aligned_market.rolling(window=window).var()

    betas = pd.DataFrame(index=aligned_stocks.index, columns=aligned_stocks.columns, dtype=float)
    for col in aligned_stocks.columns:
        cov = aligned_stocks[col].rolling(window=window).cov(aligned_market)
        betas[col] = safe_divide(cov, market_var)

    return betas


# ---------------------------------------------------------------------------
# Custom factors
# ---------------------------------------------------------------------------


def compute_custom_factor(data: pd.DataFrame, formula: str) -> pd.Series:
    """Evaluate a user-supplied formula against named columns of a DataFrame.

    This allows QuantSignal to compute factors that aren't among the
    pre-built functions above, by letting the user (or the QuantSignal
    agent, when formalizing a hypothesis in Step 1) express a factor as an
    arithmetic expression over column names.

    Args:
        data: DataFrame whose columns can be referenced by name in
            `formula` (e.g., columns like `net_income`, `market_cap`,
            `total_assets`, or any custom columns the caller has computed
            and merged in).
        formula: A string expression using column names and standard
            arithmetic operators / numpy functions, e.g.:
                - "net_income / market_cap"
                - "(total_revenue - cost_of_goods_sold) / total_assets"
                - "log(market_cap)"  (uses numpy's `log` via `engine='python'`
                  is not required — pandas' `eval` supports a restricted set
                  of functions; for arbitrary numpy functions, prefer
                  expressing the formula without function calls where
                  possible, e.g., precompute a `log_market_cap` column.)

    Returns:
        A Series containing the result of evaluating `formula` row-by-row
        against `data`, with the same index as `data`.

    Raises:
        ValueError: If `formula` is empty, or references columns not present
            in `data`.
        Exception: Re-raises any exception from `pandas.DataFrame.eval` with
            additional context about the formula that failed, to aid
            debugging of user-supplied expressions.

    Security note:
        This function uses `pandas.DataFrame.eval(formula, engine="python")`,
        which evaluates the expression using Python's `eval` under the hood
        restricted to the DataFrame's columns and a small set of numpy/python
        builtins that `pandas.eval` exposes. **Do not pass untrusted,
        user-controlled formula strings from an unauthenticated/external
        source into this function in a production system without additional
        sandboxing** — while `pandas.eval` is more restricted than raw
        `eval`, it is not a hardened sandbox. In the QuantSignal agent
        context, the formula originates from the same trusted user
        conversing with the agent, which is an acceptable trust boundary for
        a local research tool.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     "net_income": [100, 200],
        ...     "market_cap": [1000, 5000],
        ... }, index=["AAA", "BBB"])
        >>> compute_custom_factor(df, "net_income / market_cap")
        AAA    0.10
        BBB    0.04
        dtype: float64
    """
    if not formula or not formula.strip():
        raise ValueError("`formula` must be a non-empty expression string.")

    referenced_columns = _extract_identifiers(formula)
    missing = [c for c in referenced_columns if c not in data.columns]
    if missing:
        raise ValueError(
            f"Formula references column(s) not found in data: {missing}. "
            f"Available columns: {list(data.columns)}"
        )

    try:
        result = data.eval(formula, engine="python")
    except Exception as exc:  # noqa: BLE001 - surface eval errors with context
        raise type(exc)(
            f"Failed to evaluate custom factor formula '{formula}': {exc}"
        ) from exc

    if not isinstance(result, pd.Series):
        # `eval` can return a scalar if the formula doesn't reference any
        # columns (e.g., "1 + 1"); broadcast to a Series for consistency.
        result = pd.Series(result, index=data.index)

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise a clear KeyError if any of `columns` is missing from `df`."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required column(s) {missing}. Available columns: "
            f"{list(df.columns)}"
        )


def _extract_identifiers(formula: str) -> list[str]:
    """Extract likely column-name identifiers from a formula string.

    This is a lightweight heuristic (not a full parser) used to give a
    helpful error message when a formula references an unknown column,
    *before* handing the formula to `pandas.eval`. It splits on common
    arithmetic operators and parentheses, and filters out pure numbers and
    a small set of known function names.
    """
    import re

    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", formula)
    known_functions = {"log", "exp", "sqrt", "abs", "where", "min", "max"}
    return [t for t in set(tokens) if t not in known_functions]

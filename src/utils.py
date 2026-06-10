"""
utils.py
========

Shared helper functions used across QuantSignal's data, factor, and backtest
modules.

This module focuses on the "data hygiene" steps that are essential in
quantitative finance but easy to get wrong:

- **Winsorization**: capping extreme outlier values so that a few crazy data
  points (e.g., a stock with a P/E of 50,000 because its earnings are nearly
  zero) don't dominate a factor's statistics.
- **Z-score normalization**: rescaling factor values so they can be compared
  across stocks and across time on a common scale (mean 0, standard
  deviation 1).
- **Universe filtering**: applying basic liquidity / data-quality filters so
  that a backtest doesn't end up "trading" illiquid penny stocks just because
  they happen to score well on a factor.

Every function here operates on plain pandas Series/DataFrames so it can be
reused by `factor_engine.py`, `backtest.py`, and `scorer.py` without creating
circular imports.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def winsorize(
    series: pd.Series,
    lower_pct: float = 0.01,
    upper_pct: float = 0.99,
) -> pd.Series:
    """Cap extreme values in a Series at given lower/upper percentiles.

    "Winsorizing" means: instead of *removing* outliers (which throws away
    data and can bias results), we *clip* them to a reasonable maximum and
    minimum value. For example, with the default 1%/99% bounds, any value
    below the 1st percentile is set equal to the 1st percentile value, and
    any value above the 99th percentile is set equal to the 99th percentile
    value. This prevents a handful of extreme outliers (e.g., a stock that's
    about to go bankrupt with a book-to-market ratio of 500) from dominating
    statistics like the mean and standard deviation.

    Args:
        series: The input data (e.g., a column of factor values for one
            rebalance date).
        lower_pct: Lower percentile bound, expressed as a fraction (0.01 =
            1st percentile). Must be in [0, 1).
        upper_pct: Upper percentile bound, expressed as a fraction (0.99 =
            99th percentile). Must be in (0, 1] and greater than lower_pct.

    Returns:
        A new Series with the same index as the input, where values outside
        the [lower_pct, upper_pct] range have been clipped to the boundary
        values. NaN values are preserved as NaN (they are ignored when
        computing the percentile thresholds).

    Raises:
        ValueError: If `lower_pct` and `upper_pct` are not valid percentile
            bounds (i.e., not 0 <= lower_pct < upper_pct <= 1).

    Example:
        >>> s = pd.Series([1, 2, 3, 4, 5, 1000])
        >>> winsorize(s, lower_pct=0.0, upper_pct=0.8)
        0    1.0
        1    2.0
        2    3.0
        3    4.0
        4    4.0
        5    4.0
        dtype: float64
    """
    if not (0 <= lower_pct < upper_pct <= 1):
        raise ValueError(
            f"Invalid percentile bounds: lower_pct={lower_pct}, "
            f"upper_pct={upper_pct}. Require 0 <= lower_pct < upper_pct <= 1."
        )

    if series.dropna().empty:
        logger.warning("winsorize() called on a Series with no non-NaN values.")
        return series.copy()

    lower_bound = series.quantile(lower_pct)
    upper_bound = series.quantile(upper_pct)

    return series.clip(lower=lower_bound, upper=upper_bound)


def winsorize_dataframe(
    df: pd.DataFrame,
    columns: Optional[Iterable[str]] = None,
    lower_pct: float = 0.01,
    upper_pct: float = 0.99,
) -> pd.DataFrame:
    """Apply `winsorize()` to one or more columns of a DataFrame.

    Args:
        df: Input DataFrame, typically with one row per stock for a single
            rebalance date (a "cross-section").
        columns: Names of columns to winsorize. If None, all numeric columns
            are winsorized.
        lower_pct: See `winsorize`.
        upper_pct: See `winsorize`.

    Returns:
        A copy of `df` with the specified columns winsorized.
    """
    result = df.copy()
    if columns is None:
        columns = result.select_dtypes(include=[np.number]).columns

    for col in columns:
        if col not in result.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")
        result[col] = winsorize(result[col], lower_pct=lower_pct, upper_pct=upper_pct)

    return result


def zscore_normalize(series: pd.Series, ddof: int = 0) -> pd.Series:
    """Convert a Series of values into z-scores (standard scores).

    A "z-score" tells you how many standard deviations a value is away from
    the mean of the group. After this transformation, the resulting values
    have (approximately) mean 0 and standard deviation 1. This is essential
    for combining multiple factors (e.g., a value score and a momentum
    score) into a composite signal — without normalization, a factor
    measured in percent (like earnings yield, ~5%) would be swamped by a
    factor measured in raw dollars (like market cap, ~10^10).

    Formula:
        z_i = (x_i - mean(x)) / std(x)

    Args:
        series: Input data (typically a single cross-section of factor
            values, i.e., all stocks at one point in time).
        ddof: Delta degrees of freedom for the standard deviation
            calculation. ddof=0 (population std, the default) is standard
            for cross-sectional z-scoring; ddof=1 gives the sample std.

    Returns:
        A Series of z-scores with the same index as the input. If the
        standard deviation is zero (all values identical) or NaN (e.g., the
        series has fewer than 2 non-NaN values), returns a Series of zeros
        (with NaNs preserved where the input was NaN) rather than dividing
        by zero.

    Example:
        >>> s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> zscore_normalize(s).round(3).tolist()
        [-1.414, -0.707, 0.0, 0.707, 1.414]
    """
    mean = series.mean()
    std = series.std(ddof=ddof)

    if std == 0 or np.isnan(std):
        logger.warning(
            "zscore_normalize(): standard deviation is zero or NaN; "
            "returning zeros (NaNs preserved)."
        )
        return series.where(series.isna(), 0.0)

    return (series - mean) / std


def clean_returns(returns: pd.DataFrame, max_abs_return: float = 1.0) -> pd.DataFrame:
    """Clean a DataFrame of periodic returns by removing implausible values.

    Stock split/dividend adjustment errors, data vendor glitches, or
    de-listing events can sometimes produce nonsensical return values (e.g.,
    a "+5000%" single-day move that's actually a data error rather than a
    real price move). This function replaces any return whose absolute value
    exceeds `max_abs_return` with NaN, so it doesn't distort downstream
    statistics.

    Args:
        returns: DataFrame of periodic returns (e.g., monthly simple
            returns), with dates as the index and tickers as columns.
        max_abs_return: The maximum plausible absolute periodic return
            (e.g., 1.0 = 100%). Values with a larger magnitude are treated
            as data errors and set to NaN.

    Returns:
        A cleaned copy of the input DataFrame.
    """
    cleaned = returns.copy()
    mask = cleaned.abs() > max_abs_return
    n_flagged = mask.sum().sum()
    if n_flagged > 0:
        logger.warning(
            "clean_returns(): flagged %d return value(s) with |return| > %.2f "
            "as likely data errors and set to NaN.",
            n_flagged,
            max_abs_return,
        )
    cleaned[mask] = np.nan
    return cleaned


def filter_universe(
    df: pd.DataFrame,
    price_col: str = "price",
    market_cap_col: str = "market_cap",
    min_price: float = 5.0,
    min_market_cap: float = 2e9,
    required_columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Apply standard liquidity / data-quality filters to a stock universe.

    Quantitative strategies typically exclude very small or very cheap
    stocks because:
      - **Penny stocks** (price below a few dollars) often have wide
        bid-ask spreads, meaning the difference between the price you'd buy
        at and the price you'd sell at is large relative to the stock's
        price — making the strategy's theoretical returns hard or
        impossible to actually capture.
      - **Micro-cap stocks** (very small total market value) are often too
        illiquid for an institutional-size strategy to trade without moving
        the price against itself ("market impact").
      - Rows missing required data fields can't be used to compute the
        factor and should be excluded rather than silently treated as zero.

    Args:
        df: Input DataFrame with one row per stock (a single cross-section).
            Must contain at least the columns named by `price_col` and
            `market_cap_col` unless those checks are skipped (see below).
        price_col: Name of the column containing the stock's price. Pass
            None-equivalent (an empty string) is not supported; if the
            column doesn't exist, the price filter is skipped with a
            warning.
        market_cap_col: Name of the column containing market capitalization
            in dollars.
        min_price: Minimum price (in the same units as `price_col`,
            typically USD) for a stock to be included. Set to 0 to disable.
        min_market_cap: Minimum market capitalization in dollars for a stock
            to be included. Set to 0 to disable.
        required_columns: Additional columns that must be non-NaN for a row
            to be retained (e.g., the factor columns you're about to compute
            with).

    Returns:
        A filtered copy of `df` containing only rows that pass all checks.
        The original index is preserved so results can be joined back to
        other DataFrames.
    """
    result = df.copy()
    n_start = len(result)

    if price_col in result.columns and min_price > 0:
        result = result[result[price_col] >= min_price]
    elif price_col not in result.columns:
        logger.warning(
            "filter_universe(): price column '%s' not found; skipping price filter.",
            price_col,
        )

    if market_cap_col in result.columns and min_market_cap > 0:
        result = result[result[market_cap_col] >= min_market_cap]
    elif market_cap_col not in result.columns:
        logger.warning(
            "filter_universe(): market cap column '%s' not found; skipping "
            "market cap filter.",
            market_cap_col,
        )

    if required_columns:
        missing_cols = [c for c in required_columns if c not in result.columns]
        if missing_cols:
            raise KeyError(
                f"required_columns not found in DataFrame: {missing_cols}"
            )
        result = result.dropna(subset=list(required_columns))

    n_end = len(result)
    logger.info(
        "filter_universe(): retained %d/%d rows (%.1f%%) after applying filters.",
        n_end,
        n_start,
        100.0 * n_end / n_start if n_start else 0.0,
    )
    return result


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
    fill_value: float = np.nan,
) -> pd.Series:
    """Element-wise division that avoids divide-by-zero errors and warnings.

    Many financial ratios (e.g., earnings yield = earnings / market cap,
    ROE = net income / book equity) involve dividing by a quantity that can
    legitimately be zero or negative for some companies (e.g., a company
    with negative book equity due to accumulated losses or large buybacks).
    Rather than letting this produce `inf`, `-inf`, or a runtime warning,
    `safe_divide` returns `fill_value` (NaN by default) wherever the
    denominator is zero.

    Args:
        numerator: Series of numerator values.
        denominator: Series of denominator values, same index as
            `numerator`.
        fill_value: Value to use where the denominator is zero. Defaults to
            NaN, signaling "this ratio is undefined for this stock."

    Returns:
        A Series of `numerator / denominator`, with zero-denominator entries
        replaced by `fill_value`.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator
    result = result.where(denominator != 0, fill_value)
    return result


def to_monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert a DataFrame of daily prices into monthly simple returns.

    A "simple return" over a period is (P_end - P_start) / P_start, i.e.,
    the percentage change in price. This function resamples daily prices to
    month-end observations (taking the last available price in each
    calendar month) and then computes the percentage change from one
    month-end to the next.

    Args:
        prices: DataFrame of (adjusted) daily close prices, with a
            DatetimeIndex and one column per ticker.

    Returns:
        DataFrame of monthly simple returns, indexed by month-end date. The
        first row will be NaN (there is no prior month to compute a return
        from) and is dropped.
    """
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("`prices` must have a DatetimeIndex.")

    monthly_prices = prices.resample("ME").last()
    monthly_returns = monthly_prices.pct_change().dropna(how="all")
    return monthly_returns

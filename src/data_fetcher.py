"""
data_fetcher.py
================

Functions for downloading market price data and company fundamental data,
used as the raw inputs for factor computation in `factor_engine.py`.

Data sources:
    - **yfinance**: free historical price data (open/high/low/close/volume,
      adjusted for splits and dividends) and a snapshot of current
      fundamental metrics (via `Ticker.info` and `Ticker.financials` /
      `Ticker.balance_sheet` / `Ticker.cashflow`).
    - **pandas-datareader**: used as a secondary source for certain macro /
      reference series (e.g., risk-free rate proxies from FRED) that are
      useful for computing risk-adjusted metrics like the Sharpe ratio.

A note on data quality (read this before trusting results):
    Free data sources like `yfinance` are convenient but imperfect. Common
    issues include: delisted tickers returning empty data, fundamental data
    that is delayed or occasionally missing for certain fields, and
    point-in-time inconsistencies (e.g., `Ticker.info` reflects *current*
    shares outstanding, not historical). This module tries to handle these
    issues gracefully (skipping bad tickers with a warning rather than
    crashing), but for production research you would typically use a paid
    point-in-time database (e.g., Compustat, CRSP, Sharadar).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError as exc:  # pragma: no cover - import-time guard
    raise ImportError(
        "yfinance is required for data_fetcher.py. Install it with "
        "`pip install yfinance`."
    ) from exc

logger = logging.getLogger(__name__)


def fetch_price_data(
    tickers: Sequence[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    interval: str = "1d",
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Download historical price data for a list of tickers.

    Args:
        tickers: List of ticker symbols (e.g., ["AAPL", "MSFT", "GOOGL"]).
        start: Start date as "YYYY-MM-DD". If None, defaults to 5 years
            before today.
        end: End date as "YYYY-MM-DD". If None, defaults to today.
        interval: Data frequency understood by yfinance (e.g., "1d", "1wk",
            "1mo"). Defaults to daily.
        auto_adjust: If True (default), prices are adjusted for stock splits
            and dividends, so the returned "Close" series can be used
            directly to compute total returns. This is almost always what
            you want for factor research, since unadjusted prices would
            show fake "crashes" on split dates.

    Returns:
        A DataFrame of adjusted close prices with a DatetimeIndex and one
        column per successfully-downloaded ticker. Tickers that failed to
        download (e.g., delisted, typo'd) are omitted from the result and
        logged as warnings, but do NOT raise an exception — callers should
        check `result.columns` against their input `tickers` if they need
        to know which ones failed.

    Raises:
        ValueError: If `tickers` is empty, or if *no* tickers could be
            downloaded successfully.

    Example:
        >>> prices = fetch_price_data(["AAPL", "MSFT"], start="2023-01-01",
        ...                            end="2023-12-31")
        >>> prices.shape[1]  # number of tickers successfully fetched
        2
    """
    if not tickers:
        raise ValueError("`tickers` must be a non-empty sequence of ticker symbols.")

    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")
    if start is None:
        start = (datetime.today() - timedelta(days=5 * 365)).strftime("%Y-%m-%d")

    logger.info(
        "Fetching price data for %d ticker(s) from %s to %s (interval=%s)",
        len(tickers),
        start,
        end,
        interval,
    )

    raw = yf.download(
        tickers=list(tickers),
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    if raw.empty:
        raise ValueError(
            "No price data returned for any ticker. Check ticker symbols "
            "and date range."
        )

    prices = pd.DataFrame(index=raw.index)
    failed_tickers: List[str] = []

    if isinstance(raw.columns, pd.MultiIndex):
        for ticker in tickers:
            try:
                series = raw[(ticker, "Close")]
            except KeyError:
                failed_tickers.append(ticker)
                continue
            if series.dropna().empty:
                failed_tickers.append(ticker)
                continue
            prices[ticker] = series
    else:
        # yfinance collapses to a single-level column index when only one
        # ticker is requested.
        if "Close" not in raw.columns or raw["Close"].dropna().empty:
            failed_tickers.append(tickers[0])
        else:
            prices[tickers[0]] = raw["Close"]

    if failed_tickers:
        logger.warning(
            "Failed to download price data for %d ticker(s): %s. These "
            "tickers were excluded from the result.",
            len(failed_tickers),
            ", ".join(failed_tickers),
        )

    if prices.empty or prices.shape[1] == 0:
        raise ValueError(
            "No tickers could be successfully downloaded. Failed tickers: "
            f"{failed_tickers}"
        )

    prices = prices.sort_index()
    prices.index.name = "date"
    return prices


def fetch_fundamental_data(tickers: Sequence[str]) -> pd.DataFrame:
    """Fetch a snapshot of current fundamental metrics for a list of tickers.

    This pulls a curated set of fields from `yfinance`'s `Ticker.info`
    dictionary, which represents a **current snapshot** (not historical
    point-in-time data). This is suitable for `signal-quick`-style "what
    does this look like right now" checks, but using this data naively in a
    historical backtest would introduce **look-ahead bias** — the mistake of
    using information that was not actually available at the historical date
    being simulated. (For example, using *today's* shares outstanding to
    compute a market cap as of three years ago would be inconsistent if the
    company has since issued or repurchased shares.)

    Fields fetched (with the financial concept they represent):
        - `market_cap`: Market capitalization — total value of all
          outstanding shares (price x shares outstanding). Used as the
          denominator for "yield" style value factors.
        - `book_value`: Total book value of equity (assets minus
          liabilities, as reported on the balance sheet). Used for
          book-to-market.
        - `trailing_eps`: Trailing twelve-month earnings per share. Used for
          earnings yield.
        - `total_revenue`: Total revenue (sales) over the trailing twelve
          months. Used for gross profitability.
        - `gross_profits`: Gross profit (revenue minus cost of goods sold)
          over the trailing twelve months.
        - `total_assets`: Total assets, from the balance sheet.
        - `free_cashflow`: Free cash flow (operating cash flow minus capital
          expenditures) over the trailing twelve months.
        - `net_income`: Net income (the "bottom line" profit) over the
          trailing twelve months. Used for ROE.
        - `total_stockholder_equity`: Total shareholders' equity, from the
          balance sheet. Used as the denominator for ROE.
        - `beta`: yfinance's pre-computed beta estimate (sensitivity of the
          stock's returns to the overall market's returns), typically
          computed over a 3-5 year monthly window by the data provider.
        - `current_price`: The most recent trading price.

    Args:
        tickers: List of ticker symbols.

    Returns:
        A DataFrame indexed by ticker symbol, with one column per field
        listed above. Any field unavailable for a given ticker (common with
        free data) is set to NaN, and a debug-level log message is emitted.
        Tickers for which `Ticker.info` could not be retrieved at all (e.g.,
        invalid symbol) are dropped, with a warning logged.

    Raises:
        ValueError: If `tickers` is empty or if fundamental data could not
            be retrieved for *any* ticker.
    """
    if not tickers:
        raise ValueError("`tickers` must be a non-empty sequence of ticker symbols.")

    field_map = {
        "marketCap": "market_cap",
        "bookValue": "book_value_per_share",
        "trailingEps": "trailing_eps",
        "totalRevenue": "total_revenue",
        "grossProfits": "gross_profits",
        "totalCash": "total_cash",
        "freeCashflow": "free_cashflow",
        "netIncomeToCommon": "net_income",
        "returnOnEquity": "return_on_equity",
        "beta": "beta",
        "currentPrice": "current_price",
        "sharesOutstanding": "shares_outstanding",
        "totalDebt": "total_debt",
    }

    rows: Dict[str, Dict[str, float]] = {}
    failed_tickers: List[str] = []

    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
        except Exception as exc:  # noqa: BLE001 - network/library errors vary
            logger.warning("Could not fetch info for ticker '%s': %s", ticker, exc)
            failed_tickers.append(ticker)
            continue

        if not info or info.get("regularMarketPrice") is None and info.get(
            "currentPrice"
        ) is None:
            logger.warning(
                "Ticker '%s' returned empty/invalid info; excluding from results.",
                ticker,
            )
            failed_tickers.append(ticker)
            continue

        row: Dict[str, float] = {}
        for raw_field, friendly_name in field_map.items():
            value = info.get(raw_field, np.nan)
            if value is None:
                value = np.nan
                logger.debug(
                    "Field '%s' missing for ticker '%s'.", raw_field, ticker
                )
            row[friendly_name] = value

        # Derive total book equity (in dollars, not per-share) when possible,
        # since book_value from yfinance is typically reported per-share.
        bv_per_share = row.get("book_value_per_share", np.nan)
        shares = row.get("shares_outstanding", np.nan)
        if not np.isnan(bv_per_share) and not np.isnan(shares):
            row["total_book_equity"] = bv_per_share * shares
        else:
            row["total_book_equity"] = np.nan

        rows[ticker] = row

    if failed_tickers:
        logger.warning(
            "Failed to fetch fundamental data for %d ticker(s): %s.",
            len(failed_tickers),
            ", ".join(failed_tickers),
        )

    if not rows:
        raise ValueError(
            f"Could not retrieve fundamental data for any ticker. Failed: "
            f"{failed_tickers}"
        )

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "ticker"
    return df


def fetch_risk_free_rate(
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.Series:
    """Fetch a risk-free rate proxy series from FRED via pandas-datareader.

    The "risk-free rate" represents the theoretical return on an investment
    with zero risk (in practice, short-term US Treasury yields are used as a
    proxy). It's needed to compute "excess returns" (return minus risk-free
    rate), which is the correct numerator for a Sharpe ratio.

    This function fetches the 3-Month Treasury Bill secondary market rate
    (FRED series "TB3MS"), which is reported as an annualized percentage
    (e.g., 5.25 means 5.25% per year).

    Args:
        start: Start date as "YYYY-MM-DD". Defaults to 5 years before today.
        end: End date as "YYYY-MM-DD". Defaults to today.

    Returns:
        A Series of annualized risk-free rates in **decimal** form (e.g.,
        0.0525 for 5.25%), indexed by date (monthly frequency, as reported
        by FRED for this series).

    Raises:
        RuntimeError: If pandas-datareader fails to fetch the series (e.g.,
            no internet connection, FRED API change). The original exception
            is chained for debugging.

    Note:
        If you don't have internet access or FRED is unavailable, you can
        substitute a constant assumption (e.g., 0.02 for a long-run average
        ~2% risk-free rate) when calling Sharpe ratio functions in
        `backtest.py` — they accept a `risk_free_rate` parameter for exactly
        this reason.
    """
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")
    if start is None:
        start = (datetime.today() - timedelta(days=5 * 365)).strftime("%Y-%m-%d")

    try:
        import pandas_datareader.data as web

        series = web.DataReader("TB3MS", "fred", start, end)
    except Exception as exc:  # noqa: BLE001 - network/library errors vary
        raise RuntimeError(
            "Failed to fetch risk-free rate series 'TB3MS' from FRED via "
            "pandas-datareader. If you don't have internet access, pass an "
            "explicit `risk_free_rate` to backtest functions instead."
        ) from exc

    rf = series["TB3MS"] / 100.0  # convert percent to decimal
    rf.name = "risk_free_rate"
    return rf


def compute_returns(prices: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    """Compute simple percentage returns from a price DataFrame.

    A "simple return" over `periods` time steps is:

        return_t = (price_t - price_{t-periods}) / price_{t-periods}

    Args:
        prices: DataFrame of prices, indexed by date, one column per ticker.
        periods: Number of periods over which to compute the return (e.g.,
            1 for period-over-period returns, 21 for an approximate
            "1-month" return on daily data, 252 for an approximate
            "1-year" return on daily data).

    Returns:
        DataFrame of returns with the same columns as `prices`. The first
        `periods` rows will be NaN (no prior data to compute a return from).
    """
    if prices.isna().all().all():
        raise ValueError("`prices` contains no valid (non-NaN) data.")

    returns = prices.pct_change(periods=periods)
    return returns


def handle_missing_data(
    df: pd.DataFrame,
    method: str = "ffill",
    max_consecutive_gaps: int = 5,
) -> pd.DataFrame:
    """Fill or flag missing values in a price/fundamental DataFrame.

    Real-world market data has gaps: market holidays, temporary trading
    halts, or vendor outages can all produce missing rows or NaN values.
    This function provides a couple of standard, defensible ways to handle
    them.

    Args:
        df: Input DataFrame, typically prices indexed by date.
        method: One of:
            - "ffill": forward-fill missing values (carry the last known
              price forward). This is the standard approach for short gaps
              in price data — it implicitly assumes "no trading happened, so
              the price didn't change," which is a reasonable approximation
              for gaps of a few days.
            - "drop": drop any row that contains at least one NaN. Useful
              when you need a fully-aligned panel (e.g., for computing
              cross-sectional statistics where every column must have a
              value).
            - "interpolate": linearly interpolate between the last known and
              next known values. Can be useful for fundamental data that
              changes smoothly, but should be used with caution for prices
              (it can leak future information into past rows).
        max_consecutive_gaps: For "ffill", if a column has more than this
            many consecutive missing values, those values are left as NaN
            rather than forward-filled — a long gap usually indicates the
            stock was delisted or halted for an extended period, and
            blindly forward-filling could create a misleadingly flat return
            series.

    Returns:
        A cleaned copy of `df`.

    Raises:
        ValueError: If `method` is not one of "ffill", "drop", "interpolate".
    """
    if method not in {"ffill", "drop", "interpolate"}:
        raise ValueError(
            f"Unknown method '{method}'. Must be 'ffill', 'drop', or "
            "'interpolate'."
        )

    result = df.copy()

    if method == "drop":
        n_before = len(result)
        result = result.dropna(how="any")
        logger.info(
            "handle_missing_data(drop): dropped %d/%d rows containing NaNs.",
            n_before - len(result),
            n_before,
        )
        return result

    if method == "interpolate":
        return result.interpolate(method="linear", limit_direction="forward")

    # method == "ffill"
    filled = result.ffill(limit=max_consecutive_gaps)
    n_remaining_nans = filled.isna().sum().sum()
    if n_remaining_nans > 0:
        logger.info(
            "handle_missing_data(ffill): %d NaN value(s) remain after "
            "forward-filling gaps of up to %d periods (likely longer gaps "
            "such as delistings).",
            n_remaining_nans,
            max_consecutive_gaps,
        )
    return filled

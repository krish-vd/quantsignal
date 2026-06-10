---
name: signal-run
description: Run a full empirical factor backtest — fetch data, compute a factor (standard or custom), run a quintile-sort backtest, and report IC, IR, Sharpe, max drawdown, and a Q1-vs-Q5 cumulative return chart. Use this when the user wants real, computed numbers for a factor, not just a qualitative research note.
---

# signal-run — Full Factor Backtest Pipeline

This skill orchestrates the three core scripts in `src/` to produce a real, computed
backtest of an alpha factor. It is the empirical counterpart to the QuantSignal agent's
Step 5 recommendation ("run a backtest before drawing conclusions").

## When to use this skill

- The user has a formalized factor (from a QuantSignal research note, or their own
  formula) and wants to know **how it actually performed historically**.
- The user asks "does this work?", "what's the Sharpe ratio?", "show me a backtest",
  "what's the IC of this factor?".

## Inputs needed from the user (ask if not provided)

1. **Universe**: a list of tickers, or a well-known index proxy (e.g., "S&P 500
   constituents" — note that `yfinance` does not provide index constituent lists, so for
   a quick run, ask the user for an explicit ticker list, e.g. 20-50 large-cap names, or
   point them to a CSV they can supply).
2. **Factor**: one of the standard factors implemented in `factor_engine.py`
   (`momentum_12_1`, `book_to_market`, `earnings_yield`, `fcf_yield`, `roe`,
   `gross_profitability`, `accruals`, `realized_volatility`, `beta`) — or a custom
   formula expressed as a pandas-evaluable expression over named columns.
3. **Date range**: start and end dates for the backtest (default: 5 years back from
   today if unspecified — note this default explicitly to the user).
4. **Rebalancing frequency**: default monthly.
5. **Number of quintiles**: default 5 (a "quintile" is one of five equal-sized groups —
   Q1 is the lowest 20% of stocks by factor value, Q5 is the highest 20%).

## Pipeline steps

1. **Fetch data** using `src/data_fetcher.py`:
   ```bash
   python -c "
   from src.data_fetcher import fetch_price_data, fetch_fundamental_data
   prices = fetch_price_data(['AAPL','MSFT', ...], start='2019-01-01', end='2024-01-01')
   fundamentals = fetch_fundamental_data(['AAPL','MSFT', ...])
   "
   ```
   Handle and report any tickers that failed to download (delisted, typos, etc.) — do
   not silently drop them without telling the user.

2. **Compute the factor** using `src/factor_engine.py`:
   - For standard factors, call the corresponding function (e.g.,
     `compute_momentum_12_1(prices)`).
   - For custom factors, call `compute_custom_factor(data, formula)` where `formula` is
     a string expression like `"earnings / market_cap"` evaluated against the columns
     of `data`.

3. **Run the backtest** using `src/backtest.py::run_quintile_backtest()`:
   - This produces: monthly returns per quintile (Q1–Q5), the long-short Q5−Q1 spread,
     **IC** (Information Coefficient — the rank correlation between the factor value at
     time t and the stock's forward return over the next period; a higher absolute IC
     means the factor is better at ranking which stocks will do well), **IR**
     (Information Ratio — the mean IC divided by the standard deviation of IC over time;
     a measure of how *consistently* the factor's ranking ability holds up, analogous to
     a Sharpe ratio for the signal itself), the **Sharpe ratio** of the Q5−Q1 spread (a
     measure of risk-adjusted return: average return divided by the volatility of
     returns, typically annualized), and **max drawdown** (the largest peak-to-trough
     decline in cumulative returns — a measure of the worst-case loss an investor in
     this strategy would have experienced).

4. **Plot** Q1 vs Q5 (and optionally the Q5−Q1 spread) cumulative return curves using
   `backtest.py::plot_quintile_returns()`, saving to a PNG the user can view.

5. **Report results** in a table:

   | Metric | Value |
   |---|---|
   | IC (mean) | ... |
   | IC (std) | ... |
   | IR | ... |
   | Q5−Q1 annualized return | ... |
   | Q5−Q1 Sharpe ratio | ... |
   | Q5−Q1 max drawdown | ... |
   | Q1 annualized return | ... |
   | Q5 annualized return | ... |

   **Always state the exact universe, date range, and rebalancing frequency used**, and
   remind the user that:
   - Results are **gross of transaction costs** unless otherwise noted.
   - A small universe (e.g., 20-50 tickers) produces noisy, low-confidence statistics —
     this is exploratory, not production-grade.
   - Past performance (even real, computed past performance) does not guarantee future
     results.

## Error handling expectations

- If `yfinance` fails to return data for a ticker (delisted, rate-limited, etc.),
  `data_fetcher.py` should raise a clear, catchable error or return a flag — surface
  this to the user rather than silently producing a backtest on a subset.
- If the factor produces too many NaN values (e.g., missing fundamentals), report the
  percentage of the universe excluded at each rebalance date.
- If the date range requested predates available data for a ticker, clip to the
  available range and tell the user.

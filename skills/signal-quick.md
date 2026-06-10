---
name: signal-quick
description: Quickly compute one or more factor values for a small list of tickers right now (no full backtest, no historical IC/IR). Use for fast sanity checks like "what's the current 12-1 momentum and ROE for these 5 stocks?"
---

# signal-quick — Fast Single-Point Factor Check

This skill is a lightweight alternative to `signal-run` for when the user wants a quick,
**current-snapshot** view of one or more factors across a handful of tickers — not a
historical backtest.

## When to use this skill

- "What's the current momentum score for AAPL, MSFT, and NVDA?"
- "Compute book-to-market and ROE for these 5 tickers."
- "Sanity-check this custom factor formula on a couple of names before I run a full
  backtest."

This is the right tool when the user wants a **number right now**, not a historical
performance evaluation. If the user starts asking about Sharpe ratios, IC, drawdowns, or
"how would this have performed," redirect to `signal-run`.

## Steps

1. **Fetch a short window of data** using `src/data_fetcher.py` — for momentum/vol/beta
   factors, fetch enough history to compute the lookback (e.g., 13 months of price data
   for 12-1 momentum); for value/quality factors, fetch the latest available
   fundamentals via `fetch_fundamental_data()`.

2. **Compute the requested factor(s)** using `src/factor_engine.py`. For custom
   formulas, use `compute_custom_factor()` and clearly echo back the exact formula used
   so the user can verify it matches their intent.

3. **Apply cleaning** as appropriate using `src/utils.py`:
   - `winsorize()` to cap extreme outliers at the 1st/99th percentile (only meaningful
     with a larger cross-section — note this if the ticker list is very small).
   - `zscore_normalize()` if the user wants relative (cross-sectional) comparison rather
     than raw values.

4. **Report a simple table**:

   | Ticker | <Factor 1> | <Factor 2> | ... |
   |---|---|---|---|
   | AAPL | ... | ... | ... |
   | MSFT | ... | ... | ... |

   Include the **as-of date** for the data used, and flag any tickers with missing or
   stale data (e.g., fundamentals more than ~6 months old, which is common for
   `yfinance`'s free fundamental data).

## Caveats to always mention

- A handful of tickers is **not a meaningful cross-section** for ranking purposes — a
  factor "rank" only makes sense relative to a broader universe. Frame results as raw
  values, not investment recommendations.
- `yfinance` fundamental data can be delayed, restated, or occasionally missing for
  certain tickers/fields — flag any `NaN` results rather than silently omitting them.
- This is a **point-in-time snapshot using current data**, which may differ from what
  was knowable historically (e.g., current shares outstanding vs. historical) — not
  suitable for backtesting conclusions.

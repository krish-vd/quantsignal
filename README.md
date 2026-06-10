# QuantSignal

**QuantSignal** is a Claude plugin that turns a plain-English market
hypothesis into a rigorous, citation-backed alpha factor research note — and
optionally backtests it on real market data.

It pairs a senior-quant-researcher **agent** (`agents/quantsignal.md`) with a
small Python research library (`src/`) for fetching data, computing factors,
running quintile-sort backtests, and scoring factor quality.

---

## What it does

Give the agent an idea like:

> "I think companies that just announced a large stock buyback tend to
> outperform over the next year."

and it runs a five-step research pipeline:

1. **Factor Formalization** — turns the idea into a precise mathematical
   definition (inputs, formula, universe, rebalancing, edge cases).
2. **Factor Classification** — places it into the standard factor families
   (Value, Momentum, Quality, Low Volatility, Size, Alternative) with
   justification.
3. **Economic Rationale** — explains *why* the pattern should exist
   (behavioral bias or risk premium), *who* is on the other side of the
   trade, and *why it hasn't been arbitraged away*.
4. **Crowding & Novelty Assessment** — searches SSRN, ArXiv, Google Scholar,
   and the broader web for prior work, and issues a 🟢 GREEN / 🟡 YELLOW /
   🔴 RED verdict on how crowded the idea is.
5. **Quality Scorecard** — scores the factor 1-5 on five dimensions
   (Economic Strength, Novelty, Data Accessibility, Implementability, Decay
   Resistance) and recommends a next step.

The agent **never fabricates backtest statistics** and **always cites
sources**. When you want real numbers, it hands off to the `signal-run`
skill, which actually fetches data and computes results.

### Slash commands

Once the plugin is loaded, three commands are available:

| Command | What it does |
|---|---|
| `/quantsignal [hypothesis]` | Runs the full 5-step research pipeline above on a plain-English idea. |
| `/signal-run [factor + universe + dates]` | Runs a real quintile-sort backtest (IC, IR, Sharpe, max drawdown, plot) using `src/`. |
| `/signal-quick [factor + tickers]` | Computes a factor's current value for a few tickers, no backtest. |

Example:

```
/quantsignal companies that just announced large stock buybacks tend to outperform
```

---

## Project structure

```
quantsignal/
├── .claude-plugin/
│   └── plugin.json          # Plugin manifest
├── agents/
│   └── quantsignal.md        # The senior quant research agent
├── skills/
│   ├── signal-run.md          # Full empirical backtest pipeline
│   └── signal-quick.md        # Fast single-snapshot factor check
├── src/
│   ├── data_fetcher.py        # Price & fundamental data via yfinance / pandas-datareader
│   ├── factor_engine.py       # Standard + custom factor computation
│   ├── backtest.py             # Quintile-sort backtest, IC, IR, Sharpe, max drawdown
│   ├── scorer.py                # Five-dimension quality scorecard
│   └── utils.py                  # Winsorization, z-scoring, universe filters
├── requirements.txt
└── README.md
```

---

## Installation

```bash
cd quantsignal
python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Then install the plugin into Claude (refer to your Claude Code / plugin
host's documentation for loading a local plugin directory).

---

## Quick start (Python library, standalone)

The `src/` modules are fully usable on their own, independent of the agent:

```python
from src.data_fetcher import fetch_price_data, fetch_fundamental_data
from src.factor_engine import compute_momentum_12_1, compute_book_to_market
from src.backtest import run_quintile_backtest, summarize_backtest, plot_quintile_returns
from src.utils import to_monthly_returns

tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "JPM", "XOM", "JNJ", "PG", "KO"]

# 1. Fetch ~6 years of daily prices (need >253 days for 12-1 momentum)
prices = fetch_price_data(tickers, start="2018-01-01", end="2024-01-01")

# 2. Compute the 12-1 month momentum factor
momentum = compute_momentum_12_1(prices)

# 3. Resample to monthly and align factor values with forward returns
monthly_returns = to_monthly_returns(prices)
monthly_momentum = momentum.resample("ME").last()
forward_returns = monthly_returns.shift(-1)  # next month's return

# 4. Run the quintile-sort backtest
result = run_quintile_backtest(monthly_momentum, forward_returns, n_quintiles=5)

print(summarize_backtest(result))
plot_quintile_returns(result, save_path="momentum_backtest.png")
```

### Custom factors

```python
from src.factor_engine import compute_custom_factor
from src.data_fetcher import fetch_fundamental_data

fundamentals = fetch_fundamental_data(tickers)
custom_factor = compute_custom_factor(
    fundamentals, "net_income / market_cap"  # earnings yield, expressed manually
)
```

### Scoring a factor

```python
from src.scorer import score_factor

card = score_factor(
    factor_name="12-1 Month Momentum",
    economic_strength=4,
    novelty=2,
    data_accessibility=5,
    implementability=4,
    decay_resistance=3,
    rationales={
        "economic_strength": "Well-documented underreaction to news (Jegadeesh & Titman, 1993).",
        "novelty": "Extremely well-published; heavily used in commercial smart-beta products.",
        "data_accessibility": "Only requires price history, freely available via yfinance.",
        "implementability": "Moderate turnover, works well on liquid large/mid caps.",
        "decay_resistance": "Persisted for 30+ years but with long, painful drawdowns (e.g., 2009 momentum crash).",
    },
)
print(card.to_markdown())
```

---

## Important caveats

- **Free data has limits.** `yfinance` fundamental data is a *current
  snapshot*, not point-in-time historical data. Using it naively in a
  backtest can introduce look-ahead bias. See the docstrings in
  `src/data_fetcher.py` for details.
- **Small universes produce noisy statistics.** A backtest on 10-50 tickers
  is exploratory, not production-grade. IC, IR, and Sharpe ratios computed
  on small samples should be treated as directional, not definitive.
- **The agent does not invent numbers.** Any specific Sharpe ratio, IC,
  return, or drawdown figure you see should come from an actual executed run
  of `src/backtest.py` — not from the agent's narrative text.
- **This is a research tool, not investment advice.** Nothing produced by
  QuantSignal constitutes a recommendation to buy or sell any security.

---

## License

MIT

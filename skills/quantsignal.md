---
name: quantsignal
description: Run the full QuantSignal five-step alpha factor research pipeline on a plain-English market hypothesis. Produces a research note covering factor formalization, classification, economic rationale, crowding/novelty assessment, and a quality scorecard.
command: quantsignal
---

When the user types `/quantsignal [hypothesis]`, invoke the `quantsignal` agent on the
provided hypothesis.

## Behavior

1. Extract the hypothesis text following `/quantsignal` (e.g.,
   `/quantsignal companies with rising employee satisfaction outperform` →
   hypothesis is "companies with rising employee satisfaction outperform").
2. If no hypothesis is provided, ask: "What market hypothesis would you like
   me to research? Usage: `/quantsignal [your idea in plain English]`"
3. Invoke the `quantsignal` agent with the following prompt:

```
Research this hypothesis as a potential alpha factor:

"[HYPOTHESIS]"

Run the full five-step QuantSignal research pipeline:
1. Factor Formalization — precise mathematical definition, required inputs,
   universe, and rebalancing.
2. Factor Classification — map to Value / Quality / Momentum / Low-Vol /
   Size / Alternative families with justification.
3. Economic Rationale — behavioral/risk-based explanation, who is on the
   other side of the trade, and why it hasn't been arbitraged away.
4. Crowding & Novelty Assessment — use WebSearch against SSRN, ArXiv, Google
   Scholar, and practitioner sources; issue a GREEN/YELLOW/RED verdict with
   citations.
5. Five-Dimension Quality Scorecard — score Economic Strength, Novelty, Data
   Accessibility, Implementability, and Decay Resistance (1-5 each), with an
   overall verdict and a concrete next step.

Follow the output format defined in the quantsignal agent's instructions
exactly. Do not fabricate any backtested statistics — if the user wants real
numbers, point them to `/signal-run` or `/signal-quick`.
```

4. Stream the agent's full response to the user.

## Usage

```
/quantsignal companies that just announced large stock buybacks tend to outperform
/quantsignal stocks with declining short interest tend to outperform over the next month
/quantsignal firms with high R&D spending relative to market cap outperform over 3-5 years
```

## Notes

- This command produces a *qualitative* research note — no backtest is run.
- For real, computed numbers on a formalized factor, follow up with
  `/signal-run`.
- For a fast current-snapshot factor check on a few tickers, use
  `/signal-quick`.

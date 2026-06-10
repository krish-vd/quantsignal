"""
scorer.py
=========

Implements the **five-dimension factor quality scorecard** described in Step
5 of the QuantSignal research pipeline (see `agents/quantsignal.md`).

The scorecard is a structured way to summarize a *qualitative* judgment
about a factor's overall promise, across five independent dimensions:

    1. Economic Strength  — how compelling is the underlying economic story?
    2. Novelty             — how crowded / well-published is this idea?
    3. Data Accessibility  — how easy is it to get the required data?
    4. Implementability    — how practical is it to trade?
    5. Decay Resistance    — how likely is the edge to persist over time?

Each dimension is scored 1 (worst) to 5 (best). This module does **not**
attempt to compute these scores from data automatically — they require
human/LLM judgment based on research (Steps 1-4 of the pipeline). Instead,
this module provides:

    - A typed, validated container (`FactorScorecard`) for recording scores
      and rationales.
    - A `score_factor()` constructor function that validates inputs and
      computes the aggregate verdict.
    - A `to_markdown()` method for rendering the scorecard as a markdown
      table matching the format used in `agents/quantsignal.md`.

This keeps the *numbers* (which an agent or user provides based on their
research) cleanly separated from the *presentation and aggregation logic*
(which should be consistent and bug-free).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


# The five scorecard dimensions, in the canonical order used throughout
# QuantSignal's research notes.
SCORECARD_DIMENSIONS = (
    "economic_strength",
    "novelty",
    "data_accessibility",
    "implementability",
    "decay_resistance",
)

# Human-readable labels for each dimension, used in markdown output.
_DIMENSION_LABELS: Dict[str, str] = {
    "economic_strength": "Economic Strength",
    "novelty": "Novelty",
    "data_accessibility": "Data Accessibility",
    "implementability": "Implementability",
    "decay_resistance": "Decay Resistance",
}

MIN_SCORE = 1
MAX_SCORE = 5


@dataclass
class FactorScorecard:
    """A five-dimension quality scorecard for a single alpha factor.

    Attributes:
        factor_name: Short, human-readable name for the factor (e.g.,
            "12-1 Month Price Momentum").
        economic_strength: Score 1-5 for how compelling the economic
            rationale is (see Step 3 of the research pipeline).
        novelty: Score 1-5 derived from the crowding/novelty assessment
            (Step 4): 5 for GREEN, 3 for YELLOW, 1 for RED is a reasonable
            default mapping, but the score can be adjusted up/down within
            that band based on the strength of the search evidence.
        data_accessibility: Score 1-5 for how easily the required data can
            be obtained (5 = freely available via yfinance/pandas-datareader,
            1 = requires expensive proprietary data).
        implementability: Score 1-5 for how practical the factor is to
            actually trade (turnover, liquidity, complexity).
        decay_resistance: Score 1-5 for how likely the factor's edge is to
            persist over time and across regimes.
        rationales: Dict mapping each dimension name to a short text
            justification for its score. All five dimensions must have a
            non-empty rationale.

    Raises:
        ValueError: (raised by `score_factor`, not `__init__` directly,
            though dataclass validation also occurs via `__post_init__`) if
            any score is outside [1, 5], or if any dimension is missing a
            rationale.
    """

    factor_name: str
    economic_strength: int
    novelty: int
    data_accessibility: int
    implementability: int
    decay_resistance: int
    rationales: Dict[str, str]

    def __post_init__(self) -> None:
        for dim in SCORECARD_DIMENSIONS:
            score = getattr(self, dim)
            if not isinstance(score, int):
                raise TypeError(
                    f"Score for '{dim}' must be an int, got {type(score).__name__}."
                )
            if not (MIN_SCORE <= score <= MAX_SCORE):
                raise ValueError(
                    f"Score for '{dim}' must be between {MIN_SCORE} and "
                    f"{MAX_SCORE} (inclusive), got {score}."
                )

        missing_rationales = [
            dim for dim in SCORECARD_DIMENSIONS
            if not self.rationales.get(dim, "").strip()
        ]
        if missing_rationales:
            raise ValueError(
                f"Missing or empty rationale for dimension(s): "
                f"{missing_rationales}. Every dimension must have a "
                "non-empty justification."
            )

    @property
    def total_score(self) -> int:
        """Sum of all five dimension scores (range: 5-25)."""
        return sum(getattr(self, dim) for dim in SCORECARD_DIMENSIONS)

    @property
    def max_possible_score(self) -> int:
        """Maximum possible total score (always 25, for 5 dimensions x 5 points)."""
        return MAX_SCORE * len(SCORECARD_DIMENSIONS)

    @property
    def verdict(self) -> str:
        """Overall qualitative verdict derived from `total_score`.

        Bands (out of 25):
            - 20-25: "Strong candidate — proceed to backtest"
            - 13-19: "Mixed — backtest with modified construction / narrower universe"
            - 5-12:  "Weak — unlikely to be worth the engineering effort"
        """
        total = self.total_score
        if total >= 20:
            return "Strong candidate — proceed to backtest"
        if total >= 13:
            return "Mixed — backtest with modified construction / narrower universe"
        return "Weak — unlikely to be worth the engineering effort"

    def as_dict(self) -> Dict[str, object]:
        """Return the scorecard as a plain dictionary, including derived fields.

        Returns:
            A dict with keys for `factor_name`, each dimension's score and
            rationale, `total_score`, `max_possible_score`, and `verdict`.
        """
        result: Dict[str, object] = {"factor_name": self.factor_name}
        for dim in SCORECARD_DIMENSIONS:
            result[dim] = getattr(self, dim)
            result[f"{dim}_rationale"] = self.rationales[dim]
        result["total_score"] = self.total_score
        result["max_possible_score"] = self.max_possible_score
        result["verdict"] = self.verdict
        return result

    def to_markdown(self) -> str:
        """Render the scorecard as a markdown table plus verdict line.

        The output format matches the "Quality Scorecard" section of the
        research note format defined in `agents/quantsignal.md`, so it can
        be inserted directly into a research note.

        Returns:
            A markdown-formatted string.
        """
        lines = [
            f"### Quality Scorecard: {self.factor_name}",
            "",
            "| Dimension | Score (1-5) | Rationale |",
            "|---|---|---|",
        ]
        for dim in SCORECARD_DIMENSIONS:
            label = _DIMENSION_LABELS[dim]
            score = getattr(self, dim)
            rationale = self.rationales[dim].replace("\n", " ").strip()
            lines.append(f"| {label} | {score} | {rationale} |")

        lines.append("")
        lines.append(
            f"**Overall: {self.total_score}/{self.max_possible_score} — {self.verdict}**"
        )
        return "\n".join(lines)


def score_factor(
    factor_name: str,
    economic_strength: int,
    novelty: int,
    data_accessibility: int,
    implementability: int,
    decay_resistance: int,
    rationales: Dict[str, str],
) -> FactorScorecard:
    """Construct and validate a `FactorScorecard`.

    This is a thin convenience wrapper around the `FactorScorecard`
    constructor that exists primarily to give this module a clear, callable
    "entry point" matching the function name referenced in
    `agents/quantsignal.md` (Step 5: "the scorer's structured output").

    Args:
        factor_name: Short, human-readable name for the factor.
        economic_strength: Score 1-5 (see `FactorScorecard.economic_strength`).
        novelty: Score 1-5 (see `FactorScorecard.novelty`).
        data_accessibility: Score 1-5 (see `FactorScorecard.data_accessibility`).
        implementability: Score 1-5 (see `FactorScorecard.implementability`).
        decay_resistance: Score 1-5 (see `FactorScorecard.decay_resistance`).
        rationales: Dict mapping each of the five dimension names (see
            `SCORECARD_DIMENSIONS`) to a short text justification. All five
            keys are required and must map to non-empty strings.

    Returns:
        A validated `FactorScorecard` instance.

    Raises:
        ValueError: If any score is out of the [1, 5] range, or if
            `rationales` is missing an entry (or has an empty entry) for any
            of the five dimensions.
        TypeError: If any score is not an integer.

    Example:
        >>> card = score_factor(
        ...     factor_name="12-1 Month Momentum",
        ...     economic_strength=4,
        ...     novelty=2,
        ...     data_accessibility=5,
        ...     implementability=4,
        ...     decay_resistance=3,
        ...     rationales={
        ...         "economic_strength": "Well-documented underreaction to news.",
        ...         "novelty": "Extremely well-published since Jegadeesh & Titman (1993).",
        ...         "data_accessibility": "Only requires price history, freely available.",
        ...         "implementability": "Moderate turnover, works on liquid large caps.",
        ...         "decay_resistance": "Persisted for 30+ years but with long drawdowns (e.g. 2009).",
        ...     },
        ... )
        >>> card.total_score
        18
        >>> card.verdict
        'Mixed — backtest with modified construction / narrower universe'
    """
    return FactorScorecard(
        factor_name=factor_name,
        economic_strength=economic_strength,
        novelty=novelty,
        data_accessibility=data_accessibility,
        implementability=implementability,
        decay_resistance=decay_resistance,
        rationales=rationales,
    )


def novelty_score_from_verdict(verdict: str) -> int:
    """Map a Step-4 crowding/novelty verdict (GREEN/YELLOW/RED) to a default score.

    This provides a reasonable starting point for the `novelty` dimension
    score, which the agent or user can then adjust up or down by 1 point
    based on the strength of the supporting evidence (e.g., a YELLOW verdict
    backed by only one marginally-related paper might merit a 4 instead of
    the default 3).

    Args:
        verdict: One of "GREEN", "YELLOW", or "RED" (case-insensitive; emoji
            prefixes like "🟢 GREEN" are also accepted and stripped).

    Returns:
        Default novelty score: 5 for GREEN, 3 for YELLOW, 1 for RED.

    Raises:
        ValueError: If `verdict` does not contain one of "GREEN", "YELLOW",
            or "RED" (case-insensitive).
    """
    normalized = verdict.strip().upper()

    if "GREEN" in normalized:
        return 5
    if "YELLOW" in normalized:
        return 3
    if "RED" in normalized:
        return 1

    raise ValueError(
        f"Could not interpret verdict '{verdict}'. Expected it to contain "
        "'GREEN', 'YELLOW', or 'RED'."
    )

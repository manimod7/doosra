"""Odds maths: implied probability, de-vig, edge, Kelly.

All odds are decimal. A "market" is a list of decimal odds for mutually
exclusive, exhaustive outcomes (e.g. [home, away] for a cricket match winner).
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


def implied_prob(odds: float) -> float:
    if odds <= 1.0 or not isfinite(odds):
        raise ValueError(f"decimal odds must be > 1, got {odds}")
    return 1.0 / odds


def overround(odds: list[float]) -> float:
    """Sum of implied probabilities minus 1 (the bookmaker margin)."""
    return sum(implied_prob(o) for o in odds) - 1.0


def devig_proportional(odds: list[float]) -> list[float]:
    """Remove margin by scaling implied probabilities to sum to 1."""
    raw = [implied_prob(o) for o in odds]
    total = sum(raw)
    return [p / total for p in raw]


def devig_power(odds: list[float], tol: float = 1e-10) -> list[float]:
    """Power method: find k so that sum(p_i ** k) == 1.

    Shifts more margin onto longshots than the proportional method, which
    is usually closer to sharp-book reality on two-way markets.
    """
    raw = [implied_prob(o) for o in odds]
    lo, hi = 1.0, 20.0
    for _ in range(200):
        k = (lo + hi) / 2
        s = sum(p**k for p in raw)
        if abs(s - 1.0) < tol:
            break
        if s > 1.0:
            lo = k
        else:
            hi = k
    return [p**k for p in raw]


def fair_odds(prob: float) -> float:
    if not 0.0 < prob < 1.0:
        raise ValueError("prob must be in (0, 1)")
    return 1.0 / prob


def edge(model_prob: float, odds: float) -> float:
    """Expected profit per unit staked: p * odds - 1."""
    return model_prob * odds - 1.0


def kelly_fraction(model_prob: float, odds: float, fraction: float = 0.25) -> float:
    """Fractional Kelly stake as a share of bankroll (0 if no edge)."""
    b = odds - 1.0
    if b <= 0:
        return 0.0
    f = (model_prob * b - (1.0 - model_prob)) / b
    return max(0.0, f) * fraction


@dataclass(frozen=True)
class Verdict:
    label: str  # BET | SKIP | TRAP
    edge: float
    kelly: float
    reason: str


def verdict(
    model_prob: float,
    odds: float,
    sharp_prob: float | None = None,
    min_edge: float = 0.03,
    kelly_cap: float = 0.05,
) -> Verdict:
    """Decide BET / SKIP / TRAP for one outcome.

    TRAP: the model likes the price but the sharp (de-vigged) market disagrees
    by a wide margin, which usually means the model is missing information
    (injury, XI, pitch) rather than the market being wrong.
    """
    e = edge(model_prob, odds)
    k = min(kelly_fraction(model_prob, odds), kelly_cap)
    if e < min_edge:
        return Verdict("SKIP", e, 0.0, f"An edge of {e:+.1%} is below the {min_edge:.0%} Doosra needs before backing anything.")
    if sharp_prob is not None and model_prob - sharp_prob > 0.10:
        return Verdict(
            "TRAP", e, 0.0,
            f"Your {model_prob:.1%} is far above the sharp market's {sharp_prob:.1%}. A gap that big "
            "usually means something is missing, such as team news.",
        )
    return Verdict("BET", e, k, f"An edge of {e:+.1%}. A quarter-Kelly stake is {k:.2%} of bankroll.")


def clv(bet_odds: float, closing_fair_prob: float) -> float:
    """Closing line value: bet price vs closing de-vigged fair price.

    Positive means you beat the close. Expressed as bet_odds * closing_prob - 1.
    """
    return bet_odds * closing_fair_prob - 1.0

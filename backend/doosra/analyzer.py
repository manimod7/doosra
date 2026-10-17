"""Turn a market comparison (+ optional model probability) into a bet verdict.

Honest by construction:
- With no model probability the analyzer gives a market view only and says so;
  it never invents a verdict from the market alone.
- Every verdict carries the reasoning and the stated limits of the evidence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .markets import BookQuote, Comparison, compare
from .odds import clv, edge, fair_odds, kelly_fraction, verdict


@dataclass(frozen=True)
class OutcomeAnalysis:
    outcome: str
    best_book: str
    best_odds: float
    consensus_prob: float
    sharp_prob: float | None
    model_prob: float | None
    fair_odds_consensus: float
    edge_vs_best: float | None
    label: str  # BET | SKIP | TRAP | NO MODEL
    stake_fraction: float
    reasons: tuple[str, ...]


def analyze(
    outcomes: list[str],
    quotes: list[BookQuote],
    model_probs: list[float] | None = None,
    bankroll: float | None = None,
) -> dict:
    comp: Comparison = compare(outcomes, quotes)
    n = len(outcomes)
    if model_probs is not None:
        if len(model_probs) != n:
            raise ValueError("model_probs must match number of outcomes")
        if abs(sum(model_probs) - 1.0) > 1e-6 or any(not 0 < p < 1 for p in model_probs):
            raise ValueError("model_probs must be in (0,1) and sum to 1")

    results: list[OutcomeAnalysis] = []
    for i, name in enumerate(outcomes):
        best = comp.best[i]
        cons = comp.consensus_probs[i]
        sharp = comp.sharp_probs[i] if comp.sharp_probs else None
        reasons: list[str] = []
        if model_probs is None:
            label, stake, e = "NO MODEL", 0.0, None
            reasons.append("no model probability supplied: showing market view only")
        else:
            mp = model_probs[i]
            e = edge(mp, best.odds)
            v = verdict(mp, best.odds, sharp_prob=sharp)
            label, stake = v.label, v.kelly
            reasons.append(v.reason)
            if sharp is None:
                reasons.append("no sharp book (Pinnacle/Betfair) in feed: edge is judged against soft prices only")
            if comp.best_price_margin < 0:
                reasons.append("best prices across books imply an arbitrage; check the quotes are live")
        results.append(
            OutcomeAnalysis(
                outcome=name, best_book=best.book, best_odds=best.odds,
                consensus_prob=cons, sharp_prob=sharp,
                model_prob=model_probs[i] if model_probs else None,
                fair_odds_consensus=fair_odds(cons),
                edge_vs_best=e, label=label, stake_fraction=stake,
                reasons=tuple(reasons),
            )
        )

    out = {
        "outcomes": list(outcomes),
        "sharp_book": comp.sharp_book,
        "best_price_margin": comp.best_price_margin,
        "books": [
            {
                "book": b.book, "odds": list(b.odds), "margin": b.margin,
                "fair_probs": list(b.fair_probs), "is_sharp": b.is_sharp,
            }
            for b in comp.books
        ],
        "analysis": [asdict(r) for r in results],
        "limits": [
            "Probabilities come from the model you supply or the market, never from a language model.",
            "Odds are only as fresh as the feed; re-check the price at the bookmaker before acting.",
        ],
    }
    if bankroll is not None:
        for row in out["analysis"]:
            row["stake_amount"] = round(bankroll * row["stake_fraction"], 2)
    return out


def closing_line_value(bet_odds: float, closing_quotes: list[BookQuote], outcome_index: int) -> float:
    comp = compare([str(i) for i in range(len(closing_quotes[0].odds))], closing_quotes)
    p = (comp.sharp_probs or comp.consensus_probs)[outcome_index]
    return clv(bet_odds, p)

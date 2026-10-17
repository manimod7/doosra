"""Multi-bookmaker comparison for a two-way (or n-way) match-winner market.

Given quotes from several bookmakers for the same event this computes, per
book, the margin and de-vigged fair probabilities, then the best available
price per outcome, a consensus fair probability, and a sharp benchmark
(Pinnacle or Betfair when present) used as the "true price" reference.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from .odds import devig_power, implied_prob, overround

# Books treated as sharp price-setters, in order of preference.
SHARP_BOOKS = ("pinnacle", "betfair", "betfair_exchange", "matchbook")


@dataclass(frozen=True)
class BookQuote:
    book: str
    odds: tuple[float, ...]  # decimal odds, one per outcome, same order for every book

    def __post_init__(self):
        if len(self.odds) < 2:
            raise ValueError("a market needs at least two outcomes")
        for o in self.odds:
            implied_prob(o)  # validates > 1 and finite


@dataclass(frozen=True)
class BookView:
    book: str
    odds: tuple[float, ...]
    margin: float
    fair_probs: tuple[float, ...]
    is_sharp: bool


@dataclass(frozen=True)
class BestPrice:
    outcome_index: int
    book: str
    odds: float


@dataclass(frozen=True)
class Comparison:
    outcomes: tuple[str, ...]
    books: tuple[BookView, ...]
    best: tuple[BestPrice, ...]
    consensus_probs: tuple[float, ...]
    sharp_book: str | None
    sharp_probs: tuple[float, ...] | None
    best_price_margin: float  # < 0 means an arbitrage exists across books


def compare(outcomes: list[str], quotes: list[BookQuote]) -> Comparison:
    if not quotes:
        raise ValueError("no quotes supplied")
    n = len(outcomes)
    for q in quotes:
        if len(q.odds) != n:
            raise ValueError(f"{q.book}: expected {n} odds, got {len(q.odds)}")

    views = []
    for q in quotes:
        views.append(
            BookView(
                book=q.book,
                odds=q.odds,
                margin=overround(list(q.odds)),
                fair_probs=tuple(devig_power(list(q.odds))),
                is_sharp=q.book.lower() in SHARP_BOOKS,
            )
        )

    best = []
    for i in range(n):
        top = max(quotes, key=lambda q: q.odds[i])
        best.append(BestPrice(i, top.book, top.odds[i]))

    consensus = [median(v.fair_probs[i] for v in views) for i in range(n)]
    total = sum(consensus)
    consensus = tuple(p / total for p in consensus)

    sharp = None
    for name in SHARP_BOOKS:
        sharp = next((v for v in views if v.book.lower() == name), None)
        if sharp:
            break

    return Comparison(
        outcomes=tuple(outcomes),
        books=tuple(sorted(views, key=lambda v: v.margin)),
        best=tuple(best),
        consensus_probs=consensus,
        sharp_book=sharp.book if sharp else None,
        sharp_probs=sharp.fair_probs if sharp else None,
        best_price_margin=sum(1.0 / b.odds for b in best) - 1.0,
    )

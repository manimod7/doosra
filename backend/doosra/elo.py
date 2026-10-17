"""Elo ratings for cricket teams with venue and toss-aware adjustments.

Walk-forward only: predict() uses ratings as they stood before the match,
then update() is called with the result. Never fit on future matches.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import log


@dataclass
class EloModel:
    k: float = 20.0
    base: float = 1500.0
    home_adv: float = 35.0  # rating points for the team playing at its home venue
    season_regression: float = 0.25  # pull ratings toward base at season start
    ratings: dict[str, float] = field(default_factory=dict)  # plain dict so models pickle

    def rating(self, team: str) -> float:
        return self.ratings.get(team, self.base)

    def predict(self, team_a: str, team_b: str, a_home: bool = False, b_home: bool = False) -> float:
        """Probability that team_a beats team_b."""
        diff = self.rating(team_a) - self.rating(team_b)
        if a_home and not b_home:
            diff += self.home_adv
        elif b_home and not a_home:
            diff -= self.home_adv
        return 1.0 / (1.0 + 10 ** (-diff / 400.0))

    def update(self, team_a: str, team_b: str, a_won: bool, p_a: float | None = None) -> None:
        if p_a is None:
            p_a = self.predict(team_a, team_b)
        delta = self.k * ((1.0 if a_won else 0.0) - p_a)
        self.ratings[team_a] = self.rating(team_a) + delta
        self.ratings[team_b] = self.rating(team_b) - delta

    def new_season(self) -> None:
        for t in list(self.ratings):
            self.ratings[t] = self.base + (self.ratings[t] - self.base) * (1 - self.season_regression)


def log_loss(probs: list[float], outcomes: list[int]) -> float:
    eps = 1e-12
    return -sum(
        o * log(max(p, eps)) + (1 - o) * log(max(1 - p, eps))
        for p, o in zip(probs, outcomes)
    ) / len(probs)


def brier(probs: list[float], outcomes: list[int]) -> float:
    return sum((p - o) ** 2 for p, o in zip(probs, outcomes)) / len(probs)

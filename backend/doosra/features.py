"""Leak-free, walk-forward match features.

FeatureBuilder.features(m) only uses matches already passed to update().
Every feature is pre-match information you would also have before an
upcoming fixture (no toss, no venue, since odds feeds don't supply them).

Features are antisymmetric differences (team_a minus team_b) so the model
cannot learn an artefact from Cricsheet's arbitrary team order.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import date
from math import log1p

from .cricsheet import MatchRecord
from .elo import EloModel

FEATURE_NAMES = (
    "elo_diff",       # (rating_a - rating_b) / 400
    "form_diff",      # win rate over last 10, a minus b
    "exp_diff",       # log1p(games played), a minus b
    "h2h",            # a's win share over last 10 meetings, centred at 0
    "rest_diff",      # log1p(days since last match), a minus b, capped
)
FORM_N = 10
H2H_N = 10
REST_CAP = 60


@dataclass
class TeamState:
    results: deque = field(default_factory=lambda: deque(maxlen=FORM_N))
    games: int = 0
    last_date: date | None = None


@dataclass
class FeatureBuilder:
    elo_k: float = 20.0
    season_regression: float = 0.25
    pools: dict[str, EloModel] = field(default_factory=dict)
    teams: dict[tuple[str, str], TeamState] = field(default_factory=dict)
    h2h: dict[tuple[str, str, str], deque] = field(default_factory=dict)
    pool_year: dict[str, int] = field(default_factory=dict)

    def _elo(self, pool: str) -> EloModel:
        if pool not in self.pools:
            self.pools[pool] = EloModel(k=self.elo_k, season_regression=self.season_regression)
        return self.pools[pool]

    def _team(self, pool: str, team: str) -> TeamState:
        return self.teams.setdefault((pool, team), TeamState())

    def _roll_year(self, pool: str, d: date) -> None:
        last = self.pool_year.get(pool)
        if last is not None and d.year > last:
            self._elo(pool).new_season()
        self.pool_year[pool] = d.year

    def games_played(self, pool: str, team: str) -> int:
        st = self.teams.get((pool, team))
        return st.games if st else 0

    def features_for(self, pool: str, a: str, b: str, d: date) -> tuple[list[float], float]:
        """Return (feature vector, raw Elo probability for a)."""
        elo = self._elo(pool)
        ta, tb = self._team(pool, a), self._team(pool, b)

        def form(t: TeamState) -> float:
            return sum(t.results) / len(t.results) if t.results else 0.5

        def rest(t: TeamState) -> float:
            if t.last_date is None:
                return log1p(REST_CAP)
            return log1p(min(max((d - t.last_date).days, 0), REST_CAP))

        key = (pool, *sorted((a, b)))
        meetings = self.h2h.get(key, ())
        if meetings:
            a_wins = sum(1 for w in meetings if w == a)
            h2h = a_wins / len(meetings) - 0.5
        else:
            h2h = 0.0

        x = [
            (elo.rating(a) - elo.rating(b)) / 400.0,
            form(ta) - form(tb),
            log1p(ta.games) - log1p(tb.games),
            h2h,
            rest(ta) - rest(tb),
        ]
        return x, elo.predict(a, b)

    def update(self, m: MatchRecord) -> None:
        pool = m.pool
        elo = self._elo(pool)
        a_won = m.winner == m.team_a
        elo.update(m.team_a, m.team_b, a_won)
        for team, won in ((m.team_a, a_won), (m.team_b, not a_won)):
            st = self._team(pool, team)
            st.results.append(1 if won else 0)
            st.games += 1
            st.last_date = m.date
        key = (pool, *sorted((m.team_a, m.team_b)))
        self.h2h.setdefault(key, deque(maxlen=H2H_N)).append(m.winner)

    def walk(self, matches: list[MatchRecord]):
        """Yield (match, features, elo_prob) in date order, updating after each."""
        for m in matches:
            self._roll_year(m.pool, m.date)
            x, p = self.features_for(m.pool, m.team_a, m.team_b, m.date)
            yield m, x, p
            self.update(m)


def flip(x: list[float]) -> list[float]:
    """Same match seen from the other side: every feature is antisymmetric."""
    return [-v for v in x]

"""Walk-forward knowledge about players, rivalries, venues and form.

`CricketState` only ever learns from matches that are finished (update() is
called after a match's balls have been turned into features), so features
for a match never contain that match or any later one.

Every rate is shrunk toward the pool average with empirical-Bayes weights,
so a batter with 12 balls of history is treated as "about average" instead
of as a genius or a duck. Shrinkage strengths (balls of pseudo-evidence):
  players K_PLAYER, batter-v-bowler rivalries K_MATCHUP, venues K_VENUE.

Feature vector per delivery (38 columns, see STATIC_NAMES + DYNAMIC_NAMES).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .ballbyball import CLASS_NAMES, CLASS_RUNS, EXTRA, LEGAL, N_CLASSES, WICKET, BBMatch

K_PLAYER = 150.0
K_MATCHUP = 30.0
K_VENUE = 1500.0
FORM_INNINGS = 10
VENUE_PAR_N = 10
COMP_PAR_N = 60          # recent first innings that set a competition's scoring level
K_COMP = 15.0            # shrinkage (in innings) of that level towards the pool's rate

RUNS = np.array(CLASS_RUNS, dtype=float)
STATIC_NAMES = (
    [f"bat_{c}" for c in CLASS_NAMES] + [f"bowl_{c}" for c in CLASS_NAMES]
    + ["rivalry_runs", "rivalry_wkt", "bat_form", "bat_exp", "bowl_exp",
       "venue_runs", "venue_wkt", "venue_par", "comp_par", "bat_won_toss",
       "w_dew_spread", "w_humidity", "w_precip"]
)
DYNAMIC_NAMES = ["pp", "middle", "death", "over_frac", "wickets", "striker_balls", "inn2",
                 "need_rpb", "balls_left_frac"]
FEATURE_NAMES = STATIC_NAMES + DYNAMIC_NAMES
N_STATIC = len(STATIC_NAMES)
N_FEATURES = len(FEATURE_NAMES)
DEFAULT_RATES = np.array([0.36, 0.32, 0.07, 0.005, 0.11, 0.045, 0.055, 0.035])


def phase_of(over: int, overs_limit: int) -> int:
    """0 powerplay, 1 middle, 2 death (scaled for 50-over games)."""
    pp_end, death_start = (6, 15) if overs_limit <= 20 else (10, 40)
    return 0 if over < pp_end else (1 if over < death_start else 2)


@dataclass
class CricketState:
    league: dict = field(default_factory=dict)
    bat: dict = field(default_factory=dict)
    bowl: dict = field(default_factory=dict)
    rivalry: dict = field(default_factory=dict)
    venue: dict = field(default_factory=dict)
    venue_first: dict = field(default_factory=dict)
    form: dict = field(default_factory=dict)
    bowl_phase: dict = field(default_factory=dict)
    last_lineup: dict = field(default_factory=dict)
    venue_chase: dict = field(default_factory=dict)
    comp_first: dict = field(default_factory=dict)    # (pool, competition) -> recent first-innings run rates
    last_comp: dict = field(default_factory=dict)     # (pool, team) -> competition of its latest match
    matches_seen: int = 0

    # ---------------------------------------------------------- copies ---
    def slim_for(self, m) -> "CricketState":
        """A deep copy holding only what simulating match m can read: the league, the ground,
        both teams, and the records of the players involved (and their matchups).

        Held-out evaluation keeps one of these per test match; a full copy of ten thousand
        matches' history each time would not fit in memory.
        """
        import copy

        rel = {m.pool, m.venue, *m.teams, *([m.competition] if m.competition else [])}
        for names in m.players.values():
            rel.update(names)
        for t in m.teams:
            rel.update(m.batting_order(t))
            rel.update(m.bowlers_used(t))
            prev = self.last_lineup.get((m.pool, t))
            if prev:
                rel.update(prev[0]); rel.update(prev[1])

        def keep(k) -> bool:
            parts = k if isinstance(k, tuple) else (k,)
            return all(p in rel for p in parts if isinstance(p, str))

        out = CricketState(matches_seen=self.matches_seen)
        for f in ("league", "bat", "bowl", "rivalry", "venue", "venue_first", "form", "bowl_phase",
                  "last_lineup", "venue_chase", "comp_first", "last_comp"):
            setattr(out, f, copy.deepcopy({k: v for k, v in getattr(self, f).items() if keep(k)}))
        return out

    # ---------------------------------------------------------- lookups --
    def league_rates(self, pool: str) -> np.ndarray:
        c = self.league.get(pool)
        if c is None or c.sum() < 500:
            return DEFAULT_RATES
        return c / c.sum()

    def _shrunk(self, counts: np.ndarray | None, prior: np.ndarray, k: float) -> np.ndarray:
        if counts is None:
            return prior
        return (counts + k * prior) / (counts.sum() + k)

    def batter_vec(self, pool: str, name: str) -> tuple[np.ndarray, float, float]:
        """(log rate ratio per class, form, experience) for a batter."""
        lr = self.league_rates(pool)
        c = self.bat.get((pool, name))
        rates = self._shrunk(c, lr, K_PLAYER)
        f = self.form.get((pool, name))
        league_rpb = float(lr @ RUNS)
        if f:
            runs = sum(r for r, _ in f)
            balls = sum(b for _, b in f)
            form = (runs + 60 * league_rpb) / (balls + 60) / league_rpb
        else:
            form = 1.0
        exp = math.log1p(c.sum()) if c is not None else 0.0
        return np.log(rates / lr), math.log(form), exp

    def bowler_vec(self, pool: str, name: str) -> tuple[np.ndarray, float]:
        lr = self.league_rates(pool)
        c = self.bowl.get((pool, name))
        rates = self._shrunk(c, lr, K_PLAYER)
        return np.log(rates / lr), (math.log1p(c.sum()) if c is not None else 0.0)

    def rivalry_vec(self, pool: str, batter: str, bowler: str) -> tuple[float, float]:
        """How much better/worse this batter does against this bowler than expected.

        Expectation comes from the two players' own shrunk rates; the head-to-head
        record is shrunk heavily toward it, so a few balls barely move it.
        """
        c = self.rivalry.get((pool, batter, bowler))
        if c is None:
            return 0.0, 0.0
        lr = self.league_rates(pool)
        b = self._shrunk(self.bat.get((pool, batter)), lr, K_PLAYER)
        w = self._shrunk(self.bowl.get((pool, bowler)), lr, K_PLAYER)
        exp = b * w / lr
        exp = exp / exp.sum()
        shr = (c + K_MATCHUP * exp) / (c.sum() + K_MATCHUP)
        return (float(shr @ RUNS) - float(exp @ RUNS),
                math.log(shr[WICKET] / exp[WICKET]))

    def venue_vec(self, pool: str, venue: str) -> tuple[float, float, float]:
        lr = self.league_rates(pool)
        rates = self._shrunk(self.venue.get((pool, venue)), lr, K_VENUE)
        par_hist = self.venue_first.get((pool, venue))
        league_rpb = float(lr @ RUNS)
        if par_hist:
            par = (sum(par_hist) + 3 * league_rpb) / (len(par_hist) + 3) / league_rpb
        else:
            par = 1.0
        return (float(rates @ RUNS) / league_rpb - 1, math.log(rates[WICKET] / lr[WICKET]), math.log(par))

    def comp_par(self, pool: str, competition: str | None) -> float:
        """log of the competition's recent first-innings scoring rate vs the pool (0 = average).
        Captures eras and leagues: the IPL since the impact-player rule scores well above
        T20 cricket as a whole; many associate internationals score well below it."""
        hist = self.comp_first.get((pool, competition)) if competition else None
        if not hist:
            return 0.0
        league_rpb = float(self.league_rates(pool) @ RUNS)
        return math.log((sum(hist) + K_COMP * league_rpb) / (len(hist) + K_COMP) / league_rpb)

    def competition_for(self, pool: str, team_a: str, team_b: str) -> str | None:
        """The competition two teams meet in: their shared latest competition, else None."""
        a, b = self.last_comp.get((pool, team_a)), self.last_comp.get((pool, team_b))
        return a if a and a == b else None

    def bowler_phase_share(self, pool: str, name: str) -> np.ndarray:
        c = self.bowl_phase.get((pool, name))
        if c is None:
            return np.ones(3)
        return (c + 1.0) / (c.sum() + 3.0)

    def chase_win_rate(self, pool: str, venue: str) -> float:
        w, n = self.venue_chase.get((pool, venue), (0, 0))
        return (w + 5) / (n + 10)

    # ------------------------------------------------------------ update --
    def update(self, m: BBMatch) -> None:
        pool = m.pool
        lg = self.league.setdefault(pool, np.zeros(N_CLASSES))
        vk = (pool, m.venue)
        ven = self.venue.setdefault(vk, np.zeros(N_CLASSES))
        for inn_no, inn in enumerate(m.innings, start=1):
            per_batter: dict[str, list[int]] = {}
            overs_by_bowler: dict[str, set] = {}
            for b in inn:
                one = np.zeros(N_CLASSES)
                one[b.outcome] = 1
                lg += one
                ven += one
                self.bat.setdefault((pool, b.batter), np.zeros(N_CLASSES))[b.outcome] += 1
                self.bowl.setdefault((pool, b.bowler), np.zeros(N_CLASSES))[b.outcome] += 1
                self.rivalry.setdefault((pool, b.batter, b.bowler), np.zeros(N_CLASSES))[b.outcome] += 1
                rb = per_batter.setdefault(b.batter, [0, 0])
                rb[0] += b.batter_runs
                rb[1] += 1 if LEGAL[b.outcome] else 0
                overs_by_bowler.setdefault(b.bowler, set()).add(b.over)
            for name, (r, balls) in per_batter.items():
                self.form.setdefault((pool, name), deque(maxlen=FORM_INNINGS)).append((r, balls))
            for name, overs in overs_by_bowler.items():
                ph = self.bowl_phase.setdefault((pool, name), np.zeros(3))
                for o in overs:
                    ph[phase_of(o, m.overs_limit)] += 1
            if inn_no == 1:
                legal = sum(1 for b in inn if LEGAL[b.outcome])
                total = sum(b.runs for b in inn)
                if legal > 0:
                    self.venue_first.setdefault(vk, deque(maxlen=VENUE_PAR_N)).append(total / legal)
                    if m.competition:
                        self.comp_first.setdefault((pool, m.competition), deque(maxlen=COMP_PAR_N)).append(total / legal)
        for team in m.teams:
            self.last_lineup[(pool, team)] = (m.batting_order(team)[:11], m.bowlers_used(team))
            if m.competition:
                self.last_comp[(pool, team)] = m.competition
        if m.winner and len(m.innings) == 2:
            w, n = self.venue_chase.get(vk, (0, 0))
            self.venue_chase[vk] = (w + (1 if m.winner != m.batting_first else 0), n + 1)
        self.matches_seen += 1


def static_features(state: CricketState, pool: str, venue: str, batter: str, bowler: str,
                    bat_won_toss: bool, weather: dict | None, competition: str | None = None) -> np.ndarray:
    bv, form, bexp = state.batter_vec(pool, batter)
    wv, wexp = state.bowler_vec(pool, bowler)
    rr, rw = state.rivalry_vec(pool, batter, bowler)
    vr, vw, vpar = state.venue_vec(pool, venue)
    wx = weather or {}
    cpar = state.comp_par(pool, competition)
    return np.concatenate([bv, wv, [rr, rw, form, bexp, wexp, vr, vw, vpar, cpar, 1.0 if bat_won_toss else 0.0,
                                    wx.get("dew_spread", np.nan), wx.get("humidity", np.nan),
                                    wx.get("precip", np.nan)]])


def dynamic_features(over: np.ndarray, legal: np.ndarray, wickets: np.ndarray, striker_balls: np.ndarray,
                     inn2: np.ndarray, score: np.ndarray, target: np.ndarray, overs_limit: int) -> np.ndarray:
    """Vectorised so the simulator can build thousands of rows at once."""
    total_balls = overs_limit * 6
    pp_end, death_start = (6, 15) if overs_limit <= 20 else (10, 40)
    balls_left = np.maximum(total_balls - legal, 1)
    need = np.where(inn2 > 0, np.clip((target - score) / balls_left, 0, 6), 0.0)
    return np.column_stack([
        (over < pp_end).astype(float), ((over >= pp_end) & (over < death_start)).astype(float),
        (over >= death_start).astype(float), over / overs_limit, wickets.astype(float),
        np.log1p(striker_balls), inn2.astype(float), need, balls_left / total_balls,
    ])


def match_rows(state: CricketState, m: BBMatch, weather: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Feature rows and outcome labels for every delivery of one match (state = before it)."""
    rows_s, y = [], []
    dyn_cols: list[list[float]] = []
    cache: dict = {}
    for inn_no, inn in enumerate(m.innings, start=1):
        score = wk = 0
        faced: dict[str, int] = {}
        target = m.target if inn_no == 2 and m.target else 0
        for b in inn:
            key = (b.batter, b.bowler, b.batting)
            if key not in cache:
                cache[key] = static_features(state, m.pool, m.venue, b.batter, b.bowler,
                                             m.toss_winner == b.batting, weather, m.competition)
            rows_s.append(cache[key])
            dyn_cols.append([b.over, b.legal_before, wk, faced.get(b.batter, 0), inn_no == 2, score, target])
            y.append(b.outcome)
            score += b.runs
            if LEGAL[b.outcome]:
                faced[b.batter] = faced.get(b.batter, 0) + 1
            if b.outcome == WICKET:
                wk += 1
    if not y:
        return np.zeros((0, N_FEATURES)), np.zeros(0, dtype=int)
    d = np.array(dyn_cols, dtype=float)
    dyn = dynamic_features(d[:, 0], d[:, 1], d[:, 2], d[:, 3], d[:, 4], d[:, 5], d[:, 6], m.overs_limit)
    return np.hstack([np.array(rows_s), dyn]), np.array(y, dtype=int)


def build_ball_dataset(matches: list[BBMatch], weather: dict | None = None,
                       state: CricketState | None = None, snapshot_at: set[int] | None = None,
                       snapshots: dict | None = None):
    """Walk forward through matches: features from the past, then learn from the match.

    snapshot_at: match indices for which a copy of the state *before* that match is
    stored in `snapshots` (used to simulate held-out matches honestly).
    """
    state = state or CricketState()
    X, y, match_idx, dates = [], [], [], []
    for i, m in enumerate(matches):
        if snapshot_at and i in snapshot_at and snapshots is not None:
            snapshots[i] = state.slim_for(m)
        Xi, yi = match_rows(state, m, (weather or {}).get(m.match_id))
        if len(yi):
            X.append(Xi.astype(np.float32))       # half the memory; trees and logit don't need float64
            y.append(yi.astype(np.int8))
            match_idx.append(np.full(len(yi), i))
            dates.append(np.full(len(yi), np.datetime64(m.date, "D")))
        state.update(m)
    if not X:
        raise ValueError("no deliveries found")
    return (np.vstack(X), np.concatenate(y), np.concatenate(match_idx),
            np.concatenate(dates), state)

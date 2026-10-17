"""Price any match market from a simulation, and settle it from a real match.

The same `market_values` function reads a SimResult made of 2,000 simulated
matches or one made from the real scorecard (n = 1). That is what makes the
feedback loop honest: predictions and outcomes are measured the same way.

Market types (team = batting side, overs/over are 1-based):
  match_winner(team)          team_total(team)           session(team, overs)
  over_runs(team, over)       wickets_by(team, overs)    first_wicket(team)
  player_runs(player)         match_total                match_sixes / match_fours
Line markets settle Over if value > line, Under if value < line, push if equal.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .ballbyball import CLASS_RUNS, FOUR, LEGAL, SIX, WICKET, BBMatch
from .simulate import InningsSim, SimResult

MARKET_TYPES = ("match_winner", "team_total", "session", "over_runs", "wickets_by", "first_wicket",
                "player_runs", "match_total", "match_sixes", "match_fours")


@dataclass(frozen=True)
class MarketSpec:
    type: str
    team: str | None = None
    overs: int | None = None
    over: int | None = None
    player: str | None = None

    def __post_init__(self):
        if self.type not in MARKET_TYPES:
            raise ValueError(f"unknown market type {self.type!r}")
        if self.type in ("match_winner", "team_total", "session", "over_runs", "wickets_by", "first_wicket") and not self.team:
            raise ValueError(f"{self.type} needs a team")
        if self.type in ("session", "wickets_by") and not self.overs:
            raise ValueError(f"{self.type} needs overs")
        if self.type == "over_runs" and not self.over:
            raise ValueError("over_runs needs an over number")
        if self.type == "player_runs" and not self.player:
            raise ValueError("player_runs needs a player")

    def label(self) -> str:
        t = self.type
        return {
            "match_winner": f"{self.team} to win",
            "team_total": f"{self.team} innings runs",
            "session": f"{self.team} runs after {self.overs} overs",
            "over_runs": f"{self.team} runs in over {self.over}",
            "wickets_by": f"{self.team} wickets lost by over {self.overs}",
            "first_wicket": f"{self.team} score at first wicket",
            "player_runs": f"{self.player} runs",
            "match_total": "Match runs, both innings",
            "match_sixes": "Match sixes",
            "match_fours": "Match fours",
        }[t]


def market_values(sim: SimResult, spec: MarketSpec) -> np.ndarray:
    """One value per simulation (NaN where the market does not apply)."""
    t = spec.type
    if t == "match_winner":
        i = sim.teams.index(spec.team)
        return np.where(sim.winner == 0.5, 0.5, (sim.winner == i).astype(float))
    if t in ("match_total", "match_sixes", "match_fours"):
        attr = {"match_total": "total", "match_sixes": "sixes", "match_fours": "fours"}[t]
        return sum(getattr(sim.innings[tm], attr) for tm in sim.teams)
    if t == "player_runs":
        for tm in sim.teams:
            if spec.player in sim.innings[tm].batter_runs:
                return sim.innings[tm].batter_runs[spec.player].astype(float)
        raise ValueError(f"{spec.player} is not in either lineup")
    inn: InningsSim = sim.innings[spec.team]
    overs = inn.score_after_over.shape[1]
    if t == "team_total":
        return inn.total.astype(float)
    if t == "first_wicket":
        return inn.first_wicket_score.astype(float)
    if t == "session":
        k = min(spec.overs, overs)
        return inn.score_after_over[:, k - 1]
    if t == "wickets_by":
        k = min(spec.overs, overs)
        return inn.wickets_after_over[:, k - 1]
    if t == "over_runs":
        k = spec.over
        if not 1 <= k <= overs:
            raise ValueError(f"over must be between 1 and {overs}")
        prev = inn.score_after_over[:, k - 2] if k > 1 else np.zeros(sim.n)
        return inn.score_after_over[:, k - 1] - prev
    raise ValueError(t)


@dataclass(frozen=True)
class LinePrice:
    market: str
    spec: dict
    line: float | None
    p_over: float
    p_under: float
    p_push: float
    fair_over: float | None
    fair_under: float | None
    mean: float
    median: float
    q10: float
    q90: float
    n: int
    calibrated: bool = False


def price(sim: SimResult, spec: MarketSpec, line: float | None = None, calibrator=None) -> LinePrice:
    v = market_values(sim, spec)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        raise ValueError(f"no simulations where {spec.label()} applies")
    if spec.type == "match_winner":
        p = float(np.mean(v))
        p_over, p_under, push, line = p, 1 - p, 0.0, None
    else:
        if line is None:
            line = float(np.floor(np.median(v))) + 0.5   # the fair half-point line
        p_over = float(np.mean(v > line))
        p_under = float(np.mean(v < line))
        push = max(0.0, 1.0 - p_over - p_under)
        if push < 1e-12:          # float noise, not a real push chance
            push, p_under = 0.0, 1.0 - p_over
    cal = False
    if calibrator is not None and push == 0.0:
        p_over = float(np.clip(calibrator.predict(np.array([p_over]))[0], 1e-6, 1 - 1e-6))
        p_under = 1 - p_over
        cal = True
    fo = 1 / p_over if p_over > 0 else None
    fu = 1 / p_under if p_under > 0 else None
    return LinePrice(spec.label(), asdict(spec), line, p_over, p_under, push, fo, fu,
                     float(np.mean(v)), float(np.median(v)), float(np.percentile(v, 10)),
                     float(np.percentile(v, 90)), int(len(v)), cal)


def standard_card(sim: SimResult, overs_limit: int = 20, calibrators: dict | None = None) -> dict:
    """The markets people actually bet on, priced at their fair lines.

    calibrators: {market_type: fitted map} from the feedback loop; applied where present.
    """
    cals = calibrators or {}
    sessions = [6, 10, 15] if overs_limit <= 20 else [10, 20, 30, 40]
    card: dict = {"winner": {t: round(sim.win_prob(t), 4) for t in sim.teams}, "teams": {}, "players": [],
                  "match": {}}

    def p(spec):
        return price(sim, spec, calibrator=cals.get(spec.type))

    for t in sim.teams:
        live_overs = sim.innings[t].score_after_over
        rows = [p(MarketSpec("team_total", team=t))]
        for k in sessions:
            if not np.isnan(live_overs[:, k - 1]).all():   # skip sessions already finished in live mode
                rows.append(p(MarketSpec("session", team=t, overs=k)))
        if not np.isnan(sim.innings[t].wickets_after_over[:, sessions[0] - 1]).all():
            rows.append(p(MarketSpec("wickets_by", team=t, overs=sessions[0])))
        if not np.isnan(sim.innings[t].first_wicket_score).all():
            rows.append(p(MarketSpec("first_wicket", team=t)))
        card["teams"][t] = [asdict(r) for r in rows]
        inn = sim.innings[t]
        for name, runs in inn.batter_runs.items():
            balls = inn.batter_balls[name]
            batted = float(np.mean(balls > 0))
            if batted < 0.25:
                continue
            card["players"].append({
                "player": name, "team": t, "expected_runs": round(float(np.mean(runs)), 1),
                "p_bats": round(batted, 3), "p_30_plus": round(float(np.mean(runs >= 30)), 3),
                "p_50_plus": round(float(np.mean(runs >= 50)), 3),
                "fair_line": float(np.floor(np.median(runs))) + 0.5,
            })
    card["players"].sort(key=lambda r: -r["expected_runs"])
    # run progression ("the worm"): 10th/50th/90th percentile of the score after each over
    card["worm"] = {}
    for t in sim.teams:
        sao = sim.innings[t].score_after_over
        bands = {"p10": [], "p50": [], "p90": []}
        for k in range(sao.shape[1]):
            col = sao[:, k]
            col = col[~np.isnan(col)]
            for q, key in ((10, "p10"), (50, "p50"), (90, "p90")):
                bands[key].append(None if len(col) == 0 else float(np.percentile(col, q)))
        card["worm"][t] = bands
    card["match"] = {}
    for k in ("match_total", "match_sixes", "match_fours"):
        try:
            card["match"][k] = asdict(p(MarketSpec(k)))
        except ValueError:
            pass   # not defined mid-match when the first innings detail is unknown
    return card


# ------------------------------------------------------------ real matches --

def sim_from_match(m: BBMatch) -> SimResult:
    """Turn a real scorecard into a one-simulation SimResult so markets settle identically."""
    innings = {}
    overs = m.overs_limit
    first_idx = m.teams.index(m.batting_first) if m.batting_first in m.teams else 0
    for t in m.teams:
        balls = next((inn for inn in m.innings if inn and inn[0].batting == t), [])
        score = wk = legal = 0
        sao = np.full((1, overs), np.nan)
        wao = np.full((1, overs), np.nan)
        br: dict[str, float] = {}
        bb: dict[str, float] = {}
        fours = sixes = 0
        first = np.nan
        for b in balls:
            score += b.runs
            if b.outcome == WICKET:
                if wk == 0:
                    first = score
                wk += 1
            fours += b.outcome == FOUR
            sixes += b.outcome == SIX
            br[b.batter] = br.get(b.batter, 0) + b.batter_runs
            if LEGAL[b.outcome]:
                bb[b.batter] = bb.get(b.batter, 0) + 1
                legal += 1
                if legal % 6 == 0 and legal // 6 <= overs:
                    sao[0, legal // 6 - 1] = score
                    wao[0, legal // 6 - 1] = wk
        if balls:
            for k in range(overs):
                if np.isnan(sao[0, k]):
                    sao[0, k] = score
                    wao[0, k] = wk
        if np.isnan(first) and balls:
            first = score
        innings[t] = InningsSim(t, np.array([float(score) if balls else np.nan]), np.array([float(wk)]), sao, wao,
                                {k: np.array([float(v)]) for k, v in br.items()},
                                {k: np.array([float(v)]) for k, v in bb.items()},
                                np.array([float(fours)]), np.array([float(sixes)]), np.array([first]))
    if m.winner is None:
        w = 0.5
    else:
        w = float(m.teams.index(m.winner))
    return SimResult(m.teams, 1, innings, np.array([first_idx]), np.array([w]))


def settle(m: BBMatch, spec: MarketSpec, line: float | None) -> float | None:
    """Realised outcome for a prediction: 1 Over/yes, 0 Under/no, None for push / void."""
    if m.result == "no result":
        return None
    try:
        v = market_values(sim_from_match(m), spec)[0]
    except ValueError:
        return None          # e.g. player did not play
    if np.isnan(v):
        return None
    if spec.type == "match_winner":
        return None if v == 0.5 else float(v)
    if line is None or v == line:
        return None
    return 1.0 if v > line else 0.0

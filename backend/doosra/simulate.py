"""Monte Carlo match simulator driven by the per-ball model.

All simulations advance together one delivery at a time (vectorised), so
pricing a whole T20 with 2,000 simulations takes a few hundred model calls.

Cricket rules modelled: strike rotation on 1s and 3s and at the end of each
over, wides/no-balls re-bowled, bowler quotas (overs/5 each), no bowler
bowls two overs in a row, all out at 10 wickets or when the lineup runs out,
the chase stops when the target is reached, a tie is settled 50/50.
Not modelled: rain/DLS, retirements, run-outs of the non-striker, free hits.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .ballbyball import CLASS_RUNS, EXTRA, FOUR, LEGAL, ONE, SIX, THREE, WICKET
from .ballmodel import BallModel
from .ballstate import CricketState, dynamic_features, phase_of, static_features

RUNS = np.array(CLASS_RUNS)
IS_LEGAL = np.array(LEGAL)
SWAP = np.zeros(8, dtype=bool)
SWAP[[ONE, THREE]] = True
# Match-day factor. Real matches vary as a whole (a flat deck, dew, a two-paced pitch), so
# balls within one match are not independent. Each simulation draws one "day" value z,
# shared by both innings, and every ball's probabilities are tilted by exp(z * DAY_TILT):
# z > 0 is a batting day (more boundaries, fewer dots and wickets). How spread out z is
# (the volatility) is fitted on held-out matches in lifecycle.evaluate.
#                    dot   1    2    3    4    6    wkt  extra
DAY_TILT = np.array([-0.3, 0.0, 0.2, 0.3, 0.6, 0.9, -0.6, 0.0])


@dataclass
class Lineup:
    batting: list[str]
    bowlers: list[str]

    def __post_init__(self):
        if len(self.batting) < 2:
            raise ValueError("a batting lineup needs at least two players")
        if not self.bowlers:
            raise ValueError("a lineup needs at least one bowler")


@dataclass
class MatchSetup:
    pool: str
    venue: str
    teams: tuple[str, str]
    lineups: dict[str, Lineup]
    overs_limit: int = 20
    batting_first: str | None = None     # known after the toss
    toss_winner: str | None = None
    weather: dict | None = None
    competition: str | None = None       # sets the scoring level and the match-day volatility


@dataclass
class LiveState:
    """Where a match stands right now (for in-play / session pricing)."""
    batting: str
    innings: int                  # 1 or 2
    score: int
    wickets: int
    legal_balls: int              # e.g. 7.3 overs = 45
    striker: str | None = None
    non_striker: str | None = None
    out: list[str] = field(default_factory=list)
    bowler_overs: dict[str, int] = field(default_factory=dict)
    last_bowler: str | None = None
    target: int | None = None     # required when innings == 2
    first_innings_score: int | None = None


@dataclass
class InningsSim:
    team: str
    total: np.ndarray
    wickets: np.ndarray
    score_after_over: np.ndarray      # N x overs (cumulative; NaN for overs before a live start)
    wickets_after_over: np.ndarray
    batter_runs: dict[str, np.ndarray]
    batter_balls: dict[str, np.ndarray]
    fours: np.ndarray
    sixes: np.ndarray
    first_wicket_score: np.ndarray


@dataclass
class SimResult:
    teams: tuple[str, str]
    n: int
    innings: dict[str, InningsSim]   # keyed by batting team
    batted_first: np.ndarray         # N, index of team that batted first
    winner: np.ndarray               # N, 0/1 team index, 0.5 tie
    model_note: str = ""
    day: np.ndarray | None = None    # N, the match-day factor each simulation drew

    def win_prob(self, team: str) -> float:
        i = self.teams.index(team)
        return float(np.mean(np.where(self.winner == 0.5, 0.5, self.winner == i)))


def _choose_bowlers(rng, k_idx, shares, overs_bowled, last, quota, phase):
    """Pick next over's bowler for each sim in k_idx (vectorised)."""
    w = np.tile(shares[:, phase], (len(k_idx), 1))
    ob = overs_bowled[k_idx]
    elig = ob < quota
    not_last = np.arange(w.shape[1])[None, :] != last[k_idx][:, None]
    mask = elig & not_last
    none = ~mask.any(axis=1)
    mask[none] = elig[none]                      # relax "not last" when forced
    none = ~mask.any(axis=1)
    mask[none] = True                            # relax quota when lineup is too short
    w = w * mask
    cum = np.cumsum(w, axis=1)
    u = rng.random(len(k_idx)) * cum[:, -1]
    return (u[:, None] > cum).sum(axis=1).clip(0, w.shape[1] - 1)


def simulate_innings(model: BallModel, state: CricketState, setup: MatchSetup, batting: str, bowling: str,
                     n: int, rng: np.random.Generator, innings_no: int, target: np.ndarray | None,
                     live: LiveState | None = None, day: np.ndarray | None = None) -> InningsSim:
    lineup = setup.lineups[batting].batting[:11]
    bowlers = setup.lineups[bowling].bowlers
    nb, nw, overs = len(lineup), len(bowlers), setup.overs_limit
    quota = max(1, overs // 5)
    bat_won_toss = setup.toss_winner == batting
    S = np.stack([np.stack([static_features(state, setup.pool, setup.venue, b, w, bat_won_toss, setup.weather,
                                                            setup.competition)
                            for w in bowlers]) for b in lineup])          # nb x nw x F_static
    shares = np.stack([state.bowler_phase_share(setup.pool, w) for w in bowlers])  # nw x 3

    score = np.zeros(n)
    wk = np.zeros(n, dtype=int)
    legal = np.zeros(n, dtype=int)
    striker = np.zeros(n, dtype=int)
    non = np.ones(n, dtype=int)
    nxt = np.full(n, 2)
    out_mask = np.zeros((n, nb), dtype=bool)
    faced = np.zeros((n, nb))
    bruns = np.zeros((n, nb))
    overs_bowled = np.zeros((n, nw), dtype=int)
    last = np.full(n, -1)
    fours = np.zeros(n)
    sixes = np.zeros(n)
    first_wkt = np.full(n, np.nan)
    sao = np.full((n, overs), np.nan)
    wao = np.full((n, overs), np.nan)
    start_legal = 0

    if live is not None:
        score[:] = live.score
        wk[:] = live.wickets
        legal[:] = start_legal = live.legal_balls
        name_idx = {nm: i for i, nm in enumerate(lineup)}
        for nm in live.out:
            if nm in name_idx:
                out_mask[:, name_idx[nm]] = True
        s_i = name_idx.get(live.striker) if live.striker else None
        ns_i = name_idx.get(live.non_striker) if live.non_striker else None
        free = [i for i in range(nb) if not out_mask[0, i] and i not in (s_i, ns_i)]
        if s_i is None:
            s_i = free.pop(0) if free else 0
        if ns_i is None:
            ns_i = free.pop(0) if free else 1
        striker[:], non[:] = s_i, ns_i
        rest = [i for i in range(nb) if not out_mask[0, i] and i not in (s_i, ns_i)]
        nxt[:] = rest[0] if rest else nb
        order = np.array(rest + [nb] * (nb + 1))       # next-batter queue by position
        for nm, ov in live.bowler_overs.items():
            if nm in bowlers:
                overs_bowled[:, bowlers.index(nm)] = ov
        if live.last_bowler in bowlers:
            last[:] = bowlers.index(live.last_bowler)
        if live.wickets > 0:
            first_wkt[:] = np.nan  # unknown from the live state; leave out of markets
        queue_pos = np.zeros(n, dtype=int)
    else:
        order = np.arange(2, nb + 2)
        queue_pos = np.zeros(n, dtype=int)
        nxt[:] = order[0]

    total_balls = overs * 6
    tgt = target if target is not None else np.zeros(n)
    done = (legal >= total_balls) | (wk >= 10) | ((innings_no == 2) & (score >= tgt))
    bowler = np.zeros(n, dtype=int)
    need_new = ~done & (legal % 6 == 0)
    cur_over = legal // 6
    if need_new.any():
        idx = np.where(need_new)[0]
        bowler[idx] = _choose_bowlers(rng, idx, shares, overs_bowled, last,
                                      quota, phase_of(int(cur_over[idx[0]]), overs))
    if (~need_new & ~done).any():  # live start mid-over: keep last bowler bowling
        idx = np.where(~need_new & ~done)[0]
        bowler[idx] = np.where(last[idx] >= 0, last[idx], 0)
    inn2 = np.full(n, 1.0 if innings_no == 2 else 0.0)

    for _ in range(total_balls * 3):
        act = np.where(~done)[0]
        if len(act) == 0:
            break
        st, bw = striker[act], bowler[act]
        over_no = legal[act] // 6
        dyn = dynamic_features(over_no, legal[act], wk[act], faced[act, st], inn2[act],
                               score[act], tgt[act], overs)
        P = model.predict_proba(np.hstack([S[st, bw], dyn]))
        if day is not None:
            P = P * np.exp(day[act][:, None] * DAY_TILT)
            P /= P.sum(axis=1, keepdims=True)
        cls = (rng.random(len(act))[:, None] > np.cumsum(P, axis=1)).sum(axis=1).clip(0, 7)
        r = RUNS[cls]
        lg = IS_LEGAL[cls]
        score[act] += r
        fours[act] += cls == FOUR
        sixes[act] += cls == SIX
        credit = (cls != EXTRA) & (cls != WICKET)
        bruns[act[credit], st[credit]] += r[credit]
        faced[act[lg], st[lg]] += 1
        legal[act] += lg
        # wickets
        wkt = cls == WICKET
        if wkt.any():
            wi = act[wkt]
            out_mask[wi, striker[wi]] = True
            first = np.isnan(first_wkt[wi]) & (wk[wi] == 0)
            first_wkt[wi[first]] = score[wi[first]]
            wk[wi] += 1
            qp = queue_pos[wi]
            new_b = order[np.minimum(qp, len(order) - 1)]
            queue_pos[wi] += 1
            allout = (wk[wi] >= 10) | (new_b >= nb)
            striker[wi[~allout]] = new_b[~allout]
            done[wi[allout]] = True
        # strike rotation on 1s and 3s
        sw = SWAP[cls] & ~wkt
        if sw.any():
            si = act[sw]
            striker[si], non[si] = non[si].copy(), striker[si].copy()
        # chase complete
        if innings_no == 2:
            won = act[score[act] >= tgt[act]]
            done[won] = True
        # end of over
        eoo = act[lg & (legal[act] % 6 == 0)]
        if len(eoo):
            ov_done = legal[eoo] // 6 - 1
            sao[eoo, ov_done] = score[eoo]
            wao[eoo, ov_done] = wk[eoo]
            overs_bowled[eoo, bowler[eoo]] += 1
            last[eoo] = bowler[eoo]
            done[eoo[legal[eoo] >= total_balls]] = True
            cont = eoo[~done[eoo]]
            if len(cont):
                striker[cont], non[cont] = non[cont].copy(), striker[cont].copy()
                ph = phase_of(int(legal[cont[0]] // 6), overs)
                # phase can differ across sims only by over; group by over number
                for ov in np.unique(legal[cont] // 6):
                    grp = cont[legal[cont] // 6 == ov]
                    bowler[grp] = _choose_bowlers(rng, grp, shares, overs_bowled, last, quota,
                                                  phase_of(int(ov), overs))
                del ph
    # innings ended early: carry the final score/wickets through remaining overs
    for k in range(overs):
        col = sao[:, k]
        fill = np.isnan(col) & (k >= start_legal // 6) & ((legal < (k + 1) * 6) | done)
        sao[fill, k] = score[fill]
        wao[fill, k] = wk[fill]
    first_wkt = np.where(np.isnan(first_wkt) & (wk == 0), score, first_wkt)
    return InningsSim(batting, score, wk, sao, wao,
                      {nm: bruns[:, i] for i, nm in enumerate(lineup)},
                      {nm: faced[:, i] for i, nm in enumerate(lineup)}, fours, sixes, first_wkt)


def _merge(parts: list[tuple[np.ndarray, InningsSim]], n: int, team: str, overs: int) -> InningsSim:
    """Stitch per-scenario simulations back into one N-length result."""
    def stack(attr):
        out = np.full((n,) + getattr(parts[0][1], attr).shape[1:], np.nan)
        for idx, s in parts:
            out[idx] = getattr(s, attr)
        return out

    names = sorted({k for _, s in parts for k in s.batter_runs})
    br, bb = {}, {}
    for nm in names:
        br[nm] = np.zeros(n)
        bb[nm] = np.zeros(n)
        for idx, s in parts:
            if nm in s.batter_runs:
                br[nm][idx] = s.batter_runs[nm]
                bb[nm][idx] = s.batter_balls[nm]
    return InningsSim(team, stack("total"), stack("wickets"), stack("score_after_over"),
                      stack("wickets_after_over"), br, bb, stack("fours"), stack("sixes"),
                      stack("first_wicket_score"))


def simulate_match(model: BallModel, state: CricketState, setup: MatchSetup, n: int = 2000,
                   seed: int = 0, live: LiveState | None = None, volatility: float = 0.0,
                   day: np.ndarray | None = None) -> SimResult:
    """volatility: spread of the match-day factor (0 = balls independent given the features).
    day: explicit per-simulation factors, used by evaluation to sample a wide proposal."""
    rng = np.random.default_rng(seed)
    if day is None and volatility > 0:
        day = rng.normal(0.0, volatility, n)
    a, b = setup.teams
    for t in setup.teams:
        if t not in setup.lineups:
            raise ValueError(f"missing lineup for {t}")
    if live is not None:
        first = live.batting if live.innings == 1 else (b if live.batting == a else a)
        first_idx = np.full(n, setup.teams.index(first))
    elif setup.batting_first:
        first_idx = np.full(n, setup.teams.index(setup.batting_first))
    else:
        first_idx = (rng.random(n) < 0.5).astype(int)    # toss unknown: either side bats first
    parts: dict[str, list] = {a: [], b: []}
    winner = np.zeros(n)
    for fi in (0, 1):
        idx = np.where(first_idx == fi)[0]
        if len(idx) == 0:
            continue
        bat1, bat2 = setup.teams[fi], setup.teams[1 - fi]
        k = len(idx)
        dsub = day[idx] if day is not None else None
        if live is not None and live.innings == 2:
            if live.target is None:
                raise ValueError("live second innings needs a target")
            s1 = _known_innings(bat1, live.target - 1, k, setup.overs_limit)
            tgt = np.full(k, float(live.target))
            s2 = simulate_innings(model, state, setup, bat2, bat1, k, rng, 2, tgt, live, dsub)
        else:
            s1 = simulate_innings(model, state, setup, bat1, bat2, k, rng, 1, None,
                                  live if live is not None else None, dsub)
            tgt = s1.total + 1
            s2 = simulate_innings(model, state, setup, bat2, bat1, k, rng, 2, tgt, None, dsub)
        parts[bat1].append((idx, s1))
        parts[bat2].append((idx, s2))
        w2 = s2.total >= tgt
        tie = s2.total == tgt - 1
        winner[idx] = np.where(tie, 0.5, np.where(w2, 1 - fi, fi))
    innings = {t: _merge(parts[t], n, t, setup.overs_limit) for t in setup.teams}
    return SimResult(setup.teams, n, innings, first_idx, winner, day=day)


def _known_innings(team: str, total: int, n: int, overs: int) -> InningsSim:
    nan = np.full((n, overs), np.nan)
    z = np.full(n, np.nan)
    return InningsSim(team, np.full(n, float(total)), z.copy(), nan, nan.copy(), {}, {}, z.copy(),
                      z.copy(), z.copy())


def default_setup(state: CricketState, pool: str, venue: str, team_a: str, team_b: str,
                  overs_limit: int = 20, **kw) -> MatchSetup:
    """Use each team's most recent lineup and bowlers when no XI is given."""
    lineups = {}
    for t in (team_a, team_b):
        lu = state.last_lineup.get((pool, t))
        if lu is None:
            raise ValueError(f"no lineup history for {t} in {pool}; supply the XI")
        bat, bowl = lu
        lineups[t] = Lineup(list(bat), list(bowl) or list(bat[-5:]))
    kw.setdefault("competition", state.competition_for(pool, team_a, team_b))
    return MatchSetup(pool, venue, (team_a, team_b), lineups, overs_limit, **kw)

"""Ball-model lifecycle: evaluate walk-forward, gate, register, load.

Evaluation answers two questions on data the model never trained on:
1. Per delivery: is the log loss better than the pool's plain frequencies?
2. Per match: when held-out matches are simulated with their real lineups,
   do real outcomes fall where the simulations said they would?
   - 80% interval coverage for innings totals (target ~0.80)
   - PIT values (should be roughly uniform)
   - Brier score at the fair line for totals and 6-over sessions
   - log loss for the match winner

Promotion: the challenger must beat the baseline, have sane coverage, and
be trained on newer data than the champion; when there are matches the
champion never saw, the challenger must not do worse on them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from .ballbyball import BBMatch
from .ballmodel import BallModel, fit_best, multiclass_log_loss
from .ballstate import FEATURE_NAMES, CricketState, build_ball_dataset
from .props import MarketSpec, market_values, sim_from_match
from .simulate import Lineup, MatchSetup, simulate_match

MIN_BALLS = 20000
MIN_SIM_MATCHES = 60      # fewer held-out simulations than this is too noisy to judge coverage
COVERAGE_TOLERANCE = 0.08  # how far from 80% coverage may drift before it matters
MAX_REGRESSION = 0.002    # allowed log-loss slack vs champion on matches it never saw


def coverage_problem(coverage: float | None, n: int) -> tuple[bool, str]:
    """Is the 80% interval's real coverage clearly wrong (both significant and material)?

    With 36 matches, 61% coverage can happen by chance even when the true rate is
    80%, so a small sample is never used to reject, and a large one must be both
    statistically (|z| > 2.58) and practically (> 8 points) off.
    """
    if coverage is None or n < MIN_SIM_MATCHES:
        return False, ""
    se = (0.8 * 0.2 / n) ** 0.5
    z = (coverage - 0.8) / se
    if abs(z) > 2.58 and abs(coverage - 0.8) > COVERAGE_TOLERANCE:
        size = "too narrow" if coverage < 0.8 else "too wide"
        return True, (f"innings-total 80% interval caught {coverage:.0%} of {n} real totals; "
                      f"simulated ranges are {size}")
    return False, ""


@dataclass
class BallArtifact:
    version: str
    model: BallModel
    state: CricketState
    trained_through: date
    pools: tuple[str, ...]
    feature_names: tuple[str, ...]
    evaluation: dict = field(default_factory=dict)
    volatility: float | dict = 0.0   # match-day spread, per stratum (see volatility_for); old artefacts: 0


def _pit(sim_vals: np.ndarray, actual: float, rng: np.random.Generator) -> float:
    v = sim_vals[~np.isnan(sim_vals)]
    lo, eq = np.mean(v < actual), np.mean(v == actual)
    return float(lo + rng.random() * eq)   # randomised PIT for discrete outcomes


PROPOSAL = 0.5                     # spread of match-day factors sampled during evaluation
VOL_GRID = tuple(round(x, 3) for x in np.arange(0.0, 0.451, 0.025))
MIN_STRATUM_SIMS = 25              # fewer held-out matches than this: use the pooled volatility


def stratum_of(competition: str | None) -> str:
    """Internationals (often associate sides, with wild swings) and franchise/domestic
    leagues vary differently from match to match, so each gets its own volatility."""
    if not competition:
        return "all"
    return "international" if competition == "T20I" else "club"


def volatility_for(art, competition: str | None) -> float:
    """The match-day volatility to simulate with (older artefacts store a single number)."""
    v = getattr(art, "volatility", 0.0)
    if isinstance(v, dict):
        return float(v.get(stratum_of(competition), v.get("all", 0.0)))
    return float(v)
HELDOUT_MARKETS = ("total", "session6")


def simulate_heldout(model: BallModel, state: CricketState, m: BBMatch, n: int, seed: int,
                     proposal: float = PROPOSAL) -> dict | None:
    """Simulate a finished match from its real lineups, keeping the raw simulated values.

    The simulations draw match-day factors from a wide normal (the proposal), so one run
    can be re-weighted afterwards to what any narrower volatility would have predicted.
    """
    if len(m.innings) < 2 or not m.batting_first:
        return None
    lineups = {}
    for t in m.teams:
        bat = m.batting_order(t)[:11]
        bowl = m.bowlers_used(t) or bat[-5:]
        if len(bat) < 2:
            return None
        lineups[t] = Lineup(bat, bowl)
    setup = MatchSetup(m.pool, m.venue, m.teams, lineups, m.overs_limit, batting_first=m.batting_first,
                       toss_winner=m.toss_winner, competition=m.competition)
    day = np.random.default_rng(seed + 10_000).normal(0.0, proposal, n) if proposal > 0 else None
    sim = simulate_match(model, state, setup, n=n, seed=seed, day=day)
    real = sim_from_match(m)
    first = m.batting_first
    out = {"match_id": m.match_id, "year": m.date.year, "seed": seed, "proposal": proposal,
           "stratum": stratum_of(m.competition),
           "day": day if day is not None else np.zeros(n)}
    for key, spec in (("total", MarketSpec("team_total", team=first)),
                      ("session6", MarketSpec("session", team=first, overs=min(6, m.overs_limit)))):
        out[key] = market_values(sim, spec)
        out[f"{key}_actual"] = float(market_values(real, spec)[0])
    if m.winner:
        i = sim.teams.index(m.winner)
        out["winner_hit"] = np.where(sim.winner == 0.5, 0.5, (sim.winner == i).astype(float))
    return out


def _weights(day: np.ndarray, proposal: float, sigma: float) -> np.ndarray:
    """Importance weights turning proposal-spread simulations into sigma-spread ones."""
    if proposal <= 0:
        return np.ones_like(day) / len(day)
    s = max(sigma, 0.01)                  # sigma -> 0 is a point mass; 0.01 is indistinguishable
    logw = -0.5 * (day / s) ** 2 + 0.5 * (day / proposal) ** 2
    w = np.exp(logw - logw.max())
    return w / w.sum()


def _wquantile(v: np.ndarray, w: np.ndarray, q: float) -> float:
    o = np.argsort(v)
    c = np.cumsum(w[o])
    return float(v[o][min(np.searchsorted(c, q * c[-1]), len(v) - 1)])


def _wcrps(v: np.ndarray, w: np.ndarray, a: float) -> float:
    """Continuous ranked probability score of a weighted sample (lower is better)."""
    w = w / w.sum()
    # E|X - X'| over distinct pairs only: counting each draw against itself biases CRPS
    # against narrow forecasts, which rely on fewer effective draws after re-weighting.
    spread = (w @ np.abs(v[:, None] - v[None, :]) @ w) / max(1.0 - float(w @ w), 1e-9)
    return float(w @ np.abs(v - a) - 0.5 * spread)


def score_heldout(r: dict, sigma: float, rng: np.random.Generator) -> dict:
    """Coverage, PIT, Brier, CRPS and winner log loss for one held-out match at a volatility."""
    w_all = _weights(r["day"], r["proposal"], sigma)
    out: dict = {}
    for key in HELDOUT_MARKETS:
        v, a = r[key], r[f"{key}_actual"]
        ok = ~np.isnan(v)
        if not ok.any():
            continue
        v, w = v[ok], w_all[ok] / w_all[ok].sum()
        line = float(np.floor(_wquantile(v, w, 0.5))) + 0.5
        p_over = float(w[v > line].sum())
        out[f"{key}_in80"] = bool(_wquantile(v, w, 0.1) <= a <= _wquantile(v, w, 0.9))
        out[f"{key}_pit"] = float(w[v < a].sum() + rng.random() * w[v == a].sum())
        out[f"{key}_brier"] = (p_over - float(a > line)) ** 2
        out[f"{key}_crps"] = _wcrps(v, w, a)
    if "winner_hit" in r:
        out["winner_ll"] = float(-np.log(np.clip(w_all @ r["winner_hit"], 1e-6, 1)))
    return out


def _summarise(scored: list[dict]) -> dict:
    rep: dict = {"sim_matches": len(scored)}
    for key in HELDOUT_MARKETS:
        rows = [s for s in scored if f"{key}_in80" in s]
        if rows:
            rep[f"{key}_coverage80"] = float(np.mean([s[f"{key}_in80"] for s in rows]))
            rep[f"{key}_brier"] = float(np.mean([s[f"{key}_brier"] for s in rows]))
            rep[f"{key}_crps"] = float(np.mean([s[f"{key}_crps"] for s in rows]))
            rep[f"{key}_pit_hist"] = np.histogram([s[f"{key}_pit"] for s in rows], bins=5, range=(0, 1))[0].tolist()
    wl = [s["winner_ll"] for s in scored if "winner_ll" in s]
    if wl:
        rep["winner_log_loss"] = float(np.mean(wl))
    return rep


def _crps_at(raw: list[dict], sigma: float, seed: int) -> float:
    rng = np.random.default_rng(seed)
    sc = [score_heldout(r, sigma, rng) for r in raw]
    vals = [s[f"{k}_crps"] / max(np.std(r[k][~np.isnan(r[k])]) if (~np.isnan(r[k])).any() else 1, 1)
            for s, r in zip(sc, raw) for k in HELDOUT_MARKETS if f"{k}_crps" in s]
    return float(np.mean(vals)) if vals else float("inf")


def _best(raw: list[dict], seed: int) -> float:
    return min(VOL_GRID, key=lambda s: _crps_at(raw, s, seed))


def fit_volatility(raw: list[dict], seed: int = 0) -> dict:
    """Choose the match-day volatility for each stratum, and report it honestly.

    For each test season and stratum: fit on the other seasons' held-out matches (lowest
    CRPS), then score that season with it. Those cross-season scores are what the gate sees.
    The volatilities shipped with the model are fitted on every season. A stratum with too
    few held-out matches uses the pooled value.
    """
    rng = np.random.default_rng(seed)
    years = sorted({r["year"] for r in raw})
    strata = sorted({r.get("stratum", "all") for r in raw})

    def fitted(rows: list[dict], pooled: list[dict]) -> float:
        return _best(rows, seed) if len(rows) >= MIN_STRATUM_SIMS else _best(pooled, seed)

    chosen: dict = {}
    scored: list[dict] = []
    by_stratum: dict[str, list[dict]] = {}
    for yr in years:
        others = [r for r in raw if r["year"] != yr] or raw
        for st in strata:
            here = [r for r in raw if r["year"] == yr and r.get("stratum", "all") == st]
            if not here:
                continue
            sigma = fitted([r for r in others if r.get("stratum", "all") == st], others)
            chosen[f"{yr}:{st}"] = sigma
            sc = [score_heldout(r, sigma, rng) for r in here]
            scored += sc
            by_stratum.setdefault(st, []).extend(sc)
    final = {"all": float(_best(raw, seed))}
    for st in strata:
        if st != "all":
            final[st] = float(fitted([r for r in raw if r.get("stratum") == st], raw))
    rep = _summarise(scored)
    without = _summarise([score_heldout(r, 0.0, np.random.default_rng(seed)) for r in raw])
    rep |= {"volatility": final, "volatility_by_season": chosen,
            "total_coverage80_without_volatility": without.get("total_coverage80"),
            "by_stratum": {st: {k: v for k, v in _summarise(sc).items()
                                if k in ("sim_matches", "total_coverage80", "session6_coverage80", "total_crps")}
                           for st, sc in by_stratum.items()}}
    return rep


def evaluate(matches: list[BBMatch], test_years: int = 3, kinds=("hgb", "logit"), max_iter: int = 150,
             sims_per_year: int = 60, n_sims: int = 400, seed: int = 0, weather: dict | None = None) -> dict:
    years = sorted({m.date.year for m in matches})
    tests = years[-test_years:] if len(years) > test_years else years[1:]
    rng = np.random.default_rng(seed)
    sample: dict[int, list[int]] = {}
    for yr in tests:
        # half internationals, half leagues, so each stratum's volatility has enough matches
        picks: list[int] = []
        for st in ("international", "club"):
            idx = [i for i, m in enumerate(matches)
                   if m.date.year == yr and len(m.innings) == 2 and stratum_of(m.competition) == st]
            k = min(sims_per_year // 2, len(idx))
            picks += rng.choice(idx, size=k, replace=False).tolist() if k else []
        if not picks:
            idx = [i for i, m in enumerate(matches) if m.date.year == yr and len(m.innings) == 2]
            picks = rng.choice(idx, size=min(sims_per_year, len(idx)), replace=False).tolist() if idx else []
        sample[yr] = sorted(picks)
    snaps: dict = {}
    X, y, mi, dates, final_state = build_ball_dataset(
        matches, weather, snapshot_at={i for v in sample.values() for i in v}, snapshots=snaps)
    folds, ll_model, ll_base, n_balls, sims = [], [], [], 0, []
    for yr in tests:
        start = np.datetime64(f"{yr}-01-01")
        end = np.datetime64(f"{yr + 1}-01-01")
        tr = dates < start
        te = (dates >= start) & (dates < end)
        if tr.sum() < 2000 or te.sum() == 0:
            continue
        cut = int(tr.sum() * 0.8)
        tr_idx = np.where(tr)[0]
        choice = fit_best(X[tr_idx[:cut]], y[tr_idx[:cut]], X[tr_idx[cut:]], y[tr_idx[cut:]],
                          kinds=kinds, max_iter=max_iter)
        base = BallModel("baseline").fit(X[tr], y[tr])
        llm = multiclass_log_loss(choice.model.predict_proba(X[te]), y[te])
        llb = multiclass_log_loss(base.predict_proba(X[te]), y[te])
        ll_model.append(llm * te.sum())
        ll_base.append(llb * te.sum())
        n_balls += int(te.sum())
        for j, i in enumerate(sample.get(yr, [])):
            r = simulate_heldout(choice.model, snaps.pop(i), matches[i], n_sims, seed + j)
            if r:
                sims.append(r)
        folds.append({"year": yr, "train_balls": int(tr.sum()), "test_balls": int(te.sum()),
                      "model_ll": round(llm, 5), "baseline_ll": round(llb, 5), "chosen": choice.scores["chosen"]})
    if not folds:
        raise ValueError("not enough history for a walk-forward evaluation")
    report = {
        "n_balls": n_balls,
        "ball_log_loss": float(sum(ll_model) / n_balls),
        "baseline_log_loss": float(sum(ll_base) / n_balls),
        "folds": folds,
        "sim_matches": len(sims),
    }
    if sims:
        report |= fit_volatility(sims, seed)
    return {"report": report, "X": X, "y": y, "dates": dates, "state": final_state}


def gate(report: dict, champion: dict | None, trained_through: date, recent_ll: dict | None = None,
         min_balls: int = MIN_BALLS) -> tuple[bool, str]:
    if report["n_balls"] < min_balls:
        return False, f"only {report['n_balls']} out-of-sample deliveries; need {min_balls}"
    if report["ball_log_loss"] >= report["baseline_log_loss"]:
        return False, (f"per-ball log loss {report['ball_log_loss']:.4f} does not beat the plain frequencies "
                       f"{report['baseline_log_loss']:.4f}")
    bad, why = coverage_problem(report.get("total_coverage80"), report.get("sim_matches", 0))
    if bad:
        return False, why
    if champion:
        champ_through = date.fromisoformat(champion["trained_through"])
        if trained_through <= champ_through:
            return False, "no matches newer than the current model; nothing to learn from"
        if recent_ll and recent_ll["challenger"] > recent_ll["champion"] + MAX_REGRESSION:
            return False, (f"on {recent_ll['n']} deliveries the current model never saw, the new one is worse "
                           f"({recent_ll['challenger']:.4f} vs {recent_ll['champion']:.4f})")
    return True, "beats the baseline out of sample" + (" and holds up against the current model" if champion else "")


@dataclass
class BallRegistry:
    root: Path

    def __post_init__(self):
        self.root = Path(self.root) / "ball"
        self.root.mkdir(parents=True, exist_ok=True)
        self.index = self.root / "champion.json"
        self.history = self.root / "history.jsonl"

    def champion_meta(self) -> dict | None:
        return json.loads(self.index.read_text()) if self.index.exists() else None

    def load(self) -> BallArtifact | None:
        meta = self.champion_meta()
        if not meta or not (self.root / meta["file"]).exists():
            return None
        art = joblib.load(self.root / meta["file"])   # only files this tool wrote itself
        if tuple(art.feature_names) != tuple(FEATURE_NAMES):
            return None      # built for a different feature set: unusable until retrained
        return art

    def record(self, art: BallArtifact, promoted: bool, reason: str) -> dict:
        meta = {"version": art.version, "file": f"ball-{art.version}.joblib",
                "trained_through": art.trained_through.isoformat(), "pools": list(art.pools),
                "promoted": promoted, "reason": reason, "volatility": art.volatility,
                "report": {k: v for k, v in art.evaluation.items() if k != "folds"},
                "folds": art.evaluation.get("folds", []),
                "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        if promoted:
            joblib.dump(art, self.root / meta["file"])
            self.index.write_text(json.dumps(meta, indent=2))
        with self.history.open("a") as fh:
            fh.write(json.dumps(meta) + "\n")
        return meta

    def runs(self, limit: int = 20) -> list[dict]:
        if not self.history.exists():
            return []
        lines = self.history.read_text().strip().splitlines()
        return [json.loads(x) for x in lines[-limit:]][::-1]


def retrain_ball(matches: list[BBMatch], models_dir: str | Path, pools: tuple[str, ...], test_years: int = 3,
                 kinds=("hgb", "logit"), max_iter: int = 150, sims_per_year: int = 60, n_sims: int = 400,
                 min_balls: int = MIN_BALLS, weather: dict | None = None) -> dict:
    """Evaluate, fit on everything, gate against the champion, register."""
    matches = [m for m in matches if m.pool in pools]
    if not matches:
        raise ValueError(f"no matches in pools {pools}")
    ev = evaluate(matches, test_years, kinds, max_iter, sims_per_year, n_sims, weather=weather)
    X, y, dates = ev["X"], ev["y"], ev["dates"]
    cut = int(len(y) * 0.85)
    choice = fit_best(X[:cut], y[:cut], X[cut:], y[cut:], kinds=kinds, max_iter=max_iter)
    trained_through = matches[-1].date
    reg = BallRegistry(models_dir)
    champ_meta = reg.champion_meta()
    recent = None
    if champ_meta:
        champ = reg.load()
        new_rows = dates > np.datetime64(champ_meta["trained_through"])
        if champ is not None and new_rows.sum() > 500 and tuple(champ.feature_names) == tuple(FEATURE_NAMES):
            # both are out of sample on these rows: the champion never saw them, and the
            # challenger's walk-forward fold for that period was trained only on earlier data
            ch_ll = multiclass_log_loss(champ.model.predict_proba(X[new_rows]), y[new_rows])
            first_new = np.min(dates[new_rows])
            wf = BallModel(choice.model.kind, max_iter=max_iter).fit(X[dates < first_new], y[dates < first_new])
            recent = {"n": int(new_rows.sum()), "champion": ch_ll,
                      "challenger": multiclass_log_loss(wf.predict_proba(X[new_rows]), y[new_rows])}
    report = ev["report"] | {"chosen": choice.scores["chosen"], "recent_comparison": recent}
    ok, reason = gate(report, champ_meta, trained_through, recent, min_balls=min_balls)
    art = BallArtifact(datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"), choice.model, ev["state"],
                       trained_through, tuple(pools), tuple(FEATURE_NAMES), report,
                       volatility=report.get("volatility", 0.0))
    meta = reg.record(art, ok, reason)
    return {"promoted": ok, "reason": reason, "report": report, "meta": meta}

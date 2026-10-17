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


def _pit(sim_vals: np.ndarray, actual: float, rng: np.random.Generator) -> float:
    v = sim_vals[~np.isnan(sim_vals)]
    lo, eq = np.mean(v < actual), np.mean(v == actual)
    return float(lo + rng.random() * eq)   # randomised PIT for discrete outcomes


def simulate_heldout(model: BallModel, state: CricketState, m: BBMatch, n: int, seed: int) -> dict | None:
    """Simulate a finished match from its real lineups and compare with what happened."""
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
                       toss_winner=m.toss_winner)
    sim = simulate_match(model, state, setup, n=n, seed=seed)
    real = sim_from_match(m)
    rng = np.random.default_rng(seed)
    first = m.batting_first
    out = {"match_id": m.match_id}
    for key, spec in (("total", MarketSpec("team_total", team=first)),
                      ("session6", MarketSpec("session", team=first, overs=min(6, m.overs_limit)))):
        sv = market_values(sim, spec)
        actual = float(market_values(real, spec)[0])
        line = float(np.floor(np.nanmedian(sv))) + 0.5
        p_over = float(np.nanmean(sv > line))
        out[f"{key}_pit"] = _pit(sv, actual, rng)
        out[f"{key}_in80"] = bool(np.nanpercentile(sv, 10) <= actual <= np.nanpercentile(sv, 90))
        out[f"{key}_brier"] = (p_over - float(actual > line)) ** 2
    if m.winner:
        p = sim.win_prob(m.winner)
        out["winner_ll"] = float(-np.log(np.clip(p, 1e-6, 1)))
    return out


def evaluate(matches: list[BBMatch], test_years: int = 3, kinds=("hgb", "logit"), max_iter: int = 150,
             sims_per_year: int = 40, n_sims: int = 400, seed: int = 0, weather: dict | None = None) -> dict:
    years = sorted({m.date.year for m in matches})
    tests = years[-test_years:] if len(years) > test_years else years[1:]
    rng = np.random.default_rng(seed)
    sample: dict[int, list[int]] = {}
    for yr in tests:
        idx = [i for i, m in enumerate(matches) if m.date.year == yr and len(m.innings) == 2]
        sample[yr] = sorted(rng.choice(idx, size=min(sims_per_year, len(idx)), replace=False).tolist()) if idx else []
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
            r = simulate_heldout(choice.model, snaps[i], matches[i], n_sims, seed + j)
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
        for key in ("total", "session6"):
            report[f"{key}_coverage80"] = float(np.mean([s[f"{key}_in80"] for s in sims]))
            report[f"{key}_brier"] = float(np.mean([s[f"{key}_brier"] for s in sims]))
            pits = np.array([s[f"{key}_pit"] for s in sims])
            report[f"{key}_pit_hist"] = np.histogram(pits, bins=5, range=(0, 1))[0].tolist()
        wl = [s["winner_ll"] for s in sims if "winner_ll" in s]
        if wl:
            report["winner_log_loss"] = float(np.mean(wl))
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
        return joblib.load(self.root / meta["file"])   # only files this tool wrote itself

    def record(self, art: BallArtifact, promoted: bool, reason: str) -> dict:
        meta = {"version": art.version, "file": f"ball-{art.version}.joblib",
                "trained_through": art.trained_through.isoformat(), "pools": list(art.pools),
                "promoted": promoted, "reason": reason,
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
                 kinds=("hgb", "logit"), max_iter: int = 150, sims_per_year: int = 40, n_sims: int = 400,
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
                       trained_through, tuple(pools), tuple(FEATURE_NAMES), report)
    meta = reg.record(art, ok, reason)
    return {"promoted": ok, "reason": reason, "report": report, "meta": meta}

"""The feedback loop: log -> settle -> score -> recalibrate -> watch for drift.

1. log_card(): every market Doosra prices is stored with the probability it
   gave, the line, the model version and the time.
2. settle(): when the real scorecard arrives (Cricsheet), each prediction is
   settled with the same market code that priced it.
3. report(): per market type, how good the probabilities have been
   (Brier, log loss, predicted vs actual rate, calibration by bucket).
4. fit_calibrators(): once a market type has enough settled predictions,
   an isotonic map corrects systematic over- or under-confidence; it is
   used for future prices of that market type.
5. drift(): flags a market type whose recent predictions are clearly worse
   than before, which is the signal to retrain.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression

from .ballbyball import BBMatch
from .props import MarketSpec, settle as settle_market
from .storage import Store, now_iso

SCHEMA = """
CREATE TABLE IF NOT EXISTS market_predictions (
  id INTEGER PRIMARY KEY,
  event_key TEXT NOT NULL,
  team_a TEXT NOT NULL,
  team_b TEXT NOT NULL,
  match_date TEXT NOT NULL,
  market_type TEXT NOT NULL,
  spec TEXT NOT NULL,
  line REAL,
  p_over REAL NOT NULL,
  model_version TEXT NOT NULL,
  odds_over REAL,
  odds_under REAL,
  made_at TEXT NOT NULL,
  outcome REAL,            -- 1 over/yes, 0 under/no
  status TEXT NOT NULL DEFAULT 'open',   -- open | settled | void
  settled_match TEXT,
  settled_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_mp_open ON market_predictions(status, match_date);
"""
MIN_CALIBRATION = 500
DRIFT_WINDOW = 150


def init(store: Store) -> None:
    store._conn.executescript(SCHEMA)
    store._conn.commit()


def norm(team: str) -> str:
    return team.lower().replace(" women", "").replace(" w", "").strip()


def log_prediction(store: Store, event_key: str, teams: tuple[str, str], match_date: date, spec: MarketSpec,
                   line: float | None, p_over: float, version: str, odds_over: float | None = None,
                   odds_under: float | None = None) -> int:
    init(store)
    with store.tx() as c:
        cur = c.execute(
            """INSERT INTO market_predictions (event_key, team_a, team_b, match_date, market_type, spec, line,
               p_over, model_version, odds_over, odds_under, made_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (event_key, teams[0], teams[1], match_date.isoformat(), spec.type, json.dumps(asdict(spec)), line,
             float(p_over), version, odds_over, odds_under, now_iso()))
        return int(cur.lastrowid)


def log_card(store: Store, event_key: str, teams: tuple[str, str], match_date: date, card: dict, version: str) -> int:
    """Log every market on a standard card. Returns how many rows were written."""
    n = 0
    for t, p in card["winner"].items():
        log_prediction(store, event_key, teams, match_date, MarketSpec("match_winner", team=t), None, p, version)
        n += 1
    for rows in card["teams"].values():
        for r in rows:
            spec = MarketSpec(**{k: v for k, v in r["spec"].items() if v is not None})
            log_prediction(store, event_key, teams, match_date, spec, r["line"], r["p_over"], version)
            n += 1
    for r in card["match"].values():
        spec = MarketSpec(**{k: v for k, v in r["spec"].items() if v is not None})
        log_prediction(store, event_key, teams, match_date, spec, r["line"], r["p_over"], version)
        n += 1
    return n


def settle(store: Store, matches: list[BBMatch], day_tolerance: int = 1) -> dict:
    """Settle open predictions against real scorecards (same teams, date within tolerance)."""
    init(store)
    by_key: dict = {}
    for m in matches:
        by_key.setdefault(frozenset(norm(t) for t in m.teams), []).append(m)
    rows = store._conn.execute("SELECT * FROM market_predictions WHERE status = 'open'").fetchall()
    settled = void = unmatched = 0
    with store.tx() as c:
        for r in rows:
            cands = by_key.get(frozenset((norm(r["team_a"]), norm(r["team_b"]))), [])
            d = date.fromisoformat(r["match_date"])
            m = next((x for x in cands if abs((x.date - d).days) <= day_tolerance), None)
            if m is None:
                unmatched += 1
                continue
            spec_d = json.loads(r["spec"])
            spec = MarketSpec(**{k: v for k, v in spec_d.items() if v is not None})
            # map the prediction's team names onto the scorecard's spelling
            if spec.team:
                real = next((t for t in m.teams if norm(t) == norm(spec.team)), spec.team)
                spec = MarketSpec(**(asdict(spec) | {"team": real}))
            out = settle_market(m, spec, r["line"])
            status = "void" if out is None else "settled"
            c.execute("UPDATE market_predictions SET outcome = ?, status = ?, settled_match = ?, settled_at = ? WHERE id = ?",
                      (out, status, m.match_id, now_iso(), r["id"]))
            settled += out is not None
            void += out is None
    return {"settled": settled, "void": void, "still_open": unmatched}


def _settled(store: Store, market_type: str | None = None) -> list:
    init(store)
    q = "SELECT * FROM market_predictions WHERE status = 'settled'"
    args: tuple = ()
    if market_type:
        q += " AND market_type = ?"
        args = (market_type,)
    return store._conn.execute(q + " ORDER BY match_date, id", args).fetchall()


def _scores(p: np.ndarray, y: np.ndarray) -> dict:
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    buckets = []
    for lo in np.arange(0, 1, 0.2):
        mask = (p >= lo) & (p < lo + 0.2 + (1e-9 if lo >= 0.8 else 0))
        if mask.sum():
            buckets.append({"from": round(float(lo), 1), "to": round(float(lo + 0.2), 1), "n": int(mask.sum()),
                            "predicted": round(float(p[mask].mean()), 3), "actual": round(float(y[mask].mean()), 3)})
    return {"n": int(len(y)), "brier": float(np.mean((p - y) ** 2)),
            "log_loss": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))),
            "mean_predicted": float(p.mean()), "actual_rate": float(y.mean()), "buckets": buckets}


def report(store: Store) -> dict:
    """How well-calibrated each market type has been on settled predictions."""
    out = {}
    rows = _settled(store)
    types = sorted({r["market_type"] for r in rows})
    for t in types:
        rs = [r for r in rows if r["market_type"] == t]
        out[t] = _scores(np.array([r["p_over"] for r in rs]), np.array([r["outcome"] for r in rs], dtype=float))
    return out


def fit_calibrators(store: Store, models_dir: str | Path, min_n: int = MIN_CALIBRATION) -> dict:
    """Fit an isotonic correction per market type with enough settled history."""
    fitted, skipped = {}, {}
    for t in {r["market_type"] for r in _settled(store)}:
        rs = _settled(store, t)
        if len(rs) < min_n:
            skipped[t] = f"{len(rs)} settled, need {min_n}"
            continue
        p = np.array([r["p_over"] for r in rs])
        y = np.array([r["outcome"] for r in rs], dtype=float)
        cal = directional_calibrator(p, y)
        before = float(np.mean((p - y) ** 2))
        # honest check: fit on the older 70%, test on the newest 30%
        k = int(len(y) * 0.7)
        holdout = None
        if k >= min_n // 2:
            cal_tr = directional_calibrator(p[:k], y[:k])
            adj = cal_tr.predict(p[k:])
            holdout = {"raw_brier": float(np.mean((p[k:] - y[k:]) ** 2)), "calibrated_brier": float(np.mean((adj - y[k:]) ** 2))}
            if holdout["calibrated_brier"] > holdout["raw_brier"]:
                skipped[t] = "calibration did not help on the newest predictions"
                continue
        fitted[t] = {"model": cal, "n": len(y), "brier_before": before, "holdout": holdout}
    path = Path(models_dir) / "ball"
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump({t: v["model"] for t, v in fitted.items()}, path / "calibrators.joblib")
    return {"fitted": {t: {k: v for k, v in d.items() if k != "model"} for t, d in fitted.items()}, "skipped": skipped}


def directional_calibrator(p: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    """Isotonic map from predicted P(over) to observed frequency.

    Deliberately NOT symmetric: over/under markets fail in one direction
    (e.g. "sixes always predicted too high"), and mirroring each prediction
    as its opposite would cancel exactly that bias out. P(under) is always
    derived as 1 - calibrated P(over), so the pair stays consistent.
    """
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4)
    iso.fit(p, y)
    return iso


def load_calibrators(models_dir: str | Path) -> dict:
    f = Path(models_dir) / "ball" / "calibrators.joblib"
    if not f.exists():
        return {}
    return joblib.load(f)   # only files this tool wrote itself


def drift(store: Store, window: int = DRIFT_WINDOW) -> dict:
    """Compare each market type's latest `window` settled predictions with everything before."""
    flags = {}
    for t in sorted({r["market_type"] for r in _settled(store)}):
        rs = _settled(store, t)
        if len(rs) < 2 * window:
            continue
        p = np.array([r["p_over"] for r in rs])
        y = np.array([r["outcome"] for r in rs], dtype=float)
        old = np.mean((p[:-window] - y[:-window]) ** 2)
        new = np.mean((p[-window:] - y[-window:]) ** 2)
        gap = abs(p[-window:].mean() - y[-window:].mean())
        reasons = []
        if new > old * 1.15:
            reasons.append(f"Brier rose from {old:.3f} to {new:.3f}")
        if gap > 0.06:
            reasons.append(f"recent predictions average {p[-window:].mean():.0%} but {y[-window:].mean():.0%} happened")
        if reasons:
            flags[t] = reasons
    return {"drifting": flags, "retrain_recommended": bool(flags)}


def recent_days(store: Store, days: int = 30) -> list:
    init(store)
    since = (date.today() - timedelta(days=days)).isoformat()
    return store._conn.execute("SELECT * FROM market_predictions WHERE match_date >= ? ORDER BY id DESC",
                               (since,)).fetchall()

"""Ensemble win-probability model, walk-forward evaluation, calibration.

Members
- elo:   raw Elo probability (the baseline every challenger must beat)
- logit: logistic regression on the antisymmetric features
- hgb:   small, heavily regularised gradient-boosted trees

The ensemble blends the three with weights chosen on a held-out slice of the
training window (never on the test year). Predictions are symmetrised:
p(a beats b) = (f(x) + 1 - f(-x)) / 2, so swapping team order is consistent.

Calibration: isotonic regression only when at least MIN_CAL out-of-sample
predictions exist; below that it overfits, so the identity is used.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import date

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .cricsheet import MatchRecord
from .features import FEATURE_NAMES, FeatureBuilder, flip

MIN_CAL = 500
EPS = 1e-6


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    elo_p: np.ndarray
    dates: np.ndarray  # datetime64[D]
    pools: np.ndarray
    min_games: np.ndarray  # min games played by either side before the match


def build_dataset(matches: list[MatchRecord], builder: FeatureBuilder | None = None) -> tuple[Dataset, FeatureBuilder]:
    builder = builder or FeatureBuilder()
    X, y, ep, ds, pools, mg = [], [], [], [], [], []
    for m, x, p in builder.walk(matches):
        X.append(x)
        y.append(1 if m.winner == m.team_a else 0)
        ep.append(p)
        ds.append(np.datetime64(m.date, "D"))
        pools.append(m.pool)
        mg.append(min(builder.games_played(m.pool, m.team_a), builder.games_played(m.pool, m.team_b)) - 1)
    ds_arr = np.array(ds, dtype="datetime64[D]") if ds else np.array([], dtype="datetime64[D]")
    return Dataset(np.array(X, dtype=float).reshape(-1, len(FEATURE_NAMES)), np.array(y), np.array(ep),
                   ds_arr, np.array(pools), np.array(mg)), builder


def _augment(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.vstack([X, -X]), np.concatenate([y, 1 - y])


def _clip(p):
    return np.clip(p, EPS, 1 - EPS)


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = _clip(p)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error over equal-width bins."""
    idx = np.minimum((p * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            total += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(total)


def metrics(p: np.ndarray, y: np.ndarray) -> dict:
    return {
        "n": int(len(y)),
        "log_loss": log_loss(p, y),
        "brier": brier(p, y),
        "ece": ece(p, y),
        "accuracy": float(np.mean((p > 0.5) == (y == 1))),
    }


@dataclass
class Ensemble:
    logit: LogisticRegression = field(default_factory=lambda: LogisticRegression(C=1.0, max_iter=1000))
    hgb: HistGradientBoostingClassifier = field(default_factory=lambda: HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=80, l2_regularization=1.0,
        early_stopping=False, random_state=0))
    weights: tuple[float, float, float] = (1.0, 0.0, 0.0)  # elo, logit, hgb
    calibrator: IsotonicRegression | None = None
    fitted: bool = False

    def _members(self, X: np.ndarray, elo_p: np.ndarray) -> np.ndarray:
        def sym(model):
            return (model.predict_proba(X)[:, 1] + 1 - model.predict_proba(-X)[:, 1]) / 2

        return np.column_stack([elo_p, sym(self.logit), sym(self.hgb)])

    def fit(self, X: np.ndarray, y: np.ndarray, elo_p: np.ndarray, val_frac: float = 0.2) -> "Ensemble":
        n = len(y)
        if n < 50:
            raise ValueError(f"need at least 50 training matches, got {n}")
        cut = int(n * (1 - val_frac))
        # 1) fit members on the early part, choose blend weights on the later part (time order kept)
        Xa, ya = _augment(X[:cut], y[:cut])
        self.logit.fit(Xa, ya)
        self.hgb.fit(Xa, ya)
        P = self._members(X[cut:], elo_p[cut:])
        self.weights = _best_weights(P, y[cut:])
        # 2) refit members on the whole window with the chosen weights
        Xa, ya = _augment(X, y)
        self.logit.fit(Xa, ya)
        self.hgb.fit(Xa, ya)
        self.fitted = True
        return self

    def predict_raw(self, X: np.ndarray, elo_p: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("model not fitted")
        return _clip(self._members(X, elo_p) @ np.array(self.weights))

    def predict(self, X: np.ndarray, elo_p: np.ndarray) -> np.ndarray:
        p = self.predict_raw(X, elo_p)
        if self.calibrator is not None:
            # symmetric application keeps p(a beats b) + p(b beats a) == 1
            c = self.calibrator.predict
            p = _clip((c(p) + 1 - c(1 - p)) / 2)
        return p

    def member_predictions(self, X: np.ndarray, elo_p: np.ndarray) -> dict[str, np.ndarray]:
        P = self._members(X, elo_p)
        return {"elo": P[:, 0], "logit": P[:, 1], "hgb": P[:, 2], "ensemble": _clip(P @ np.array(self.weights))}


def _best_weights(P: np.ndarray, y: np.ndarray, step: float = 0.1) -> tuple[float, float, float]:
    grid = np.round(np.arange(0, 1 + 1e-9, step), 2)
    best, best_w = float("inf"), (1.0, 0.0, 0.0)
    for w0, w1 in itertools.product(grid, grid):
        w2 = round(1 - w0 - w1, 2)
        if w2 < -1e-9:
            continue
        w = np.array([w0, w1, max(w2, 0.0)])
        ll = log_loss(P @ w, y)
        if ll < best - 1e-12:
            best, best_w = ll, (float(w0), float(w1), float(max(w2, 0.0)))
    return best_w


def fit_calibrator(p: np.ndarray, y: np.ndarray) -> IsotonicRegression | None:
    """Isotonic recalibration, but only with enough out-of-sample data."""
    if len(y) < MIN_CAL:
        return None
    iso = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1 - EPS)
    # fit on both orientations of every match so the map is centred at 0.5
    iso.fit(np.concatenate([p, 1 - p]), np.concatenate([y, 1 - y]))
    return iso


def walk_forward_eval(ds: Dataset, first_test_year: int, min_games: int = 5, min_train: int = 200) -> dict:
    """Expanding window by calendar year: train on everything before year Y, test on Y.

    Only matches where both sides have >= min_games history are scored,
    because nobody should bet a model on teams it has barely seen.
    """
    years = sorted({int(str(d)[:4]) for d in ds.dates})
    preds = {k: [] for k in ("elo", "logit", "hgb", "ensemble")}
    ys, fold_info = [], []
    for year in [y for y in years if y >= first_test_year]:
        start = np.datetime64(f"{year}-01-01")
        end = np.datetime64(f"{year + 1}-01-01")
        tr = ds.dates < start
        te = (ds.dates >= start) & (ds.dates < end) & (ds.min_games >= min_games)
        if tr.sum() < min_train or te.sum() == 0:
            continue
        model = Ensemble().fit(ds.X[tr], ds.y[tr], ds.elo_p[tr])
        mp = model.member_predictions(ds.X[te], ds.elo_p[te])
        for k in preds:
            preds[k].append(mp[k])
        ys.append(ds.y[te])
        fold_info.append({"year": year, "train": int(tr.sum()), "test": int(te.sum()), "weights": model.weights})
    if not ys:
        return {"folds": [], "models": {}, "n": 0}
    y = np.concatenate(ys)
    out = {k: metrics(np.concatenate(v), y) for k, v in preds.items()}
    out["coinflip"] = metrics(np.full(len(y), 0.5), y)
    ens = np.concatenate(preds["ensemble"])
    cal = fit_calibrator(ens, y)
    return {
        "n": int(len(y)),
        "folds": fold_info,
        "models": out,
        "calibration_available": cal is not None,
        "oos_ensemble": ens,  # kept for calibrator fitting by the trainer
        "oos_y": y,
    }


@dataclass
class Predictor:
    """Everything needed to price an upcoming match: fitted model + feature state."""
    version: str
    builder: FeatureBuilder
    model: Ensemble
    trained_through: date
    min_games: int = 5
    eval_summary: dict = field(default_factory=dict)

    def predict(self, pool: str, a: str, b: str, d: date) -> tuple[float | None, str]:
        ga, gb = self.builder.games_played(pool, a), self.builder.games_played(pool, b)
        if min(ga, gb) < self.min_games:
            missing = a if ga < gb else b
            return None, f"not enough history for {missing} in {pool} ({min(ga, gb)} matches)"
        x, ep = self.builder.features_for(pool, a, b, d)
        p = float(self.model.predict(np.array([x]), np.array([ep]))[0])
        return p, "ok"

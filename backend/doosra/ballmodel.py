"""Per-delivery outcome model: P(dot, 1, 2, 3, 4, 6, wicket, extra | situation).

Two learners are trained and the better one (by out-of-sample log loss) is
kept; the trivial baseline is the pool's class frequencies. Everything else
(totals, sessions, player runs, the winner) is derived by simulation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .ballbyball import N_CLASSES

EPS = 1e-9


def multiclass_log_loss(P: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], EPS, 1))))


@dataclass
class BallModel:
    kind: str = "hgb"            # "hgb" | "logit" | "baseline"
    max_iter: int = 150
    model: object | None = None
    prior: np.ndarray | None = None
    keep: np.ndarray | None = None   # feature columns with any data at fit time
    fitted: bool = False

    def _make(self):
        if self.kind == "hgb":
            return HistGradientBoostingClassifier(
                max_iter=self.max_iter, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=200,
                l2_regularization=1.0, early_stopping=False, random_state=0)
        if self.kind == "logit":
            return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=600))
        return None

    def _prep(self, X: np.ndarray) -> np.ndarray:
        X = X[:, self.keep]
        return X if self.kind == "hgb" else np.nan_to_num(X)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BallModel":
        self.prior = np.bincount(y, minlength=N_CLASSES) + 1.0
        self.prior = self.prior / self.prior.sum()
        # Columns that are entirely missing (e.g. weather before any is fetched) carry no
        # information and break tree binning; drop them and remember the choice.
        self.keep = ~np.all(np.isnan(X), axis=0)
        if self.kind != "baseline":
            self.model = self._make()
            self.model.fit(self._prep(X), y)
        self.fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("ball model not fitted")
        if self.kind == "baseline":
            return np.tile(self.prior, (len(X), 1))
        raw = self.model.predict_proba(self._prep(X))
        P = np.full((len(X), N_CLASSES), EPS)
        P[:, self.model.classes_ if self.kind == "hgb" else self.model[-1].classes_] = raw
        return P / P.sum(axis=1, keepdims=True)


@dataclass
class BallModelChoice:
    """The fitted winner plus the evidence for choosing it."""
    model: BallModel
    scores: dict = field(default_factory=dict)


def fit_best(X_tr, y_tr, X_val, y_val, kinds=("hgb", "logit"), max_iter: int = 150) -> BallModelChoice:
    scores, fitted = {}, {}
    for k in ("baseline",) + tuple(kinds):
        m = BallModel(kind=k, max_iter=max_iter).fit(X_tr, y_tr)
        scores[k] = multiclass_log_loss(m.predict_proba(X_val), y_val)
        fitted[k] = m
    best = min(kinds, key=lambda k: scores[k])
    final = BallModel(kind=best, max_iter=max_iter).fit(np.vstack([X_tr, X_val]), np.concatenate([y_tr, y_val]))
    return BallModelChoice(final, scores | {"chosen": best})

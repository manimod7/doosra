"""Champion / challenger model registry with a promotion gate.

A retrained model (the challenger) replaces the live one (the champion) only
when it is measurably better out of sample. Without this gate, "keep
retraining" quietly turns into "keep overfitting".
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib

from .model import Predictor

MIN_N = 300            # out-of-sample matches required before any promotion
MIN_GAIN = 0.002       # log-loss improvement required over the champion
MUST_BEAT_ELO = True   # never ship something worse than the simple baseline


@dataclass(frozen=True)
class Decision:
    promote: bool
    reason: str


def gate(challenger: dict, champion: dict | None, elo: dict | None, clv_challenger: float | None = None,
         clv_champion: float | None = None) -> Decision:
    """challenger/champion/elo are metrics dicts with 'n' and 'log_loss'."""
    if challenger["n"] < MIN_N:
        return Decision(False, f"only {challenger['n']} out-of-sample matches; need {MIN_N}")
    if MUST_BEAT_ELO and elo is not None and challenger["log_loss"] >= elo["log_loss"]:
        return Decision(False, f"log loss {challenger['log_loss']:.4f} does not beat Elo {elo['log_loss']:.4f}")
    if champion is not None:
        gain = champion["log_loss"] - challenger["log_loss"]
        if gain < MIN_GAIN:
            return Decision(False, f"gain over champion {gain:+.4f} below required {MIN_GAIN}")
        if clv_challenger is not None and clv_champion is not None and clv_challenger < clv_champion:
            return Decision(False, f"closing-line value worse than champion ({clv_challenger:+.3f} vs {clv_champion:+.3f})")
    return Decision(True, "beats Elo" + (" and the champion" if champion else "") + " out of sample")


@dataclass
class Registry:
    root: str | Path

    def __post_init__(self):
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index = self.root / "champion.json"

    def champion_meta(self) -> dict | None:
        if not self.index.exists():
            return None
        return json.loads(self.index.read_text())

    def load_champion(self) -> Predictor | None:
        meta = self.champion_meta()
        if not meta:
            return None
        path = self.root / meta["file"]
        if not path.exists():
            return None
        # Only ever load artefacts this tool wrote itself into its own models folder.
        return joblib.load(path)

    def save(self, predictor: Predictor, metrics: dict, decision: Decision) -> dict:
        fname = f"model-{predictor.version}.joblib"
        joblib.dump(predictor, self.root / fname)
        meta = {
            "version": predictor.version,
            "file": fname,
            "trained_through": predictor.trained_through.isoformat(),
            "metrics": metrics,
            "decision": decision.reason,
        }
        (self.root / f"model-{predictor.version}.json").write_text(json.dumps(meta, indent=2))
        if decision.promote:
            self.index.write_text(json.dumps(meta, indent=2))
        return meta

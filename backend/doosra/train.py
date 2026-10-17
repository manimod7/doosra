"""Train, evaluate and (maybe) promote a model from Cricsheet data.

Usage:
  python -I -m doosra.train /path/to/all_json.zip --models-dir models \
      [--pools T20:male ODI:male] [--first-test-year 2016]

Steps
1. Parse decided matches, keep the requested rating pools.
2. Walk-forward evaluation by calendar year (no future data in any fold).
3. Print every member vs Elo vs a coin flip.
4. Fit the final model on all data, add isotonic calibration if enough
   out-of-sample predictions exist, and run the promotion gate.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

from .cricsheet import load_matches
from .model import Ensemble, Predictor, build_dataset, fit_calibrator, walk_forward_eval
from .registry import Registry, gate

DEFAULT_POOLS = ("T20:male", "ODI:male", "T20:female", "ODI:female")


def run(path: str, models_dir: str, pools: tuple[str, ...], first_test_year: int, min_games: int = 5) -> dict:
    matches = [m for m in load_matches(path) if m.pool in pools]
    if len(matches) < 300:
        raise SystemExit(f"only {len(matches)} decided matches in pools {pools}; need at least 300")
    ds, builder = build_dataset(matches)
    ev = walk_forward_eval(ds, first_test_year=first_test_year, min_games=min_games)
    if not ev["n"]:
        raise SystemExit("no evaluation folds produced; lower --first-test-year")

    model = Ensemble().fit(ds.X, ds.y, ds.elo_p)
    model.calibrator = fit_calibrator(ev["oos_ensemble"], ev["oos_y"])
    version = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    predictor = Predictor(version=version, builder=builder, model=model,
                          trained_through=matches[-1].date, min_games=min_games,
                          eval_summary={k: v for k, v in ev["models"].items()})

    reg = Registry(models_dir)
    champ = reg.champion_meta()
    decision = gate(ev["models"]["ensemble"], champ["metrics"] if champ else None, ev["models"]["elo"])
    meta = reg.save(predictor, ev["models"]["ensemble"], decision)
    return {"matches": len(matches), "eval": ev, "decision": decision, "meta": meta}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="doosra.train")
    ap.add_argument("data")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--pools", nargs="+", default=list(DEFAULT_POOLS))
    ap.add_argument("--first-test-year", type=int, default=2016)
    ap.add_argument("--min-games", type=int, default=5)
    a = ap.parse_args(argv[1:])
    res = run(a.data, a.models_dir, tuple(a.pools), a.first_test_year, a.min_games)
    ev = res["eval"]
    print(f"decided matches used: {res['matches']}, scored out of sample: {ev['n']}")
    print(f"{'model':<10}{'log loss':>10}{'brier':>9}{'ECE':>8}{'acc':>8}")
    for name in ("coinflip", "elo", "logit", "hgb", "ensemble"):
        m = ev["models"][name]
        print(f"{name:<10}{m['log_loss']:>10.4f}{m['brier']:>9.4f}{m['ece']:>8.3f}{m['accuracy']:>8.1%}")
    print("last fold weights (elo, logit, hgb):", ev["folds"][-1]["weights"])
    print("isotonic calibration:", "on" if ev["calibration_available"] else f"off (fewer than 500 OOS matches)")
    d = res["decision"]
    print(("PROMOTED: " if d.promote else "NOT promoted: ") + d.reason)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

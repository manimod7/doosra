"""One learning cycle, end to end. Run after each new Cricsheet download:

  python -I -m doosra.pipeline cycle  /path/to/t20s_json.zip [--models-dir models] [--weather]

Steps
1. Load ball-by-ball matches for the chosen pools.
2. Settle every open prediction against the new scorecards.
3. Re-fit per-market calibration where enough predictions have settled.
4. Check for drift in each market type.
5. Retrain the ball model (walk-forward evaluation + promotion gate).
6. Retrain the match-winner model the same way.
Each step reports what it did; nothing is promoted without passing its gate.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import httpx

from . import feedback
from . import modelsync
from .ballbyball import load_bb_matches
from .conditions import Weather
from .lifecycle import MIN_BALLS, retrain_ball
from .storage import Store
from .train import run as retrain_match_model

CRICSHEET_ALLOWED = "https://cricsheet.org/downloads/"


def download_cricsheet(name: str, dest_dir: str | Path, transport: httpx.BaseTransport | None = None) -> Path:
    """Fetch a Cricsheet archive (e.g. 't20s_json.zip'). Only cricsheet.org downloads are allowed."""
    if "/" in name or not name.endswith(".zip"):
        raise ValueError("give an archive name like t20s_json.zip")
    url = CRICSHEET_ALLOWED + name
    dest = Path(dest_dir) / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=120, transport=transport, follow_redirects=False) as c:
        r = c.get(url)
        r.raise_for_status()
        dest.write_bytes(r.content)
    return dest


def cycle(data_path: str | Path, store: Store, models_dir: str | Path, pools: tuple[str, ...] = ("T20:male",),
          use_weather: bool = False, test_years: int = 3, kinds=("hgb", "logit"), max_iter: int = 150,
          sims_per_year: int = 40, n_sims: int = 400, min_balls: int = MIN_BALLS,
          retrain_winner_model: bool = True) -> dict:
    matches = load_bb_matches(data_path, pools)
    out: dict = {"matches": len(matches), "through": matches[-1].date.isoformat() if matches else None}
    if not matches:
        raise ValueError(f"no matches for pools {pools} in {data_path}")
    out["settle"] = feedback.settle(store, matches)
    out["calibration"] = feedback.fit_calibrators(store, models_dir)
    out["drift"] = feedback.drift(store)
    out["market_report"] = feedback.report(store)
    weather = Weather(store).for_matches(matches) if use_weather else None
    out["weather_matches"] = len(weather or {})
    res = retrain_ball(matches, models_dir, pools, test_years, kinds, max_iter, sims_per_year, n_sims,
                       min_balls, weather)
    out["ball_model"] = {"promoted": res["promoted"], "reason": res["reason"],
                         "report": {k: v for k, v in res["report"].items() if k not in ("folds",)},
                         "folds": res["report"].get("folds", [])}
    if retrain_winner_model:
        try:
            wm = retrain_match_model(str(data_path), str(models_dir), pools, first_test_year=matches[0].date.year + 2)
            out["winner_model"] = {"promoted": wm["decision"].promote, "reason": wm["decision"].reason,
                                   "ensemble": wm["eval"]["models"]["ensemble"], "elo": wm["eval"]["models"]["elo"]}
        except (SystemExit, ValueError) as e:
            out["winner_model"] = {"promoted": False, "reason": str(e)}
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="doosra.pipeline")
    ap.add_argument("command", choices=["cycle", "download"])
    ap.add_argument("data", help="Cricsheet zip path (cycle) or archive name (download)")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--db", default=os.environ.get("DOOSRA_DB", "data/doosra.sqlite"),
                    help="SQLite path or postgresql:// URL (default: $DOOSRA_DB)")
    ap.add_argument("--pools", nargs="+", default=["T20:male"])
    ap.add_argument("--weather", action="store_true")
    ap.add_argument("--test-years", type=int, default=3)
    a = ap.parse_args(argv[1:])
    if a.command == "download":
        print(download_cricsheet(a.data, "data/raw"))
        return 0
    store = Store(a.db)
    if store.is_remote:                 # start from the live champion, and publish whatever wins
        res = {"models_pulled": modelsync.pull(store, a.models_dir)}
        res |= cycle(a.data, store, a.models_dir, tuple(a.pools), a.weather, a.test_years)
        res["models_pushed"] = modelsync.push(store, a.models_dir)
    else:
        res = cycle(a.data, store, a.models_dir, tuple(a.pools), a.weather, a.test_years)
    print(json.dumps(res, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

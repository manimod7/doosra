"""Walk-forward Elo backtest over parsed Cricsheet matches.

Usage:  python -I -m doosra.backtest /path/to/ipl_json.zip
"""
from __future__ import annotations

import sys

from .cricsheet import MatchRecord, load_matches
from .elo import EloModel, brier, log_loss


def walk_forward(matches: list[MatchRecord], warmup: int = 100, **elo_kwargs) -> dict:
    model = EloModel(**elo_kwargs)
    probs: list[float] = []
    outcomes: list[int] = []
    current_season = None
    for i, m in enumerate(matches):
        if current_season is not None and m.season != current_season:
            model.new_season()
        current_season = m.season
        p_a = model.predict(m.team_a, m.team_b)
        a_won = m.winner == m.team_a
        if i >= warmup:  # score only after ratings have had time to form
            probs.append(p_a)
            outcomes.append(1 if a_won else 0)
        model.update(m.team_a, m.team_b, a_won, p_a)
    n = len(probs)
    baseline = [0.5] * n
    return {
        "n_scored": n,
        "log_loss": log_loss(probs, outcomes) if n else float("nan"),
        "brier": brier(probs, outcomes) if n else float("nan"),
        "coinflip_log_loss": log_loss(baseline, outcomes) if n else float("nan"),
        "coinflip_brier": brier(baseline, outcomes) if n else float("nan"),
        "accuracy": sum((p > 0.5) == bool(o) for p, o in zip(probs, outcomes)) / n if n else float("nan"),
        "final_ratings": dict(sorted(model.ratings.items(), key=lambda kv: -kv[1])),
    }


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python -I -m doosra.backtest <cricsheet zip or dir>")
        return 2
    matches = load_matches(argv[1])
    if not matches:
        print("no decided matches found")
        return 1
    res = walk_forward(matches)
    print(f"matches parsed: {len(matches)}, scored: {res['n_scored']}")
    print(f"Elo   log loss {res['log_loss']:.4f}  brier {res['brier']:.4f}  acc {res['accuracy']:.1%}")
    print(f"Coin  log loss {res['coinflip_log_loss']:.4f}  brier {res['coinflip_brier']:.4f}")
    print("Top ratings:", list(res["final_ratings"].items())[:5])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

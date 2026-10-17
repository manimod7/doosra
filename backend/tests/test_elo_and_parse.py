import json
import random
import zipfile

import pytest

from doosra.backtest import walk_forward
from doosra.cricsheet import load_matches, parse_match
from doosra.elo import EloModel, brier, log_loss


def _match(mid, d, a, b, winner, season="2020"):
    info = {
        "teams": [a, b], "dates": [d], "season": season,
        "venue": "Test Ground", "toss": {"winner": a, "decision": "bat"},
        "event": {"name": "Test League"},
    }
    if winner:
        info["outcome"] = {"winner": winner}
    else:
        info["outcome"] = {"result": "no result"}
    return mid, {"info": info}


def test_parse_skips_undecided():
    mid, data = _match("1", "2020-03-01", "A", "B", None)
    assert parse_match(mid, data) is None


def test_parse_decided():
    mid, data = _match("2", "2020-03-02", "A", "B", "B")
    rec = parse_match(mid, data)
    assert rec and rec.winner == "B" and rec.season == "2020"


def test_load_from_zip_sorted(tmp_path):
    zpath = tmp_path / "x.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        for mid, d, w in [("b", "2020-05-02", "A"), ("a", "2020-05-01", "B")]:
            _, data = _match(mid, d, "A", "B", w)
            zf.writestr(f"{mid}.json", json.dumps(data))
    recs = load_matches(zpath)
    assert [r.match_id for r in recs] == ["a", "b"]


def test_elo_update_is_zero_sum():
    m = EloModel()
    m.update("A", "B", True)
    assert m.rating("A") + m.rating("B") == pytest.approx(3000.0)
    assert m.rating("A") > 1500 > m.rating("B")


def test_home_advantage_shifts_probability():
    m = EloModel()
    assert m.predict("A", "B", a_home=True) > 0.5 > m.predict("A", "B", b_home=True)


def test_walk_forward_beats_coinflip_on_synthetic_strengths():
    """Synthetic league with known team strengths: Elo must beat 0.5."""
    rng = random.Random(7)
    strength = {"A": 0.8, "B": 0.6, "C": 0.5, "D": 0.4, "E": 0.2}
    teams = list(strength)
    rows = []
    for i in range(600):
        a, b = rng.sample(teams, 2)
        p = strength[a] / (strength[a] + strength[b])
        winner = a if rng.random() < p else b
        day = 1 + i % 28
        month = 1 + (i // 28) % 12
        year = 2015 + i // 336
        _, data = _match(str(i), f"{year}-{month:02d}-{day:02d}", a, b, winner, season=str(year))
        rows.append(parse_match(str(i), data))
    res = walk_forward(sorted(rows, key=lambda r: (r.date, r.match_id)), warmup=100)
    assert res["n_scored"] == 500
    assert res["log_loss"] < res["coinflip_log_loss"]
    assert res["brier"] < res["coinflip_brier"]
    top = list(res["final_ratings"])[0]
    assert top == "A"


def test_metrics_perfect_and_worst():
    assert log_loss([0.999999], [1]) < 1e-5
    assert brier([0.0], [1]) == 1.0

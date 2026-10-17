import json
import random
import zipfile
from datetime import date, timedelta

import numpy as np
import pytest

from doosra.cricsheet import MatchRecord
from doosra.features import FeatureBuilder, flip
from doosra.model import Ensemble, build_dataset, fit_calibrator, walk_forward_eval
from doosra.registry import Registry, gate
from doosra.train import run


def synthetic_league(n=2400, seed=3, pool=("T20", "male")):
    """Teams with drifting latent strength. Outcomes follow a known logistic law."""
    rng = random.Random(seed)
    teams = [f"T{i}" for i in range(10)]
    s = {t: rng.gauss(0, 1) for t in teams}
    d = date(2010, 1, 1)
    out = []
    for i in range(n):
        for t in teams:
            s[t] += rng.gauss(0, 0.03)
        a, b = rng.sample(teams, 2)
        p = 1 / (1 + np.exp(-(s[a] - s[b])))
        winner = a if rng.random() < p else b
        d += timedelta(days=rng.choice([1, 2, 3]))
        out.append(MatchRecord(str(i), d, str(d.year), a, b, winner, None, None, None, None, pool[0], pool[1]))
    return out


def test_features_are_leak_free_and_antisymmetric():
    ms = synthetic_league(50)
    fb = FeatureBuilder()
    rows = list(fb.walk(ms))
    first_x = rows[0][1]
    assert first_x == [0.0, 0.0, 0.0, 0.0, 0.0]  # nothing known before match 1
    x, _ = fb.features_for("T20:male", "T0", "T1", date(2030, 1, 1))
    y, _ = fb.features_for("T20:male", "T1", "T0", date(2030, 1, 1))
    assert np.allclose(x, flip(y))


def test_pools_do_not_share_ratings():
    t20 = synthetic_league(200, pool=("T20", "male"))
    fb = FeatureBuilder()
    list(fb.walk(t20))
    assert "ODI:male" not in fb.pools
    assert fb.games_played("ODI:male", "T0") == 0


def test_ensemble_symmetry_and_beats_coinflip():
    ds, _ = build_dataset(synthetic_league())
    cut = 1800
    m = Ensemble().fit(ds.X[:cut], ds.y[:cut], ds.elo_p[:cut])
    p = m.predict(ds.X[cut:], ds.elo_p[cut:])
    q = m.predict(-ds.X[cut:], 1 - ds.elo_p[cut:])
    assert np.allclose(p + q, 1.0, atol=1e-9)  # swapping teams is consistent
    y = ds.y[cut:]
    ll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    assert ll < np.log(2)
    assert abs(sum(m.weights) - 1) < 1e-9


def test_walk_forward_has_no_future_training_and_reports_all_models():
    ds, _ = build_dataset(synthetic_league())
    ev = walk_forward_eval(ds, first_test_year=2013)
    assert ev["n"] > 300
    for f in ev["folds"]:
        assert f["train"] > 0 and f["test"] > 0
    years = [f["year"] for f in ev["folds"]]
    assert years == sorted(years)
    for k in ("elo", "logit", "hgb", "ensemble", "coinflip"):
        assert k in ev["models"]
    assert ev["models"]["elo"]["log_loss"] < ev["models"]["coinflip"]["log_loss"]


def test_calibrator_needs_enough_data():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.2, 0.8, 100)
    assert fit_calibrator(p, (rng.uniform(size=100) < p).astype(int)) is None
    p = rng.uniform(0.2, 0.8, 800)
    assert fit_calibrator(p, (rng.uniform(size=800) < p).astype(int)) is not None


def test_gate_rules():
    elo = {"n": 1000, "log_loss": 0.66}
    assert not gate({"n": 100, "log_loss": 0.60}, None, elo).promote            # too few matches
    assert not gate({"n": 1000, "log_loss": 0.67}, None, elo).promote           # worse than Elo
    assert gate({"n": 1000, "log_loss": 0.65}, None, elo).promote               # first champion
    champ = {"n": 1000, "log_loss": 0.650}
    assert not gate({"n": 1000, "log_loss": 0.649}, champ, elo).promote         # gain too small
    assert gate({"n": 1000, "log_loss": 0.640}, champ, elo).promote
    assert not gate({"n": 1000, "log_loss": 0.640}, champ, elo, clv_challenger=-0.01, clv_champion=0.01).promote


def _write_zip(path, matches):
    with zipfile.ZipFile(path, "w") as zf:
        for m in matches:
            info = {"teams": [m.team_a, m.team_b], "dates": [m.date.isoformat()], "season": m.season,
                    "outcome": {"winner": m.winner}, "match_type": m.match_type, "gender": m.gender}
            zf.writestr(f"{m.match_id}.json", json.dumps({"info": info}))


@pytest.mark.slow
def test_train_end_to_end_promotes_then_requires_real_gain(tmp_path):
    z = tmp_path / "all.zip"
    _write_zip(z, synthetic_league())
    res = run(str(z), str(tmp_path / "models"), ("T20:male",), first_test_year=2013)
    assert res["eval"]["n"] > 300
    reg = Registry(tmp_path / "models")
    assert res["decision"].promote, res["decision"].reason  # synthetic league has real signal
    if res["decision"].promote:
        pred = reg.load_champion()
        p, why = pred.predict("T20:male", "T0", "T1", date(2030, 1, 1))
        assert why == "ok" and 0 < p < 1
        p2, _ = pred.predict("T20:male", "T1", "T0", date(2030, 1, 1))
        assert p + p2 == pytest.approx(1.0, abs=1e-6)
        unknown, reason = pred.predict("T20:male", "T0", "Nobody", date(2030, 1, 1))
        assert unknown is None and "not enough history" in reason
        # retraining on identical data must NOT replace the champion (no real gain)
        res2 = run(str(z), str(tmp_path / "models"), ("T20:male",), first_test_year=2013)
        assert not res2["decision"].promote
        assert reg.champion_meta()["version"] == res["meta"]["version"]

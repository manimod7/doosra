from datetime import date

import httpx
import numpy as np
import pytest

from doosra import feedback
from doosra.conditions import Weather, coords_for, summarise
from doosra.lifecycle import BallRegistry, coverage_problem, gate, retrain_ball
from doosra.pipeline import cycle, download_cricsheet
from doosra.props import MarketSpec
from doosra.storage import Store

GOOD = {"n_balls": 50000, "ball_log_loss": 1.60, "baseline_log_loss": 1.62, "sim_matches": 100,
        "total_coverage80": 0.79}


# ---------------------------------------------------------------- gate

def test_gate_accepts_a_sound_first_model():
    ok, why = gate(GOOD, None, date(2020, 1, 1), min_balls=20000)
    assert ok, why


@pytest.mark.parametrize("change, fragment", [
    ({"n_balls": 100}, "out-of-sample deliveries"),
    ({"ball_log_loss": 1.63}, "does not beat"),
    ({"total_coverage80": 0.55}, "too narrow"),
    ({"total_coverage80": 0.97}, "too wide"),
])
def test_gate_rejects(change, fragment):
    ok, why = gate(GOOD | change, None, date(2020, 1, 1), min_balls=20000)
    assert not ok and fragment in why


def test_gate_needs_new_data_and_no_regression():
    champ = {"trained_through": "2020-01-01"}
    ok, why = gate(GOOD, champ, date(2020, 1, 1), min_balls=20000)
    assert not ok and "nothing to learn" in why
    worse = {"n": 5000, "champion": 1.600, "challenger": 1.610}
    ok, why = gate(GOOD, champ, date(2020, 6, 1), worse, min_balls=20000)
    assert not ok and "worse" in why
    ok, _ = gate(GOOD, champ, date(2020, 6, 1), {"n": 5000, "champion": 1.600, "challenger": 1.601}, min_balls=20000)
    assert ok


def test_coverage_check_ignores_small_noisy_samples():
    assert not coverage_problem(0.61, 36)[0]        # the false alarm we hit during development
    assert coverage_problem(0.61, 200)[0]
    assert not coverage_problem(0.76, 200)[0]
    assert not coverage_problem(None, 500)[0]


# ---------------------------------------------------------------- feedback loop

def _log(store, p, outcome_team_total, line=150.5, day=date(2021, 1, 1), key="e"):
    return feedback.log_prediction(store, key, ("Team A", "Team B"), day,
                                   MarketSpec("team_total", team="Team A"), line, p, "v1")


def test_log_settle_and_void(bb_matches, tmp_path):
    store = Store(tmp_path / "db.sqlite")
    m = next(x for x in bb_matches if len(x.innings) == 2 and x.winner)
    first = m.batting_first
    total = sum(b.runs for b in m.innings[0])
    feedback.log_prediction(store, "k", m.teams, m.date, MarketSpec("team_total", team=first), total - 0.5, 0.6, "v1")
    feedback.log_prediction(store, "k", m.teams, m.date, MarketSpec("player_runs", player="Nobody"), 10.5, 0.5, "v1")
    feedback.log_prediction(store, "k", ("Team Q", "Team R"), m.date, MarketSpec("match_total"), 300.5, 0.5, "v1")
    res = feedback.settle(store, bb_matches)
    assert res == {"settled": 1, "void": 1, "still_open": 1}
    rep = feedback.report(store)
    assert rep["team_total"]["n"] == 1 and rep["team_total"]["actual_rate"] == 1.0
    assert feedback.settle(store, bb_matches)["settled"] == 0     # nothing settles twice


def test_settle_matches_team_names_loosely(bb_matches, tmp_path):
    store = Store(tmp_path / "db.sqlite")
    m = next(x for x in bb_matches if len(x.innings) == 2)
    upper = tuple(t.upper() for t in m.teams)
    feedback.log_prediction(store, "k", upper, m.date, MarketSpec("match_total"), 0.5, 0.9, "v1")
    assert feedback.settle(store, bb_matches)["settled"] == 1


def _seed_biased(store, n, bias, rng):
    """Predictions that are systematically too high by `bias`."""
    for i in range(n):
        p_true = rng.uniform(0.2, 0.8)
        y = float(rng.random() < p_true)
        rid = feedback.log_prediction(store, f"k{i}", ("A", "B"), date(2021, 1, 1),
                                      MarketSpec("match_sixes"), 10.5, min(p_true + bias, 0.99), "v1")
        with store.tx() as c:
            c.execute("UPDATE market_predictions SET outcome = ?, status = 'settled' WHERE id = ?", (y, rid))


def test_calibration_corrects_a_systematic_bias(tmp_path):
    store = Store(tmp_path / "db.sqlite")
    _seed_biased(store, 1500, 0.12, np.random.default_rng(1))
    res = feedback.fit_calibrators(store, tmp_path / "models", min_n=500)
    fit = res["fitted"]["match_sixes"]
    assert fit["holdout"]["calibrated_brier"] < fit["holdout"]["raw_brier"]
    cal = feedback.load_calibrators(tmp_path / "models")["match_sixes"]
    assert cal.predict(np.array([0.62]))[0] < 0.56                  # pulls a 12-point-high 62% back down
    assert cal.predict(np.array([0.32]))[0] < 0.27                  # the bias is one-directional at every level


def test_calibrated_price_keeps_over_and_under_consistent(logit_model, state, tmp_path):
    from doosra.props import price
    from doosra.simulate import default_setup, simulate_match
    store = Store(tmp_path / "db.sqlite")
    _seed_biased(store, 900, 0.12, np.random.default_rng(4))
    feedback.fit_calibrators(store, tmp_path / "models", min_n=500)
    cal = feedback.load_calibrators(tmp_path / "models")["match_sixes"]
    sim = simulate_match(logit_model, state, default_setup(state, "T20:male", "Neutral Park", "Team A", "Team B"),
                         n=300, seed=1)
    raw = price(sim, MarketSpec("match_sixes"))
    adj = price(sim, MarketSpec("match_sixes"), calibrator=cal)
    assert adj.calibrated and adj.p_over + adj.p_under == pytest.approx(1)
    assert adj.p_over < raw.p_over


def test_calibration_waits_for_enough_data(tmp_path):
    store = Store(tmp_path / "db.sqlite")
    _seed_biased(store, 100, 0.12, np.random.default_rng(2))
    res = feedback.fit_calibrators(store, tmp_path / "models", min_n=500)
    assert "match_sixes" in res["skipped"] and not res["fitted"]


def test_drift_flags_a_recent_shift(tmp_path):
    store = Store(tmp_path / "db.sqlite")
    rng = np.random.default_rng(3)
    _seed_biased(store, 400, 0.0, rng)
    assert not feedback.drift(store, window=150)["retrain_recommended"]
    _seed_biased(store, 150, 0.25, rng)
    d = feedback.drift(store, window=150)
    assert d["retrain_recommended"] and "match_sixes" in d["drifting"]


# ---------------------------------------------------------------- end-to-end lifecycle

@pytest.fixture(scope="module")
def two_period_data(tmp_path_factory):
    from doosra.synth import generate, write_zip
    root = tmp_path_factory.mktemp("cycle")
    allm = generate(420, seed=8)
    p1 = write_zip(root / "p1.zip", allm[:300])
    p2 = write_zip(root / "p2.zip", allm)
    return root, p1, p2


FAST = dict(min_balls=5000, max_iter=30, kinds=("logit",), sims_per_year=8, n_sims=150, test_years=2)


@pytest.mark.slow
def test_full_learning_cycle(two_period_data):
    root, p1, p2 = two_period_data
    store = Store(root / "db.sqlite")
    models = root / "models"
    r1 = cycle(p1, store, models, retrain_winner_model=False, **FAST)
    assert r1["ball_model"]["promoted"], r1["ball_model"]["reason"]
    reg = BallRegistry(models)
    art = reg.load()
    assert art is not None and art.trained_through.isoformat() == r1["through"]

    # predict the unseen matches, then let their results arrive
    from doosra.ballbyball import load_bb_matches
    from doosra.props import standard_card
    from doosra.simulate import default_setup, simulate_match
    future = [m for m in load_bb_matches(p2) if m.date > art.trained_through][:15]
    for m in future:
        card = standard_card(simulate_match(art.model, art.state, default_setup(art.state, m.pool, m.venue, *m.teams),
                                            n=150, seed=1))
        feedback.log_card(store, m.match_id, m.teams, m.date, card, art.version)

    r2 = cycle(p2, store, models, retrain_winner_model=False, **FAST)
    assert r2["settle"]["settled"] >= 15 * 15 and r2["settle"]["still_open"] == 0
    assert r2["ball_model"]["promoted"], r2["ball_model"]["reason"]
    assert reg.champion_meta()["trained_through"] == r2["through"]
    assert "team_total" in r2["market_report"]

    r3 = cycle(p2, store, models, retrain_winner_model=False, **FAST)
    assert not r3["ball_model"]["promoted"] and "nothing to learn" in r3["ball_model"]["reason"]
    history = reg.runs()
    assert [h["promoted"] for h in history[:3]] == [False, True, True]


def test_retrain_rejects_unknown_pool(two_period_data):
    root, p1, _ = two_period_data
    from doosra.ballbyball import load_bb_matches
    with pytest.raises(ValueError):
        retrain_ball(load_bb_matches(p1), root / "m2", ("ODI:female",), **{k: v for k, v in FAST.items() if k != "test_years"})


# ---------------------------------------------------------------- weather + downloads

def test_weather_summary_and_cache(tmp_path):
    hourly = {"time": [f"2021-01-01T{h:02d}:00" for h in range(24)],
              "temperature_2m": [20.0] * 24, "dew_point_2m": [12.0] * 18 + [18.5] * 6,
              "relative_humidity_2m": [60.0] * 24, "precipitation": [0.0] * 23 + [2.0]}
    s = summarise(hourly)
    assert s == {"dew_spread": 1.5, "humidity": 60.0, "precip": 2.0}
    calls = []

    def handler(req):
        calls.append(req.url.host)
        return httpx.Response(200, json={"hourly": hourly})

    w = Weather(Store(tmp_path / "db.sqlite"), transport=httpx.MockTransport(handler))
    assert w.for_match("Wankhede Stadium, Mumbai", date(2021, 1, 1))["dew_spread"] == 1.5
    w.for_match("Wankhede Stadium, Mumbai", date(2021, 1, 1))
    assert calls == ["archive-api.open-meteo.com"]                     # second call came from cache
    assert w.for_match("Unknown Ground", date(2021, 1, 1)) is None
    assert coords_for("M Chinnaswamy Stadium, Bengaluru") is not None


def test_weather_failure_never_breaks_a_prediction(tmp_path):
    w = Weather(Store(tmp_path / "db.sqlite"), transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    assert w.for_match("Eden Gardens", date(2021, 1, 1)) is None


def test_download_is_restricted_to_cricsheet(tmp_path):
    seen = []

    def handler(req):
        seen.append(str(req.url))
        return httpx.Response(200, content=b"PK")

    p = download_cricsheet("t20s_json.zip", tmp_path, transport=httpx.MockTransport(handler))
    assert p.read_bytes() == b"PK" and seen == ["https://cricsheet.org/downloads/t20s_json.zip"]
    for bad in ("../etc/passwd", "https://evil.example/x.zip", "t20s.json"):
        with pytest.raises(ValueError):
            download_cricsheet(bad, tmp_path, transport=httpx.MockTransport(handler))


def _fake_heldout(n_matches, true_sigma, seed, proposal=0.35, n=400):
    """Held-out 'simulations' where a total is 150 * (1 + 0.8 z) + noise, and the real
    total came from a day factor with spread true_sigma. Fitting must find true_sigma."""
    import numpy as np
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_matches):
        mu = rng.normal(155, 10)
        day = rng.normal(0, proposal, n)
        v = np.round(mu * (1 + 0.8 * day) + rng.normal(0, 14, n))
        z = rng.normal(0, true_sigma)
        a = round(mu * (1 + 0.8 * z) + rng.normal(0, 14))
        s6 = np.round(v * 0.3); a6 = round(a * 0.3)
        out.append({"match_id": str(i), "year": 2024 + i % 3, "seed": i, "proposal": proposal, "day": day,
                    "total": v, "total_actual": float(a), "session6": s6, "session6_actual": float(a6),
                    "winner_hit": (rng.random(n) < 0.5).astype(float)})
    return out


def test_volatility_fit_recovers_the_planted_spread():
    from doosra.lifecycle import fit_volatility
    rep = fit_volatility(_fake_heldout(150, 0.15, seed=1))
    assert abs(rep["volatility"] - 0.15) <= 0.05, rep["volatility"]
    assert abs(rep["total_coverage80"] - 0.8) < 0.1          # judged on seasons it was not fitted on
    assert rep["total_coverage80_without_volatility"] < 0.65  # ignoring match days is clearly too narrow


def test_no_planted_spread_fits_near_zero():
    from doosra.lifecycle import fit_volatility
    rep = fit_volatility(_fake_heldout(150, 0.0, seed=2))
    assert rep["volatility"] <= 0.05


def test_weights_at_the_proposal_are_uniform():
    import numpy as np
    from doosra.lifecycle import _weights
    day = np.random.default_rng(0).normal(0, 0.35, 500)
    w = _weights(day, 0.35, 0.35)
    assert np.allclose(w, 1 / 500)

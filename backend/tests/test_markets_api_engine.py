"""HTTP contract for the ball-by-ball engine. Uses one small trained model."""
import time

import pytest
from fastapi.testclient import TestClient

from doosra.api import Settings, create_app
from doosra.lifecycle import retrain_ball

TOKEN = "t0k"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture(scope="module")
def client(tmp_path_factory, bb_matches, synth_zip):
    root = tmp_path_factory.mktemp("api")
    res = retrain_ball(bb_matches, root / "models", ("T20:male",), test_years=2, kinds=("logit",), max_iter=20,
                       sims_per_year=4, n_sims=100, min_balls=2000)
    assert res["promoted"], res["reason"]
    s = Settings(db_path=str(root / "db.sqlite"), admin_token=TOKEN, models_dir=str(root / "models"),
                 cricsheet_path=str(synth_zip))
    return TestClient(create_app(s, llm=None))


@pytest.fixture()
def bare_client(tmp_path):
    return TestClient(create_app(Settings(db_path=str(tmp_path / "d.sqlite"), models_dir=str(tmp_path / "m")), llm=None))


MATCH = {"team_a": "Team A", "team_b": "Team B", "venue": "Neutral Park", "sims": 300}


def test_engine_without_model_explains_itself(bare_client):
    e = bare_client.get("/engine").json()
    assert e["model"] is None and e["runs"] == []
    r = bare_client.post("/markets/card", json=MATCH)
    assert r.status_code == 409 and "learning cycle" in r.json()["detail"]


def test_engine_status(client):
    e = client.get("/engine").json()
    assert e["model"]["kind"] == "logit" and e["runs"][0]["promoted"]
    assert e["model"]["evaluation"]["ball_log_loss"] < e["model"]["evaluation"]["baseline_log_loss"]


def test_catalog(client):
    c = client.get("/catalog").json()
    assert "Team A" in c["teams"] and "Flat Deck" in c["venues"]
    assert len(c["lineups"]["Team A"]["batting"]) == 11


def test_card(client):
    r = client.post("/markets/card", json=MATCH)
    assert r.status_code == 200
    c = r.json()["card"]
    assert sum(c["winner"].values()) == pytest.approx(1, abs=1e-3)
    labels = [row["market"] for row in c["teams"]["Team A"]]
    assert "Team A runs after 6 overs" in labels and c["players"]


def test_card_validation(client):
    assert client.post("/markets/card", json=MATCH | {"team_b": "Team A"}).status_code == 422
    assert client.post("/markets/card", json=MATCH | {"batting_first": "Team Z"}).status_code == 422
    assert client.post("/markets/card", json=MATCH | {"team_b": "Unknown XI"}).status_code == 422
    assert client.post("/markets/card", json=MATCH | {"sims": 10}).status_code == 422


def test_card_with_custom_lineup(client):
    lu = {"Team A": {"batting": ["A_Bat0", "A_Bat1", "A_Bat2"], "bowlers": ["A_Bowl0", "A_Bowl1"]}}
    r = client.post("/markets/card", json=MATCH | {"lineups": lu, "batting_first": "Team A"})
    assert r.status_code == 200
    assert r.json()["lineups"]["Team A"]["batting"] == ["A_Bat0", "A_Bat1", "A_Bat2"]


def test_live_card_skips_finished_sessions(client):
    live = {"batting": "Team A", "innings": 1, "score": 90, "wickets": 2, "overs": 11.2}
    r = client.post("/markets/card", json=MATCH | {"live": live})
    assert r.status_code == 200
    labels = [row["market"] for row in r.json()["card"]["teams"]["Team A"]]
    assert "Team A runs after 15 overs" in labels and "Team A runs after 6 overs" not in labels
    bad = client.post("/markets/card", json=MATCH | {"live": live | {"overs": 11.7}})
    assert bad.status_code == 422
    chase = client.post("/markets/card", json=MATCH | {"live": {"batting": "Team B", "innings": 2, "score": 10,
                                                                   "wickets": 0, "overs": 1.0}})
    assert chase.status_code == 422 and "target" in chase.json()["detail"]


def test_price_with_verdict(client):
    body = MATCH | {"market": "team_total", "team": "Team A", "line": 150.5, "odds_over": 1.9, "odds_under": 1.9,
                    "bankroll": 10000}
    r = client.post("/markets/price", json=body).json()
    assert 0 < r["price"]["p_over"] < 1
    assert r["verdict"]["outcomes"] == ["Over 150.5", "Under 150.5"]
    assert {a["label"] for a in r["verdict"]["analysis"]} <= {"BET", "SKIP", "TRAP"}
    push = client.post("/markets/price", json=body | {"line": 150}).json()
    assert "verdict" not in push and "push" in push["note"]
    bad = client.post("/markets/price", json=MATCH | {"market": "session", "team": "Team A"})
    assert bad.status_code == 422


def test_logging_needs_admin(client):
    r = client.post("/markets/card", json=MATCH | {"log_as": "evt1"})
    assert r.status_code == 401
    r = client.post("/markets/card", json=MATCH | {"log_as": "evt1"}, headers=AUTH)
    assert r.status_code == 200 and r.json()["logged"] > 10


@pytest.mark.slow
def test_cycle_endpoint_runs_in_background(client):
    assert client.post("/admin/cycle").status_code == 401
    assert client.post("/admin/cycle", headers=AUTH).json()["state"] == "running"
    assert client.post("/admin/cycle", headers=AUTH).status_code == 409        # one at a time
    for _ in range(300):
        st = client.get("/admin/cycle", headers=AUTH).json()
        if st["state"] != "running":
            break
        time.sleep(1)
    assert st["state"] == "done", st.get("error")
    assert "promoted" in st["summary"]

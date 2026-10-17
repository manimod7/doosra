import json
import zipfile

import httpx
import pytest
from fastapi.testclient import TestClient

from doosra.api import Settings, create_app, pool_for
from doosra.providers.the_odds_api import TheOddsApi
from tests.test_ml import synthetic_league

TOKEN = "t0ken"
AUTH = {"Authorization": f"Bearer {TOKEN}"}

EVENTS = [{
    "id": "ev1", "sport_key": "cricket_international_t20", "commence_time": "2099-01-01T09:00:00Z",
    "home_team": "T0", "away_team": "T1",
    "bookmakers": [
        {"key": "pinnacle", "title": "Pinnacle", "markets": [{"key": "h2h", "outcomes": [
            {"name": "T0", "price": 1.80}, {"name": "T1", "price": 2.10}]}]},
        {"key": "betway", "title": "Betway", "markets": [{"key": "h2h", "outcomes": [
            {"name": "T0", "price": 1.75}, {"name": "T1", "price": 2.05}]}]},
    ],
}]


def handler(request):
    if request.url.path.endswith("/odds"):
        body = EVENTS if "international_t20" in request.url.path else []
        return httpx.Response(200, json=body, headers={"x-requests-remaining": "300", "x-requests-used": "2", "x-requests-last": "2"})
    return httpx.Response(200, json=[])


def make(tmp_path, *, token=TOKEN, models_dir=None):
    s = Settings(db_path=str(tmp_path / "db.sqlite"), odds_api_key="k", admin_token=token,
                 models_dir=str(models_dir or tmp_path / "models"))
    factory = lambda st, store: TheOddsApi(api_key="k", store=store, transport=httpx.MockTransport(handler))
    return TestClient(create_app(s, provider_factory=factory, llm=None))


def test_pool_mapping():
    assert pool_for("cricket_odi", ["India Women", "England Women"]) == ("ODI:female", ["India", "England"])
    assert pool_for("cricket_international_t20", ["India", "Australia"])[0] == "T20:male"
    assert pool_for("soccer_epl", ["A", "B"])[0] is None


def test_admin_endpoints_need_token(tmp_path):
    c = make(tmp_path)
    assert c.post("/admin/refresh").status_code == 401
    assert c.post("/admin/refresh", headers={"Authorization": "Bearer wrong"}).status_code == 401
    off = make(tmp_path / "x", token=None)
    assert off.post("/admin/refresh", headers=AUTH).status_code == 403


def test_live_refresh_then_board(tmp_path):
    c = make(tmp_path)
    assert c.get("/matches").json()["matches"] == []
    r = c.post("/admin/refresh", headers=AUTH).json()
    assert r["stored_rows"] == 2 and r["remaining"] == 300
    ms = c.get("/matches").json()
    assert ms["data_mode"] == "live" and ms["matches"][0]["id"] == "ev1" and ms["matches"][0]["books"] == 2
    odds = c.get("/matches/ev1/odds").json()
    assert {b["book"] for b in odds["books"]} == {"Pinnacle", "Betway"}
    assert odds["sharp_book"] == "Pinnacle"
    assert odds["model"]["available"] is False  # no trained model in this test
    assert c.get("/matches/ev1/odds", params={"use_model": True}).status_code == 409
    assert c.get("/budget").json()["remaining"] == 300


def test_explain_falls_back_without_llm(tmp_path):
    c = make(tmp_path)
    c.post("/admin/refresh", headers=AUTH)
    r = c.post("/matches/ev1/explain", json={"model_probs": [0.6, 0.4]}, headers=AUTH).json()
    assert r["explanation"]["source"] == "template"
    assert "T0 is best priced at 1.80 with Pinnacle" in r["explanation"]["text"]


def test_bets_ledger_with_closing_line(tmp_path):
    c = make(tmp_path)
    c.post("/admin/refresh", headers=AUTH)
    bid = c.post("/bets", json={"event_id": "ev1", "outcome": "T0", "book": "Pinnacle", "odds": 1.95, "stake": 100},
                 headers=AUTH).json()["id"]
    s = c.post(f"/bets/{bid}/settle", json={"result": "won"}, headers=AUTH).json()
    assert s["closing_fair_prob"] is not None and s["clv"] > 0  # 1.95 beat the 1.80 close
    led = c.get("/bets", headers=AUTH).json()["summary"]
    assert led["settled"] == 1 and led["profit"] == 95.0 and led["avg_clv"] > 0
    assert c.post(f"/bets/{bid}/settle", json={"result": "maybe"}, headers=AUTH).status_code == 422


def test_model_used_when_promoted(tmp_path):
    from doosra.train import run
    from tests.test_ml import _write_zip
    z = tmp_path / "all.zip"
    _write_zip(z, synthetic_league())
    res = run(str(z), str(tmp_path / "models"), ("T20:male",), first_test_year=2013)
    assert res["decision"].promote
    c = make(tmp_path, models_dir=tmp_path / "models")
    c.post("/admin/refresh", headers=AUTH)
    odds = c.get("/matches/ev1/odds", params={"use_model": True}).json()
    assert odds["model"]["available"] and odds["probs_source"].startswith("model")
    assert abs(sum(odds["model"]["probs"]) - 1) < 1e-9
    assert all(a["label"] in {"BET", "SKIP", "TRAP"} for a in odds["analysis"])
    store = c.app.state.store
    assert store._conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 1  # prediction logged for CLV

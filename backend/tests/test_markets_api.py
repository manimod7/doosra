import pytest
from fastapi.testclient import TestClient

from doosra.analyzer import analyze, closing_line_value
from doosra.api import app
from doosra.markets import BookQuote, compare

client = TestClient(app)

QUOTES = [
    BookQuote("pinnacle", (1.78, 2.14)),
    BookQuote("bet365", (1.72, 2.10)),
    BookQuote("stake", (1.76, 2.12)),
]


def test_compare_best_price_and_sharp():
    c = compare(["A", "B"], QUOTES)
    assert c.best[0].book == "pinnacle" and c.best[0].odds == 1.78
    assert c.best[1].book == "pinnacle"
    assert c.sharp_book == "pinnacle"
    assert sum(c.consensus_probs) == pytest.approx(1.0)
    assert c.books[0].margin <= c.books[-1].margin  # sorted by margin
    assert c.best_price_margin > 0  # no arbitrage in this set


def test_compare_detects_arbitrage():
    arb = [BookQuote("x", (2.20, 1.60)), BookQuote("y", (1.60, 2.20))]
    assert compare(["A", "B"], arb).best_price_margin < 0


def test_compare_rejects_mismatched_outcomes():
    with pytest.raises(ValueError):
        compare(["A", "B", "C"], QUOTES)


def test_analyze_no_model_never_gives_verdict():
    out = analyze(["A", "B"], QUOTES)
    assert all(r["label"] == "NO MODEL" for r in out["analysis"])
    assert all(r["stake_fraction"] == 0.0 for r in out["analysis"])


def test_analyze_with_model_flags_bet_skip():
    # model thinks A is 60%: best price 1.78 -> edge +6.8%
    out = analyze(["A", "B"], QUOTES, model_probs=[0.60, 0.40], bankroll=1000)
    a, b = out["analysis"]
    assert a["label"] == "BET" and a["stake_amount"] > 0
    assert b["label"] == "SKIP"


def test_analyze_trap_when_model_far_above_sharp():
    out = analyze(["A", "B"], QUOTES, model_probs=[0.80, 0.20])
    assert out["analysis"][0]["label"] == "TRAP"


def test_analyze_validates_model_probs():
    with pytest.raises(ValueError):
        analyze(["A", "B"], QUOTES, model_probs=[0.7, 0.7])


def test_closing_line_value_sign():
    assert closing_line_value(2.10, [BookQuote("pinnacle", (1.95, 1.95))], 0) > 0


def test_api_health_and_matches():
    assert client.get("/health").json()["status"] == "ok"
    r = client.get("/matches").json()
    assert r["data_mode"] == "demo" and len(r["matches"]) == 2


def test_api_match_odds_and_404():
    r = client.get("/matches/demo-1/odds")
    assert r.status_code == 200 and r.json()["data_mode"] == "demo"
    assert client.get("/matches/nope/odds").status_code == 404
    r = client.get("/matches/demo-1/odds", params={"model_probs": "0.6,0.4", "bankroll": 1000})
    assert r.json()["analysis"][0]["label"] in {"BET", "SKIP", "TRAP"}


def test_api_analyze_validation():
    body = {"outcomes": ["A", "B"], "quotes": [{"book": "x", "odds": [1.9, 1.9]}]}
    assert client.post("/analyze", json=body).status_code == 200
    bad = {"outcomes": ["A", "B"], "quotes": [{"book": "x", "odds": [1.0, 1.9]}]}
    assert client.post("/analyze", json=bad).status_code == 422
    mismatch = {"outcomes": ["A", "B"], "quotes": [{"book": "x", "odds": [1.9, 1.9, 3.0]}]}
    assert client.post("/analyze", json=mismatch).status_code == 422
    badprobs = {**body, "model_probs": [0.9, 0.9]}
    assert client.post("/analyze", json=badprobs).status_code == 422

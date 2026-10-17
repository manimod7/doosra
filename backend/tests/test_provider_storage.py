import httpx
import pytest

from doosra.providers.the_odds_api import BudgetExceeded, ProviderError, TheOddsApi, parse_events
from doosra.storage import Store

# Shape follows The Odds API v4 documented /odds response.
EVENTS = [
    {
        "id": "ev1", "sport_key": "cricket_odi", "commence_time": "2099-01-01T09:00:00Z",
        "home_team": "India", "away_team": "Australia",
        "bookmakers": [
            {"key": "pinnacle", "title": "Pinnacle", "last_update": "2099-01-01T00:00:00Z",
             "markets": [{"key": "h2h", "outcomes": [
                 {"name": "Australia", "price": 2.30}, {"name": "India", "price": 1.66}]}]},
            {"key": "betway", "title": "Betway",
             "markets": [{"key": "h2h", "outcomes": [{"name": "India", "price": 1.60}]}]},  # one side missing
        ],
    },
    {
        "id": "ev2", "sport_key": "cricket_test_match", "commence_time": "2099-01-05T04:00:00Z",
        "home_team": "England", "away_team": "Pakistan",
        "bookmakers": [{"key": "onexbet", "title": "1xBet", "markets": [{"key": "h2h", "outcomes": [
            {"name": "England", "price": 1.9}, {"name": "Draw", "price": 4.5}, {"name": "Pakistan", "price": 3.6}]}]}],
    },
]


def test_parse_fixes_order_and_drops_incomplete():
    rows = parse_events(EVENTS, "2098-12-31T00:00:00+00:00")
    assert len(rows) == 2
    odi = rows[0]
    assert odi["outcomes"] == ["India", "Australia"] and odi["odds"] == [1.66, 2.30]
    test = rows[1]
    assert test["outcomes"] == ["England", "Pakistan", "Draw"] and test["odds"] == [1.9, 3.6, 4.5]


def _api(store, handler, **kw):
    return TheOddsApi(api_key="k", store=store, transport=httpx.MockTransport(handler), **kw)


def _ok(remaining):
    def handler(request):
        assert request.url.params["apiKey"] == "k"
        if request.url.path.endswith("/sports"):
            return httpx.Response(200, json=[{"key": "cricket_odi", "group": "Cricket", "active": True},
                                             {"key": "soccer_epl", "group": "Soccer", "active": True}],
                                  headers={"x-requests-remaining": str(remaining), "x-requests-used": "0", "x-requests-last": "0"})
        return httpx.Response(200, json=EVENTS[:1],
                              headers={"x-requests-remaining": str(remaining - 2), "x-requests-used": "2", "x-requests-last": "2"})
    return handler


def test_refresh_stores_rows_and_tracks_credits():
    store = Store(":memory:")
    api = _api(store, _ok(400))
    assert [s["key"] for s in api.active_cricket_sports()] == ["cricket_odi"]
    res = api.refresh(("cricket_odi",))
    assert res["stored_rows"] == 1 and res["remaining"] == 398
    assert store.upcoming_events(after_iso="2000-01-01")[0]["event_id"] == "ev1"
    quotes = store.latest_quotes("ev1")
    assert quotes[0]["book_title"] == "Pinnacle"


def test_budget_guard_blocks_before_calling():
    store = Store(":memory:")
    store.record_credits("/seed", 0, 450, 51)
    calls = []

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(200, json=[])

    api = _api(store, handler, reserve=50)
    with pytest.raises(BudgetExceeded):
        api.fetch_odds("cricket_odi")  # costs 2 with uk,eu; 51-2 < 50
    assert calls == []
    res = api.refresh(("cricket_odi", "cricket_test_match"))
    assert res["stored_rows"] == 0 and len(res["skipped"]) == 1 and calls == []


def test_bad_key_is_reported_clearly():
    store = Store(":memory:")
    api = _api(store, lambda r: httpx.Response(401, json={"message": "invalid"}))
    with pytest.raises(ProviderError, match="rejected the key"):
        api.fetch_odds("cricket_odi")


def test_latest_quotes_as_of_gives_closing_line():
    store = Store(":memory:")
    early = parse_events(EVENTS[:1], "2098-12-30T00:00:00+00:00")
    late = parse_events(EVENTS[:1], "2098-12-31T08:00:00+00:00")
    late[0]["odds"] = [1.55, 2.55]
    store.add_snapshot_rows(early + late)
    assert store.latest_quotes("ev1")[0]["odds"] == [1.55, 2.55]
    assert store.latest_quotes("ev1", before_iso="2098-12-30T12:00:00+00:00")[0]["odds"] == [1.66, 2.30]


def test_bets_ledger_roundtrip():
    store = Store(":memory:")
    bid = store.add_bet("ev1", "India", "pinnacle", 1.66, 500)
    assert store.settle_bet(bid, "won", 0.62)
    b = store.bets()[0]
    assert b["result"] == "won" and b["closing_fair_prob"] == 0.62
    with pytest.raises(ValueError):
        store.settle_bet(bid, "maybe", None)

"""The same storage behaviour on SQLite and on Postgres (what Neon runs in the cloud).

Postgres tests run when DOOSRA_TEST_PG_URL points at a server where the user may create
databases, e.g. postgresql://doosra@localhost:55432/postgres. Each test gets its own
throwaway database. CI starts a Postgres service and sets the variable.
"""
from __future__ import annotations

import os
import uuid
from datetime import date

import pytest

from doosra import feedback, modelsync
from doosra.props import MarketSpec
from doosra.storage import Store

PG = os.environ.get("DOOSRA_TEST_PG_URL")


@pytest.fixture(params=["sqlite", "postgres"])
def store(request, tmp_path):
    if request.param == "sqlite":
        yield Store(tmp_path / "t.sqlite")
        return
    if not PG:
        pytest.skip("set DOOSRA_TEST_PG_URL to run the Postgres variants")
    import psycopg
    name = f"doosra_t_{uuid.uuid4().hex[:10]}"
    with psycopg.connect(PG, autocommit=True) as admin:
        admin.execute(f'CREATE DATABASE "{name}"')
    url = PG.rsplit("/", 1)[0] + "/" + name
    s = Store(url)
    yield s
    s._conn.close()
    with psycopg.connect(PG, autocommit=True) as admin:
        admin.execute(f'DROP DATABASE "{name}" WITH (FORCE)')


def snap(event="e1", book="b1", at="2026-10-01T10:00:00+00:00", odds=(1.9, 1.95), start="2099-01-01T14:00:00+00:00"):
    return {"event_id": event, "sport_key": "cricket_ipl", "commence_time": start, "home_team": "Mumbai Indians",
            "away_team": "Chennai Super Kings", "book_key": book, "book_title": book.upper(),
            "outcomes": ["Mumbai Indians", "Chennai Super Kings"], "odds": list(odds),
            "book_last_update": at, "fetched_at": at}


def test_bets_get_ids_and_settle(store):
    a = store.add_bet("e1", "Mumbai Indians", "book", 1.9, 250, note="100% sure")   # % must not break SQL
    b = store.add_bet("e1", "Chennai Super Kings", "book", 2.1, 100)
    assert b == a + 1
    assert store.settle_bet(a, "won", 0.55) is True
    assert store.settle_bet(999, "won", None) is False
    bets = store.bets()
    assert [x["id"] for x in bets] == [b, a]
    assert bets[1]["result"] == "won" and bets[1]["note"] == "100% sure"
    with pytest.raises(ValueError):
        store.settle_bet(a, "maybe", None)


def test_snapshots_latest_quote_per_book_and_upcoming(store):
    store.add_snapshot_rows([snap(at="2026-10-01T10:00:00+00:00", odds=(1.8, 2.0)),
                             snap(at="2026-10-01T12:00:00+00:00", odds=(1.9, 1.9)),
                             snap(book="b2", at="2026-10-01T11:00:00+00:00", odds=(1.85, 1.95)),
                             snap(event="old", start="2000-01-01T00:00:00+00:00")])
    q = {r["book_key"]: r["odds"] for r in store.latest_quotes("e1")}
    assert q == {"b1": [1.9, 1.9], "b2": [1.85, 1.95]}
    closing = {r["book_key"]: r["odds"] for r in store.latest_quotes("e1", before_iso="2026-10-01T11:30:00+00:00")}
    assert closing["b1"] == [1.8, 2.0]
    up = store.upcoming_events()
    assert [e["event_id"] for e in up] == ["e1"] and up[0]["home_team"] == "Mumbai Indians"
    assert store.event("e1")["fetched_at"] == "2026-10-01T12:00:00+00:00"


def test_credits(store):
    store.record_credits("/odds", 2, 10, 490)
    store.record_credits("/odds", 2, 12, None)
    assert store.last_known_remaining() == 490
    assert len(store.credit_log()) == 2


def test_market_prediction_log_and_settlement(store, bb_matches):
    m = bb_matches[-1]
    first = m.batting_first or m.teams[0]
    pid = feedback.log_prediction(store, "k1", m.teams, m.date, MarketSpec("team_total", team=first), 150.5, 0.5, "v1")
    feedback.log_prediction(store, "k1", m.teams, m.date, MarketSpec("match_winner", team=m.teams[0]), None, 0.6, "v1")
    assert pid >= 1
    res = feedback.settle(store, [m])
    assert res["settled"] + res["void"] == 2 and res["still_open"] == 0
    assert len(feedback._settled(store)) == res["settled"]


def test_weather_cache_upserts(store):
    from doosra.conditions import Weather
    w = Weather(store)
    for data in ('{"humidity": 60}', '{"humidity": 70}'):
        store._conn.execute("INSERT OR REPLACE INTO weather VALUES (?,?,?)", ("Wankhede", "2026-10-01", data))
    row = store._conn.execute("SELECT data FROM weather WHERE venue = ?", ("Wankhede",)).fetchall()
    assert [r["data"] for r in row] == ['{"humidity": 70}']
    assert w.for_match("Nowhere Ground", date(2026, 10, 1)) is None


def test_models_round_trip_through_the_database(store, tmp_path):
    import json
    src = tmp_path / "models"
    (src / "ball").mkdir(parents=True)
    (src / "ball" / "ball-1.joblib").write_bytes(b"\x00\x01binary")
    (src / "ball" / "champion.json").write_text(json.dumps({"file": "ball-1.joblib", "version": "1"}))
    (src / "ball" / "history.jsonl").write_text('{"v": 1}\n')
    (src / "ball" / "ball-0.joblib").write_bytes(b"old, not referenced")
    pushed = modelsync.push(store, src)
    assert sorted(pushed["uploaded"]) == ["ball/ball-1.joblib", "ball/champion.json", "ball/history.jsonl"]

    dst = tmp_path / "fresh"
    got = modelsync.pull(store, dst)
    assert got["downloaded"] == sorted(pushed["uploaded"])
    assert (dst / "ball" / "ball-1.joblib").read_bytes() == b"\x00\x01binary"
    assert not (dst / "ball" / "ball-0.joblib").exists()

    # a new champion replaces the old file in the database
    (src / "ball" / "ball-2.joblib").write_bytes(b"newer")
    (src / "ball" / "champion.json").write_text(json.dumps({"file": "ball-2.joblib", "version": "2"}))
    again = modelsync.push(store, src)
    assert again["removed"] == ["ball/ball-1.joblib"]


def test_pull_never_writes_outside_the_models_folder(store, tmp_path):
    store._conn.execute(modelsync.SCHEMA)
    store._conn.execute("INSERT INTO artifacts (name, data, updated_at) VALUES (?,?,?)", ("../evil.txt", b"x", "now"))
    store._conn.commit()
    got = modelsync.pull(store, tmp_path / "m")
    assert got["downloaded"] == [] and not (tmp_path / "evil.txt").exists()


@pytest.mark.skipif(not PG, reason="Postgres only")
def test_postgres_reads_leave_no_open_transaction_and_survive_a_dropped_connection(store):
    if not store.is_remote:
        pytest.skip("Postgres only")
    import psycopg
    store.add_bet("e1", "x", "b", 2.0, 10)
    store.bets()
    assert store._conn._db.info.transaction_status == psycopg.pq.TransactionStatus.IDLE
    store._conn._db.close()                      # what Neon does to an idle connection
    assert len(store.bets()) == 1
    assert "postgres" in repr(store) and "@" not in repr(store)


@pytest.mark.skipif(not PG, reason="Postgres only")
def test_failed_statement_does_not_poison_the_connection(store):
    if not store.is_remote:
        pytest.skip("Postgres only")
    with pytest.raises(Exception):
        store._conn.execute("SELECT * FROM no_such_table")
    assert store.bets() == []

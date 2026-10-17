"""Storage: a local SQLite file, or Postgres (Neon) in the cloud.

DOOSRA_DB is either a file path (local runs, tests) or a postgresql:// URL
(Cloud Run, GitHub Actions). The SQL is written once, in SQLite style; for
Postgres a thin adapter rewrites placeholders and schema types, so every
module above this one works unchanged on both.

Tables
- snapshots: every odds quote ever fetched (append-only, so closing lines survive)
- credits:   API credit ledger from provider response headers
- predictions: what the model said, and the price, at the moment it said it
- bets:      your own bets, for ROI and closing-line value
"""
from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
  id INTEGER PRIMARY KEY,
  event_id TEXT NOT NULL,
  sport_key TEXT NOT NULL,
  commence_time TEXT NOT NULL,
  home_team TEXT NOT NULL,
  away_team TEXT NOT NULL,
  book_key TEXT NOT NULL,
  book_title TEXT NOT NULL,
  outcomes TEXT NOT NULL,      -- JSON list of names, fixed order
  odds TEXT NOT NULL,          -- JSON list of decimal odds, same order
  book_last_update TEXT,
  fetched_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_snap_event ON snapshots(event_id, fetched_at);

CREATE TABLE IF NOT EXISTS credits (
  id INTEGER PRIMARY KEY,
  at TEXT NOT NULL,
  endpoint TEXT NOT NULL,
  cost INTEGER,
  used INTEGER,
  remaining INTEGER
);

CREATE TABLE IF NOT EXISTS predictions (
  id INTEGER PRIMARY KEY,
  event_id TEXT NOT NULL,
  model_version TEXT NOT NULL,
  outcomes TEXT NOT NULL,
  probs TEXT NOT NULL,
  best_odds TEXT NOT NULL,
  made_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS bets (
  id INTEGER PRIMARY KEY,
  event_id TEXT NOT NULL,
  outcome TEXT NOT NULL,
  book TEXT NOT NULL,
  odds REAL NOT NULL,
  stake REAL NOT NULL,
  placed_at TEXT NOT NULL,
  result TEXT,                -- won | lost | void | NULL while open
  closing_fair_prob REAL,
  note TEXT
);
"""


_EVENT_COLS = ("event_id, MAX(sport_key) AS sport_key, MAX(commence_time) AS commence_time, "
               "MAX(home_team) AS home_team, MAX(away_team) AS away_team, MAX(fetched_at) AS fetched_at")


def is_postgres_url(target: str | Path) -> bool:
    return str(target).startswith(("postgresql://", "postgres://"))


class _PgCursor:
    """The bits of a sqlite3 cursor the code uses: fetchone/fetchall/rowcount/lastrowid."""

    def __init__(self, cur, lastrowid: int | None = None):
        self._cur, self.lastrowid, self.rowcount = cur, lastrowid, cur.rowcount

    def fetchone(self):
        return self._cur.fetchone() if self._cur.description else None

    def fetchall(self):
        return self._cur.fetchall() if self._cur.description else []


def _is_disconnect(e: Exception) -> bool:
    import psycopg
    return isinstance(e, psycopg.OperationalError)


class PgConnection:
    """sqlite3.Connection look-alike over psycopg, for the SQL this project writes.

    Rewrites: ? and :name placeholders, INTEGER PRIMARY KEY -> BIGSERIAL, REAL -> DOUBLE
    PRECISION, INSERT OR REPLACE -> ON CONFLICT DO UPDATE (on the table's primary key),
    and adds RETURNING id to plain INSERTs so lastrowid works. Rows come back as dicts,
    which support row["col"] and dict(row) exactly like sqlite3.Row.
    """

    _PK = {"weather": ("venue", "day"), "artifacts": ("name",)}

    def __init__(self, url: str):
        self._url, self._in_tx = url, False
        self._db = self._connect()

    def _connect(self):
        import psycopg                      # optional dependency: pip install ".[cloud]"
        from psycopg.rows import dict_row
        return psycopg.connect(self._url, row_factory=dict_row, autocommit=False)

    # Outside Store.tx() every statement commits at once, so no transaction is ever left
    # open between requests (an open one would stop Neon from sleeping and use up free hours).
    def begin(self):
        self._in_tx = True

    def end(self):
        self._in_tx = False

    @staticmethod
    def _sql(sql: str) -> str:
        sql = sql.replace("%", "%%")
        sql = re.sub(r"(?<![:\w]):([A-Za-z_]\w*)", r"%(\1)s", sql)
        return sql.replace("?", "%s")

    @classmethod
    def _ddl(cls, sql: str) -> str:
        sql = re.sub(r"\bid INTEGER PRIMARY KEY\b", "id BIGSERIAL PRIMARY KEY", sql)
        sql = re.sub(r"\bREAL\b", "DOUBLE PRECISION", sql)
        return re.sub(r"\bBLOB\b", "BYTEA", sql)

    def _upsert(self, sql: str) -> str:
        m = re.match(r"\s*INSERT OR REPLACE INTO (\w+)\s*(\(([^)]*)\))?\s*VALUES", sql, re.I)
        if not m:
            return sql
        table, cols = m.group(1), m.group(3)
        names = [c.strip() for c in cols.split(",")] if cols else self._columns(table)
        pk = self._PK[table]
        sets = ", ".join(f"{c} = EXCLUDED.{c}" for c in names if c not in pk) or f"{pk[0]} = EXCLUDED.{pk[0]}"
        body = re.sub(r"INSERT OR REPLACE INTO", "INSERT INTO", sql, count=1, flags=re.I)
        return f"{body} ON CONFLICT ({', '.join(pk)}) DO UPDATE SET {sets}"

    def _columns(self, table: str) -> list[str]:
        rows = self._db.execute("SELECT column_name FROM information_schema.columns WHERE table_name = %s "
                                "ORDER BY ordinal_position", (table,)).fetchall()
        return [r["column_name"] for r in rows]

    def execute(self, sql: str, params=()):
        if re.match(r"\s*CREATE ", sql, re.I):
            sql = self._ddl(sql)
        sql = self._upsert(sql)
        returning = bool(re.match(r"\s*INSERT INTO (snapshots|credits|predictions|bets|market_predictions)\b", sql, re.I)) \
            and "RETURNING" not in sql.upper()
        q = self._sql(sql) + (" RETURNING id" if returning else "")
        args = params if params != () else None
        try:
            cur = self._db.execute(q, args)
        except Exception as e:
            if self._in_tx:
                raise                       # Store.tx() rolls back
            if not _is_disconnect(e):
                self._reset()               # leave the connection usable for the next statement
                raise
            self._db = self._connect()      # Neon drops idle connections: reconnect once
            cur = self._db.execute(q, args)
        last = cur.fetchone()["id"] if returning else None
        if not self._in_tx:
            self._db.commit()
        return _PgCursor(cur, last)

    def executemany(self, sql: str, seq):
        with self._db.cursor() as cur:
            cur.executemany(self._sql(self._upsert(sql)), list(seq))
        if not self._in_tx:
            self._db.commit()

    def executescript(self, script: str):
        for stmt in (x.strip() for x in script.split(";")):
            if stmt:
                self.execute(stmt)

    def _reset(self):
        try:
            self._db.rollback()
        except Exception:
            self._db = self._connect()

    def commit(self):
        self._db.commit()

    def rollback(self):
        self._db.rollback()

    def close(self):
        self._db.close()


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class Store:
    path: str | Path

    def __post_init__(self):
        if is_postgres_url(self.path):
            self._conn = PgConnection(str(self.path))
        else:
            p = Path(self.path)
            if str(p) != ":memory:":
                p.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(p), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    @property
    def is_remote(self) -> bool:
        return is_postgres_url(self.path)

    def __repr__(self) -> str:          # never print a connection string (it holds the password)
        return f"Store({'postgres' if self.is_remote else self.path!s})"

    @contextmanager
    def tx(self) -> Iterator["sqlite3.Connection | PgConnection"]:
        begin, end = getattr(self._conn, "begin", None), getattr(self._conn, "end", None)
        if begin:
            begin()
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            if end:
                end()

    # ---- snapshots -----------------------------------------------------
    def add_snapshot_rows(self, rows: list[dict]) -> int:
        with self.tx() as c:
            c.executemany(
                """INSERT INTO snapshots (event_id, sport_key, commence_time, home_team,
                   away_team, book_key, book_title, outcomes, odds, book_last_update, fetched_at)
                   VALUES (:event_id, :sport_key, :commence_time, :home_team, :away_team,
                   :book_key, :book_title, :outcomes, :odds, :book_last_update, :fetched_at)""",
                [{**r, "outcomes": json.dumps(r["outcomes"]), "odds": json.dumps(r["odds"])} for r in rows],
            )
        return len(rows)

    def upcoming_events(self, after_iso: str | None = None) -> list[dict]:
        after_iso = after_iso or now_iso()
        cur = self._conn.execute(
            f"""SELECT {_EVENT_COLS} FROM snapshots WHERE commence_time >= ?
               GROUP BY event_id ORDER BY MAX(commence_time)""",
            (after_iso,),
        )
        return [dict(r) for r in cur.fetchall()]

    def event(self, event_id: str) -> dict | None:
        r = self._conn.execute(
            f"SELECT {_EVENT_COLS} FROM snapshots WHERE event_id = ? GROUP BY event_id",
            (event_id,),
        ).fetchone()
        return dict(r) if r else None

    def latest_quotes(self, event_id: str, before_iso: str | None = None) -> list[dict]:
        """Most recent quote per bookmaker, optionally as of a moment (e.g. kick-off for closing line)."""
        params: list = [event_id]
        cond = ""
        if before_iso:
            cond = "AND fetched_at <= ?"
            params.append(before_iso)
        cur = self._conn.execute(
            f"""SELECT s.* FROM snapshots s JOIN (
                  SELECT book_key, MAX(fetched_at) AS m FROM snapshots
                  WHERE event_id = ? {cond} GROUP BY book_key
                ) last ON s.book_key = last.book_key AND s.fetched_at = last.m
                WHERE s.event_id = ?""",
            params + [event_id],
        )
        out = []
        for r in cur.fetchall():
            d = dict(r)
            d["outcomes"] = json.loads(d["outcomes"])
            d["odds"] = json.loads(d["odds"])
            out.append(d)
        return out

    # ---- credits -------------------------------------------------------
    def record_credits(self, endpoint: str, cost: int | None, used: int | None, remaining: int | None) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT INTO credits (at, endpoint, cost, used, remaining) VALUES (?,?,?,?,?)",
                (now_iso(), endpoint, cost, used, remaining),
            )

    def last_known_remaining(self) -> int | None:
        r = self._conn.execute(
            "SELECT remaining FROM credits WHERE remaining IS NOT NULL ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return r["remaining"] if r else None

    def credit_log(self, limit: int = 20) -> list[dict]:
        cur = self._conn.execute("SELECT * FROM credits ORDER BY id DESC LIMIT ?", (limit,))
        return [dict(r) for r in cur.fetchall()]

    # ---- predictions ---------------------------------------------------
    def log_prediction(self, event_id: str, version: str, outcomes: list[str], probs: list[float], best_odds: list[float]) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT INTO predictions (event_id, model_version, outcomes, probs, best_odds, made_at) VALUES (?,?,?,?,?,?)",
                (event_id, version, json.dumps(outcomes), json.dumps(probs), json.dumps(best_odds), now_iso()),
            )

    # ---- bets ----------------------------------------------------------
    def add_bet(self, event_id: str, outcome: str, book: str, odds: float, stake: float, note: str | None = None) -> int:
        with self.tx() as c:
            cur = c.execute(
                "INSERT INTO bets (event_id, outcome, book, odds, stake, placed_at, note) VALUES (?,?,?,?,?,?,?)",
                (event_id, outcome, book, odds, stake, now_iso(), note),
            )
            return int(cur.lastrowid)

    def settle_bet(self, bet_id: int, result: str, closing_fair_prob: float | None) -> bool:
        if result not in {"won", "lost", "void"}:
            raise ValueError("result must be won, lost or void")
        with self.tx() as c:
            cur = c.execute(
                "UPDATE bets SET result = ?, closing_fair_prob = COALESCE(?, closing_fair_prob) WHERE id = ?",
                (result, closing_fair_prob, bet_id),
            )
            return cur.rowcount == 1

    def set_closing(self, bet_id: int, closing_fair_prob: float) -> None:
        with self.tx() as c:
            c.execute("UPDATE bets SET closing_fair_prob = ? WHERE id = ?", (closing_fair_prob, bet_id))

    def bets(self) -> list[dict]:
        return [dict(r) for r in self._conn.execute("SELECT * FROM bets ORDER BY id DESC").fetchall()]

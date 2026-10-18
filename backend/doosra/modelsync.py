"""Keep the trained models in the database, so every machine sees the same champion.

Cloud Run has no disk that survives a restart, and the weekly learning cycle runs on a
fresh GitHub runner. Both point DOOSRA_DB at the same Postgres, so:

- push(): after a learning cycle, upload the files the registries need (current
  champions, their index files, the run history and the market calibrators).
- pull(): on API start (and before a cycle), write those files back into the local
  models folder. The file-based registries then work exactly as they do locally.

Only the live set is kept: files that are no longer referenced are removed from the
table, so the free database tier never fills up with old models.
"""
from __future__ import annotations

import json
from pathlib import Path

from .storage import Store, now_iso

SCHEMA = """
CREATE TABLE IF NOT EXISTS artifacts (
  name TEXT PRIMARY KEY,
  data BLOB NOT NULL,
  updated_at TEXT NOT NULL
)
"""
INDEXES = ("ball/champion.json", "champion.json")
ALWAYS = ("ball/history.jsonl", "ball/calibrators.joblib")


def _init(store: Store) -> None:
    store._conn.execute(SCHEMA)
    store._conn.commit()


def live_files(models_dir: str | Path) -> list[str]:
    """Relative paths of the files the API needs: champions, their indexes, history, calibrators."""
    root = Path(models_dir)
    out = []
    for idx in INDEXES:
        p = root / idx
        if p.exists():
            out.append(idx)
            meta = json.loads(p.read_text())
            model = (p.parent / meta["file"]).relative_to(root).as_posix()
            if (root / model).exists():
                out.append(model)
            sidecar = Path(model).with_suffix(".json").as_posix()
            if sidecar != idx and (root / sidecar).exists():
                out.append(sidecar)
    out += [f for f in ALWAYS if (root / f).exists()]
    return out


def push(store: Store, models_dir: str | Path) -> dict:
    _init(store)
    root = Path(models_dir)
    names = live_files(root)
    with store.tx() as c:
        for name in names:
            c.execute("INSERT OR REPLACE INTO artifacts (name, data, updated_at) VALUES (?, ?, ?)",
                      (name, (root / name).read_bytes(), now_iso()))
        stale = [r["name"] for r in c.execute("SELECT name FROM artifacts").fetchall() if r["name"] not in names]
        for name in stale:
            c.execute("DELETE FROM artifacts WHERE name = ?", (name,))
    return {"uploaded": names, "removed": stale}


def pull(store: Store, models_dir: str | Path) -> dict:
    _init(store)
    root = Path(models_dir).resolve()
    written = []
    for r in store._conn.execute("SELECT name, data FROM artifacts").fetchall():
        target = (root / r["name"]).resolve()
        if root not in target.parents:          # never write outside the models folder
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(bytes(r["data"]))
        written.append(r["name"])
    return {"downloaded": sorted(written)}


def stamp(store: Store) -> str | None:
    """When the live champion was last published (None if nothing has been)."""
    _init(store)
    row = store._conn.execute("SELECT updated_at FROM artifacts WHERE name = ?", ("ball/champion.json",)).fetchone()
    return row["updated_at"] if row else None


class ModelWatcher:
    """Keeps the API on the newest published model without a restart.

    The weekly learning cycle publishes to the database; at most every `interval` seconds
    the API checks the publish time and, if it changed, pulls the files and reloads.
    `load(models_dir)` builds whatever the API keeps in memory (model, calibrators).
    """

    def __init__(self, store: Store, models_dir: str | Path, load, interval: float = 600.0, clock=None):
        import threading
        import time
        self.store, self.models_dir, self._load, self.interval = store, Path(models_dir), load, interval
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._stamp = stamp(store) if store.is_remote else None
        self._checked = self._clock()
        self.value = load(self.models_dir)

    def current(self):
        if self.store.is_remote and self._clock() - self._checked >= self.interval:
            with self._lock:
                if self._clock() - self._checked >= self.interval:
                    self._checked = self._clock()
                    try:
                        s = stamp(self.store)
                        if s and s != self._stamp:
                            pull(self.store, self.models_dir)
                            self.value = self._load(self.models_dir)
                            self._stamp = s
                    except Exception:        # database asleep or unreachable: keep serving the loaded model
                        pass
        return self.value

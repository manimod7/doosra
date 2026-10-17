"""Parse Cricsheet JSON match files into flat match records.

Input: a directory (or zip) of Cricsheet *.json files, e.g. ipl_json.zip.
Output: list[MatchRecord] sorted by date, containing only decided matches
between two named teams (no-results, ties and abandoned games are skipped
for the win-probability baseline).
"""
from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class MatchRecord:
    match_id: str
    date: date
    season: str
    team_a: str
    team_b: str
    winner: str
    venue: str | None
    toss_winner: str | None
    toss_decision: str | None
    event: str | None
    match_type: str = "unknown"  # Cricsheet: T20, ODI, Test, IT20, ODM, MDM
    gender: str = "male"

    @property
    def pool(self) -> str:
        """Rating pool key. Formats and genders never share one Elo table."""
        return f"{self.match_type}:{self.gender}"


def _iter_json(path: Path) -> Iterator[tuple[str, dict]]:
    if path.is_file() and path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                if name.endswith(".json"):
                    with zf.open(name) as fh:
                        yield Path(name).stem, json.load(fh)
    else:
        for f in sorted(path.glob("*.json")):
            with f.open() as fh:
                yield f.stem, json.load(fh)


def parse_match(match_id: str, data: dict) -> MatchRecord | None:
    info = data.get("info", {})
    teams = info.get("teams") or []
    outcome = info.get("outcome", {})
    winner = outcome.get("winner")
    if len(teams) != 2 or not winner or winner not in teams:
        return None
    dates = info.get("dates") or []
    if not dates:
        return None
    toss = info.get("toss", {})
    event = info.get("event", {})
    return MatchRecord(
        match_id=match_id,
        date=date.fromisoformat(dates[0]),
        season=str(info.get("season", dates[0][:4])),
        team_a=teams[0],
        team_b=teams[1],
        winner=winner,
        venue=info.get("venue"),
        toss_winner=toss.get("winner"),
        toss_decision=toss.get("decision"),
        event=event.get("name") if isinstance(event, dict) else None,
        match_type=str(info.get("match_type") or "unknown"),
        gender=str(info.get("gender") or "male"),
    )


def load_matches(path: str | Path) -> list[MatchRecord]:
    records = []
    for match_id, data in _iter_json(Path(path)):
        rec = parse_match(match_id, data)
        if rec is not None:
            records.append(rec)
    records.sort(key=lambda r: (r.date, r.match_id))
    return records

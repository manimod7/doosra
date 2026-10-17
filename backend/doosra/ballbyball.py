"""Cricsheet ball-by-ball parsing.

Every delivery is reduced to one of eight outcome classes. The simplifications
are deliberate and listed here so nobody mistakes them for data:

  0 dot   1 one   2 two   3 three   4 four   5 six   6 wicket   7 extra
- byes and leg-byes count as runs off a legal ball (scored to the total only)
- wides and no-balls are "extra": 1 run, the ball is re-bowled
- 5 runs is counted as a four, 7+ as a six (both very rare)
- a wicket ball scores its runs to the total but is classed as "wicket"
"""
from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterator

DOT, ONE, TWO, THREE, FOUR, SIX, WICKET, EXTRA = range(8)
N_CLASSES = 8
CLASS_RUNS = (0, 1, 2, 3, 4, 6, 0, 1)
LEGAL = (True, True, True, True, True, True, True, False)
CLASS_NAMES = ("dot", "one", "two", "three", "four", "six", "wicket", "extra")


@dataclass(frozen=True)
class Ball:
    innings: int          # 1 or 2 (super overs dropped)
    over: int             # 0-based over number
    legal_before: int     # legal balls already bowled in this innings
    batting: str
    bowling: str
    batter: str
    bowler: str
    non_striker: str
    outcome: int
    runs: int             # runs added to the total by this delivery
    batter_runs: int
    player_out: str | None


@dataclass
class BBMatch:
    match_id: str
    date: date
    match_type: str
    gender: str
    venue: str
    city: str | None
    teams: tuple[str, str]
    toss_winner: str | None
    toss_decision: str | None
    winner: str | None            # None for tie / no result
    result: str                   # "win", "tie", "no result"
    overs_limit: int
    players: dict[str, list[str]]
    innings: list[list[Ball]] = field(default_factory=list)
    batting_first: str | None = None
    target: int | None = None

    @property
    def pool(self) -> str:
        return f"{self.match_type}:{self.gender}"

    def batting_order(self, team: str) -> list[str]:
        """Order of first appearance at the crease, then unused XI members."""
        seen: list[str] = []
        for inn in self.innings:
            for b in inn:
                if b.batting != team:
                    continue
                for name in (b.batter, b.non_striker):
                    if name not in seen:
                        seen.append(name)
        for name in self.players.get(team, []):
            if name not in seen:
                seen.append(name)
        return seen

    def bowlers_used(self, team: str) -> list[str]:
        out: list[str] = []
        for inn in self.innings:
            for b in inn:
                if b.bowling == team and b.bowler not in out:
                    out.append(b.bowler)
        return out


def classify(delivery: dict) -> tuple[int, int, int]:
    """Return (outcome class, runs to total, batter runs)."""
    runs = delivery.get("runs", {})
    total = int(runs.get("total", 0))
    batter_runs = int(runs.get("batter", 0))
    extras = delivery.get("extras", {}) or {}
    if delivery.get("wickets"):
        return WICKET, total, batter_runs
    if "wides" in extras or "noballs" in extras:
        return EXTRA, total, batter_runs
    if total <= 0:
        return DOT, 0, batter_runs
    if total >= 7:
        return SIX, total, batter_runs
    return {1: ONE, 2: TWO, 3: THREE, 4: FOUR, 5: FOUR, 6: SIX}[total], total, batter_runs


def parse_match(match_id: str, data: dict) -> BBMatch | None:
    info = data.get("info", {})
    teams = info.get("teams") or []
    dates = info.get("dates") or []
    if len(teams) != 2 or not dates or not data.get("innings"):
        return None
    outcome = info.get("outcome", {}) or {}
    winner = outcome.get("winner")
    result = "win" if winner else str(outcome.get("result", "no result"))
    toss = info.get("toss", {}) or {}
    m = BBMatch(
        match_id=match_id,
        date=date.fromisoformat(dates[0]),
        match_type=str(info.get("match_type", "unknown")),
        gender=str(info.get("gender", "male")),
        venue=str(info.get("venue", "unknown")),
        city=info.get("city"),
        teams=(teams[0], teams[1]),
        toss_winner=toss.get("winner"),
        toss_decision=toss.get("decision"),
        winner=winner if winner in teams else None,
        result=result,
        overs_limit=int(info.get("overs") or (20 if info.get("match_type") == "T20" else 50)),
        players={k: list(v) for k, v in (info.get("players") or {}).items()},
    )
    regular = [inn for inn in data["innings"] if not inn.get("super_over")][:2]
    for idx, inn in enumerate(regular, start=1):
        batting = inn.get("team")
        if batting not in teams:
            return None
        bowling = teams[1] if batting == teams[0] else teams[0]
        balls: list[Ball] = []
        legal = 0
        for ov in inn.get("overs", []):
            for d in ov.get("deliveries", []):
                cls, total, bruns = classify(d)
                outs = d.get("wickets") or []
                balls.append(Ball(idx, int(ov.get("over", 0)), legal, batting, bowling,
                                  d.get("batter", ""), d.get("bowler", ""), d.get("non_striker", ""),
                                  cls, total, bruns, outs[0].get("player_out") if outs else None))
                if LEGAL[cls]:
                    legal += 1
        if idx == 1:
            m.batting_first = batting
        if idx == 2 and isinstance(inn.get("target"), dict):
            m.target = inn["target"].get("runs")
        m.innings.append(balls)
    if m.batting_first and m.target is None and m.innings:
        m.target = sum(b.runs for b in m.innings[0]) + 1
    return m


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


def _standard_format(info: dict) -> bool:
    """Six-ball overs and the format's normal length. Drops The Hundred (5-ball sets), which
    Cricsheet files under T20, and the odd mislabelled match, so the simulator only learns
    from the game it simulates."""
    if int(info.get("balls_per_over", 6)) != 6:
        return False
    expected = {"T20": 20, "IT20": 20, "ODI": 50, "ODM": 50}.get(info.get("match_type"))
    overs = info.get("overs")
    return expected is None or overs is None or int(overs) == expected


def load_bb_matches(path: str | Path, pools: tuple[str, ...] | None = None) -> list[BBMatch]:
    if not Path(path).exists():
        raise FileNotFoundError(f"Cricsheet data not found at {path}")
    out = []
    for mid, data in _iter_json(Path(path)):
        info = data.get("info", {})
        pool = f"{info.get('match_type')}:{info.get('gender', 'male')}"
        if pools and pool not in pools:
            continue
        if not _standard_format(info):
            continue
        m = parse_match(mid, data)
        if m is not None and m.innings:
            out.append(m)
    out.sort(key=lambda m: (m.date, m.match_id))
    return out

"""Synthetic T20 matches in exact Cricsheet JSON format, with KNOWN truths.

Used only for tests and demos: because we choose every player's skill, the
venue effects, a batter-v-bowler rivalry and an evening dew effect, we can
check that the pipeline actually recovers them. Never use this to bet.

Ground truths planted
- batters: aggression a in [-0.6, 0.8]; bowlers: control c in [-0.5, 0.6]
- venues: "Flat Deck" scores ~20% more, "Slow Turner" ~20% less
- rivalry: "B_Rival" is dismissed far more often by "W_Nemesis"
- dew: at "Dew Bowl" chasing sides score more in the second innings
"""
from __future__ import annotations

import json
import math
import random
import zipfile
from datetime import date, timedelta
from pathlib import Path

VENUES = {"Flat Deck": 1.25, "Slow Turner": 0.78, "Dew Bowl": 1.0, "Neutral Park": 1.0}


def _team_rosters(rng: random.Random, n_teams: int = 6):
    teams = {}
    for t in range(n_teams):
        name = f"Team {chr(65 + t)}"
        bats = [(f"{name[-1]}_Bat{i}", rng.uniform(-0.6, 0.8) - i * 0.06) for i in range(7)]
        alls = [(f"{name[-1]}_All{i}", rng.uniform(-0.4, 0.3)) for i in range(2)]
        bowls = [(f"{name[-1]}_Bowl{i}", rng.uniform(-0.5, 0.6)) for i in range(4)]
        teams[name] = {"bat": bats + alls + [(b, -1.0) for b, _ in bowls[:2]],
                       "bowl": bowls + [(a, s * 0.5) for a, s in alls]}
    # the planted rivalry
    teams["Team A"]["bat"][1] = ("B_Rival", 0.5)
    teams["Team B"]["bowl"][0] = ("W_Nemesis", 0.1)
    return teams


def _ball_probs(agg: float, ctrl: float, venue_f: float, phase: int, wkts: int, dew: float, rival: bool):
    # base T20 per-ball rates: dot, 1, 2, 3, 4, 6, wicket, extra
    base = [0.36, 0.33, 0.07, 0.005, 0.11, 0.045, 0.05, 0.03]
    if phase == 0:
        base[4] += 0.03; base[0] += 0.02; base[1] -= 0.05
    elif phase == 2:
        base[5] += 0.03; base[4] += 0.02; base[6] += 0.02; base[0] -= 0.04; base[1] -= 0.03
    run_mult = math.exp(0.35 * agg - 0.35 * ctrl) * venue_f * (1 + dew)
    w = list(base)
    for c in (4, 5):
        w[c] *= run_mult
    w[2] *= math.sqrt(run_mult)
    w[6] *= math.exp(0.2 * agg + 0.45 * ctrl + 0.05 * wkts) * (3.0 if rival else 1.0)
    s = sum(w)
    return [x / s for x in w]


def generate(n_matches: int = 400, seed: int = 11, start: date = date(2014, 1, 1)) -> list[dict]:
    rng = random.Random(seed)
    teams = _team_rosters(rng)
    names = list(teams)
    out = []
    d = start
    for mi in range(n_matches):
        d += timedelta(days=rng.choice([2, 3, 4]))
        a, b = rng.sample(names, 2)
        venue = rng.choice(list(VENUES))
        toss_w = rng.choice([a, b])
        decision = "field" if (venue == "Dew Bowl" or rng.random() < 0.55) else "bat"
        first = toss_w if decision == "bat" else (b if toss_w == a else a)
        second = b if first == a else a
        innings_json, target = [], None
        totals = []
        for inn_no, (bat, bowl) in enumerate(((first, second), (second, first)), start=1):
            order = teams[bat]["bat"]
            bowlers = teams[bowl]["bowl"]
            quota = {n: 0 for n, _ in bowlers}
            striker, non, nxt = 0, 1, 2
            score = wk = legal = 0
            overs_json = []
            last_bowler = None
            done = False
            for ov in range(20):
                if done:
                    break
                choices = [x for x in bowlers if quota[x[0]] < 4 and x[0] != last_bowler]
                bname, bctrl = rng.choice(choices)
                quota[bname] += 1
                last_bowler = bname
                dels = []
                legal_in_over = 0
                while legal_in_over < 6:
                    phase = 0 if ov < 6 else (1 if ov < 15 else 2)
                    dew = 0.12 if (venue == "Dew Bowl" and inn_no == 2) else 0.0
                    sname, sagg = order[striker]
                    rival = sname == "B_Rival" and bname == "W_Nemesis"
                    p = _ball_probs(sagg, bctrl, VENUES[venue], phase, wk, dew, rival)
                    cls = rng.choices(range(8), weights=p)[0]
                    runs = (0, 1, 2, 3, 4, 6, 0, 1)[cls]
                    dj = {"batter": sname, "bowler": bname, "non_striker": order[non][0],
                          "runs": {"batter": runs if cls < 6 else 0, "extras": 1 if cls == 7 else 0, "total": runs}}
                    if cls == 7:
                        dj["extras"] = {"wides": 1}
                    if cls == 6:
                        dj["wickets"] = [{"player_out": sname, "kind": "caught"}]
                    dels.append(dj)
                    score += runs
                    if cls != 7:
                        legal_in_over += 1
                        legal += 1
                    if cls == 6:
                        wk += 1
                        if wk == 10:
                            done = True
                            break
                        striker = nxt
                        nxt += 1
                    elif runs in (1, 3) and cls != 7:
                        striker, non = non, striker
                    if target is not None and score >= target:
                        done = True
                        break
                overs_json.append({"over": ov, "deliveries": dels})
                striker, non = non, striker
            inn = {"team": bat, "overs": overs_json}
            if inn_no == 2:
                inn["target"] = {"overs": 20, "runs": target}
            innings_json.append(inn)
            totals.append(score)
            if inn_no == 1:
                target = score + 1
        if totals[1] >= target:
            outcome = {"winner": second, "by": {"wickets": 1}}
        elif totals[1] == target - 1:
            outcome = {"result": "tie"}
        else:
            outcome = {"winner": first, "by": {"runs": target - 1 - totals[1]}}
        info = {
            "dates": [d.isoformat()], "match_type": "T20", "gender": "male", "overs": 20,
            "teams": [a, b], "venue": venue, "city": venue,
            "toss": {"winner": toss_w, "decision": decision}, "outcome": outcome,
            "players": {t: [n for n, _ in teams[t]["bat"]] for t in (a, b)},
            "season": str(d.year),
        }
        out.append({"meta": {"synthetic": True}, "info": info, "innings": innings_json, "_id": f"s{mi:05d}"})
    return out


def write_zip(path: str | Path, matches: list[dict]) -> Path:
    path = Path(path)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for m in matches:
            mid = m.pop("_id")
            zf.writestr(f"{mid}.json", json.dumps(m))
            m["_id"] = mid
    return path

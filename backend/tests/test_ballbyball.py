import json
import zipfile

import pytest

from doosra.ballbyball import (DOT, EXTRA, FOUR, ONE, SIX, TWO, WICKET, classify, load_bb_matches, parse_match)


def d(batter_runs=0, extras=0, extra_kind=None, wicket=False, total=None):
    out = {"batter": "A", "bowler": "X", "non_striker": "B",
           "runs": {"batter": batter_runs, "extras": extras, "total": batter_runs + extras if total is None else total}}
    if extra_kind:
        out["extras"] = {extra_kind: extras}
    if wicket:
        out["wickets"] = [{"player_out": "A", "kind": "bowled"}]
    return out


@pytest.mark.parametrize("delivery, expected", [
    (d(0), DOT), (d(1), ONE), (d(2), TWO), (d(4), FOUR), (d(6), SIX),
    (d(0, 1, "wides"), EXTRA), (d(0, 1, "noballs"), EXTRA), (d(4, 1, "noballs"), EXTRA),
    (d(0, 1, "legbyes"), ONE), (d(0, 4, "byes"), FOUR), (d(0, wicket=True), WICKET),
    (d(1, wicket=True), WICKET), (d(5), FOUR), (d(7), SIX),
])
def test_classify(delivery, expected):
    assert classify(delivery)[0] == expected


def test_classify_keeps_true_runs():
    cls, total, batter = classify(d(4, 1, "noballs"))
    assert (cls, total, batter) == (EXTRA, 5, 4)


def _match(innings, outcome=None, **info):
    base = {"teams": ["A", "B"], "dates": ["2020-01-01"], "match_type": "T20", "gender": "male", "overs": 20,
            "venue": "V", "toss": {"winner": "A", "decision": "bat"}, "outcome": outcome or {"winner": "A"},
            "players": {"A": ["a1", "a2", "a3"], "B": ["b1", "b2"]}}
    base.update(info)
    return {"info": base, "innings": innings}


def _inn(team, overs, target=None, super_over=False):
    inn = {"team": team, "overs": overs}
    if target:
        inn["target"] = {"overs": 20, "runs": target}
    if super_over:
        inn["super_over"] = True
    return inn


def test_parse_counts_legal_balls_and_target():
    over = [{"over": 0, "deliveries": [d(1) | {"batter": "a1", "non_striker": "a2"},
                                       d(0, 1, "wides") | {"batter": "a2", "non_striker": "a1"},
                                       d(4) | {"batter": "a2", "non_striker": "a1"}]}]
    m = parse_match("m", _match([_inn("A", over), _inn("B", over, target=7)]))
    assert m.batting_first == "A" and m.target == 7
    legal_before = [b.legal_before for b in m.innings[0]]
    assert legal_before == [0, 1, 1]          # the wide does not advance the count
    assert m.batting_order("A")[:2] == ["a1", "a2"] and m.batting_order("A")[2] == "a3"


def test_parse_drops_super_overs_and_bad_teams():
    over = [{"over": 0, "deliveries": [d(1)]}]
    m = parse_match("m", _match([_inn("A", over), _inn("B", over), _inn("B", over, super_over=True)],
                                outcome={"result": "tie"}))
    assert len(m.innings) == 2 and m.winner is None and m.result == "tie"
    assert parse_match("m", _match([_inn("Z", over)])) is None
    assert parse_match("m", {"info": {"teams": ["A"]}, "innings": []}) is None


def test_target_inferred_when_missing():
    over = [{"over": 0, "deliveries": [d(6), d(4)]}]
    m = parse_match("m", _match([_inn("A", over), _inn("B", over)]))
    assert m.target == 11


def test_load_filters_pools_and_sorts(tmp_path):
    z = tmp_path / "x.zip"
    over = [{"over": 0, "deliveries": [d(1)]}]
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("2.json", json.dumps(_match([_inn("A", over)], dates=["2021-01-01"])))
        zf.writestr("1.json", json.dumps(_match([_inn("A", over)], dates=["2020-01-01"])))
        zf.writestr("3.json", json.dumps(_match([_inn("A", over)], match_type="ODI")))
    ms = load_bb_matches(z, ("T20:male",))
    assert [m.match_id for m in ms] == ["1", "2"]


def test_load_missing_path_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_bb_matches(tmp_path / "nope.zip")


def test_loader_skips_hundred_and_mislabelled_lengths(tmp_path, synth_zip):
    """The Hundred (5-ball sets) and 50-over games filed as T20 must not reach the 20-over model."""
    import json, zipfile
    from doosra.ballbyball import load_bb_matches
    src = zipfile.ZipFile(synth_zip)
    first = next(n for n in src.namelist() if n.endswith(".json"))
    base = json.loads(src.read(first))
    out = tmp_path / "mixed.zip"
    with zipfile.ZipFile(out, "w") as z:
        for name, patch in [("ok", {}), ("hundred", {"balls_per_over": 5}), ("fifty", {"overs": 50})]:
            d = json.loads(json.dumps(base)); d["info"].update(patch)
            z.writestr(f"{name}.json", json.dumps(d))
    got = load_bb_matches(out, (f"{base['info']['match_type']}:{base['info'].get('gender', 'male')}",))
    assert [m.match_id for m in got] == ["ok"]


def test_ground_names_merge_but_generic_names_keep_their_city():
    from doosra.ballbyball import canonical_venue
    assert canonical_venue("Wankhede Stadium, Mumbai", "Mumbai") == canonical_venue("Wankhede Stadium", None)
    assert canonical_venue("M.Chinnaswamy Stadium", "Bangalore") == canonical_venue("M Chinnaswamy Stadium, Bengaluru", "Bengaluru")
    assert canonical_venue("County Ground", "Taunton") != canonical_venue("County Ground, Chelmsford", "Chelmsford")
    assert canonical_venue("National Stadium", "Karachi") != canonical_venue("National Stadium, Hamilton", "Hamilton")
    assert canonical_venue(None) == "unknown"


def test_competition_names_survive_sponsor_changes():
    from doosra.ballbyball import competition_of
    blast = {competition_of({"team_type": "club", "event": {"name": n}})
             for n in ("NatWest T20 Blast", "Vitality Blast", "Vitality Blast Men")}
    assert blast == {"T20 Blast"}
    assert competition_of({"team_type": "international", "match_type": "T20", "event": {"name": "Asia Cup"}}) == "T20I"
    assert competition_of({"team_type": "club", "event": {"name": "Indian Premier League"}}) == "Indian Premier League"

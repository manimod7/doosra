import numpy as np
import pytest

from doosra.props import MarketSpec, market_values, price, settle, sim_from_match, standard_card
from doosra.simulate import LiveState, Lineup, MatchSetup, default_setup, simulate_match


@pytest.fixture()
def sim(logit_model, state):
    setup = default_setup(state, "T20:male", "Neutral Park", "Team A", "Team B", batting_first="Team A")
    return simulate_match(logit_model, state, setup, n=500, seed=7)


# ---------------------------------------------------------------- cricket rules

def test_innings_rules_hold_in_every_simulation(sim):
    for t in sim.teams:
        inn = sim.innings[t]
        assert (inn.wickets <= 10).all() and (inn.total >= 0).all()
        sao = inn.score_after_over
        assert np.all(np.diff(sao, axis=1) >= 0)                   # scores never go down
        np.testing.assert_allclose(sao[:, -1], inn.total)          # last over equals the total
        assert (np.diff(inn.wickets_after_over, axis=1) >= 0).all()
        bat_runs = sum(inn.batter_runs.values())
        assert (bat_runs <= inn.total + 1e-9).all()                # extras only add to the total
        assert all((b <= 120).all() for b in inn.batter_balls.values())


def test_chase_stops_at_the_target(sim):
    first, second = sim.innings["Team A"], sim.innings["Team B"]
    assert (second.total <= first.total + 1 + 6).all()            # can overshoot by one shot at most
    won = second.total >= first.total + 1
    assert np.allclose(sim.winner[won], 1)


def test_winner_probabilities_sum_to_one(sim):
    assert sim.win_prob("Team A") + sim.win_prob("Team B") == pytest.approx(1.0)


def test_same_seed_same_result(logit_model, state):
    setup = default_setup(state, "T20:male", "Flat Deck", "Team C", "Team D")
    a = simulate_match(logit_model, state, setup, n=200, seed=3)
    b = simulate_match(logit_model, state, setup, n=200, seed=3)
    np.testing.assert_array_equal(a.innings["Team C"].total, b.innings["Team C"].total)


def test_venue_effect_survives_simulation(logit_model, state):
    def mean_total(venue):
        s = simulate_match(logit_model, state, default_setup(state, "T20:male", venue, "Team A", "Team B",
                                                             batting_first="Team A"), n=600, seed=1)
        return s.innings["Team A"].total.mean()
    assert mean_total("Flat Deck") > mean_total("Slow Turner") + 15


def test_live_state_is_respected(logit_model, state):
    setup = default_setup(state, "T20:male", "Neutral Park", "Team A", "Team B")
    live = LiveState(batting="Team B", innings=2, score=150, wickets=4, legal_balls=108, target=160)
    s = simulate_match(logit_model, state, setup, n=400, seed=2, live=live)
    inn = s.innings["Team B"]
    assert (inn.total >= 150).all() and (inn.wickets >= 4).all()
    assert np.isnan(inn.score_after_over[:, 17]).all()              # overs before the live start are unknown
    assert 0.3 < s.win_prob("Team B") < 0.98                        # 10 off 12 with 6 wickets: favoured
    with pytest.raises(ValueError):
        simulate_match(logit_model, state, setup, n=10, live=LiveState("Team B", 2, 10, 0, 6))


def test_lineup_validation():
    with pytest.raises(ValueError):
        Lineup(["only one"], ["x"])
    with pytest.raises(ValueError):
        Lineup(["a", "b"], [])


def test_missing_lineup_history_is_explained(state):
    with pytest.raises(ValueError, match="supply the XI"):
        default_setup(state, "T20:male", "V", "Team A", "Nobody XI")


def test_short_bowling_lineup_still_completes(logit_model, state):
    setup = MatchSetup("T20:male", "Neutral Park", ("Team A", "Team B"),
                       {"Team A": Lineup(["A_Bat0", "A_Bat1", "A_Bat2"], ["A_Bowl0"]),
                        "Team B": Lineup(["B_Bat0", "B_Bat1"], ["B_Bowl0", "B_Bowl1"])}, batting_first="Team A")
    s = simulate_match(logit_model, state, setup, n=100, seed=0)
    assert (s.innings["Team A"].wickets <= 2).all()                # 3 batters: all out at 2 wickets


# ---------------------------------------------------------------- markets

@pytest.mark.parametrize("bad", [dict(type="session", team="A"), dict(type="nope"), dict(type="team_total"),
                                 dict(type="player_runs"), dict(type="over_runs", team="A")])
def test_market_spec_validation(bad):
    with pytest.raises(ValueError):
        MarketSpec(**bad)


def test_line_probabilities_are_coherent(sim):
    spec = MarketSpec("team_total", team="Team A")
    lines = [120.5, 140.5, 160.5, 180.5]
    ps = [price(sim, spec, ln).p_over for ln in lines]
    assert ps == sorted(ps, reverse=True)                          # higher line, lower P(over)
    lp = price(sim, spec, 150)                                     # integer line can push
    assert lp.p_over + lp.p_under + lp.p_push == pytest.approx(1)
    half = price(sim, spec)
    assert half.p_push == 0 and half.line % 1 == 0.5


def test_sessions_nest_inside_the_innings(sim):
    s6 = market_values(sim, MarketSpec("session", team="Team A", overs=6))
    s10 = market_values(sim, MarketSpec("session", team="Team A", overs=10))
    tot = market_values(sim, MarketSpec("team_total", team="Team A"))
    assert (s6 <= s10).all() and (s10 <= tot).all()
    over_sum = sum(market_values(sim, MarketSpec("over_runs", team="Team A", over=k)) for k in range(1, 21))
    np.testing.assert_allclose(over_sum, tot)


def test_match_markets_add_up(sim):
    tot = market_values(sim, MarketSpec("match_total"))
    np.testing.assert_allclose(tot, sim.innings["Team A"].total + sim.innings["Team B"].total)


def test_unknown_player_is_an_error(sim):
    with pytest.raises(ValueError):
        market_values(sim, MarketSpec("player_runs", player="Nobody"))


def test_standard_card_shape(sim):
    card = standard_card(sim)
    assert set(card["winner"]) == {"Team A", "Team B"}
    assert sum(card["winner"].values()) == pytest.approx(1, abs=1e-3)
    assert len(card["teams"]["Team A"]) == 6
    assert card["players"] == sorted(card["players"], key=lambda r: -r["expected_runs"])
    assert all(0 <= p["p_50_plus"] <= p["p_30_plus"] <= 1 for p in card["players"])
    worm = card["worm"]["Team A"]
    assert len(worm["p50"]) == 20
    assert all(a <= b <= c for a, b, c in zip(worm["p10"], worm["p50"], worm["p90"]))
    assert worm["p50"] == sorted(worm["p50"])          # the median worm only climbs


def test_real_match_rebuilds_exactly(bb_matches):
    for m in bb_matches[:30]:
        s = sim_from_match(m)
        for inn in m.innings:
            team = inn[0].batting
            assert s.innings[team].total[0] == sum(b.runs for b in inn)
            assert s.innings[team].fours[0] == sum(1 for b in inn if b.outcome == 4)


def test_settle_rules(bb_matches):
    m = next(x for x in bb_matches if x.winner and len(x.innings) == 2)
    first = m.batting_first
    total = sum(b.runs for b in m.innings[0])
    assert settle(m, MarketSpec("team_total", team=first), total - 0.5) == 1.0
    assert settle(m, MarketSpec("team_total", team=first), total + 0.5) == 0.0
    assert settle(m, MarketSpec("team_total", team=first), float(total)) is None     # push
    assert settle(m, MarketSpec("match_winner", team=m.winner), None) == 1.0
    assert settle(m, MarketSpec("player_runs", player="Nobody"), 10.5) is None        # void, not a loss


def test_match_day_volatility_widens_totals_but_zero_changes_nothing(logit_model, state):
    import numpy as np
    from doosra.simulate import simulate_match, default_setup
    setup = default_setup(state, "T20:male", "Neutral Park", "Team A", "Team B", batting_first="Team A")
    base = simulate_match(logit_model, state, setup, n=600, seed=11)
    zero = simulate_match(logit_model, state, setup, n=600, seed=11, volatility=0.0)
    wide = simulate_match(logit_model, state, setup, n=600, seed=11, volatility=0.25)
    assert np.array_equal(base.innings["Team A"].total, zero.innings["Team A"].total)
    assert wide.day is not None and base.day is None
    assert wide.innings["Team A"].total.std() > base.innings["Team A"].total.std() * 1.15
    # batting days really score more
    hi = wide.day > 0.15
    lo = wide.day < -0.15
    assert wide.innings["Team A"].total[hi].mean() > wide.innings["Team A"].total[lo].mean() + 10

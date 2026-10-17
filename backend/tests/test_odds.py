import pytest

from doosra.odds import (
    clv, devig_power, devig_proportional, edge, fair_odds,
    implied_prob, kelly_fraction, overround, verdict,
)


def test_implied_prob():
    assert implied_prob(2.0) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        implied_prob(1.0)


def test_overround_two_way():
    assert overround([1.90, 1.90]) == pytest.approx(0.0526, abs=1e-4)


@pytest.mark.parametrize("fn", [devig_proportional, devig_power])
def test_devig_sums_to_one(fn):
    probs = fn([1.60, 2.40])
    assert sum(probs) == pytest.approx(1.0, abs=1e-8)
    assert probs[0] > probs[1]


def test_power_shifts_margin_to_longshot():
    prop = devig_proportional([1.30, 4.00])
    power = devig_power([1.30, 4.00])
    assert power[1] < prop[1]  # longshot gets less probability under power method


def test_edge_and_fair_odds():
    assert edge(0.55, 2.0) == pytest.approx(0.10)
    assert fair_odds(0.5) == pytest.approx(2.0)


def test_kelly_no_edge_is_zero():
    assert kelly_fraction(0.45, 2.0) == 0.0


def test_kelly_quarter():
    # full Kelly for p=0.6, odds 2.0 is 0.2; quarter is 0.05
    assert kelly_fraction(0.6, 2.0) == pytest.approx(0.05)


def test_verdict_skip_bet_trap():
    assert verdict(0.50, 2.0).label == "SKIP"
    v = verdict(0.58, 2.0, sharp_prob=0.52)
    assert v.label == "BET" and 0 < v.kelly <= 0.05
    assert verdict(0.70, 2.0, sharp_prob=0.50).label == "TRAP"


def test_clv_positive_when_beating_close():
    assert clv(2.10, 0.50) == pytest.approx(0.05)

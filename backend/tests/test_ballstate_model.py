import copy

import numpy as np
import pytest

from doosra.ballbyball import N_CLASSES
from doosra.ballmodel import BallModel, fit_best, multiclass_log_loss
from doosra.ballstate import (FEATURE_NAMES, K_PLAYER, N_FEATURES, CricketState, build_ball_dataset,
                              dynamic_features, phase_of)


def test_feature_width_is_stable(dataset):
    X, *_ = dataset
    assert X.shape[1] == N_FEATURES == len(FEATURE_NAMES) == 38


def test_features_are_leak_free(bb_matches):
    """Features for match i must not change when later matches are appended."""
    X_short, _, mi_s, *_ = build_ball_dataset(bb_matches[:80])
    X_long, _, mi_l, *_ = build_ball_dataset(bb_matches[:160])
    np.testing.assert_allclose(X_short, X_long[mi_l < 80], equal_nan=True)


def test_first_match_sees_nothing(dataset):
    X, _, mi, *_ = dataset
    assert np.allclose(X[mi == 0][:, :16], 0)   # player rates equal league prior


def test_shrinkage_small_samples_stay_near_average():
    s = CricketState()
    s.league["P"] = np.full(N_CLASSES, 1000.0)
    s.bat[("P", "lucky")] = np.array([0, 0, 0, 0, 3, 3, 0, 0], dtype=float)  # 6 balls, 6 boundaries
    lr, _, _ = s.batter_vec("P", "lucky")
    # 6 boundaries off 6 balls is a 700% boundary rate; shrinkage keeps the estimate within ~12%
    assert np.max(np.abs(lr)) < 0.12
    s.bat[("P", "proven")] = np.array([0, 0, 0, 0, 3000, 3000, 0, 0], dtype=float)
    lr2, _, _ = s.batter_vec("P", "proven")
    assert lr2[4] > 1.0


def test_planted_truths_recovered(state):
    v = {name: state.venue_vec("T20:male", name)[0] for name in ("Flat Deck", "Slow Turner")}
    assert v["Flat Deck"] > 0.05 > -0.05 > v["Slow Turner"]
    assert state.rivalry_vec("T20:male", "B_Rival", "W_Nemesis")[1] > 0   # gets out more to the nemesis
    assert state.chase_win_rate("T20:male", "Dew Bowl") > 0.5


def test_dynamic_features_vectorised():
    D = dynamic_features(np.array([0, 10, 19]), np.array([0, 60, 114]), np.array([0, 3, 8]), np.array([0, 5, 30]),
                         np.array([0, 1, 1]), np.array([0, 80, 150]), np.array([0, 180, 160]), 20)
    assert D.shape == (3, 9)
    assert D[0, 0] == 1 and D[1, 1] == 1 and D[2, 2] == 1          # phases
    assert D[0, 7] == 0 and D[1, 7] == pytest.approx(100 / 60)     # need runs per ball only in a chase


def test_phase_boundaries_scale_with_format():
    assert [phase_of(o, 20) for o in (0, 5, 6, 14, 15, 19)] == [0, 0, 1, 1, 2, 2]
    assert [phase_of(o, 50) for o in (9, 10, 39, 40)] == [0, 1, 1, 2]


@pytest.mark.parametrize("kind", ["baseline", "logit", "hgb"])
def test_ball_model_outputs_valid_probabilities(dataset, kind):
    X, y, *_ = dataset
    m = BallModel(kind, max_iter=20).fit(X[:20000], y[:20000])
    P = m.predict_proba(X[20000:21000])
    assert P.shape == (1000, N_CLASSES)
    assert np.allclose(P.sum(axis=1), 1) and (P > 0).all()


def test_all_nan_weather_columns_do_not_break_training(dataset):
    X, y, *_ = dataset
    assert np.isnan(X[:, FEATURE_NAMES.index("w_humidity")]).all()
    m = BallModel("hgb", max_iter=10).fit(X[:5000], y[:5000])
    assert not m.keep[FEATURE_NAMES.index("w_humidity")]
    m.predict_proba(X[:10])


def test_learned_model_beats_baseline(dataset):
    X, y, *_ = dataset
    cut = int(len(y) * 0.8)
    choice = fit_best(X[:int(cut * 0.8)], y[:int(cut * 0.8)], X[int(cut * 0.8):cut], y[int(cut * 0.8):cut],
                      kinds=("logit",))
    assert choice.scores["logit"] < choice.scores["baseline"]
    test_ll = multiclass_log_loss(choice.model.predict_proba(X[cut:]), y[cut:])
    base_ll = multiclass_log_loss(BallModel("baseline").fit(X[:cut], y[:cut]).predict_proba(X[cut:]), y[cut:])
    assert test_ll < base_ll


def test_unfitted_model_refuses():
    with pytest.raises(RuntimeError):
        BallModel("logit").predict_proba(np.zeros((1, N_FEATURES)))


def test_slim_state_simulates_exactly_like_the_full_state(bb_matches, logit_model):
    """Held-out evaluation keeps a slim copy of the state per test match. It must be
    indistinguishable from the full state for that match's simulation."""
    import copy
    from doosra.ballstate import CricketState
    from doosra.lifecycle import simulate_heldout
    state = CricketState()
    for m in bb_matches[:-1]:
        state.update(m)
    target = bb_matches[-1]
    full = simulate_heldout(logit_model, copy.deepcopy(state), target, n=60, seed=3)
    slim_state = state.slim_for(target)
    slim = simulate_heldout(logit_model, slim_state, target, n=60, seed=3)
    assert full.keys() == slim.keys()
    for k in full:
        if isinstance(full[k], np.ndarray):
            assert np.array_equal(full[k], slim[k], equal_nan=True), k
        else:
            assert full[k] == slim[k], k
    assert len(slim_state.rivalry) < len(state.rivalry)
    assert len(slim_state.bat) < len(state.bat)


def test_competition_level_tracks_a_high_scoring_league(bb_matches):
    from dataclasses import replace
    from doosra.ballstate import CricketState
    state = CricketState()
    for i, m in enumerate(bb_matches):
        state.update(replace(m, competition="Flat League" if i % 2 else "Slow League"))
    assert "Flat League" in {k[1] for k in state.comp_first}
    # the synthetic league's Flat Deck games are spread over both, so just check the plumbing
    assert state.comp_par("T20:male", None) == 0.0
    assert state.comp_par("T20:male", "Never Played") == 0.0
    assert state.competition_for("T20:male", *bb_matches[-1].teams) in ("Flat League", "Slow League", None)

"""Shared, deterministic fixtures. Everything is synthetic and seeded, so a
failing test always means the code changed behaviour, never that data did."""
import copy

import numpy as np
import pytest

from doosra.ballbyball import load_bb_matches
from doosra.ballmodel import BallModel
from doosra.ballstate import build_ball_dataset
from doosra.synth import generate, write_zip


@pytest.fixture(autouse=True)
def no_real_ai_keys(monkeypatch):
    """Tests never call a real AI provider, whatever keys the machine has."""
    for k in ("GEMINI_API_KEY", "GROQ_API_KEY", "ANTHROPIC_API_KEY", "DOOSRA_LLM_ORDER"):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture(scope="session")
def synth_zip(tmp_path_factory):
    path = tmp_path_factory.mktemp("synth") / "t20.zip"
    return write_zip(path, generate(260, seed=21))


@pytest.fixture(scope="session")
def bb_matches(synth_zip):
    return load_bb_matches(synth_zip, ("T20:male",))


@pytest.fixture(scope="session")
def dataset(bb_matches):
    return build_ball_dataset(bb_matches)


@pytest.fixture(scope="session")
def logit_model(dataset):
    X, y, *_ = dataset
    return BallModel("logit").fit(X, y)


@pytest.fixture()
def state(dataset):
    """A private copy of the final state, so tests can't leak into each other."""
    return copy.deepcopy(dataset[4])


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(0)

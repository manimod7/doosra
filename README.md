# Doosra

*Read the match before it's bowled.*

A personal cricket forecasting tool. Doosra plays each T20 out thousands of
times, ball by ball, prices every common market (winner, innings runs,
6/10/15-over sessions, wickets, opening stand, batter runs, match runs, sixes,
fours), checks any bookmaker price against that, and learns from every result.
It never places bets and never logs in to or scrapes a bookmaker.

## How it works

```
Cricsheet ball-by-ball ──► player / rivalry / venue / form features (walk-forward, no leakage)
                          ──► per-ball model: P(dot, 1, 2, 3, 4, 6, wicket, extra)
                          ──► simulator: 2,000 matches with real XIs, bowler quotas, chase, toss, live state
                          ──► market prices + verdicts (BET / SKIP / TRAP, quarter-Kelly stake)
results arrive ──► settle every logged prediction ──► per-market correction ──► drift check ──► retrain behind a gate
```

- **Features** (`ballstate.py`): batter and bowler outcome rates shrunk toward the league, batter-v-bowler history,
  venue scoring and recent par, batter form, toss, phase, wickets, balls faced, required rate. Weather columns
  switch on automatically once Open-Meteo data has been fetched (`conditions.py`).
- **Model** (`ballmodel.py`): boosted trees and a logistic model are both trained; whichever scores better on held-out
  balls is used.
- **Simulator** (`simulate.py`): strike rotation, wides re-bowled, quotas, no consecutive overs, all-out, chase stop,
  ties, and in-play starts from any score.
- **Markets** (`props.py`): one function prices a market from simulations and settles it from a real scorecard, so
  predictions and results are measured the same way.
- **Lifecycle** (`lifecycle.py`, `feedback.py`, `pipeline.py`): walk-forward evaluation by year, interval-coverage
  check on simulated held-out matches, promotion only when the new model is better on unseen data, a prediction log,
  automatic settlement, one-directional isotonic correction per market type, drift flags.
- **AI layer** (`llm.py`, `explain.py`): Gemini, then Groq, then Claude, whichever keys exist. It only rewords numbers the
  simulator produced; any invented number gets the text rejected. Without keys, built-in summaries are used.
- **Odds feed** (`providers/the_odds_api.py`): The Odds API with a hard credit reserve.

## Run it

```bash
# 1. API
cd backend
cp .env.example .env                       # fill in what you have; everything is optional
pip install -e ".[dev]"
python -m doosra.pipeline download all_json.zip         # or put Cricsheet zips in data/raw yourself
python -I -m doosra.pipeline cycle data/raw/all_json.zip    # first learning cycle (uses men's T20 matches)
set -a; . ./.env; set +a
uvicorn doosra.api:app --port 8000

# 2. Dashboard (second terminal)
cd frontend
cp .env.example .env.local                 # same DOOSRA_ADMIN_TOKEN as the API
npm install && npm run dev                 # http://localhost:3000
```

## Tests

```bash
cd backend
python -m pytest -q -m "not slow"          # ~1 minute, run while editing
python -m pytest -q                        # everything, including full learning cycles (CI runs this)
```

Every test runs on seeded synthetic Cricsheet data with planted truths (a high-scoring ground, a rivalry, dew),
so a failure always means behaviour changed. Tests never call real AI providers or odds APIs.

## Free cloud setup

| Piece | Service | Notes |
|---|---|---|
| Dashboard | Vercel Hobby | `frontend/`; set `DOOSRA_API_URL`, `DOOSRA_ADMIN_TOKEN`, `DOOSRA_DASHBOARD_PASSWORD` |
| API + simulator | Google Cloud Run free tier | `backend/Dockerfile`; set a ₹0 budget alert |
| Weekly learning | GitHub Actions | `.github/workflows/learning-cycle.yml`; needs the `DOOSRA_DB` secret |
| Odds refresh | GitHub Actions | `.github/workflows/refresh-odds.yml` |
| AI explanations | Gemini API (free tier or Google AI Pro credit), Groq free tier | optional |

| Database | Neon Postgres free tier | Set `DOOSRA_DB` to the connection string on Cloud Run and as a GitHub secret |

Set `DOOSRA_DB` to a `postgresql://` URL and everything durable moves into Postgres: odds history, the
prediction log, your bets and the trained models. The weekly cycle on GitHub publishes each new champion
there, and the API loads it when it starts, so Cloud Run needs no disk.

## Honest limits

- Probabilities are calibrated estimates, not certainties; a 60% call should lose about 4 times in 10.
- Beating the closing price over many bets is the real test of an edge, not short-run profit.
- Pitch reports and team news have no historical archive to learn from yet; they are shown as context only.
- Check whether betting is legal where you live before acting on anything.

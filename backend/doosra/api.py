"""FastAPI service. Run:  uvicorn doosra.api:app --port 8000

Personal-use tool. Analysis only: no bet placement, no bookmaker logins.

Modes
- demo: no DOOSRA_ODDS_API_KEY set -> invented odds, clearly labelled
- live: odds come from the local store, filled by POST /admin/refresh
  (or the `python -m doosra.jobs refresh` cron job)

Anything that spends credits (odds refresh, LLM calls) or reads/writes your
bets needs `Authorization: Bearer <DOOSRA_ADMIN_TOKEN>`. If no admin token is
configured those endpoints are switched off.
"""
from __future__ import annotations

import hmac
import os
from dataclasses import dataclass
from datetime import date, datetime
from typing import Callable

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from . import modelsync
from .analyzer import analyze
from .demo import DEMO_MATCHES
from .explain import LLMClient, default_client, explain, extract_flags
from .markets import BookQuote, compare
from .model import Predictor
from .providers.the_odds_api import FORMAT_FOR_SPORT, TheOddsApi
from .registry import Registry
from .storage import Store, now_iso

SPORT_TITLES = {
    "cricket_international_t20": "International T20",
    "cricket_odi": "One Day International",
    "cricket_test_match": "Test match",
    "cricket_ipl": "IPL",
}


@dataclass
class Settings:
    db_path: str = "data/doosra.sqlite"
    odds_api_key: str | None = None
    admin_token: str | None = None
    models_dir: str = "models"
    cors: tuple[str, ...] = ("http://localhost:3000",)
    credit_reserve: int = 50
    cricsheet_path: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            db_path=os.environ.get("DOOSRA_DB", "data/doosra.sqlite"),
            odds_api_key=os.environ.get("DOOSRA_ODDS_API_KEY") or None,
            admin_token=os.environ.get("DOOSRA_ADMIN_TOKEN") or None,
            models_dir=os.environ.get("DOOSRA_MODELS_DIR", "models"),
            cors=tuple(os.environ.get("DOOSRA_CORS", "http://localhost:3000").split(",")),
            credit_reserve=int(os.environ.get("DOOSRA_CREDIT_RESERVE", "50")),
            cricsheet_path=os.environ.get("DOOSRA_CRICSHEET") or None,
        )


# ---------------------------------------------------------------- models ---

class QuoteIn(BaseModel):
    book: str = Field(min_length=1, max_length=40)
    odds: list[float] = Field(min_length=2, max_length=4)

    @field_validator("odds")
    @classmethod
    def odds_gt_one(cls, v: list[float]) -> list[float]:
        if any(o <= 1.0 for o in v):
            raise ValueError("decimal odds must be greater than 1")
        return v


class AnalyzeIn(BaseModel):
    outcomes: list[str] = Field(min_length=2, max_length=4)
    quotes: list[QuoteIn] = Field(min_length=1, max_length=40)
    model_probs: list[float] | None = None
    bankroll: float | None = Field(default=None, gt=0)


class ExplainIn(BaseModel):
    model_probs: list[float] | None = None
    bankroll: float | None = Field(default=None, gt=0)
    use_model: bool = False


class FlagsIn(BaseModel):
    notes: str = Field(min_length=1, max_length=4000)


class BetIn(BaseModel):
    event_id: str = Field(min_length=1, max_length=80)
    outcome: str = Field(min_length=1, max_length=80)
    book: str = Field(min_length=1, max_length=40)
    odds: float = Field(gt=1.0)
    stake: float = Field(gt=0)
    note: str | None = Field(default=None, max_length=300)


class SettleIn(BaseModel):
    result: str = Field(pattern="^(won|lost|void)$")


# ------------------------------------------------------------ helpers ---

def pool_for(sport_key: str, teams: list[str]) -> tuple[str | None, list[str]]:
    """Map an odds-feed event to a Cricsheet rating pool and normalised team names."""
    fmt = FORMAT_FOR_SPORT.get(sport_key)
    if not fmt:
        return None, teams
    female = any(t.endswith((" Women", " W")) for t in teams)
    names = [t.removesuffix(" Women").removesuffix(" W") for t in teams]
    return f"{fmt}:{'female' if female else 'male'}", names


def create_app(settings: Settings | None = None, provider_factory: Callable[[Settings, Store], TheOddsApi] | None = None,
               llm: LLMClient | None | str = "default") -> FastAPI:
    s = settings or Settings.from_env()
    store = Store(s.db_path)
    if store.is_remote:                 # Cloud Run: the models live in the database
        modelsync.pull(store, s.models_dir)
    live = bool(s.odds_api_key)
    registry = Registry(s.models_dir)
    predictor: Predictor | None = registry.load_champion()
    llm_client = default_client() if llm == "default" else llm

    def provider() -> TheOddsApi:
        if not live:
            raise HTTPException(409, "no odds API key configured (DOOSRA_ODDS_API_KEY)")
        if provider_factory:
            return provider_factory(s, store)
        return TheOddsApi(api_key=s.odds_api_key, store=store, reserve=s.credit_reserve)

    def require_admin(authorization: str | None = Header(default=None)) -> None:
        if not s.admin_token:
            raise HTTPException(403, "admin endpoints are off: set DOOSRA_ADMIN_TOKEN")
        given = (authorization or "").removeprefix("Bearer ").strip()
        if not hmac.compare_digest(given, s.admin_token):
            raise HTTPException(401, "missing or wrong admin token")

    app = FastAPI(title="Doosra", version="0.2.0")
    app.add_middleware(CORSMiddleware, allow_origins=list(s.cors), allow_methods=["GET", "POST"],
                       allow_headers=["*"])

    # -- event lookup (live store or demo) ---------------------------------
    def load_event(match_id: str) -> tuple[dict, list[BookQuote], dict]:
        if live:
            ev = store.event(match_id)
            if ev is None:
                raise HTTPException(404, "match not found")
            rows = store.latest_quotes(match_id)
            outcomes = rows[0]["outcomes"]
            quotes = [BookQuote(r["book_title"], tuple(r["odds"])) for r in rows if r["outcomes"] == outcomes]
            info = {"id": ev["event_id"], "league": SPORT_TITLES.get(ev["sport_key"], ev["sport_key"]),
                    "start": ev["commence_time"], "sport_key": ev["sport_key"], "outcomes": outcomes,
                    "odds_fetched_at": ev["fetched_at"]}
            return info, quotes, ev
        m = next((x for x in DEMO_MATCHES if x["id"] == match_id), None)
        if m is None:
            raise HTTPException(404, "match not found")
        quotes = [BookQuote(b, tuple(o)) for b, o in m["quotes"].items()]
        info = {k: m[k] for k in ("id", "league", "start", "outcomes")} | {"sport_key": "demo"}
        return info, quotes, m

    def model_view(info: dict) -> dict:
        if predictor is None:
            return {"available": False, "reason": "no promoted model yet: run doosra.train on Cricsheet data"}
        outcomes = info["outcomes"]
        if len(outcomes) != 2:
            return {"available": False, "reason": "draws are not modelled; Test match model unavailable"}
        pool, names = pool_for(info.get("sport_key", ""), outcomes)
        if pool is None:
            return {"available": False, "reason": "this competition is not mapped to a rating pool"}
        try:
            d = datetime.fromisoformat(info["start"].replace("Z", "+00:00")).date()
        except ValueError:
            d = date.today()
        p, why = predictor.predict(pool, names[0], names[1], d)
        if p is None:
            return {"available": False, "reason": why}
        return {"available": True, "probs": [p, 1 - p], "version": predictor.version,
                "trained_through": predictor.trained_through.isoformat(), "pool": pool}

    def run_analysis(match_id: str, model_probs: list[float] | None, bankroll: float | None, use_model: bool) -> dict:
        info, quotes, _ = load_event(match_id)
        mv = model_view(info)
        probs = model_probs
        source = "you" if model_probs else None
        if probs is None and use_model:
            if not mv["available"]:
                raise HTTPException(409, f"model unavailable: {mv['reason']}")
            probs, source = mv["probs"], f"model {mv['version']}"
        try:
            result = analyze(info["outcomes"], quotes, probs, bankroll)
        except ValueError as e:
            raise HTTPException(422, str(e))
        if live and source and source.startswith("model"):
            comp = compare(info["outcomes"], quotes)
            store.log_prediction(match_id, mv["version"], info["outcomes"], probs, [b.odds for b in comp.best])
        return {"data_mode": "live" if live else "demo",
                "match": {k: info[k] for k in ("id", "league", "start")} | {"odds_fetched_at": info.get("odds_fetched_at")},
                "probs_source": source, "model": mv, **result}

    # -- routes -------------------------------------------------------------
    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "data_mode": "live" if live else "demo",
                "model": predictor.version if predictor else None,
                "llm": llm_client is not None, "admin": bool(s.admin_token)}

    @app.get("/matches")
    def matches() -> dict:
        if live:
            evs = store.upcoming_events()
            out = []
            for e in evs:
                rows = store.latest_quotes(e["event_id"])
                out.append({"id": e["event_id"], "league": SPORT_TITLES.get(e["sport_key"], e["sport_key"]),
                            "start": e["commence_time"], "outcomes": rows[0]["outcomes"] if rows else
                            [e["home_team"], e["away_team"]], "books": len(rows)})
            notice = "" if out else "No upcoming matches stored yet. Run a refresh to fetch odds."
            return {"data_mode": "live", "notice": notice, "matches": out}
        return {"data_mode": "demo", "notice": "DEMO odds only. Connect an odds provider for real prices.",
                "matches": [{k: m[k] for k in ("id", "league", "start", "outcomes")} for m in DEMO_MATCHES]}

    @app.get("/matches/{match_id}/odds")
    def match_odds(match_id: str, model_probs: str | None = None, bankroll: float | None = None,
                   use_model: bool = False) -> dict:
        mp = None
        if model_probs:
            try:
                mp = [float(x) for x in model_probs.split(",")]
            except ValueError:
                raise HTTPException(422, "model_probs must be comma-separated numbers")
        return run_analysis(match_id, mp, bankroll, use_model)

    @app.post("/analyze")
    def analyze_endpoint(body: AnalyzeIn) -> dict:
        n = len(body.outcomes)
        if any(len(q.odds) != n for q in body.quotes):
            raise HTTPException(422, f"every quote needs exactly {n} odds")
        quotes = [BookQuote(q.book, tuple(q.odds)) for q in body.quotes]
        try:
            return analyze(body.outcomes, quotes, body.model_probs, body.bankroll)
        except ValueError as e:
            raise HTTPException(422, str(e))

    @app.post("/matches/{match_id}/explain", dependencies=[Depends(require_admin)])
    def explain_endpoint(match_id: str, body: ExplainIn) -> dict:
        res = run_analysis(match_id, body.model_probs, body.bankroll, body.use_model)
        return {"explanation": explain(res["match"], res, client=llm_client), "analysis": res}

    @app.post("/matches/{match_id}/flags", dependencies=[Depends(require_admin)])
    def flags_endpoint(match_id: str, body: FlagsIn) -> dict:
        info, _, _ = load_event(match_id)
        return extract_flags(body.notes, info["outcomes"], client=llm_client)

    @app.post("/admin/refresh", dependencies=[Depends(require_admin)])
    def refresh() -> dict:
        return provider().refresh()

    @app.get("/budget")
    def budget() -> dict:
        return {"data_mode": "live" if live else "demo", "remaining": store.last_known_remaining(),
                "reserve": s.credit_reserve, "recent": store.credit_log(10)}

    @app.get("/bets", dependencies=[Depends(require_admin)])
    def list_bets() -> dict:
        bets = store.bets()
        settled = [b for b in bets if b["result"] in ("won", "lost")]
        staked = sum(b["stake"] for b in settled)
        profit = sum(b["stake"] * (b["odds"] - 1) if b["result"] == "won" else -b["stake"] for b in settled)
        clvs = [b["odds"] * b["closing_fair_prob"] - 1 for b in bets if b["closing_fair_prob"]]
        return {"bets": bets, "summary": {
            "settled": len(settled), "open": sum(1 for b in bets if b["result"] is None),
            "staked": round(staked, 2), "profit": round(profit, 2),
            "roi": round(profit / staked, 4) if staked else None,
            "avg_clv": round(sum(clvs) / len(clvs), 4) if clvs else None, "clv_count": len(clvs),
            "note": "Judge the method by closing-line value over many bets, not by short-run profit."}}

    @app.post("/bets", dependencies=[Depends(require_admin)])
    def add_bet(body: BetIn) -> dict:
        return {"id": store.add_bet(body.event_id, body.outcome, body.book, body.odds, body.stake, body.note)}

    @app.post("/bets/{bet_id}/settle", dependencies=[Depends(require_admin)])
    def settle(bet_id: int, body: SettleIn) -> dict:
        bet = next((b for b in store.bets() if b["id"] == bet_id), None)
        if bet is None:
            raise HTTPException(404, "bet not found")
        closing = None
        ev = store.event(bet["event_id"]) if live else None
        if ev:
            rows = store.latest_quotes(bet["event_id"], before_iso=ev["commence_time"])
            if rows and bet["outcome"] in rows[0]["outcomes"]:
                outcomes = rows[0]["outcomes"]
                comp = compare(outcomes, [BookQuote(r["book_title"], tuple(r["odds"])) for r in rows
                                          if r["outcomes"] == outcomes])
                probs = comp.sharp_probs or comp.consensus_probs
                closing = probs[outcomes.index(bet["outcome"])]
        store.settle_bet(bet_id, body.result, closing)
        return {"id": bet_id, "result": body.result, "closing_fair_prob": closing,
                "clv": None if closing is None else round(bet["odds"] * closing - 1, 4)}

    from . import markets_api

    markets_api.register(app, store, s.models_dir, require_admin, s.cricsheet_path)
    app.state.store = store
    app.state.settings = s
    return app


app = create_app()

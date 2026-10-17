"""HTTP routes for the ball-by-ball engine and the learning loop."""
from __future__ import annotations

import threading
import traceback
from dataclasses import asdict
from datetime import date, datetime, timezone

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field, field_validator

from . import feedback
from . import modelsync
from .analyzer import analyze
from .lifecycle import BallRegistry, volatility_for
from .markets import BookQuote
from .props import MARKET_TYPES, MarketSpec, price, standard_card
from .simulate import LiveState, Lineup, MatchSetup, default_setup, simulate_match
from .storage import Store

MAX_SIMS = 5000


class LineupIn(BaseModel):
    batting: list[str] = Field(min_length=2, max_length=11)
    bowlers: list[str] = Field(min_length=1, max_length=11)


class MatchIn(BaseModel):
    team_a: str = Field(min_length=1, max_length=60)
    team_b: str = Field(min_length=1, max_length=60)
    venue: str = Field(min_length=1, max_length=120)
    pool: str = "T20:male"
    overs_limit: int = Field(20, ge=5, le=50)
    batting_first: str | None = None
    toss_winner: str | None = None
    lineups: dict[str, LineupIn] | None = None
    sims: int = Field(2000, ge=100, le=MAX_SIMS)
    seed: int = 0

    @field_validator("batting_first", "toss_winner")
    @classmethod
    def blank_is_none(cls, v):
        return v or None


class LiveIn(BaseModel):
    batting: str
    innings: int = Field(ge=1, le=2)
    score: int = Field(ge=0, le=600)
    wickets: int = Field(ge=0, le=9)
    overs: float = Field(ge=0, le=50, description="cricket notation: 7.3 = 7 overs and 3 balls")
    striker: str | None = None
    non_striker: str | None = None
    out: list[str] = []
    target: int | None = Field(None, ge=1)

    @field_validator("overs")
    @classmethod
    def valid_overs(cls, v):
        balls = round((v % 1) * 10)
        if balls > 5:
            raise ValueError("overs use cricket notation; the part after the dot is balls (0-5)")
        return v


class CardIn(MatchIn):
    live: LiveIn | None = None
    log_as: str | None = Field(None, max_length=80, description="event key to log predictions under (admin)")


class PriceIn(MatchIn):
    market: str
    team: str | None = None
    overs: int | None = None
    over: int | None = None
    player: str | None = None
    line: float | None = None
    odds_over: float | None = Field(None, gt=1)
    odds_under: float | None = Field(None, gt=1)
    book: str = "Bookmaker"
    bankroll: float | None = Field(None, gt=0)
    live: LiveIn | None = None


def _legal_balls(overs: float) -> int:
    whole = int(overs)
    return whole * 6 + round((overs - whole) * 10)


def register(app: FastAPI, store: Store, models_dir: str, require_admin, cricsheet_path: str | None) -> None:
    feedback.init(store)
    holder: dict = {"art": BallRegistry(models_dir).load(), "cal": feedback.load_calibrators(models_dir)}
    job: dict = {"state": "idle"}
    lock = threading.Lock()

    def art():
        if holder["art"] is None:
            raise HTTPException(409, "no ball-by-ball model yet: run a learning cycle on Cricsheet data first")
        return holder["art"]

    def setup_from(body: MatchIn):
        a = art()
        if body.team_a == body.team_b:
            raise HTTPException(422, "pick two different teams")
        for t in (body.batting_first, body.toss_winner):
            if t and t not in (body.team_a, body.team_b):
                raise HTTPException(422, f"{t} is not one of the two teams")
        try:
            if body.lineups:
                lus = {t: Lineup(l.batting, l.bowlers) for t, l in body.lineups.items()}
                missing = {body.team_a, body.team_b} - set(lus)
                if missing:
                    base = default_setup(a.state, body.pool, body.venue, body.team_a, body.team_b, body.overs_limit)
                    for t in missing:
                        lus[t] = base.lineups[t]
                return MatchSetup(body.pool, body.venue, (body.team_a, body.team_b), lus, body.overs_limit,
                                  body.batting_first, body.toss_winner)
            return default_setup(a.state, body.pool, body.venue, body.team_a, body.team_b, body.overs_limit,
                                 batting_first=body.batting_first, toss_winner=body.toss_winner)
        except ValueError as e:
            raise HTTPException(422, str(e))

    def live_from(lv: LiveIn | None, body: MatchIn) -> LiveState | None:
        if lv is None:
            return None
        if lv.batting not in (body.team_a, body.team_b):
            raise HTTPException(422, "the batting side must be one of the two teams")
        if lv.innings == 2 and not lv.target:
            raise HTTPException(422, "a second-innings situation needs the target")
        if _legal_balls(lv.overs) > body.overs_limit * 6:
            raise HTTPException(422, "more overs than the match has")
        return LiveState(lv.batting, lv.innings, lv.score, lv.wickets, _legal_balls(lv.overs), lv.striker,
                         lv.non_striker, list(lv.out), target=lv.target)

    def run(body: MatchIn, live: LiveState | None):
        setup = setup_from(body)
        try:
            return setup, simulate_match(art().model, art().state, setup, n=body.sims, seed=body.seed, live=live,
                                         volatility=volatility_for(art(), setup.competition))
        except ValueError as e:
            raise HTTPException(422, str(e))

    @app.get("/engine")
    def engine() -> dict:
        reg = BallRegistry(models_dir)
        a = holder["art"]
        return {
            "model": None if a is None else {"version": a.version, "trained_through": a.trained_through.isoformat(),
                                             "pools": list(a.pools), "kind": a.model.kind,
                                             "evaluation": {k: v for k, v in a.evaluation.items() if k != "folds"},
                                             "folds": a.evaluation.get("folds", [])},
            "runs": reg.runs(12),
            "market_report": feedback.report(store),
            "drift": feedback.drift(store),
            "calibrated_markets": sorted(holder["cal"]),
            "cycle": {k: v for k, v in job.items() if k != "result"},
        }

    @app.get("/catalog")
    def catalog(pool: str = "T20:male") -> dict:
        a = art()
        teams = sorted(t for (p, t) in a.state.last_lineup if p == pool)
        venues = sorted({v for (p, v) in a.state.venue if p == pool})
        lineups = {t: {"batting": a.state.last_lineup[(pool, t)][0], "bowlers": a.state.last_lineup[(pool, t)][1]}
                   for t in teams}
        return {"pool": pool, "teams": teams, "venues": venues, "lineups": lineups, "market_types": list(MARKET_TYPES)}

    @app.post("/markets/card")
    def card(body: CardIn, authorization: str | None = Header(default=None)) -> dict:
        live = live_from(body.live, body)
        setup, sim = run(body, live)
        c = standard_card(sim, body.overs_limit, holder["cal"])
        logged = 0
        if body.log_as:
            require_admin(authorization)
            logged = feedback.log_card(store, body.log_as, setup.teams, date.today(), c, art().version)
        return {"model_version": art().version, "sims": sim.n, "live": body.live is not None,
                "lineups": {t: asdict(l) for t, l in setup.lineups.items()}, "card": c, "logged": logged}

    @app.post("/markets/price")
    def price_line(body: PriceIn) -> dict:
        try:
            spec = MarketSpec(body.market, body.team, body.overs, body.over, body.player)
        except ValueError as e:
            raise HTTPException(422, str(e))
        _, sim = run(body, live_from(body.live, body))
        try:
            lp = price(sim, spec, body.line, holder["cal"].get(spec.type))
        except ValueError as e:
            raise HTTPException(422, str(e))
        out = {"price": asdict(lp)}
        if body.odds_over and body.odds_under and lp.p_push == 0:
            names = ["Yes", "No"] if spec.type == "match_winner" else [f"Over {lp.line}", f"Under {lp.line}"]
            out["verdict"] = analyze(names, [BookQuote(body.book, (body.odds_over, body.odds_under))],
                                     [lp.p_over, lp.p_under], body.bankroll)
        elif body.odds_over or body.odds_under:
            out["note"] = ("enter both prices for a verdict" if lp.p_push == 0
                           else "this line can push (whole number); use a half-point line for a verdict")
        return out

    def _cycle_worker(path: str):
        from .pipeline import cycle
        try:
            res = cycle(path, store, models_dir)
            if store.is_remote:
                res["models_pushed"] = modelsync.push(store, models_dir)
            holder["art"] = BallRegistry(models_dir).load()
            holder["cal"] = feedback.load_calibrators(models_dir)
            job.update(state="done", finished=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       summary={"promoted": res["ball_model"]["promoted"], "reason": res["ball_model"]["reason"],
                                "settled": res["settle"]["settled"]}, result=res)
        except Exception as e:  # report, never crash the API
            job.update(state="failed", error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-800:])

    @app.post("/admin/cycle", dependencies=[Depends(require_admin)])
    def start_cycle() -> dict:
        if not cricsheet_path:
            raise HTTPException(409, "set DOOSRA_CRICSHEET to the Cricsheet zip the cycle should learn from")
        with lock:
            if job.get("state") == "running":
                raise HTTPException(409, "a learning cycle is already running")
            job.clear()
            job.update(state="running", started=datetime.now(timezone.utc).isoformat(timespec="seconds"))
            threading.Thread(target=_cycle_worker, args=(cricsheet_path,), daemon=True).start()
        return {"state": "running"}

    @app.get("/admin/cycle", dependencies=[Depends(require_admin)])
    def cycle_status() -> dict:
        return job

    app.state.engine = holder

"""The Odds API (v4) adapter with a hard credit budget.

Cost model (per their docs): one /odds call costs markets x regions credits.
/sports is free. Usage is reported in response headers:
x-requests-remaining, x-requests-used, x-requests-last.

The budget guard refuses a call that would push remaining credits below the
reserve, so a refresh job can never silently burn the monthly allowance.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import httpx

from ..storage import Store, now_iso

BASE_URL = "https://api.the-odds-api.com/v4"

# Sport keys seen live in the user's /sports response (Oct 2026). IPL appears in season.
CRICKET_SPORT_KEYS = ("cricket_international_t20", "cricket_odi", "cricket_test_match")

FORMAT_FOR_SPORT = {
    "cricket_international_t20": "T20",
    "cricket_ipl": "T20",
    "cricket_odi": "ODI",
    "cricket_test_match": "Test",
}


class BudgetExceeded(RuntimeError):
    pass


class ProviderError(RuntimeError):
    pass


def _int(v: str | None) -> int | None:
    try:
        return int(float(v)) if v is not None else None
    except ValueError:
        return None


@dataclass
class TheOddsApi:
    api_key: str
    store: Store
    reserve: int = 50  # never spend the last N credits
    monthly_allowance: int = 500
    regions: tuple[str, ...] = ("uk", "eu")
    timeout: float = 20.0
    transport: httpx.BaseTransport | None = None  # injectable for tests
    _client: httpx.Client | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if not self.api_key:
            raise ValueError("The Odds API key is missing (set DOOSRA_ODDS_API_KEY)")
        self._client = httpx.Client(base_url=BASE_URL, timeout=self.timeout, transport=self.transport)

    # -- budget ---------------------------------------------------------
    def remaining(self) -> int | None:
        return self.store.last_known_remaining()

    def cost_of(self, markets: int = 1) -> int:
        return markets * len(self.regions)

    def check_budget(self, cost: int) -> None:
        rem = self.remaining()
        if rem is not None and rem - cost < self.reserve:
            raise BudgetExceeded(
                f"call costs {cost} credits, {rem} remaining, reserve is {self.reserve}: skipped"
            )

    def _get(self, path: str, params: dict, cost: int) -> list | dict:
        if cost:
            self.check_budget(cost)
        try:
            r = self._client.get(path, params={**params, "apiKey": self.api_key})
        except httpx.HTTPError as e:
            raise ProviderError(f"network error calling The Odds API: {type(e).__name__}") from e
        h = r.headers
        self.store.record_credits(
            path, _int(h.get("x-requests-last")), _int(h.get("x-requests-used")), _int(h.get("x-requests-remaining"))
        )
        if r.status_code in (401, 403):
            raise ProviderError("The Odds API rejected the key (401/403). Check DOOSRA_ODDS_API_KEY.")
        if r.status_code == 429:
            raise ProviderError("The Odds API quota or rate limit hit (429).")
        if r.status_code >= 400:
            raise ProviderError(f"The Odds API returned {r.status_code}: {r.text[:200]}")
        return r.json()

    # -- endpoints --------------------------------------------------------
    def active_cricket_sports(self) -> list[dict]:
        sports = self._get("/sports", {}, cost=0)
        return [s for s in sports if s.get("group") == "Cricket" and s.get("active")]

    def fetch_odds(self, sport_key: str) -> list[dict]:
        events = self._get(
            f"/sports/{sport_key}/odds",
            {"regions": ",".join(self.regions), "markets": "h2h", "oddsFormat": "decimal", "dateFormat": "iso"},
            cost=self.cost_of(1),
        )
        if not isinstance(events, list):
            raise ProviderError("unexpected /odds response shape")
        return events

    def refresh(self, sport_keys: tuple[str, ...] = CRICKET_SPORT_KEYS) -> dict:
        """Fetch and store odds for each sport. Stops cleanly when the budget says so."""
        stored, skipped, errors = 0, [], []
        fetched_at = now_iso()
        for key in sport_keys:
            try:
                rows = parse_events(self.fetch_odds(key), fetched_at)
                stored += self.store.add_snapshot_rows(rows)
            except BudgetExceeded as e:
                skipped.append({"sport": key, "reason": str(e)})
                break
            except ProviderError as e:
                errors.append({"sport": key, "reason": str(e)})
        return {"stored_rows": stored, "skipped": skipped, "errors": errors, "remaining": self.remaining()}


def parse_events(events: list[dict], fetched_at: str) -> list[dict]:
    """Flatten v4 events into one row per (event, bookmaker) h2h quote.

    Outcome order is fixed to [home, away] plus "Draw" when a book offers it
    (Test matches). Quotes missing a side are dropped rather than guessed.
    """
    rows = []
    for ev in events:
        home, away = ev.get("home_team"), ev.get("away_team")
        if not home or not away:
            continue
        for bm in ev.get("bookmakers", []):
            market = next((m for m in bm.get("markets", []) if m.get("key") == "h2h"), None)
            if not market:
                continue
            prices = {o.get("name"): o.get("price") for o in market.get("outcomes", [])}
            order = [home, away] + (["Draw"] if "Draw" in prices else [])
            odds = [prices.get(n) for n in order]
            if any(not isinstance(o, (int, float)) or o <= 1.0 for o in odds):
                continue
            rows.append({
                "event_id": ev["id"],
                "sport_key": ev.get("sport_key", ""),
                "commence_time": ev.get("commence_time", ""),
                "home_team": home,
                "away_team": away,
                "book_key": bm.get("key", ""),
                "book_title": bm.get("title", bm.get("key", "")),
                "outcomes": order,
                "odds": [float(o) for o in odds],
                "book_last_update": market.get("last_update") or bm.get("last_update"),
                "fetched_at": fetched_at,
            })
    return rows

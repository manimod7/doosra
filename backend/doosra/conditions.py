"""Weather conditions for a venue and date, from Open-Meteo (free, no key).

Cricsheet gives a date but no start time, so features use daily summaries:
  dew_spread  smallest (temperature - dew point) in the 18:00-23:00 local
              window; small values mean dew is likely in an evening game
  humidity    mean relative humidity across the day (%)
  precip      total rain (mm)
Historical matches use the archive endpoint; upcoming ones the forecast.

Venue coordinates below are approximate (city-level, good enough for a
weather grid cell). Unknown venues simply get no weather features, and the
model treats them as missing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, timedelta

import httpx

from .storage import Store

ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
FORECAST = "https://api.open-meteo.com/v1/forecast"
HOURLY = "temperature_2m,dew_point_2m,relative_humidity_2m,precipitation"

# Approximate coordinates (lat, lon) for major venues; match by substring of the Cricsheet venue name.
VENUE_COORDS: dict[str, tuple[float, float]] = {
    "Wankhede": (18.94, 72.83), "Brabourne": (18.93, 72.82), "DY Patil": (19.04, 73.03),
    "Eden Gardens": (22.56, 88.34), "Chinnaswamy": (12.98, 77.60), "Chepauk": (13.06, 80.28),
    "Chidambaram": (13.06, 80.28), "Arun Jaitley": (28.64, 77.24), "Feroz Shah Kotla": (28.64, 77.24),
    "Narendra Modi": (23.09, 72.60), "Motera": (23.09, 72.60), "Sawai Mansingh": (26.89, 75.80),
    "Rajiv Gandhi International": (17.41, 78.55), "Punjab Cricket Association": (30.69, 76.74),
    "Ekana": (26.81, 81.00), "Dharamsala": (32.20, 76.33), "Barsapara": (26.14, 91.73),
    "Melbourne Cricket Ground": (-37.82, 144.98), "Sydney Cricket Ground": (-33.89, 151.22),
    "Adelaide Oval": (-34.92, 138.60), "Gabba": (-27.49, 153.04), "Perth Stadium": (-31.95, 115.89),
    "Lord's": (51.53, -0.17), "Kennington Oval": (51.48, -0.11), "Edgbaston": (52.46, -1.90),
    "Old Trafford": (53.46, -2.29), "Headingley": (53.82, -1.58), "Trent Bridge": (52.94, -1.13),
    "Rose Bowl": (50.92, -1.32), "Sophia Gardens": (51.49, -3.19),
    "Wanderers": (-26.13, 28.06), "Newlands": (-33.97, 18.47), "Kingsmead": (-29.85, 31.03),
    "SuperSport Park": (-25.86, 28.18), "Dubai International": (25.05, 55.22), "Sharjah": (25.33, 55.42),
    "Sheikh Zayed": (24.40, 54.47), "Gaddafi": (31.51, 74.33), "National Stadium, Karachi": (24.89, 67.08),
    "Rawalpindi": (33.65, 73.08), "Shere Bangla": (23.81, 90.36), "R Premadasa": (6.94, 79.87),
    "Pallekele": (7.28, 80.72), "Kensington Oval": (13.10, -59.62), "Queen's Park Oval": (10.67, -61.52),
    "Eden Park": (-36.87, 174.74), "Hagley Oval": (-43.53, 172.62), "Basin Reserve": (-41.30, 174.78),
    "Harare Sports Club": (-17.81, 31.05),
}


def coords_for(venue: str) -> tuple[float, float] | None:
    v = venue.lower()
    for key, c in VENUE_COORDS.items():
        if key.lower() in v:
            return c
    return None


def summarise(hourly: dict) -> dict:
    times = hourly.get("time", [])
    t = hourly.get("temperature_2m", [])
    dp = hourly.get("dew_point_2m", [])
    rh = hourly.get("relative_humidity_2m", [])
    pr = hourly.get("precipitation", [])
    evening = [i for i, ts in enumerate(times) if 18 <= int(ts[11:13]) <= 23]
    spreads = [t[i] - dp[i] for i in evening if t[i] is not None and dp[i] is not None]
    hums = [x for x in rh if x is not None]
    rain = [x for x in pr if x is not None]
    return {
        "dew_spread": min(spreads) if spreads else None,
        "humidity": sum(hums) / len(hums) if hums else None,
        "precip": sum(rain) if rain else None,
    }


@dataclass
class Weather:
    store: Store
    transport: httpx.BaseTransport | None = None
    _client: httpx.Client | None = field(default=None, init=False)

    def __post_init__(self):
        self.store._conn.execute(
            "CREATE TABLE IF NOT EXISTS weather (venue TEXT, day TEXT, data TEXT, PRIMARY KEY (venue, day))")
        self._client = httpx.Client(timeout=20, transport=self.transport)

    def for_match(self, venue: str, day: date) -> dict | None:
        row = self.store._conn.execute("SELECT data FROM weather WHERE venue = ? AND day = ?",
                                       (venue, day.isoformat())).fetchone()
        if row:
            return json.loads(row["data"]) or None
        c = coords_for(venue)
        if c is None:
            return None
        url = ARCHIVE if day < date.today() - timedelta(days=5) else FORECAST
        try:
            r = self._client.get(url, params={"latitude": c[0], "longitude": c[1], "start_date": day.isoformat(),
                                              "end_date": day.isoformat(), "hourly": HOURLY, "timezone": "auto"})
            r.raise_for_status()
            data = summarise(r.json().get("hourly", {}))
        except (httpx.HTTPError, ValueError):
            return None            # weather is optional; never block a prediction on it
        with self.store.tx() as conn:
            conn.execute("INSERT OR REPLACE INTO weather VALUES (?,?,?)", (venue, day.isoformat(), json.dumps(data)))
        return {k: v for k, v in data.items() if v is not None} or None

    def for_matches(self, matches) -> dict:
        """{match_id: features} for training, using the cache wherever possible."""
        out = {}
        for m in matches:
            w = self.for_match(m.venue, m.date)
            if w:
                out[m.match_id] = w
        return out

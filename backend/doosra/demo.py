"""DEMO DATA ONLY. These odds are invented so the UI has something to render.

They are NOT real bookmaker prices and must never be used to place a bet.
Real prices arrive through a provider adapter once an odds API key is set.
"""
from __future__ import annotations

DEMO_MATCHES = [
    {
        "id": "demo-1",
        "league": "DEMO T20 League",
        "start": "2026-10-18T14:00:00Z",
        "outcomes": ["Team Alpha", "Team Bravo"],
        "quotes": {
            "pinnacle": [1.78, 2.14],
            "bet365": [1.72, 2.10],
            "betway": [1.70, 2.05],
            "1xbet": [1.80, 2.10],
            "stake": [1.76, 2.12],
            "rainbet": [1.77, 2.11],
        },
    },
    {
        "id": "demo-2",
        "league": "DEMO ODI Series",
        "start": "2026-10-19T09:30:00Z",
        "outcomes": ["Team Charlie", "Team Delta"],
        "quotes": {
            "pinnacle": [1.45, 2.85],
            "bet365": [1.40, 2.80],
            "betway": [1.42, 2.75],
            "1xbet": [1.47, 2.78],
            "stake": [1.44, 2.82],
        },
    },
]

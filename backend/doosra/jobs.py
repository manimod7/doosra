"""Scheduled jobs (run from cron / Railway cron).

  python -m doosra.jobs refresh     # fetch odds within the credit budget
  python -m doosra.jobs budget      # print remaining credits

Suggested schedule on the free 500-credit plan, 3 cricket feeds, uk+eu
regions (2 credits per feed): twice a day costs about 12 credits/day,
roughly 360/month, leaving headroom for a pre-match refresh.
"""
from __future__ import annotations

import json
import sys

from .api import Settings
from .providers.the_odds_api import TheOddsApi
from .storage import Store


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "refresh"
    s = Settings.from_env()
    store = Store(s.db_path)
    if cmd == "budget":
        print(json.dumps({"remaining": store.last_known_remaining(), "reserve": s.credit_reserve}))
        return 0
    if cmd == "refresh":
        if not s.odds_api_key:
            print("DOOSRA_ODDS_API_KEY is not set", file=sys.stderr)
            return 2
        res = TheOddsApi(api_key=s.odds_api_key, store=store, reserve=s.credit_reserve).refresh()
        print(json.dumps(res))
        return 1 if res["errors"] else 0
    print(f"unknown command {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

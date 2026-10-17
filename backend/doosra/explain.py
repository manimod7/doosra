"""Plain-language explanations and news flags from Claude, under strict guards.

Rules this module enforces
1. The LLM never produces probabilities, prices or stakes. It only rewords
   numbers that the analyzer already computed.
2. Every number in the LLM's text must match a number in the facts; if one
   doesn't, the text is discarded and a deterministic template is used.
3. News "flags" come from a fixed vocabulary, are shown to you as context,
   and never change any probability (they can't be backtested yet).
4. No key, network error or bad output -> template. The app never breaks
   because the LLM is unavailable.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Protocol


FLAG_VOCAB = {
    "key_batter_out": "A key batter is missing",
    "key_bowler_out": "A key bowler is missing",
    "captain_change": "The captain has changed",
    "xi_confirmed": "The playing XI is confirmed",
    "rain_risk": "Rain is forecast, so a shortened match is possible",
    "dew_expected": "Dew is expected, which usually helps the chasing side",
    "pitch_spin": "The pitch is expected to help spin",
    "pitch_pace": "The pitch is expected to help pace",
    "pitch_flat": "A flat, high-scoring pitch is expected",
    "dead_rubber": "The match has little at stake for one side",
}

NUM_RE = re.compile(r"(?<![\w.])[-−+]?\d+(?:\.\d+)?")


class LLMClient(Protocol):
    def complete(self, system: str, prompt: str, max_tokens: int) -> str: ...


def default_client() -> LLMClient | None:
    """Gemini, then Groq, then Claude, whichever keys are set (see llm.py)."""
    from .llm import from_env

    return from_env()


# ---------------------------------------------------------------- facts ---

def facts_from_analysis(match: dict, analysis: dict) -> dict:
    rows = []
    for a in analysis["analysis"]:
        rows.append({
            "side": a["outcome"],
            "verdict": a["label"],
            "best_odds": round(a["best_odds"], 2),
            "best_book": a["best_book"],
            "fair_chance_pct": round(a["consensus_prob"] * 100, 1),
            "sharp_chance_pct": None if a["sharp_prob"] is None else round(a["sharp_prob"] * 100, 1),
            "your_chance_pct": None if a["model_prob"] is None else round(a["model_prob"] * 100, 1),
            "edge_pct": None if a["edge_vs_best"] is None else round(a["edge_vs_best"] * 100, 1),
            "stake_pct": round(a["stake_fraction"] * 100, 2),
            "stake_amount": a.get("stake_amount"),
        })
    return {
        "match": " v ".join(analysis["outcomes"]),
        "league": match.get("league"),
        "books_compared": len(analysis["books"]),
        "best_price_margin_pct": round(analysis["best_price_margin"] * 100, 1),
        "sharp_book": analysis.get("sharp_book"),
        "sides": rows,
    }


def _allowed_numbers(facts: dict) -> list[float]:
    vals: list[float] = []

    def walk(v: Any):
        if isinstance(v, bool):
            return
        if isinstance(v, (int, float)):
            vals.extend([float(v), round(float(v)), round(float(v), 1), round(float(v), 2)])
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)

    walk(facts)
    vals.extend(float(i) for i in range(0, 12))  # small counts like "two books", "1 side"
    return vals


def numbers_are_grounded(text: str, facts: dict) -> tuple[bool, list[str]]:
    allowed = _allowed_numbers(facts)
    bad = []
    for tok in NUM_RE.findall(text.replace(",", "")):
        v = abs(float(tok.replace("−", "-")))
        if not any(abs(v - abs(a)) < 0.0051 for a in allowed):
            bad.append(tok)
    return (not bad), bad


def template_explanation(facts: dict) -> str:
    parts = [f"{facts['match']}: prices from {facts['books_compared']} bookmakers compared."]
    for s in facts["sides"]:
        line = (f"{s['side']} is best priced at {s['best_odds']:.2f} with {s['best_book']}, "
                f"a fair chance of about {s['fair_chance_pct']:.1f}% once margins are removed.")
        if s["your_chance_pct"] is None:
            line += " No verdict, because no win chance was given."
        elif s["verdict"] == "BET":
            line += (f" At your {s['your_chance_pct']:.1f}% that is an edge of {s['edge_pct']:+.1f}%, "
                     f"so the suggested stake is {s['stake_pct']:.2f}% of bankroll.")
        elif s["verdict"] == "TRAP":
            line += (f" Your {s['your_chance_pct']:.1f}% is far above the sharp market's "
                     f"{s['sharp_chance_pct']:.1f}%, so treat the edge as suspect.")
        else:
            line += f" At your {s['your_chance_pct']:.1f}% there is no worthwhile edge."
        parts.append(line)
    return " ".join(parts)


SYSTEM = (
    "You explain cricket betting analysis to one person in plain English. "
    "Use ONLY the facts in the JSON you are given. Do not add statistics, form, injuries, "
    "players or predictions that are not in the JSON. Never invent or recompute numbers; "
    "copy them exactly as given. Do not encourage betting; if a verdict is SKIP or TRAP say so plainly. "
    "Write 3 to 5 short sentences, no lists, no headings."
)


def explain(match: dict, analysis: dict, client: LLMClient | None = None) -> dict:
    facts = facts_from_analysis(match, analysis)
    client = client if client is not None else default_client()
    if client is None:
        return {"text": template_explanation(facts), "source": "template", "reason": "no AI provider key set"}
    try:
        text = client.complete(SYSTEM, json.dumps(facts, ensure_ascii=False), max_tokens=300)
    except Exception as e:  # network, auth, model name: fall back, never break the page
        return {"text": template_explanation(facts), "source": "template", "reason": f"LLM call failed: {type(e).__name__}"}
    ok, bad = numbers_are_grounded(text, facts)
    if not text or not ok:
        return {"text": template_explanation(facts), "source": "template",
                "reason": f"LLM output used numbers not in the analysis: {', '.join(bad[:5])}" if bad else "empty LLM output"}
    return {"text": text, "source": "llm", "reason": "ok", "provider": getattr(client, "last_used", None) or getattr(client, "name", None)}


# ---------------------------------------------------------------- flags ---

FLAG_SYSTEM = (
    "You read cricket team news and pitch or weather notes and return JSON only: "
    '{"flags": [{"code": <one of the allowed codes>, "team": <team name or null>, "evidence": <short quote from the notes>}]}. '
    "Only use a code when the notes clearly state it. Never guess. Return {\"flags\": []} if nothing applies. "
    "Allowed codes: " + ", ".join(FLAG_VOCAB)
)


def extract_flags(notes: str, teams: list[str], client: LLMClient | None = None) -> dict:
    notes = (notes or "").strip()[:4000]
    if not notes:
        return {"flags": [], "source": "none", "reason": "no notes given"}
    client = client if client is not None else default_client()
    if client is None:
        return {"flags": [], "source": "none", "reason": "no AI provider key set"}
    try:
        raw = client.complete(FLAG_SYSTEM, json.dumps({"teams": teams, "notes": notes}), max_tokens=400)
        data = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
    except Exception as e:
        return {"flags": [], "source": "none", "reason": f"could not read flags: {type(e).__name__}"}
    flags = []
    for f in data.get("flags", []):
        code = f.get("code")
        evidence = str(f.get("evidence") or "")[:200]
        team = f.get("team") if f.get("team") in teams else None
        # keep a flag only if it is in the vocabulary and its evidence really appears in the notes
        if code in FLAG_VOCAB and evidence and evidence.lower() in notes.lower():
            flags.append({"code": code, "label": FLAG_VOCAB[code], "team": team, "evidence": evidence})
    return {"flags": flags, "source": "llm", "reason": "ok",
            "note": "Flags are context only. They do not change any probability."}

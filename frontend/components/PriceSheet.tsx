"use client";

import { useEffect, useRef, useState } from "react";
import { bridge, inr, odds, pct, signedPct, type Analysis, type LinePrice, type MatchQuery } from "@/lib/api";

export type SheetTarget = {
  title: string;            // e.g. "Team A · Runs after 6 overs"
  spec: { type: string; team?: string | null; overs?: number | null; over?: number | null; player?: string | null };
  line: number | null;      // fair line to start from (null for match winner)
  pOver?: number;          // chance of over at the fair line, when known
  q10?: number;
  q90?: number;
};

type Result = { price: LinePrice; verdict?: Analysis; note?: string };

const CALL = { BET: "Worth backing", SKIP: "No edge here", TRAP: "Too good to trust", "NO MODEL": "No verdict" };

export default function PriceSheet({ target, query, onClose }: { target: SheetTarget; query: MatchQuery; onClose: () => void }) {
  const isWinner = target.spec.type === "match_winner";
  const [line, setLine] = useState<number | null>(target.line);
  const [oOver, setOOver] = useState("");
  const [oUnder, setOUnder] = useState("");
  const [bank, setBank] = useState("");
  const [state, setState] = useState<{ k: "idle" } | { k: "busy" } | { k: "err"; m: string } | { k: "ok"; r: Result }>({ k: "idle" });
  const [logged, setLogged] = useState<string | null>(null);
  const first = useRef<HTMLInputElement>(null);

  useEffect(() => {
    first.current?.focus();
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    document.addEventListener("keydown", onKey);
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => { document.removeEventListener("keydown", onKey); document.body.style.overflow = prev; };
  }, [onClose]);

  const sideNames = isWinner ? ["Yes", "No"] : ["Over", "Under"];
  const valid = Number(oOver) > 1 && Number(oUnder) > 1;

  async function check(e: React.FormEvent) {
    e.preventDefault();
    if (!valid) return;
    setState({ k: "busy" });
    setLogged(null);
    try {
      const r = await bridge<Result>("markets/price", {
        method: "POST",
        body: { ...query, sims: 2000, market: target.spec.type, team: target.spec.team ?? null,
          overs: target.spec.overs ?? null, over: target.spec.over ?? null, player: target.spec.player ?? null,
          line, odds_over: Number(oOver), odds_under: Number(oUnder), book: "My bookmaker",
          bankroll: Number(bank) > 0 ? Number(bank) : null },
      });
      setState({ k: "ok", r });
    } catch (err) {
      setState({ k: "err", m: (err as Error).message });
    }
  }

  async function log(side: 0 | 1, stake: number) {
    if (state.k !== "ok" || !state.r.verdict) return;
    const a = state.r.verdict.analysis[side];
    try {
      await bridge("bets", { method: "POST", body: {
        event_id: `${query.team_a} v ${query.team_b}`.slice(0, 80),
        outcome: `${target.title} · ${a.outcome}`.slice(0, 80), book: "My bookmaker", odds: a.best_odds, stake,
      } });
      setLogged(`Logged ${inr(stake)} on ${a.outcome}.`);
    } catch (err) {
      setLogged((err as Error).message);
    }
  }

  const step = (d: number) => {
    if (line === null) return;
    setLine(Math.max(0.5, line + d));
    setState({ k: "idle" });
  };

  const r = state.k === "ok" ? state.r : null;
  const best = r?.verdict?.analysis.find((a) => a.label === "BET") ?? r?.verdict?.analysis.find((a) => a.label === "TRAP");

  return (
    <div className="scrim" onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className="sheet" role="dialog" aria-modal="true" aria-labelledby="sheet-title">
        <div className="sheet-grab" aria-hidden="true" />
        <div className="sheet-head">
          <div style={{ display: "grid", gap: 4 }}>
            <span className="t-micro">Check a price</span>
            <h2 id="sheet-title" className="t-head">{target.title}</h2>
          </div>
          <button className="icon-btn" type="button" onClick={onClose} aria-label="Close">
            <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><path d="M1 1l10 10M11 1 1 11" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" /></svg>
          </button>
        </div>

        <div className="inset" style={{ padding: 16, display: "grid", gap: 10 }}>
          {isWinner ? (
            <p className="t-small">Doosra gives this a <b style={{ color: "var(--ink)" }}>{pct(target.pOver ?? 0)}</b> chance, so a fair price is <b style={{ color: "var(--ink)" }}>{odds(1 / (target.pOver ?? 1))}</b>.</p>
          ) : (
            <>
              <div className="stepper" aria-label="Line">
                <button type="button" onClick={() => step(-1)} aria-label="Lower the line by one">−</button>
                <span className="num-l" style={{ minWidth: "4ch", textAlign: "center" }}>{line}</span>
                <button type="button" onClick={() => step(1)} aria-label="Raise the line by one">+</button>
                <span className="t-small" style={{ marginLeft: 4 }}>Set this to the bookmaker&apos;s line.</span>
              </div>
              <p className="t-small">
                {target.pOver !== undefined ? `At the fair line ${target.line}, over is ${pct(target.pOver)}.` : `The fair line is ${target.line}: half the simulations finish above it.`}
                {target.q10 !== undefined && ` 8 in 10 simulations land between ${Math.round(target.q10)} and ${Math.round(target.q90!)}.`}
              </p>
            </>
          )}
        </div>

        <form onSubmit={check} style={{ display: "grid", gap: 14 }}>
          <div className="two">
            <label className="field"><span>{sideNames[0]} price (decimal)</span>
              <input ref={first} className="input" type="number" inputMode="decimal" step="0.01" min="1.01" value={oOver}
                onChange={(e) => { setOOver(e.target.value); setState({ k: "idle" }); }} placeholder="e.g. 1.90" />
            </label>
            <label className="field"><span>{sideNames[1]} price (decimal)</span>
              <input className="input" type="number" inputMode="decimal" step="0.01" min="1.01" value={oUnder}
                onChange={(e) => { setOUnder(e.target.value); setState({ k: "idle" }); }} placeholder="e.g. 1.90" />
            </label>
          </div>
          <label className="field"><span>Bankroll in ₹ (optional, for a stake size)</span>
            <input className="input" type="number" inputMode="decimal" min="1" value={bank} onChange={(e) => setBank(e.target.value)} placeholder="10000" />
          </label>
          <button className="btn btn-primary" type="submit" disabled={!valid || state.k === "busy"}>
            {state.k === "busy" ? "Simulating 2,000 matches" : "Check this price"}
          </button>
        </form>

        <div aria-live="polite" style={{ display: "grid", gap: 12 }}>
          {state.k === "err" && <p className="note">{state.m}</p>}
          {r?.note && <p className="note">{r.note}</p>}
          {r?.verdict && (
            <>
              <div className={`verdict ${best?.label === "BET" ? "good" : best?.label === "TRAP" ? "critical" : ""}`}>
                <div className="verdict-top">
                  <span className="verdict-call">{best ? `${CALL[best.label]}: ${best.outcome}` : CALL.SKIP}</span>
                  {best && <span className={`pill ${best.label === "BET" ? "pill-good" : "pill-critical"}`}>{best.label}</span>}
                </div>
                <p className="t-small">{(best ?? r.verdict.analysis[0]).reasons[0]}</p>
                {best?.label === "BET" && best.stake_fraction > 0 && (
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 10, alignItems: "center", justifyContent: "space-between" }}>
                    <span className="t-small">Suggested stake: <b style={{ color: "var(--ink)" }}>{(best.stake_fraction * 100).toFixed(2)}% of bankroll{best.stake_amount ? `, ${inr(best.stake_amount)}` : ""}</b></span>
                    <button type="button" className="btn btn-quiet" onClick={() => log(r.verdict!.analysis.indexOf(best) as 0 | 1, best.stake_amount || 100)}>Log this bet</button>
                  </div>
                )}
                {logged && <p className="t-small">{logged}</p>}
              </div>
              <ul className="list">
                {r.verdict.analysis.map((a) => (
                  <li key={a.outcome} className="row">
                    <div className="row-main">
                      <span className="row-title">{a.outcome}</span>
                      <span className="t-micro">Doosra {pct(a.model_prob ?? 0, 1)} · price {odds(a.best_odds)}</span>
                    </div>
                    <span className="num" style={{ fontWeight: 700 }}>{a.edge_vs_best === null ? "–" : signedPct(a.edge_vs_best)}</span>
                  </li>
                ))}
              </ul>
              {r.price.calibrated && <p className="t-micro">This market&apos;s chance was corrected using Doosra&apos;s past results on it.</p>}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

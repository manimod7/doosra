"use client";

import { useState } from "react";
import { bridge, inr, odds, pct, signedPct, type MatchOdds } from "@/lib/api";

const CALL = { BET: "Worth backing", SKIP: "No edge here", TRAP: "Too good to trust", "NO MODEL": "No verdict" };

/** Your chance vs the best bookmaker price, for an odds-feed match. */
export default function PriceCheck({ matchId, outcomes }: { matchId: string; outcomes: string[] }) {
  const two = outcomes.length === 2;
  const [vals, setVals] = useState<string[]>(outcomes.map(() => ""));
  const [bank, setBank] = useState("");
  const [state, setState] = useState<{ k: "idle" } | { k: "busy" } | { k: "err"; m: string } | { k: "ok"; d: MatchOdds }>({ k: "idle" });

  const nums = two ? (vals[0] === "" ? [NaN, NaN] : [Number(vals[0]), 100 - Number(vals[0])]) : vals.map((v) => (v === "" ? NaN : Number(v)));
  const total = nums.reduce((s, v) => s + v, 0);
  const valid = nums.every((v) => v > 0 && v < 100) && Math.abs(total - 100) < 0.01;

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!valid) return;
    setState({ k: "busy" });
    const q = new URLSearchParams({ model_probs: nums.map((v) => (v / 100).toFixed(4)).join(",") });
    if (Number(bank) > 0) q.set("bankroll", bank);
    try {
      setState({ k: "ok", d: await bridge<MatchOdds>(`matches/${encodeURIComponent(matchId)}/odds?${q}`) });
    } catch (err) {
      setState({ k: "err", m: (err as Error).message });
    }
  }

  return (
    <section className="card card-pad" style={{ display: "grid", gap: 16 }} aria-labelledby="pc">
      <div style={{ display: "grid", gap: 4 }}>
        <h2 id="pc" className="t-head">Check a bet</h2>
        <p className="t-small">Say how likely you think each side is. Doosra compares it with the best price on the board.</p>
      </div>
      <form onSubmit={submit} style={{ display: "grid", gap: 12 }}>
        {outcomes.map((o, i) => (
          <label key={o} className="field"><span>{o} wins (%)</span>
            <input className="input" type="number" inputMode="decimal" min="1" max="99" step="0.1"
              readOnly={two && i === 1}
              value={two && i === 1 ? (isNaN(nums[1]) ? "" : String(+nums[1].toFixed(1))) : vals[i]}
              placeholder={two && i === 1 ? "Filled in for you" : "e.g. 58"}
              onChange={(e) => setVals((p) => p.map((v, j) => (j === i ? e.target.value : v)))} />
          </label>
        ))}
        {!two && !isNaN(total) && !valid && <p className="t-micro">These add up to {total.toFixed(1)}%. They need to total 100%.</p>}
        <label className="field"><span>Bankroll in ₹ (optional)</span>
          <input className="input" type="number" inputMode="decimal" min="1" value={bank} onChange={(e) => setBank(e.target.value)} placeholder="10000" />
        </label>
        <button className="btn btn-primary" disabled={!valid || state.k === "busy"}>{state.k === "busy" ? "Checking" : "Check this bet"}</button>
      </form>
      <div aria-live="polite" style={{ display: "grid", gap: 10 }}>
        {state.k === "err" && <p className="note">{state.m}</p>}
        {state.k === "ok" && state.d.analysis.map((a) => (
          <div key={a.outcome} className={`verdict ${a.label === "BET" ? "good" : a.label === "TRAP" ? "critical" : ""}`}>
            <div className="verdict-top">
              <span className="verdict-call" style={{ fontSize: "1.125rem" }}>{a.outcome}: {CALL[a.label]}</span>
              <span className={`pill ${a.label === "BET" ? "pill-good" : a.label === "TRAP" ? "pill-critical" : ""}`}>{a.label}</span>
            </div>
            <p className="t-small">Best price {odds(a.best_odds)} at {a.best_book}. Your chance {pct(a.model_prob ?? 0, 1)} gives an edge of {a.edge_vs_best === null ? "–" : signedPct(a.edge_vs_best)}.</p>
            {a.stake_fraction > 0 && <p className="t-small">Suggested stake {(a.stake_fraction * 100).toFixed(2)}% of bankroll{a.stake_amount ? `, ${inr(a.stake_amount)}` : ""}.</p>}
          </div>
        ))}
      </div>
    </section>
  );
}

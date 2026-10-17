import { Suspense } from "react";
import Problem from "@/components/Problem";
import SettleButtons from "@/components/SettleButtons";
import { fetchBets, formatDay, inr, signedPct } from "@/lib/api";

export default function LedgerPage() {
  return (
    <Suspense fallback={<div className="page"><div className="skeleton" style={{ height: 360 }} /></div>}>
      <Ledger />
    </Suspense>
  );
}

async function Ledger() {
  let data;
  try { data = await fetchBets(); } catch (e) { return <Problem e={e} title="Your ledger can't load" />; }
  const { bets, summary: s } = data;
  return (
    <main className="page">
      <header className="hero" style={{ paddingTop: 0 }}>
        <h1 className="t-title">Ledger</h1>
        <p className="t-sub">Every bet you log, settled against the final price before the match. Beating that closing price over many bets is the real sign of an edge.</p>
      </header>
      <dl className="card tally">
        <div><dt className="t-micro">Settled</dt><dd className="num-l">{s.settled}</dd></div>
        <div><dt className="t-micro">Open</dt><dd className="num-l">{s.open}</dd></div>
        <div><dt className="t-micro">Profit</dt><dd className={`num-l ${s.profit >= 0 ? "pos" : "neg"}`}>{inr(s.profit)}</dd></div>
        <div><dt className="t-micro">Return on stakes</dt><dd className="num-l">{s.roi === null ? "–" : signedPct(s.roi)}</dd></div>
        <div className="hl"><dt className="t-micro">Beat the closing price by</dt>
          <dd className="num-l">{s.avg_clv === null ? "–" : signedPct(s.avg_clv)}{s.clv_count > 0 && <span className="t-micro" style={{ fontFamily: "var(--font-text)" }}> over {s.clv_count} {s.clv_count === 1 ? "bet" : "bets"}</span>}</dd></div>
      </dl>
      {bets.length === 0 ? (
        <p className="note">Nothing logged yet. When Doosra calls a price worth backing, log it from the price sheet.</p>
      ) : (
        <ul className="list">
          {bets.map((b) => (
            <li key={b.id} className="row" style={{ flexWrap: "wrap" }}>
              <div className="row-main" style={{ minWidth: 200 }}>
                <span className="row-title">{b.outcome}</span>
                <span className="t-micro">{b.event_id} · {b.book} · {formatDay(b.placed_at)}</span>
              </div>
              <div style={{ display: "grid", justifyItems: "end", gap: 2 }}>
                <span className="num" style={{ fontWeight: 700 }}>{inr(b.stake)} at {b.odds.toFixed(2)}</span>
                <span className="t-micro">{b.closing_fair_prob ? `${signedPct(b.odds * b.closing_fair_prob - 1)} vs close` : "Close not known"}</span>
              </div>
              <div style={{ minWidth: 150, display: "flex", justifyContent: "flex-end" }}>
                {b.result ? <span className={`pill ${b.result === "won" ? "pill-good" : b.result === "lost" ? "pill-critical" : ""}`} style={{ textTransform: "capitalize" }}>{b.result}</span> : <SettleButtons betId={b.id} />}
              </div>
            </li>
          ))}
        </ul>
      )}
    </main>
  );
}

import { Suspense, type CSSProperties } from "react";
import PriceCheck from "@/components/PriceCheck";
import Problem from "@/components/Problem";
import Split from "@/components/Split";
import { teamVars } from "@/lib/teams";
import { fetchMatchOdds, formatStart, matchTitle, odds, pct } from "@/lib/api";

export default function PricesPage({ params }: { params: Promise<{ id: string }> }) {
  return (
    <Suspense fallback={<div className="page"><div className="skeleton" style={{ height: 420 }} /></div>}>
      <Board params={params} />
    </Suspense>
  );
}

async function Board({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  let m;
  try { m = await fetchMatchOdds(id); } catch (e) { return <Problem e={e} title="These prices can't load" />; }
  const best = (i: number) => Math.max(...m.books.map((b) => b.odds[i]));
  const fair = m.analysis.map((a) => a.consensus_prob);
  return (
    <main className="page">
      <header style={{ display: "grid", gap: 6 }}>
        <span className="t-micro">{m.match.league} · {formatStart(m.match.start)} IST{m.data_mode === "demo" ? " · sample prices" : ""}</span>
        <h1 className="t-title">{matchTitle(m.outcomes)}</h1>
      </header>
      <div className="fc-grid team-scope" style={teamVars([m.outcomes[0], m.outcomes[m.outcomes.length - 1]]) as CSSProperties}>
        <div className="fc-main">
          <section className="card card-pad" style={{ display: "grid", gap: 18 }}>
            <div className="section-head"><h2 className="t-head">The market&apos;s fair view</h2></div>
            <Split names={m.outcomes} probs={fair} size="l"
              caption={`Bookmakers' margins removed${m.sharp_book ? `, led by ${m.sharp_book}, the sharpest book here` : ""}.`} />
          </section>
          <section className="section" aria-labelledby="bk">
            <div className="section-head">
              <h2 id="bk" className="t-head">Every bookmaker</h2>
              <span className="t-micro">Best price for each side highlighted</span>
            </div>
            <div className="card scroll-x">
              <table className="board">
                <caption className="sr-only">Decimal odds by bookmaker, with each one&apos;s margin.</caption>
                <thead><tr><th scope="col">Bookmaker</th>{m.outcomes.map((o) => <th key={o} scope="col">{o}</th>)}<th scope="col">Margin</th></tr></thead>
                <tbody>
                  {m.books.map((b) => (
                    <tr key={b.book}>
                      <th scope="row" style={{ fontWeight: 600 }}>{b.book}{b.is_sharp && <span className="pill" style={{ marginLeft: 8 }}>Sharp</span>}</th>
                      {b.odds.map((o, i) => (
                        <td key={i}><span className={`price${o === best(i) ? " best" : ""}`}>{odds(o)}</span>{o === best(i) && <span className="sr-only"> best price</span>}</td>
                      ))}
                      <td className="t-small">{pct(b.margin, 1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="t-micro">{m.best_price_margin < 0 ? "The best prices add up to a guaranteed profit. That usually means one price is stale, so re-check it." : `Even the best price on each side keeps a ${pct(m.best_price_margin, 1)} margin.`}</p>
          </section>
        </div>
        <aside className="fc-side"><PriceCheck matchId={m.match.id} outcomes={m.outcomes} /></aside>
      </div>
    </main>
  );
}

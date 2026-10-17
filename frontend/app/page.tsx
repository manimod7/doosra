import Link from "next/link";
import { Suspense, type CSSProperties } from "react";
import Composer from "@/components/Composer";
import RefreshOdds from "@/components/RefreshOdds";
import Split from "@/components/Split";
import { teamVars } from "@/lib/teams";
import {
  fetchBudget, fetchCard, fetchCatalog, fetchEngine, fetchFixtures, forecastHref, formatDay, formatStart,
  matchTitle, type Catalog, type Engine,
} from "@/lib/api";

export default function Today() {
  return (
    <main className="page">
      <header className="hero rise">
        <h1 className="t-display">Read the match<br />before it&apos;s bowled.</h1>
        <p className="t-sub">
          Doosra plays each match out thousands of times, ball by ball, then tells you what is likely and whether a
          price is fair. Every result is checked against what it said, and it keeps learning.
        </p>
      </header>
      <Suspense fallback={<div className="skeleton" style={{ height: 320 }} />}>
        <Featured />
      </Suspense>
      <Suspense fallback={<div className="skeleton" style={{ height: 220 }} />}>
        <Prices />
      </Suspense>
      <Suspense fallback={<div className="skeleton" style={{ height: 110 }} />}>
        <Learning />
      </Suspense>
    </main>
  );
}

async function Featured() {
  let catalog: Catalog;
  try {
    catalog = await fetchCatalog();
  } catch {
    return (
      <section className="card card-pad" style={{ display: "grid", gap: 8 }}>
        <h2 className="t-head">Forecasts start after the first learning cycle</h2>
        <p className="t-small">Doosra hasn&apos;t learned from any matches yet. Run a learning cycle on Cricsheet data and forecasts appear here.</p>
      </section>
    );
  }
  const [a, b] = catalog.teams;
  const venue = catalog.venues[0];
  let preview = null;
  if (a && b && venue) {
    try {
      const res = await fetchCard({ team_a: a, team_b: b, venue, sims: 1000 });
      preview = res.card;
    } catch { /* the composer still works without a preview */ }
  }
  return (
    <section className="section" aria-labelledby="fc">
      <div className="card card-pad" style={{ display: "grid", gap: 22 }}>
        <div className="section-head">
          <h2 id="fc" className="t-head">Forecast a match</h2>
          <span className="t-micro">{catalog.teams.length} teams · {catalog.venues.length} grounds</span>
        </div>
        <Composer teams={catalog.teams} venues={catalog.venues} />
        {preview && (
          <Link href={forecastHref({ team_a: a, team_b: b, venue })} className="inset team-scope"
            style={{ display: "grid", gap: 14, padding: 18, ...(teamVars([a, b]) as CSSProperties) }}>
            <div className="section-head">
              <span className="t-small">For example, {a} v {b} at {venue}</span>
              <span className="link">Open forecast</span>
            </div>
            <Split names={[a, b]} probs={[preview.winner[a], preview.winner[b]]} size="l" />
          </Link>
        )}
      </div>
    </section>
  );
}

async function Prices() {
  let fixtures;
  try {
    fixtures = await fetchFixtures();
  } catch {
    return null;
  }
  const live = fixtures.data_mode === "live";
  const budget = live ? await fetchBudget().catch(() => null) : null;
  return (
    <section className="section" aria-labelledby="pr">
      <div className="section-head">
        <h2 id="pr" className="t-head">Bookmaker prices</h2>
        {!live && <span className="pill">Sample prices</span>}
      </div>
      {live && <RefreshOdds remaining={budget?.remaining ?? null} />}
      {fixtures.matches.length === 0 ? (
        <p className="note">No upcoming matches stored. Refresh odds to fetch them.</p>
      ) : (
        <ul className="list">
          {fixtures.matches.map((m) => (
            <li key={m.id}>
              <Link href={`/prices/${m.id}`} className="row fixture-row">
                <div className="row-main">
                  <span className="row-title">{matchTitle(m.outcomes)}</span>
                  <span className="t-micro">{m.league} · {formatStart(m.start)}</span>
                </div>
                {m.books ? <span className="t-micro">{m.books} books</span> : null}
                <svg className="chev" viewBox="0 0 8 13" aria-hidden="true"><path d="M1.5 1.5 6.5 6.5l-5 5" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" /></svg>
              </Link>
            </li>
          ))}
        </ul>
      )}
      {!live && <p className="t-micro">{fixtures.notice}</p>}
    </section>
  );
}

async function Learning() {
  let engine: Engine;
  try {
    engine = await fetchEngine();
  } catch {
    return null;
  }
  const settled = Object.values(engine.market_report).reduce((s, m) => s + m.n, 0);
  const last = engine.runs[0];
  const ev = engine.model?.evaluation;
  const gain = ev?.ball_log_loss && ev.baseline_log_loss ? (1 - ev.ball_log_loss / ev.baseline_log_loss) * 100 : null;
  return (
    <section className="section" aria-labelledby="ln">
      <div className="section-head">
        <h2 id="ln" className="t-head">How Doosra is doing</h2>
        <Link href="/learning" className="link">See how it learns</Link>
      </div>
      <div className="card strip">
        <div><span className="t-micro">Learned up to</span><span className="num-l" style={{ fontSize: "1.5rem" }}>{engine.model ? formatDay(engine.model.trained_through) : "Not yet"}</span></div>
        <div><span className="t-micro">Predictions checked</span><span className="num-l" style={{ fontSize: "1.5rem" }}>{settled.toLocaleString("en-IN")}</span></div>
        <div><span className="t-micro">Better than league averages</span><span className="num-l" style={{ fontSize: "1.5rem" }}>{gain === null ? "–" : `${gain.toFixed(1)}%`}</span></div>
        <div><span className="t-micro">Last learning cycle</span><span style={{ fontWeight: 650 }}>{last ? (last.promoted ? "New model adopted" : "Kept the current model") : "None yet"}</span></div>
      </div>
      {engine.drift.retrain_recommended && <p className="note note-strong">Some markets have drifted lately. The next learning cycle will retrain.</p>}
    </section>
  );
}

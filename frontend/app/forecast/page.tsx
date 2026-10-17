import { Suspense, type CSSProperties } from "react";
import Composer from "@/components/Composer";
import MatchControls from "@/components/MatchControls";
import Markets, { WinnerButton } from "@/components/Markets";
import Split from "@/components/Split";
import Worm from "@/components/Worm";
import Problem from "@/components/Problem";
import { teamVar, teamVars } from "@/lib/teams";
import { fetchCard, fetchCatalog, fetchEngine, formatDay, type LiveInput, type MatchQuery } from "@/lib/api";

type SP = Promise<{ [k: string]: string | string[] | undefined }>;
const one = (v: string | string[] | undefined) => (typeof v === "string" ? v : undefined);

export default function ForecastPage({ searchParams }: { searchParams: SP }) {
  return (
    <Suspense fallback={<Skeleton />}>
      <Forecast searchParams={searchParams} />
    </Suspense>
  );
}

async function Forecast({ searchParams }: { searchParams: SP }) {
  const sp = await searchParams;
  const a = one(sp.a), b = one(sp.b), venue = one(sp.venue);

  if (!a || !b || !venue) {
    let catalog;
    try { catalog = await fetchCatalog(); } catch (e) { return <Problem e={e} />; }
    return (
      <main className="page">
        <header className="hero">
          <h1 className="t-title">Forecast a match</h1>
          <p className="t-sub">Pick the two sides and the ground. Doosra uses each side&apos;s most recent eleven unless you change it.</p>
        </header>
        <div className="card card-pad"><Composer teams={catalog.teams} venues={catalog.venues} /></div>
      </main>
    );
  }

  let live: LiveInput | null = null;
  if (one(sp.mode) === "live" && one(sp.l_score) && one(sp.l_ov)) {
    live = {
      batting: one(sp.l_bat) ?? a, innings: one(sp.l_inn) === "2" ? 2 : 1, score: Number(one(sp.l_score)),
      wickets: Number(one(sp.l_wk) ?? 0), overs: Number(one(sp.l_ov)), target: one(sp.l_tgt) ? Number(one(sp.l_tgt)) : null,
    };
  }
  const query: MatchQuery = { team_a: a, team_b: b, venue, batting_first: one(sp.first) ?? null, live };

  return (
    <main className="page">
      <header style={{ display: "grid", gap: 18 }}>
        <div style={{ display: "grid", gap: 6 }}>
          <span className="t-micro">{venue}</span>
          <h1 className="t-title">{a} <span style={{ color: "var(--ink-3)", fontWeight: 500 }}>v</span> {b}</h1>
        </div>
        <MatchControls teams={[a, b]} />
      </header>
      <Suspense key={JSON.stringify(query)} fallback={<Skeleton />}>
        <Body query={query} />
      </Suspense>
    </main>
  );
}

async function Body({ query }: { query: MatchQuery }) {
  let res, engine;
  try {
    [res, engine] = await Promise.all([fetchCard({ ...query, sims: 2000 }), fetchEngine()]);
  } catch (e) {
    return <Problem e={e} title="This forecast can't be made right now" />;
  }
  const { card } = res;
  const teams = Object.keys(card.winner);
  const fav = teams.reduce((x, y) => (card.winner[x] >= card.winner[y] ? x : y));
  const firstTotals = teams.map((t) => card.teams[t].find((r) => r.spec.type === "team_total"));

  return (
    <div className="fc-grid team-scope" style={teamVars([teams[0], teams[1]]) as CSSProperties}>
      <div className="fc-main">
        <section className="card card-pad" style={{ display: "grid", gap: 20 }} aria-labelledby="win">
          <div className="section-head">
            <h2 id="win" className="t-head">Chance to win</h2>
            <span className="t-micro">{res.sims.toLocaleString("en-IN")} simulated matches{res.live ? " from the current score" : ""}</span>
          </div>
          <Split names={teams} probs={teams.map((t) => card.winner[t])} />
          <div style={{ display: "flex", flexWrap: "wrap", gap: 10 }}>
            {teams.map((t) => <WinnerButton key={t} team={t} p={card.winner[t]} query={query} />)}
          </div>
        </section>

        <section className="card card-pad" style={{ display: "grid", gap: 14 }} aria-labelledby="worm">
          <div className="section-head">
            <h2 id="worm" className="t-head">How the innings should unfold</h2>
          </div>
          <p className="t-small">
            {firstTotals.map((r, i) => r && `${teams[i]} about ${Math.round(r.median)}`).filter(Boolean).join(", ")}.
            {" "}{fav} {card.winner[fav] > 0.6 ? "are clear favourites" : card.winner[fav] > 0.53 ? "edge it" : "and their opponents are close to level"}.
          </p>
          <Worm worm={card.worm} teams={teams} />
        </section>

        <Markets card={card} query={query} />
      </div>

      <aside className="fc-side">
        <section className="card card-pad" style={{ display: "grid", gap: 14 }} aria-labelledby="xi">
          <h2 id="xi" className="t-head" style={{ fontSize: "1.0625rem" }}>Elevens used</h2>
          {teams.map((t) => (
            <div key={t} style={{ display: "grid", gap: 6 }}>
              <span className="row-title"><span className="team-dot" style={{ ["--k" as string]: teamVar(teams, t) }} />{t}</span>
              <ol className="lineup">
                {res.lineups[t].batting.map((p, k) => <li key={p}><span>{k + 1}</span><span>{p}</span></li>)}
              </ol>
            </div>
          ))}
          <p className="t-micro">Each side&apos;s most recent batting order. Bowlers: the ones they used last time.</p>
        </section>
        <section className="card card-pad" style={{ display: "grid", gap: 12 }} aria-labelledby="about">
          <h2 id="about" className="t-head" style={{ fontSize: "1.0625rem" }}>About this forecast</h2>
          <dl className="dl">
            <div><dt>Model</dt><dd>{res.model_version.slice(0, 8)}</dd></div>
            {engine.model && <div><dt>Learned up to</dt><dd>{formatDay(engine.model.trained_through)}</dd></div>}
            <div><dt>Corrected markets</dt><dd>{engine.calibrated_markets.length || "None yet"}</dd></div>
          </dl>
          <p className="t-micro">Every ball is predicted from both players&apos; records, their history against each other, the ground, form, the toss and the match situation. Weather and pitch reports are not in the model yet.</p>
        </section>
      </aside>
    </div>
  );
}

function Skeleton() {
  return (
    <div className="fc-grid" aria-busy="true" aria-label="Simulating the match">
      <div className="fc-main">
        <div className="skeleton" style={{ height: 210 }} />
        <div className="skeleton" style={{ height: 380 }} />
        <div className="skeleton" style={{ height: 300 }} />
      </div>
      <div className="fc-side"><div className="skeleton" style={{ height: 420 }} /></div>
    </div>
  );
}

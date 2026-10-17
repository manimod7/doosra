import { Suspense } from "react";
import Problem from "@/components/Problem";
import { fetchEngine, formatDay, pct, type MarketScore } from "@/lib/api";

const MARKET_NAMES: Record<string, string> = {
  match_winner: "Match winner", team_total: "Innings runs", session: "Sessions", wickets_by: "Wickets by an over",
  first_wicket: "Score at first wicket", over_runs: "Runs in an over", player_runs: "Batter runs",
  match_total: "Match runs", match_sixes: "Match sixes", match_fours: "Match fours",
};

export default function LearningPage() {
  return (
    <Suspense fallback={<div className="page"><div className="skeleton" style={{ height: 480 }} /></div>}>
      <Learning />
    </Suspense>
  );
}

async function Learning() {
  let e;
  try { e = await fetchEngine(); } catch (err) { return <Problem e={err} title="Learning history can't load" />; }
  const ev = e.model?.evaluation;
  const gain = ev?.ball_log_loss && ev.baseline_log_loss ? (1 - ev.ball_log_loss / ev.baseline_log_loss) * 100 : null;
  const markets = Object.entries(e.market_report).sort((a, b) => b[1].n - a[1].n);

  return (
    <main className="page">
      <header className="hero" style={{ paddingTop: 0 }}>
        <h1 className="t-title">How Doosra learns</h1>
        <p className="t-sub">
          Every price Doosra gives is saved. When the scorecard arrives, each one is checked. Where a market has been
          consistently off, its chances are corrected. A retrained model only replaces the current one if it does better
          on matches it has never seen.
        </p>
      </header>

      <section className="section" aria-labelledby="now">
        <h2 id="now" className="t-head">The model in use</h2>
        {e.model ? (
          <div className="card strip">
            <div className="metric"><span className="t-micro">Learned up to</span><span className="num-l" style={{ fontSize: "1.5rem" }}>{formatDay(e.model.trained_through)}</span></div>
            <div className="metric"><span className="t-micro">Sharper than league averages</span><span className="num-l" style={{ fontSize: "1.5rem" }}>{gain === null ? "–" : `${gain.toFixed(1)}%`}</span><span className="t-micro">per ball, on unseen matches</span></div>
            <div className="metric"><span className="t-micro">Innings totals inside its 80% range</span><span className="num-l" style={{ fontSize: "1.5rem" }}>{ev?.total_coverage80 === undefined ? "–" : pct(ev.total_coverage80)}</span><span className="t-micro">aim is 80%, over {ev?.sim_matches ?? 0} matches</span></div>
            <div className="metric"><span className="t-micro">Learner</span><span style={{ fontWeight: 650 }}>{e.model.kind === "hgb" ? "Gradient-boosted trees" : e.model.kind === "logit" ? "Logistic model" : e.model.kind}</span><span className="t-micro">picked by out-of-sample score</span></div>
          </div>
        ) : (
          <p className="note">No model yet. The first learning cycle on Cricsheet data creates one.</p>
        )}
      </section>

      <section className="section" aria-labelledby="cal">
        <div className="section-head">
          <h2 id="cal" className="t-head">Predicted against what happened</h2>
          <span className="t-micro" style={{ display: "inline-flex", gap: 14 }}>
            <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}><span className="calib-mark pred" style={{ position: "static", transform: "none", display: "inline-block" }} />Predicted</span>
            <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}><span className="calib-mark act" style={{ position: "static", transform: "none", display: "inline-block" }} />Happened</span>
          </span>
        </div>
        {markets.length === 0 ? (
          <p className="note">Nothing settled yet. Log forecasts before matches and they are checked here once results arrive.</p>
        ) : (
          <div className="list">
            {markets.map(([k, m]) => <CalibRow key={k} name={MARKET_NAMES[k] ?? k} m={m} corrected={e.calibrated_markets.includes(k)} drift={e.drift.drifting[k]} />)}
            <div className="calib" style={{ paddingTop: 6, paddingBottom: 10 }}>
              <span />
              <div className="calib-axis"><span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span></div>
              <span />
            </div>
          </div>
        )}
        <p className="t-micro">Each row compares the average chance Doosra gave with how often it actually happened. The closer the two dots, the more you can trust the numbers. Score is the Brier score: lower is better, 0.25 is a coin flip.</p>
      </section>

      <section className="section" aria-labelledby="hist">
        <h2 id="hist" className="t-head">Learning cycles</h2>
        {e.runs.length === 0 ? <p className="note">No cycles yet.</p> : (
          <ol className="list timeline">
            {e.runs.map((r) => (
              <li key={r.version + r.at}>
                <span className={`tl-dot${r.promoted ? " ok" : ""}`} aria-hidden="true" />
                <div style={{ display: "grid", gap: 2, minWidth: 0 }}>
                  <span style={{ fontWeight: 650 }}>{r.promoted ? "New model adopted" : "Kept the current model"}</span>
                  <span className="t-small">{sentence(r.reason)}</span>
                  <span className="t-micro">Data up to {formatDay(r.trained_through)}{r.report.n_balls ? ` · judged on ${r.report.n_balls.toLocaleString("en-IN")} unseen balls` : ""}</span>
                </div>
                <span className="t-micro">{formatDay(r.at)}</span>
              </li>
            ))}
          </ol>
        )}
      </section>
    </main>
  );
}

function CalibRow({ name, m, corrected, drift }: { name: string; m: MarketScore; corrected: boolean; drift?: string[] }) {
  const lo = Math.min(m.mean_predicted, m.actual_rate);
  const hi = Math.max(m.mean_predicted, m.actual_rate);
  return (
    <div className="calib">
      <div style={{ display: "grid", gap: 2 }}>
        <span style={{ fontWeight: 600 }}>{name}</span>
        <span className="t-micro">{m.n.toLocaleString("en-IN")} checked{corrected ? " · corrected" : ""}</span>
      </div>
      <div className="calib-track" role="img" aria-label={`${name}: predicted ${pct(m.mean_predicted, 1)}, happened ${pct(m.actual_rate, 1)}`}>
        <span className="calib-gap" style={{ left: `${lo * 100}%`, width: `${(hi - lo) * 100}%` }} />
        <span className="calib-mark pred" style={{ left: `${m.mean_predicted * 100}%` }} />
        <span className="calib-mark act" style={{ left: `${m.actual_rate * 100}%` }} />
      </div>
      <div style={{ display: "grid", justifyItems: "end", gap: 2 }}>
        <span className="num" style={{ fontWeight: 700 }}>{m.brier.toFixed(3)}</span>
        <span className="t-micro">{drift ? "Drifting" : Math.abs(m.mean_predicted - m.actual_rate) > 0.05 ? `Off by ${pct(Math.abs(m.mean_predicted - m.actual_rate))}` : "On target"}</span>
      </div>
    </div>
  );
}

function sentence(s: string) {
  return s.charAt(0).toUpperCase() + s.slice(1) + (s.endsWith(".") ? "" : ".");
}

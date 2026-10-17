"use client";

import { useState } from "react";
import { teamVar } from "@/lib/teams";
import PriceSheet, { type SheetTarget } from "@/components/PriceSheet";
import { odds, pct, shortMarket, type Card, type LinePrice, type MatchQuery } from "@/lib/api";


function Chevron() {
  return (
    <svg className="chev" viewBox="0 0 8 13" aria-hidden="true"><path d="M1.5 1.5 6.5 6.5l-5 5" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" /></svg>
  );
}

export default function Markets({ card, query }: { card: Card; query: MatchQuery }) {
  const [open, setOpen] = useState<SheetTarget | null>(null);
  const teams = Object.keys(card.winner);   // same order as the page, so colours match
  const color = (t: string) => teamVar(teams, t);

  const byType = (type: string) => teams.flatMap((t) => card.teams[t].filter((r) => r.spec.type === type).map((r) => ({ t, r })));
  const sessions = teams.flatMap((t) => card.teams[t].filter((r) => r.spec.type === "session").map((r) => ({ t, r })))
    .sort((a, b) => (a.r.spec.overs ?? 0) - (b.r.spec.overs ?? 0) || teams.indexOf(a.t) - teams.indexOf(b.t));

  const openLine = (r: LinePrice, team: string | null) => setOpen({
    title: team ? `${team} · ${shortMarket(r)}` : shortMarket(r),
    spec: r.spec, line: r.line, pOver: r.p_over, q10: r.q10, q90: r.q90,
  });

  const Row = ({ r, t }: { r: LinePrice; t: string | null }) => (
    <li>
      <button type="button" className="row" onClick={() => openLine(r, t)}>
        <div className="row-main mkt">
          <span className="row-title">
            {t && <span className="team-dot" style={{ ["--k" as string]: color(t) }} />}
            {t ? `${t}` : shortMarket(r)}
            {t && <span className="t-small" style={{ fontWeight: 500 }}> · {shortMarket(r)}</span>}
          </span>
          <span className="mkt-line" aria-label={`Fair line ${r.line}`}>{r.line}</span>
          <span className="mkt-odds">
            <span className="odd"><small>Over</small><b>{odds(r.fair_over)}</b></span>
            <span className="odd"><small>Under</small><b>{odds(r.fair_under)}</b></span>
          </span>
          <span className="mkt-range">8 in 10 between {Math.round(r.q10)} and {Math.round(r.q90)} · over {pct(r.p_over)}{r.calibrated ? " · corrected by past results" : ""}</span>
        </div>
        <Chevron />
      </button>
    </li>
  );

  const groups: { title: string; rows: { t: string | null; r: LinePrice }[] }[] = [
    { title: "Innings runs", rows: byType("team_total") },
    { title: "Sessions", rows: sessions },
    { title: "Wickets and openers", rows: [...byType("wickets_by"), ...byType("first_wicket")] },
    { title: "Whole match", rows: Object.values(card.match).map((r) => ({ t: null, r })) },
  ].filter((g) => g.rows.length);

  return (
    <section className="section" aria-labelledby="mk">
      <div className="section-head">
        <h2 id="mk" className="t-head">Fair prices</h2>
        <span className="t-micro">Tap a market to check a bookmaker&apos;s price</span>
      </div>
      {groups.map((g) => (
        <div key={g.title} className="mkt-group">
          <h3>{g.title}</h3>
          <ul className="list">{g.rows.map(({ t, r }) => <Row key={`${t}-${r.market}`} r={r} t={t} />)}</ul>
        </div>
      ))}
      <div className="mkt-group">
        <h3>Batters</h3>
        <ul className="list">
          {card.players.slice(0, 8).map((p) => (
            <li key={p.player}>
              <button type="button" className="row" onClick={() => setOpen({
                title: `${p.player} · runs`, spec: { type: "player_runs", player: p.player },
                line: p.fair_line,
              })}>
                <div className="row-main player">
                  <span className="row-title"><span className="team-dot" style={{ ["--k" as string]: color(p.team) }} />{p.player}</span>
                  <span className="num" style={{ fontWeight: 700, fontSize: "1.125rem" }}>{p.expected_runs.toFixed(0)}<span className="t-micro"> runs</span></span>
                  <span className="meter" aria-hidden="true"><i style={{ width: `${Math.min(100, p.p_30_plus * 100)}%`, ["--k" as string]: color(p.team) }} /></span>
                  <span className="t-micro" style={{ gridColumn: "1 / -1" }}>30+ in {pct(p.p_30_plus)} of simulations · 50+ in {pct(p.p_50_plus)} · fair line {p.fair_line}</span>
                </div>
                <Chevron />
              </button>
            </li>
          ))}
        </ul>
      </div>
      {open && <PriceSheet target={open} query={query} onClose={() => setOpen(null)} />}
    </section>
  );
}

export function WinnerButton({ team, p, query }: { team: string; p: number; query: MatchQuery }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <button type="button" className="btn btn-quiet" onClick={() => setOpen(true)}>Check a price on {team}</button>
      {open && (
        <PriceSheet query={query} onClose={() => setOpen(false)}
          target={{ title: `${team} to win`, spec: { type: "match_winner", team }, line: null, pOver: p }} />
      )}
    </>
  );
}

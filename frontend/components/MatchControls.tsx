"use client";

import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useState, useTransition } from "react";

/** Toss and live-situation controls. State lives in the URL so a forecast can be shared or revisited. */
export default function MatchControls({ teams }: { teams: [string, string] }) {
  const router = useRouter();
  const path = usePathname();
  const params = useSearchParams();
  const [pending, start] = useTransition();
  const live = params.get("mode") === "live";
  const first = params.get("first") ?? "";

  const [bat, setBat] = useState(params.get("l_bat") ?? teams[0]);
  const [inn, setInn] = useState(params.get("l_inn") ?? "1");
  const [score, setScore] = useState(params.get("l_score") ?? "");
  const [wk, setWk] = useState(params.get("l_wk") ?? "");
  const [ov, setOv] = useState(params.get("l_ov") ?? "");
  const [tgt, setTgt] = useState(params.get("l_tgt") ?? "");

  const go = (patch: Record<string, string | null>) => {
    const p = new URLSearchParams(params.toString());
    Object.entries(patch).forEach(([k, v]) => (v === null || v === "" ? p.delete(k) : p.set(k, v)));
    start(() => router.push(`${path}?${p}`, { scroll: false }));
  };

  const overs = Number(ov);
  const ballsPart = Math.round((overs % 1) * 10);
  const liveValid = score !== "" && wk !== "" && ov !== "" && Number(wk) <= 9 && ballsPart <= 5 && overs <= 20 && (inn === "1" || Number(tgt) > 0);

  return (
    <div style={{ display: "grid", gap: 16 }} aria-busy={pending}>
      <div className="controls">
        <div>
          <span className="t-micro">Batting first</span>
          <div className="seg" role="group" aria-label="Batting first">
            <button aria-pressed={!first} onClick={() => go({ first: null })}>Not known yet</button>
            {teams.map((t) => <button key={t} aria-pressed={first === t} onClick={() => go({ first: t })}>{t}</button>)}
          </div>
        </div>
        <div>
          <span className="t-micro">Situation</span>
          <div className="seg" role="group" aria-label="Situation">
            <button aria-pressed={!live} onClick={() => go({ mode: null, l_bat: null, l_inn: null, l_score: null, l_wk: null, l_ov: null, l_tgt: null })}>Before the match</button>
            <button aria-pressed={live} onClick={() => go({ mode: "live" })}>In play</button>
          </div>
        </div>
        {pending && <span className="t-micro" role="status">Re-running the simulations</span>}
      </div>
      {live && (
        <form className="card card-pad rise" style={{ display: "grid", gap: 14 }}
          onSubmit={(e) => { e.preventDefault(); if (liveValid) go({ l_bat: bat, l_inn: inn, l_score: score, l_wk: wk, l_ov: ov, l_tgt: inn === "2" ? tgt : null }); }}>
          <p className="t-small">Enter the score as it stands. Doosra plays out the rest of the match from here.</p>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(130px, 1fr))", gap: 12 }}>
            <label className="field"><span>Batting now</span>
              <select className="select" value={bat} onChange={(e) => setBat(e.target.value)}>
                {teams.map((t) => <option key={t}>{t}</option>)}
              </select>
            </label>
            <label className="field"><span>Innings</span>
              <select className="select" value={inn} onChange={(e) => setInn(e.target.value)}>
                <option value="1">First</option><option value="2">Second (chasing)</option>
              </select>
            </label>
            <label className="field"><span>Runs</span><input className="input" type="number" min="0" inputMode="numeric" value={score} onChange={(e) => setScore(e.target.value)} placeholder="84" /></label>
            <label className="field"><span>Wickets</span><input className="input" type="number" min="0" max="9" inputMode="numeric" value={wk} onChange={(e) => setWk(e.target.value)} placeholder="2" /></label>
            <label className="field"><span>Overs, e.g. 7.3</span><input className="input" type="number" min="0" max="20" step="0.1" inputMode="decimal" value={ov} onChange={(e) => setOv(e.target.value)} placeholder="10.2" /></label>
            {inn === "2" && <label className="field"><span>Target</span><input className="input" type="number" min="1" inputMode="numeric" value={tgt} onChange={(e) => setTgt(e.target.value)} placeholder="171" /></label>}
          </div>
          {ov !== "" && ballsPart > 5 && <p className="t-micro" role="alert">After the dot comes balls in the over, 0 to 5.</p>}
          <div><button className="btn btn-primary" disabled={!liveValid || pending}>Update the forecast</button></div>
        </form>
      )}
    </div>
  );
}

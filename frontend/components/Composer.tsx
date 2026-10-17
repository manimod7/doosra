"use client";

import { useRouter } from "next/navigation";
import { useState, useTransition } from "react";
import { forecastHref } from "@/lib/api";

/** Pick two sides and a ground. The only form on the home screen. */
export default function Composer({ teams, venues, initial }: {
  teams: string[]; venues: string[]; initial?: { a?: string; b?: string; venue?: string };
}) {
  const router = useRouter();
  const [pending, start] = useTransition();
  const [a, setA] = useState(initial?.a ?? teams[0] ?? "");
  const [b, setB] = useState(initial?.b ?? teams[1] ?? "");
  const [venue, setVenue] = useState(initial?.venue ?? venues[0] ?? "");
  const same = a === b;

  return (
    <form className="compose" onSubmit={(e) => {
      e.preventDefault();
      if (!same) start(() => router.push(forecastHref({ team_a: a, team_b: b, venue })));
    }}>
      <label className="field"><span>Team</span>
        <select className="select" value={a} onChange={(e) => setA(e.target.value)}>{teams.map((t) => <option key={t}>{t}</option>)}</select>
      </label>
      <label className="field"><span>Opponent</span>
        <select className="select" value={b} onChange={(e) => setB(e.target.value)}>{teams.map((t) => <option key={t}>{t}</option>)}</select>
      </label>
      <label className="field"><span>Ground</span>
        <select className="select" value={venue} onChange={(e) => setVenue(e.target.value)}>{venues.map((v) => <option key={v}>{v}</option>)}</select>
      </label>
      <button className="btn btn-primary" disabled={same || pending}>
        {pending ? "Simulating" : same ? "Pick two teams" : "Forecast the match"}
      </button>
    </form>
  );
}

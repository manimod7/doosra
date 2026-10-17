"use client";

import { useMemo, useRef, useState } from "react";
import type { Worm as WormData } from "@/lib/api";
import { teamVar } from "@/lib/teams";

const W = 680;
const H = 300;
const M = { l: 40, r: 58, t: 26, b: 36 };
const SESSIONS = [6, 10, 15];

type Series = { team: string; data: WormData; color: string };

/**
 * "The worm": projected run progression for both innings, the way cricket
 * broadcasts show it. Line = median simulation, shaded = middle 80%.
 */
export default function Worm({ worm, teams, overs = 20 }: { worm: Record<string, WormData>; teams?: string[]; overs?: number }) {
  // Colour by team, not by position: the worm lists sides in batting order, the page in fixture order.
  const order = teams ?? Object.keys(worm);
  const series: Series[] = Object.entries(worm).map(([team, data]) => ({ team, data, color: teamVar(order, team) }));
  const [hover, setHover] = useState<number | null>(null);
  const box = useRef<SVGSVGElement>(null);

  const yMax = useMemo(() => {
    const all = series.flatMap((s) => s.data.p90.filter((v): v is number => v !== null));
    return Math.max(50, Math.ceil((Math.max(...all, 0) + 10) / 50) * 50);
  }, [series]);

  const x = (o: number) => M.l + (o / overs) * (W - M.l - M.r);
  const y = (v: number) => H - M.b - (v / yMax) * (H - M.t - M.b);

  const points = (vals: (number | null)[]) => {
    const pts: [number, number][] = [];
    const firstKnown = vals.findIndex((v) => v !== null);
    if (firstKnown === 0) pts.push([0, 0]);
    vals.forEach((v, k) => { if (v !== null) pts.push([k + 1, v]); });
    return pts;
  };
  const line = (pts: [number, number][]) => pts.map(([o, v], i) => `${i ? "L" : "M"}${x(o).toFixed(1)},${y(v).toFixed(1)}`).join("");
  const band = (s: Series) => {
    const hi = points(s.data.p90);
    const lo = points(s.data.p10).reverse();
    return hi.length ? `${line(hi)}${lo.map(([o, v]) => `L${x(o).toFixed(1)},${y(v).toFixed(1)}`).join("")}Z` : "";
  };

  // end labels: keep them at least 16px apart
  const ends = series.map((s) => {
    const pts = points(s.data.p50);
    const last = pts[pts.length - 1] ?? [0, 0];
    return { s, o: last[0], v: last[1], ly: y(last[1]) };
  });
  if (ends.length === 2 && Math.abs(ends[0].ly - ends[1].ly) < 16) {
    const [a, b] = ends[0].ly < ends[1].ly ? [ends[0], ends[1]] : [ends[1], ends[0]];
    const mid = (a.ly + b.ly) / 2;
    a.ly = mid - 8;
    b.ly = mid + 8;
  }

  const ticks = Array.from({ length: yMax / 50 + 1 }, (_, i) => i * 50);

  function onMove(e: React.PointerEvent<SVGRectElement>) {
    const r = box.current?.getBoundingClientRect();
    if (!r) return;
    const px = ((e.clientX - r.left) / r.width) * W;
    const o = Math.round(((px - M.l) / (W - M.l - M.r)) * overs);
    setHover(Math.min(overs, Math.max(1, o)));
  }

  const tip = hover !== null && (
    <div className="worm-tip" style={{ left: `${(x(hover) / W) * 100}%`, top: `${(M.t / H) * 100}%` }} role="status">
      <span className="t-micro">After over {hover}</span>
      {series.map((s) => {
        const v = s.data.p50[hover - 1];
        return (
          <span key={s.team} style={{ display: "flex", justifyContent: "space-between", gap: 12, alignItems: "baseline" }}>
            <span><span className="team-dot" style={{ ["--k" as string]: s.color }} />{s.team}</span>
            {v === null ? <span className="t-micro">already bowled</span> : (
              <span><b>{Math.round(v)}</b> <span className="t-micro">{Math.round(s.data.p10[hover - 1] ?? 0)}–{Math.round(s.data.p90[hover - 1] ?? 0)}</span></span>
            )}
          </span>
        );
      })}
    </div>
  );

  return (
    <figure style={{ display: "grid", gap: 12 }}>
      <div style={{ display: "flex", flexWrap: "wrap", gap: "8px 18px", alignItems: "center" }}>
        {series.map((s) => <span key={s.team} className="key" style={{ ["--k" as string]: s.color }}>{s.team}</span>)}
        <span className="t-micro">Line is the middle simulation; shading covers 8 in 10.</span>
      </div>
      <div style={{ position: "relative" }}>
        <svg ref={box} className="worm" viewBox={`0 0 ${W} ${H}`} role="img"
          aria-label={`Projected runs by over. ${ends.map((e) => `${e.s.team} about ${Math.round(e.v)}`).join(", ")}.`}>
          {ticks.map((t) => (
            <g key={t}>
              <line className="grid" x1={M.l} x2={W - M.r} y1={y(t)} y2={y(t)} />
              <text x={M.l - 8} y={y(t) + 4} textAnchor="end">{t}</text>
            </g>
          ))}
          {SESSIONS.filter((s) => s < overs).map((s) => (
            <g key={s}>
              <line className="grid" x1={x(s)} x2={x(s)} y1={M.t - 6} y2={H - M.b} />
              <text x={x(s)} y={M.t - 12} textAnchor="middle">{s === 6 ? "Powerplay" : `${s} ov`}</text>
            </g>
          ))}
          {[0, 5, 10, 15, 20].filter((o) => o <= overs).map((o) => (
            <text key={o} x={x(o)} y={H - M.b + 20} textAnchor="middle">{o}</text>
          ))}
          <text x={W - M.r} y={H - 4} textAnchor="end">Overs</text>
          {series.map((s) => <path key={`b${s.team}`} className="band fade-in" d={band(s)} fill={s.color} />)}
          {series.map((s) => (
            <path key={`m${s.team}`} className="median draw" d={line(points(s.data.p50))} stroke={s.color} pathLength={1} />
          ))}
          {ends.map((e) => (
            <g key={`e${e.s.team}`} className="fade-in">
              <circle cx={x(e.o)} cy={y(e.v)} r={5} fill={e.s.color} stroke="var(--surface)" strokeWidth={2} />
              <text className="end-label" x={x(e.o) + 10} y={e.ly + 5}>{Math.round(e.v)}</text>
            </g>
          ))}
          {hover !== null && (
            <g>
              <line className="cross" x1={x(hover)} x2={x(hover)} y1={M.t} y2={H - M.b} />
              {series.map((s) => {
                const v = s.data.p50[hover - 1];
                return v === null ? null : (
                  <circle key={s.team} cx={x(hover)} cy={y(v)} r={4.5} fill={s.color} stroke="var(--surface)" strokeWidth={2} />
                );
              })}
            </g>
          )}
          <rect x={M.l} y={M.t} width={W - M.l - M.r} height={H - M.t - M.b} fill="transparent"
            onPointerMove={onMove} onPointerLeave={() => setHover(null)} style={{ touchAction: "pan-y" }} />
        </svg>
        {tip}
      </div>
      <div className="sr-only"><table>
        <caption>Projected score after each over (middle simulation, with the range covering 8 in 10)</caption>
        <thead><tr><th>Over</th>{series.map((s) => <th key={s.team}>{s.team}</th>)}</tr></thead>
        <tbody>
          {Array.from({ length: overs }, (_, k) => (
            <tr key={k}><td>{k + 1}</td>{series.map((s) => (
              <td key={s.team}>{s.data.p50[k] === null ? "–" : `${Math.round(s.data.p50[k]!)} (${Math.round(s.data.p10[k]!)}–${Math.round(s.data.p90[k]!)})`}</td>
            ))}</tr>
          ))}
        </tbody>
      </table></div>
    </figure>
  );
}

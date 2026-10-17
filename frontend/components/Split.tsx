import { pct } from "@/lib/api";

const COLORS = ["var(--team-a)", "var(--team-b)", "var(--ink-3)"];

/** Win probability as two big numbers over one shared track. */
export default function Split({ names, probs, size = "xl", caption }: {
  names: string[]; probs: number[]; size?: "xl" | "l"; caption?: string;
}) {
  const label = names.map((n, i) => `${n} ${pct(probs[i])}`).join(", ");
  const first = 0;
  const last = names.length - 1;
  return (
    <figure className="split" aria-label={`Chance to win: ${label}`}>
      <div className="split-nums">
        <div className="split-side">
          <span className="split-team" style={{ ["--k" as string]: COLORS[first] }}><span>{names[first]}</span></span>
          <span className={size === "xl" ? "num-xl" : "num-l"}>{pct(probs[first])}</span>
        </div>
        {names.length > 2 && (
          <div className="split-side" style={{ textAlign: "center", justifyItems: "center" }}>
            <span className="t-micro">Draw</span>
            <span className="num-l" style={{ color: "var(--ink-2)" }}>{pct(probs[1])}</span>
          </div>
        )}
        <div className="split-side right">
          <span className="split-team" style={{ ["--k" as string]: COLORS[1] }}><span>{names[last]}</span></span>
          <span className={size === "xl" ? "num-xl" : "num-l"}>{pct(probs[last])}</span>
        </div>
      </div>
      <div className="split-track" aria-hidden="true">
        {probs.map((p, i) => (
          <i key={i} className="grow" style={{ flexGrow: Math.max(p, 0.005),
            ["--k" as string]: names.length > 2 && i === 1 ? COLORS[2] : COLORS[i === last ? 1 : 0] }} />
        ))}
      </div>
      {caption && <figcaption className="t-micro">{caption}</figcaption>}
    </figure>
  );
}

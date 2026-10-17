"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { bridge } from "@/lib/api";

type RefreshResult = { stored_rows: number; remaining: number | null; skipped: unknown[]; errors: { reason: string }[] };

export default function RefreshOdds({ remaining }: { remaining: number | null }) {
  const router = useRouter();
  const [state, setState] = useState<{ k: "idle" } | { k: "busy" } | { k: "msg"; text: string }>({ k: "idle" });

  async function run() {
    setState({ k: "busy" });
    try {
      const r = await bridge<RefreshResult>("admin/refresh", { method: "POST" });
      const parts = [`Saved ${r.stored_rows} prices.`];
      if (r.skipped.length) parts.push("Stopped early to keep your monthly reserve.");
      if (r.errors.length) parts.push(r.errors.map((e) => e.reason).join(" "));
      setState({ k: "msg", text: parts.join(" ") });
      router.refresh();
    } catch (err) {
      setState({ k: "msg", text: (err as Error).message });
    }
  }

  return (
    <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 12 }}>
      <button type="button" className="btn btn-quiet" onClick={run} disabled={state.k === "busy"}>
        {state.k === "busy" ? "Fetching prices" : "Refresh prices"}
      </button>
      <span className="t-micro">{remaining === null ? "Credits left: known after the first refresh" : `${remaining} credits left this month`}</span>
      {state.k === "msg" && <span className="t-small" role="status">{state.text}</span>}
    </div>
  );
}

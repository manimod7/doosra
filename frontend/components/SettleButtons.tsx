"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { bridge } from "@/lib/api";

export default function SettleButtons({ betId }: { betId: number }) {
  const router = useRouter();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function settle(result: "won" | "lost" | "void") {
    setBusy(true);
    setError(null);
    try {
      await bridge(`bets/${betId}/settle`, { method: "POST", body: { result } });
      router.refresh();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <span className="seg" role="group" aria-label="Settle this bet">
      <button type="button" disabled={busy} onClick={() => settle("won")}>Won</button>
      <button type="button" disabled={busy} onClick={() => settle("lost")}>Lost</button>
      <button type="button" disabled={busy} onClick={() => settle("void")}>Void</button>
      {error && <span className="t-micro" role="alert" style={{ alignSelf: "center", padding: "0 8px" }}>{error}</span>}
    </span>
  );
}

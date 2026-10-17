import { ApiError, ApiUnreachable } from "@/lib/api";

export default function Problem({ e, title = "This can't load right now" }: { e: unknown; title?: string }) {
  const msg = e instanceof ApiUnreachable
    ? "Doosra's engine isn't running. Start the API, then reload."
    : e instanceof ApiError ? e.message : "Something went wrong. Try again in a moment.";
  return (
    <main className="page">
      <div className="card card-pad" style={{ display: "grid", gap: 8, maxWidth: 620 }}>
        <h1 className="t-head">{title}</h1>
        <p className="t-small">{msg}</p>
      </div>
    </main>
  );
}

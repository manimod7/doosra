export const SERVER_API = process.env.DOOSRA_API_URL ?? "http://localhost:8000";
// Browser calls go through the server-side bridge in app/api/doosra.
export const BRIDGE = "/api/doosra";

/* ---------------------------------------------------------------- odds feed */

export type Fixture = { id: string; league: string; start: string; outcomes: string[]; books?: number };
export type FixtureList = { data_mode: "demo" | "live"; notice: string; matches: Fixture[] };
export type BookRow = { book: string; odds: number[]; margin: number; fair_probs: number[]; is_sharp: boolean };
export type Verdict = "BET" | "SKIP" | "TRAP" | "NO MODEL";
export type OutcomeAnalysis = {
  outcome: string; best_book: string; best_odds: number; consensus_prob: number; sharp_prob: number | null;
  model_prob: number | null; fair_odds_consensus: number; edge_vs_best: number | null; label: Verdict;
  stake_fraction: number; stake_amount?: number; reasons: string[];
};
export type Analysis = {
  outcomes: string[]; sharp_book: string | null; best_price_margin: number; books: BookRow[];
  analysis: OutcomeAnalysis[]; limits: string[];
};
export type MatchOdds = Analysis & {
  data_mode: "demo" | "live";
  match: { id: string; league: string; start: string; odds_fetched_at?: string | null };
};
export type Budget = { data_mode: "demo" | "live"; remaining: number | null; reserve: number };

/* ---------------------------------------------------------------- ledger */

export type Bet = {
  id: number; event_id: string; outcome: string; book: string; odds: number; stake: number; placed_at: string;
  result: "won" | "lost" | "void" | null; closing_fair_prob: number | null; note: string | null;
};
export type BetLedger = {
  bets: Bet[];
  summary: { settled: number; open: number; staked: number; profit: number; roi: number | null;
    avg_clv: number | null; clv_count: number; note: string };
};

/* ---------------------------------------------------------------- engine */

export type LinePrice = {
  market: string; spec: { type: string; team: string | null; overs: number | null; over: number | null; player: string | null };
  line: number | null; p_over: number; p_under: number; p_push: number; fair_over: number | null;
  fair_under: number | null; mean: number; median: number; q10: number; q90: number; n: number; calibrated: boolean;
};
export type PlayerLine = {
  player: string; team: string; expected_runs: number; p_bats: number; p_30_plus: number; p_50_plus: number; fair_line: number;
};
export type Worm = { p10: (number | null)[]; p50: (number | null)[]; p90: (number | null)[] };
export type Card = {
  winner: Record<string, number>; teams: Record<string, LinePrice[]>; players: PlayerLine[];
  match: Record<string, LinePrice>; worm: Record<string, Worm>;
};
export type CardResponse = {
  model_version: string; sims: number; live: boolean;
  lineups: Record<string, { batting: string[]; bowlers: string[] }>; card: Card; logged: number;
};
export type Catalog = {
  pool: string; teams: string[]; venues: string[];
  lineups: Record<string, { batting: string[]; bowlers: string[] }>; market_types: string[];
};
export type EngineRun = {
  version: string; promoted: boolean; reason: string; at: string; trained_through: string;
  report: { n_balls?: number; ball_log_loss?: number; baseline_log_loss?: number; total_coverage80?: number;
    sim_matches?: number; winner_log_loss?: number; chosen?: string };
};
export type MarketScore = {
  n: number; brier: number; log_loss: number; mean_predicted: number; actual_rate: number;
  buckets: { from: number; to: number; n: number; predicted: number; actual: number }[];
};
export type Engine = {
  model: null | { version: string; trained_through: string; pools: string[]; kind: string;
    evaluation: EngineRun["report"] & { folds?: unknown } };
  runs: EngineRun[];
  market_report: Record<string, MarketScore>;
  drift: { drifting: Record<string, string[]>; retrain_recommended: boolean };
  calibrated_markets: string[];
  cycle: { state: string; started?: string; finished?: string; summary?: { promoted: boolean; reason: string; settled: number } };
};
export type LiveInput = {
  batting: string; innings: 1 | 2; score: number; wickets: number; overs: number; target?: number | null;
};
export type MatchQuery = {
  team_a: string; team_b: string; venue: string; batting_first?: string | null; live?: LiveInput | null; sims?: number;
};

/* ---------------------------------------------------------------- transport */

export class ApiUnreachable extends Error {}
export class ApiError extends Error {
  constructor(public status: number, message: string) { super(message); }
}

async function call<T>(path: string, init?: { method?: "GET" | "POST"; body?: unknown; admin?: boolean }): Promise<T> {
  const headers: Record<string, string> = { "content-type": "application/json" };
  if (init?.admin && process.env.DOOSRA_ADMIN_TOKEN) headers.authorization = `Bearer ${process.env.DOOSRA_ADMIN_TOKEN}`;
  let res: Response;
  try {
    res = await fetch(`${SERVER_API}${path}`, {
      method: init?.method ?? "GET", headers, cache: "no-store",
      body: init?.body === undefined ? undefined : JSON.stringify(init.body),
    });
  } catch {
    throw new ApiUnreachable(SERVER_API);
  }
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    const d = body?.detail;
    throw new ApiError(res.status, typeof d === "string" ? d : `The API answered ${res.status}.`);
  }
  return (await res.json()) as T;
}

export const fetchFixtures = () => call<FixtureList>("/matches");
export const fetchMatchOdds = (id: string) => call<MatchOdds>(`/matches/${encodeURIComponent(id)}/odds`);
export const fetchBudget = () => call<Budget>("/budget");
export const fetchBets = () => call<BetLedger>("/bets", { admin: true });
export const fetchEngine = () => call<Engine>("/engine");
export const fetchCatalog = () => call<Catalog>("/catalog");
export const fetchCard = (q: MatchQuery) => call<CardResponse>("/markets/card", { method: "POST", body: q });

export async function bridge<T>(path: string, init?: { method?: "GET" | "POST"; body?: unknown }): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BRIDGE}/${path}`, {
      method: init?.method ?? "GET", headers: { "content-type": "application/json" }, cache: "no-store",
      body: init?.body === undefined ? undefined : JSON.stringify(init.body),
    });
  } catch {
    throw new Error("Can't reach Doosra. Check your connection and try again.");
  }
  const data = await res.json().catch(() => null);
  if (!res.ok) {
    const d = data?.detail;
    throw new Error(typeof d === "string" ? d : Array.isArray(d) ? d.map((x) => x.msg).join(". ") : `Doosra answered ${res.status}.`);
  }
  return data as T;
}

/* ---------------------------------------------------------------- formatting */

export const pct = (p: number, digits = 0) => `${(p * 100).toFixed(digits)}%`;
export const signedPct = (p: number) => `${p >= 0 ? "+" : "−"}${Math.abs(p * 100).toFixed(1)}%`;
export const odds = (o: number | null) => (o === null || !isFinite(o) ? "–" : o >= 100 ? "99+" : o.toFixed(2));
export const inr = (n: number) =>
  `${n < 0 ? "−" : ""}₹${Math.abs(n).toLocaleString("en-IN", { minimumFractionDigits: Number.isInteger(n) ? 0 : 2, maximumFractionDigits: 2 })}`;
export const matchTitle = (outcomes: string[]) => outcomes.filter((o) => o !== "Draw").join(" v ");

export function formatStart(iso: string) {
  return new Intl.DateTimeFormat("en-IN", {
    timeZone: "Asia/Kolkata", weekday: "short", day: "numeric", month: "short", hour: "numeric", minute: "2-digit", hour12: true,
  }).format(new Date(iso));
}

export function formatDay(iso: string) {
  return new Intl.DateTimeFormat("en-IN", { day: "numeric", month: "short", year: "numeric" }).format(new Date(iso));
}

export function forecastHref(q: MatchQuery) {
  const p = new URLSearchParams({ a: q.team_a, b: q.team_b, venue: q.venue });
  if (q.batting_first) p.set("first", q.batting_first);
  return `/forecast?${p}`;
}

/** Friendly market label relative to a team (the team is shown separately). */
export function shortMarket(lp: LinePrice) {
  const s = lp.spec;
  switch (s.type) {
    case "team_total": return "Innings runs";
    case "session": return `Runs after ${s.overs} overs`;
    case "wickets_by": return `Wickets by over ${s.overs}`;
    case "first_wicket": return "Score at first wicket";
    case "over_runs": return `Runs in over ${s.over}`;
    case "match_total": return "Match runs";
    case "match_sixes": return "Match sixes";
    case "match_fours": return "Match fours";
    default: return lp.market;
  }
}

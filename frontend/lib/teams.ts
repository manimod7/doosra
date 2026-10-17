/**
 * Team colours: what each side actually wears, made readable in both themes,
 * and kept apart when two sides wear similar colours.
 *
 * - Every team has a main kit colour and a second one from the same kit.
 * - Raw kit colours are adjusted (lightness only, hue kept) until they stand out
 *   against Doosra's light and dark backgrounds. IPL colours are the hand-tuned
 *   set from Chase and are used as they are.
 * - When two teams look alike (India v Sri Lanka, MI v DC), the less iconic one
 *   switches to its second colour, the same rule Chase uses.
 *
 * Names are matched on Cricsheet's spellings, including old franchise names,
 * so historical matches get today's colours. To fix or add a team, edit KITS.
 */

export type Theme = "light" | "dark";
type Hex = `#${string}`;
type Kit = { main: Hex; second: Hex; names: string[]; iconic?: boolean; tuned?: [Hex, Hex, Hex, Hex] };

// Light theme background and dark theme surface: the colours a team colour must stand out from.
export const BACKDROP: Record<Theme, Hex[]> = { light: ["#ffffff", "#f3f5f2"], dark: ["#131815", "#090c0a"] };
// Fills (bars, dots, chart lines) need 3:1 in dark; in light, yellows and golds may sit at 2.2:1 so they
// stay gold rather than turning olive. Text is never drawn in a team colour.
export const MIN_CONTRAST: Record<Theme, number> = { light: 2.2, dark: 3 };
export const DEFAULT_PAIR: Record<Theme, [Hex, Hex]> = { light: ["#2a78d6", "#eb6834"], dark: ["#3987e5", "#d95926"] };
// Below this OKLab distance two colours read as "the same colour" on a small dot or a thin line.
export const CLASH = 0.13;

const IPL = {
  csk: ["#d99a00", "#ffcb05", "#0b7bd1", "#4fb0ff"],
  mi: ["#005db4", "#4a9bea", "#c99a2e", "#e8c060"],
  rcb: ["#d3161c", "#ff5a52", "#b8892b", "#e5be5c"],
  kkr: ["#4b2a83", "#a27be0", "#b8962e", "#e3c467"],
  dc: ["#1a56b0", "#5c97e6", "#e3202b", "#ff6b6f"],
  pbks: ["#d71920", "#ff5656", "#7c828a", "#c4c9cf"],
  rr: ["#e4007c", "#ff5cad", "#2a4ba8", "#6f8fea"],
  srh: ["#f26522", "#ff8a3d", "#2b2b2b", "#e8e1d6"],
  gt: ["#1c2b5e", "#6e8ae8", "#b58a33", "#e1b85a"],
  lsg: ["#0a9aae", "#2fc8dc", "#e8731a", "#ff9a4d"],
} as const satisfies Record<string, [Hex, Hex, Hex, Hex]>;

/** Franchise families: sister teams abroad wear the parent's colours. */
const fam = (k: keyof typeof IPL, names: string[], iconic = false): Kit =>
  ({ main: IPL[k][0], second: IPL[k][2], tuned: IPL[k], names, iconic });

export const KITS: Record<string, Kit> = {
  // ---- International (men's and women's sides share a name in Cricsheet)
  india: { main: "#1d5bbf", second: "#f58220", names: ["India"], iconic: true },
  australia: { main: "#ffcd00", second: "#00843d", names: ["Australia"], iconic: true },
  england: { main: "#1b2a5c", second: "#d0202e", names: ["England"] },
  south_africa: { main: "#007749", second: "#ffb81c", names: ["South Africa"], iconic: true },
  new_zealand: { main: "#111111", second: "#2fb5aa", names: ["New Zealand"], iconic: true },
  pakistan: { main: "#0b7a3e", second: "#a8d84e", names: ["Pakistan"], iconic: true },
  sri_lanka: { main: "#1c3f94", second: "#ffbe29", names: ["Sri Lanka"] },
  west_indies: { main: "#7a0019", second: "#f9b51c", names: ["West Indies"], iconic: true },
  bangladesh: { main: "#006a4e", second: "#f42a41", names: ["Bangladesh"] },
  afghanistan: { main: "#0b4ea2", second: "#d32011", names: ["Afghanistan"] },
  ireland: { main: "#009a44", second: "#1d3c78", names: ["Ireland"] },
  zimbabwe: { main: "#d21034", second: "#ffd200", names: ["Zimbabwe"] },
  netherlands: { main: "#ff6f00", second: "#21468b", names: ["Netherlands"], iconic: true },
  scotland: { main: "#5c2d91", second: "#0065bd", names: ["Scotland"] },
  nepal: { main: "#003893", second: "#dc143c", names: ["Nepal"] },
  usa: { main: "#0a3161", second: "#b31942", names: ["United States of America", "USA"] },
  uae: { main: "#c8102e", second: "#00843d", names: ["United Arab Emirates", "UAE"] },
  oman: { main: "#db161b", second: "#008000", names: ["Oman"] },
  namibia: { main: "#003580", second: "#d21034", names: ["Namibia"] },
  canada: { main: "#d80621", second: "#1a1a1a", names: ["Canada"] },
  png: { main: "#ce1126", second: "#fcd116", names: ["Papua New Guinea"] },
  uganda: { main: "#fcdc04", second: "#d90000", names: ["Uganda"] },
  hong_kong: { main: "#de2910", second: "#1a1a1a", names: ["Hong Kong"] },
  kenya: { main: "#006600", second: "#bb0000", names: ["Kenya"] },

  // ---- IPL and WPL (Chase's hand-tuned colours), with old names and overseas sister teams
  csk: fam("csk", ["Chennai Super Kings", "Joburg Super Kings", "Texas Super Kings"], true),
  mi: fam("mi", ["Mumbai Indians", "MI Cape Town", "MI Emirates", "MI New York", "MI London"], true),
  rcb: fam("rcb", ["Royal Challengers Bengaluru", "Royal Challengers Bangalore"], true),
  kkr: fam("kkr", ["Kolkata Knight Riders", "Abu Dhabi Knight Riders", "Los Angeles Knight Riders"]),
  dc: fam("dc", ["Delhi Capitals", "Delhi Daredevils", "Pretoria Capitals", "Dubai Capitals"]),
  pbks: fam("pbks", ["Punjab Kings", "Kings XI Punjab", "Saint Lucia Kings", "St Lucia Kings"]),
  rr: fam("rr", ["Rajasthan Royals", "Paarl Royals", "Barbados Royals"], true),
  srh: fam("srh", ["Sunrisers Hyderabad", "Sunrisers Eastern Cape", "Sunrisers Leeds"]),
  gt: fam("gt", ["Gujarat Titans"]),
  lsg: fam("lsg", ["Lucknow Super Giants", "Durban's Super Giants", "Durban Super Giants", "Manchester Super Giants"]),
  deccan: { main: "#1f3a68", second: "#c0c6cc", names: ["Deccan Chargers"] },
  gujarat_lions: { main: "#e04f16", second: "#1f2f5c", names: ["Gujarat Lions"] },
  pune: { main: "#6e3e97", second: "#c99a2e", names: ["Rising Pune Supergiant", "Rising Pune Supergiants"] },
  pune_warriors: { main: "#2f9bd5", second: "#1a1a1a", names: ["Pune Warriors"] },
  kochi: { main: "#e2621b", second: "#5a2d82", names: ["Kochi Tuskers Kerala"] },
  gujarat_giants: { main: "#f26f21", second: "#1c2b5e", names: ["Gujarat Giants"] },
  up_warriorz: { main: "#6c2c8f", second: "#ffd100", names: ["UP Warriorz"] },

  // ---- Big Bash League
  adelaide_strikers: { main: "#00a0e3", second: "#0a1f44", names: ["Adelaide Strikers"] },
  brisbane_heat: { main: "#00a5b5", second: "#ff6a13", names: ["Brisbane Heat"] },
  hobart_hurricanes: { main: "#6a2c91", second: "#00a99d", names: ["Hobart Hurricanes"] },
  melbourne_renegades: { main: "#d6001c", second: "#1a1a1a", names: ["Melbourne Renegades"] },
  melbourne_stars: { main: "#00a651", second: "#1a1a1a", names: ["Melbourne Stars"] },
  perth_scorchers: { main: "#f47920", second: "#1a1a1a", names: ["Perth Scorchers"] },
  sydney_sixers: { main: "#e6007e", second: "#1a1a1a", names: ["Sydney Sixers"] },
  sydney_thunder: { main: "#8cc63f", second: "#1a1a1a", names: ["Sydney Thunder"] },

  // ---- Pakistan Super League
  islamabad_united: { main: "#e2231a", second: "#1a1a1a", names: ["Islamabad United"] },
  karachi_kings: { main: "#1157b0", second: "#e6007e", names: ["Karachi Kings"] },
  lahore_qalandars: { main: "#36b54a", second: "#e31e24", names: ["Lahore Qalandars"] },
  multan_sultans: { main: "#00a19a", second: "#f9a01b", names: ["Multan Sultans"] },
  peshawar_zalmi: { main: "#fdb913", second: "#1a1a1a", names: ["Peshawar Zalmi"] },
  quetta_gladiators: { main: "#5c2d91", second: "#ffd200", names: ["Quetta Gladiators"] },

  // ---- Caribbean Premier League (Royals, Kings and Knight Riders sides are in the families above)
  guyana: { main: "#00a651", second: "#f5a800", names: ["Guyana Amazon Warriors"] },
  jamaica: { main: "#009b3a", second: "#fed100", names: ["Jamaica Tallawahs"] },
  trinbago: { main: "#c8102e", second: "#1a1a1a", names: ["Trinbago Knight Riders"] },
  st_kitts: { main: "#d0102e", second: "#009e49", names: ["St Kitts and Nevis Patriots", "St Kitts & Nevis Patriots"] },

  // ---- ILT20, Major League Cricket, The Hundred (others are in the families above)
  desert_vipers: { main: "#d7192d", second: "#1a1a1a", names: ["Desert Vipers"] },
  gulf_giants: { main: "#f26f21", second: "#1c2b5e", names: ["Gulf Giants"] },
  seattle_orcas: { main: "#0a7d72", second: "#0b2a5b", names: ["Seattle Orcas"] },
  washington_freedom: { main: "#1f3a93", second: "#d1202f", names: ["Washington Freedom"] },
  birmingham_phoenix: { main: "#f15a22", second: "#1d1d1b", names: ["Birmingham Phoenix"] },
  london_spirit: { main: "#0d2c6c", second: "#c9a227", names: ["London Spirit"] },
  trent_rockets: { main: "#ffcd00", second: "#d0021b", names: ["Trent Rockets"] },
  welsh_fire: { main: "#d0021b", second: "#1a1a1a", names: ["Welsh Fire"] },
};

/* ------------------------------------------------------------------ name lookup */

const norm = (s: string) => s.toLowerCase().replace(/&/g, "and").replace(/[^a-z0-9]+/g, " ").trim();
const BY_NAME = new Map<string, string>();
for (const [key, kit] of Object.entries(KITS)) for (const n of kit.names) BY_NAME.set(norm(n), key);
// Cricsheet sometimes appends " Women" or uses an "A" side name; fall back to the base side.
const SUFFIX = / (women|woman|w|a|xi|u19|under 19s?)$/;

export function teamKey(name: string | null | undefined): string | null {
  if (!name) return null;
  let n = norm(name);
  for (let i = 0; i < 2; i++) {
    const hit = BY_NAME.get(n);
    if (hit) return hit;
    if (!SUFFIX.test(n)) break;
    n = n.replace(SUFFIX, "");
  }
  return null;
}

/* ------------------------------------------------------------------ colour maths (sRGB, WCAG, OKLab) */

const hexRgb = (h: string) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16) / 255);
const toHex = (rgb: number[]) => `#${rgb.map((v) => Math.round(Math.min(1, Math.max(0, v)) * 255).toString(16).padStart(2, "0")).join("")}` as Hex;
const lin = (c: number) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
const gam = (c: number) => (c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055);

export function contrast(a: string, b: string): number {
  const L = (h: string) => { const [r, g, bl] = hexRgb(h).map(lin); return 0.2126 * r + 0.7152 * g + 0.0722 * bl; };
  const [x, y] = [L(a), L(b)].sort((p, q) => q - p);
  return (x + 0.05) / (y + 0.05);
}

function oklab(h: string): [number, number, number] {
  const [r, g, b] = hexRgb(h).map(lin);
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  return [0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s];
}

function fromOklab([L, A, B]: number[]): number[] {
  const l = (L + 0.3963377774 * A + 0.2158037573 * B) ** 3;
  const m = (L - 0.1055613458 * A - 0.0638541728 * B) ** 3;
  const s = (L - 0.0894841775 * A - 1.291485548 * B) ** 3;
  return [4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s];
}

/** Same hue, new lightness; chroma is reduced only as far as needed to stay a real sRGB colour. */
function withLightness(h: string, L: number): Hex {
  const [, a, b] = oklab(h);
  for (let k = 1; k >= 0; k -= 0.02) {
    const rgb = fromOklab([L, a * k, b * k]);
    if (rgb.every((v) => v >= -1e-4 && v <= 1 + 1e-4)) return toHex(rgb.map(gam));
  }
  return toHex(fromOklab([L, 0, 0]).map(gam));
}

export const distance = (x: string, y: string) => { const p = oklab(x), q = oklab(y); return Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]); };

/** Nudge a kit colour lighter (dark theme) or darker (light theme) until it stands out from the background. */
export function readable(hex: Hex, theme: Theme): Hex {
  const ok = (c: string) => BACKDROP[theme].every((bg) => contrast(c, bg) >= MIN_CONTRAST[theme]);
  if (ok(hex)) return hex;
  const L0 = oklab(hex)[0], step = theme === "dark" ? 0.01 : -0.01;
  for (let L = L0 + step; L > 0 && L < 1; L += step) {
    const c = withLightness(hex, L);
    if (ok(c)) return c;
  }
  return DEFAULT_PAIR[theme][0];
}

/* ------------------------------------------------------------------ pairs */

const cache = new Map<string, Hex>();
function shade(key: string, which: 0 | 1, theme: Theme): Hex {
  const id = `${key}/${which}/${theme}`;
  let c = cache.get(id);
  if (!c) {
    const kit = KITS[key];
    c = kit.tuned ? kit.tuned[which * 2 + (theme === "dark" ? 1 : 0)] : readable(which ? kit.second : kit.main, theme);
    cache.set(id, c);
  }
  return c;
}

/** The colour of one team on its own, or null when Doosra doesn't know the side. */
export function teamColour(name: string, theme: Theme): Hex | null {
  const k = teamKey(name);
  return k ? shade(k, 0, theme) : null;
}

/**
 * Colours for two sides in one theme. Unknown sides get Doosra's default blue or orange,
 * whichever is further from the known side. When the two main colours clash, the less iconic
 * side (or the second-named, if both are equally iconic) switches to its second colour.
 */
export function pairColours(teams: [string, string], theme: Theme): [Hex, Hex] {
  const keys = teams.map(teamKey) as [string | null, string | null];
  const main = keys.map((k) => (k ? shade(k, 0, theme) : null));
  const [d1, d2] = DEFAULT_PAIR[theme];
  const fill = (other: Hex | null): Hex => (other && distance(other, d1) < distance(other, d2) ? d2 : d1);
  if (!main[0] && !main[1]) return [d1, d2];
  if (!main[0]) return [fill(main[1]), main[1]!];
  if (!main[1]) return [main[0], fill(main[0])];
  let [a, b] = main as [Hex, Hex];
  if (distance(a, b) >= CLASH) return [a, b];

  const iconic = keys.map((k) => !!KITS[k!].iconic);
  const order: (0 | 1)[] = iconic[1] && !iconic[0] ? [0, 1] : [1, 0];   // who switches first
  for (const side of order) {
    const alt = shade(keys[side]!, 1, theme);
    const trial: [Hex, Hex] = side === 0 ? [alt, b] : [a, alt];
    if (distance(trial[0], trial[1]) >= CLASH) return trial;
  }
  [a, b] = [shade(keys[0]!, 0, theme), shade(keys[1]!, 1, theme)];
  if (distance(a, b) >= CLASH) return [a, b];
  return [shade(keys[0]!, 1, theme), shade(keys[1]!, 0, theme)];
}

/** CSS custom properties for a match: the stylesheet picks the light or dark pair. */
export function teamVars(teams: [string, string]): Record<string, string> {
  const l = pairColours(teams, "light"), d = pairColours(teams, "dark");
  return { "--ta-l": l[0], "--tb-l": l[1], "--ta-d": d[0], "--tb-d": d[1] };
}

/** "var(--team-a)" or "var(--team-b)" for a team in the current match. */
export const teamVar = (teams: readonly string[], name: string | null | undefined) =>
  teams.indexOf(name ?? "") === 1 ? "var(--team-b)" : "var(--team-a)";

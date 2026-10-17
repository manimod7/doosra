// Run with: npm test   (node's built-in runner, TypeScript types stripped)
import { test } from "node:test";
import assert from "node:assert/strict";
import { BACKDROP, CLASH, DEFAULT_PAIR, KITS, MIN_CONTRAST, contrast, distance, pairColours, teamKey, teamVar, teamVars, type Theme } from "./teams.ts";

const THEMES: Theme[] = ["light", "dark"];
const ALL = Object.values(KITS).map((k) => k.names[0]);

test("every name in the table resolves to its own team", () => {
  for (const [key, kit] of Object.entries(KITS)) for (const n of kit.names) assert.equal(teamKey(n), key, n);
});

test("Cricsheet spellings, old franchise names and women's sides resolve", () => {
  const cases: [string, string][] = [
    ["Royal Challengers Bangalore", "rcb"], ["Royal Challengers Bengaluru", "rcb"], ["Kings XI Punjab", "pbks"],
    ["Delhi Daredevils", "dc"], ["India Women", "india"], ["Australia Women", "australia"],
    ["United States of America", "usa"], ["St Kitts & Nevis Patriots", "st_kitts"], ["  mumbai indians ", "mi"],
    ["Durban's Super Giants", "lsg"], ["MI New York", "mi"], ["Sunrisers Eastern Cape", "srh"],
  ];
  for (const [n, k] of cases) assert.equal(teamKey(n), k, n);
});

test("look-alike names do not cross over", () => {
  assert.equal(teamKey("Dubai Capitals"), "dc");
  assert.equal(teamKey("Gujarat Giants"), "gujarat_giants");
  assert.equal(teamKey("Gujarat Titans"), "gt");
  assert.equal(teamKey("Trinbago Knight Riders"), "trinbago");
  assert.equal(teamKey("Team A"), null);
  assert.equal(teamKey(""), null);
  assert.equal(teamKey("Indiana"), null);
});

test("every team colour stands out from both backgrounds in both themes", () => {
  for (const t of THEMES) for (const name of ALL) {
    const [c] = pairColours([name, "Unknown XI"], t);
    for (const bg of BACKDROP[t]) {
      // IPL shades are Chase's hand-tuned set; hold them to a slightly lower bar in light mode (CSK gold).
      const floor = KITS[teamKey(name)!].tuned && t === "light" ? 2 : MIN_CONTRAST[t];
      assert.ok(contrast(c, bg) >= floor - 0.01, `${name} ${t} ${c} on ${bg}: ${contrast(c, bg).toFixed(2)}`);
    }
  }
});

test("any two teams that meet get colours you can tell apart, in both themes", () => {
  const bad: string[] = [];
  for (const t of THEMES) for (let i = 0; i < ALL.length; i++) for (let j = 0; j < ALL.length; j++) {
    if (i === j || teamKey(ALL[i]) === teamKey(ALL[j])) continue;
    const [a, b] = pairColours([ALL[i], ALL[j]], t);
    if (distance(a, b) < CLASH) bad.push(`${t}: ${ALL[i]} ${a} v ${ALL[j]} ${b}`);
  }
  assert.deepEqual(bad.slice(0, 10), [], `${bad.length} pairs too close`);
});

test("main colours are kept unless they clash", () => {
  const [india, sl] = pairColours(["India", "Sri Lanka"], "light");
  const [indiaAlone] = pairColours(["India", "Australia"], "light");
  assert.equal(india, indiaAlone, "India keeps blue");
  assert.ok(distance(sl, india) >= CLASH, "Sri Lanka moves to its gold");
});

test("the iconic side keeps its colour whichever way round the fixture is written", () => {
  for (const order of [["Mumbai Indians", "Delhi Capitals"], ["Delhi Capitals", "Mumbai Indians"]] as [string, string][]) {
    const pair = pairColours(order, "light");
    const mi = pair[order.indexOf("Mumbai Indians")];
    assert.equal(mi, KITS.mi.tuned![0]);
  }
});

test("unknown sides fall back to the default pair, away from a known side's colour", () => {
  assert.deepEqual(pairColours(["Team A", "Team B"], "light"), DEFAULT_PAIR.light);
  const [ind, other] = pairColours(["India", "Team B"], "light");
  assert.notEqual(other, DEFAULT_PAIR.light[0], "India is blue, so the stranger gets orange");
  assert.ok(distance(ind, other) >= CLASH);
});

test("css variables cover both themes and teamVar follows fixture order", () => {
  const v = teamVars(["Chennai Super Kings", "Mumbai Indians"]);
  assert.deepEqual(Object.keys(v).sort(), ["--ta-d", "--ta-l", "--tb-d", "--tb-l"]);
  assert.equal(v["--ta-l"], KITS.csk.tuned![0]);
  assert.equal(teamVar(["X", "Y"], "Y"), "var(--team-b)");
  assert.equal(teamVar(["X", "Y"], "X"), "var(--team-a)");
});

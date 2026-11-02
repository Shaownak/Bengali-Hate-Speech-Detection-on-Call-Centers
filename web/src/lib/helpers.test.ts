import { describe, expect, it } from "vitest";

import { callApply, emptyCall, lane } from "@/lib/call-live";
import { parseCsv, toCsv } from "@/lib/csv";
import { clock, maskTerm, median } from "@/lib/format";
import { rowToLexicon } from "@/lib/lexicon";
import { keyAction } from "@/lib/review-keys";
import { checkRules, leaves, setLeaf } from "@/lib/rules-yaml";
import { bandOf } from "@/lib/severity";

describe("format", () => {
  it("clock, mask, median", () => {
    expect(clock(65)).toBe("1:05");
    expect(clock(3725)).toBe("1:02:05");
    expect(clock(undefined)).toBe("—");
    expect(maskTerm("<GALI_1>")).toBe("<•••");
    expect(maskTerm("ক")).toBe("•••");
    expect(median([3, 1, 2])).toBe(2);
    expect(median([4, 1, 2, 3])).toBe(2.5);
    expect(median([])).toBeUndefined();
    expect(bandOf(0.9)).toBe(3);
    expect(bandOf(0)).toBeUndefined();
  });
});

describe("csv", () => {
  it("escapes, neutralises formulas and round-trips Bangla", () => {
    const out = toCsv([{ a: 'x,"y"', b: "=SUM(A1)", c: "বাংলা" }]);
    expect(out.startsWith("﻿")).toBe(true);
    expect(out).toContain('"x,""y"""');
    expect(out).toContain("'=SUM(A1)");
    expect(parseCsv(out)).toEqual([{ a: 'x,"y"', b: "'=SUM(A1)", c: "বাংলা" }]);
  });

  it("lexicon rows → API bodies, rejecting bad rows", () => {
    const [row] = parseCsv("term,category,severity,variants,context_sensitive\r\n<GALI_2>,insult,2,\"a, b\",yes\r\n");
    expect(rowToLexicon(row!)).toMatchObject({
      term: "<GALI_2>",
      category: "insult",
      severity: 2,
      variants: ["a", "b"],
      context_sensitive: true,
      active: true,
    });
    expect(() => rowToLexicon({ term: "x", category: "nope", severity: "2" })).toThrow();
    expect(() => rowToLexicon({ term: "x", category: "insult", severity: "9" })).toThrow();
  });
});

describe("review keys", () => {
  it("maps the docs/04 §2.4 shortcuts and ignores typing in fields", () => {
    expect(keyAction("j")).toEqual({ type: "next" });
    expect(keyAction("K")).toEqual({ type: "prev" });
    expect(keyAction("1")).toEqual({ type: "verdict", verdict: "correct" });
    expect(keyAction("3")).toEqual({ type: "verdict", verdict: "false_positive" });
    expect(keyAction("2")).toEqual({ type: "pick", field: "category" });
    expect(keyAction("4")).toEqual({ type: "pick", field: "severity" });
    expect(keyAction(" ")).toEqual({ type: "play" });
    expect(keyAction("x")).toBeUndefined();
    expect(keyAction("1", document.createElement("textarea"))).toBeUndefined();
  });
});

describe("rules yaml", () => {
  const y = "version: 2\nrules:\n  - name: lex\n    when: {source: lexicon, score_gte: 0.7}\n";
  it("validates shape and edits scalars keeping types", () => {
    const c = checkRules(y);
    expect(c.ok).toBe(true);
    if (!c.ok) return;
    expect(leaves(c.doc.rules[0]!)).toEqual([
      ["name", "lex"],
      ["when.source", "lexicon"],
      ["when.score_gte", 0.7],
    ]);
    const next = checkRules(setLeaf(c.doc, "lex", "when.score_gte", "0.9"));
    expect(next.ok && next.doc.rules[0]!.when).toEqual({ source: "lexicon", score_gte: 0.9 });
    expect(checkRules("version: x\nrules: []").ok).toBe(false);
    expect(checkRules("version: 1\nrules: [{name: a}, {name: a}]").ok).toBe(false);
    expect(checkRules(": :").ok).toBe(false);
  });
});

describe("live call", () => {
  it("partials are replaced by finals and never regress; signals feed timelines", () => {
    let c = callApply(emptyCall(), [
      { stream: "asr.segments", data: { segment_id: "s1", channel: "agent", ts_start: 1, ts_end: 2, text_redacted: "আমি", is_final: false } },
    ]);
    expect(lane(c, "agent")[0]).toMatchObject({ final: false, text: "আমি" });
    c = callApply(c, [
      { stream: "asr.segments", data: { segment_id: "s1", channel: "agent", ts_start: 1, ts_end: 3, text_redacted: "আমি বলছি", is_final: true } },
      { stream: "asr.segments", data: { segment_id: "s1", channel: "agent", ts_start: 1, ts_end: 2, text_redacted: "আমি", is_final: false } },
      { stream: "signals", data: { channel: "customer", source: "dsp_shout", ts_start: 5, score: 0.8 } },
      { stream: "signals", data: { channel: "customer", source: "ser", ts_start: 5, score: 0.6 } },
      { stream: "signals", data: { channel: "agent", source: "register", ts_start: 2, score: 0.9 } },
      { stream: "call.lifecycle", data: { call_id: "c", event: "ended", ts: 9 } },
    ]);
    expect(lane(c, "agent")[0]).toMatchObject({ final: true, text: "আমি বলছি" });
    expect(c.loudness.customer).toEqual([[5, 0.8]]);
    expect(c.anger.customer).toEqual([[5, 0.6]]);
    expect(c.markers).toEqual([{ t: 2, channel: "agent", kind: "register" }]);
    expect(c.status).toBe("ended");
  });
});

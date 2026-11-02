import { parse, stringify } from "yaml";

export interface RuleDoc {
  version: number;
  rules: Array<Record<string, unknown> & { name: string }>;
}

export type Check = { ok: true; doc: RuleDoc } | { ok: false; error: string };

/** Client-side shape check before the server's full validation (docs/04 §2.8). */
export function checkRules(text: string): Check {
  let doc: unknown;
  try {
    doc = parse(text);
  } catch (e) {
    return { ok: false, error: e instanceof Error ? e.message : String(e) };
  }
  const d = doc as Partial<RuleDoc> | null;
  if (!d || typeof d.version !== "number") return { ok: false, error: "`version` must be a number" };
  if (!Array.isArray(d.rules)) return { ok: false, error: "`rules` must be a list" };
  const names = new Set<string>();
  for (const r of d.rules) {
    if (!r || typeof r.name !== "string") return { ok: false, error: "every rule needs a `name`" };
    if (names.has(r.name)) return { ok: false, error: `duplicate rule \`${r.name}\`` };
    names.add(r.name);
  }
  return { ok: true, doc: d as RuleDoc };
}

/** Scalar leaves of a rule (path → value) for the form view. */
export function leaves(obj: Record<string, unknown>, prefix = ""): Array<[string, string | number | boolean]> {
  const out: Array<[string, string | number | boolean]> = [];
  for (const [k, v] of Object.entries(obj)) {
    const path = prefix ? `${prefix}.${k}` : k;
    if (v && typeof v === "object" && !Array.isArray(v)) out.push(...leaves(v as Record<string, unknown>, path));
    else if (typeof v === "number" || typeof v === "string" || typeof v === "boolean") out.push([path, v]);
  }
  return out;
}

/** Set a scalar by dotted path, keeping the value's type; returns new YAML. */
export function setLeaf(doc: RuleDoc, rule: string, path: string, raw: string): string {
  const copy = structuredClone(doc);
  const target = copy.rules.find((r) => r.name === rule);
  if (!target) return stringify(copy);
  const keys = path.split(".");
  let cur = target as Record<string, unknown>;
  for (const k of keys.slice(0, -1)) cur = cur[k] as Record<string, unknown>;
  const last = keys[keys.length - 1] as string;
  const old = cur[last];
  cur[last] =
    typeof old === "number" ? Number(raw) : typeof old === "boolean" ? raw === "true" : raw;
  return stringify(copy);
}

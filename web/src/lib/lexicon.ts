import type { Schemas } from "@/lib/api";
import { CATEGORIES } from "@/lib/categories";

export type LexIn = Schemas["LexiconIn"];

const split = (s: string) =>
  s
    .split(",")
    .map((x) => x.trim())
    .filter(Boolean);

/** CSV row → API body (columns per lexicon.csvHelp). Exported for tests. */
export function rowToLexicon(r: Record<string, string>): LexIn {
  const sev = Number(r.severity);
  if (!r.term || !CATEGORIES.includes(r.category as never) || ![1, 2, 3].includes(sev)) {
    throw new Error("invalid row");
  }
  return {
    term: r.term,
    category: r.category as LexIn["category"],
    severity: sev as LexIn["severity"],
    variants: split(r.variants ?? ""),
    romanized: split(r.romanized ?? ""),
    dialect: r.dialect || null,
    context_sensitive: ["1", "true", "yes"].includes((r.context_sensitive ?? "").toLowerCase()),
    active: true,
  };
}

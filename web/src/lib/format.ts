/** mm:ss (or h:mm:ss) for a duration in seconds. */
export function clock(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds)) return "—";
  const s = Math.max(0, Math.floor(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const ss = String(s % 60).padStart(2, "0");
  return h ? `${h}:${String(m).padStart(2, "0")}:${ss}` : `${m}:${ss}`;
}

export function pct(x: number | null | undefined, digits = 0): string {
  return x == null ? "—" : `${(x * 100).toFixed(digits)}%`;
}

/** Lexicon terms are masked by default (docs/04 §2.7): first character + •••. */
export function maskTerm(term: string): string {
  const chars = Array.from(term);
  return chars.length <= 1 ? "•••" : `${chars[0]}•••`;
}

export function dateTime(iso: string | null | undefined, locale: string): string {
  if (!iso) return "—";
  return new Intl.DateTimeFormat(locale === "bn" ? "bn-BD" : "en-GB", {
    dateStyle: "short",
    timeStyle: "medium",
  }).format(new Date(iso));
}

export function median(xs: number[]): number | undefined {
  if (!xs.length) return undefined;
  const s = [...xs].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  const hi = s[mid] ?? 0;
  return s.length % 2 ? hi : ((s[mid - 1] ?? hi) + hi) / 2;
}

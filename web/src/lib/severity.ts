import { Info, type LucideIcon, OctagonAlert, TriangleAlert } from "lucide-react";

export type Severity = 1 | 2 | 3;

/** docs/04 §4 + WCAG: severity is shown as icon + text + colour, never colour alone. */
export const SEVERITY: Record<Severity, { icon: LucideIcon; className: string; bar: string }> = {
  1: {
    icon: Info,
    className: "bg-sky-100 text-sky-950 border-sky-400 dark:bg-sky-950 dark:text-sky-100",
    bar: "bg-sky-600",
  },
  2: {
    icon: TriangleAlert,
    className:
      "bg-amber-100 text-amber-950 border-amber-500 dark:bg-amber-950 dark:text-amber-100",
    bar: "bg-amber-500",
  },
  3: {
    icon: OctagonAlert,
    className: "bg-red-100 text-red-950 border-red-500 dark:bg-red-950 dark:text-red-100",
    bar: "bg-red-600",
  },
};

export function asSeverity(n: number | null | undefined): Severity | undefined {
  return n === 1 || n === 2 || n === 3 ? n : undefined;
}

/** Probability → severity band for risk meters (display only, not an alert threshold). */
export function bandOf(p: number): Severity | undefined {
  if (p >= 0.85) return 3;
  if (p >= 0.6) return 2;
  if (p > 0) return 1;
  return undefined;
}

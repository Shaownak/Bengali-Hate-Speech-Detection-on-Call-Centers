import { useTranslations } from "next-intl";

import type { Schemas } from "@/lib/api";

/** 7 × 24 grid; every cell carries its number as text too (colour is not the only cue). */
export function Heatmap({
  cells,
  metric,
}: {
  cells: Schemas["HeatCell"][];
  metric: "agent_incidents" | "avg_customer_hostility";
}) {
  const t = useTranslations("analytics");
  const days = t("weekdays").split(",");
  const value = (c?: Schemas["HeatCell"]) => (c ? (c[metric] ?? 0) : 0);
  const max = Math.max(1, ...cells.map((c) => value(c)));
  const at = new Map(cells.map((c) => [`${c.weekday}-${c.hour}`, c]));
  return (
    <div className="overflow-x-auto">
      <table className="border-separate border-spacing-0.5 text-[10px]" data-testid="heatmap">
        <thead>
          <tr>
            <th />
            {Array.from({ length: 24 }, (_, h) => (
              <th key={h} className="w-6 font-normal opacity-70">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {days.map((d, w) => (
            <tr key={d}>
              <th className="pr-1 text-right font-normal">{d}</th>
              {Array.from({ length: 24 }, (_, h) => {
                const c = at.get(`${w}-${h}`);
                const v = value(c);
                return (
                  <td
                    key={h}
                    className="h-6 w-6 rounded text-center tabular-nums"
                    style={{ background: v ? `rgba(220, 38, 38, ${0.15 + (0.75 * v) / max})` : undefined }}
                    title={c ? `${d} ${h}:00 · ${v} · ${c.calls}` : undefined}
                  >
                    {v ? (Number.isInteger(v) ? v : v.toFixed(1)) : ""}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

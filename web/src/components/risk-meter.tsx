import { useTranslations } from "next-intl";

import { CategoryLabel } from "@/components/badges";
import type { RiskPoint } from "@/lib/live-store";
import { bandOf, SEVERITY } from "@/lib/severity";
import { cn } from "@/lib/utils";

/** Rolling max fused probability over the last 60 s, as bar + percentage + band icon/text. */
export function RiskMeter({ label, point }: { label: string; point?: RiskPoint }) {
  const t = useTranslations("severity");
  const p = point?.p ?? 0;
  const band = bandOf(p);
  const Icon = band ? SEVERITY[band].icon : undefined;
  return (
    <div className="flex min-w-36 flex-col gap-0.5" data-risk={band ?? 0}>
      <div className="flex items-center justify-between text-xs">
        <span>{label}</span>
        <span className="inline-flex items-center gap-1 tabular-nums">
          {Icon && <Icon aria-hidden className="size-3.5" />}
          {Math.round(p * 100)}%{band ? ` · ${t(String(band) as "1" | "2" | "3")}` : ""}
        </span>
      </div>
      <div
        role="meter"
        aria-label={label}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(p * 100)}
        className="h-2 overflow-hidden rounded bg-muted"
      >
        <div
          className={cn("h-full transition-[width]", band ? SEVERITY[band].bar : "bg-transparent")}
          style={{ width: `${Math.round(p * 100)}%` }}
        />
      </div>
      {point && <CategoryLabel category={point.category} className="text-xs opacity-80" />}
    </div>
  );
}

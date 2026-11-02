"use client";

import { useQuery } from "@tanstack/react-query";
import { useTranslations } from "next-intl";

import { AuthGate } from "@/components/auth-gate";
import { Card, CardTitle } from "@/components/ui/card";
import { API_BASE } from "@/lib/api";

/** Grafana panel ids on the GAL-E dashboard (infra/observability, P6). */
const PANELS: ReadonlyArray<{ id: number; title: string }> = [
  { id: 1, title: "Bus lag" },
  { id: 2, title: "GPU utilisation" },
  { id: 3, title: "Stage latency p50/p95" },
  { id: 4, title: "Error rates" },
  { id: 5, title: "Alert volume" },
];

export default function HealthPage() {
  return (
    <AuthGate perm="models.read">
      <Health />
    </AuthGate>
  );
}

function Health() {
  const t = useTranslations("health");
  const grafana = process.env.NEXT_PUBLIC_GRAFANA_URL;
  const dashboard = process.env.NEXT_PUBLIC_GRAFANA_DASHBOARD ?? "gale-overview";
  const ready = useQuery({
    queryKey: ["readyz"],
    queryFn: async () => {
      const r = await fetch(`${API_BASE}/readyz`);
      return (await r.json()) as { status: string; checks: Record<string, boolean> };
    },
    refetchInterval: 10_000,
  });
  return (
    <div className="flex flex-col gap-4">
      <h1 className="text-xl font-semibold">{t("title")}</h1>
      <Card>
        <CardTitle>{t("api")}</CardTitle>
        <ul className="flex gap-4 text-sm" data-testid="readiness">
          {Object.entries(ready.data?.checks ?? {}).map(([k, ok]) => (
            <li key={k}>
              {ok ? "✔" : "✖"} {k}
            </li>
          ))}
        </ul>
      </Card>
      <Card>
        <CardTitle>{t("panels")}</CardTitle>
        {grafana ? (
          <div className="grid gap-4 lg:grid-cols-2">
            {PANELS.map((p) => (
              <iframe
                key={p.id}
                title={p.title}
                className="h-64 w-full rounded border border-border"
                src={`${grafana}/d-solo/${dashboard}?orgId=1&panelId=${p.id}&refresh=30s`}
              />
            ))}
          </div>
        ) : (
          <p className="text-sm opacity-80">{t("grafanaMissing")}</p>
        )}
      </Card>
    </div>
  );
}

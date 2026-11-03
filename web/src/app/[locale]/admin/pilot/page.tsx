"use client";

import { useQuery } from "@tanstack/react-query";
import { useLocale, useTranslations } from "next-intl";
import { useState } from "react";

import { AuthGate } from "@/components/auth-gate";
import { QueryState } from "@/components/query-state";
import { Button } from "@/components/ui/button";
import { Card, CardTitle } from "@/components/ui/card";
import type { Schemas } from "@/lib/api";
import { dateTime, pct } from "@/lib/format";
import { unwrap, useApi } from "@/lib/use-api";

export default function PilotPage() {
  return (
    <AuthGate perm="pilot.read">
      <Pilot />
    </AuthGate>
  );
}

const STATUS_ICON: Record<string, string> = { pass: "✅", fail: "❌", unknown: "❔", manual: "📝" };

function fmt(v: unknown): string {
  if (v == null) return "—";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(3);
  return String(v);
}

function ExitList({ exit }: { exit: Schemas["PilotReportBrief"]["exit"] }) {
  const t = useTranslations("pilot");
  return (
    <ul className="flex flex-col gap-1 text-sm">
      {(exit ?? []).map((x) => {
        const status = String(x.status);
        return (
          <li key={String(x.criterion)} className="flex flex-wrap gap-2" data-status={status}>
            <span aria-label={t(`status_${status}` as "status_pass")}>{STATUS_ICON[status] ?? "?"}</span>
            <span className="font-medium">{String(x.criterion)}</span>
            <span className="opacity-80">
              {fmt(x.value)} ({String(x.target)})
            </span>
          </li>
        );
      })}
    </ul>
  );
}

function Pilot() {
  const t = useTranslations("pilot");
  const locale = useLocale();
  const api = useApi();
  const [week, setWeek] = useState<string | null>(null);
  const status = useQuery({
    queryKey: ["pilot-status"],
    queryFn: async () => unwrap(await api.GET("/api/v1/pilot/status")),
  });
  const reports = useQuery({
    queryKey: ["pilot-reports"],
    queryFn: async () => unwrap(await api.GET("/api/v1/pilot/reports")),
  });
  const html = useQuery({
    queryKey: ["pilot-report-html", week],
    enabled: week != null,
    queryFn: async () => {
      const r = await api.GET("/api/v1/pilot/reports/{week}/html", {
        params: { path: { week: week ?? "" } },
        parseAs: "text",
      });
      return unwrap(r) as unknown as string;
    },
  });
  const s = status.data;

  return (
    <div className="flex flex-col gap-4">
      <h1 className="text-xl font-semibold">{t("title")}</h1>
      <p className="text-sm opacity-80">{t("intro")}</p>
      <QueryState isLoading={status.isLoading} error={status.error}>
        {s && (
          <div className="grid gap-4 md:grid-cols-3">
            <Card>
              <CardTitle>{t("stage")}</CardTitle>
              <p className="text-2xl font-semibold" data-testid="pilot-stage">
                {s.stage}
              </p>
              <p className="text-sm">{t(`stage_${s.stage.replace("-", "")}` as "stage_P1")}</p>
              <p className="text-xs opacity-80">{t("week", { week: s.week })}</p>
            </Card>
            <Card>
              <CardTitle>{t("alerts")}</CardTitle>
              <p className="text-sm">
                {s.alerts_default === "hidden" ? t("alertsHidden") : t("alertsVisible")}
              </p>
              {s.visible_queues.length > 0 && (
                <p className="text-sm">
                  {t("visibleQueues")}: {s.visible_queues.join(", ")}
                </p>
              )}
            </Card>
            <Card>
              <CardTitle>{t("candidates")}</CardTitle>
              {s.candidates.length === 0 ? (
                <p className="text-sm opacity-80">{t("noCandidates")}</p>
              ) : (
                <ul className="text-sm">
                  {s.candidates.map((c) => (
                    <li key={c.component}>
                      <span className="font-medium">{c.component}</span> <span className="font-mono">{c.version}</span> ·{" "}
                      {t(c.role === "shadow" ? "roleShadow" : "roleCanary")} · {c.percent}%
                      {c.queues.length > 0 && ` · ${c.queues.join(", ")}`}
                    </li>
                  ))}
                </ul>
              )}
            </Card>
          </div>
        )}
      </QueryState>

      <h2 className="text-lg font-semibold">{t("reports")}</h2>
      <QueryState
        isLoading={reports.isLoading}
        error={reports.error}
        isEmpty={!reports.data?.length}
        emptyText={t("noReports")}
      >
        <div className="grid gap-4 lg:grid-cols-2">
          {reports.data?.map((r) => {
            const h = r.headline as Record<string, unknown>;
            return (
              <Card key={r.week} data-testid="pilot-report" data-week={r.week}>
                <CardTitle className="flex items-center justify-between gap-2">
                  <span>
                    {r.week} · {r.stage}
                  </span>
                  <span className="text-xs font-normal opacity-70">{dateTime(r.created_at, locale)}</span>
                </CardTitle>
                <dl className="mb-2 grid grid-cols-2 gap-x-4 text-sm sm:grid-cols-4">
                  <dt className="opacity-70">{t("calls")}</dt>
                  <dd>{fmt(h.calls)}</dd>
                  <dt className="opacity-70">{t("alertsN")}</dt>
                  <dd>{fmt(h.alerts)}</dd>
                  <dt className="opacity-70">{t("precision")}</dt>
                  <dd>{typeof h.precision === "number" ? pct(h.precision) : "—"}</dd>
                  <dt className="opacity-70">{t("acceptance")}</dt>
                  <dd>{typeof h.alert_acceptance === "number" ? pct(h.alert_acceptance) : "—"}</dd>
                  <dt className="opacity-70">{t("usefulness")}</dt>
                  <dd>{fmt(h.usefulness)}</dd>
                  <dt className="opacity-70">{t("fairnessFlags")}</dt>
                  <dd>{fmt(h.fairness_red_flags)}</dd>
                  <dt className="opacity-70">{t("missed")}</dt>
                  <dd>{fmt(h.missed_events)}</dd>
                </dl>
                <h3 className="text-sm font-medium">{t("exit")}</h3>
                <ExitList exit={r.exit} />
                {r.has_html && (
                  <Button size="sm" variant="outline" className="mt-2 self-start" onClick={() => setWeek(r.week)}>
                    {t("open")}
                  </Button>
                )}
              </Card>
            );
          })}
        </div>
      </QueryState>

      {week && (
        <Card>
          <CardTitle className="flex items-center justify-between">
            {t("fullReport", { week })}
            <Button size="sm" variant="ghost" onClick={() => setWeek(null)}>
              {t("close")}
            </Button>
          </CardTitle>
          <QueryState isLoading={html.isLoading} error={html.error}>
            {/* the API sends a script-free, self-contained page; the sandbox keeps it inert */}
            <iframe title={t("fullReport", { week })} sandbox="" srcDoc={html.data ?? ""} className="h-[70vh] w-full rounded border" />
          </QueryState>
        </Card>
      )}
    </div>
  );
}

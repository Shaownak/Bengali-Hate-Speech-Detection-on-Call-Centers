"use client";

import { useQuery } from "@tanstack/react-query";
import { useTranslations } from "next-intl";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { CategoryLabel, SeverityBadge } from "@/components/badges";
import { QueryState } from "@/components/query-state";
import { Card, CardTitle } from "@/components/ui/card";
import { Link } from "@/i18n/navigation";
import { clock } from "@/lib/format";
import { unwrap, useApi } from "@/lib/use-api";

/** Agent scorecard (docs/04 §2.5, also the agent's own view §2.6). */
export function AgentScorecard({ agentId }: { agentId: string }) {
  const t = useTranslations("analytics");
  const tcat = useTranslations("categories");
  const api = useApi();
  const q = useQuery({
    queryKey: ["agent-analytics", agentId],
    queryFn: async () =>
      unwrap(
        await api.GET("/api/v1/analytics/agents/{agent_id}", {
          params: { path: { agent_id: agentId } },
        }),
      ),
  });
  const a = q.data;
  const byCat = (a?.events_by_category ?? [])
    .filter((c) => c.speaker === "agent")
    .map((c) => ({
      name: tcat.has(c.category) ? tcat(c.category) : c.category,
      per100: a?.calls ? Math.round((1000 * c.count) / a.calls) / 10 : 0,
    }));
  return (
    <QueryState isLoading={q.isLoading} error={q.error} onRetry={() => void q.refetch()}>
      {a && (
        <div className="flex flex-col gap-4" data-testid="agent-scorecard">
          <dl className="grid grid-cols-2 gap-3 sm:grid-cols-6">
            {[
              [t("calls"), a.calls],
              [t("talkMinutes"), a.talk_minutes],
              [t("per100"), a.alerts_per_100_calls ?? "—"],
              [t("teamMedian"), a.team_median_alerts_per_100_calls ?? "—"],
              [t("avgProfessionalism"), a.avg_professionalism ?? "—"],
              [t("disputes"), a.disputes],
            ].map(([k, v]) => (
              <div key={String(k)} className="rounded-lg border border-border p-3">
                <dt className="text-xs opacity-70">{k}</dt>
                <dd className="text-xl font-semibold tabular-nums">{v}</dd>
              </div>
            ))}
          </dl>
          <div className="grid gap-4 lg:grid-cols-2">
            <Card>
              <CardTitle>{t("byCategory")}</CardTitle>
              <figure aria-label={t("byCategory")} className="h-56">
                <ResponsiveContainer>
                  <BarChart data={byCat} layout="vertical" margin={{ left: 40 }}>
                    <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                    <XAxis type="number" fontSize={11} />
                    <YAxis type="category" dataKey="name" width={120} fontSize={11} />
                    <Tooltip />
                    <Bar dataKey="per100" name={t("per100")} fill="#dc2626" />
                  </BarChart>
                </ResponsiveContainer>
              </figure>
            </Card>
            <Card>
              <CardTitle>{t("trend")}</CardTitle>
              <figure aria-label={t("trend")} className="h-56">
                <ResponsiveContainer>
                  <LineChart data={a.trend}>
                    <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                    <XAxis dataKey="day" fontSize={11} />
                    <YAxis allowDecimals={false} fontSize={11} />
                    <Tooltip />
                    <Line dataKey="calls" name={t("calls")} stroke="#2563eb" />
                    <Line dataKey="alerts" name={t("alerts")} stroke="#dc2626" />
                  </LineChart>
                </ResponsiveContainer>
              </figure>
            </Card>
          </div>
          <Card>
            <CardTitle>{t("coachingMoments")}</CardTitle>
            <ul className="flex flex-col gap-1 text-sm">
              {a.coaching_moments?.map((m) => (
                <li key={m.event_id} className="flex flex-wrap items-center gap-2">
                  <Link href={`/calls/${m.call_id}`} className="tabular-nums underline">
                    {clock(m.ts_start)}
                  </Link>
                  <CategoryLabel category={m.category} />
                  <SeverityBadge level={m.severity} />
                  <span lang="bn">{m.explanation_bn}</span>
                </li>
              ))}
            </ul>
            {(a.coaching_points_bn?.length ?? 0) > 0 && (
              <ul className="mt-2 list-disc pl-5 text-sm" lang="bn">
                {a.coaching_points_bn?.map((p, i) => (
                  <li key={i}>{p}</li>
                ))}
              </ul>
            )}
          </Card>
        </div>
      )}
    </QueryState>
  );
}

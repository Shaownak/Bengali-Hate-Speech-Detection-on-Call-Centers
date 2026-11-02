"use client";

import { useQuery } from "@tanstack/react-query";
import { useTranslations } from "next-intl";
import { useState } from "react";

import { AgentScorecard } from "@/components/agent-scorecard";
import { AuthGate } from "@/components/auth-gate";
import { Heatmap } from "@/components/heatmap";
import { QueryState } from "@/components/query-state";
import { Button } from "@/components/ui/button";
import { Card, CardTitle } from "@/components/ui/card";
import { Label, Select } from "@/components/ui/field";
import { useAuth } from "@/lib/auth";
import { useDirectory } from "@/lib/hooks";
import { unwrap, useApi } from "@/lib/use-api";
import { cn } from "@/lib/utils";

type Tab = "agent" | "team" | "queues" | "fairness";

export default function AnalyticsPage() {
  return (
    <AuthGate perm="analytics.read">
      <Analytics />
    </AuthGate>
  );
}

function Analytics() {
  const t = useTranslations("analytics");
  const me = useAuth((s) => s.me);
  const dir = useDirectory();
  const isAgent = me?.role === "agent";
  const tabs: Tab[] = isAgent
    ? ["agent"]
    : ["agent", "team", "queues", ...(me?.role === "admin" || me?.role === "qa_analyst" ? (["fairness"] as Tab[]) : [])];
  const [tab, setTab] = useState<Tab>("agent");
  const [agentId, setAgentId] = useState<string>("");
  const [teamId, setTeamId] = useState<string>("");
  const agent = agentId || (isAgent ? (me?.agent_id ?? "") : (dir.data?.agents[0]?.id ?? ""));
  const team = teamId || (dir.data?.teams[0]?.id ?? "");

  return (
    <div className="flex flex-col gap-4">
      <h1 className="text-xl font-semibold">{t("title")}</h1>
      <div role="tablist" className="flex gap-1">
        {tabs.map((k) => (
          <Button
            key={k}
            role="tab"
            aria-selected={tab === k}
            variant={tab === k ? "default" : "ghost"}
            size="sm"
            onClick={() => setTab(k)}
          >
            {t(k === "agent" ? "agent" : k === "team" ? "team" : k === "queues" ? "queues" : "fairness")}
          </Button>
        ))}
      </div>
      {tab === "agent" && (
        <>
          {!isAgent && (
            <Label className="w-48">
              {t("pickAgent")}
              <Select value={agent} onChange={(e) => setAgentId(e.target.value)}>
                {(dir.data?.agents ?? []).map((a) => (
                  <option key={a.id} value={a.id}>
                    {a.employee_code}
                  </option>
                ))}
              </Select>
            </Label>
          )}
          {agent && <AgentScorecard agentId={agent} />}
        </>
      )}
      {tab === "team" && (
        <>
          <Label className="w-48">
            {t("pickTeam")}
            <Select value={team} onChange={(e) => setTeamId(e.target.value)}>
              {(dir.data?.teams ?? []).map((tm) => (
                <option key={tm.id} value={tm.id}>
                  {tm.name}
                </option>
              ))}
            </Select>
          </Label>
          {team && <TeamView teamId={team} />}
        </>
      )}
      {tab === "queues" && <Queues />}
      {tab === "fairness" && (
        <Card>
          <CardTitle>{t("fairness")}</CardTitle>
          <p className="text-sm" data-testid="fairness-pending">
            {t("fairnessPending")}
          </p>
        </Card>
      )}
    </div>
  );
}

function TeamView({ teamId }: { teamId: string }) {
  const t = useTranslations("analytics");
  const api = useApi();
  const [metric, setMetric] = useState<"agent_incidents" | "avg_customer_hostility">("agent_incidents");
  const q = useQuery({
    queryKey: ["team-analytics", teamId],
    queryFn: async () =>
      unwrap(
        await api.GET("/api/v1/analytics/teams/{team_id}", { params: { path: { team_id: teamId } } }),
      ),
  });
  const d = q.data;
  return (
    <QueryState isLoading={q.isLoading} error={q.error} onRetry={() => void q.refetch()}>
      {d && (
        <div className="grid gap-4 lg:grid-cols-2">
          <Card>
            <CardTitle>
              {d.name} · {d.calls} {t("calls")} · {d.alerts} {t("alerts")}
            </CardTitle>
            <table className="w-full text-sm">
              <thead className="text-left text-xs opacity-70">
                <tr>
                  <th className="p-1">{t("pickAgent")}</th>
                  <th className="p-1">{t("calls")}</th>
                  <th className="p-1">{t("alerts")}</th>
                  <th className="p-1">{t("avgProfessionalism")}</th>
                </tr>
              </thead>
              <tbody>
                {d.agents.map((a) => (
                  <tr key={a.agent_id} className="border-t border-border">
                    <td className="p-1">{a.employee_code}</td>
                    <td className="p-1 tabular-nums">{a.calls}</td>
                    <td className="p-1 tabular-nums">{a.alerts}</td>
                    <td className="p-1 tabular-nums">{a.avg_professionalism ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
          <Card>
            <CardTitle className="flex items-center gap-2">
              {t("heatmap")}
              <span className="ml-auto flex gap-1">
                {(["agent_incidents", "avg_customer_hostility"] as const).map((m) => (
                  <button
                    key={m}
                    type="button"
                    aria-pressed={metric === m}
                    onClick={() => setMetric(m)}
                    className={cn("rounded px-2 py-0.5 text-xs", metric === m && "bg-muted font-medium")}
                  >
                    {m === "agent_incidents" ? t("heatIncidents") : t("heatHostility")}
                  </button>
                ))}
              </span>
            </CardTitle>
            <Heatmap cells={d.heatmap ?? []} metric={metric} />
          </Card>
        </div>
      )}
    </QueryState>
  );
}

function Queues() {
  const t = useTranslations("analytics");
  const tc = useTranslations("common");
  const api = useApi();
  const q = useQuery({
    queryKey: ["queue-analytics"],
    queryFn: async () => unwrap(await api.GET("/api/v1/analytics/queues")),
  });
  return (
    <QueryState isLoading={q.isLoading} error={q.error} isEmpty={!q.data?.length}>
      <Card>
        <CardTitle>{t("queues")}</CardTitle>
        <table className="w-full text-sm" data-testid="queue-table">
          <thead className="text-left text-xs opacity-70">
            <tr>
              <th className="p-1">{tc("queue")}</th>
              <th className="p-1">{t("calls")}</th>
              <th className="p-1">{t("agentAlerts")}</th>
              <th className="p-1">{t("customerAlerts")}</th>
              <th className="p-1">{t("avgHostility")}</th>
            </tr>
          </thead>
          <tbody>
            {q.data?.map((r) => (
              <tr key={r.queue} className="border-t border-border">
                <td className="p-1">{r.queue}</td>
                <td className="p-1 tabular-nums">{r.calls}</td>
                <td className="p-1 tabular-nums">{r.agent_alerts}</td>
                <td className="p-1 tabular-nums">{r.customer_alerts}</td>
                <td className="p-1 tabular-nums">{r.avg_customer_hostility ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </QueryState>
  );
}

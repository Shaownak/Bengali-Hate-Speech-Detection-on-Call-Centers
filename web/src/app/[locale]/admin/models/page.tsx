"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useLocale, useTranslations } from "next-intl";

import { AuthGate } from "@/components/auth-gate";
import { QueryState } from "@/components/query-state";
import { Button } from "@/components/ui/button";
import { Card, CardTitle } from "@/components/ui/card";
import type { Schemas } from "@/lib/api";
import { can, useAuth } from "@/lib/auth";
import { dateTime } from "@/lib/format";
import { ApiError, unwrap, useApi } from "@/lib/use-api";

export default function ModelsPage() {
  return (
    <AuthGate perm="models.read">
      <Models />
    </AuthGate>
  );
}

function Models() {
  const t = useTranslations("models");
  const locale = useLocale();
  const me = useAuth((s) => s.me);
  const admin = can(me, "models.promote");
  const api = useApi();
  const qc = useQueryClient();
  const models = useQuery({
    queryKey: ["models"],
    queryFn: async () => unwrap(await api.GET("/api/v1/models")),
  });
  const pending = useQuery({
    queryKey: ["promotions"],
    queryFn: async () =>
      unwrap(await api.GET("/api/v1/models/promotions", { params: { query: { status: "pending" } } })),
  });
  const refresh = () => {
    void qc.invalidateQueries({ queryKey: ["models"] });
    void qc.invalidateQueries({ queryKey: ["promotions"] });
  };
  const promote = useMutation({
    mutationFn: async (m: Schemas["ModelVersionOut"]) =>
      unwrap(
        await api.POST("/api/v1/models/{component}/promote", {
          params: { path: { component: m.component } },
          body: { model_version_id: m.id },
        }),
      ),
    onSuccess: refresh,
  });
  const decide = useMutation({
    mutationFn: async ({ id, approve }: { id: string; approve: boolean }) => {
      const path = { params: { path: { promotion_id: id } } };
      return unwrap(
        approve
          ? await api.POST("/api/v1/models/promotions/{promotion_id}/approve", path)
          : await api.POST("/api/v1/models/promotions/{promotion_id}/reject", path),
      );
    },
    onSuccess: refresh,
  });
  const byComponent = new Map<string, Schemas["ModelVersionOut"][]>();
  for (const m of models.data ?? []) byComponent.set(m.component, [...(byComponent.get(m.component) ?? []), m]);
  const version = new Map((models.data ?? []).map((m) => [m.id, m]));
  const err = promote.error ?? decide.error;

  return (
    <div className="flex flex-col gap-4">
      <h1 className="text-xl font-semibold">{t("title")}</h1>
      <p className="text-sm opacity-80">{t("twoPerson")}</p>
      {err instanceof ApiError && (
        <p role="alert" className="text-sm">
          {String(err.detail)}
        </p>
      )}
      {(pending.data?.length ?? 0) > 0 && (
        <Card>
          <CardTitle>{t("pending")}</CardTitle>
          <ul className="flex flex-col gap-2 text-sm" data-testid="pending-promotions">
            {pending.data?.map((p) => {
              const mine = p.requested_by === me?.user_id;
              const mv = version.get(p.model_version_id);
              return (
                <li key={p.id} className="flex flex-wrap items-center gap-2">
                  <span className="font-medium">{p.component}</span>
                  <span>{mv?.version ?? p.model_version_id}</span>
                  <span className="opacity-70">
                    {t("requestedBy")} {mine ? me?.sub : p.requested_by.slice(0, 8)} · {dateTime(p.created_at, locale)}
                  </span>
                  {p.reason && <span>“{p.reason}”</span>}
                  {admin && (
                    <>
                      <Button size="sm" disabled={mine} title={mine ? t("twoPerson") : undefined} onClick={() => decide.mutate({ id: p.id, approve: true })}>
                        {t("approve")}
                      </Button>
                      <Button size="sm" variant="ghost" disabled={mine} onClick={() => decide.mutate({ id: p.id, approve: false })}>
                        {t("reject")}
                      </Button>
                    </>
                  )}
                </li>
              );
            })}
          </ul>
        </Card>
      )}
      <QueryState isLoading={models.isLoading} error={models.error} isEmpty={!models.data?.length}>
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {[...byComponent.entries()].map(([component, versions]) => (
            <Card key={component}>
              <CardTitle>{component}</CardTitle>
              <ul className="flex flex-col gap-2 text-sm">
                {versions.map((m) => {
                  const gate = m.metrics.gate as { passed?: boolean } | undefined;
                  const numeric = Object.entries(m.metrics).filter(([, v]) => typeof v === "number");
                  return (
                    <li key={m.id} className="flex flex-wrap items-center gap-2" data-model={m.version}>
                      <span className="font-mono">{m.version}</span>
                      {m.active && <span className="rounded border px-1 text-xs">{t("activeBadge")}</span>}
                      {m.shadow && (
                        <span className="rounded border border-dashed px-1 text-xs" title={t("shadowHint")}>
                          {t("shadowBadge")}
                        </span>
                      )}
                      {gate && (
                        <span className="text-xs" data-testid="gate">
                          {gate.passed ? t("gatePassed") : t("gateFailed")}
                        </span>
                      )}
                      {m.research_only && <span className="text-xs text-amber-700 dark:text-amber-300">{t("researchOnly")}</span>}
                      {numeric.length > 0 && (
                        <span className="text-xs opacity-80">
                          {numeric.map(([k, v]) => `${k}=${(v as number).toFixed(3)}`).join(" · ")}
                        </span>
                      )}
                      {admin && !m.active && !m.research_only && gate?.passed !== false && (
                        <Button size="sm" variant="outline" onClick={() => promote.mutate(m)}>
                          {t("promote")}
                        </Button>
                      )}
                    </li>
                  );
                })}
              </ul>
            </Card>
          ))}
        </div>
      </QueryState>
    </div>
  );
}

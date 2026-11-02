"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useLocale, useTranslations } from "next-intl";
import { useEffect, useState } from "react";

import { AuthGate } from "@/components/auth-gate";
import { CategoryLabel } from "@/components/badges";
import { QueryState } from "@/components/query-state";
import { Button } from "@/components/ui/button";
import { Card, CardTitle } from "@/components/ui/card";
import { Input, Label, Textarea } from "@/components/ui/field";
import { Link } from "@/i18n/navigation";
import type { Schemas } from "@/lib/api";
import { can, useAuth } from "@/lib/auth";
import { dateTime } from "@/lib/format";
import { checkRules, leaves, setLeaf } from "@/lib/rules-yaml";
import { ApiError, unwrap, useApi } from "@/lib/use-api";

export default function RulesPage() {
  return (
    <AuthGate perm="rules.read">
      <Rules />
    </AuthGate>
  );
}

function Rules() {
  const t = useTranslations("rules");
  const tc = useTranslations("common");
  const locale = useLocale();
  const me = useAuth((s) => s.me);
  const admin = can(me, "rules.write");
  const api = useApi();
  const qc = useQueryClient();
  const [view, setView] = useState<"yaml" | "form">("yaml");
  const [text, setText] = useState("");
  const [n, setN] = useState(50);
  const history = useQuery({
    queryKey: ["rules-history"],
    queryFn: async () => unwrap(await api.GET("/api/v1/rules", { params: { query: { history: true } } })),
  });
  const thresholds = useQuery({
    queryKey: ["thresholds"],
    queryFn: async () => unwrap(await api.GET("/api/v1/thresholds")),
  });
  const active = history.data?.find((r) => r.active);
  useEffect(() => {
    if (active && !text) setText(active.yaml);
  }, [active, text]);
  const check = checkRules(text);

  const simulate = useMutation({
    mutationFn: async () =>
      unwrap(await api.POST("/api/v1/rules/simulate", { body: { yaml: text, limit: n } })),
  });
  const save = useMutation({
    mutationFn: async (yaml: string) => unwrap(await api.PUT("/api/v1/rules", { body: { yaml } })),
    onSuccess: (r) => {
      setText(r.yaml);
      void qc.invalidateQueries({ queryKey: ["rules-history"] });
    },
  });
  const sim = simulate.data;

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center gap-2">
        <h1 className="text-xl font-semibold">{t("title")}</h1>
        {active && <span className="text-sm opacity-80">{t("version", { v: active.version })}</span>}
        <div role="tablist" className="ml-auto flex gap-1">
          {(["yaml", "form"] as const).map((v) => (
            <Button key={v} role="tab" aria-selected={view === v} size="sm" variant={view === v ? "default" : "ghost"} onClick={() => setView(v)}>
              {t(v)}
            </Button>
          ))}
        </div>
      </div>
      <QueryState isLoading={history.isLoading} error={history.error}>
        <div className="grid gap-4 xl:grid-cols-[3fr_2fr]">
          <Card>
            {view === "yaml" ? (
              <Textarea
                aria-label={t("yaml")}
                className="h-[60vh] w-full font-mono text-xs"
                spellCheck={false}
                readOnly={!admin}
                value={text}
                onChange={(e) => setText(e.target.value)}
                data-testid="rules-yaml"
              />
            ) : check.ok ? (
              <div className="flex max-h-[60vh] flex-col gap-3 overflow-auto text-sm">
                {check.doc.rules.map((r) => (
                  <fieldset key={r.name} className="rounded border border-border p-2">
                    <legend className="px-1 font-medium">{r.name}</legend>
                    <div className="grid gap-2 sm:grid-cols-2">
                      {leaves(r)
                        .filter(([k]) => k !== "name" && k !== "description")
                        .map(([k, v]) => (
                          <Label key={k}>
                            {k}
                            <Input
                              disabled={!admin}
                              defaultValue={String(v)}
                              type={typeof v === "number" ? "number" : "text"}
                              step="any"
                              onBlur={(e) => setText(setLeaf(check.doc, r.name, k, e.target.value))}
                            />
                          </Label>
                        ))}
                    </div>
                  </fieldset>
                ))}
              </div>
            ) : (
              <p role="alert">{t("invalid")}</p>
            )}
            <p className="mt-2 text-xs" role="status">
              {check.ok ? t("valid") : `${t("invalid")}: ${check.error}`}
              {save.error instanceof ApiError && ` · ${String(save.error.detail)}`}
            </p>
            {admin && (
              <div className="mt-2 flex flex-wrap items-end gap-2">
                <Label>
                  {t("lastN")}
                  <Input type="number" min={1} max={500} value={n} onChange={(e) => setN(Number(e.target.value))} className="w-24" />
                </Label>
                <Button variant="outline" disabled={!check.ok || simulate.isPending} onClick={() => simulate.mutate()}>
                  {t("simulate")}
                </Button>
                <Button disabled={!check.ok || save.isPending || text === active?.yaml} onClick={() => save.mutate(text)}>
                  {t("saveVersion")}
                </Button>
              </div>
            )}
          </Card>
          <div className="flex flex-col gap-4">
            {sim && (
              <Card data-testid="simulation">
                <CardTitle>{t("simulate")}</CardTitle>
                <p className="text-sm">
                  {t("simResult", { base: sim.baseline_alerts, cand: sim.candidate_alerts, calls: sim.calls })}
                </p>
                <ul className="text-sm">
                  {Array.from(new Set([...Object.keys(sim.baseline_by_category), ...Object.keys(sim.candidate_by_category)])).map((c) => (
                    <li key={c} className="flex gap-2">
                      <CategoryLabel category={c} />
                      {sim.baseline_by_category[c] ?? 0} → {sim.candidate_by_category[c] ?? 0}
                    </li>
                  ))}
                </ul>
                {sim.changed_calls.length > 0 && (
                  <>
                    <h3 className="mt-2 text-sm font-medium">{t("changedCalls")}</h3>
                    <ul className="text-xs">
                      {sim.changed_calls.map((d: Schemas["CallDiff"]) => (
                        <li key={d.call_id}>
                          <Link href={`/calls/${d.call_id}`} className="underline">
                            {d.call_id.slice(0, 8)}
                          </Link>{" "}
                          +{d.added_alerts?.length ?? 0} {t("added")} / −{d.removed_alerts?.length ?? 0} {t("removed")}
                        </li>
                      ))}
                    </ul>
                  </>
                )}
                <p className="mt-2 text-xs opacity-70">{t("noteGold")}</p>
              </Card>
            )}
            <Card>
              <CardTitle>{t("history")}</CardTitle>
              <ul className="flex flex-col gap-1 text-sm">
                {history.data?.map((r) => (
                  <li key={r.id} className="flex items-center gap-2">
                    <span>{t("version", { v: r.version })}</span>
                    <span className="text-xs opacity-70">{dateTime(r.updated_at, locale)}</span>
                    {r.active ? (
                      <span className="text-xs font-medium">{t("active")}</span>
                    ) : (
                      admin && (
                        <Button size="sm" variant="ghost" onClick={() => save.mutate(r.yaml)}>
                          {t("rollback")}
                        </Button>
                      )
                    )}
                  </li>
                ))}
              </ul>
            </Card>
            <Card>
              <CardTitle>{t("thresholds")}</CardTitle>
              <QueryState isLoading={thresholds.isLoading} error={thresholds.error}>
                <table className="w-full text-xs" data-testid="thresholds">
                  <thead className="text-left opacity-70">
                    <tr>
                      <th className="p-1">{tc("category")}</th>
                      <th className="p-1">log</th>
                      <th className="p-1">{t("realtime")} A / C</th>
                      <th className="p-1">{t("postcall")} A / C</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(thresholds.data?.categories ?? {}).map(([c, v]) => (
                      <tr key={c} className="border-t border-border">
                        <td className="p-1">
                          <CategoryLabel category={c} />
                        </td>
                        <td className="p-1 tabular-nums">{v.log}</td>
                        <td className="p-1 tabular-nums">
                          {v.alert_realtime.agent} / {v.alert_realtime.customer}
                        </td>
                        <td className="p-1 tabular-nums">
                          {v.alert_postcall.agent} / {v.alert_postcall.customer}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <p className="mt-1 text-xs opacity-70">{t("thresholdsNote")}</p>
              </QueryState>
            </Card>
          </div>
        </div>
      </QueryState>
    </div>
  );
}

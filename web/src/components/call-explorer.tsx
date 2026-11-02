"use client";

import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { useLocale, useTranslations } from "next-intl";
import { useState } from "react";

import { SeverityBadge, StatusBadge } from "@/components/badges";
import { QueryState } from "@/components/query-state";
import { Button } from "@/components/ui/button";
import { Input, Label, Select } from "@/components/ui/field";
import { Link } from "@/i18n/navigation";
import type { Schemas } from "@/lib/api";
import { CATEGORIES } from "@/lib/categories";
import { downloadText, toCsv } from "@/lib/csv";
import { clock, dateTime } from "@/lib/format";
import { useDirectory } from "@/lib/hooks";
import { unwrap, useApi } from "@/lib/use-api";

type Query = NonNullable<
  NonNullable<Parameters<ReturnType<typeof useApi>["GET"]>[1]>["params"]
>["query"];

interface Filters {
  q: string;
  date_from: string;
  date_to: string;
  agent_id: string;
  team_id: string;
  queue: string;
  category: string;
  min_severity: string;
  speaker: string;
  reviewed: string;
  alert: string;
  agent_score_max: string;
  hostility_min: string;
}

const EMPTY: Filters = {
  q: "",
  date_from: "",
  date_to: "",
  agent_id: "",
  team_id: "",
  queue: "",
  category: "",
  min_severity: "",
  speaker: "",
  reviewed: "",
  alert: "",
  agent_score_max: "",
  hostility_min: "",
};

const PAGE = 50;

/** Filters → API query (empty fields omitted). Exported for tests. */
export function toQuery(f: Filters, offset: number): Record<string, string | number | boolean> {
  const out: Record<string, string | number | boolean> = { limit: PAGE, offset };
  for (const [k, v] of Object.entries(f)) {
    if (v === "") continue;
    if (k === "alert" || k === "reviewed") out[k] = v === "true";
    else if (k === "date_from" || k === "date_to") out[k] = new Date(v).toISOString();
    else if (["min_severity", "agent_score_max", "hostility_min"].includes(k)) out[k] = Number(v);
    else out[k] = v;
  }
  return out;
}

export function CallExplorer({ agentOnly = false }: { agentOnly?: boolean }) {
  const t = useTranslations("calls");
  const tc = useTranslations("common");
  const tcat = useTranslations("categories");
  const locale = useLocale();
  const api = useApi();
  const dir = useDirectory();
  const [draft, setDraft] = useState<Filters>(EMPTY);
  const [filters, setFilters] = useState<Filters>(EMPTY);
  const [offset, setOffset] = useState(0);
  const calls = useQuery({
    queryKey: ["calls", filters, offset],
    queryFn: async () =>
      unwrap(
        await api.GET("/api/v1/calls", { params: { query: toQuery(filters, offset) as Query } }),
      ),
    placeholderData: keepPreviousData,
  });
  const page = calls.data;
  const set = (k: keyof Filters) => (e: { target: { value: string } }) =>
    setDraft({ ...draft, [k]: e.target.value });

  const exportCsv = (items: Schemas["CallSummary"][]) =>
    downloadText(
      "calls.csv",
      toCsv(
        items.map((c) => ({
          id: c.id,
          started_at: c.started_at,
          agent: c.agent_code,
          queue: c.queue,
          duration_s: c.duration_s,
          mode: c.mode,
          status: c.status,
          events: c.event_count,
          alerts: c.alert_count,
          max_severity: c.max_severity,
        })),
      ),
    );

  return (
    <div className="flex flex-col gap-4">
      {!agentOnly && <h1 className="text-xl font-semibold">{t("title")}</h1>}
      <form
        className="flex flex-wrap items-end gap-3"
        onSubmit={(e) => {
          e.preventDefault();
          setOffset(0);
          setFilters(draft);
        }}
      >
        <Label className="min-w-64 flex-1">
          {tc("search")}
          <Input
            type="search"
            placeholder={t("searchPlaceholder")}
            value={draft.q}
            onChange={set("q")}
          />
        </Label>
        <Label>
          {t("dateFrom")}
          <Input type="date" value={draft.date_from} onChange={set("date_from")} />
        </Label>
        <Label>
          {t("dateTo")}
          <Input type="date" value={draft.date_to} onChange={set("date_to")} />
        </Label>
        {!agentOnly && (
          <>
            <Label>
              {tc("team")}
              <Select value={draft.team_id} onChange={set("team_id")}>
                <option value="">{tc("all")}</option>
                {(dir.data?.teams ?? []).map((tm) => (
                  <option key={tm.id} value={tm.id}>
                    {tm.name}
                  </option>
                ))}
              </Select>
            </Label>
            <Label>
              {tc("agent")}
              <Select value={draft.agent_id} onChange={set("agent_id")}>
                <option value="">{tc("all")}</option>
                {(dir.data?.agents ?? []).map((a) => (
                  <option key={a.id} value={a.id}>
                    {a.employee_code}
                  </option>
                ))}
              </Select>
            </Label>
          </>
        )}
        <Label>
          {tc("queue")}
          <Input value={draft.queue} onChange={set("queue")} className="w-28" />
        </Label>
        <Label>
          {tc("category")}
          <Select value={draft.category} onChange={set("category")}>
            <option value="">{tc("all")}</option>
            {CATEGORIES.map((c) => (
              <option key={c} value={c}>
                {tcat(c)}
              </option>
            ))}
          </Select>
        </Label>
        <Label>
          {t("maxSeverity")} ≥
          <Select value={draft.min_severity} onChange={set("min_severity")}>
            <option value="">{tc("any")}</option>
            {[1, 2, 3].map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </Select>
        </Label>
        <Label>
          {tc("speaker")}
          <Select value={draft.speaker} onChange={set("speaker")}>
            <option value="">{tc("both")}</option>
            <option value="agent">{tc("agent")}</option>
            <option value="customer">{tc("customer")}</option>
          </Select>
        </Label>
        <Label>
          {t("alertOnly")}
          <Select value={draft.alert} onChange={set("alert")}>
            <option value="">{tc("any")}</option>
            <option value="true">{tc("yes")}</option>
            <option value="false">{tc("no")}</option>
          </Select>
        </Label>
        <Label>
          {tc("status")}
          <Select value={draft.reviewed} onChange={set("reviewed")}>
            <option value="">{tc("any")}</option>
            <option value="true">{t("reviewed")}</option>
            <option value="false">{t("unreviewed")}</option>
          </Select>
        </Label>
        <Label>
          {t("agentScoreMax")}
          <Input
            type="number"
            min={1}
            max={5}
            className="w-20"
            value={draft.agent_score_max}
            onChange={set("agent_score_max")}
          />
        </Label>
        <Label>
          {t("hostilityMin")}
          <Input
            type="number"
            min={1}
            max={5}
            className="w-20"
            value={draft.hostility_min}
            onChange={set("hostility_min")}
          />
        </Label>
        <Button type="submit">{tc("apply")}</Button>
        <Button
          type="button"
          variant="ghost"
          onClick={() => {
            setDraft(EMPTY);
            setFilters(EMPTY);
            setOffset(0);
          }}
        >
          {tc("reset")}
        </Button>
        {page && page.items.length > 0 && (
          <Button type="button" variant="outline" onClick={() => exportCsv(page.items)}>
            {tc("exportCsv")}
          </Button>
        )}
      </form>

      <QueryState
        isLoading={calls.isLoading}
        error={calls.error}
        isEmpty={page?.items.length === 0}
        emptyText={t("noResults")}
        onRetry={() => void calls.refetch()}
      >
        <div className="overflow-x-auto">
          <table className="w-full text-sm" data-testid="calls-table">
            <thead className="text-left text-xs opacity-70">
              <tr>
                <th className="p-2">{t("started")}</th>
                <th className="p-2">{tc("agent")}</th>
                <th className="p-2">{tc("queue")}</th>
                <th className="p-2">{tc("duration")}</th>
                <th className="p-2">{t("mode")}</th>
                <th className="p-2">{tc("status")}</th>
                <th className="p-2">{t("events")}</th>
                <th className="p-2">{t("alerts")}</th>
                <th className="p-2">{t("maxSeverity")}</th>
              </tr>
            </thead>
            <tbody>
              {page?.items.map((c) => (
                <tr key={c.id} className="border-t border-border" data-call-id={c.id}>
                  <td className="p-2">
                    <Link href={`/calls/${c.id}`} className="underline-offset-2 hover:underline">
                      {dateTime(c.started_at, locale)}
                    </Link>
                  </td>
                  <td className="p-2">{c.agent_code ?? "—"}</td>
                  <td className="p-2">{c.queue ?? "—"}</td>
                  <td className="p-2 tabular-nums">{clock(c.duration_s)}</td>
                  <td className="p-2">{c.mode}</td>
                  <td className="p-2">
                    <StatusBadge status={c.status} />
                  </td>
                  <td className="p-2 tabular-nums">{c.event_count}</td>
                  <td className="p-2 tabular-nums">{c.alert_count}</td>
                  <td className="p-2">
                    {c.max_severity ? <SeverityBadge level={c.max_severity} /> : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {page && (
          <div className="flex items-center gap-3 text-sm">
            <Button
              size="sm"
              variant="outline"
              disabled={offset === 0}
              onClick={() => setOffset(Math.max(0, offset - PAGE))}
            >
              {tc("previous")}
            </Button>
            <span>
              {tc("pageOf", {
                from: page.total ? offset + 1 : 0,
                to: offset + page.items.length,
                total: page.total,
              })}
            </span>
            <Button
              size="sm"
              variant="outline"
              disabled={offset + PAGE >= page.total}
              onClick={() => setOffset(offset + PAGE)}
            >
              {tc("next")}
            </Button>
          </div>
        )}
      </QueryState>
    </div>
  );
}

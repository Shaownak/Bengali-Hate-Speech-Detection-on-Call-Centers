"use client";

import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { useLocale, useTranslations } from "next-intl";
import { useState } from "react";

import { AuthGate } from "@/components/auth-gate";
import { QueryState } from "@/components/query-state";
import { Button } from "@/components/ui/button";
import { Input, Label } from "@/components/ui/field";
import { downloadText, toCsv } from "@/lib/csv";
import { dateTime } from "@/lib/format";
import { unwrap, useApi } from "@/lib/use-api";

const PAGE = 100;

export default function AuditPage() {
  return (
    <AuthGate perm="audit.read">
      <Audit />
    </AuthGate>
  );
}

function Audit() {
  const t = useTranslations("audit");
  const tc = useTranslations("common");
  const tcalls = useTranslations("calls");
  const locale = useLocale();
  const api = useApi();
  const [draft, setDraft] = useState({ action: "", object_type: "", object_id: "", date_from: "", date_to: "" });
  const [f, setF] = useState(draft);
  const [offset, setOffset] = useState(0);
  const q = useQuery({
    queryKey: ["audit", f, offset],
    queryFn: async () =>
      unwrap(
        await api.GET("/api/v1/audit", {
          params: {
            query: {
              action: f.action || undefined,
              object_type: f.object_type || undefined,
              object_id: f.object_id || undefined,
              date_from: f.date_from ? new Date(f.date_from).toISOString() : undefined,
              date_to: f.date_to ? new Date(f.date_to).toISOString() : undefined,
              limit: PAGE,
              offset,
            },
          },
        }),
      ),
    placeholderData: keepPreviousData,
  });
  const set = (k: keyof typeof draft) => (e: { target: { value: string } }) => setDraft({ ...draft, [k]: e.target.value });
  return (
    <div className="flex flex-col gap-4">
      <h1 className="text-xl font-semibold">{t("title")}</h1>
      <form
        className="flex flex-wrap items-end gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          setOffset(0);
          setF(draft);
        }}
      >
        <Label>
          {t("action")}
          <Input value={draft.action} onChange={set("action")} placeholder="transcript.view" />
        </Label>
        <Label>
          {t("object")}
          <Input value={draft.object_type} onChange={set("object_type")} placeholder="call" className="w-28" />
        </Label>
        <Label>
          ID
          <Input value={draft.object_id} onChange={set("object_id")} />
        </Label>
        <Label>
          {tcalls("dateFrom")}
          <Input type="date" value={draft.date_from} onChange={set("date_from")} />
        </Label>
        <Label>
          {tcalls("dateTo")}
          <Input type="date" value={draft.date_to} onChange={set("date_to")} />
        </Label>
        <Button type="submit">{tc("apply")}</Button>
        <Button
          type="button"
          variant="outline"
          disabled={!q.data?.length}
          onClick={() =>
            downloadText(
              "audit.csv",
              toCsv(
                (q.data ?? []).map((r) => ({
                  id: r.id,
                  ts: r.ts,
                  user_id: r.user_id,
                  action: r.action,
                  object_type: r.object_type,
                  object_id: r.object_id,
                  ip: r.ip,
                  details: r.details,
                })),
              ),
            )
          }
        >
          {tc("exportCsv")}
        </Button>
      </form>
      <QueryState isLoading={q.isLoading} error={q.error} isEmpty={!q.data?.length}>
        <table className="w-full text-sm" data-testid="audit-table">
          <thead className="text-left text-xs opacity-70">
            <tr>
              <th className="p-2">{tc("time")}</th>
              <th className="p-2">{t("user")}</th>
              <th className="p-2">{t("action")}</th>
              <th className="p-2">{t("object")}</th>
              <th className="p-2">{t("ip")}</th>
              <th className="p-2">{tc("status")}</th>
            </tr>
          </thead>
          <tbody>
            {q.data?.map((r) => (
              <tr key={r.id} className="border-t border-border">
                <td className="p-2 tabular-nums">{dateTime(r.ts, locale)}</td>
                <td className="p-2 font-mono text-xs">{r.user_id?.slice(0, 8) ?? "—"}</td>
                <td className="p-2">{r.action}</td>
                <td className="p-2">
                  {r.object_type} <span className="font-mono text-xs">{r.object_id?.slice(0, 36)}</span>
                </td>
                <td className="p-2">{r.ip ?? "—"}</td>
                <td className="p-2">{String((r.details as { status?: number }).status ?? "")}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="flex gap-2">
          <Button size="sm" variant="outline" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE))}>
            {tc("previous")}
          </Button>
          <Button size="sm" variant="outline" disabled={(q.data?.length ?? 0) < PAGE} onClick={() => setOffset(offset + PAGE)}>
            {tc("next")}
          </Button>
        </div>
      </QueryState>
    </div>
  );
}

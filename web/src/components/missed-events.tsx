"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useTranslations } from "next-intl";
import { useState } from "react";

import { CategoryLabel, SeverityBadge, SpeakerLabel } from "@/components/badges";
import { Button } from "@/components/ui/button";
import { Card, CardTitle } from "@/components/ui/card";
import { Input, Label, Select, Textarea } from "@/components/ui/field";
import type { Schemas } from "@/lib/api";
import { can, useAuth } from "@/lib/auth";
import { CATEGORIES } from "@/lib/categories";
import { clock } from "@/lib/format";
import { ApiError, unwrap, useApi } from "@/lib/use-api";

type Body = Schemas["MissedEventIn"];

/** False negatives found by QA (ADR-020): they feed the active-learning sampler and retraining. */
export function MissedEvents({ callId, time }: { callId: string; time: number }) {
  const t = useTranslations("missed");
  const tcat = useTranslations("categories");
  const me = useAuth((s) => s.me);
  const api = useApi();
  const qc = useQueryClient();
  const path = { params: { path: { call_id: callId } } };
  const list = useQuery({
    queryKey: ["missed", callId],
    queryFn: async () => unwrap(await api.GET("/api/v1/calls/{call_id}/missed", path)),
  });
  const [open, setOpen] = useState(false);
  const [form, setForm] = useState<Body>({
    speaker: "agent",
    category: "insult",
    severity: 2,
    ts_start: 0,
    ts_end: 3,
    comment: "",
  });
  const add = useMutation({
    mutationFn: async (body: Body) =>
      unwrap(await api.POST("/api/v1/calls/{call_id}/missed", { ...path, body })),
    onSuccess: () => {
      setOpen(false);
      void qc.invalidateQueries({ queryKey: ["missed", callId] });
    },
  });
  const set = <K extends keyof Body>(k: K, v: Body[K]) => setForm((f) => ({ ...f, [k]: v }));
  const items = list.data ?? [];
  const mayReport = can(me, "events.report_missed");
  if (!items.length && !mayReport) return null;

  return (
    <Card data-testid="missed-events">
      <CardTitle className="flex items-center justify-between gap-2">
        {t("title")}
        {mayReport && !open && (
          <Button
            size="sm"
            variant="outline"
            onClick={() => {
              const start = Math.max(0, Math.floor(time));
              setForm((f) => ({ ...f, ts_start: start, ts_end: start + 3 }));
              setOpen(true);
            }}
          >
            {t("report")}
          </Button>
        )}
      </CardTitle>
      <p className="text-xs opacity-80">{t("hint")}</p>
      {open && (
        <form
          className="my-2 grid gap-2 sm:grid-cols-3"
          onSubmit={(ev) => {
            ev.preventDefault();
            add.mutate({ ...form, comment: form.comment || null });
          }}
        >
          <Label>
            {t("speaker")}
            <Select value={form.speaker} onChange={(e) => set("speaker", e.target.value as Body["speaker"])}>
              <option value="agent">{t("agent")}</option>
              <option value="customer">{t("customer")}</option>
            </Select>
          </Label>
          <Label>
            {t("category")}
            <Select value={form.category} onChange={(e) => set("category", e.target.value as Body["category"])}>
              {CATEGORIES.map((c) => (
                <option key={c} value={c}>
                  {tcat.has(c) ? tcat(c) : c}
                </option>
              ))}
            </Select>
          </Label>
          <Label>
            {t("severity")}
            <Select value={form.severity} onChange={(e) => set("severity", Number(e.target.value) as Body["severity"])}>
              <option value={1}>1</option>
              <option value={2}>2</option>
              <option value={3}>3</option>
            </Select>
          </Label>
          <Label>
            {t("start")}
            <Input type="number" min={0} step={0.5} value={form.ts_start} onChange={(e) => set("ts_start", Number(e.target.value))} />
          </Label>
          <Label>
            {t("end")}
            <Input type="number" min={0} step={0.5} value={form.ts_end} onChange={(e) => set("ts_end", Number(e.target.value))} />
          </Label>
          <Label className="sm:col-span-3">
            {t("comment")}
            <Textarea value={form.comment ?? ""} maxLength={2000} onChange={(e) => set("comment", e.target.value)} />
          </Label>
          {add.error instanceof ApiError && (
            <p role="alert" className="text-sm sm:col-span-3">
              {String(add.error.detail)}
            </p>
          )}
          <div className="flex gap-2 sm:col-span-3">
            <Button type="submit" size="sm" disabled={add.isPending}>
              {t("save")}
            </Button>
            <Button type="button" size="sm" variant="ghost" onClick={() => setOpen(false)}>
              {t("cancel")}
            </Button>
          </div>
        </form>
      )}
      {items.length > 0 && (
        <ul className="flex flex-col gap-1 text-sm">
          {items.map((m: Schemas["MissedEventOut"]) => (
            <li key={m.id} className="flex flex-wrap items-center gap-2">
              <span className="tabular-nums">
                {clock(m.ts_start)}–{clock(m.ts_end)}
              </span>
              <SpeakerLabel speaker={m.speaker} />
              <CategoryLabel category={m.category} />
              <SeverityBadge level={m.severity} />
              {m.comment && <span className="opacity-80">“{m.comment}”</span>}
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}

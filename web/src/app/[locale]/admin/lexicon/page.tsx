"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Eye, EyeOff } from "lucide-react";
import { useTranslations } from "next-intl";
import { useState } from "react";

import { AuthGate } from "@/components/auth-gate";
import { CategoryLabel, SeverityBadge } from "@/components/badges";
import { QueryState } from "@/components/query-state";
import { Button } from "@/components/ui/button";
import { Card, CardTitle } from "@/components/ui/card";
import { Input, Label, Select } from "@/components/ui/field";
import type { Schemas } from "@/lib/api";
import { can, useAuth } from "@/lib/auth";
import { CATEGORIES } from "@/lib/categories";
import { parseCsv } from "@/lib/csv";
import { type LexIn, rowToLexicon } from "@/lib/lexicon";
import { maskTerm } from "@/lib/format";
import { unwrap, useApi } from "@/lib/use-api";

export default function LexiconPage() {
  return (
    <AuthGate perm="lexicon.read">
      <Lexicon />
    </AuthGate>
  );
}

const BLANK = { term: "", category: "insult", severity: "2", variants: "", romanized: "", dialect: "", context: false, notes: "" };

function Lexicon() {
  const t = useTranslations("lexicon");
  const tc = useTranslations("common");
  const tcat = useTranslations("categories");
  const me = useAuth((s) => s.me);
  const admin = can(me, "lexicon.write");
  const api = useApi();
  const qc = useQueryClient();
  const [reveal, setReveal] = useState(false);
  const [category, setCategory] = useState("");
  const [active, setActive] = useState("");
  const [form, setForm] = useState(BLANK);
  const [importMsg, setImportMsg] = useState<string>();
  const [matches, setMatches] = useState<Record<string, number>>({});

  const list = useQuery({
    queryKey: ["lexicon", category, active],
    queryFn: async () =>
      unwrap(
        await api.GET("/api/v1/lexicon", {
          params: {
            query: {
              category: (category || undefined) as LexIn["category"] | undefined,
              active: active === "" ? undefined : active === "true",
            },
          },
        }),
      ),
  });
  const refresh = () => void qc.invalidateQueries({ queryKey: ["lexicon"] });
  const create = useMutation({
    mutationFn: async (body: LexIn) => unwrap(await api.POST("/api/v1/lexicon", { body })),
    onSuccess: refresh,
  });
  const approve = useMutation({
    mutationFn: async (e: Schemas["LexiconOut"]) =>
      unwrap(
        await api.PUT("/api/v1/lexicon/{entry_id}", {
          params: { path: { entry_id: e.id } },
          body: {
            term: e.term,
            category: e.category as LexIn["category"],
            severity: e.severity as LexIn["severity"],
            variants: e.variants,
            romanized: e.romanized,
            dialect: e.dialect,
            context_sensitive: e.context_sensitive,
            notes: e.notes,
            active: true,
          },
        }),
      ),
    onSuccess: refresh,
  });
  const deactivate = useMutation({
    mutationFn: async (id: string) =>
      api.DELETE("/api/v1/lexicon/{entry_id}", { params: { path: { entry_id: id } } }),
    onSuccess: refresh,
  });

  async function preview(e: Schemas["LexiconOut"]) {
    const since = new Date(Date.now() - 7 * 86_400_000).toISOString();
    const r = unwrap(
      await api.GET("/api/v1/calls", { params: { query: { q: e.normalized, date_from: since, limit: 1 } } }),
    );
    setMatches((m) => ({ ...m, [e.id]: r.total }));
  }

  async function importFile(file: File) {
    const rows = parseCsv(await file.text());
    let ok = 0;
    let bad = 0;
    for (const r of rows) {
      try {
        await create.mutateAsync(rowToLexicon(r));
        ok++;
      } catch {
        bad++;
      }
    }
    setImportMsg(`${t("imported", { n: ok })}${bad ? ` · ${t("importFailed", { n: bad })}` : ""}`);
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center gap-3">
        <h1 className="text-xl font-semibold">{t("title")}</h1>
        <Button variant="outline" size="sm" onClick={() => setReveal(!reveal)} aria-pressed={reveal}>
          {reveal ? <EyeOff className="size-4" aria-hidden /> : <Eye className="size-4" aria-hidden />}
          {reveal ? t("hide") : t("reveal")}
        </Button>
        <Label>
          {tc("category")}
          <Select value={category} onChange={(e) => setCategory(e.target.value)}>
            <option value="">{tc("all")}</option>
            {CATEGORIES.map((c) => (
              <option key={c} value={c}>
                {tcat(c)}
              </option>
            ))}
          </Select>
        </Label>
        <Label>
          {tc("status")}
          <Select value={active} onChange={(e) => setActive(e.target.value)}>
            <option value="">{tc("all")}</option>
            <option value="true">{t("active")}</option>
            <option value="false">{t("proposed")}</option>
          </Select>
        </Label>
        <Label>
          {t("importCsv")}
          <Input
            type="file"
            accept=".csv,text/csv"
            title={t("csvHelp")}
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) void importFile(f);
            }}
          />
        </Label>
        {importMsg && <span role="status">{importMsg}</span>}
      </div>

      <Card>
        <CardTitle>{admin ? t("add") : t("propose")}</CardTitle>
        <form
          className="flex flex-wrap items-end gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            create.mutate(
              {
                ...rowToLexicon({
                  term: form.term,
                  category: form.category,
                  severity: form.severity,
                  variants: form.variants,
                  romanized: form.romanized,
                  dialect: form.dialect,
                  context_sensitive: form.context ? "1" : "0",
                }),
                notes: form.notes || null,
              },
              { onSuccess: () => setForm(BLANK) },
            );
          }}
        >
          <Label>
            {t("term")}
            <Input
              required
              type={reveal ? "text" : "password"}
              autoComplete="off"
              value={form.term}
              onChange={(e) => setForm({ ...form, term: e.target.value })}
            />
          </Label>
          <Label>
            {tc("category")}
            <Select value={form.category} onChange={(e) => setForm({ ...form, category: e.target.value })}>
              {CATEGORIES.map((c) => (
                <option key={c} value={c}>
                  {tcat(c)}
                </option>
              ))}
            </Select>
          </Label>
          <Label>
            {tc("severity")}
            <Select value={form.severity} onChange={(e) => setForm({ ...form, severity: e.target.value })}>
              {["1", "2", "3"].map((s) => (
                <option key={s}>{s}</option>
              ))}
            </Select>
          </Label>
          <Label>
            {t("variants")} ({t("commaSeparated")})
            <Input value={form.variants} onChange={(e) => setForm({ ...form, variants: e.target.value })} />
          </Label>
          <Label>
            {t("romanized")}
            <Input value={form.romanized} onChange={(e) => setForm({ ...form, romanized: e.target.value })} />
          </Label>
          <Label>
            {t("dialect")}
            <Input value={form.dialect} onChange={(e) => setForm({ ...form, dialect: e.target.value })} className="w-28" />
          </Label>
          <label className="inline-flex items-center gap-1 text-xs">
            <input
              type="checkbox"
              checked={form.context}
              onChange={(e) => setForm({ ...form, context: e.target.checked })}
            />
            {t("contextSensitive")}
          </label>
          <Label>
            {t("notes")}
            <Input value={form.notes} onChange={(e) => setForm({ ...form, notes: e.target.value })} />
          </Label>
          <Button type="submit" disabled={create.isPending}>
            {admin ? t("add") : t("propose")}
          </Button>
        </form>
      </Card>

      <QueryState isLoading={list.isLoading} error={list.error} isEmpty={!list.data?.length}>
        <table className="w-full text-sm" data-testid="lexicon-table">
          <thead className="text-left text-xs opacity-70">
            <tr>
              <th className="p-2">{t("term")}</th>
              <th className="p-2">{tc("category")}</th>
              <th className="p-2">{tc("severity")}</th>
              <th className="p-2">{t("variants")}</th>
              <th className="p-2">{t("dialect")}</th>
              <th className="p-2">{tc("status")}</th>
              <th className="p-2">{t("matches7d")}</th>
              <th className="p-2" />
            </tr>
          </thead>
          <tbody>
            {list.data?.map((e) => (
              <tr key={e.id} className="border-t border-border" data-active={e.active}>
                <td className="p-2 font-medium" data-masked={!reveal}>
                  {reveal ? e.term : maskTerm(e.term)}
                </td>
                <td className="p-2">
                  <CategoryLabel category={e.category} />
                </td>
                <td className="p-2">
                  <SeverityBadge level={e.severity} />
                </td>
                <td className="p-2">{reveal ? e.variants.join(", ") : e.variants.map(maskTerm).join(", ")}</td>
                <td className="p-2">{e.dialect ?? "—"}</td>
                <td className="p-2">{e.active ? t("active") : t("proposed")}</td>
                <td className="p-2">
                  {matches[e.id] !== undefined ? (
                    matches[e.id]
                  ) : (
                    <Button size="sm" variant="ghost" onClick={() => void preview(e)}>
                      {tc("search")}
                    </Button>
                  )}
                </td>
                <td className="flex gap-1 p-2">
                  {admin && !e.active && (
                    <Button size="sm" variant="outline" onClick={() => approve.mutate(e)}>
                      {t("approve")}
                    </Button>
                  )}
                  {admin && e.active && (
                    <Button size="sm" variant="ghost" onClick={() => deactivate.mutate(e.id)}>
                      {t("deactivate")}
                    </Button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </QueryState>
    </div>
  );
}

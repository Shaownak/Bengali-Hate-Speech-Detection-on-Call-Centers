"use client";

import { useMutation } from "@tanstack/react-query";
import { useVirtualizer } from "@tanstack/react-virtual";
import { BellRing, Volume2 } from "lucide-react";
import { useTranslations } from "next-intl";
import { useEffect, useMemo, useRef, useState } from "react";

import { AuthGate } from "@/components/auth-gate";
import { CategoryLabel, SeverityBadge, SpeakerLabel, StatusBadge } from "@/components/badges";
import { RiskMeter } from "@/components/risk-meter";
import { Button } from "@/components/ui/button";
import { Card, CardTitle } from "@/components/ui/card";
import { Label, Select } from "@/components/ui/field";
import { Link } from "@/i18n/navigation";
import { can, useAuth } from "@/lib/auth";
import { CATEGORIES } from "@/lib/categories";
import { clock } from "@/lib/format";
import { agentLookup, beep, useDirectory, useNow, usePref } from "@/lib/hooks";
import { kpis, type LiveAlert, type LiveCall, riskOf, useLiveStore } from "@/lib/live-store";
import { useApi } from "@/lib/use-api";

interface Filters {
  team: string;
  queue: string;
  minSeverity: number;
  speaker: "" | "agent" | "customer";
  category: string;
}

const NO_FILTERS: Filters = { team: "", queue: "", minSeverity: 1, speaker: "", category: "" };

export default function LivePage() {
  return (
    <AuthGate perm="live.read" live>
      <Wallboard />
    </AuthGate>
  );
}

function Wallboard() {
  const t = useTranslations("live");
  const tc = useTranslations("common");
  const tcat = useTranslations("categories");
  const api = useApi();
  const me = useAuth((s) => s.me);
  const { calls, alerts, connect, markAcked, markFalseAlarm, setOnNewAlert } = useLiveStore();
  const now = useNow(1000);
  const dir = useDirectory();
  const agentOf = useMemo(() => agentLookup(dir.data), [dir.data]);
  const teamName = useMemo(
    () => new Map((dir.data?.teams ?? []).map((tm) => [tm.id, tm.name])),
    [dir.data],
  );
  const [f, setF] = useState<Filters>(NO_FILTERS);
  const [sound, setSound] = usePref("gale-alert-sound", false);
  const [notify, setNotify] = usePref("gale-alert-notify", false);

  useEffect(() => connect(() => useAuth.getState().token), [connect]);
  useEffect(() => {
    setOnNewAlert((a: LiveAlert) => {
      if (a.event.severity !== 3) return;
      if (sound) beep();
      if (notify && "Notification" in window && Notification.permission === "granted") {
        new Notification(t("newAlert"), { tag: a.event.event_id });
      }
    });
    return () => setOnNewAlert(undefined);
  }, [sound, notify, setOnNewAlert, t]);

  const callPasses = (c: LiveCall) => {
    const agent = agentOf(c.agent);
    if (f.team && agent?.team_id !== f.team) return false;
    if (f.queue && c.queue !== f.queue) return false;
    return true;
  };
  const alertPasses = (a: LiveAlert) => {
    const e = a.event;
    if (e.severity < f.minSeverity) return false;
    if (f.speaker && e.speaker !== f.speaker) return false;
    if (f.category && e.category !== f.category) return false;
    const call = calls[e.call_id];
    return call ? callPasses(call) : !f.team && !f.queue;
  };

  const callList = Object.values(calls)
    .filter(callPasses)
    .sort((a, b) => {
      const ra = Math.max(riskOf(a, "agent", now)?.p ?? 0, riskOf(a, "customer", now)?.p ?? 0);
      const rb = Math.max(riskOf(b, "agent", now)?.p ?? 0, riskOf(b, "customer", now)?.p ?? 0);
      return rb - ra || a.startedAt - b.startedAt;
    });
  const feed = Object.values(alerts)
    .filter(alertPasses)
    .sort((a, b) => b.receivedAt - a.receivedAt);
  const k = kpis({ calls, alerts }, now);
  const queues = Array.from(new Set(Object.values(calls).map((c) => c.queue).filter(Boolean)));

  const ack = useMutation({
    mutationFn: async (eventId: string) => {
      const r = await api.POST("/api/v1/events/{event_id}/ack", {
        params: { path: { event_id: eventId } },
      });
      if (!r.response.ok) throw new Error(String(r.response.status));
      markAcked(eventId);
    },
  });
  const falseAlarm = useMutation({
    mutationFn: async (eventId: string) => {
      const r = await api.POST("/api/v1/events/{event_id}/reviews", {
        params: { path: { event_id: eventId } },
        body: { kind: "verdict", verdict: "false_positive" },
      });
      if (!r.response.ok) throw new Error(String(r.response.status));
      markFalseAlarm(eventId);
    },
  });

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h1 className="text-xl font-semibold">{t("title")}</h1>
        <div className="flex gap-3 text-sm">
          <label className="inline-flex items-center gap-1">
            <input type="checkbox" checked={sound} onChange={(e) => setSound(e.target.checked)} />
            <Volume2 aria-hidden className="size-4" /> {t("sound")}
          </label>
          <label className="inline-flex items-center gap-1">
            <input
              type="checkbox"
              checked={notify}
              onChange={(e) => {
                setNotify(e.target.checked);
                if (e.target.checked && "Notification" in window) void Notification.requestPermission();
              }}
            />
            <BellRing aria-hidden className="size-4" /> {t("notify")}
          </label>
        </div>
      </div>

      <dl className="grid grid-cols-2 gap-3 sm:grid-cols-5" data-testid="kpis">
        {[
          [t("activeCalls"), k.active],
          [t("alertsLastHour"), k.alertsLastHour],
          [t("agentCaused"), k.agentCaused],
          [t("customerCaused"), k.customerCaused],
          [t("medianAck"), k.medianAckMs === undefined ? "—" : clock(k.medianAckMs / 1000)],
        ].map(([label, value]) => (
          <div key={String(label)} className="rounded-lg border border-border p-3">
            <dt className="text-xs opacity-70">{label}</dt>
            <dd className="text-2xl font-semibold tabular-nums">{value}</dd>
          </div>
        ))}
      </dl>

      <fieldset className="flex flex-wrap items-end gap-3">
        <legend className="sr-only">{tc("filters")}</legend>
        <Label>
          {tc("team")}
          <Select value={f.team} onChange={(e) => setF({ ...f, team: e.target.value })}>
            <option value="">{tc("all")}</option>
            {(dir.data?.teams ?? []).map((tm) => (
              <option key={tm.id} value={tm.id}>
                {tm.name}
              </option>
            ))}
          </Select>
        </Label>
        <Label>
          {tc("queue")}
          <Select value={f.queue} onChange={(e) => setF({ ...f, queue: e.target.value })}>
            <option value="">{tc("all")}</option>
            {queues.map((q) => (
              <option key={q} value={q ?? ""}>
                {q}
              </option>
            ))}
          </Select>
        </Label>
        <Label>
          {t("minSeverity")}
          <Select
            value={f.minSeverity}
            onChange={(e) => setF({ ...f, minSeverity: Number(e.target.value) })}
          >
            {[1, 2, 3].map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </Select>
        </Label>
        <Label>
          {tc("speaker")}
          <Select
            value={f.speaker}
            onChange={(e) => setF({ ...f, speaker: e.target.value as Filters["speaker"] })}
          >
            <option value="">{tc("both")}</option>
            <option value="agent">{tc("agent")}</option>
            <option value="customer">{tc("customer")}</option>
          </Select>
        </Label>
        <Label>
          {tc("category")}
          <Select value={f.category} onChange={(e) => setF({ ...f, category: e.target.value })}>
            <option value="">{tc("all")}</option>
            {CATEGORIES.map((c) => (
              <option key={c} value={c}>
                {tcat(c)}
              </option>
            ))}
          </Select>
        </Label>
        <Button variant="ghost" size="sm" onClick={() => setF(NO_FILTERS)}>
          {tc("reset")}
        </Button>
      </fieldset>

      <div className="grid gap-4 lg:grid-cols-[2fr_1fr]">
        <Card>
          <CardTitle>{t("activeCalls")}</CardTitle>
          {callList.length === 0 ? (
            <p className="text-sm opacity-70">{t("noCalls")}</p>
          ) : (
            <CallGrid
              calls={callList}
              now={now}
              agentName={(c) => {
                const a = agentOf(c.agent);
                return a ? `${a.name ?? a.employee_code}` : (c.agent ?? "—");
              }}
              teamName={(c) => teamName.get(agentOf(c.agent)?.team_id ?? "") ?? ""}
            />
          )}
        </Card>

        <Card aria-live="polite">
          <CardTitle>{t("alertFeed")}</CardTitle>
          {feed.length === 0 ? (
            <p className="text-sm opacity-70">{t("noAlerts")}</p>
          ) : (
            <ol className="flex max-h-[70vh] flex-col gap-2 overflow-auto" data-testid="alert-feed">
              {feed.map((a) => {
                const e = a.event;
                const agent = agentOf(calls[e.call_id]?.agent);
                return (
                  <li
                    key={e.event_id}
                    className="rounded border border-border p-2 text-sm"
                    data-event-id={e.event_id}
                    data-status={e.status}
                  >
                    <div className="flex flex-wrap items-center gap-2">
                      <time className="tabular-nums opacity-70">
                        {new Date(a.receivedAt).toLocaleTimeString()}
                      </time>
                      <span className="font-medium">{agent?.employee_code ?? "—"}</span>
                      <SpeakerLabel speaker={e.speaker} />
                      <CategoryLabel category={e.category} />
                      <SeverityBadge level={e.severity} />
                      <StatusBadge status={a.falseAlarm ? "reviewed_false" : (e.status ?? "provisional")} />
                    </div>
                    {e.explanation_bn && <p className="mt-1 line-clamp-1">{e.explanation_bn}</p>}
                    <div className="mt-2 flex flex-wrap gap-2">
                      <Button asChild size="sm" variant="outline">
                        <Link href={`/live/${e.call_id}`}>{t("openCall")}</Link>
                      </Button>
                      {can(me, "events.ack") && (
                        <Button
                          size="sm"
                          variant="outline"
                          disabled={a.ackedAt !== undefined}
                          onClick={() => ack.mutate(e.event_id)}
                        >
                          {a.ackedAt !== undefined ? t("acknowledged") : t("acknowledge")}
                        </Button>
                      )}
                      {can(me, "events.review") && (
                        <Button
                          size="sm"
                          variant="ghost"
                          disabled={a.falseAlarm}
                          onClick={() => falseAlarm.mutate(e.event_id)}
                        >
                          {t("falseAlarm")}
                        </Button>
                      )}
                    </div>
                  </li>
                );
              })}
            </ol>
          )}
        </Card>
      </div>
    </div>
  );
}

/** Virtualised so 1,000 active calls at 2 updates/s stay smooth (docs/04 §4). */
function CallGrid({
  calls,
  now,
  agentName,
  teamName,
}: {
  calls: LiveCall[];
  now: number;
  agentName: (c: LiveCall) => string;
  teamName: (c: LiveCall) => string;
}) {
  const t = useTranslations("live");
  const parent = useRef<HTMLDivElement>(null);
  const v = useVirtualizer({
    count: calls.length,
    getScrollElement: () => parent.current,
    estimateSize: () => 72,
    overscan: 8,
  });
  return (
    <div ref={parent} className="max-h-[70vh] overflow-auto" data-testid="call-grid">
      <ul style={{ height: v.getTotalSize(), position: "relative" }}>
        {v.getVirtualItems().map((row) => {
          const c = calls[row.index];
          if (!c) return null;
          return (
            <li
              key={c.callId}
              data-call-id={c.callId}
              className="absolute inset-x-0 flex items-center gap-4 border-b border-border px-1 py-2"
              style={{ transform: `translateY(${row.start}px)`, height: row.size }}
            >
              <Link href={`/live/${c.callId}`} className="min-w-32 hover:underline">
                <div className="font-medium">{agentName(c)}</div>
                <div className="text-xs opacity-70">
                  {[teamName(c), c.queue].filter(Boolean).join(" · ")} ·{" "}
                  <span className="tabular-nums">{clock((now - c.startedAt) / 1000)}</span>
                </div>
              </Link>
              <RiskMeter label={t("riskAgent")} point={riskOf(c, "agent", now)} />
              <RiskMeter label={t("riskCustomer")} point={riskOf(c, "customer", now)} />
            </li>
          );
        })}
      </ul>
    </div>
  );
}

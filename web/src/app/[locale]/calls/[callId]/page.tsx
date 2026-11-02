"use client";

import { useQuery } from "@tanstack/react-query";
import { useLocale, useTranslations } from "next-intl";
import { use, useMemo, useRef, useState } from "react";

import { AuthGate } from "@/components/auth-gate";
import { SeverityBadge, StatusBadge } from "@/components/badges";
import { EventTable } from "@/components/event-table";
import { QueryState } from "@/components/query-state";
import { Scorecard } from "@/components/scorecard";
import { TimelineChart } from "@/components/timeline-chart";
import { Transcript } from "@/components/transcript";
import { Button } from "@/components/ui/button";
import { Card, CardTitle } from "@/components/ui/card";
import { type PlayerApi, type RegionSpec, Waveform } from "@/components/waveform";
import { downloadText, toCsv } from "@/lib/csv";
import { clock, dateTime } from "@/lib/format";
import { unwrap, useApi } from "@/lib/use-api";

export default function CallDetailPage({ params }: { params: Promise<{ callId: string }> }) {
  const { callId } = use(params);
  return (
    <AuthGate perm="calls.read">
      <CallDetail callId={callId} />
    </AuthGate>
  );
}

const SEV_COLOR = { 1: "rgba(14,165,233,0.25)", 2: "rgba(245,158,11,0.3)", 3: "rgba(220,38,38,0.3)" };
const LINE = {
  agent: "#2563eb",
  customer: "#dc2626",
};

function CallDetail({ callId }: { callId: string }) {
  const t = useTranslations("call");
  const tc = useTranslations("common");
  const tl = useTranslations("live");
  const tcat = useTranslations("categories");
  const locale = useLocale();
  const api = useApi();
  const player = useRef<PlayerApi>(null);
  const [time, setTime] = useState(0);
  const path = { params: { path: { call_id: callId } } };

  const detail = useQuery({
    queryKey: ["call", callId],
    queryFn: async () => unwrap(await api.GET("/api/v1/calls/{call_id}", path)),
  });
  const events = useQuery({
    queryKey: ["events", callId],
    queryFn: async () => unwrap(await api.GET("/api/v1/calls/{call_id}/events", path)),
  });
  const segments = useQuery({
    queryKey: ["segments", callId],
    queryFn: async () => unwrap(await api.GET("/api/v1/calls/{call_id}/segments", path)),
  });
  const timelines = useQuery({
    queryKey: ["timelines", callId],
    queryFn: async () => unwrap(await api.GET("/api/v1/calls/{call_id}/timelines", path)),
  });
  const audio = useQuery({
    queryKey: ["audio", callId],
    queryFn: async () => {
      const r = await api.GET("/api/v1/calls/{call_id}/audio-url", path);
      return r.response.status === 404 ? null : unwrap(r);
    },
    staleTime: 60_000, // links expire after the API's signed-URL TTL
    retry: false,
  });

  const evs = useMemo(() => events.data ?? [], [events.data]);
  const regions: RegionSpec[] = useMemo(
    () =>
      evs
        .filter((e) => e.status !== "dismissed_by_llm")
        .map((e) => ({
          id: e.event_id,
          start: e.ts_start,
          end: Math.max(e.ts_end, e.ts_start + 0.3),
          channel: e.speaker,
          color: SEV_COLOR[e.severity as 1 | 2 | 3] ?? SEV_COLOR[1],
          label: tcat.has(e.category) ? tcat(e.category) : e.category,
        })),
    [evs, tcat],
  );
  const seek = (s: number) => {
    setTime(s);
    player.current?.seek(s);
  };

  const series = useMemo(() => {
    const tl0 = timelines.data;
    if (!tl0) return [];
    const out = [];
    for (const [ch, byName] of Object.entries(tl0.series)) {
      for (const [name, values] of Object.entries(byName)) {
        out.push({
          key: `${ch}-${name}`,
          label: `${name} · ${tc(ch === "agent" ? "agent" : "customer")}`,
          color: LINE[ch as "agent" | "customer"] ?? "#666",
          dashed: name !== "loudness",
          points: values.map((v, i) => [i * tl0.bin_s, v] as [number, number | null]),
        });
      }
    }
    return out;
  }, [timelines.data, tc]);

  const exportCsv = () =>
    downloadText(
      `call-${callId}-events.csv`,
      toCsv(
        evs.map((e) => ({
          event_id: e.event_id,
          ts_start: e.ts_start,
          ts_end: e.ts_end,
          speaker: e.speaker,
          category: e.category,
          severity: e.severity,
          probability: e.probability,
          status: e.status,
          alert: e.alert,
          verdict: e.latest_verdict,
          explanation_bn: e.explanation_bn,
        })),
      ),
    );

  const d = detail.data;
  const curve = d?.scores?.escalation_curve ?? [];
  return (
    <QueryState isLoading={detail.isLoading} error={detail.error} onRetry={() => void detail.refetch()}>
      {d && (
        <div className="flex flex-col gap-4">
          <div className="flex flex-wrap items-center gap-3">
            <h1 className="text-xl font-semibold">{t("title", { id: d.id.slice(0, 8) })}</h1>
            <StatusBadge status={d.status} />
            {d.max_severity ? <SeverityBadge level={d.max_severity} /> : null}
            <span className="text-sm opacity-80">
              {d.agent_code ?? "—"} · {d.queue ?? "—"} · {dateTime(d.started_at, locale)} ·{" "}
              {clock(d.duration_s)}
            </span>
            <div className="ml-auto flex gap-2 no-print">
              <Button size="sm" variant="outline" onClick={exportCsv} disabled={!evs.length}>
                {tc("exportCsv")}
              </Button>
              <Button size="sm" variant="outline" onClick={() => window.print()}>
                {tc("exportPdf")}
              </Button>
            </div>
          </div>

          <Card className="no-print">
            <CardTitle>{t("waveform")}</CardTitle>
            {audio.data ? (
              <Waveform
                urls={audio.data.urls}
                regions={regions}
                onTime={setTime}
                onReady={(a) => void (player.current = a)}
              />
            ) : (
              <p className="text-sm opacity-70">{audio.isLoading ? tc("loading") : t("noAudio")}</p>
            )}
          </Card>

          <div className="grid gap-4 xl:grid-cols-2">
            <Card>
              <CardTitle>{t("transcript")}</CardTitle>
              <QueryState isLoading={segments.isLoading} error={segments.error} isEmpty={!segments.data?.length}>
                <Transcript
                  segments={segments.data ?? []}
                  time={time}
                  onSeek={seek}
                  flagged={(s) =>
                    evs.find(
                      (e) =>
                        e.speaker === s.channel &&
                        e.status !== "dismissed_by_llm" &&
                        e.ts_start < s.ts_end &&
                        s.ts_start < e.ts_end,
                    )?.severity
                  }
                />
              </QueryState>
            </Card>
            <Card>
              <CardTitle>{t("scorecard")}</CardTitle>
              {d.scores ? <Scorecard scores={d.scores} /> : <p className="text-sm opacity-70">{t("noScorecard")}</p>}
            </Card>
          </div>

          <Card>
            <CardTitle>{t("timelines")}</CardTitle>
            <QueryState isLoading={timelines.isLoading} error={timelines.error} isEmpty={!series.length}>
              <TimelineChart
                ariaLabel={t("timelines")}
                series={series}
                onSeek={seek}
                markers={time > 0 ? [{ t: Math.round(time), label: "▶" }] : []}
              />
            </QueryState>
            {curve.length > 0 && (
              <div className="mt-4">
                <h3 className="mb-1 text-sm font-medium">
                  {t("escalationCurve")}
                  {d.scores && (
                    <span className="ml-2 font-normal opacity-80">
                      · {t("whoFirst")}: {t(`who.${d.scores.scorecard.escalation.who_escalated_first}`)}
                    </span>
                  )}
                </h3>
                <TimelineChart
                  ariaLabel={t("escalationCurve")}
                  height={140}
                  series={[
                    {
                      key: "ea",
                      label: tl("agentCaused"),
                      color: LINE.agent,
                      points: curve.map((r) => [r[0] ?? 0, Math.min(1, (r[1] ?? 0) / 5)]),
                    },
                    {
                      key: "ec",
                      label: tl("customerCaused"),
                      color: LINE.customer,
                      points: curve.map((r) => [r[0] ?? 0, Math.min(1, (r[2] ?? 0) / 5)]),
                    },
                  ]}
                />
              </div>
            )}
          </Card>

          <Card>
            <CardTitle>{t("eventTable")}</CardTitle>
            <QueryState isLoading={events.isLoading} error={events.error} isEmpty={!evs.length}>
              <EventTable events={evs} onSeek={seek} />
            </QueryState>
          </Card>
        </div>
      )}
    </QueryState>
  );
}

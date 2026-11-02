"use client";

import { useTranslations } from "next-intl";
import { use, useEffect, useMemo, useState } from "react";

import { AuthGate } from "@/components/auth-gate";
import { CategoryLabel, SeverityBadge, SpeakerLabel, StatusBadge } from "@/components/badges";
import { EventActions } from "@/components/event-actions";
import { TimelineChart } from "@/components/timeline-chart";
import { Card, CardTitle } from "@/components/ui/card";
import { useAuth } from "@/lib/auth";
import { type CallLive, callApply, callFromSnapshot, emptyCall, lane } from "@/lib/call-live";
import { clock } from "@/lib/format";
import { LiveSocket, type SocketStatus } from "@/lib/live-socket";
import { SEVERITY, asSeverity } from "@/lib/severity";
import { cn } from "@/lib/utils";
import type { FusedEvent } from "@/types/generated";

export default function LiveCallPage({ params }: { params: Promise<{ callId: string }> }) {
  const { callId } = use(params);
  return (
    <AuthGate perm="live_call.read" live>
      <LiveCall callId={callId} />
    </AuthGate>
  );
}

function LiveCall({ callId }: { callId: string }) {
  const t = useTranslations("live");
  const tc = useTranslations("common");
  const [state, setState] = useState<CallLive>(emptyCall);
  const [status, setStatus] = useState<SocketStatus>("idle");
  const [code, setCode] = useState<number>();

  useEffect(() => {
    const sock = new LiveSocket(`/ws/calls/${callId}`, () => useAuth.getState().token, {
      onStatus: (s, c) => {
        setStatus(s);
        setCode(c);
      },
      onSnapshot: (snap) => setState(callFromSnapshot(snap)),
      onBatch: (items) => setState((prev) => callApply(prev, items)),
    });
    sock.start();
    return () => sock.stop();
  }, [callId]);

  const events = useMemo(
    () => Object.values(state.events).sort((a, b) => a.ts_start - b.ts_start),
    [state.events],
  );

  if (status === "denied") {
    return <p role="alert">{code === 4404 ? tc("notAvailable") : tc("notAllowed")}</p>;
  }

  return (
    <div className="flex flex-col gap-4">
      <h1 className="text-xl font-semibold">
        {t("transcript")} <span className="font-mono text-sm opacity-70">{callId}</span>
      </h1>
      {status === "reconnecting" && (
        <p role="status" className="text-sm">
          {tc("offline")}
        </p>
      )}
      {state.status === "ended" && (
        <p role="status" className="text-sm opacity-80">
          {t("ended")}
        </p>
      )}
      <div className="grid gap-4 xl:grid-cols-[3fr_2fr]">
        <Card>
          <CardTitle>{t("transcript")}</CardTitle>
          <div className="grid grid-cols-2 gap-3" data-testid="live-transcript">
            {(["agent", "customer"] as const).map((ch) => (
              <section key={ch} aria-label={tc(ch)}>
                <h3 className="mb-2 text-sm font-medium">
                  <SpeakerLabel speaker={ch} />
                </h3>
                <ol className="flex max-h-[60vh] flex-col gap-1 overflow-auto text-sm" aria-live="polite">
                  {lane(state, ch).map((g) => {
                    const hit = events.find(
                      (e) =>
                        e.speaker === ch &&
                        e.status !== "dismissed_by_llm" &&
                        e.ts_start < g.tsEnd &&
                        g.tsStart < e.ts_end,
                    );
                    const sev = asSeverity(hit?.severity);
                    const switched = state.markers.some(
                      (m) => m.kind === "register" && m.channel === ch && m.t >= g.tsStart && m.t <= g.tsEnd,
                    );
                    return (
                      <li
                        key={g.id}
                        className={cn(
                          "rounded px-1",
                          !g.final && "italic opacity-60",
                          sev && "underline decoration-2 underline-offset-4",
                          sev === 3 && "decoration-double",
                          sev && SEVERITY[sev].className,
                        )}
                        data-final={g.final}
                      >
                        <span className="mr-1 text-xs tabular-nums opacity-60">{clock(g.tsStart)}</span>
                        {switched && (
                          <span title={t("registerSwitch")} aria-label={t("registerSwitch")} className="mr-1">
                            আপনি→তুমি ⇣
                          </span>
                        )}
                        {g.text}
                        {!g.final && <span className="sr-only"> ({t("partial")})</span>}
                      </li>
                    );
                  })}
                </ol>
              </section>
            ))}
          </div>
        </Card>
        <div className="flex flex-col gap-4">
          <Card>
            <CardTitle>{t("events")}</CardTitle>
            <ol className="flex flex-col gap-2" data-testid="live-events">
              {events.map((e) => (
                <EventItem key={e.event_id} e={e} />
              ))}
            </ol>
          </Card>
          <Card>
            <TimelineChart
              ariaLabel={`${t("loudness")} / ${t("anger")}`}
              markers={state.markers
                .filter((m) => m.kind === "overtalk")
                .map((m) => ({ t: m.t, label: t("overtalk") }))}
              series={[
                { key: "la", label: `${t("loudness")} · ${tc("agent")}`, color: "#2563eb", points: state.loudness.agent ?? [] },
                { key: "lc", label: `${t("loudness")} · ${tc("customer")}`, color: "#dc2626", points: state.loudness.customer ?? [] },
                { key: "aa", label: `${t("anger")} · ${tc("agent")}`, color: "#2563eb", dashed: true, points: state.anger.agent ?? [] },
                { key: "ac", label: `${t("anger")} · ${tc("customer")}`, color: "#dc2626", dashed: true, points: state.anger.customer ?? [] },
              ]}
            />
          </Card>
        </div>
      </div>
    </div>
  );
}

function EventItem({ e }: { e: FusedEvent }) {
  const tc = useTranslations("call");
  return (
    <li
      className={cn(
        "rounded border p-2 text-sm transition-colors duration-700",
        e.status === "provisional" ? "border-dashed border-border" : "border-solid border-foreground/40",
      )}
      data-event-id={e.event_id}
      data-status={e.status}
    >
      <div className="flex flex-wrap items-center gap-2">
        <span className="tabular-nums opacity-70">{clock(e.ts_start)}</span>
        <SpeakerLabel speaker={e.speaker} />
        <CategoryLabel category={e.category} />
        <SeverityBadge level={e.severity} />
        <StatusBadge status={e.status ?? "provisional"} />
      </div>
      {(e.explanation_bn || e.explanation_en) && (
        <details className="mt-1">
          <summary className="cursor-pointer text-xs">{tc("rationale")}</summary>
          <p>{e.explanation_bn}</p>
          {e.explanation_en && <p className="opacity-80">{e.explanation_en}</p>}
        </details>
      )}
      <div className="mt-2">
        <EventActions eventId={e.event_id} alert={!!e.alert} />
      </div>
    </li>
  );
}

"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useTranslations } from "next-intl";
import { useEffect, useRef, useState } from "react";

import { AuthGate } from "@/components/auth-gate";
import { CategoryLabel, SeverityBadge, SpeakerLabel, StatusBadge } from "@/components/badges";
import { useReview } from "@/components/event-actions";
import { QueryState } from "@/components/query-state";
import { RedactedText } from "@/components/transcript";
import { Button } from "@/components/ui/button";
import { Card, CardTitle } from "@/components/ui/card";
import { Label, Select } from "@/components/ui/field";
import { type PlayerApi, Waveform } from "@/components/waveform";
import { Link } from "@/i18n/navigation";
import type { Schemas } from "@/lib/api";
import { CATEGORIES } from "@/lib/categories";
import { clock, pct } from "@/lib/format";
import { keyAction } from "@/lib/review-keys";
import { unwrap, useApi } from "@/lib/use-api";

export default function ReviewPage() {
  return (
    <AuthGate perm="review_queue.read">
      <ReviewQueue />
    </AuthGate>
  );
}

function ReviewQueue() {
  const t = useTranslations("review");
  const api = useApi();
  const qc = useQueryClient();
  const [i, setI] = useState(0);
  const queue = useQuery({
    queryKey: ["review-queue"],
    queryFn: async () =>
      unwrap(await api.GET("/api/v1/review-queue", { params: { query: { limit: 100 } } })),
  });
  const agreement = useQuery({
    queryKey: ["agreement"],
    queryFn: async () => unwrap(await api.GET("/api/v1/reviews/agreement")),
  });
  const items = queue.data?.items ?? [];
  const idx = Math.min(i, Math.max(0, items.length - 1));
  const item = items[idx];
  const a = agreement.data;

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h1 className="text-xl font-semibold">{t("title")}</h1>
        <Card className="py-2 text-sm" aria-label={t("agreement")}>
          <span className="font-medium">{t("agreement")}: </span>
          {a?.pairwise_agreement != null
            ? t("agreementValue", { pct: pct(a.pairwise_agreement), n: a.events_with_multiple_verdicts })
            : t("agreementNone")}
        </Card>
      </div>
      <p className="text-xs opacity-80" id="review-help">
        {t("help")}
      </p>
      <QueryState
        isLoading={queue.isLoading}
        error={queue.error}
        isEmpty={!items.length}
        emptyText={t("empty")}
        onRetry={() => void queue.refetch()}
      >
        {item && (
          <ReviewItem
            key={item.event.event_id}
            item={item}
            position={t("position", { i: idx + 1, n: queue.data?.total ?? items.length })}
            onNext={() => setI(Math.min(items.length - 1, idx + 1))}
            onPrev={() => setI(Math.max(0, idx - 1))}
            onDone={() => {
              void qc.invalidateQueries({ queryKey: ["review-queue"] });
              void qc.invalidateQueries({ queryKey: ["agreement"] });
            }}
          />
        )}
      </QueryState>
    </div>
  );
}

function ReviewItem({
  item,
  position,
  onNext,
  onPrev,
  onDone,
}: {
  item: Schemas["ReviewQueueItem"];
  position: string;
  onNext: () => void;
  onPrev: () => void;
  onDone: () => void;
}) {
  const t = useTranslations("review");
  const tc = useTranslations("call");
  const tcat = useTranslations("categories");
  const api = useApi();
  const e = item.event;
  const review = useReview(e.event_id);
  const player = useRef<PlayerApi>(null);
  const catRef = useRef<HTMLSelectElement>(null);
  const sevRef = useRef<HTMLSelectElement>(null);
  const [cat, setCat] = useState<string>(e.category);
  const [sev, setSev] = useState<1 | 2 | 3>(e.severity === 3 ? 2 : 3);
  const path = { params: { path: { call_id: e.call_id } } };
  const audio = useQuery({
    queryKey: ["audio", e.call_id],
    queryFn: async () => {
      const r = await api.GET("/api/v1/calls/{call_id}/audio-url", path);
      return r.response.status === 404 ? null : unwrap(r);
    },
    retry: false,
    staleTime: 60_000,
  });
  const segments = useQuery({
    queryKey: ["segments", e.call_id],
    queryFn: async () => unwrap(await api.GET("/api/v1/calls/{call_id}/segments", path)),
  });
  const context = (segments.data ?? []).filter(
    (s) => s.ts_end >= e.ts_start - 10 && s.ts_start <= e.ts_end + 10,
  );

  const submit = (body: Parameters<typeof review.mutate>[0]) =>
    review.mutate(body, { onSuccess: onDone });

  useEffect(() => {
    const onKey = (ev: KeyboardEvent) => {
      const act = keyAction(ev.key, ev.target);
      if (!act) return;
      ev.preventDefault();
      if (act.type === "next") onNext();
      else if (act.type === "prev") onPrev();
      else if (act.type === "verdict") submit({ kind: "verdict", verdict: act.verdict });
      else if (act.type === "pick") (act.field === "category" ? catRef : sevRef).current?.focus();
      else if (act.type === "play") player.current?.playSpan(e.ts_start - 3, e.ts_end + 3);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  return (
    <Card aria-describedby="review-help" data-testid="review-item" data-event-id={e.event_id} data-call-id={e.call_id}>
      <CardTitle className="flex flex-wrap items-center gap-2">
        <span className="text-sm font-normal opacity-70">{position}</span>
        <SpeakerLabel speaker={e.speaker} />
        <CategoryLabel category={e.category} />
        <SeverityBadge level={e.severity} />
        <StatusBadge status={e.status ?? "provisional"} />
        <span className="text-sm font-normal">
          {tc("evidence")} {clock(e.ts_start)} · {t("uncertainty")} {pct(1 - Math.abs(e.probability - 0.5) * 2)}
        </span>
        <Link href={`/calls/${e.call_id}`} className="ml-auto text-sm font-normal underline">
          {item.call.agent_code ?? "—"} · {item.call.queue ?? "—"}
        </Link>
      </CardTitle>
      {e.explanation_bn && <p lang="bn">{e.explanation_bn}</p>}
      <ul className="my-2 text-xs">
        {e.top_signals?.map((s) => (
          <li key={s.signal_id}>
            <span className="font-mono">{s.source}</span> · {pct(s.score)}
            {s.text_span && (
              <>
                {" "}
                · <RedactedText text={s.text_span} />
              </>
            )}
          </li>
        ))}
      </ul>
      <ol className="mb-3 flex flex-col gap-0.5 rounded bg-muted/50 p-2 text-sm">
        {context.map((s) => (
          <li key={s.id} className={s.ts_start <= e.ts_end && e.ts_start <= s.ts_end ? "font-semibold" : ""}>
            <span className="mr-2 tabular-nums opacity-60">{clock(s.ts_start)}</span>
            <SpeakerLabel speaker={s.channel} />: <RedactedText text={s.text_redacted} />
          </li>
        ))}
      </ol>
      {audio.data && (
        <Waveform urls={audio.data.urls} regions={[]} onReady={(p) => void (player.current = p)} />
      )}
      <div className="mt-3 flex flex-wrap items-end gap-2">
        <Button onClick={() => submit({ kind: "verdict", verdict: "correct" })}>1 · {tc("correct")}</Button>
        <Label>
          {t("pickCategory")}
          <Select
            ref={catRef}
            value={cat}
            onChange={(ev) => setCat(ev.target.value)}
            onKeyDown={(ev) => {
              if (ev.key === "Enter")
                submit({ kind: "verdict", verdict: "wrong_category", corrected_category: cat });
            }}
          >
            {CATEGORIES.map((c) => (
              <option key={c} value={c}>
                {tcat(c)}
              </option>
            ))}
          </Select>
        </Label>
        <Button
          variant="outline"
          onClick={() => submit({ kind: "verdict", verdict: "wrong_category", corrected_category: cat })}
        >
          2 · {tc("wrongCategory")}
        </Button>
        <Button variant="outline" onClick={() => submit({ kind: "verdict", verdict: "false_positive" })}>
          3 · {tc("falsePositive")}
        </Button>
        <Label>
          {t("pickSeverity")}
          <Select
            ref={sevRef}
            value={sev}
            onChange={(ev) => setSev(Number(ev.target.value) as 1 | 2 | 3)}
            onKeyDown={(ev) => {
              if (ev.key === "Enter")
                submit({ kind: "verdict", verdict: "wrong_severity", corrected_severity: sev });
            }}
          >
            {[1, 2, 3].map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </Select>
        </Label>
        <Button
          variant="outline"
          onClick={() => submit({ kind: "verdict", verdict: "wrong_severity", corrected_severity: sev })}
        >
          4 · {tc("wrongSeverity")}
        </Button>
        {review.isError && <span role="alert">{String(review.error)}</span>}
      </div>
    </Card>
  );
}

"use client";

import { ChevronDown, ChevronRight } from "lucide-react";
import { useTranslations } from "next-intl";
import { Fragment, useState } from "react";

import { CategoryLabel, SeverityBadge, SpeakerLabel, StatusBadge } from "@/components/badges";
import { EventActions, useReview } from "@/components/event-actions";
import { RedactedText } from "@/components/transcript";
import { Button } from "@/components/ui/button";
import { Select, Textarea } from "@/components/ui/field";
import type { Schemas } from "@/lib/api";
import { can, useAuth } from "@/lib/auth";
import { CATEGORIES } from "@/lib/categories";
import { clock, pct } from "@/lib/format";

type EventOut = Schemas["EventOut"];

/** QA verdict buttons (docs/04 §2.3–2.4) and the agent's dispute form. */
export function ReviewControls({ e }: { e: EventOut }) {
  const t = useTranslations("call");
  const tcat = useTranslations("categories");
  const me = useAuth((s) => s.me);
  const review = useReview(e.event_id);
  const [cat, setCat] = useState<string>(e.category);
  const [sev, setSev] = useState<1 | 2 | 3>(e.severity === 3 ? 2 : 3);
  const [dispute, setDispute] = useState("");
  if (can(me, "events.review")) {
    return (
      <div className="flex flex-wrap items-center gap-2" data-testid="review-controls">
        <Button size="sm" variant="outline" onClick={() => review.mutate({ kind: "verdict", verdict: "correct" })}>
          {t("correct")}
        </Button>
        <span className="inline-flex items-center gap-1">
          <Select aria-label={t("wrongCategory")} value={cat} onChange={(ev) => setCat(ev.target.value)}>
            {CATEGORIES.map((c) => (
              <option key={c} value={c}>
                {tcat(c)}
              </option>
            ))}
          </Select>
          <Button
            size="sm"
            variant="outline"
            onClick={() =>
              review.mutate({ kind: "verdict", verdict: "wrong_category", corrected_category: cat })
            }
          >
            {t("wrongCategory")}
          </Button>
        </span>
        <Button
          size="sm"
          variant="outline"
          onClick={() => review.mutate({ kind: "verdict", verdict: "false_positive" })}
        >
          {t("falsePositive")}
        </Button>
        <span className="inline-flex items-center gap-1">
          <Select
            aria-label={t("wrongSeverity")}
            value={sev}
            onChange={(ev) => setSev(Number(ev.target.value) as 1 | 2 | 3)}
          >
            {[1, 2, 3].map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </Select>
          <Button
            size="sm"
            variant="outline"
            onClick={() =>
              review.mutate({ kind: "verdict", verdict: "wrong_severity", corrected_severity: sev })
            }
          >
            {t("wrongSeverity")}
          </Button>
        </span>
        {review.isSuccess && <span role="status">{t("saved")}</span>}
      </div>
    );
  }
  if (can(me, "events.dispute") && e.speaker === "agent") {
    return (
      <form
        className="flex flex-col gap-2"
        onSubmit={(ev) => {
          ev.preventDefault();
          review.mutate({ kind: "dispute", comment: dispute }, { onSuccess: () => setDispute("") });
        }}
      >
        <Textarea
          aria-label={t("dispute")}
          placeholder={t("disputePlaceholder")}
          value={dispute}
          onChange={(ev) => setDispute(ev.target.value)}
        />
        <div className="flex items-center gap-2">
          <Button size="sm" type="submit" disabled={!dispute.trim()}>
            {t("dispute")}
          </Button>
          {review.isSuccess && <span role="status">{t("saved")}</span>}
        </div>
      </form>
    );
  }
  return null;
}

/** Event table with evidence drill-down: top signals, LLM rationale, reviews (docs/04 §2.3). */
export function EventTable({ events, onSeek }: { events: EventOut[]; onSeek?: (t: number) => void }) {
  const t = useTranslations("call");
  const tc = useTranslations("common");
  const [open, setOpen] = useState<string>();
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm" data-testid="event-table">
        <thead className="text-left text-xs opacity-70">
          <tr>
            <th className="p-2" />
            <th className="p-2">{tc("time")}</th>
            <th className="p-2">{tc("speaker")}</th>
            <th className="p-2">{tc("category")}</th>
            <th className="p-2">{tc("severity")}</th>
            <th className="p-2">{tc("probability")}</th>
            <th className="p-2">{tc("status")}</th>
            <th className="p-2">{t("verdict")}</th>
          </tr>
        </thead>
        <tbody>
          {events.map((e) => {
            const isOpen = open === e.event_id;
            return (
              <Fragment key={e.event_id}>
                <tr className="border-t border-border" data-event-id={e.event_id} data-status={e.status}>
                  <td className="p-2">
                    <button
                      type="button"
                      aria-expanded={isOpen}
                      aria-label={tc("details")}
                      onClick={() => setOpen(isOpen ? undefined : e.event_id)}
                    >
                      {isOpen ? <ChevronDown className="size-4" /> : <ChevronRight className="size-4" />}
                    </button>
                  </td>
                  <td className="p-2">
                    <button
                      type="button"
                      className="tabular-nums hover:underline"
                      onClick={() => onSeek?.(e.ts_start)}
                      aria-label={t("seek", { t: clock(e.ts_start) })}
                    >
                      {clock(e.ts_start)}
                    </button>
                  </td>
                  <td className="p-2">
                    <SpeakerLabel speaker={e.speaker} />
                  </td>
                  <td className="p-2">
                    <CategoryLabel category={e.category} />
                  </td>
                  <td className="p-2">
                    <SeverityBadge level={e.severity} />
                  </td>
                  <td className="p-2 tabular-nums">{pct(e.probability)}</td>
                  <td className="p-2">
                    <StatusBadge status={e.status ?? "provisional"} />
                    {e.alert && <span className="ml-1 text-xs">🔔</span>}
                  </td>
                  <td className="p-2">{e.latest_verdict ?? "—"}</td>
                </tr>
                {isOpen && (
                  <tr className="bg-muted/40">
                    <td />
                    <td colSpan={7} className="p-2">
                      <div className="flex flex-col gap-3">
                        <div>
                          <h4 className="text-xs font-medium">{t("signals")}</h4>
                          <ul className="text-xs">
                            {e.top_signals?.map((s) => (
                              <li key={s.signal_id}>
                                <span className="font-mono">{s.source}</span> · {s.category} · {pct(s.score)}
                                {s.text_span && (
                                  <>
                                    {" "}
                                    · {t("evidence")}: <RedactedText text={s.text_span} />
                                  </>
                                )}
                              </li>
                            ))}
                          </ul>
                        </div>
                        {(e.explanation_bn || e.explanation_en) && (
                          <div>
                            <h4 className="text-xs font-medium">{t("rationale")}</h4>
                            <p lang="bn">{e.explanation_bn}</p>
                            {e.explanation_en && <p lang="en">{e.explanation_en}</p>}
                          </div>
                        )}
                        {(e.reviews?.length ?? 0) > 0 && (
                          <div>
                            <h4 className="text-xs font-medium">{t("notes")}</h4>
                            <ul className="text-xs">
                              {e.reviews?.map((r, i) => (
                                <li key={i}>
                                  {r.kind}
                                  {r.verdict ? `: ${r.verdict}` : ""} {r.comment ? `— ${r.comment}` : ""}
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                        <ReviewControls e={e} />
                        <EventActions eventId={e.event_id} alert={!!e.alert} acked={!!e.acked_by} />
                      </div>
                    </td>
                  </tr>
                )}
              </Fragment>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

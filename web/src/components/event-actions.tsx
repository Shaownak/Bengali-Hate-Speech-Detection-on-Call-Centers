"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useTranslations } from "next-intl";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/field";
import { can, useAuth } from "@/lib/auth";
import { useApi } from "@/lib/use-api";

type ReviewBody = {
  kind: "verdict" | "dispute" | "note" | "flag";
  verdict?: "correct" | "wrong_category" | "false_positive" | "wrong_severity";
  corrected_category?: string;
  corrected_severity?: 1 | 2 | 3;
  comment?: string;
};

/** POST /events/{id}/reviews, then refresh anything that shows events. */
export function useReview(eventId: string) {
  const api = useApi();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (body: ReviewBody) => {
      const r = await api.POST("/api/v1/events/{event_id}/reviews", {
        params: { path: { event_id: eventId } },
        // the generated type narrows corrected_category to the Category enum
        body: body as never,
      });
      if (!r.response.ok) throw new Error(String(r.response.status));
      return r.data;
    },
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ["events"] });
      void qc.invalidateQueries({ queryKey: ["review-queue"] });
    },
  });
}

/** Supervisor actions on a live/recorded event: acknowledge, note, flag for QA (docs/04 §2.2). */
export function EventActions({
  eventId,
  alert,
  acked,
  onAcked,
}: {
  eventId: string;
  alert: boolean;
  acked?: boolean;
  onAcked?: () => void;
}) {
  const t = useTranslations("live");
  const tc = useTranslations("common");
  const me = useAuth((s) => s.me);
  const api = useApi();
  const review = useReview(eventId);
  const [note, setNote] = useState("");
  const [open, setOpen] = useState(false);
  const [done, setDone] = useState<string>();
  const ack = useMutation({
    mutationFn: async () => {
      const r = await api.POST("/api/v1/events/{event_id}/ack", {
        params: { path: { event_id: eventId } },
      });
      if (!r.response.ok) throw new Error(String(r.response.status));
      onAcked?.();
    },
  });
  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap gap-2">
        {alert && can(me, "events.ack") && (
          <Button
            size="sm"
            variant="outline"
            disabled={acked || ack.isSuccess}
            onClick={() => ack.mutate()}
          >
            {acked || ack.isSuccess ? t("acknowledged") : t("acknowledge")}
          </Button>
        )}
        {can(me, "events.review") && (
          <>
            <Button size="sm" variant="outline" onClick={() => setOpen((o) => !o)}>
              {t("addNote")}
            </Button>
            <Button
              size="sm"
              variant="ghost"
              disabled={done === "flag"}
              onClick={() => review.mutate({ kind: "flag" }, { onSuccess: () => setDone("flag") })}
            >
              {done === "flag" ? t("flagged") : t("flagQa")}
            </Button>
          </>
        )}
      </div>
      {open && (
        <form
          className="flex flex-col gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            review.mutate(
              { kind: "note", comment: note },
              {
                onSuccess: () => {
                  setNote("");
                  setOpen(false);
                  setDone("note");
                },
              },
            );
          }}
        >
          <Textarea
            aria-label={t("addNote")}
            placeholder={t("notePlaceholder")}
            value={note}
            onChange={(e) => setNote(e.target.value)}
            required
          />
          <div className="flex gap-2">
            <Button size="sm" type="submit" disabled={!note.trim() || review.isPending}>
              {tc("save")}
            </Button>
            <Button size="sm" variant="ghost" type="button" onClick={() => setOpen(false)}>
              {tc("cancel")}
            </Button>
          </div>
        </form>
      )}
    </div>
  );
}

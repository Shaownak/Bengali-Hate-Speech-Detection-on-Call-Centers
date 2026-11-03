"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useTranslations } from "next-intl";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Card, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/field";
import { can, useAuth } from "@/lib/auth";
import { unwrap, useApi } from "@/lib/use-api";

/** Weekly usefulness rating (pilot P-2 exit: supervisors rate ≥ 4/5; ADR-020). */
export function UsefulnessCard() {
  const t = useTranslations("pilot");
  const me = useAuth((s) => s.me);
  const api = useApi();
  const qc = useQueryClient();
  const allowed = can(me, "pilot.feedback");
  const mine = useQuery({
    queryKey: ["pilot-feedback"],
    queryFn: async () => unwrap(await api.GET("/api/v1/pilot/feedback")),
    enabled: allowed,
  });
  const [comment, setComment] = useState("");
  const rate = useMutation({
    mutationFn: async (rating: number) =>
      unwrap(await api.POST("/api/v1/pilot/feedback", { body: { rating, comment: comment || null } })),
    onSuccess: () => void qc.invalidateQueries({ queryKey: ["pilot-feedback"] }),
  });
  if (!allowed) return null;
  const current = mine.data?.rating;

  return (
    <Card data-testid="usefulness">
      <CardTitle>{t("rateTitle")}</CardTitle>
      <p className="text-sm opacity-80">{t("rateQuestion")}</p>
      <div role="radiogroup" aria-label={t("rateQuestion")} className="my-2 flex gap-1">
        {[1, 2, 3, 4, 5].map((n) => (
          <Button
            key={n}
            role="radio"
            aria-checked={current === n}
            size="sm"
            variant={current === n ? "default" : "outline"}
            disabled={rate.isPending}
            onClick={() => rate.mutate(n)}
          >
            {n}
          </Button>
        ))}
      </div>
      <Textarea
        aria-label={t("rateComment")}
        placeholder={t("rateComment")}
        value={comment}
        maxLength={2000}
        onChange={(e) => setComment(e.target.value)}
      />
      {current != null && (
        <p className="mt-1 text-xs opacity-80" role="status">
          {t("rateSaved", { rating: current, week: mine.data?.week ?? "" })}
        </p>
      )}
    </Card>
  );
}

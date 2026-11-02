"use client";

import { useTranslations } from "next-intl";

import { AgentScorecard } from "@/components/agent-scorecard";
import { AuthGate } from "@/components/auth-gate";
import { CallExplorer } from "@/components/call-explorer";
import { Card, CardTitle } from "@/components/ui/card";
import { useAuth } from "@/lib/auth";

/** Agent self-view (docs/04 §2.6): own calls and scorecard only; the API enforces the scope. */
export default function MePage() {
  return (
    <AuthGate perm="events.dispute">
      <Me />
    </AuthGate>
  );
}

function Me() {
  const t = useTranslations("me");
  const me = useAuth((s) => s.me);
  return (
    <div className="flex flex-col gap-4">
      <h1 className="text-xl font-semibold">{t("title")}</h1>
      <p className="text-sm opacity-80">{t("disputeHint")}</p>
      {me?.agent_id && (
        <Card>
          <CardTitle>{t("scorecard")}</CardTitle>
          <AgentScorecard agentId={me.agent_id} />
        </Card>
      )}
      <Card>
        <CardTitle>{t("title")}</CardTitle>
        <CallExplorer agentOnly />
      </Card>
    </div>
  );
}

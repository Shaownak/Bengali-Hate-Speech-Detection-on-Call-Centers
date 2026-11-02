import { useTranslations } from "next-intl";

import type { Schemas } from "@/lib/api";

function Score({ label, value }: { label: string; value: number }) {
  return (
    <div className="flex items-center justify-between gap-2">
      <dt>{label}</dt>
      <dd className="flex items-center gap-2">
        <meter min={1} max={5} low={2.5} high={3.5} optimum={5} value={value} aria-label={label} />
        <span className="tabular-nums">{value}/5</span>
      </dd>
    </div>
  );
}

/** LLM call-mode scorecard (docs/04 §2.3, docs/08). */
export function Scorecard({ scores }: { scores: Schemas["CallScoreOut"] }) {
  const t = useTranslations("call");
  const tc = useTranslations("common");
  const c = scores.scorecard;
  return (
    <div className="grid gap-4 text-sm md:grid-cols-2" data-testid="scorecard">
      <dl className="flex flex-col gap-1">
        <h3 className="font-medium">{tc("agent")}</h3>
        <Score label={t("professionalism")} value={c.agent.professionalism} />
        <Score label={t("empathy")} value={c.agent.empathy} />
        <Score label={t("deEscalation")} value={c.agent.de_escalation} />
        <div className="flex justify-between">
          <dt>{t("register")}</dt>
          <dd>{c.agent.respectful_register_maintained ? tc("yes") : tc("no")}</dd>
        </div>
        {c.agent.violations.length > 0 && (
          <div>
            <dt>{t("violations")}</dt>
            <dd>{c.agent.violations.join(", ")}</dd>
          </div>
        )}
      </dl>
      <dl className="flex flex-col gap-1">
        <h3 className="font-medium">{tc("customer")}</h3>
        <Score label={t("hostility")} value={c.customer.hostility} />
        <div className="flex justify-between">
          <dt>{t("abuseTowardAgent")}</dt>
          <dd>{c.customer.abuse_toward_agent ? tc("yes") : tc("no")}</dd>
        </div>
        <div className="flex justify-between gap-2">
          <dt>{t("mainIssue")}</dt>
          <dd>{c.customer.main_issue_bn}</dd>
        </div>
        <div className="flex justify-between">
          <dt>{t("escalated")}</dt>
          <dd>{c.escalation.escalated ? tc("yes") : tc("no")}</dd>
        </div>
        <div className="flex justify-between">
          <dt>{t("whoFirst")}</dt>
          <dd data-testid="who-first">{t(`who.${c.escalation.who_escalated_first}`)}</dd>
        </div>
      </dl>
      <div className="md:col-span-2">
        <h3 className="font-medium">{t("summary")}</h3>
        <p lang="bn">{c.summary_bn}</p>
        <p lang="en" className="opacity-80">
          {c.summary_en}
        </p>
      </div>
      {c.coaching_points_bn.length > 0 && (
        <div className="md:col-span-2">
          <h3 className="font-medium">{t("coaching")}</h3>
          <ul className="list-disc pl-5" lang="bn">
            {c.coaching_points_bn.map((p, i) => (
              <li key={i}>{p}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

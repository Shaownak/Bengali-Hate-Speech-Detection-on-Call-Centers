import { Ban, CircleCheck, CircleDashed, Ear, EyeOff, Flame, Hand, Megaphone, MessageSquareWarning, ShieldAlert, Skull, Smile, Swords, UserX, Volume2, Zap, type LucideIcon } from "lucide-react";
import { useTranslations } from "next-intl";

import { asSeverity, SEVERITY } from "@/lib/severity";
import { cn } from "@/lib/utils";

const CATEGORY_ICON: Record<string, LucideIcon> = {
  slang_profanity: MessageSquareWarning,
  insult: UserX,
  threat: Skull,
  sexual_harassment: ShieldAlert,
  hate_identity: Swords,
  disrespect_register: Ear,
  harsh_tone: Flame,
  shouting: Volume2,
  sarcasm_mockery: Smile,
  dismissive_rude: Hand,
  interruption_overtalk: Megaphone,
};

export function SeverityBadge({ level, className }: { level: number; className?: string }) {
  const t = useTranslations("severity");
  const s = asSeverity(level);
  if (!s) return <span>—</span>;
  const { icon: Icon, className: tone } = SEVERITY[s];
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded border px-1.5 py-0.5 text-xs font-medium",
        tone,
        className,
      )}
      data-severity={s}
      title={t("label", { level: s })}
    >
      <Icon aria-hidden className="size-3.5" />
      {t(String(s) as "1" | "2" | "3")}
    </span>
  );
}

export function CategoryLabel({ category, className }: { category: string; className?: string }) {
  const t = useTranslations("categories");
  const Icon = CATEGORY_ICON[category] ?? Zap;
  return (
    <span className={cn("inline-flex items-center gap-1", className)} data-category={category}>
      <Icon aria-hidden className="size-4 shrink-0" />
      {t.has(category) ? t(category) : category}
    </span>
  );
}

export function StatusBadge({ status }: { status: string }) {
  const t = useTranslations("status");
  const provisional = status === "provisional";
  const Icon = provisional ? CircleDashed : status.startsWith("dismissed") || status === "reviewed_false" ? Ban : CircleCheck;
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs",
        provisional ? "border-dashed" : "border-solid",
      )}
      data-status={status}
      title={provisional ? t("fastPath") : undefined}
    >
      <Icon aria-hidden className="size-3.5" />
      {t.has(status) ? t(status) : status}
    </span>
  );
}

/** Shadow mode (ADR-020): computed but not shown to supervisors yet — QA/admin only. */
export function ShadowBadge() {
  const t = useTranslations("status");
  return (
    <span
      className="inline-flex items-center gap-1 rounded-full border border-dotted px-2 py-0.5 text-xs opacity-80"
      data-shadow
      title={t("shadowHint")}
    >
      <EyeOff aria-hidden className="size-3.5" />
      {t("shadow")}
    </span>
  );
}

export function SpeakerLabel({ speaker }: { speaker: string }) {
  const t = useTranslations("speaker");
  return <span data-speaker={speaker}>{t.has(speaker) ? t(speaker) : speaker}</span>;
}

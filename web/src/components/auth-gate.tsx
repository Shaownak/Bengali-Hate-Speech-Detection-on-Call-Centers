"use client";

import { useTranslations } from "next-intl";
import { type ReactNode, useEffect } from "react";

import { AppShell } from "@/components/app-shell";
import { useRouter } from "@/i18n/navigation";
import { can, useAuth, useHydrated } from "@/lib/auth";

/**
 * Page guard: no session → /login; session without ``perm`` → "not allowed". The server enforces
 * the same rules (ADR-016); this only avoids showing screens the role can't use.
 */
export function AuthGate({
  perm,
  children,
  live = false,
}: {
  perm?: string;
  children: ReactNode;
  live?: boolean;
}) {
  const t = useTranslations("common");
  const { token, me } = useAuth();
  const hydrated = useHydrated();
  const router = useRouter();
  useEffect(() => {
    if (hydrated && !token) router.replace("/login");
  }, [hydrated, token, router]);
  if (!hydrated || !token || !me) {
    return (
      <p role="status" className="p-8 text-sm opacity-70">
        {t("loading")}
      </p>
    );
  }
  return (
    <AppShell live={live}>
      {perm && !can(me, perm) ? (
        <p role="alert" className="p-4">
          {t("notAllowed")}
        </p>
      ) : (
        children
      )}
    </AppShell>
  );
}

"use client";

import { useLocale, useTranslations } from "next-intl";
import { useEffect, useState } from "react";

import { useRouter } from "@/i18n/navigation";
import { homeFor, useAuth } from "@/lib/auth";
import { oidcCallback } from "@/lib/oidc";

/** Keycloak redirect target (authorization code + PKCE). */
export default function AuthCallback() {
  const t = useTranslations("auth");
  const locale = useLocale();
  const router = useRouter();
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    oidcCallback(locale)
      .then(() => {
        const me = useAuth.getState().me;
        router.replace(me ? homeFor(me) : "/login");
      })
      .catch(() => setFailed(true));
  }, [locale, router]);
  return (
    <p role="status" className="p-8">
      {failed ? t("failed") : t("callback")}
    </p>
  );
}

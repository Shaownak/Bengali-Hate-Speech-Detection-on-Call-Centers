"use client";

import { useLocale, useTranslations } from "next-intl";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Card, CardTitle } from "@/components/ui/card";
import { Link, usePathname, useRouter } from "@/i18n/navigation";
import { AUTH_MODE, DEV_USERS, devSignIn, homeFor } from "@/lib/auth";
import { oidcSignIn } from "@/lib/oidc";

export default function LoginPage() {
  const t = useTranslations("auth");
  const th = useTranslations("home");
  const tr = useTranslations("roles");
  const locale = useLocale();
  const router = useRouter();
  const pathname = usePathname();
  const [error, setError] = useState<string>();
  const [busy, setBusy] = useState(false);

  async function dev(sub: string, role: (typeof DEV_USERS)[number]["role"]) {
    setBusy(true);
    setError(undefined);
    try {
      const me = await devSignIn(sub, role);
      router.replace(homeFor(me));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="mx-auto flex max-w-xl flex-col gap-4 p-8">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">{th("title")}</h1>
        <Button asChild variant="outline" size="sm">
          <Link href={pathname} locale={locale === "bn" ? "en" : "bn"}>
            {th("switchLanguage")}
          </Link>
        </Button>
      </div>
      <Card>
        <CardTitle>{t("title")}</CardTitle>
        {AUTH_MODE === "oidc" ? (
          <Button onClick={() => oidcSignIn(locale)}>{t("oidc")}</Button>
        ) : (
          <>
            <p className="mb-3 text-sm opacity-80">{t("devNotice")}</p>
            <ul className="grid gap-2 sm:grid-cols-2">
              {DEV_USERS.map((u) => (
                <li key={u.sub}>
                  <Button
                    variant="outline"
                    className="w-full justify-between"
                    disabled={busy}
                    onClick={() => dev(u.sub, u.role)}
                    data-testid={`login-${u.sub}`}
                  >
                    <span>{u.sub}</span>
                    <span className="text-xs opacity-70">{tr(u.role)}</span>
                  </Button>
                </li>
              ))}
            </ul>
          </>
        )}
        {error && (
          <p role="alert" className="mt-3 text-sm text-red-700 dark:text-red-300">
            {t("failed")} ({error})
          </p>
        )}
      </Card>
    </main>
  );
}

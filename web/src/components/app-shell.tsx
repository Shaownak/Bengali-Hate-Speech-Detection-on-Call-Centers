"use client";

import { LogOut, Moon, Sun, WifiOff } from "lucide-react";
import { useLocale, useTranslations } from "next-intl";
import { type ReactNode, useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import { Link, usePathname, useRouter } from "@/i18n/navigation";
import { can, homeFor, useAuth } from "@/lib/auth";
import { useLiveStore } from "@/lib/live-store";
import { cn } from "@/lib/utils";

const NAV: ReadonlyArray<{ href: string; key: string; perm: string; agentOnly?: boolean }> = [
  { href: "/live", key: "live", perm: "live.read" },
  { href: "/calls", key: "calls", perm: "calls.read" },
  { href: "/review", key: "review", perm: "review_queue.read" },
  { href: "/analytics", key: "analytics", perm: "analytics.read" },
  { href: "/me", key: "me", perm: "events.dispute", agentOnly: true },
  { href: "/admin/lexicon", key: "lexicon", perm: "lexicon.read" },
  { href: "/admin/rules", key: "rules", perm: "rules.read" },
  { href: "/admin/models", key: "models", perm: "models.read" },
  { href: "/admin/health", key: "health", perm: "models.read" },
  { href: "/admin/audit", key: "audit", perm: "audit.read" },
];

export function useTheme(): [boolean, () => void] {
  const [dark, setDark] = useState(false);
  useEffect(() => setDark(document.documentElement.classList.contains("dark")), []);
  const toggle = () => {
    const next = !document.documentElement.classList.contains("dark");
    document.documentElement.classList.toggle("dark", next);
    try {
      localStorage.setItem("gale-theme", next ? "dark" : "light");
    } catch {
      /* storage unavailable: theme lasts for this page only */
    }
    setDark(next);
  };
  return [dark, toggle];
}

/** Signed-in chrome: role-based navigation, language and theme toggles, offline banner. */
export function AppShell({ children, live = false }: { children: ReactNode; live?: boolean }) {
  const t = useTranslations("nav");
  const tc = useTranslations("common");
  const tr = useTranslations("roles");
  const me = useAuth((s) => s.me);
  const clear = useAuth((s) => s.clear);
  const locale = useLocale();
  const pathname = usePathname();
  const router = useRouter();
  const [dark, toggleTheme] = useTheme();
  const status = useLiveStore((s) => s.status);
  const offline = live && (status === "reconnecting" || status === "closed");
  const other = locale === "bn" ? "en" : "bn";

  return (
    <div className="flex min-h-screen flex-col">
      <a href="#main" className="sr-only focus:not-sr-only focus:p-2">
        {t("skip")}
      </a>
      <header className="flex flex-wrap items-center gap-3 border-b border-border px-4 py-2 print:hidden">
        <Link href={me ? homeFor(me) : "/"} className="font-semibold">
          GAL-E
        </Link>
        <nav aria-label="main" className="flex flex-wrap gap-1">
          {NAV.filter(
            (n) => can(me, n.perm) && (!n.agentOnly || me?.role === "agent"),
          ).map((n) => (
            <Link
              key={n.href}
              href={n.href}
              aria-current={pathname.startsWith(n.href) ? "page" : undefined}
              className={cn(
                "rounded px-2 py-1 text-sm hover:bg-muted",
                pathname.startsWith(n.href) && "bg-muted font-medium",
              )}
            >
              {t(n.key)}
            </Link>
          ))}
        </nav>
        <div className="ml-auto flex items-center gap-2 text-sm">
          {me && (
            <span className="opacity-80" data-testid="whoami">
              {me.sub} · {tr(me.role)}
            </span>
          )}
          <Button asChild variant="ghost" size="sm" aria-label={tc("language")}>
            <Link href={pathname} locale={other}>
              {other === "en" ? "English" : "বাংলা"}
            </Link>
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={toggleTheme}
            aria-label={`${tc("theme")}: ${dark ? tc("themeLight") : tc("themeDark")}`}
          >
            {dark ? <Sun className="size-4" aria-hidden /> : <Moon className="size-4" aria-hidden />}
          </Button>
          {me && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => {
                clear();
                router.replace("/login");
              }}
            >
              <LogOut className="size-4" aria-hidden />
              {t("signOut")}
            </Button>
          )}
        </div>
      </header>
      {offline && (
        <div
          role="status"
          className="flex items-center gap-2 bg-amber-200 px-4 py-1 text-sm text-amber-950"
        >
          <WifiOff className="size-4" aria-hidden />
          {tc("offline")}
        </div>
      )}
      <main id="main" className="flex-1 p-4">
        {children}
      </main>
      <footer className="border-t border-border px-4 py-2 text-xs opacity-70 print:hidden">
        {tc("advisory")}
      </footer>
    </div>
  );
}

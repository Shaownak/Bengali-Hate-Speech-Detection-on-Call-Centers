"use client";

import { useTranslations } from "next-intl";
import { useEffect } from "react";

import { CategoryList } from "@/components/category-list";
import { useRouter } from "@/i18n/navigation";
import { homeFor, useAuth, useHydrated } from "@/lib/auth";

/** Entry: signed-in users go to their role's home screen, others to sign-in. */
export default function Home() {
  const t = useTranslations("home");
  const me = useAuth((s) => s.me);
  const hydrated = useHydrated();
  const router = useRouter();
  useEffect(() => {
    if (hydrated) router.replace(me ? homeFor(me) : "/login");
  }, [hydrated, me, router]);
  return (
    <main className="mx-auto flex max-w-3xl flex-col gap-6 p-8">
      <h1 className="text-2xl font-semibold">{t("title")}</h1>
      <p>{t("subtitle")}</p>
      <section aria-labelledby="cats">
        <h2 id="cats" className="mb-2 font-medium">
          {t("categoriesHeading")}
        </h2>
        <CategoryList />
      </section>
    </main>
  );
}

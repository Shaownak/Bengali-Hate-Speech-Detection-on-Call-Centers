import Link from "next/link";
import { useTranslations } from "next-intl";
import { setRequestLocale } from "next-intl/server";
import { use } from "react";

import { CategoryList } from "@/components/category-list";
import { Button } from "@/components/ui/button";

export default function Home({ params }: { params: Promise<{ locale: string }> }) {
  const { locale } = use(params);
  setRequestLocale(locale);
  const t = useTranslations("home");
  const other = locale === "bn" ? "en" : "bn";
  return (
    <main className="mx-auto flex max-w-3xl flex-col gap-6 p-8">
      <header className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold">{t("title")}</h1>
        <Button asChild variant="outline" size="sm">
          <Link href={`/${other}`}>{t("switchLanguage")}</Link>
        </Button>
      </header>
      <p>{t("subtitle")}</p>
      <section aria-labelledby="cats">
        <h2 id="cats" className="mb-2 font-medium">
          {t("categoriesHeading")}
        </h2>
        <CategoryList />
      </section>
      <p className="text-sm opacity-70">{t("stubNotice")}</p>
    </main>
  );
}

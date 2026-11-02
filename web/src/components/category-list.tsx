import { useTranslations } from "next-intl";

import { CATEGORIES } from "@/lib/categories";

export function CategoryList() {
  const t = useTranslations("categories");
  return (
    <ul className="grid gap-2 sm:grid-cols-2">
      {CATEGORIES.map((c) => (
        <li key={c} className="rounded-md border border-border px-3 py-2" data-category={c}>
          {t(c)}
        </li>
      ))}
    </ul>
  );
}

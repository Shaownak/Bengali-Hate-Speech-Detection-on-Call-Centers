import { defineRouting } from "next-intl/routing";

// docs/04: Bangla is the default UI language, English is the toggle.
export const routing = defineRouting({
  locales: ["bn", "en"],
  defaultLocale: "bn",
});

export type Locale = (typeof routing.locales)[number];

import type { Metadata } from "next";
import { Hind_Siliguri, Inter, Noto_Sans_Bengali } from "next/font/google";
import { notFound } from "next/navigation";
import { hasLocale, NextIntlClientProvider } from "next-intl";
import { setRequestLocale } from "next-intl/server";
import type { ReactNode } from "react";

import { Providers } from "@/components/providers";
import { routing } from "@/i18n/routing";

import "../globals.css";

// docs/04: Inter for Latin; Bangla glyphs fall through to Noto Sans Bengali / Hind Siliguri.
const noto = Noto_Sans_Bengali({
  subsets: ["bengali", "latin"],
  variable: "--font-noto-bengali",
  display: "swap",
});
const hind = Hind_Siliguri({
  subsets: ["bengali", "latin"],
  weight: ["400", "600"],
  variable: "--font-hind",
  display: "swap",
});
const inter = Inter({ subsets: ["latin"], variable: "--font-inter", display: "swap" });

export const metadata: Metadata = { title: "GAL-E" };

export function generateStaticParams() {
  return routing.locales.map((locale) => ({ locale }));
}

// Apply the saved theme before first paint (no flash). Reads only a "dark"/"light" flag.
const THEME_SCRIPT = `try{var t=localStorage.getItem("gale-theme");if(t==="dark"||(!t&&matchMedia("(prefers-color-scheme: dark)").matches))document.documentElement.classList.add("dark")}catch(e){}`;

export default async function LocaleLayout({
  children,
  params,
}: {
  children: ReactNode;
  params: Promise<{ locale: string }>;
}) {
  const { locale } = await params;
  if (!hasLocale(routing.locales, locale)) notFound();
  setRequestLocale(locale);
  return (
    <html
      lang={locale}
      suppressHydrationWarning
      className={`${noto.variable} ${hind.variable} ${inter.variable}`}
    >
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />
      </head>
      <body className="min-h-screen antialiased">
        <NextIntlClientProvider>
          <Providers>{children}</Providers>
        </NextIntlClientProvider>
      </body>
    </html>
  );
}

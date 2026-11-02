import type { NextConfig } from "next";
import createNextIntlPlugin from "next-intl/plugin";

const withNextIntl = createNextIntlPlugin("./src/i18n/request.ts");

const nextConfig: NextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
  // container image (infra/docker/web.Dockerfile): self-contained server, no node_modules copy
  ...(process.env.GALE_STANDALONE ? { output: "standalone" as const } : {}),
};

export default withNextIntl(nextConfig);

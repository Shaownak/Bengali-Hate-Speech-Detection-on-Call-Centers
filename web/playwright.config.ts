import { defineConfig, devices } from "@playwright/test";

// E2E against `make dev-up` (stub models) + the call simulator (Prompt 13). The stack must be up:
// tests replay synthetic fixture calls through the real ingest → workers → API → web path.
export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  workers: 1, // scenarios build on each other's calls (one dev stack, one database)
  timeout: 180_000,
  expect: { timeout: 30_000 },
  reporter: [["list"]],
  outputDir: "test-results",
  use: {
    baseURL: process.env.GALE_WEB_URL ?? "http://127.0.0.1:3000",
    viewport: { width: 1440, height: 900 },
    locale: "bn-BD",
    trace: "retain-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 900 } } }],
});

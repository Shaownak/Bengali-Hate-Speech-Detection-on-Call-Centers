import { execFile } from "node:child_process";
import { randomUUID } from "node:crypto";
import { existsSync, mkdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";

import type { Page } from "@playwright/test";

const run = promisify(execFile);
export const ROOT = fileURLToPath(new URL("../..", import.meta.url));
export const SHOTS = path.join(ROOT, "docs", "screenshots");
export const API = process.env.GALE_API_URL ?? "http://127.0.0.1:8000";

const PY =
  process.platform === "win32"
    ? path.join(ROOT, ".venv", "Scripts", "python.exe")
    : path.join(ROOT, ".venv", "bin", "python");

/** Replay a synthetic fixture call as live RTP through the gateway; returns its call id. */
export async function simulate(
  fixture: string,
  agent: string,
  speed = 4,
  callId: string = randomUUID(),
): Promise<string> {
  await run(
    PY,
    [
      "-m",
      "tools.simulator",
      path.join("tests", "fixtures", "calls", fixture),
      "--mode",
      "rtp",
      "--speed",
      String(speed),
      "--agent",
      agent,
      "--queue",
      "billing",
      "--call-id",
      callId,
    ],
    { cwd: ROOT, env: { ...process.env, PYTHONUTF8: "1" }, timeout: 170_000 },
  );
  return callId;
}

/** Run a GAL-E CLI command against the dev stack (e.g. a pilot job) and return its stdout. */
export async function python(args: string[]): Promise<string> {
  // the dev stack's embedded Postgres listens on a random port, written by `make dev-up`
  const dbUri = path.join(ROOT, ".gale", "devstack", "db_uri");
  const db = existsSync(dbUri) ? { GALE_DB_URL: readFileSync(dbUri, "utf-8").trim() } : {};
  const { stdout } = await run(PY, args, {
    cwd: ROOT,
    env: { ...process.env, ...db, GALE_ENV: "dev", PYTHONUTF8: "1" },
    timeout: 120_000,
  });
  return stdout;
}

/** ISO week (YYYY-Www) of a date in the call centre's time zone (UTC+6, no DST). */
export function isoWeekDhaka(d = new Date()): string {
  const local = new Date(d.getTime() + 6 * 3600_000);
  const t = new Date(Date.UTC(local.getUTCFullYear(), local.getUTCMonth(), local.getUTCDate()));
  t.setUTCDate(t.getUTCDate() + 4 - (t.getUTCDay() || 7)); // the week's Thursday decides the year
  const yearStart = Date.UTC(t.getUTCFullYear(), 0, 1);
  const week = Math.ceil(((t.getTime() - yearStart) / 86_400_000 + 1) / 7);
  return `${t.getUTCFullYear()}-W${String(week).padStart(2, "0")}`;
}

/** Start a simulation without waiting for it (to watch the wallboard while it plays). */
export function simulateInBackground(fixture: string, agent: string, speed = 4) {
  const callId = randomUUID();
  return { callId, done: simulate(fixture, agent, speed, callId) };
}

export async function login(page: Page, sub: string, locale = "bn") {
  await page.goto(`/${locale}/login`);
  await page.getByTestId(`login-${sub}`).click();
  await page.getByTestId("whoami").waitFor();
}

export async function apiToken(sub: string, role: string): Promise<string> {
  const r = await fetch(`${API}/api/v1/dev/token`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sub, role }),
  });
  return ((await r.json()) as { access_token: string }).access_token;
}

export async function apiGet(token: string, p: string): Promise<{ status: number; body: unknown }> {
  const r = await fetch(`${API}${p}`, { headers: { Authorization: `Bearer ${token}` } });
  return { status: r.status, body: r.ok ? await r.json() : await r.text() };
}

export async function shot(page: Page, name: string) {
  mkdirSync(SHOTS, { recursive: true });
  await page.screenshot({ path: path.join(SHOTS, `${name}.png`), fullPage: true });
}

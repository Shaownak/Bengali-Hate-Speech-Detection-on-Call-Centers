// End-to-end against the dev stack + simulator (Prompt 13). Fixture calls are synthetic; screenshots
// go to docs/screenshots/ and contain no real call data.
import { expect, test } from "@playwright/test";

import { apiGet, apiToken, isoWeekDhaka, login, python, shot, simulate, simulateInBackground } from "./helpers";

test.describe.configure({ mode: "serial" });

let liveCall = "";
let otherCall = "";

test("an alert appears on the supervisor wallboard while the call is live", async ({ page }) => {
  await login(page, "dev-supervisor");
  await page.goto("/bn/live");
  await expect(page.getByRole("heading", { name: "লাইভ ওয়ালবোর্ড" })).toBeVisible();
  const sim = simulateInBackground("demo_01", "AG-001", 4);
  liveCall = sim.callId;

  const row = page.locator(`[data-call-id="${liveCall}"]`);
  await expect(row).toBeVisible({ timeout: 60_000 });
  const alert = page.getByTestId("alert-feed").locator("li").first();
  await expect(alert).toBeVisible({ timeout: 90_000 });
  await expect(alert.locator("[data-severity]")).toBeVisible(); // icon + text severity chip
  await shot(page, "01-live-wallboard");

  await page.goto(`/bn/live/${liveCall}`);
  await expect(page.getByTestId("live-transcript").locator("li").first()).toBeVisible({ timeout: 60_000 });
  await expect(page.getByTestId("live-events").locator("li").first()).toBeVisible({ timeout: 60_000 });
  await shot(page, "02-live-call");
  await sim.done;
});

test("call detail shows waveform, transcript, scorecard and evidence", async ({ page }) => {
  await login(page, "dev-qa");
  await page.goto(`/bn/calls/${liveCall}`);
  await expect(page.getByTestId("event-table").locator("tbody tr").first()).toBeVisible();
  await expect(page.getByTestId("transcript").locator("li").first()).toBeVisible();
  await expect(page.getByTestId("scorecard")).toBeVisible({ timeout: 30_000 });
  await expect(page.getByTestId("waveform")).toBeVisible();
  await page.getByTestId("event-table").getByRole("button", { name: "বিস্তারিত" }).first().click();
  await expect(page.getByTestId("review-controls")).toBeVisible();
  await shot(page, "03-call-detail");

  await page.goto("/bn/calls");
  await expect(page.locator(`[data-call-id="${liveCall}"]`)).toBeVisible();
  await shot(page, "04-call-explorer");
});

test("a keyboard review in the queue updates the event's status", async ({ page }) => {
  await login(page, "dev-qa");
  await page.goto("/bn/review");
  const item = page.getByTestId("review-item");
  await expect(item).toBeVisible();
  const eventId = await item.getAttribute("data-event-id");
  const callId = await item.getAttribute("data-call-id");
  expect(eventId).toBeTruthy();
  await shot(page, "05-review-queue");
  await page.keyboard.press("3"); // false positive
  await expect(page.getByTestId("review-item")).not.toHaveAttribute("data-event-id", eventId!, {
    timeout: 15_000,
  });

  const token = await apiToken("dev-qa", "qa_analyst");
  const events = (await apiGet(token, `/api/v1/calls/${callId}/events`)).body as Array<{
    event_id: string;
    status: string;
    latest_verdict: string | null;
  }>;
  const ev = events.find((e) => e.event_id === eventId);
  expect(ev).toMatchObject({ status: "reviewed_false", latest_verdict: "false_positive" });
});

test("an agent cannot see another agent's calls", async ({ page }) => {
  otherCall = await simulate("demo_03", "AG-003", 20);
  await login(page, "dev-agent");
  await page.goto("/bn/me");
  await expect(page.getByTestId("calls-table")).toBeVisible();
  await expect(page.locator(`[data-call-id="${liveCall}"]`)).toBeVisible();
  await expect(page.locator(`[data-call-id="${otherCall}"]`)).toHaveCount(0);
  await shot(page, "06-agent-self-view");

  await page.goto(`/bn/calls/${otherCall}`);
  await expect(page.getByRole("alert")).toBeVisible(); // API answers 404 for out-of-scope calls
  await expect(page.getByTestId("event-table")).toHaveCount(0);

  const token = await apiToken("dev-agent", "agent");
  expect((await apiGet(token, `/api/v1/calls/${otherCall}`)).status).toBe(404);
  expect((await apiGet(token, `/api/v1/calls/${otherCall}/segments`)).status).toBe(404);
  // and the agent's own nav has no wallboard or admin screens
  await expect(page.getByRole("link", { name: "লাইভ" })).toHaveCount(0);
});

test("admin screens: analytics, rules, lexicon (masked), models, audit — and English", async ({
  page,
}) => {
  await login(page, "dev-admin", "en");
  await page.goto("/en/analytics");
  await expect(page.getByTestId("agent-scorecard")).toBeVisible();
  await shot(page, "07-analytics");
  await page.getByRole("tab", { name: "Team" }).click();
  await expect(page.getByTestId("heatmap")).toBeVisible();
  await shot(page, "08-team-heatmap");

  await page.goto("/en/admin/rules");
  await expect(page.getByTestId("rules-yaml")).toBeVisible();
  await page.getByRole("button", { name: "Simulate on last N calls" }).click();
  await expect(page.getByTestId("simulation")).toBeVisible();
  await shot(page, "09-rules-simulate");

  await page.goto("/en/admin/lexicon");
  await expect(page.getByTestId("lexicon-table").or(page.getByText("Nothing to show."))).toBeVisible();
  await shot(page, "10-lexicon");

  await page.goto("/en/admin/models");
  await expect(page.getByText("asr_postcall")).toBeVisible();
  await shot(page, "11-models");

  await page.goto("/en/admin/audit");
  await expect(page.getByTestId("audit-table")).toBeVisible();
  await expect(page.getByTestId("audit-table")).toContainText("transcript.view");
  await shot(page, "12-audit");
});

test("pilot: QA reports a miss, a supervisor rates the week, the weekly report shows exit criteria", async ({
  browser,
}) => {
  const qa = await (await browser.newContext()).newPage();
  await login(qa, "dev-qa", "en");
  await qa.goto(`/en/calls/${liveCall}`);
  await qa.getByRole("button", { name: "Report a miss" }).click();
  await qa.getByRole("button", { name: "Save" }).click();
  await expect(qa.getByTestId("missed-events")).toContainText("Insult");

  const sup = await (await browser.newContext()).newPage();
  await login(sup, "dev-supervisor", "en");
  await sup.goto("/en/analytics");
  await sup.getByTestId("usefulness").getByRole("radio", { name: "4" }).click();
  await expect(sup.getByTestId("usefulness").getByRole("status")).toContainText("4/5");

  const week = isoWeekDhaka();
  const out = JSON.parse(await python(["-m", "gale_api", "pilot-report", "--week", week])) as {
    week: string;
    exit: Record<string, string>;
  };
  expect(out.week).toBe(week);

  const admin = await (await browser.newContext()).newPage();
  await login(admin, "dev-admin", "en");
  await admin.goto("/en/admin/pilot");
  await expect(admin.getByTestId("pilot-stage")).toHaveText("P-3"); // dev default: rollout
  const card = admin.locator(`[data-testid="pilot-report"][data-week="${week}"]`);
  await expect(card).toContainText("Exit criteria");
  await expect(card).toContainText("pilot lead"); // P-3 criteria are signed off by the pilot lead
  await card.getByRole("button", { name: "Open full report" }).click();
  await expect(admin.locator("iframe")).toBeVisible();
  await shot(admin, "13-pilot");
  await admin.goto(`/en/calls/${liveCall}`);
  await expect(admin.getByTestId("missed-events")).toContainText("Insult");
});

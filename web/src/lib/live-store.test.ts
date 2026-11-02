import { describe, expect, it } from "vitest";

import type { LiveItem } from "@/lib/live-socket";
import { applyItems, applySnapshot, emptyLive, kpis, riskOf } from "@/lib/live-store";

const T0 = 1_800_000_000_000;

function ev(over: Record<string, unknown> = {}): LiveItem {
  return {
    stream: "events.fused",
    data: {
      event_id: "c1:e1",
      call_id: "c1",
      speaker: "agent",
      category: "insult",
      severity: 2,
      probability: 0.7,
      ts_start: 3,
      ts_end: 4,
      status: "provisional",
      alert: false,
      ...over,
    },
  };
}

const started: LiveItem = {
  stream: "call.lifecycle",
  data: { call_id: "c1", event: "started", ts: 0, wall_clock: T0 / 1000, agent_id: "AG-001", queue: "billing" },
};

describe("wallboard reducers", () => {
  it("tracks call start/end from call.lifecycle", () => {
    let d = applyItems(emptyLive(), [started], T0);
    expect(d.calls.c1).toMatchObject({ agent: "AG-001", queue: "billing", startedAt: T0 });
    d = applyItems(d, [{ stream: "call.lifecycle", data: { call_id: "c1", event: "ended", ts: 30 } }], T0);
    expect(d.calls.c1).toBeUndefined();
  });

  it("de-duplicates alerts per event and keeps the latest state", () => {
    let d = applyItems(emptyLive(), [started, ev()], T0);
    expect(Object.keys(d.alerts)).toEqual([]); // not an alert yet
    d = applyItems(d, [ev({ alert: true }), ev({ alert: true, status: "confirmed" })], T0 + 100);
    expect(Object.keys(d.alerts)).toEqual(["c1:e1"]);
    expect(d.alerts["c1:e1"]?.event.status).toBe("confirmed");
    expect(d.alerts["c1:e1"]?.receivedAt).toBe(T0 + 100);
    // a later update without the alert flag (e.g. dismissed) stays in the feed, still flagged
    d = applyItems(d, [ev({ alert: false, status: "dismissed_by_llm" })], T0 + 200);
    expect(d.alerts["c1:e1"]?.event).toMatchObject({ alert: true, status: "dismissed_by_llm" });
  });

  it("risk meters are a rolling 60 s max and drop dismissed events", () => {
    let d = applyItems(emptyLive(), [started, ev({ probability: 0.9 })], T0);
    const call = () => d.calls.c1!;
    expect(riskOf(call(), "agent", T0 + 1000)?.p).toBe(0.9);
    expect(riskOf(call(), "customer", T0 + 1000)).toBeUndefined();
    expect(riskOf(call(), "agent", T0 + 61_000)).toBeUndefined(); // decayed out of the window
    d = applyItems(d, [ev({ status: "dismissed_by_llm" })], T0 + 2000);
    expect(riskOf(call(), "agent", T0 + 2000)).toBeUndefined();
  });

  it("builds state from the snapshot and computes KPIs", () => {
    const d = applySnapshot(
      {
        calls: [{ call_id: "c1", started_at: new Date(T0).toISOString(), queue: "q", agent_id: "u1", mode: "realtime" }],
        alerts: [ev({ alert: true }).data, ev({ event_id: "c1:e2", speaker: "customer", alert: true }).data],
      },
      T0,
    );
    expect(Object.keys(d.calls)).toEqual(["c1"]);
    d.alerts["c1:e1"]!.ackedAt = T0 + 4000;
    d.alerts["c1:e2"]!.ackedAt = T0 + 8000;
    expect(kpis(d, T0 + 10_000)).toEqual({
      active: 1,
      alertsLastHour: 2,
      agentCaused: 1,
      customerCaused: 1,
      medianAckMs: 6000,
    });
    expect(kpis(d, T0 + 3_700_000).alertsLastHour).toBe(0);
  });
});

describe("wallboard load (docs/04 §4: 1,000 active calls at 2 updates/s)", () => {
  it("applies one 250 ms batch (500 updates) over 1,000 calls well inside a frame budget", () => {
    let d = emptyLive();
    const starts: LiveItem[] = Array.from({ length: 1000 }, (_, i) => ({
      stream: "call.lifecycle",
      data: { call_id: `c${i}`, event: "started", ts: 0, agent_id: `AG-${i}` },
    }));
    d = applyItems(d, starts, T0);
    const batch = (k: number): LiveItem[] =>
      Array.from({ length: 500 }, (_, j) => {
        const i = (k * 500 + j) % 1000;
        return ev({ event_id: `c${i}:e${k}`, call_id: `c${i}`, probability: 0.5 + (j % 50) / 100, alert: j % 10 === 0 });
      });
    const t0 = performance.now();
    for (let k = 0; k < 20; k++) d = applyItems(d, batch(k), T0 + k * 250); // 5 s of traffic
    const perBatch = (performance.now() - t0) / 20;
    expect(Object.keys(d.calls)).toHaveLength(1000);
    expect(perBatch).toBeLessThan(50);
  });
});

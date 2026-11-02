// Wallboard state (docs/04 §2.1) fed by /ws/live. The reducers are pure (tested in isolation);
// the Zustand store wires them to a LiveSocket.
import { create } from "zustand";

import type { LiveItem, SocketStatus } from "@/lib/live-socket";
import { median } from "@/lib/format";
import { LiveSocket } from "@/lib/live-socket";
import type { FusedEvent } from "@/types/generated";

export const RISK_WINDOW_MS = 60_000;

export interface RiskPoint {
  eventId: string;
  t: number; // wall time received (ms)
  p: number;
  severity: number;
  category: string;
}

export interface LiveCall {
  callId: string;
  startedAt: number; // ms epoch
  queue?: string | null;
  agent?: string | null; // agent uuid (snapshot) or employee code (lifecycle)
  mode?: string | null;
  risk: { agent: RiskPoint[]; customer: RiskPoint[] };
}

export interface LiveAlert {
  event: FusedEvent;
  receivedAt: number;
  ackedAt?: number;
  falseAlarm?: boolean;
}

export interface LiveData {
  calls: Record<string, LiveCall>;
  alerts: Record<string, LiveAlert>;
}

export const emptyLive = (): LiveData => ({ calls: {}, alerts: {} });

interface SnapCall {
  call_id: string;
  started_at: string;
  queue?: string | null;
  agent_id?: string | null;
  mode?: string | null;
}

export function applySnapshot(snap: Record<string, unknown>, now: number): LiveData {
  const out = emptyLive();
  for (const c of (snap.calls as SnapCall[] | undefined) ?? []) {
    out.calls[c.call_id] = {
      callId: c.call_id,
      startedAt: Date.parse(c.started_at),
      queue: c.queue,
      agent: c.agent_id,
      mode: c.mode,
      risk: { agent: [], customer: [] },
    };
  }
  for (const e of (snap.alerts as FusedEvent[] | undefined) ?? []) {
    out.alerts[e.event_id] = { event: e, receivedAt: now };
    addRisk(out, e, now);
  }
  return out;
}

function addRisk(d: LiveData, e: FusedEvent, now: number): void {
  const call = d.calls[e.call_id];
  if (!call) return;
  const lane = call.risk[e.speaker];
  const kept = lane.filter((r) => r.eventId !== e.event_id && now - r.t <= RISK_WINDOW_MS);
  if (e.status !== "dismissed_by_llm" && e.status !== "reviewed_false") {
    kept.push({
      eventId: e.event_id,
      t: now,
      p: e.probability,
      severity: e.severity,
      category: e.category,
    });
  }
  call.risk = { ...call.risk, [e.speaker]: kept };
}

/** Apply one 250 ms batch. Returns a new object (immutable update for React). */
export function applyItems(prev: LiveData, items: LiveItem[], now: number): LiveData {
  const d: LiveData = { calls: { ...prev.calls }, alerts: { ...prev.alerts } };
  for (const { stream, data } of items) {
    if (stream === "call.lifecycle") {
      const id = String(data.call_id);
      if (data.event === "started") {
        const wall = typeof data.wall_clock === "number" ? data.wall_clock * 1000 : now;
        d.calls[id] = {
          callId: id,
          startedAt: wall,
          queue: (data.queue as string | null) ?? null,
          agent: (data.agent_id as string | null) ?? null,
          mode: (data.mode as string | null) ?? null,
          risk: { agent: [], customer: [] },
        };
      } else if (data.event === "ended" || data.event === "hangup") {
        delete d.calls[id];
      }
    } else if (stream === "events.fused" || stream === "alerts") {
      const e = data as unknown as FusedEvent;
      const existing = d.calls[e.call_id];
      if (existing) d.calls[e.call_id] = { ...existing };
      addRisk(d, e, now);
      const known = d.alerts[e.event_id];
      if (e.alert || stream === "alerts" || known) {
        // de-duplicated per event: later updates (provisional → confirmed / dismissed) replace it
        d.alerts[e.event_id] = known
          ? { ...known, event: { ...e, alert: known.event.alert || e.alert } }
          : { event: e, receivedAt: now };
      }
    }
  }
  return d;
}

export function riskOf(call: LiveCall, speaker: "agent" | "customer", now: number): RiskPoint | undefined {
  let best: RiskPoint | undefined;
  for (const r of call.risk[speaker]) {
    if (now - r.t <= RISK_WINDOW_MS && (!best || r.p > best.p)) best = r;
  }
  return best;
}

export interface Kpis {
  active: number;
  alertsLastHour: number;
  agentCaused: number;
  customerCaused: number;
  medianAckMs?: number;
}

export function kpis(d: LiveData, now: number): Kpis {
  const recent = Object.values(d.alerts).filter((a) => now - a.receivedAt <= 3_600_000);
  const acks = Object.values(d.alerts)
    .filter((a) => a.ackedAt !== undefined)
    .map((a) => (a.ackedAt as number) - a.receivedAt);
  return {
    active: Object.keys(d.calls).length,
    alertsLastHour: recent.length,
    agentCaused: recent.filter((a) => a.event.speaker === "agent").length,
    customerCaused: recent.filter((a) => a.event.speaker === "customer").length,
    medianAckMs: median(acks),
  };
}

interface LiveState extends LiveData {
  status: SocketStatus;
  connected: boolean;
  setConnected: (connected: boolean) => void;
  connect: (getToken: () => string | undefined) => () => void;
  markAcked: (eventId: string, at?: number) => void;
  markFalseAlarm: (eventId: string) => void;
  onNewAlert?: (a: LiveAlert) => void;
  setOnNewAlert: (fn?: (a: LiveAlert) => void) => void;
}

export const useLiveStore = create<LiveState>((set, get) => ({
  ...emptyLive(),
  status: "idle",
  connected: false,
  setConnected: (connected) => set({ connected }),
  setOnNewAlert: (fn) => set({ onNewAlert: fn }),
  connect: (getToken) => {
    const sock = new LiveSocket("/ws/live", getToken, {
      onStatus: (status) => set({ status, connected: status === "open" }),
      onSnapshot: (snap) => set(applySnapshot(snap, Date.now())),
      onBatch: (items) => {
        const before = get().alerts;
        const next = applyItems(get(), items, Date.now());
        set(next);
        const notify = get().onNewAlert;
        if (notify) {
          for (const [id, a] of Object.entries(next.alerts)) if (!before[id]) notify(a);
        }
      },
    });
    sock.start();
    return () => sock.stop();
  },
  markAcked: (eventId, at = Date.now()) =>
    set((s) => {
      const a = s.alerts[eventId];
      return a ? { alerts: { ...s.alerts, [eventId]: { ...a, ackedAt: a.ackedAt ?? at } } } : s;
    }),
  markFalseAlarm: (eventId) =>
    set((s) => {
      const a = s.alerts[eventId];
      return a ? { alerts: { ...s.alerts, [eventId]: { ...a, falseAlarm: true } } } : s;
    }),
}));

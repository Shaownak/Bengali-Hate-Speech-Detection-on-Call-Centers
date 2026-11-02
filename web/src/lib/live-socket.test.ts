import { afterEach, describe, expect, it, vi } from "vitest";

import { backoff, LiveSocket, type SocketStatus } from "@/lib/live-socket";

class FakeWs {
  static all: FakeWs[] = [];
  sent: string[] = [];
  onopen?: () => void;
  onmessage?: (e: { data: string }) => void;
  onclose?: (e: { code: number }) => void;
  constructor(readonly url: string) {
    FakeWs.all.push(this);
  }
  send(s: string) {
    this.sent.push(s);
  }
  close() {}
  open() {
    this.onopen?.();
  }
  msg(m: object) {
    this.onmessage?.({ data: JSON.stringify(m) });
  }
}

afterEach(() => {
  FakeWs.all = [];
  vi.useRealTimers();
});

function make() {
  const statuses: SocketStatus[] = [];
  const snaps: unknown[] = [];
  const batches: unknown[] = [];
  const s = new LiveSocket(
    "/ws/live",
    () => "tok",
    {
      onStatus: (st) => statuses.push(st),
      onSnapshot: (x) => snaps.push(x),
      onBatch: (x) => batches.push(x),
    },
    FakeWs as unknown as new (url: string) => WebSocket,
    "https://api.test",
  );
  return { s, statuses, snaps, batches };
}

describe("LiveSocket", () => {
  it("backoff doubles, caps and jitters", () => {
    expect(backoff(0, 1000, 30_000, () => 1)).toBe(1000);
    expect(backoff(3, 1000, 30_000, () => 1)).toBe(8000);
    expect(backoff(10, 1000, 30_000, () => 1)).toBe(30_000);
    expect(backoff(3, 1000, 30_000, () => 0)).toBe(4000);
  });

  it("authenticates with the first message, never the URL", () => {
    const { s, statuses, snaps, batches } = make();
    s.start();
    const ws = FakeWs.all[0]!;
    expect(ws.url).toBe("wss://api.test/ws/live");
    ws.open();
    expect(JSON.parse(ws.sent[0]!)).toEqual({ type: "auth", token: "tok" });
    ws.msg({ type: "snapshot", calls: [] });
    ws.msg({ type: "batch", seq: 1, items: [{ stream: "alerts", data: {} }] });
    expect(statuses).toEqual(["connecting", "open"]);
    expect(snaps).toHaveLength(1);
    expect(batches).toHaveLength(1);
  });

  it("reconnects with backoff after a drop, but not after a refusal", () => {
    vi.useFakeTimers();
    const { s, statuses } = make();
    s.start();
    FakeWs.all[0]!.onclose?.({ code: 1006 });
    expect(statuses.at(-1)).toBe("reconnecting");
    vi.advanceTimersByTime(1100);
    expect(FakeWs.all).toHaveLength(2);
    FakeWs.all[1]!.onclose?.({ code: 4403 });
    expect(statuses.at(-1)).toBe("denied");
    vi.advanceTimersByTime(60_000);
    expect(FakeWs.all).toHaveLength(2);
    s.stop();
    expect(statuses.at(-1)).toBe("closed");
  });
});

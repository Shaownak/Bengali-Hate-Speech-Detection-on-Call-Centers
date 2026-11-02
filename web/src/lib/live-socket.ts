// WebSocket client for /ws/live and /ws/calls/{id} (docs/04 §3, ADR-016): sends the token as the
// first message, then receives one snapshot and 250 ms batches. Reconnects with exponential
// backoff + jitter; auth/scope refusals (4401/4403/4404) are final.
import { wsUrl } from "@/lib/api";

export type SocketStatus = "idle" | "connecting" | "open" | "reconnecting" | "denied" | "closed";

export interface LiveItem {
  stream: string;
  data: Record<string, unknown>;
}

export interface SocketHandlers {
  onSnapshot: (snap: Record<string, unknown>) => void;
  onBatch: (items: LiveItem[]) => void;
  onStatus: (status: SocketStatus, code?: number) => void;
}

export const FINAL_CODES = new Set([4401, 4403, 4404]);

/** Delay before reconnect attempt ``attempt`` (0-based): base·2^n capped, with ±50 % jitter. */
export function backoff(attempt: number, base = 1000, max = 30_000, rand = Math.random): number {
  const d = Math.min(max, base * 2 ** attempt);
  return Math.round(d * (0.5 + rand() / 2));
}

type WsCtor = new (url: string) => WebSocket;

export class LiveSocket {
  private ws?: WebSocket;
  private attempt = 0;
  private timer?: ReturnType<typeof setTimeout>;
  private stopped = true;

  constructor(
    private readonly path: string,
    private readonly getToken: () => string | undefined,
    private readonly handlers: SocketHandlers,
    private readonly Ws: WsCtor = WebSocket,
    private readonly base?: string,
  ) {}

  start(): void {
    this.stopped = false;
    this.connect();
  }

  stop(): void {
    this.stopped = true;
    clearTimeout(this.timer);
    this.ws?.close(1000);
    this.ws = undefined;
    this.handlers.onStatus("closed");
  }

  private connect(): void {
    const token = this.getToken();
    if (!token) {
      this.handlers.onStatus("denied", 4401);
      return;
    }
    this.handlers.onStatus(this.attempt ? "reconnecting" : "connecting");
    const ws = new this.Ws(this.base ? wsUrl(this.path, this.base) : wsUrl(this.path));
    this.ws = ws;
    ws.onopen = () => ws.send(JSON.stringify({ type: "auth", token }));
    ws.onmessage = (ev: MessageEvent) => {
      const msg = JSON.parse(String(ev.data)) as { type: string; items?: LiveItem[] };
      if (msg.type === "snapshot") {
        this.attempt = 0;
        this.handlers.onStatus("open");
        this.handlers.onSnapshot(msg as Record<string, unknown>);
      } else if (msg.type === "batch" && msg.items) {
        this.handlers.onBatch(msg.items);
      }
    };
    ws.onclose = (ev: CloseEvent) => {
      if (this.stopped || this.ws !== ws) return;
      if (FINAL_CODES.has(ev.code)) {
        this.handlers.onStatus("denied", ev.code);
        return;
      }
      this.handlers.onStatus("reconnecting", ev.code);
      this.timer = setTimeout(() => this.connect(), backoff(this.attempt++));
    };
  }
}

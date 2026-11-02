// One live call (docs/04 §2.2) fed by /ws/calls/{id}: two transcript lanes with partial text,
// events with provisional → confirmed transitions, and loudness / anger / overtalk timelines.
import type { LiveItem } from "@/lib/live-socket";
import type { FusedEvent } from "@/types/generated";

export interface LiveSegment {
  id: string;
  channel: string;
  tsStart: number;
  tsEnd: number;
  text: string;
  final: boolean;
}

export interface Marker {
  t: number;
  channel: string;
  kind: "register" | "overtalk";
}

export interface CallLive {
  status: string;
  segments: Record<string, LiveSegment>;
  events: Record<string, FusedEvent>;
  loudness: Record<string, Array<[number, number]>>;
  anger: Record<string, Array<[number, number]>>;
  markers: Marker[];
}

export const emptyCall = (): CallLive => ({
  status: "active",
  segments: {},
  events: {},
  loudness: {},
  anger: {},
  markers: [],
});

interface SnapSeg {
  segment_id: string;
  channel: string;
  ts_start: number;
  ts_end: number;
  text_redacted: string;
}

export function callFromSnapshot(snap: Record<string, unknown>): CallLive {
  const c = emptyCall();
  c.status = String(snap.status ?? "active");
  for (const g of (snap.segments as SnapSeg[] | undefined) ?? []) {
    c.segments[g.segment_id] = {
      id: g.segment_id,
      channel: g.channel,
      tsStart: g.ts_start,
      tsEnd: g.ts_end,
      text: g.text_redacted,
      final: true,
    };
  }
  for (const e of (snap.events as FusedEvent[] | undefined) ?? []) c.events[e.event_id] = e;
  return c;
}

const MAX_POINTS = 600;

function push(series: Record<string, Array<[number, number]>>, ch: string, t: number, v: number) {
  const arr = [...(series[ch] ?? []), [t, v] as [number, number]];
  series[ch] = arr.length > MAX_POINTS ? arr.slice(-MAX_POINTS) : arr;
}

export function callApply(prev: CallLive, items: LiveItem[]): CallLive {
  const c: CallLive = {
    ...prev,
    segments: { ...prev.segments },
    events: { ...prev.events },
    loudness: { ...prev.loudness },
    anger: { ...prev.anger },
    markers: [...prev.markers],
  };
  for (const { stream, data } of items) {
    if (stream === "asr.segments") {
      const id = String(data.segment_id);
      if (c.segments[id]?.final && !data.is_final) continue; // never regress final → partial
      c.segments[id] = {
        id,
        channel: String(data.channel),
        tsStart: Number(data.ts_start),
        tsEnd: Number(data.ts_end),
        text: String(data.text_redacted ?? ""),
        final: Boolean(data.is_final),
      };
    } else if (stream === "events.fused" || stream === "alerts") {
      const e = data as unknown as FusedEvent;
      const known = c.events[e.event_id];
      c.events[e.event_id] = known ? { ...e, alert: known.alert || e.alert } : e;
    } else if (stream === "signals") {
      const ch = String(data.channel);
      const t = Number(data.ts_start);
      const score = Number(data.score);
      if (data.source === "dsp_shout") push(c.loudness, ch, t, score);
      else if (data.source === "ser") push(c.anger, ch, t, score);
      else if (data.source === "register") c.markers.push({ t, channel: ch, kind: "register" });
      else if (data.source === "vad_overtalk") c.markers.push({ t, channel: ch, kind: "overtalk" });
    } else if (stream === "call.lifecycle") {
      if (data.event === "ended" || data.event === "hangup") c.status = "ended";
    }
  }
  return c;
}

export function lane(c: CallLive, channel: string): LiveSegment[] {
  return Object.values(c.segments)
    .filter((s) => s.channel === channel)
    .sort((a, b) => a.tsStart - b.tsStart);
}

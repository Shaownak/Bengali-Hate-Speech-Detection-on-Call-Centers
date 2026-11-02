"use client";

import { Pause, Play } from "lucide-react";
import { useTranslations } from "next-intl";
import { useEffect, useRef, useState } from "react";
import type WaveSurfer from "wavesurfer.js";

import { Button } from "@/components/ui/button";
import { clock } from "@/lib/format";

export interface PlayerApi {
  seek: (t: number) => void;
  /** Play [t0, t1] and stop (review: Space plays the span ±3 s). */
  playSpan: (t0: number, t1: number) => void;
  toggle: () => void;
}

export interface RegionSpec {
  id: string;
  start: number;
  end: number;
  channel: "agent" | "customer";
  color: string;
  label: string;
}

type Channel = "agent" | "customer";
const CHANNEL_COLOR: Record<Channel, string> = { agent: "#2563eb", customer: "#dc2626" };

/**
 * Stereo waveform with event regions (docs/04 §2.3). An uploaded recording is one stereo file
 * (split channels); gateway-retained live audio is one file per channel, played in sync.
 */
export function Waveform({
  urls,
  regions,
  onTime,
  onReady,
}: {
  urls: Record<string, string>;
  regions: RegionSpec[];
  onTime?: (t: number) => void;
  onReady?: (api: PlayerApi) => void;
}) {
  const t = useTranslations("call");
  const ts = useTranslations("speaker");
  const refs = useRef<Record<string, HTMLDivElement | null>>({});
  const players = useRef<WaveSurfer[]>([]);
  const [playing, setPlaying] = useState(false);
  const [now, setNow] = useState(0);
  const lanes: Array<"stereo" | Channel> = urls.stereo
    ? ["stereo"]
    : (["agent", "customer"] as const).filter((c) => urls[c]);
  const key = JSON.stringify(urls);

  useEffect(() => {
    let cancelled = false;
    let stopAt: number | undefined;
    (async () => {
      const { default: WS } = await import("wavesurfer.js");
      const { default: Regions } = await import("wavesurfer.js/dist/plugins/regions.esm.js");
      if (cancelled) return;
      const made: WaveSurfer[] = [];
      for (const lane of lanes) {
        const container = refs.current[lane];
        const url = urls[lane];
        if (!container || !url) continue;
        const reg = Regions.create();
        const ws = WS.create({
          container,
          url,
          height: lane === "stereo" ? 64 : 56,
          plugins: [reg],
          ...(lane === "stereo"
            ? {
                splitChannels: [
                  { waveColor: CHANNEL_COLOR.agent, progressColor: "#1e3a8a" },
                  { waveColor: CHANNEL_COLOR.customer, progressColor: "#7f1d1d" },
                ],
              }
            : { waveColor: CHANNEL_COLOR[lane as Channel], progressColor: "#444" }),
        });
        ws.on("decode", () => {
          for (const r of regions) {
            if (lane !== "stereo" && r.channel !== lane) continue;
            reg.addRegion({
              id: `${lane}-${r.id}`,
              start: r.start,
              end: r.end,
              color: r.color,
              content: r.label,
              drag: false,
              resize: false,
            });
          }
        });
        reg.on("region-clicked", (region, e) => {
          e.stopPropagation();
          api.playSpan(region.start, region.end);
        });
        made.push(ws);
      }
      players.current = made;
      const lead = made[0];
      if (!lead) return;
      lead.on("timeupdate", (time) => {
        setNow(time);
        onTime?.(time);
        if (stopAt !== undefined && time >= stopAt) {
          stopAt = undefined;
          made.forEach((p) => p.pause());
        }
      });
      lead.on("play", () => setPlaying(true));
      lead.on("pause", () => setPlaying(false));
      lead.on("interaction", (time) => made.slice(1).forEach((p) => p.setTime(time)));
      const api: PlayerApi = {
        seek: (time) => made.forEach((p) => p.setTime(time)),
        playSpan: (t0, t1) => {
          made.forEach((p) => p.setTime(Math.max(0, t0)));
          stopAt = t1;
          made.forEach((p) => void p.play());
        },
        toggle: () => made.forEach((p) => void p.playPause()),
      };
      onReady?.(api);
    })();
    return () => {
      cancelled = true;
      players.current.forEach((p) => p.destroy());
      players.current = [];
    };
    // regions/onTime/onReady are read once per audio load
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);

  return (
    <div className="flex flex-col gap-2" data-testid="waveform">
      {lanes.map((lane) => (
        <div key={lane}>
          {lane !== "stereo" && (
            <span className="text-xs" style={{ color: CHANNEL_COLOR[lane as Channel] }}>
              {ts(lane)}
            </span>
          )}
          <div ref={(el) => void (refs.current[lane] = el)} />
        </div>
      ))}
      <div className="flex items-center gap-2 text-sm">
        <Button
          size="sm"
          variant="outline"
          onClick={() => players.current.forEach((p) => void p.playPause())}
          aria-label={playing ? t("pause") : t("play")}
        >
          {playing ? <Pause className="size-4" aria-hidden /> : <Play className="size-4" aria-hidden />}
        </Button>
        <span className="tabular-nums">{clock(now)}</span>
      </div>
    </div>
  );
}

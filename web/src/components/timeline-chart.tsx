"use client";

import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { clock } from "@/lib/format";

export interface Series {
  key: string;
  label: string;
  color: string;
  points: Array<[number, number | null]>;
  dashed?: boolean;
}

/** Score series over call time (0–1), with optional vertical markers (overtalk, register). */
export function TimelineChart({
  series,
  markers = [],
  height = 180,
  onSeek,
  ariaLabel,
}: {
  series: Series[];
  markers?: Array<{ t: number; label: string }>;
  height?: number;
  onSeek?: (t: number) => void;
  ariaLabel: string;
}) {
  const rows = new Map<number, Record<string, number | null>>();
  for (const s of series) {
    for (const [t, v] of s.points) {
      const r = rows.get(t) ?? { t };
      r[s.key] = v;
      rows.set(t, r);
    }
  }
  const data = [...rows.values()].sort((a, b) => (a.t ?? 0) - (b.t ?? 0));
  return (
    <figure aria-label={ariaLabel} style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart
          data={data}
          margin={{ top: 4, right: 8, left: -20, bottom: 0 }}
          onClick={(e) => {
            const label = (e as { activeLabel?: number | string } | null)?.activeLabel;
            if (onSeek && label !== undefined) onSeek(Number(label));
          }}
        >
          <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
          <XAxis
            dataKey="t"
            type="number"
            domain={["dataMin", "dataMax"]}
            tickFormatter={(v: number) => clock(v)}
            fontSize={11}
          />
          <YAxis domain={[0, 1]} fontSize={11} />
          <Tooltip labelFormatter={(v) => clock(Number(v))} />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          {markers.map((m, i) => (
            <ReferenceLine key={i} x={m.t} stroke="#888" strokeDasharray="2 2" label={m.label} />
          ))}
          {series.map((s) => (
            <Line
              key={s.key}
              type="monotone"
              dataKey={s.key}
              name={s.label}
              stroke={s.color}
              strokeDasharray={s.dashed ? "4 3" : undefined}
              dot={false}
              connectNulls
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </figure>
  );
}

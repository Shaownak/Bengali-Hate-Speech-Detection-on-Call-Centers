"use client";

import { Fragment, useEffect, useRef } from "react";

import { SpeakerLabel } from "@/components/badges";
import type { Schemas } from "@/lib/api";
import { clock } from "@/lib/format";
import { cn } from "@/lib/utils";

const TAG = /(<[A-Z][A-Z0-9_]*>)/;

/** Redacted text with PII placeholders (<PHONE>, <NAME>…) rendered as tags (docs/04 §2.3). */
export function RedactedText({ text }: { text: string }) {
  return (
    <>
      {text.split(TAG).map((part, i) =>
        TAG.test(part) ? (
          <span
            key={i}
            className="mx-0.5 rounded bg-muted px-1 font-mono text-[0.7rem]"
            data-pii-tag={part}
          >
            {part.slice(1, -1)}
          </span>
        ) : (
          <Fragment key={i}>{part}</Fragment>
        ),
      )}
    </>
  );
}

/** Index of the word playing at ``t`` (or -1). Exported for tests. */
export function activeWord(words: ReadonlyArray<{ start: number; end: number }>, t: number): number {
  return words.findIndex((w) => t >= w.start && t < w.end);
}

/** Chronological transcript, following playback (word-level highlight when words exist). */
export function Transcript({
  segments,
  time,
  onSeek,
  flagged,
}: {
  segments: Schemas["SegmentOut"][];
  time: number;
  onSeek?: (t: number) => void;
  flagged?: (seg: Schemas["SegmentOut"]) => number | undefined;
}) {
  const active = useRef<HTMLLIElement>(null);
  const current = segments.find((s) => time >= s.ts_start && time < s.ts_end)?.id;
  useEffect(() => {
    active.current?.scrollIntoView({ block: "nearest" });
  }, [current]);
  return (
    <ol className="flex max-h-[50vh] flex-col gap-1 overflow-auto text-sm" data-testid="transcript">
      {segments.map((s) => {
        const isCurrent = s.id === current;
        const words = s.words ?? [];
        const w = isCurrent ? activeWord(words, time) : -1;
        const sev = flagged?.(s);
        return (
          <li
            key={s.id}
            ref={isCurrent ? active : undefined}
            className={cn(
              "grid grid-cols-[4rem_5rem_1fr] gap-2 rounded px-1",
              isCurrent && "bg-muted",
              sev && "underline decoration-red-600 decoration-2 underline-offset-4",
              sev === 3 && "decoration-double",
            )}
            data-segment-id={s.id}
          >
            <button
              type="button"
              className="text-left tabular-nums opacity-70 hover:underline"
              onClick={() => onSeek?.(s.ts_start)}
            >
              {clock(s.ts_start)}
            </button>
            <span className={s.channel === "agent" ? "text-blue-700 dark:text-blue-300" : "text-red-700 dark:text-red-300"}>
              <SpeakerLabel speaker={s.channel} />
            </span>
            <span>
              {w >= 0 && words.length ? (
                words.map((word, i) => (
                  <span key={i} className={cn(i === w && "rounded bg-yellow-200 dark:bg-yellow-700")}>
                    <RedactedText text={word.word} />{" "}
                  </span>
                ))
              ) : (
                <RedactedText text={s.text_redacted} />
              )}
            </span>
          </li>
        );
      })}
    </ol>
  );
}

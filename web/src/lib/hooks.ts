"use client";

import { useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";

import type { Schemas } from "@/lib/api";
import { useApi, unwrap } from "@/lib/use-api";

/** Wall-clock that re-renders every ``ms`` (durations, risk decay). */
export function useNow(ms = 1000): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), ms);
    return () => clearInterval(id);
  }, [ms]);
  return now;
}

/** A boolean preference kept in localStorage (per browser, not per user). */
export function usePref(key: string, initial: boolean): [boolean, (v: boolean) => void] {
  const [v, setV] = useState(initial);
  useEffect(() => {
    try {
      const s = localStorage.getItem(key);
      if (s !== null) setV(s === "1");
    } catch {
      /* storage unavailable */
    }
  }, [key]);
  const set = (next: boolean) => {
    setV(next);
    try {
      localStorage.setItem(key, next ? "1" : "0");
    } catch {
      /* storage unavailable */
    }
  };
  return [v, set];
}

export type Directory = Schemas["Directory"];

/** Teams and agents the caller may see (filters, names). */
export function useDirectory() {
  const api = useApi();
  return useQuery({
    queryKey: ["directory"],
    queryFn: async () => unwrap(await api.GET("/api/v1/directory")),
    staleTime: 60_000,
  });
}

/** Resolve an agent reference (uuid from the DB or employee code from the gateway). */
export function agentLookup(dir: Directory | undefined) {
  const byKey = new Map<string, Schemas["AgentOut"]>();
  for (const a of dir?.agents ?? []) {
    byKey.set(a.id, a);
    byKey.set(a.employee_code, a);
  }
  return (ref: string | null | undefined) => (ref ? byKey.get(ref) : undefined);
}

/** Short beep for severity-3 alerts (Web Audio; no asset to load). */
export function beep(): void {
  try {
    const ctx = new AudioContext();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.frequency.value = 880;
    gain.gain.value = 0.1;
    osc.connect(gain).connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + 0.25);
  } catch {
    /* audio unavailable */
  }
}

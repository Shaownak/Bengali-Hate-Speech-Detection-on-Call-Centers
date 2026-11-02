// Session for the dashboard: the API access token + who the user is (GET /api/v1/me).
// Dev mode signs in through the API's dev issuer; OIDC mode uses Keycloak (see oidc.ts).
// The token lives in sessionStorage only (cleared when the tab closes), never in a URL.
import { useEffect, useState } from "react";
import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";

import type { Schemas } from "@/lib/api";
import { API_BASE } from "@/lib/api";

export type Me = Schemas["MeOut"];
export type Role = "admin" | "qa_analyst" | "supervisor" | "agent" | "auditor";
export type AuthMode = "dev" | "oidc";

export const AUTH_MODE: AuthMode =
  process.env.NEXT_PUBLIC_GALE_AUTH_MODE === "oidc" ? "oidc" : "dev";

/** Seeded dev users (services/api/src/gale_api/seed.py). */
export const DEV_USERS: ReadonlyArray<{ sub: string; role: Role }> = [
  { sub: "dev-supervisor", role: "supervisor" },
  { sub: "dev-qa", role: "qa_analyst" },
  { sub: "dev-admin", role: "admin" },
  { sub: "dev-admin2", role: "admin" },
  { sub: "dev-agent", role: "agent" },
  { sub: "dev-supervisor-ctg", role: "supervisor" },
  { sub: "dev-agent-3", role: "agent" },
  { sub: "dev-auditor", role: "auditor" },
];

interface AuthState {
  token?: string;
  me?: Me;
  setSession: (token: string, me: Me) => void;
  clear: () => void;
}

export const useAuth = create<AuthState>()(
  persist(
    (set) => ({
      setSession: (token, me) => set({ token, me }),
      clear: () => set({ token: undefined, me: undefined }),
    }),
    {
      name: "gale-session",
      storage: createJSONStorage(() => sessionStorage),
      partialize: (s) => ({ token: s.token, me: s.me }),
    },
  ),
);

export function can(me: Me | undefined, perm: string): boolean {
  return !!me && me.permissions.includes(perm);
}

/** Landing page per role (docs/04 §2: the wallboard is the supervisor default). */
export function homeFor(me: Me): string {
  switch (me.role) {
    case "supervisor":
      return "/live";
    case "qa_analyst":
      return "/review";
    case "agent":
      return "/me";
    case "auditor":
      return "/admin/audit";
    default:
      return "/calls";
  }
}

export async function fetchMe(token: string, base = API_BASE): Promise<Me> {
  const r = await fetch(`${base}/api/v1/me`, { headers: { Authorization: `Bearer ${token}` } });
  if (!r.ok) throw new Error(`me: ${r.status}`);
  return (await r.json()) as Me;
}

export async function devSignIn(sub: string, role: Role, base = API_BASE): Promise<Me> {
  const r = await fetch(`${base}/api/v1/dev/token`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sub, role }),
  });
  if (!r.ok) throw new Error(`dev token: ${r.status}`);
  const { access_token: token } = (await r.json()) as { access_token: string };
  const me = await fetchMe(token, base);
  useAuth.getState().setSession(token, me);
  return me;
}

/** True once the session has been restored from sessionStorage (client only). */
export function useHydrated(): boolean {
  const [done, setDone] = useState(false);
  useEffect(() => {
    setDone(useAuth.persist.hasHydrated());
    return useAuth.persist.onFinishHydration(() => setDone(true));
  }, []);
  return done;
}

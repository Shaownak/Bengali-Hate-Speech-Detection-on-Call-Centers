// Typed REST client for the GAL-E API. Paths and bodies come from the generated OpenAPI types
// (`make gen-types` → src/types/generated/api.ts); this file only adds the base URL and token.
import createClient, { type Middleware } from "openapi-fetch";

import type { components, paths } from "@/types/generated/api";

export type Schemas = components["schemas"];
export type CallSummary = Schemas["CallSummary"];
export type CallDetail = Schemas["CallDetail"];
export type EventOut = Schemas["EventOut"];

export const API_BASE = process.env.NEXT_PUBLIC_GALE_API_URL ?? "http://127.0.0.1:8000";

export function createApiClient(getToken: () => string | undefined, baseUrl = API_BASE) {
  const client = createClient<paths>({ baseUrl });
  const auth: Middleware = {
    onRequest({ request }) {
      const token = getToken();
      if (token) request.headers.set("Authorization", `Bearer ${token}`);
      return request;
    },
  };
  client.use(auth);
  return client;
}

export type ApiClient = ReturnType<typeof createApiClient>;

/** WebSocket URL for `/ws/live` or `/ws/calls/{id}`; send `{type:"auth", token}` first. */
export function wsUrl(path: string, baseUrl = API_BASE): string {
  return baseUrl.replace(/^http/, "ws") + path;
}

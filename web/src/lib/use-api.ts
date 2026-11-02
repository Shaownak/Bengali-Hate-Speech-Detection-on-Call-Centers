"use client";

import { useMemo } from "react";

import { createApiClient } from "@/lib/api";
import { useAuth } from "@/lib/auth";

/** The typed API client with the current session's token; a 401 ends the session. */
export function useApi() {
  const token = useAuth((s) => s.token);
  return useMemo(() => {
    const client = createApiClient(() => token);
    client.use({
      onResponse({ response }) {
        if (response.status === 401) useAuth.getState().clear();
        return response;
      },
    });
    return client;
  }, [token]);
}

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly detail: unknown,
  ) {
    super(typeof detail === "string" ? detail : `HTTP ${status}`);
  }
}

/** openapi-fetch result → data, or throw (for TanStack Query). */
export function unwrap<T>(res: { data?: T; error?: unknown; response: Response }): T {
  if (res.error !== undefined || (res.data === undefined && res.response.status !== 204)) {
    const detail = (res.error as { detail?: unknown } | undefined)?.detail ?? res.error;
    throw new ApiError(res.response.status, detail);
  }
  return res.data as T;
}

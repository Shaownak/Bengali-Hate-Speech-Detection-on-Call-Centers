import { describe, expect, it, vi } from "vitest";

import { createApiClient, wsUrl } from "./api";

describe("api client", () => {
  it("sends the bearer token and hits typed paths", async () => {
    const fetchMock = vi.fn(async (req: Request) => {
      expect(req.headers.get("Authorization")).toBe("Bearer t0k");
      expect(new URL(req.url).pathname).toBe("/api/v1/calls/abc/events");
      return new Response("[]", { headers: { "Content-Type": "application/json" } });
    });
    vi.stubGlobal("fetch", fetchMock);
    const api = createApiClient(() => "t0k", "http://api.test");
    const { data } = await api.GET("/api/v1/calls/{call_id}/events", {
      params: { path: { call_id: "abc" } },
    });
    expect(data).toEqual([]);
    expect(fetchMock).toHaveBeenCalledOnce();
    vi.unstubAllGlobals();
  });

  it("builds websocket URLs", () => {
    expect(wsUrl("/ws/live", "https://gale.example")).toBe("wss://gale.example/ws/live");
  });
});

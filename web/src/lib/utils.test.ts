import { describe, expect, it } from "vitest";

import { useLiveStore } from "./live-store";
import { cn } from "./utils";

describe("cn", () => {
  it("merges and dedupes tailwind classes", () => {
    expect(cn("px-2", "px-4", false, "text-sm")).toBe("px-4 text-sm");
  });
});

describe("live store", () => {
  it("tracks connection state", () => {
    expect(useLiveStore.getState().connected).toBe(false);
    useLiveStore.getState().setConnected(true);
    expect(useLiveStore.getState().connected).toBe(true);
  });
});

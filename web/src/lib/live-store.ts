import { create } from "zustand";

/** Live connection state for the wallboard (filled by the WebSocket client in P5). */
interface LiveState {
  connected: boolean;
  setConnected: (connected: boolean) => void;
}

export const useLiveStore = create<LiveState>((set) => ({
  connected: false,
  setConnected: (connected) => set({ connected }),
}));

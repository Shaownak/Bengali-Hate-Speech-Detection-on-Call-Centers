"use client";

import { AuthGate } from "@/components/auth-gate";
import { CallExplorer } from "@/components/call-explorer";

export default function CallsPage() {
  return (
    <AuthGate perm="calls.read">
      <CallExplorer />
    </AuthGate>
  );
}

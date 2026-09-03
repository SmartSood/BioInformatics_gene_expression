"use client";

import { useEffect, useState, type ReactNode } from "react";
import { useTier2Wake } from "../../../hooks/useTier2Wake";

/**
 * Wrap any feature that calls the model/embedding/depmap/affinity backends
 * with this. It triggers Tier 2 to wake on mount and only renders children
 * once it's actually ready — pods can take 30-90s to schedule after the
 * EC2 instance itself boots, so this isn't just an "instance running" check.
 *
 * Usage: <Tier2WakeGate><YourFeature /></Tier2WakeGate>
 */
export function Tier2WakeGate({ children }: { children: ReactNode }) {
  const { status, ensureAwake } = useTier2Wake();
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (status === "asleep") {
      ensureAwake().catch((e) =>
        setError(e instanceof Error ? e.message : "Failed to wake backend")
      );
    }
  }, [status, ensureAwake]);

  if (status === "ready") return <>{children}</>;

  if (error || status === "error") {
    return (
      <div style={{ padding: 24, textAlign: "center" }}>
        <p>Couldn&apos;t start the compute backend. Try refreshing.</p>
        {error && <p style={{ opacity: 0.6, fontSize: 12 }}>{error}</p>}
      </div>
    );
  }

  return (
    <div style={{ padding: 24, textAlign: "center" }}>
      <p>Warming up the ML backend — usually 60-90s on first use.</p>
      <p style={{ opacity: 0.6, fontSize: 12 }}>
        It sleeps when idle to keep costs down; this only happens on the
        first request after a while.
      </p>
    </div>
  );
}

import type { ReactNode } from "react";
import { Tier2WakeGate } from "./components/Tier2WakeGate";

// Every dashboard page (dataset list/upload, experiments, depmap, affinity,
// embeddings) calls a Tier 2 backend directly, so gate the whole section
// here rather than per-page - this is what actually triggers the wake-on-
// visit behavior; Tier2WakeGate/useTier2Wake existed but nothing rendered
// them before this.
export default function DashboardLayout({ children }: { children: ReactNode }) {
  return <Tier2WakeGate>{children}</Tier2WakeGate>;
}

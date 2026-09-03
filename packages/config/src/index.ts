// Every URL here is overridable via a NEXT_PUBLIC_* env var, read at build
// time by Next.js and inlined into the client bundle - required for the
// deployed frontend (Tier 1) to reach the real Tier 1/Tier 2 public
// addresses instead of localhost, which only ever works when the browser
// and the backend are on the same machine.
export const AUTH_BACKEND_URL =
  process.env.NEXT_PUBLIC_AUTH_BACKEND_URL || "http://localhost:4000";
export const MODEL_BACKEND_URL =
  process.env.NEXT_PUBLIC_MODEL_BACKEND_URL || "http://localhost:8000";
export const DEPMAP_BACKEND_URL =
  process.env.NEXT_PUBLIC_DEPMAP_BACKEND_URL || "http://localhost:8001";
export const AFFINITY_BACKEND_URL =
  process.env.NEXT_PUBLIC_AFFINITY_BACKEND_URL || "http://localhost:8003";
export const EMBEDDING_BACKEND_URL =
  process.env.NEXT_PUBLIC_EMBEDDING_BACKEND_URL || "http://localhost:8002";

// Tier 1 (always-on) service that starts/stops the on-demand Tier 2 EC2+k3s
// node. Runs alongside the auth backend on the small always-on instance.
export const WAKE_GATEWAY_URL =
  process.env.NEXT_PUBLIC_WAKE_GATEWAY_URL || "http://localhost:4100";

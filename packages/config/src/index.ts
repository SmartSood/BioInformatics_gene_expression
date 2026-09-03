export const AUTH_BACKEND_URL = "http://localhost:4000";
export const MODEL_BACKEND_URL = "http://localhost:8000";
export const DEPMAP_BACKEND_URL = "http://localhost:8001";
export const AFFINITY_BACKEND_URL = "http://localhost:8003";

// Tier 1 (always-on) service that starts/stops the on-demand Tier 2 EC2+k3s
// node. Runs alongside the auth backend on the small always-on instance.
export const WAKE_GATEWAY_URL = "http://localhost:4100";

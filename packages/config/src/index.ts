// Production uses same-origin paths so an HTTPS frontend never calls an HTTP
// backend directly. Tier 1 Nginx proxies these paths to the appropriate API.
// Local development keeps the existing localhost service ports.
const isProduction = process.env.NODE_ENV === "production";

export const AUTH_BACKEND_URL =
  process.env.NEXT_PUBLIC_AUTH_BACKEND_URL ||
  (isProduction ? "/auth" : "http://localhost:4000");
export const MODEL_BACKEND_URL =
  process.env.NEXT_PUBLIC_MODEL_BACKEND_URL ||
  (isProduction ? "/api/models" : "http://localhost:8000");
export const DEPMAP_BACKEND_URL =
  process.env.NEXT_PUBLIC_DEPMAP_BACKEND_URL ||
  (isProduction ? "/api/depmap" : "http://localhost:8001");
export const AFFINITY_BACKEND_URL =
  process.env.NEXT_PUBLIC_AFFINITY_BACKEND_URL ||
  (isProduction ? "/api/affinity" : "http://localhost:8003");
export const EMBEDDING_BACKEND_URL =
  process.env.NEXT_PUBLIC_EMBEDDING_BACKEND_URL ||
  (isProduction ? "/api/embeddings" : "http://localhost:8002");

// Tier 1 (always-on) service that starts/stops the on-demand Tier 2 EC2+k3s
// node. Runs alongside the auth backend on the small always-on instance.
export const WAKE_GATEWAY_URL =
  process.env.NEXT_PUBLIC_WAKE_GATEWAY_URL ||
  (isProduction ? "/wake" : "http://localhost:4100");

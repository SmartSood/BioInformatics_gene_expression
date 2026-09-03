// PM2 process list for Tier 1 (the always-on box). Only the lightweight
// pieces run here: frontend, auth, and the wake-gateway that starts/stops
// Tier 2. Everything ML/compute-heavy lives on Tier 2 instead.
module.exports = {
  apps: [
    {
      name: "gene-web-frontend",
      cwd: "/opt/gene-web/apps/web",
      script: "npm",
      args: "run start",
      env: { NODE_ENV: "production" },
    },
    {
      name: "gene-auth-backend",
      cwd: "/opt/gene-web/apps/auth_backend",
      script: "dist/index.js",
      env: { NODE_ENV: "production" },
    },
    {
      name: "gene-wake-gateway",
      cwd: "/opt/gene-web/apps/wake_gateway",
      script: "dist/index.js",
      env: { NODE_ENV: "production" },
    },
  ],
};

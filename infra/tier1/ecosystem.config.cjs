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
    // Run via tsx, not `node dist/index.js`: @repo/dotenv-path, @repo/db and
    // @repo/zod-scemma all export raw .ts source directly (no build step of
    // their own), which Next.js transpiles fine for apps/web but which
    // plain `node` cannot load at all (ERR_UNKNOWN_FILE_EXTENSION) - tsx
    // resolves and transpiles those workspace imports on the fly instead.
    {
      name: "gene-auth-backend",
      cwd: "/opt/gene-web/apps/auth_backend",
      script: "npx",
      args: "tsx src/index.ts",
      env: { NODE_ENV: "production" },
    },
    {
      name: "gene-wake-gateway",
      cwd: "/opt/gene-web/apps/wake_gateway",
      script: "npx",
      args: "tsx src/index.ts",
      env: { NODE_ENV: "production" },
    },
  ],
};

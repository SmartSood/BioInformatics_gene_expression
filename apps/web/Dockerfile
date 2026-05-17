FROM node:20-slim AS base

WORKDIR /app

COPY package.json turbo.json tsconfig.json ./
COPY packages ./packages
COPY apps/web ./apps/web
COPY apps/auth_backend/package.json ./apps/auth_backend/package.json

RUN npm install

FROM base AS build

COPY . .
RUN npm run build --workspace apps/web

FROM node:20-slim AS runner

WORKDIR /app
ENV NODE_ENV=production

COPY --from=base /app/node_modules ./node_modules
COPY --from=build /app/apps/web/.next ./apps/web/.next
COPY --from=build /app/apps/web/public ./apps/web/public
COPY --from=build /app/apps/web/package.json ./apps/web/package.json
COPY --from=build /app/apps/web/next.config.js ./apps/web/next.config.js

EXPOSE 3000

CMD ["npm", "run", "start", "--workspace", "apps/web"]

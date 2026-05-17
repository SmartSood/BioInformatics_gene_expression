# Build stage
FROM node:20-slim AS builder

WORKDIR /app

COPY package.json turbo.json tsconfig.json ./
COPY packages ./packages
COPY apps/auth_backend ./apps/auth_backend

RUN npm install
RUN npm run build --workspace apps/auth_backend

# Runtime stage
FROM node:20-slim

WORKDIR /app
ENV NODE_ENV=production

COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/packages ./packages
COPY --from=builder /app/apps/auth_backend ./apps/auth_backend
COPY package.json turbo.json tsconfig.json ./

EXPOSE 3001

CMD ["npm", "run", "start", "--workspace", "apps/auth_backend"]

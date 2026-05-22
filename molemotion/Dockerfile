# Multi-stage build for the MoleMotion frontend.
# Build context = molemotion/ (set in docker-compose.yml).

# ---------- stage 1: build the Vite bundle ----------
FROM node:20-alpine AS builder

WORKDIR /build

# npm ci needs package.json AND package-lock.json — both go in the cache layer.
COPY package.json package-lock.json ./
RUN npm ci

COPY . .

# Same-origin proxy in prod: the nginx in stage 2 fronts /api → inference:8000,
# so the bundle just hits /api directly.
ARG VITE_API_BASE_URL=/api
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL
RUN npm run build

# ---------- stage 2: serve via nginx ----------
FROM nginx:alpine

# Replace default config with our SPA + /api proxy config
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /build/dist /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD wget -q -O- http://localhost/ >/dev/null 2>&1 || exit 1

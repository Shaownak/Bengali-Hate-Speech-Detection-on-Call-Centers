# Dashboard image: Next.js standalone server, non-root, pinned base (docs/05 §2).
#   docker build -f infra/docker/web.Dockerfile \
#     --build-arg NEXT_PUBLIC_GALE_API_URL=https://gale-api.example.bd \
#     --build-arg NEXT_PUBLIC_GALE_AUTH_MODE=oidc .
# NEXT_PUBLIC_* values are compiled into the client bundle, so build one image per environment.

FROM node:22.23.0-bookworm-slim@sha256:d9f850096136edbc402debdd8729579a288aac64574ada0ff4db26b6ae58b0b2 AS build
ARG NEXT_PUBLIC_GALE_API_URL=http://127.0.0.1:8000
ARG NEXT_PUBLIC_GALE_AUTH_MODE=dev
ARG NEXT_PUBLIC_OIDC_ISSUER=""
ARG NEXT_PUBLIC_OIDC_CLIENT_ID=gale-web
ARG NEXT_PUBLIC_GRAFANA_URL=""
ENV NEXT_TELEMETRY_DISABLED=1 \
    NEXT_PUBLIC_GALE_API_URL=${NEXT_PUBLIC_GALE_API_URL} \
    NEXT_PUBLIC_GALE_AUTH_MODE=${NEXT_PUBLIC_GALE_AUTH_MODE} \
    NEXT_PUBLIC_OIDC_ISSUER=${NEXT_PUBLIC_OIDC_ISSUER} \
    NEXT_PUBLIC_OIDC_CLIENT_ID=${NEXT_PUBLIC_OIDC_CLIENT_ID} \
    NEXT_PUBLIC_GRAFANA_URL=${NEXT_PUBLIC_GRAFANA_URL}
RUN npm install -g pnpm@12.16.1
WORKDIR /web
COPY web/package.json web/pnpm-lock.yaml web/pnpm-workspace.yaml ./
# the dev store/cache paths in pnpm-workspace.yaml point at ../.gale; use the image's own
RUN sed -i -E '/^(storeDir|cacheDir):/d' pnpm-workspace.yaml && pnpm install --frozen-lockfile
COPY web/ ./
RUN sed -i -E '/^(storeDir|cacheDir):/d' pnpm-workspace.yaml && GALE_STANDALONE=1 pnpm build

FROM node:22.23.0-bookworm-slim@sha256:d9f850096136edbc402debdd8729579a288aac64574ada0ff4db26b6ae58b0b2
ENV NODE_ENV=production NEXT_TELEMETRY_DISABLED=1 HOSTNAME=0.0.0.0 PORT=3000
WORKDIR /web
COPY --from=build --chown=root:root /web/.next/standalone ./
COPY --from=build --chown=root:root /web/.next/static ./.next/static
USER 1000
EXPOSE 3000
CMD ["node", "server.js"]

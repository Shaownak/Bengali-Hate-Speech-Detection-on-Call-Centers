# ADR-017: Web dashboard (Prompt 13)

- **Status:** Accepted
- **Date:** 2026-11-02
- **Deciders:** Claude Code (Prompt 13), per CLAUDE.md "spec ambiguous → simplest option + ADR"
- **Spec refs:** docs/04 §1–5; ADR-016

## Context
Prompt 13 builds every docs/04 §2 screen. Choices the spec leaves open: how the browser signs in, where session state lives, how live data reaches React, and what to do with spec items whose data doesn't exist yet.

## Decision
1. **Client-rendered screens over the API.**
   - Every screen is a client component that talks to the API with the generated `openapi-fetch` client and TanStack Query.
   - Next.js only serves the app shell, fonts and i18n.
   - No Next.js API routes, so no second backend to secure.
2. **Sign-in.**
   - `NEXT_PUBLIC_GALE_AUTH_MODE=oidc` uses Keycloak through `oidc-client-ts` (authorization code + PKCE).
   - `dev` (the default) uses the API's dev issuer, with a picker of seeded users.
   - The access token is kept in `sessionStorage` (cleared when the tab closes), never in a URL. A 401 ends the session.
   - This replaces the `next-auth` mention in docs/04: with no server-side session there is nothing for `next-auth` to do. The OIDC path is exercised only in staging.
3. **Live data.**
   - One `LiveSocket` class handles both WebSockets: token as the first message, a snapshot, then 250 ms batches.
   - Reconnects use exponential backoff with jitter. Refusals (4401/4403/4404) are final.
   - The wallboard state is a Zustand store with pure reducers, unit-tested including a 1,000-call / 2-updates-per-second load test.
   - Risk meters are the rolling 60 s maximum of fused probability per speaker, computed client-side from event updates.
   - The call grid is virtualised (`@tanstack/react-virtual`).
4. **Audio.**
   - wavesurfer.js: one split-channel waveform for an uploaded stereo recording, or two synced waveforms for gateway-retained per-channel audio.
   - Event regions sit on the speaker's channel.
   - The review queue plays the span ±3 s.
5. **Exports.**
   - CSV is built in the browser (UTF-8 BOM for Bangla in Excel; formula cells neutralised).
   - PDF uses the browser's print-to-PDF with print styles. The full HTML report from the post-call worker stays available in the object store.
6. **Small API additions for the screens:**
   - `GET /me`, `GET /directory`, `GET /reviews/agreement`, `GET /thresholds` (read-only);
   - review kinds `note` and `flag`;
   - team heatmap (hour × weekday in `GALE_DISPLAY_TZ`, bucketed in Python so the database's timezone tables don't matter);
   - agent team median and coaching moments;
   - more call-list filters: min severity, speaker, reviewed, score ranges.
   All are covered by the RBAC matrix.
7. **Accessibility and i18n.**
   - Severity is always icon + text + colour; heatmap cells print their values; meters expose `aria-valuenow`.
   - There is a skip link, `aria-live` on feeds, and reduced-motion handling.
   - Bangla is the default locale. A unit test enforces identical bn/en message keys.
   - Inter for Latin; Noto Sans Bengali / Hind Siliguri for Bangla.

## Consequences
- Items waiting on missing data, shown honestly in the UI:
  - the fairness panel waits on Q-05 (agent gender and dialect aren't collected);
  - rules simulation shows alert deltas, not precision/recall, until labelled dev calls exist;
  - threshold editing stays in `configs/thresholds.yaml` (read-only on screen);
  - Grafana panels need `NEXT_PUBLIC_GRAFANA_URL`.
- Not done yet: visual-regression snapshots and a Lighthouse/axe accessibility score (docs/04 §5). The e2e suite writes reference screenshots to `docs/screenshots/` that a visual-regression step can build on.

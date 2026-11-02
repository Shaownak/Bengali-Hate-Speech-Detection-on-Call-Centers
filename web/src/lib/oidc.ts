// Keycloak sign-in (authorization code + PKCE) for NEXT_PUBLIC_GALE_AUTH_MODE=oidc. Exercised only
// against a real Keycloak in staging (ADR-017); dev and tests use the API's dev issuer.
import { UserManager, WebStorageStateStore } from "oidc-client-ts";

import { fetchMe, useAuth } from "@/lib/auth";

let manager: UserManager | undefined;

function userManager(locale: string): UserManager {
  manager ??= new UserManager({
    authority: process.env.NEXT_PUBLIC_OIDC_ISSUER ?? "http://127.0.0.1:8080/realms/gale",
    client_id: process.env.NEXT_PUBLIC_OIDC_CLIENT_ID ?? "gale-web",
    redirect_uri: `${window.location.origin}/${locale}/auth/callback`,
    response_type: "code",
    scope: "openid profile email",
    userStore: new WebStorageStateStore({ store: window.sessionStorage }),
  });
  return manager;
}

export function oidcSignIn(locale: string): Promise<void> {
  return userManager(locale).signinRedirect();
}

export async function oidcCallback(locale: string): Promise<void> {
  const user = await userManager(locale).signinRedirectCallback();
  const me = await fetchMe(user.access_token);
  useAuth.getState().setSession(user.access_token, me);
}

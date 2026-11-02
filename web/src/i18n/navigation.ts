import { createNavigation } from "next-intl/navigation";

import { routing } from "./routing";

// Locale-aware Link / router (keeps /bn or /en in every URL).
export const { Link, redirect, usePathname, useRouter } = createNavigation(routing);

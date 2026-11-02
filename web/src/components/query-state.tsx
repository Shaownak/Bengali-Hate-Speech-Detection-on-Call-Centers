"use client";

import { useTranslations } from "next-intl";
import type { ReactNode } from "react";

import { Button } from "@/components/ui/button";

/** Loading / error / empty states for every view (docs/04 §4). */
export function QueryState({
  isLoading,
  error,
  isEmpty,
  emptyText,
  onRetry,
  children,
}: {
  isLoading: boolean;
  error: unknown;
  isEmpty?: boolean;
  emptyText?: string;
  onRetry?: () => void;
  children: ReactNode;
}) {
  const t = useTranslations("common");
  if (isLoading)
    return (
      <p role="status" aria-live="polite" className="p-4 text-sm opacity-70">
        {t("loading")}
      </p>
    );
  if (error)
    return (
      <div role="alert" className="flex items-center gap-3 p-4 text-sm">
        <span>
          {t("error")} {error instanceof Error ? `(${error.message})` : null}
        </span>
        {onRetry && (
          <Button size="sm" variant="outline" onClick={onRetry}>
            {t("retry")}
          </Button>
        )}
      </div>
    );
  if (isEmpty) return <p className="p-4 text-sm opacity-70">{emptyText ?? t("empty")}</p>;
  return <>{children}</>;
}

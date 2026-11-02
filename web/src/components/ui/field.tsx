import type { ComponentProps } from "react";

import { cn } from "@/lib/utils";

const base =
  "h-9 rounded-md border border-border bg-background px-2 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring";

export function Input({ className, ...props }: ComponentProps<"input">) {
  return <input className={cn(base, className)} {...props} />;
}

export function Select({ className, ...props }: ComponentProps<"select">) {
  return <select className={cn(base, "pr-6", className)} {...props} />;
}

export function Textarea({ className, ...props }: ComponentProps<"textarea">) {
  return <textarea className={cn(base, "h-auto min-h-20 py-1", className)} {...props} />;
}

export function Label({ className, ...props }: ComponentProps<"label">) {
  return <label className={cn("flex flex-col gap-1 text-xs font-medium", className)} {...props} />;
}

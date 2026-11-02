import type { Category } from "@/types/generated";

/** Category display order (docs/04 §4). Severity is never conveyed by colour alone. */
export const CATEGORIES: readonly Category[] = [
  "slang_profanity",
  "insult",
  "threat",
  "sexual_harassment",
  "hate_identity",
  "disrespect_register",
  "harsh_tone",
  "shouting",
  "sarcasm_mockery",
  "dismissive_rude",
  "interruption_overtalk",
] as const;

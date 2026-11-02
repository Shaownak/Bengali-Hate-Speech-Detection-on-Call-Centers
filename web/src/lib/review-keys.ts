/** Review-queue keyboard map (docs/04 §2.4). Pure so it can be unit-tested. */
export type ReviewAction =
  | { type: "next" }
  | { type: "prev" }
  | { type: "verdict"; verdict: "correct" | "false_positive" }
  | { type: "pick"; field: "category" | "severity" }
  | { type: "play" };

export function keyAction(key: string, target?: EventTarget | null): ReviewAction | undefined {
  // typing into a field must not trigger shortcuts
  const el = target as HTMLElement | null | undefined;
  if (el && ["INPUT", "TEXTAREA", "SELECT"].includes(el.tagName ?? "")) return undefined;
  switch (key) {
    case "j":
    case "J":
      return { type: "next" };
    case "k":
    case "K":
      return { type: "prev" };
    case "1":
      return { type: "verdict", verdict: "correct" };
    case "2":
      return { type: "pick", field: "category" };
    case "3":
      return { type: "verdict", verdict: "false_positive" };
    case "4":
      return { type: "pick", field: "severity" };
    case " ":
      return { type: "play" };
    default:
      return undefined;
  }
}

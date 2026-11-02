import { render, screen } from "@testing-library/react";
import { NextIntlClientProvider } from "next-intl";
import type { ReactNode } from "react";
import { describe, expect, it } from "vitest";

import { CategoryLabel, SeverityBadge, StatusBadge } from "@/components/badges";
import { Heatmap } from "@/components/heatmap";
import { RiskMeter } from "@/components/risk-meter";
import { activeWord, RedactedText } from "@/components/transcript";

import bn from "../../messages/bn.json";
import en from "../../messages/en.json";

const wrap = (node: ReactNode, locale: "bn" | "en" = "bn") =>
  render(
    <NextIntlClientProvider locale={locale} messages={locale === "bn" ? bn : en}>
      {node}
    </NextIntlClientProvider>,
  );

describe("accessible severity", () => {
  it("shows icon + text, not colour alone", () => {
    const { container } = wrap(<SeverityBadge level={3} />);
    expect(screen.getByText("উচ্চ")).toBeInTheDocument();
    expect(container.querySelector("svg")).not.toBeNull();
    expect(container.querySelector("[data-severity='3']")).toHaveAttribute("title", "তীব্রতা 3");
  });

  it("labels categories and statuses in the chosen language", () => {
    wrap(
      <>
        <CategoryLabel category="shouting" />
        <StatusBadge status="provisional" />
      </>,
      "en",
    );
    expect(screen.getByText("Shouting")).toBeInTheDocument();
    expect(screen.getByText("Provisional")).toHaveAttribute(
      "title",
      expect.stringContaining("Fast path"),
    );
  });

  it("risk meter exposes its value to assistive tech", () => {
    wrap(
      <RiskMeter
        label="Agent risk"
        point={{ eventId: "e", t: 0, p: 0.9, severity: 3, category: "insult" }}
      />,
      "en",
    );
    const m = screen.getByRole("meter", { name: "Agent risk" });
    expect(m).toHaveAttribute("aria-valuenow", "90");
    expect(screen.getByText(/90% · High/)).toBeInTheDocument();
  });
});

describe("transcript", () => {
  it("renders PII placeholders as tags and finds the active word", () => {
    const { container } = wrap(<RedactedText text="call me at <PHONE> now" />);
    expect(container.querySelector("[data-pii-tag='<PHONE>']")).toHaveTextContent("PHONE");
    expect(activeWord([{ start: 0, end: 1 }, { start: 1, end: 2 }], 1.5)).toBe(1);
    expect(activeWord([{ start: 0, end: 1 }], 3)).toBe(-1);
  });
});

describe("heatmap", () => {
  it("prints values in cells (colour is not the only cue)", () => {
    wrap(
      <Heatmap
        metric="agent_incidents"
        cells={[{ weekday: 0, hour: 20, calls: 3, agent_incidents: 2, avg_customer_hostility: 4 }]}
      />,
      "en",
    );
    const table = screen.getByTestId("heatmap");
    expect(table).toHaveTextContent("Mon");
    expect(table.querySelector("td[title^='Mon 20:00']")).toHaveTextContent("2");
  });
});

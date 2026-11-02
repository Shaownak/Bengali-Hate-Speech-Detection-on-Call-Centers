import { render, screen } from "@testing-library/react";
import { NextIntlClientProvider } from "next-intl";
import { describe, expect, it } from "vitest";

import bn from "../../messages/bn.json";
import { CategoryList } from "./category-list";

describe("CategoryList", () => {
  it("renders all 11 categories in Bangla", () => {
    render(
      <NextIntlClientProvider locale="bn" messages={bn}>
        <CategoryList />
      </NextIntlClientProvider>,
    );
    expect(screen.getAllByRole("listitem")).toHaveLength(11);
    expect(screen.getByText("চিৎকার")).toBeInTheDocument();
  });
});

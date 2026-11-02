import { describe, expect, it } from "vitest";

import bn from "../../messages/bn.json";
import en from "../../messages/en.json";
import schemas from "../types/generated/schemas.json";
import { CATEGORIES } from "./categories";

describe("categories", () => {
  const backendEnum = [...schemas.models.Signal.$defs.Category.enum].sort();

  it("matches the backend Category enum", () => {
    expect([...CATEGORIES].sort()).toEqual(backendEnum);
  });

  it("has a Bangla and English label for every category", () => {
    for (const c of CATEGORIES) {
      expect(bn.categories[c]).toBeTruthy();
      expect(en.categories[c]).toBeTruthy();
    }
    expect(Object.keys(bn.categories).sort()).toEqual(backendEnum);
    expect(Object.keys(en.categories).sort()).toEqual(backendEnum);
  });

  it("bn and en message files have exactly the same keys", () => {
    const keys = (o: object, p = ""): string[] =>
      Object.entries(o).flatMap(([k, v]) =>
        v && typeof v === "object" ? keys(v as object, `${p}${k}.`) : [`${p}${k}`],
      );
    expect(keys(bn).sort()).toEqual(keys(en).sort());
  });
});

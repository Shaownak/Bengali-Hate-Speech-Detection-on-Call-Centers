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

  it("bn and en message files have the same keys", () => {
    expect(Object.keys(bn.home).sort()).toEqual(Object.keys(en.home).sort());
  });
});

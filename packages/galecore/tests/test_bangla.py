import unicodedata

import pytest

from galecore.bangla import (
    find_canonical,
    phonetic_key,
    to_ascii_digits,
    to_bangla_digits,
    tokenize,
    words,
)
from galecore.bangla import (
    normalize as n,
)

RRA_PRE, RRA_DEC = "ড়", "ড়"  # ড়
RHA_PRE, RHA_DEC = "ঢ়", "ঢ়"  # ঢ়
YYA_PRE, YYA_DEC = "য়", "য়"  # য়


class TestNormalize:
    def test_empty(self) -> None:
        assert n("") == ""

    @pytest.mark.parametrize(
        ("pre", "dec"), [(RRA_PRE, RRA_DEC), (RHA_PRE, RHA_DEC), (YYA_PRE, YYA_DEC)]
    )
    def test_nukta_letters_unify_to_precomposed(self, pre: str, dec: str) -> None:
        assert n("প" + dec) == n("প" + pre) == "প" + pre

    def test_nfc_composes_o_kar(self) -> None:
        # ো written as ে + া must equal the single code point U+09CB
        assert n("কো") == "কো"
        assert n("কৌ") == "কৌ"

    def test_legacy_khanda_ta(self) -> None:
        assert n("উত্‍সব") == "উৎসব"

    def test_zero_width_chars_removed(self) -> None:
        assert n("ক্‌ষ") == "ক্ষ"
        assert n("ক্‍ষ") == "ক্ষ"
        assert n("﻿আমি​ তুমি­") == "আমি তুমি"

    def test_digits(self) -> None:
        assert n("০১৭১২৩৪৫৬৭৮") == "01712345678"
        assert n("01712", digits="bangla") == "০১৭১২"
        assert n("০১৭12", digits="keep") == "০১৭12"

    def test_whitespace_and_latin_case(self) -> None:
        assert n("  Recharge\t\nকরেন   OTP ") == "recharge করেন otp"
        assert n("ABC", lowercase_latin=False) == "ABC"

    def test_placeholder_tags_keep_case(self) -> None:
        # tags (<PHONE>, <GALI_1>) must survive lower-casing so lexicon/tag matching works
        assert n("এই <GALI_1> AND <PHONE>") == "এই <GALI_1> and <PHONE>"

    def test_canonical_map(self) -> None:
        assert n("রিচার্য করেন", canonical={"রিচার্য": "রিচার্জ"}) == "রিচার্জ করেন"

    @pytest.mark.parametrize(
        "text",
        ["প" + YYA_DEC + "সা", "উত্‍সব", "০১৭ AB  c", "ক্‌ষমা", ""],
    )
    def test_idempotent(self, text: str) -> None:
        assert n(n(text)) == n(text)

    def test_output_has_no_decomposed_nukta(self) -> None:
        out = n("আমার " + RRA_DEC + YYA_DEC + RHA_DEC)
        assert "়" not in out
        assert out == "আমার " + RRA_PRE + YYA_PRE + RHA_PRE
        # NFC alone would leave them decomposed (composition exclusions)
        assert unicodedata.normalize("NFC", RRA_PRE) == RRA_DEC


class TestDigits:
    def test_round_trip(self) -> None:
        assert to_bangla_digits(to_ascii_digits("৪২")) == "৪২"
        assert len(to_ascii_digits("০১২৩")) == 4


class TestPhoneticKey:
    @pytest.mark.parametrize(
        ("a", "b"),
        [
            ("শালা", "সালা"),
            ("ষাট", "সাট"),
            ("যাও", "জাও"),
            ("কারণ", "কারন"),
            ("দীন", "দিন"),
            ("ঈদ", "ইদ"),
            ("ভূত", "ভুত"),
            ("ঊষা", "উসা"),
            ("পড়া", "পরা"),
            ("আষাঢ়", "আসার"),
        ],
    )
    def test_merges(self, a: str, b: str) -> None:
        assert phonetic_key(a) == phonetic_key(b)

    def test_decomposed_input(self) -> None:
        assert phonetic_key("প" + RRA_DEC + "া") == phonetic_key("পরা")

    def test_distinct_words_stay_distinct(self) -> None:
        assert phonetic_key("কাল") != phonetic_key("গাল")


class TestTokenize:
    def test_offsets_are_exact_substrings(self) -> None:
        text = "আপনি কি পাগল? আমার <PHONE> নম্বরে কল দিন।"
        toks = tokenize(text)
        assert [t.text for t in toks] == [
            "আপনি",
            "কি",
            "পাগল",
            "আমার",
            "<PHONE>",
            "নম্বরে",
            "কল",
            "দিন",
        ]
        for t in toks:
            assert text[t.start : t.end] == t.text

    def test_combining_marks_stay_in_token(self) -> None:
        assert words("কিন্তু চাঁদ") == ["কিন্তু", "চাঁদ"]

    def test_banglish_and_digits(self) -> None:
        assert words("recharge 500 taka, ok?") == ["recharge", "500", "taka", "ok"]

    def test_gali_placeholder_is_one_token(self) -> None:
        assert words("তুই <GALI_12>!") == ["তুই", "<GALI_12>"]


def test_find_canonical_matches_across_unicode_forms() -> None:
    """LLM tokenizers (Qwen) NFC-normalise, which decomposes ড় / য় (composition exclusions):
    evidence copied by the model must still ground on a transcript in the precomposed form,
    and the span returned is the transcript's own text."""
    yya_pre, yya_dec = "য়", "য়"
    text = f"আপনি কি {RRA_PRE}া দি{yya_pre}েছেন? না <GALI_3> বলছি"
    assert find_canonical(text, "<GALI_3>") == (text.index("<GALI_3>"), text.index("<GALI_3>") + 8)
    needle = f"{RRA_DEC}া দি{yya_dec}েছেন"  # decomposed, as the model emits it
    assert needle not in text
    span = find_canonical(text, needle)
    assert span is not None
    assert text[span[0] : span[1]] == f"{RRA_PRE}া দি{yya_pre}েছেন"  # verbatim transcript text
    assert find_canonical(text, "কেউ বলেনি") is None
    assert find_canonical(text, "") is None
    assert find_canonical(text, "   ") is None

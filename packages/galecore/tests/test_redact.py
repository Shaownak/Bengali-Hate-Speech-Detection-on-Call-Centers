"""Redaction edge cases. All numbers are synthetic."""

import pytest

from galecore.bangla import normalize
from galecore.redact import (
    ACCT,
    EMAIL,
    NID,
    PHONE,
    Span,
    classify_digits,
    contains_pii,
    redact,
)


def r(text: str) -> str:
    return redact(normalize(text, digits="keep")).text


class TestPhones:
    @pytest.mark.parametrize(
        "number",
        [
            "01712345678",
            "+8801712345678",
            "8801912345678",
            "০১৭১২৩৪৫৬৭৮",  # Bangla digits
            "+৮৮০১৮১২৩৪৫৬৭৮",
            "017-1234-5678",
            "017 1234 5678",
            "01712 345678",
        ],
    )
    def test_bd_mobile_forms(self, number: str) -> None:
        assert r(f"আমার নম্বর {number} এ কল দেন") == f"আমার নম্বর {PHONE} এ কল দেন"

    def test_operator_prefix_must_be_3_to_9(self) -> None:
        # 012… is not a valid BD mobile prefix; still long enough to be treated as an account
        assert classify_digits("01212345678") == ACCT

    def test_spoken_bangla_digits(self) -> None:
        text = "নম্বরটা হলো জিরো ওয়ান সেভেন ওয়ান টু থ্রি ফোর ফাইভ সিক্স সেভেন এইট ঠিক আছে"
        assert r(text) == f"নম্বরটা হলো {PHONE} ঠিক আছে"

    def test_spoken_bangla_words(self) -> None:
        text = "শূন্য এক সাত এক দুই তিন চার পাঁচ ছয় সাত আট"
        assert r(text) == PHONE

    def test_spoken_double_triple(self) -> None:
        # 0 1 7 1 1 2 2 2 3 4 5 → 01711222345
        text = "জিরো ওয়ান সেভেন ডাবল ওয়ান ট্রিপল টু থ্রি ফোর ফাইভ"
        assert r(text) == PHONE

    def test_spoken_english(self) -> None:
        assert r("zero one eight one two three four five six seven eight") == PHONE

    def test_mixed_spoken_and_digits(self) -> None:
        assert r("জিরো 1 সাত ১ ২ ৩ ৪ ৫ ৬ ৭ ৮") == PHONE


class TestNidAndAccounts:
    @pytest.mark.parametrize("n", ["1234567890", "1234567890123", "12345678901234567"])
    def test_nid_lengths(self, n: str) -> None:
        assert r(f"NID {n}") == f"nid {NID}"

    def test_bangla_digit_nid(self) -> None:
        assert r("এনআইডি ১২৩৪৫৬৭৮৯০") == f"এনআইডি {NID}"

    def test_card_or_wallet_number(self) -> None:
        assert r("কার্ড 4111 1111 1111 1111") == f"কার্ড {ACCT}"

    def test_short_numbers_kept(self) -> None:
        # amounts, package ids, OTP-length codes are not PII by this rule
        for text in ["৫০০ টাকা রিচার্জ", "প্যাকেজ 149", "কোড 123456"]:
            assert r(text) == normalize(text, digits="keep")

    def test_spoken_short_number_kept(self) -> None:
        assert r("এক দুই তিন") == "এক দুই তিন"
        assert r("এক মিনিট দাঁড়ান") == normalize("এক মিনিট দাঁড়ান", digits="keep")


class TestEmail:
    def test_email(self) -> None:
        assert r("মেইল test.user+1@example.com দিন") == f"মেইল {EMAIL} দিন"


class TestResult:
    def test_spans_and_counts(self) -> None:
        text = "01712345678 আর 1234567890"
        res = redact(text)
        assert res.text == f"{PHONE} আর {NID}"
        assert res.spans == [Span(0, 11, PHONE), Span(15, 25, NID)]
        assert res.counts == {PHONE: 1, NID: 1}

    def test_empty(self) -> None:
        assert redact("").text == ""

    def test_idempotent(self) -> None:
        once = redact("ফোন 01712345678").text
        assert redact(once).text == once

    def test_ner_backend_plugs_in(self) -> None:
        class FakeNER:
            def find(self, text: str) -> list[Span]:
                i = text.find("করিম")
                return [Span(i, i + 4, "<NAME>")] if i >= 0 else []

        assert redact("আমি করিম বলছি", ner=FakeNER()).text == "আমি <NAME> বলছি"

    def test_contains_pii(self) -> None:
        assert contains_pii("call 01712345678")
        assert contains_pii("a@b.co")
        assert not contains_pii("৫০০ টাকা")

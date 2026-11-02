"""Typed view of ``configs/rules.yaml`` for fusion v0.

Each rule name maps to one handler in ``fusion.py``. Parameters are read from YAML and
validated here, so a typo in the config fails at start-up instead of silently disabling a rule.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from galecore.config import RulesConfig
from galecore.schemas import Category, Speaker

KNOWN = {
    "lexicon_fast_path",
    "kws_fast_path",
    "text_clf_event",
    "sustained_shout",
    "harsh_tone_ser",
    "register_drop",
    "overtalk",
    "llm_window_event",
    "shout_plus_abuse",
    "llm_dismiss",
}


class _P(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ScoreRule(_P):
    score_gte: float
    severity: int | None = None  # fixed severity, else from the signal's evidence


class TextClfRule(ScoreRule):
    category_in: frozenset[Category]


class DurationRule(_P):
    score_gte: float
    min_duration_s: float
    severity: int


class RegisterRule(_P):
    severity_by_level: dict[Speaker, dict[str, int]]


class CountRule(_P):
    count_gte: int
    window_s: float
    severity: int


class ShoutAbuseRule(_P):
    categories: frozenset[Category]
    within_s: float
    severity_delta: int


class DismissRule(_P):
    unless_kws_gte: float


class FusionRules(_P):
    lexicon: ScoreRule
    kws: ScoreRule
    text_clf: TextClfRule
    shout: DurationRule
    harsh_tone: DurationRule
    register_drop: RegisterRule
    overtalk: CountRule
    llm_event: ScoreRule
    shout_abuse: ShoutAbuseRule
    llm_dismiss: DismissRule


def parse_rules(cfg: RulesConfig) -> FusionRules:
    by_name = {r.name: r for r in cfg.rules}
    unknown = set(by_name) - KNOWN
    missing = KNOWN - set(by_name)
    if unknown or missing:
        raise ValueError(f"rules.yaml: unknown rules {sorted(unknown)}, missing {sorted(missing)}")

    def w(name: str) -> dict[str, Any]:
        return by_name[name].when

    def t(name: str) -> dict[str, Any]:
        return by_name[name].then

    abuse_when = w("shout_plus_abuse")["all"]
    abuse = next(c for c in abuse_when if "event_in" in c)
    return FusionRules(
        lexicon=ScoreRule(score_gte=w("lexicon_fast_path")["score_gte"]),
        kws=ScoreRule(
            score_gte=w("kws_fast_path")["score_gte"], severity=t("kws_fast_path").get("severity")
        ),
        text_clf=TextClfRule(
            score_gte=w("text_clf_event")["score_gte"],
            category_in=frozenset(w("text_clf_event")["category_in"]),
        ),
        shout=DurationRule(
            score_gte=w("sustained_shout")["score_gte"],
            min_duration_s=w("sustained_shout")["min_duration_s"],
            severity=t("sustained_shout")["severity"],
        ),
        harsh_tone=DurationRule(
            score_gte=w("harsh_tone_ser")["score_gte"],
            min_duration_s=w("harsh_tone_ser")["min_duration_s"],
            severity=t("harsh_tone_ser")["severity"],
        ),
        register_drop=RegisterRule(severity_by_level=t("register_drop")["severity_by_level"]),
        overtalk=CountRule(
            count_gte=w("overtalk")["count_gte"],
            window_s=w("overtalk")["window_s"],
            severity=t("overtalk")["severity"],
        ),
        llm_event=ScoreRule(score_gte=w("llm_window_event")["score_gte"]),
        shout_abuse=ShoutAbuseRule(
            categories=frozenset(abuse["event_in"]),
            within_s=abuse["within_s"],
            severity_delta=t("shout_plus_abuse")["severity_delta"],
        ),
        llm_dismiss=DismissRule(unless_kws_gte=t("llm_dismiss")["unless"]["score_gte"]),
    )

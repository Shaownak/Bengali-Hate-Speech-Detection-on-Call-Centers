"""Text cleaning (docs/09 §6.4): URLs, @mentions, hashtags, emoji (kept as a feature), runs of the
same character collapsed to two, then ``galecore.bangla.normalize``."""

from __future__ import annotations

import re

from galecore.bangla import normalize

URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
MENTION = re.compile(r"(?<!\w)@\w+")
HASHTAG = re.compile(r"(?<!\w)#(\w+)")
EMOJI = re.compile(
    "[\U0001f000-\U0001faff\U00002600-\U000027bf\U0001f1e6-\U0001f1ff\U0000fe0f\U0000200d\U00002300-\U000023ff]+"
)
REPEAT = re.compile(r"(.)\1{2,}")


def clean(text: str) -> tuple[str, bool]:
    """Return (cleaned normalised text, emoji_present)."""
    emoji = bool(EMOJI.search(text))
    t = URL.sub(" ", text)
    t = MENTION.sub(" ", t)
    t = HASHTAG.sub(r"\1", t)  # keep the hashtag word, drop the sign
    t = EMOJI.sub(" ", t)
    t = REPEAT.sub(r"\1\1", t)
    return normalize(t), emoji

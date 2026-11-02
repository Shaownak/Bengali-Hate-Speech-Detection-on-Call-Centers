from pathlib import Path

import pytest

from gale_text_worker.lexicon import LexiconMatcher, _edit_distance_le1
from gale_text_worker.register import RuleRegisterDetector, observe_level, word_index_at
from gale_text_worker.steps import TextWorker
from galecore.bangla import normalize
from galecore.config import LexiconEntry, Settings, load_lexicon, load_thresholds
from galecore.models import load_models
from galecore.schemas import AsrSegment, Category, Channel, SignalSource, Word

CONFIGS = Path(__file__).resolve().parents[3] / "configs"


def entry(term: str, cat: str = "slang_profanity", sev: int = 2, **kw: object) -> LexiconEntry:
    return LexiconEntry(id=f"t_{term}", term=term, category=cat, severity=sev, **kw)  # type: ignore[arg-type]


def seg(text: str, ch: Channel = Channel.AGENT, i: int = 0, t0: float = 10.0) -> AsrSegment:
    words = [
        Word(word=w, start=t0 + k * 0.5, end=t0 + k * 0.5 + 0.4, conf=0.9)
        for k, w in enumerate(text.split())
    ]
    return AsrSegment(
        call_id="c1",
        channel=ch,
        segment_id=f"s{i}",
        ts_start=t0,
        ts_end=t0 + 0.5 * len(words),
        text_redacted=text,
        words=words,
        is_final=True,
        avg_conf=0.9,
        model_version="t",
    )


def test_edit_distance() -> None:
    assert _edit_distance_le1("abcde", "abcde")
    assert _edit_distance_le1("abcde", "abxde")
    assert _edit_distance_le1("abcde", "abde")
    assert _edit_distance_le1("abde", "abcde")
    assert not _edit_distance_le1("abcde", "axcxe")
    assert not _edit_distance_le1("abc", "abcde")


def test_lexicon_exact_fuzzy_phonetic_and_multiword() -> None:
    m = LexiconMatcher(
        [
            entry("<GALI_1>", sev=3),
            entry("<GALI_11>"),
            entry("শয়তানি", "insult", romanized=["shoytani"]),
            entry("ফালতু লোক", "insult", 1),
        ]
    )
    hits = m.match(normalize("তুই <GALI_11> আর <GALI_1>, একটা ফালতু লোক!"))
    assert [(h.entry.term, h.match) for h in hits] == [
        ("<GALI_11>", "exact"),
        ("<GALI_1>", "exact"),
        ("ফালতু লোক", "exact"),
    ]
    assert hits[2].surface == "ফালতু লোক"
    fuzzy = m.match(normalize("শয়তানী"))  # ি → ী: one edit
    assert [(h.entry.term, h.match) for h in fuzzy] == [("শয়তানি", "fuzzy")]
    assert m.match(normalize("shoytani"))[0].match == "exact"  # romanised form
    # placeholders never match approximately
    assert m.match("<GALI_2>") == []


def test_lexicon_phonetic_key() -> None:
    m = LexiconMatcher([entry("শালা", sev=1)])  # short: phonetic only (no fuzzy below 5 letters)
    hits = m.match(normalize("সালা"))
    assert [(h.entry.term, h.match, h.score) for h in hits] == [("শালা", "phonetic", 0.8)]


def test_observe_level_and_verbs() -> None:
    assert observe_level("আপনি একটু বলুন") is not None
    o = observe_level("তুমি শুনছ না কেন?")
    assert o is not None
    assert (o.level, o.pronoun, o.verb_agrees) == ("tumi", "তুমি", True)
    mixed = observe_level("আপনি আর তুই")
    assert mixed is not None
    assert mixed.level == "tui"  # the lowest register in the utterance wins
    assert observe_level("জি, ঠিক আছে") is None


def test_word_index_at() -> None:
    s = seg("এক দুই তিন")
    assert [word_index_at(s, p) for p in (0, 1, 2, 3, 7, 30)] == [0, 0, None, 1, 2, None]


def test_register_agent_drops_fire_and_baseline_suppresses() -> None:
    r = RuleRegisterDetector()
    assert r.observe(seg("আপনি একটু অপেক্ষা করুন", i=0)) == []
    sig = r.observe(seg("তুমি বুঝতেছ না কেন?", i=1, t0=20.0))
    assert len(sig) == 1
    assert sig[0].category is Category.DISRESPECT_REGISTER
    assert (sig[0].ts_start, sig[0].ts_end) == (20.0, 20.4)  # the pronoun's word timing
    assert sig[0].evidence.features["to_level"] == 1.0
    assert sig[0].score == 0.95  # verb agrees
    assert r.observe(seg("তুমি শোনো", i=2)) == []  # same level again: no new signal
    tui = r.observe(seg("তুই ফোন রাখ", i=3))
    assert [s.evidence.features["to_level"] for s in tui] == [2.0]
    # a customer who starts with তুমি uses it neutrally (dialect): never fires for তুমি
    assert r.observe(seg("তুমি কেমন আছ", Channel.CUSTOMER, 4)) == []
    assert r.observe(seg("তুমি আবার বলো", Channel.CUSTOMER, 5), negative_context=True) == []


def test_register_customer_needs_negative_context() -> None:
    r = RuleRegisterDetector()
    r.observe(seg("আপনি বলেন", Channel.CUSTOMER, 0))
    assert r.observe(seg("তুমি শুনছ?", Channel.CUSTOMER, 1)) == []
    r2 = RuleRegisterDetector()
    r2.observe(seg("আপনি বলেন", Channel.CUSTOMER, 0))
    sig = r2.observe(seg("তুমি <GALI_19>", Channel.CUSTOMER, 1), negative_context=True)
    assert len(sig) == 1
    assert sig[0].score == 0.8


@pytest.fixture
def worker() -> TextWorker:
    s = Settings(env="test", config_dir=CONFIGS)
    models = load_models(s)
    return TextWorker(
        LexiconMatcher(load_lexicon(s).entries),
        models.text_clf,
        RuleRegisterDetector(),
        load_thresholds(CONFIGS),
    )


def test_text_worker_signals(worker: TextWorker) -> None:
    assert worker.process(seg("আপনি একটু অপেক্ষা করুন", Channel.CUSTOMER, 0)) == []
    sigs = worker.process(seg("তোকে আমি দেখে নেব, <GALI_17>!", Channel.CUSTOMER, 1, t0=40.0))
    by = {(s.source, s.category) for s in sigs}
    assert (SignalSource.LEXICON, Category.THREAT) in by
    assert (SignalSource.TEXT_CLF, Category.THREAT) in by
    assert (
        SignalSource.REGISTER,
        Category.DISRESPECT_REGISTER,
    ) in by  # negative context via lexicon
    lex = next(s for s in sigs if s.source is SignalSource.LEXICON)
    assert (lex.ts_start, lex.ts_end) == (42.0, 42.4)  # 5th word
    assert lex.evidence.term_id == "gali_17"
    assert lex.evidence.features["severity"] == 3.0
    assert not any(
        s.source is SignalSource.TEXT_CLF and s.category is Category.DISRESPECT_REGISTER
        for s in sigs
    )

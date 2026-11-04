"""SFT windows for the LLM judge from the open data (docs/03 §7.2, docs/08 §4, docs/09 §7).

Before in-house gold exists, each window is composed around open-data utterances: a labelled
comment becomes one speaker's turn inside a templated call (``templates_bn.yaml``), with simulated
fast-path signals and provisional events rendered through the production user-message format
(``gale_llm_judge.windows.render_user_message``). The target JSON follows the conventions fusion
relies on (``gale_fusion_engine.fusion.apply_judgements`` / ``llm_events``):

- a provisional event gets a decision (confirm / dismiss / change_category / change_severity) and
  is not repeated in ``events``; ``events`` holds only findings no provisional event covers;
- ``shouting`` / ``harsh_tone`` appear only when ACOUSTIC_SIGNALS support them (rule 5);
- reported speech, banter and self-directed swearing carry ``context_flags`` and severity 1, and a
  provisional event on them is dismissed (D-22: gold with context flags ends dismissed/absent);
- spoken prompt-injection lines are data: the window is judged as if they were not there.

Scenarios (``SCENARIOS``) set the mix: abuse by either speaker, keyword-free agent behaviour
(register, dismissive, sarcasm), hard negatives (complaints, policy, open-data negatives, banter),
reported speech and injection. Every abusive utterance comes from the open data; nothing here
contains a real slur. Text is redacted before it is rendered (D-01).

    python -m ml.llm.sft --train 12000 --dev 400 --test 400   # + --redteam 200 --calls 1500
"""

from __future__ import annotations

import argparse
import json
import random
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from gale_llm_judge.windows import render_user_message, system_prompt, utterance_ids
from galecore.bangla import normalize
from galecore.config import REPO_ROOT
from galecore.models.base import WindowInput
from galecore.redact import redact
from galecore.schemas import (
    AsrSegment,
    Category,
    Channel,
    Evidence,
    FusedEvent,
    JudgedEvent,
    ProvisionalDecision,
    Signal,
    SignalSource,
    Speaker,
    WindowJudgement,
)
from ml.data.open.schema import UnifiedRow

TEMPLATES = Path(__file__).with_name("templates_bn.yaml")
UNIFIED = REPO_ROOT / ".gale" / "data" / "open" / "unified.jsonl"
OUT = REPO_ROOT / ".gale" / "data" / "llm"

TEXT_CATEGORIES = (
    "slang_profanity",
    "insult",
    "threat",
    "sexual_harassment",
    "hate_identity",
    "sarcasm_mockery",
    "dismissive_rude",
)
SFT_SOURCES = {
    "banglacat",
    "banglamultihate",
    "violens",
    "banglaharassnet",
    "codemixed_chaos",
    "cyberbully_multilabel",
    "karim_bn_hate",
    "trac2_bn",
    "brac_online_comments",
}
MAX_CHARS = 280  # keeps a window inside the 1,500-token input budget (docs/08 §1)
SHOUT_P, SHOUT_Z, ANGER_P = 0.8, 2.0, 0.75  # rule-5 thresholds the targets follow
FLAGGED = ("reported_speech", "banter", "self_directed")
CONFUSABLE = {  # categories the text classifier mixes up (the judge corrects them)
    "insult": ["slang_profanity", "sarcasm_mockery"],
    "slang_profanity": ["insult"],
    "sarcasm_mockery": ["insult"],
    "hate_identity": ["insult"],
    "threat": ["insult"],
    "sexual_harassment": ["slang_profanity", "insult"],
}
TEXT_LIKE = (*TEXT_CATEGORIES, "disrespect_register")  # categories a text provisional can become
TARGETS = {"individual": None, "organization": "company", "group": "third_party"}

# scenario → share of the mix (docs/08 §4: gold-like 55–65 %, hard negatives 15–20 %)
SCENARIOS: dict[str, float] = {
    "customer_abuse": 0.36,
    "agent_abuse": 0.10,
    "agent_behaviour": 0.10,
    "customer_register": 0.02,
    "reported": 0.05,
    "open_negative": 0.15,
    "complaint": 0.10,
    "banter_policy": 0.06,
    "injection": 0.03,
    "two_abusive": 0.03,
}


@dataclass
class Finding:
    """One gold behaviour on one utterance."""

    seg: int  # index into the window's segments
    speaker: Speaker
    category: str
    severity: int
    confidence: float
    target: str | None = None
    flags: list[str] = field(default_factory=list)
    rationale: str = ""
    quote: str | None = None  # evidence when only part of the utterance is the behaviour


@dataclass
class Draft:
    segs: list[AsrSegment]
    findings: list[Finding]
    signals: list[Signal] = field(default_factory=list)
    provisional: list[tuple[FusedEvent, int]] = field(default_factory=list)  # (event, seg index)
    meta: dict[str, Any] = field(default_factory=dict)


class Builder:
    def __init__(self, rng: random.Random, templates: dict[str, Any]) -> None:
        self.rng, self.t = rng, templates

    # --- utterances --------------------------------------------------------------------------

    def _segs(self, call_id: str, turns: list[tuple[Channel, str]]) -> list[AsrSegment]:
        t = self.rng.uniform(15, 420)
        out = []
        for i, (ch, text) in enumerate(turns):
            dur = max(1.2, 0.35 * len(text.split()) + self.rng.uniform(0.2, 1.0))
            out.append(
                AsrSegment(
                    call_id=call_id,
                    channel=ch,
                    segment_id=f"{call_id}:{i}",
                    ts_start=round(t, 1),
                    ts_end=round(t + dur, 1),
                    text_redacted=normalize(text),  # as the ASR worker does (CLAUDE.md)
                    is_final=True,
                    avg_conf=round(self.rng.uniform(0.7, 0.97), 2),
                    model_version="open_synthetic",
                )
            )
            t += dur + self.rng.uniform(0.2, 1.2)
        return out

    def pick(self, key: str) -> Any:
        return self.rng.choice(self.t[key])

    def _context(self, n: int) -> list[tuple[Channel, str]]:
        """``n`` neutral turns, alternating and starting with either side."""
        ch = self.rng.choice([Channel.AGENT, Channel.CUSTOMER])
        turns = []
        for _ in range(n):
            pool = "agent_neutral" if ch == Channel.AGENT else "customer_neutral"
            turns.append((ch, self.pick(pool)))
            ch = Channel.CUSTOMER if ch == Channel.AGENT else Channel.AGENT
        return turns

    def _frame(
        self, middle: list[tuple[Channel, str]], before: int | None = None
    ) -> tuple[list[tuple[Channel, str]], int]:
        """Neutral context, the utterance(s) of interest, maybe an agent reply. Returns the turns
        and the index of the first utterance of interest."""
        before = self.rng.choice([1, 1, 2, 3]) if before is None else before
        turns = self._context(before)
        at = len(turns)
        turns += middle
        if self.rng.random() < 0.6:
            turns.append((Channel.AGENT, self.pick("agent_reply")))
        return turns, at

    # --- labels ------------------------------------------------------------------------------

    def rationale(self, cat: str, speaker: Speaker, p: float | None = None) -> str:
        who = self.t["who"][speaker.value]
        return str(self.rng.choice(self.t["rationale"][cat])).format(
            who=who, p=f"{p:.2f}" if p is not None else ""
        )

    def row_findings(self, r: UnifiedRow, seg: int, speaker: Speaker) -> list[Finding]:
        cats = [c for c in TEXT_CATEGORIES if r.labels.get(c) == 1]
        base = r.severity if r.severity and 1 <= r.severity <= 3 else 2
        out = []
        for c in cats:
            sev = base
            if c in ("threat", "sexual_harassment"):
                sev = 3  # docs/08 rule 7
            elif speaker == Speaker.AGENT and c in ("slang_profanity", "insult", "hate_identity"):
                sev = min(3, base + 1)  # severe gali by the agent
            conf = r.label_confidence.get(c)
            if conf is not None and conf < 1:
                sev = min(sev, 1)  # weak mapping → mild/ambiguous
            tgt = TARGETS.get(r.target or "")
            if tgt is None:
                tgt = "agent" if speaker == Speaker.CUSTOMER else "customer"
                if c == "hate_identity":
                    tgt = "third_party"
            out.append(
                Finding(
                    seg=seg,
                    speaker=speaker,
                    category=c,
                    severity=sev,
                    confidence=round(conf if conf is not None else self.rng.uniform(0.78, 0.96), 2),
                    target=tgt,
                    rationale=self.rationale(c, speaker),
                )
            )
        return out

    # --- signals & provisional events --------------------------------------------------------

    def _sig(
        self,
        d: Draft,
        seg: int,
        source: SignalSource,
        cat: str,
        p: float,
        **features: float,
    ) -> None:
        s = d.segs[seg]
        d.signals.append(
            Signal(
                call_id=s.call_id,
                channel=s.channel,
                signal_id=str(uuid.UUID(int=self.rng.getrandbits(128))),
                source=source,
                category=Category(cat),
                score=round(p, 2),
                ts_start=s.ts_start,
                ts_end=s.ts_end,
                evidence=Evidence(segment_id=s.segment_id, features=features),
                model_version="sim",
            )
        )

    def acoustics(self, d: Draft, loud_segs: set[int]) -> None:
        """Per-utterance acoustic summary; ``loud_segs`` get shouting-level values. Rule 5: the
        target lists shouting / harsh_tone exactly when the signals cross the thresholds."""
        r = self.rng
        for i, s in enumerate(d.segs):
            loud = i in loud_segs
            if not loud and r.random() < 0.45:
                continue  # not every utterance carries acoustic signals
            shout = r.uniform(0.8, 0.98) if loud else r.uniform(0.02, 0.45)
            z = r.uniform(2.1, 3.6) if loud else r.uniform(-0.9, 1.6)
            anger = r.uniform(0.7, 0.96) if loud else r.uniform(0.03, 0.55)
            if not loud and r.random() < 0.03:
                anger = r.uniform(0.76, 0.92)  # tense but not loud
            self._sig(d, i, SignalSource.DSP_SHOUT, "shouting", shout, loudness_z=round(z, 2))
            self._sig(d, i, SignalSource.SER, "harsh_tone", anger)
            sp = Speaker(s.channel.value)
            shouting = shout >= SHOUT_P and z >= SHOUT_Z
            if shouting:
                sev = 2 if sp == Speaker.AGENT else 1
                d.findings.append(
                    Finding(
                        i, sp, "shouting", sev, round(shout, 2),
                        rationale=self.rationale("shouting", sp, shout),
                    )
                )  # fmt: skip
            elif anger >= ANGER_P:
                d.findings.append(
                    Finding(
                        i, sp, "harsh_tone", 1, round(anger, 2),
                        rationale=self.rationale("harsh_tone", sp, anger),
                    )
                )  # fmt: skip

    def text_signals(self, d: Draft) -> None:
        """Fast-path text signals: real findings mostly score high, clean utterances sometimes
        get a spurious score (the hard negatives the judge must dismiss)."""
        r = self.rng
        by_seg: dict[int, list[Finding]] = defaultdict(list)
        for f in d.findings:
            by_seg[f.seg].append(f)
        for i in range(len(d.segs)):
            fs = [f for f in by_seg[i] if f.category in TEXT_CATEGORIES]
            for f in fs:
                if r.random() < 0.85:
                    cat = f.category
                    if r.random() < 0.15 and cat in CONFUSABLE:  # fast path picks a neighbour
                        cat = r.choice(CONFUSABLE[cat])
                    self._sig(d, i, SignalSource.TEXT_CLF, cat, r.uniform(0.55, 0.97))
                if f.category == "slang_profanity" and r.random() < 0.5:
                    self._sig(d, i, SignalSource.LEXICON, "slang_profanity", 1.0)
                    d.signals[-1].evidence.term_id = f"term_{r.randint(1, 400)}"
            if [f for f in by_seg[i] if f.category == "disrespect_register"]:
                self._sig(d, i, SignalSource.REGISTER, "disrespect_register", r.uniform(0.6, 0.95))
            if not fs and r.random() < 0.18:
                c = r.choice(["insult", "slang_profanity", "threat", "sarcasm_mockery"])
                self._sig(d, i, SignalSource.TEXT_CLF, c, r.uniform(0.45, 0.8))

    def provisionals(self, d: Draft) -> None:
        """Provisional events where the fast path would raise one: strong text signals and loud
        utterances, sometimes with a different category or severity than the gold."""
        r = self.rng
        seen: set[tuple[int, str]] = set()
        for sig in list(d.signals):
            i = next(k for k, s in enumerate(d.segs) if s.segment_id == sig.evidence.segment_id)
            cat = sig.category.value
            if (i, cat) in seen:
                continue
            strong = sig.score >= 0.6 or sig.source == SignalSource.LEXICON
            if sig.source == SignalSource.DSP_SHOUT:
                strong = sig.score >= SHOUT_P and sig.evidence.features["loudness_z"] >= SHOUT_Z
            if sig.source == SignalSource.SER or not strong or r.random() > 0.7:
                continue
            seen.add((i, cat))
            gold = [f for f in d.findings if f.seg == i]
            sev = next((f.severity for f in gold if f.category == cat), r.choice([1, 2]))
            if r.random() < 0.2:
                sev = r.choice([x for x in (1, 2, 3) if x != sev])
            s = d.segs[i]
            ev = FusedEvent(
                event_id=str(uuid.UUID(int=r.getrandbits(128))),
                call_id=s.call_id,
                speaker=Speaker(s.channel.value),
                category=Category(cat),
                severity=sev,  # type: ignore[arg-type]
                probability=round(sig.score, 2),
                ts_start=s.ts_start,
                ts_end=s.ts_end,
            )
            d.provisional.append((ev, i))

    # --- target ------------------------------------------------------------------------------

    def target(self, d: Draft, uids: dict[str, str]) -> WindowJudgement:
        covered: set[int] = set()  # finding indexes handled by a decision
        decisions = []
        reason = self.t["reason"]
        for ev, i in d.provisional:
            gold = [(k, f) for k, f in enumerate(d.findings) if f.seg == i and k not in covered]
            same = [(k, f) for k, f in gold if f.category == ev.category.value]
            if same and any(fl in FLAGGED for fl in same[0][1].flags):
                covered.add(same[0][0])
                decisions.append(
                    ProvisionalDecision(
                        event_id=ev.event_id,
                        decision="dismiss",
                        reason_bn=self.rng.choice(reason["dismiss_flagged"]),
                    )
                )
            elif same:
                k, f = same[0]
                covered.add(k)
                if f.severity != ev.severity:
                    decisions.append(
                        ProvisionalDecision(
                            event_id=ev.event_id,
                            decision="change_severity",
                            new_severity=f.severity,  # type: ignore[arg-type]
                            reason_bn=self.rng.choice(reason["change_severity"]).format(
                                new=f.severity
                            ),
                        )
                    )
                else:
                    decisions.append(
                        ProvisionalDecision(
                            event_id=ev.event_id,
                            decision="confirm",
                            reason_bn=self.rng.choice(reason["confirm"]),
                        )
                    )
            elif (text_gold := [(k, f) for k, f in gold if f.category in TEXT_LIKE]) and not any(
                fl in FLAGGED for _, f in gold for fl in f.flags
            ):
                k, f = text_gold[0]
                covered.add(k)
                decisions.append(
                    ProvisionalDecision(
                        event_id=ev.event_id,
                        decision="change_category",
                        new_category=Category(f.category),
                        new_severity=f.severity if f.severity != ev.severity else None,  # type: ignore[arg-type]
                        reason_bn=self.rng.choice(reason["change_category"]).format(new=f.category),
                    )
                )
            else:
                flagged = any(fl in FLAGGED for _, f in gold for fl in f.flags)
                decisions.append(
                    ProvisionalDecision(
                        event_id=ev.event_id,
                        decision="dismiss",
                        reason_bn=self.rng.choice(
                            reason["dismiss_flagged" if flagged else "dismiss_clean"]
                        ),
                    )
                )
        events = []
        for k, f in sorted(enumerate(d.findings), key=lambda kf: (kf[1].seg, kf[1].category)):
            if k in covered:
                continue
            s = d.segs[f.seg]
            events.append(
                JudgedEvent(
                    speaker=f.speaker,
                    category=Category(f.category),
                    severity=f.severity,  # type: ignore[arg-type]
                    target=f.target,  # type: ignore[arg-type]
                    confidence=f.confidence,
                    evidence=self.evidence(f, s.text_redacted),
                    utterance_id=uids[s.segment_id],
                    context_flags=f.flags,  # type: ignore[arg-type]
                    rationale_bn=f.rationale,
                )
            )
        return WindowJudgement(events=events, provisional_decisions=decisions)

    @staticmethod
    def evidence(f: Finding, text: str) -> str:
        return f.quote if f.quote and f.quote in text else text

    # --- scenarios ---------------------------------------------------------------------------

    def window(
        self, scenario: str, rows: Iterator[UnifiedRow], negs: Iterator[UnifiedRow]
    ) -> Draft:
        r = self.rng
        call_id = str(uuid.UUID(int=r.getrandbits(128)))
        findings: list[Finding] = []
        meta: dict[str, Any] = {"scenario": scenario}
        loud: set[int] = set()

        def text_of(row: UnifiedRow) -> str:
            return normalize(redact(row.text_bn).text)

        if scenario in ("customer_abuse", "agent_abuse", "injection", "two_abusive"):
            row = next(rows)
            sp = Speaker.AGENT if scenario == "agent_abuse" else Speaker.CUSTOMER
            ch = Channel(sp.value)
            middle = [(ch, text_of(row))]
            ab = 0  # offset of the abusive turn inside ``middle``
            if scenario == "injection":
                ab = r.randrange(2)
                middle.insert(1 - ab, (Channel.CUSTOMER, self.pick("injection")))
            row2 = None
            if scenario == "two_abusive":
                row2 = next(rows)
                middle += [(Channel.AGENT, self.pick("agent_reply")), (ch, text_of(row2))]
            turns, at = self._frame(middle)
            ab += at
            findings += self.row_findings(row, ab, sp)
            if row2 is not None:
                findings += self.row_findings(row2, at + 2, sp)
            if r.random() < (0.35 if sp == Speaker.CUSTOMER else 0.2):
                loud.add(ab)
            meta.update(dataset=row.source, uid=row.uid, split=row.split)
        elif scenario == "reported":
            row = next(rows)
            quote = text_of(row)
            wrapped = str(self.pick("reported_wrappers")).format(text=quote)
            turns, at = self._frame([(Channel.CUSTOMER, wrapped)])
            for f in self.row_findings(row, at, Speaker.CUSTOMER):
                f.flags, f.severity, f.target = ["reported_speech"], 1, "third_party"
                f.rationale = str(self.t["flagged"]).format(who=self.t["who"]["customer"])
                f.quote = quote
                findings.append(f)
            meta.update(dataset=row.source, uid=row.uid, split=row.split)
        elif scenario in ("agent_behaviour", "customer_register"):
            key = (
                "customer_register"
                if scenario == "customer_register"
                else r.choice(
                    ["agent_register", "agent_register", "agent_dismissive", "agent_sarcasm"]
                )
            )
            item = self.pick(key)
            sp = Speaker.CUSTOMER if key == "customer_register" else Speaker.AGENT
            cat = {
                "agent_register": "disrespect_register",
                "customer_register": "disrespect_register",
                "agent_dismissive": "dismissive_rude",
                "agent_sarcasm": "sarcasm_mockery",
            }[key]
            pre = [(Channel.AGENT, self.pick("agent_neutral"))] if sp == Speaker.AGENT else []
            middle = [
                *pre,
                (Channel.CUSTOMER, self.pick("customer_complaint")),
                (Channel(sp.value), item["text"]),
            ]
            turns, at = self._frame(middle, before=r.choice([0, 1]))
            i = at + len(middle) - 1
            other = "customer" if sp == Speaker.AGENT else "agent"
            for c in [cat, *([item["also"]] if item.get("also") else [])]:
                findings.append(
                    Finding(i, sp, c, int(item["severity"]), round(r.uniform(0.75, 0.93), 2),
                            target=other, rationale=self.rationale(c, sp))
                )  # fmt: skip
            meta["dataset"] = "templates"
        elif scenario == "open_negative":
            row = next(negs)
            turns, at = self._frame([(Channel.CUSTOMER, text_of(row))])
            if r.random() < 0.15:
                loud.add(at)
            meta.update(dataset=row.source, uid=row.uid, split=row.split)
        elif scenario == "injection_clean":  # red-team only: an injection in a clean window
            middle = [(Channel.CUSTOMER, self.pick("customer_complaint"))]
            middle.insert(r.randrange(2), (Channel.CUSTOMER, self.pick("injection")))
            turns, at = self._frame(middle)
            meta["dataset"] = "templates"
        elif scenario == "complaint":
            turns, at = self._frame([(Channel.CUSTOMER, self.pick("customer_complaint"))])
            if r.random() < 0.4:
                loud.add(at)  # an angry, loud complaint: shouting, but not abuse
            meta["dataset"] = "templates"
        else:  # banter_policy
            kind = r.choice(["customer_self_directed", "customer_banter", "agent_policy"])
            ch = Channel.AGENT if kind == "agent_policy" else Channel.CUSTOMER
            turns, at = self._frame([(ch, self.pick(kind))])
            meta["dataset"] = "templates"
        d = Draft(self._segs(call_id, turns), findings, meta=meta)
        self.acoustics(d, loud)
        self.text_signals(d)
        self.provisionals(d)
        return d

    def example(self, d: Draft, queue: str) -> dict[str, Any]:
        window = WindowInput(
            d.segs[0].call_id, d.segs, d.signals, [e for e, _ in d.provisional], queue
        )
        ids = utterance_ids(d.segs)
        target = self.target(d, ids)
        return {
            "messages": [
                {"role": "system", "content": system_prompt("window")},
                {"role": "user", "content": render_user_message(window, ids)},
                {"role": "assistant", "content": target.model_dump_json(exclude_none=True)},
            ],
            "meta": {
                "source": "open_synthetic" if d.meta.get("dataset") != "templates" else "synthetic",
                "research_only": True,
                **d.meta,
            },
        }


QUEUES = ["billing", "technical", "sales", "cards", "delivery", "general", "unknown"]


def _cycle(rows: list[UnifiedRow], rng: random.Random) -> Iterator[UnifiedRow]:
    while True:
        rng.shuffle(rows)
        yield from rows


def balanced(rows: list[UnifiedRow], rng: random.Random) -> list[UnifiedRow]:
    """Abusive rows, re-weighted so rare categories (threat, sexual harassment, sarcasm) are seen
    often enough: each row is kept with probability ∝ 1/√(frequency of its rarest category)."""
    freq = Counter(c for r in rows for c in TEXT_CATEGORIES if r.labels.get(c) == 1)
    top = max(freq.values()) if freq else 1
    out = []
    for r in rows:
        cats = [c for c in TEXT_CATEGORIES if r.labels.get(c) == 1]
        keep = max((top / freq[c]) ** 0.5 for c in cats) / (top / min(freq.values())) ** 0.5
        if rng.random() < max(keep, 0.25):
            out.append(r)
    return out


def load_rows(path: Path = UNIFIED) -> dict[str, tuple[list[UnifiedRow], list[UnifiedRow]]]:
    """split → (abusive rows, clean rows) from the SFT sources, originals only (no aug copies),
    short enough for the window budget, no context-dependent rows."""
    out: dict[str, tuple[list[UnifiedRow], list[UnifiedRow]]] = defaultdict(lambda: ([], []))
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = UnifiedRow.model_validate_json(line)
            if r.source not in SFT_SOURCES or r.synthetic or r.split is None:
                continue
            if len(r.text_bn) > MAX_CHARS or len(r.text_bn) < 6:
                continue
            pos = any(r.labels.get(c) == 1 for c in TEXT_CATEGORIES)
            if pos:
                out[r.split][0].append(r)
            elif r.toxic_any == 0:
                out[r.split][1].append(r)
    return dict(out)


def build(
    rows: dict[str, tuple[list[UnifiedRow], list[UnifiedRow]]],
    sizes: dict[str, int],
    seed: int = 17,
    templates: dict[str, Any] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    tpl = templates or yaml.safe_load(TEMPLATES.read_text(encoding="utf-8"))
    out: dict[str, list[dict[str, Any]]] = {}
    for split, n in sizes.items():
        rng = random.Random(f"{seed}:{split}")
        pos, neg = rows.get(split, ([], []))
        if not pos or not neg:
            out[split] = []
            continue
        b = Builder(rng, tpl)
        prow, nrow = _cycle(balanced(list(pos), rng), rng), _cycle(list(neg), rng)
        names, weights = zip(*SCENARIOS.items(), strict=True)
        recs = []
        for _ in range(n):
            sc = rng.choices(names, weights)[0]
            recs.append(b.example(b.window(sc, prow, nrow), rng.choice(QUEUES)))
        out[split] = recs
    return out


def build_redteam(
    rows: dict[str, tuple[list[UnifiedRow], list[UnifiedRow]]],
    n: int,
    seed: int = 17,
    templates: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Red-team windows (docs/08 §6) from the test split, with the held-out
    ``redteam_injection`` phrases only (never seen in training): half inside abusive windows
    (the abuse must still be found), half inside clean ones (nothing may be flagged, including
    the agent the injection tries to frame)."""
    tpl = dict(templates or yaml.safe_load(TEMPLATES.read_text(encoding="utf-8")))
    tpl["injection"] = tpl["redteam_injection"]
    rng = random.Random(f"{seed}:redteam")
    pos, neg = rows.get("test", ([], []))
    b = Builder(rng, tpl)
    prow, nrow = _cycle(list(pos), rng), _cycle(list(neg), rng)
    out = []
    for i in range(n):
        sc = "injection" if i % 2 == 0 else "injection_clean"
        ex = b.example(b.window(sc, prow, nrow), rng.choice(QUEUES))
        ex["meta"]["scenario"] = f"redteam_{sc}"
        out.append(ex)
    return out


def stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    sc: Counter[str] = Counter()
    cats: Counter[str] = Counter()
    dec: Counter[str] = Counter()
    empty = 0
    for x in records:
        sc[x["meta"]["scenario"]] += 1
        t = json.loads(x["messages"][2]["content"])
        empty += not t["events"] and not t["provisional_decisions"]
        cats.update(e["category"] for e in t["events"])
        dec.update(p["decision"] for p in t["provisional_decisions"])
    return {
        "n": len(records),
        "empty_targets": empty,
        "scenarios": dict(sc),
        "event_categories": dict(cats),
        "decisions": dict(dec),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--unified", type=Path, default=UNIFIED)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--train", type=int, default=12000)
    ap.add_argument("--dev", type=int, default=400)
    ap.add_argument("--test", type=int, default=400)
    ap.add_argument("--calls", type=int, default=1500, help="call-mode scorecards (train)")
    ap.add_argument("--calls-eval", type=int, default=100)
    ap.add_argument("--redteam", type=int, default=200, help="held-out injection windows")
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args(argv)
    rows = load_rows(a.unified)
    sets = build(rows, {"train": a.train, "dev": a.dev, "test": a.test}, a.seed)
    if a.redteam:
        sets["redteam"] = build_redteam(rows, a.redteam, a.seed)
    from ml.llm.sft_call import build_calls

    calls = build_calls(rows, {"train": a.calls, "dev": a.calls_eval, "test": a.calls_eval}, a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    report = {}
    for split, recs in sets.items():
        p = a.out / f"sft_window_{split}.jsonl"
        p.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in recs), "utf-8")
        report[split] = stats(recs)
        print(f"{split}: {len(recs)} windows → {p}")
    for split, recs in calls.items():
        p = a.out / f"sft_call_{split}.jsonl"
        p.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in recs), "utf-8")
        report[f"call_{split}"] = {"n": len(recs)}
        print(f"call {split}: {len(recs)} scorecards → {p}")
    (a.out / "sft_stats.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

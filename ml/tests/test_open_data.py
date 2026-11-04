"""Open-data pipeline tests on tiny synthetic files in each raw format. No real content: abusive
words are <GALI_n> placeholders (CLAUDE.md rule 2)."""

import csv
import io
import json
import random
import zipfile
from pathlib import Path

import httpx
import numpy as np
import pytest
import soundfile as sf

from galecore.schemas import WindowJudgement
from ml.data.open.augment import AsrNoise, rewrite_spoken
from ml.data.open.build import build, write_outputs
from ml.data.open.clean import clean
from ml.data.open.dedupe import duplicate_groups, jaccard, shingles
from ml.data.open.download import ChecksumError, download
from ml.data.open.harmonise import check_no_leakage, harmonise
from ml.data.open.label_map import LabelMapper
from ml.data.open.lid import keep, language_of, script_of
from ml.data.open.loaders import TEXT_LOADERS, MissingDataError
from ml.data.open.readers import ColumnError, read_table, resolve_columns
from ml.data.open.registry import (
    DATASETS_MD,
    LicenceError,
    check_allowed,
    load_register,
    register_markdown,
)
from ml.data.open.schema import LABELS, RawRow, UnifiedRow
from ml.data.open.translit import RuleTransliterator, rule_word
from ml.lexicon.mine import mine, write_csv

REG = load_register()
BN_SPLIT = {"train": "প্রশিক্ষণ", "dev": "যাচাই", "test": "পরীক্ষা"}
WORDS = [
    "আমি",
    "তুমি",
    "সে",
    "আমরা",
    "বাড়ি",
    "অফিস",
    "বাজার",
    "নদী",
    "আকাশ",
    "গাছ",
    "বই",
    "কলম",
    "টেবিল",
    "চেয়ার",
    "দরজা",
    "জানালা",
    "রাস্তা",
    "গাড়ি",
    "ট্রেন",
    "বাস",
]
NEUTRAL = [
    "আমার ইন্টারনেট সংযোগ কাজ করছে না",
    "বিলটা এই মাসে অনেক বেশি এসেছে",
    "রিচার্জ করার পরেও ব্যালেন্স আসেনি",
    "প্যাকেজটা কীভাবে বন্ধ করা যায় বলুন",
    "সন্ধ্যার দিকে নেট একদম ধীর হয়ে যায়",
    "এসএমএসটা এখনো আসেনি ভাই",
]


def write_csv_rows(
    path: Path, header: list[str], rows: list[list[object]], delim: str = ","
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(
            f,
            delimiter=delim,
            quoting=csv.QUOTE_NONE if delim == "\t" else csv.QUOTE_MINIMAL,
            escapechar="\\",
        )
        w.writerow(header)
        w.writerows(rows)


@pytest.fixture
def raw(tmp_path: Path) -> Path:
    r = tmp_path / "raw"
    # T1: unquoted TSV with official splits
    for split, n in (("1C_train.tsv", 12), ("1C_dev.tsv", 3), ("1C_test_with_labels.tsv", 4)):
        rows = []
        for i in range(n):
            ht = ["None", "Profane", "Abusive", "Political Hate"][i % 4]
            tag = {"1C_t": "প্রশিক্ষণ", "1C_d": "যাচাই", "1C_te": "পরীক্ষা"}.get(
                split[:5] if "test" in split else split[:4], "অন্য"
            )
            txt = (
                f"{NEUTRAL[i % 6]} {tag} {i}"
                if ht == "None"
                else f"তুই একটা <GALI_{i % 3 + 1}> {tag} {i}"
            )
            rows.append(
                [
                    f"{split}{i}",
                    txt,
                    ht,
                    ["Little to None", "Mild", "Severe", "Mild"][i % 4],
                    "Individual",
                ]
            )
        write_csv_rows(
            r / "banglamultihate" / split,
            ["id", "text", "hate_type", "hate_severity", "to_whom"],
            rows,
            "\t",
        )
    # T6: CSV 0/1/2
    for split in ("train", "dev", "test"):
        write_csv_rows(
            r / "violens" / f"{split}.csv",
            ["text", "label"],
            [[f"{NEUTRAL[i]} {BN_SPLIT[split]} ভায়োলেন্স {i}", i % 3] for i in range(6)],
        )
    # T8: parquet with integer classes
    import pyarrow as pa
    import pyarrow.parquet as pq

    (r / "karim_bn_hate").mkdir(parents=True)
    pq.write_table(
        pa.table({"text": [f"<GALI_13> করিম {i}" for i in range(5)], "label": [0, 1, 2, 3, 4]}),
        r / "karim_bn_hate" / "train.parquet",
    )
    # T5: one 0/1 column per class (incl. spam → dropped)
    write_csv_rows(
        r / "cyberbully_multilabel" / "cyberbully_multilabel.csv",
        ["Gender", "Profession", "comment", "bully", "sexual", "religious", "threat", "spam"],
        [
            ["female", "social", "<GALI_14> সাইবার এক", 1, 0, 0, 0, 0],
            ["male", "actor", "অফার পেতে লিংকে ক্লিক করুন সাইবার দুই", 0, 0, 0, 0, 1],
            ["female", "social", "সাধারণ মন্তব্য সাইবার তিন", 0, 0, 0, 0, 0],
        ],
    )
    # T12: one xlsx per harassment type, with a title row above the header
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.append(["Trolling — Moderate Abuse Level (Severity: 1)"])
    ws.append(["Post_ID", "Platform", "Text_Content", "Harassment_Type", "Severity_Level"])
    ws.append([1, "Facebook", "<GALI_16> ট্রল মন্তব্য এক", "Trolling ", 1.0])
    (r / "banglaharassnet").mkdir(parents=True)
    wb.save(r / "banglaharassnet" / "trolling.xlsx")
    # T3: unpacked Mendeley archive (nested folder), Bangla file + its English twin
    for name in ("bangla_context_toxicity_train", "bangla_context_toxicity_train_ENGLISH"):
        write_csv_rows(
            r / "banglacat" / "BanglaCAT_Mendeley 2" / "train" / f"{name}.csv",
            ["row_id", "previous_reply", "target_reply", "level_1_class", "toxicity_type"],
            [["a1", "আগের মন্তব্য", "<GALI_18> উপহাস ক্যাট", "Toxic", "Mockery / Ridicule"]],
        )
    # T2 + T4: BD-SHS parents and BIDWESH dialect children
    write_csv_rows(
        r / "bdshs" / "test.csv",
        ["sentence", "hate", "target", "type"],
        [["<GALI_17> বিডিএসএইচএস টেস্ট", 1, "ind", "callToViolence,slander"]],
    )
    write_csv_rows(
        r / "bdshs" / "train.csv",
        ["sentence", "hate", "target", "type"],
        [["সুন্দর একটা দিন বিডিএসএইচএস", 0, "", ""]],
    )
    write_csv_rows(
        r / "bidwesh" / "noakhali.csv",
        ["text", "bdshs_id", "dialect", "hate", "type"],
        [["<GALI_17> নোয়াখালী রূপ", "test:0", "Noakhali", 1, "callToViolence"]],
    )
    # R1: Banglish (Roman) rows to back-transliterate, plus English and Hindi rows to drop
    write_csv_rows(
        r / "banth" / "banth_train.csv",
        ["text", "hate", "labels"],
        [
            ["tumi ekta <GALI_2> bhai", 1, "Personal Offence"],
            ["this is a very good video and i like it", 0, ""],
            ["यह बहुत अच्छा है", 0, ""],
        ],
    )
    # R2: the Bangla workbook, one 0/1 column per class (xlsx floats)
    wb = Workbook()
    ws = wb.active
    ws.append(["Text", "Vulgar-based", "Troll-based", "Insult-based", "Non-Toxic"])
    ws.append(["তুমি একটা <GALI_15> কোডমিক্স", 0.0, 1.0, 0.0, 0.0])
    ws.append(["ভালো ভিডিও কোডমিক্স দুই", 0.0, 0.0, 0.0, 1.0])
    (r / "codemixed_chaos").mkdir(parents=True)
    wb.save(r / "codemixed_chaos" / "bangla.xlsx")
    # R3 eval-only, R4 pairs
    write_csv_rows(
        r / "mixtox_sa" / "mixtox.csv", ["text", "label"], [["মিক্সটক্স মন্তব্য একটি", "toxic"]]
    )
    write_csv_rows(
        r / "banglatlit" / "banglatlit_train.csv",
        ["romanized", "bangla"],
        [["ami valo achi", "আমি ভালো আছি"]],
    )
    # A1: BAAD-style wav folders
    for term in ("term_a", "term_b"):
        d = r / "baad" / term
        d.mkdir(parents=True)
        sf.write(d / "s01_1.wav", np.zeros(1600, np.float32), 16000)
    return r


def test_register_covers_docs09_and_is_rendered() -> None:
    ids = {d.id for d in REG.datasets}
    assert ids == {
        *(f"T{i}" for i in range(1, 16)),
        "R1",
        "R2",
        "R3",
        "R4",
        "A1",
        "A2",
        "A3",
        "A4",
        *(f"S{i}" for i in range(1, 9)),
    }
    mapper = LabelMapper()
    for d in REG.datasets:
        assert d.commercial_ok in ("true", "false", "unknown")
        if d.kind == "text":
            assert d.key in TEXT_LOADERS
            assert d.key in mapper.sources
        if d.download_allowed:  # every download lands as a `files` entry or an archive
            assert set(d.download) <= set(d.files) | set(d.archives)
            assert set(d.archives) <= set(d.download)
    assert DATASETS_MD.read_text(encoding="utf-8") == register_markdown(REG)


def test_licence_gate() -> None:
    t1, r3 = REG.get("T1"), REG.get("mixtox_sa")
    with pytest.raises(LicenceError, match="commercial_ok"):
        check_allowed(t1, "prod")
    assert check_allowed(t1, "research") is True
    ok = t1.model_copy(update={"commercial_ok": "true"})
    assert check_allowed(ok, "prod") is False
    with pytest.raises(LicenceError, match="evaluation-only"):
        check_allowed(r3.model_copy(update={"commercial_ok": "true"}), "prod")


def test_readers_and_columns(tmp_path: Path) -> None:
    (tmp_path / "a.jsonl").write_text('{"text": "x"}\n\n{"text": "y"}\n', encoding="utf-8")
    assert len(read_table(tmp_path / "a.jsonl")) == 2
    (tmp_path / "b.json").write_text(json.dumps({"data": [{"t": 1}]}), encoding="utf-8")
    assert read_table(tmp_path / "b.json") == [{"t": 1}]
    with pytest.raises(ValueError, match="unsupported"):
        read_table(tmp_path / "c.pdf")
    assert resolve_columns({"text": ["Sentence", "text"]}, ["sentence", "x"], ("text",), "s") == {
        "text": "sentence"
    }
    with pytest.raises(ColumnError, match="file has"):
        resolve_columns({"text": "comment"}, ["foo"], ("text",), "s")


def test_label_mapper_partial_weak_weight_drop() -> None:
    m = LabelMapper()

    def row(src: str, **fields: object) -> RawRow:
        return RawRow(source=src, source_id="1", text="x", split=None, fields=dict(fields))

    t1 = m.map(
        row(
            "banglamultihate",
            hate_type="Political Hate",
            hate_severity="Severe",
            to_whom="Community",
        )
    )
    assert t1.labels["hate_identity"] == 1
    assert t1.labels["insult"] == 0  # annotated by this source, not hit
    assert t1.labels["threat"] is None  # not annotated: no loss
    assert (t1.weight, t1.severity, t1.target, t1.toxic_any) == (0.3, 3, "group", 1)
    neg = m.map(row("banglamultihate", hate_type="None", hate_severity="Little to None"))
    assert (neg.toxic_any, neg.severity, neg.labels["slang_profanity"]) == (0, 0, 0)
    trac = m.map(row("trac2_bn", aggression="OAG", gendered="NGEN"))
    assert trac.labels["aggression"] == 1
    assert trac.confidence == {"insult": 0.6}  # weak mapping → soft target
    assert m.map(row("cyberbully_multilabel", bully="0", spam="1")).drop
    bidwesh = m.map(row("bidwesh", hate="1", type=["callToViolence"]))  # same_as bdshs
    assert bidwesh.labels["threat"] == 1
    cat = m.map(row("banglacat", toxic="1", types=["Dehumanization"], context_case="quoted speech"))
    assert cat.context_flags == ["reported_speech"]
    assert m.map(row("karim_bn_hate", label="Personal")).toxic_any == 1


def test_clean_lid_translit() -> None:
    text, emoji = clean("দেখুন https://x.co @user #অফার দারুণণণণ 😀")
    assert text == "দেখুন অফার দারুণণ"
    assert emoji
    assert script_of("আমি ami") == "mixed"
    assert script_of("১২৩") == "bangla"
    assert language_of("यह अच्छा है") == "hi"
    assert language_of("this is a very good video") == "en"
    assert language_of("tumi ekta kotha bolo") == "banglish"
    assert keep("ami valo achi")
    assert not keep("what is this and why")
    assert RuleTransliterator()(["ami tumi, <GALI_1>!"]) == ["আমি তুমি, <GALI_1>!"]
    assert all("ঀ" <= ch <= "৿" for ch in rule_word("bangladesh"))


def test_dedupe() -> None:
    a = "আমার ইন্টারনেট সংযোগ তিন দিন ধরে কাজ করছে না, খুব সমস্যা হচ্ছে"
    texts = [a, a + "।", a.replace("হচ্ছে", "হচ্ছেে"), "সম্পূর্ণ আলাদা একটি বাক্য এখানে আছে যা মিলবে না"]
    groups = duplicate_groups(texts)
    assert sorted(map(sorted, groups)) == [[0, 1, 2]]
    assert jaccard(shingles("abcdef"), shingles("abcdef")) == 1.0


def urow(
    uid: str,
    text: str,
    split: str | None,
    src: str = "bdshs",
    parent: str | None = None,
    **kw: object,
) -> UnifiedRow:
    labels = dict.fromkeys(LABELS)
    labels.update(kw.pop("labels", {}))  # type: ignore[arg-type]
    return UnifiedRow(
        uid=uid,
        source=src,
        source_split=split,
        source_id=uid,
        parent_uid=parent,
        text_orig=text,  # type: ignore[arg-type]
        script_orig="bangla",
        text_bn=text,
        labels=labels,
        toxic_any=kw.pop("toxic", 1),  # type: ignore[arg-type]
        licence_id=src,
        commercial_ok=False,
        research_only=True,
    )


def test_harmonise_votes_splits_and_leakage() -> None:
    dup = "এই বাক্যটি দুটি উৎসে প্রায় একই রকম ভাবে আছে এবং মিলবে"
    rows = [
        urow("bdshs:1", dup, "test", labels={"insult": 1}),
        urow("hsban:9", dup + "।", "train", src="hsban", labels={"insult": 0}),
        urow("romim2020:3", dup + "!", None, src="romim2020", labels={"insult": 0}),
        urow("bidwesh:1", "সম্পূর্ণ অন্য উপভাষার বাক্য", None, src="bidwesh", parent="bdshs:1"),
        urow("bidwesh:2", "অভিভাবকহীন উপভাষার বাক্য", None, src="bidwesh", parent="bdshs:404"),
        *[
            urow(f"violens:{i}", " ".join(random.Random(i).sample(WORDS, 8)), None, src="violens")
            for i in range(20)
        ],
    ]
    out, stats = harmonise(rows)
    by = {r.uid: r for r in out}
    assert stats["duplicates_removed"] == 2
    kept = by["bdshs:1"]
    assert kept.split == "test"  # any official test member keeps the group in test
    assert (
        kept.labels["insult"] == 0
    )  # weighted vote: bdshs 1.0 for 1 vs hsban 0.8 + romim 0.8 for 0
    assert by["bidwesh:1"].split == "test"  # inherits the parent's split
    assert stats["orphan_children"] == 1
    assert {r.split for r in out if r.source == "violens"} <= {"train", "dev"}
    assert check_no_leakage(out) == []


def test_asr_noise_and_rewrite() -> None:
    noise = AsrNoise(char_rate=0.25, word_drop=0.0, word_merge=0.0)
    src = "শাসন ষাঁড় যাত্রা কারণ শীতল পূর্ণ"
    a, b = noise(src, random.Random(1)), noise(src, random.Random(1))
    assert a == b
    assert a != src
    fitted = AsrNoise.fit([("শাল শাক", "সাল সাক")] * 3, min_count=3)
    assert fitted.conf["শ"][0][0] == "স"

    def handler(req: httpx.Request) -> httpx.Response:
        if b"fail" in req.content:
            return httpx.Response(500)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "আপনি এটা করলেন কেন?"}}]}
        )

    out = rewrite_spoken(
        ["কমেন্ট", "fail"], "http://vllm.local/v1", "m", transport=httpx.MockTransport(handler)
    )
    assert out == ["আপনি এটা করলেন কেন?", None]


def test_build_end_to_end(raw: Path, tmp_path: Path) -> None:
    res = build(REG, raw, "research", asr_noise=0.5, seed=3)
    st = res.stats
    assert "trac2_bn" in st["missing"]
    assert st["dropped_language"] == 2  # the English and the Hindi BanTH rows
    assert st["transliterated"] == 1
    assert st["leakage_pairs"] == 0
    rows = {r.uid: r for r in res.rows}
    assert rows["banth:banth_train:0"].text_bn.startswith("তুমি একটা <GALI_2>")
    chaos = rows["codemixed_chaos:bangla:0"]
    assert chaos.text_bn.startswith("তুমি একটা <GALI_15>")
    assert (chaos.labels["sarcasm_mockery"], chaos.labels["insult"], chaos.toxic_any) == (1, 0, 1)
    assert rows["codemixed_chaos:bangla:1"].toxic_any == 0  # Non-Toxic = 1
    troll = rows["banglaharassnet:trolling:0"]  # title row skipped
    assert (troll.labels["sarcasm_mockery"], troll.severity) == (1, 1)
    cat = [r for r in res.rows if r.source == "banglacat" and not r.synthetic]
    assert len(cat) == 1  # the _ENGLISH twin is not loaded
    assert cat[0].labels["sarcasm_mockery"] == 1
    assert cat[0].context == ["আগের মন্তব্য"]
    assert rows["mixtox_sa:mixtox:0"].split == "test"  # eval only
    assert rows["bidwesh:noakhali:0"].split == "test"  # parent is in BD-SHS test
    assert rows["karim_bn_hate:train:1"].weight == 0.3  # Political → low weight
    assert not any(
        r.uid.startswith("cyberbully_multilabel:cyberbully_multilabel:1") for r in res.rows
    )  # spam dropped
    assert all(r.research_only for r in res.rows)
    assert any(r.synthetic and r.aug == "asr_noise" for r in res.rows)
    assert {a.label for a in res.audio} == {"term_a", "term_b"}
    assert res.translit_pairs == [
        {"split": "train", "roman": "ami valo achi", "bangla": "আমি ভালো আছি"}
    ]
    assert len([r for r in res.rows if r.source == "banglamultihate" and r.split == "test"]) == 4

    paths = write_outputs(res, REG, tmp_path / "out", "research")
    card = paths["card"].read_text(encoding="utf-8")
    assert "research_only" in card
    assert "<GALI" not in card  # counts only, never text
    sft = [json.loads(line) for line in paths["llm_sft"].read_text(encoding="utf-8").splitlines()]
    assert sft
    for ex in sft:
        user, target = (
            ex["messages"][1]["content"],
            WindowJudgement.model_validate_json(ex["messages"][2]["content"]),
        )
        for e in target.events:
            assert e.evidence in user  # grounded: evidence is verbatim in the prompt (docs/08 §2)
        assert ex["meta"]["source"] == "open_synthetic"
    assert any(
        WindowJudgement.model_validate_json(ex["messages"][2]["content"]).events == [] for ex in sft
    )

    with pytest.raises(LicenceError):
        check_allowed(REG.get("T1"), "prod")
    prod = build(REG, raw, "prod")
    assert prod.rows == []
    assert "banglamultihate" in prod.stats["refused_licence"]


def test_missing_data_message(tmp_path: Path) -> None:
    with pytest.raises(MissingDataError, match="place the files"):
        list(TEXT_LOADERS["trac2_bn"](REG.get("trac2_bn"), tmp_path))


def test_download_with_checksum(tmp_path: Path) -> None:
    ds = REG.get("violens")
    transport = httpx.MockTransport(lambda req: httpx.Response(200, content=b"text,label\nx,0\n"))
    got = download(ds, tmp_path, transport=transport)
    assert set(got) == {"train", "dev", "test"}
    assert (
        json.loads((tmp_path / "violens" / "SOURCE.json").read_text(encoding="utf-8"))["sha256"]
        == got
    )
    bad = ds.model_copy(update={"sha256": {"train": "0" * 64}})
    with pytest.raises(ChecksumError):
        download(bad, tmp_path / "x", transport=transport)
    with pytest.raises(PermissionError):
        download(REG.get("trac2_bn"), tmp_path)


def _zip(members: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, text in members.items():
            z.writestr(name, text)
    return buf.getvalue()


def test_download_unpacks_archives(tmp_path: Path) -> None:
    ds = REG.get("banglacat").model_copy(update={"sha256": {}})
    member = "BanglaCAT_Mendeley 2/train/bangla_context_toxicity_train.csv"
    body = _zip({member: "row_id,target_reply,level_1_class,toxicity_type\na,x,Toxic,Insult\n"})
    ok = httpx.MockTransport(lambda r: httpx.Response(200, content=body))
    download(ds, tmp_path, transport=ok)
    assert (tmp_path / "banglacat" / "banglacat.zip").is_file()
    assert (tmp_path / "banglacat" / member).is_file()
    rows = list(TEXT_LOADERS["banglacat"](ds, tmp_path))
    assert [(r.split, r.fields["types"]) for r in rows] == [("train", ["Insult"])]
    evil = _zip({"../escape.csv": "x"})
    bad = httpx.MockTransport(lambda r: httpx.Response(200, content=evil))
    with pytest.raises(ValueError, match="escapes"):
        download(ds, tmp_path / "e", transport=bad)
    assert not (tmp_path / "e" / "escape.csv").exists()


def test_lexicon_mining(tmp_path: Path) -> None:
    rows = [
        urow(
            f"t:{i}",
            f"তুই একটা <GALI_1> গালিশব্দ {NEUTRAL[i % 6]}",
            None,
            src="banglamultihate",
            labels={"slang_profanity": 1},
        )
        for i in range(12)
    ]
    rows += [
        urow(
            f"n:{i}",
            f"তুই একটা {NEUTRAL[i % 6]}",
            None,
            src="banglamultihate",
            toxic=0,
            labels={"slang_profanity": 0},
        )
        for i in range(30)
    ]
    rows += [
        urow(
            f"v:{i}",
            "তুই একটা গালিশব্দো মন্তব্য",
            None,
            src="banglamultihate",
            labels={"slang_profanity": 1},
        )
        for i in range(6)
    ]
    cands = mine(rows, top=10, min_count=3, z_min=1.0)
    assert cands[0].term in ("গালিশব্দ", "গালিশব্দো")
    assert {cands[0].term, *cands[0].variants} >= {"গালিশব্দ", "গালিশব্দো"}  # variants grouped
    assert all("<" not in c.term for c in cands)  # placeholders never become candidates
    write_csv(cands, tmp_path / "c.csv")
    assert (tmp_path / "c.csv").read_text(encoding="utf-8").startswith("candidate_id,term,variants")


def _mendeley(files: dict[str, dict[str, bytes]], bad_hash: bool = False) -> httpx.MockTransport:
    """A fake Mendeley public API: folder name → {file name: content}; "" is the root."""
    import hashlib

    folders = [{"id": f"f{i}", "name": n, "parent_id": "top"} for i, n in enumerate(files) if n]
    ids = {"": "root", **{f["name"]: f["id"] for f in folders}}

    def handler(req: httpx.Request) -> httpx.Response:
        path = req.url.path
        if path.endswith("/folders/3"):
            return httpx.Response(200, json=folders)
        if path.endswith("/files"):
            name = next(n for n, i in ids.items() if i == req.url.params["folder_id"])
            return httpx.Response(
                200,
                json=[
                    {
                        "filename": fn,
                        "content_details": {
                            "sha256_hash": "0" * 64
                            if bad_hash
                            else hashlib.sha256(body).hexdigest(),
                            "download_url": f"https://files.test/{ids[name]}/{fn}",
                        },
                    }
                    for fn, body in files[name].items()
                ],
            )
        folder, fn = path.strip("/").split("/")
        name = next(n for n, i in ids.items() if i == folder)
        return httpx.Response(200, content=files[name][fn])

    return httpx.MockTransport(handler)


def test_mendeley_tree_mirror_resumes_and_verifies(tmp_path: Path) -> None:
    pinned = REG.get("baad")
    assert pinned.sha256["manifest"]  # the real release is pinned
    ds = pinned.model_copy(update={"sha256": {}})  # this fake tree is not that release
    tree = {"": {"readme.txt": b"r"}, "1. term_a": {"a1.wav": b"aa"}, "2. term_b": {"b1.wav": b"b"}}
    got = download(ds, tmp_path, transport=_mendeley(tree))
    assert (got["files"], got["fetched"]) == ("3", "3")
    assert (tmp_path / "baad" / "1. term_a" / "a1.wav").read_bytes() == b"aa"
    again = download(ds, tmp_path, transport=_mendeley(tree))
    assert (again["skipped"], again["manifest"]) == ("3", got["manifest"])  # resumable, stable
    src = json.loads((tmp_path / "baad" / "SOURCE.json").read_text(encoding="utf-8"))
    assert "term_a" not in json.dumps(src)  # names never recorded
    with pytest.raises(ChecksumError):
        download(ds, tmp_path / "x", transport=_mendeley(tree, bad_hash=True))
    with pytest.raises(ChecksumError, match="manifest"):  # upstream release changed
        download(pinned, tmp_path / "z", transport=_mendeley(tree))
    with pytest.raises(ValueError, match="unsafe"):
        download(ds, tmp_path / "y", transport=_mendeley({"": {}, "..": {"x.wav": b"x"}}))

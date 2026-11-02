"""Data engine end to end on the synthetic fixture calls (Prompt 4: "everything must run on the
synthetic fixtures in CI"). Label Studio is a mock; annotations are simulated from the fixtures'
ground truth by two "annotators" who disagree a little."""

import csv
import json
import shutil
import uuid
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
import soundfile as sf

from gale_text_worker.lexicon import LexiconMatcher
from galecore.config import Settings, load_lexicon
from galecore.models import FixtureOracle
from ml.data import augment_audio as aug
from ml.data.__main__ import build_tasks, main
from ml.data.agreement import agreement, krippendorff_alpha_nominal
from ml.data.ingest import FolderSource, RecorderColumns, RecorderCsvSource, S3Source
from ml.data.labelstudio import convert as cv
from ml.data.labelstudio.client import LabelStudioClient
from ml.data.labelstudio.templates import TEMPLATES
from ml.data.manifest import (
    ChannelRow,
    PseudonymSaltError,
    call_uuid,
    load_salt,
    pseudonymise,
    read_manifest,
    write_manifest,
)
from ml.data.process import process, stereo_review_path
from ml.data.splits import assign_splits, check_golden, write_lock
from ml.data.transcribe import load_transcript, transcribe_rows
from tools.fixtures_gen.__main__ import CALLS_DIR, ensure_wav

SALT = b"test-salt-not-a-secret-0123456789"
CALLS = ["demo_01", "demo_03", "demo_04"]
SETTINGS = Settings(env="test", bus_backend="memory")


@pytest.fixture(scope="module")
def recordings(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Fixture calls copied into a 'recorder folder' with CDR sidecars (synthetic numbers)."""
    root = tmp_path_factory.mktemp("rec")
    for i, c in enumerate(CALLS):
        d = root / c
        d.mkdir()
        shutil.copy(ensure_wav(CALLS_DIR / c), d / "call.wav")
        shutil.copy(CALLS_DIR / c / "script.json", d / "script.json")
        shutil.copy(CALLS_DIR / c / "expected_events.json", d / "expected_events.json")
        (d / "call.json").write_text(
            json.dumps(
                {
                    "customer_number": f"+880 171 000 00{i:02d}",
                    "agent_id": "agent_A" if i < 2 else "agent_B",
                    "queue": "billing" if i % 2 else "network",
                    "date": "2026-10-1" + str(i),
                    "dialect": "dhaka",
                }
            ),
            encoding="utf-8",
        )
    return root


@pytest.fixture(scope="module")
def ingested(
    recordings: Path, tmp_path_factory: pytest.TempPathFactory
) -> tuple[list[ChannelRow], Path]:
    out = tmp_path_factory.mktemp("audio")
    src = FolderSource(recordings, SALT)
    rows = [r for rec in src.recordings() for r in process(rec, out, synthetic=True)]
    return rows, out


def test_pseudonymisation_and_salt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    a = pseudonymise("+880 1712-345678", SALT)
    assert a == pseudonymise("01712345678", SALT)  # same number, different formatting
    assert a != pseudonymise("01712345678", b"another-salt-0123456789")
    assert "1712345678" not in (a or "")
    assert pseudonymise(None, SALT) is None
    monkeypatch.delenv("GALE_PSEUDONYM_SALT", raising=False)
    monkeypatch.delenv("GALE_PSEUDONYM_SALT_FILE", raising=False)
    with pytest.raises(PseudonymSaltError):
        load_salt()
    (tmp_path / "salt").write_text("s" * 32, encoding="utf-8")
    monkeypatch.setenv("GALE_PSEUDONYM_SALT_FILE", str(tmp_path / "salt"))
    assert load_salt() == b"s" * 32
    assert str(uuid.UUID(call_uuid("folder:/x", "a.wav"))) == call_uuid("folder:/x", "a.wav")


def test_ingest_and_process(ingested: tuple[list[ChannelRow], Path]) -> None:
    rows, out = ingested
    assert len(rows) == 6
    assert all(r.dropped is None for r in rows)
    r = rows[0]
    assert r.sample_rate == 8000
    assert r.codec == "pcm_16"
    assert r.queue == "network"
    assert r.customer_hash is not None
    assert r.customer_hash.startswith("c_")
    assert "171" not in r.model_dump_json()  # no customer number anywhere in the manifest
    assert sf.info(r.audio_16k_path).samplerate == 16000  # type: ignore[arg-type]
    assert sf.info(stereo_review_path(out, r.call_id)).channels == 2
    assert 0 <= r.quality["silence_ratio"] < 0.8


def test_quality_filters(tmp_path: Path) -> None:
    from ml.data.ingest import Recording

    sr = 8000
    t = np.arange(sr * 30) / sr
    speech = (0.3 * np.sin(2 * np.pi * 200 * t) * (np.sin(2 * np.pi * 0.5 * t) > 0)).astype(
        np.float32
    )
    cases = {
        "short.wav": np.stack([speech[: sr * 5]] * 2, axis=1),
        "silent.wav": np.stack([speech, np.zeros_like(speech)], axis=1),
        "clipped.wav": np.stack([np.clip(speech * 20, -1, 1), speech], axis=1),
    }
    got = {}
    for name, x in cases.items():
        sf.write(tmp_path / name, x, sr, subtype="PCM_16")
        rows = process(Recording("c-" + name, "t", name, tmp_path / name), tmp_path / "out")
        got[name] = [r.dropped for r in rows]
    assert got == {
        "short.wav": ["too_short", "too_short"],
        "silent.wav": [None, "silence"],
        "clipped.wav": ["clipping", None],
    }


def test_recorder_csv_and_s3_sources(recordings: Path, tmp_path: Path) -> None:
    with (tmp_path / "export.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "recording_file",
                "recording_id",
                "caller_number",
                "agent_id",
                "queue",
                "start_time",
                "direction",
            ]
        )
        w.writerow(
            [
                "demo_01/call.wav",
                "R-1",
                "01700000001",
                "ag1",
                "billing",
                "2026-10-30T10:00:00",
                "Inbound",
            ]
        )
        w.writerow(
            [
                "missing.wav",
                "R-2",
                "01700000002",
                "ag1",
                "billing",
                "2026-10-30T11:00:00",
                "Inbound",
            ]
        )
    src = RecorderCsvSource(tmp_path / "export.csv", recordings, SALT, RecorderColumns())
    recs = list(src.recordings())
    assert [r.meta["date"] for r in recs] == ["2026-10-30"]
    assert recs[0].meta["direction"] == "inbound"
    assert src.missing == ["R-2"]

    class FakeS3:
        def __init__(self) -> None:
            self.pages = [
                {
                    "Contents": [{"Key": "p/a.wav"}],
                    "IsTruncated": True,
                    "NextContinuationToken": "t",
                },
                {"Contents": [{"Key": "p/a.json"}, {"Key": "p/readme.txt"}]},
            ]

        def list_objects_v2(self, **kw: Any) -> dict[str, Any]:
            return self.pages[1 if kw.get("ContinuationToken") else 0]

        def download_file(self, bucket: str, key: str, dest: str) -> None:
            if key.endswith(".json"):
                Path(dest).write_text(
                    json.dumps({"agent_id": "s3agent", "customer_number": "01711111111"}),
                    encoding="utf-8",
                )
            else:
                shutil.copy(recordings / "demo_01" / "call.wav", dest)

    s3 = list(S3Source("bucket", "p/", tmp_path / "cache", SALT, client=FakeS3()).recordings())
    assert [(r.source_ref, r.meta["agent_id"]) for r in s3] == [("p/a.wav", "s3agent")]
    assert s3[0].meta["customer_hash"].startswith("c_")


def test_transcribe_is_resumable(
    ingested: tuple[list[ChannelRow], Path], recordings: Path, tmp_path: Path
) -> None:
    rows, _ = ingested

    def oracle_for(r: ChannelRow) -> FixtureOracle | None:
        return FixtureOracle.for_audio(recordings / r.source_ref)

    st = transcribe_rows(rows, tmp_path, SETTINGS, oracle_for)
    assert (st.done, st.skipped, st.failed) == (6, 0, 0)
    segs = load_transcript(tmp_path, rows[0])
    assert segs
    assert segs[0].text_redacted.startswith("আসসালামু")
    again = transcribe_rows(rows, tmp_path, SETTINGS, oracle_for)
    assert (again.done, again.skipped) == (0, 6)


def test_templates_and_client() -> None:
    import xml.etree.ElementTree as ET

    t2 = ET.fromstring(TEMPLATES["T2"]())
    labels = {el.attrib["value"] for el in t2.iter("Label")}
    assert "interruption_overtalk" in labels
    assert len(labels) == 11
    calls: list[tuple[str, str]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append((req.method, req.url.path))
        assert req.headers["Authorization"] == "Token tok"
        if req.url.path == "/api/projects" and req.method == "GET":
            return httpx.Response(200, json={"results": [{"id": 3, "title": "GAL-E T2 · b1"}]})
        if req.url.path == "/api/projects":
            return httpx.Response(201, json={"id": 9})
        if req.url.path.endswith("/import"):
            return httpx.Response(201, json={"task_count": 1})
        return httpx.Response(200, json=[{"id": 1}])

    c = LabelStudioClient("http://ls.local", "tok", transport=httpx.MockTransport(handler))
    assert c.find_project("GAL-E T2 · b1") == 3
    assert c.find_project("other") is None
    assert c.create_project("x", "<View/>") == 9
    assert c.import_tasks(9, [{"data": {}}] * 3, batch=2) == 3
    assert c.export(9) == [{"id": 1}]
    assert ("POST", "/api/projects/9/import") in calls


def _annotation(spans: list[dict[str, Any]], who: int, shift: float = 0.0) -> dict[str, Any]:
    result = []
    for i, e in enumerate(spans):
        val = {
            "start": e["t_start"] + shift,
            "end": e["t_end"] + shift,
            "channel": 0 if e["speaker"] == "agent" else 1,
        }
        result += [
            {
                "id": f"r{i}",
                "from_name": "category",
                "to_name": "audio",
                "type": "labels",
                "value": {**val, "labels": [e["category"]]},
            },
            {
                "id": f"r{i}",
                "from_name": "severity",
                "to_name": "audio",
                "type": "choices",
                "value": {**val, "choices": [str(e["severity"])]},
            },
            {
                "id": f"r{i}",
                "from_name": "note",
                "to_name": "audio",
                "type": "textarea",
                "value": {**val, "text": ["call me on 01712345678"]},
            },
        ]
    return {"completed_by": who, "result": result}


def test_tasks_export_convert_agreement_splits(
    ingested: tuple[list[ChannelRow], Path], recordings: Path, tmp_path: Path
) -> None:
    rows, audio_out = ingested
    transcribe_rows(
        rows,
        tmp_path / "tr",
        SETTINGS,
        lambda r: FixtureOracle.for_audio(recordings / r.source_ref),
    )
    t1 = build_tasks("T1", rows, tmp_path / "tr", audio_out, SETTINGS, "/data/local-files/?d=")
    assert len(t1) == 6
    assert t1[0]["predictions"][0]["result"][1]["from_name"] == "transcription"
    t2 = build_tasks("T2", rows, tmp_path / "tr", audio_out, SETTINGS, "/data/local-files/?d=")
    assert len(t2) == 3
    by_call = {t["data"]["call_id"]: t for t in t2}
    first = by_call[rows[0].call_id]["predictions"][0]["result"]
    assert {r["value"]["labels"][0] for r in first if r["from_name"] == "category"} == {
        "slang_profanity",
        "shouting",
    }
    assert len(build_tasks("T3", rows, tmp_path / "tr", audio_out, SETTINGS, "x")) == 3

    # simulated Label Studio export: annotator 1 = ground truth; annotator 2 shifted 0.2 s and misses one span
    gold = {
        r.call_id: json.loads(
            (recordings / r.source_ref)
            .with_name("expected_events.json")
            .read_text(encoding="utf-8")
        )["events"]
        for r in rows
    }
    export2 = []
    for cid, events in gold.items():
        export2.append(
            {
                "data": {"call_id": cid},
                "annotations": [
                    _annotation(events, 1),
                    _annotation(events[:-1], 2, 0.2),
                    {"completed_by": 3, "was_cancelled": True, "result": []},
                ],
            }
        )
    spans = cv.parse_t2(export2)
    assert all(s.note is None or "01712345678" not in s.note for s in spans)  # notes are redacted
    export1 = [
        {
            "data": t["data"],
            "annotations": [
                {
                    "completed_by": 1,
                    "result": t["predictions"][0]["result"]
                    + [
                        {
                            "from_name": "contains_slang",
                            "to_name": "audio",
                            "type": "choices",
                            "value": {"choices": ["contains slang"]},
                        }
                    ],
                }
            ],
        }
        for t in t1
        if t["predictions"]
    ]
    segs = cv.parse_t1(export1)
    assert segs
    assert any("<GALI_" in s.text for s in segs)
    t3 = cv.parse_t3(
        [
            {
                "data": {"call_id": "c"},
                "annotations": [
                    {
                        "completed_by": 1,
                        "result": [
                            {"from_name": "professionalism", "value": {"rating": 4}},
                            {"from_name": "escalated_by_agent", "value": {"choices": ["no"]}},
                        ],
                    }
                ],
            }
        ]
    )
    assert (t3[0].professionalism, t3[0].escalated_by_agent) == (4, False)

    split = assign_splits(rows, fixed={rows[4].call_id: "golden"})
    rows2 = [r.model_copy(update={"split": split[r.call_id]}) for r in rows]
    # calls 0 and 1 share agent_A, so they always land in the same split
    assert split[rows[0].call_id] == split[rows[2].call_id]
    train_rows = [r for r in rows2 if r.split != "golden"]
    sets = {
        "asr": cv.asr_set(segs, train_rows),
        "kws": cv.kws_set(segs, train_rows, LexiconMatcher(load_lexicon(SETTINGS).entries)),
        "shout": cv.shout_set(spans, train_rows),
        "ser": cv.ser_set(spans, train_rows),
        "audio_tox": cv.audio_tox_set(spans, train_rows),
        "textclf": cv.textclf_set(segs, spans, train_rows),
        "llm_sft": cv.llm_sft_set(segs, spans, train_rows),
        "fusion": cv.fusion_set(spans, train_rows),
    }
    assert all(sets.values()), {k: len(v) for k, v in sets.items()}
    assert {r["shouting"] for r in sets["shout"]} == {0, 1}
    assert any(r["labels"]["slang_profanity"] == 1 for r in sets["textclf"])
    with pytest.raises(cv.GoldenLeakError):
        cv.asr_set(segs, rows2)

    anns = {cid: {"1", "2"} for cid in gold}
    rep = agreement(
        spans, anns, {r.call_id: r.duration for r in rows}, ["slang_profanity", "shouting"]
    )
    assert rep.double_annotated_calls == 3
    assert rep.span_mean_iou is not None and rep.span_mean_iou > 0.5  # noqa: PT018
    assert any(d["issue"] == "missing" for d in rep.adjudication)
    assert rep.alpha["slang_profanity"] is not None


def test_krippendorff_alpha() -> None:
    assert krippendorff_alpha_nominal([[1, 1], [0, 0], [1, 0]]) == pytest.approx(0.4444, abs=1e-4)
    assert krippendorff_alpha_nominal([[1, 1], [0, 0]]) == 1.0
    assert krippendorff_alpha_nominal([[1], [0]]) is None


def test_golden_lock_and_overlap(ingested: tuple[list[ChannelRow], Path], tmp_path: Path) -> None:
    rows, _ = ingested
    split = assign_splits(rows, fixed={rows[4].call_id: "golden"})
    rows2 = [r.model_copy(update={"split": split[r.call_id]}) for r in rows]
    lock = write_lock(rows2, tmp_path / "golden.lock")
    assert lock.calls == 1
    assert check_golden(rows2, tmp_path / "golden.lock") == []
    leak = [*rows2, rows2[4].model_copy(update={"split": "train"})]
    assert any("golden call" in p for p in check_golden(leak))
    moved = [
        r.model_copy(update={"split": "golden"}) if r.call_id == rows[0].call_id else r
        for r in rows2
    ]
    assert any("golden.lock" in p for p in check_golden(moved, tmp_path / "golden.lock"))
    with pytest.raises(ValueError, match="share a speaker"):
        assign_splits(rows, fixed={rows[0].call_id: "golden", rows[2].call_id: "train"})


def test_augmentation_is_seeded() -> None:
    sr = 16000
    x = (0.2 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr)).astype(np.float32)
    a = aug.telephony_chain(x, sr, np.random.default_rng(7))
    b = aug.telephony_chain(x, sr, np.random.default_rng(7))
    assert np.array_equal(a, b)
    lost = aug.packet_loss(np.ones(sr, np.float32), sr, np.random.default_rng(1), rate=0.5)
    assert 0.3 < float(np.mean(lost == 0)) < 0.7
    low = (0.2 * np.sin(2 * np.pi * 100 * np.arange(sr) / sr)).astype(np.float32)
    assert float(np.abs(aug.band_limit(low, sr)).max()) < 0.01
    noisy = aug.add_noise(x, np.random.default_rng(0), snr_db=10)
    snr = 10 * np.log10(np.mean(x**2) / np.mean((noisy - x) ** 2))
    assert snr == pytest.approx(10, abs=0.5)
    assert len(aug.speed(x, sr, 1.1)) < len(x)
    assert aug.speed(x, sr, 1.0) is x
    with pytest.raises(ValueError, match="shouting"):
        aug.gain(x, 6, shout=True)
    assert aug.codec_available("ulaw")
    if not aug.codec_available("amr_nb"):
        with pytest.raises(RuntimeError, match="ffmpeg"):
            aug.codec(x, sr, "amr_nb")


def test_cli_end_to_end(
    recordings: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("GALE_PSEUDONYM_SALT", SALT.decode())
    m = tmp_path / "manifests" / "calls.jsonl"
    assert (
        main(
            [
                "ingest",
                "--folder",
                str(recordings),
                "--audio-out",
                str(tmp_path / "audio"),
                "--manifest",
                str(m),
                "--synthetic",
            ]
        )
        == 0
    )
    (tmp_path / "golden.txt").write_text(read_manifest(m)[0].call_id + "\n", encoding="utf-8")
    lock = tmp_path / "manifests" / "golden.lock"
    assert (
        main(
            [
                "split",
                "--manifest",
                str(m),
                "--golden",
                str(tmp_path / "golden.txt"),
                "--lock",
                "--lock-path",
                str(lock),
            ]
        )
        == 0
    )
    assert main(["golden-check", "--manifest-dir", str(m.parent), "--lock-path", str(lock)]) == 0
    assert main(["transcribe", "--manifest", str(m), "--out", str(tmp_path / "tr")]) == 0
    assert main(["ls-templates", "--out", str(tmp_path / "ls")]) == 0
    assert sorted(p.name for p in (tmp_path / "ls").iterdir()) == ["T1.xml", "T2.xml", "T3.xml"]
    monkeypatch.delenv("GALE_LS_URL", raising=False)
    with pytest.raises(SystemExit, match="GALE_LS_URL"):
        main(["ls-pull", "--project-id", "1", "--out", str(tmp_path / "x.json")])
    rows = read_manifest(m)
    events = [
        {"t_start": 10.0, "t_end": 12.0, "speaker": "customer", "category": "insult", "severity": 2}
    ]
    export = [
        {
            "data": {"call_id": r.call_id},
            "annotations": [_annotation(events, 1), _annotation(events, 2)],
        }
        for r in rows
        if r.channel == "agent"
    ]
    (tmp_path / "t2.json").write_text(json.dumps(export), encoding="utf-8")
    assert (
        main(
            [
                "convert",
                "--manifest",
                str(m),
                "--t2",
                str(tmp_path / "t2.json"),
                "--out",
                str(tmp_path / "train"),
            ]
        )
        == 0
    )
    fusion = [
        json.loads(line)
        for line in (tmp_path / "train" / "fusion.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0].call_id not in {
        f["call_id"] for f in fusion
    }  # golden never reaches a training set
    assert (
        main(
            [
                "agreement",
                "--manifest",
                str(m),
                "--t2",
                str(tmp_path / "t2.json"),
                "--out",
                str(tmp_path / "rep"),
            ]
        )
        == 0
    )
    assert (
        json.loads((tmp_path / "rep" / "agreement.json").read_text(encoding="utf-8"))["span_f1"]
        == 1.0
    )
    out = capsys.readouterr().out
    assert "0171" not in out  # nothing personal printed
    write_manifest(rows, m)  # round-trip

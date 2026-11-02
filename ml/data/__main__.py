"""Data engine CLI (docs/02 §2–3, Prompt 4). Paths default to ``$GALE_DATA_DIR/data``.

    python -m ml.data ingest --folder /mnt/recordings                 # or --s3 bucket/prefix, --recorder-csv
    python -m ml.data split --manifest calls.jsonl [--golden ids.txt] --lock
    python -m ml.data golden-check                                    # all manifests; non-zero on overlap
    python -m ml.data transcribe --manifest calls.jsonl              # resumable seed transcripts
    python -m ml.data ls-templates                                    # write T1–T3 XML
    python -m ml.data ls-push --task T2 --manifest calls.jsonl --batch 2026-11-a
    python -m ml.data ls-pull --project-id 7 --out t2.json
    python -m ml.data convert --manifest calls.jsonl --t1 t1.json --t2 t2.json   # training sets
    python -m ml.data agreement --manifest calls.jsonl --t2 t2.json            # α, IoU, adjudication

Needs ``GALE_PSEUDONYM_SALT`` (ingest), ``GALE_LS_URL`` + ``GALE_LS_TOKEN`` (Label Studio), and for S3
the usual ``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY`` (MinIO via ``--endpoint``).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

from galecore.config import REPO_ROOT, Settings, get_settings, load_lexicon
from ml.data.manifest import ChannelRow, load_salt, read_manifest, write_manifest

DATA = Path(os.environ.get("GALE_DATA_DIR", REPO_ROOT / ".gale")) / "data"


def _dump(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )


def cmd_ingest(a: argparse.Namespace) -> int:
    from ml.data.ingest import FolderSource, RecorderCsvSource, S3Source, Source
    from ml.data.process import process

    salt = load_salt()
    src: Source
    if a.folder:
        src = FolderSource(a.folder, salt)
    elif a.s3:
        bucket, _, prefix = a.s3.partition("/")
        src = S3Source(bucket, prefix, DATA / "cache", salt, endpoint_url=a.endpoint)
    else:
        src = RecorderCsvSource(a.recorder_csv, a.audio_dir, salt)
    rows: list[ChannelRow] = []
    for rec in src.recordings():
        rows += process(rec, a.audio_out, synthetic=a.synthetic)
    n = write_manifest(rows, a.manifest)
    dropped = sum(r.dropped is not None for r in rows)
    print(
        f"{src.name}: {len({r.call_id for r in rows})} calls, {n} channels ({dropped} dropped by quality) → {a.manifest}"
    )
    if isinstance(src, RecorderCsvSource) and src.missing:
        print(f"  {len(src.missing)} CSV rows had no audio file")
    return 0


def cmd_split(a: argparse.Namespace) -> int:
    from ml.data.splits import assign_splits, check_golden, write_lock

    rows = read_manifest(a.manifest)
    fixed = {}
    if a.golden:
        fixed = {
            line.strip(): "golden"
            for line in a.golden.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    fixed |= {
        r.call_id: r.split for r in rows if r.split == "golden"
    }  # an existing golden set never moves
    split = assign_splits(rows, fixed=fixed)
    rows = [r.model_copy(update={"split": split[r.call_id]}) for r in rows]
    write_manifest(rows, a.manifest)
    if a.lock:
        lock = write_lock(rows, a.lock_path)
        print(f"golden.lock: {lock.calls} calls, {lock.speakers} speakers")
    problems = check_golden(rows, a.lock_path)
    counts: dict[str, int] = {}
    for r in rows:
        counts[str(r.split)] = counts.get(str(r.split), 0) + 1
    print(f"splits (channels): {counts}")
    for p in problems:
        print(f"PROBLEM: {p}")
    return 1 if problems else 0


def cmd_golden_check(a: argparse.Namespace) -> int:
    from ml.data.splits import check_golden

    rows = [r for p in sorted(a.manifest_dir.glob("*.jsonl")) for r in read_manifest(p)]
    problems = check_golden(rows, a.lock_path)
    for p in problems:
        print(f"PROBLEM: {p}")
    print(f"{len(rows)} manifest rows checked: {'OK' if not problems else 'FAILED'}")
    return 1 if problems else 0


def cmd_transcribe(a: argparse.Namespace) -> int:
    from galecore.models import FixtureOracle
    from ml.data.transcribe import transcribe_rows

    rows = read_manifest(a.manifest)

    def oracle_for(r: ChannelRow) -> FixtureOracle | None:  # synthetic fixtures only
        return (
            FixtureOracle.for_audio(Path(r.source.split(":", 1)[1]) / r.source_ref)
            if r.synthetic
            else None
        )

    st = transcribe_rows(rows, a.out, get_settings(), oracle_for, a.limit)
    print(
        f"transcribed {st.done} channels ({st.segments} segments), skipped {st.skipped} done, {st.failed} failed"
    )
    return 1 if st.failed else 0


def cmd_ls_templates(a: argparse.Namespace) -> int:
    from ml.data.labelstudio.templates import TEMPLATES

    a.out.mkdir(parents=True, exist_ok=True)
    for name, fn in TEMPLATES.items():
        (a.out / f"{name}.xml").write_text(fn(), encoding="utf-8")
        print(f"wrote {a.out / f'{name}.xml'}")
    return 0


def _ls_client() -> Any:
    from ml.data.labelstudio.client import LabelStudioClient

    url, token = os.environ.get("GALE_LS_URL"), os.environ.get("GALE_LS_TOKEN")
    if not url or not token:
        raise SystemExit("set GALE_LS_URL (our self-hosted Label Studio) and GALE_LS_TOKEN")
    return LabelStudioClient(url, token)


def build_tasks(
    task: str,
    rows: list[ChannelRow],
    transcripts: Path,
    audio_out: Path,
    settings: Settings,
    url_prefix: str,
) -> list[dict[str, Any]]:
    from gale_postcall.pipeline import run_postcall
    from galecore.models import FixtureOracle
    from ml.data.labelstudio.tasks import local_files_url, t1_task, t2_task, t3_task
    from ml.data.process import stereo_review_path
    from ml.data.transcribe import load_transcript

    url = local_files_url(url_prefix)
    out = []
    if task == "T1":
        # every kept channel, golden included: golden calls need human transcripts as the ASR test set
        return [
            t1_task(r, load_transcript(transcripts, r), url, "asr-seed")
            for r in rows
            if not r.dropped
        ]
    calls = sorted({r.call_id for r in rows if not r.dropped})
    for cid in calls:
        stereo = stereo_review_path(audio_out, cid)
        if not stereo.exists():
            continue
        if task == "T3":
            out.append(t3_task(cid, str(stereo), url))
            continue
        src = next(r for r in rows if r.call_id == cid)
        s = settings
        if (
            src.synthetic and s.model_mode == "stub"
        ):  # fixture oracle keyed to the original fixture file
            from gale_postcall.models import build_models

            oracle = FixtureOracle.for_audio(Path(src.source.split(":", 1)[1]) / src.source_ref)
            res = run_postcall(stereo, s, call_id=cid, models=build_models(s, oracle))
        else:
            res = run_postcall(stereo, s, call_id=cid)
        out.append(
            t2_task(cid, str(stereo), res.events, url, f"gale-postcall:{res.models.get('mode')}")
        )
    return out


def cmd_ls_push(a: argparse.Namespace) -> int:
    from ml.data.labelstudio.templates import TEMPLATES, instruction_html, project_title

    rows = read_manifest(a.manifest)
    tasks = build_tasks(a.task, rows, a.transcripts, a.audio_out, get_settings(), a.url_prefix)
    client = _ls_client()
    title = project_title(a.task, a.batch)
    pid = client.find_project(title) or client.create_project(
        title, TEMPLATES[a.task](), instruction_html(a.task)
    )
    n = client.import_tasks(pid, tasks)
    print(
        f"project {pid} ({title}): imported {n} tasks ({sum(t['meta']['double'] for t in tasks)} for double annotation)"
    )
    return 0


def cmd_ls_pull(a: argparse.Namespace) -> int:
    export = _ls_client().export(a.project_id)
    a.out.write_text(json.dumps(export, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(export)} tasks → {a.out}")
    return 0


def cmd_convert(a: argparse.Namespace) -> int:
    from gale_text_worker.lexicon import LexiconMatcher
    from ml.data.labelstudio import convert as cv

    rows = [
        r for r in read_manifest(a.manifest) if r.split != "golden"
    ]  # rule 3: never train on golden
    t1 = cv.parse_t1(json.loads(a.t1.read_text(encoding="utf-8"))) if a.t1 else []
    t2 = cv.parse_t2(json.loads(a.t2.read_text(encoding="utf-8"))) if a.t2 else []
    keep = {r.call_id for r in rows}
    t1 = [s for s in t1 if s.call_id in keep]
    t2 = [s for s in t2 if s.call_id in keep]
    sets = {
        "asr": cv.asr_set(t1, rows),
        "kws": cv.kws_set(t1, rows, LexiconMatcher(load_lexicon().entries)),
        "shout": cv.shout_set(t2, rows),
        "ser": cv.ser_set(t2, rows),
        "audio_tox": cv.audio_tox_set(t2, rows),
        "textclf": cv.textclf_set(t1, t2, rows),
        "llm_sft": cv.llm_sft_set(t1, t2, rows),
        "fusion": cv.fusion_set(t2, rows),
    }
    for name, recs in sets.items():
        _dump(a.out / f"{name}.jsonl", recs)
        print(f"  {name:<10} {len(recs):>6} rows → {a.out / f'{name}.jsonl'}")
    return 0


def cmd_agreement(a: argparse.Namespace) -> int:
    from galecore.schemas import Category
    from ml.data.agreement import agreement
    from ml.data.labelstudio.convert import parse_t2

    rows = read_manifest(a.manifest)
    export = json.loads(a.t2.read_text(encoding="utf-8"))
    spans = parse_t2(export)
    anns = {
        t["data"]["call_id"]: {
            str(x.get("completed_by"))
            for x in t.get("annotations", [])
            if not x.get("was_cancelled")
        }
        for t in export
    }
    dur = {r.call_id: r.duration for r in rows}
    rep = agreement(spans, anns, dur, [c.value for c in Category])
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "agreement.json").write_text(
        json.dumps({k: v for k, v in rep.__dict__.items() if k != "adjudication"}, indent=2) + "\n",
        encoding="utf-8",
    )
    with (a.out / "adjudication.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["call_id", "channel", "category", "start", "end", "issue", "annotators", "decision"]
        )
        for d in rep.adjudication:
            w.writerow(
                [
                    d["call_id"],
                    d["channel"],
                    d["category"],
                    d["start"],
                    d["end"],
                    d["issue"],
                    json.dumps(d["annotators"]),
                    "",
                ]
            )
    failing = [c for c, ok in rep.gates.items() if ok is False]
    print(
        f"double-annotated calls: {rep.double_annotated_calls}; span IoU {rep.span_mean_iou}, F1 {rep.span_f1}; "
        f"{len(rep.adjudication)} items to adjudicate; below gate: {failing or 'none'}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m ml.data",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("ingest")
    g = i.add_mutually_exclusive_group(required=True)
    g.add_argument("--folder", type=Path)
    g.add_argument("--s3", help="bucket/prefix")
    g.add_argument("--recorder-csv", type=Path)
    i.add_argument("--audio-dir", type=Path, help="WAV folder for --recorder-csv")
    i.add_argument("--endpoint", help="MinIO / S3 endpoint URL")
    i.add_argument("--audio-out", type=Path, default=DATA / "audio")
    i.add_argument("--manifest", type=Path, default=DATA / "manifests" / "calls.jsonl")
    i.add_argument("--synthetic", action="store_true", help="mark rows as synthetic fixtures")
    s = sub.add_parser("split")
    s.add_argument("--manifest", type=Path, default=DATA / "manifests" / "calls.jsonl")
    s.add_argument(
        "--golden", type=Path, help="file with one call id per line to pin to the golden split"
    )
    s.add_argument("--lock", action="store_true")
    s.add_argument("--lock-path", type=Path, default=DATA / "manifests" / "golden.lock")
    gc = sub.add_parser("golden-check")
    gc.add_argument("--manifest-dir", type=Path, default=DATA / "manifests")
    gc.add_argument("--lock-path", type=Path, default=DATA / "manifests" / "golden.lock")
    t = sub.add_parser("transcribe")
    t.add_argument("--manifest", type=Path, default=DATA / "manifests" / "calls.jsonl")
    t.add_argument("--out", type=Path, default=DATA / "transcripts")
    t.add_argument("--limit", type=int)
    lt = sub.add_parser("ls-templates")
    lt.add_argument("--out", type=Path, default=DATA / "labelstudio")
    lp = sub.add_parser("ls-push")
    lp.add_argument("--task", choices=["T1", "T2", "T3"], required=True)
    lp.add_argument("--manifest", type=Path, default=DATA / "manifests" / "calls.jsonl")
    lp.add_argument("--transcripts", type=Path, default=DATA / "transcripts")
    lp.add_argument("--audio-out", type=Path, default=DATA / "audio")
    lp.add_argument("--batch", required=True)
    lp.add_argument("--url-prefix", default="/data/local-files/?d=")
    lpl = sub.add_parser("ls-pull")
    lpl.add_argument("--project-id", type=int, required=True)
    lpl.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("convert")
    c.add_argument("--manifest", type=Path, default=DATA / "manifests" / "calls.jsonl")
    c.add_argument("--t1", type=Path)
    c.add_argument("--t2", type=Path)
    c.add_argument("--out", type=Path, default=DATA / "training")
    ag = sub.add_parser("agreement")
    ag.add_argument("--manifest", type=Path, default=DATA / "manifests" / "calls.jsonl")
    ag.add_argument("--t2", type=Path, required=True)
    ag.add_argument("--out", type=Path, default=DATA / "reports")
    a = ap.parse_args(argv)
    if a.cmd == "ingest" and a.recorder_csv and not a.audio_dir:
        ap.error("--recorder-csv needs --audio-dir")
    handler = {
        "ingest": cmd_ingest,
        "split": cmd_split,
        "golden-check": cmd_golden_check,
        "transcribe": cmd_transcribe,
        "ls-templates": cmd_ls_templates,
        "ls-push": cmd_ls_push,
        "ls-pull": cmd_ls_pull,
        "convert": cmd_convert,
        "agreement": cmd_agreement,
    }[a.cmd]
    return handler(a)


if __name__ == "__main__":
    sys.exit(main())

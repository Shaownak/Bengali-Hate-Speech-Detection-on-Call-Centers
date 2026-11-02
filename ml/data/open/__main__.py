"""Open-data CLI (docs/09).

python -m ml.data.open register [--write-md]        # list datasets / regenerate DATASETS.md
python -m ml.data.open download [KEY ...]           # fetch download_allowed datasets
python -m ml.data.open status                       # which raw files are in place
python -m ml.data.open build --run-tag research     # unified.jsonl + views + data card
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from galecore.config import REPO_ROOT
from ml.data.open.build import build, write_outputs
from ml.data.open.download import download
from ml.data.open.loaders import _files
from ml.data.open.registry import DATASETS_MD, load_register, register_markdown
from ml.data.open.translit import load_transliterator

DATA = Path(os.environ.get("GALE_DATA_DIR", REPO_ROOT / ".gale")) / "data"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m ml.data.open",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--raw", type=Path, default=DATA / "raw")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("register")
    r.add_argument("--write-md", action="store_true")
    d = sub.add_parser("download")
    d.add_argument("keys", nargs="*")
    sub.add_parser("status")
    b = sub.add_parser("build")
    b.add_argument("--run-tag", choices=["prod", "research"], required=True)
    b.add_argument("--only", nargs="*")
    b.add_argument("--out", type=Path, default=DATA / "open")
    b.add_argument(
        "--asr-noise",
        type=float,
        default=0.15,
        help="fraction of train rows to add an ASR-noised copy of",
    )
    b.add_argument(
        "--model-dir",
        type=Path,
        default=Path(os.environ.get("GALE_MODEL_DIR", REPO_ROOT / ".gale" / "models")),
    )
    a = ap.parse_args(argv)
    reg = load_register()

    if a.cmd == "register":
        for ds in reg.datasets:
            print(
                f"{ds.id:<4} {ds.key:<24} commercial_ok={ds.commercial_ok:<8} "
                f"{'download' if ds.download_allowed else 'manual  '} {ds.kind:<9} {ds.format}"
            )
        if a.write_md:
            DATASETS_MD.write_text(register_markdown(reg), encoding="utf-8", newline="\n")
            print(f"wrote {DATASETS_MD.relative_to(REPO_ROOT)}")
        return 0
    if a.cmd == "download":
        targets = (
            [reg.get(k) for k in a.keys]
            if a.keys
            else [x for x in reg.datasets if x.download_allowed]
        )
        for ds in targets:
            got = download(ds, a.raw)
            print(
                f"{ds.id} {ds.key}: " + ", ".join(f"{s} sha256={h[:12]}…" for s, h in got.items())
            )
        return 0
    if a.cmd == "status":
        for ds in reg.datasets:
            try:
                n = len(_files(ds, a.raw))
                print(f"{ds.id:<4} {ds.key:<24} {n} file(s)")
            except FileNotFoundError:
                how = "download" if ds.download_allowed else f"manual: {ds.url}"
                print(f"{ds.id:<4} {ds.key:<24} MISSING ({how})")
        return 0
    res = build(
        reg,
        a.raw,
        a.run_tag,
        only=a.only,
        translit=load_transliterator(a.model_dir),
        asr_noise=a.asr_noise,
    )
    paths = write_outputs(res, reg, a.out, a.run_tag)
    st = res.stats
    print(
        f"run_tag={a.run_tag}: {st['loaded']} rows loaded → {st['output_rows']} unified "
        f"({st['duplicates_removed']} duplicates removed, "
        f"{st['dropped_language']} dropped by language ID, "
        f"{st.get('asr_noise_rows', 0)} ASR-noised copies); leakage pairs: {st['leakage_pairs']}"
    )
    if st["missing"]:
        print(f"missing raw files (skipped): {', '.join(st['missing'])}")
    if st.get("refused_licence"):
        print(f"refused by the licence gate: {', '.join(st['refused_licence'])}")
    for k, p in paths.items():
        print(f"  {k:<14} {p}")
    return 1 if st["leakage_pairs"] else 0


if __name__ == "__main__":
    sys.exit(main())

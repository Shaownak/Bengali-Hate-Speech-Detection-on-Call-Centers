"""Offline post-call report.

    python -m gale_postcall report tests/fixtures/calls/demo_01/call.wav [--out DIR] [--open]

Writes ``report.json`` and ``report.html`` to ``--out`` (default ``.gale/reports/<file>-<id>``)
and the same files into the object store under ``reports/<call_id>/``.
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

from gale_postcall.pipeline import run_postcall
from gale_postcall.report import to_html, to_json_text
from galecore.config import REPO_ROOT, get_settings
from galecore.logging import configure_logging
from galecore.objectstore import LocalObjectStore


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m gale_postcall",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    rp = sub.add_parser("report", help="run the post-call pipeline on one stereo WAV")
    rp.add_argument("file", type=Path)
    rp.add_argument("--out", type=Path)
    rp.add_argument("--call-id", help="default: a new UUID (D-18)")
    rp.add_argument(
        "--layout",
        choices=["stereo_agent_left", "stereo_agent_right", "mono"],
        default="stereo_agent_left",
    )
    rp.add_argument("--queue")
    rp.add_argument("--open", action="store_true", help="open report.html in the browser")
    a = ap.parse_args(argv)

    if not a.file.is_file():
        if a.file.name == "call.wav" and (a.file.parent / "script.json").is_file():
            from tools.fixtures_gen.__main__ import ensure_wav

            ensure_wav(a.file.parent)  # synthetic fixture: render its audio on demand (ADR-010)
        else:
            print(f"no such file: {a.file}", file=sys.stderr)
            return 2
    s = get_settings()
    configure_logging("postcall", s.log_level, s.log_json)
    r = run_postcall(a.file, s, call_id=a.call_id, layout=a.layout, queue=a.queue)
    out = (
        a.out
        or REPO_ROOT
        / ".gale"
        / "reports"
        / f"{a.file.parent.name if a.file.name == 'call.wav' else a.file.stem}-{r.call_id[:8]}"
    )
    out.mkdir(parents=True, exist_ok=True)
    json_text, html_text = to_json_text(r), to_html(r)
    (out / "report.json").write_text(json_text, encoding="utf-8", newline="\n")
    (out / "report.html").write_text(html_text, encoding="utf-8", newline="\n")
    if s.object_store == "local":
        store = LocalObjectStore(s.object_store_dir)
        store.put(f"reports/{r.call_id}/report.json", json_text.encode())
        store.put(f"reports/{r.call_id}/report.html", html_text.encode())

    print(
        f"call {r.call_id}: {r.duration_s:.1f} s, {len(r.segments)} segments, "
        f"{len(r.signals)} signals, "
        f"{len(r.events)} events ({sum(e.alert for e in r.events)} alerts), LLM {r.llm_status}, "
        f"{sum(r.stages_s.values()):.2f} s"
    )
    for e in r.events:
        print(
            f"  {e.ts_start:7.2f}-{e.ts_end:7.2f}s  {e.speaker.value:<8} {e.category.value:<22} "
            f"sev {e.severity}"
            f"  p={e.probability:.2f}  {e.status.value}{'  ALERT' if e.alert else ''}"
        )
    for n in r.notes:
        print(f"  note: {n}")
    print(f"wrote {out / 'report.json'}\nwrote {out / 'report.html'}")
    if a.open:
        webbrowser.open((out / "report.html").resolve().as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())

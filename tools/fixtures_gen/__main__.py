"""Synthetic fixture calls (ADR-010).

    python -m tools.fixtures_gen gen [--only demo_01 ...] [--tts auto|stub|local]
    python -m tools.fixtures_gen render tests/fixtures/calls/demo_01 [--force]
    python -m tools.fixtures_gen check     # committed scripts match the generator (stub TTS)

``script.json`` and ``expected_events.json`` are committed; ``call.wav`` is rendered on demand
(git-ignored) and is bit-for-bit reproducible with the stub TTS.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import soundfile as sf

from galecore.config import REPO_ROOT
from tools.fixtures_gen.compose import compose, render
from tools.fixtures_gen.models import ExpectedEvents, Script
from tools.fixtures_gen.scenarios import SCENARIOS
from tools.fixtures_gen.tts import SR, StubTTS, TTSBackend, load_tts

CALLS_DIR = REPO_ROOT / "tests" / "fixtures" / "calls"


def _dump(model: Script | ExpectedEvents) -> str:
    return json.dumps(model.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"


def _rel(p: Path) -> Path:
    return p.relative_to(REPO_ROOT) if p.is_relative_to(REPO_ROOT) else p


def _tts_dir(arg: str | None) -> Path | None:
    if arg:
        return Path(arg)
    if os.environ.get("GALE_TTS_DIR"):
        return Path(os.environ["GALE_TTS_DIR"])
    model_dir = os.environ.get("GALE_MODEL_DIR")
    return Path(model_dir) / "tts" if model_dir else None


def write_call(out: Path, script: Script, events: ExpectedEvents, tts: TTSBackend) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    (out / "script.json").write_text(_dump(script), encoding="utf-8", newline="\n")
    (out / "expected_events.json").write_text(_dump(events), encoding="utf-8", newline="\n")
    return render_wav(out, script, tts)


def render_wav(out: Path, script: Script, tts: TTSBackend) -> Path:
    wav = out / "call.wav"
    sf.write(wav, render(script, tts), SR, subtype="PCM_16")
    return wav


def ensure_wav(call_dir: Path, *, force: bool = False) -> Path:
    """Render ``call.wav`` from ``script.json`` if it is missing (stub TTS scripts only)."""
    wav = call_dir / "call.wav"
    if wav.exists() and not force:
        return wav
    script = Script.model_validate_json((call_dir / "script.json").read_text(encoding="utf-8"))
    if script.tts.get("backend") != "stub":
        raise RuntimeError(f"{call_dir} was generated with a real TTS; rerun `gen --tts local`")
    return render_wav(call_dir, script, StubTTS())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m tools.fixtures_gen",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen", help="compose scripts, ground truth and audio")
    g.add_argument("--only", nargs="*", help="call ids (default: all)")
    g.add_argument("--out", type=Path, default=CALLS_DIR)
    g.add_argument("--tts", choices=["auto", "stub", "local"], default="auto")
    g.add_argument(
        "--tts-dir", help="local TTS checkpoint dir (default $GALE_TTS_DIR or $GALE_MODEL_DIR/tts)"
    )
    r = sub.add_parser("render", help="render call.wav from an existing script.json")
    r.add_argument("call_dirs", nargs="*", type=Path)
    r.add_argument("--force", action="store_true")
    sub.add_parser("check", help="fail if committed scripts differ from the generator")
    a = ap.parse_args(argv)

    if a.cmd == "gen":
        tts = load_tts(a.tts, _tts_dir(a.tts_dir))
        for sc in SCENARIOS:
            if a.only and sc.call_id not in a.only:
                continue
            script, events = compose(sc, tts)
            wav = write_call(a.out / sc.call_id, script, events, tts)
            print(
                f"{sc.call_id}: {script.duration_s:6.1f} s, {len(script.turns):2d} turns, "
                f"{len(events.events):2d} events, tts={tts.name} → {_rel(wav)}"
            )
        return 0
    if a.cmd == "render":
        dirs = a.call_dirs or sorted(p.parent for p in CALLS_DIR.glob("*/script.json"))
        for d in dirs:
            print(ensure_wav(d, force=a.force))
        return 0
    # check
    stale = []
    for sc in SCENARIOS:
        script, events = compose(sc, StubTTS())
        d = CALLS_DIR / sc.call_id
        for name, model in (("script.json", script), ("expected_events.json", events)):
            f = d / name
            if not f.exists() or f.read_text(encoding="utf-8") != _dump(model):
                stale.append(str(f.relative_to(REPO_ROOT)))
    if stale:
        print(
            "fixtures are stale, run `python -m tools.fixtures_gen gen --tts stub`:\n  "
            + "\n  ".join(stale)
        )
        return 1
    print(f"{len(SCENARIOS)} fixture scripts up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Download pinned model revisions into ``$GALE_MODEL_DIR`` and record licences.

    python scripts/download_models.py --list
    python scripts/download_models.py whisper_bn emotion2vec_plus_large     # needs huggingface_hub
    python scripts/download_models.py --write-licenses     # rewrite ml/MODEL_LICENSES.md

Rules (CLAUDE.md 6, docs/05 §7): every model is pinned to an HF **commit hash**, never a branch;
licences are recorded in ``ml/MODEL_LICENSES.md``; services load weights only from
``GALE_MODEL_DIR`` and never reach the internet at runtime. Run this on a build host, then mirror
the directory to the internal store (Q-08).

``licence_status``:
- ``commercial_ok`` — permissive, may ship;
- ``verify`` — needs the legal review (Q-03/Q-04) before anything trained on it ships;
- ``research_only`` — non-commercial: research runs only, blocks promotion.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LICENSES_MD = REPO_ROOT / "ml" / "MODEL_LICENSES.md"


@dataclass(frozen=True)
class ModelPin:
    key: str
    repo_id: str
    revision: str  # HF commit hash, looked up 2026-11-02
    licence: str
    licence_status: str
    target: str  # sub-directory of GALE_MODEL_DIR
    used_by: str
    notes: str = ""
    gated: bool = False


PINS: tuple[ModelPin, ...] = (
    ModelPin(
        "silero_vad",
        "onnx-community/silero-vad",
        "e71cae966052b992a7eca6b17738916ce0eca4ec",
        "MIT",
        "commercial_ok",
        "vad",
        "VAD (audio-worker, asr-worker)",
        "copy onnx/model.onnx to vad/silero_vad.onnx",
    ),
    ModelPin(
        "whisper_bn",
        "bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium",
        "76cce9a874e8752d561a7aff28f01383df270db4",
        "Apache-2.0",
        "commercial_ok",
        "asr/whisper_hf",
        "ASR backend whisper_ct2 (post-call)",
        "convert: ct2-transformers-converter --model asr/whisper_hf --output_dir asr/whisper_ct2 "
        "--quantization int8_float16",
    ),
    ModelPin(
        "indic_conformer",
        "ai4bharat/indic-conformer-600m-multilingual",
        "2a77d305038dfc502fd9464e7d028b90381976cb",
        "MIT",
        "commercial_ok",
        "asr/indic_conformer",
        "ASR backend indic_conformer; CTC KWS (D-11)",
        "gated: accept the terms on HF first",
        gated=True,
    ),
    ModelPin(
        "qwen3_asr_bn",
        "SayedShaun/Qwen3-ASR-0.6B-Bengali",
        "bf3ceae2da07edf71736772f2cdf29f23419a93b",
        "Apache-2.0",
        "commercial_ok",
        "asr/qwen3_asr_bn",
        "ASR bake-off candidate (docs/03 §1.1)",
        "served via vLLM; not wired into the pipeline yet",
    ),
    ModelPin(
        "emotion2vec_plus_large",
        "emotion2vec/emotion2vec_plus_large",
        "6c303ba987b86b93193de93e34bb2b077a6bedc4",
        "other (model card licence)",
        "verify",
        "ser/emotion2vec_plus_large",
        "SER backbone (audio-worker)",
        "check the model-card licence terms (Q-03)",
    ),
    ModelPin(
        "banglabert",
        "csebuetnlp/banglabert",
        "9ce791f330578f50da6bc52b54205166fb5d1c8c",
        "not declared on HF; CC BY-NC-SA 4.0 per Q-03",
        "research_only",
        "encoders/banglabert",
        "text classifier encoder (bake-off)",
        "non-commercial: research runs only until legal says otherwise",
    ),
    ModelPin(
        "banglabert_large",
        "csebuetnlp/banglabert_large",
        "a64fb146d81ec4d7f8838b85e084b0c6a325a22f",
        "not declared on HF; CC BY-NC-SA 4.0 per Q-03",
        "research_only",
        "encoders/banglabert_large",
        "text classifier encoder (bake-off)",
        "non-commercial: research runs only until legal says otherwise",
    ),
    ModelPin(
        "muril",
        "google/muril-base-cased",
        "afd9f36c7923d54e97903922ff1b260d091d202f",
        "Apache-2.0",
        "commercial_ok",
        "encoders/muril",
        "text classifier encoder (bake-off, commercial fallback)",
    ),
    ModelPin(
        "indicbert_v2",
        "ai4bharat/IndicBERTv2-MLM-only",
        "8598f13fe52443bc3fc054fcd665944560145b5c",
        "MIT",
        "commercial_ok",
        "encoders/indicbert_v2",
        "text classifier encoder (bake-off)",
    ),
    ModelPin(
        "qwen3_5_9b",
        "Qwen/Qwen3.5-9B",
        "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "Apache-2.0",
        "commercial_ok",
        "llm/qwen3_5_9b",
        "LLM judge base (served by vLLM, docs/03 §7)",
    ),
    ModelPin(
        "qwen3_5_4b",
        "Qwen/Qwen3.5-4B",
        "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "Apache-2.0",
        "commercial_ok",
        "llm/qwen3_5_4b",
        "LLM judge base on a 12 GB GPU (ADR-021)",
        "looked up 2026-11-03",
    ),
    ModelPin(
        "qwen3_5_0_8b",
        "Qwen/Qwen3.5-0.8B",
        "2fc06364715b967f1860aea9cf38778875588b17",
        "Apache-2.0",
        "commercial_ok",
        "llm/qwen3_5_0_8b",
        "LLM judge trainer smoke runs (docs prompts §9)",
        "looked up 2026-11-03",
    ),
)
BY_KEY = {p.key: p for p in PINS}


def licenses_markdown() -> str:
    rows = "\n".join(
        f"| `{p.key}` | [{p.repo_id}](https://huggingface.co/{p.repo_id}) | `{p.revision[:12]}` "
        f"| {p.licence} | **{p.licence_status}** | {p.used_by} | {p.notes}"
        f"{' (gated)' if p.gated and 'gated' not in p.notes else ''} |"
        for p in PINS
    )
    return f"""# Model licences and pinned revisions

Generated by `python scripts/download_models.py --write-licenses`. Do not edit by hand: change
`PINS` in `scripts/download_models.py`. CLAUDE.md rule 6: every model is pinned to an HF commit
hash.

Licence status gates promotion: `commercial_ok` may ship; `verify` needs the legal review
(Q-03/Q-04) first; `research_only` may be used in research runs only and blocks promotion.
Licence fields come from the HF repo metadata (looked up 2026-11-02) and are **not legal advice**.

| key | repo | revision | licence | status | used by | notes |
|---|---|---|---|---|---|---|
{rows}

Not models but bundled at runtime: none. DSP shouting is code (rule fallback); its LightGBM model
is trained in-house (P3) and stored in the model registry, not on HF.
"""


def download(pin: ModelPin, model_dir: Path) -> Path:  # pragma: no cover - network
    from huggingface_hub import snapshot_download

    dest = model_dir / pin.target
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=pin.repo_id, revision=pin.revision, local_dir=dest)
    (dest / "REVISION").write_text(f"{pin.repo_id}@{pin.revision}\n", encoding="utf-8")
    if pin.key == "silero_vad":
        onnx = dest / "onnx" / "model.onnx"
        if onnx.is_file():
            (dest / "silero_vad.onnx").write_bytes(onnx.read_bytes())
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("keys", nargs="*", help="models to download (default: none)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--write-licenses", action="store_true")
    ap.add_argument(
        "--allow-research-only", action="store_true", help="also download research_only models"
    )
    ap.add_argument(
        "--model-dir",
        type=Path,
        default=Path(os.environ.get("GALE_MODEL_DIR", REPO_ROOT / ".gale" / "models")),
    )
    a = ap.parse_args(argv)
    if a.list:
        for p in PINS:
            print(f"{p.key:<24} {p.licence_status:<14} {p.repo_id}@{p.revision[:12]} → {p.target}")
    if a.write_licenses:
        LICENSES_MD.write_text(licenses_markdown(), encoding="utf-8", newline="\n")
        print(f"wrote {LICENSES_MD.relative_to(REPO_ROOT)}")
    for key in a.keys:
        if key not in BY_KEY:
            print(f"unknown model {key!r}; see --list", file=sys.stderr)
            return 2
        pin = BY_KEY[key]
        if pin.licence_status == "research_only" and not a.allow_research_only:
            print(
                f"{key} is research_only (licence); pass --allow-research-only for research runs",
                file=sys.stderr,
            )
            return 3
        print(f"downloaded {key} → {download(pin, a.model_dir)}")  # pragma: no cover
    return 0


if __name__ == "__main__":
    sys.exit(main())

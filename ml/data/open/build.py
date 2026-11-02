"""Open-data build (docs/09 §6): raw files → ``unified.jsonl`` + per-model views + data card.

    python -m ml.data.open build --run-tag research [--only violens] [--asr-noise 0.15]

Datasets whose files are missing are skipped and listed in the card. Text never reaches logs.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ml.data.open.augment import AsrNoise
from ml.data.open.card import data_card
from ml.data.open.clean import clean
from ml.data.open.harmonise import PARENT_SOURCE, check_no_leakage, harmonise
from ml.data.open.label_map import LabelMapper
from ml.data.open.lid import keep, script_of
from ml.data.open.loaders import TEXT_LOADERS, MissingDataError, load_audio_folders, load_banglatlit
from ml.data.open.registry import Dataset, Register, RunTag, check_allowed
from ml.data.open.schema import AudioRow, RawRow, UnifiedRow
from ml.data.open.translit import RuleTransliterator, Transliterator
from ml.data.open.views import dumps, llm_sft_records, textclf_record


@dataclass
class BuildResult:
    rows: list[UnifiedRow]
    audio: list[AudioRow]
    translit_pairs: list[dict[str, str | None]]
    stats: dict[str, Any] = field(default_factory=dict)


def to_unified(
    raw: RawRow, ds: Dataset, mapper: LabelMapper, text_bn: str, research_only: bool
) -> UnifiedRow | None:
    m = mapper.map(raw)
    if m.drop:
        return None
    cleaned, emoji = clean(text_bn)
    if not cleaned:
        return None
    parent = PARENT_SOURCE.get(ds.key)
    return UnifiedRow(
        uid=f"{ds.key}:{raw.source_id}",
        source=ds.key,
        source_split=raw.split,
        source_id=raw.source_id,
        parent_uid=f"{parent}:{raw.parent_id}" if parent and raw.parent_id else None,
        text_orig=raw.text,
        script_orig=script_of(raw.text),
        text_bn=cleaned,
        dialect=raw.dialect,
        context=[clean(c)[0] for c in raw.context][-3:],
        labels=m.labels,
        label_confidence=m.confidence,
        toxic_any=m.toxic_any,
        severity=m.severity,
        target=m.target,
        context_flags=m.context_flags,
        emoji_present=emoji,
        licence_id=ds.key,
        commercial_ok=ds.commercial_ok == "true",
        research_only=research_only,
        weight=m.weight,
    )


def build(
    reg: Register,
    raw_root: Path,
    run_tag: RunTag,
    *,
    only: list[str] | None = None,
    translit: Transliterator | None = None,
    asr_noise: float = 0.0,
    seed: int = 1,
) -> BuildResult:
    mapper = LabelMapper()
    tl = translit or RuleTransliterator()
    stats: dict[str, Any] = {
        "missing": [],
        "loaded": 0,
        "dropped_language": 0,
        "dropped_label_map": 0,
        "transliterated": 0,
    }
    rows: list[UnifiedRow] = []
    audio: list[AudioRow] = []
    pairs: list[dict[str, str | None]] = []
    for ds in reg.datasets:
        if only and ds.key not in only and ds.id not in only:
            continue
        if ds.kind in ("asr", "ser"):  # speech corpora: register only, recipes in P3
            continue
        try:
            research_only = check_allowed(ds, run_tag)
        except PermissionError:
            stats.setdefault("refused_licence", []).append(ds.key)
            continue
        try:
            if ds.kind == "audio":
                audio += list(load_audio_folders(ds, raw_root, research_only))
                continue
            if ds.kind == "translit":
                pairs += [
                    {"split": s, "roman": r, "bangla": b}
                    for s, r, b in load_banglatlit(ds, raw_root)
                ]
                continue
            raws = list(TEXT_LOADERS[ds.key](ds, raw_root))
        except MissingDataError:
            stats["missing"].append(ds.key)
            continue
        stats["loaded"] += len(raws)
        # script handling: official Bangla version first, else back-transliterate Roman/mixed rows
        roman = [r for r in raws if r.text_bn is None and script_of(r.text) in ("roman", "mixed")]
        kept_roman = [r for r in roman if keep(r.text)]
        stats["dropped_language"] += len(roman) - len(kept_roman)
        converted = dict(
            zip((id(r) for r in kept_roman), tl([r.text for r in kept_roman]), strict=True)
        )
        stats["transliterated"] += len(converted)
        for r in raws:
            if (
                r.text_bn is None
                and script_of(r.text) in ("roman", "mixed")
                and id(r) not in converted
            ):
                continue
            if script_of(r.text) == "other" and r.text_bn is None:
                stats["dropped_language"] += 1
                continue
            u = to_unified(r, ds, mapper, r.text_bn or converted.get(id(r), r.text), research_only)
            if u is None:
                stats["dropped_label_map"] += 1
                continue
            if ds.eval_only:
                u.source_split = "test"
            rows.append(u)
    rows, hstats = harmonise(rows)
    stats.update(hstats)
    stats["leakage_pairs"] = len(check_no_leakage(rows))
    if asr_noise > 0:
        rng = random.Random(seed)
        noise = AsrNoise()
        extra = [
            r.model_copy(
                update={
                    "uid": f"{r.uid}#asr",
                    "text_bn": noise(r.text_bn, rng),
                    "synthetic": True,
                    "aug": "asr_noise",
                }
            )
            for r in rows
            if r.split == "train" and rng.random() < asr_noise
        ]
        stats["asr_noise_rows"] = len(extra)
        rows += extra
    stats["output_rows"] = len(rows)
    return BuildResult(rows, audio, pairs, stats)


def write_outputs(res: BuildResult, reg: Register, out: Path, run_tag: RunTag) -> dict[str, Path]:
    (out / "views").mkdir(parents=True, exist_ok=True)
    (out / "reports").mkdir(parents=True, exist_ok=True)
    paths = {"unified": out / "unified.jsonl", "card": out / "reports" / "open_data_card.md"}
    paths["unified"].write_text(
        "".join(r.model_dump_json() + "\n" for r in res.rows), encoding="utf-8"
    )
    for split in ("train", "dev", "test"):
        p = out / "views" / f"textclf_{split}.jsonl"
        p.write_text(
            dumps([textclf_record(r) for r in res.rows if r.split == split]), encoding="utf-8"
        )
        paths[f"textclf_{split}"] = p
    paths["llm_sft"] = out / "views" / "llm_sft_open_synthetic.jsonl"
    paths["llm_sft"].write_text(
        dumps(llm_sft_records([r for r in res.rows if r.split != "test"])), encoding="utf-8"
    )
    if res.audio:
        paths["audio"] = out / "views" / "audio_manifest.jsonl"
        paths["audio"].write_text(
            "".join(a.model_dump_json() + "\n" for a in res.audio), encoding="utf-8"
        )
    if res.translit_pairs:
        paths["translit"] = out / "views" / "translit_pairs.jsonl"
        paths["translit"].write_text(dumps(list(res.translit_pairs)), encoding="utf-8")  # type: ignore[arg-type]
    paths["card"].write_text(data_card(res.rows, reg, res.stats, run_tag), encoding="utf-8")
    (out / "build_stats.json").write_text(
        json.dumps(
            {**res.stats, "run_tag": run_tag, "splits": dict(Counter(r.split for r in res.rows))},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return paths

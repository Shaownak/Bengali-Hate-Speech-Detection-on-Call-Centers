import json
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from galecore.schemas import (
    AsrSegment,
    AudioFrame,
    CallScorecard,
    Category,
    Channel,
    FusedEvent,
    Review,
    Signal,
    SignalSource,
    WindowJudgement,
    json_schemas,
)

DOCS = Path(__file__).resolve().parents[3] / "docs"


def test_category_enum_matches_annotation_guide() -> None:
    guide = (DOCS / "02_DATA_PLAN.md").read_text(encoding="utf-8")
    table = guide.split("### 3.4")[1].split("**Severity**")[0]
    in_guide = set(re.findall(r"^\| `([a-z_]+)` \|", table, flags=re.M))
    assert in_guide == {c.value for c in Category}


def test_category_enum_matches_llm_schema() -> None:
    spec = (DOCS / "08_LLM_JUDGE_SPEC.md").read_text(encoding="utf-8")
    block = spec.split('"category":  {"enum": [')[1].split("]}")[0]
    in_spec = set(re.findall(r'"([a-z_]+)"', block))
    assert in_spec == {c.value for c in Category}


def test_audio_frame_validation() -> None:
    AudioFrame(call_id="c", channel=Channel.AGENT, seq=0, ts_start=0, ts_end=0.2, pcm_s16le=b"\0\0")
    with pytest.raises(ValidationError, match="whole 16-bit"):
        AudioFrame(call_id="c", channel="agent", seq=0, ts_start=0, ts_end=0.2, pcm_s16le=b"\0")
    with pytest.raises(ValidationError, match="ts_end"):
        AudioFrame(call_id="c", channel="agent", seq=0, ts_start=1, ts_end=0.2, pcm_s16le=b"")
    with pytest.raises(ValidationError):
        AudioFrame(
            call_id="c",
            channel="agent",
            seq=0,
            ts_start=0,
            ts_end=0.2,
            pcm_s16le=b"",
            sample_rate=8000,  # type: ignore[arg-type]
        )


def test_asr_segment_has_no_raw_text_field() -> None:
    # D-01: only redacted text exists on the bus
    assert "text" not in AsrSegment.model_fields
    with pytest.raises(ValidationError, match="extra"):
        AsrSegment.model_validate(
            {"call_id": "c", "channel": "agent", "segment_id": "s", "ts_start": 0, "ts_end": 1,
             "text_redacted": "x", "text": "raw", "is_final": True, "avg_conf": 0.9,
             "model_version": "v"}
        )  # fmt: skip


def test_signal_round_trip_and_shadow_default() -> None:
    sig = Signal(
        call_id="c", channel="customer", signal_id="s1", source=SignalSource.VAD_OVERTALK,
        category=Category.INTERRUPTION_OVERTALK, score=0.7, ts_start=1, ts_end=2,
        model_version="v",
    )  # fmt: skip
    assert sig.shadow is False
    assert Signal.model_validate_json(sig.model_dump_json()) == sig
    with pytest.raises(ValidationError):
        Signal(**{**sig.model_dump(), "score": 1.5})


def test_fused_event_speaker_cannot_be_mixed() -> None:
    with pytest.raises(ValidationError):
        FusedEvent(
            event_id="e", call_id="c", speaker="mixed", category="insult",  # type: ignore[arg-type]
            severity=2, probability=0.9, ts_start=0, ts_end=1,
        )  # fmt: skip
    with pytest.raises(ValidationError):
        FusedEvent(
            event_id="e", call_id="c", speaker="agent", category="insult",
            severity=4, probability=0.9, ts_start=0, ts_end=1,  # type: ignore[arg-type]
        )  # fmt: skip


def test_review_rules() -> None:
    Review(event_id="e", reviewer_user_id="u", verdict="correct")
    Review(event_id="e", reviewer_user_id="u", kind="dispute", comment="not me")
    with pytest.raises(ValidationError, match="needs a verdict"):
        Review(event_id="e", reviewer_user_id="u")
    with pytest.raises(ValidationError, match="corrected_category"):
        Review(event_id="e", reviewer_user_id="u", verdict="wrong_category")
    with pytest.raises(ValidationError, match="corrected_severity"):
        Review(event_id="e", reviewer_user_id="u", verdict="wrong_severity")


def test_window_judgement_new_category_is_enum() -> None:
    ok = {"events": [], "provisional_decisions": [
        {"event_id": "e", "decision": "change_category", "new_category": "insult"}]}  # fmt: skip
    WindowJudgement.model_validate(ok)
    bad = json.loads(json.dumps(ok))
    bad["provisional_decisions"][0]["new_category"] = "rudeness"
    with pytest.raises(ValidationError):
        WindowJudgement.model_validate(bad)


def test_scorecard_requires_all_sub_fields() -> None:
    schema = CallScorecard.model_json_schema()
    for sub in ("AgentScore", "CustomerScore", "Escalation"):
        props = schema["$defs"][sub]["properties"]
        assert set(schema["$defs"][sub]["required"]) == set(props)


def test_json_schemas_exported() -> None:
    schemas = json_schemas()
    assert {"AudioFrame", "Signal", "FusedEvent", "WindowJudgement", "CallScorecard"} <= set(
        schemas
    )
    assert schemas["AudioFrame"]["properties"]["pcm_s16le"]["format"] in ("base64", "base64url")

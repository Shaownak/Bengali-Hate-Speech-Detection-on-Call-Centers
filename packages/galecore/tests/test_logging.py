import io
import json

from galecore.logging import configure_logging, get_logger


def capture(**kwargs: object) -> dict[str, object]:
    buf = io.StringIO()
    configure_logging("test-svc", stream=buf)
    get_logger("t").info("event_name", **kwargs)
    return json.loads(buf.getvalue().strip().splitlines()[-1])


def test_content_keys_are_dropped() -> None:
    out = capture(
        text="raw words", text_redacted="still content", transcript="x", audio_b64="AAAA",
        pcm_s16le=b"\0\0", call_id="c-1", latency_ms=12.5,
    )  # fmt: skip
    for key in ("text", "text_redacted", "transcript", "audio_b64", "pcm_s16le"):
        assert key not in out
    assert out["call_id"] == "c-1"
    assert out["latency_ms"] == 12.5
    assert out["service"] == "test-svc"
    assert out["level"] == "info"


def test_pii_patterns_redacted_in_values_and_message() -> None:
    buf = io.StringIO()
    configure_logging("svc", stream=buf)
    get_logger().warning("failed for 01712345678", detail="nid 1234567890", nested={"n": "a@b.co"})
    out = json.loads(buf.getvalue())
    assert "01712345678" not in buf.getvalue()
    assert out["event"] == "failed for <PHONE>"
    assert out["detail"] == "nid <NID>"
    assert out["nested"] == {"n": "<EMAIL>"}


def test_nested_forbidden_keys_and_bytes() -> None:
    out = capture(meta={"text": "x", "ok": 1, "items": ["01712345678", b"raw"]})
    assert out["meta"] == {"text": "[dropped]", "ok": 1, "items": ["<PHONE>", "[dropped]"]}


def test_level_filtering() -> None:
    buf = io.StringIO()
    configure_logging("svc", level="WARNING", stream=buf)
    get_logger().info("hidden")
    assert buf.getvalue() == ""
    configure_logging("svc", level="nonsense", json=False, stream=buf)
    get_logger().info("shown")
    assert "shown" in buf.getvalue()

import json
import math
import zlib

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from gale_ingest_gateway.app import build_app
from gale_ingest_gateway.gateway import GatewayError, GatewaySettings, Registry, RtpSessionRequest
from gale_ingest_gateway.sessions import FRAME_SAMPLES, ChannelAssembler, CountingSink
from galecore.audio import encode_alaw, encode_ulaw, s16le_to_float
from galecore.config import Settings
from galecore.rtp import PT_PCMA, PT_PCMU, RtpPacket
from galecore.schemas import AudioFrame, Channel

SETTINGS = Settings(env="test", bus_backend="memory")
GW = GatewaySettings(rtp_port=0)


def sine(n: int, sr: int = 8000, hz: float = 1000.0, amp: float = 0.3) -> np.ndarray:
    return np.round(amp * 32767 * np.sin(2 * np.pi * hz * np.arange(n) / sr)).astype(np.int16)


class ListSink:
    def __init__(self) -> None:
        self.frames: list[AudioFrame] = []

    def __call__(self, frame: AudioFrame) -> None:
        self.frames.append(frame)


def test_assembler_resamples_and_frames_continuously() -> None:
    sink = ListSink()
    asm = ChannelAssembler("c1", Channel.AGENT, 8000, sink, "rtp")
    x = sine(8000 + 123)  # ~1.015 s in odd-sized chunks
    for i in range(0, len(x), 77):
        asm.push(x[i : i + 77])
    stats = asm.finish()
    out = np.concatenate([s16le_to_float(f.pcm_s16le) for f in sink.frames])
    assert stats.frames_out == len(sink.frames) == math.ceil(len(out) / FRAME_SAMPLES)
    assert abs(len(out) - 2 * len(x)) <= 2
    assert [f.seq for f in sink.frames] == list(range(len(sink.frames)))
    assert all(
        a.ts_end == pytest.approx(b.ts_start)
        for a, b in zip(sink.frames, sink.frames[1:], strict=False)
    )
    assert all(len(f.pcm_s16le) == 2 * FRAME_SAMPLES for f in sink.frames[:-1])
    # the 16 kHz output is still a clean 1 kHz sine (ignore filter edges)
    mid = out[2000:-2000]
    ref = 0.3 * np.sin(2 * np.pi * 1000 * (np.arange(len(out)) / 16000))[2000:-2000]
    best = max(np.corrcoef(mid, np.roll(ref, k))[0, 1] for k in range(-40, 41))
    assert best > 0.999
    assert stats.samples_in == len(x)
    assert stats.duration_s == pytest.approx(len(x) / 8000, abs=1e-3)
    assert stats.pcm_crc32 == zlib.crc32(x.astype("<i2").tobytes())


def test_assembler_16k_passthrough() -> None:
    sink = ListSink()
    asm = ChannelAssembler("c1", Channel.MIXED, 16000, sink, "ws")
    asm.push(sine(4000, sr=16000))
    asm.finish()
    assert [len(f.pcm_s16le) // 2 for f in sink.frames] == [3200, 800]


def _registry(**kw: int) -> Registry:
    return Registry(GatewaySettings(rtp_port=0, **kw), CountingSink())


def test_registry_rejects_conflicts() -> None:
    reg = _registry(max_sessions=2)
    reg.open_rtp(RtpSessionRequest(call_id="a", ssrc={Channel.AGENT: 1, Channel.CUSTOMER: 2}))
    with pytest.raises(GatewayError) as e:
        reg.open_rtp(RtpSessionRequest(call_id="a", ssrc={Channel.AGENT: 3}))
    assert e.value.status == 409
    with pytest.raises(GatewayError) as e:
        reg.open_rtp(RtpSessionRequest(call_id="b", ssrc={Channel.AGENT: 2}))
    assert e.value.status == 409
    with pytest.raises(GatewayError) as e:
        reg.open_rtp(RtpSessionRequest(call_id="b", ssrc={Channel.MIXED: 5, Channel.AGENT: 6}))
    assert e.value.status == 422
    reg.open_rtp(RtpSessionRequest(call_id="b", ssrc={Channel.MIXED: 7}))
    with pytest.raises(GatewayError) as e:
        reg.open_rtp(RtpSessionRequest(call_id="c", ssrc={Channel.MIXED: 8}))
    assert e.value.status == 503
    with pytest.raises(GatewayError) as e:
        reg.get("nope")
    assert e.value.status == 404
    reg.close("a")
    assert 1 not in reg.by_ssrc
    reg.open_rtp(RtpSessionRequest(call_id="a", ssrc={Channel.AGENT: 1}))  # reusable after close


def test_datagrams_decode_and_bad_packets_are_dropped() -> None:
    reg = _registry()
    reg.open_rtp(
        RtpSessionRequest(call_id="a", codec="pcma", ssrc={Channel.AGENT: 11, Channel.CUSTOMER: 12})
    )
    x = sine(800)
    payload = encode_alaw(x)
    for k in range(5):
        reg.on_datagram(
            RtpPacket(PT_PCMA, k, 160 * k, 11, payload[160 * k : 160 * (k + 1)]).to_bytes()
        )
    reg.on_datagram(RtpPacket(PT_PCMA, 4, 640, 11, payload[640:800]).to_bytes())  # duplicate
    reg.on_datagram(RtpPacket(PT_PCMA, 0, 0, 99, payload[:160]).to_bytes())  # unknown ssrc
    reg.on_datagram(RtpPacket(PT_PCMU, 0, 0, 12, encode_ulaw(x[:160])).to_bytes())  # wrong codec
    reg.on_datagram(b"\x00\x01")  # malformed
    stats = reg.close("a").channels
    assert stats[Channel.AGENT].packets == 5
    assert stats[Channel.AGENT].duplicates == 1
    assert stats[Channel.AGENT].samples_in == 800
    assert stats[Channel.CUSTOMER].packets == 0


def test_rtp_http_session_lifecycle() -> None:
    app = build_app(SETTINGS, GW)
    with TestClient(app) as client:
        r = client.post(
            "/v1/rtp/sessions", json={"call_id": "c9", "ssrc": {"agent": 1, "customer": 2}}
        )
        assert r.status_code == 201
        assert r.json()["rtp_port"] > 0
        assert (
            client.post(
                "/v1/rtp/sessions", json={"call_id": "c9", "ssrc": {"agent": 3}}
            ).status_code
            == 409
        )
        assert client.get("/v1/sessions/c9").json()["closed"] is False
        final = client.delete("/v1/rtp/sessions/c9").json()
        assert final["closed"] is True
        assert set(final["channels"]) == {"agent", "customer"}
        assert client.get("/v1/sessions/missing").status_code == 404
        assert client.delete("/v1/rtp/sessions/missing").status_code == 404


def _ws_stream(
    client: TestClient, call_id: str, start: dict[str, object], chunks: list[bytes]
) -> dict[str, object]:
    with client.websocket_connect(f"/v1/stream/{call_id}") as ws:
        ws.send_text(json.dumps(start))
        for c in chunks:
            ws.send_bytes(c)
        ws.send_text(json.dumps({"event": "stop"}))
        reply: dict[str, object] = ws.receive_json()
    return reply


def test_ws_stereo_with_split_samples_and_right_layout() -> None:
    app = build_app(SETTINGS, GW)
    agent, customer = sine(1600, hz=500), sine(1600, hz=1500)
    inter = np.stack([customer, agent], axis=1).astype("<i2").tobytes()  # agent on the right
    chunks = [inter[i : i + 333] for i in range(0, len(inter), 333)]  # splits mid-sample
    with TestClient(app) as client:
        reply = _ws_stream(client, "w1", {"event": "start", "layout": "stereo_agent_right"}, chunks)
    ch = reply["channels"]
    assert reply["event"] == "stats"
    assert reply["transport"] == "ws"
    assert ch["agent"]["pcm_crc32"] == zlib.crc32(agent.astype("<i2").tobytes())  # type: ignore[index]
    assert ch["customer"]["pcm_crc32"] == zlib.crc32(customer.astype("<i2").tobytes())  # type: ignore[index]
    assert ch["agent"]["frames_out"] == 1  # type: ignore[index]  # 0.2 s at 8 kHz → one 200 ms frame


def test_ws_mono_and_bad_start() -> None:
    app = build_app(SETTINGS, GW)
    with TestClient(app) as client:
        reply = _ws_stream(
            client,
            "m1",
            {"event": "start", "channels": 1, "sample_rate": 16000},
            [sine(4800, sr=16000).astype("<i2").tobytes()],
        )
        assert list(reply["channels"]) == ["mixed"]  # type: ignore[call-overload]
        assert reply["channels"]["mixed"]["frames_out"] == 2  # type: ignore[index]
        with client.websocket_connect("/v1/stream/bad") as ws:
            ws.send_text(json.dumps({"event": "start", "sample_rate": 44100}))
            assert ws.receive_json()["event"] == "error"
            with pytest.raises(WebSocketDisconnect):
                ws.receive_json()


def test_ws_disconnect_without_stop_closes_session() -> None:
    app = build_app(SETTINGS, GW)
    with TestClient(app) as client:
        with client.websocket_connect("/v1/stream/d1") as ws:
            ws.send_text(json.dumps({"event": "start"}))
            ws.send_bytes(np.zeros(320, dtype="<i2").tobytes())
        assert client.get("/v1/sessions/d1").json()["closed"] is True

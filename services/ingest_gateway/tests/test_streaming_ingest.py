import asyncio
import json
from pathlib import Path

import httpx
import numpy as np
import pytest
import soundfile as sf
from fakeredis import FakeAsyncRedis
from fastapi.testclient import TestClient

from gale_ingest_gateway.app import build_app, watch_drop_dir
from gale_ingest_gateway.ari import AriApp
from gale_ingest_gateway.gateway import GatewayError, GatewaySettings, Registry, RtpSessionRequest
from gale_ingest_gateway.publisher import BusPublisher
from gale_ingest_gateway.sessions import CountingSink
from galecore.audio import encode_ulaw
from galecore.bus import CALL_LIFECYCLE, POSTCALL_JOBS, BusClient, audio_stream, decode
from galecore.config import Settings
from galecore.objectstore import LocalObjectStore
from galecore.rtp import PT_PCMU, RtpPacket
from galecore.schemas import AudioFrame, CallLifecycle, Channel, PostcallJob


class Events:
    def __init__(self) -> None:
        self.log: list[tuple[str, str]] = []

    def started(self, call_id: str, external_ref: str | None, layout: str | None) -> None:
        self.log.append(("started", f"{call_id}|{external_ref}|{layout}"))

    def ended(self, call_id: str, duration_s: float) -> None:
        self.log.append(("ended", f"{call_id}|{duration_s}"))


def test_rtp_by_source_address_and_lifecycle() -> None:
    ev = Events()
    reg = Registry(GatewaySettings(rtp_port=0), CountingSink(), ev)
    reg.open_rtp(
        RtpSessionRequest(
            call_id="c1",
            remote={Channel.AGENT: "10.0.0.5:4000", Channel.CUSTOMER: "10.0.0.5:4002"},
            external_ref="asterisk:1700.1",
        )
    )
    payload = encode_ulaw(np.zeros(160, np.int16))
    reg.on_datagram(RtpPacket(PT_PCMU, 1, 0, 999, payload).to_bytes(), ("10.0.0.5", 4002))
    reg.on_datagram(
        RtpPacket(PT_PCMU, 1, 0, 999, payload).to_bytes(), ("10.0.0.9", 1)
    )  # unknown source
    stats = reg.close("c1")
    assert stats.channels[Channel.CUSTOMER].packets == 1
    assert stats.channels[Channel.AGENT].packets == 0
    assert ev.log == [("started", "c1|asterisk:1700.1|stereo_agent_left"), ("ended", "c1|0.02")]
    assert reg.by_addr == {}
    with pytest.raises(GatewayError, match="either ssrc or remote"):
        reg.open_rtp(RtpSessionRequest(call_id="c2"))
    reg.open_rtp(RtpSessionRequest(call_id="c3", remote={Channel.AGENT: "h:1"}))
    with pytest.raises(GatewayError, match="already in use"):
        reg.open_rtp(RtpSessionRequest(call_id="c4", remote={Channel.AGENT: "h:1"}))


async def test_publisher_publishes_and_retains(tmp_path: Path) -> None:
    bus = BusClient(FakeAsyncRedis())
    store = LocalObjectStore(tmp_path)
    pub = BusPublisher(bus, store)
    stop = asyncio.Event()
    task = asyncio.create_task(pub.run(stop))
    pub.started("c1", "fixture:demo_01", "stereo_agent_left")
    pcm = (np.ones(3200) * 1000).astype("<i2").tobytes()
    for i in range(3):
        pub.frame(
            AudioFrame(
                call_id="c1",
                channel=Channel.AGENT,
                seq=i,
                ts_start=i * 0.2,
                ts_end=(i + 1) * 0.2,
                pcm_s16le=pcm,
            )
        )
    pub.ended("c1", 0.6)
    stop.set()
    await task
    life = [decode(f, CallLifecycle) for _, f in await bus.redis.xrange(CALL_LIFECYCLE)]
    assert [e.event for e in life] == ["started", "ended"]
    assert life[0].external_ref == "fixture:demo_01"
    assert life[0].wall_clock is not None
    assert await bus.redis.xlen(audio_stream("c1")) == 3
    x, sr = sf.read(tmp_path / "audio" / "c1" / "agent.wav")
    assert (sr, len(x)) == (16000, 9600)


def test_freeswitch_fork_endpoint() -> None:
    ev = Events()
    app = build_app(
        Settings(env="test", bus_backend="memory", workers=False),
        GatewaySettings(rtp_port=0),
        sink=CountingSink(),
        events=ev,
    )
    pcm = np.zeros((1600, 2), "<i2").tobytes()
    with TestClient(app) as client:
        with client.websocket_connect("/v1/fork") as ws:
            ws.send_text(
                json.dumps(
                    {
                        "fs_uuid": "fs-123",
                        "sample_rate": 8000,
                        "channels": 2,
                        "agent_channel": "right",
                    }
                )
            )
            accepted = ws.receive_json()
            assert accepted["event"] == "accepted"
            ws.send_bytes(pcm)
            ws.send_text(json.dumps({"event": "stop"}))
            stats = ws.receive_json()
        assert stats["channels"]["agent"]["duration_s"] == 0.2
        with client.websocket_connect("/v1/fork") as ws:
            ws.send_text("not json")
            assert ws.receive_json()["event"] == "error"
    assert ev.log[0] == ("started", f"{accepted['call_id']}|fs-123|stereo_agent_right")


async def test_drop_folder_queues_postcall_jobs(tmp_path: Path) -> None:
    bus = BusClient(FakeAsyncRedis())
    drop = tmp_path / "incoming"
    drop.mkdir()
    sf.write(drop / "rec1.wav", np.zeros((800, 2), np.int16), 8000)
    (drop / "rec1.json").write_text(
        json.dumps({"queue": "billing", "script": str(tmp_path / "fx" / "script.json")}),
        encoding="utf-8",
    )
    stop = asyncio.Event()
    task = asyncio.create_task(watch_drop_dir(bus, drop, stop, 0.05))
    await asyncio.sleep(0.3)
    stop.set()
    await task
    jobs = [decode(f, PostcallJob) for _, f in await bus.redis.xrange(POSTCALL_JOBS)]
    assert len(jobs) == 1
    assert jobs[0].queue == "billing"
    assert jobs[0].external_ref == f"fixture:{tmp_path / 'fx'}"
    assert Path(jobs[0].uri).is_file()  # noqa: ASYNC240
    assert list(drop.iterdir()) == []  # moved to processed/, queued once


async def test_ari_app_snoops_and_registers() -> None:
    calls: list[tuple[str, str, dict[str, str]]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        params = dict(req.url.params)
        calls.append((req.method, req.url.path, params))
        path = req.url.path.removeprefix("/ari")
        if path.endswith("/variable"):
            return httpx.Response(
                200, json={"value": "agent" if path.startswith("/channels/agent-1") else ""}
            )
        if path.endswith("/snoop"):
            return httpx.Response(200, json={"id": params["snoopId"]})
        if path == "/channels/externalMedia":
            port = "5000" if "-in-" in params["channelId"] else "5002"
            return httpx.Response(
                200,
                json={
                    "id": params["channelId"],
                    "channelvars": {
                        "UNICASTRTP_LOCAL_ADDRESS": "10.1.1.1",
                        "UNICASTRTP_LOCAL_PORT": port,
                    },
                },
            )
        if path == "/bridges":
            return httpx.Response(200, json={"id": params["name"]})
        return httpx.Response(204)

    ev = Events()
    reg = Registry(GatewaySettings(rtp_port=0), CountingSink(), ev)
    app = AriApp(
        reg,
        "http://asterisk:8088",
        "gale",
        "secret",
        rtp_host="10.2.2.2",
        rtp_port=40000,
        transport=httpx.MockTransport(handler),
    )
    agent = {"id": "agent-1", "name": "PJSIP/agent-0001", "linkedid": "1700.9"}
    await app.on_event(
        {"type": "ChannelEnteredBridge", "channel": {"id": "cust-1", "name": "PJSIP/trunk-1"}}
    )
    assert app.monitors == {}  # not marked as the agent leg
    await app.on_event({"type": "ChannelEnteredBridge", "channel": agent})
    mon = app.monitors["agent-1"]
    session = reg.sessions[mon.call_id]
    assert set(session.channels) == {Channel.AGENT, Channel.CUSTOMER}
    assert reg.by_addr["10.1.1.1:5000"][1] == Channel.AGENT  # spy=in is the agent's voice
    assert reg.by_addr["10.1.1.1:5002"][1] == Channel.CUSTOMER
    snoops = [c for c in calls if c[1].endswith("/snoop")]
    assert [c[2]["spy"] for c in snoops] == ["in", "out"]
    em = [c for c in calls if c[1].endswith("/externalMedia")]
    assert em[0][2]["external_host"] == "10.2.2.2:40000"
    assert ev.log[0][1].endswith("|asterisk:1700.9|stereo_agent_left")
    await app.on_event({"type": "ChannelLeftBridge", "channel": agent})
    assert app.monitors == {}
    assert reg.sessions[mon.call_id].closed
    assert sum(1 for c in calls if c[0] == "DELETE") == 6  # 2 snoops, 2 external media, 2 bridges

    def failing(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/variable"):
            return httpx.Response(200, json={"value": "agent"})
        return httpx.Response(500)

    bad = AriApp(reg, "http://asterisk:8088", "gale", "x", transport=httpx.MockTransport(failing))
    await bad.on_event(
        {"type": "ChannelEnteredBridge", "channel": {"id": "agent-2", "name": "PJSIP/a"}}
    )
    assert bad.monitors == {}  # failure tears down cleanly and never touches the call


def test_settings_disable_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GALE_WORKERS", "false")
    assert Settings(env="test").workers is False

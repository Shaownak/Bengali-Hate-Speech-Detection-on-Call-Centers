import asyncio
import json
import math
import socket
import time
from pathlib import Path

import httpx
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from gale_ingest_gateway.app import build_app
from gale_ingest_gateway.gateway import GatewaySettings
from galecore.audio import decode_alaw, decode_ulaw
from galecore.config import Settings
from galecore.rtp import parse
from tools.simulator.__main__ import main
from tools.simulator.core import (
    Call,
    Pacer,
    expected_crc,
    file_drop,
    g711_roundtrip,
    load_call,
    run_rtp,
    send_rtp,
    ssrcs_for,
    ws_chunks,
)


def _call(tmp_path: Path, seconds: float = 1.3, sr: int = 8000) -> Call:
    rng = np.random.default_rng(7)
    n = round(seconds * sr)
    t = np.arange(n) / sr
    agent = 0.25 * np.sin(2 * np.pi * 220 * t) * (1 + np.sin(2 * np.pi * 4 * t)) / 2
    customer = 0.2 * rng.standard_normal(n).clip(-3, 3) / 3
    audio = np.round(np.stack([agent, customer], axis=1) * 32767).astype(np.int16)
    wav = tmp_path / "call.wav"
    sf.write(wav, audio, sr, subtype="PCM_16")
    return load_call(tmp_path, "sim-test")


@pytest.mark.parametrize("codec", ["pcmu", "pcma"])
async def test_rtp_output_decodes_back_to_the_original(tmp_path: Path, codec: str) -> None:
    call = _call(tmp_path)
    got: list[bytes] = []

    class Rx(asyncio.DatagramProtocol):
        def datagram_received(self, data: bytes, addr: object) -> None:
            got.append(data)

    loop = asyncio.get_running_loop()
    transport, _ = await loop.create_datagram_endpoint(Rx, local_addr=("127.0.0.1", 0))
    port = transport.get_extra_info("sockname")[1]
    ssrc = ssrcs_for(call.call_id)
    sent = await send_rtp(call, "127.0.0.1", port, ssrc=ssrc, codec=codec, speed=0)  # type: ignore[arg-type]
    for _ in range(100):
        if len(got) >= sum(sent.values()):
            break
        await asyncio.sleep(0.01)
    transport.close()

    n_pkts = math.ceil(len(call.audio) / 160)
    assert sent == {"agent": n_pkts, "customer": n_pkts}
    assert len(got) == 2 * n_pkts
    pkts = [parse(d) for d in got]
    for i, ch in enumerate(["agent", "customer"]):
        mine = sorted((p for p in pkts if p.ssrc == ssrc[ch]), key=lambda p: p.timestamp)
        assert mine[0].marker
        dec = decode_ulaw if codec == "pcmu" else decode_alaw
        decoded = np.concatenate([dec(p.payload) for p in mine])
        orig = call.audio[:, i]
        # bit-exact with the G.711 reference, and close to the original PCM
        assert np.array_equal(decoded, g711_roundtrip(orig, codec))  # type: ignore[arg-type]
        err = decoded.astype(float) - orig
        snr = 10 * np.log10(np.sum(orig.astype(float) ** 2) / np.sum(err**2))
        assert snr > 30


async def test_pacer_follows_speed() -> None:
    p = Pacer(10.0)
    t0 = time.monotonic()
    for k in range(11):
        await p.wait(k * 0.1)  # 1 s of media at 10× → ~0.1 s
    assert 0.09 <= time.monotonic() - t0 < 0.6


def test_ws_chunks_reassemble(tmp_path: Path) -> None:
    call = _call(tmp_path)
    chunks = ws_chunks(call)
    assert len(chunks[0]) == 160 * 2 * 2
    back = np.frombuffer(b"".join(chunks), dtype="<i2").reshape(-1, 2)
    assert np.array_equal(back, call.audio)


def test_load_call_rejects_mono(tmp_path: Path) -> None:
    wav = tmp_path / "mono.wav"
    sf.write(wav, np.zeros(800, np.int16), 8000)
    with pytest.raises(ValueError, match="stereo"):
        load_call(wav)


def test_file_drop_is_atomic(tmp_path: Path) -> None:
    call = _call(tmp_path)
    out = file_drop(call, tmp_path / "drop")
    files = sorted(p.name for p in (tmp_path / "drop" / "incoming").iterdir())
    assert files == ["sim-test.json", "sim-test.wav"]
    assert out.read_bytes() == call.path.read_bytes()
    meta = json.loads(
        (tmp_path / "drop" / "incoming" / "sim-test.json").read_text(encoding="utf-8")
    )
    assert meta["synthetic"] is True
    assert meta["layout"] == "stereo_agent_left"


async def test_end_to_end_rtp_into_gateway(tmp_path: Path) -> None:
    call = _call(tmp_path, seconds=2.0)
    app = build_app(Settings(env="test", bus_backend="memory"), GatewaySettings(rtp_port=0))
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://gw") as client:
            sent, stats = await run_rtp(
                client, call, rtp_host="127.0.0.1", codec="pcmu", speed=0, drain_s=0.2
            )
    exp = expected_crc(call, "rtp", "pcmu")
    for ch in ("agent", "customer"):
        cs = stats["channels"][ch]
        assert cs["packets"] == sent[ch] == 100
        assert cs["lost"] == 0
        assert cs["pcm_crc32"] == exp[ch]
        assert cs["frames_out"] == 10  # 2 s → ten 200 ms frames at 16 kHz


def test_end_to_end_ws_into_gateway(tmp_path: Path) -> None:
    call = _call(tmp_path, seconds=1.0)
    app = build_app(Settings(env="test", bus_backend="memory"), GatewaySettings(rtp_port=0))
    with TestClient(app) as client, client.websocket_connect(f"/v1/stream/{call.call_id}") as ws:
        ws.send_text(json.dumps({"event": "start", "sample_rate": 8000, "channels": 2}))
        for c in ws_chunks(call):
            ws.send_bytes(c)
        ws.send_text(json.dumps({"event": "stop"}))
        stats = ws.receive_json()
    exp = expected_crc(call, "ws")
    assert {ch: stats["channels"][ch]["pcm_crc32"] for ch in exp} == exp


def test_cli_file_mode(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _call(tmp_path)
    assert (
        main(
            [
                str(tmp_path),
                "--mode",
                "file",
                "--call-id",
                "cli-1",
                "--drop-dir",
                str(tmp_path / "d"),
            ]
        )
        == 0
    )
    assert (tmp_path / "d" / "incoming" / "cli-1.wav").exists()
    assert "dropped" in capsys.readouterr().out


def test_cli_rtp_needs_a_gateway_when_spawn_never(tmp_path: Path) -> None:
    _call(tmp_path)
    with socket.socket() as s:  # a port nobody listens on
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with pytest.raises(SystemExit, match="not reachable"):
        main([str(tmp_path), "--spawn", "never", "--gateway", f"http://127.0.0.1:{port}"])

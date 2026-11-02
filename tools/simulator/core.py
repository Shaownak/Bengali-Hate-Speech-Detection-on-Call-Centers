"""Replay a stereo call as live RTP, WebSocket PCM, or a post-call file drop."""

from __future__ import annotations

import asyncio
import json
import shutil
import socket
import time
import uuid
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import httpx
import numpy as np
import numpy.typing as npt
import soundfile as sf

from galecore.audio import decode_alaw, decode_ulaw, encode_alaw, encode_ulaw
from galecore.rtp import PT_PCMA, PT_PCMU, SAMPLES_PER_PACKET, TELEPHONY_SR, packetize

Codec = Literal["pcmu", "pcma"]
Int16 = npt.NDArray[np.int16]
CHANNELS = ("agent", "customer")  # stereo_agent_left: column 0 = agent, 1 = customer


@dataclass
class Call:
    call_id: str
    path: Path
    sample_rate: int
    audio: Int16  # samples × 2 (agent left)
    external_ref: str | None = (
        None  # "fixture:<dir>" for synthetic fixtures (stub models key off it)
    )

    @property
    def duration_s(self) -> float:
        return len(self.audio) / self.sample_rate


def load_call(path: Path, call_id: str | None = None) -> Call:
    """``path`` is a fixture dir (with call.wav) or a stereo WAV file."""
    wav = path / "call.wav" if path.is_dir() else path
    audio, sr = sf.read(wav, dtype="int16", always_2d=True)
    if audio.shape[1] != 2:
        raise ValueError(
            f"{wav}: expected a stereo (agent left) recording, got {audio.shape[1]} channel(s)"
        )
    # GAL-E mints UUID call ids (D-18), which also keeps them readable in scrubbed logs
    ref = (
        f"fixture:{path.resolve()}" if path.is_dir() and (path / "script.json").is_file() else None
    )
    return Call(call_id or str(uuid.uuid4()), wav, int(sr), np.ascontiguousarray(audio), ref)


class Pacer:
    """Schedules media time against the wall clock. ``speed`` 2.0 = twice real time, 0 = unpaced."""

    def __init__(self, speed: float):
        self.speed = speed
        self._t0: float | None = None

    async def wait(self, media_t: float) -> None:
        if self.speed <= 0:
            await asyncio.sleep(0)
            return
        now = time.monotonic()
        if self._t0 is None:
            self._t0 = now
        delay = self._t0 + media_t / self.speed - now
        if delay > 0:
            await asyncio.sleep(delay)


def g711(pcm: Int16, codec: Codec) -> bytes:
    return encode_ulaw(pcm) if codec == "pcmu" else encode_alaw(pcm)


def g711_roundtrip(pcm: Int16, codec: Codec) -> Int16:
    """What a receiver decodes after G.711 coding (the reference for round-trip checks)."""
    data = g711(pcm, codec)
    return decode_ulaw(data) if codec == "pcmu" else decode_alaw(data)


def crc32(pcm: Int16) -> int:
    return zlib.crc32(pcm.astype("<i2").tobytes())


def ssrcs_for(call_id: str) -> dict[str, int]:
    base = zlib.crc32(call_id.encode()) & 0xFFFFFFF0
    return {"agent": base | 1, "customer": base | 2}


async def send_rtp(
    call: Call,
    host: str,
    port: int,
    *,
    ssrc: dict[str, int],
    codec: Codec = "pcmu",
    speed: float = 1.0,
) -> dict[str, int]:
    """Stream both channels as 20 ms G.711 RTP packets, one UDP socket per channel (PBX-style).

    Returns packets sent per channel.
    """
    if call.sample_rate != TELEPHONY_SR:
        raise ValueError("RTP mode needs 8 kHz audio")
    pt = PT_PCMU if codec == "pcmu" else PT_PCMA
    streams = {
        ch: list(
            packetize(
                g711(call.audio[:, i], codec),
                ssrc=ssrc[ch],
                payload_type=pt,
                start_seq=ssrc[ch] & 0xFFFF,
                start_ts=ssrc[ch] & 0xFFFF,
            )
        )
        for i, ch in enumerate(CHANNELS)
    }
    socks = {ch: socket.socket(socket.AF_INET, socket.SOCK_DGRAM) for ch in CHANNELS}
    pacer = Pacer(speed)
    try:
        for k in range(len(streams["agent"])):
            await pacer.wait(k * SAMPLES_PER_PACKET / TELEPHONY_SR)
            for ch in CHANNELS:
                socks[ch].sendto(streams[ch][k].to_bytes(), (host, port))
    finally:
        for s in socks.values():
            s.close()
    return {ch: len(streams[ch]) for ch in CHANNELS}


def ws_chunks(call: Call, chunk_ms: int = 20) -> list[bytes]:
    step = call.sample_rate * chunk_ms // 1000
    data = call.audio.astype("<i2")
    return [data[i : i + step].tobytes() for i in range(0, len(data), step)]


async def send_ws(
    call: Call, url: str, *, speed: float = 1.0, chunk_ms: int = 20
) -> dict[str, Any]:
    """Stream interleaved s16le stereo over a WebSocket; return the gateway's final stats."""
    import websockets

    pacer = Pacer(speed)
    async with websockets.connect(url, max_size=2**22) as ws:
        await ws.send(
            json.dumps(
                {
                    "event": "start",
                    "sample_rate": call.sample_rate,
                    "channels": 2,
                    "encoding": "pcm_s16le",
                    "layout": "stereo_agent_left",
                    "external_ref": call.external_ref,
                }
            )
        )
        for k, chunk in enumerate(ws_chunks(call, chunk_ms)):
            await pacer.wait(k * chunk_ms / 1000)
            await ws.send(chunk)
        await ws.send(json.dumps({"event": "stop"}))
        reply: dict[str, Any] = json.loads(await ws.recv())
    if reply.get("event") != "stats":
        raise RuntimeError(f"gateway rejected the stream: {reply}")
    return reply


def file_drop(call: Call, drop_dir: Path, script: Path | None = None) -> Path:
    """Post-call mode: atomically place ``<call_id>.wav`` + ``.json`` in ``drop_dir/incoming``."""
    incoming = drop_dir / "incoming"
    incoming.mkdir(parents=True, exist_ok=True)
    meta = {
        "call_id": call.call_id,
        "layout": "stereo_agent_left",
        "sample_rate": call.sample_rate,
        "duration_s": round(call.duration_s, 3),
        "source": "simulator",
        "synthetic": True,
        "script": str(script) if script and script.exists() else None,
    }
    tmp = incoming / f"{call.call_id}.wav.part"
    shutil.copyfile(call.path, tmp)
    (incoming / f"{call.call_id}.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    final = incoming / f"{call.call_id}.wav"
    tmp.replace(final)  # the wav appears last, so a watcher never sees a partial file
    return final


async def run_rtp(
    client: httpx.AsyncClient,
    call: Call,
    *,
    rtp_host: str,
    codec: Codec,
    speed: float,
    drain_s: float = 0.5,
) -> tuple[dict[str, int], dict[str, Any]]:
    ssrc = ssrcs_for(call.call_id)
    r = await client.post(
        "/v1/rtp/sessions",
        json={
            "call_id": call.call_id,
            "codec": codec,
            "ssrc": ssrc,
            "external_ref": call.external_ref,
        },
    )
    r.raise_for_status()
    sent = await send_rtp(
        call, rtp_host, int(r.json()["rtp_port"]), ssrc=ssrc, codec=codec, speed=speed
    )
    await asyncio.sleep(drain_s)  # let the gateway drain its UDP buffer before hang-up
    r = await client.delete(f"/v1/rtp/sessions/{call.call_id}")
    r.raise_for_status()
    stats: dict[str, Any] = r.json()
    return sent, stats


def expected_crc(
    call: Call, transport: Literal["rtp", "ws"], codec: Codec = "pcmu"
) -> dict[str, int]:
    return {
        ch: crc32(
            g711_roundtrip(call.audio[:, i], codec) if transport == "rtp" else call.audio[:, i]
        )
        for i, ch in enumerate(CHANNELS)
    }

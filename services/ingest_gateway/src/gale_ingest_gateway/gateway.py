"""Session registry, RTP/UDP receiver and WebSocket PCM handling for the ingest-gateway.

P0 stub transport layer (Prompt 2): the simulator registers a call's SSRCs over HTTP, then
streams G.711 RTP to one UDP port. In P4 the Asterisk ARI / FreeSWITCH integrations register
sessions the same way (ExternalMedia gives the SSRC/port per channel).
"""

from __future__ import annotations

import asyncio
import socket
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from gale_ingest_gateway.sessions import LOST, PACKETS, REJECTED, FrameSink, Session, SessionStats
from galecore.audio import decode_alaw, decode_ulaw
from galecore.logging import get_logger
from galecore.rtp import PT_PCMA, PT_PCMU, TELEPHONY_SR, RtpError, parse
from galecore.schemas import Channel

log = get_logger(__name__)


class GatewaySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GALE_INGEST_", extra="ignore")

    rtp_host: str = "127.0.0.1"
    rtp_port: int = Field(default=40000, ge=0, le=65535)  # 0 = pick a free port (tests)
    max_sessions: int = Field(default=500, ge=1)
    udp_rcvbuf_bytes: int = 4 * 1024 * 1024


class RtpSessionRequest(BaseModel):
    call_id: str = Field(min_length=1, max_length=128)
    codec: Literal["pcmu", "pcma"] = "pcmu"
    ssrc: dict[Channel, int]  # one RTP stream per speaker channel


class WsStart(BaseModel):
    event: Literal["start"]
    sample_rate: Literal[8000, 16000] = 8000
    channels: Literal[1, 2] = 2
    encoding: Literal["pcm_s16le"] = "pcm_s16le"
    layout: Literal["stereo_agent_left", "stereo_agent_right", "mono"] = "stereo_agent_left"


class GatewayError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status, self.detail = status, detail


class Registry:
    def __init__(self, settings: GatewaySettings, sink: FrameSink):
        self.settings, self.sink = settings, sink
        self.sessions: dict[str, Session] = {}
        self.by_ssrc: dict[
            int, tuple[Session, Channel, int]
        ] = {}  # ssrc → session, channel, payload type
        self.rtp_port: int | None = None

    def _new(
        self, call_id: str, transport: Literal["rtp", "ws"], sr: int, channels: list[Channel]
    ) -> Session:
        if call_id in self.sessions and not self.sessions[call_id].closed:
            raise GatewayError(409, "session already open")
        if sum(not s.closed for s in self.sessions.values()) >= self.settings.max_sessions:
            raise GatewayError(503, "too many sessions")
        session = Session(call_id, transport, sr, channels, self.sink)
        self.sessions[call_id] = session
        return session

    def open_rtp(self, req: RtpSessionRequest) -> Session:
        if Channel.MIXED in req.ssrc and len(req.ssrc) > 1:
            raise GatewayError(422, "mixed channel cannot be combined with speaker channels")
        clash = [
            s for s in req.ssrc.values() if s in self.by_ssrc and not self.by_ssrc[s][0].closed
        ]
        if clash:
            raise GatewayError(409, "ssrc already in use")
        session = self._new(req.call_id, "rtp", TELEPHONY_SR, list(req.ssrc))
        pt = PT_PCMU if req.codec == "pcmu" else PT_PCMA
        for ch, ssrc in req.ssrc.items():
            self.by_ssrc[ssrc] = (session, ch, pt)
        log.info("rtp_session_open", call_id=req.call_id, channels=len(req.ssrc), codec=req.codec)
        return session

    def open_ws(self, call_id: str, start: WsStart) -> Session:
        channels = (
            [Channel.MIXED]
            if start.channels == 1 or start.layout == "mono"
            else [Channel.AGENT, Channel.CUSTOMER]
        )
        return self._new(call_id, "ws", start.sample_rate, channels)

    def close(self, call_id: str) -> SessionStats:
        session = self.get(call_id)
        stats = session.close()
        for ssrc in [k for k, v in self.by_ssrc.items() if v[0] is session]:
            del self.by_ssrc[ssrc]
        for ch, cs in stats.channels.items():
            if cs.lost:
                LOST.labels(ch.value).inc(cs.lost)
        log.info(
            "session_closed",
            call_id=call_id,
            transport=session.transport,
            frames={c.value: s.frames_out for c, s in stats.channels.items()},
        )
        return stats

    def get(self, call_id: str) -> Session:
        if call_id not in self.sessions:
            raise GatewayError(404, "unknown session")
        return self.sessions[call_id]

    # --- RTP ---------------------------------------------------------------------------------

    def on_datagram(self, data: bytes) -> None:
        try:
            pkt = parse(data)
        except RtpError:
            REJECTED.labels("malformed").inc()
            return
        entry = self.by_ssrc.get(pkt.ssrc)
        if entry is None:
            REJECTED.labels("unknown_ssrc").inc()
            return
        session, ch, pt = entry
        if pkt.payload_type != pt:
            REJECTED.labels("payload_type").inc()
            return
        asm = session.channels[ch]
        before = asm.seq.duplicates
        asm.seq.update(pkt.seq)
        if asm.seq.duplicates != before:
            return
        # P0: decode in arrival order (localhost, no jitter buffer yet; reordering lands in P4)
        asm.stats.packets += 1
        PACKETS.labels(ch.value).inc()
        asm.push(decode_ulaw(pkt.payload) if pt == PT_PCMU else decode_alaw(pkt.payload))

    # --- WebSocket ---------------------------------------------------------------------------

    @staticmethod
    def on_ws_bytes(session: Session, start: WsStart, data: bytes, carry: bytes) -> bytes:
        """Feed interleaved s16le PCM; return any incomplete sample frame to carry over."""
        width = 2 * len(session.channels)
        buf = carry + data
        usable = len(buf) - len(buf) % width
        pcm = np.frombuffer(buf[:usable], dtype="<i2").astype(np.int16)
        chans = list(session.channels.values())
        if len(chans) == 2:
            pcm2 = pcm.reshape(-1, 2)
            left, right = pcm2[:, 0], pcm2[:, 1]
            agent, customer = (
                (left, right) if start.layout != "stereo_agent_right" else (right, left)
            )
            session.channels[Channel.AGENT].push(agent)
            session.channels[Channel.CUSTOMER].push(customer)
        else:
            chans[0].push(pcm)
        for asm in chans:
            asm.stats.packets += 1
        return buf[usable:]


class _RtpProtocol(asyncio.DatagramProtocol):
    def __init__(self, registry: Registry):
        self.registry = registry

    def datagram_received(self, data: bytes, addr: Any) -> None:
        self.registry.on_datagram(data)


async def start_rtp(registry: Registry) -> asyncio.DatagramTransport:
    s = registry.settings
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, s.udp_rcvbuf_bytes)
    sock.bind((s.rtp_host, s.rtp_port))
    sock.setblocking(False)
    loop = asyncio.get_running_loop()
    transport, _ = await loop.create_datagram_endpoint(lambda: _RtpProtocol(registry), sock=sock)
    registry.rtp_port = sock.getsockname()[1]
    log.info("rtp_listening", host=s.rtp_host, port=registry.rtp_port)
    return transport

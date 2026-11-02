"""Session registry, RTP/UDP receiver and WebSocket PCM handling for the ingest-gateway.

Sessions are registered over HTTP (simulator), by the Asterisk ARI app (``ari.py``: RTP mapped
by the ExternalMedia source address) or over WebSocket (generic PCM, FreeSWITCH ``mod_audio_fork``).
``CallEvents`` (``publisher.BusPublisher`` in the service) publishes ``call.lifecycle`` on open and
close; frames go to the ``FrameSink``.
"""

from __future__ import annotations

import asyncio
import socket
from typing import Any, Literal, Protocol

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
    drop_dir: str | None = None  # post-call file mode: watch this folder for <name>.wav (+ .json)
    drop_poll_s: float = 2.0
    # Asterisk ARI (Prompt 11a): set ari_url to enable
    ari_url: str | None = None  # e.g. http://asterisk.internal:8088
    ari_user: str = "gale"
    ari_password: str = ""
    ari_app: str = "gale"
    ari_rtp_host: str = "127.0.0.1"  # the gateway address as Asterisk reaches it


class RtpSessionRequest(BaseModel):
    call_id: str = Field(min_length=1, max_length=128)
    codec: Literal["pcmu", "pcma"] = "pcmu"
    ssrc: dict[Channel, int] = Field(default_factory=dict)  # one RTP stream per speaker channel
    # alternative to SSRCs: map by source address "host:port" (Asterisk ExternalMedia)
    remote: dict[Channel, str] = Field(default_factory=dict)
    external_ref: str | None = Field(default=None, max_length=256)
    agent_id: str | None = Field(default=None, max_length=64)  # employee code (RBAC scoping)
    queue: str | None = Field(default=None, max_length=100)


class WsStart(BaseModel):
    event: Literal["start"]
    sample_rate: Literal[8000, 16000] = 8000
    channels: Literal[1, 2] = 2
    encoding: Literal["pcm_s16le"] = "pcm_s16le"
    layout: Literal["stereo_agent_left", "stereo_agent_right", "mono"] = "stereo_agent_left"
    external_ref: str | None = None
    agent_id: str | None = Field(default=None, max_length=64)
    queue: str | None = Field(default=None, max_length=100)


class CallEvents(Protocol):
    def started(
        self,
        call_id: str,
        external_ref: str | None,
        layout: str | None,
        *,
        agent_id: str | None = None,
        queue: str | None = None,
    ) -> None: ...

    def ended(self, call_id: str, duration_s: float) -> None: ...


class GatewayError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status, self.detail = status, detail


class Registry:
    def __init__(
        self, settings: GatewaySettings, sink: FrameSink, events: CallEvents | None = None
    ):
        self.settings, self.sink, self.events = settings, sink, events
        self.by_addr: dict[str, tuple[Session, Channel, int]] = {}  # "host:port" → …
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

    def _started(
        self,
        session: Session,
        external_ref: str | None,
        layout: str | None,
        agent_id: str | None = None,
        queue: str | None = None,
    ) -> None:
        if self.events is not None:
            self.events.started(
                session.call_id, external_ref, layout, agent_id=agent_id, queue=queue
            )

    def open_rtp(self, req: RtpSessionRequest) -> Session:
        streams = req.ssrc or req.remote
        if not streams or (req.ssrc and req.remote):
            raise GatewayError(422, "give either ssrc or remote per channel")
        if Channel.MIXED in streams and len(streams) > 1:
            raise GatewayError(422, "mixed channel cannot be combined with speaker channels")
        clash = [
            str(s) for s in req.ssrc.values() if s in self.by_ssrc and not self.by_ssrc[s][0].closed
        ]
        clash += [
            a for a in req.remote.values() if a in self.by_addr and not self.by_addr[a][0].closed
        ]
        if clash:
            raise GatewayError(409, "ssrc / remote address already in use")
        session = self._new(req.call_id, "rtp", TELEPHONY_SR, list(streams))
        pt = PT_PCMU if req.codec == "pcmu" else PT_PCMA
        for ch, ssrc in req.ssrc.items():
            self.by_ssrc[ssrc] = (session, ch, pt)
        for ch, addr in req.remote.items():
            self.by_addr[addr] = (session, ch, pt)
        log.info("rtp_session_open", call_id=req.call_id, channels=len(streams), codec=req.codec)
        self._started(
            session,
            req.external_ref,
            "mono" if Channel.MIXED in streams else "stereo_agent_left",
            req.agent_id,
            req.queue,
        )
        return session

    def open_ws(self, call_id: str, start: WsStart) -> Session:
        channels = (
            [Channel.MIXED]
            if start.channels == 1 or start.layout == "mono"
            else [Channel.AGENT, Channel.CUSTOMER]
        )
        session = self._new(call_id, "ws", start.sample_rate, channels)
        self._started(session, start.external_ref, start.layout, start.agent_id, start.queue)
        return session

    def close(self, call_id: str) -> SessionStats:
        session = self.get(call_id)
        was_open = not session.closed
        stats = session.close()
        for ssrc in [k for k, v in self.by_ssrc.items() if v[0] is session]:
            del self.by_ssrc[ssrc]
        for addr in [k for k, v in self.by_addr.items() if v[0] is session]:
            del self.by_addr[addr]
        if was_open and self.events is not None:
            self.events.ended(
                call_id, max((c.duration_s for c in stats.channels.values()), default=0.0)
            )
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

    def on_datagram(self, data: bytes, addr: tuple[str, int] | None = None) -> None:
        try:
            pkt = parse(data)
        except RtpError:
            REJECTED.labels("malformed").inc()
            return
        entry = self.by_ssrc.get(pkt.ssrc)
        if entry is None and addr is not None:
            entry = self.by_addr.get(f"{addr[0]}:{addr[1]}")
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
        self.registry.on_datagram(data, (str(addr[0]), int(addr[1])) if addr else None)


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

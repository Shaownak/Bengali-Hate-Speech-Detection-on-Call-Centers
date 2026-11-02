"""Asterisk ARI integration (Prompt 11a): passive per-speaker audio from live calls.

The dialplan marks the agent leg (``Set(__GALE_ROLE=agent)``, see docs/runbooks/pbx-integration.md).
This app subscribes to all channel events; when a marked channel enters a bridge it:

1. creates two **snoop** channels on the agent channel: ``spy=in`` (what the agent says) and
   ``spy=out`` (what the agent hears = the customer);
2. creates two **ExternalMedia** channels that send RTP (``ulaw``) to the gateway's RTP port;
3. bridges each snoop with its ExternalMedia channel;
4. registers a gateway session whose streams are keyed by the RTP **source address** Asterisk
   reports for each ExternalMedia channel (``UNICASTRTP_LOCAL_ADDRESS:PORT``).

When the agent channel leaves the bridge or hangs up, everything is torn down and the session
closes (``call.lifecycle: ended``). The call is never moved into Stasis: monitoring is passive and
a failure here never affects the call itself.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx

from gale_ingest_gateway.gateway import GatewayError, Registry, RtpSessionRequest
from galecore.logging import get_logger
from galecore.schemas import Channel

log = get_logger(__name__)
ROLE_VAR = "GALE_ROLE"


@dataclass
class Monitor:
    call_id: str
    agent_channel: str
    created: list[str] = field(default_factory=list)  # channel ids to hang up
    bridges: list[str] = field(default_factory=list)


class AriApp:
    def __init__(
        self,
        registry: Registry,
        base_url: str,
        user: str,
        password: str,
        *,
        app: str = "gale",
        rtp_host: str = "127.0.0.1",
        rtp_port: int | None = None,
        fmt: str = "ulaw",
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.registry, self.app, self.rtp_host, self.fmt = registry, app, rtp_host, fmt
        self.rtp_port = rtp_port
        self.base_url = base_url.rstrip("/")
        self.http = httpx.AsyncClient(
            base_url=f"{self.base_url}/ari",
            auth=(user, password),
            transport=transport,
            timeout=10.0,
        )
        self.auth = (user, password)
        self.monitors: dict[str, Monitor] = {}  # agent channel id → monitor

    async def _post(self, path: str, **params: Any) -> dict[str, Any]:
        r = await self.http.post(path, params={k: v for k, v in params.items() if v is not None})
        r.raise_for_status()
        return r.json() if r.content else {}

    async def _role(self, channel_id: str) -> str | None:
        r = await self.http.get(f"/channels/{channel_id}/variable", params={"variable": ROLE_VAR})
        return r.json().get("value") if r.status_code == 200 else None

    async def on_event(self, evt: dict[str, Any]) -> None:
        kind = evt.get("type")
        ch = evt.get("channel") or {}
        cid = ch.get("id")
        if (
            kind == "ChannelEnteredBridge"
            and cid
            and cid not in self.monitors
            and not ch.get("name", "").startswith(("Snoop/", "UnicastRTP/"))
        ):
            if (await self._role(cid)) == "agent":
                await self.start(ch)
        elif (
            kind in ("ChannelLeftBridge", "ChannelDestroyed", "StasisEnd") and cid in self.monitors
        ):
            await self.stop(cid)

    async def start(self, ch: dict[str, Any]) -> None:
        cid = ch["id"]
        mon = Monitor(call_id=str(uuid.uuid4()), agent_channel=cid)  # D-18
        self.monitors[cid] = mon
        port = self.rtp_port or self.registry.rtp_port
        remotes: dict[Channel, str] = {}
        try:
            for spy, speaker in (("in", Channel.AGENT), ("out", Channel.CUSTOMER)):
                snoop = await self._post(
                    f"/channels/{cid}/snoop",
                    spy=spy,
                    whisper="none",
                    app=self.app,
                    snoopId=f"gale-{mon.call_id[:8]}-{spy}",
                )
                em = await self._post(
                    "/channels/externalMedia",
                    app=self.app,
                    external_host=f"{self.rtp_host}:{port}",
                    format=self.fmt,
                    encapsulation="rtp",
                    transport="udp",
                    direction="both",
                    channelId=f"gale-{mon.call_id[:8]}-{spy}-em",
                )
                mon.created += [snoop["id"], em["id"]]
                bridge = await self._post(
                    "/bridges", type="mixing", name=f"gale-{mon.call_id[:8]}-{spy}"
                )
                mon.bridges.append(bridge["id"])
                await self._post(
                    f"/bridges/{bridge['id']}/addChannel", channel=f"{snoop['id']},{em['id']}"
                )
                vars_ = em.get("channelvars", {})
                remotes[speaker] = (
                    f"{vars_.get('UNICASTRTP_LOCAL_ADDRESS')}:{vars_.get('UNICASTRTP_LOCAL_PORT')}"
                )
            ref = f"asterisk:{ch.get('linkedid') or ch.get('id')}"
            self.registry.open_rtp(
                RtpSessionRequest(
                    call_id=mon.call_id,
                    codec="pcmu" if self.fmt == "ulaw" else "pcma",
                    remote=remotes,
                    external_ref=ref,
                )
            )
            log.info("ari_monitor_started", call_id=mon.call_id)
        except (httpx.HTTPError, KeyError, GatewayError) as e:
            log.warning("ari_monitor_failed", call_id=mon.call_id, error=type(e).__name__)
            await self.stop(cid)

    async def stop(self, agent_channel: str) -> None:
        mon = self.monitors.pop(agent_channel, None)
        if mon is None:
            return
        for chan in mon.created:
            await self.http.delete(f"/channels/{chan}")
        for b in mon.bridges:
            await self.http.delete(f"/bridges/{b}")
        if mon.call_id in self.registry.sessions and not self.registry.sessions[mon.call_id].closed:
            self.registry.close(mon.call_id)
        log.info("ari_monitor_stopped", call_id=mon.call_id)

    async def run(self, stop: asyncio.Event) -> None:  # pragma: no cover - needs an Asterisk
        import websockets

        ws_url = (
            self.base_url.replace("http", "ws", 1)
            + f"/ari/events?app={self.app}&subscribeAll=true&api_key={self.auth[0]}:{self.auth[1]}"
        )
        backoff = 1.0
        while not stop.is_set():
            try:
                async with websockets.connect(ws_url) as ws:
                    backoff = 1.0
                    while not stop.is_set():
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                        except TimeoutError:
                            continue
                        await self.on_event(json.loads(msg))
            except (OSError, websockets.WebSocketException):
                log.warning("ari_disconnected", retry_s=backoff)
                await asyncio.sleep(backoff)
                backoff = min(30.0, backoff * 2)

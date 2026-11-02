# Runbook: PBX integration (for the PBX / telephony team)

GAL-E listens passively. It never answers, routes, records for the business, or changes a call. It needs
the agent's voice and the customer's voice as two separate audio streams, plus a call identifier. Three
options; choose the one that matches your platform.

The **ingest-gateway** is the only thing the PBX talks to. Below:
- `GW_HOST` = the gateway address as the PBX reaches it;
- RTP port `40000/udp` (`GALE_INGEST_RTP_PORT`);
- HTTP/WS port `8001`.

## Option A — Asterisk (ARI ExternalMedia + snoop) — preferred for Asterisk ≥ 16.6

How it works: the gateway runs an ARI application `gale`. When an agent channel enters a bridge, the app:
1. creates two snoop channels on it (`spy=in` = agent voice, `spy=out` = customer voice);
2. creates two ExternalMedia channels that send µ-law RTP to `GW_HOST:40000`;
3. bridges each pair.

The call itself is never put into Stasis.

**1. `ari.conf`** — a user for GAL-E (read/write needed for snoop/externalMedia):
```ini
[general]
enabled = yes
[gale]
type = user
read_only = no
password = <secret, give to the GAL-E admin out of band>
```
**2. `http.conf`** — ARI over HTTP(S) reachable from the gateway only (firewall to GW_HOST):
```ini
[general]
enabled = yes
bindaddr = 0.0.0.0
bindport = 8088
; tlsenable = yes  (recommended; then use https/wss in GALE_INGEST_ARI_URL)
```
**3. Dialplan** — mark the agent leg so the app knows which side is the agent. The `__` prefix makes
the variable inherit to the agent channel created by `Dial()` / `Queue()`:
```ini
[from-customers]
exten => _X.,1,NoOp(GAL-E: inbound call)
 same => n,Set(__GALE_ROLE=agent)          ; inherited by the agent leg
 same => n,Set(GALE_ROLE=customer)          ; this (customer) channel
 same => n,Queue(support)                   ; or Dial(PJSIP/${AGENT})
```
Optional, but needed for supervisor scoping in the dashboard: also set the agent's employee code
and the queue on the agent leg, e.g. in the queue member's subroutine:
`Set(GALE_AGENT_ID=${MEMBERNAME})` and `Set(GALE_QUEUE=${QUEUENAME})`. Calls without an agent code are
visible to admins and QA only.
For queue members (`Local/` or `PJSIP/agent-*`) you can also set it on the member channel:
`Queue(support,,,,,,,gale-mark)` with a subroutine `[gale-mark] exten => s,1,Set(GALE_ROLE=agent)`.

**4. Network**: Asterisk → `GW_HOST:40000/udp` (RTP); gateway → Asterisk `8088/tcp` (ARI). ExternalMedia
sends from Asterisk's RTP range (`rtp.conf rtpstart–rtpend`), so allow that range → GW_HOST:40000.

**5. Gateway settings** (GAL-E side):
```
GALE_INGEST_ARI_URL=http://asterisk.internal:8088
GALE_INGEST_ARI_USER=gale
GALE_INGEST_ARI_PASSWORD=<secret>
GALE_INGEST_ARI_RTP_HOST=<GW_HOST as Asterisk reaches it>
```

Check:
- `asterisk -rx "ari show apps"` lists `gale`.
- During a test call, `core show channels` shows `Snoop/…` and `UnicastRTP/…` channels.
- The gateway's `/metrics` shows `gale_ingest_rtp_packets_total` rising.

## Option B — FreeSWITCH (`mod_audio_fork`)

The fork streams L16 audio over WebSocket to `ws://GW_HOST:8001/v1/fork`. The first text frame is
GAL-E metadata.
```xml
<!-- in the dialplan of the agent leg, after answer -->
<action application="set" data="api_on_answer=uuid_audio_fork ${uuid} start ws://GW_HOST:8001/v1/fork stereo 16k {&quot;fs_uuid&quot;:&quot;${uuid}&quot;,&quot;agent_id&quot;:&quot;${agent_code}&quot;,&quot;agent_channel&quot;:&quot;left&quot;,&quot;sample_rate&quot;:16000,&quot;channels&quot;:2}"/>
```
- `stereo` puts the leg's read audio on one side and its write audio on the other. On the **agent** leg, read = agent and write = customer, so `"agent_channel":"left"`. If you fork the customer leg instead, use `"right"`.
- Build/load `mod_audio_fork` (drachtio `freeswitch-modules`); allow FreeSWITCH → `GW_HOST:8001/tcp`.
- The gateway answers `{"event":"accepted","call_id":…}`; the fork stops on hang-up (`uuid_audio_fork <uuid> stop`).

## Option C — any platform: generic WebSocket PCM or SIPREC → recorder

- **WebSocket**: `ws://GW_HOST:8001/v1/stream/<call_id>`. Send the JSON `{"event":"start","sample_rate":8000|16000,"channels":2,"layout":"stereo_agent_left","external_ref":"<your call id>","agent_id":"<employee code>","queue":"<queue>"}`, then binary interleaved s16le, then `{"event":"stop"}`.
- **Post-call only (no live alerts)**: drop stereo recordings into the gateway's watched folder (`GALE_INGEST_DROP_DIR`): `<name>.wav` plus an optional `<name>.json` (`{"external_ref":…,"queue":…}`). Write the WAV last or rename it into place. Each file becomes a `postcall.jobs` entry.

## What the PBX team must configure (checklist)

| # | Item | Asterisk | FreeSWITCH | Other |
|---|---|---|---|---|
| 1 | Agent-leg marking | `Set(__GALE_ROLE=agent)` in dialplan | fork on the agent leg | `layout` in the start message |
| 2 | API / stream access | ARI user `gale` (read/write), `http.conf` | `mod_audio_fork` loaded | WebSocket client or recorder export |
| 3 | Network | Asterisk RTP range → GW_HOST:40000/udp; GW → Asterisk:8088/tcp | FS → GW_HOST:8001/tcp | → GW_HOST:8001/tcp, or a shared folder |
| 4 | Codec | µ-law (default) or A-law (`GALE_INGEST` codec) | L16 16 kHz | s16le 8/16 kHz |
| 5 | A staging PBX + 2 test extensions (agent, customer) | ✓ | ✓ | ✓ |
| 6a | Agent employee code (dashboard scoping) | `GALE_AGENT_ID`, `GALE_QUEUE` channel vars | `agent_id` in metadata | `agent_id` in the start message |
| 6 | Call identifier to correlate with CDRs | `linkedid` (sent automatically) | `${uuid}` in metadata | `external_ref` |
| 7 | Legal: monitoring notice in the IVR greeting (PDPO 2025) | ✓ | ✓ | ✓ |

Load: each monitored call costs Asterisk two snoop + two ExternalMedia channels (about 2 × 64 kbit/s of
µ-law RTP to the gateway). Size the gateway at roughly 500 concurrent calls per instance (CPU-bound
decode + resample); scale horizontally with one ARI app per instance.

# Runbook: PBX stream drop (`GalePBXStreamsStopped`, `GaleIngestPacketLoss`)

**Impact:** live calls are not analysed. Post-call analysis still works if recordings are exported
(drop folder / `POST /api/v1/ingest/recordings`).

1. **Is it us or the PBX?** `kubectl -n gale-ingest logs deploy/gale-ingest-gateway --tail=200`
   (ids and counts only), and Grafana Ops → "RTP ingest".
2. **Asterisk (ARI):** `asterisk -rx "ari show apps"` must list `gale`. The gateway reconnects with
   backoff; if the ARI password rotated, see key-rotation.md. `core show channels` should show
   `Snoop/` and `UnicastRTP/` channels during calls.
3. **Network:** RTP is UDP from the PBX media range to the gateway's `-rtp` LoadBalancer on 40000.
   Check `loadBalancerSourceRanges`, the `gateway-pbx` NetworkPolicy (`pbxCidrs`) and any firewall
   change on the PBX side.
4. **Packet loss > 2 %:** usually network congestion on the PBX link; G.711 tolerates a little, but
   ASR quality drops. Ask the network team; it is not fixable in GAL-E.
5. One gateway serves one PBX (in-memory RTP/ARI sessions, ADR-018). A gateway restart ends the
   streams of calls in progress; new calls are picked up automatically.

Setup reference: [pbx-integration.md](pbx-integration.md).

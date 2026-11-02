"""Call simulator: replay a synthetic stereo call into the ingest-gateway as a "live" call.

    python -m tools.simulator tests/fixtures/calls/demo_01             # RTP G.711 µ-law, real time
    python -m tools.simulator tests/fixtures/calls/demo_01 --speed 20      # 20× faster
    python -m tools.simulator tests/fixtures/calls/demo_01 --mode ws       # WebSocket PCM
    python -m tools.simulator tests/fixtures/calls/demo_01 --mode file     # post-call file drop

If no gateway answers at --gateway, one is started in-process (stub mode) unless --spawn never.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from galecore.config import REPO_ROOT
from tools.simulator.core import (
    CHANNELS,
    Call,
    expected_crc,
    file_drop,
    load_call,
    run_rtp,
    send_ws,
)


@contextmanager
def gateway(url: str, spawn: str) -> Iterator[str]:
    if spawn != "always":
        try:
            httpx.get(f"{url}/healthz", timeout=1.0).raise_for_status()
            yield url
            return
        except httpx.HTTPError:
            if spawn == "never":
                raise SystemExit(
                    f"ingest-gateway not reachable at {url} (run `make dev-up`)"
                ) from None
    import uvicorn

    os.environ.setdefault("GALE_BUS_BACKEND", "memory")
    os.environ.setdefault("GALE_LOG_LEVEL", "WARNING")
    from gale_ingest_gateway.app import build_app

    u = urlparse(url)
    server = uvicorn.Server(
        uvicorn.Config(
            build_app(),
            host=u.hostname or "127.0.0.1",
            port=u.port or 8001,
            log_config=None,
            access_log=False,
            lifespan="on",
        )
    )
    thread = threading.Thread(target=server.run, name="ingest-gateway", daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started:
        if not thread.is_alive() or time.monotonic() > deadline:
            raise SystemExit("could not start an in-process ingest-gateway")
        time.sleep(0.05)
    print(f"(started an in-process ingest-gateway on {url})")
    try:
        yield url
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def report(
    call: Call, transport: str, sent: dict[str, int] | None, stats: dict[str, Any], codec: str
) -> bool:
    exp = expected_crc(call, "rtp" if transport == "rtp" else "ws", codec)  # type: ignore[arg-type]
    unit = "packets" if transport == "rtp" else "chunks"
    print(
        f"\ningest-gateway received ({transport}{', ' + codec if transport == 'rtp' else ''}), "
        f"call {stats['call_id']}, {call.duration_s:.1f} s of audio:"
    )
    print(
        f"  {'channel':<9} {'sent':>7} {unit:>9} {'lost':>5} "
        f"{'frames (200 ms @16 kHz)':>24} {'audio':>8}  integrity"
    )
    ok = True
    for ch in CHANNELS:
        cs = stats["channels"][ch]
        good = cs["pcm_crc32"] == exp[ch] and cs["lost"] == 0
        ok &= good
        s = str(sent[ch]) if sent else "-"
        print(
            f"  {ch:<9} {s:>7} {cs['packets']:>9} {cs['lost']:>5} {cs['frames_out']:>24} "
            f"{cs['duration_s']:>7.1f}s  {'OK' if good else 'MISMATCH'}"
        )
    return ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m tools.simulator",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "call", type=Path, help="fixture dir (tests/fixtures/calls/<name>) or a stereo WAV"
    )
    ap.add_argument("--mode", choices=["rtp", "ws", "file"], default="rtp")
    ap.add_argument(
        "--speed", type=float, default=1.0, help="1 = real time, 10 = 10× faster, 0 = unpaced"
    )
    ap.add_argument("--codec", choices=["pcmu", "pcma"], default="pcmu")
    ap.add_argument("--gateway", default="http://127.0.0.1:8001", help="ingest-gateway base URL")
    ap.add_argument("--rtp-host", default="127.0.0.1")
    ap.add_argument("--spawn", choices=["auto", "never", "always"], default="auto")
    ap.add_argument("--call-id")
    ap.add_argument("--drop-dir", type=Path, default=REPO_ROOT / ".gale" / "drop")
    a = ap.parse_args(argv)

    if a.call.is_dir() and not (a.call / "call.wav").exists():
        from tools.fixtures_gen.__main__ import ensure_wav

        print(f"rendering {a.call / 'call.wav'} from script.json …")
        ensure_wav(a.call)
    call = load_call(a.call, a.call_id)
    print(
        f"simulating {call.path} as {call.call_id}: {call.duration_s:.1f} s, "
        f"mode={a.mode}, speed={a.speed:g}"
    )

    if a.mode == "file":
        path = file_drop(call, a.drop_dir, a.call / "script.json" if a.call.is_dir() else None)
        print(f"dropped {path} (+ .json sidecar) for post-call processing")
        return 0

    with gateway(a.gateway.rstrip("/"), a.spawn) as url:
        if a.mode == "rtp":

            async def go_rtp() -> tuple[dict[str, int], dict[str, Any]]:
                async with httpx.AsyncClient(base_url=url, timeout=10) as client:
                    return await run_rtp(
                        client, call, rtp_host=a.rtp_host, codec=a.codec, speed=a.speed
                    )

            sent, stats = asyncio.run(go_rtp())
            ok = report(call, "rtp", sent, stats, a.codec)
        else:
            ws_url = url.replace("http", "ws", 1) + f"/v1/stream/{call.call_id}"
            stats = asyncio.run(send_ws(call, ws_url, speed=a.speed))
            ok = report(call, "ws", None, stats, a.codec)
    print("round trip OK" if ok else "round trip FAILED (loss or corrupted audio)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

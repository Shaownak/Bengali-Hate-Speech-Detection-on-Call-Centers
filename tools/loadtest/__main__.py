"""Load test (docs/05 §3): replay synthetic fixture calls concurrently at 25/50/100/200 % of the
expected peak through the gateway, scrape every service's /metrics, and write a capacity report.

    make dev-up                         # or point --host at staging
    python -m tools.loadtest --peak 4 --levels 25,50,100,200 --speed 1
    python -m tools.loadtest --peak 300 --host staging-gw.gale.internal --speed 1   # staging

Pass criteria per level (docs/01 §5, docs/05 §8): provisional p95 ≤ 2 s, confirmed p95 ≤ 6 s,
no frames dropped, no dead letters, bus lag < 10 s. Stub-model runs prove the pipeline, not GPU
capacity: GPU numbers in the report stay "TBD" until measured on the staging GPU pool.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import math
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx

from galecore.service import SERVICE_PORTS
from tools.fixtures_gen.__main__ import CALLS_DIR, ensure_wav
from tools.loadtest.metrics import Snapshot, parse, window
from tools.simulator.core import load_call, send_ws

ROOT = Path(__file__).resolve().parents[2]
BUDGET = {"provisional_event": 2.0, "confirmed_event": 6.0, "alert": 8.0}
METRIC_SERVICES = [s for s in SERVICE_PORTS if s != "postcall_worker"]


@dataclass
class LevelResult:
    level_pct: int
    calls: int
    wall_s: float
    call_minutes: float
    p95: dict[str, float | None] = field(default_factory=dict)
    samples: dict[str, int] = field(default_factory=dict)
    frames_dropped: float = 0.0
    dead_letters: float = 0.0
    max_bus_lag_s: float = 0.0
    errors: int = 0
    passed: bool = False


async def scrape(client: httpx.AsyncClient, host: str) -> Snapshot:
    snap = Snapshot()
    for svc in METRIC_SERVICES:
        try:
            r = await client.get(f"http://{host}:{SERVICE_PORTS[svc]}/metrics", timeout=5)
            parse(r.text, snap)
        except httpx.HTTPError:
            continue
    return snap


async def watch_lag(
    client: httpx.AsyncClient, host: str, stop: asyncio.Event, out: list[float]
) -> None:
    while not stop.is_set():
        out.append((await scrape(client, host)).gauges_max.get("bus_lag_s", 0.0))
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), 2.0)


async def run_level(
    client: httpx.AsyncClient,
    host: str,
    gw: str,
    pct: int,
    peak: int,
    fixtures: list[str],
    speed: float,
) -> LevelResult:
    n = max(1, math.ceil(peak * pct / 100))
    calls = []
    for i in range(n):
        name = fixtures[i % len(fixtures)]
        c = load_call(ensure_wav(CALLS_DIR / name), str(uuid.uuid4()))
        c.external_ref = f"fixture:{CALLS_DIR / name}"
        c.agent_id, c.queue = f"AG-00{1 + i % 3}", "loadtest"
        calls.append(c)
    before = await scrape(client, host)
    stop, lags = asyncio.Event(), []
    watcher = asyncio.create_task(watch_lag(client, host, stop, lags))
    t0 = time.monotonic()
    results = await asyncio.gather(
        *(send_ws(c, f"ws://{gw}/v1/stream/{c.call_id}", speed=speed) for c in calls),
        return_exceptions=True,
    )
    await asyncio.sleep(8.0)  # let the LLM debounce / confirmations land
    stop.set()
    await watcher
    after = await scrape(client, host)
    res = LevelResult(
        level_pct=pct,
        calls=n,
        wall_s=round(time.monotonic() - t0, 1),
        call_minutes=round(sum(c.duration_s for c in calls) / 60, 2),
        errors=sum(isinstance(r, Exception) for r in results),
        max_bus_lag_s=round(max(lags, default=0.0), 2),
        frames_dropped=after.counters["gale_ingest_frames_dropped"]
        - before.counters["gale_ingest_frames_dropped"],
        dead_letters=after.counters["gale_bus_dead_lettered"]
        - before.counters["gale_bus_dead_lettered"],
    )
    for (svc, stage), (p, cnt) in sorted(window(before, after).items()):
        res.p95[f"{svc}/{stage}"] = round(p, 3) if p is not None else None
        res.samples[f"{svc}/{stage}"] = cnt
    res.passed = (
        res.errors == 0
        and res.frames_dropped == 0
        and res.dead_letters == 0
        and res.max_bus_lag_s < 10
        and (
            speed != 1
            or all(
                (p is None or p <= BUDGET[k.split("/")[1]])
                for k, p in res.p95.items()
                if k.split("/")[1] in BUDGET
            )
        )
    )
    return res


def render_report(results: list[LevelResult], meta: dict[str, Any]) -> str:
    stages = sorted({k for r in results for k in r.p95})
    lines = [
        "# GAL-E capacity report",
        "",
        f"Generated {meta['generated']} · peak = {meta['peak']} concurrent calls · speed {meta['speed']}× · "
        f"models: **{meta['model_mode']}** · target: `{meta['host']}`",
        "",
        "> Stub-model runs exercise the pipeline (bus, workers, fusion, persistence) only. Fill the GPU",
        "> rows below from a staging run with real models before sizing production (docs/05 §3).",
        *(
            [
                "",
                f"> **Speed {meta['speed']}×:** stage latencies are measured against real time and are",
                "> not meaningful when playback is faster; only lag / drops / errors count here.",
            ]
            if meta["speed"] != 1
            else []
        ),
        "",
        "## Results per level",
        "",
        "| level | calls | call-min | errors | frames dropped | dead letters | max bus lag (s) | pass |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.level_pct}% | {r.calls} | {r.call_minutes} | {r.errors} | {r.frames_dropped:g} | "
            f"{r.dead_letters:g} | {r.max_bus_lag_s} | {'✅' if r.passed else '❌'} |"
        )
    lines += [
        "",
        "## p95 latency per stage (s)",
        "",
        "| stage | " + " | ".join(f"{r.level_pct}%" for r in results) + " | budget |",
        "|---|" + "---|" * (len(results) + 1),
    ]
    for st in stages:
        budget = BUDGET.get(st.split("/")[1])
        cells = [str(r.p95.get(st, "—")) for r in results]
        lines.append(f"| {st} | " + " | ".join(cells) + f" | {budget if budget else ''} |")
    lines += [
        "",
        "## Sizing (docs/05 §3)",
        "",
        "```",
        f"channels = 2 × peak_concurrent_calls = 2 × {meta['peak']} = {2 * meta['peak']}",
        "GPU_ASR  = channels × RTF_asr / streams_per_gpu_at_latency_target  = TBD (measure RTF_asr on L4)",
        "GPU_SER  = channels / windows_per_sec_capacity                    = TBD",
        "LLM_rps  = active_calls × (utterances_per_min / 60) × 0.3         = TBD (utterances/min from staging)",
        "GPU_LLM  = LLM_rps / measured_rps_per_gpu_at_p95≤2.5s             = TBD",
        "cost per 1,000 call-minutes                                       = TBD",
        "```",
        "",
        "Worker shards (Helm `services.*.shards`): set so the 200 % level passes with bus lag < 10 s.",
    ]
    return "\n".join(lines) + "\n"


async def amain(a: argparse.Namespace) -> tuple[list[LevelResult], dict[str, Any]]:
    levels = [int(x) for x in a.levels.split(",")]
    gw = f"{a.host}:{a.gateway_port}"
    results = []
    async with httpx.AsyncClient() as client:
        for pct in levels:
            r = await run_level(client, a.host, gw, pct, a.peak, a.fixtures.split(","), a.speed)
            print(f"{pct:>4}%: {r.calls} calls, pass={r.passed}, p95={r.p95}", flush=True)
            results.append(r)
    meta = {
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "peak": a.peak,
        "speed": a.speed,
        "model_mode": a.model_mode,
        "host": a.host,
    }
    return results, meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--peak", type=int, default=4, help="expected peak concurrent calls (100 %%)")
    ap.add_argument("--levels", default="25,50,100,200")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed (1 = real time)")
    ap.add_argument("--fixtures", default="demo_01,demo_03,demo_04,demo_09")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--gateway-port", type=int, default=SERVICE_PORTS["ingest_gateway"])
    ap.add_argument("--model-mode", default="stub")
    ap.add_argument("--out", default=str(ROOT / ".gale" / "reports" / "loadtest"))
    a = ap.parse_args(argv)
    results, meta = asyncio.run(amain(a))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "loadtest.json").write_text(
        json.dumps({"meta": meta, "levels": [asdict(r) for r in results]}, indent=2) + "\n",
        encoding="utf-8",
    )
    (out / "capacity-report.md").write_text(render_report(results, meta), encoding="utf-8")
    print(f"wrote {out / 'capacity-report.md'}")
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

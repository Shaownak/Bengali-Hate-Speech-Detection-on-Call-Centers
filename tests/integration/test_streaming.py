"""Streaming stack end to end (Prompt 11): gateway → bus → asr/audio/text/llm/fusion workers, all
in-process on one fake Redis, stub models keyed to synthetic fixture calls.

- real-time pacing on a short synthetic call → events within the docs/01 §5 latency budget;
- unpaced fixtures (two calls at once) → events match ``expected_events.json``;
- chaos: text-worker and fusion-engine crash mid-call and restart → no event lost;
- post-call jobs wait while the real-time workers have a backlog;
- two replicas of every real-time worker (call-affine sharding, ADR-018) → same events, no
  call split across replicas.
"""

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from galecore.bus import ASR_SEGMENTS, POSTCALL_JOBS, audio_stream, call_shard
from galecore.schemas import AsrSegment, AudioFrame, EventStatus, PostcallJob, Speaker
from tests.integration.streaming_harness import (
    SETTINGS,
    Stack,
    gold,
    matches,
    p95,
    play_call,
    wait_quiet,
)
from tools.fixtures_gen.__main__ import CALLS_DIR, ensure_wav, write_call
from tools.fixtures_gen.compose import compose
from tools.fixtures_gen.scenarios import A, C, Scenario
from tools.fixtures_gen.tts import StubTTS

WORKERS = ("asr", "audio", "text", "llm", "fusion")


async def _stack(**kw: int) -> Stack:
    st = Stack(**kw)
    for n in WORKERS:
        await st.start(n)
    return st


def _short_call(tmp: Path) -> Path:
    """~25 s: customer slang, agent register drops to তুমি then তুই."""
    sc = Scenario(
        "rt_short",
        "real-time latency",
        "short synthetic call",
        seed=211,
        agent_f0=120,
        customer_f0=205,
        lines=[
            A("আসসালামু আলাইকুম, গ্রাহক সেবায় স্বাগতম। আমি কীভাবে আপনাকে সাহায্য করতে পারি?", register="apni"),
            C("আমার ইন্টারনেট দুই দিন ধরে কাজ করছে না।"),
            A("জি, আপনি একটু অপেক্ষা করুন।", register="apni"),
            C("এই সার্ভিসটা পুরাই <GALI_4>, কিছুই ঠিক নেই।"),
            A("তুমি এত কথা বলছ কেন?", register="tumi"),
            C("এভাবে কথা বলছেন কেন?"),
            A("তুই ফোন রাখ।", register="tui"),
            C("জি, ঠিক আছে।"),
        ],
    )
    script, events = compose(sc, StubTTS())
    write_call(tmp / "rt_short", script, events, StubTTS())
    return tmp / "rt_short"


async def test_realtime_latency_budget(tmp_path: Path) -> None:
    d = _short_call(tmp_path)
    st = await _stack()
    cid = str(uuid.uuid4())
    await play_call(st, d / "call.wav", cid, f"fixture:{d}", speed=1.0)
    await wait_quiet(st, 30, settle=4.0)
    events = await st.events()
    await st.shutdown()
    want = [g for g in gold(d) if g["alert"]]
    assert {(g["category"], g["speaker"]) for g in want} == {
        ("slang_profanity", "customer"),
        ("disrespect_register", "agent"),
    }
    for g in want:
        assert any(matches(e, g) for e in events.values()), g
    # docs/01 §5 budgets per evidence type: lexicon hits count from the word (partial ASR
    # segments make that possible); utterance-level steps (register, classifier) count from the
    # end of the utterance ("ASR final 1.2 s after end of utterance" + "100 ms after segment").
    script = json.loads((d / "script.json").read_text(encoding="utf-8"))

    def ref_end(e: Any) -> float:
        if e.category.value == "disrespect_register":
            turn = next(
                t
                for t in script["turns"]
                if t["t_start"] <= e.ts_start <= t["t_end"] and t["speaker"] == e.speaker.value
            )
            return float(turn["t_end"])
        return float(e.ts_end)

    lat = st.workers["fusion"].latencies  # type: ignore[attr-defined]
    prov = [
        x[3] - (ref_end(events[x[1]]) - events[x[1]].ts_end)
        for x in lat
        if x[2] == "provisional_event"
    ]
    conf = [
        x[3] - (ref_end(events[x[1]]) - events[x[1]].ts_end)
        for x in lat
        if x[2] == "confirmed_event"
    ]
    assert prov
    assert conf
    assert p95(prov) <= 2.0, prov  # fast-path provisional alert ≤ 2 s
    assert p95(conf) <= 6.0, conf  # confirmed (after the LLM, 3 s debounce) ≤ 6 s
    assert min(prov) >= 0


async def test_unpaced_fixtures_match_ground_truth() -> None:
    st = await _stack()
    calls = {name: str(uuid.uuid4()) for name in ("demo_04", "demo_09")}
    await asyncio.gather(
        *(
            play_call(st, ensure_wav(CALLS_DIR / n), cid, f"fixture:{n}", speed=0)
            for n, cid in calls.items()
        )
    )
    for n in ("demo_01", "demo_03"):
        calls[n] = str(uuid.uuid4())
        await play_call(st, ensure_wav(CALLS_DIR / n), calls[n], f"fixture:{n}", speed=0)
    await wait_quiet(st, 120, settle=5.0)
    events = await st.events()
    scores = await st.scores()
    await st.shutdown()
    for name, cid in calls.items():
        mine = [e for e in events.values() if e.call_id == cid]
        for g in gold(CALLS_DIR / name):
            hit = [e for e in mine if matches(e, g)]
            assert hit, (name, g)
            dismissed = all(e.status == EventStatus.DISMISSED_BY_LLM for e in hit)
            assert dismissed == (not g["alert"]), (name, g, [e.status for e in hit])
        live = [e for e in mine if e.status != EventStatus.DISMISSED_BY_LLM]
        assert len(live) == sum(g["alert"] for g in gold(CALLS_DIR / name)), (
            name,
            [(e.category, e.ts_start) for e in live],
        )
    assert {s.call_id for s in scores} == set(calls.values())
    agent_card = next(s for s in scores if s.call_id == calls["demo_03"]).scorecard
    assert agent_card.agent.respectful_register_maintained is False


def _owned_by(replica: int, n: int = 2) -> str:
    while True:
        cid = str(uuid.uuid4())
        if call_shard(cid) % n == replica:
            return cid


async def test_two_replicas_per_worker_shard_calls_without_splitting_them() -> None:
    st = Stack()
    for n in WORKERS:
        for i in range(2):
            await st.start(n, shard=(i, 2))
    calls = {"demo_04": _owned_by(0), "demo_09": _owned_by(1), "demo_03": _owned_by(1)}
    await asyncio.gather(
        *(
            play_call(st, ensure_wav(CALLS_DIR / n), cid, f"fixture:{n}", speed=0)
            for n, cid in calls.items()
        )
    )
    await wait_quiet(st, 120, settle=5.0)
    events = await st.events()
    segments = await st.read(ASR_SEGMENTS, AsrSegment)
    scores = await st.scores()
    seen = {k: set(w.clock.started) for k, w in st.workers.items() if k.startswith("asr#")}
    await st.shutdown()
    for name, cid in calls.items():
        mine = [e for e in events.values() if e.call_id == cid]
        for g in gold(CALLS_DIR / name):
            assert any(matches(e, g) for e in mine), (name, g)
        finals = [g.segment_id for g in segments if g.call_id == cid and g.is_final]
        assert finals
        assert len(finals) == len(set(finals)), "a call's ASR ran on two replicas"
    assert {s.call_id for s in scores} == set(calls.values())
    # each replica tracked exactly the calls it owns
    assert seen["asr#0"] == {calls["demo_04"]}
    assert seen["asr#1"] == {calls["demo_09"], calls["demo_03"]}


async def test_chaos_workers_crash_mid_call_lose_no_events() -> None:
    st = await _stack(min_idle_ms=300)
    name, cid = "demo_03", str(uuid.uuid4())
    state = {"killed": False}

    async def chaos(progress: float) -> None:
        if progress > 0.5 and not state["killed"]:  # register drops come at ~56–60 % of the call
            state["killed"] = True
            await st.kill("text")
            await st.kill("fusion")
            await asyncio.sleep(0.5)  # messages pile up while they are down
            await st.start("text")
            await st.start("fusion")

    await play_call(
        st, ensure_wav(CALLS_DIR / name), cid, f"fixture:{name}", speed=0, on_progress=chaos
    )
    await wait_quiet(st, 120, settle=5.0)
    events = await st.events()
    replayed = st.workers["text"].replayed  # type: ignore[attr-defined]
    await st.shutdown()
    assert state["killed"]
    assert replayed >= 1  # the restarted text worker rebuilt the register state from the bus
    for g in gold(CALLS_DIR / name):
        assert any(matches(e, g) and e.status == EventStatus.CONFIRMED for e in events.values()), g
    assert all(e.speaker == Speaker.AGENT for e in events.values())


async def test_postcall_jobs_yield_to_realtime_backlog(tmp_path: Path) -> None:
    from gale_postcall.worker import PostcallWorker

    st = Stack()
    bus = st.bus()
    # a real-time backlog: frames the asr/audio groups have not consumed yet
    stream = audio_stream("busy")
    for g in ("asr_worker", "audio_worker"):
        await bus.ensure_group(stream, g)
    pcm = np.zeros(3200, "<i2").tobytes()
    for i in range(30):
        await bus.publish(
            stream,
            AudioFrame(
                call_id="busy",
                channel="agent",
                seq=i,
                ts_start=i * 0.2,  # type: ignore[arg-type]
                ts_end=(i + 1) * 0.2,
                pcm_s16le=pcm,
            ),
        )
    worker = PostcallWorker(
        st.bus(),
        SETTINGS.model_copy(update={"object_store_dir": tmp_path / "obj"}),
        backlog_limit=10,
    )
    await worker.setup()
    job_id = str(uuid.uuid4())
    await bus.publish(
        POSTCALL_JOBS,
        PostcallJob(
            call_id=job_id,
            uri=str(ensure_wav(CALLS_DIR / "demo_06")),
            external_ref="fixture:demo_06",
        ),
    )
    assert await worker.poll() == 0
    assert worker.yielded == 1
    assert worker.done == []
    await bus.redis.delete(stream)  # backlog drained
    assert await worker.poll() >= 1
    assert worker.done == [job_id]
    assert (tmp_path / "obj" / "reports" / job_id / "report.html").is_file()
    published = [e for e in await st.events() if True]
    assert published

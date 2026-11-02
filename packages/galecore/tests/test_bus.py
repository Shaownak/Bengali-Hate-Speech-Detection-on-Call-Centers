import asyncio

import pytest
from fakeredis import FakeAsyncRedis

from galecore.bus import (
    DLQ_SUFFIX,
    SIGNALS,
    BusClient,
    audio_stream,
    decode,
    encode,
)
from galecore.config import Settings
from galecore.schemas import AudioFrame, Category, Channel, Signal, SignalSource


def make_signal(i: int = 1) -> Signal:
    return Signal(
        call_id="call-1", channel=Channel.AGENT, signal_id=f"sig-{i}",
        source=SignalSource.LEXICON, category=Category.SLANG_PROFANITY, score=0.9,
        ts_start=1.0, ts_end=1.5, model_version="test",
    )  # fmt: skip


@pytest.fixture
def bus() -> BusClient:
    return BusClient(FakeAsyncRedis())


def test_encode_decode_audio_frame_binary_field() -> None:
    pcm = bytes(range(256)) * 2
    frame = AudioFrame(
        call_id="c", channel=Channel.CUSTOMER, seq=3, ts_start=0.6, ts_end=0.8, pcm_s16le=pcm
    )
    fields = encode(frame, idem_key="c:customer:3")
    assert b"pcm" not in fields["data"]  # D-02: PCM is a raw field, not inside JSON
    assert fields["bin:pcm_s16le"] == pcm
    back = decode({k.encode(): v for k, v in fields.items()}, AudioFrame)
    assert back == frame


def test_audio_stream_shard_is_stable() -> None:
    assert audio_stream("call-42") == audio_stream("call-42")
    assert audio_stream("call-42").startswith("audio.frames.")


async def test_publish_consume_ack(bus: BusClient) -> None:
    await bus.ensure_group(SIGNALS, "fusion")
    await bus.ensure_group(SIGNALS, "fusion")  # idempotent (BUSYGROUP ignored)
    await bus.publish(SIGNALS, make_signal(1), idem_key="sig-1")
    got: list[Signal] = []

    async def handler(s: Signal) -> None:
        got.append(s)

    n = await bus.poll_once(SIGNALS, "fusion", "w1", Signal, handler, block_ms=None)
    assert n == 1
    assert got == [make_signal(1)]
    pending = await bus.redis.xpending(SIGNALS, "fusion")
    assert pending["pending"] == 0


async def test_duplicate_idem_key_skipped(bus: BusClient) -> None:
    await bus.ensure_group(SIGNALS, "fusion")
    for _ in range(3):
        await bus.publish(SIGNALS, make_signal(1), idem_key="sig-1")
    calls = 0

    async def handler(_s: Signal) -> None:
        nonlocal calls
        calls += 1

    await bus.poll_once(SIGNALS, "fusion", "w1", Signal, handler, block_ms=None)
    assert calls == 1
    assert (await bus.redis.xpending(SIGNALS, "fusion"))["pending"] == 0


async def test_failure_retries_then_dead_letters(bus: BusClient) -> None:
    await bus.ensure_group(SIGNALS, "fusion")
    await bus.publish(SIGNALS, make_signal(7), idem_key="sig-7")
    attempts = 0

    async def flaky(_s: Signal) -> None:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("boom")

    for _ in range(3):
        await bus.poll_once(
            SIGNALS, "fusion", "w1", Signal, flaky, block_ms=None, max_retries=3, min_idle_ms=0
        )
    assert attempts == 3
    assert (await bus.redis.xpending(SIGNALS, "fusion"))["pending"] == 0
    dead = await bus.redis.xrange(SIGNALS + DLQ_SUFFIX)
    assert len(dead) == 1
    assert dead[0][1][b"error"] == b"RuntimeError"
    assert decode(dead[0][1], Signal) == make_signal(7)


async def test_failure_then_success_on_retry(bus: BusClient) -> None:
    await bus.ensure_group(SIGNALS, "fusion")
    await bus.publish(SIGNALS, make_signal(2), idem_key="sig-2")
    seen: list[str] = []

    async def once_flaky(s: Signal) -> None:
        seen.append(s.signal_id)
        if len(seen) == 1:
            raise ValueError("transient")

    for _ in range(2):
        await bus.poll_once(
            SIGNALS, "fusion", "w1", Signal, once_flaky, block_ms=None, min_idle_ms=0
        )
    assert seen == ["sig-2", "sig-2"]
    assert (await bus.redis.xpending(SIGNALS, "fusion"))["pending"] == 0
    assert await bus.redis.xlen(SIGNALS + DLQ_SUFFIX) == 0


async def test_consume_loop_stops(bus: BusClient) -> None:
    stop = asyncio.Event()
    got: list[Signal] = []

    async def handler(s: Signal) -> None:
        got.append(s)
        stop.set()

    await bus.publish(SIGNALS, make_signal(3))
    await asyncio.wait_for(
        bus.consume(SIGNALS, "g", "w", Signal, handler, stop=stop, block_ms=10), timeout=5
    )
    assert len(got) == 1


async def test_from_settings_memory_and_ping() -> None:
    bus = BusClient.from_settings(Settings(env="test", bus_backend="memory"))
    assert await bus.ping()
    await bus.close()


async def test_ping_false_when_unreachable() -> None:
    bus = BusClient.from_settings(Settings(env="test", bus_url="redis://127.0.0.1:1/0"))
    assert not await bus.ping()

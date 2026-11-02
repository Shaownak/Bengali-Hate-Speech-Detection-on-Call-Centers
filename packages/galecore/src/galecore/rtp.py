"""Minimal RTP (RFC 3550) packet codec for G.711 telephony streams.

Used by ``tools/simulator`` (sender) and ``ingest-gateway`` (receiver). Only what PBX media
forks send is supported: fixed header, CSRC list, one header extension, padding.
"""

from __future__ import annotations

import struct
from collections.abc import Iterator
from dataclasses import dataclass, field

PT_PCMU = 0  # G.711 µ-law, 8 kHz
PT_PCMA = 8  # G.711 A-law, 8 kHz
TELEPHONY_SR = 8000
PTIME_MS = 20
SAMPLES_PER_PACKET = TELEPHONY_SR * PTIME_MS // 1000  # 160 (one byte per sample in G.711)

_HEADER = struct.Struct("!BBHII")


class RtpError(ValueError):
    """Malformed RTP packet."""


@dataclass(frozen=True, slots=True)
class RtpPacket:
    payload_type: int
    seq: int
    timestamp: int
    ssrc: int
    payload: bytes
    marker: bool = False

    def to_bytes(self) -> bytes:
        b1 = (0x80 if self.marker else 0) | (self.payload_type & 0x7F)
        header = _HEADER.pack(0x80, b1, self.seq & 0xFFFF, self.timestamp & 0xFFFFFFFF, self.ssrc)
        return header + self.payload


def parse(data: bytes) -> RtpPacket:
    if len(data) < _HEADER.size:
        raise RtpError("packet shorter than the RTP header")
    b0, b1, seq, ts, ssrc = _HEADER.unpack_from(data)
    if b0 >> 6 != 2:
        raise RtpError("not RTP version 2")
    offset = _HEADER.size + 4 * (b0 & 0x0F)
    if b0 & 0x10:  # header extension: 16-bit profile, 16-bit length in 32-bit words
        if len(data) < offset + 4:
            raise RtpError("truncated header extension")
        (words,) = struct.unpack_from("!H", data, offset + 2)
        offset += 4 + 4 * words
    end = len(data)
    if b0 & 0x20:  # padding: last byte holds the pad length
        end -= data[-1]
    if offset > end or end > len(data):
        raise RtpError("header, extension or padding overruns the packet")
    return RtpPacket(
        payload_type=b1 & 0x7F,
        seq=seq,
        timestamp=ts,
        ssrc=ssrc,
        payload=bytes(data[offset:end]),
        marker=bool(b1 & 0x80),
    )


def packetize(
    payload: bytes,
    *,
    ssrc: int,
    payload_type: int = PT_PCMU,
    start_seq: int = 0,
    start_ts: int = 0,
    samples_per_packet: int = SAMPLES_PER_PACKET,
) -> Iterator[RtpPacket]:
    """Cut an encoded G.711 byte stream into 20 ms RTP packets. The first one has the marker bit."""
    for i, start in enumerate(range(0, len(payload), samples_per_packet)):
        yield RtpPacket(
            payload_type=payload_type,
            seq=(start_seq + i) & 0xFFFF,
            timestamp=(start_ts + start) & 0xFFFFFFFF,
            ssrc=ssrc,
            payload=payload[start : start + samples_per_packet],
            marker=i == 0,
        )


@dataclass(slots=True)
class SeqTracker:
    """Loss / reorder / duplicate accounting over 16-bit sequence numbers (handles wrap)."""

    received: int = 0
    duplicates: int = 0
    out_of_order: int = 0
    _base: int | None = None
    _max: int = 0
    _seen: set[int] = field(default_factory=set)

    def update(self, seq: int) -> int:
        """Record one packet; return its extended (unwrapped) sequence number."""
        self.received += 1
        if self._base is None:
            self._base = self._max = seq
            self._seen.add(seq)
            return seq
        delta = (seq - self._max) & 0xFFFF
        ext = self._max + delta if delta < 0x8000 else self._max - (0x10000 - delta)
        if ext in self._seen:
            self.duplicates += 1
            return ext
        self._seen.add(ext)
        if ext < self._max:
            self.out_of_order += 1
            self._base = min(self._base, ext)
        else:
            self._max = ext
        if len(self._seen) > 4096:  # keep the duplicate window bounded
            self._seen = {s for s in self._seen if s > self._max - 2048}
        return ext

    @property
    def expected(self) -> int:
        return 0 if self._base is None else self._max - self._base + 1

    @property
    def lost(self) -> int:
        return max(0, self.expected - (self.received - self.duplicates))

import pytest

from galecore.rtp import (
    PT_PCMA,
    PT_PCMU,
    RtpError,
    RtpPacket,
    SeqTracker,
    packetize,
    parse,
)


def test_build_parse_roundtrip() -> None:
    p = RtpPacket(
        payload_type=PT_PCMU,
        seq=65535,
        timestamp=4_000_000_000,
        ssrc=0xDEADBEEF,
        payload=b"\x7f" * 160,
        marker=True,
    )
    raw = p.to_bytes()
    assert len(raw) == 12 + 160
    assert raw[0] == 0x80  # V=2, no padding/extension/CSRC
    assert raw[1] == 0x80 | PT_PCMU  # marker bit
    assert parse(raw) == p


def test_parse_rejects_bad_version_and_short() -> None:
    with pytest.raises(RtpError):
        parse(b"\x00" * 12)
    with pytest.raises(RtpError):
        parse(b"\x80\x00")


def test_parse_skips_csrc_extension_and_padding() -> None:
    # V=2, P=1, X=1, CC=1
    header = bytes([0x80 | 0x20 | 0x10 | 0x01, PT_PCMA]) + (7).to_bytes(2, "big")
    header += (160).to_bytes(4, "big") + (42).to_bytes(4, "big")
    csrc = (99).to_bytes(4, "big")
    ext = (0xBEDE).to_bytes(2, "big") + (1).to_bytes(2, "big") + b"\x01\x02\x03\x04"
    payload = b"\xd5" * 8
    padding = b"\x00\x00\x03"
    p = parse(header + csrc + ext + payload + padding)
    assert p.payload == payload
    assert (p.payload_type, p.seq, p.timestamp, p.ssrc) == (PT_PCMA, 7, 160, 42)


def test_parse_rejects_truncated_extension_and_bad_padding() -> None:
    header = bytes([0x80 | 0x10, PT_PCMU]) + bytes(10)
    with pytest.raises(RtpError):
        parse(header + (0xBEDE).to_bytes(2, "big") + (5).to_bytes(2, "big"))
    with pytest.raises(RtpError):
        parse(bytes([0x80 | 0x20, PT_PCMU]) + bytes(10) + b"\x00\x09")


def test_packetize_20ms_with_wrapping_seq() -> None:
    payload = bytes(range(256)) * 2  # 512 bytes → 160, 160, 160, 32
    pkts = list(packetize(payload, ssrc=1, payload_type=PT_PCMU, start_seq=65534, start_ts=10))
    assert [len(p.payload) for p in pkts] == [160, 160, 160, 32]
    assert [p.seq for p in pkts] == [65534, 65535, 0, 1]
    assert [p.timestamp for p in pkts] == [10, 170, 330, 490]
    assert pkts[0].marker
    assert not any(p.marker for p in pkts[1:])
    assert b"".join(p.payload for p in pkts) == payload


def test_seq_tracker_counts_loss_reorder_and_wrap() -> None:
    t = SeqTracker()
    for s in [65533, 65534, 65535, 0, 3, 2, 4]:
        t.update(s)
    assert t.received == 7
    assert t.lost == 1  # 1 missing (2 arrived late and was counted as out of order)
    assert t.out_of_order == 1
    assert t.expected == 8  # 65533..4 inclusive


def test_seq_tracker_duplicate() -> None:
    t = SeqTracker()
    for s in [10, 11, 11, 12]:
        t.update(s)
    assert t.duplicates == 1
    assert t.lost == 0

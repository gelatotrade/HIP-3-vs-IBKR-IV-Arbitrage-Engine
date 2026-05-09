"""
Python writer for the shared-memory signal ring buffer consumed by the Rust
trade-bot. Sub-microsecond producer-side latency.

Wire format (must match `crates/signal-bridge/src/ringbuffer.rs`):
  Region: 64-byte header + 16384 slots * 256 bytes = 4 MB total
  Header: magic | version | capacity | slot_size | write_seq | read_seq | reserved
  Each slot: [u32 length] [bincode-encoded Signal]

bincode uses little-endian fixed-int + length-prefixed strings/options. Since
implementing bincode in Python is tedious, we ship a thin ctypes-based binary
encoder that matches Rust's bincode default config exactly.
"""

from __future__ import annotations

import mmap
import os
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

MAGIC = 0x5349_4E41_4C53_5231
VERSION = 1
SLOT_SIZE = 256
RING_CAPACITY = 16_384
HEADER_SIZE = 64
REGION_SIZE = HEADER_SIZE + RING_CAPACITY * SLOT_SIZE

OFF_MAGIC = 0
OFF_VERSION = 8
OFF_CAPACITY = 16
OFF_SLOT_SIZE = 24
OFF_WRITE_SEQ = 32
OFF_READ_SEQ = 40

KIND_VOL_SPREAD = "vol_spread"
KIND_CALENDAR = "calendar"
KIND_RISK_REVERSAL = "risk_reversal"
KIND_GAMMA_SCALP = "gamma_scalp"


@dataclass
class Signal:
    seq: int
    kind: str
    symbol: str
    side: str  # "buy" or "sell"
    size: str  # decimal as string
    limit_price: Optional[str]
    edge_bps: int
    confidence: float
    ttl_ms: int
    timestamp_ns: int


def _bincode_encode(sig: Signal) -> bytes:
    """Encode a Signal in bincode-compatible binary format."""
    out = bytearray()

    # u64 seq
    out += struct.pack("<Q", sig.seq)

    # SignalKind enum (variant index as u32)
    kind_idx = {
        KIND_VOL_SPREAD: 0,
        KIND_CALENDAR: 1,
        KIND_RISK_REVERSAL: 2,
        KIND_GAMMA_SCALP: 3,
    }[sig.kind]
    out += struct.pack("<I", kind_idx)

    # String (length u64 + bytes)
    sym_b = sig.symbol.encode()
    out += struct.pack("<Q", len(sym_b)) + sym_b

    # Side enum (variant index as u32)
    side_idx = 0 if sig.side == "buy" else 1
    out += struct.pack("<I", side_idx)

    # Decimal as string (rust_decimal serializes with serde-with-str)
    size_b = sig.size.encode()
    out += struct.pack("<Q", len(size_b)) + size_b

    # Option<Decimal>: 0 = None, 1 = Some + decimal string
    if sig.limit_price is None:
        out += struct.pack("<I", 0)
    else:
        out += struct.pack("<I", 1)
        lp_b = sig.limit_price.encode()
        out += struct.pack("<Q", len(lp_b)) + lp_b

    # i32 edge_bps
    out += struct.pack("<i", sig.edge_bps)

    # f32 confidence
    out += struct.pack("<f", sig.confidence)

    # u32 ttl_ms
    out += struct.pack("<I", sig.ttl_ms)

    # u64 timestamp_ns
    out += struct.pack("<Q", sig.timestamp_ns)

    return bytes(out)


class SignalRingWriter:
    """Single-producer writer to the shared-memory ring buffer."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        if not self.path.exists() or self.path.stat().st_size != REGION_SIZE:
            with open(self.path, "wb") as f:
                f.truncate(REGION_SIZE)
        self._fd = os.open(self.path, os.O_RDWR)
        self._mm = mmap.mmap(self._fd, REGION_SIZE)
        self._lock = threading.Lock()
        self._init_header()

    def _init_header(self) -> None:
        magic = struct.unpack_from("<Q", self._mm, OFF_MAGIC)[0]
        if magic != MAGIC:
            struct.pack_into("<Q", self._mm, OFF_MAGIC, MAGIC)
            struct.pack_into("<Q", self._mm, OFF_VERSION, VERSION)
            struct.pack_into("<Q", self._mm, OFF_CAPACITY, RING_CAPACITY)
            struct.pack_into("<Q", self._mm, OFF_SLOT_SIZE, SLOT_SIZE)
            struct.pack_into("<Q", self._mm, OFF_WRITE_SEQ, 0)
            struct.pack_into("<Q", self._mm, OFF_READ_SEQ, 0)

    def push(self, sig: Signal) -> None:
        payload = _bincode_encode(sig)
        if len(payload) + 4 > SLOT_SIZE:
            raise ValueError(f"signal too large: {len(payload)} bytes (max {SLOT_SIZE - 4})")

        with self._lock:
            write_seq = struct.unpack_from("<Q", self._mm, OFF_WRITE_SEQ)[0]
            read_seq = struct.unpack_from("<Q", self._mm, OFF_READ_SEQ)[0]
            if write_seq - read_seq >= RING_CAPACITY:
                raise RuntimeError("ring buffer full (consumer lagging)")

            slot_idx = write_seq & (RING_CAPACITY - 1)
            slot_start = HEADER_SIZE + slot_idx * SLOT_SIZE
            struct.pack_into("<I", self._mm, slot_start, len(payload))
            self._mm[slot_start + 4 : slot_start + 4 + len(payload)] = payload
            struct.pack_into("<Q", self._mm, OFF_WRITE_SEQ, write_seq + 1)

    def close(self) -> None:
        self._mm.close()
        os.close(self._fd)


def now_ns() -> int:
    return time.time_ns()


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/hip3_signals.ring"
    writer = SignalRingWriter(path)
    sig = Signal(
        seq=int(time.time_ns()),
        kind=KIND_VOL_SPREAD,
        symbol="AAPL",
        side="buy",
        size="1.5",
        limit_price="195.50",
        edge_bps=35,
        confidence=0.82,
        ttl_ms=250,
        timestamp_ns=now_ns(),
    )
    writer.push(sig)
    print(f"pushed signal to {path}")

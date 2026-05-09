//! Lock-free SPSC shared-memory ring buffer for sub-microsecond signal transport
//! between Python (producer) and Rust (consumer).
//!
//! Wire format: 256-byte fixed-size slots, 16384 slots = 4 MB region.
//! Layout (header is 64 bytes, cache-line aligned):
//!   [0..8)   : magic (u64 = 0x53494E414C535231 = "SIGNALSR1")
//!   [8..16)  : version (u64)
//!   [16..24) : capacity (u64, must be power of 2)
//!   [24..32) : slot_size (u64, fixed 256)
//!   [32..40) : write_seq (u64, monotonic atomic)
//!   [40..48) : read_seq  (u64, monotonic atomic)
//!   [48..64) : reserved
//!   [64..)   : slots, each 256 bytes
//!
//! Each slot starts with [0..4) = payload length (u32), then bincode-encoded `Signal`.

pub mod ringbuffer;

pub use ringbuffer::{SignalReader, SignalWriter, RING_CAPACITY, SLOT_SIZE};

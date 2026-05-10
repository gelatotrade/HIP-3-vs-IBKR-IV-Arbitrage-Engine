use anyhow::{anyhow, Result};
use common::Signal;
use memmap2::{MmapMut, MmapOptions};
use std::fs::OpenOptions;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

pub const MAGIC: u64 = 0x5349_4E41_4C53_5231_u64;
pub const VERSION: u64 = 1;
pub const SLOT_SIZE: usize = 256;
pub const RING_CAPACITY: usize = 16_384;
pub const HEADER_SIZE: usize = 64;
pub const REGION_SIZE: usize = HEADER_SIZE + RING_CAPACITY * SLOT_SIZE;

const OFF_MAGIC: usize = 0;
const OFF_VERSION: usize = 8;
const OFF_CAPACITY: usize = 16;
const OFF_SLOT_SIZE: usize = 24;
const OFF_WRITE_SEQ: usize = 32;
const OFF_READ_SEQ: usize = 40;

fn write_u64(buf: &mut [u8], offset: usize, val: u64) {
    buf[offset..offset + 8].copy_from_slice(&val.to_le_bytes());
}

fn read_u64(buf: &[u8], offset: usize) -> u64 {
    let mut tmp = [0u8; 8];
    tmp.copy_from_slice(&buf[offset..offset + 8]);
    u64::from_le_bytes(tmp)
}

fn atomic_at(buf: &mut [u8], offset: usize) -> &AtomicU64 {
    let ptr = unsafe { buf.as_mut_ptr().add(offset) } as *const AtomicU64;
    unsafe { &*ptr }
}

fn open_or_create(path: &Path) -> Result<MmapMut> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)?;
    file.set_len(REGION_SIZE as u64)?;
    let mmap = unsafe { MmapOptions::new().len(REGION_SIZE).map_mut(&file)? };
    Ok(mmap)
}

fn init_header(buf: &mut [u8]) {
    if read_u64(buf, OFF_MAGIC) != MAGIC {
        write_u64(buf, OFF_MAGIC, MAGIC);
        write_u64(buf, OFF_VERSION, VERSION);
        write_u64(buf, OFF_CAPACITY, RING_CAPACITY as u64);
        write_u64(buf, OFF_SLOT_SIZE, SLOT_SIZE as u64);
        write_u64(buf, OFF_WRITE_SEQ, 0);
        write_u64(buf, OFF_READ_SEQ, 0);
    }
}

/// Writer side (typically on the Python producer; provided here for tests
/// and for Rust-side replay tools). Single-producer.
pub struct SignalWriter {
    mmap: MmapMut,
}

impl SignalWriter {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let mut mmap = open_or_create(path.as_ref())?;
        init_header(&mut mmap);
        Ok(Self { mmap })
    }

    pub fn push(&mut self, signal: &Signal) -> Result<()> {
        let payload = bincode::serialize(signal)?;
        if payload.len() + 4 > SLOT_SIZE {
            return Err(anyhow!(
                "signal too large: {} bytes (max {})",
                payload.len(),
                SLOT_SIZE - 4
            ));
        }

        let buf = &mut self.mmap[..];
        let write_seq = atomic_at(buf, OFF_WRITE_SEQ).load(Ordering::Relaxed);
        let read_seq = atomic_at(buf, OFF_READ_SEQ).load(Ordering::Acquire);
        if write_seq.saturating_sub(read_seq) >= RING_CAPACITY as u64 {
            return Err(anyhow!("ring buffer full (consumer lagging)"));
        }

        let slot_idx = (write_seq as usize) & (RING_CAPACITY - 1);
        let slot_start = HEADER_SIZE + slot_idx * SLOT_SIZE;
        let len = payload.len() as u32;
        buf[slot_start..slot_start + 4].copy_from_slice(&len.to_le_bytes());
        buf[slot_start + 4..slot_start + 4 + payload.len()].copy_from_slice(&payload);

        atomic_at(buf, OFF_WRITE_SEQ).store(write_seq + 1, Ordering::Release);
        Ok(())
    }
}

/// Reader side. Single-consumer.
pub struct SignalReader {
    mmap: MmapMut,
    next_read: u64,
}

impl SignalReader {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let mut mmap = open_or_create(path.as_ref())?;
        init_header(&mut mmap);

        let buf = &mmap[..];
        if read_u64(buf, OFF_MAGIC) != MAGIC {
            return Err(anyhow!("invalid ring buffer magic"));
        }

        let next_read = atomic_at(&mut mmap, OFF_READ_SEQ).load(Ordering::Relaxed);
        Ok(Self { mmap, next_read })
    }

    pub fn try_pop(&mut self) -> Result<Option<Signal>> {
        let buf = &mut self.mmap[..];
        let write_seq = atomic_at(buf, OFF_WRITE_SEQ).load(Ordering::Acquire);
        if self.next_read >= write_seq {
            return Ok(None);
        }

        let slot_idx = (self.next_read as usize) & (RING_CAPACITY - 1);
        let slot_start = HEADER_SIZE + slot_idx * SLOT_SIZE;
        let mut len_bytes = [0u8; 4];
        len_bytes.copy_from_slice(&buf[slot_start..slot_start + 4]);
        let len = u32::from_le_bytes(len_bytes) as usize;
        if len > SLOT_SIZE - 4 {
            return Err(anyhow!("corrupt slot: length {} > slot size", len));
        }
        let payload = &buf[slot_start + 4..slot_start + 4 + len];
        let signal: Signal = bincode::deserialize(payload)?;

        self.next_read += 1;
        atomic_at(buf, OFF_READ_SEQ).store(self.next_read, Ordering::Release);
        Ok(Some(signal))
    }

    pub fn lag(&mut self) -> u64 {
        let buf = &mut self.mmap[..];
        let write_seq = atomic_at(buf, OFF_WRITE_SEQ).load(Ordering::Relaxed);
        write_seq.saturating_sub(self.next_read)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::{Side, SignalKind};
    use rust_decimal_macros::dec;
    use tempfile::NamedTempFile;

    fn sample_signal(seq: u64) -> Signal {
        Signal {
            seq,
            kind: SignalKind::VolSpread,
            symbol: "AAPL".to_string(),
            side: Side::Buy,
            size: dec!(1.5),
            limit_price: Some(dec!(195.50)),
            edge_bps: 35,
            confidence: 0.82,
            ttl_ms: 250,
            timestamp_ns: 1_700_000_000_000_000_000,
        }
    }

    #[test]
    fn round_trip_one_signal() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path();
        let mut writer = SignalWriter::open(path).unwrap();
        let mut reader = SignalReader::open(path).unwrap();

        writer.push(&sample_signal(1)).unwrap();
        let got = reader.try_pop().unwrap().unwrap();
        assert_eq!(got.seq, 1);
        assert_eq!(got.symbol, "AAPL");
    }

    #[test]
    fn empty_returns_none() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path();
        let _writer = SignalWriter::open(path).unwrap();
        let mut reader = SignalReader::open(path).unwrap();
        assert!(reader.try_pop().unwrap().is_none());
    }

    #[test]
    fn many_signals_in_order() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path();
        let mut writer = SignalWriter::open(path).unwrap();
        let mut reader = SignalReader::open(path).unwrap();
        for i in 0..1000 {
            writer.push(&sample_signal(i)).unwrap();
        }
        for i in 0..1000 {
            let got = reader.try_pop().unwrap().unwrap();
            assert_eq!(got.seq, i);
        }
        assert!(reader.try_pop().unwrap().is_none());
    }
}

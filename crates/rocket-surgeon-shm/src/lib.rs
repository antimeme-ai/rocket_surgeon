pub mod cleanup;
pub mod region;
pub mod ring;

pub const CONTROL_SIZE: usize = 4096;
pub const MAGIC: &[u8; 8] = b"DOOMRING";
pub const MAGIC_SIZE: usize = 8;
pub const VERSION: u32 = 1;

pub const MAGIC_OFFSET: usize = 0;
pub const VERSION_OFFSET: usize = 8;
pub const SLOT_COUNT_OFFSET: usize = 12;
pub const SLOT_SIZE_OFFSET: usize = 16;
pub const REGION_SIZE_OFFSET: usize = 24;

pub const MAKETIC_OFFSET: usize = 128;
pub const NETTICS_OFFSET: usize = 256;

pub const PROBE_FRAME_HEADER_SIZE: usize = 128;

// Probe frame field offsets (must match probe_frame.rs in rocket-surgeon-python)
pub const FRAME_OFFSET_RANK: usize = 0;
pub const FRAME_OFFSET_LAYER: usize = 4;
pub const FRAME_OFFSET_COMP_ID: usize = 8;
pub const FRAME_OFFSET_DTYPE: usize = 10;
pub const FRAME_OFFSET_NDIM: usize = 11;
pub const FRAME_OFFSET_SHAPE: usize = 12;
pub const FRAME_OFFSET_TICK_ID: usize = 48;
pub const FRAME_OFFSET_DATA_OFF: usize = 56;
pub const FRAME_OFFSET_SIZE: usize = 64;
pub const FRAME_OFFSET_FLAGS: usize = 72;
pub const FRAME_OFFSET_GENERATION: usize = 76;

#[derive(Debug, Clone, Copy)]
pub struct RingConfig {
    pub backuptics: u32,
    pub slot_size: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum ShmError {
    #[error("backuptics ({0}) must be a power of two and > 0")]
    NotPowerOfTwo(u32),
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
    #[error("shm_open failed for '{name}': {source}")]
    Open {
        name: String,
        source: std::io::Error,
    },
    #[error("mmap failed: {0}")]
    Mmap(std::io::Error),
    #[error("ftruncate failed: {0}")]
    Truncate(std::io::Error),
    #[error("shm_unlink failed for '{name}': {source}")]
    Unlink {
        name: String,
        source: std::io::Error,
    },
    #[error("offset {offset} + length {length} exceeds region size {region_size}")]
    OutOfBounds {
        offset: usize,
        length: usize,
        region_size: usize,
    },
    #[error("read out of bounds: offset {offset} + length {length} exceeds capacity {capacity}")]
    ReadOutOfBounds {
        offset: usize,
        length: usize,
        capacity: usize,
    },
    #[error("offset {offset} is not {alignment}-byte aligned")]
    Unaligned { offset: usize, alignment: usize },
    #[error("magic mismatch: expected DOOMRING, got {0:?}")]
    BadMagic([u8; 8]),
    #[error("version mismatch: expected {expected}, got {got}")]
    BadVersion { expected: u32, got: u32 },
    #[error("ring full: maketic={maketic}, nettics={nettics}, capacity={capacity}")]
    RingFull {
        maketic: u64,
        nettics: u64,
        capacity: u32,
    },
    #[error("tensor size {tensor_size} exceeds slot data capacity {slot_capacity}")]
    TensorTooLarge {
        tensor_size: usize,
        slot_capacity: usize,
    },
    #[error("shm name '{name}' exceeds max length {max_len}")]
    NameTooLong { name: String, max_len: usize },
    #[error("region size mismatch: expected at least {expected}, got {actual}")]
    RegionTooSmall { expected: usize, actual: usize },
    #[error("stale slot: expected generation {expected}, got {actual}")]
    StaleSlot { expected: u32, actual: u32 },
}

impl RingConfig {
    pub fn new(backuptics: u32, slot_size: u64) -> Result<Self, ShmError> {
        if backuptics == 0 || (backuptics & (backuptics - 1)) != 0 {
            return Err(ShmError::NotPowerOfTwo(backuptics));
        }
        if slot_size < PROBE_FRAME_HEADER_SIZE as u64 {
            return Err(ShmError::InvalidConfig(format!(
                "slot_size ({slot_size}) must be >= PROBE_FRAME_HEADER_SIZE ({PROBE_FRAME_HEADER_SIZE})"
            )));
        }
        let slot_total = u64::from(backuptics)
            .checked_mul(slot_size)
            .and_then(|v| v.checked_add(CONTROL_SIZE as u64))
            .ok_or_else(|| ShmError::InvalidConfig("region_size overflows u64".into()))?;
        if slot_total > usize::MAX as u64 {
            return Err(ShmError::InvalidConfig(
                "region_size exceeds addressable memory".into(),
            ));
        }
        Ok(Self {
            backuptics,
            slot_size,
        })
    }

    pub fn region_size(&self) -> usize {
        CONTROL_SIZE + (self.backuptics as usize) * (self.slot_size as usize)
    }

    pub fn slot_offset(&self, maketic: u64) -> usize {
        let slot_index = (maketic & (u64::from(self.backuptics) - 1)) as usize;
        CONTROL_SIZE + slot_index * (self.slot_size as usize)
    }

    pub fn slot_data_capacity(&self) -> usize {
        (self.slot_size as usize) - PROBE_FRAME_HEADER_SIZE
    }

    pub fn mask(&self) -> u64 {
        u64::from(self.backuptics) - 1
    }
}

#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn serialize_probe_frame(
    rank: u32,
    layer: u32,
    comp_id: u16,
    dtype: u8,
    ndim: u8,
    shape: &[u32; 8],
    tick_id: u64,
    data_off: u64,
    size: u64,
    flags: u32,
    generation: u32,
) -> [u8; PROBE_FRAME_HEADER_SIZE] {
    let mut buf = [0u8; PROBE_FRAME_HEADER_SIZE];

    buf[FRAME_OFFSET_RANK..FRAME_OFFSET_RANK + 4].copy_from_slice(&rank.to_le_bytes());
    buf[FRAME_OFFSET_LAYER..FRAME_OFFSET_LAYER + 4].copy_from_slice(&layer.to_le_bytes());
    buf[FRAME_OFFSET_COMP_ID..FRAME_OFFSET_COMP_ID + 2].copy_from_slice(&comp_id.to_le_bytes());
    buf[FRAME_OFFSET_DTYPE] = dtype;
    buf[FRAME_OFFSET_NDIM] = ndim;

    for (i, &dim) in shape.iter().enumerate() {
        let start = FRAME_OFFSET_SHAPE + i * 4;
        buf[start..start + 4].copy_from_slice(&dim.to_le_bytes());
    }

    buf[FRAME_OFFSET_TICK_ID..FRAME_OFFSET_TICK_ID + 8].copy_from_slice(&tick_id.to_le_bytes());
    buf[FRAME_OFFSET_DATA_OFF..FRAME_OFFSET_DATA_OFF + 8].copy_from_slice(&data_off.to_le_bytes());
    buf[FRAME_OFFSET_SIZE..FRAME_OFFSET_SIZE + 8].copy_from_slice(&size.to_le_bytes());
    buf[FRAME_OFFSET_FLAGS..FRAME_OFFSET_FLAGS + 4].copy_from_slice(&flags.to_le_bytes());
    buf[FRAME_OFFSET_GENERATION..FRAME_OFFSET_GENERATION + 4]
        .copy_from_slice(&generation.to_le_bytes());

    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn control_size_is_4096() {
        assert_eq!(CONTROL_SIZE, 4096);
    }

    #[test]
    fn magic_is_doomring() {
        assert_eq!(MAGIC, b"DOOMRING");
    }

    #[test]
    fn cursors_on_separate_cache_lines() {
        assert_eq!(MAKETIC_OFFSET, 128);
        assert_eq!(NETTICS_OFFSET, 256);
        const { assert!(NETTICS_OFFSET - MAKETIC_OFFSET >= 128) };
    }

    #[test]
    fn ring_config_validates_power_of_two() {
        assert!(RingConfig::new(16, 64 * 1024 * 1024).is_ok());
        assert!(RingConfig::new(15, 64 * 1024 * 1024).is_err());
        assert!(RingConfig::new(0, 64 * 1024 * 1024).is_err());
        assert!(RingConfig::new(1, 64 * 1024 * 1024).is_ok());
    }

    #[test]
    fn slot_offset_computation() {
        let config = RingConfig::new(16, 64 * 1024 * 1024).unwrap();
        assert_eq!(config.slot_offset(0), CONTROL_SIZE);
        assert_eq!(config.slot_offset(1), CONTROL_SIZE + 64 * 1024 * 1024);
        assert_eq!(config.slot_offset(16), config.slot_offset(0));
    }

    #[test]
    fn region_size_computation() {
        let config = RingConfig::new(16, 64 * 1024 * 1024).unwrap();
        assert_eq!(config.region_size(), CONTROL_SIZE + 16 * 64 * 1024 * 1024);
    }

    #[test]
    fn slot_data_capacity() {
        let config = RingConfig::new(16, 64 * 1024 * 1024).unwrap();
        assert_eq!(config.slot_data_capacity(), 64 * 1024 * 1024 - 128);
    }
}

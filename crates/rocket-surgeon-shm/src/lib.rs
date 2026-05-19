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

#[derive(Debug, Clone, Copy)]
pub struct RingConfig {
    pub backuptics: u32,
    pub slot_size: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum ShmError {
    #[error("backuptics ({0}) must be a power of two and > 0")]
    NotPowerOfTwo(u32),
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
}

impl RingConfig {
    pub fn new(backuptics: u32, slot_size: u64) -> Result<Self, ShmError> {
        if backuptics == 0 || (backuptics & (backuptics - 1)) != 0 {
            return Err(ShmError::NotPowerOfTwo(backuptics));
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

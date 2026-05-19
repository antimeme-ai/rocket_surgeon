use crate::region::ShmRegion;
use crate::{
    CONTROL_SIZE, FRAME_OFFSET_GENERATION, FRAME_OFFSET_SIZE, MAGIC, MAKETIC_OFFSET,
    NETTICS_OFFSET, PROBE_FRAME_HEADER_SIZE, REGION_SIZE_OFFSET, RingConfig, SLOT_COUNT_OFFSET,
    SLOT_SIZE_OFFSET, ShmError, VERSION, VERSION_OFFSET,
};

pub struct ConsumedFrame {
    pub header: Vec<u8>,
    pub data: Vec<u8>,
}

pub struct DoomRingProducer {
    region: ShmRegion,
    config: RingConfig,
    maketic: u64,
}

impl std::fmt::Debug for DoomRingProducer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DoomRingProducer")
            .field("name", &self.region.name())
            .field("maketic", &self.maketic)
            .finish_non_exhaustive()
    }
}

impl DoomRingProducer {
    pub fn create(name: &str, config: RingConfig) -> Result<Self, ShmError> {
        let region_size = config.region_size();
        let region = ShmRegion::create(name, region_size)?;

        region.write_bytes(VERSION_OFFSET, &VERSION.to_le_bytes())?;
        region.write_bytes(SLOT_COUNT_OFFSET, &config.backuptics.to_le_bytes())?;
        region.write_bytes(SLOT_SIZE_OFFSET, &config.slot_size.to_le_bytes())?;
        region.write_bytes(REGION_SIZE_OFFSET, &(region_size as u64).to_le_bytes())?;

        region.atomic_store_u64(MAKETIC_OFFSET, 0)?;
        region.atomic_store_u64(NETTICS_OFFSET, 0)?;

        // Write magic LAST — this is the init barrier
        region.write_magic()?;

        Ok(Self {
            region,
            config,
            maketic: 0,
        })
    }

    pub fn publish(&mut self, header: &[u8], data: &[u8]) -> Result<u64, ShmError> {
        assert_eq!(header.len(), PROBE_FRAME_HEADER_SIZE);

        if data.len() > self.config.slot_data_capacity() {
            return Err(ShmError::TensorTooLarge {
                tensor_size: data.len(),
                slot_capacity: self.config.slot_data_capacity(),
            });
        }

        let nettics = self.region.atomic_load_u64(NETTICS_OFFSET)?;
        if (self.maketic - nettics) >= u64::from(self.config.backuptics) {
            return Err(ShmError::RingFull {
                maketic: self.maketic,
                nettics,
                capacity: self.config.backuptics,
            });
        }

        let slot_offset = self.config.slot_offset(self.maketic);
        self.region.write_bytes(slot_offset, header)?;
        self.region
            .write_bytes(slot_offset + PROBE_FRAME_HEADER_SIZE, data)?;

        self.maketic += 1;
        self.region.atomic_store_u64(MAKETIC_OFFSET, self.maketic)?;

        Ok(self.maketic - 1)
    }

    pub fn is_full(&self) -> bool {
        let nettics = self.region.atomic_load_u64(NETTICS_OFFSET).unwrap_or(0);
        (self.maketic - nettics) >= u64::from(self.config.backuptics)
    }

    pub fn maketic(&self) -> u64 {
        self.maketic
    }

    pub fn config(&self) -> &RingConfig {
        &self.config
    }

    pub fn shm_name(&self) -> &str {
        self.region.name()
    }
}

pub struct DoomRingConsumer {
    region: ShmRegion,
    config: RingConfig,
    nettics: u64,
}

impl std::fmt::Debug for DoomRingConsumer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DoomRingConsumer")
            .field("name", &self.region.name())
            .field("nettics", &self.nettics)
            .finish_non_exhaustive()
    }
}

impl DoomRingConsumer {
    pub fn open(name: &str) -> Result<Self, ShmError> {
        let probe_region = ShmRegion::open(name, CONTROL_SIZE)?;

        let mut magic = [0u8; 8];
        probe_region.read_bytes(0, &mut magic)?;
        if &magic != MAGIC {
            return Err(ShmError::BadMagic(magic));
        }

        let mut version_bytes = [0u8; 4];
        probe_region.read_bytes(VERSION_OFFSET, &mut version_bytes)?;
        let version = u32::from_le_bytes(version_bytes);
        if version != VERSION {
            return Err(ShmError::BadVersion {
                expected: VERSION,
                got: version,
            });
        }

        let mut slot_count_bytes = [0u8; 4];
        probe_region.read_bytes(SLOT_COUNT_OFFSET, &mut slot_count_bytes)?;
        let slot_count = u32::from_le_bytes(slot_count_bytes);

        let mut slot_size_bytes = [0u8; 8];
        probe_region.read_bytes(SLOT_SIZE_OFFSET, &mut slot_size_bytes)?;
        let slot_size = u64::from_le_bytes(slot_size_bytes);

        let config = RingConfig::new(slot_count, slot_size)?;
        drop(probe_region);

        let region_size = config.region_size();
        let region = ShmRegion::open(name, region_size)?;

        let nettics = region.atomic_load_u64(NETTICS_OFFSET)?;

        Ok(Self {
            region,
            config,
            nettics,
        })
    }

    pub fn try_consume(&self) -> Result<Option<ConsumedFrame>, ShmError> {
        let maketic = self.region.atomic_load_u64(MAKETIC_OFFSET)?;
        if maketic == self.nettics {
            return Ok(None);
        }

        let slot_offset = self.config.slot_offset(self.nettics);

        let mut header = vec![0u8; PROBE_FRAME_HEADER_SIZE];
        self.region.read_bytes(slot_offset, &mut header)?;

        // Read and validate size from header
        let size = u64::from_le_bytes(
            header[FRAME_OFFSET_SIZE..FRAME_OFFSET_SIZE + 8]
                .try_into()
                .expect("8 bytes"),
        );

        let slot_data_cap = self.config.slot_data_capacity();
        if size as usize > slot_data_cap {
            return Err(ShmError::ReadOutOfBounds {
                offset: PROBE_FRAME_HEADER_SIZE,
                length: size as usize,
                capacity: slot_data_cap,
            });
        }

        // Validate generation field to detect stale/torn reads
        let generation = u32::from_le_bytes(
            header[FRAME_OFFSET_GENERATION..FRAME_OFFSET_GENERATION + 4]
                .try_into()
                .expect("4 bytes"),
        );
        let expected_generation = (self.nettics & 0xFFFF_FFFF) as u32;
        if generation != expected_generation {
            return Err(ShmError::StaleSlot {
                expected: expected_generation,
                actual: generation,
            });
        }

        let mut data = vec![0u8; size as usize];
        self.region
            .read_bytes(slot_offset + PROBE_FRAME_HEADER_SIZE, &mut data)?;

        Ok(Some(ConsumedFrame { header, data }))
    }

    pub fn advance(&mut self) -> Result<(), ShmError> {
        self.advance_by(1)
    }

    pub fn advance_by(&mut self, count: u64) -> Result<(), ShmError> {
        self.nettics += count;
        self.region.atomic_store_u64(NETTICS_OFFSET, self.nettics)
    }

    pub fn nettics(&self) -> u64 {
        self.nettics
    }

    pub fn config(&self) -> &RingConfig {
        &self.config
    }

    pub fn read_slot_bytes(
        &self,
        slot_maketic: u64,
        offset_in_slot: usize,
        len: usize,
    ) -> Result<Vec<u8>, ShmError> {
        let slot_offset = self.config.slot_offset(slot_maketic);
        let mut buf = vec![0u8; len];
        self.region
            .read_bytes(slot_offset + offset_in_slot, &mut buf)?;
        Ok(buf)
    }

    pub fn read_absolute(&self, offset: usize, len: usize) -> Result<Vec<u8>, ShmError> {
        let mut buf = vec![0u8; len];
        self.region.read_bytes(offset, &mut buf)?;
        Ok(buf)
    }

    /// Returns a slice into a slot in the shared memory region.
    ///
    /// # Safety
    ///
    /// Caller must ensure no concurrent writes to the requested range
    /// within the slot.
    pub unsafe fn slot_as_slice(
        &self,
        slot_maketic: u64,
        offset_in_slot: usize,
        len: usize,
    ) -> Result<&[u8], ShmError> {
        let slot_offset = self.config.slot_offset(slot_maketic);
        // SAFETY: caller is responsible for ensuring no concurrent writes.
        unsafe { self.region.as_slice(slot_offset + offset_in_slot, len) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PROBE_FRAME_HEADER_SIZE;

    fn test_ring_name() -> String {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        format!("/rs-r-{}-{n}", std::process::id())
    }

    fn small_config() -> RingConfig {
        RingConfig::new(4, 4096).unwrap()
    }

    fn make_header_with_size(size: u64) -> Vec<u8> {
        make_header(size, 0)
    }

    fn make_header(size: u64, generation: u32) -> Vec<u8> {
        let mut header = vec![0u8; PROBE_FRAME_HEADER_SIZE];
        header[FRAME_OFFSET_SIZE..FRAME_OFFSET_SIZE + 8].copy_from_slice(&size.to_le_bytes());
        header[FRAME_OFFSET_GENERATION..FRAME_OFFSET_GENERATION + 4]
            .copy_from_slice(&generation.to_le_bytes());
        header
    }

    #[test]
    fn producer_creates_valid_control_block() {
        let name = test_ring_name();
        let config = small_config();
        let producer = DoomRingProducer::create(&name, config).unwrap();
        assert_eq!(producer.maketic(), 0);
        assert!(!producer.is_full());
        drop(producer);
        ShmRegion::unlink(&name).unwrap();
    }

    #[test]
    fn consumer_validates_magic() {
        let name = test_ring_name();
        let config = small_config();
        let producer = DoomRingProducer::create(&name, config).unwrap();
        let consumer = DoomRingConsumer::open(&name).unwrap();
        assert_eq!(consumer.nettics(), 0);
        drop(consumer);
        drop(producer);
        ShmRegion::unlink(&name).unwrap();
    }

    #[test]
    fn consumer_rejects_bad_magic() {
        let name = test_ring_name();
        let config = small_config();
        let region_size = config.region_size();
        let region = ShmRegion::create(&name, region_size).unwrap();
        region.write_bytes(0, b"NOTADOOM").unwrap();
        drop(region);
        let err = DoomRingConsumer::open(&name).unwrap_err();
        assert!(matches!(err, ShmError::BadMagic(_)));
        ShmRegion::unlink(&name).unwrap();
    }

    #[test]
    fn publish_and_consume_round_trip() {
        let name = test_ring_name();
        let config = small_config();
        let mut producer = DoomRingProducer::create(&name, config).unwrap();
        let mut consumer = DoomRingConsumer::open(&name).unwrap();

        let data = vec![0xBB_u8; 100];
        // generation=0 matches consumer.nettics=0
        let header = make_header(data.len() as u64, 0);
        producer.publish(&header, &data).unwrap();

        assert_eq!(producer.maketic(), 1);

        let frame = consumer.try_consume().unwrap().unwrap();
        assert_eq!(&frame.header[..], &header[..]);
        assert_eq!(&frame.data[..], &data[..]);
        consumer.advance().unwrap();
        assert_eq!(consumer.nettics(), 1);

        drop(consumer);
        drop(producer);
        ShmRegion::unlink(&name).unwrap();
    }

    #[test]
    fn ring_full_detection() {
        let name = test_ring_name();
        let config = small_config();
        let mut producer = DoomRingProducer::create(&name, config).unwrap();

        let data = vec![0u8; 100];
        let header = make_header_with_size(data.len() as u64);
        for _ in 0..4 {
            producer.publish(&header, &data).unwrap();
        }
        assert!(producer.is_full());
        let err = producer.publish(&header, &data).unwrap_err();
        assert!(matches!(err, ShmError::RingFull { .. }));

        drop(producer);
        ShmRegion::unlink(&name).unwrap();
    }

    #[test]
    fn consume_empty_returns_none() {
        let name = test_ring_name();
        let config = small_config();
        let producer = DoomRingProducer::create(&name, config).unwrap();
        let consumer = DoomRingConsumer::open(&name).unwrap();
        assert!(consumer.try_consume().unwrap().is_none());
        drop(consumer);
        drop(producer);
        ShmRegion::unlink(&name).unwrap();
    }

    #[test]
    fn tensor_too_large_rejected() {
        let name = test_ring_name();
        let config = small_config();
        let mut producer = DoomRingProducer::create(&name, config).unwrap();

        let too_big = vec![0u8; 4096];
        let header = make_header_with_size(too_big.len() as u64);
        let err = producer.publish(&header, &too_big).unwrap_err();
        assert!(matches!(err, ShmError::TensorTooLarge { .. }));

        drop(producer);
        ShmRegion::unlink(&name).unwrap();
    }

    #[test]
    fn slots_wrap_around() {
        let name = test_ring_name();
        let config = small_config();
        let mut producer = DoomRingProducer::create(&name, config).unwrap();
        let mut consumer = DoomRingConsumer::open(&name).unwrap();

        let mut tick: u64 = 0;
        for round in 0u8..3 {
            for i in 0u8..4 {
                let data = vec![round * 10 + i; 50];
                let generation = (tick & 0xFFFF_FFFF) as u32;
                let header = make_header(data.len() as u64, generation);
                producer.publish(&header, &data).unwrap();
                tick += 1;
            }
            for i in 0u8..4 {
                let frame = consumer.try_consume().unwrap().unwrap();
                assert_eq!(frame.data[0], round * 10 + i);
                consumer.advance().unwrap();
            }
        }
        assert_eq!(producer.maketic(), 12);
        assert_eq!(consumer.nettics(), 12);

        drop(consumer);
        drop(producer);
        ShmRegion::unlink(&name).unwrap();
    }
}

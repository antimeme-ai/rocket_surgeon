pub const HEADER_SIZE: usize = 128;
const MAX_DIMS: usize = 8;

const OFFSET_RANK: usize = 0;
const OFFSET_LAYER: usize = 4;
const OFFSET_COMP_ID: usize = 8;
const OFFSET_DTYPE: usize = 10;
const OFFSET_NDIM: usize = 11;
const OFFSET_SHAPE: usize = 12;
const OFFSET_TICK_ID: usize = 44;
const OFFSET_OFFSET: usize = 52;
const OFFSET_SIZE: usize = 60;
const OFFSET_FLAGS: usize = 68;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbeFrameHeader {
    pub rank: u32,
    pub layer: u32,
    pub comp_id: u16,
    pub dtype: u8,
    pub ndim: u8,
    pub shape: [u32; MAX_DIMS],
    pub tick_id: u64,
    pub offset: u64,
    pub size: u64,
    pub flags: u32,
}

impl ProbeFrameHeader {
    pub fn serialize(&self) -> [u8; HEADER_SIZE] {
        let mut buf = [0u8; HEADER_SIZE];

        buf[OFFSET_RANK..OFFSET_RANK + 4].copy_from_slice(&self.rank.to_le_bytes());
        buf[OFFSET_LAYER..OFFSET_LAYER + 4].copy_from_slice(&self.layer.to_le_bytes());
        buf[OFFSET_COMP_ID..OFFSET_COMP_ID + 2].copy_from_slice(&self.comp_id.to_le_bytes());
        buf[OFFSET_DTYPE] = self.dtype;
        buf[OFFSET_NDIM] = self.ndim;

        for (i, &dim) in self.shape.iter().enumerate() {
            let start = OFFSET_SHAPE + i * 4;
            buf[start..start + 4].copy_from_slice(&dim.to_le_bytes());
        }

        buf[OFFSET_TICK_ID..OFFSET_TICK_ID + 8].copy_from_slice(&self.tick_id.to_le_bytes());
        buf[OFFSET_OFFSET..OFFSET_OFFSET + 8].copy_from_slice(&self.offset.to_le_bytes());
        buf[OFFSET_SIZE..OFFSET_SIZE + 8].copy_from_slice(&self.size.to_le_bytes());
        buf[OFFSET_FLAGS..OFFSET_FLAGS + 4].copy_from_slice(&self.flags.to_le_bytes());

        buf
    }

    pub fn parse(data: &[u8]) -> Result<Self, ProbeFrameError> {
        if data.len() < HEADER_SIZE {
            return Err(ProbeFrameError::BufferTooSmall {
                expected: HEADER_SIZE,
                got: data.len(),
            });
        }

        let rank = u32::from_le_bytes(data[OFFSET_RANK..OFFSET_RANK + 4].try_into().unwrap());
        let layer = u32::from_le_bytes(data[OFFSET_LAYER..OFFSET_LAYER + 4].try_into().unwrap());
        let comp_id =
            u16::from_le_bytes(data[OFFSET_COMP_ID..OFFSET_COMP_ID + 2].try_into().unwrap());
        let dtype = data[OFFSET_DTYPE];
        let ndim = data[OFFSET_NDIM];

        let mut shape = [0u32; MAX_DIMS];
        for (i, dim) in shape.iter_mut().enumerate() {
            let start = OFFSET_SHAPE + i * 4;
            *dim = u32::from_le_bytes(data[start..start + 4].try_into().unwrap());
        }

        let tick_id =
            u64::from_le_bytes(data[OFFSET_TICK_ID..OFFSET_TICK_ID + 8].try_into().unwrap());
        let offset = u64::from_le_bytes(data[OFFSET_OFFSET..OFFSET_OFFSET + 8].try_into().unwrap());
        let size = u64::from_le_bytes(data[OFFSET_SIZE..OFFSET_SIZE + 8].try_into().unwrap());
        let flags = u32::from_le_bytes(data[OFFSET_FLAGS..OFFSET_FLAGS + 4].try_into().unwrap());

        Ok(Self {
            rank,
            layer,
            comp_id,
            dtype,
            ndim,
            shape,
            tick_id,
            offset,
            size,
            flags,
        })
    }
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum ProbeFrameError {
    #[error("buffer too small: expected {expected} bytes, got {got}")]
    BufferTooSmall { expected: usize, got: usize },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_header() -> ProbeFrameHeader {
        ProbeFrameHeader {
            rank: 0,
            layer: 12,
            comp_id: 3,
            dtype: 2, // Float32
            ndim: 3,
            shape: [2, 4096, 4096, 0, 0, 0, 0, 0],
            tick_id: 42,
            offset: 0x1000,
            size: 2 * 4096 * 4096 * 4,
            flags: 0,
        }
    }

    #[test]
    fn header_size_is_128() {
        let header = sample_header();
        let bytes = header.serialize();
        assert_eq!(bytes.len(), 128);
    }

    #[test]
    fn round_trip() {
        let original = sample_header();
        let bytes = original.serialize();
        let parsed = ProbeFrameHeader::parse(&bytes).unwrap();
        assert_eq!(original, parsed);
    }

    #[test]
    fn reserved_bytes_are_zero() {
        let header = sample_header();
        let bytes = header.serialize();
        assert!(bytes[72..HEADER_SIZE].iter().all(|&b| b == 0));
    }

    #[test]
    fn endianness_is_little() {
        let header = ProbeFrameHeader {
            rank: 0x0102_0304,
            ..sample_header()
        };
        let bytes = header.serialize();
        assert_eq!(bytes[0], 0x04);
        assert_eq!(bytes[1], 0x03);
        assert_eq!(bytes[2], 0x02);
        assert_eq!(bytes[3], 0x01);
    }

    #[test]
    fn parse_too_small_buffer() {
        let err = ProbeFrameHeader::parse(&[0u8; 64]).unwrap_err();
        assert!(matches!(
            err,
            ProbeFrameError::BufferTooSmall {
                expected: 128,
                got: 64
            }
        ));
    }

    #[test]
    fn shape_padding_preserved() {
        let header = ProbeFrameHeader {
            ndim: 2,
            shape: [1024, 768, 0, 0, 0, 0, 0, 0],
            ..sample_header()
        };
        let bytes = header.serialize();
        let parsed = ProbeFrameHeader::parse(&bytes).unwrap();
        assert_eq!(parsed.shape[0], 1024);
        assert_eq!(parsed.shape[1], 768);
        for &dim in &parsed.shape[2..] {
            assert_eq!(dim, 0);
        }
    }

    #[test]
    fn all_fields_at_correct_offsets() {
        let header = ProbeFrameHeader {
            rank: 7,
            layer: 31,
            comp_id: 256,
            dtype: 9,
            ndim: 1,
            shape: [42, 0, 0, 0, 0, 0, 0, 0],
            tick_id: 0xDEAD_BEEF_CAFE_BABE,
            offset: 0x1234_5678_9ABC_DEF0,
            size: 0xFEDC_BA98_7654_3210,
            flags: 0xABCD_EF01,
        };
        let bytes = header.serialize();

        assert_eq!(
            u32::from_le_bytes(bytes[0..4].try_into().unwrap()),
            7,
            "rank"
        );
        assert_eq!(
            u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
            31,
            "layer"
        );
        assert_eq!(
            u16::from_le_bytes(bytes[8..10].try_into().unwrap()),
            256,
            "comp_id"
        );
        assert_eq!(bytes[10], 9, "dtype");
        assert_eq!(bytes[11], 1, "ndim");
        assert_eq!(
            u32::from_le_bytes(bytes[12..16].try_into().unwrap()),
            42,
            "shape[0]"
        );
        assert_eq!(
            u64::from_le_bytes(bytes[44..52].try_into().unwrap()),
            0xDEAD_BEEF_CAFE_BABE,
            "tick_id"
        );
        assert_eq!(
            u64::from_le_bytes(bytes[52..60].try_into().unwrap()),
            0x1234_5678_9ABC_DEF0,
            "offset"
        );
        assert_eq!(
            u64::from_le_bytes(bytes[60..68].try_into().unwrap()),
            0xFEDC_BA98_7654_3210,
            "size"
        );
        assert_eq!(
            u32::from_le_bytes(bytes[68..72].try_into().unwrap()),
            0xABCD_EF01,
            "flags"
        );
    }
}

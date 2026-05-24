use std::cell::RefCell;
use std::collections::HashMap;

pub const SLOT_MAGIC: u32 = 0x434B_5054; // "CKPT"
pub const SLOT_HEADER_SIZE: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DtypeTag {
    Float16 = 0,
    Bfloat16 = 1,
    Float32 = 2,
    Float64 = 3,
}

impl DtypeTag {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Float16),
            1 => Some(Self::Bfloat16),
            2 => Some(Self::Float32),
            3 => Some(Self::Float64),
            _ => None,
        }
    }

    pub fn from_torch_str(s: &str) -> Option<Self> {
        match s {
            "torch.float16" => Some(Self::Float16),
            "torch.bfloat16" => Some(Self::Bfloat16),
            "torch.float32" => Some(Self::Float32),
            "torch.float64" => Some(Self::Float64),
            _ => None,
        }
    }

    pub fn to_torch_str(self) -> &'static str {
        match self {
            Self::Float16 => "torch.float16",
            Self::Bfloat16 => "torch.bfloat16",
            Self::Float32 => "torch.float32",
            Self::Float64 => "torch.float64",
        }
    }
}

#[derive(Debug, Clone)]
pub struct SlotHeader {
    pub magic: u32,
    pub dtype: DtypeTag,
    pub ndim: u8,
    pub shape: [u64; 6],
    pub byte_len: u64,
}

impl SlotHeader {
    pub fn write_to(&self, buf: &mut [u8; SLOT_HEADER_SIZE]) {
        buf[0..4].copy_from_slice(&self.magic.to_le_bytes());
        buf[4] = self.dtype as u8;
        buf[5] = self.ndim;
        buf[6..8].copy_from_slice(&[0u8; 2]);
        for (i, &dim) in self.shape.iter().enumerate() {
            let offset = 8 + i * 8;
            buf[offset..offset + 8].copy_from_slice(&dim.to_le_bytes());
        }
        buf[56..64].copy_from_slice(&self.byte_len.to_le_bytes());
    }

    pub fn read_from(buf: &[u8; SLOT_HEADER_SIZE]) -> Self {
        let magic = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
        let dtype = DtypeTag::from_u8(buf[4]).unwrap_or(DtypeTag::Float32);
        let ndim = buf[5];
        let mut shape = [0u64; 6];
        for (i, dim) in shape.iter_mut().enumerate() {
            let offset = 8 + i * 8;
            *dim = u64::from_le_bytes(buf[offset..offset + 8].try_into().unwrap());
        }
        let byte_len = u64::from_le_bytes(buf[56..64].try_into().unwrap());
        Self {
            magic,
            dtype,
            ndim,
            shape,
            byte_len,
        }
    }
}

pub fn align_up(val: usize, align: usize) -> usize {
    (val + align - 1) & !(align - 1)
}

struct SlotDescriptor {
    offset: usize,
    checkpoint_id: String,
    layer_idx: u32,
}

#[derive(Clone, Copy)]
pub struct ArenaSnapshot {
    free_count: usize,
}

pub struct CheckpointArena {
    ptr: *mut u8,
    capacity: usize,
    slot_size: usize,
    num_slots: usize,
    inner: RefCell<ArenaInner>,
}

struct ArenaInner {
    free_list: Vec<usize>,
    slots: Vec<SlotDescriptor>,
    index: HashMap<(String, u32), usize>,
    checkpoint_slots: HashMap<String, Vec<usize>>,
}

// SAFETY: The arena is single-threaded (worker dispatch loop is serial).
// The mmap'd region is accessed only through &self methods that use
// RefCell for interior mutability of bookkeeping.
unsafe impl Send for CheckpointArena {}

impl CheckpointArena {
    pub fn new(slot_size: usize, num_slots: usize) -> anyhow::Result<Self> {
        assert!(slot_size >= SLOT_HEADER_SIZE);
        let capacity = slot_size * num_slots;
        if capacity == 0 {
            anyhow::bail!("arena capacity must be non-zero");
        }

        #[cfg(target_os = "linux")]
        let flags = libc::MAP_ANONYMOUS | libc::MAP_PRIVATE | libc::MAP_POPULATE;
        #[cfg(not(target_os = "linux"))]
        let flags = libc::MAP_ANONYMOUS | libc::MAP_PRIVATE;

        // SAFETY: mmap with MAP_ANONYMOUS creates a fresh zero-filled mapping.
        // No file descriptor needed (-1). We check MAP_FAILED before use.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                capacity,
                libc::PROT_READ | libc::PROT_WRITE,
                flags,
                -1,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            anyhow::bail!(
                "mmap failed for checkpoint arena ({capacity} bytes): {}",
                std::io::Error::last_os_error()
            );
        }

        let free_list: Vec<usize> = (0..num_slots).rev().collect();

        Ok(Self {
            ptr: ptr.cast::<u8>(),
            capacity,
            slot_size,
            num_slots,
            inner: RefCell::new(ArenaInner {
                free_list,
                slots: Vec::new(),
                index: HashMap::new(),
                checkpoint_slots: HashMap::new(),
            }),
        })
    }

    pub fn slot_size(&self) -> usize {
        self.slot_size
    }

    pub fn num_slots(&self) -> usize {
        self.num_slots
    }

    pub fn available(&self) -> usize {
        self.inner.borrow().free_list.len()
    }

    pub fn base_ptr(&self) -> (*mut u8, usize) {
        (self.ptr, self.capacity)
    }

    pub fn alloc_slot(
        &self,
        checkpoint_id: &str,
        layer_idx: u32,
    ) -> anyhow::Result<(*mut u8, SlotHeader)> {
        let mut inner = self.inner.borrow_mut();
        let slot_idx = inner
            .free_list
            .pop()
            .ok_or_else(|| anyhow::anyhow!("checkpoint arena exhausted"))?;
        let offset = slot_idx * self.slot_size;

        let header = SlotHeader {
            magic: SLOT_MAGIC,
            dtype: DtypeTag::Float32,
            ndim: 0,
            shape: [0; 6],
            byte_len: (self.slot_size - SLOT_HEADER_SIZE) as u64,
        };

        // SAFETY: offset is slot_idx * slot_size, both bounded by capacity.
        // ptr was returned by mmap and is valid for capacity bytes.
        let slot_ptr = unsafe { self.ptr.add(offset) };
        let mut hdr_buf = [0u8; SLOT_HEADER_SIZE];
        header.write_to(&mut hdr_buf);
        // SAFETY: slot_ptr points to a valid slot within the mmap region.
        unsafe {
            std::ptr::copy_nonoverlapping(hdr_buf.as_ptr(), slot_ptr, SLOT_HEADER_SIZE);
        }

        let desc_idx = inner.slots.len();
        inner.slots.push(SlotDescriptor {
            offset,
            checkpoint_id: checkpoint_id.to_owned(),
            layer_idx,
        });
        inner
            .index
            .insert((checkpoint_id.to_owned(), layer_idx), desc_idx);
        inner
            .checkpoint_slots
            .entry(checkpoint_id.to_owned())
            .or_default()
            .push(slot_idx);

        Ok((slot_ptr, header))
    }

    pub fn get_slot(&self, checkpoint_id: &str, layer_idx: u32) -> Option<(*const u8, SlotHeader)> {
        let inner = self.inner.borrow();
        let &desc_idx = inner.index.get(&(checkpoint_id.to_owned(), layer_idx))?;
        let desc = &inner.slots[desc_idx];
        // SAFETY: desc.offset was computed from a valid slot index during alloc.
        let slot_ptr = unsafe { self.ptr.add(desc.offset) };
        let mut hdr_buf = [0u8; SLOT_HEADER_SIZE];
        // SAFETY: slot_ptr is within the mmap region, SLOT_HEADER_SIZE fits in any slot.
        unsafe {
            std::ptr::copy_nonoverlapping(slot_ptr, hdr_buf.as_mut_ptr(), SLOT_HEADER_SIZE);
        }
        let header = SlotHeader::read_from(&hdr_buf);
        Some((slot_ptr, header))
    }

    /// # Safety
    ///
    /// `slot_ptr` must point to a valid slot within this arena's mmap region.
    #[allow(clippy::unused_self)]
    pub unsafe fn update_header(&self, slot_ptr: *mut u8, header: &SlotHeader) {
        let mut hdr_buf = [0u8; SLOT_HEADER_SIZE];
        header.write_to(&mut hdr_buf);
        // SAFETY: caller guarantees slot_ptr is within the arena.
        unsafe {
            std::ptr::copy_nonoverlapping(hdr_buf.as_ptr(), slot_ptr, SLOT_HEADER_SIZE);
        }
    }

    pub fn free_checkpoint(&self, checkpoint_id: &str) {
        let mut inner = self.inner.borrow_mut();
        if let Some(slot_indices) = inner.checkpoint_slots.remove(checkpoint_id) {
            for &slot_idx in &slot_indices {
                inner.free_list.push(slot_idx);
            }
        }
        inner.index.retain(|k, _| k.0 != checkpoint_id);
    }

    pub fn snapshot(&self) -> ArenaSnapshot {
        ArenaSnapshot {
            free_count: self.inner.borrow().free_list.len(),
        }
    }

    pub fn rollback(&self, snap: ArenaSnapshot, checkpoint_id: &str) {
        self.free_checkpoint(checkpoint_id);
        debug_assert!(self.inner.borrow().free_list.len() >= snap.free_count);
    }

    pub fn oldest_checkpoint(&self) -> Option<String> {
        let inner = self.inner.borrow();
        inner.checkpoint_slots.keys().next().cloned()
    }

    pub fn slot_info_for_checkpoint(&self, checkpoint_id: &str) -> Vec<(u32, usize, SlotHeader)> {
        let inner = self.inner.borrow();
        inner
            .slots
            .iter()
            .filter(|d| d.checkpoint_id == checkpoint_id)
            .map(|d| {
                // SAFETY: d.offset was computed from a valid slot index during alloc.
                let slot_ptr = unsafe { self.ptr.add(d.offset) };
                let mut hdr_buf = [0u8; SLOT_HEADER_SIZE];
                // SAFETY: slot_ptr is within the mmap region.
                unsafe {
                    std::ptr::copy_nonoverlapping(slot_ptr, hdr_buf.as_mut_ptr(), SLOT_HEADER_SIZE);
                }
                let header = SlotHeader::read_from(&hdr_buf);
                (d.layer_idx, d.offset, header)
            })
            .collect()
    }
}

impl Drop for CheckpointArena {
    fn drop(&mut self) {
        if !self.ptr.is_null() && self.capacity > 0 {
            // SAFETY: ptr and capacity were set in new() from a successful mmap call.
            unsafe {
                libc::munmap(self.ptr.cast(), self.capacity);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// NVMe spill / load
// ---------------------------------------------------------------------------

use std::io::{Read, Write};
use std::path::Path;

const SPILL_MAGIC: &[u8; 8] = b"CKPTSPIL";
const SPILL_VERSION: u32 = 1;
const SPILL_INDEX_ENTRY_SIZE: usize = 80;

struct SpillIndexEntry {
    layer_idx: u32,
    dtype: DtypeTag,
    ndim: u8,
    shape: [u64; 6],
    data_offset: u64,
    data_len: u64,
    crc32: u32,
}

impl SpillIndexEntry {
    fn write_to(&self, buf: &mut [u8; SPILL_INDEX_ENTRY_SIZE]) {
        buf[0..4].copy_from_slice(&self.layer_idx.to_le_bytes());
        buf[4] = self.dtype as u8;
        buf[5] = self.ndim;
        buf[6..8].copy_from_slice(&[0u8; 2]);
        for (i, &dim) in self.shape.iter().enumerate() {
            let off = 8 + i * 8;
            buf[off..off + 8].copy_from_slice(&dim.to_le_bytes());
        }
        buf[56..64].copy_from_slice(&self.data_offset.to_le_bytes());
        buf[64..72].copy_from_slice(&self.data_len.to_le_bytes());
        buf[72..76].copy_from_slice(&self.crc32.to_le_bytes());
        buf[76..80].copy_from_slice(&[0u8; 4]);
    }

    fn read_from(buf: &[u8; SPILL_INDEX_ENTRY_SIZE]) -> Self {
        let layer_idx = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
        let dtype = DtypeTag::from_u8(buf[4]).unwrap_or(DtypeTag::Float32);
        let ndim = buf[5];
        let mut shape = [0u64; 6];
        for (i, dim) in shape.iter_mut().enumerate() {
            let off = 8 + i * 8;
            *dim = u64::from_le_bytes(buf[off..off + 8].try_into().unwrap());
        }
        let data_offset = u64::from_le_bytes(buf[56..64].try_into().unwrap());
        let data_len = u64::from_le_bytes(buf[64..72].try_into().unwrap());
        let crc32 = u32::from_le_bytes(buf[72..76].try_into().unwrap());
        Self {
            layer_idx,
            dtype,
            ndim,
            shape,
            data_offset,
            data_len,
            crc32,
        }
    }
}

pub fn spill_checkpoint(
    arena: &CheckpointArena,
    checkpoint_id: &str,
    dir: &Path,
) -> anyhow::Result<String> {
    let slot_infos = arena.slot_info_for_checkpoint(checkpoint_id);
    if slot_infos.is_empty() {
        anyhow::bail!("checkpoint {checkpoint_id} not found in arena");
    }

    let header_size = 8 + 4 + 4 + slot_infos.len() * SPILL_INDEX_ENTRY_SIZE;
    let mut data_offset = align_up(header_size, 64) as u64;
    let mut index_entries = Vec::with_capacity(slot_infos.len());
    let mut slot_data: Vec<Vec<u8>> = Vec::new();

    for &(layer_idx, offset, ref hdr) in &slot_infos {
        let data_len = hdr
            .byte_len
            .min((arena.slot_size - SLOT_HEADER_SIZE) as u64);
        let mut data = vec![0u8; data_len as usize];
        // SAFETY: offset is a valid slot offset within the arena mmap region.
        unsafe {
            std::ptr::copy_nonoverlapping(
                arena.ptr.add(offset + SLOT_HEADER_SIZE),
                data.as_mut_ptr(),
                data_len as usize,
            );
        }
        let crc = crc32fast::hash(&data);
        index_entries.push(SpillIndexEntry {
            layer_idx,
            dtype: hdr.dtype,
            ndim: hdr.ndim,
            shape: hdr.shape,
            data_offset,
            data_len,
            crc32: crc,
        });
        slot_data.push(data);
        data_offset += align_up(data_len as usize, 64) as u64;
    }

    let path = dir.join(format!("{checkpoint_id}.ckpt"));
    let mut file = std::fs::File::create(&path)?;

    file.write_all(SPILL_MAGIC)?;
    file.write_all(&SPILL_VERSION.to_le_bytes())?;
    file.write_all(&(index_entries.len() as u32).to_le_bytes())?;

    for entry in &index_entries {
        let mut buf = [0u8; SPILL_INDEX_ENTRY_SIZE];
        entry.write_to(&mut buf);
        file.write_all(&buf)?;
    }

    let current_pos = 8 + 4 + 4 + index_entries.len() * SPILL_INDEX_ENTRY_SIZE;
    let padding = align_up(current_pos, 64) - current_pos;
    if padding > 0 {
        file.write_all(&vec![0u8; padding])?;
    }

    for data in &slot_data {
        file.write_all(data)?;
        let pad = align_up(data.len(), 64) - data.len();
        if pad > 0 {
            file.write_all(&vec![0u8; pad])?;
        }
    }

    file.flush()?;
    arena.free_checkpoint(checkpoint_id);

    Ok(checkpoint_id.to_owned())
}

pub fn load_spilled_checkpoint(
    arena: &CheckpointArena,
    path: &Path,
    checkpoint_id: &str,
) -> anyhow::Result<()> {
    let mut file = std::fs::File::open(path)?;

    let mut magic = [0u8; 8];
    file.read_exact(&mut magic)?;
    if &magic != SPILL_MAGIC {
        anyhow::bail!("invalid spill file magic");
    }

    let mut version_buf = [0u8; 4];
    file.read_exact(&mut version_buf)?;
    let version = u32::from_le_bytes(version_buf);
    if version != SPILL_VERSION {
        anyhow::bail!("unsupported spill version {version}");
    }

    let mut count_buf = [0u8; 4];
    file.read_exact(&mut count_buf)?;
    let num_slots = u32::from_le_bytes(count_buf) as usize;

    let mut entries = Vec::with_capacity(num_slots);
    for _ in 0..num_slots {
        let mut buf = [0u8; SPILL_INDEX_ENTRY_SIZE];
        file.read_exact(&mut buf)?;
        entries.push(SpillIndexEntry::read_from(&buf));
    }

    let file_bytes = std::fs::read(path)?;

    for entry in &entries {
        let start = entry.data_offset as usize;
        let end = start + entry.data_len as usize;
        if end > file_bytes.len() {
            anyhow::bail!(
                "spill file truncated: need {} bytes, got {}",
                end,
                file_bytes.len()
            );
        }
        let data = &file_bytes[start..end];
        let actual_crc = crc32fast::hash(data);
        if actual_crc != entry.crc32 {
            anyhow::bail!(
                "CRC32 mismatch for layer {}: expected {:#010x}, got {:#010x}",
                entry.layer_idx,
                entry.crc32,
                actual_crc
            );
        }

        let (slot_ptr, _) = arena.alloc_slot(checkpoint_id, entry.layer_idx)?;
        let header = SlotHeader {
            magic: SLOT_MAGIC,
            dtype: entry.dtype,
            ndim: entry.ndim,
            shape: entry.shape,
            byte_len: entry.data_len,
        };
        // SAFETY: slot_ptr was just returned by alloc_slot and is valid.
        unsafe {
            arena.update_header(slot_ptr, &header);
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                slot_ptr.add(SLOT_HEADER_SIZE),
                data.len(),
            );
        }
    }

    Ok(())
}

pub use rocket_surgeon_protocol::checkpoint_layers;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_header_roundtrip() {
        let header = SlotHeader {
            magic: SLOT_MAGIC,
            dtype: DtypeTag::Bfloat16,
            ndim: 3,
            shape: [2, 1024, 4096, 0, 0, 0],
            byte_len: 2 * 1024 * 4096,
        };
        let mut buf = [0u8; SLOT_HEADER_SIZE];
        header.write_to(&mut buf);
        let parsed = SlotHeader::read_from(&buf);
        assert_eq!(parsed.magic, SLOT_MAGIC);
        assert_eq!(parsed.dtype, DtypeTag::Bfloat16);
        assert_eq!(parsed.ndim, 3);
        assert_eq!(parsed.shape[0], 2);
        assert_eq!(parsed.shape[1], 1024);
        assert_eq!(parsed.shape[2], 4096);
        assert_eq!(parsed.byte_len, 2 * 1024 * 4096);
    }

    #[test]
    fn slot_header_size_is_64() {
        assert_eq!(SLOT_HEADER_SIZE, 64);
    }

    #[test]
    fn dtype_tag_roundtrip() {
        for tag in [
            DtypeTag::Float16,
            DtypeTag::Bfloat16,
            DtypeTag::Float32,
            DtypeTag::Float64,
        ] {
            assert_eq!(DtypeTag::from_u8(tag as u8), Some(tag));
        }
        assert_eq!(DtypeTag::from_u8(99), None);
    }

    #[test]
    fn arena_new_and_available() {
        let arena = CheckpointArena::new(128, 4).unwrap();
        assert_eq!(arena.available(), 4);
        assert_eq!(arena.slot_size(), 128);
        assert_eq!(arena.num_slots(), 4);
    }

    #[test]
    fn arena_alloc_and_free() {
        let arena = CheckpointArena::new(128, 4).unwrap();
        assert_eq!(arena.available(), 4);

        let (ptr, _) = arena.alloc_slot("ckpt-1", 5).unwrap();
        assert!(!ptr.is_null());
        assert_eq!(arena.available(), 3);

        let (ptr2, _) = arena.alloc_slot("ckpt-1", 10).unwrap();
        assert!(!ptr2.is_null());
        assert_eq!(arena.available(), 2);

        arena.free_checkpoint("ckpt-1");
        assert_eq!(arena.available(), 4);
    }

    #[test]
    fn arena_alloc_exhaustion() {
        let arena = CheckpointArena::new(128, 2).unwrap();
        arena.alloc_slot("a", 0).unwrap();
        arena.alloc_slot("a", 1).unwrap();
        assert!(arena.alloc_slot("a", 2).is_err());
    }

    #[test]
    fn arena_get_slot_returns_written_data() {
        let arena = CheckpointArena::new(128, 4).unwrap();
        let (ptr, _) = arena.alloc_slot("ckpt-1", 5).unwrap();
        // SAFETY: ptr points to a valid slot, writing 64 bytes after header is within slot_size.
        unsafe {
            let data_ptr = ptr.add(SLOT_HEADER_SIZE);
            std::ptr::write_bytes(data_ptr, 0xAB, 64);
        }
        let (rptr, header) = arena.get_slot("ckpt-1", 5).unwrap();
        assert_eq!(header.magic, SLOT_MAGIC);
        // SAFETY: rptr points to a valid slot, reading after header is safe.
        unsafe {
            let data_ptr = rptr.add(SLOT_HEADER_SIZE);
            assert_eq!(*data_ptr, 0xAB);
        }
    }

    #[test]
    fn arena_get_slot_missing_returns_none() {
        let arena = CheckpointArena::new(128, 4).unwrap();
        assert!(arena.get_slot("nope", 0).is_none());
    }

    #[test]
    fn arena_checkpoint_slots_tracks_ownership() {
        let arena = CheckpointArena::new(128, 8).unwrap();
        arena.alloc_slot("a", 0).unwrap();
        arena.alloc_slot("a", 1).unwrap();
        arena.alloc_slot("b", 0).unwrap();
        assert_eq!(arena.available(), 5);

        arena.free_checkpoint("a");
        assert_eq!(arena.available(), 7);
        assert!(arena.get_slot("a", 0).is_none());
        assert!(arena.get_slot("b", 0).is_some());
    }

    #[test]
    fn arena_transactional_rollback() {
        let arena = CheckpointArena::new(128, 4).unwrap();
        let snap = arena.snapshot();
        arena.alloc_slot("ckpt-1", 0).unwrap();
        arena.alloc_slot("ckpt-1", 1).unwrap();
        assert_eq!(arena.available(), 2);
        arena.rollback(snap, "ckpt-1");
        assert_eq!(arena.available(), 4);
        assert!(arena.get_slot("ckpt-1", 0).is_none());
    }

    #[test]
    fn arena_ptr_returns_base_address() {
        let arena = CheckpointArena::new(128, 4).unwrap();
        let (ptr, len) = arena.base_ptr();
        assert!(!ptr.is_null());
        assert_eq!(len, 128 * 4);
    }

    #[test]
    fn spill_and_load_roundtrip() {
        let dir = std::env::temp_dir().join(format!("rs-ckpt-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let arena = CheckpointArena::new(128, 8).unwrap();

        let (ptr0, _) = arena.alloc_slot("ckpt-1", 5).unwrap();
        // SAFETY: ptr0 is valid, writing 64 bytes after header within slot_size.
        unsafe {
            std::ptr::write_bytes(ptr0.add(SLOT_HEADER_SIZE), 0xAA, 64);
            arena.update_header(
                ptr0,
                &SlotHeader {
                    magic: SLOT_MAGIC,
                    dtype: DtypeTag::Float16,
                    ndim: 2,
                    shape: [1, 64, 0, 0, 0, 0],
                    byte_len: 64,
                },
            );
        }

        let (ptr1, _) = arena.alloc_slot("ckpt-1", 10).unwrap();
        // SAFETY: ptr1 is valid, same reasoning.
        unsafe {
            std::ptr::write_bytes(ptr1.add(SLOT_HEADER_SIZE), 0xBB, 64);
            arena.update_header(
                ptr1,
                &SlotHeader {
                    magic: SLOT_MAGIC,
                    dtype: DtypeTag::Bfloat16,
                    ndim: 3,
                    shape: [1, 8, 8, 0, 0, 0],
                    byte_len: 64,
                },
            );
        }

        assert_eq!(arena.available(), 6);

        let spill_id = spill_checkpoint(&arena, "ckpt-1", &dir).unwrap();
        assert_eq!(spill_id, "ckpt-1");
        assert_eq!(arena.available(), 8);

        let path = dir.join("ckpt-1.ckpt");
        assert!(path.exists());

        load_spilled_checkpoint(&arena, &path, "ckpt-1-restored").unwrap();
        assert_eq!(arena.available(), 6);

        let (rptr0, hdr0) = arena.get_slot("ckpt-1-restored", 5).unwrap();
        assert_eq!(hdr0.dtype, DtypeTag::Float16);
        assert_eq!(hdr0.ndim, 2);
        // SAFETY: rptr0 is a valid slot pointer.
        unsafe {
            assert_eq!(*rptr0.add(SLOT_HEADER_SIZE), 0xAA);
        }

        let (rptr1, hdr1) = arena.get_slot("ckpt-1-restored", 10).unwrap();
        assert_eq!(hdr1.dtype, DtypeTag::Bfloat16);
        assert_eq!(hdr1.ndim, 3);
        // SAFETY: rptr1 is a valid slot pointer.
        unsafe {
            assert_eq!(*rptr1.add(SLOT_HEADER_SIZE), 0xBB);
        }

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn spill_file_detects_corruption() {
        let dir = std::env::temp_dir().join(format!("rs-ckpt-corrupt-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let arena = CheckpointArena::new(128, 4).unwrap();
        let (ptr, _) = arena.alloc_slot("bad", 0).unwrap();
        // SAFETY: ptr is valid.
        unsafe {
            std::ptr::write_bytes(ptr.add(SLOT_HEADER_SIZE), 0xFF, 64);
            arena.update_header(
                ptr,
                &SlotHeader {
                    magic: SLOT_MAGIC,
                    dtype: DtypeTag::Float32,
                    ndim: 1,
                    shape: [16, 0, 0, 0, 0, 0],
                    byte_len: 64,
                },
            );
        }

        spill_checkpoint(&arena, "bad", &dir).unwrap();

        let path = dir.join("bad.ckpt");
        let mut bytes = std::fs::read(&path).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0x01;
        std::fs::write(&path, &bytes).unwrap();

        let result = load_spilled_checkpoint(&arena, &path, "bad-restored");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("CRC32"));

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn checkpoint_layers_32() {
        let layers = checkpoint_layers(32);
        assert_eq!(layers.len(), 5);
        assert_eq!(layers, vec![5, 10, 16, 21, 26]);
    }

    #[test]
    fn checkpoint_layers_80() {
        let layers = checkpoint_layers(80);
        assert_eq!(layers.len(), 8);
        assert_eq!(layers, vec![8, 17, 26, 35, 44, 53, 62, 71]);
    }

    #[test]
    fn checkpoint_layers_1_returns_empty() {
        assert!(checkpoint_layers(1).is_empty());
    }

    #[test]
    fn checkpoint_layers_2_returns_single() {
        let layers = checkpoint_layers(2);
        assert_eq!(layers, vec![1]);
    }

    #[test]
    fn checkpoint_layers_excludes_last() {
        for n in [16, 32, 48, 64, 80, 128] {
            let layers = checkpoint_layers(n);
            assert!(
                layers.iter().all(|&l| l < n - 1),
                "n={n}: layers {layers:?} should exclude last layer {}",
                n - 1
            );
        }
    }

    #[test]
    fn checkpoint_layers_excludes_zero() {
        for n in [16, 32, 48, 64, 80, 128] {
            let layers = checkpoint_layers(n);
            assert!(
                layers.iter().all(|&l| l > 0),
                "n={n}: layers {layers:?} should exclude layer 0"
            );
        }
    }
}

# Checkpoint State Tier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire up worker-side tensor capture and storage so the daemon's existing checkpoint metadata tier has actual activation data behind it via a zero-copy mmap arena with CUDA-pinned DMA.

**Architecture:** Rust-owned mmap arena in the worker process, registered with CUDA via `cudaHostRegister` for pinned DMA. Four Python bridge functions wrap arena pointers as PyTorch tensors — `copy_()` lands GPU data directly in Rust's address space. The `_host/checkpoint` dispatch handler connects the daemon's existing metadata tier to real tensor capture/restore. Auto-checkpointing fires at √L layer boundaries after each step.

**Tech Stack:** Rust (mmap, libc), PyO3, PyTorch (torch.frombuffer, torch.cuda), CRC32 (Rust `crc32fast` crate)

---

### Task 1: √L boundary selector — pure function, no dependencies

**Files:**
- Create: `crates/rocket-surgeon-worker/src/checkpoint.rs`
- Modify: `crates/rocket-surgeon-worker/src/main.rs` (add `mod checkpoint;`)

- [ ] **Step 1: Write the failing tests**

In `crates/rocket-surgeon-worker/src/checkpoint.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

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
```

- [ ] **Step 2: Write the implementation above the tests**

At the top of `crates/rocket-surgeon-worker/src/checkpoint.rs`:

```rust
pub fn checkpoint_layers(num_layers: u32) -> Vec<u32> {
    if num_layers <= 1 {
        return Vec::new();
    }
    let sqrt_l = (num_layers as f64).sqrt().ceil() as u32;
    let interval = num_layers as f64 / sqrt_l as f64;
    (1..sqrt_l)
        .map(|i| (i as f64 * interval).floor() as u32)
        .collect()
}
```

- [ ] **Step 3: Add `mod checkpoint;` to worker lib**

In `crates/rocket-surgeon-worker/src/main.rs`, find the existing `mod` declarations and add:

```rust
mod checkpoint;
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p rocket-surgeon-worker checkpoint`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add crates/rocket-surgeon-worker/src/checkpoint.rs crates/rocket-surgeon-worker/src/main.rs
git commit -m "feat(worker): √L checkpoint boundary selector"
```

---

### Task 2: Slot header types and serialization

**Files:**
- Modify: `crates/rocket-surgeon-worker/src/checkpoint.rs`

- [ ] **Step 1: Write the failing tests**

Append to the `tests` module in `checkpoint.rs`:

```rust
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
        for tag in [DtypeTag::Float16, DtypeTag::Bfloat16, DtypeTag::Float32, DtypeTag::Float64] {
            assert_eq!(DtypeTag::from_u8(tag as u8), Some(tag));
        }
        assert_eq!(DtypeTag::from_u8(99), None);
    }
```

- [ ] **Step 2: Write the types and serialization**

Add above `checkpoint_layers` in `checkpoint.rs`:

```rust
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
        buf[6..8].copy_from_slice(&[0u8; 2]); // reserved
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
        for i in 0..6 {
            let offset = 8 + i * 8;
            shape[i] = u64::from_le_bytes(buf[offset..offset + 8].try_into().unwrap());
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
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p rocket-surgeon-worker checkpoint`
Expected: All 9 tests PASS (6 from Task 1 + 3 new)

- [ ] **Step 4: Commit**

```bash
git add crates/rocket-surgeon-worker/src/checkpoint.rs
git commit -m "feat(worker): checkpoint slot header types and serialization"
```

---

### Task 3: CheckpointArena — mmap allocation and free-list

**Files:**
- Modify: `crates/rocket-surgeon-worker/src/checkpoint.rs`
- Modify: `crates/rocket-surgeon-worker/Cargo.toml` (add `libc` dependency)

- [ ] **Step 1: Add libc dependency**

In `crates/rocket-surgeon-worker/Cargo.toml`, add under `[dependencies]`:

```toml
libc.workspace = true
```

Verify `libc` is in the workspace `Cargo.toml` — run:
```bash
grep 'libc' Cargo.toml
```
If not present, add `libc = "0.2"` to `[workspace.dependencies]` in the root `Cargo.toml`.

- [ ] **Step 2: Write the failing tests**

Append to the `tests` module in `checkpoint.rs`:

```rust
    #[test]
    fn arena_new_and_available() {
        let slot_size = 128;
        let num_slots = 4;
        let arena = CheckpointArena::new(slot_size, num_slots).unwrap();
        assert_eq!(arena.available(), 4);
        assert_eq!(arena.slot_size(), 128);
        assert_eq!(arena.num_slots(), 4);
    }

    #[test]
    fn arena_alloc_and_free() {
        let slot_size = 128;
        let arena = CheckpointArena::new(slot_size, 4).unwrap();
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
        // Write test data into the data region (after the 64-byte header)
        unsafe {
            let data_ptr = ptr.add(SLOT_HEADER_SIZE);
            std::ptr::write_bytes(data_ptr, 0xAB, 64);
        }
        let (rptr, header) = arena.get_slot("ckpt-1", 5).unwrap();
        assert_eq!(header.magic, SLOT_MAGIC);
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
```

- [ ] **Step 3: Write the arena implementation**

Add to `checkpoint.rs` (after `SlotHeader`):

```rust
use std::cell::RefCell;
use std::collections::HashMap;

struct SlotDescriptor {
    offset: usize,
    checkpoint_id: String,
    layer_idx: u32,
}

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

        let mut flags = libc::MAP_ANONYMOUS | libc::MAP_PRIVATE;
        #[cfg(target_os = "linux")]
        {
            flags |= libc::MAP_POPULATE;
        }

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
            dtype: DtypeTag::Float32, // placeholder — caller overwrites after capture
            ndim: 0,
            shape: [0; 6],
            byte_len: (self.slot_size - SLOT_HEADER_SIZE) as u64,
        };

        let slot_ptr = unsafe { self.ptr.add(offset) };
        let mut hdr_buf = [0u8; SLOT_HEADER_SIZE];
        header.write_to(&mut hdr_buf);
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

    pub fn get_slot(
        &self,
        checkpoint_id: &str,
        layer_idx: u32,
    ) -> Option<(*const u8, SlotHeader)> {
        let inner = self.inner.borrow();
        let &desc_idx = inner
            .index
            .get(&(checkpoint_id.to_owned(), layer_idx))?;
        let desc = &inner.slots[desc_idx];
        let slot_ptr = unsafe { self.ptr.add(desc.offset) };
        let mut hdr_buf = [0u8; SLOT_HEADER_SIZE];
        unsafe {
            std::ptr::copy_nonoverlapping(slot_ptr, hdr_buf.as_mut_ptr(), SLOT_HEADER_SIZE);
        }
        let header = SlotHeader::read_from(&hdr_buf);
        Some((slot_ptr, header))
    }

    pub fn update_header(&self, slot_ptr: *mut u8, header: &SlotHeader) {
        let mut hdr_buf = [0u8; SLOT_HEADER_SIZE];
        header.write_to(&mut hdr_buf);
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
        inner
            .index
            .retain(|k, _| k.0 != checkpoint_id);
    }

    pub fn snapshot(&self) -> ArenaSnapshot {
        ArenaSnapshot {
            free_count: self.inner.borrow().free_list.len(),
        }
    }

    pub fn rollback(&self, snap: ArenaSnapshot, checkpoint_id: &str) {
        self.free_checkpoint(checkpoint_id);
        // free_checkpoint already restores slots to free_list.
        // Verify invariant: free_count should be at least snap level.
        debug_assert!(self.inner.borrow().free_list.len() >= snap.free_count);
    }

    pub fn oldest_checkpoint(&self) -> Option<String> {
        let inner = self.inner.borrow();
        inner
            .checkpoint_slots
            .keys()
            .next()
            .cloned()
    }
}

impl Drop for CheckpointArena {
    fn drop(&mut self) {
        if !self.ptr.is_null() && self.capacity > 0 {
            unsafe {
                libc::munmap(self.ptr.cast(), self.capacity);
            }
        }
    }
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p rocket-surgeon-worker checkpoint`
Expected: All 17 tests PASS (9 from Tasks 1-2 + 8 new)

- [ ] **Step 5: Commit**

```bash
git add crates/rocket-surgeon-worker/src/checkpoint.rs crates/rocket-surgeon-worker/Cargo.toml
git commit -m "feat(worker): CheckpointArena — mmap allocation with free-list pool"
```

---

### Task 4: NVMe spill and load

**Files:**
- Modify: `crates/rocket-surgeon-worker/src/checkpoint.rs`
- Modify: `crates/rocket-surgeon-worker/Cargo.toml` (add `crc32fast`)

- [ ] **Step 1: Add crc32fast dependency**

In root `Cargo.toml`, add to `[workspace.dependencies]`:
```toml
crc32fast = "1"
```

In `crates/rocket-surgeon-worker/Cargo.toml`, add under `[dependencies]`:
```toml
crc32fast.workspace = true
```

- [ ] **Step 2: Write the failing tests**

Append to the `tests` module in `checkpoint.rs`:

```rust
    #[test]
    fn spill_and_load_roundtrip() {
        let dir = std::env::temp_dir().join(format!("rs-ckpt-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let arena = CheckpointArena::new(128, 8).unwrap();

        // Alloc two slots for checkpoint "ckpt-1"
        let (ptr0, _) = arena.alloc_slot("ckpt-1", 5).unwrap();
        unsafe {
            let data = ptr0.add(SLOT_HEADER_SIZE);
            std::ptr::write_bytes(data, 0xAA, 64);
        }
        arena.update_header(ptr0, &SlotHeader {
            magic: SLOT_MAGIC,
            dtype: DtypeTag::Float16,
            ndim: 2,
            shape: [1, 64, 0, 0, 0, 0],
            byte_len: 64,
        });

        let (ptr1, _) = arena.alloc_slot("ckpt-1", 10).unwrap();
        unsafe {
            let data = ptr1.add(SLOT_HEADER_SIZE);
            std::ptr::write_bytes(data, 0xBB, 64);
        }
        arena.update_header(ptr1, &SlotHeader {
            magic: SLOT_MAGIC,
            dtype: DtypeTag::Bfloat16,
            ndim: 3,
            shape: [1, 8, 8, 0, 0, 0],
            byte_len: 64,
        });

        assert_eq!(arena.available(), 6);

        // Spill
        let spill_id = spill_checkpoint(&arena, "ckpt-1", &dir).unwrap();
        assert_eq!(spill_id, "ckpt-1");
        assert_eq!(arena.available(), 8); // slots freed

        // Verify file exists
        let path = dir.join("ckpt-1.ckpt");
        assert!(path.exists());

        // Load back
        load_spilled_checkpoint(&arena, &path, "ckpt-1-restored").unwrap();
        assert_eq!(arena.available(), 6);

        // Verify data
        let (rptr0, hdr0) = arena.get_slot("ckpt-1-restored", 5).unwrap();
        assert_eq!(hdr0.dtype, DtypeTag::Float16);
        assert_eq!(hdr0.ndim, 2);
        unsafe {
            assert_eq!(*rptr0.add(SLOT_HEADER_SIZE), 0xAA);
        }

        let (rptr1, hdr1) = arena.get_slot("ckpt-1-restored", 10).unwrap();
        assert_eq!(hdr1.dtype, DtypeTag::Bfloat16);
        assert_eq!(hdr1.ndim, 3);
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
        unsafe {
            std::ptr::write_bytes(ptr.add(SLOT_HEADER_SIZE), 0xFF, 64);
        }
        arena.update_header(ptr, &SlotHeader {
            magic: SLOT_MAGIC,
            dtype: DtypeTag::Float32,
            ndim: 1,
            shape: [16, 0, 0, 0, 0, 0],
            byte_len: 64,
        });

        spill_checkpoint(&arena, "bad", &dir).unwrap();

        // Corrupt one byte of the data
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
```

- [ ] **Step 3: Write the spill/load implementation**

Add to `checkpoint.rs`:

```rust
use std::io::{Read, Write};
use std::path::Path;

const SPILL_MAGIC: &[u8; 8] = b"CKPTSPIL";
const SPILL_VERSION: u32 = 1;
const SPILL_INDEX_ENTRY_SIZE: usize = 80;

#[derive(Debug)]
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
        for i in 0..6 {
            let off = 8 + i * 8;
            shape[i] = u64::from_le_bytes(buf[off..off + 8].try_into().unwrap());
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

fn align_up(val: usize, align: usize) -> usize {
    (val + align - 1) & !(align - 1)
}

pub fn spill_checkpoint(
    arena: &CheckpointArena,
    checkpoint_id: &str,
    dir: &Path,
) -> anyhow::Result<String> {
    let inner = arena.inner.borrow();
    let slot_indices = inner
        .checkpoint_slots
        .get(checkpoint_id)
        .ok_or_else(|| anyhow::anyhow!("checkpoint {checkpoint_id} not found in arena"))?;

    let descs: Vec<&SlotDescriptor> = inner
        .slots
        .iter()
        .filter(|d| d.checkpoint_id == checkpoint_id)
        .collect();

    let header_size = 8 + 4 + 4 + descs.len() * SPILL_INDEX_ENTRY_SIZE;
    let mut data_offset = align_up(header_size, 64) as u64;
    let mut index_entries = Vec::with_capacity(descs.len());
    let mut slot_data: Vec<(u64, Vec<u8>)> = Vec::new();

    for desc in &descs {
        let slot_ptr = unsafe { arena.ptr.add(desc.offset) };
        let mut hdr_buf = [0u8; SLOT_HEADER_SIZE];
        unsafe {
            std::ptr::copy_nonoverlapping(slot_ptr, hdr_buf.as_mut_ptr(), SLOT_HEADER_SIZE);
        }
        let hdr = SlotHeader::read_from(&hdr_buf);
        let data_len = hdr.byte_len.min((arena.slot_size - SLOT_HEADER_SIZE) as u64);
        let mut data = vec![0u8; data_len as usize];
        unsafe {
            std::ptr::copy_nonoverlapping(
                slot_ptr.add(SLOT_HEADER_SIZE),
                data.as_mut_ptr(),
                data_len as usize,
            );
        }
        let crc = crc32fast::hash(&data);
        index_entries.push(SpillIndexEntry {
            layer_idx: desc.layer_idx,
            dtype: hdr.dtype,
            ndim: hdr.ndim,
            shape: hdr.shape,
            data_offset,
            data_len,
            crc32: crc,
        });
        slot_data.push((data_offset, data));
        data_offset += align_up(data_len as usize, 64) as u64;
    }
    drop(inner);

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

    for (expected_offset, data) in &slot_data {
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
        arena.update_header(slot_ptr, &header);
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                slot_ptr.add(SLOT_HEADER_SIZE),
                data.len(),
            );
        }
    }

    Ok(())
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p rocket-surgeon-worker checkpoint`
Expected: All 19 tests PASS

- [ ] **Step 5: Commit**

```bash
git add crates/rocket-surgeon-worker/src/checkpoint.rs crates/rocket-surgeon-worker/Cargo.toml Cargo.toml
git commit -m "feat(worker): NVMe spill/load with CRC32 integrity checks"
```

---

### Task 5: Python checkpoint bridge

**Files:**
- Create: `python/rocket_surgeon/checkpoint.py`

- [ ] **Step 1: Write the Python bridge module**

Create `python/rocket_surgeon/checkpoint.py`:

```python
"""Checkpoint tensor capture/restore bridge.

Called from Rust worker via PyO3. Wraps arena pointers as PyTorch
tensors for zero-copy CUDA DMA. No logic, no state — just the
thinnest bridge to torch.

The arena memory is Rust-owned. Tensors created by frombuffer
must not escape these functions.
"""

from __future__ import annotations

import ctypes
import struct

import torch

from rocket_surgeon.bridge import _models


def _get_residual_stream(layer_idx: int, handle: int) -> torch.Tensor:
    model = _models[handle]
    layers = model.model.layers  # type: ignore[union-attr]
    hook_output = layers[layer_idx]._forward_hooks_result
    if hook_output is not None:
        return hook_output
    return layers[layer_idx].output[0] if hasattr(layers[layer_idx], "output") else layers[layer_idx](torch.empty(0))


def capture_activation(
    handle: int,
    layer_idx: int,
    dst_ptr: int,
    dst_len: int,
) -> tuple[str, list[int]]:
    """Copy residual stream at layer_idx into arena memory at dst_ptr.

    Returns (dtype_string, shape_list).
    """
    model = _models[handle]
    layers = model.model.layers  # type: ignore[union-attr]
    tensor = layers[layer_idx]._rs_last_output
    t = tensor.detach().contiguous()
    nbytes = t.nelement() * t.element_size()
    if nbytes > dst_len:
        msg = f"tensor {nbytes} bytes exceeds slot capacity {dst_len}"
        raise ValueError(msg)
    buf = (ctypes.c_byte * dst_len).from_address(dst_ptr)
    cpu_view = torch.frombuffer(buf, dtype=t.dtype).reshape(t.shape)
    cpu_view.copy_(t)
    if t.is_cuda:
        torch.cuda.synchronize()
    del cpu_view
    return (str(t.dtype), list(t.shape))


def restore_activation(
    handle: int,
    layer_idx: int,
    src_ptr: int,
    src_len: int,
    dtype_str: str,
    shape: list[int],
) -> None:
    """Copy activation from arena memory at src_ptr back to GPU."""
    torch_dtype = getattr(torch, dtype_str.replace("torch.", ""))
    nelement = 1
    for s in shape:
        nelement *= s
    nbytes = nelement * torch.tensor([], dtype=torch_dtype).element_size()
    buf = (ctypes.c_byte * nbytes).from_address(src_ptr)
    cpu_view = torch.frombuffer(buf, dtype=torch_dtype).reshape(shape)
    model = _models[handle]
    layers = model.model.layers  # type: ignore[union-attr]
    target = layers[layer_idx]._rs_last_output
    target.copy_(cpu_view)
    if target.is_cuda:
        torch.cuda.synchronize()
    del cpu_view


def register_cuda_pinned(ptr: int, size: int) -> bool:
    """Register mmap'd memory with CUDA for pinned DMA.

    Returns True on success, False if CUDA is unavailable or registration fails.
    """
    if not torch.cuda.is_available():
        return False
    result = torch.cuda.cudart().cudaHostRegister(ptr, size, 0)
    return result.value == 0 if hasattr(result, "value") else result == 0


def unregister_cuda_pinned(ptr: int) -> bool:
    """Unregister mmap'd memory from CUDA. Call before munmap."""
    if not torch.cuda.is_available():
        return False
    result = torch.cuda.cudart().cudaHostUnregister(ptr)
    return result.value == 0 if hasattr(result, "value") else result == 0


def capture_rng_state() -> bytes:
    """Capture CUDA RNG state for all devices as length-prefixed raw bytes."""
    if not torch.cuda.is_available():
        return struct.pack("<I", 0)
    parts: list[bytes] = []
    device_count = torch.cuda.device_count()
    parts.append(struct.pack("<I", device_count))
    for i in range(device_count):
        rng_bytes = torch.cuda.get_rng_state(i).numpy().tobytes()
        parts.append(struct.pack("<II", i, len(rng_bytes)))
        parts.append(rng_bytes)
    return b"".join(parts)


def restore_rng_state(state: bytes) -> None:
    """Restore CUDA RNG state from bytes captured by capture_rng_state."""
    offset = 0
    (device_count,) = struct.unpack_from("<I", state, offset)
    offset += 4
    for _ in range(device_count):
        device_id, length = struct.unpack_from("<II", state, offset)
        offset += 8
        rng_bytes = state[offset : offset + length]
        offset += length
        t = torch.frombuffer(bytearray(rng_bytes), dtype=torch.uint8)
        torch.cuda.set_rng_state(t, device_id)
```

- [ ] **Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('python/rocket_surgeon/checkpoint.py').read()); print('OK')"` from repo root.
Expected: `OK`

- [ ] **Step 3: Run ruff**

Run: `ruff check python/rocket_surgeon/checkpoint.py && ruff format python/rocket_surgeon/checkpoint.py`
Expected: No errors

- [ ] **Step 4: Commit**

```bash
git add python/rocket_surgeon/checkpoint.py
git commit -m "feat(python): checkpoint bridge — capture/restore activation + RNG"
```

---

### Task 6: Rust bridge wrappers for Python checkpoint functions

**Files:**
- Modify: `crates/rocket-surgeon-worker/src/bridge.rs`

- [ ] **Step 1: Add the Rust-side bridge functions**

Append to `crates/rocket-surgeon-worker/src/bridge.rs`:

```rust
pub fn register_arena_cuda(ptr: usize, size: usize) -> anyhow::Result<bool> {
    Python::with_gil(|py| {
        let ckpt = py.import("rocket_surgeon.checkpoint")?;
        let result: bool = ckpt
            .getattr("register_cuda_pinned")?
            .call1((ptr, size))?
            .extract()?;
        Ok(result)
    })
}

pub fn unregister_arena_cuda(ptr: usize) -> anyhow::Result<bool> {
    Python::with_gil(|py| {
        let ckpt = py.import("rocket_surgeon.checkpoint")?;
        let result: bool = ckpt
            .getattr("unregister_cuda_pinned")?
            .call1((ptr,))?
            .extract()?;
        Ok(result)
    })
}

pub fn capture_activation(
    handle: u64,
    layer_idx: u32,
    dst_ptr: usize,
    dst_len: usize,
) -> anyhow::Result<(String, Vec<i64>)> {
    Python::with_gil(|py| {
        let ckpt = py.import("rocket_surgeon.checkpoint")?;
        let result = ckpt
            .getattr("capture_activation")?
            .call1((handle, layer_idx, dst_ptr, dst_len))?;
        let tuple = result
            .downcast::<PyTuple>()
            .map_err(|e| anyhow::anyhow!("expected tuple: {e}"))?;
        let dtype: String = tuple.get_item(0)?.extract()?;
        let shape: Vec<i64> = tuple.get_item(1)?.extract()?;
        Ok((dtype, shape))
    })
}

pub fn restore_activation(
    handle: u64,
    layer_idx: u32,
    src_ptr: usize,
    src_len: usize,
    dtype: &str,
    shape: &[i64],
) -> anyhow::Result<()> {
    Python::with_gil(|py| {
        let ckpt = py.import("rocket_surgeon.checkpoint")?;
        let shape_list = PyList::new(py, shape)?;
        ckpt.getattr("restore_activation")?
            .call1((handle, layer_idx, src_ptr, src_len, dtype, shape_list))?;
        Ok(())
    })
}

pub fn capture_rng_state() -> anyhow::Result<Vec<u8>> {
    Python::with_gil(|py| {
        let ckpt = py.import("rocket_surgeon.checkpoint")?;
        let result: Vec<u8> = ckpt
            .getattr("capture_rng_state")?
            .call0()?
            .extract()?;
        Ok(result)
    })
}

pub fn restore_rng_state(state: &[u8]) -> anyhow::Result<()> {
    Python::with_gil(|py| {
        let ckpt = py.import("rocket_surgeon.checkpoint")?;
        ckpt.getattr("restore_rng_state")?
            .call1((state,))?;
        Ok(())
    })
}
```

- [ ] **Step 2: Verify compilation**

Run: `cargo check -p rocket-surgeon-worker`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add crates/rocket-surgeon-worker/src/bridge.rs
git commit -m "feat(worker): Rust bridge wrappers for Python checkpoint functions"
```

---

### Task 7: Worker dispatch handler for `_host/checkpoint`

**Files:**
- Modify: `crates/rocket-surgeon-worker/src/dispatch.rs`
- Modify: `crates/rocket-surgeon-worker/src/checkpoint.rs` (add to `WorkerState`)

- [ ] **Step 1: Add arena and checkpoint_layers to WorkerState**

In `crates/rocket-surgeon-worker/src/dispatch.rs`, add a field to `WorkerState`:

```rust
pub checkpoint_arena: Option<crate::checkpoint::CheckpointArena>,
```

And in `WorkerState::new()`, add:

```rust
checkpoint_arena: None,
```

Add the import at the top of dispatch.rs:

```rust
use rocket_surgeon_protocol::messages::{HostCheckpointRequest, HostCheckpointResponse};
```

- [ ] **Step 2: Initialize arena on attach**

In `handle_host_attach`, after the shm ring setup (after `state.shm_ring = shm_ring;`), add arena initialization:

```rust
    let sqrt_layers = crate::checkpoint::checkpoint_layers(info.num_layers);
    let num_checkpoint_slots = sqrt_layers.len() * 2 + 2; // 2 checkpoints + RNG slots
    let dtype_size = match req.dtype {
        Some(rocket_surgeon_protocol::types::DType::Float16)
        | Some(rocket_surgeon_protocol::types::DType::Bfloat16) => 2usize,
        _ => 4usize,
    };
    let max_seq_len: usize = config
        .raw
        .as_ref()
        .and_then(|v| v.get("max_position_embeddings"))
        .and_then(|v| v.as_u64())
        .unwrap_or(2048) as usize;
    let slot_data_size = info.hidden_dim as usize * max_seq_len * dtype_size;
    let slot_size = crate::checkpoint::SLOT_HEADER_SIZE + crate::checkpoint::align_up(slot_data_size, 64);

    // RS_CHECKPOINT_ARENA_MB overrides computed sizing
    let (final_slot_size, final_num_slots) = match std::env::var("RS_CHECKPOINT_ARENA_MB") {
        Ok(mb_str) => {
            if let Ok(mb) = mb_str.parse::<usize>() {
                let total = mb * 1024 * 1024;
                let ns = total / slot_size;
                tracing::info!("RS_CHECKPOINT_ARENA_MB={mb} -> {ns} slots of {slot_size} bytes");
                (slot_size, ns.max(2))
            } else {
                (slot_size, num_checkpoint_slots)
            }
        }
        Err(_) => (slot_size, num_checkpoint_slots),
    };

    match crate::checkpoint::CheckpointArena::new(final_slot_size, final_num_slots) {
        Ok(arena) => {
            let (ptr, len) = arena.base_ptr();
            match bridge::register_arena_cuda(ptr as usize, len) {
                Ok(true) => tracing::info!("checkpoint arena CUDA-pinned ({len} bytes, {final_num_slots} slots)"),
                Ok(false) => tracing::warn!("checkpoint arena unpinned (no CUDA)"),
                Err(e) => tracing::warn!("cudaHostRegister failed: {e}"),
            }
            state.checkpoint_arena = Some(arena);
        }
        Err(e) => tracing::warn!("checkpoint arena creation failed: {e}"),
    }
```

- [ ] **Step 3: Tear down arena on detach**

In `handle_host_detach`, before `match bridge::unload_model(...)`, add:

```rust
    if let Some(ref arena) = state.checkpoint_arena {
        let (ptr, _) = arena.base_ptr();
        if let Err(e) = bridge::unregister_arena_cuda(ptr as usize) {
            tracing::warn!("cudaHostUnregister failed: {e}");
        }
    }
    state.checkpoint_arena = None;
```

- [ ] **Step 4: Add dispatch arm and handler**

In the `dispatch` function, add before the `_ =>` fallthrough:

```rust
        internal::HOST_CHECKPOINT => handle_host_checkpoint(state, request),
```

Add the handler function:

```rust
fn handle_host_checkpoint(state: &mut WorkerState, request: &Request) -> Response {
    let req: HostCheckpointRequest = match parse_params(request) {
        Ok(r) => r,
        Err(resp) => return *resp,
    };

    let Some(handle) = state.model_handle else {
        return internal_error(request.id.clone(), "No model loaded".to_owned());
    };

    let Some(ref arena) = state.checkpoint_arena else {
        return internal_error(request.id.clone(), "No checkpoint arena".to_owned());
    };

    match req {
        HostCheckpointRequest::Create {
            model_handle,
            checkpoint_id,
            tier,
            tick_id,
            layer_idx: _,
        } => {
            if model_handle != handle {
                return internal_error(
                    request.id.clone(),
                    format!("model handle mismatch: expected {handle}, got {model_handle}"),
                );
            }

            let num_layers = state
                .component_map
                .as_ref()
                .map(|m| {
                    m.components
                        .iter()
                        .filter_map(|c| c.layer_index)
                        .max()
                        .unwrap_or(0)
                        + 1
                })
                .unwrap_or(32);

            let layers = crate::checkpoint::checkpoint_layers(num_layers);
            let snap = arena.snapshot();
            let mut total_bytes: u64 = 0;

            for &layer in &layers {
                let (slot_ptr, _) = match arena.alloc_slot(&checkpoint_id, layer) {
                    Ok(r) => r,
                    Err(e) => {
                        arena.rollback(snap, &checkpoint_id);
                        return internal_error(
                            request.id.clone(),
                            format!("arena alloc failed for layer {layer}: {e}"),
                        );
                    }
                };

                let data_ptr = unsafe { slot_ptr.add(crate::checkpoint::SLOT_HEADER_SIZE) };
                let data_len = arena.slot_size() - crate::checkpoint::SLOT_HEADER_SIZE;

                match bridge::capture_activation(handle, layer, data_ptr as usize, data_len) {
                    Ok((dtype_str, shape)) => {
                        let dtype = crate::checkpoint::DtypeTag::from_torch_str(&dtype_str)
                            .unwrap_or(crate::checkpoint::DtypeTag::Float32);
                        let mut shape_arr = [0u64; 6];
                        for (i, &s) in shape.iter().enumerate().take(6) {
                            shape_arr[i] = s as u64;
                        }
                        let byte_len = shape.iter().product::<i64>().unsigned_abs()
                            * match dtype {
                                crate::checkpoint::DtypeTag::Float16
                                | crate::checkpoint::DtypeTag::Bfloat16 => 2,
                                crate::checkpoint::DtypeTag::Float32 => 4,
                                crate::checkpoint::DtypeTag::Float64 => 8,
                            };
                        let header = crate::checkpoint::SlotHeader {
                            magic: crate::checkpoint::SLOT_MAGIC,
                            dtype,
                            ndim: shape.len().min(6) as u8,
                            shape: shape_arr,
                            byte_len,
                        };
                        arena.update_header(slot_ptr, &header);
                        total_bytes += byte_len;
                    }
                    Err(e) => {
                        arena.rollback(snap, &checkpoint_id);
                        return internal_error(
                            request.id.clone(),
                            format!("capture_activation failed for layer {layer}: {e}"),
                        );
                    }
                }
            }

            // Capture RNG state as a sentinel slot (layer_idx = u32::MAX)
            match bridge::capture_rng_state() {
                Ok(rng_bytes) => {
                    if let Ok((rng_ptr, _)) = arena.alloc_slot(&checkpoint_id, u32::MAX) {
                        let header = crate::checkpoint::SlotHeader {
                            magic: crate::checkpoint::SLOT_MAGIC,
                            dtype: crate::checkpoint::DtypeTag::Float32,
                            ndim: 1,
                            shape: [rng_bytes.len() as u64, 0, 0, 0, 0, 0],
                            byte_len: rng_bytes.len() as u64,
                        };
                        arena.update_header(rng_ptr, &header);
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                rng_bytes.as_ptr(),
                                rng_ptr.add(crate::checkpoint::SLOT_HEADER_SIZE),
                                rng_bytes.len(),
                            );
                        }
                    }
                }
                Err(e) => tracing::warn!("RNG state capture failed: {e}"),
            }

            // Spill oldest checkpoint if arena > 80% utilized
            let utilization = 1.0 - (arena.available() as f64 / arena.num_slots() as f64);
            if utilization > 0.8 {
                if let Some(oldest) = arena.oldest_checkpoint() {
                    if oldest.starts_with("auto-") {
                        let spill_dir = std::env::temp_dir().join("rocket-surgeon-spill");
                        std::fs::create_dir_all(&spill_dir).ok();
                        match crate::checkpoint::spill_checkpoint(arena, &oldest, &spill_dir) {
                            Ok(_) => tracing::info!("spilled auto-checkpoint {oldest}"),
                            Err(e) => tracing::warn!("spill failed for {oldest}: {e}"),
                        }
                    }
                }
            }

            let tier = match tier {
                rocket_surgeon_protocol::messages::CreateCheckpointTier::Activation => {
                    rocket_surgeon_protocol::types::CheckpointTier::Activation
                }
                rocket_surgeon_protocol::messages::CreateCheckpointTier::FullSnapshot => {
                    rocket_surgeon_protocol::types::CheckpointTier::FullSnapshot
                }
            };

            let resp = HostCheckpointResponse {
                checkpoint_id,
                tier,
                restored_to: None,
                bytes_captured: Some(total_bytes),
            };

            match serde_json::to_value(resp) {
                Ok(value) => Response::success(request.id.clone(), value),
                Err(e) => internal_error(request.id.clone(), format!("serialization failed: {e}")),
            }
        }
        HostCheckpointRequest::Restore {
            model_handle,
            checkpoint_id,
        } => {
            if model_handle != handle {
                return internal_error(
                    request.id.clone(),
                    format!("model handle mismatch: expected {handle}, got {model_handle}"),
                );
            }

            let inner = arena.inner.borrow();
            let slot_indices = match inner.checkpoint_slots.get(&checkpoint_id) {
                Some(s) => s.clone(),
                None => {
                    return internal_error(
                        request.id.clone(),
                        format!("checkpoint {checkpoint_id} not found in arena"),
                    );
                }
            };
            drop(inner);

            // Restore RNG state first
            if let Some((rng_ptr, rng_header)) = arena.get_slot(&checkpoint_id, u32::MAX) {
                let rng_len = rng_header.byte_len as usize;
                let mut rng_bytes = vec![0u8; rng_len];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        rng_ptr.add(crate::checkpoint::SLOT_HEADER_SIZE),
                        rng_bytes.as_mut_ptr(),
                        rng_len,
                    );
                }
                if let Err(e) = bridge::restore_rng_state(&rng_bytes) {
                    tracing::warn!("RNG state restore failed: {e}");
                }
            }

            // Restore activations
            let inner = arena.inner.borrow();
            for desc in inner.slots.iter().filter(|d| d.checkpoint_id == checkpoint_id && d.layer_idx != u32::MAX) {
                let slot_ptr = unsafe { arena.ptr.add(desc.offset) };
                let mut hdr_buf = [0u8; crate::checkpoint::SLOT_HEADER_SIZE];
                unsafe {
                    std::ptr::copy_nonoverlapping(slot_ptr, hdr_buf.as_mut_ptr(), crate::checkpoint::SLOT_HEADER_SIZE);
                }
                let header = crate::checkpoint::SlotHeader::read_from(&hdr_buf);
                let data_ptr = unsafe { slot_ptr.add(crate::checkpoint::SLOT_HEADER_SIZE) };
                let shape: Vec<i64> = header
                    .shape
                    .iter()
                    .take(header.ndim as usize)
                    .map(|&s| s as i64)
                    .collect();

                drop(inner);
                if let Err(e) = bridge::restore_activation(
                    handle,
                    desc.layer_idx,
                    data_ptr as usize,
                    header.byte_len as usize,
                    header.dtype.to_torch_str(),
                    &shape,
                ) {
                    return internal_error(
                        request.id.clone(),
                        format!("restore_activation failed for layer {}: {e}", desc.layer_idx),
                    );
                }
                // Re-borrow for next iteration — safe because dispatch is single-threaded
                // and restore_activation doesn't touch the arena.
                let inner_reborrow = arena.inner.borrow();
                // Need to restructure this loop to avoid borrow issues
            }

            let resp = HostCheckpointResponse {
                checkpoint_id,
                tier: rocket_surgeon_protocol::types::CheckpointTier::Activation,
                restored_to: None,
                bytes_captured: None,
            };

            match serde_json::to_value(resp) {
                Ok(value) => Response::success(request.id.clone(), value),
                Err(e) => internal_error(request.id.clone(), format!("serialization failed: {e}")),
            }
        }
    }
}
```

**Note:** The restore loop has a borrow issue with `arena.inner`. The implementing engineer should restructure by collecting all slot info into a `Vec` first (while holding the borrow), then iterating over the collected vec after dropping the borrow. Like this:

```rust
let restore_info: Vec<_> = {
    let inner = arena.inner.borrow();
    inner.slots.iter()
        .filter(|d| d.checkpoint_id == checkpoint_id && d.layer_idx != u32::MAX)
        .map(|d| {
            let slot_ptr = unsafe { arena.ptr.add(d.offset) };
            let mut hdr_buf = [0u8; crate::checkpoint::SLOT_HEADER_SIZE];
            unsafe { std::ptr::copy_nonoverlapping(slot_ptr, hdr_buf.as_mut_ptr(), crate::checkpoint::SLOT_HEADER_SIZE); }
            let header = crate::checkpoint::SlotHeader::read_from(&hdr_buf);
            (d.layer_idx, slot_ptr, header)
        })
        .collect()
};
// inner is dropped here

for (layer_idx, slot_ptr, header) in restore_info {
    // ... call bridge::restore_activation
}
```

- [ ] **Step 5: Write dispatch test**

Append to the `tests` module in `dispatch.rs`:

```rust
    #[test]
    fn dispatch_host_checkpoint_no_model_returns_error() {
        let mut state = make_state();
        let req = make_request(
            internal::HOST_CHECKPOINT,
            serde_json::json!({
                "action": "create",
                "model_handle": 1,
                "checkpoint_id": "test-ckpt",
                "tier": "activation",
                "tick_id": 5,
                "layer_idx": 3
            }),
        );
        let resp = dispatch(&mut state, &req);
        assert!(resp.error.is_some());
        assert!(
            resp.error
                .as_ref()
                .unwrap()
                .message
                .contains("No model loaded")
        );
    }

    #[test]
    fn dispatch_host_checkpoint_invalid_params() {
        let mut state = make_state();
        let req = make_request(
            internal::HOST_CHECKPOINT,
            serde_json::json!({"wrong_field": 42}),
        );
        let resp = dispatch(&mut state, &req);
        assert!(resp.error.is_some());
        assert_eq!(resp.error.as_ref().unwrap().code, INVALID_PARAMS);
    }
```

- [ ] **Step 6: Run tests**

Run: `cargo test -p rocket-surgeon-worker`
Expected: All tests PASS (including new dispatch tests)

- [ ] **Step 7: Run clippy**

Run: `cargo clippy -p rocket-surgeon-worker -- -D warnings`
Expected: No warnings

- [ ] **Step 8: Commit**

```bash
git add crates/rocket-surgeon-worker/src/dispatch.rs crates/rocket-surgeon-worker/src/checkpoint.rs
git commit -m "feat(worker): _host/checkpoint Create/Restore dispatch handler"
```

---

### Task 8: Orchestrator handle method for `_host/checkpoint`

**Files:**
- Modify: `crates/rocket-surgeon/src/orchestrator_handle.rs`

- [ ] **Step 1: Add the checkpoint method**

In `crates/rocket-surgeon/src/orchestrator_handle.rs`, add after the `export_env` method:

```rust
    pub fn checkpoint(
        &mut self,
        req: &HostCheckpointRequest,
    ) -> anyhow::Result<HostCheckpointResponse> {
        let id = self.next_id();
        let params = serde_json::to_value(req)?;
        let request = Request::new(RequestId::Number(id), internal::HOST_CHECKPOINT, params);

        self.send(&request)?;
        let response = self.recv()?;

        if let Some(err) = response.error {
            anyhow::bail!(
                "orchestrator checkpoint failed (code {}): {}",
                err.code,
                err.message
            );
        }

        let result = response
            .result
            .ok_or_else(|| anyhow::anyhow!("orchestrator checkpoint: missing result"))?;
        let host_resp: HostCheckpointResponse = serde_json::from_value(result)?;
        Ok(host_resp)
    }
```

Add the import at the top if not already present:

```rust
use rocket_surgeon_protocol::messages::{HostCheckpointRequest, HostCheckpointResponse};
```

- [ ] **Step 2: Verify compilation**

Run: `cargo check -p rocket-surgeon`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add crates/rocket-surgeon/src/orchestrator_handle.rs
git commit -m "feat(daemon): orchestrator handle method for _host/checkpoint"
```

---

### Task 9: Wire daemon `rocket/checkpoint` to worker via orchestrator

**Files:**
- Modify: `crates/rocket-surgeon/src/main.rs`
- Modify: `crates/rocket-surgeon/src/dispatch.rs` (daemon-side)

- [ ] **Step 1: Add `try_orchestrator_checkpoint` function**

In `crates/rocket-surgeon/src/main.rs`, add a new function (following the pattern of `try_orchestrator_step`):

```rust
fn try_orchestrator_checkpoint_create(
    orchestrator: &mut Option<OrchestratorHandle>,
    model_handle: Option<u64>,
    checkpoint_id: &str,
    tier: rocket_surgeon_protocol::messages::CreateCheckpointTier,
    tick_id: u64,
    layer_idx: u32,
) -> Option<rocket_surgeon_protocol::messages::HostCheckpointResponse> {
    let (orch, mh) = (orchestrator.as_mut()?, model_handle?);
    let host_req = rocket_surgeon_protocol::messages::HostCheckpointRequest::Create {
        model_handle: mh,
        checkpoint_id: checkpoint_id.to_owned(),
        tier,
        tick_id,
        layer_idx,
    };
    match orch.checkpoint(&host_req) {
        Ok(hr) => Some(hr),
        Err(e) => {
            warn!("orchestrator checkpoint create failed: {e}");
            None
        }
    }
}

fn try_orchestrator_checkpoint_restore(
    orchestrator: &mut Option<OrchestratorHandle>,
    model_handle: Option<u64>,
    checkpoint_id: &str,
) -> Option<rocket_surgeon_protocol::messages::HostCheckpointResponse> {
    let (orch, mh) = (orchestrator.as_mut()?, model_handle?);
    let host_req = rocket_surgeon_protocol::messages::HostCheckpointRequest::Restore {
        model_handle: mh,
        checkpoint_id: checkpoint_id.to_owned(),
    };
    match orch.checkpoint(&host_req) {
        Ok(hr) => Some(hr),
        Err(e) => {
            warn!("orchestrator checkpoint restore failed: {e}");
            None
        }
    }
}
```

- [ ] **Step 2: Update main loop checkpoint routing**

The daemon needs to coordinate checkpoint_id between the session metadata tier and the worker. The daemon generates the UUID, sends it to the worker, then passes the same id to `session.checkpoint_create_with_id()`. This avoids id mismatch.

The approach: refactor `handle_checkpoint` in the daemon-side `dispatch.rs` to accept the orchestrator handle and forward Create/Restore to the worker before touching session metadata. The `try_orchestrator_checkpoint_*` helper functions in `main.rs` (defined in Step 1) are NOT needed — the orchestrator call lives directly inside `handle_checkpoint`.

- [ ] **Step 3: Modify `session.checkpoint_create` to accept explicit id**

In `crates/rocket-surgeon/src/session.rs`, change `checkpoint_create`:

```rust
    pub fn checkpoint_create(
        &mut self,
        tier: Option<CreateCheckpointTier>,
    ) -> ResponseEnvelope<CheckpointResponse> {
        self.checkpoint_create_with_id(tier, None)
    }

    pub fn checkpoint_create_with_id(
        &mut self,
        tier: Option<CreateCheckpointTier>,
        explicit_id: Option<String>,
    ) -> ResponseEnvelope<CheckpointResponse> {
        let checkpoint_id = explicit_id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
```

Then move the existing body of `checkpoint_create` into `checkpoint_create_with_id`. The existing `checkpoint_create` becomes a thin wrapper. This preserves all existing call sites.

- [ ] **Step 4: Update daemon main loop**

In `crates/rocket-surgeon/src/main.rs`, update the checkpoint routing:

```rust
        } else if request.method == method::CHECKPOINT {
            handle_checkpoint(&mut session, &request, &mut orchestrator, model_handle)
```

And update `handle_checkpoint` in `crates/rocket-surgeon/src/dispatch.rs` to accept the orchestrator and forward Create/Restore:

```rust
pub fn handle_checkpoint(
    session: &mut Session,
    request: &Request,
    orchestrator: &mut Option<crate::orchestrator_handle::OrchestratorHandle>,
    model_handle: Option<u64>,
) -> Response {
    let req: CheckpointRequest = match parse_params(request) {
        Ok(r) => r,
        Err(e) => return invalid_params_response(request.id.clone(), &e),
    };

    if let Err(ref e) = session.require_stopped("rocket/checkpoint") {
        return session_error_to_response(request.id.clone(), e);
    }

    let result = match req {
        CheckpointRequest::Create { tier } => {
            let checkpoint_id = uuid::Uuid::new_v4().to_string();
            let tick_id = session.state().tick_id.unwrap_or(0);
            let layer = session.state().position.as_ref().map_or(0, |p| p.layer);
            let create_tier = tier.unwrap_or(CreateCheckpointTier::Activation);
            if let (Some(orch), Some(mh)) = (orchestrator.as_mut(), model_handle) {
                let host_req = HostCheckpointRequest::Create {
                    model_handle: mh,
                    checkpoint_id: checkpoint_id.clone(),
                    tier: create_tier,
                    tick_id,
                    layer_idx: layer,
                };
                if let Err(e) = orch.checkpoint(&host_req) {
                    tracing::warn!("worker checkpoint create failed: {e}");
                }
            }
            Ok(session.checkpoint_create_with_id(tier, Some(checkpoint_id)))
        }
        CheckpointRequest::Restore { checkpoint_id } => {
            if let (Some(orch), Some(mh)) = (orchestrator.as_mut(), model_handle) {
                let host_req = HostCheckpointRequest::Restore {
                    model_handle: mh,
                    checkpoint_id: checkpoint_id.clone(),
                };
                if let Err(e) = orch.checkpoint(&host_req) {
                    tracing::warn!("worker checkpoint restore failed: {e}");
                }
            }
            session.checkpoint_restore(&checkpoint_id)
        }
        CheckpointRequest::List {} => Ok(session.checkpoint_list()),
        CheckpointRequest::Delete { checkpoint_id } => session.checkpoint_delete(&checkpoint_id),
        CheckpointRequest::Bookmark { tick_id, name } => {
            Ok(session.checkpoint_bookmark(tick_id, &name))
        }
    };

    match result {
        Ok(envelope) => serialize_envelope(request.id.clone(), envelope),
        Err(ref e) => session_error_to_response(request.id.clone(), e),
    }
}
```

- [ ] **Step 5: Update all callers of handle_checkpoint**

In `main.rs`, update the call site to pass the new parameters.

- [ ] **Step 6: Run tests**

Run: `cargo test --workspace`
Expected: All tests PASS. The existing `checkpoint.feature` TCK scenarios should still pass since the daemon metadata tier is unchanged.

- [ ] **Step 7: Commit**

```bash
git add crates/rocket-surgeon/src/main.rs crates/rocket-surgeon/src/dispatch.rs crates/rocket-surgeon/src/session.rs crates/rocket-surgeon/src/orchestrator_handle.rs
git commit -m "feat(daemon): wire rocket/checkpoint Create/Restore through to worker"
```

---

### Task 10: Auto-checkpoint at √L boundaries after step

**Files:**
- Modify: `crates/rocket-surgeon/src/main.rs`
- Modify: `crates/rocket-surgeon/src/session.rs` (add `auto_checkpoint_layers`)

- [ ] **Step 1: Add checkpoint layer tracking to session**

In `crates/rocket-surgeon/src/session.rs`, add to the `Session` struct (near the `checkpoint_positions` field):

```rust
    auto_checkpoint_layers: Vec<u32>,
```

Initialize to empty in `new()`:

```rust
    auto_checkpoint_layers: Vec::new(),
```

Add a setter method:

```rust
    pub fn set_auto_checkpoint_layers(&mut self, layers: Vec<u32>) {
        self.auto_checkpoint_layers = layers;
    }

    pub fn auto_checkpoint_layers(&self) -> &[u32] {
        &self.auto_checkpoint_layers
    }
```

- [ ] **Step 2: Set auto-checkpoint layers on attach**

In `crates/rocket-surgeon/src/main.rs`, after a successful attach (where `session.attach(...)` is called), add:

```rust
    // Configure √L auto-checkpoint boundaries
    let ckpt_layers = rocket_surgeon_worker::checkpoint::checkpoint_layers(
        attach_resp.num_layers,
    );
    session.set_auto_checkpoint_layers(ckpt_layers);
```

Wait — the daemon crate can't import from the worker crate. The function `checkpoint_layers` is pure math — it should live in the protocol crate or be duplicated. Since the user's preference is minimal code, put it in the protocol crate as a utility.

Actually, better: put it in `rocket-surgeon-protocol` since it's a pure function of `num_layers`:

In `crates/rocket-surgeon-protocol/src/lib.rs`, add:

```rust
pub fn checkpoint_layers(num_layers: u32) -> Vec<u32> {
    if num_layers <= 1 {
        return Vec::new();
    }
    let sqrt_l = (num_layers as f64).sqrt().ceil() as u32;
    let interval = num_layers as f64 / sqrt_l as f64;
    (1..sqrt_l)
        .map(|i| (i as f64 * interval).floor() as u32)
        .collect()
}
```

Then the worker's `checkpoint.rs` re-exports it:
```rust
pub use rocket_surgeon_protocol::checkpoint_layers;
```

And the daemon uses it directly:
```rust
    let ckpt_layers = rocket_surgeon_protocol::checkpoint_layers(attach_resp.num_layers);
    session.set_auto_checkpoint_layers(ckpt_layers);
```

- [ ] **Step 3: Auto-checkpoint after step**

In `crates/rocket-surgeon/src/main.rs`, after the step handling (after `handle_step` returns), add auto-checkpoint logic:

```rust
        // After step: auto-checkpoint if we crossed a √L boundary
        if let Some(ref hr) = step_host_response {
            let current_layer = hr.position.layer;
            if session.auto_checkpoint_layers().contains(&current_layer) {
                let auto_id = format!("auto-{}", uuid::Uuid::new_v4());
                let tick_id = session.state().tick_id.unwrap_or(0);
                if let (Some(orch), Some(mh)) = (orchestrator.as_mut(), model_handle) {
                    let host_req = HostCheckpointRequest::Create {
                        model_handle: mh,
                        checkpoint_id: auto_id.clone(),
                        tier: CreateCheckpointTier::Activation,
                        tick_id,
                        layer_idx: current_layer,
                    };
                    if let Err(e) = orch.checkpoint(&host_req) {
                        tracing::debug!("auto-checkpoint failed: {e}");
                    } else {
                        session.checkpoint_create_with_id(
                            Some(CreateCheckpointTier::Activation),
                            Some(auto_id),
                        );
                        tracing::debug!(layer = current_layer, "auto-checkpoint captured");
                    }
                }
            }
        }
```

- [ ] **Step 4: Verify compilation**

Run: `cargo check --workspace`
Expected: No errors

- [ ] **Step 5: Run TCK tests**

Run: `cd python && python -m pytest tests/tck/test_checkpoint.py -v`
Expected: All 9 checkpoint scenarios still PASS (auto-checkpoint is transparent)

- [ ] **Step 6: Commit**

```bash
git add crates/rocket-surgeon-protocol/src/lib.rs crates/rocket-surgeon/src/session.rs crates/rocket-surgeon/src/main.rs crates/rocket-surgeon-worker/src/checkpoint.rs
git commit -m "feat(daemon): auto-checkpoint at √L layer boundaries after step"
```

---

### Task 11: Make `align_up` public and verify full workspace build

**Files:**
- Modify: `crates/rocket-surgeon-worker/src/checkpoint.rs` (ensure `align_up` is `pub`)

- [ ] **Step 1: Verify `align_up` visibility**

Ensure `align_up` in `checkpoint.rs` is `pub`:

```rust
pub fn align_up(val: usize, align: usize) -> usize {
    (val + align - 1) & !(align - 1)
}
```

This is needed by `dispatch.rs` when computing slot sizes.

- [ ] **Step 2: Full workspace build**

Run: `cargo build --workspace --exclude rocket-surgeon-python --exclude rocket-surgeon-worker`
Expected: No errors

Run: `cargo build -p rocket-surgeon-worker`
Expected: No errors (requires Python linkage via venv)

- [ ] **Step 3: Full workspace test**

Run: `cargo test --workspace --exclude rocket-surgeon-python --exclude rocket-surgeon-worker`
Expected: All tests PASS

Run: `cargo test -p rocket-surgeon-worker`
Expected: All tests PASS

- [ ] **Step 4: Run TCK test suite**

Run: `cd python && python -m pytest tests/tck/ -v --timeout=120`
Expected: 150+ passed, 178 deferred, 0 failed. Existing checkpoint scenarios still green.

- [ ] **Step 5: Commit (if any fixes needed)**

```bash
git add -A
git commit -m "fix(worker): workspace build fixes for checkpoint integration"
```

---

### Task 12: Final cleanup and push

**Files:**
- Verify all files are committed
- Push branch

- [ ] **Step 1: Run clippy on full workspace**

Run: `cargo clippy --workspace --exclude rocket-surgeon-python -- -D warnings`
Expected: No warnings

- [ ] **Step 2: Run ruff on all Python**

Run: `ruff check python/ && ruff format --check python/`
Expected: No issues

- [ ] **Step 3: Verify git status is clean**

Run: `git status`
Expected: Nothing to commit, working tree clean

- [ ] **Step 4: Push branch**

```bash
git push -u origin HEAD
```

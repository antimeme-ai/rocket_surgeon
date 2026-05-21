use std::ffi::CString;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::{MAGIC, MAGIC_OFFSET, ShmError};

#[cfg(target_os = "macos")]
pub const PSHMNAMLEN_MAX: usize = 30;
#[cfg(not(target_os = "macos"))]
pub const PSHMNAMLEN_MAX: usize = 255;

pub struct ShmRegion {
    ptr: *mut u8,
    len: usize,
    fd: i32,
    name: String,
}

impl std::fmt::Debug for ShmRegion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShmRegion")
            .field("name", &self.name)
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

// SAFETY: The mmap'd region is MAP_SHARED and all access is through
// atomic ops or explicit copy (no unsynchronized aliased writes).
unsafe impl Send for ShmRegion {}

impl ShmRegion {
    pub fn create(name: &str, size: usize) -> Result<Self, ShmError> {
        if size == 0 {
            return Err(ShmError::InvalidConfig(
                "region size must be non-zero".into(),
            ));
        }
        validate_name(name)?;
        let c_name = CString::new(name).expect("shm name must not contain null bytes");

        // SAFETY: POSIX shm_open with O_CREAT|O_EXCL creates a new region.
        let fd = unsafe {
            libc::shm_open(
                c_name.as_ptr(),
                libc::O_CREAT | libc::O_RDWR | libc::O_EXCL,
                0o600,
            )
        };
        if fd < 0 {
            return Err(ShmError::Open {
                name: name.to_owned(),
                source: std::io::Error::last_os_error(),
            });
        }

        // SAFETY: fd is valid from shm_open above.
        if unsafe { libc::ftruncate(fd, size as libc::off_t) } != 0 {
            let err = std::io::Error::last_os_error();
            // SAFETY: fd is valid.
            unsafe {
                libc::close(fd);
            }
            // SAFETY: c_name is a valid CString.
            let _ = unsafe { libc::shm_unlink(c_name.as_ptr()) };
            return Err(ShmError::Truncate(err));
        }

        // SAFETY: fd is valid, size > 0, MAP_SHARED for cross-process visibility.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            let err = std::io::Error::last_os_error();
            // SAFETY: fd is valid.
            unsafe {
                libc::close(fd);
            }
            // SAFETY: c_name is a valid CString.
            let _ = unsafe { libc::shm_unlink(c_name.as_ptr()) };
            return Err(ShmError::Mmap(err));
        }

        Ok(Self {
            ptr: ptr.cast::<u8>(),
            len: size,
            fd,
            name: name.to_owned(),
        })
    }

    pub fn open(name: &str, size: usize) -> Result<Self, ShmError> {
        if size == 0 {
            return Err(ShmError::InvalidConfig(
                "region size must be non-zero".into(),
            ));
        }
        validate_name(name)?;
        let c_name = CString::new(name).expect("shm name must not contain null bytes");

        // SAFETY: POSIX shm_open with O_RDWR opens an existing region.
        let fd = unsafe { libc::shm_open(c_name.as_ptr(), libc::O_RDWR, 0) };
        if fd < 0 {
            return Err(ShmError::Open {
                name: name.to_owned(),
                source: std::io::Error::last_os_error(),
            });
        }

        // Validate actual region size via fstat to prevent SIGBUS
        // SAFETY: fd is valid from shm_open above, stat is zero-initialized.
        let actual_size = unsafe {
            let mut stat: libc::stat = std::mem::zeroed();
            if libc::fstat(fd, std::ptr::addr_of_mut!(stat)) != 0 {
                let err = std::io::Error::last_os_error();
                libc::close(fd);
                return Err(ShmError::Open {
                    name: name.to_owned(),
                    source: err,
                });
            }
            stat.st_size
        };
        if (actual_size as usize) < size {
            // SAFETY: fd is valid.
            unsafe {
                libc::close(fd);
            }
            return Err(ShmError::RegionTooSmall {
                expected: size,
                actual: actual_size as usize,
            });
        }

        // SAFETY: fd is valid, MAP_SHARED for cross-process visibility.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            let err = std::io::Error::last_os_error();
            // SAFETY: fd is valid.
            unsafe {
                libc::close(fd);
            }
            return Err(ShmError::Mmap(err));
        }

        Ok(Self {
            ptr: ptr.cast::<u8>(),
            len: size,
            fd,
            name: name.to_owned(),
        })
    }

    pub fn unlink(name: &str) -> Result<(), ShmError> {
        let c_name = CString::new(name).expect("shm name must not contain null bytes");
        // SAFETY: c_name is a valid CString for a POSIX shm region name.
        if unsafe { libc::shm_unlink(c_name.as_ptr()) } != 0 {
            return Err(ShmError::Unlink {
                name: name.to_owned(),
                source: std::io::Error::last_os_error(),
            });
        }
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn write_bytes(&self, offset: usize, data: &[u8]) -> Result<(), ShmError> {
        self.bounds_check(offset, data.len())?;
        // SAFETY: bounds_check verified offset+len is within the mmap region.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), self.ptr.add(offset), data.len());
        }
        Ok(())
    }

    pub fn read_bytes(&self, offset: usize, buf: &mut [u8]) -> Result<(), ShmError> {
        self.bounds_check(offset, buf.len())?;
        // SAFETY: bounds_check verified offset+len is within the mmap region.
        unsafe {
            std::ptr::copy_nonoverlapping(self.ptr.add(offset), buf.as_mut_ptr(), buf.len());
        }
        Ok(())
    }

    /// Returns a slice into the shared memory region.
    ///
    /// # Safety
    ///
    /// Caller must ensure no concurrent writes to the range
    /// `[offset, offset+len)` in the shared memory region.
    pub unsafe fn as_slice(&self, offset: usize, len: usize) -> Result<&[u8], ShmError> {
        self.bounds_check(offset, len)?;
        // SAFETY: bounds_check verified offset+len is within the mmap region.
        // Caller guarantees no concurrent writes to this range.
        Ok(unsafe { std::slice::from_raw_parts(self.ptr.add(offset), len) })
    }

    #[allow(clippy::cast_ptr_alignment)]
    pub fn atomic_store_u64(&self, offset: usize, value: u64) -> Result<(), ShmError> {
        Self::alignment_check(offset, 8)?;
        self.bounds_check(offset, 8)?;
        // SAFETY: alignment and bounds verified, AtomicU64 has same layout as u64.
        let atomic = unsafe { &*(self.ptr.add(offset).cast::<AtomicU64>()) };
        atomic.store(value, Ordering::Release);
        Ok(())
    }

    #[allow(clippy::cast_ptr_alignment)]
    pub fn atomic_load_u64(&self, offset: usize) -> Result<u64, ShmError> {
        Self::alignment_check(offset, 8)?;
        self.bounds_check(offset, 8)?;
        // SAFETY: alignment and bounds verified, AtomicU64 has same layout as u64.
        let atomic = unsafe { &*(self.ptr.add(offset).cast::<AtomicU64>()) };
        Ok(atomic.load(Ordering::Acquire))
    }

    pub fn write_magic(&self) -> Result<(), ShmError> {
        self.bounds_check(MAGIC_OFFSET, MAGIC.len())?;
        // SAFETY: bounds verified, writing MAGIC bytes as the init barrier.
        unsafe {
            std::ptr::copy_nonoverlapping(MAGIC.as_ptr(), self.ptr.add(MAGIC_OFFSET), MAGIC.len());
        }
        std::sync::atomic::fence(Ordering::Release);
        Ok(())
    }

    fn bounds_check(&self, offset: usize, length: usize) -> Result<(), ShmError> {
        if offset.checked_add(length).is_none_or(|end| end > self.len) {
            return Err(ShmError::OutOfBounds {
                offset,
                length,
                region_size: self.len,
            });
        }
        Ok(())
    }

    fn alignment_check(offset: usize, alignment: usize) -> Result<(), ShmError> {
        if !offset.is_multiple_of(alignment) {
            return Err(ShmError::Unaligned { offset, alignment });
        }
        Ok(())
    }
}

impl Drop for ShmRegion {
    fn drop(&mut self) {
        // SAFETY: ptr and len are from a successful mmap, fd from shm_open.
        unsafe {
            libc::munmap(self.ptr.cast(), self.len);
            libc::close(self.fd);
        }
    }
}

fn validate_name(name: &str) -> Result<(), ShmError> {
    if name.len() > PSHMNAMLEN_MAX {
        return Err(ShmError::NameTooLong {
            name: name.to_owned(),
            max_len: PSHMNAMLEN_MAX,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_region_name() -> String {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        format!("/rs-t-{}-{n}", std::process::id())
    }

    #[test]
    fn create_and_unlink() {
        let name = test_region_name();
        let region = ShmRegion::create(&name, 4096).unwrap();
        assert_eq!(region.len(), 4096);
        drop(region);
        ShmRegion::unlink(&name).unwrap();
    }

    #[test]
    fn create_then_open_sees_same_bytes() {
        let name = test_region_name();
        let creator = ShmRegion::create(&name, 4096).unwrap();
        creator.write_bytes(0, b"hello").unwrap();

        let opener = ShmRegion::open(&name, 4096).unwrap();
        let mut buf = [0u8; 5];
        opener.read_bytes(0, &mut buf).unwrap();
        assert_eq!(&buf, b"hello");

        drop(opener);
        drop(creator);
        ShmRegion::unlink(&name).unwrap();
    }

    #[test]
    fn write_bytes_out_of_bounds() {
        let name = test_region_name();
        let region = ShmRegion::create(&name, 64).unwrap();
        let err = region.write_bytes(60, &[0u8; 10]).unwrap_err();
        assert!(matches!(err, ShmError::OutOfBounds { .. }));
        drop(region);
        ShmRegion::unlink(&name).unwrap();
    }

    #[test]
    fn atomic_store_load_round_trip() {
        let name = test_region_name();
        let region = ShmRegion::create(&name, 4096).unwrap();
        region.atomic_store_u64(128, 0xDEAD_BEEF_CAFE_BABE).unwrap();
        let value = region.atomic_load_u64(128).unwrap();
        assert_eq!(value, 0xDEAD_BEEF_CAFE_BABE);
        drop(region);
        ShmRegion::unlink(&name).unwrap();
    }

    #[test]
    fn atomic_unaligned_offset_rejected() {
        let name = test_region_name();
        let region = ShmRegion::create(&name, 4096).unwrap();
        let err = region.atomic_store_u64(5, 42).unwrap_err();
        assert!(matches!(err, ShmError::Unaligned { .. }));
        drop(region);
        ShmRegion::unlink(&name).unwrap();
    }

    #[test]
    fn read_bytes_returns_correct_data() {
        let name = test_region_name();
        let region = ShmRegion::create(&name, 4096).unwrap();
        let data = [1u8, 2, 3, 4, 5, 6, 7, 8];
        region.write_bytes(100, &data).unwrap();
        let mut buf = [0u8; 8];
        region.read_bytes(100, &mut buf).unwrap();
        assert_eq!(buf, data);
        drop(region);
        ShmRegion::unlink(&name).unwrap();
    }

    #[test]
    fn name_too_long_rejected() {
        let long_name = format!("/{}", "a".repeat(PSHMNAMLEN_MAX));
        let err = ShmRegion::create(&long_name, 4096).unwrap_err();
        assert!(matches!(err, ShmError::NameTooLong { .. }));
    }

    #[test]
    fn write_magic_with_release_fence() {
        let name = test_region_name();
        let region = ShmRegion::create(&name, 4096).unwrap();
        region.write_magic().unwrap();
        let mut magic = [0u8; 8];
        region.read_bytes(0, &mut magic).unwrap();
        assert_eq!(&magic, b"DOOMRING");
        drop(region);
        ShmRegion::unlink(&name).unwrap();
    }

    #[test]
    fn as_slice_returns_valid_view() {
        let name = test_region_name();
        let region = ShmRegion::create(&name, 4096).unwrap();
        region.write_bytes(0, b"DOOMRING").unwrap();
        // SAFETY: single-threaded test, no concurrent writes.
        let slice = unsafe { region.as_slice(0, 8) }.unwrap();
        assert_eq!(slice, b"DOOMRING");
        drop(region);
        ShmRegion::unlink(&name).unwrap();
    }
}

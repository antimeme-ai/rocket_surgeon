use crate::region::ShmRegion;
use tracing::info;

pub fn sweep_stale_regions(known_names: &[String]) -> usize {
    let mut count = 0;
    for name in known_names {
        if ShmRegion::unlink(name).is_ok() {
            info!(name = %name, "unlinked stale shm region");
            count += 1;
        }
    }
    count
}

#[cfg(target_os = "linux")]
pub fn discover_stale_region_names() -> Vec<String> {
    let shm_dir = std::path::Path::new("/dev/shm");
    let mut names = Vec::new();
    if let Ok(entries) = std::fs::read_dir(shm_dir) {
        for entry in entries.flatten() {
            if let Some(name) = entry.file_name().to_str() {
                if name.starts_with("rs-") {
                    // Format: rs-{pid_hex}-{n}
                    // Only include regions whose owning PID is dead.
                    if let Some(pid) = parse_pid_from_region_name(name) {
                        if !is_pid_alive(pid) {
                            names.push(format!("/{name}"));
                        }
                    } else {
                        // Can't parse PID — include it as potentially stale
                        names.push(format!("/{name}"));
                    }
                }
            }
        }
    }
    names
}

/// Parse the PID from a region name of the form `rs-{pid_hex}-{n}`.
#[cfg(target_os = "linux")]
fn parse_pid_from_region_name(name: &str) -> Option<i32> {
    let rest = name.strip_prefix("rs-")?;
    let pid_hex = rest.split('-').next()?;
    u32::from_str_radix(pid_hex, 16).ok().map(|p| p as i32)
}

/// Check if a PID is still alive using `kill(pid, 0)`.
#[cfg(target_os = "linux")]
fn is_pid_alive(pid: i32) -> bool {
    // SAFETY: kill with signal 0 performs error checking without sending a signal.
    unsafe { libc::kill(pid, 0) == 0 }
}

#[cfg(target_os = "macos")]
pub fn discover_stale_region_names() -> Vec<String> {
    let state_file = dirs_or_home().join("shm_regions.json");
    match std::fs::read_to_string(&state_file) {
        Ok(contents) => serde_json::from_str::<Vec<String>>(&contents).unwrap_or_default(),
        Err(_) => Vec::new(),
    }
}

#[cfg(target_os = "macos")]
fn dirs_or_home() -> std::path::PathBuf {
    if let Some(home) = std::env::var_os("HOME") {
        std::path::PathBuf::from(home)
            .join(".local")
            .join("state")
            .join("rocket_surgeon")
    } else {
        std::path::PathBuf::from("/tmp/rocket_surgeon")
    }
}

#[cfg(target_os = "macos")]
pub fn register_region_name(name: &str) {
    let state_dir = dirs_or_home();
    let _ = std::fs::create_dir_all(&state_dir);
    let state_file = state_dir.join("shm_regions.json");
    let lock_file = state_dir.join("shm_regions.lock");

    let _lock = AdvisoryLock::acquire(&lock_file);
    let mut names = match std::fs::read_to_string(&state_file) {
        Ok(contents) => serde_json::from_str::<Vec<String>>(&contents).unwrap_or_default(),
        Err(_) => Vec::new(),
    };
    if !names.contains(&name.to_owned()) {
        names.push(name.to_owned());
    }
    let _ = std::fs::write(
        &state_file,
        serde_json::to_string(&names).unwrap_or_default(),
    );
    // _lock drops here, releasing the advisory lock
}

#[cfg(target_os = "macos")]
pub fn deregister_region_name(name: &str) {
    let state_dir = dirs_or_home();
    let state_file = state_dir.join("shm_regions.json");
    let lock_file = state_dir.join("shm_regions.lock");

    let _lock = AdvisoryLock::acquire(&lock_file);
    if let Ok(contents) = std::fs::read_to_string(&state_file) {
        let mut names: Vec<String> = serde_json::from_str(&contents).unwrap_or_default();
        names.retain(|n| n != name);
        let _ = std::fs::write(
            &state_file,
            serde_json::to_string(&names).unwrap_or_default(),
        );
    }
    // _lock drops here, releasing the advisory lock
}

/// RAII advisory file lock using `flock(2)`.
#[cfg(target_os = "macos")]
struct AdvisoryLock {
    fd: i32,
}

#[cfg(target_os = "macos")]
impl AdvisoryLock {
    fn acquire(path: &std::path::Path) -> Self {
        use std::os::unix::io::IntoRawFd;
        let file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(path)
            .expect("failed to open lock file");
        let fd = file.into_raw_fd();
        // SAFETY: fd is valid from OpenOptions::open above.
        unsafe {
            libc::flock(fd, libc::LOCK_EX);
        }
        Self { fd }
    }
}

#[cfg(target_os = "macos")]
impl Drop for AdvisoryLock {
    fn drop(&mut self) {
        // SAFETY: fd is valid, acquired in AdvisoryLock::acquire.
        unsafe {
            libc::flock(self.fd, libc::LOCK_UN);
            libc::close(self.fd);
        }
    }
}

#[cfg(target_os = "linux")]
pub fn register_region_name(_name: &str) {}

#[cfg(target_os = "linux")]
pub fn deregister_region_name(_name: &str) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::region::ShmRegion;

    #[test]
    fn sweep_stale_unlinks_matching_regions() {
        let name = format!("/rs-sw-{}", std::process::id());
        let region = ShmRegion::create(&name, 4096).unwrap();
        drop(region);

        let unlinked = sweep_stale_regions(std::slice::from_ref(&name));
        assert_eq!(unlinked, 1);
        assert!(ShmRegion::open(&name, 4096).is_err());
    }

    #[test]
    fn sweep_stale_ignores_missing() {
        let unlinked = sweep_stale_regions(&["/rs-nonexist-9999".to_owned()]);
        assert_eq!(unlinked, 0);
    }
}

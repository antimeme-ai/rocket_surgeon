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
                    names.push(format!("/{name}"));
                }
            }
        }
    }
    names
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
}

#[cfg(target_os = "macos")]
pub fn deregister_region_name(name: &str) {
    let state_dir = dirs_or_home();
    let state_file = state_dir.join("shm_regions.json");
    if let Ok(contents) = std::fs::read_to_string(&state_file) {
        let mut names: Vec<String> = serde_json::from_str(&contents).unwrap_or_default();
        names.retain(|n| n != name);
        let _ = std::fs::write(
            &state_file,
            serde_json::to_string(&names).unwrap_or_default(),
        );
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

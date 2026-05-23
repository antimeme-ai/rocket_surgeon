use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result};
use flate2::Compression;
use flate2::write::GzEncoder;
use tar::Builder;

pub struct BundleArtifact {
    pub name: String,
    pub data: Vec<u8>,
}

pub fn assemble_bundle(path: &Path, artifacts: &[BundleArtifact]) -> Result<u64> {
    let tmp_path = path.with_extension("tar.gz.tmp");
    let file = File::create(&tmp_path).with_context(|| format!("create {}", tmp_path.display()))?;
    let enc = GzEncoder::new(file, Compression::default());
    let mut tar = Builder::new(enc);

    for artifact in artifacts {
        let mut header = tar::Header::new_gnu();
        header.set_size(artifact.data.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        tar.append_data(&mut header, &artifact.name, artifact.data.as_slice())
            .with_context(|| format!("append {}", artifact.name))?;
    }

    let enc = tar.into_inner().context("finalize tar")?;
    enc.finish().context("finalize gzip")?;

    std::fs::rename(&tmp_path, path)
        .with_context(|| format!("rename {} → {}", tmp_path.display(), path.display()))?;

    let meta = std::fs::metadata(path).context("stat bundle")?;
    Ok(meta.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assemble_and_read_back() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test-bundle.tar.gz");

        let artifacts = vec![
            BundleArtifact {
                name: "manifest.json".into(),
                data: br#"{"version":"0.1.0"}"#.to_vec(),
            },
            BundleArtifact {
                name: "env.json".into(),
                data: br#"{"gpu":"none"}"#.to_vec(),
            },
        ];

        let size = assemble_bundle(&path, &artifacts).unwrap();
        assert!(size > 0);
        assert!(path.exists());

        let file = File::open(&path).unwrap();
        let dec = flate2::read::GzDecoder::new(file);
        let mut archive = tar::Archive::new(dec);
        let entries: Vec<String> = archive
            .entries()
            .unwrap()
            .map(|e| e.unwrap().path().unwrap().to_string_lossy().into_owned())
            .collect();
        assert!(entries.contains(&"manifest.json".to_string()));
        assert!(entries.contains(&"env.json".to_string()));
    }
}

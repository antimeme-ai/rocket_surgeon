// Register `--cfg loom` as a known cfg so the loom-gated interleaving test
// (`tests/loom_ring.rs`, `#![cfg(loom)]`) does not trip `unexpected_cfgs` on a
// normal build. The test only compiles under `RUSTFLAGS="--cfg loom"`.
fn main() {
    println!("cargo::rustc-check-cfg=cfg(loom)");
}

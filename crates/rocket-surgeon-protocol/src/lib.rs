#![forbid(unsafe_code)]

pub mod errors;
pub mod jsonrpc;
pub mod messages;
pub mod types;

pub fn checkpoint_layers(num_layers: u32) -> Vec<u32> {
    if num_layers <= 1 {
        return Vec::new();
    }
    let sqrt_l = f64::from(num_layers).sqrt().ceil() as u32;
    let interval = f64::from(num_layers) / f64::from(sqrt_l);
    (1..sqrt_l)
        .map(|i| (f64::from(i) * interval).floor() as u32)
        .collect()
}

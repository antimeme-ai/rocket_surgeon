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

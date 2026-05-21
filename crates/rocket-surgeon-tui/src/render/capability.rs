use std::env;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum GraphicsTier {
    /// No pixel graphics — half-block characters only
    HalfBlock = 1,
    /// Sixel graphics support
    Sixel = 2,
    /// Kitty graphics protocol
    Kitty = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorDepth {
    Basic,
    Color256,
    TrueColor,
}

#[derive(Debug, Clone)]
pub struct TerminalCapabilities {
    pub graphics: GraphicsTier,
    pub color: ColorDepth,
    pub width: u16,
    pub height: u16,
}

pub fn detect() -> TerminalCapabilities {
    let (width, height) = crossterm::terminal::size().unwrap_or((80, 24));
    let color = detect_color_depth();
    let graphics = detect_graphics_tier();

    TerminalCapabilities {
        graphics,
        color,
        width,
        height,
    }
}

fn detect_color_depth() -> ColorDepth {
    if let Ok(ct) = env::var("COLORTERM")
        && (ct == "truecolor" || ct == "24bit")
    {
        return ColorDepth::TrueColor;
    }

    if let Ok(term) = env::var("TERM")
        && term.contains("256color")
    {
        return ColorDepth::Color256;
    }

    ColorDepth::Basic
}

fn detect_graphics_tier() -> GraphicsTier {
    if env::var("TERM_PROGRAM").as_deref() == Ok("WezTerm")
        || env::var("TERM").as_deref() == Ok("xterm-kitty")
        || env::var("KITTY_WINDOW_ID").is_ok()
    {
        return GraphicsTier::Kitty;
    }

    if env::var("TERM_PROGRAM").as_deref() == Ok("mlterm") || env::var("SIXEL_SUPPORT").is_ok() {
        return GraphicsTier::Sixel;
    }

    GraphicsTier::HalfBlock
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier_ordering() {
        assert!(GraphicsTier::Kitty > GraphicsTier::Sixel);
        assert!(GraphicsTier::Sixel > GraphicsTier::HalfBlock);
    }

    #[test]
    fn detect_returns_something() {
        let caps = detect();
        assert!(caps.width > 0 || caps.width == 80);
    }
}

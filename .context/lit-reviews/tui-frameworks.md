---
topic: TUI frameworks, dual-interface design, terminal graphics capabilities
status: draft
created: 2026-05-14
sources: Ratatui, Textual, Cursive, GDB-MI, Neovim RPC, DAP, tmux control mode
---

# TUI Frameworks + Dual-Interface Design: Lit Review

How to build a terminal debugger UI that serves both humans and LLMs.

## Ratatui (Rust) — Primary Candidate

### Architecture
- **Immediate-mode rendering**: redraw entire interface every frame based on current state. No persistent scene graphs.
- **Double-buffer diff**: efficient diffing between frames, only changed cells written. Typically <1ms per frame.
- **Backend abstraction**: Backend trait implemented by crossterm/termion/termwiz. Swap terminals without touching app code.
- **Constraint-based layouts**: proportional or fixed-size specifications resolved algorithmically.

### Performance
- Sub-millisecond rendering, 60+ FPS even with complex layouts
- Zero-cost abstractions
- Max terminal cell count ~65,535 — rendering bottleneck is write syscall, not compute

### Widget Ecosystem
- Core: charts, sparklines, tables, gauges, scrollable lists, progress bars, scrollbars
- **ratatui-image**: supports Sixel, Kitty protocol, iTerm2 protocol, unicode half-blocks
- Used in production: BugStalker (Rust debugger), gitui (VCS), xplr (file explorer)

### Why Ratatui for rocket_surgeon
1. Immediate-mode = precise control for complex debugging UIs
2. Rust performance = no lag stepping through 100+ layer transformers
3. Terminal graphics via ratatui-image = attention heatmaps in terminal
4. Stateless rendering = TUI is purely a view layer, state lives in core engine

## Textual (Python) — Secondary/Orchestration Layer

### Architecture
- React/web-like: component-based, CSS-like styling, reactive state management
- 120 FPS renders vs 20 FPS for curses-based
- Delta updates via Rich's segment trees
- Tradeoff: Python VM overhead (~14ms startup) vs immediate-mode Rust

### When to use
- Python orchestration layer (if we have one)
- Rapid prototyping of UI concepts
- Not for the primary production TUI

## Cursive (Rust)
- Retained-mode rendering: maintains widget state and scene graph
- More opinionated, less flexible for custom rendering
- Not recommended for our use case

## Notable TUI Design Patterns

### From lazygit
- Box/view-based layout with most views visible simultaneously
- Consistent keyboard navigation, deliberate color schemes (WCAG 2.0 AA)

### From btop/bottom
- Real-time ASCII charts using Braille characters and block elements
- Combine text-based visualization + color for temporal trends

### From helix editor
- Components in Compositor stack, each renders in order
- Decoupled event definitions (helix-event crate)
- Layered architecture: UI overlays without coupling to core logic

### From gitui
- Tab-based interface (status, logs, files, stashing)
- Fast startup, low memory with Rust + immediate-mode

## Dual-Interface Design: The Critical Pattern

### GDB: MI + TUI
- Machine Interface (MI): line-based, structured output, sequence tokens
- TUI: curses-based synchronized windows
- Multiple interpreters simultaneously on different I/O streams
- TUI and MI operate independently but sync state

### Neovim: RPC API
- MessagePack-RPC protocol
- Channel layer: stdio, TCP, named pipes, internal Lua
- C API functions auto-exposed via RPC
- **Stability guarantee: RPC API never breaks**

### tmux: Control Mode
- -C flag activates control mode: text protocol instead of drawing
- Standard tmux commands + async notifications (%-prefixed)
- Works over SSH, easy to parse

### DAP (Debug Adapter Protocol)
- JSON-based, header/content structure
- Request/Response/Event message types
- Machine-processable JSON-schema specification
- Async events for reactive debugging

### Command Result Pattern
- `CommandResult { success, human_message, structured_data }`
- Single backend supports both TUI and programmatic access
- LLM-friendly: structured output for reliable parsing

## Terminal Graphics Capabilities

### Sixel
- 6-pixel-high horizontal strips, older DEC format
- No true 24-bit color
- Safe fallback for heterogeneous environments
- Support: VS Code 1.80+, Konsole, tmux (with --enable-sixel)

### Kitty Graphics Protocol
- Arbitrary pixel graphics, GPU-accelerated, alpha blending with text
- Stateful: re-render images at different positions
- Modern, trending toward broad adoption
- Support: WezTerm, Ghostty, Konsole, Rio

### iTerm2 Inline Images
- OSC 1337 sequences with base64-encoded data
- Width/height in pixels/percent/cells
- Simple, widely supported

### ratatui-image widget
- Abstracts over Sixel, Kitty, iTerm2, unicode half-blocks
- Handles font-size detection, protocol availability
- Coordinates immediate-mode TUI with stateful graphics protocols

## Three-Layer Architecture for rocket_surgeon

### Layer 1: Core Engine (Pure Rust)
- Transformer state representation
- Step semantics (forward, backward, inspect, modify)
- No UI concerns — purely state transformation + inspection

### Layer 2: Machine Interface (Structured Protocol)
- JSON-RPC or similar over stdio/TCP
- Message types: step_forward, inspect_activation, query_attention, modify_state, etc.
- Supports LLM clients, Python scripts, notebooks, web dashboards
- **This is the primary interface. The TUI is a consumer of it.**

### Layer 3: TUI (Ratatui)
- Reads from core engine via protocol
- Renders: activation heatmaps, attention matrices, computational graph, token step-through
- Terminal graphics via ratatui-image for rich visualization
- Keyboard/mouse interaction translated to protocol commands

This three-layer design is proven by GDB, Neovim, tmux, and DAP. The debugger remains useful whether called from a terminal or an LLM orchestrator.

## Sources

- ratatui.rs, github.com/ratatui/ratatui
- textual.textualize.io, github.com/Textualize/textual
- github.com/gyscos/cursive
- Neovim API docs (neovim.io)
- GDB/MI Protocol (sourceware.org)
- DAP specification (microsoft.github.io)
- tmux Control Mode wiki
- Kitty graphics protocol docs
- ratatui-image widget repo
- helix architecture docs
- Building AI Coding Agents for the Terminal (arxiv 2603.05344)

# TUI + Rust Infrastructure Analysis for rocket_surgeon

Source material: ratatui, taskwarrior-tui, trippy, git-cliff, safetensors, pyo3
(all from quarantine/)

---

## 1. Ratatui Patterns

### Workspace Architecture (v0.30+)

Ratatui is now a modular workspace with separated concerns:

```
ratatui           -- umbrella re-export crate (apps depend on this)
ratatui-core      -- Widget/StatefulWidget traits, Buffer, Layout, Style, Text
ratatui-widgets   -- built-in widgets (Table, Chart, Canvas, Sparkline, etc.)
ratatui-crossterm -- crossterm backend
ratatui-macros    -- convenience macros
```

**Implication for rocket_surgeon**: our custom widgets should depend on
`ratatui-core` only (stable API, fast compile), not the full `ratatui` crate.
The app binary depends on `ratatui` for the re-exports and backend.

### The Widget Trait

```rust
pub trait Widget {
    fn render(self, area: Rect, buf: &mut Buffer) where Self: Sized;
}

pub trait StatefulWidget {
    type State: ?Sized;
    fn render(self, area: Rect, buf: &mut Buffer, state: &mut Self::State);
}
```

Key patterns observed in the codebase:
- Widgets are **consumed** on render (`self`, not `&self`). This is the
  immediate-mode model: widgets are ephemeral structs constructed each frame.
- To avoid cloning app state, implement `Widget for &MyWidget` (reference impl).
  The demo2 example does exactly this: `impl Widget for &App`.
- `StatefulWidget` is for widgets that track scroll offset, selection, etc.
  between frames. The `State` lives in the app, not the widget.
- Widgets compose by nesting: a parent widget calls `child.render(sub_area, buf)`
  inside its own `render` method. There is no widget tree -- it's just function
  calls.

### The Buffer Abstraction

`Buffer` is a flat `Vec<Cell>` mapped to a `Rect`. Every widget writes into
a Buffer; no widget touches the terminal directly. The Terminal holds two
Buffers and diffs them each frame, sending only changed cells to the backend.

This is a **retained-mode diff on an immediate-mode API**: you rebuild the
whole UI each frame, but only the delta is flushed. This is the key performance
characteristic.

### Layout Engine

Layout uses the **Cassowary constraint solver** (via `kasuari` crate).
Constraints are:
- `Length(n)` -- fixed cells
- `Percentage(n)` -- fraction of parent
- `Ratio(a, b)` -- proportional
- `Fill(weight)` -- absorb remaining space proportionally
- `Min(n)`, `Max(n)` -- bounds

Idiomatic pattern: destructure at compile time when count is known:
```rust
let [header, body, footer] = Layout::vertical([
    Constraint::Length(3),
    Constraint::Fill(1),
    Constraint::Length(1),
]).areas(area);
```

`Flex` controls distribution of excess space: Start, End, Center,
SpaceBetween, SpaceAround, SpaceEvenly. This is CSS-flexbox-like.

### Rendering Model

The canonical event loop (from demo2, the most idiomatic example):

```rust
fn run(mut self, mut terminal: DefaultTerminal) -> Result<()> {
    while self.is_running() {
        terminal.draw(|frame| self.render(frame))?;
        self.handle_events()?;
    }
    Ok(())
}
```

`terminal.draw()` is the key call. It:
1. Checks terminal size, resizes buffers if needed
2. Creates a Frame backed by the current buffer
3. Runs your closure to populate the buffer
4. Diffs current vs previous buffer
5. Writes changes to backend
6. Applies cursor state
7. Swaps buffers, flushes backend

The render callback is **synchronous**. Async data fetching must happen
outside the draw call (see async patterns below).

### Event Handling

Crossterm provides `event::poll(timeout)` and `event::read()`. The pattern:

```rust
let timeout = Duration::from_secs_f64(1.0 / 50.0);
if !event::poll(timeout)? { return Ok(()); }
if let Some(key) = event::read()?.as_key_press_event() {
    match key.code { ... }
}
```

For async apps (tokio), use `crossterm::event::EventStream`:
```rust
tokio::select! {
    _ = interval.tick() => { terminal.draw(...)?; },
    Some(Ok(event)) = events.next() => self.handle_event(&event),
}
```

### Available Widgets for Data Visualization

| Widget | Use for rocket_surgeon |
|--------|----------------------|
| `Table` + `TableState` | Layer/module listing, tensor metadata, expert routing tables |
| `Chart` (line/scatter) | Activation distributions over time, loss curves, gradient norms |
| `BarChart` | Expert load distribution, attention head activity, token frequency |
| `Sparkline` | Inline activation magnitude sparklines per layer |
| `Canvas` + `Shape` trait | Custom heatmaps for attention matrices, activation grids |
| `Paragraph` | Log output, tensor value display, command input |
| `Gauge`/`LineGauge` | Progress bars for forward pass position |
| `Tabs` | Mode switching (layers, heads, experts, tensors) |

The `Canvas` widget is particularly relevant: it provides a pixel-grid
abstraction with Braille, block, half-block, sextant, and octant markers.
Custom `Shape` implementations can render attention heatmaps at sub-character
resolution. The `Points` shape is a scatter plot. The `Line` shape draws
between two coordinate points.

---

## 2. Real-World TUI Applications

### taskwarrior-tui Architecture

**State structure**: One massive `TaskwarriorTui` struct holding ALL app state
(~50 fields): task data, selection state, mode, config, UI state, event loop
handle, history, completion state. This is the "God object" pattern -- it works
for medium complexity but becomes unwieldy.

**Event loop**: Uses tokio with an `EventLoop` struct that wraps a
`mpsc::UnboundedChannel<Event<KeyCode>>`. A background task reads crossterm
events and converts them to app-level `Event` variants (Input, Paste, Tick,
Closed). The main loop receives from the channel.

**Mode system**: Uses `enum Mode { Tasks(Action), Projects, Timesheet, Calendar }`
where `Action` is a sub-enum for task-specific modes (Report, Filter, Add,
Annotate, etc.). Input handling dispatches on mode first, then keybinding.

**Keybinding system**: `KeyConfig` struct with one field per action, deserialized
from config. Simple approach -- each action maps to exactly one key. Trippy
does it better (see below).

**Lessons**:
- Flat state struct works for single-purpose apps but won't scale to
  rocket_surgeon's complexity (layers, heads, experts, multiple tensors,
  forward/backward stepping, surgical interventions).
- The event channel pattern is solid: decouple terminal events from app logic.
- The mode/action enum pattern is correct for modal TUIs.

### trippy Architecture

**Much cleaner separation of concerns**. Workspace with:
- `trippy-core` -- tracing engine (no TUI dependency)
- `trippy-tui` -- frontend (ratatui + crossterm)
  - `frontend/` -- event loop, binding config, rendering
  - `frontend/render/` -- one file per UI region (app, header, body, table,
    chart, histogram, footer, help, settings, etc.)

**State**: `TuiApp` struct is focused (~20 fields): trace data snapshot,
config, table state, selection indices, flow state, show/hide toggles, zoom.
The actual trace data lives in `trippy-core::State` and is snapshotted each
frame via `snapshot_trace_data()`.

**Real-time data flow**:
1. Background tracer threads write into a shared `State` (lock-free or
   mutex-protected).
2. Each frame, `app.snapshot_trace_data()` takes a consistent read of the
   current data.
3. Render functions receive `&TuiApp` immutably.

This is the correct pattern for rocket_surgeon: the Python-side forward pass
engine writes tensor data into shared state, the TUI snapshots it each frame.

**Rendering decomposition**:
```
render/app.rs      -- top-level layout
render/header.rs   -- title bar
render/body.rs     -- dispatches to table/chart/map/splash/bsod
render/table.rs    -- main data table
render/chart.rs    -- Chart widget for ping history
render/histogram.rs -- BarChart for frequency distribution
render/footer.rs   -- status bar
render/help.rs     -- help overlay
render/settings.rs -- settings dialog
```

Each render function takes `(f: &mut Frame, app: &TuiApp, rect: Rect)`.
No widget structs for the top-level layout -- just free functions that
compose built-in widgets.

**Keybinding system**: `TuiBindings` struct with named binding fields, each
a `KeyBinding` that can match against `KeyEvent`:
```rust
if bindings.toggle_chart.check(key) { app.toggle_chart(); }
```
This is more extensible than taskwarrior-tui's approach and supports
configurable bindings.

**Data visualization**:
- Chart widget with Braille markers for line graphs of ping latency
- BarChart for frequency histograms
- Canvas with world map for geographic visualization
- Dynamic zoom via `zoom_factor`

**Freeze/unfreeze**: `frozen_start: Option<SystemTime>` -- when set, stops
snapshotting new data. This is exactly what rocket_surgeon needs for
stepping through ticks.

---

## 3. Data Visualization in Terminal

### Available Ratatui Primitives

**Chart (line/scatter)**:
- Plots `Dataset`s of `(f64, f64)` points
- `GraphType::Line` or `GraphType::Scatter`
- `Marker::Braille` for highest resolution (2x4 dots per cell = 160x96 in 80x24)
- `Marker::HalfBlock` for solid filled areas
- X/Y axes with labels, bounds, title
- Multiple datasets with different styles

**Canvas (freeform drawing)**:
- Coordinate space mapped to terminal cells
- Multiple marker resolutions: Braille (2x4), HalfBlock (1x2), Block (1x1),
  Sextant (2x3), Octant (2x4 blocks)
- Custom `Shape` trait for arbitrary drawing
- Layer system for compositing shapes
- Labels can be placed at arbitrary (x, y) coordinates

**BarChart**:
- Vertical or horizontal bars
- Grouped bars via `BarGroup`
- Customizable bar width, gap, style, value display
- Good for expert load distribution, attention head activity

**Sparkline**:
- Compact inline bar chart
- Uses 8-level or 9-level bar characters
- Good for per-layer activation magnitude in a table column

### Tensor/Attention Visualization Strategy

For rocket_surgeon's specific needs:

1. **Attention heatmap**: Custom `Shape` implementation on Canvas.
   Map attention weights [0,1] to color intensity. Use Braille markers
   for maximum resolution. A 40x20 cell area gives 80x80 "pixels" with
   Braille, enough for small attention matrices. For larger matrices,
   downsample or aggregate.

2. **Activation distribution**: Chart with histogram overlay. Use BarChart
   for binned distribution, Sparkline for inline per-layer summary.

3. **Expert routing**: BarChart showing token-to-expert assignment counts.
   Color-code by expert utilization.

4. **Gradient flow**: Chart widget with line graph showing gradient magnitude
   per layer (x = layer index, y = gradient norm).

5. **Tensor value inspector**: Table with cells showing formatted float values.
   Highlight cells by magnitude using style colors (blue for negative, red
   for positive, intensity by magnitude).

6. **Layer-by-layer sparklines**: Table where each row is a layer, and one
   column contains a Sparkline of activation magnitudes.

---

## 4. safetensors: Checkpoint Serialization

### Format Structure

```
[8 bytes: header_size as u64 LE]
[header_size bytes: JSON header, padded to 8-byte alignment]
[tensor data: contiguous byte buffers, ordered by offset]
```

The JSON header maps tensor names to `{dtype, shape, data_offsets: [start, end]}`.
Tensors are ordered by alignment (largest dtype first) for optimal memory layout.

### Rust API

**Serialization**:
```rust
// In-memory
let bytes: Vec<u8> = safetensors::serialize(&tensor_map, metadata)?;

// Direct to file (avoids full allocation)
safetensors::serialize_to_file(&tensor_map, metadata, path)?;
```

The `View` trait is the key abstraction:
```rust
pub trait View {
    fn dtype(&self) -> Dtype;
    fn shape(&self) -> &[usize];
    fn data(&self) -> Cow<'_, [u8]>;
    fn data_len(&self) -> usize;
}
```

This allows zero-copy serialization from any backing store (CPU, GPU via
explicit copy). The `Cow` return allows both borrowed (mmap) and owned (GPU
copy) data paths.

**Deserialization (zero-copy with mmap)**:
```rust
let file = File::open(path)?;
let mmap = unsafe { MmapOptions::new().map(&file)? };
let tensors = SafeTensors::deserialize(&mmap)?;
let tensor = tensors.tensor("layer.0.weight")?;
// tensor.data() is a &[u8] slice into the mmap -- zero copy
```

The `SafeTensors<'data>` struct borrows the backing buffer. With mmap, tensor
data is accessed on-demand via page faults -- no upfront copy.

**Slicing**: `TensorView::sliced_data()` returns a `SliceIterator` yielding
byte chunks, enabling extraction of sub-tensors without copying the whole thing.

### Dtype Coverage

safetensors supports all modern ML dtypes:
- Standard: BOOL, U8, I8, U16, I16, U32, I32, U64, I64, F16, BF16, F32, F64
- FP8 variants: F8_E5M2, F8_E4M3, F8_E4M3FNUZ, F8_E5M2FNUZ, F8_E8M0
- Sub-byte: F4 (4-bit float), F6_E2M3, F6_E3M2
- Complex: C64

This covers everything rocket_surgeon will encounter in modern transformer
checkpoints.

### Usage Pattern for rocket_surgeon

1. Load checkpoint via mmap for read-only inspection
2. For surgical modifications: deserialize affected tensors, modify in-place
   or copy-on-write, serialize modified state back
3. For snapshots during stepping: serialize current activation state to
   safetensors for logging/replay
4. The `View` trait can wrap our internal tensor representation directly

---

## 5. PyO3 Patterns

### Tensor Data Sharing

The safetensors Python binding demonstrates the critical pattern for zero-copy
tensor sharing:

```rust
#[pyclass(frozen, from_py_object)]
struct TensorSpec {
    dtype: Dtype,
    shape: Vec<usize>,
    data_ptr: u64,    // raw pointer to tensor data
    data_len: usize,
}
```

The `data_ptr` is a raw memory address. This is how frameworks pass tensor
data without copying: the Python side provides a pointer + length, the Rust
side creates a slice from it. **SAFETY**: the caller must ensure the buffer
stays alive.

For rocket_surgeon, the pattern would be:
1. Python engine holds PyTorch tensors
2. Pass `.data_ptr()` and `.numel() * element_size` to Rust
3. Rust creates a `&[u8]` view for read-only inspection
4. For modifications, write directly into the buffer (requires mut pointer)

### GIL Release Patterns

From the safetensors bindings:
```rust
#[pyfunction]
fn serialize(py: Python, tensor_dict: ..., metadata: ...) -> PyResult<PyBytes> {
    let out = py.detach(|| {
        safetensors::tensor::serialize(...)
    })?;
    Ok(PyBytes::new(py, &out))
}
```

`py.detach(|| ...)` releases the GIL while the closure runs, allowing Python
threads to proceed. This is critical for:
- Long-running Rust computations (TUI rendering, data processing)
- Parallel Python+Rust work (forward pass in Python, TUI in Rust)

From the PyO3 guide:
- `Python::detach` costs < 1ms for attach/detach
- Any work > a few ms benefits from detaching
- **Free-threaded Python (3.14+)**: `detach` is still best practice for
  "stop the world" GC events

### Class Definitions

```rust
#[pyclass]
struct MyClass {
    inner: SomeRustType,
}

#[pymethods]
impl MyClass {
    #[new]
    fn new(args...) -> Self { ... }

    #[getter]
    fn some_property(&self) -> ... { ... }

    fn some_method(&self, ...) -> PyResult<...> { ... }
}
```

Restrictions: no lifetime parameters, no generics, must be `Send` (thread-safe).
Use `Arc` for shared ownership between Python and Rust.

### Async Bridging

For rocket_surgeon, the TUI runs in a Rust thread/task while the Python
forward pass engine runs in Python threads. Communication options:

1. **Shared memory**: Use raw pointers (as safetensors does) for tensor data.
   Use `Arc<Mutex<>>` or `Arc<RwLock<>>` for control state.

2. **Channels**: `tokio::sync::mpsc` or `std::sync::mpsc` for commands
   (step, modify, set breakpoint) from TUI to engine.

3. **JSON-RPC over channels**: Since rocket_surgeon uses JSON-RPC as its
   protocol, the Rust TUI acts as a JSON-RPC client. The Python engine
   is the server. Communication can go through:
   - Unix domain sockets (for separate processes)
   - In-process channels (for embedded mode via PyO3)

### Performance Tips from PyO3 Guide

- Use `cast` instead of `extract` for type checks (avoids PyErr allocation)
- Access to `Bound<'py, T>` gives zero-cost `py` token via `.py()`
- Prefer Rust tuples over `PyTuple` for `vectorcall` protocol
- Disable global reference pool (`pyo3_disable_reference_pool`) for
  high-frequency Python-Rust boundary crossing
- Free-threaded Python (3.14+) removes GIL bottleneck entirely

---

## 6. Concrete Architecture Recommendation

### Process Model

```
[Python Process]                    [Rust TUI Process]
   |                                      |
   | (forward pass engine)                | (ratatui terminal UI)
   |                                      |
   +--- JSON-RPC server ---+--- JSON-RPC client ---+
   |    (Unix socket /     |    (protocol layer)   |
   |     stdio pair)       |                       |
   +--- shared mmap ---+--- tensor reader --------+
        (safetensors)       (zero-copy inspection)
```

Two processes communicating via:
1. **JSON-RPC** over Unix socket or stdio for control commands (step, set
   breakpoint, modify activations, query state)
2. **Shared memory** (mmap'd safetensors files) for tensor data inspection
   (zero-copy reads of activation tensors, weights, gradients)

This matches the spec's dual-interface design: the JSON-RPC protocol serves
both the TUI and LLM clients identically.

### Rust Crate Organization (git-cliff-inspired workspace)

```
rocket_surgeon/
  Cargo.toml                    # workspace root
  rs/
    rocket-surgeon-protocol/    # JSON-RPC message types, (de)serialization
      src/lib.rs
    rocket-surgeon-core/        # core debugger state machine, breakpoints, stepping
      src/lib.rs
    rocket-surgeon-tensors/     # safetensors integration, tensor views, dtype handling
      src/lib.rs
    rocket-surgeon-tui/         # ratatui TUI application
      src/
        main.rs                 # entry point, terminal setup
        app.rs                  # App state struct
        event.rs                # event loop, channel setup
        protocol_client.rs      # JSON-RPC client
        render/
          mod.rs
          layout.rs             # top-level layout
          header.rs             # title bar, mode indicator, tick counter
          layer_tree.rs         # layer/module tree sidebar
          tensor_view.rs        # tensor value inspector
          activation_chart.rs   # activation distribution charts
          attention_heatmap.rs  # attention matrix visualization
          expert_panel.rs       # MoE expert routing view
          command_bar.rs        # command input / status bar
          help.rs               # help overlay
        widgets/
          mod.rs
          heatmap.rs            # custom Canvas-based heatmap widget
          tensor_table.rs       # formatted tensor value table
          sparkline_column.rs   # sparkline as table column
    rocket-surgeon-python/      # PyO3 bindings
      src/lib.rs
```

### TUI App State (trippy-inspired)

```rust
pub struct App {
    // Connection
    client: ProtocolClient,         // JSON-RPC client
    connection_state: ConnectionState,

    // Debugger state (snapshotted from server)
    model_info: ModelInfo,          // architecture, layer names, shapes
    current_tick: TickState,        // current position in forward pass
    breakpoints: Vec<Breakpoint>,

    // Tensor data (mmap'd)
    tensor_store: TensorStore,      // safetensors mmap handles

    // UI state
    mode: Mode,                     // Navigate, Inspect, Command, Help
    focus: Focus,                   // which pane has focus
    layer_tree_state: TreeState,    // tree selection state
    tensor_view_state: TensorViewState,
    chart_state: ChartState,
    command_input: LineBuffer,

    // Display toggles
    show_help: bool,
    show_attention: bool,
    show_experts: bool,
    frozen: bool,                   // pause data updates (like trippy)

    // Config
    config: TuiConfig,
    bindings: KeyBindings,
}
```

### Event Loop (async, tokio-based)

```rust
async fn run(mut self, mut terminal: DefaultTerminal) -> Result<()> {
    let mut events = EventStream::new();
    let mut tick = tokio::time::interval(Duration::from_millis(33)); // ~30fps

    loop {
        // Snapshot server state if not frozen
        if !self.app.frozen {
            self.app.poll_server_state().await?;
        }

        tokio::select! {
            _ = tick.tick() => {
                terminal.draw(|f| render::layout::render(f, &self.app))?;
            }
            Some(Ok(event)) = events.next() => {
                if self.app.handle_event(event)? == Action::Quit {
                    break;
                }
            }
            msg = self.app.client.next_notification() => {
                self.app.handle_notification(msg?)?;
            }
        }
    }
    Ok(())
}
```

Three event sources:
1. **Timer tick**: triggers re-render
2. **Terminal events**: keyboard/mouse input
3. **Server notifications**: debugger state changes pushed from Python

### Rendering Pipeline (trippy-inspired decomposition)

```rust
// render/layout.rs
pub fn render(f: &mut Frame, app: &App) {
    let [header, body, footer] = Layout::vertical([
        Constraint::Length(2),
        Constraint::Fill(1),
        Constraint::Length(1),
    ]).areas(f.area());

    header::render(f, app, header);

    let [sidebar, main] = Layout::horizontal([
        Constraint::Length(30),
        Constraint::Fill(1),
    ]).areas(body);

    layer_tree::render(f, app, sidebar);

    match app.focus {
        Focus::TensorInspector => tensor_view::render(f, app, main),
        Focus::ActivationChart => activation_chart::render(f, app, main),
        Focus::AttentionHeatmap => attention_heatmap::render(f, app, main),
        Focus::ExpertPanel => expert_panel::render(f, app, main),
    }

    if app.mode == Mode::Command {
        command_bar::render(f, app, footer);
    } else {
        status_bar::render(f, app, footer);
    }

    if app.show_help {
        help::render_overlay(f, app);
    }
}
```

### Key Design Decisions

1. **Immediate-mode rendering**: Rebuild the full UI each frame. Ratatui diffs
   automatically. This is simpler and more correct than tracking dirty state.

2. **Snapshot-based data flow**: Each frame, snapshot the current debugger state
   (like trippy's `snapshot_trace_data()`). This avoids holding locks during
   rendering and guarantees a consistent view.

3. **Widget for &App pattern**: Implement `Widget for &App` (or for specific
   view structs) to avoid cloning state every frame.

4. **Canvas-based heatmaps**: For attention matrices, implement a custom
   `Shape` that maps f32 values to colored Braille dots. This gives 80x80
   "pixel" resolution in a 40x20 cell area.

5. **Configurable keybindings**: Follow trippy's pattern with a `KeyBindings`
   struct and `.check(key)` method on each binding.

6. **Mode stack**: Use a mode stack rather than flat enum to support nested
   modes (e.g., Navigate > Inspect > Edit > Confirm).

7. **safetensors for tensor exchange**: Use mmap'd safetensors files as the
   shared-memory mechanism for tensor data. The Python engine writes snapshots,
   the Rust TUI reads them zero-copy. This is simpler and safer than raw
   shared memory.

8. **Separate processes**: Run TUI and engine as separate processes connected
   via JSON-RPC. This isolates GPU crashes from the TUI, allows connecting to
   remote engines, and is cleaner than in-process PyO3 embedding for the
   primary use case. PyO3 embedding remains an option for tight integration.
